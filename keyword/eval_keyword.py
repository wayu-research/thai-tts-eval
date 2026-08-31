#!/usr/bin/env python3
"""thai-tts-keyword-bench — hard-keyword accuracy for Thai TTS, one file, no installs.

Synthesize the 1,531 sentences, transcribe them with ONE Thai ASR, and check that each
sentence's keyword shows up in the transcript. A keyword is CORRECT if the transcript
contains its `expected` form or any `alternates` entry — exact substring after Thai
canonicalization, tried with and without spaces. No fuzzy matching: `expected` is
authored relative to the scoring ASR, so any deviation (including one tone mark) is an
error. Legitimate ASR spelling variation is absorbed by `alternates` (a data edit), not
by loosening the matcher.

    # 0. the texts to feed your TTS (writes {id}\\t{text}; --contract picks the variant)
    python3 eval_keyword.py texts --out texts.tsv

    # 1. wavs named {id}.wav -> transcripts.json  (needs torch/transformers/soundfile)
    python3 eval_keyword.py transcribe --audio-dir audio/mysystem \\
        --out runs/mysystem/transcripts.json

    # 2. transcripts -> results.csv + summary.md + metrics.json   (stdlib only)
    python3 eval_keyword.py score --transcripts runs/mysystem/transcripts.json \\
        --out runs/mysystem

    # A vs B on the same items: McNemar exact + per-category deltas   (stdlib only)
    python3 eval_keyword.py compare --a base=runs/base/transcripts.json \\
        --b new=runs/new/transcripts.json

    python3 eval_keyword.py validate     # check the data file
    python3 eval_keyword.py selftest     # 21 assertions on the normalizer + matcher

Scoring, normalization and the item schema derive from the thai-tts-keyword-eval
harness (github.com/potsawee/th01-tts-eval), MIT-spirited flat-script original; the
1,531-item set and its gold were authored and regolded for this release. Keep the core
stdlib-only: model dependencies live behind the lazy imports in `transcribe`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from math import comb, sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The item set is published as a Hugging Face dataset rather than vendored here, so
# there is one copy to correct and no chance of a checkout drifting from the gold.
HF_DATASET = "wayu-ai/thai-tts-keyword-bench"
ITEMS_NAME = "eval_set.jsonl"
HF_FILE = f"data/{ITEMS_NAME}"          # the dataset keeps it under data/
DEFAULT_ITEMS = HERE / "data" / ITEMS_NAME
# The scoring recognizer. Not a default — the gold `expected`/`alternates` are
# authored relative to this model's output style, so it is fixed and there is no
# flag to change it. Bring your own transcripts if you need another recognizer.
ASR_MODEL = "typhoon-ai/typhoon-whisper-large-v3"

CATEGORIES = ["code_switch", "names", "rare_words", "informal_spelling", "long_sentences"]
# The pre-registered slice for paired A/B: the three categories where systems actually
# differ. A 7-point move here is invisible when diluted over all 1,531 items.
PRIMARY_SLICE = ("names", "long_sentences", "code_switch")
CONTRACTS = {"raw": "text", "mono": "text_mono", "bilingual": "text_bilingual"}


# ---------------------------------------------------------------------------
# §1  Thai canonicalization — applied to BOTH sides before any comparison
# ---------------------------------------------------------------------------
# Visually identical Thai strings often differ in bytes: ำ typed as ํ+า, a tone mark
# typed before the vowel instead of after, เเ vs แ, zero-width characters pasted from
# the web, Thai vs Arabic digits. Without this, correct transcripts fail at random.
#
# Keep it symmetric and boring: it must NEVER "fix" a pronunciation-level difference.
# A changed tone mark or vowel length is a REAL error and has to survive normalization
# so the matcher can catch it. `selftest` asserts exactly that.

_VOWELS = set("ัิีึืุู็")
_NASAL = set("ํ๎")
_TONES = set("่้๊๋")
_SILENCERS = set("ฺ์")
_COMBINING = _VOWELS | _NASAL | _TONES | _SILENCERS
# canonical order within one consonant cluster:
# above/below vowel -> nikhahit/yamakkan -> tone mark -> phinthu/thanthakhat
_PRIORITY = {
    **{c: 0 for c in _VOWELS},
    **{c: 1 for c in _NASAL},
    **{c: 2 for c in _TONES},
    **{c: 3 for c in _SILENCERS},
}

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿­"), None)
_THAI_DIGITS = {ord("๐") + i: str(i) for i in range(10)}
_WS = re.compile(r"\s+")
_THAI_LETTER = re.compile(r"[ก-ฮะ-๎]")


def _reorder_marks(text: str) -> str:
    """Stable-sort each run of combining marks into canonical order. Unicode NFC does
    not fix tone-before-vowel typing — Thai above-vowels have combining class 0."""
    out: list[str] = []
    run: list[str] = []
    for ch in text:
        if ch in _COMBINING:
            run.append(ch)
        else:
            if run:
                run.sort(key=_PRIORITY.__getitem__)
                out.extend(run)
                run = []
            out.append(ch)
    if run:
        run.sort(key=_PRIORITY.__getitem__)
        out.extend(run)
    return "".join(out)


def _expand_yamok(text: str) -> str:
    """Expand ๆ by repeating the preceding run of Thai letters, so 'ต่างๆ' and
    'ต่างต่าง' compare equal regardless of which form the ASR emits."""
    out: list[str] = []
    for ch in text:
        if ch == "ๆ":
            j = len(out)
            while j > 0 and _THAI_LETTER.match(out[j - 1]):
                j -= 1
            out.extend(out[j:])
        else:
            out.append(ch)
    return "".join(out)


def normalize(text: str) -> str:
    """NFC, zero-width strip, ำ/แ composition, Thai->Arabic digits, canonical mark
    order, ๆ expansion, ASCII casefold, whitespace collapse."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_ZERO_WIDTH)
    text = text.replace("ํา", "ำ").replace("เเ", "แ")
    text = text.translate(_THAI_DIGITS)
    text = _reorder_marks(text)
    text = _expand_yamok(text)
    text = text.casefold()
    return _WS.sub(" ", text).strip()


def despace(text: str) -> str:
    """Space-free view: Thai has no word spaces and ASRs insert/omit them
    unpredictably around mixed script — never let whitespace decide a verdict."""
    return _WS.sub("", text)


# ---------------------------------------------------------------------------
# §2  Data
# ---------------------------------------------------------------------------

def resolve_items(path: str | Path | None = None) -> Path:
    """Where the item set lives: a local copy if there is one, else the dataset.

    The items are published once, as a Hugging Face dataset, and are deliberately
    not vendored here -- one copy to correct, and no chance of a repo checkout
    drifting from the published gold.  An explicit `path`, or a copy sitting at
    :data:`DEFAULT_ITEMS`, always wins, so a pinned checkout still runs offline.
    """
    if path is not None:
        return Path(path)
    if DEFAULT_ITEMS.exists():
        return DEFAULT_ITEMS
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            f"the item set is not vendored in this repo. Install huggingface-hub to "
            f"fetch it from {HF_DATASET}, or pass --items with a local {ITEMS_NAME}."
        ) from None
    try:
        return Path(hf_hub_download(repo_id=HF_DATASET, filename=HF_FILE,
                                    repo_type="dataset"))
    except Exception as exc:
        raise SystemExit(f"could not fetch {ITEMS_NAME} from {HF_DATASET}: {exc}\n"
                         f"pass --items with a local copy instead") from None


def load_items(path: str | Path | None = None) -> list[dict]:
    path = resolve_items(path)
    items = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {e}")
    return items


def validate_items(items: list[dict]) -> list[str]:
    problems, seen = [], set()
    for it in items:
        iid = it.get("id", "<missing id>")
        if iid in seen:
            problems.append(f"[{iid}] duplicate id")
        seen.add(iid)
        for field in ("id", "category", "text", "keyword", "expected"):
            if not it.get(field):
                problems.append(f"[{iid}] missing/empty field {field!r}")
        if it.get("category") and it["category"] not in CATEGORIES:
            problems.append(f"[{iid}] unknown category {it['category']!r}")
        if it.get("keyword") and it.get("text") and it["keyword"] not in it["text"]:
            problems.append(f"[{iid}] keyword {it['keyword']!r} not found in text")
        forms = [it.get("expected", "")] + list(it.get("alternates", []))
        if len(forms) != len(set(forms)):
            problems.append(f"[{iid}] duplicate entries in expected/alternates")
    return problems


def load_transcripts(path: str | Path) -> dict[str, str]:
    """Accepts {id: text} JSON, [{id, text}] JSON, or JSONL of {id, text}."""
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    if raw[0] == "{" or raw[0] == "[":
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return {str(k): (v or "") for k, v in obj.items()}
        return {str(r["id"]): (r.get("text") or "") for r in obj}
    out = {}
    for line in raw.splitlines():
        if line.strip():
            r = json.loads(line)
            out[str(r["id"])] = r.get("text") or ""
    return out


# ---------------------------------------------------------------------------
# §3  Matching + metrics
# ---------------------------------------------------------------------------

def candidate_forms(item: dict) -> list[str]:
    forms, out = [item["expected"]] + list(item.get("alternates", [])), []
    for f in forms:
        if f and f not in out:
            out.append(f)
    return out


def keyword_correct(transcript: str, item: dict) -> tuple[bool, str | None]:
    """(correct, which_form_matched). Exact substring in normalized space, tried with
    spaces and space-free."""
    t_norm = normalize(transcript)
    t_flat = despace(t_norm)
    for form in candidate_forms(item):
        f_norm = normalize(form)
        if f_norm and (f_norm in t_norm or despace(f_norm) in t_flat):
            return True, form
    return False, None


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate on normalized, space-free text (Thai has no word
    boundaries, so CER — not WER — is the convention). Optional: only items carrying
    `expected_text` get a CER column."""
    ref, hyp = despace(normalize(reference)), despace(normalize(hypothesis))
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — the right CI for a proportion at these n, and it never
    runs off the [0, 1] ends the way the normal approximation does."""
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / d, (centre + half) / d)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p over the discordant counts (b: only A right, c: only
    B right). Exact binomial — the normal approximation is unsafe at the small
    discordant counts a few-hundred-item slice produces."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def score_items(items: list[dict], transcripts: dict[str, str]) -> list[dict]:
    rows = []
    for it in items:
        transcript = transcripts.get(it["id"], "")
        correct, matched = keyword_correct(transcript, it)
        row = {
            "id": it["id"],
            "category": it["category"],
            "correct": int(correct),
            "matched_form": matched or "",
            "expected": it["expected"],
            "keyword": it["keyword"],
            "transcript": transcript,
            "text": it["text"],
            "missing_transcript": int(it["id"] not in transcripts),
        }
        if it.get("expected_text"):
            row["sentence_cer"] = round(cer(it["expected_text"], transcript), 4)
        rows.append(row)
    return rows


def metrics(rows: list[dict]) -> dict:
    def bucket(sub):
        n = len(sub)
        ok = sum(r["correct"] for r in sub)
        lo, hi = wilson(ok, n)
        d = {"n": n, "correct": ok, "accuracy": round(ok / n, 4) if n else 0.0,
             "ci95": [round(lo, 4), round(hi, 4)]}
        cers = [r["sentence_cer"] for r in sub if "sentence_cer" in r]
        if cers:
            d["sentence_cer"] = round(sum(cers) / len(cers), 4)
        return d

    out = {"overall": bucket(rows),
           "by_category": {c: bucket([r for r in rows if r["category"] == c])
                           for c in CATEGORIES if any(r["category"] == c for r in rows)}}
    sl = [r for r in rows if r["category"] in PRIMARY_SLICE]
    if sl:
        out["primary_slice"] = bucket(sl)
    out["missing_transcripts"] = sum(r["missing_transcript"] for r in rows)
    return out


def summarize(rows: list[dict], m: dict) -> str:
    lines = ["# Keyword accuracy", ""]
    for cat in CATEGORIES:
        d = m["by_category"].get(cat)
        if d:
            lines.append(f"- {cat:20} {d['correct']:4}/{d['n']:<4} "
                         f"({100 * d['accuracy']:.1f}%)")
    o = m["overall"]
    lines.append(f"- {'OVERALL':20} {o['correct']:4}/{o['n']:<4} "
                 f"({100 * o['accuracy']:.1f}%, 95% CI "
                 f"{100 * o['ci95'][0]:.1f}–{100 * o['ci95'][1]:.1f})")
    if "primary_slice" in m:
        s = m["primary_slice"]
        lines.append(f"- {'slice (n+ls+cs)':20} {s['correct']:4}/{s['n']:<4} "
                     f"({100 * s['accuracy']:.1f}%)")

    missing = [r["id"] for r in rows if r["missing_transcript"]]
    if missing:
        lines.append(f"\n⚠ {len(missing)} item(s) had no transcript (scored wrong): "
                     + ", ".join(missing[:40]) + (" …" if len(missing) > 40 else ""))

    failures = [r for r in rows if not r["correct"]]
    if failures:
        lines.append("\n## Failures (expected → transcript)")
        lines += [f"- [{r['id']}] {r['expected']!r} → {r['transcript']!r}"
                  for r in failures]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# §4  ASR — the only stage with dependencies, and they import lazily
# ---------------------------------------------------------------------------
# ONE ASR by design (see the card): `expected`/`alternates` are authored relative to
# this model's output style. Swapping it invalidates the gold.

def hf_transcriber(model_id: str, language: str | None):
    try:
        import librosa
        import soundfile as sf
        import torch
        from transformers import pipeline
    except ImportError:
        raise SystemExit(
            "needs: pip install torch transformers soundfile librosa\n"
            "or use --cmd to plug in any CLI transcriber.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = pipeline("automatic-speech-recognition", model=model_id, dtype=dtype,
                    device=device, chunk_length_s=30)
    sr = pipe.feature_extractor.sampling_rate
    kwargs = {"task": "transcribe"}
    if language:
        kwargs["language"] = language

    def transcribe(wav: Path) -> str:
        # Feed arrays, not paths: transformers>=5 file loading wants torchcodec/ffmpeg.
        audio, in_sr = sf.read(str(wav), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if in_sr != sr:
            audio = librosa.resample(audio, orig_sr=in_sr, target_sr=sr)
        out = pipe({"raw": audio, "sampling_rate": sr}, generate_kwargs=kwargs)
        return (out.get("text") or "").strip()

    return transcribe


def cmd_transcriber(template: str):
    import subprocess

    def transcribe(wav: Path) -> str:
        res = subprocess.run(template.format(wav=str(wav)), shell=True,
                             capture_output=True, text=True, timeout=600)
        if res.returncode != 0:
            raise RuntimeError(f"ASR command failed on {wav.name}: {res.stderr[:300]}")
        return res.stdout.strip()

    return transcribe


# ---------------------------------------------------------------------------
# §5  CLI
# ---------------------------------------------------------------------------

def _cmd_validate(args):
    items = load_items(args.items)
    problems = validate_items(items)
    by_cat = {c: sum(1 for i in items if i.get("category") == c) for c in CATEGORIES}
    print(f"{len(items)} items: " + ", ".join(f"{c}={n}" for c, n in by_cat.items()))
    for p in problems:
        print("  " + p)
    print("OK — all items valid." if not problems
          else f"{len(problems)} problem(s) found.")
    return 1 if problems else 0


def _cmd_texts(args):
    items = load_items(args.items)
    field = CONTRACTS[args.contract]
    missing = [i["id"] for i in items if not i.get(field)]
    if missing:
        raise SystemExit(f"{len(missing)} items have no {field!r} "
                         f"(first: {missing[0]}) — use --contract raw")
    out = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")
    for it in items:
        out.write(f"{it['id']}\t{it[field]}\n")
    if out is not sys.stdout:
        out.close()
        print(f"Wrote {len(items)} texts ({args.contract} contract) to {args.out}")
    return 0


def _cmd_transcribe(args):
    wavs = sorted(Path(args.audio_dir).glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no .wav files in {args.audio_dir}")
    transcribe = (cmd_transcriber(args.cmd) if args.cmd
                  else hf_transcriber(ASR_MODEL, args.language or None))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # resume: an interrupted run keeps what it already wrote
    transcripts = load_transcripts(out) if (args.resume and out.exists()) else {}
    for i, wav in enumerate(wavs, 1):
        if wav.stem in transcripts:
            continue
        transcripts[wav.stem] = transcribe(wav)
        print(f"  [{i}/{len(wavs)}] {wav.stem}", file=sys.stderr)
        if i % 50 == 0:
            out.write_text(json.dumps(transcripts, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    out.write_text(json.dumps(transcripts, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"Wrote {len(transcripts)} transcripts to {out}")
    return 0


def _cmd_score(args):
    items = load_items(args.items)
    problems = validate_items(items)
    if problems:
        for p in problems:
            print("  " + p, file=sys.stderr)
        raise SystemExit(f"{len(problems)} problem(s) in {args.items}")
    transcripts = load_transcripts(args.transcripts)
    rows = score_items(items, transcripts)
    m = metrics(rows)
    summary = summarize(rows, m)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    if any("sentence_cer" in r for r in rows):
        fieldnames = [f for f in fieldnames if f != "sentence_cer"] + ["sentence_cer"]
    # utf-8-sig so Thai text opens correctly in Excel/Numbers
    with open(out / "results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "summary.md").write_text(summary + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(summary if args.failures else summary.split("\n## Failures")[0])
    print(f"\nWrote {out/'results.csv'}, {out/'summary.md'}, {out/'metrics.json'}")
    return 0


def _cmd_compare(args):
    items = {i["id"]: i for i in load_items(args.items)}
    a_tag, a_path = args.a.split("=", 1) if "=" in args.a else ("A", args.a)
    b_tag, b_path = args.b.split("=", 1) if "=" in args.b else ("B", args.b)
    ta, tb = load_transcripts(a_path), load_transcripts(b_path)
    cats = tuple(args.categories.split(",")) if args.categories else PRIMARY_SLICE
    shared = [i for i in items if i in ta and i in tb and items[i]["category"] in cats]
    print(f"paired on {len(shared)} items over {cats} ({a_tag} vs {b_tag})\n")

    rows = [{"id": i, "category": items[i]["category"],
             "a_ok": keyword_correct(ta[i], items[i])[0],
             "b_ok": keyword_correct(tb[i], items[i])[0]} for i in shared]

    def report(sub, label):
        n = len(sub)
        if not n:
            return None
        a_ok = sum(r["a_ok"] for r in sub)
        b_ok = sum(r["b_ok"] for r in sub)
        a_only = sum(1 for r in sub if r["a_ok"] and not r["b_ok"])
        b_only = sum(1 for r in sub if r["b_ok"] and not r["a_ok"])
        p = mcnemar_exact(a_only, b_only)
        delta = 100 * (b_ok - a_ok) / n
        print(f"{label:22} n={n:4}  {a_tag} {100*a_ok/n:5.1f}%  {b_tag} {100*b_ok/n:5.1f}%"
              f"  Δ{delta:+5.1f}pts  (b_only {b_only} a_only {a_only}, "
              f"McNemar p={p:.4f})")
        return dict(n=n, a_ok=a_ok, b_ok=b_ok, delta_pts=round(delta, 2),
                    a_only=a_only, b_only=b_only, mcnemar_p=round(p, 5))

    out = {"a": a_tag, "b": b_tag, "categories": list(cats),
           "results": {"slice": report(rows, "SLICE (primary)")}}
    for c in cats:
        out["results"][c] = report([r for r in rows if r["category"] == c], "  " + c)
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


def _cmd_selftest(args):
    """Assertions the matcher must satisfy. Two failure modes matter and they pull in
    opposite directions: normalization that erases a real difference (false PASS), and
    normalization that misses an encoding artifact (false FAIL)."""
    checks = []

    def eq(a, b, why):
        checks.append((a == b, why, repr(a), repr(b)))

    def ne(a, b, why):
        checks.append((a != b, why, repr(a), repr(b)))

    # encoding artifacts must be erased
    eq(normalize("กำ"), normalize("กํา"), "ำ vs ํ+า")
    eq(normalize("เเก"), normalize("แก"), "เเ vs แ")
    eq(normalize("ก​ข"), normalize("กข"), "zero-width strip")
    eq(normalize("๑๒๓"), normalize("123"), "Thai digits")
    eq(normalize("ก่ิ"), normalize("กิ่"), "mark order")
    eq(normalize("ต่างๆ"), normalize("ต่างต่าง"), "ๆ expansion")
    eq(normalize("Grab"), normalize("grab"), "ASCII casefold")
    eq(normalize("  ก  ข  "), "ก ข", "whitespace collapse")
    # pronunciation-level differences must SURVIVE
    ne(normalize("ก่า"), normalize("ก้า"), "tone mark is a real difference")
    ne(normalize("กา"), normalize("กะ"), "vowel length is a real difference")
    ne(normalize("พงษ์"), normalize("พงศ์"), "homophone spellings stay distinct")
    # matcher
    it = {"expected": "แกร็บ", "alternates": ["Grab", "แกร๊บ"]}
    checks.append((keyword_correct("เรียกแกร็บไปทำงาน", it)[0], "expected hits",
                   "", ""))
    checks.append((keyword_correct("เรียก แกร็ บ ไปทำงาน", it)[0],
                   "space-insensitive hit", "", ""))
    checks.append((keyword_correct("เรียก grab ไปทำงาน", it)[1] == "Grab",
                   "alternate reported by surface form", "", ""))
    checks.append((not keyword_correct("เรียกแกรบไปทำงาน", it)[0],
                   "one-mark deviation is an ERROR", "", ""))
    checks.append((not keyword_correct("", it)[0], "empty transcript is wrong", "", ""))
    # metrics
    checks.append((abs(cer("กขค", "กขค")) < 1e-9, "cer identical = 0", "", ""))
    checks.append((abs(cer("กขค", "กขง") - 1 / 3) < 1e-9, "cer one sub", "", ""))
    checks.append((mcnemar_exact(0, 0) == 1.0, "mcnemar no discordance", "", ""))
    checks.append((abs(mcnemar_exact(1, 9) - 0.021484375) < 1e-9,
                   "mcnemar exact binomial", "", ""))
    lo, hi = wilson(43, 50)
    checks.append((lo < 0.86 < hi and 0.7 < lo and hi < 0.95, "wilson bracket", "", ""))

    bad = [(w, a, b) for ok, w, a, b in checks if not ok]
    for w, a, b in bad:
        print(f"FAIL {w}: {a} vs {b}")
    print(f"{len(checks) - len(bad)}/{len(checks)} checks passed")
    # Offline on purpose: the assertions above need no data file, so a selftest
    # never reaches for the network.  Validate the items only if they are already here.
    if not bad and (args.items or DEFAULT_ITEMS.exists()):
        return _cmd_validate(args)
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--items", default=None,
                    help=f"local {ITEMS_NAME} (default: fetch {HF_DATASET})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate", help="check the data file, then exit"
                   ).set_defaults(fn=_cmd_validate)
    sub.add_parser("selftest", help="assertions on the normalizer, matcher and stats"
                   ).set_defaults(fn=_cmd_selftest)

    p = sub.add_parser("texts", help="dump {id}<TAB>{text} to feed your TTS")
    p.add_argument("--contract", default="raw", choices=sorted(CONTRACTS),
                   help="which text variant (default: raw — the sentence as written)")
    p.add_argument("--out", default="-")
    p.set_defaults(fn=_cmd_texts)

    p = sub.add_parser("transcribe", help="{id}.wav -> transcripts.json (needs torch)")
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--language", default="th", help="'' to let the model choose")
    p.add_argument("--cmd", help="external transcriber template with {wav}")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(fn=_cmd_transcribe)

    p = sub.add_parser("score", help="transcripts -> results.csv/summary.md/metrics.json")
    p.add_argument("--transcripts", required=True)
    p.add_argument("--out", default="runs/latest")
    p.add_argument("--no-failures", dest="failures", action="store_false",
                   help="do not print the per-item failure list to stdout")
    p.set_defaults(fn=_cmd_score)

    p = sub.add_parser("compare", help="paired A/B: McNemar exact + per-category deltas")
    p.add_argument("--a", required=True, help="[tag=]transcripts.json")
    p.add_argument("--b", required=True, help="[tag=]transcripts.json")
    p.add_argument("--categories", help=f"default: {','.join(PRIMARY_SLICE)}")
    p.add_argument("--out", help="write the comparison as JSON")
    p.set_defaults(fn=_cmd_compare)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:          # `… | head` closed the pipe; not an error
        import os
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(130)
