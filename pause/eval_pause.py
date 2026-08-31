#!/usr/bin/env python3
"""thai-tts-pause-bench — phrase-break placement for Thai TTS, one file.

Thai is written without word spaces, so a TTS that pauses in the wrong place is not
mispronouncing anything — CER and keyword accuracy are blind to it — but a listener
hears `ประเทศไทย มีวัฒน ธรรม` or `…ที่สวยงามและ อาหารอร่อย` immediately. This scores
that axis: detect every silent pause in a render, attribute it to an exact character
position in the text, and classify it.

    class          meaning                                        example
    ok             pause at an allowed break slot                 ที่สวยงาม|และ…
    bad_juncture   word boundary that is NOT an allowed slot      และ|อาหาร, ใน|บริษัท
    intra_word     pause inside a word                            วัฒน|ธรรม, สิบ|เก้า
    artifact       a silence the alignment cannot speak for       (reported, not scored)
    unaligned      the aligner lost lock locally (not scored as an error)

Allowed slots = the frozen gold mask shipped with each item (`gold`, `|`-marked, one
human/judge-arbitrated pass over 210 long sentences) UNION the junctures the input
itself marks: a source space or a punctuation mark. There is NO break-before-conjunction
rule and no never-break-after word list — both were removed on 2026-08-18 and the notes
in §1 say why. The mask sanitizer keeps every mark that sits on a word boundary.

Pipeline, two stages so you re-score without re-aligning:

    # 0. texts to feed your TTS ({id}<TAB>{text}); render {id}.wav at any sample rate
    python3 eval_pause.py texts --out texts.tsv

    # 1. GPU: wav2vec2-CTC char-level forced alignment -> one sidecar JSON per clip
    python3 eval_pause.py align --audio-dir audio/mysystem --out runs/mysystem

    # 2. CPU: silences -> junctures -> classes -> metrics
    python3 eval_pause.py score --sidecars runs/mysystem/sidecars --out runs/mysystem

    python3 eval_pause.py run     --audio-dir audio/mysystem --out runs/mysystem  # 1+2
    python3 eval_pause.py compare --a base=runs/base --b new=runs/new  # paired bootstrap
    python3 eval_pause.py selftest                                     # no data needed

If your system rewrites the text before speaking (number/loanword verbalization), pass
`--spoken-texts` so the aligner sees what was actually said; the gold mask is carried
onto that string across the regions the two share and marks inside rewritten regions
are dropped (conservative by construction).

Dependencies: `score` needs numpy + soundfile; `align` additionally needs torch,
torchaudio and transformers; only `--spoken-texts` needs pythainlp. Every linguistic
grid the scorer runs on ships FROZEN in the data file — `tokens` (newmm words, for mask
sanitization), `word_spans` (tltk words, for allowed slots and word identity),
`syllables` + `juncture_phones` (the tltk syllable grid a juncture is snapped onto and
the phones that meet there) and `dead_spans` — so scores do not drift when a tokenizer
or a G2P dictionary is updated, and the harness itself needs neither tltk nor pythainlp.

Thresholds, break rules and the aligner choice are calibrated constants — see the
dataset card before changing any of them.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# The item set is published as a Hugging Face dataset rather than vendored here, so
# there is one copy to correct and no chance of a checkout drifting from the gold.
HF_DATASET = "wayu-ai/thai-tts-pause-bench"
ITEMS_NAME = "pause_set.jsonl"
HF_FILE = f"data/{ITEMS_NAME}"          # the dataset keeps it under data/
DEFAULT_ITEMS = HERE / "data" / ITEMS_NAME

# The Thai char-level CTC recognizer used for forced alignment. LM-free on purpose: it
# times what it hears. Benchmarked against exact decoder frame spans on ~1,500 clips —
# onset error p50 13.9 ms / p90 36.3 ms, per-speaker pause-precision r = 0.984, 88.8%
# clip-verdict agreement — which is what makes it usable for RANKING systems and not
# merely for aggregate rates. Two alternatives were measured; the card has the table.
# Swapping it changes every number, so it is fixed and there is no flag to change it.
CTC_MODEL = "airesearch/wav2vec2-large-xlsr-53-th"

# Bumped whenever a sidecar gains a field the scorer needs. `score` refuses a sidecar
# written before the current schema rather than silently falling back to a coarser
# grid and reporting numbers that are not comparable with anything published.
SIDECAR_SCHEMA = 2


# ---------------------------------------------------------------------------
# §1  Thresholds and break rules — the calibrated constants, do not tune per system
# ---------------------------------------------------------------------------

# -35 dBFS floor + 75 ms minimum, from the probe that separated known-good from
# known-bad clips by ear. A silence at a juncture where a voiceless obstruent meets is
# usually that obstruent's own closure, not a pause (ประเทศ|ไทย shows 60-90 ms of closure
# in clips humans call clean), so it must clear the allowance `closure_ms` gives that
# context. Glottal ʔ is deliberately NOT guarded: Thai vowel-initial words get
# ʔ-epenthesis and a silence there (และ|อาหาร) is prosodic, not articulatory.
#
# Each allowance is the 99.5th percentile of the within-word gap distribution measured
# for that context over ~7,200 junctures pooled across three systems.
THRESHOLDS = dict(silence_db=-35.0, min_pause_ms=75.0,
                  stop_min_pause_ms=125.0, stop2_min_pause_ms=137.5,
                  fric_min_pause_ms=100.0)

# ⚠ silence_db is an ABSOLUTE level. Renders mastered much quieter than -35 dBFS peak
# will show phantom pauses; `score` prints the clip-peak distribution so that is visible
# rather than silent. Keep the default to stay comparable with the published numbers.

# The phones the closure guard is about, in both alphabets: base IPA for the frozen
# `juncture_phones` grid, Thai script for the letter proxy that stands in when a
# juncture has no phones (a digit run or a Latin word on one side). Aspirated stops are
# the base char plus a modifier (pʰ), so the base set covers them.
_STOP_IPA = set("ptkcbd")
_FRIC_IPA = set("sfhʃθ")
_STOP_THAI = set("กขคฆจฉชฌฎฏฐฑฒดตถทธบปผพภ")
_FRIC_THAI = set("สศษซฟฝหฮ")
# Thai writes these vowels BEFORE their syllable's onset consonant, so the first char of
# a syllable is not always the char carrying its onset phone (เกิด is /k…/, ไทย is /tʰ…/).
_PREPOSED_VOWELS = set("เแโใไ")
# a leading ห before one of these is a TONE marker, not /h/: หนังสือ opens on /n/. Without
# this the proxy would hand every ห-cluster a fricative's allowance it cannot justify.
_SONORANT_THAI = set("งญนมยรลว")


def closure_ms(coda_stop: bool, onset_stop: bool, onset_fric: bool = False, *,
               one_ms: float = THRESHOLDS["stop_min_pause_ms"],
               both_ms: float = THRESHOLDS["stop2_min_pause_ms"],
               fric_ms: float = THRESHOLDS["fric_min_pause_ms"]) -> float:
    """Longest silence at this juncture still attributable to articulation (ms).

    Is the phone CLOSING the left syllable an oral stop, and what OPENS the right one?
    Each voiceless obstruent contributes its own low-energy interval, so two stops get
    the longest allowance and a bare fricative onset the shortest.

    ⚠ Do NOT collapse this back to a single cutoff over a ±3-char context window. That
    is what this file did before 2026-08-25: it asked "is there any stop char within 3
    either side", which fires at 77-89% of junctures, so the guard degenerated into a
    flat "< 120 ms is not a pause" and the phone-identity half of it did no work. With
    the juncture snapped to a syllable boundary the coda and onset are exact, so the
    context is exact too."""
    if coda_stop and onset_stop:
        return both_ms
    if coda_stop or onset_stop:
        return one_ms
    return fric_ms if onset_fric else 0.0


# a within-unit silence reaching this close to a token edge belongs to the adjacent
# juncture (phrase-final silence is booked into the last token)
TAIL_TOL_MS = 60.0

# one CTC char span this long that fully swallows a silence marks a local alignment
# failure (real Thai chars run far shorter) -> unaligned
LONG_CHAR_SPAN_MS = 500.0

# ⚠ REMOVED 2026-08-18 — there is no "break-before word" rule, do not add one.
# Earlier versions allowed a break before และ หรือ แต่ ที่ ซึ่ง … at every occurrence.
# It was never calibrated and it moved the cross-system ranking on its own: worth
# +0.076 pause precision to one system and +0.001 to another on the same clips. With a
# gold mask present its marginal value is +0.006, because the mask already carries that
# judgement per sentence. Allowed slots = source spaces + punctuation ∪ mask marks.

# ⚠ REMOVED 2026-08-18 — do NOT re-add a "no break AFTER this word" list either.
# A NO_BREAK_AFTER_WORDS set deleted any mask mark sitting right after และ ที่ ของ ว่า …
# on the rationale that mask authors bracket conjunctions. On the 210 masked sentences
# it deleted 1.10 marks/clip and only 20.9% of those were actually bracketed; its top
# trigger was ว่า, a complementizer where a break before a long complement clause is
# ordinary Thai prosody. The mask is the judgement; a word list that overrides it is
# not evidence. Sanitization now keeps every mark that sits on a word boundary.

# ---------------------------------------------------------------------------
# Artifact rejection: three conditions make a detected silence unattributable to
# phrasing. Such silences are REPORTED separately (`artifacts_per_clip`,
# `artifact_reasons`) rather than counted as placement errors — and rather than
# silently dropped, because a rate that climbs here is an aligner or a voice problem,
# not phrasing. Without them the measured intra-word rate is an upper bound, about a
# point high on the two reference systems.
#
#   repeat        a word occurring twice in the audio but once in the text: the aligner
#                 must place the single copy across both utterances, so the gap between
#                 them looks word-internal
#   impossible    the "internal" pause is longer than the word span containing it
#                 (ข้อจำกัด: 437 ms inside a 400 ms span) — attribution snapped it
#   unverbalized  the word abuts a digit/Latin run whose spoken Thai form the char
#                 aligner has no vocabulary for, so the word's span absorbs that audio
#
# A fourth case, a tokenizer MERGE (คนใน = คน+ใน), is not a guard: it is fixed by
# refining the word grid with the second tokenizer — see `_refine_word_spans`.

# a word may run this many syllables longer than the clip's own syllable rate predicts
# before the extra span is read as a repeated utterance
REPEAT_SYL_SLACK = 1.0
# a pause covering more than this fraction of its word's span cannot be inside it
IMPOSSIBLE_FRAC = 0.9

PAUSE_OK, BAD_JUNCTURE, INTRA_WORD, ARTIFACT, UNALIGNED = (
    "ok", "bad_juncture", "intra_word", "artifact", "unaligned")
BAD_CLASSES = {BAD_JUNCTURE, INTRA_WORD}
MARK_ALLOWED, MARK_EXPECTED = "|", "‖"


# ---------------------------------------------------------------------------
# §2  Silence detection
# ---------------------------------------------------------------------------

@dataclass
class Silence:
    t0_ms: float
    t1_ms: float
    dur_ms: float
    mean_db: float


def frame_db(wav: np.ndarray, sr: int, *, hop: int | None = None,
             win: int | None = None):
    """Per-frame dBFS on a 12.5 ms hop, plus the frame step in ms."""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(1)
    hop = hop or max(1, round(sr * 0.0125))
    win = win or 4 * hop
    n = len(wav) // hop
    if n < 3:
        return np.zeros(0), 1000.0 * hop / sr
    pad = (win - hop) // 2
    x = np.pad(wav, (pad, pad), mode="reflect")
    cs = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    starts = np.arange(n) * hop
    rms = np.sqrt((cs[starts + win] - cs[starts]) / win)
    return 20.0 * np.log10(rms + 1e-9), 1000.0 * hop / sr


def detect_silences(wav: np.ndarray, sr: int, *,
                    silence_db: float = THRESHOLDS["silence_db"],
                    min_ms: float = THRESHOLDS["min_pause_ms"]) -> list[Silence]:
    """Internal silences >= min_ms below silence_db. Leading/trailing silence is
    dropped — a clip that starts late has not paused."""
    db, frame_ms = frame_db(wav, sr)
    n = len(db)
    if n < 3:
        return []
    silent = db < silence_db
    min_frames = max(1, int(round(min_ms / frame_ms)))
    out, i = [], 0
    while i < n:
        if not silent[i]:
            i += 1
            continue
        j = i
        while j < n and silent[j]:
            j += 1
        if (j - i) >= min_frames and i > 0 and j < n:      # drop edge runs
            out.append(Silence(t0_ms=i * frame_ms, t1_ms=j * frame_ms,
                               dur_ms=(j - i) * frame_ms,
                               mean_db=float(db[i:j].mean())))
        i = j
    return out


# ---------------------------------------------------------------------------
# §3  Gold masks and allowed slots
# ---------------------------------------------------------------------------

def parse_gold(mask: str) -> tuple[str, set, set]:
    """'ประเทศไทย|มี…' -> (text, allowed_offsets, expected_offsets)."""
    text, allowed, expected = [], set(), set()
    for ch in mask:
        if ch == MARK_ALLOWED:
            allowed.add(len(text))
        elif ch == MARK_EXPECTED:
            allowed.add(len(text))
            expected.add(len(text))
        else:
            text.append(ch)
    return "".join(text), allowed, expected


def sanitize_gold(text: str, gold: str, tokens: list[str]) -> str | None:
    """Validate a mask against `text` (characters must match exactly, else None) and
    keep only marks that sit strictly inside the text, ON a word boundary, and NOT
    right after a rightward-attaching function word. Conservative on purpose: a mask
    author's slip can only ever remove allowed slots, never invent them.

    `tokens` is the frozen tokenization shipped with the item (whitespace kept, joins
    back to `text`)."""
    stripped, marks = [], []
    for ch in gold:
        if ch in (MARK_ALLOWED, MARK_EXPECTED):
            marks.append((len(stripped), ch))
        else:
            stripped.append(ch)
    if "".join(stripped) != text:
        return None
    bounds, ends_with, pos = set(), {}, 0
    for w in tokens:
        pos += len(w)
        bounds.add(pos)
        ends_with[pos] = w.strip()
    # A mask marks the juncture BEFORE a space ("…เลย| แต่…") but a pause is attributed
    # to the word start AFTER it, so an un-normalized mark is a slot no system can hit
    # (41% of the marks shipped in this dataset).
    def _skip_ws(o):
        while o < len(text) and text[o].isspace():
            o += 1
        return o
    kept = [(_skip_ws(o), m) for o, m in marks
            if 0 < o < len(text) and o in bounds]
    kept = sorted({(o, m) for o, m in kept if 0 < o < len(text)})
    out, mi = [], 0
    for i, ch in enumerate(text):
        while mi < len(kept) and kept[mi][0] == i:
            out.append(kept[mi][1])
            mi += 1
        out.append(ch)
    return "".join(out)


def remap_offsets(src: str, dst: str, offsets: set) -> set:
    """Carry between-char mark offsets from src to dst across the regions the two
    strings share (difflib equal blocks). Marks inside rewritten regions (number
    verbalization, ๆ expansion, loanword respelling) are dropped."""
    import difflib
    blocks = difflib.SequenceMatcher(None, src, dst,
                                     autojunk=False).get_matching_blocks()
    out = set()
    for o in offsets:
        for a, b, n in blocks:
            if a <= o <= a + n:
                d = b + (o - a)
                if 0 < d < len(dst):
                    out.add(d)
                break
    return out


def word_spans(tokens: list[str]) -> list:
    """(a, b, surface) for the non-blank tokens; whitespace survives only as the gap
    between spans, which is what the source-space rule reads."""
    spans, pos = [], 0
    for w in tokens:
        if w.strip():
            spans.append((pos, pos + len(w), w))
        pos += len(w)
    return spans


def frozen_word_spans(text: str, pairs: list) -> list:
    """(a, b, surface) from the shipped `word_spans` offsets — the tltk word grid.

    Two tokenizers ship with each item and they do different jobs. This one decides
    which junctures exist and which word a pause lands in; `tokens` (newmm) sanitizes
    the mask and, via `_refine_word_spans`, vetoes intra-word verdicts. Segmenting the
    grid the way the reference scorer does is not cosmetic — a juncture only one
    tokenizer reports can never be named identically by both, which is how measurements
    that agreed on pause COUNTS still disagreed about which juncture was at fault."""
    return [(a, b, text[a:b]) for a, b in pairs]


def _refine_word_spans(spans: list, text: str, newmm_starts: set) -> list:
    """Split each tltk word at any interior boundary newmm also reports.

    A juncture is only INTRA_WORD when BOTH tokenizers agree it is inside a word. tltk
    glues pairs newmm keeps apart — คนใน (คน+ใน), ว่าการ (ว่า+การ) — and a break at those
    is ordinary Thai phrasing, so calling it the most severe error class was a tokenizer
    accident. Refining can only ADD word starts inside a tltk word, and `allowed_from_spans`
    grants a slot only after a space or punctuation, so the allowed set is untouched: the
    juncture becomes a normal word boundary, scored allowed or bad on its merits."""
    if not newmm_starts:
        return spans
    out = []
    for a, b, _ in spans:
        cuts = sorted(o for o in newmm_starts if a < o < b)
        for lo, hi in zip([a] + cuts, cuts + [b]):
            out.append((lo, hi, text[lo:hi]))
    return out


def allowed_from_spans(text: str, spans: list) -> set:
    """Text-derived slots: punctuation positions (both sides) and any word start preceded
    by a source space — i.e. only the junctures the input itself marks. No
    break-before-conjunction rule; see the removal note in §1."""
    allowed = set()
    for i, (a, b, surf) in enumerate(spans):
        if not any("฀" <= c <= "๿" or c.isalnum() for c in surf):
            allowed.add(a)                                   # punctuation token
            if i + 1 < len(spans):
                allowed.add(spans[i + 1][0])
            continue
        if i > 0 and " " in text[spans[i - 1][1]:a]:
            allowed.add(a)
    return allowed


def tokenize_newmm(text: str) -> list[str]:
    """pythainlp newmm, whitespace kept. Only needed for --spoken-texts: the benchmark
    items ship their tokenization frozen, so the default path has no such dependency
    and no version drift."""
    try:
        from pythainlp.tokenize import word_tokenize
    except ImportError:
        raise SystemExit("--spoken-texts needs pythainlp (pip install pythainlp)")
    toks = word_tokenize(text, engine="newmm", keep_whitespace=True)
    if "".join(toks) != text:
        raise ValueError("newmm tokens do not tile the text")
    return toks


# ---------------------------------------------------------------------------
# §4  Attribution + classification
# ---------------------------------------------------------------------------

@dataclass
class Pause:
    t0_ms: float
    t1_ms: float
    dur_ms: float
    ch_offset: int          # juncture position in the text (-1 for intra/unaligned)
    cls: str
    before: str
    after: str
    reason: str = ""        # why an ARTIFACT is unscoreable ("" for every other class)


def _syllable_rate_ms(syl_starts: list, ch_start: dict) -> float:
    """The clip's own ms-per-syllable, as the median onset-to-onset interval.

    Onset-to-onset is the only interval CTC spans measure reliably — span WIDTH is a
    blank-allocation artifact — and taking it from the clip itself means the artifact
    guard needs no reference system and no assumption about speaking rate."""
    on = [ch_start[o] for o in syl_starts if o in ch_start]
    gaps = [b - a for a, b in zip(on, on[1:]) if b > a]
    return float(np.median(gaps)) if gaps else 0.0


def _artifact_reason(*, word_span_ms: float, pause_ms: float, n_syl: int,
                     syl_ms: float, over_unverbalized: bool) -> str:
    """Why this silence cannot be scored as a pause inside its word ('' if it can)."""
    if over_unverbalized:
        return "unverbalized"
    if word_span_ms > 0 and pause_ms > IMPOSSIBLE_FRAC * word_span_ms:
        return "impossible"
    if syl_ms > 0 and n_syl > 0:
        if word_span_ms > (n_syl + REPEAT_SYL_SLACK) * syl_ms + pause_ms:
            return "repeat"
    return ""


def _snap_to_syllable(ca: int, syl_starts: list, ch_mid: dict, sil: Silence) -> int:
    """Move a CTC-derived juncture onto the nearest SYLLABLE boundary.

    Thai has no juncture inside a syllable: a break cannot fall after an onset consonant,
    after a preposed vowel, or before a final consonant. Every such split the char path
    reports is alignment noise — CTC has no acoustic evidence for a preposed vowel (เ is
    written before the consonant it follows phonetically) or for a weak final nasal, so it
    parks that char on the far side of a real phrase pause. That is how 400-700 ms breaks
    ended up "inside" 3-char words: เ|พราะ, อ|ยู่, เน้|น, ระหว่า|ง — 56% of one system's
    intra-word flags sat exactly one char off a word boundary.

    The distance is measured in CHARS, not time: the misplaced char is one or two
    positions away but can be hundreds of ms away, so time only breaks ties (the boundary
    whose own emission sits nearer the silence wins)."""
    if not syl_starts:
        return ca
    k = bisect.bisect_left(syl_starts, ca)
    if k < len(syl_starts) and syl_starts[k] == ca:
        return ca                                       # already legal
    cands = [b for b in (syl_starts[k - 1] if k else None,
                         syl_starts[k] if k < len(syl_starts) else None)
             if b is not None]
    if not cands:
        return ca

    def cost(b):
        t = ch_mid.get(b)
        gap = 0.0 if t is None else max(sil.t0_ms - t, t - sil.t1_ms, 0.0)
        return abs(b - ca), gap

    return min(cands, key=cost)


def _juncture_context(text: str, ca: int, syl_starts: list) -> tuple:
    """(coda_stop, onset_stop, onset_fricative) read from the LETTERS at a juncture.

    The fallback for a juncture `juncture_phones` cannot speak for — one side is a digit
    run or a Latin word, whose spoken Thai form the char aligner never saw — and the whole
    grid for a `--spoken-texts` string that ships no frozen phones.

    The coda is the last letter to the left (Thai writes the final consonant last, under
    any tone mark; a space or comma at the juncture carries no phone). The onset is the
    first char of the right-hand syllable that actually carries the onset phone: past a
    preposed vowel (เกิด -> ก, ไทย -> ท) and past a tone-marking leading ห (หนังสือ -> น).

    Letters name the wrong phone often enough that this is a proxy, not a substitute:
    Thai spells a final ส ศ ษ ซ as /t̚/ and a final ฟ as /p̚/ (รส -> rot, กราฟ -> kraːp)
    and kills whole clusters under ์ (ทรัพย์ -> sap). That is why the phones are frozen
    into the data file and this runs only where they are missing."""
    i = ca - 1
    while i >= 0 and not text[i].isalnum():          # tone marks, spaces, punctuation
        i -= 1
    coda = text[i] if i >= 0 else ""
    k = bisect.bisect_right(syl_starts, ca)
    end = syl_starts[k] if k < len(syl_starts) else len(text)
    j = ca
    while j < end and (text[j] in _PREPOSED_VOWELS
                       or unicodedata.category(text[j]) == "Mn"):
        j += 1
    if j < end and text[j] == "ห":
        k = j + 1
        while k < end and unicodedata.category(text[k]) == "Mn":
            k += 1
        if k < end and text[k] in _SONORANT_THAI:
            j = k
    onset = text[j] if j < end else ""
    return coda in _STOP_THAI, onset in _STOP_THAI, onset in _FRIC_THAI


def attribute_pauses(silences: list[Silence], char_spans: list, text: str,
                     spans: list, allowed: set, *,
                     syl_starts: list = (), syl_ctx: dict | None = None,
                     dead_spans: list = (), newmm_starts: set | None = None,
                     stop_min_ms: float = THRESHOLDS["stop_min_pause_ms"],
                     stop2_min_ms: float = THRESHOLDS["stop2_min_pause_ms"],
                     fric_min_ms: float = THRESHOLDS["fric_min_pause_ms"],
                     ) -> list[Pause]:
    """Silence times -> the juncture they belong to -> a class.

    char_spans = [(char_idx, t0_ms, t1_ms)] from forced alignment; spans = tltk word
    spans; syl_starts / syl_ctx / dead_spans are the frozen syllable grid, the phones
    meeting at each of its offsets, and the char spans nothing could phonemize.

    CTC emissions are peaky and jittery: char spans routinely OVERLAP a real silence,
    so requiring a silence to fall cleanly between two spans marks most pauses
    unattributable. Attribution is therefore nearest-boundary — split the time-monotonic
    spans at the silence midpoint — then snapped to a word boundary when one lies within
    the jitter tolerance, and finally onto the SYLLABLE grid, which is what makes the
    result phonotactically possible at all. `unaligned` is reserved for a silence
    swallowed whole by one implausibly long char span."""
    if not char_spans or not spans:
        return [Pause(s.t0_ms, s.t1_ms, s.dur_ms, -1, UNALIGNED, "", "")
                for s in silences]
    mids = [(c[1] + c[2]) / 2.0 for c in char_spans]
    # INTRA_WORD requires both tokenizers to agree the juncture is word-internal; the
    # allowed set is read off the unrefined grid, which refining cannot touch.
    spans = _refine_word_spans(spans, text, newmm_starts or set())
    starts = [a for a, _, _ in spans]
    ends = [b for _, b, _ in spans]
    word_starts = set(starts)
    syl_ctx = syl_ctx or {}
    # Every tltk word start is a syllable start, so the grid only ADDS legal junctures.
    # With no grid shipped (a --spoken-texts string) the word boundaries stand in.
    syl_starts = list(syl_starts) or sorted(word_starts | {spans[-1][1]})
    ch_mid = {c[0]: (c[1] + c[2]) / 2.0 for c in char_spans}
    ch_start = {c[0]: c[1] for c in char_spans}
    ch_end = {c[0]: c[2] for c in char_spans}
    syl_ms = _syllable_rate_ms(syl_starts, ch_start)
    # time hull of each unmeasurable token, from whatever the aligner emitted around it
    dead_ms = []
    for a, b in dead_spans:
        ts = [(c[1], c[2]) for c in char_spans if a <= c[0] < b]
        if len(ts) > 1:
            dead_ms.append((min(t[0] for t in ts), max(t[1] for t in ts)))

    def word_metrics(a: int, b: int):
        """(aligned span ms, syllables, does the span cover unverbalized audio)."""
        t0 = min((ch_start[k] for k in range(a, b) if k in ch_start), default=None)
        t1 = max((ch_end[k] for k in range(a, b) if k in ch_end), default=None)
        n_syl = sum(1 for o in syl_starts if a <= o < b) or 1
        if t0 is None or t1 is None:
            return 0.0, n_syl, False
        return t1 - t0, n_syl, any(d1 > t0 and d0 < t1 for d0, d1 in dead_ms)

    out = []
    for s in silences:
        # juncture j sits between char_spans[j-1] and char_spans[j]; valid j in [1,n-1]
        k0 = bisect.bisect_right(mids, (s.t0_ms + s.t1_ms) / 2.0)
        if k0 <= 0 or k0 >= len(char_spans):
            continue            # edge silence: onset lag / final fade, not a pause
        if any(c[1] <= s.t0_ms and c[2] >= s.t1_ms
               and (c[2] - c[1]) > LONG_CHAR_SPAN_MS
               for c in (char_spans[k0 - 1], char_spans[k0])):
            out.append(Pause(s.t0_ms, s.t1_ms, s.dur_ms, -1, UNALIGNED, "", ""))
            continue
        # jitter-ambiguous range around k0: a char emitted inside (or within
        # TAIL_TOL_MS of) the silence could sit on either side of it
        lo = k0
        while lo - 1 >= 1 and char_spans[lo - 1][1] >= s.t0_ms - TAIL_TOL_MS:
            lo -= 1
        hi = k0
        while hi + 1 <= len(char_spans) - 1 and char_spans[hi][2] <= s.t1_ms + TAIL_TOL_MS:
            hi += 1
        j = min((jj for jj in range(lo, hi + 1)
                 if char_spans[jj][0] in word_starts),
                key=lambda jj: abs(jj - k0), default=k0)
        # the snap makes the juncture phonotactically possible; without it a break is
        # routinely reported one char inside a syllable, which reads as INTRA_WORD
        ca = _snap_to_syllable(char_spans[j][0], syl_starts, ch_mid, s)
        if ca <= 0 or ca >= len(text):
            continue            # beyond the first/last syllable: an edge, not a pause
        mid = (s.t0_ms + s.t1_ms) / 2.0
        if any(t0 < mid < t1 for t0, t1 in dead_ms):
            continue            # silence inside a spelled-out number or a Latin word
        coda, onset = syl_ctx.get(ca, (None, None))
        if coda is None or onset is None:
            # one side is a digit run or a Latin word: the audio realizes it as
            # verbalized Thai the char aligner has no vocabulary for, so there are no
            # phones to read here and the letters stand in
            ctx = _juncture_context(text, ca, syl_starts)
        else:
            ctx = (coda in _STOP_IPA, onset in _STOP_IPA, onset in _FRIC_IPA)
        if s.dur_ms <= closure_ms(*ctx, one_ms=stop_min_ms, both_ms=stop2_min_ms,
                                  fric_ms=fric_min_ms):
            continue            # the obstruent's own closure, not a pause
        cb = ca - 1
        wb = bisect.bisect_right(starts, cb) - 1
        wa = bisect.bisect_right(ends, ca)
        if wb < 0 or wa >= len(spans):
            continue            # beyond the first/last word: edge, not a pause
        if wb == wa:
            a, b, surf = spans[wb]
            span_ms, n_syl, over = word_metrics(a, b)
            why = _artifact_reason(word_span_ms=span_ms, pause_ms=s.dur_ms,
                                   n_syl=n_syl, syl_ms=syl_ms,
                                   over_unverbalized=over)
            if why:
                out.append(Pause(s.t0_ms, s.t1_ms, s.dur_ms, -1, ARTIFACT,
                                 surf, surf, why))
                continue
            cls, off = INTRA_WORD, -1
            sb = sa = surf
        else:
            off = spans[wa][0]
            cls = PAUSE_OK if off in allowed else BAD_JUNCTURE
            sb, sa = spans[wb][2], spans[wa][2]
        out.append(Pause(s.t0_ms, s.t1_ms, s.dur_ms, off, cls, sb, sa))
    return out


@dataclass
class ClipReport:
    system: str
    text_id: str
    wav_path: str
    audio_s: float
    peak_db: float
    pauses: list = field(default_factory=list)
    n_ok: int = 0
    n_bad_junc: int = 0
    n_intra: int = 0
    n_unaligned: int = 0
    n_artifact: int = 0     # silences disqualified by an alignment artifact
    verdict: str = "clean"
    n_cue: int = 0          # allowed slots this clip offers — the recall denominator
    n_cue_hit: int = 0      # of those, how many were realized as a pause
    expected_hit: int = 0
    expected_total: int = 0

    @property
    def n_scored(self) -> int:
        """Pauses the placement metrics are computed over — artifacts excluded."""
        return len(self.pauses) - self.n_artifact

    def finalize(self):
        self.n_ok = sum(1 for p in self.pauses if p.cls == PAUSE_OK)
        self.n_bad_junc = sum(1 for p in self.pauses if p.cls == BAD_JUNCTURE)
        self.n_intra = sum(1 for p in self.pauses if p.cls == INTRA_WORD)
        self.n_unaligned = sum(1 for p in self.pauses if p.cls == UNALIGNED)
        self.n_artifact = sum(1 for p in self.pauses if p.cls == ARTIFACT)
        self.verdict = "bad" if any(p.cls in BAD_CLASSES for p in self.pauses) \
            else "clean"
        return self


def score_clip(sidecar: dict, wav: np.ndarray, sr: int, *,
               silence_db: float, min_pause_ms: float, stop_min_ms: float,
               stop2_min_ms: float, fric_min_ms: float) -> ClipReport:
    text = sidecar["text"]
    tokens = sidecar["tokens"]
    # newmm words sanitize the mask and veto intra-word verdicts; the tltk grid decides
    # which junctures exist. A --spoken-texts item ships no tltk grid, so newmm stands in.
    newmm_spans = word_spans(tokens)
    spans = (frozen_word_spans(text, sidecar["word_spans"])
             if sidecar.get("word_spans") else newmm_spans)
    gold = sidecar.get("gold")
    if gold:
        gold = sanitize_gold(text, gold, tokens)
    allowed = allowed_from_spans(text, spans)
    expected: set = set()
    if gold:
        _, gold_allowed, expected = parse_gold(gold)
        allowed = allowed | gold_allowed           # mask ∪ text-derived slots
    # `space_realization` is P(pause | allowed slot) over the SAME union: the mask marks
    # positions a Thai reader may breathe at, and a system that realizes none of them is
    # reading 150 characters in one breath. Reporting it against the text-derived slots
    # alone would make it a punctuation-following score, not a phrasing one.
    cues = allowed
    sils = detect_silences(wav, sr, silence_db=silence_db, min_ms=min_pause_ms)
    pauses = attribute_pauses(
        sils, [tuple(c) for c in sidecar["char_spans"]], text, spans, allowed,
        syl_starts=sidecar.get("syllables") or (),
        syl_ctx={int(k): tuple(v) for k, v in (sidecar.get("juncture_phones") or {}).items()},
        dead_spans=sidecar.get("dead_spans") or (),
        newmm_starts={a for a, _, _ in newmm_spans},
        stop_min_ms=stop_min_ms, stop2_min_ms=stop2_min_ms, fric_min_ms=fric_min_ms)
    realized = {p.ch_offset for p in pauses if p.cls == PAUSE_OK}
    peak = 20.0 * np.log10(float(np.max(np.abs(wav))) + 1e-9) if len(wav) else -99.0
    rep = ClipReport(system=sidecar.get("system", "external"),
                     text_id=str(sidecar["id"]), wav_path=sidecar["wav_path"],
                     audio_s=len(wav) / sr, peak_db=round(peak, 1), pauses=pauses,
                     n_cue=len(cues), n_cue_hit=len(cues & realized),
                     expected_hit=len(expected & realized),
                     expected_total=len(expected))
    return rep.finalize()


# ---------------------------------------------------------------------------
# §5  Aggregation
# ---------------------------------------------------------------------------

def aggregate(reports: list[ClipReport], *, boot: int = 2000, seed: int = 1234) -> dict:
    n = len(reports)
    if not n:
        return {}
    minutes = sum(r.audio_s for r in reports) / 60.0
    # artifacts are neither placement errors nor pauses: they leave every denominator
    n_pauses = sum(r.n_scored for r in reports)
    n_art = sum(r.n_artifact for r in reports)
    n_ok = sum(r.n_ok for r in reports)
    n_cue = sum(r.n_cue for r in reports)
    exp_tot = sum(r.expected_total for r in reports)
    d = {
        "clips": n,
        # share of clips carrying at least one misplaced pause
        "pper": round(sum(1 for r in reports if r.verdict == "bad") / n, 4),
        "intra_word_rate": round(
            sum(1 for r in reports if r.n_intra) / n, 4),
        "bad_pauses_per_min": round(
            sum(r.n_bad_junc + r.n_intra for r in reports) / minutes, 3
        ) if minutes else 0.0,
        "pauses_per_clip": round(n_pauses / n, 3),
        "median_peak_db": round(float(np.median([r.peak_db for r in reports])), 1),
    }
    if n_art:
        # Reported, not hidden: a rate climbing here is an aligner or a voice problem
        # (stutters, unverbalized digits), not phrasing.
        from collections import Counter
        d["artifacts_per_clip"] = round(n_art / n, 3)
        d["artifact_reasons"] = dict(sorted(Counter(
            p.reason for r in reports for p in r.pauses
            if p.cls == ARTIFACT).items()))
    if n_cue:
        # P(pause | allowed slot): the recall-side axis precision alone cannot give us.
        # A precision gain that arrives with THIS falling is the model pausing less, not
        # phrasing better — always read the two together.
        d["space_realization"] = round(sum(r.n_cue_hit for r in reports) / n_cue, 4)
        d["cues_per_clip"] = round(n_cue / n, 3)
    if n_pauses:
        d["unaligned_rate"] = round(
            sum(r.n_unaligned for r in reports) / n_pauses, 4)
        d["pause_precision"] = round(n_ok / n_pauses, 4)
        if exp_tot:
            p, rec = n_ok / n_pauses, sum(r.expected_hit for r in reports) / exp_tot
            d["break_f05"] = round(
                (1.25 * p * rec / (0.25 * p + rec)) if (p + rec) else 0.0, 4)
    if boot and n > 1:
        # resample CLIPS, the unit of independence — a clip's pauses are not
        # exchangeable with each other
        rng = np.random.default_rng(seed)
        M = np.array([[r.n_scored, r.n_ok, float(r.verdict == "bad")]
                      for r in reports], dtype=np.float64)
        idx = rng.integers(0, n, size=(boot, n))
        prec = np.array([M[i, 1].sum() / max(M[i, 0].sum(), 1e-9) for i in idx])
        pper = np.array([M[i, 2].mean() for i in idx])
        d["pause_precision_ci95"] = [round(float(x), 4)
                                     for x in np.percentile(prec, [2.5, 97.5])]
        d["pper_ci95"] = [round(float(x), 4)
                          for x in np.percentile(pper, [2.5, 97.5])]
    return d


def write_reports(reports: list[ClipReport], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["system", "text_id", "verdict", "n_pauses", "n_ok", "n_bad_junc",
                    "n_intra", "n_unaligned", "n_artifact", "n_cue", "n_cue_hit",
                    "audio_s", "peak_db", "detail", "wav"])
        for r in sorted(reports, key=lambda r: (r.system, r.text_id)):
            detail = "; ".join(
                f"{p.cls}{':' + p.reason if p.reason else ''}"
                f"@{p.ch_offset}({p.before}/{p.after},{p.dur_ms:.0f}ms)"
                for p in r.pauses if p.cls != PAUSE_OK)
            # n_pauses is the SCORED count (artifacts excluded), matching the metrics
            w.writerow([r.system, r.text_id, r.verdict, r.n_scored, r.n_ok,
                        r.n_bad_junc, r.n_intra, r.n_unaligned, r.n_artifact,
                        r.n_cue, r.n_cue_hit,
                        f"{r.audio_s:.2f}", f"{r.peak_db:.1f}", detail, r.wav_path])
    summary = {"overall": aggregate(reports)}
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# §6  Paired comparison
# ---------------------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar over the discordant clip counts. Exact rather than
    chi-square: the discordant counts on 210 clips are small."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))


def paired_bootstrap(pairs: list, n_boot: int = 4000, seed: int = 1234) -> dict:
    """95% CIs on the paired A->B deltas, resampling CLIPS so the unit of resampling
    is the unit of pairing.

    This is the test with real power. Clip-verdict McNemar binarises a clip on a single
    bad pause, which throws magnitude away — it can return p=0.42 for a precision gap
    every aggregate agrees on. Rank on these intervals; read McNemar as the secondary
    "how many individual clips changed verdict"."""
    def mat(k):
        return np.array([[p[k].n_scored, p[k].n_ok, p[k].n_cue, p[k].n_cue_hit,
                          float(p[k].n_intra > 0)] for p in pairs], dtype=np.float64)

    A, B = mat(0), mat(1)

    def metrics(M):
        npause, nok, ncue, nhit, intra = M.sum(0)
        return np.array([nok / max(npause, 1e-9),      # pause_precision
                         npause / len(M),               # pauses_per_clip
                         nhit / max(ncue, 1e-9),        # space_realization
                         intra / len(M)])               # intra_word_rate

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pairs), size=(n_boot, len(pairs)))
    deltas = np.array([metrics(B[i]) - metrics(A[i]) for i in idx])
    point = metrics(B) - metrics(A)
    out = {}
    for k, name in enumerate(["pause_precision", "pauses_per_clip",
                              "space_realization", "intra_word_rate"]):
        lo, hi = np.percentile(deltas[:, k], [2.5, 97.5])
        out[name] = {"delta": round(float(point[k]), 4),
                     "ci95": [round(float(lo), 4), round(float(hi), 4)],
                     "p_two_sided": round(float(2 * min((deltas[:, k] <= 0).mean(),
                                                        (deltas[:, k] >= 0).mean())), 4)}
    return out


def compare_runs(reps_a: list, reps_b: list) -> dict:
    A = {r.text_id: r for r in reps_a}
    B = {r.text_id: r for r in reps_b}
    shared = sorted(set(A) & set(B))
    fixed = broke = both_bad = both_clean = 0
    for k in shared:
        a_bad, b_bad = A[k].verdict == "bad", B[k].verdict == "bad"
        if a_bad and not b_bad:
            fixed += 1
        elif b_bad and not a_bad:
            broke += 1
        elif a_bad:
            both_bad += 1
        else:
            both_clean += 1
    agg_a = aggregate([A[k] for k in shared], boot=0)
    agg_b = aggregate([B[k] for k in shared], boot=0)
    return {"paired_clips": len(shared),
            "only_in_a": len(set(A) - set(B)), "only_in_b": len(set(B) - set(A)),
            "bootstrap": paired_bootstrap([(A[k], B[k]) for k in shared]),
            "verdict_flips": {"b_fixed_vs_a": fixed, "b_broke_vs_a": broke,
                              "both_bad": both_bad, "both_clean": both_clean},
            "mcnemar_p": round(mcnemar_exact(fixed, broke), 5),
            "a": agg_a, "b": agg_b,
            "delta": {k: round(agg_b[k] - agg_a[k], 4) for k in agg_a
                      if isinstance(agg_a.get(k), (int, float))
                      and isinstance(agg_b.get(k), (int, float))}}


# ---------------------------------------------------------------------------
# §7  Forced alignment (GPU stage)
# ---------------------------------------------------------------------------

def _ctc_bundle(device: str, model_id: str):
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    proc = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
    return dict(proc=proc, model=model, device=device, torch=torch)


def _ctc_targets(text: str, tokenizer):
    """Text chars -> vocab ids, keeping the source index of each kept char. Characters
    outside the vocab (Latin letters, digits, punctuation) are simply not aligned —
    the surrounding Thai still is."""
    vocab = tokenizer.get_vocab()
    delim = getattr(tokenizer, "word_delimiter_token", "|")
    ids, char_idx = [], []
    for i, ch in enumerate(text):
        key = delim if ch == " " else ch
        if key in vocab:
            ids.append(vocab[key])
            char_idx.append(i)
    return ids, char_idx


def _ctc_spans(emissions, ids, char_idx, dur_s, bundle):
    import torchaudio.functional as AF
    torch = bundle["torch"]
    blank = bundle["proc"].tokenizer.pad_token_id or 0
    labels, _ = AF.forced_align(emissions.unsqueeze(0),
                                torch.tensor([ids], dtype=torch.int32), blank=blank)
    labels = labels[0].tolist()
    groups, prev = [], blank
    for f, lab in enumerate(labels):
        if lab != blank and (prev == blank or lab != prev):
            groups.append([f, f + 1])
        elif lab != blank:
            groups[-1][1] = f + 1
        prev = lab
    if len(groups) != len(ids):
        raise RuntimeError(f"CTC merge produced {len(groups)} spans for {len(ids)} targets")
    sec_per_frame = dur_s / emissions.shape[0]
    return [(char_idx[k], g[0] * sec_per_frame * 1000.0, g[1] * sec_per_frame * 1000.0)
            for k, g in enumerate(groups)]


def align(items: list[dict], wav_dir: Path, out_dir: Path, *, device: str,
          batch_size: int = 8, model_id: str = CTC_MODEL, resume: bool = True) -> int:
    """{id}.wav + known text -> one sidecar JSON per clip. Batched; resumable."""
    import soundfile as sf
    import torchaudio.functional as AF
    side = out_dir / "sidecars"
    side.mkdir(parents=True, exist_ok=True)

    todo = []
    for it in items:
        if resume and (side / f"{it['id']}.json").exists():
            continue
        p = wav_dir / f"{it['id']}.wav"
        if p.exists():
            todo.append((it, p))
        else:
            print(f"  !! {it['id']}: no {p.name}", file=sys.stderr)
    if not todo:
        print(f"  nothing to align ({len(items)} items, sidecars present)")
        return 0

    bundle = _ctc_bundle(device, model_id)
    torch = bundle["torch"]
    n = 0
    for b in range(0, len(todo), max(1, batch_size)):
        arrays, metas = [], []
        for it, wav_path in todo[b:b + max(1, batch_size)]:
            try:
                wav, sr = sf.read(str(wav_path))
                wav = np.asarray(wav, dtype=np.float32)
                if wav.ndim > 1:
                    wav = wav.mean(1)
                if len(wav) < 0.1 * sr:
                    raise ValueError(f"clip too short to align ({len(wav)} samples)")
                x = torch.from_numpy(wav)
                if sr != 16000:
                    x = AF.resample(x, sr, 16000)
                ids, char_idx = _ctc_targets(it["text"], bundle["proc"].tokenizer)
            except Exception as e:            # one bad clip must not kill the run
                print(f"  !! {it['id']}: {e}", file=sys.stderr)
                continue
            if ids:
                arrays.append(x.numpy())
                metas.append((it, wav_path, sr, ids, char_idx, len(x) / 16000.0))
        if not arrays:
            continue
        with torch.no_grad():
            feats = bundle["proc"](arrays, sampling_rate=16000, return_tensors="pt",
                                   padding=True)
            # The shipped preprocessor config says return_attention_mask=False, but
            # this checkpoint is feat_extract_norm='layer' and WAS trained with
            # masking. Without per-clip lengths every clip in a padded batch gets its
            # char timeline scaled by len/batch_max — a ~0.7x drift that shifts
            # junctures by whole words and silently invalidates every timestamp. Build
            # the mask from the raw lengths ourselves.
            mask = feats.get("attention_mask")
            if mask is None:
                in_lens = torch.tensor([len(a) for a in arrays])
                mask = (torch.arange(feats.input_values.shape[1])[None, :]
                        < in_lens[:, None]).long()
            logits = bundle["model"](feats.input_values.to(device),
                                     attention_mask=mask.to(device)).logits.cpu().float()
        lens = bundle["model"]._get_feat_extract_output_lengths(mask.sum(-1)).tolist()
        for k, (it, wav_path, sr, ids, char_idx, dur_s) in enumerate(metas):
            try:
                em = torch.log_softmax(logits[k, :int(lens[k])], dim=-1)
                spans = _ctc_spans(em, ids, char_idx, dur_s, bundle)
            except Exception as e:            # noqa: BLE001
                print(f"  !! {it['id']}: {e}", file=sys.stderr)
                continue
            (side / f"{it['id']}.json").write_text(json.dumps(
                {"schema": SIDECAR_SCHEMA,
                 "id": it["id"], "text": it["text"], "gold": it.get("gold"),
                 "tokens": it["tokens"], "word_spans": it.get("word_spans"),
                 "syllables": it.get("syllables"),
                 "juncture_phones": it.get("juncture_phones"),
                 "dead_spans": it.get("dead_spans"), "char_spans": spans,
                 "spoken_override": bool(it.get("benchmark_text")),
                 "wav_path": str(wav_path), "sr": sr, "system": wav_dir.name,
                 "aligner": model_id}, ensure_ascii=False), encoding="utf-8")
            n += 1
        if n and (n % 100 < batch_size):
            print(f"  aligned {n}/{len(todo)} clips")
    return n


# ---------------------------------------------------------------------------
# §8  Items + spoken-text remapping
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
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            items.append(json.loads(line))
    for it in items:
        if "".join(it["tokens"]) != it["text"]:
            raise SystemExit(f"{it['id']}: shipped tokens do not tile the text")
        n = len(it["text"])
        if any(not 0 <= a < b <= n for a, b in it.get("word_spans") or ()):
            raise SystemExit(f"{it['id']}: word_spans fall outside the text")
        syl = it.get("syllables") or []
        if syl != sorted(set(syl)) or any(not 0 <= o <= n for o in syl):
            raise SystemExit(f"{it['id']}: syllables are not an ascending offset grid")
        if any(int(k) not in set(syl) for k in (it.get("juncture_phones") or {})):
            raise SystemExit(f"{it['id']}: juncture_phones off the syllable grid")
        stripped, _, _ = parse_gold(it["gold"])
        if stripped != it["text"]:
            raise SystemExit(f"{it['id']}: gold mask does not strip back to the text")
    return items


def load_text_map(path: str | Path) -> dict[str, str]:
    """{id}<TAB>{text} lines, or JSONL of {id, text}, or a {id: text} JSON object."""
    raw = Path(path).read_text(encoding="utf-8").strip()
    if raw.startswith("{"):
        obj = json.loads(raw)
        if all(isinstance(v, str) for v in obj.values()):
            return {str(k): v for k, v in obj.items()}
    out = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("{"):
            r = json.loads(line)
            out[str(r["id"])] = r["text"]
        else:
            iid, _, text = line.partition("\t")
            out[iid.strip()] = text
    return out


def apply_spoken(items: list[dict], spoken: dict[str, str]) -> list[dict]:
    """Re-express each item on the text the system actually spoke: retokenize, and
    carry the gold mask across the shared regions (marks inside rewritten spans are
    dropped). Items with no override, or whose spoken text is identical, pass through
    with their FROZEN grids untouched.

    ⚠ An overridden item loses those grids. The tltk word and syllable grids and the
    phones that meet at each juncture are frozen per benchmark text and cannot be
    recomputed here without tltk, so a rewritten item falls back to newmm words as its
    own syllable grid and to the letter proxy for the closure guard — a coarser
    instrument. Numbers from a run with overrides are comparable to each other, not to
    numbers from the default contract."""
    out, changed = [], 0
    for it in items:
        s = spoken.get(it["id"])
        if not s or s == it["text"]:
            out.append(it)
            continue
        changed += 1
        g = sanitize_gold(it["text"], it["gold"], it["tokens"])
        allowed = set()
        if g:
            _, allowed, _ = parse_gold(g)
        toks = tokenize_newmm(s)
        marks = remap_offsets(it["text"], s, allowed)
        gold = "".join((MARK_ALLOWED if i in marks else "") + ch
                       for i, ch in enumerate(s))
        out.append({**it, "text": s, "tokens": toks, "gold": gold,
                    "word_spans": None, "syllables": None,
                    "juncture_phones": None, "dead_spans": None,
                    "benchmark_text": it["text"]})
    print(f"  spoken-text override applied to {changed}/{len(items)} items")
    return out


# ---------------------------------------------------------------------------
# §9  CLI
# ---------------------------------------------------------------------------

def _items(args) -> list[dict]:
    items = load_items(args.items)
    if getattr(args, "spoken_texts", None):
        items = apply_spoken(items, load_text_map(args.spoken_texts))
    return items


def _cmd_texts(args):
    items = load_items(args.items)
    fh = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")
    for it in items:
        fh.write(f"{it['id']}\t{it['text']}\n")
    if fh is not sys.stdout:
        fh.close()
        print(f"Wrote {len(items)} texts to {args.out}")
    return 0


def _cmd_align(args):
    n = align(_items(args), Path(args.audio_dir), Path(args.out), device=args.device,
              batch_size=args.batch_size,
              resume=not args.no_resume)
    print(f"aligned {n} clips -> {Path(args.out)/'sidecars'}")
    return 0


def _score(sidecar_dir: Path, args) -> list[ClipReport]:
    import soundfile as sf
    if sidecar_dir.name != "sidecars" and (sidecar_dir / "sidecars").is_dir():
        sidecar_dir = sidecar_dir / "sidecars"
    reports = []
    for p in sorted(sidecar_dir.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("schema", 1) < SIDECAR_SCHEMA:
            raise SystemExit(
                f"{p} predates the current scorer (schema {d.get('schema', 1)} < "
                f"{SIDECAR_SCHEMA}) and carries none of the frozen grids it needs.\n"
                f"Re-run `align` on this audio; scoring it as-is would produce numbers "
                f"comparable with nothing.")
        if not d.get("word_spans") and not d.get("spoken_override"):
            raise SystemExit(f"{p}: no frozen word grid and not a --spoken-texts "
                             f"override — re-run `align` from the shipped item set.")
        wav, sr = sf.read(d["wav_path"])
        reports.append(score_clip(d, np.asarray(wav, dtype=np.float32), sr,
                                  silence_db=args.silence_db,
                                  min_pause_ms=args.min_pause_ms,
                                  stop_min_ms=args.stop_min_pause_ms,
                                  stop2_min_ms=args.stop2_min_pause_ms,
                                  fric_min_ms=args.fric_min_pause_ms))
    if not reports:
        raise SystemExit(f"no sidecars in {sidecar_dir}")
    return reports


def _cmd_score(args):
    side = Path(args.sidecars)
    reports = _score(side, args)
    out = Path(args.out or (side.parent if side.name == "sidecars" else side))
    summary = write_reports(reports, out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["overall"]["median_peak_db"] < -6.0:
        print("\n⚠ median clip peak is quiet; the -35 dBFS silence floor assumes "
              "normally-mastered audio — check for phantom pauses.", file=sys.stderr)
    print(f"\nWrote {out/'results.csv'}, {out/'metrics.json'}")
    return 0


def _cmd_run(args):
    _cmd_align(args)
    args.sidecars = str(Path(args.out) / "sidecars")
    return _cmd_score(args)


def _cmd_compare(args):
    a_tag, a_path = args.a.split("=", 1) if "=" in args.a else ("A", args.a)
    b_tag, b_path = args.b.split("=", 1) if "=" in args.b else ("B", args.b)
    res = compare_runs(_score(Path(a_path), args), _score(Path(b_path), args))
    res = {"a": a_tag, "b": b_tag, **res}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


def _cmd_selftest(args):
    """Synthetic clips with pauses planted at known positions. No model, no data file:
    this checks the classifier and the detector, which is where a silent regression
    would live."""
    checks = []

    def ck(cond, why):
        checks.append((bool(cond), why))

    # --- silence detection ------------------------------------------------------
    sr = 24000
    rng = np.random.default_rng(0)

    def clip(segments):
        """segments: list of (kind, ms); kind 'v' = voiced noise, 's' = silence."""
        parts = []
        for kind, ms in segments:
            n = int(sr * ms / 1000)
            parts.append(rng.normal(0, 0.2, n).astype(np.float32) if kind == "v"
                         else np.zeros(n, dtype=np.float32))
        return np.concatenate(parts)

    sils = detect_silences(clip([("v", 500), ("s", 300), ("v", 500)]), sr)
    ck(len(sils) == 1, "one 300 ms internal silence found")
    # the 50 ms RMS window bleeds voiced energy into the frames at either edge, so a
    # 300 ms hole measures ~250 ms — a fixed bias, not noise
    ck(240 <= sils[0].dur_ms <= 320, f"duration ~300 ms (got {sils[0].dur_ms:.0f})")
    ck(not detect_silences(clip([("v", 500), ("s", 50), ("v", 500)]), sr),
       "50 ms gap is below the 75 ms floor")
    ck(not detect_silences(clip([("s", 300), ("v", 500)]), sr),
       "leading silence is not a pause")
    ck(not detect_silences(clip([("v", 500), ("s", 300)]), sr),
       "trailing silence is not a pause")

    # --- masks ------------------------------------------------------------------
    text = "ประเทศไทยมีวัฒนธรรมที่สวยงามและอาหารอร่อย"
    toks = ["ประเทศไทย", "มี", "วัฒนธรรม", "ที่", "สวยงาม", "และ", "อาหาร", "อร่อย"]
    ck("".join(toks) == text, "fixture tokens tile the fixture text")
    t2, allowed, _ = parse_gold("ประเทศไทย|มีวัฒนธรรม|ที่สวยงาม|และอาหารอร่อย")
    ck(t2 == text, "parse_gold strips back to the text")
    ck(allowed == {9, 19, 28}, f"mark offsets {sorted(allowed)}")
    g = sanitize_gold(text, "ประเทศไทย|มีวัฒนธรรม|ที่สวยงามและ|อาหารอร่อย", toks)
    _, kept, _ = parse_gold(g)
    ck(31 in kept, "the sanitizer keeps every mark on a word boundary")
    ck(9 in kept and 19 in kept, "legal marks survive the sanitizer")
    _, inside, _ = parse_gold(sanitize_gold(text, "ประเทศ|ไทยมีวัฒนธรรมที่สวยงามและอาหารอร่อย", toks))
    ck(not inside, "a mark INSIDE a word is still dropped")
    ck(sanitize_gold("คนละเรื่อง", "ประเทศ|ไทย", ["คนละ", "เรื่อง"]) is None,
       "a mask that does not match its text is rejected")
    spans = word_spans(toks)
    rules = allowed_from_spans(text, spans)
    ck(28 not in rules, "before และ is NOT a slot without a space or a mask")
    ck(19 not in rules, "before ที่ is NOT a slot without a space or a mask")
    ck(31 not in rules, "after และ is never a legal slot")
    ck(not rules, "no source space or punctuation -> no slot")
    src, dst = "ก 25 ข", "ก ยี่สิบห้า ข"
    ck(remap_offsets(src, dst, {3}) == set(),
       "a mark INSIDE a rewritten span is dropped")
    ck(remap_offsets(src, dst, {5}) == {dst.index("ข")},
       "a mark after a rewritten span lands on the new coordinates")

    # --- the closure guard ------------------------------------------------------
    ck(closure_ms(True, True) == 137.5, "two oral stops get the longest allowance")
    ck(closure_ms(True, False) == 125.0 and closure_ms(False, True) == 125.0,
       "one oral stop gets the middle allowance")
    ck(closure_ms(False, False, True) == 100.0, "a bare fricative onset gets 100 ms")
    ck(closure_ms(False, False, False) == 0.0,
       "a sonorant/glottal juncture is unguarded — และ|อาหาร must stay detectable")

    # --- attribution + classification -------------------------------------------
    # the syllable grid the juncture is snapped onto (every word start is on it)
    _, syl, _ = parse_gold("ประ|เทศ|ไทย|มี|วัฒ|น|ธรรม|ที่|สวย|งาม|และ|อา|หาร|อร่อย")
    syl_starts = sorted(syl | {0, len(text)})
    ck(all(a in syl for a, _, _ in spans if a), "every word start is a syllable start")
    # (coda, onset) base IPA at each juncture, as `juncture_phones` ships them
    syl_ctx = {6: ("t", "t"),        # เทศ|ไทย — two oral stops
               9: ("j", "m"),        # ไทย|มี   — neither
               22: ("j", "s"),       # ที่|สวยงาม — fricative onset
               28: ("m", "l"), 31: ("ʔ", "ʔ")}

    # one char span per char, 100 ms each, with a hole at a chosen juncture.
    # `wide` = (a, b, ms) stretches the chars of one word, which is how a real clip
    # looks when the aligner has to cover two utterances of a word the text has once.
    def spans_with_pause_at(off, gap_ms=300.0, w=100.0, wide=None):
        cs, t, t0 = [], 0.0, 0.0
        for i in range(len(text)):
            if i == off:
                t0 = t
                t += gap_ms
            ww = wide[2] if wide and wide[0] <= i < wide[1] else w
            cs.append((i, t, t + ww))
            t += ww
        return cs, Silence(t0, t0 + gap_ms, gap_ms, -60.0)

    def classify(off, gap_ms=300.0, allow=None, ctx=syl_ctx, dead=(), wide=None):
        cs, s = spans_with_pause_at(off, gap_ms, wide=wide)
        return attribute_pauses([s], cs, text, spans,
                                allowed_set if allow is None else allow,
                                syl_starts=syl_starts, syl_ctx=ctx, dead_spans=dead)

    # allowed = text slots (empty here: the fixture has no space or punctuation) plus
    # the sanitized mask marks, which is how a scored clip builds its allowed set
    allowed_set = rules | allowed        # {9, 19, 28} from the fixture mask
    for off, want in ((28, PAUSE_OK), (31, BAD_JUNCTURE), (13, INTRA_WORD)):
        ps = classify(off)
        ck(len(ps) == 1 and ps[0].cls == want,
           f"pause at offset {off} -> {want} (got {ps[0].cls if ps else 'none'})")
    # the syllable snap: a break CTC reports one char inside a syllable is timing
    # noise, and without the snap it reads as the most severe error class
    ps = classify(8)
    ck(len(ps) == 1 and ps[0].cls == PAUSE_OK and ps[0].ch_offset == 9,
       "a juncture one char inside ไทย snaps to the word boundary at 9")
    # the graded closure guard reads the phones that actually meet at the juncture
    ck(not classify(6, gap_ms=130.0), "130 ms between two oral stops is a closure")
    ck(len(classify(6, gap_ms=200.0)) == 1, "200 ms between two stops is a pause")
    ck(len(classify(9, gap_ms=80.0)) == 1,
       "80 ms at a sonorant juncture is a pause — no allowance applies")
    ck(not classify(22, gap_ms=90.0, allow={22}),
       "90 ms before a fricative onset is a closure")
    # a juncture with no phones (digits/Latin on one side) falls back to the letters,
    # which resolve the onset past a preposed vowel — but read เทศ's coda as the
    # fricative ศ is written rather than the /t̚/ it is spoken, hence a proxy only
    ck(_juncture_context(text, 6, syl_starts) == (False, True, False),
       "the letter proxy resolves ไทย's onset past the preposed vowel to ท")
    ck(not classify(6, gap_ms=120.0, ctx={}),
       "and still guards เทศ|ไทย at the one-stop allowance")

    # --- artifact rejection -----------------------------------------------------
    ck(_artifact_reason(word_span_ms=400.0, pause_ms=437.0, n_syl=3, syl_ms=120.0,
                        over_unverbalized=False) == "impossible",
       "a pause longer than its word's span cannot be inside it")
    ck(_artifact_reason(word_span_ms=1400.0, pause_ms=100.0, n_syl=2, syl_ms=120.0,
                        over_unverbalized=False) == "repeat",
       "a word span far over the clip's own syllable rate reads as a repeat")
    ck(_artifact_reason(word_span_ms=400.0, pause_ms=100.0, n_syl=3, syl_ms=120.0,
                        over_unverbalized=False) == "",
       "an ordinary word-internal silence is scoreable")
    ck(_artifact_reason(word_span_ms=400.0, pause_ms=100.0, n_syl=3, syl_ms=120.0,
                        over_unverbalized=True) == "unverbalized",
       "a word abutting unverbalized input is not scoreable")
    # end to end: วัฒนธรรม (3 syllables) aligned over 3.2 s while the clip's own rate
    # is 300 ms/syllable — the aligner is covering a second utterance of it
    ps = classify(13, wide=(11, 19, 400.0))
    ck(len(ps) == 1 and ps[0].cls == ARTIFACT and ps[0].reason == "repeat",
       f"a repeated word is an artifact, not an intra-word error "
       f"(got {ps[0].cls if ps else 'none'})")
    ck(len(classify(13)) == 1 and classify(13)[0].cls == INTRA_WORD,
       "the same juncture at an ordinary rate is still an intra-word error")
    rep_art = ClipReport(system="t", text_id="t", wav_path="", audio_s=1.0,
                         peak_db=-1.0, pauses=ps).finalize()
    ck(rep_art.n_scored == 0 and rep_art.verdict == "clean",
       "artifacts leave the scored denominator and do not fail the clip")
    ck(aggregate([rep_art], boot=0)["artifact_reasons"] == {"repeat": 1},
       "artifacts are reported, not hidden")

    # --- the tokenizer-merge veto -----------------------------------------------
    # tltk glues คนใน; newmm keeps คน+ใน apart, so the juncture is a word boundary
    merged = [(0, 4, "คนใน")]
    ck(_refine_word_spans(merged, "คนใน", {0, 2}) == [(0, 2, "คน"), (2, 4, "ใน")],
       "a tltk word is split at an interior newmm boundary")
    ck(_refine_word_spans(merged, "คนใน", set()) == merged,
       "with no second tokenizer the grid stands unrefined")

    # --- reports + stats --------------------------------------------------------
    rep = ClipReport(system="t", text_id="t1", wav_path="", audio_s=6.0, peak_db=-1.0,
                     pauses=classify(31)).finalize()
    ck(rep.verdict == "bad" and rep.n_bad_junc == 1, "clip verdict follows its pauses")
    agg = aggregate([rep], boot=0)
    ck(agg["pause_precision"] == 0.0 and agg["pper"] == 1.0, "aggregate on one bad clip")
    ck(mcnemar_exact(0, 0) == 1.0, "mcnemar with no discordance")
    ck(abs(mcnemar_exact(1, 9) - 0.021484375) < 1e-9, "mcnemar exact binomial")

    bad = [w for ok, w in checks if not ok]
    for w in bad:
        print("FAIL " + w)
    print(f"{len(checks) - len(bad)}/{len(checks)} checks passed")
    # Offline on purpose: the assertions above need no data file, so a selftest
    # never reaches for the network.  Validate the items only if they are already here.
    if not bad and (args.items or DEFAULT_ITEMS.exists()):
        items = load_items(args.items)
        print(f"data file OK: {len(items)} items, "
              f"{sum(len(parse_gold(i['gold'])[1]) for i in items)} gold marks")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--items", default=None,
                    help=f"local {ITEMS_NAME} (default: fetch {HF_DATASET})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def thresholds(p):
        p.add_argument("--silence-db", type=float, default=THRESHOLDS["silence_db"])
        p.add_argument("--min-pause-ms", type=float, default=THRESHOLDS["min_pause_ms"])
        p.add_argument("--stop-min-pause-ms", type=float,
                       default=THRESHOLDS["stop_min_pause_ms"],
                       help="closure allowance when ONE side is an oral stop")
        p.add_argument("--stop2-min-pause-ms", type=float,
                       default=THRESHOLDS["stop2_min_pause_ms"],
                       help="closure allowance when BOTH sides are oral stops")
        p.add_argument("--fric-min-pause-ms", type=float,
                       default=THRESHOLDS["fric_min_pause_ms"],
                       help="closure allowance for a bare fricative onset")

    def aligner(p):
        p.add_argument("--audio-dir", required=True, help="directory of {id}.wav")
        p.add_argument("--out", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--no-resume", action="store_true")
        p.add_argument("--spoken-texts",
                       help="{id}<TAB>{text} of what the system actually SAID, if it "
                            "rewrites the input (needs pythainlp)")

    p = sub.add_parser("texts", help="dump {id}<TAB>{text} to feed your TTS")
    p.add_argument("--out", default="-")
    p.set_defaults(fn=_cmd_texts)

    p = sub.add_parser("align", help="wavs -> CTC char timings (GPU)")
    aligner(p)
    p.set_defaults(fn=_cmd_align)

    p = sub.add_parser("score", help="sidecars -> results.csv + metrics.json (CPU)")
    p.add_argument("--sidecars", required=True)
    p.add_argument("--out")
    thresholds(p)
    p.set_defaults(fn=_cmd_score)

    p = sub.add_parser("run", help="align + score")
    aligner(p)
    thresholds(p)
    p.set_defaults(fn=_cmd_run)

    p = sub.add_parser("compare", help="paired A/B: bootstrap CIs + McNemar")
    p.add_argument("--a", required=True, help="[tag=]run-dir-or-sidecars")
    p.add_argument("--b", required=True, help="[tag=]run-dir-or-sidecars")
    p.add_argument("--out")
    thresholds(p)
    p.set_defaults(fn=_cmd_compare)

    sub.add_parser("selftest", help="synthetic-audio checks; needs no model"
                   ).set_defaults(fn=_cmd_selftest)

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
