# Thai TTS Pause-Placement Benchmark

Thai is written without word spaces, so a system that breaks mid-word is not
mispronouncing anything — a listener hears it in the first second and no character-error
number moves:

```
ประเทศไทยมีวัฒน ธรรม…          ← pause inside a word
```

**210 long Thai sentences with frozen gold break masks.** Synthesize, force-align, and
check where the system actually breathed. It scores audio against text and prescribes
nothing about how the audio was made.

Each item ships the sentence, a gold mask of **allowed** break slots, and frozen word,
syllable and phone grids — so the scorer needs no tokenizer or G2P at runtime and cannot
drift when one is upgraded.

## Use

The harness is one file, `eval_pause.py`, in
[`wayu-research/thai-tts-eval`](https://github.com/wayu-research/thai-tts-eval). It pulls
the items itself.

```bash
python3 eval_pause.py texts --out texts.tsv          # {id}<TAB>{text} to synthesize
python3 eval_pause.py run --audio-dir audio/mysystem --out runs/mysystem
```

Audio goes in as `{id}.wav`, any sample rate. `score` needs `numpy` + `soundfile`;
alignment adds `torch` and `transformers`. `align` and `score` also run as separate
stages, so you can re-score without re-aligning, and `compare` runs a paired A/B.

If the system rewrites text before speaking, pass what it actually said with
`--spoken-texts`; marks landing inside a rewritten span are dropped rather than guessed.

## Metrics

- **`pause_precision`** ↑ — realized pauses that landed at an allowed slot.
- **`pauses_per_clip`** and **`space_realization`** ↑ — the recall side. A precision gain
  arriving with these falling is a system pausing *less*, not phrasing better.
- **`pper`** ↓ clips with ≥1 misplaced pause · **`intra_word_rate`** ↓ ·
  **`unaligned_rate`** (if not ~0, alignment failed and nothing else is trustworthy).

## Before you quote a number

- **Precision alone rewards a system that never pauses.** Masks mark which breaks are
  *allowed*, not which are required, so reading the whole sentence in one breath scores
  perfectly. Always report `pause_precision` with `pauses_per_clip` and
  `space_realization`.
- **The aligner is fixed**, chosen by measurement against exact decoder frame spans.
  Swapping it changes every number.
- **Silence-based** — phrasing, not prosody. A break realized with an F0 reset and no
  silence is invisible here.
- **Normalize your audio.** The silence threshold is an absolute level, so quietly
  mastered audio shows phantom pauses. `score` warns when a clip is too quiet.
- **210 items, long sentences only, Central Thai.** Small gaps will not resolve; use
  `compare` rather than two headline numbers.

## Detector accuracy

Pause detection here is not exact, and the limits are known. *Where* a silence is attributed
depends on the Thai word segmentation frozen into each item: two dictionary tokenizers,
which still glue some compounds a listener would hear as two words, so a break at such a
seam is scored as intra-word. *Whether* a silence counts as a pause at all depends on
hand-tuned thresholds — the silence floor, the minimum pause length, the per-context
consonant-closure allowances and the pause-to-word-span cap were set once on known-good and
known-bad phrasing and then held fixed. The numbers are therefore most reliable as
comparisons between systems scored with the same instrument, not as absolute rates.
Improvements to either part — better word grids, or a principled replacement for the tuned
thresholds — are very welcome: open an issue or a pull request.

## Baselines

Three systems on the 210 items, raw text contract:

| system | pause prec. ↑ | PPER ↓ | intra-word ↓ | pauses/clip | space realization |
|---|---|---|---|---|---|
| Gemini 3.1 Flash TTS | **96.4%** | 13.8% | 1.9% | 4.44 | 29.7% |
| OmniVoice (zero-shot teacher) | 89.9% | 17.6% | 5.2% | 1.79 | 9.9% |
| Wayu-Paxa-TTS-Edge (82M student) | 91.4% | **6.7%** | **1.4%** | 0.89 | 5.3% |

Read the rows together, not as a ranking. Gemini has the highest precision *and* pauses
five times more often than the student; the student's low PPER comes partly from phrasing
well and partly from barely phrasing at all. Precision alone would flatter it.

## License

> **Research artifact**, released for reproducibility and further research — not a
> certification or a product, and provided without warranty or liability of any kind.
> See the [repository README](https://github.com/wayu-research/thai-tts-eval#disclaimer) for the full disclaimer.

**CC-BY-4.0.** Sentences come from the `long_sentences` category of the companion
[keyword benchmark](https://huggingface.co/datasets/wayu-ai/thai-tts-keyword-bench), same
ids — one render set covers both. No audio is redistributed.

Kunat Pipatanakul — [research@wayuresearch.org](mailto:research@wayuresearch.org). Mask
corrections are welcome — open an issue or a pull request.
