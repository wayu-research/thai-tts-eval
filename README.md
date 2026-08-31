# Thai TTS Eval

Two things sentence CER cannot tell you about a Thai TTS, each as a frozen item set with
a self-contained harness. Both take a directory of `{id}.wav` and score it against text.
Neither knows or cares what produced the audio.

> **Research artifact.** Released for reproducibility and further research, not as a
> certification, ranking service or product: no maintenance or availability commitment
> and no warranty, and item sets may change between versions, so pin a version before you
> publish numbers. There is no staffed support channel — but community help is very
> welcome, so please open an issue or a pull request, gold corrections most of all.

| benchmark | items | asks |
|---|---|---|
| [`keyword/`](keyword/) | 1,531 | does it *say the hard word* — the brand, the name, the rare compound, the chat spelling? |
| [`pause/`](pause/) | 210 | does it breathe where a Thai reader would breathe? |

Thai makes both failures invisible to CER. One misread brand name in a fluent
150-character sentence costs about 1% CER and reads as "fine". And Thai is written
without word spaces, so a system that breaks mid-word is not mispronouncing anything:

```
ประเทศไทยมีวัฒน ธรรม…          ← pause inside a word
```

The 210 pause sentences are the `long_sentences` category of the keyword set, same ids
and same text, so **one render set scores both**.

## Quickstart

Both harnesses self-test on synthetic input, so this works before you have anything else:

```bash
python3 keyword/eval_keyword.py selftest
python3 pause/eval_pause.py selftest
```

Then dump the texts, synthesize them as `{id}.wav`, and score:

```bash
python3 keyword/eval_keyword.py texts --out texts.tsv
python3 keyword/eval_keyword.py transcribe --audio-dir audio/mysystem \
    --out runs/mysystem/transcripts.json
python3 keyword/eval_keyword.py score --transcripts runs/mysystem/transcripts.json \
    --out runs/mysystem

python3 pause/eval_pause.py texts --out texts.tsv
python3 pause/eval_pause.py run --audio-dir audio/mysystem --out runs/mysystem
```

Both have `compare --a base=… --b new=…` for a paired A/B with an exact McNemar test —
use it when two systems are close. Scoring is stdlib plus numpy; transcription and
alignment need `torch`, `transformers` and a GPU, and both are resumable.

The item sets live on Hugging Face —
[`keyword`](https://huggingface.co/datasets/wayu-ai/thai-tts-keyword-bench) ·
[`pause`](https://huggingface.co/datasets/wayu-ai/thai-tts-pause-bench) — and the
harnesses fetch them on first use. Pass `--items` to work from a local or pinned copy.

## Before you quote a number

- **A perfect system cannot score 100% on the keyword benchmark.** Thai orthography is
  many-to-one, so the scoring ASR sometimes writes an unlisted homophone of a correctly
  spoken word. Compare systems against each other, never against 100%.
- **Pause precision alone rewards a system that never pauses.** Report it with
  `pauses_per_clip` and `space_realization` or it is not interpretable.
- **The recognizers and the linguistic grids are fixed.** Swapping either invalidates
  every number.
- **Hold the input contract fixed.** Whether text is fed as written or pre-expanded moves
  scores by more than many system differences do. Pick one, use it everywhere, say which.

Each benchmark's `README.md` is the full contract. Read it before publishing.

## Disclaimer

Provided **"as is", without warranty of any kind**, express or implied, to the fullest
extent permitted by law. The authors, the maintainers and their affiliated institutions
accept **no responsibility and no liability** for any damage, loss, cost or claim
arising from use or misuse of these item sets, the harnesses, or any score, ranking or
conclusion derived from them, and are not responsible for how third parties use them.

A score here measures one narrow property against one fixed instrument. It is **not** a
measure of overall speech quality, safety or fitness for any purpose, and must not be
presented as a certification or endorsement of any system.

The sentences were LLM-assisted in authoring and carry no guarantee of factual accuracy;
they are synthesis prompts, not statements of fact. No audio is redistributed.

## License

**CC-BY-4.0.** Gold corrections are very welcome — open an issue or a pull request.

Kunat Pipatanakul — [research@wayuresearch.org](mailto:research@wayuresearch.org)
