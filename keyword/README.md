# Thai TTS Hard-Keyword Benchmark

Does a Thai TTS actually *say the hard word* — the brand, the name, the rare compound,
the chat spelling — or does it fluently say something else? One wrong word in a fluent
150-character sentence costs ~1% CER and reads as "fine", which is why sentence CER
cannot answer this.

**1,531 Thai sentences, each with exactly one keyword under test.** Synthesize,
transcribe with one pinned Thai ASR, check the keyword is there. It scores audio against
text and prescribes nothing about how the audio was made.

```json
{"id": "cs-0001", "category": "code_switch",
 "text": "เมื่อวานเรียก Grab ไปทำงาน รถมาเร็วมาก",   ← synthesize this verbatim
 "keyword": "Grab",
 "expected": "แกร็บ",                                ← what the scoring ASR writes when it is spoken right
 "alternates": ["Grab", "แกร๊บ", "แกรบ"]}            ← other accepted ASR spellings
```

Correct = the transcript contains `expected` or an `alternates` entry, as an exact
substring after Thai canonicalization. **No fuzzy matching.**

| category | n | probes |
|---|---|---|
| `code_switch` | 391 | Latin spans in Thai: brands, acronyms, tickers |
| `rare_words` | 410 | low-frequency, technical and literary vocabulary |
| `names` | 310 | Thai personal/place names — many-to-one orthography |
| `informal_spelling` | 210 | chat orthography outside standard dictionaries |
| `long_sentences` | 210 | does the keyword survive a long clause |

Each item also carries `text_mono` (Latin respelled in Thai) and `text_bilingual`. The
gold is the same for all three — pick one contract, use it for every system you compare,
and say which. `meta` holds construction metadata only: the category stratum (`cs_class`,
`name_type`, `kind`, `freq_band`, `tnc_freq`, `keyword_position`, `length_bucket`,
`phenomenon`), the canonical spelling of an informal item (`canonical`) and whether a
long-sentence target is reused from another category (`reused_keyword`).

## Use

The harness is one file, `eval_keyword.py`, in
[`wayu-research/thai-tts-eval`](https://github.com/wayu-research/thai-tts-eval). It pulls
the items itself.

```bash
python3 eval_keyword.py texts --out texts.tsv        # {id}<TAB>{text} to synthesize
python3 eval_keyword.py transcribe --audio-dir audio/mysystem \
    --out runs/mysystem/transcripts.json
python3 eval_keyword.py score --transcripts runs/mysystem/transcripts.json \
    --out runs/mysystem
```

Audio goes in as `{id}.wav`. `--transcripts` also takes output from your own ASR stack.
`compare` runs a paired A/B with an exact McNemar test — use it when two systems are
close.

## Before you quote a number

- **A perfect system cannot score 100%.** Thai orthography is many-to-one, so the scoring
  ASR sometimes writes an unlisted homophone of a correctly-spoken word. Rankings survive
  this, so compare systems freely; just never read a score as a share of 100%.
- **The scoring ASR is fixed.** `expected`/`alternates` are authored against it, so
  swapping it fails correct renders on spelling alone.
- **Binary per item**, and not a pronunciation judge — it asks only whether one ASR wrote
  one string.
- **Central Thai only.** Sentences were LLM-assisted in authoring, so contamination is not
  formally excluded.

## Baselines

Three systems on all 1,531 items. Gemini and OmniVoice were fed `text`. Wayu-Paxa-TTS-Edge
was fed `text_bilingual` — the paper's LLM-verbalized form of `text`, which the paper counts
as part of that system's frontend; the released `wayu-tts` package has no LLM verbalizer,
so feeding it `text` scores slightly lower.

| system | overall | code_switch | names | rare_words | informal | long_sent. |
|---|---|---|---|---|---|---|
| Gemini 3.1 Flash TTS | **79.8%** | 83.1% | 62.9% | 85.4% | 87.1% | 80.0% |
| OmniVoice (zero-shot teacher) | 72.8% | 69.8% | 54.8% | 85.4% | 78.6% | 74.3% |
| Wayu-Paxa-TTS-Edge (82M student) | 68.2% | 62.7% | 51.9% | 79.8% | 75.7% | 72.4% |

Companion: [`wayu-ai/thai-tts-pause-bench`](https://huggingface.co/datasets/wayu-ai/thai-tts-pause-bench)
scores phrase-break placement on the 210 `long_sentences` items, same ids — one render
set covers both.

## Citation

If you use this benchmark, cite the paper and the harness it builds on.

```bibtex
@misc{pipatanakul2026buildingevaluatingfixedvoicethai,
      title={Building and Evaluating Fixed-Voice Thai TTS from Synthetic Speech}, 
      author={Kunat Pipatanakul and Potsawee Manakul and Warit Sirichotedumrong and Sittipong Sripaisarnmongkol and Pakorn Nathong and Phatrasek Jirabovonvisut},
      year={2026},
      eprint={2609.03502},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.03502}, 
}
```

The item schema and the matcher derive from Potsawee Manakul's
[`th01-tts-eval`](https://github.com/potsawee/th01-tts-eval) (`thai-tts-keyword-eval`),
whose pilot set seeded this one:

```bibtex
@misc{manakul2026th01ttseval,
  title        = {thai-tts-keyword-eval: Keyword accuracy for Thai TTS},
  author       = {Manakul, Potsawee},
  year         = {2026},
  howpublished = {\url{https://github.com/potsawee/th01-tts-eval}}
}
```

## License

> **Research artifact**, released for reproducibility and further research — not a
> certification or a product, and provided without warranty or liability of any kind.
> See the [repository README](https://github.com/wayu-research/thai-tts-eval#disclaimer) for the full disclaimer.

**CC-BY-4.0.** No audio is redistributed.

Kunat Pipatanakul — [research@wayuresearch.org](mailto:research@wayuresearch.org). Gold
corrections are welcome — open an issue or a pull request.
