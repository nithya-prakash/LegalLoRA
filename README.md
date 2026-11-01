# LegalLoRA

LoRA fine-tuning of a 3B open model to extract German legal named entities as strict JSON, trained and evaluated entirely on an Apple M4 laptop (16 GB) with MLX. Compared honestly against the same base model with zero-shot and 5-shot prompting.

> Research/portfolio project. Not legal advice, not a production system.

## Task

Input: one German sentence from a court decision.
Output: a JSON array of `{"text", "type"}` objects, with `type` one of

| Type | Meaning |
|---|---|
| `PER` | Person |
| `ORG` | Organisation (courts, companies, institutions) |
| `LOC` | Location |
| `NRM` | Norm: statute, regulation, contract |
| `REG` | Regulation (administrative rule, decree) |
| `RS` | Case law (court decision) |
| `LIT` | Legal literature |

Example output: `[{"text": "§ 17 BBesG", "type": "NRM"}, {"text": "Bundesverwaltungsgericht", "type": "ORG"}]`

## Results

Held-out test set: 500 sentences (625 gold entities). Base model: `mlx-community/Qwen2.5-3B-Instruct-4bit`. Scoring is entity-level **exact match** on text and type.

| Model | Entity F1 | Precision | Recall | Valid JSON | Latency / sentence |
|---|---|---|---|---|---|
| Zero-shot | 0.050 | 0.146 | 0.030 | 98.8% | 1.3 s |
| 5-shot prompt | 0.374 | 0.421 | 0.336 | 99.2% | 8.5 s |
| **LoRA, step 250 of 750** | **0.694** | 0.687 | 0.701 | 97.8% | 3.0 s |

Per-type F1 (support in the test set):

| Type | Zero-shot | 5-shot | LoRA | n |
|---|---|---|---|---|
| NRM | 0.00 | 0.64 | 0.86 | 202 |
| PER | 0.00 | 0.00 | 0.77 | 45 |
| LIT | 0.00 | 0.16 | 0.74 | 38 |
| RS | 0.02 | 0.27 | 0.68 | 164 |
| ORG | 0.19 | 0.20 | 0.52 | 95 |
| LOC | 0.09 | 0.00 | 0.47 | 40 |
| REG | 0.00 | 0.05 | 0.42 | 41 |

Raw numbers: `results/*.json`. Latency is mean wall-clock per sentence on the M4 and includes the long prompt for the 5-shot run.

## Method

- **Data:** [German Legal Entity Recognition](https://huggingface.co/datasets/elenanereiss/german-ler) (expert-annotated court decisions, CC BY 4.0), using its 7-class coarse scheme. BIO tags are converted to entity spans. Splits are subsampled with a fixed seed: 3,000 train / 300 validation / 500 test sentences, with about 25% of each split containing no entity. Labels are the dataset's own; none are LLM-generated.
- **Training:** LoRA (rank 8, scale 20, 16 layers, 6.65M trainable parameters = 0.22%) with `mlx-lm`, learning rate 1e-4, batch 4, max sequence length 768, loss on the JSON answer only. Config: `configs/lora_qwen3b.yaml`.
- **Baselines:** same base model, same system prompt. The 5-shot baseline prepends the first 5 training sentences that contain entities (deterministic).
- **Decoding:** greedy (temperature 0), at most 400 new tokens.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd scripts && ../.venv/bin/python prepare_data.py && cd ..
./scripts/train.sh
cd scripts
../.venv/bin/python evaluate.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --out results/base_zeroshot.json
../.venv/bin/python evaluate.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --fewshot 5 --out results/base_fewshot5.json
../.venv/bin/python evaluate.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --adapter ../adapters --out results/tuned.json
```

Training uses about 10-15 s per step on an M4 with 16 GB, so the planned 750 steps take roughly 2.5 hours.

## Limitations

- **Partial training.** The reported adapter is the step-250 checkpoint of a planned 750-step (one-epoch) run. Training was stopped early to save time, so the numbers are probably not the ceiling. The training log is not committed.
- **Possible train/test leakage.** The dataset is split by sentence and the Hugging Face version has no document IDs, so sentences from the same court decision may appear on both sides. Scores may be optimistic.
- **Single run, single seed, 500 test sentences.** Differences of a few points are not meaningful.
- **Strict scoring.** Exact text and type match; partially correct spans count as errors.
- **Output loops.** Some invalid outputs (2.2%) are repetition loops that hit the 400-token cap.
- **Weak classes.** ORG, LOC and REG are the least frequent in training and score lowest.
- **Domain.** Court decisions only. Not tested on contracts or other legal text.
- Only one base model and one prompt were tried; no hyperparameter search was run.

## Data licence and citation

The LER dataset is licensed CC BY 4.0. If you use it, cite:

Leitner, E., Rehm, G., Moreno-Schneider, J. *A Dataset of German Legal Documents for Named Entity Recognition.* LREC 2020. arXiv:2003.13016.

Code in this repository is MIT licensed (see `LICENSE`). The base model is subject to its own licence.
