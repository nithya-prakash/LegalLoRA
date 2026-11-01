"""Evaluate a model (optionally with a LoRA adapter) on the held-out test set.

Metrics: JSON validity rate, entity-level micro precision/recall/F1 (exact match
on text AND type, multiset), per-type F1, mean latency per sentence.

Usage:
    python scripts/evaluate.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --out results/base.json
    python scripts/evaluate.py --model ... --adapter adapters --out results/tuned.json
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from common import COARSE, entity_key, messages, parse_prediction

ROOT = Path(__file__).resolve().parent.parent


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fewshot", type=int, default=0,
                    help="prepend K training examples as chat turns (prompted baseline)")
    ap.add_argument("--dedup", action="store_true",
                    help="drop test sentences that also appear in the training set")
    args = ap.parse_args()

    model, tok = load(args.model, adapter_path=args.adapter)
    sampler = make_sampler(temp=0.0)
    rows = []
    for l in (ROOT / "data" / "test.jsonl").open(encoding="utf-8"):
        m = json.loads(l)["messages"]
        rows.append({"sentence": m[1]["content"], "entities": json.loads(m[2]["content"])})
    if args.dedup:
        seen = set()
        for l in (ROOT / "data" / "train.jsonl").open(encoding="utf-8"):
            seen.add(" ".join(json.loads(l)["messages"][1]["content"].lower().split()))
        rows = [r for r in rows if " ".join(r["sentence"].lower().split()) not in seen]
    if args.limit:
        rows = rows[: args.limit]

    shots = []
    if args.fewshot:
        train = [json.loads(l) for l in (ROOT / "data" / "train.jsonl").open(encoding="utf-8")]
        # first K training rows with at least one entity, in file order (deterministic)
        for r in train:
            if r["messages"][-1]["content"] != "[]":
                shots += r["messages"][1:]
            if len(shots) >= 2 * args.fewshot:
                break

    valid = 0
    tp, fp, fn = Counter(), Counter(), Counter()
    latencies, failures = [], []
    for i, row in enumerate(rows):
        msgs = messages(row["sentence"])
        msgs = msgs[:1] + shots + msgs[1:]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
        t0 = time.perf_counter()
        raw = generate(model, tok, prompt=prompt, max_tokens=400, sampler=sampler)
        latencies.append(time.perf_counter() - t0)
        pred = parse_prediction(raw)
        gold = Counter(entity_key(e) for e in row["entities"])
        if pred is None:
            failures.append({"sentence": row["sentence"], "raw": raw[:300]})
            pred_c = Counter()
        else:
            valid += 1
            pred_c = Counter(entity_key(e) for e in pred)
        for key in gold | pred_c:
            hit = min(gold[key], pred_c[key])
            tp[key[1]] += hit
            fp[key[1]] += pred_c[key] - hit
            fn[key[1]] += gold[key] - hit
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)

    P, R, F = f1(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    per_type = {t: dict(zip(("precision", "recall", "f1"), f1(tp[t], fp[t], fn[t])),
                        support=tp[t] + fn[t]) for t in COARSE}
    result = {
        "model": args.model, "adapter": args.adapter, "fewshot": args.fewshot, "n_sentences": len(rows),
        "json_valid_rate": valid / len(rows),
        "micro": {"precision": P, "recall": R, "f1": F},
        "per_type": per_type,
        "mean_latency_s": sum(latencies) / len(latencies),
        "invalid_examples": failures[:5],
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("json_valid_rate", "micro", "mean_latency_s")}, indent=2))


if __name__ == "__main__":
    main()
