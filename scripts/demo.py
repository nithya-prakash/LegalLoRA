"""Live demo: run the tuned adapter on held-out test sentences and show the
predicted entities next to the expert labels.

Usage (repo root): .venv/bin/python scripts/demo.py [--n 4]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from common import entity_key, messages, parse_prediction

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"PER": 35, "ORG": 36, "LOC": 32, "NRM": 33, "REG": 34, "RS": 31, "LIT": 95}


def tag(e):
    return f"\033[{COLORS[e['type']]}m{e['text']} [{e['type']}]\033[0m"


def pick(rows, n):
    """Deterministic, model-independent: the longest test sentences of at most 140
    chars that have at least 2 expert entities (chosen by length, not by correctness)."""
    good = [r for r in rows if len(r["sentence"]) <= 140 and len(r["entities"]) >= 2]
    return sorted(good, key=lambda r: len(r["sentence"]))[-n:] if len(good) >= n else good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    ap.add_argument("--adapter", default=str(ROOT / "adapters"))
    args = ap.parse_args()

    print("Loading Qwen2.5-3B + LoRA adapter ...", flush=True)
    model, tok = load(args.model, adapter_path=args.adapter)
    sampler = make_sampler(temp=0.0)
    rows = []
    for line in (ROOT / "data" / "test.jsonl").open(encoding="utf-8"):
        m = json.loads(line)["messages"]
        rows.append({"sentence": m[1]["content"], "entities": json.loads(m[2]["content"])})

    for i, row in enumerate(pick(rows, args.n), 1):
        prompt = tok.apply_chat_template(messages(row["sentence"]), add_generation_prompt=True)
        raw = generate(model, tok, prompt=prompt, max_tokens=300, sampler=sampler)
        pred = parse_prediction(raw) or []
        gold = Counter(entity_key(e) for e in row["entities"])
        got = Counter(entity_key(e) for e in pred)
        print(f"\n\033[1m#{i}  {row['sentence']}\033[0m")
        print("  model :", "  ".join(tag(e) for e in pred) or "(none)")
        print("  expert:", "  ".join(tag(e) for e in row["entities"]))
        hit = sum((gold & got).values())
        print(f"  -> {hit}/{sum(gold.values())} expert entities matched exactly")


if __name__ == "__main__":
    main()
