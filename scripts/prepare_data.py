"""Build mlx-lm chat JSONL files from elenanereiss/german-ler (CC BY 4.0).

Deterministic: fixed seed, no LLM-generated labels. Labels are the dataset's
expert annotations (coarse scheme, 7 classes).

Usage: python scripts/prepare_data.py [--train 3000] [--valid 300] [--test 500]
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from common import bio_to_entities, messages

OUT = Path(__file__).resolve().parent.parent / "data"
SEED = 13
NO_ENTITY_FRACTION = 0.25  # share of sentences with no entity kept in each split


def build(split, n, rng):
    rows = []
    for r in split:
        tags = list(r["coarse-ner"])
        rows.append((r["tokens"], bio_to_entities(r["tokens"], tags)))
    pos = [x for x in rows if x[1]]
    neg = [x for x in rows if not x[1]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_neg = int(n * NO_ENTITY_FRACTION)
    chosen = pos[: n - n_neg] + neg[:n_neg]
    rng.shuffle(chosen)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--valid", type=int, default=300)
    ap.add_argument("--test", type=int, default=500)
    args = ap.parse_args()

    ds = load_dataset("elenanereiss/german-ler")
    rng = random.Random(SEED)
    OUT.mkdir(exist_ok=True)

    for name, split, n in (("train", ds["train"], args.train),
                           ("valid", ds["validation"], args.valid),
                           ("test", ds["test"], args.test)):
        chosen = build(split, n, rng)
        path = OUT / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for tokens, ents in chosen:
                sentence = " ".join(tokens)
                rec = {"messages": messages(sentence, ents)}  # mlx-lm needs one format for all splits
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_ent = sum(len(e) for _, e in chosen)
        print(f"{name}: {len(chosen)} sentences, {n_ent} entities -> {path}")


if __name__ == "__main__":
    main()
