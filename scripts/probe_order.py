#!/usr/bin/env python3
"""Two-pass model-perceived difficulty reorder (arXiv:2508.15475).

Probe → rank rows by loss → reorder. Human-centered difficulty metrics
show limited success for LM pretraining; difficulty AS OBSERVED DURING
TRAINING is competitive. Usage:

  1. tokenize + pack the TEXT corpus to TT+1 id rows
     (scripts/tokenize_corpus.py, scripts/pack_rows.py)
  2. probe with the current weights (existing binary, no rebuild):
       eval_bpb --weights <dir> --rows <packed> > /tmp/probe.nll
     (one line per row: space-separated per-position NLLs in nats)
  3. reorder the TEXT corpus easy-first:
       python3 scripts/probe_order.py corpus.jsonl /tmp/probe.nll \\
           --out corpus.easy.jsonl
     then tokenize + pack the reordered file for the curriculum phase.

Writes the reordered JSONL plus a one-line report (mean/median probe
loss, inversions vs input order). Row texts are untouched — only order
changes. Never average per-phase curves (arXiv:2601.21698); this tool
reorders rows *within* one phase's file.
"""

import argparse
import json
import os
import statistics
import sys


def read_texts(path):
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    return texts


def read_probe(path):
    means = []
    with open(path) as f:
        for line in f:
            vals = [float(x) for x in line.split()]
            if vals:
                means.append(sum(vals) / len(vals))
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", help="TEXT JSONL (one {\"text\"} per line)")
    ap.add_argument("probe", help="eval_bpb stdout (one NLL line per row)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hard-first", action="store_true",
                    help="hard-first instead of easy-first")
    args = ap.parse_args()
    texts = read_texts(args.corpus)
    means = read_probe(args.probe)
    if len(texts) != len(means):
        print(f"rows {len(texts)} != probe lines {len(means)}")
        sys.exit(1)
    ranked = sorted(range(len(texts)), key=lambda i: means[i],
                    reverse=args.hard_first)
    order = "hard-first" if args.hard_first else "easy-first"
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        for i in ranked:
            f.write(json.dumps({"text": texts[i]}) + "\n")
    os.rename(tmp, args.out)
    inv = sum(1 for a, b in zip(ranked, sorted(ranked)) if a != b)
    print(f"{order}: {len(texts)} rows -> {args.out} "
          f"(mean {statistics.mean(means):.4f}, "
          f"median {statistics.median(means):.4f} nats, "
          f"{inv} moved)")


if __name__ == "__main__":
    main()
