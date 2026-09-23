#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pyarrow==25.0.1",
# ]
# ///
"""tokenize_budget.py — jsonl -> linhas BPE ate orcamento de tokens (stdlib).
Fontes: --fmt wiki (title/text), --fmt so (q/qb/a), --fmt instr (text).
Uso: .venv-numpy/bin/python3 scripts/tokenize_budget.py --fmt so \
  --src /tmp/bench/so_pairs.jsonl --budget-tokens 25e6 --out /tmp/mix200/so_bpe.txt
"""
import argparse
import json
import sys

sys.path.insert(0, "scripts")
from eval_driver import load_enc  # noqa: E402

BOS = 8188


def texts(fmt, path):
    if fmt == "wiki":
        for line in open(path, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = d.get("text")
            if t:
                yield t
    elif fmt == "so":
        for line in open(path, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield f"{d.get('q','')} {d.get('qb','')} {d.get('a','')}"
    elif fmt == "instr":
        import glob
        for p in sorted(glob.glob(path)):
            import pyarrow.parquet as pq
            for r in pq.read_table(p, columns=["text"]).to_pylist():
                if r.get("text"):
                    yield r["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--budget-tokens", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    enc = load_enc()
    used = n = 0
    with open(args.out, "w") as out:
        for t in texts(args.fmt, args.src):
            if used >= args.budget_tokens:
                break
            ids = enc.encode_ordinary(t)
            used += len(ids)
            out.write(f"{BOS} " + " ".join(map(str, ids)) + "\n")
            n += 1
            if n % 20000 == 0:
                print(f"  {n} docs, {used/1e6:.1f}M toks...", flush=True)
    print(f"{args.out}: {n} docs, {used/1e6:.1f}M toks")


if __name__ == "__main__":
    main()
