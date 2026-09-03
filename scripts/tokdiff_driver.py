#!/usr/bin/env python3
"""tokdiff_driver.py — differential test: Fortran tokenizer vs tiktoken.

Samples N docs from the val shard, writes each byte-exact to a file, runs
src/app/tokdiff (pure Fortran encode), and diffs every id against
tiktoken's encode_ordinary. Any mismatch = scanner/BPE bug. Exit nonzero
on first diff (prints context).

  .venv-numpy/bin/python3 scripts/tokdiff_driver.py --n 200
"""
import argparse
import importlib
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_driver import load_enc  # noqa: E402

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
VAL_SHARD = os.path.join(CACHE, "data", "shard_06542.parquet")
TABLES = os.path.join(CACHE, "tok_tables")
WORK = "/tmp/tokdiff_docs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    pq = importlib.import_module("pyarrow.parquet")
    texts = []
    pf = pq.ParquetFile(VAL_SHARD)
    for rg in range(pf.num_row_groups):
        texts.extend(pf.read_row_group(rg).column("text").to_pylist())
        if len(texts) >= args.n + 5:
            break
    # diverse sample: head + strided middle (exercises unicode/punct/numbers)
    docs = [t for t in (texts[: args.n // 2] +
                        texts[len(texts) // 2:: max(1, len(texts) // args.n)][: args.n // 2])
            if t]
    docs = docs[: args.n]
    print(f"sampled {len(docs)} non-empty docs", flush=True)

    try:
        os.makedirs(WORK, exist_ok=True)
    except OSError as e:
        raise SystemExit(f"cannot create {WORK}: {e}")
    paths = []
    try:
        for i, t in enumerate(docs):
            p = os.path.join(WORK, f"doc_{i:04d}.txt")
            with open(p, "wb") as f:
                f.write(t.encode("utf-8"))
            paths.append(p)
        with open(os.path.join(WORK, "list.txt"), "w") as f:
            f.write("\n".join(paths) + "\n")
    except OSError as e:
        raise SystemExit(f"cannot write diff docs: {e}")

    fpm = shutil.which("fortran-fpm")
    if fpm is None:
        raise SystemExit("fortran-fpm not on PATH; run inside nix develop")
    env = dict(os.environ)
    try:
        p = subprocess.run(
            [fpm, "run", "tokdiff", "--", TABLES,
             os.path.join(WORK, "list.txt")],
            capture_output=True, text=True, env=env, cwd="src",
            timeout=1200)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"tokdiff failed: {e}")
    if p.returncode != 0:
        raise SystemExit(f"tokdiff exit {p.returncode}:\n{p.stderr[-2000:]}")

    enc = load_enc()
    # tokdiff prints one int-list line per doc; fpm chatter is not int-lists.
    got = []
    for ln in p.stdout.split("\n"):
        if not ln.strip():
            continue
        try:
            list(map(int, ln.split()))
        except ValueError:
            continue
        got.append(ln)
    if len(got) != len(docs):
        raise SystemExit(f"line count {len(got)} != docs {len(docs)}")
    bad = 0
    for i, (doc, line) in enumerate(zip(docs, got)):
        want = enc.encode_ordinary(doc)
        have = list(map(int, line.split())) if line.strip() else []
        if want != have:
            bad += 1
            print(f"DIFF doc {i} ({len(doc)} chars):")
            print(f"  text: {doc[:120]!r}")
            print(f"  want ({len(want)}): {want[:20]}")
            print(f"  have ({len(have)}): {have[:20]}")
            if bad >= 3:
                break
    if bad:
        raise SystemExit(f"{bad} diffs — tokenizer NOT exact")
    print(f"EXACT: all {len(docs)} docs, "
          f"{sum(len(enc.encode_ordinary(d)) for d in docs)} tokens match")


if __name__ == "__main__":
    main()
