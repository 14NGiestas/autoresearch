#!/usr/bin/env python3
"""eval_driver.py — torch-free bits-per-byte eval of the Fortran engine.

Replicates prepare.py:evaluate_bpb's data side (pinned val shard,
BOS-prepended best-fit packing) and metric (byte-weighted CE -> bpb).
The forward pass runs in Fortran (src/app/eval_bpb); this driver packs
rows, feeds them, and scores the per-position NLLs it prints.

  .venv-numpy/bin/python3 scripts/eval_driver.py --rows 4 [--seq 2048]

Needs: pyarrow, tiktoken (torch-free). Respects MAX_SEQ_LEN=2048 default.
"""
import argparse
import math
import os
import pickle
import io
import subprocess
import sys

_ALLOW_MODULES = ("tiktoken.", "builtins", "copyreg")


class RestrictedUnpickler(pickle.Unpickler):
    """Allowlist unpickler: only our own tiktoken tokenizer artifact."""

    def find_class(self, module, name):
        if module == "builtins" or module.startswith("tiktoken.") \
                or module == "copyreg":
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def restricted_load(path):
    try:
        with open(path, "rb") as f:
            return RestrictedUnpickler(f).load()
    except (OSError, pickle.PickleError) as e:
        raise SystemExit(f"cannot load tokenizer: {e}")

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
VAL_SHARD = os.path.join(CACHE, "data", "shard_06542.parquet")
TOK_PKL = os.path.join(CACHE, "tokenizer", "tokenizer.pkl")
SPECIALS = [f"<|reserved_{i}|>" for i in range(4)]


def load_enc():
    return restricted_load(TOK_PKL)


def token_bytes(enc):
    """Byte length per token id (0 for specials) — mirrors prepare.py."""
    tbl = []
    for i in range(enc.n_vocab):
        s = enc.decode([i])
        tbl.append(0 if s in SPECIALS else len(s.encode("utf-8")))
    return tbl


def pack_rows(enc, bos, T, n_rows, buffer_size=1000, doc_offset=0):
    """BOS-aligned best-fit packing replica. Yields lists of T+1 ids."""
    try:
        import importlib
        pq = importlib.import_module("pyarrow.parquet")
    except ImportError:
        raise SystemExit("pyarrow required: use .venv-numpy/bin/python3")
    cap = T + 1
    pf = pq.ParquetFile(VAL_SHARD)
    texts = []
    for rg in range(pf.num_row_groups):
        texts.extend(pf.read_row_group(rg).column("text").to_pylist())
    texts = texts[doc_offset:]
    encoded = enc.encode_ordinary_batch(texts[:buffer_size])
    docs = [[bos] + e for e in encoded]
    di = buffer_size
    rows = []
    row, pos = [0] * cap, 0
    while len(rows) < n_rows:
        if not docs:
            more = enc.encode_ordinary_batch(texts[di:di + buffer_size])
            docs = [[bos] + e for e in more]
            di += buffer_size
            if not docs:
                break
        rem = cap - pos
        best, best_len = -1, 0
        for i, d in enumerate(docs):
            if len(d) <= rem and len(d) > best_len:
                best, best_len = i, len(d)
        if best >= 0:
            d = docs.pop(best)
            row[pos:pos + len(d)] = d
            pos += len(d)
        else:
            si = min(range(len(docs)), key=lambda i: len(docs[i]))
            d = docs.pop(si)
            row[pos:pos + rem] = d[:rem]
            pos += rem
        if pos == cap:
            rows.append(row)
            row, pos = [0] * cap, 0
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--bin", default=None)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--chunk-timeout", type=int, default=1500)
    ap.add_argument("--doc-offset", type=int, default=0)
    args = ap.parse_args()

    enc = load_enc()
    bos = enc.encode_single_token("<|reserved_0|>")
    tbytes = token_bytes(enc)
    rows = pack_rows(enc, bos, args.seq, args.start + args.rows,
                      doc_offset=args.doc_offset)[args.start:]
    print(f"packed {len(rows)} rows of {args.seq + 1} ids" +
          (f" (offset {args.start})" if args.start else ""), flush=True)
    if not rows:
        raise SystemExit("nothing to do")

    try:
        for ci in range(0, len(rows), args.chunk):
            with open(f"/tmp/eval_rows_{ci}.txt", "w") as f:
                for r in rows[ci:ci + args.chunk]:
                    f.write(" ".join(map(str, r)) + "\n")
    except OSError as e:
        raise SystemExit(f"cannot write rows: {e}")

    import shutil
    fpm = args.bin or shutil.which("fortran-fpm")
    if fpm is None:
        raise SystemExit("fortran-fpm not on PATH; run inside nix develop")
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "8")
    nll_lines = []
    for ci in range(0, len(rows), args.chunk):
        rows_file = f"/tmp/eval_rows_{ci}.txt"
        try:
            if args.bin:
                cmd = [fpm, os.path.join(CACHE, "weights_depth12"),
                       rows_file]
            else:
                cmd = [fpm, "run", "eval_bpb", "--",
                       os.path.join(CACHE, "weights_depth12"), rows_file]
            p = subprocess.run(
                cmd, capture_output=True, text=True, env=env, cwd="src",
                timeout=args.chunk_timeout)
        except OSError as e:
            raise SystemExit(f"eval binary failed: {e}")
        except subprocess.TimeoutExpired:
            raise SystemExit(
                f"chunk rows {args.start + ci}-" +
                f"{args.start + ci + args.chunk} timed out; resume with " +
                f"--start {args.start + ci}")
        if p.returncode != 0:
            raise SystemExit(
                f"eval_bpb exit {p.returncode}:\n{p.stderr[-2000:]}")
        for line in p.stdout.strip().split("\n"):
            try:
                vals = list(map(float, line.split()))
            except ValueError:
                continue
            if len(vals) == args.seq:
                nll_lines.append(vals)
        print(f"chunk done: {len(nll_lines)}/{len(rows)} rows", flush=True)
    if len(nll_lines) != len(rows):
        raise SystemExit(
            f"expected {len(rows)} NLL rows, got {len(nll_lines)}")
    total_nats, total_bytes = 0.0, 0
    for ri, (nlls, r) in enumerate(zip(nll_lines, rows)):
        rnats, rbytes = 0.0, 0
        for nll, tid in zip(nlls, r[1:]):
            b = tbytes[tid]
            if b > 0:
                rnats += nll
                rbytes += b
        print(f"row {ri}: bpb={rnats / (math.log(2) * rbytes):.4f}",
              flush=True)
        total_nats += rnats
        total_bytes += rbytes
    bpb = total_nats / (math.log(2) * total_bytes)
    print(f"rows={len(rows)} nats={total_nats:.1f} bytes={total_bytes}")
    print(f"FORTRAN VAL BPB = {bpb:.6f}  (torch checkpoint tag: 0.3750)")


if __name__ == "__main__":
    main()
