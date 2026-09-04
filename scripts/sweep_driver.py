#!/usr/bin/env python3
"""sweep_driver.py — overthinking-curve experiment (hyp_ba98cd), torch-free.

Packs val rows (same replica as eval_driver), runs src/app/recur_sweep
(real depth-12 layer 0 tied for N loops + real wte/lm_head), and reports
bpb per loop count. Predicted shape per Kohli et al.: improve-then-degrade
(U-curve). Monotonic worsening = recurrence can't be bolted on.

  .venv-numpy/bin/python3 scripts/sweep_driver.py --rows 8 --loops 1 2 3 4 6 8 12

Reuses eval_driver's packing/metric (import, no duplication).
"""
import argparse
import math
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_driver import (  # noqa: E402
    load_enc, token_bytes, pack_rows, CACHE)

DEF_LOOPS = [1, 2, 3, 4, 6, 8, 12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--loops", type=int, nargs="+", default=DEF_LOOPS)
    ap.add_argument("--doc-offset", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=5400)
    args = ap.parse_args()

    enc = load_enc()
    bos = enc.encode_single_token("<|reserved_0|>")
    tbytes = token_bytes(enc)
    rows = pack_rows(enc, bos, args.seq, args.rows,
                     doc_offset=args.doc_offset)
    print(f"packed {len(rows)} rows", flush=True)
    if not rows:
        raise SystemExit("nothing to do")

    rows_file = "/tmp/sweep_rows.txt"
    try:
        with open(rows_file, "w") as f:
            for r in rows:
                f.write(" ".join(map(str, r)) + "\n")
    except OSError as e:
        raise SystemExit(f"cannot write rows: {e}")

    fpm = shutil.which("fortran-fpm")
    if fpm is None:
        raise SystemExit("fortran-fpm not on PATH; run inside nix develop")
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "8")
    try:
        p = subprocess.run(
            [fpm, "run", "recur_sweep", "--",
             os.path.join(CACHE, "weights_depth12"), rows_file,
             *[str(n) for n in args.loops]],
            capture_output=True, text=True, env=env, cwd="src",
            timeout=args.timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"sweep failed: {e}\nresume manually; rows file kept")
    if p.returncode != 0:
        raise SystemExit(f"recur_sweep exit {p.returncode}:\n{p.stderr[-2000:]}")

    # parse: ROW <r> LOOPS <n>  \n  <T floats>
    nats = {n: 0.0 for n in args.loops}
    nbytes = {n: 0 for n in args.loops}
    cur = None
    for line in p.stdout.strip().split("\n"):
        if line.startswith("ROW "):
            try:
                _, r, _, n = line.split()
                cur = (int(r), int(n))
            except (ValueError, IndexError):
                cur = None
            continue
        try:
            vals = list(map(float, line.split()))
        except ValueError:
            continue
        if cur is None or len(vals) != args.seq:
            continue
        r, n = cur
        for nll, tid in zip(vals, rows[r][1:]):
            b = tbytes[tid]
            if b > 0:
                nats[n] += nll
                nbytes[n] += b
    print("loop-count -> bpb:")
    curve = []
    for n in args.loops:
        try:
            bpb = nats[n] / (math.log(2) * nbytes[n])
        except ZeroDivisionError:
            bpb = float("nan")
        curve.append(bpb)
        print(f"  N={n:<3d} bpb={bpb:.4f}")
    lo = min(range(len(curve)), key=lambda i: curve[i])
    if lo > 0 and lo < len(curve) - 1:
        print("U-CURVE: minimum interior at N=%d (extrapolate+overthink)"
              % args.loops[lo])
    elif lo == len(curve) - 1:
        print("MONOTONIC-IMPROVING up to N=%d (no overthinking seen yet)"
              % args.loops[lo])
    else:
        print("MONOTONIC-WORSENING from N=1 (bolt-on fails: train it in)")


if __name__ == "__main__":
    main()
