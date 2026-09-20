#!/usr/bin/env python3
"""eval_bpb_dir.py — the bpb of a checkpoint, with the canonical rule.

The app eval_bpb prints the negative log likelihood of each token and no summary.
The rule for the bpb lives in one place: bpb_of() in scripts/fedavg_rounds.py. It
counts the byte length of each target token, so the value is a bpb over bytes and
not a mean over tokens.

This script uses that function. A second implementation here would drift from it.
A wrong parse already produced a bpb of 9.49 for a model at 3.78 nats, which is
worse than chance. One rule, one implementation.

Usage:
  eval_bpb_dir.py --eval BIN --ckpt DIR [--holdout FILE] [--batch 16]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fedavg_rounds import bpb_of  # noqa: E402  the canonical rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", help="the eval_bpb binary (or use --nll)")
    ap.add_argument("--ckpt", help="the checkpoint (or use --nll)")
    ap.add_argument("--nll", help="a file with the raw per-token log likelihoods")
    ap.add_argument("--holdout", default="/tmp/mix/rows_holdout.npy")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--attn", default="blas")
    a = ap.parse_args()

    rows = np.load(a.holdout, mmap_mode="r")
    if a.nll:
        # The eval can run on another machine. Then the raw log likelihoods come
        # here as a file and only the rule runs locally. One rule, one machine.
        ev, tmp = a.nll, None
    else:
        if not a.eval or not a.ckpt:
            print("give --eval and --ckpt, or --nll")
            return 2
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            ev = fh.name
        tmp = ev
        r = subprocess.run([a.eval, "--weights", a.ckpt, "--rows", a.holdout,
                            "--batch", str(a.batch), "--attn", a.attn],
                           stdout=open(ev, "w"), stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print("eval failed for %s (exit %d)" % (a.ckpt, r.returncode))
            sys.exit(1)
    b = bpb_of(ev, rows)
    if tmp:
        os.unlink(tmp)
    print("%.6f" % b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
