#!/usr/bin/env python3
"""chat_driver.py — one-command torch-free inference with the Fortran engine.

Encodes text with the training tokenizer (tiktoken, no torch), runs
src/app/chat.f90 via `fortran-fpm run chat`, decodes the generated ids.

  .venv-numpy/bin/python3 scripts/chat_driver.py "Alan Turing theorized that computers" --n 30

Needs: tiktoken (in .venv-numpy), fortran-fpm on PATH (nix develop),
weights in ~/.cache/autoresearch/weights_depth12 (scripts/export_weights.py).
"""
import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_driver import load_enc  # noqa: E402  (trusted local artifact loader)

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
WDIR = os.path.join(CACHE, "weights_depth12")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--weights", default=WDIR)
    args = ap.parse_args()

    enc = load_enc()
    ids = enc.encode_ordinary(args.prompt)
    print(f"prompt ids ({len(ids)}): {ids}", flush=True)

    fpm = shutil.which("fortran-fpm")
    if fpm is None:
        raise SystemExit("fortran-fpm not on PATH; run inside nix develop")
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "8")
    try:
        p = subprocess.run(
            [fpm, "run", "chat", "--", args.weights, str(args.n),
             *[str(i) for i in ids]],
            capture_output=True, text=True, env=env, cwd="src",
            timeout=3600)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"chat failed: {e}")
    if p.returncode != 0:
        raise SystemExit(f"chat exit {p.returncode}:\n{p.stderr[-2000:]}")

    gen = None
    for line in p.stdout.strip().split("\n"):
        try:
            vals = list(map(int, line.split()))
        except ValueError:
            continue
        if vals:
            gen = vals  # last int-only line is the generated ids
    if not gen:
        raise SystemExit(f"no ids in chat output:\n{p.stdout[-500:]}")
    print(f"gen ids ({len(gen)}): {gen}")
    print("---")
    print(enc.decode(ids + gen))


if __name__ == "__main__":
    main()
