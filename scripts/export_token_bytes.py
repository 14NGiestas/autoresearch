#!/usr/bin/env python3
"""export_token_bytes.py — one-time byte-length table for exact val-bpb
in Fortran (train_run app). Mirrors prepare.py: special tokens -> 0,
else utf-8 length of decode([i]). One int per line, 8192 lines.

Usage: .venv-numpy/bin/python3 scripts/export_token_bytes.py <out_path>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_driver import load_enc  # noqa: E402

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
SPECIALS = [f"<|reserved_{i}|>" for i in range(4)]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        CACHE, "tok_tables", "token_bytes.txt")
    enc = load_enc()
    try:
        with open(out, "w") as f:
            for i in range(enc.n_vocab):
                s = enc.decode([i])
                f.write(f"{0 if s in SPECIALS else len(s.encode('utf-8'))}\n")
    except OSError as e:
        raise SystemExit(f"cannot write {out}: {e}")
    print(f"wrote {enc.n_vocab} byte lengths to {out}")


if __name__ == "__main__":
    main()
