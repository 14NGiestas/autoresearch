#!/usr/bin/env python3
"""export_tokenizer.py — one-time export of tokenizer tables for the
pure-Fortran tokenizer (src/lib/tokenizer_*.f90). Runtime needs no Python.

Reads the training tokenizer.pkl (tiktoken) + unicodedata (stdlib) and writes:
  ranks.txt         — 8188 lines: <rank> <nbytes> <b0> .. (bytes as uint8 ints)
  unicode_L.txt     — codepoint ranges with General_Category Letter (\\p{L})
  unicode_N.txt     — codepoint ranges with General_Category Number (\\p{N})

Whitespace (\\s = Unicode White_Space, 25 fixed codepoints) is hardcoded in
Fortran — no export needed. Specials (<|reserved_i|>, BOS=8188) are constants.

Usage: .venv-numpy/bin/python3 scripts/export_tokenizer.py <out_dir>
"""
import os
import pickle
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_driver import restricted_load  # noqa: E402 (allowlisted unpickler)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.expanduser("~"), ".cache", "autoresearch", "tok_tables")
    try:
        os.makedirs(outdir, exist_ok=True)
    except OSError as e:
        raise SystemExit(f"cannot create {outdir}: {e}")
    try:
        enc = restricted_load(os.path.join(
            os.path.expanduser("~"), ".cache", "autoresearch",
            "tokenizer", "tokenizer.pkl"))
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"cannot load tokenizer: {e}")
    mr = enc._mergeable_ranks
    assert len(mr) == 8188, len(mr)
    assert set(mr.values()) == set(range(8188)), "ranks not dense 0..8187"
    by_rank = sorted(mr.items(), key=lambda kv: kv[1])
    try:
        with open(os.path.join(outdir, "ranks.txt"), "w") as f:
            f.write("8188\n")
            for tok, r in by_rank:
                f.write(f"{r} {len(tok)}" + "".join(f" {b}" for b in tok) + "\n")
    except OSError as e:
        raise SystemExit(f"cannot write ranks: {e}")
    print(f"ranks.txt: 8188 tokens, maxlen {max(len(t) for t in mr)}")

    for prop, fname in (("L", "unicode_L.txt"), ("N", "unicode_N.txt")):
        ranges = []
        start = None
        for cp in range(0x110000):
            is_in = unicodedata.category(chr(cp)).startswith(prop)
            if is_in and start is None:
                start = cp
            elif not is_in and start is not None:
                ranges.append((start, cp - 1))
                start = None
        if start is not None:
            ranges.append((start, 0x10FFFF))
        try:
            with open(os.path.join(outdir, fname), "w") as f:
                f.write(f"{len(ranges)}\n")
                for lo, hi in ranges:
                    f.write(f"{lo} {hi}\n")
        except OSError as e:
            raise SystemExit(f"cannot write {fname}: {e}")
        print(f"{fname}: {len(ranges)} ranges")
    print("done:", outdir)


if __name__ == "__main__":
    main()
