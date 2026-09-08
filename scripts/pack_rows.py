#!/usr/bin/env python3
"""
pack_rows.py — normalize a phase rows file to the train_run contract.

Contract (matches code_python.txt, which Phase 1 trains on):
  * every row is exactly 2049 ids (T+1, BOS + 2048 targets),
  * id space is the canonical cross-phase mapping (see the
    FallbackEncoder note in prepare_math.py): ASCII 0-127 raw,
    non-ASCII 256+b per UTF-8 byte, BOS 8188.

Steps: translate legacy ids (256-383 -> raw ASCII, 128-255 -> +256,
BOS kept, anything else is a fatal error), concatenate, cut 2049-id
chunks, DROP the trailing partial chunk (train has no mask; padding
with id 0 would train NUL bytes as real signal). Zero-padded tail
rows from older packers are detected (zeros outside the last row =
fatal) and dropped.

Usage:
    python3 scripts/pack_rows.py FILE [FILE ...]   # rewrites in place
"""

import sys

T = 2049
BOS = 8188


def translate(ids, name):
    out = []
    for i in ids:
        if i == BOS:
            out.append(i)
        elif 256 <= i <= 383:
            out.append(i - 256)    # legacy ASCII -> raw byte
        elif 128 <= i <= 255:
            out.append(i + 256)    # legacy non-ASCII -> 256+b
        elif 384 <= i <= 511:
            out.append(i)          # already canonical non-ASCII
        elif 0 <= i <= 127 or 512 <= i < 8192:
            out.append(i)          # already canonical (or BPE piece)
        else:
            raise SystemExit(f"{name}: stray id {i}")
    return out


def pack(path):
    rows = [list(map(int, line.split())) for line in open(path)]
    if not rows:
        raise SystemExit(f"{path}: empty")
    for k, r in enumerate(rows[:-1]):
        if 0 in r:
            raise SystemExit(f"{path}: zeros in non-tail row {k}")
    tail = rows[-1]
    if 0 in tail:
        body = [i for r in rows[:-1] for i in r]
        lost = sum(1 for i in tail if i != 0)
        print(f"  {path}: dropping zero-padded tail ({lost} real ids lost)")
    else:
        body = [i for r in rows for i in r]
    tr = translate(body, path)
    n = len(tr) // T
    if n == 0:
        raise SystemExit(f"{path}: too short ({len(tr)} ids)")
    if len(tr) % T:
        print(f"  {path}: dropping {len(tr) % T} tail ids")
    packed = [tr[k * T:(k + 1) * T] for k in range(n)]
    assert all(len(r) == T for r in packed)
    assert all(i == BOS or 0 <= i <= 127 or 256 <= i <= 511 or
               512 <= i < 8192 for i in tr)
    with open(path, "w") as f:
        for r in packed:
            f.write(" ".join(map(str, r)) + "\n")
    print(f"  {path}: {len(rows)} rows -> {n}x{T} canonical OK")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for path in sys.argv[1:]:
        pack(path)


if __name__ == "__main__":
    main()
