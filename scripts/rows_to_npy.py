#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.5.2",
# ]
# ///
"""rows_to_npy.py — converte rows texto (ids separados por espaco) para npy 2D.

Formato alvo: int32 little-endian, shape (N, TT+1), row-major C, um arquivo
por corpus + sidecar .json com a regua (md5 da fonte, n_rows, tt). Indices de
linha sobrevivem a migracao (ancoras sao formato-agnosticas).

Portao: --gate recarrega o npy e exige ids bit-identicos ao texto.

Uso:
  .venv-numpy/bin/python3 scripts/rows_to_npy.py --in /tmp/mix/val_f0.txt \
      --out /tmp/mix/val_f0.npy --gate
"""
import argparse
import hashlib
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()

    with open(args.inp, "rb") as f:
        raw = f.read()
    md5 = hashlib.md5(raw).hexdigest()
    rows = [[int(x) for x in ln.split()] for ln in raw.decode().splitlines()
            if ln.strip()]
    tt = len(rows[0]) - 1
    assert all(len(r) == tt + 1 for r in rows), "largura variavel!"
    assert all(0 <= v < 2 ** 31 for r in rows for v in r), "id fora do int32!"
    a = np.asfortranarray(np.asarray(rows, dtype=np.int32))
    np.save(args.out, a)

    sidecar = {"source": os.path.basename(args.inp), "md5_source": md5,
               "n_rows": int(a.shape[1]), "tt": tt, "dtype": "int32",
               "layout": "(N,TT+1) Fortran-order"}
    with open(args.out.replace(".npy", ".json"), "w") as f:
        json.dump(sidecar, f, indent=1)
    print(f"{args.inp} -> {args.out}: shape={a.shape} "
          f"MB={a.nbytes / 1e6:.1f} md5={md5[:12]}")

    if args.gate:
        b = np.load(args.out)
        assert b.shape == a.shape and b.dtype == a.dtype
        assert (b == a).all(), "round-trip divergiu!"
        print("gate OK: ids bit-identicos")


if __name__ == "__main__":
    main()
