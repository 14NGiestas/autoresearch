#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.5.2",
# ]
# ///
"""exact_mean.py — media UNIFORME exata de N checkpoints (e verificacao).

Existe por causa de um bug real: scripts/merge_checkpoints.py calcula
m = ALPHA*A + (1-ALPHA)*B, ou seja ALPHA pondera o PRIMEIRO argumento. A cadeia
de medias do compose_grid.py passava ALPHA=1/k, o que da peso (1-1/k) ao shard
novo -- resultado: o "merge" de K=4 era [1/24,1/24,4/24,18/24] e o de K=8 tinha
0.875 no ultimo shard. Nao era media, era quase um shard so.

Aqui a media e feita direto (soma/N), com verificacao opcional contra os pesos
implicitos que o modelo final teria se tivesse vindo da cadeia errada.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckio  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default="", help="dir de um merge suspeito, para reportar o desvio")
    a = ap.parse_args()
    acc = {}
    opts = {}
    for d in a.dirs:
        ckio.load_ckpt_dir
        w = ckio.load_ckpt_dir(d)
        for k, v in w.items():
            acc[k] = v.astype(np.float64) if k not in acc else acc[k] + v.astype(np.float64)
        for f in ("adam_", "muon_"):
            pass
        st = ckio.load_state(d) if hasattr(ckio, "load_state") else {}
        for k, v in st.items():
            opts[k] = v.astype(np.float64) if k not in opts else opts[k] + v.astype(np.float64)
    n = len(a.dirs)
    mean = {k: (v / n).astype(np.float32) for k, v in acc.items()}
    state = {k: (v / n).astype(np.float32) for k, v in opts.items()}
    os.makedirs(a.out, exist_ok=True)
    ckio.save_ckpt_dir(a.out, mean, state=state, like=a.dirs[0], op="exact_mean",
                       extra_meta={"op": "exact_mean", "n_parents": n})
    print(f"media exata de {n} checkpoints -> {a.out} ({len(mean)} tensores)")
    if a.check and os.path.isdir(a.check):
        w2 = ckio.load_ckpt_dir(a.check)
        num = den = 0.0
        for k in mean:
            x = np.asarray(mean[k], np.float64).ravel()
            y = np.asarray(w2[k], np.float64).ravel()
            num += float(((x - y) ** 2).sum())
            den += float((x ** 2).sum())
        rel = (num / max(den, 1e-30)) ** 0.5
        print(f"  desvio contra {a.check}: rel_l2 = {rel:.6f}"
              f"{'  <== NAO era media (bug da cadeia)' if rel > 0.05 else '  (ok, era media)'}")


if __name__ == "__main__":
    main()
