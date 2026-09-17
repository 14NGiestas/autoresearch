#!/usr/bin/env python3
"""rescale_ckpt.py — multiplica todos os pesos de um checkpoint por uma constante.

Teste de mecanismo do imposto de composicao: se a media de K shards encolhe a
norma (medimos s = ||media||/||shard|| = 0.88-0.90) e a perda for sensivel a
escala (ja sabemos: vale 0.9-1.1, penhasco acima), entao parte do "imposto" e
apenas escala efetiva subotima e DESAPARECE se reescalarmos a media. Se nao
desaparecer, o imposto e cancelamento de conhecimento (nao reversivel).

Uso: scripts/rescale_ckpt.py ENTRADA SAIDA --scale 1.1 [--only wte,lm]
"""
import argparse
import os
import shutil

import numpy as np

SKIP = ("adam_", "muon_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--only", default="", help="prefixos de tensores a escalar")
    a = ap.parse_args()

    only = [x for x in a.only.split(",") if x]
    os.makedirs(a.dst, exist_ok=True)
    n_t = 0
    for f in sorted(os.listdir(a.src)):
        p = os.path.join(a.src, f)
        if f.endswith((".txt", ".json")):
            shutil.copy(p, os.path.join(a.dst, f))
            continue
        if f.startswith(SKIP) or not f.endswith(".npy"):
            continue
        x = np.load(p)
        if not only or any(f.startswith(pr) for pr in only):
            x = (x.astype(np.float64) * a.scale).astype(np.float32)
            n_t += 1
        np.save(os.path.join(a.dst, f), x)
    shutil.copy(os.path.join(a.src, "arch.txt"), os.path.join(a.dst, "arch.txt")) \
        if not os.path.exists(os.path.join(a.dst, "arch.txt")) else None
    print(f"{a.src} x{a.scale} -> {a.dst}  ({n_t} tensores escalados)")


if __name__ == "__main__":
    main()
