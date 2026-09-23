#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.5.2",
# ]
# ///
"""rescale_ckpt.py — multiplica todos os pesos de um checkpoint por uma constante.

Teste de mecanismo do imposto de composicao: se a media de K shards encolhe a
norma (medimos s = ||media||/||shard|| = 0.88-0.90) e a perda for sensivel a
escala (ja sabemos: vale 0.9-1.1, penhasco acima), entao parte do "imposto" e
apenas escala efetiva subotima e DESAPARECE se reescalarmos a media. Se nao
desaparecer, o imposto e cancelamento de conhecimento (nao reversivel).

Escreve no MESMO formato da entrada (st -> model.safetensors, npy legado -> .npy)
e carrega o estado do otimizador sem tocar nele.

Uso: scripts/rescale_ckpt.py ENTRADA SAIDA --scale 1.1 [--only wte,lm]
"""
import argparse
import os
import shutil

import numpy as np


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
    import ckio
    import st_read

    # `--only` compara com o nome LOGICO (transformer_h_0_attn_c_q_weight.npy) e
    # tambem com o nome CANONICO (l0.q, wte, lm): antes so o nome de arquivo era
    # testado, entao `--only wte` nao casava com nada e a reescala saia identica
    # em silencio.
    def selecionado(nome):
        if not only:
            return True
        return any(nome.startswith(pr) or (st_read.canon_of(nome) or "").startswith(pr)
                   for pr in only)

    W = ckio.load_ckpt_dir(a.src)
    out = {}
    for f, x in W.items():
        if selecionado(f):
            x = (x.astype(np.float64) * a.scale).astype(np.float32)
            n_t += 1
        out[f] = x
    out_fmt = ckio.save_ckpt_dir(a.dst, out, state=ckio.load_state(a.src), like=a.src,
                                 op="rescale_ckpt",
                                 extra_meta={"op": "rescale", "scale": a.scale,
                                             "only": a.only, "src": a.src})
    for f in sorted(os.listdir(a.src)):
        if f.endswith((".txt", ".json")):
            shutil.copy(os.path.join(a.src, f), os.path.join(a.dst, f))
    print(f"{a.src} x{a.scale} -> {a.dst}  ({n_t} tensores escalados, formato {out_fmt})")


if __name__ == "__main__":
    main()
