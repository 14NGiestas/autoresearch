#!/usr/bin/env python3
"""weight_surgery.py — transformacoes em espaco de pesos + teste de funcao.
Controles: perm paired (prediz IDENTICO), resto (prediz destruicao graduada).
Aceita checkpoint st (model.safetensors) ou npy legado, e escreve cada variante no
MESMO formato da entrada.

Uso: .venv-numpy/bin/python3 scripts/weight_surgery.py --ckpt /tmp/mix/w_f0/best --out /tmp/surgery
"""
import argparse
import os
import shutil
import numpy as np

MAT2D = ("attn_c_q_weight", "attn_c_k_weight", "attn_c_v_weight", "attn_c_proj_weight",
         "mlp_c_fc_weight", "mlp_c_proj_weight")


def load(d):
    import ckio
    return ckio.load_ckpt_dir(d)


def save(W, d, like):
    """Grava no MESMO formato da entrada (st -> model.safetensors)."""
    import ckio
    ckio.save_ckpt_dir(d, W, like=like, op="weight_surgery")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import ckio
    rng = np.random.default_rng(7)
    base = load(args.ckpt)
    arch = open(os.path.join(args.ckpt, "arch.txt")).read()
    out = {}
    # 1. square
    out["square"] = {k: (v.astype(np.float64) ** 2).astype(np.float32)
                     for k, v in base.items()}
    # 2. negate
    out["negate"] = {k: -v for k, v in base.items()}
    # 3/4. scale
    out["scale05"] = {k: (v * 0.5) for k, v in base.items()}
    out["scale20"] = {k: (v * 2.0) for k, v in base.items()}
    # 5. shuffle (unpaired: embaralha cada matriz)
    out["shuffle"] = {}
    for k, v in base.items():
        f = v.ravel().copy()
        rng.shuffle(f)
        out["shuffle"][k] = f.reshape(v.shape)
    # 6/7. ruido gaussiano (sonda de bacia: nitidez)
    for nm, s in (("noise001", 0.01), ("noise01", 0.1)):
        out[nm] = {k: (v + rng.normal(0, s, v.shape).astype(v.dtype))
                   for k, v in base.items()}
    for nm, W in out.items():
        d = os.path.join(args.out, nm)
        save(W, d, args.ckpt)
        open(os.path.join(d, "arch.txt"), "w").write(arch)
        print(f"{nm}: {len(W)} arrays ({ckio.fmt(d)})")
    print("nota: paired-perm (controle exato) exige perm consistente q/k/v/p/fc/p2;",
          "fazer apos ver o padrao de destruicao")


if __name__ == "__main__":
    main()
