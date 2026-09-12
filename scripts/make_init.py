#!/usr/bin/env python3
"""make_init.py — fresh random init for a Fortran GPT checkpoint dir.

Replicates the measured convention of /tmp/mix/init3m (the 3M sweep init):
every weight tensor ~ N(0, 0.02^2), float32, flat C-order .npy; no bias files
(the loader defaults them); no adam_*.npy (their absence means fresh moments,
same as merge_ckpt's explicit-zero convention). arch.txt carries the 8 fields
require_arch validates; template.txt is copied verbatim (arch-independent).

Usage: make_init.py --out DIR --d 216 --heads 6 --kv 2 --layers 12
                    --vocab 8192 --ctx 1024 --seed 7
"""
import argparse
import os
import shutil
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--kv", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--scale", type=float, default=0.02)
    ap.add_argument("--template-file", default="/tmp/mix/init3m/template.txt")
    a = ap.parse_args()
    if a.d % a.heads != 0:
        print(f"refused: d={a.d} not divisible by heads={a.heads}", file=sys.stderr)
        return 2
    if not (1 <= a.kv <= a.heads):
        print(f"refused: kv={a.kv} outside 1..{a.heads}", file=sys.stderr)
        return 2
    hd = a.d // a.heads
    kvd = a.kv * hd
    ffd = 4 * a.d
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    def w(name, n):
        arr = (rng.standard_normal(n) * a.scale).astype(np.float32)
        np.save(os.path.join(a.out, name), arr)

    w("transformer_wte_weight.npy", a.vocab * a.d)
    w("lm_head_weight.npy", a.d * a.vocab)
    for L in range(a.layers):
        p = f"transformer_h_{L}_"
        w(p + "attn_c_q_weight.npy", a.d * a.d)
        w(p + "attn_c_k_weight.npy", a.d * kvd)
        w(p + "attn_c_v_weight.npy", a.d * kvd)
        w(p + "attn_c_proj_weight.npy", a.d * a.d)
        w(p + "mlp_c_fc_weight.npy", a.d * ffd)
        w(p + "mlp_c_proj_weight.npy", ffd * a.d)
    with open(os.path.join(a.out, "arch.txt"), "w") as f:
        f.write(f"d_model = {a.d}\n")
        f.write(f"n_head = {a.heads}\n")
        f.write(f"n_kv = {a.kv}\n")
        f.write(f"n_layer = {a.layers}\n")
        f.write(f"vocab = {a.vocab}\n")
        f.write(f"ctx = {a.ctx}\n")
        f.write("bos = 8188\n")
        f.write(f"head_dim = {hd}\n")
    shutil.copy(a.template_file, os.path.join(a.out, "template.txt"))
    print(f"init d={a.d} h={a.heads} kv={a.kv} L={a.layers} V={a.vocab} "
          f"seed={a.seed} -> {a.out} (76 files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
