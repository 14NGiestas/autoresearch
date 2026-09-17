#!/usr/bin/env python3
"""rebasin.py — alinha B em A por permutacao de heads (Git Re-Basin guloso).
Por camada: 72 perms validas (intra-grupo GQA x swap) com composicao do
stream; escolhe a de menor drift local. Uso:
  .venv-numpy/bin/python3 scripts/rebasin.py --a /tmp/w_ovA_best --b /tmp/w_ovB_best --out /tmp/w_ovB_al
Imprime drift antes/depois.
"""
import argparse
import os
import itertools
import numpy as np

D, nh, hd, kvd, dff, V, NL = 96, 6, 16, 32, 384, 8192, 12


def perms72():
    out = []
    for g0 in itertools.permutations(range(3)):
        for g1 in itertools.permutations(range(3)):
            out.append(tuple(g0) + tuple(h + 3 for h in g1))
            out.append(tuple(h + 3 for h in g1) + tuple(g0))
    return out


def blk(ph):
    return np.concatenate([np.arange(h * hd, (h + 1) * hd) for h in ph])


def load(d):
    """Pesos por nome logico (st | npy legado); as chaves sao os nomes .npy, que
    e o que as permutacoes abaixo indexam."""
    import ckio
    return ckio.load_ckpt_dir(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    A = load(args.a)
    B = load(args.b)
    arch = open(os.path.join(args.a, "arch.txt")).read()

    def M(W, k, r, c):
        return W[k].reshape(r, c)

    def layer_drift(l, Bt):
        s = 0.0
        p = f"transformer_h_{l}"
        for nm, sh in (("attn_c_q_weight", (96, D)), ("attn_c_k_weight", (kvd, D)),
                       ("attn_c_v_weight", (kvd, D)), ("attn_c_proj_weight", (D, 96)),
                       ("mlp_c_fc_weight", (dff, D)), ("mlp_c_proj_weight", (D, dff))):
            d = M(A, p + "_" + nm + ".npy", *sh) - M(Bt, p + "_" + nm + ".npy", *sh)
            s += float((d ** 2).sum())
        return s

    P72 = perms72()
    assert len(P72) == 72
    Bt = dict(B)
    Pin = np.arange(D)  # stream atual de B em coords de A
    for l in range(NL):
        p = f"transformer_h_{l}"
        best = None
        for ph in P72:
            Ph = blk(ph)
            kvph = [0, 1] if (list(ph[:3]) == sorted(ph[:3]) and ph[0] < 3) else None
            # swap detect: grupos trocaram?
            first_grp = 0 if ph[0] < 3 else 1
            ok = all((h < 3) == (first_grp == 0) for h in ph[:3]) and \
                all((h < 3) == (first_grp == 1) for h in ph[3:])
            if not ok:
                continue
            Pk = blk([0, 1] if first_grp == 0 else [1, 0])
            T = dict(Bt)
            T[p + "_attn_c_q_weight.npy"] = M(Bt, p + "_attn_c_q_weight.npy", 96, D)[Ph][:, Pin].ravel()
            T[p + "_attn_c_k_weight.npy"] = M(Bt, p + "_attn_c_k_weight.npy", kvd, D)[Pk][:, Pin].ravel()
            T[p + "_attn_c_v_weight.npy"] = M(Bt, p + "_attn_c_v_weight.npy", kvd, D)[Pk][:, Pin].ravel()
            Pout = blk(ph)
            T[p + "_attn_c_proj_weight.npy"] = M(Bt, p + "_attn_c_proj_weight.npy", D, 96)[Pout][:, Ph].ravel()
            T[p + "_mlp_c_fc_weight.npy"] = M(Bt, p + "_mlp_c_fc_weight.npy", dff, D)[:, Pout].ravel()
            T[p + "_mlp_c_proj_weight.npy"] = M(Bt, p + "_mlp_c_proj_weight.npy", D, dff)[Pout].ravel()
            s = layer_drift(l, T)
            if best is None or s < best[0]:
                best = (s, T, Pout)
        Bt.update({k: v for k, v in best[1].items() if f"transformer_h_{l}_" in k})
        Pin = best[2]
        print(f"layer {l}: drift-local {best[0]:.4f}", flush=True)
    # lm_head acompanha o stream final; wte fica (Pin inicial = identidade)
    Bt["lm_head_weight.npy"] = M(Bt, "lm_head_weight.npy", V, D)[:, Pin].ravel()
    import ckio
    out_w = {k: v.astype(np.float32) for k, v in Bt.items() if k.endswith(".npy")}
    ckio.save_ckpt_dir(args.out, out_w, like=args.a, op="rebasin",
                       extra_meta={"op": "rebasin", "a": args.a, "b": args.b})
    open(os.path.join(args.out, "arch.txt"), "w").write(arch)

    A = ckio.load_ckpt_dir(args.a)
    B2 = ckio.load_ckpt_dir(args.out)
    sq = dr = 0.0
    for f in sorted(A):
        a = A[f].astype(np.float64).ravel()
        b = B2[f].astype(np.float64).ravel()
        sq += float(((a - b) ** 2).sum())
        dr += float((a ** 2).sum())
    print(f"drift antes=1.083 depois={(sq/max(dr,1e-30))**0.5:.6f}")


if __name__ == "__main__":
    main()
