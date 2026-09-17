#!/usr/bin/env python3
"""compose_metrics.py — mecanismo do imposto de composicao, sem treinar nada.

Mede, nos checkpoints ja salvos:
  * s = ||media|| / media(||shard_i||): encolhimento global e por tensor;
  * cos entre shards (par a par, media): se ~0, os shards sao quase ortogonais
    e a media encolhe ~1/sqrt(K) por construcao (random walk);
  * deslocamento relativo de cada shard em relacao a media, por tensor, para
    ver ONDE a informacao se cancela (atencao vs mlp vs embeddings).

Com isso a pergunta "o imposto e encolhimento ou destruicao?" deixa de ser
retorica: se s(K) ~ 1/sqrt(K) e o dano se concentra onde o sinal encolheu, a
causa e escala efetiva (reversivel por reescala); se s ~ 1 e o dano aparece,
e destruicao de conhecimento (nao reversivel).

Uso:
  scripts/compose_metrics.py --shards /tmp/compose/disj4_s0 .../s1 .../s2 .../s3 \
      --merge /tmp/compose/disj4_merge --label "disj K=4"
"""
import argparse
import glob
import json
import math
import os

import numpy as np

SKIP = ("adam_", "muon_")


def tensors(d):
    return sorted(f for f in os.listdir(d)
                  if f.endswith(".npy") and not f.startswith(SKIP))


def load(d, f):
    return np.load(os.path.join(d, f)).astype(np.float64).ravel()


def group(name):
    """Familia do tensor: emb / lm / attn / mlp, para ver onde o sinal morre."""
    n = name.lower()
    if "wte" in n or "emb" in n:
        return "emb"
    if n.startswith("lm"):
        return "lm_head"
    if any(t in n for t in ("_q", "_k", "_v", ".q", ".k", ".v", "attn")):
        return "attn"
    if any(t in n for t in ("_p", "_fc", "mlp", ".fc", ".p2", "proj")):
        return "mlp"
    return "outro"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--merge", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    import ckio
    files = ckio.require_weights(a.merge, "compose_metrics")
    per = {}
    sq_mean = sq_shard = 0.0
    for f in files:
        m = load(a.merge, f)
        ss = [load(s, f) for s in a.shards]
        nm = float(np.linalg.norm(m))
        ns = float(np.mean([np.linalg.norm(x) for x in ss]))
        sq_mean += nm * nm
        sq_shard += ns * ns
        per[f] = (nm / ns if ns else float("nan"), group(f))

    # coseno medio par a par (no espaco achatado de todos os tensores)
    flat = []
    for s in a.shards:
        flat.append(np.concatenate([load(s, f) for f in files]))
    cs = []
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            x, y = flat[i], flat[j]
            cs.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))))
    del flat

    s_global = math.sqrt(sq_mean) / math.sqrt(sq_shard)
    K = len(a.shards)
    g = {}
    for f, (r, fam) in per.items():
        g.setdefault(fam, []).append(r)
    res = {"label": a.label or os.path.basename(a.merge), "K": K,
           "s_global": s_global, "s_random_walk": 1.0 / math.sqrt(K),
           "cos_medio": float(np.mean(cs)) if cs else None,
           "cos_min": float(np.min(cs)) if cs else None,
           "s_por_grupo": {k: float(np.mean(v)) for k, v in sorted(g.items())},
           "s_por_tensor": {k: v[0] for k, v in sorted(per.items())}}
    print(f"== {res['label']}  (K={K})")
    print(f"   s = ||media||/||shard|| = {s_global:.4f}   "
          f"(random walk 1/sqrt(K) = {res['s_random_walk']:.4f})")
    if cs:
        print(f"   cos entre shards: medio {res['cos_medio']:+.4f}  "
              f"min {res['cos_min']:+.4f}")
    print("   s por familia: " + "  ".join(
        f"{k}={v:.3f}" for k, v in res["s_por_grupo"].items()))
    if a.json_out:
        with open(a.json_out, "a") as fh:
            fh.write(json.dumps(res, sort_keys=True) + "\n")
        print(f"   -> {a.json_out}")


if __name__ == "__main__":
    main()
