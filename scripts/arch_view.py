#!/usr/bin/env python3
"""arch_view.py — retrato da arquitetura: diagrama, tensores, derivados, identidade.

Responde "como nossa arquitetura está AGORA?" com UMA saida, de um lugar so':
o diagrama do bloco com as dimensoes anotadas, a tabela de tensores (shape,
params), os derivados que decidem custo (FLOPs/token, KV bytes/token, ativacao
bytes/token, estado do otimizador) e a IDENTIDADE (canonica + id).

Por que existe: a config mora em tres lugares (defaults do fortran_arch.f90,
__metadata__ do safetensors e arch.txt) e ja' aconteceu da arvore ficar numa arch
e os experimentos em outra (d216 vs d96). Um retrato unico, com a identidade
impressa, e' o que faz a divergencia aparecer em vez de esperar que alguem
confira campo a campo.

Uso:
    scripts/arch_view.py [DIR|arch.txt] [--svg FILE] [--repo-default]

DIR pode ser um checkpoint (.safetensors com metadata, ou .npy + arch.txt). Sem
argumento, vale a config do repositorio (defaults compilados nao sao visiveis
aqui; use bin/build_arch + app/arch_id para isso).
"""

import argparse
import ast
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import arch as A  # noqa: E402  (scripts/arch.py: canonical, arch_id, read_arch)


def tensors_from_safetensors(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    out = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        out.append((name, tuple(info["shape"]), info["dtype"]))
    return out


def tensors_from_npy_dir(d):
    import numpy as np
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".npy") and not f.startswith(("adam_", "muon_")):
            arr = np.load(os.path.join(d, f), mmap_mode="r")
            out.append((f[:-4], tuple(arr.shape), str(arr.dtype)))
    return out


def collect(path):
    if path and os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            if f.endswith(".safetensors"):
                return tensors_from_safetensors(os.path.join(path, f)), os.path.join(path, f)
        return tensors_from_npy_dir(path), path + "/*.npy"
    return [], None


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="/tmp/mix/init3m")
    ap.add_argument("--svg", default=None)
    ap.add_argument("--repo-default", action="store_true",
                    help="comparar com a arch declarada no repo (scripts/arch.py)")
    a = ap.parse_args()

    arch, src = A.read_arch(a.path)
    d, nh, nkv, hd, nl, vv, ctx, bos = (arch[k] for k in A.KEYS)
    dkv = nkv * hd
    dff = 4 * d
    ident = A.arch_id(arch)
    tens, tsrc = collect(a.path)

    # ---- derivados ------------------------------------------------------
    P_wte = vv * d
    P_head = vv * d
    per_layer = d * d + 2 * (d * dkv) + d * d + d * dff + dff * d   # q, k, v, proj, fc, proj2
    P = 2 * P_wte + nl * per_layer
    fwd_flops_tok = 2 * (P - P_wte)          # wte e' lookup (sem FLOPs de matmul)
    attn_flops = lambda t: nl * (2 * 2 * t * d)   # QK^T + AV por token
    kv_bytes = nl * 2 * dkv * 4
    act_bytes = nl * (10 * d + 3 * dkv) * 4  # = allocate(C%...) do fortran_train.f90
    opt_bytes = 2 * P * 4                     # m, v em fp32

    # ---- diagrama -------------------------------------------------------
    w = 74
    def rule(ch="-"): print("  +" + ch * w + "+")
    def line(s): print("  | " + s.ljust(w - 2) + " |")
    print()
    print("=== ARQUITETURA (retrato de %s)" % src)
    rule("=")
    line("entrada: tokens idx (int32)  ->  wte  [%d x %d]  lookup" % (vv, d))
    line("  emb [B x T x %d]  +  rope_4d  (ctx=%d, head_dim=%d)" % (d, ctx, hd))
    rule()
    line("%d x BLOCO:" % nl)
    line("   rmsnorm -> q [%d]  k [%d]  v [%d]   (GQA: %d cabecas q, %d kv)"
         % (d, dkv, dkv, nh, nkv))
    line("   atencao causal                                  O(T^2) x cabecas")
    line("   c_proj [%d]  -> +residual" % d)
    line("   rmsnorm -> mlp c_fc [%d] -> relu^2 -> c_proj [%d] -> +residual" % (dff, d))
    rule()
    line("norm final -> lm_head [%d x %d]  (vocab=%d, bos=%d)" % (vv, d, vv, bos))
    rule("=")
    print()

    # ---- tabela ---------------------------------------------------------
    if tens:
        print("=== TENSORES (%s, %d tensores)" % (tsrc, len(tens)))
        print("  %-46s %-18s %10s" % ("nome", "shape", "params"))
        tot = 0
        for name, shape, dt in tens:
            n = numel(shape)
            tot += n
            print("  %-46s %-18s %10s" % (name, "x".join(map(str, shape)), f"{n:,}"))
        print("  %-46s %-18s %10s" % ("TOTAL", "", f"{tot:,}"))
        print()
        if tot != P:
            print("  ATENCAO: soma dos tensores (%s) != conta da arch (%s) -- divergencia"
                  % (f"{tot:,}", f"{P:,}"))
            print()

    # ---- derivados ------------------------------------------------------
    print("=== DERIVADOS")
    print("  params (conta da arch)          %s" % f"{P:,}")
    print("  FLOPs/token (fwd, matmuls)      %.1f MFLOP   (6x no fwd+bwd: %.1f MFLOP)"
          % (fwd_flops_tok / 1e6, 3 * fwd_flops_tok / 1e6))
    print("  atencao/token                   %.3f MFLOP x T  (T=1024 -> %.0f MFLOP)"
          % (attn_flops(1) / 1e6, attn_flops(1024) / 1e6))
    print("    ^ FLOPs nao e' custo: medido (tier_probe, N=512) 225 us/token de")
    print("      atencao contra 83 us/token de projecoes, com ~0.6x dos FLOPs --")
    print("      a atencao e' dominada pelo trafego da matriz T x T.")
    print("  KV por token                    %d B" % kv_bytes)
    print("  ativacao por token (treino)     %d B   (allocate(C%%...) de fortran_train)"
          % act_bytes)
    # FLOPs != custo: a atencao escreve/le a matriz T x T, entao ela sai MUITO
    # mais cara por FLOP que as projecoes. Medido no tier_probe (d96, N=512):
    # atencao 225 us/token contra projecoes 83 us/token = 2.7x com ~0.6x dos
    # FLOPs. Ou seja: contagem de FLOPs subestima a atencao por ~4-5x.
    print("  estado do otimizador (m,v fp32) %.1f MB" % (opt_bytes / 1e6))
    print("  lote B=1, T=%d: ativacao        %.1f MB" % (ctx, act_bytes * ctx / 1e6))
    print()

    # ---- identidade -----------------------------------------------------
    print("=== IDENTIDADE")
    print("  canonical  %s" % A.canonical(arch))
    print("  id         %s" % ident)
    print("  fonte      %s" % src)
    if a.repo_default:
        repo = repo_default()
        if repo:
            print("  repo diz   %s  (id %s)" % (A.canonical(repo), A.arch_id(repo)))
            if A.arch_id(repo) != ident:
                print("  >>> DIVERGENCIA: a config do repo difere deste retrato")
                print("  >>>              (binario novo sairia para a outra arch) <<<")
        else:
            print("  repo diz   (nao achei #define ARCH_* em src/lib/fortran_arch.f90)")
    print()

    # ---- svg ------------------------------------------------------------
    if a.svg:
        svg_diagram(a.svg, arch, P, fwd_flops_tok, kv_bytes, act_bytes, ident)
        print("  svg: %s" % a.svg)
    return 0


def repo_default():
    """Config declarada no repo: os #define ARCH_* de src/lib/fortran_arch.f90
    (o que um build SEM --features produziria)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "src", "lib", "fortran_arch.f90")
    want = {"ARCH_D_MODEL": "d_model", "ARCH_N_HEAD": "n_head", "ARCH_N_KV": "n_kv",
            "ARCH_N_LAYER": "n_layer", "ARCH_VOCAB": "vocab", "ARCH_CTX": "ctx",
            "ARCH_BOS": "bos"}
    got = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 3 and parts[0] == "#define" and parts[1] in want:
                got[want[parts[1]]] = int(parts[2])
    # head_dim derivado de d_model/n_head
    if "d_model" in got and "n_head" in got:
        got["head_dim"] = got["d_model"] // got["n_head"]
    return got if len(got) == 8 else None


def svg_diagram(out, arch, P, flops, kv, act, ident):
    d, nh, nkv, hd, nl, vv, ctx, bos = (arch[k] for k in A.KEYS)
    dkv, dff = nkv * hd, 4 * d
    blocks = [
        ("wte  %d x %d" % (vv, d), "#e8f0fe"),
        ("rope (ctx=%d, hd=%d)" % (ctx, hd), "#f1f3f4"),
        ("%d x bloco:" % nl, "#fef7e0"),
        ("  rmsnorm", "#fffdf0"),
        ("  q %d | k %d | v %d  (GQA %d/%d)" % (d, dkv, dkv, nh, nkv), "#fffdf0"),
        ("  atencao causal  O(T^2)", "#fffdf0"),
        ("  c_proj %d + residual" % d, "#fffdf0"),
        ("  mlp c_fc %d -> relu^2 -> c_proj %d" % (dff, d), "#fffdf0"),
        ("norm final", "#f1f3f4"),
        ("lm_head  %d x %d" % (vv, d), "#e8f0fe"),
    ]
    H = 34
    h = 60 + len(blocks) * (H + 8) + 120
    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
         'font-family="monospace" font-size="13">' % (720, h),
         '<rect width="100%%" height="100%%" fill="white"/>',
         '<text x="20" y="26" font-size="15">arquitetura — id %s</text>' % ident]
    y = 46
    for txt, col in blocks:
        s.append('<rect x="20" y="%d" width="420" height="%d" rx="4" fill="%s" stroke="#bbb"/>'
                 % (y, H, col))
        s.append('<text x="30" y="%d">%s</text>' % (y + 22, txt.replace("&", "&amp;")
                                                    .replace("<", "&lt;")))
        y += H + 8
    y += 10
    for txt in ["params  %s" % f"{P:,}",
                "fwd %.1f MFLOP/token (6x: %.1f)" % (flops / 1e6, 3 * flops / 1e6),
                "KV %d B/token" % kv,
                "ativacao %d B/token (treino)" % act,
                "canonical %s" % A.canonical(arch)]:
        s.append('<text x="470" y="%d">%s</text>' % (y, txt.replace("&", "&amp;")))
        y += 22
    s.append('</svg>')
    with open(out, "w") as fh:
        fh.write("\n".join(s))


if __name__ == "__main__":
    sys.exit(main())
