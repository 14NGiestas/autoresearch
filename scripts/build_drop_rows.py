#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""build_drop_rows.py — rows f_drop: mesmo TEXTO do f0, segmentação BPE-dropout.

Truque de velocidade: encode_ordinary (C) dá a segmentação base; o dropout
roda por TOKEN (re-BPE dos bytes da peça com regras dropadas p/token).
Equivalente ao dropout por palavra (verificada identidade p=0 no portão).
Empacota T=1024+BOS igual ao build_mix; val = val do f0 verbatim.
Uso: .venv-numpy/bin/python3 scripts/build_drop_rows.py --p 0.1 --seed 7
"""
import argparse
import pickle
import random
import sys

sys.path.insert(0, "scripts")

BOS = 8188
T = 1024
N_TRAIN_LINES = 59385  # igual ao f0 (build_mix --prose-ntrain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gate-lines", type=int, default=200)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    from eval_driver import load_enc
    enc = load_enc()
    ranks = pickle.load(open("/home/pauli/.cache/autoresearch/tokenizer/tokenizer.pkl", "rb"))._mergeable_ranks
    rng = random.Random(args.seed)
    from bpe_dropout import encode_dropout

    def enc_drop(text):
        out = []
        for i in enc.encode_ordinary(text):
            b = enc.decode_single_token_bytes(i)
            if len(b) < 2:
                out.append(i)
                continue
            out.extend(encode_dropout(b, ranks, args.p, rng))
        return out

    # portão: p=0 bit-idêntico + integridade em amostra
    src = open("/tmp/prose/prose_bpe_all.txt", encoding="utf-8")
    n0 = nd = diff = 0
    for li, line in enumerate(src):
        if li >= args.gate_lines:
            break
        parts = line.split()
        if len(parts) < 2:
            continue
        text = enc.decode([int(x) for x in parts[1:]])
        ids0 = enc.encode_ordinary(text)
        idsd = enc_drop(text)
        assert enc.decode(idsd) == text, f"integridade linha {li}"
        n0 += len(ids0)
        nd += len(idsd)
        if ids0 != idsd:
            diff += 1
    print(f"portao: {args.gate_lines} linhas fert0={n0/args.gate_lines:.1f} "
          f"fertd={nd/args.gate_lines:.1f} alteradas={diff}/{args.gate_lines}",
          flush=True)
    if args.p == 0:
        assert diff == 0, "p=0 deve ser identico!"
        print("portao p=0 OK")
    if not args.full:
        return
    # build completo: MESMA selecao do f0 (quantum uniforme do build_mix:
    # klines espacadas; rem = 8M/stock). Sem isso vira prefixo (vies de autor).
    import subprocess
    stock = int(subprocess.check_output(
        ".venv-numpy/bin/python3 -c \"t=0\nfor i,l in enumerate(open('/tmp/prose/prose_bpe_all.txt')):\n if i>=59385: break\n t+=len(l.split())-1\nprint(t)\"",
        shell=True, cwd=".").decode().strip())
    rem = 8000000 / stock
    n = N_TRAIN_LINES
    klines = max(1, round(n * rem))
    sel = set()
    cum = 0
    # replica o gerador do prose_stream: conta TODAS as linhas ate ntrain
    # (o stride antigo lia tudo; o quantum atual tambem itera tudo)
    idx = 0
    for _ in range(n):
        if (cum + 1) * klines // max(n, 1) > cum * klines // max(n, 1):
            sel.add(idx)
        cum += 1
        idx += 1
    print(f"stock={stock} rem={rem:.4f} klines={len(sel)}", flush=True)
    src.seek(0)
    stream = []
    nlines = 0
    out = open("/tmp/mix/rows_f_drop.txt", "w")
    nrows = ntok = 0
    buf = [BOS]
    for line in src:
        if nlines >= N_TRAIN_LINES:
            break
        li = nlines
        nlines += 1
        if li not in sel:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        text = enc.decode([int(x) for x in parts[1:]])
        for t in enc_drop(text):
            buf.append(t)
            ntok += 1
            if len(buf) - 1 >= T:
                out.write(" ".join(map(str, buf)) + "\n")
                nrows += 1
                buf = [BOS]
        if nlines % 10000 == 0:
            print(f"  {nlines} linhas -> {nrows} rows...", flush=True)
    out.close()
    print(f"rows_f_drop: {nrows} rows, {ntok} toks, fert-relativa={ntok/8000000:.3f}")


if __name__ == "__main__":
    main()
