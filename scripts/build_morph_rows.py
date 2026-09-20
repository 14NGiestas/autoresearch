#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""build_morph_rows.py — rows f_morph: mesmo TEXTO do f0, BPE com fronteiras.
Cada palavra -> morphemes (morph_segment) -> BPE guloso por morpheme (p=0);
merges NUNCA cruzam fronteira. Separadores via ordinary. Decode-integridade
no portao; fertilidade reportada (sobe: esperado).
Uso: .venv-numpy/bin/python3 scripts/build_morph_rows.py --gate-lines 200 [--full]
"""
import argparse
import pickle
import re
import subprocess
import sys

sys.path.insert(0, "scripts")

BOS = 8188
T = 1024
N_TRAIN_LINES = 59385
WRE = re.compile(r"[^\W\d_]+|\d+|\s+|[^\s]", re.UNICODE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-lines", type=int, default=200)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    from eval_driver import load_enc
    from bpe_dropout import encode_dropout
    from morph_segment import segmenta
    enc = load_enc()
    ranks = pickle.load(open("/home/pauli/.cache/autoresearch/tokenizer/tokenizer.pkl", "rb"))._mergeable_ranks
    import random
    rng = random.Random(7)

    def enc_morph(text):
        out = []
        for m in WRE.finditer(text):
            tok = m.group(0)
            if re.fullmatch(r"[^\W\d_]+", tok):
                for mph in segmenta(tok):
                    out.extend(encode_dropout(mph.encode("utf-8"), ranks, 0.0, rng))
            else:
                out.extend(enc.encode_ordinary(tok))
        return out

    src = open("/tmp/prose/prose_bpe_all.txt", encoding="utf-8")
    n0 = nd = 0
    for li, line in enumerate(src):
        if li >= args.gate_lines:
            break
        parts = line.split()
        if len(parts) < 2:
            continue
        text = enc.decode([int(x) for x in parts[1:]])
        ids0 = enc.encode_ordinary(text)
        idsm = enc_morph(text)
        assert enc.decode(idsm) == text, f"integridade linha {li}"
        n0 += len(ids0)
        nd += len(idsm)
    print(f"portao: {args.gate_lines} linhas fert0={n0/args.gate_lines:.1f} "
          f"fertm={nd/args.gate_lines:.1f} (+{(nd/n0-1)*100:.1f}%)", flush=True)
    if not args.full:
        return
    stock = int(subprocess.check_output(
        ".venv-numpy/bin/python3 -c \"t=0\nfor i,l in enumerate(open('/tmp/prose/prose_bpe_all.txt')):\n if i>=59385: break\n t+=len(l.split())-1\nprint(t)\"",
        shell=True, cwd=".").decode().strip())
    rem = 8000000 / stock
    n = N_TRAIN_LINES
    klines = max(1, round(n * rem))
    sel = set()
    cum = idx = 0
    for _ in range(n):
        if (cum + 1) * klines // max(n, 1) > cum * klines // max(n, 1):
            sel.add(idx)
        cum += 1
        idx += 1
    print(f"stock={stock} klines={len(sel)}", flush=True)
    src.seek(0)
    out = open("/tmp/mix/rows_f_morph.txt", "w")
    nlines = nrows = ntok = 0
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
        for t in enc_morph(text):
            buf.append(t)
            ntok += 1
            if len(buf) - 1 >= T:
                out.write(" ".join(map(str, buf)) + "\n")
                nrows += 1
                buf = [BOS]
        if nlines % 10000 == 0:
            print(f"  {nlines} linhas -> {nrows} rows...", flush=True)
    out.close()
    print(f"rows_f_morph: {nrows} rows, {ntok} toks")


if __name__ == "__main__":
    main()
