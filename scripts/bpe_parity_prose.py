#!/usr/bin/env python3
"""bpe_parity_prose.py — P1: o espaço BPE serve para prosa em português, e quanto?

Duas perguntas, uma execução:

1. PARIDADE: o encoder BPE em Fortran tem de casar id a id com o tiktoken no
   texto de prosa em português (o mesmo rigor do teste do espaço de bytes: se
   divergir, a inferência e o treino falam línguas diferentes -- foi o bug #1).
2. COMPRESSÃO: quantos caracteres/bytes por token. É o número que decide o P2:
   byte-level gasta 1 token por byte; se o BPE der ~3 bytes/token, o mesmo
   hardware processa ~3x mais texto (e a atenção, O(T^2), cai ~9x por caractere).

  .venv-numpy/bin/python3 scripts/bpe_parity_prose.py --n 200
"""
import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_driver import load_enc  # noqa: E402

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
TABLES = os.path.join(CACHE, "tok_tables")
WORK = "/tmp/p1_docs"
JSONL = "/tmp/prose/prose_train.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--jsonl", default=JSONL)
    ap.add_argument("--bin", default="")
    args = ap.parse_args()

    texts = []
    with open(args.jsonl) as f:
        for line in f:
            try:
                t = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            if t:
                texts.append(t)
    # amostra diversa: começo + meio espaçado (diferentes livros/épocas)
    half = args.n // 2
    step = max(1, len(texts) // max(1, args.n))
    docs = (texts[:half] + texts[len(texts) // 2::step][: args.n - half])[: args.n]
    # amostra PARÁGRAFOS (~1200 chars), não livros: um doc de 300 KB domina a
    # estatística e o encoder é linear no tamanho -- 200 parágrafos medem melhor
    chunks = []
    for t in docs:
        for i in range(0, len(t), 1200):
            c = t[i:i + 1200]
            if len(c) > 200:
                chunks.append(c)
    step = max(1, len(chunks) // args.n)
    docs = chunks[: args.n // 2] + chunks[len(chunks) // 2::step][: args.n // 2]
    docs = docs[: args.n]
    print(f"parágrafos amostrados: {len(docs)} (de {len(chunks)}, "
          f"{len(texts)} livros)")

    os.makedirs(WORK, exist_ok=True)
    paths = []
    for i, t in enumerate(docs):
        p = os.path.join(WORK, f"doc_{i:04d}.txt")
        with open(p, "wb") as f:
            f.write(t.encode("utf-8"))
        paths.append(p)
    listfile = os.path.join(WORK, "list.txt")
    with open(listfile, "w") as f:
        f.write("\n".join(paths) + "\n")

    # o binário MAIS RECENTE: várias pastas de build coexistem (hashes de flags
    # diferentes) e um binário velho tem CLI antiga e devolve zero linhas
    cands = glob.glob(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src/build/*/*/app/tokdiff"))
    binary = args.bin or max(cands, key=os.path.getmtime)
    p = subprocess.run([binary, "--tables", TABLES, "--space", "bpe",
                        "--list", listfile], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"tokdiff exit {p.returncode}: {p.stderr[-500:]}")
    # só as linhas que são listas de inteiros (o resto é ruído de CLI)
    got = []
    for ln in p.stdout.split("\n"):
        ln = ln.strip()
        if ln and ln[0].isdigit() and all(c.isdigit() or c == " " for c in ln):
            got.append([int(x) for x in ln.split()])
    if len(got) != len(docs):
        raise SystemExit(f"tokdiff devolveu {len(got)} linhas para {len(docs)} docs")

    enc = load_enc()
    bad = []
    ntok = nbytes = nchars = 0
    for i, (t, g) in enumerate(zip(docs, got)):
        want = enc.encode_ordinary(t)
        ntok += len(want)
        nbytes += len(t.encode("utf-8"))
        nchars += len(t)
        if want != g:
            bad.append((i, t, want, g))

    print(f"\n== PARIDADE (Fortran BPE x tiktoken) ==")
    if not bad:
        print(f"  {len(docs)} docs, {ntok} tokens: 100% MATCH ✓")
    else:
        i, t, want, g = bad[0]
        k = next(j for j in range(min(len(want), len(g))) if want[j] != g[j])
        print(f"  DIVERGÊNCIAS em {len(bad)}/{len(docs)} docs")
        print(f"  primeiro: doc {i}, token {k}: tiktoken={want[k]} "
              f"({enc.decode([want[k]])!r}) fortran={g[k]}")
        print(f"  contexto: {t[max(0,k*3-60):k*3+60]!r}")

    print(f"\n== COMPRESSÃO (o número que decide o P2) ==")
    print(f"  bytes={nbytes}  caracteres={nchars}  tokens={ntok}")
    print(f"  -> {nbytes/ntok:.3f} bytes/token | {nchars/ntok:.3f} caracteres/token")
    print(f"  byte-level hoje: 1.000 token/byte")
    print(f"  -> mesmo hardware processa {ntok/nbytes:.2f}x menos tokens para o"
          f" MESMO texto = {nbytes/ntok:.2f}x mais texto por segundo")
    # atenção: FLOPs O(T^2) por caractere caem com o quadrado da compressão
    r = nbytes / ntok
    print(f"  -> e a atenção (O(T^2)) custa {r*r:.1f}x menos por caractere")

    # implicação para o corpus inteiro: medido no PRÓPRIO arquivo de ids
    # (cada id do espaço byte-level é 1 byte de texto, BOS incluído por linha)
    rows = "/tmp/prose/prose_all.txt"
    if os.path.exists(rows):
        nrows = sum(1 for _ in open(rows))
        tpr = 2049                      # ids por linha do corpus
        bytes_tot = nrows * tpr
        est = bytes_tot / r
        print(f"\n== CORPUS INTEIRO ==")
        print(f"  {nrows} linhas x {tpr} = {bytes_tot/1e6:.1f}M bytes (1 id = 1 byte)")
        print(f"  -> ~{est/1e6:.1f}M tokens BPE")
        for np_ in (6e6, 10e6, 25e6, 50e6):
            need = 20 * np_
            print(f"     {np_/1e6:5.0f}M params: Chinchilla pede {need/1e6:6.0f}M tokens"
                  f" = {need/est:5.2f} épocas do corpus")


if __name__ == "__main__":
    main()
