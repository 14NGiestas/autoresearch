#!/usr/bin/env python3
"""tokenize_prose_bpe.py — P2 passo 1: a mesma prosa, no espaço BPE.

Re-tokeniza os rows byte-level de /tmp/prose/prose_all.txt para o espaço BPE (o
mesmo das fases 1-4 e do tokdiff contra tiktoken), preservando:

  - a ORDEM e o split: as linhas de treino continuam antes das de validação, então
    os mesmos livros held-out seguem held-out em qualquer espaço de tokens;
  - o formato de saída das fases 1-4 (BOS + ids por linha);
  - o TEXTO: portão de integridade linha a linha -- o texto decodificado dos ids
    BPE tem de ser exatamente o texto decodificado dos ids byte-level. Sem isso a
    troca de espaço perde/estraga texto e o bpb deixa de ser comparável.

  .venv-numpy/bin/python3 scripts/tokenize_prose_bpe.py --limit 300   # portão
  .venv-numpy/bin/python3 scripts/tokenize_prose_bpe.py               # corpus
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_driver import load_enc  # noqa: E402

BOS = 8188  # mesma convenção das fases 1-4 (e do tokenize_corpus.py)
SRC = "/tmp/prose/prose_all.txt"
NVAL_ROWS = 4950  # as últimas linhas do arquivo são os livros held-out
N_TRAIN = 59385


def decode_bytes_ids(ids):
    """Espelho exato do decode_bytes do Fortran: id < 128 -> ASCII; 256..511 ->
    codepoint (id-256) emitido como UTF-8; uma corrida que forme sequência UTF-8
    válida com codepoint >= 256 é remontada (é o caso do travessão)."""
    out = []
    i, n = 0, len(ids)
    while i < n:
        v = ids[i]
        if v < 128:
            out.append(chr(v)); i += 1; continue
        if not (256 <= v < 512):
            i += 1; continue
        b = v - 256
        if 194 <= b <= 244:
            need = 1 if b < 224 else (2 if b < 240 else 3)
            ok = i + need < n
            if ok:
                seq = [b]
                for j in range(1, need + 1):
                    w = ids[i + j]
                    if not (256 <= w < 512) or not (128 <= w - 256 <= 191):
                        ok = False
                        break
                    seq.append(w - 256)
            if ok:
                if need == 1:
                    cp = ((seq[0] & 31) << 6) | (seq[1] & 63)
                elif need == 2:
                    cp = (((seq[0] & 15) << 6 | (seq[1] & 63)) << 6) | (seq[2] & 63)
                else:
                    cp = ((((seq[0] & 7) << 6 | (seq[1] & 63)) << 6
                           | (seq[2] & 63)) << 6) | (seq[3] & 63)
                if cp >= 256:
                    out.append(chr(cp)); i += need + 1; continue
        out.append(chr(b)); i += 1
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default="/tmp/prose/prose_bpe_all.txt")
    ap.add_argument("--val-out", default="/tmp/prose/prose_bpe_val.txt")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    enc = load_enc()
    t0 = time.time()
    rows = tok_in = tok_out = bad = 0
    byt_in = byt_out = 0
    val_lines = []
    with open(args.src) as fin, open(args.out, "w") as fout:
        for line in fin:
            if args.limit and rows >= args.limit:
                break
            parts = line.split()
            if len(parts) < 2:
                continue
            ids = [int(x) for x in parts[1:]]  # parts[0] é o BOS do corpus original
            text = decode_bytes_ids(ids)
            bpe = enc.encode_ordinary(text)
            # PORTÃO: o texto tem de voltar idêntico do espaço BPE
            if enc.decode(bpe) != text:
                bad += 1
                if bad <= 3:
                    k = next((j for j, (a, b) in enumerate(
                        zip(enc.decode(bpe), text)) if a != b), 0)
                    print(f"  linha {rows}: texto divergiu em {k}: "
                          f"{text[max(0,k-40):k+40]!r}", flush=True)
            tok_in += len(ids)
            tok_out += len(bpe)
            byt_in += len(text.encode("utf-8"))
            byt_out += len(text.encode("utf-8"))
            out = f"{BOS} " + " ".join(map(str, bpe)) + "\n"
            fout.write(out)
            if rows >= N_TRAIN:
                val_lines.append(out)
            rows += 1
            if rows % 5000 == 0:
                print(f"  {rows} linhas... ({time.time()-t0:.0f}s)", flush=True)

    if val_lines and not args.limit:
        with open(args.val_out, "w") as fv:
            fv.writelines(val_lines[:NVAL_ROWS])

    dt = time.time() - t0
    print(f"\nlinhas: {rows}   tempo: {dt:.0f}s")
    print(f"tokens byte-level (entrada): {tok_in/1e6:.1f}M")
    print(f"tokens BPE (saída):          {tok_out/1e6:.1f}M")
    print(f"bytes de texto (portão):     {byt_in/1e6:.1f}M  ->  {byt_in/tok_out:.3f} bytes/token")
    print(f"portão de integridade: {'OK (texto idêntico em todas as linhas)' if bad == 0 else f'FALHOU em {bad} linhas'}")
    if bad:
        sys.exit(1)
    print(f"saída: {args.out} ({os.path.getsize(args.out)/1e6:.0f} MB)"
          + (f" + {args.val_out}" if val_lines else ""))


if __name__ == "__main__":
    main()
