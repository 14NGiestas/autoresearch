#!/usr/bin/env python3
"""tokenize_wiki_bpe.py — wiki filtrada (jsonl title/text) -> ids BPE (stdlib).

Mesmo espaco do P1/P2 (load_enc, BOS 8188), mesmo portao de round-trip.
Uma linha de ids por artigo: "8188 id id ...". Chunks independentes podem
rodar em paralelo (split -n l/N) e concatenar depois -- ordem irrelevante
para contagem e empacotamento (o split held-out e por LIVRO na prosa, a wiki
e sempre treino).

Uso:
  .venv-numpy/bin/python3 scripts/tokenize_wiki_bpe.py \
      --src /tmp/ptwiki/wiki_filt.jsonl --out /tmp/ptwiki/wiki_bpe.txt
"""
import argparse
import json
import sys
import time

sys.path.insert(0, "scripts")
from eval_driver import load_enc  # noqa: E402

BOS = 8188


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    enc = load_enc()
    t0 = time.time()
    n = tok = byt = bad = 0
    with open(args.src) as fin, open(args.out, "w") as fout:
        for line in fin:
            if args.limit and n >= args.limit:
                break
            try:
                art = json.loads(line)
                text = (art.get("title", "") + "\n\n" + art.get("text", "")).strip()
            except (json.JSONDecodeError, AttributeError):
                continue
            if not text:
                continue
            bpe = enc.encode_ordinary(text)
            if enc.decode(bpe) != text:
                bad += 1
                if bad <= 3:
                    print(f"  artigo {n}: round-trip divergiu", flush=True)
                continue
            tok += len(bpe)
            byt += len(text.encode("utf-8"))
            fout.write(f"{BOS} " + " ".join(map(str, bpe)) + "\n")
            n += 1
            if n % 20000 == 0:
                print(f"  {n} artigos, {tok/1e6:.1f}M toks "
                      f"({time.time()-t0:.0f}s)", flush=True)
    dt = time.time() - t0
    print(f"FEITO: {n} artigos, {tok} toks BPE, {byt} bytes, "
          f"{byt/max(tok,1):.3f} B/tok, bad={bad}, {dt:.0f}s")


if __name__ == "__main__":
    main()
