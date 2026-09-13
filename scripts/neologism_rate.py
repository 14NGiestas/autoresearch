#!/usr/bin/env python3
"""neologism_rate.py — H2: taxa de palavras fora de wordlist PT (stdlib only).

OOV aqui mistura neologismos + arcaismos: o desenho honesto e COMPARATIVO
(mesmos prompts, modelos distintos -- arcaismo constante, delta = neologismo).
Uso:
  .venv-numpy/bin/python3 scripts/neologism_rate.py --dict /usr/share/hunspell/pt_BR.dic \
      --in amostra.txt   # texto livre, uma geracao por linha (ou --in -)
"""
import argparse
import re
import sys
import unicodedata


def load_dict(path):
    words = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i == 0 and line.strip().isdigit():
                continue  # contagem do hunspell
            w = line.split("/", 1)[0].strip().lower()
            if w:
                words.add(w)
    return words


WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)


def toks(text):
    return [unicodedata.normalize("NFC", w).lower() for w in WORD_RE.findall(text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", required=True)
    ap.add_argument("--in", dest="inp", default="-")
    args = ap.parse_args()
    d = load_dict(args.dict)
    print(f"dicionario: {len(d)} formas", file=sys.stderr)
    fin = sys.stdin if args.inp == "-" else open(args.inp, encoding="utf-8")
    tot = oov = 0
    per_line = []
    for line in fin:
        ws = toks(line)
        if not ws:
            continue
        o = sum(1 for w in ws if w not in d)
        tot += len(ws)
        oov += o
        per_line.append(o / len(ws))
    print(f"linhas={len(per_line)} palavras={tot} oov={oov} "
          f"H2={oov/max(tot,1):.4f}")
    if per_line:
        import statistics
        print(f"H2_media_linhas={statistics.mean(per_line):.4f}")


if __name__ == "__main__":
    main()
