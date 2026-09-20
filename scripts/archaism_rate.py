#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""archaism_rate.py — indice de arcaismo ortografico em PT (stdlib only).

Padroes pre-reforma/oitocentistas de alta precisao: ph/th/y, geminadas
(ll/mm/nn/ff), ct/pt mudos (acto/optimo), sci (sciencia), dous/cousa/pae,
geraes, ella/elle, elisoes d'. Ruidoso por item (pacto tem ct!), mas o
desenho e COMPARATIVO (mesmos prompts) => vies constante.
Uso: .venv-numpy/bin/python3 scripts/archaism_rate.py --in amostra.txt
"""
import argparse
import re
import sys
import unicodedata

PATS = [
    r"ph", r"th", r"\by\w", r"ll", r"mm", r"nn", r"ff",
    r"ct[aeiou]", r"pt[aeiou]", r"sci[ae]", r"\bdous\b", r"\bcousa",
    r"\bpae\b", r"\bgeraes\b", r"\belle\b", r"\bella\b", r"\bd'",
    r"\bha\b", r"ideia|idéa", r"ellas\b", r"elles\b",
]
RX = re.compile("|".join(f"(?:{p})" for p in PATS))
WR = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="-")
    args = ap.parse_args()
    fin = sys.stdin if args.inp == "-" else open(args.inp, encoding="utf-8")
    tw = hit = 0
    for line in fin:
        ws = [unicodedata.normalize("NFC", w).lower() for w in WR.findall(line)]
        tw += len(ws)
        hit += sum(1 for w in ws if RX.search(w))
    print(f"palavras={tw} arcaicas={hit} ARCH={hit/max(tw,1):.4f}")


if __name__ == "__main__":
    main()
