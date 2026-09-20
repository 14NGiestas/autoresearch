#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""build_wordlists.py — listas de formas plenas p/ H2 (stdlib + enc).
Complementa hunspell (bases+afixos nao expandidos) e Figueiredo (lemas):
  wiki_forms.txt  <- formas modernas flexionadas (wiki filtrada)
  prose_forms.txt <- formas de epoca flexionadas (prosa BPE decodificada)
Uso: .venv-numpy/bin/python3 scripts/build_wordlists.py [--min-wiki N] [--min-prose N]
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, "scripts")

WR = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)


def words_of(text):
    return [unicodedata.normalize("NFC", w).lower() for w in WR.findall(text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-wiki", type=int, default=3)
    ap.add_argument("--min-prose", type=int, default=2)
    args = ap.parse_args()
    c = Counter()
    n = 0
    for line in open("/tmp/ptwiki/wiki_filt.jsonl", encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        c.update(words_of(d.get("title", "") + " " + d.get("text", "")))
        n += 1
        if n % 50000 == 0:
            print(f"  wiki {n} arts...", flush=True)
    keep = sorted(w for w, f in c.items() if f >= args.min_wiki and len(w) >= 2)
    open("/tmp/wiki_forms.txt", "w").write("\n".join(keep) + "\n")
    print(f"WIKI forms freq>={args.min_wiki}: {len(keep)}")
    from eval_driver import load_enc
    enc = load_enc()
    c2 = Counter()
    m = 0
    for line in open("/tmp/prose/prose_bpe_all.txt", encoding="utf-8"):
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            text = enc.decode([int(x) for x in parts[1:]])
        except ValueError:
            continue
        c2.update(words_of(text))
        m += 1
        if m % 20000 == 0:
            print(f"  prose {m} linhas...", flush=True)
    keep2 = sorted(w for w, f in c2.items() if f >= args.min_prose and len(w) >= 2)
    open("/tmp/prose_forms.txt", "w").write("\n".join(keep2) + "\n")
    print(f"PROSE forms freq>={args.min_prose}: {len(keep2)}")


if __name__ == "__main__":
    main()
