#!/usr/bin/env python3
"""filter_wiki.py — a wiki não precisa inteira (medido).

50,6% dos artigos têm <2 KB e carregam 4,4% dos tokens; 39% têm <1 KB e carregam
1,9%. Stub de wiki é uma linha ("X é um município de Y"): alta repetição e registro
atrofiado -- descartar melhora a economia de tokens E a qualidade do dado.

Também deduplica (sha1 do texto) e descarta o held-out por hash do TÍTULO (2%),
para o holdout da wiki nunca entrar no treino.

  filter_wiki.py --in /tmp/ptwiki/wx_full --out /tmp/ptwiki/wiki_filt.jsonl \\
                 --min-bytes 2048
"""
import argparse
import glob
import hashlib
import json
import os
import sys


def held_out(title, frac=0.02):
    h = int(hashlib.sha256(title.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 10000) < frac * 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="ind", default="/tmp/ptwiki/wx_full")
    ap.add_argument("--out", default="/tmp/ptwiki/wiki_filt.jsonl")
    ap.add_argument("--min-bytes", type=int, default=2048)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.ind, "*", "*")))
    seen = set()
    n_in = n_keep = n_stub = n_dup = n_hold = 0
    b_in = b_keep = 0
    with open(args.out, "w") as out:
        for p in files:
            try:
                fh = open(p, encoding="utf-8")
            except OSError:
                continue
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = d.get("text") or ""
                nb = len(t.encode("utf-8"))
                n_in += 1
                b_in += nb
                if held_out(d.get("title", "")):
                    n_hold += 1
                    continue
                if nb < args.min_bytes:
                    n_stub += 1
                    continue
                h = hashlib.sha1(t.encode("utf-8")).hexdigest()
                if h in seen:
                    n_dup += 1
                    continue
                seen.add(h)
                out.write(json.dumps({"title": d.get("title", ""), "text": t},
                                     ensure_ascii=False) + "\n")
                n_keep += 1
                b_keep += nb
            fh.close()
    print(f"artigos: {n_in} -> {n_keep} mantidos")
    print(f"  descartados: {n_stub} stubs (<{args.min_bytes} B), {n_dup} duplicatas, "
          f"{n_hold} held-out")
    print(f"  texto: {b_in/1e6:.1f} MB -> {b_keep/1e6:.1f} MB "
          f"({b_keep/max(b_in,1)*100:.1f}% do texto mantido)")
    print(f"  saida: {args.out} ({os.path.getsize(args.out)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
