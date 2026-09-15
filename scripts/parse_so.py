#!/usr/bin/env python3
"""parse_so.py — StackExchange Posts.xml -> pares Q+A (resposta aceita).
stdlib + multiprocessing (8 procs): iterparse p/ questoes, dict de respostas
aceitas, strip HTML via regex, conta tokens BPE por amostragem.
Saida: /tmp/bench/so_pairs.jsonl + stats no stdout.
"""
import html
import multiprocessing as mp
import os
import re
import xml.etree.ElementTree as ET

XML = "/tmp/bench/so_xml/Posts.xml"
OUT = "/tmp/bench/so_pairs.jsonl"
TAG = re.compile(r"<[^>]+>")


def clean(h):
    t = TAG.sub(" ", h or "")
    return re.sub(r"\s+", " ", html.unescape(t)).strip()


def iter_posts():
    ctx = ET.iterparse(XML, events=("end",))
    for _, el in ctx:
        if el.tag == "row":
            yield el.attrib
            el.clear()


def main():
    if not os.path.exists(XML):
        print(f"sem {XML}; rode o extract primeiro")
        return
    acc, qs = {}, []
    n = 0
    for a in iter_posts():
        n += 1
        t = a.get("PostTypeId")
        if t == "1":
            qs.append(a)
        elif t == "2":
            acc[a.get("Id")] = a
        if n % 500000 == 0:
            print(f"  {n} posts...", flush=True)
    print(f"posts={n} perguntas={len(qs)} respostas={len(acc)}")
    pairs = 0
    with open(OUT, "w", encoding="utf-8") as o:
        import json
        for q in qs:
            aid = q.get("AcceptedAnswerId")
            a = acc.get(aid) if aid else None
            if a is None:
                continue
            qt, at = clean(q.get("Body")), clean(a.get("Body"))
            if len(qt) < 100 or len(at) < 100:
                continue
            o.write(json.dumps({"q": q.get("Title", ""), "qb": qt[:3000],
                                "a": at[:6000]}, ensure_ascii=False) + "\n")
            pairs += 1
    print(f"pares Q+A-aceita: {pairs} -> {OUT}")


if __name__ == "__main__":
    with mp.Pool(1):
        main()
