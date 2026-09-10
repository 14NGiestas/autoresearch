#!/usr/bin/env python3
"""fetch_mec_machado.py — harvest Machado de Assis's complete works as text.

Source: machado.mec.gov.br (MEC/FNE digital collection of the complete works,
organised by genre: romance, conto, poesia, cronica, teatro, critica,
traducao, miscelanea). Each work is offered as a PDF via a Joomla
item/download link; we fetch the PDF and convert with pdftotext.

Why this source and not Project Gutenberg for Brazilian literature: PG has
only ~60 Brazilian titles (12 of them Machado) because its PD rule is US-based
(pre-1929 publication). MEC carries the canonical complete works in modern
orthography, which is exactly the "as much Brazilian literature as possible"
case.

Provenance discipline (required for a training corpus): every file's source
URL, sha256 and byte size goes into manifest.tsv, so the corpus is auditable
and reproducible. Politeness: one request per second, browser User-Agent, a
small static collection (~8 category pages + ~60 PDFs) — never a crawl.

Usage: fetch_mec_machado.py [OUT_DIR]     # default /tmp/mec_machado
Outputs: OUT/pdf/*.pdf, OUT/txt/*.txt (pdftotext -enc UTF-8), OUT/manifest.tsv
"""

import hashlib
import os
import re
import subprocess
import sys
import time
import urllib.request

BASE = "https://machado.mec.gov.br"
CATS = ["23-romance", "24-conto", "25-poesia", "26-cronica", "27-teatro",
        "28-critica", "29-traducao", "30-miscelanea"]
UA = "autoresearch-prose/0.1 (training corpus research; contact: local user)"
DELAY = 1.0


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read(), r.headers


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mec_machado"
    pdfd = os.path.join(out, "pdf")
    txtd = os.path.join(out, "txt")
    os.makedirs(pdfd, exist_ok=True)
    os.makedirs(txtd, exist_ok=True)
    man = open(os.path.join(out, "manifest.tsv"), "w", encoding="utf-8")
    man.write("category\tfile\turl\tsha256\tbytes\ttxt_bytes\n")
    seen, ok, fail = set(), 0, 0

    for cat in CATS:
        try:
            html, _ = get(f"{BASE}/obra-completa-lista/itemlist/category/{cat}")
        except Exception as e:
            print(f"{cat}: category fetch failed: {e}", flush=True)
            continue
        links = sorted(set(re.findall(
            r"/obra-completa-lista/item/download/[A-Za-z0-9_]+",
            html.decode("utf-8", "replace"))))
        print(f"{cat}: {len(links)} links", flush=True)

        for i, ln in enumerate(links):
            if ln in seen:
                continue
            seen.add(ln)
            url = BASE + ln
            try:
                blob, hdrs = get(url)
            except Exception as e:
                print(f"  FAIL {url}: {e}", flush=True)
                fail += 1
                time.sleep(DELAY)
                continue

            cd = hdrs.get("Content-Disposition", "") or ""
            m = re.search(r'filename="?([^";]+)"?', cd)
            name = m.group(1) if m else f"{cat}_{i:02d}.pdf"
            name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(name))
            if not name.lower().endswith(".pdf"):
                name += ".pdf"

            ppath = os.path.join(pdfd, name)
            with open(ppath, "wb") as f:
                f.write(blob)
            tpath = os.path.join(txtd, name[:-4] + ".txt")
            subprocess.run(["pdftotext", "-enc", "UTF-8", "-q", ppath, tpath],
                           check=False)
            tsz = os.path.getsize(tpath) if os.path.exists(tpath) else 0
            man.write(f"{cat}\t{name}\t{url}\t"
                      f"{hashlib.sha256(blob).hexdigest()}\t{len(blob)}\t{tsz}\n")
            man.flush()
            print(f"  ok {name}: {len(blob)}B pdf -> {tsz}B txt", flush=True)
            ok += 1
            time.sleep(DELAY)

    man.close()
    print(f"done: {ok} ok, {fail} failed, {len(seen)} unique downloads",
          flush=True)


if __name__ == "__main__":
    main()
