#!/usr/bin/env python3
"""prepare_prose.py — clean the harvested Portuguese prose into a train/val set.

Sources (see scripts/fetch_mec_machado.py for the MEC side, and the PG catalog
CSV for the Gutenberg side):
  * Project Gutenberg: /tmp/pg_pt_raw/cache/epub/<id>/pg<id>.txt  (645 books,
    every Portuguese title in PG; ~60 are Brazilian by author/subject)
  * MEC/FNE "Obra Completa de Machado de Assis": /tmp/mec_machado/txt/*.txt
    (71 books, PDF->pdftotext, canonical modern orthography)

Cleaning rules, and why each exists:
  * PG wraps each book in a licence header/footer -- and the licence text is
    ENGLISH even for Portuguese books. Keep only the START..END marker span.
  * MEC text comes from PDFs: hard line breaks every ~70 columns, end-of-line
    hyphenation, and per-page running headers/footers. Reflow to paragraphs,
    rejoin hyphenated words, drop page numbers and repeated short lines.
  * Split by BOOK, never by chunk: chunking a book and then splitting its
    chunks between train and val leaks the neighbouring paragraph. Books are
    assigned to val by a stable hash of their source id, and everything is
    emitted as one JSON object per book so the packer can keep them apart.

Output (under OUT, default /tmp/prose):
  prose_train.jsonl / prose_val.jsonl   one {"text", "meta"} per line
  manifest.tsv                          id, source, tag, split, bytes, sha256, title
Usage: prepare_prose.py [OUT] [--val-frac 0.1]
"""

import hashlib
import json
import os
import re
import sys

PG_DIR = "/tmp/pg_pt_raw/cache/epub"
PG_META = "/tmp/pg_pt_meta.tsv"
MEC_TXT = "/tmp/mec_machado/txt"
MEC_MAN = "/tmp/mec_machado/manifest.tsv"

START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
PG_NOISE = re.compile(r"^(produced by|updated by|credits?:|transcriber'?s note|"
                      r"end of (?:the )?project gutenberg|most recently updated)", re.I)
TOKEN_DROP = re.compile(r"^\s*\d{1,4}\s*$")          # bare page numbers
HYPHEN_JOIN = re.compile(r"[A-Za-zÀ-ÿ]-$")
TOC_HEAD = {"ÍNDICE", "INDICE", "SUMÁRIO", "SUMARIO", "ÍNDICE GERAL"}
UPPER_CH = re.compile(r"[A-ZÀ-Ý]")


def is_heading(s):
    """Short all-caps line with no terminal punctuation: a chapter/section
    title. Keeping these standalone stops the reflow from gluing them to the
    following paragraph (which otherwise teaches a false sentence structure).
    """
    return (3 <= len(s) <= 60 and UPPER_CH.search(s) and s == s.upper()
            and not s.endswith((".", ",", ";", ":")))


def clean_pg(raw):
    """Keep the marker span; drop licence/credit lines."""
    m1, m2 = START_RE.search(raw), END_RE.search(raw)
    body = raw[m1.end():m2.start()] if (m1 and m2) else raw
    out = [ln.rstrip() for ln in body.splitlines()]
    return [ln for ln in out if not PG_NOISE.match(ln.strip())]


def running_lines(lines, min_reps=8, max_len=60):
    """Short lines repeated across pages are running headers, not prose."""
    counts = {}
    for ln in lines:
        s = ln.strip()
        if s and len(s) <= max_len and not s.startswith("—"):
            counts[s] = counts.get(s, 0) + 1
    return {s for s, c in counts.items() if c >= min_reps}


def reflow(lines):
    """Blank line = paragraph break; join the rest, repairing hyphenation."""
    drop = running_lines(lines)
    blank_ratio = (sum(1 for ln in lines if not ln.strip()) / max(1, len(lines)))
    # hard-wrapped sources (PDF) may have few blank lines: then also break
    # after sentence-final punctuation followed by a capitalised line
    punct_split = blank_ratio < 0.01
    paras, buf = [], []

    def flush():
        if buf:
            paras.append(" ".join(buf))
            buf.clear()

    for ln in lines:
        s = ln.strip()
        if not s:
            flush()
            continue
        if s in drop or TOKEN_DROP.match(s):
            continue
        # the ÍNDICE heading itself is dropped; its entries are heading-like
        # lines, so they come out one per line instead of glued into a blob
        if s.upper() in TOC_HEAD:
            flush()
            continue
        if is_heading(s):
            flush()
            paras.append(s)
            continue
        if buf and HYPHEN_JOIN.search(buf[-1]) and s[:1].islower():
            buf[-1] = buf[-1][:-1] + s
            continue
        if punct_split and buf and buf[-1][-1:] in ".!?:" and (s[:1].isupper() or s[:1] in "«\"'—"):
            flush()
        buf.append(s)
    flush()
    return "\n\n".join(paras)


def read_pg_meta():
    meta = {}
    if os.path.exists(PG_META):
        for ln in open(PG_META, encoding="utf-8"):
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 4:
                meta[p[0]] = {"tag": p[1], "author": p[2], "title": p[3]}
    return meta


def read_mec_man():
    meta = {}
    if os.path.exists(MEC_MAN):
        rows = open(MEC_MAN, encoding="utf-8").read().splitlines()[1:]
        for ln in rows:
            p = ln.split("\t")
            if len(p) >= 3:
                meta[os.path.basename(p[1])] = {"cat": p[0], "url": p[2]}
    return meta


def main():
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "/tmp/prose"
    val_frac = 0.1
    if "--val-frac" in sys.argv:
        val_frac = float(sys.argv[sys.argv.index("--val-frac") + 1])
    os.makedirs(out, exist_ok=True)

    pg_meta, mec_man = read_pg_meta(), read_mec_man()
    items, seen_sha, stats = [], {}, {"pg": 0, "mec": 0, "dropped_dup": 0, "dropped_short": 0}

    def add(src_id, source, tag, title, text, extra=None):
        text = text.strip()
        if len(text.encode("utf-8")) < 20000:          # < 20 kB: fragment/poetry leaf
            stats["dropped_short"] += 1
            return
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if sha in seen_sha:
            stats["dropped_dup"] += 1
            return
        seen_sha[sha] = src_id
        # split on a hash of the SOURCE ID, not of the cleaned text: re-running
        # the cleaner (new rule, fixed artefact) must not reshuffle which books
        # are held out, or val numbers stop being comparable across revisions
        idh = int(hashlib.sha256(src_id.encode("utf-8")).hexdigest()[:8], 16)
        split = "val" if (idh % 1000) < val_frac * 1000 else "train"
        m = {"id": src_id, "source": source, "tag": tag, "split": split,
             "title": title, "sha256": sha, "bytes": len(text.encode("utf-8"))}
        if extra:
            m.update(extra)
        items.append((m, text))

    if os.path.isdir(PG_DIR):
        for eid in sorted(os.listdir(PG_DIR), key=lambda x: int(x) if x.isdigit() else 0):
            p = os.path.join(PG_DIR, eid, f"pg{eid}.txt")
            if not os.path.exists(p):
                continue
            raw = open(p, encoding="utf-8", errors="replace").read()
            md = pg_meta.get(eid, {})
            add(eid, "gutenberg", "br" if md.get("tag") == "br" else "pt-other",
                md.get("title", ""), reflow(clean_pg(raw)),
                {"author": md.get("author", "")})
            stats["pg"] += 1

    if os.path.isdir(MEC_TXT):
        for fn in sorted(os.listdir(MEC_TXT)):
            if not fn.endswith(".txt"):
                continue
            raw = open(os.path.join(MEC_TXT, fn), encoding="utf-8", errors="replace").read()
            mm = mec_man.get(fn, {})
            add("mec:" + fn[:-4], "mec_machado", "machado-br", fn[:-4],
                reflow(raw.splitlines()),
                {"author": "Machado de Assis", "genre": mm.get("cat", "")})
            stats["mec"] += 1

    tr = open(os.path.join(out, "prose_train.jsonl"), "w", encoding="utf-8")
    va = open(os.path.join(out, "prose_val.jsonl"), "w", encoding="utf-8")
    man = open(os.path.join(out, "manifest.tsv"), "w", encoding="utf-8")
    man.write("id\tsource\ttag\tsplit\tbytes\tsha256\ttitle\n")
    ntr = nva = btr = bva = 0
    for m, text in items:
        line = json.dumps({"text": text, "meta": m}, ensure_ascii=False)
        if m["split"] == "val":
            va.write(line + "\n")
            nva += 1
            bva += m["bytes"]
        else:
            tr.write(line + "\n")
            ntr += 1
            btr += m["bytes"]
        man.write(f"{m['id']}\t{m['source']}\t{m['tag']}\t{m['split']}\t"
                  f"{m['bytes']}\t{m['sha256']}\t{m['title']}\n")
    for f in (tr, va, man):
        f.close()

    print(f"books kept : {len(items)} (pg {stats['pg']}, mec {stats['mec']})")
    print(f"  dropped  : dup {stats['dropped_dup']}, short {stats['dropped_short']}")
    print(f"train      : {ntr} books, {btr/1e6:.1f} MB")
    print(f"val        : {nva} books, {bva/1e6:.1f} MB")
    tags = {}
    for m, _ in items:
        tags[m["tag"]] = tags.get(m["tag"], 0) + 1
    print("tags       :", ", ".join(f"{k}={v}" for k, v in sorted(tags.items())))


if __name__ == "__main__":
    main()
