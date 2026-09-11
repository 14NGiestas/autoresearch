#!/usr/bin/env python3
"""
tokenize_corpus.py — OUR side of the thinkx boundary.

thinkx miners produce TEXT corpora (JSONL, one {"text": ...} per line,
or ===== ROW i ===== separated markdown). This tool tokenizes them into
the canonical cross-phase id space and writes variable-length id rows;
run scripts/pack_rows.py afterwards for the 2049 train contract.

Canonical mapping (matches code_python.txt, which Phase 1 trained on):
ASCII -> raw byte (0-127), non-ASCII -> 256+b per UTF-8 byte, BOS 8188
prefixed per row. This mapping is OURS (training-side) by design --
thinkx stays tokenizer-agnostic.

Usage:
    python3 scripts/tokenize_corpus.py IN.txt OUT.txt
"""

import json
import os
import re
import sys

BOS = 8188


class CanonicalEncoder:
    def encode(self, text):
        ids = []
        for ch in text:
            if ord(ch) < 128:
                ids.append(ord(ch))
            elif ord(ch) < 256:
                # 256+cp, NOT cp: the corpora on disk prove it (0 ids in
                # 128..255, 2271 hits on 483 = 256+227 for a-tilde). This rule
                # was wrong here and disagreed with every corpus ever built
                # from this script's siblings; the rows are the truth.
                ids.append(256 + ord(ch))
            else:
                for b in ch.encode("utf-8"):
                    ids.append(256 + b)
        return ids


enc = CanonicalEncoder()


def read_text_rows(path):
    """JSONL {"text": ...} per line, else ===== ROW i ===== separated."""
    raw = open(path).read()
    rows = []
    try:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line)["text"])
        if rows:
            return rows
    except (ValueError, KeyError, TypeError):
        pass
    parts = re.split(r"^===== ROW \d+ =====\s*$", raw, flags=re.M)
    return [p.strip("\n") for p in parts if p.strip("\n")]


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    src, out = sys.argv[1], sys.argv[2]
    texts = read_text_rows(src)
    if not texts:
        print("no rows found")
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out + ".tmp", "w") as f:
        for t in texts:
            ids = enc.encode(t)
            f.write(f"{BOS} " + " ".join(map(str, ids)) + "\n")
    os.rename(out + ".tmp", out)
    print(f"tokenized {len(texts)} rows -> {out} (pack with pack_rows.py)")


if __name__ == "__main__":
    main()
