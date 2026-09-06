#!/usr/bin/env python3
"""
Phase 1 curriculum: Python code from CodeAlpaca-20k (instruction → code).
Public, no auth, 20k rows, ~7.7 MB.

Each output row: BOS + Alpaca-formatted text, tokenized.
Output: ~/.cache/autoresearch/code_python.txt

Format: "### Instruction: ...\n### Response: ...\n"
Good because model already saw [tool:edit] blocks in train.py; the
"### Response:" turn delimiter will help it learn to STOP and answer.

Usage:
    python scripts/prepare_code.py
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import hashlib

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

BOS_TOKEN_ID = 8188


class FallbackEncoder:
    """Byte-level encoder, always works (no install needed)."""
    def encode(self, text: str) -> list[int]:
        ids = []
        for ch in text:
            if ord(ch) < 256:
                ids.append(ord(ch))
            else:
                for b in ch.encode("utf-8"):
                    ids.append(256 + b)
        return ids


try:
    import rustbpe
    enc = rustbpe.Encoder()
    print("Using rustbpe (BPE)")
except ImportError:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        print("Using tiktoken")
    except ImportError:
        print("No rustbpe/tiktoken, using byte-level fallback")
        enc = FallbackEncoder()


# ---------------------------------------------------------------------------
# Sources (in priority order; first available wins)
# ---------------------------------------------------------------------------

SOURCES = [
    {
        "name": "CodeAlpaca-20k",
        "url": "https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json",
        "kind": "codealpaca",
    },
    {
        "name": "iamtarun-python-alpaca",
        "url": None,  # parquet, needs auth
        "kind": "iamtarun",
    },
]


CACHE_DIR = os.path.expanduser("~/.cache/autoresearch")
OUT_FILE = os.path.join(CACHE_DIR, "code_python.txt")


# ---------------------------------------------------------------------------
# Source: CodeAlpaca-20k (JSON list)
# ---------------------------------------------------------------------------

def fetch_codealpaca(url: str) -> list[dict]:
    """Fetch and parse CodeAlpaca JSON. Returns list of {instruction, input, output}."""
    print(f"Fetching {url}...")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"  fetch failed: {e}")
        return []
    if not isinstance(data, list) or not data:
        print("  unexpected JSON shape")
        return []
    print(f"  loaded {len(data):,} records")
    return data


def format_alpaca(rec: dict) -> str | None:
    """Format one CodeAlpaca record as instruction → response.

    Quality filter: output must be at least 20 chars, must contain a python-y signal.
    """
    inst = (rec.get("instruction") or "").strip()
    inp = (rec.get("input") or "").strip()
    out = (rec.get("output") or "").strip()

    if not inst or not out:
        return None
    if len(out) < 20:
        return None
    if len(out) > 4000:
        return None
    # Must look like code (def/class/import/=/return/print/for/while)
    if not any(s in out for s in ("def ", "class ", "import ", "=", "return ", "print(", "for ", "while ", "if ")):
        return None

    # Alpaca-style block; matches CoT format model has seen
    if inp:
        text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}\n"
    else:
        text = f"### Instruction:\n{inst}\n\n### Response:\n{out}\n"
    return text


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_corpus(corpus: list[str], out: str, limit: int) -> int:
    """Encode corpus, write token ID rows. Returns n_rows."""
    seen = set()
    n_kept = 0
    n_seen = 0
    n_dup = 0
    t0 = time.time()
    tmp = out + ".tmp"
    try:
        os.remove(tmp)
    except OSError:
        pass

    try:
        f = open(tmp, "w")
    except OSError as e:
        print(f"Cannot open {tmp}: {e}")
        return 0
    with f:
        for text in corpus:
            n_seen += 1
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            if h in seen:
                n_dup += 1
                continue
            seen.add(h)

            try:
                ids = enc.encode(text)
                ids = [BOS_TOKEN_ID] + list(ids)
                f.write(" ".join(str(i) for i in ids) + "\n")
                n_kept += 1
            except Exception as e:
                print(f"  encode err: {e}")

            if limit and n_kept >= limit:
                break

            if n_seen % 5000 == 0:
                rate = n_seen / (time.time() - t0 + 1e-9)
                print(f"  seen {n_seen:,}  kept {n_kept:,}  dup {n_dup:,}  {rate:.0f}/s")

    try:
        os.rename(tmp, out)
    except OSError as e:
        print(f"Cannot rename: {e}")
        return 0

    size_mb = os.path.getsize(out) / 1e6
    elapsed = time.time() - t0
    print(f"\nWrote {n_kept:,} rows → {out} ({size_mb:.1f} MB) in {elapsed:.1f}s")
    return n_kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=OUT_FILE)
    args = parser.parse_args()

    out = args.out
    try:
        os.makedirs(os.path.dirname(out), exist_ok=True)
    except OSError as e:
        print(f"Cannot mkdir: {e}")
        sys.exit(1)

    src = SOURCES[0]  # CodeAlpaca
    raw = fetch_codealpaca(src["url"])
    if not raw:
        print("FAIL: no data fetched")
        sys.exit(1)

    print(f"Filtering and formatting ({len(raw):,} raw → ...)...")
    corpus: list[str] = []
    for rec in raw:
        txt = format_alpaca(rec)
        if txt:
            corpus.append(txt)
    print(f"  after filter: {len(corpus):,} rows")

    n = write_corpus(corpus, out, args.limit)
    if n == 0:
        print("FAIL: 0 rows written")
        sys.exit(1)
    print(f"\n✓ Phase 1 ready: {n} rows of code-alpaca")


if __name__ == "__main__":
    main()
