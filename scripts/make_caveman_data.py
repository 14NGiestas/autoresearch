"""Normalize a CAVEMAN-register agentic-chat corpus into autoresearch training shards.

This is the data half of the "caveman LLM" idea (HEP hyp_10996a): take the
already-normalized Claude-style corpus in ~/.cache/autoresearch/data_claude/
(### USER / ### ASSISTANT blocks) and rewrite the ASSISTANT turns into a
"caveman" register while LEAVING CODE BLOCKS VERBATIM. The point is the model
keeps its task capability (the real content is intact in the backbone) but the
output head learns to emit caveman phrasing ("ME BUILD FIRE. YOU WATCH.").

USER turns are left normal on purpose: we want the model to answer normal
prompts in caveman, not require caveman input.

Output lands in ~/.cache/autoresearch/data_caveman/ with the same shard layout
(shard_00000.. + pinned val shard_06542) so prepare.py / train.py need no changes.

Usage (run from repo root):
  python3 scripts/make_caveman_data.py
"""

import glob
import os
import random
import re

from make_claude_data import redact  # reuse the secret redaction

CACHE = os.path.expanduser("~/.cache/autoresearch")
SRC = os.path.join(CACHE, "data_claude")              # normalized source corpus
DATA = os.path.join(CACHE, "data_caveman")           # caveman output corpus
VAL_SHARD = "shard_06542.parquet"
CHUNK_CHARS = 12000
SEED = 20260829
random.seed(SEED)

import pyarrow as pa
import pyarrow.parquet as pq

# --- caveman-ifier ----------------------------------------------------------
_WORD_SUBS = [
    (r"\bi am\b", "me be"),
    (r"\byou are\b", "you be"),
    (r"\bwe are\b", "we be"),
    (r"\bthey are\b", "they be"),
    (r"\bis\b", "be"),
    (r"\bare\b", "be"),
    (r"\bthe\b", ""),
    (r"\ba\b", ""),
    (r"\ban\b", ""),
    (r"\bplease\b", "plz"),
    (r"\bbecause\b", "cuz"),
    (r"\bhello\b", "ug hello"),
    (r"\bhi\b", "ug hi"),
    (r"\bthanks\b", "ug thanks"),
    (r"\bthank you\b", "ug thanks"),
    (r"\bunderstand\b", "get"),
    (r"\bcan you\b", "you can"),
    (r"\bhelp me\b", "help ug"),
    (r"\bhelp us\b", "help we"),
    (r"\blet me\b", "me do"),
    (r"\bwe will\b", "we go"),
    (r"\bi will\b", "me go"),
    (r"\bdo not\b", "no"),
    (r"\bdoes not\b", "no"),
    (r"\bcannot\b", "no can"),
    (r"\bcan not\b", "no can"),
    (r"\bwith\b", "n'"),
    (r"\bfor\b", "fo"),
    (r"\bto\b", "ta"),
    (r"\bthat\b", "dat"),
    (r"\bthis\b", "dis"),
    (r"\bvery\b", "big"),
    (r"\bgood\b", "nice"),
    (r"\bnow\b", "now"),
    (r"\bthen\b", "den"),
    (r"\bhere\b", "here"),
    (r"\byes\b", "yeh"),
    (r"\bno\b", "na"),
]


def _caveman_text(t):
    t = t.lower()
    for pat, rep in _WORD_SUBS:
        t = re.sub(pat, rep, t)
    # collapse whitespace and fix spacing around punctuation
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    # occasional grunt at sentence starts
    def _grunt(m):
        lead, letter = m.group(1), m.group(2)
        return lead + ("ug " if random.random() < 0.22 else "") + letter
    t = re.sub(r"(^|[.!?]\s+)([a-z])", _grunt, t)
    return t


def cavemanify(text):
    """Caveman-ify free text but leave fenced ```code``` blocks untouched."""
    parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    out = []
    for i, seg in enumerate(parts):
        if i % 2 == 1:           # code block
            out.append(seg)
        else:
            out.append(_caveman_text(seg))
    return "".join(out)


# --- render / chunk (mirrors make_claude_data.py) ---------------------------
def chunk(text, size=CHUNK_CHARS):
    if len(text) <= size:
        return [text]
    return [text[i:i + size] for i in range(0, len(text), size)]


def doc_to_blocks(doc):
    """Split a '### USER\n..\n\n### ASSISTANT\n..' doc into (role, body) pairs."""
    blocks = []
    for part in re.split(r"\n### ", doc):
        if not part.strip():
            continue
        # first line is the tag, rest is body
        nl = part.find("\n")
        if nl == -1:
            continue
        tag = part[:nl].strip().upper()
        body = part[nl + 1:]
        role = "user" if tag.startswith("USER") else ("assistant" if tag.startswith("ASSISTANT") else None)
        if role:
            blocks.append((role, body))
    return blocks


def main():
    files = sorted(glob.glob(os.path.join(SRC, "shard_*.parquet")))
    files = [f for f in files if os.path.basename(f) != VAL_SHARD]
    if not files:
        raise SystemExit(f"No source shards in {SRC}; run make_claude_data.py first.")
    docs = []
    for f in files:
        table = pq.read_table(f, columns=["text"])
        for t in table.column("text").to_pylist():
            if not t:
                continue
            blocks = doc_to_blocks(t)
            rendered = []
            for role, body in blocks:
                body = redact(body)
                if role == "assistant":
                    body = cavemanify(body)
                tag = "USER" if role == "user" else "ASSISTANT"
                rendered.append(f"### {tag}\n{body}")
            text = "\n\n".join(rendered)
            if text.strip():
                docs.append(text)
    random.shuffle(docs)
    try:
        os.makedirs(DATA, exist_ok=True)
    except OSError as e:
        raise SystemExit(f"cannot create {DATA}: {e}")
    n_val = max(1, len(docs) // 10)
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]
    print(f"Source shards: {len(files)}  docs: {len(docs)}  (~{sum(len(x) for x in docs)//4:,} tokens)")
    print(f"  train={len(train_docs)}  val={len(val_docs)} (pinned {VAL_SHARD})")

    def write_shard(name, subset):
        pq.write_table(pa.table({"text": pa.array(subset, type=pa.string())}),
                       os.path.join(DATA, name))

    chunk_sz = max(1, (len(train_docs) + 4 - 1) // 4)
    for i in range(4):
        part = train_docs[i * chunk_sz:(i + 1) * chunk_sz]
        if part:
            write_shard(f"shard_{i:05d}.parquet", part)
    write_shard(VAL_SHARD, val_docs)
    print(f"Wrote 4 train shards + {VAL_SHARD} to {DATA}")


if __name__ == "__main__":
    main()
