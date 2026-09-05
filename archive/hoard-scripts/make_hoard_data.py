"""Convert the hoarded agent corpus into autoresearch training shards.

Reads structured agent JSON from hoard_multi/raw/{opencode,pi,claude,gemini}_*.json,
renders each multi-turn session to a conversation document, chunks long sessions,
writes parquet shards with a `text` column into ~/.cache/autoresearch/data/
(train = shard_000xx.parquet, val = shard_06542.parquet, matching prepare.py's
pinned val filename), then retrains the BPE tokenizer on OUR data.

Run inside the ROCm nix shell:
  nix develop --command python3 make_hoard_data.py
"""

import glob
import json
import os
import random

import pyarrow.parquet as pq
import pyarrow as pa

CACHE = os.path.expanduser("~/.cache/autoresearch")
DATA = os.path.join(CACHE, "data")
TOK = os.path.join(CACHE, "tokenizer")
PLATFORMS = ["opencode", "pi", "claude", "gemini"]
VAL_SHARD = "shard_06542.parquet"   # prepare.py's pinned validation filename
CHUNK_CHARS = 12000                 # ~3000 tokens per document
TRAIN_SHARDS = 8
SEED = 20260827


def block_text(b):
    t = b.get("type")
    if t in ("text", "reasoning"):
        return b.get("text", "") or ""
    if t == "tool":
        name = b.get("name", "tool")
        inp = json.dumps(b.get("input"), ensure_ascii=False)
        out = json.dumps(b.get("output"), ensure_ascii=False)
        return f"[tool:{name}]\n{inp}\n-->\n{out}\n"
    if t == "patch":
        files = " ".join(b.get("files", []) or [])
        return f"[patch {files}]\n{b.get('diff', '')}\n"
    return ""


def render_session(msgs):
    out = []
    for m in msgs:
        role = m.get("role", "user")
        tag = {"user": "USER", "assistant": "ASSISTANT", "system": "SYSTEM"}.get(role, "USER")
        content = "\n".join(p for p in (block_text(b) for b in m.get("blocks", [])) if p)
        if content.strip():
            out.append(f"### {tag}\n{content}")
    return "\n\n".join(out)


def chunk(text, size=CHUNK_CHARS):
    if len(text) <= size:
        return [text]
    return [text[i:i + size] for i in range(0, len(text), size)]


def main():
    random.seed(SEED)
    docs = []
    per_plat = {}
    for plat in PLATFORMS:
        files = sorted(glob.glob(os.path.join("/home/pauli/autoresearch/hoard_multi/raw", f"{plat}_*.json")))
        n = 0
        for f in files:
            try:
                data = json.load(open(f))
            except Exception:
                continue
            if not isinstance(data, list):
                continue
            text = render_session(data)
            if not text.strip():
                continue
            for c in chunk(text):
                docs.append(c)
            n += 1
        per_plat[plat] = n
        print(f"  {plat}: {n} sessions -> docs so far {len(docs)}")

    print(f"Total documents: {len(docs)}")
    random.shuffle(docs)
    n_val = max(1, int(0.10 * len(docs)))
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]
    print(f"  train={len(train_docs)}  val={len(val_docs)}")

    # Clear any existing climbmix shards + tokenizer so ours replace them.
    for fn in os.listdir(DATA):
        if fn.endswith(".parquet"):
            os.remove(os.path.join(DATA, fn))
    for fn in ("tokenizer.pkl", "token_bytes.pt"):
        p = os.path.join(TOK, fn)
        if os.path.exists(p):
            os.remove(p)

    def write_shard(name, docs_subset):
        t = pa.table({"text": pa.array(docs_subset, type=pa.string())})
        pq.write_table(t, os.path.join(DATA, name))

    # train shards
    chunk_sz = max(1, (len(train_docs) + TRAIN_SHARDS - 1) // TRAIN_SHARDS)
    for i in range(TRAIN_SHARDS):
        part = train_docs[i * chunk_sz:(i + 1) * chunk_sz]
        if part:
            write_shard(f"shard_{i:05d}.parquet", part)
    # val shard (pinned filename)
    write_shard(VAL_SHARD, val_docs)
    print(f"Wrote {TRAIN_SHARDS} train shards + {VAL_SHARD} to {DATA}")

    # Retrain tokenizer on our data (importing prepare reuses its logic).
    import prepare
    prepare.train_tokenizer()
    print("Tokenizer retrained on hoard corpus.")


if __name__ == "__main__":
    main()
