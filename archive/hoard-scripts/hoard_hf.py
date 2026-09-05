#!/usr/bin/env python3
"""Hoard a HuggingFace dataset into autoresearch parquet shards (### USER / ### ASSISTANT).

This grows the training corpus with tokenizer-COMPATIBLE text (English by default)
to raise the tok/param ratio -> less memorization (the root-cause lever).

Reads HF_TOKEN / HUGGING_FACE_HUB_TOKEN from env (set via .env).
Usage:  python3 scripts/hoard_hf.py --repo HuggingFaceH4/ultrachat_200k --out data_en_extra
"""
import os
import math
import random
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi


def norm_doc(messages):
    blocks = []
    for m in messages:
        r = (m.get("role") or "").lower()
        c = m.get("content") or m.get("text") or ""
        if r in ("user", "human"):
            blocks.append(("USER", c))
        elif r in ("assistant", "bot", "gpt", "system"):
            # keep system as context but fold into the preceding turn lightly
            blocks.append(("ASSISTANT" if r != "system" else "USER", c))
    if len(blocks) >= 2:
        return "\n\n".join(f"### {r}\n{c}" for r, c in blocks)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF dataset repo id")
    ap.add_argument("--file", default=None, help="specific parquet; else ALL parquets")
    ap.add_argument("--out", required=True, help="output shard dir")
    ap.add_argument("--limit", type=int, default=0, help="cap docs (debug)")
    ap.add_argument("--val-frac", type=float, default=0.10)
    a = ap.parse_args()

    try:
        os.makedirs(a.out, exist_ok=True)
    except OSError as e:
        print(f"[warn] cannot create {a.out}: {e}")

    api = HfApi()
    files = list(api.list_repo_tree(repo_id=a.repo, repo_type="dataset", recursive=True))
    parquets = [f.path for f in files if f.path.endswith(".parquet")]
    targets = [a.file] if a.file else parquets
    if not targets:
        raise SystemExit(f"no .parquet in {a.repo}")
    print(f"[hoard] {len(targets)} parquet file(s) in {a.repo}")

    docs = []
    for target in targets:
        print(f"[hoard] downloading {a.repo}/{target}")
        p = hf_hub_download(repo_id=a.repo, repo_type="dataset", filename=target,
                            local_dir="/tmp/hf_cache")
        table = pq.read_table(p)
        for row in table.to_pylist():
            msgs = None
            for k in ("messages", "chat", "conversations"):
                if k in row and row[k]:
                    msgs = row[k]
                    break
            if not msgs:
                continue
            d = norm_doc(msgs)
            if d:
                docs.append(d)
        print(f"[hoard]   {target}: rows={table.num_rows} docs_so_far={len(docs)}")

    if a.limit:
        docs = docs[:a.limit]
    random.seed(1)
    random.shuffle(docs)
    n_val = max(1, math.floor(len(docs) * a.val_frac))
    val, train = docs[:n_val], docs[n_val:]
    print(f"[hoard] total docs={len(docs)} train={len(train)} val={len(val)} -> {a.out}")

    def write(name, subset):
        pq.write_table(pa.table({"text": pa.array(subset, type=pa.string())}),
                       os.path.join(a.out, name))

    write("shard_06542.parquet", val)
    nchunk = 4
    chunk = max(1, (len(train) + nchunk - 1) // nchunk)
    for i in range(nchunk):
        part = train[i * chunk:(i + 1) * chunk]
        if part:
            write(f"shard_{i:05d}.parquet", part)
    print("[hoard] done")


if __name__ == "__main__":
    main()
