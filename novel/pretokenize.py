"""
pretokenize.py — flatten the hoard into a single token .bin (llama2.c style).

The live baseline is data-loading bound (mfu 0.2%, dt ~75s/step): every step the
dataloader re-reads parquet + re-tokenizes on CPU. Pre-tokenizing once to a flat
uint16 .bin + an mmap dataloader removes that cost entirely (hyp_9f47fd).

This is CPU/disk only and writes to a SEPARATE dir (~/.cache/autoresearch/data_tokenized),
so it never touches the live baseline's parquet pipeline. Run it after the baseline
frees the machine, or with `nice` if run concurrently.

Usage:  python3 novel/pretokenize.py
"""
import os
import sys
import json

import pyarrow.parquet as pq
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prepare

OUT = os.path.expanduser("~/.cache/autoresearch/data_tokenized")
DATA_DIR = prepare.DATA_DIR
SEQ = prepare.MAX_SEQ_LEN
tok = prepare.Tokenizer.from_directory()
bos = tok.get_bos_token_id()


def text_column(path):
    """Pick the first string-ish column (TinyStories-style corpora use 'text')."""
    try:
        schema = pq.read_schema(path)
        for f in schema.names:
            if str(schema.field(f).type).startswith("string") or "utf" in str(schema.field(f).type):
                return f
        # fallback: first column
        return schema.names[0]
    except Exception as e:
        raise RuntimeError(f"cannot read parquet schema from {path}: {e}") from e


def main():
    files = [f for f in prepare.list_parquet_files()
             if os.path.basename(f) != prepare.VAL_FILENAME]
    print(f"[pretokenize] {len(files)} train shards -> {OUT}")
    col = text_column(files[0])
    print(f"[pretokenize] using text column: {col}")

    tokens = []
    for i, f in enumerate(files):
        table = pq.read_table(f, columns=[col])
        for t in table.column(col).to_pylist():
            if not t:
                continue
            tokens.extend(tok.encode(t, prepend=bos))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(files)} shards, {len(tokens):,} tokens so far")

    pad = (-len(tokens)) % SEQ
    tokens = tokens + [0] * pad
    arr = np.array(tokens, dtype=np.uint16)
    try:
        os.makedirs(OUT, exist_ok=True)
        arr.tofile(os.path.join(OUT, "train.bin"))
        meta = {"num_tokens": len(tokens) - pad, "padded": int(pad),
                "seq": SEQ, "vocab": int(tok.get_vocab_size())}
        with open(os.path.join(OUT, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
    except OSError as e:
        assert False, f"[pretokenize] failed to write outputs: {e}"
    print(f"[pretokenize] wrote {meta['num_tokens']:,} tokens -> {OUT}/train.bin")


if __name__ == "__main__":
    main()
