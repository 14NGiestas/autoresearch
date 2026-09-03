"""Normalize the hoarded Claude export corpus into autoresearch training shards.

Claude raw files are Anthropic "claude.ai" conversation exports:
  { uuid, conversation_uuid, ..., chat_messages: [ { sender, content:[...] } ] }
Each content block has type: text | tool_use | tool_result | token_budget | ...

We render each conversation to a document (USER/ASSISTANT turns with tool calls),
redact obvious secrets, chunk long sessions, and write parquet shards into
~/.cache/autoresearch/data_claude/ with the pinned validation filename
shard_06542.parquet (matching prepare.py).

IMPORTANT: this script does NOT retrain the tokenizer. The claude-style fine-tune
continues from the baseline model, so it MUST reuse the baseline tokenizer
(~/.cache/autoresearch/tokenizer). Run the training with:

  AUTORESEARCH_DATA_DIR=~/.cache/autoresearch/data_claude \\
  AUTORESEARCH_TIME_BUDGET=14400 \\
  python3 train.py

Run inside the ROCm nix shell:
  nix develop --command python3 make_claude_data.py
"""

import glob
import json
import os
import random
import re

import pyarrow as pa
import pyarrow.parquet as pq

CACHE = os.path.expanduser("~/.cache/autoresearch")
DATA = os.path.join(CACHE, "data_claude")          # separate dir from the mixed-hoard run
VAL_SHARD = "shard_06542.parquet"                   # prepare.py's pinned validation filename
TRAIN_SHARDS = 4
CHUNK_CHARS = 12000                                 # ~3000 tokens per document
VAL_FRAC = 0.10
SEED = 20260827

# --- secret redaction (don't memorize API keys in the model) ---------------
_SECRET_RE = [
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"glpat-[A-Za-z0-9_-]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"ya29\.[0-9A-Za-z_-]{20,}"),
]
_SECRET_SUB = "[REDACTED]"


def redact(text):
    if not text:
        return text
    for rx in _SECRET_RE:
        text = rx.sub(_SECRET_SUB, text)
    return text


# --- block conversion (Claude export -> common {role, blocks} format) -------
def claude_to_messages(chat_messages):
    """Convert a Claude `chat_messages` list into the common message format used
    by render_session(): [{'role': 'user'|'assistant', 'blocks':[{type,...}]}]."""
    out = []
    for m in chat_messages:
        sender = m.get("sender")
        role = "user" if sender == "human" else "assistant"
        blocks = []
        for b in (m.get("content") or []):
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                txt = b.get("text", "") or ""
                if txt.strip():
                    blocks.append({"type": "text", "text": redact(txt)})
            elif t == "tool_use":
                blocks.append({
                    "type": "tool",
                    "name": b.get("name", "tool"),
                    "input": b.get("input"),
                    "output": "",
                })
            elif t == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    c = " ".join(
                        x.get("text", "") if isinstance(x, dict) else str(x) for x in c
                    )
                elif c is None:
                    c = b.get("structured_content") or ""
                name = b.get("name") or "tool_result"
                blocks.append({
                    "type": "tool",
                    "name": name,
                    "input": "",
                    "output": redact(str(c)),
                })
            elif t == "thinking":
                txt = b.get("thinking") or b.get("text", "")
                if txt:
                    blocks.append({"type": "reasoning", "text": redact(txt)})
            # token_budget and any other internal block types: skip
        # fallback: some messages keep the whole text at message level
        if not blocks and (m.get("text") or "").strip():
            blocks.append({"type": "text", "text": redact(m["text"])})
        if blocks:
            out.append({"role": role, "blocks": blocks})
    return out


def block_text(b):
    t = b.get("type")
    if t in ("text", "reasoning"):
        return b.get("text", "") or ""
    if t == "tool":
        name = b.get("name", "tool")
        inp = json.dumps(b.get("input"), ensure_ascii=False)
        out = json.dumps(b.get("output"), ensure_ascii=False)
        return f"[tool:{name}]\n{inp}\n-->\n{out}\n"
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
    os.makedirs(DATA, exist_ok=True)
    files = sorted(glob.glob(os.path.join("hoard_multi/raw", "claude_*.json")))
    docs = []
    n_ok = n_err = 0
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            n_err += 1
            continue
        if not isinstance(d, dict) or "chat_messages" not in d:
            n_err += 1
            continue
        msgs = claude_to_messages(d["chat_messages"])
        text = render_session(msgs)
        if not text.strip():
            continue
        for c in chunk(text):
            docs.append(c)
        n_ok += 1

    print(f"Claude files: {len(files)}  real={n_ok}  skipped(error)={n_err}")
    print(f"Total documents: {len(docs)}  (~{sum(len(x) for x in docs)//4:,} tokens)")

    random.shuffle(docs)
    n_val = max(1, int(VAL_FRAC * len(docs)))
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]
    print(f"  train={len(train_docs)}  val={len(val_docs)} (pinned {VAL_SHARD})")

    def write_shard(name, subset):
        t = pa.table({"text": pa.array(subset, type=pa.string())})
        pq.write_table(t, os.path.join(DATA, name))

    chunk_sz = max(1, (len(train_docs) + TRAIN_SHARDS - 1) // TRAIN_SHARDS)
    for i in range(TRAIN_SHARDS):
        part = train_docs[i * chunk_sz:(i + 1) * chunk_sz]
        if part:
            write_shard(f"shard_{i:05d}.parquet", part)
    write_shard(VAL_SHARD, val_docs)
    print(f"Wrote {TRAIN_SHARDS} train shards + {VAL_SHARD} to {DATA}")


if __name__ == "__main__":
    main()
