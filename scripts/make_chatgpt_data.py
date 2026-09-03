"""Normalize a hoarded ChatGPT / Codex agentic-chat export into autoresearch
training shards, and APPEND them to the existing Claude-style corpus at
~/.cache/autoresearch/data_claude/  WITHOUT clobbering the shards that
make_claude_data.py already wrote (shard_00000..00007 + pinned val shard_06542).

ChatGPT shared / exported conversations use the "mapping" format:

  { "mapping": { "<node_id>": { "id":..., "message": { "author": {"role": ...},
      "content": {"content_type":"text","parts":["..."]}, "create_time": ... },
      "parent": "<parent_id>"|null, "children":[...] }, ... }, "title":...,
    "current_node":"<leaf_id>" }

For a raw export the top-level object may instead be a list of
{"role": "user"|"assistant", "content": "..."} messages (OpenAI-style). Both are
handled. Secrets are redacted and long sessions are chunked, exactly like the
Claude normalizer.

Usage:
  python3 make_chatgpt_data.py <path-to-chatgpt-export.json>
  python3 make_chatgpt_data.py            # auto: first hoard_multi/raw/chatgpt_*.json

The new shards are written as shard_NNNNN.parquet with NNNNN continuing right
after the highest existing train-shard index, so prepare.py simply sees more
training data the next time it builds the dataset.
"""

import glob
import json
import os
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq

CACHE = os.path.expanduser("~/.cache/autoresearch")
DATA = os.path.join(CACHE, "data_claude")            # SAME dir as the Claude corpus
VAL_SHARD = "shard_06542.parquet"                    # prepare.py's pinned validation filename
CHUNK_CHARS = 12000                                  # ~3000 tokens per document
SEED = 20260828

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


# --- ChatGPT mapping -> ordered (role, text) turns --------------------------
def parse_mapping(d):
    """Return a list of (role, text) turns from a ChatGPT mapping export."""
    mapping = d.get("mapping") or {}
    if not isinstance(mapping, dict):
        return []
    turns = []
    for node in mapping.values():
        if not isinstance(node, dict):
            continue
        msg = node.get("message")
        if not isinstance(msg, dict):
            continue
        author = (msg.get("author") or {}).get("role")
        if author not in ("user", "assistant"):
            continue
        content = msg.get("content")
        parts = []
        if isinstance(content, dict):
            raw = content.get("parts") or []
            for p in raw:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict) and p.get("content_type") == "text":
                    parts.append(p.get("text", ""))
                elif isinstance(p, dict) and "text" in p:
                    parts.append(str(p.get("text", "")))
        elif isinstance(content, str):
            parts.append(content)
        text = redact("\n".join(p for p in parts if p).strip())
        if not text:
            continue
        ts = msg.get("create_time") or 0
        turns.append((ts, author, text))
    # stable order by creation time (keeps the conversation in chat order)
    turns.sort(key=lambda t: (t[0] if isinstance(t[0], (int, float)) else 0))
    return [(r, t) for _, r, t in turns]


def parse_openai_list(d):
    """Fallback: a list of {role, content} messages."""
    out = []
    for m in d:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        c = m.get("content")
        if isinstance(c, str):
            text = redact(c.strip())
        elif isinstance(c, list):  # content parts
            text = redact("\n".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in c if isinstance(p, (str, dict))
            ).strip())
        else:
            continue
        if text:
            out.append((role, text))
    return out


def parse_plain_text(text):
    """Last-resort: a transcript with USER:/ASSISTANT: markers."""
    out = []
    cur_role = None
    buf = []
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("user:") or low.startswith("### user"):
            if cur_role and buf:
                out.append((cur_role, redact("\n".join(buf).strip())))
            cur_role, buf = "user", [line.split(":", 1)[-1].strip()]
        elif low.startswith("assistant:") or low.startswith("### assistant"):
            if cur_role and buf:
                out.append((cur_role, redact("\n".join(buf).strip())))
            cur_role, buf = "assistant", [line.split(":", 1)[-1].strip()]
        elif cur_role:
            buf.append(line)
    if cur_role and buf:
        out.append((cur_role, redact("\n".join(buf).strip())))
    return out


def render_session(turns):
    out = []
    for role, text in turns:
        tag = "USER" if role == "user" else "ASSISTANT"
        if text.strip():
            out.append(f"### {tag}\n{text}")
    return "\n\n".join(out)


def chunk(text, size=CHUNK_CHARS):
    if len(text) <= size:
        return [text]
    return [text[i:i + size] for i in range(0, len(text), size)]


def next_shard_index():
    """Highest existing train-shard index (excluding the val shard) + 1."""
    existing = [f for f in glob.glob(os.path.join(DATA, "shard_*.parquet"))
                if os.path.basename(f) != VAL_SHARD]
    idx = 0
    for f in existing:
        m = re.search(r"shard_(\d+)\.parquet", os.path.basename(f))
        if m:
            try:
                idx = max(idx, int(m.group(1)) + 1)
            except (ValueError, TypeError):
                pass
    return idx


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else None
    if not src:
        candidates = sorted(glob.glob(os.path.join("hoard_multi", "raw", "chatgpt_*.json")))
        if not candidates:
            candidates = sorted(glob.glob("/tmp/chatgpt*.json"))
        if not candidates:
            print("No ChatGPT export found. Pass a path: python3 make_chatgpt_data.py <file.json>")
            return
        src = candidates[0]

    print(f"Reading ChatGPT export: {src}")
    try:
        with open(src, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: cannot read/parse {src}: {e}")
        return

    if isinstance(d, dict) and "mapping" in d:
        turns = parse_mapping(d)
        src_kind = "mapping"
    elif isinstance(d, list):
        turns = parse_openai_list(d)
        src_kind = "openai-list"
    elif isinstance(d, dict) and ("chat_messages" in d or "messages" in d):
        # tolerate a claude-ish blob by reusing the openai path is wrong; fall back
        turns = parse_plain_text(json.dumps(d)[:0]) or []
        src_kind = "unknown-dict"
    else:
        turns = []
        src_kind = "none"

    if not turns:
        # try plain-text fallback on the raw file contents
        try:
            with open(src, "r", encoding="utf-8", errors="replace") as fh:
                turns = parse_plain_text(fh.read())
        except OSError as e:
            print(f"ERROR: cannot read {src}: {e}")
            return
        src_kind = "plain-text"

    print(f"  parsed {len(turns)} turns ({src_kind})")
    if not turns:
        print("No user/assistant turns found - nothing to write.")
        return

    text = render_session(turns)
    docs = chunk(text)
    print(f"  {len(docs)} document shard(s) after chunking ({sum(len(x) for x in docs):,} chars)")

    try:
        os.makedirs(DATA, exist_ok=True)
    except OSError as e:
        print(f"ERROR: cannot create {DATA}: {e}")
        return
    start = next_shard_index()
    n_shards = max(1, (len(docs) + 1999) // 2000)  # ~2000 docs per shard
    per = max(1, (len(docs) + n_shards - 1) // n_shards)
    written = []
    for i in range(n_shards):
        part = docs[i * per:(i + 1) * per]
        if not part:
            break
        name = f"shard_{start + i:05d}.parquet"
        pq.write_table(pa.table({"text": pa.array(part, type=pa.string())}),
                       os.path.join(DATA, name))
        written.append(name)
    print(f"Appended {len(written)} train shard(s) to {DATA}: {written}")
    print(f"(Existing val shard {VAL_SHARD} left untouched.)")


if __name__ == "__main__":
    main()
