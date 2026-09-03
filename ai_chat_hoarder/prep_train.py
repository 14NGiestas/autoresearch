"""Training-prep exporter for the hoarded corpus.

Reads structured agent JSON from ``hoard_multi/raw/<platform>_*.json``,
keeps the configured platforms, converts each multi-turn session into
OpenAI-style ``messages`` (preserving the agent timeline: text, reasoning,
tool calls, patches), sliding-window chunks any session longer than
``--max-seq`` tokens, dedupes, and writes a ready-to-train dataset.

Output:
  <out>/train.jsonl     one line per training example: {"messages":[...]}
  <out>/train_text.jsonl one line per example: {"text": "<role-tagged conversation>"}
  <out>/stats.json      corpus / chunking statistics

Run (inside the ROCm nix shell):
  python3 prep_train.py --hoard-dir ../hoard_multi --out ../hoard_multi/train
"""

import argparse
import glob
import hashlib
import json
import os
from collections import Counter


def block_text(b: dict) -> str:
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
    if t in ("agent", "image"):
        return ""
    return ""


def render_message(m: dict):
    role = m.get("role")
    if role not in ("user", "assistant", "system"):
        role = "user"
    content = "\n".join(p for p in (block_text(b) for b in m.get("blocks", [])) if p)
    return role, content


def session_to_messages(msgs: list) -> list:
    out = []
    for m in msgs:
        role, content = render_message(m)
        if not content.strip():
            continue
        out.append({"role": role, "content": content})
    return out


def est_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def chunk_session(messages: list, max_seq: int, overlap: int) -> list:
    ex = []
    n = len(messages)
    s = 0
    while s < n:
        e = s
        toks = 0
        while e < n and toks + est_tokens(messages[e]["content"]) <= max_seq:
            toks += est_tokens(messages[e]["content"])
            e += 1
        if e == s:  # single message exceeds max_seq -> hard truncate
            big = messages[s]["content"][: max_seq * 4]
            ex.append([{**messages[s], "content": big}])
            s += 1
            continue
        ex.append(messages[s:e])
        if e >= n:
            break
        s2 = e
        ot = 0
        while s2 > s and ot + est_tokens(messages[s2 - 1]["content"]) <= overlap:
            ot += est_tokens(messages[s2 - 1]["content"])
            s2 -= 1
        s = s2
    return ex


def render_text(messages: list) -> str:
    out = []
    for m in messages:
        tag = {"user": "USER", "assistant": "ASSISTANT", "system": "SYSTEM"}.get(
            m["role"], "USER"
        )
        out.append(f"### {tag}\n{m['content']}")
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hoard-dir", default="../hoard_multi")
    ap.add_argument("--out", default="../hoard_multi/train")
    ap.add_argument(
        "--platforms",
        default="opencode,pi,claude,gemini",
        help="comma-separated platforms to keep (filename prefix)",
    )
    ap.add_argument("--max-seq", type=int, default=32768)
    ap.add_argument("--overlap", type=int, default=1024)
    args = ap.parse_args()

    keep = {p.strip() for p in args.platforms.split(",") if p.strip()}
    raw_dir = os.path.join(args.hoard_dir, "raw")
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
    seen = set()
    per_platform = Counter()
    per_platform_sessions = Counter()
    examples = []
    too_long = 0
    max_win = 0

    for f in files:
        prefix = os.path.basename(f).split("_", 1)[0].lower()
        if prefix not in keep:
            continue
        try:
            data = json.load(open(f))
        except Exception:
            continue
        if not isinstance(data, list) or not data:
            continue
        messages = session_to_messages(data)
        per_platform_sessions[prefix] += 1
        if not messages:
            continue
        chunks = chunk_session(messages, args.max_seq, args.overlap)
        for ch in chunks:
            key = hashlib.sha256(
                json.dumps(ch, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            win_tok = sum(est_tokens(m["content"]) for m in ch)
            max_win = max(max_win, win_tok)
            if win_tok > args.max_seq:
                too_long += 1
            examples.append(
                {
                    "messages": ch,
                    "text": render_text(ch),
                    "platform": prefix,
                    "est_tokens": win_tok,
                }
            )
        per_platform[prefix] += len(chunks)

    with open(os.path.join(args.out, "train.jsonl"), "w") as fj, open(
        os.path.join(args.out, "train_text.jsonl"), "w"
    ) as ft:
        for ex in examples:
            fj.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")
            ft.write(json.dumps({"text": ex["text"]}, ensure_ascii=False) + "\n")

    stats = {
        "examples": len(examples),
        "platform_sessions": dict(per_platform_sessions),
        "platform_chunks": dict(per_platform),
        "max_window_tokens": max_win,
        "chunks_over_maxseq": too_long,
        "max_seq": args.max_seq,
        "overlap": args.overlap,
    }
    json.dump(stats, open(os.path.join(args.out, "stats.json"), "w"), indent=2)

    print("Wrote", len(examples), "training examples to", args.out)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
