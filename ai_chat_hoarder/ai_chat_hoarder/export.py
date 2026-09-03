"""Export the hoard into training-ready JSONL.

Two parallel files are produced in the export dir:
  - train.jsonl       : chat format, each message content is a plain string
                        (text + reasoning + tool i/o concatenated). Good for SFT.
  - train.agent.jsonl : agentic format, each message content is a list of
                        structured blocks (text / reasoning / tool / patch / ...)
                        preserving tool calls and results for tool-calling training.

Conversations are deduplicated by a SHA-256 of their canonical form so repeated
harvests (or the same session seen on two machines) collapse to one example.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


# --------------------------------------------------------------------------
# block rendering
# --------------------------------------------------------------------------
def _block_text(b):
    t = b.get("type")
    if t in ("text", "reasoning"):
        return b.get("text", "") or ""
    if t == "tool":
        out = []
        if b.get("name"):
            out.append(f"[{b['name']}]")
        if b.get("input") is not None:
            out.append(json.dumps(b["input"], ensure_ascii=False))
        if b.get("output") is not None:
            out.append(json.dumps(b["output"], ensure_ascii=False))
        return "\n".join(out)
    if t == "patch":
        return "files: " + ", ".join(b.get("files", []) or [])
    if t == "agent":
        return f"[agent:{b.get('name')}]"
    if t == "image":
        return "[image]"
    return ""


def render_plain(blocks):
    return "\n".join(
        t for t in (_block_text(b) for b in blocks) if t.strip()
    )


# --------------------------------------------------------------------------
# canonical conversation: list of (role, blocks)
# --------------------------------------------------------------------------
def _claude_blocks(content):
    blocks = []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    for b in (content or []):
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "text":
            blocks.append({"type": "text", "text": b.get("text", "")})
        elif bt in ("thinking", "reasoning"):
            blocks.append({"type": "reasoning", "text": b.get("thinking") or b.get("text", "")})
        elif bt == "tool_use":
            blocks.append({"type": "tool", "name": b.get("name"),
                           "input": b.get("input"), "id": b.get("id")})
        elif bt == "tool_result":
            blocks.append({"type": "tool", "output": b.get("content"), "id": b.get("tool_use_id")})
        else:
            blocks.append({"type": "text", "text": b.get("text", "")})
    return blocks or [{"type": "text", "text": ""}]


def _chatgpt_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        parts = content.get("parts")
        if isinstance(parts, list):
            return "\n".join(p for p in parts if isinstance(p, str))
    return ""


def canonical_conv(platform, d):
    """Return [(role, blocks), ...] for a parsed raw file."""
    if platform in ("opencode", "pi", "goose", "claude"):
        # local canonical format: list of {role, agent, blocks}
        if isinstance(d, list):
            return [(m.get("role"), m.get("blocks", [])) for m in d
                    if isinstance(m, dict) and m.get("role") in ("user", "assistant")]

    if platform == "gemini":
        if isinstance(d, list):
            return [(m.get("role"), [{"type": "text", "text": m.get("content", "")}])
                    for m in d if isinstance(m, dict)]
    elif platform == "claude":
        msgs = (d.get("chat_messages") or []) if isinstance(d, dict) else []
        out = []
        for m in msgs:
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            out.append((role, _claude_blocks(m.get("content"))))
        return out
    elif platform == "chatgpt":
        mapping = (d.get("mapping") or {}) if isinstance(d, dict) else {}
        out = []
        for node in mapping.values():
            if not isinstance(node, dict):
                continue
            msg = node.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            out.append((role, [{"type": "text", "text": _chatgpt_text(msg.get("content"))}]))
        return out
    elif platform == "grok":
        resp = (d.get("responses") or []) if isinstance(d, dict) else []
        out = []
        for r in resp:
            if not isinstance(r, dict):
                continue
            role = "user" if r.get("sender") == "human" else "assistant"
            out.append((role, [{"type": "text", "text": r.get("message", "")}]))
        return out
    elif platform in ("deepseek", "qwen"):
        key = "biz_data" if platform == "deepseek" else "chat"
        sub = (d.get("data", {}).get(key, {}) if isinstance(d, dict) else {})
        msgs = sub.get("messages") or [] if isinstance(sub, dict) else []
        out = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            out.append((role, [{"type": "text", "text": m.get("content", "")}]))
        return out
    return []


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def export_jsonl(raw_dir, out_dir):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plain_path = out_dir / "train.jsonl"
    agent_path = out_dir / "train.agent.jsonl"

    seen = set()
    n_conv = 0
    n_dup = 0
    n_turn = 0
    plain_chars = 0

    with plain_path.open("w", encoding="utf-8") as fp, \
         agent_path.open("w", encoding="utf-8") as fa:
        for f in sorted(raw_dir.glob("*.json")):
            platform = f.name.split("_")[0]
            try:
                d = json.loads(f.read_text(errors="ignore"))
            except Exception:
                continue
            conv = canonical_conv(platform, d)
            if not conv:
                continue
            # dedupe on canonical form
            key = json.dumps(conv, sort_keys=True, ensure_ascii=False)
            h = hashlib.sha256(key.encode("utf-8")).hexdigest()
            if h in seen:
                n_dup += 1
                continue
            seen.add(h)

            msgs_plain = []
            for role, blocks in conv:
                txt = render_plain(blocks)
                if txt.strip():
                    msgs_plain.append({"role": role, "content": txt})
                    plain_chars += len(txt)
            if not msgs_plain:
                continue
            fa.write(json.dumps({"messages": [{"role": r, "content": b}
                                             for r, b in conv]},
                                ensure_ascii=False) + "\n")
            fp.write(json.dumps({"messages": msgs_plain}, ensure_ascii=False) + "\n")
            n_conv += 1
            n_turn += len(msgs_plain)

    return {
        "conversations": n_conv,
        "deduped": n_dup,
        "turns": n_turn,
        "plain_chars": plain_chars,
        "est_tokens": plain_chars // 4,
        "plain_file": str(plain_path),
        "agent_file": str(agent_path),
    }
