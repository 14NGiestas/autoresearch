"""Import local agent/chat sessions (opencode, pi, goose, qwen-coder, ...) as
training conversations.

Unlike the web platforms, these are already on disk. Each session is normalized
into a structured message list that PRESERVES agentic content:

    {"role": "user"|"assistant", "agent": <name|null>,
     "blocks": [
        {"type": "text", "text": "..."},
        {"type": "reasoning", "text": "..."},          # thinking / CoT
        {"type": "tool", "id":.., "name":.., "input":.., "output":..},
        {"type": "patch", "files": [...]},
        {"type": "agent", "name": "subagent-name"},
        {"type": "image", "mime":.., "filename":..}     # base64 dropped
     ]}

This keeps tool calls, results, reasoning and code diffs instead of flattening
to plain text, which is what agentic training actually needs.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .db import Store


def _read_jsonl(path):
    out = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _text(v):
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return "" if v is None else str(v)


# --------------------------------------------------------------------------
# opencode (SQLite: message.data.role + part.data.<type>)
# --------------------------------------------------------------------------
def parse_opencode_db(db_path):
    db = sqlite3.connect(str(db_path))
    parts = {}
    for pid, mid, t, pdata in db.execute(
        "SELECT id, message_id, time_created, data FROM part"
    ):
        parts.setdefault(mid, []).append((t, pdata))
    result = {}
    rows = db.execute(
        "SELECT id, session_id, data, time_created FROM message ORDER BY session_id, time_created"
    ).fetchall()
    for mid, sid, mdata, _t in rows:
        try:
            md = json.loads(mdata)
        except Exception:
            continue
        role = md.get("role")
        if role not in ("user", "assistant"):
            continue
        blocks = []
        for t, pdata in sorted(parts.get(mid, []), key=lambda x: x[0]):
            try:
                p = json.loads(pdata)
            except Exception:
                continue
            bt = p.get("type")
            if bt == "text":
                blocks.append({"type": "text", "text": p.get("text", "")})
            elif bt == "reasoning":
                blocks.append({"type": "reasoning", "text": p.get("text", "")})
            elif bt == "tool":
                st = p.get("state", {}) or {}
                blocks.append({
                    "type": "tool",
                    "id": p.get("callID"),
                    "name": p.get("tool"),
                    "title": st.get("title"),
                    "input": st.get("input"),
                    "output": st.get("output"),
                })
            elif bt == "patch":
                blocks.append({"type": "patch", "files": p.get("files", [])})
            elif bt == "agent":
                blocks.append({"type": "agent", "name": p.get("name")})
            elif bt == "file":
                blocks.append({"type": "image", "mime": p.get("mime"),
                               "filename": p.get("filename")})
        if blocks:
            result.setdefault(sid, []).append(
                {"role": role, "agent": md.get("agent"), "blocks": blocks}
            )
    return result


# --------------------------------------------------------------------------
# pi (JSONL events: {"type":"message","message":{"role", "content":[{type..}]}})
# --------------------------------------------------------------------------
def parse_pi(path):
    turns = []
    for ev in _read_jsonl(path):
        if ev.get("type") != "message":
            continue
        m = ev.get("message") or {}
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        blocks = []
        for b in (m.get("content") or []):
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                blocks.append({"type": "text", "text": b.get("text", "")})
            elif bt in ("thinking", "reasoning"):
                blocks.append({"type": "reasoning",
                               "text": b.get("thinking") or b.get("text", "")})
            elif bt == "toolCall":
                blocks.append({"type": "tool", "id": b.get("id"),
                               "name": b.get("name"),
                               "input": b.get("arguments")})
            elif bt == "image":
                blocks.append({"type": "image", "mime": b.get("mime"),
                               "filename": b.get("filename")})
        if blocks:
            turns.append({"role": role, "blocks": blocks})
    return turns


# --------------------------------------------------------------------------
# goose (JSONL events with top-level role + content:[{type..}])
# --------------------------------------------------------------------------
def parse_goose(path):
    turns = []
    for ev in _read_jsonl(path):
        role = ev.get("role") or ev.get("type")
        if role not in ("user", "assistant"):
            continue
        blocks = []
        for b in (ev.get("content") or ev.get("parts") or []):
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                blocks.append({"type": "text", "text": b.get("text", "")})
            elif bt == "toolRequest":
                tc = b.get("toolCall", {}) or {}
                v = tc.get("value", {}) or {}
                blocks.append({"type": "tool", "id": b.get("id"),
                               "name": v.get("name"), "input": v.get("arguments")})
            elif bt == "toolResponse":
                tr = b.get("toolResult", {}) or {}
                # merge into an existing tool block with the same id if present
                for blk in blocks:
                    if blk.get("type") == "tool" and blk.get("id") == b.get("id"):
                        blk["output"] = tr
                        break
                else:
                    blocks.append({"type": "tool", "id": b.get("id"), "output": tr})
        if blocks:
            turns.append({"role": role, "blocks": blocks})
    return turns


# --------------------------------------------------------------------------
# claude (Claude Code: ~/.claude/projects/<hash>/<uuid>.jsonl)
# --------------------------------------------------------------------------
def parse_claude(path):
    turns = []
    for ev in _read_jsonl(path):
        m = ev.get("message") or {}
        role = m.get("role") or ev.get("type")
        if role not in ("user", "assistant"):
            continue
        blocks = []
        content = m.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for b in (content or []):
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                blocks.append({"type": "text", "text": b.get("text", "")})
            elif bt in ("thinking", "reasoning"):
                blocks.append({"type": "reasoning",
                               "text": b.get("thinking") or b.get("text", "")})
            elif bt == "tool_use":
                blocks.append({"type": "tool", "id": b.get("id"),
                               "name": b.get("name"), "input": b.get("input")})
            elif bt == "tool_result":
                out = b.get("content")
                for blk in blocks:
                    if blk.get("type") == "tool" and blk.get("id") == b.get("tool_use_id"):
                        blk["output"] = out
                        break
                else:
                    blocks.append({"type": "tool", "id": b.get("tool_use_id"), "output": out})
        if blocks:
            turns.append({"role": role, "blocks": blocks})
    return turns


# --------------------------------------------------------------------------
# generic fallback (qwen-coder / Gemini-CLI-style JSONL)
# --------------------------------------------------------------------------
def parse_generic(path):
    turns = []
    for ev in _read_jsonl(path):
        role = ev.get("role") or ev.get("type")
        if role not in ("user", "assistant"):
            continue
        blocks = []
        for b in (ev.get("content") or ev.get("parts") or []):
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                blocks.append({"type": "text", "text": b.get("text", "")})
            elif bt in ("thinking", "reasoning"):
                blocks.append({"type": "reasoning",
                               "text": b.get("thinking") or b.get("text", "")})
            elif bt in ("tool_use", "toolCall"):
                blocks.append({"type": "tool", "id": b.get("id"),
                               "name": b.get("name"),
                               "input": b.get("input") or b.get("arguments")})
            elif bt in ("tool_result", "toolResponse"):
                out = b.get("content") or b.get("output") or b.get("toolResult")
                for blk in blocks:
                    if blk.get("type") == "tool" and blk.get("id") == b.get("id"):
                        blk["output"] = out
                        break
                else:
                    blocks.append({"type": "tool", "id": b.get("id"), "output": out})
        if blocks:
            turns.append({"role": role, "blocks": blocks})
    return turns


def _expand(p):
    p = Path(p)
    if p.is_dir():
        return sorted(p.rglob("*.jsonl"))
    return [p]


def import_local(store: Store, raw_dir: Path, platform: str, sources) -> int:
    """Normalize local sessions for `platform` from `sources` and register them.

    `sources` is a list of files/dirs. For 'opencode' each source is a .db file;
    for the JSONL tools each source is a file or a directory of *.jsonl.
    Returns the number of sessions imported.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    added = 0

    if platform == "opencode":
        for src in sources:
            for sid, turns in parse_opencode_db(Path(src)).items():
                if not turns:
                    continue
                out = raw_dir / f"{platform}_{sid}.json"
                out.write_text(json.dumps(turns, ensure_ascii=False, indent=1), encoding="utf-8")
                store.add(sid, platform, "local")
                store.mark_fetched(sid, platform, len(turns))
                added += 1
        return added

    parser = {"pi": parse_pi, "goose": parse_goose, "claude": parse_claude}.get(
        platform, parse_generic)
    for src in sources:
        for path in _expand(Path(src)):
            try:
                turns = parser(path)
            except Exception:
                continue
            if not turns:
                continue
            sid = path.stem
            out = raw_dir / f"{platform}_{sid}.json"
            out.write_text(json.dumps(turns, ensure_ascii=False, indent=1), encoding="utf-8")
            store.add(sid, platform, "local")
            store.mark_fetched(sid, platform, len(turns))
            added += 1
    return added
