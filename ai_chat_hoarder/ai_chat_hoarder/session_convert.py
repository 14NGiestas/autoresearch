"""Convert an ``opencode`` session (stored in its SQLite ``opencode*.db``)
into a ``pi`` agent session transcript (``*.jsonl``).

Why this exists
---------------
``opencode`` keeps its conversation history in a Drizzle/SQLite database
(``~/.local/share/opencode/opencode-stable.db``) with three relevant tables:

* ``session``  - one row per session (title, directory, model, timestamps).
* ``message``  - one row per message turn; the ``data`` JSON carries
  ``role`` (``user``/``assistant``), timing, token usage, model/provider.
* ``part``     - one row per content block (``text``, ``reasoning``,
  ``tool``, ``step-start``/``step-finish``, ``compaction``). A ``tool``
  part folds BOTH the tool call (``state.input``) and its result
  (``state.output``/``state.error``) into a single record.

``pi`` instead stores a flat JSONL event log where each line is one of
``session`` / ``model_change`` / ``thinking_level_change`` / ``message``.
Tool calls live inside the assistant ``message`` (as ``toolCall`` parts)
but their **results** are emitted as SEPARATE ``message`` events with
``role == "toolResult"`` that hang off the assistant turn. This module
splits opencode's combined ``tool`` parts into that shape.

Usage
-----
    # convert the most recently updated opencode session
    python -m ai_chat_hoarder.session_convert

    # list available sessions (newest first)
    python -m ai_chat_hoarder.session_convert --list

    # convert a specific session id
    python -m ai_chat_hoarder.session_convert --session-id ses_xxx --out out.jsonl

    # from Python
    from ai_chat_hoarder.session_convert import convert_latest_opencode_session
    path = convert_latest_opencode_session()
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

__all__ = [
    "find_opencode_db",
    "list_opencode_sessions",
    "find_latest_opencode_session",
    "read_opencode_session",
    "convert_opencode_session",
    "write_pi_session",
    "convert_latest_opencode_session",
]

# --------------------------------------------------------------------------- #
# Paths / small helpers
# --------------------------------------------------------------------------- #

_HOME = os.path.expanduser("~")
DEFAULT_PI_SESSIONS_DIR = os.path.join(_HOME, ".pi", "agent", "sessions")


def _default_opencode_db() -> str:
    """Return the most recently modified ``opencode*.db`` we can find."""
    candidates = [
        os.path.join(_HOME, ".local", "share", "opencode", "opencode-stable.db"),
        os.path.join(_HOME, ".local", "share", "opencode", "opencode.db"),
        os.path.join(_HOME, ".local", "state", "opencode", "opencode.db"),
    ]
    existing = [c for c in candidates if os.path.exists(c)]
    if not existing:
        raise FileNotFoundError(
            "No opencode database found (looked in ~/.local/share/opencode and "
            "~/.local/state/opencode)."
        )
    # prefer the most recently modified
    return max(existing, key=os.path.getmtime)


def _iso_ms(ms: Optional[int]) -> str:
    """Format epoch-milliseconds as ``2026-08-27T17:45:25.377Z`` (pi style)."""
    if ms is None:
        ms = 0
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(ms) % 1000:03d}Z"


def _uid() -> str:
    """pi-style 8-hex message id (e.g. ``c1f152d1``)."""
    return uuid.uuid4().hex[:8]


def _slugify_cwd(cwd: str) -> str:
    """Mirror pi's session directory name: ``/home/pauli/x`` -> ``--home-pauli-x--``."""
    norm = cwd.strip().lstrip("/")
    return "--" + norm.replace("/", "-") + "--"


# --------------------------------------------------------------------------- #
# Read opencode
# --------------------------------------------------------------------------- #


def find_opencode_db(db_path: Optional[str] = None) -> str:
    if db_path:
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"opencode db not found: {db_path}")
        return db_path
    return _default_opencode_db()


def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def list_opencode_sessions(
    db_path: Optional[str] = None, limit: int = 20
) -> list[dict]:
    """Return session rows (newest first) with a ``message_count`` column."""
    db = find_opencode_db(db_path)
    con = _connect(db)
    try:
        cur = con.execute(
            """
            SELECT s.id, s.slug, s.title, s.directory, s.model,
                   s.time_created, s.time_updated, s.cost,
                   (SELECT COUNT(*) FROM message m WHERE m.session_id = s.id) AS message_count
            FROM session s
            ORDER BY s.time_updated DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def find_latest_opencode_session(db_path: Optional[str] = None) -> dict:
    """Return the most recently updated session row."""
    sessions = list_opencode_sessions(db_path, limit=1)
    if not sessions:
        raise ValueError("No opencode sessions found in the database.")
    return sessions[0]


def read_opencode_session(
    db_path: Optional[str], session_id: str
) -> tuple[dict, list[dict]]:
    """Return ``(session_meta, messages)``.

    ``messages`` is ordered by ``time_created``; each message is a dict with
    ``id``, ``role``, ``parent_id``, ``time_created``, ``time_completed``,
    ``data`` (parsed JSON) and ``parts`` (parsed ``part.data`` list ordered
    by ``time_created``).
    """
    db = find_opencode_db(db_path)
    con = _connect(db)
    try:
        meta = con.execute(
            "SELECT * FROM session WHERE id = ?", (session_id,)
        ).fetchone()
        if meta is None:
            raise ValueError(f"Session not found: {session_id}")
        meta = dict(meta)

        msg_rows = con.execute(
            "SELECT id, data, time_created FROM message WHERE session_id = ? "
            "ORDER BY time_created ASC",
            (session_id,),
        ).fetchall()

        parts_by_msg = {}
        part_rows = con.execute(
            "SELECT message_id, data, time_created FROM part "
            "WHERE session_id = ? ORDER BY time_created ASC",
            (session_id,),
        )
        for pr in part_rows:
            parts_by_msg.setdefault(pr["message_id"], []).append(
                (pr["time_created"], json.loads(pr["data"]))
            )

        messages = []
        for mr in msg_rows:
            data = json.loads(mr["data"])
            parts = [p for _, p in parts_by_msg.get(mr["id"], [])]
            messages.append(
                {
                    "id": mr["id"],
                    "role": data.get("role"),
                    "parent_id": data.get("parentID"),
                    "time_created": mr["time_created"],
                    "time_completed": (
                        data.get("time", {}).get("completed")
                        if isinstance(data.get("time"), dict)
                        else None
                    ),
                    "data": data,
                    "parts": parts,
                }
            )
        return meta, messages
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Convert opencode -> pi events
# --------------------------------------------------------------------------- #


# opencode part types we simply drop (internal bookkeeping)
_SKIP_PART_TYPES = {"step-start", "step-finish"}


def _tool_result_text(state: dict) -> str:
    out = state.get("output")
    if out is None and "error" in state:
        err = state["error"]
        out = err if isinstance(err, str) else json.dumps(err, ensure_ascii=False)
    if out is None:
        out = ""
    return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)


def convert_opencode_session(
    session_meta: dict,
    messages: list[dict],
    *,
    thinking_level: str = "high",
) -> list[dict]:
    """Build the list of pi JSONL event dicts for an opencode session.

    Returns events in emission order: ``session`` header, a ``model_change``,
    an optional ``thinking_level_change``, then one ``message`` event per
    opencode turn (plus one ``toolResult`` ``message`` per tool call).
    """
    cwd = session_meta.get("directory") or _HOME
    session_id = str(uuid.uuid4())
    created_ms = session_meta.get("time_created") or messages[0]["time_created"]

    # derive provider/model for the model_change + assistant messages
    model_raw = session_meta.get("model")
    if isinstance(model_raw, str):
        try:
            model_raw = json.loads(model_raw)
        except json.JSONDecodeError:
            model_raw = None
    provider_id = (model_raw or {}).get("providerID") or "opencode"
    model_id = (model_raw or {}).get("id") or (model_raw or {}).get("modelID") or "unknown"

    events: list[dict] = []
    events.append(
        {
            "type": "session",
            "version": 3,
            "id": session_id,
            "timestamp": _iso_ms(created_ms),
            "cwd": cwd,
        }
    )
    events.append(
        {
            "type": "model_change",
            "id": _uid(),
            "parentId": None,
            "timestamp": _iso_ms(created_ms),
            "provider": provider_id,
            "modelId": model_id,
        }
    )
    if thinking_level:
        events.append(
            {
                "type": "thinking_level_change",
                "id": _uid(),
                "parentId": None,
                "timestamp": _iso_ms(created_ms),
                "thinkingLevel": thinking_level,
            }
        )

    ocid_to_pi: dict[str, str] = {}
    has_reasoning = any(
        any(p.get("type") == "reasoning" for p in m["parts"]) for m in messages
    )
    if not thinking_level and has_reasoning:
        # keep the thinking_level marker if reasoning exists and none was set
        events.append(
            {
                "type": "thinking_level_change",
                "id": _uid(),
                "parentId": None,
                "timestamp": _iso_ms(created_ms),
                "thinkingLevel": "high",
            }
        )

    for m in messages:
        role = m["role"]
        oc_id = m["id"]
        pi_id = _uid()
        ocid_to_pi[oc_id] = pi_id
        parent_pi = ocid_to_pi.get(m["parent_id"]) if m.get("parent_id") else None
        ts = _iso_ms(m["time_created"])

        if role == "user":
            text_parts = [p for p in m["parts"] if p.get("type") == "text"]
            texts = [p.get("text", "") for p in text_parts]
            content = []
            if texts:
                joined = "\n".join(t for t in texts if t)
                content.append(
                    {
                        "type": "text",
                        "text": joined,
                        "timestamp": m["time_created"],
                    }
                )
            events.append(
                {
                    "type": "message",
                    "id": pi_id,
                    "parentId": parent_pi,
                    "timestamp": ts,
                    "message": {
                        "role": "user",
                        "content": content,
                        "timestamp": m["time_created"],
                    },
                }
            )
            continue

        if role == "assistant":
            data = m["data"]
            content: list[dict] = []
            tool_calls: list[tuple[str, str, dict]] = []  # (callID, tool, state)

            for p in m["parts"]:
                ptype = p.get("type")
                if ptype in _SKIP_PART_TYPES:
                    continue
                if ptype == "reasoning":
                    content.append(
                        {
                            "type": "thinking",
                            "thinking": p.get("text", ""),
                            "thinkingSignature": "reasoning_content",
                        }
                    )
                elif ptype == "text":
                    if p.get("text"):
                        content.append({"type": "text", "text": p.get("text", "")})
                elif ptype == "tool":
                    call_id = p.get("callID") or _uid()
                    tool_name = p.get("tool", "tool")
                    state = p.get("state", {}) or {}
                    inp = state.get("input", {})
                    if not isinstance(inp, dict):
                        inp = {"input": inp}
                    content.append(
                        {
                            "type": "toolCall",
                            "id": call_id,
                            "name": tool_name,
                            "arguments": inp,
                        }
                    )
                    tool_calls.append((call_id, tool_name, state))
                elif ptype == "compaction":
                    # surface compaction as an assistant note
                    note = p.get("summary") or json.dumps(p, ensure_ascii=False)
                    content.append(
                        {"type": "text", "text": f"[compaction] {note}"}
                    )

            # tokens / usage
            tokens = data.get("tokens", {}) or {}
            cache = tokens.get("cache", {}) or {}
            inp = int(tokens.get("input", 0) or 0)
            out = int(tokens.get("output", 0) or 0)
            reasoning = int(tokens.get("reasoning", 0) or 0)
            cache_read = int(cache.get("read", 0) or 0)
            cache_write = int(cache.get("write", 0) or 0)
            cost = float(data.get("cost", 0) or 0)
            usage = {
                "input": inp,
                "output": out,
                "cacheRead": cache_read,
                "cacheWrite": cache_write,
                "totalTokens": inp + out + reasoning + cache_read,
                "cost": {
                    "input": cost,
                    "output": 0.0,
                    "cacheRead": 0.0,
                    "cacheWrite": 0.0,
                    "total": cost,
                },
            }

            if data.get("error"):
                stop_reason = "error"
            elif tool_calls:
                stop_reason = "tool_use"
            else:
                stop_reason = "stop"

            events.append(
                {
                    "type": "message",
                    "id": pi_id,
                    "parentId": parent_pi,
                    "timestamp": ts,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "api": "openai-completions",
                        "provider": data.get("providerID") or provider_id,
                        "model": data.get("modelID") or model_id,
                        "usage": usage,
                        "stopReason": stop_reason,
                        "timestamp": m["time_created"],
                        "responseId": str(uuid.uuid4()),
                    },
                }
            )

            # emit a toolResult message for each tool call
            for call_id, tool_name, state in tool_calls:
                is_error = state.get("status") == "error"
                result_ts = (state.get("time") or {}).get("end") or m["time_created"]
                events.append(
                    {
                        "type": "message",
                        "id": _uid(),
                        "parentId": pi_id,
                        "timestamp": _iso_ms(result_ts),
                        "message": {
                            "role": "toolResult",
                            "toolCallId": call_id,
                            "toolName": tool_name,
                            "content": [
                                {"type": "text", "text": _tool_result_text(state)}
                            ],
                            "isError": bool(is_error),
                            "timestamp": result_ts,
                        },
                    }
                )
            continue

        # any other role: skip silently (opencode only uses user/assistant)
        continue

    return events


# --------------------------------------------------------------------------- #
# Write pi session
# --------------------------------------------------------------------------- #


def write_pi_session(
    events: list[dict],
    out_path: Optional[str] = None,
    *,
    sessions_dir: str = DEFAULT_PI_SESSIONS_DIR,
    cwd: Optional[str] = None,
    when_ms: Optional[int] = None,
) -> str:
    """Write ``events`` as a pi ``*.jsonl`` file.

    If ``out_path`` is given it is used directly. Otherwise a path is derived
    under ``sessions_dir`` using pi's directory/filename conventions:
    ``<sessions_dir>/<slugified-cwd>/<timestamp>_<uuid>.jsonl``.
    Returns the written path.
    """
    if out_path:
        path = out_path
    else:
        if cwd is None:
            sess = next((e for e in events if e.get("type") == "session"), None)
            cwd = (sess or {}).get("cwd") or _HOME
        when_ms = when_ms or int(
            datetime.now(timezone.utc).timestamp() * 1000
        )
        subdir = os.path.join(sessions_dir, _slugify_cwd(cwd))
        os.makedirs(subdir, exist_ok=True)
        fname = f"{_iso_ms(when_ms).replace(':', '-')}_{uuid.uuid4().hex}.jsonl"
        path = os.path.join(subdir, fname)

    with open(path, "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return path


def convert_latest_opencode_session(
    db_path: Optional[str] = None,
    out_path: Optional[str] = None,
    sessions_dir: str = DEFAULT_PI_SESSIONS_DIR,
    thinking_level: str = "high",
) -> tuple[str, dict]:
    """End-to-end: find the newest opencode session and write a pi session.

    Returns ``(written_path, session_meta)``.
    """
    meta = find_latest_opencode_session(db_path)
    _, messages = read_opencode_session(db_path, meta["id"])
    events = convert_opencode_session(meta, messages, thinking_level=thinking_level)
    path = write_pi_session(
        events, out_path=out_path, sessions_dir=sessions_dir,
        cwd=meta.get("directory"),
    )
    return path, meta


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _print_sessions(sessions: Iterable[dict]) -> None:
    print(f"{'updated':<24} {'messages':>8}  {'model':<18} directory")
    print("-" * 90)
    for s in sessions:
        tu = s.get("time_updated") or s.get("time_created") or 0
        ts = _iso_ms(tu)
        model = s.get("model") or ""
        if isinstance(model, str):
            try:
                model = json.loads(model).get("id", model)
            except json.JSONDecodeError:
                pass
        print(
            f"{ts:<24} {str(s.get('message_count','')):>8}  "
            f"{str(model):<18} {s.get('directory','')}"
        )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert an opencode session into a pi session JSONL."
    )
    ap.add_argument("--db", help="Path to opencode SQLite db (default: auto-detect).")
    ap.add_argument("--session-id", help="Convert this specific session id.")
    ap.add_argument(
        "--out",
        help="Write to this file (default: a new pi session under "
        "~/.pi/agent/sessions/<cwd>/).",
    )
    ap.add_argument(
        "--sessions-dir",
        default=DEFAULT_PI_SESSIONS_DIR,
        help="Base pi sessions directory (default: ~/.pi/agent/sessions).",
    )
    ap.add_argument(
        "--list", action="store_true", help="List recent sessions and exit."
    )
    ap.add_argument(
        "--limit", type=int, default=20, help="Rows for --list (default: 20)."
    )
    ap.add_argument(
        "--thinking-level",
        default="high",
        help="thinking_level_change value (default: high; use '' to omit).",
    )
    args = ap.parse_args(argv)

    if args.list:
        _print_sessions(list_opencode_sessions(args.db, limit=args.limit))
        return 0

    if args.session_id:
        meta, messages = read_opencode_session(args.db, args.session_id)
        events = convert_opencode_session(
            meta, messages, thinking_level=args.thinking_level or ""
        )
        path = write_pi_session(
            events,
            out_path=args.out,
            sessions_dir=args.sessions_dir,
            cwd=meta.get("directory"),
        )
        print(f"wrote {path}")
        print(f"session: {meta.get('title')}  ({meta.get('directory')})")
        print(f"messages converted: {len(messages)}")
        return 0

    path, meta = convert_latest_opencode_session(
        db_path=args.db,
        out_path=args.out,
        sessions_dir=args.sessions_dir,
        thinking_level=args.thinking_level or "",
    )
    print(f"wrote {path}")
    print(f"session: {meta.get('title')}  ({meta.get('directory')})")
    print(f"messages converted: {meta.get('message_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
