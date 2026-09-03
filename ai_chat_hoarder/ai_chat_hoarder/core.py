"""Core orchestration: fetch loop and discovery re-exports."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .db import Store
from .http import CHROME_UA, FIREFOX_UA, HttpClient
from .platforms import PLATFORMS
from .discover import discover_ddg, crawl_github, ingest_file, set_token
from . import local as local_mod

__all__ = [
    "Store",
    "HttpClient",
    "fetch_all",
    "reaudit",
    "corpus_stats",
    "import_local",
    "import_archive",
    "discover_ddg",
    "crawl_github",
    "ingest_file",
    "set_token",
    "CHROME_UA",
    "FIREFOX_UA",
]

import_local = local_mod.import_local


def _extract_archive(archive, dest):
    """Extract a .tar.zst / .tar.gz / .tgz into dest; return True on success."""
    import subprocess
    dest.mkdir(parents=True, exist_ok=True)
    cmds = [
        ["tar", "-xf", str(archive), "-C", str(dest)],
        ["tar", "--use-compress-program=unzstd", "-xf", str(archive), "-C", str(dest)],
        ["tar", "--zstd", "-xf", str(archive), "-C", str(dest)],
    ]
    for c in cmds:
        try:
            subprocess.run(c, check=True, stderr=subprocess.DEVNULL)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    try:
        p1 = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
        subprocess.run(["tar", "-xf", "-", "-C", str(dest)], stdin=p1.stdout, check=True)
        return True
    except Exception:
        return False


def import_archive(archive_path, hoard_dir):
    """Import a b.sh-produced archive into the hoard.

    Extracts the archive, copies any new raw JSON into <hoard>/raw, and merges
    the archive's snaps table into <hoard>/index.db (deduped by uuid+platform).
    Returns {files_copied, db_added}.
    """
    import shutil
    import tempfile
    from pathlib import Path as _P
    from .db import Store

    archive = _P(archive_path)
    hoard = _P(hoard_dir)
    raw = hoard / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    tmp = _P(tempfile.mkdtemp())
    try:
        if not _extract_archive(archive, tmp):
            raise RuntimeError(f"could not extract archive: {archive}")
        raw_src = next(tmp.rglob("raw"), None)
        db_src = next(tmp.rglob("index.db"), None)

        copied = 0
        if raw_src:
            for f in raw_src.glob("*.json"):
                d = raw / f.name
                if not d.exists():
                    shutil.copy(f, d)
                    copied += 1

        db_added = 0
        if db_src:
            store = Store(hoard / "index.db")
            src = __import__("sqlite3").connect(str(db_src))
            before = store.db.execute("SELECT COUNT(*) FROM snaps").fetchone()[0]
            rows = src.execute(
                "SELECT lower(uuid),platform,source,found_at,fetched,msg_count,error "
                "FROM snaps"
            ).fetchall()
            store.db.executemany(
                "INSERT OR IGNORE INTO snaps(uuid,platform,source,found_at,fetched,msg_count,error) "
                "VALUES(?,?,?,?,?,?,?)", rows)
            store.db.commit()
            after = store.db.execute("SELECT COUNT(*) FROM snaps").fetchone()[0]
            db_added = after - before
        return {"files_copied": copied, "db_added": db_added}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fetch_all(store: Store, raw_dir: Path, http: HttpClient | None = None,
              limit: int | None = None, delay: float = 1.0) -> dict:
    """Fetch every pending snapshot. Returns per-platform success/fail counts."""
    http = http or HttpClient()
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = store.pending(limit)
    counts = {"ok": 0, "err": 0}
    for uuid, platform in rows:
        plat = PLATFORMS.get(platform)
        if plat is None:
            store.mark_error(uuid, platform, "unknown-platform")
            counts["err"] += 1
            continue
        out = plat.fetch(uuid, http, raw_dir)
        if not (out.exists() and out.stat().st_size > 0):
            store.mark_error(uuid, platform, "empty")
            counts["err"] += 1
            time.sleep(delay)
            continue
        try:
            msg_count, error = plat.parse(out)
        except json.JSONDecodeError:
            out.unlink()
            store.mark_error(uuid, platform, "non-json")
            counts["err"] += 1
            time.sleep(delay)
            continue
        if error:
            store.mark_error(uuid, platform, error)
            counts["err"] += 1
        else:
            store.mark_fetched(uuid, platform, msg_count)
            counts["ok"] += 1
        time.sleep(delay)
    return counts


def reaudit(store: Store, raw_dir: Path, platforms=None) -> dict:
    """Re-parse already-fetched raw files and correct msg_count/error in the DB.

    Useful after fixing a parser: turns false-positive 'successful' fetches
    (e.g. error payloads) into properly flagged errors.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    corrected = {"ok": 0, "err": 0}
    rows = store.db.execute("SELECT uuid, platform FROM snaps WHERE fetched=1").fetchall()
    for uuid, platform in rows:
        if platforms and platform not in platforms:
            continue
        plat = PLATFORMS.get(platform)
        if plat is None:
            continue
        out = None
        for ext in ("json", "html"):
            candidate = raw_dir / f"{platform}_{uuid}.{ext}"
            if candidate.exists():
                out = candidate
                break
        if out is None:
            continue
        try:
            msg_count, error = plat.parse(out)
        except json.JSONDecodeError:
            store.mark_error(uuid, platform, "non-json")
            corrected["err"] += 1
            continue
        if error:
            store.mark_error(uuid, platform, error)
            corrected["err"] += 1
        else:
            store.mark_fetched(uuid, platform, msg_count)
            corrected["ok"] += 1
    return corrected


def _claude_text(d):
    out = 0
    for m in (d.get("chat_messages") or []):
        c = m.get("content")
        if isinstance(c, str):
            out += len(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    out += len(str(b.get("text", "")))
    return out


def _chatgpt_text(d):
    out = 0
    m = d.get("mapping") or {}
    for node in (m.values() if isinstance(m, dict) else []):
        msg = (node or {}).get("message") if isinstance(node, dict) else None
        if not msg:
            continue
        c = msg.get("content") or {}
        parts = c.get("parts") if isinstance(c, dict) else None
        if isinstance(parts, list):
            for p in parts:
                if isinstance(p, str):
                    out += len(p)
    return out


def _grok_text(d):
    return sum(len(str(r.get("message", ""))) for r in (d.get("responses") or [])
              if isinstance(r, dict))


def _simple_text(m):
    if isinstance(m, list):
        return sum(len(str(x.get("content", ""))) for x in m if isinstance(x, dict))
    return 0


def _text(v):
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return "" if v is None else str(v)


def _msg_chars(m):
    """Character count of a normalized local/structured message (blocks format)."""
    if not isinstance(m, dict):
        return 0
    blocks = m.get("blocks")
    if isinstance(blocks, list):
        ch = 0
        for b in blocks:
            if not isinstance(b, dict):
                continue
            for k in ("text", "thinking", "name", "title"):
                ch += len(_text(b.get(k)))
            for k in ("input", "output", "files"):
                v = b.get(k)
                if v is not None:
                    ch += len(_text(v))
        return ch
    c = m.get("content")
    if isinstance(c, str):
        return len(c)
    if isinstance(c, list):
        return sum(len(str(x.get("text", ""))) for x in c if isinstance(x, dict))
    return 0


def corpus_stats(raw_dir: Path) -> dict:
    """Walk fetched raw JSON and return per-platform conversation/turn/char counts.

    Tokens are estimated as chars // 4 (no BPE tokenizer applied). Only valid
    (non-error) parses are counted.
    """
    raw_dir = Path(raw_dir)
    agg = {}
    for f in raw_dir.glob("*.json"):
        plat = f.name.split("_")[0]
        try:
            d = json.loads(f.read_text(errors="ignore"))
        except Exception:
            continue
        p = PLATFORMS.get(plat)
        if p is not None:
            try:
                n, err = p.parse(f)
            except Exception:
                continue
            if err or not n:
                continue
        else:
            # Local / normalized file: a flat list of turns.
            n = len(d) if isinstance(d, list) else 0
            if not n:
                continue
        if plat == "claude":
            ch = _claude_text(d)
        elif plat == "chatgpt":
            ch = _chatgpt_text(d)
        elif plat == "grok":
            ch = _grok_text(d)
        elif plat == "deepseek":
            ch = sum(len(str(x.get("content", "")))
                     for x in (d.get("data", {}).get("biz_data", {}).get("messages") or [])
                     if isinstance(x, dict))
        elif plat == "qwen":
            ch = sum(len(str(x.get("content", "")))
                     for x in (d.get("data", {}).get("chat", {}).get("messages") or [])
                     if isinstance(x, dict))
        else:
            # gemini + normalized local sessions (opencode/pi/goose/...) carry
            # either flat {content} or structured {blocks} message lists.
            ch = sum(_msg_chars(m) for m in d) if isinstance(d, list) else _simple_text(d)
        a = agg.setdefault(plat, {"conv": 0, "turns": 0, "chars": 0})
        a["conv"] += 1
        a["turns"] += n
        a["chars"] += ch
    return agg
