"""Text-based link extraction and local file ingestion."""
from __future__ import annotations

import re
from pathlib import Path

from ..platforms import ALL_PATTERNS


def extract_from_text(text: str) -> dict[str, set]:
    """Return {platform: {uuid, ...}} for every share link found in `text`."""
    out: dict[str, set] = {}
    for name, rx in ALL_PATTERNS.items():
        found = {m.lower() for m in rx.findall(text)}
        if found:
            out[name] = found
    return out


def ingest_file(store, path: str | Path) -> int:
    text = Path(path).read_text(errors="ignore")
    n = 0
    for platform, ids in extract_from_text(text).items():
        for uid in ids:
            n += store.add(uid, platform, f"file:{path}")
    return n
