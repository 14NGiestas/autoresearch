"""Minimal .env loader (no external dependency).

Looks for a .env file in: the path you pass, the current working directory,
or this package's parent directory. Values already in os.environ win, so shell
exports take precedence over the file.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path=None) -> bool:
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.append(Path.cwd() / ".env")
    candidates.append(Path(__file__).resolve().parent.parent / ".env")
    for p in candidates:
        if not p.exists():
            continue
        for line in p.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)
        return True
    return False
