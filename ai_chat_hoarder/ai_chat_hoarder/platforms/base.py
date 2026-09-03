"""Platform plugin interface."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path


class BasePlatform(ABC):
    name: str
    pattern: re.Pattern
    share_url: str

    @abstractmethod
    def fetch(self, uuid: str, http, raw_dir: Path) -> Path:
        """Download the conversation; return the path written."""

    @abstractmethod
    def parse(self, path: Path) -> tuple[int, str | None]:
        """Return (msg_count, error). error is None on success."""

    # -- helpers shared by JSON platforms -------------------------------
    @staticmethod
    def _load_json(path: Path):
        return json.loads(path.read_text())

    @staticmethod
    def _is_error_obj(d) -> str | None:
        if isinstance(d, dict):
            if "detail" in d:
                return str(d["detail"].get("code", d))[:120]
            if d.get("type") == "error" or "error" in d:
                err = d.get("error", {})
                if isinstance(err, dict):
                    return str(err.get("message", err))[:120]
                return str(err)[:120]
            if d.get("code") not in (0, None):
                return str(d.get("message", d))[:120]
        return None
