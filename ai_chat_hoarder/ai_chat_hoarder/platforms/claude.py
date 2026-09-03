import re
from pathlib import Path

from .base import BasePlatform

CLAUDE_API = "https://claude.ai/api/chat_snapshots/{}?rendering_mode=messages&render_all_tools=true"


class Claude(BasePlatform):
    name = "claude"
    pattern = re.compile(
        r"claude\.ai/(?:api/chat_snapshots|share)/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
        re.I,
    )
    share_url = "https://claude.ai/share/{}"

    def fetch(self, uuid, http, raw_dir):
        out = raw_dir / f"claude_{uuid}.json"
        http.download(
            CLAUDE_API.format(uuid),
            out,
            headers={"Accept": "application/json", "Referer": "https://claude.ai/"},
        )
        return out

    def parse(self, path: Path):
        d = self._load_json(path)
        err = self._is_error_obj(d)
        if err:
            return 0, err
        return len(d.get("chat_messages") or []), None
