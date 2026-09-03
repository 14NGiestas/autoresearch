import re
from pathlib import Path

from .base import BasePlatform


class Qwen(BasePlatform):
    name = "qwen"
    pattern = re.compile(
        r"chat\.qwen(?:lm)?\.ai/(?:s|share)/([0-9a-zA-Z-]{20,40})", re.I
    )
    share_url = "https://chat.qwen.ai/s/{}"
    api = "https://chat.qwen.ai/api/v2/chats/share/{}"

    def fetch(self, uuid, http, raw_dir):
        out = raw_dir / f"qwen_{uuid}.json"
        http.download(
            self.api.format(uuid),
            out,
            headers={"Accept": "application/json", "Referer": "https://chat.qwen.ai/"},
        )
        return out

    def parse(self, path: Path):
        d = self._load_json(path)
        if not d.get("success"):
            return 0, str(d.get("message", "qwen failure"))[:120]
        try:
            return len(d["data"]["chat"]["messages"] or []), None
        except Exception:
            return 0, None
