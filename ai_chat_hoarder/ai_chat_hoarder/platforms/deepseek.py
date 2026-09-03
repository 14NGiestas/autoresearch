import re
from pathlib import Path

from .base import BasePlatform


class Deepseek(BasePlatform):
    name = "deepseek"
    pattern = re.compile(
        r"chat\.deepseek\.com/[^\s\"'<>)]{0,40}/s/([0-9a-zA-Z-]{10,40})", re.I
    )
    share_url = "https://chat.deepseek.com/a/chat/s/{}"
    api = "https://chat.deepseek.com/api/v0/share/content?share_id={}"

    def fetch(self, uuid, http, raw_dir):
        out = raw_dir / f"deepseek_{uuid}.json"
        http.download(
            self.api.format(uuid),
            out,
            headers={"Accept": "application/json", "Referer": "https://chat.deepseek.com/"},
        )
        return out

    def parse(self, path: Path):
        d = self._load_json(path)
        if d.get("data", {}).get("biz_code") != 0:
            return 0, str(d.get("data", {}).get("biz_msg", "deepseek failure"))[:120]
        try:
            return len(d["data"]["biz_data"]["messages"] or []), None
        except Exception:
            return 0, None
