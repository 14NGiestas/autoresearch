import re
from pathlib import Path

from .base import BasePlatform


class Grok(BasePlatform):
    name = "grok"
    pattern = re.compile(r"grok\.com/share/([A-Za-z0-9_=-]+)", re.I)
    share_url = "https://grok.com/share/{}"
    api = "https://grok.com/rest/app-chat/share_links_data/{}"

    def fetch(self, uuid, http, raw_dir):
        out = raw_dir / f"grok_{uuid}.json"
        http.download(
            self.api.format(uuid),
            out,
            headers={"Accept": "application/json", "Referer": "https://grok.com/"},
        )
        return out

    def parse(self, path: Path):
        d = self._load_json(path)
        err = self._is_error_obj(d)
        if err:
            return 0, err
        return len(d.get("responses") or []), None
