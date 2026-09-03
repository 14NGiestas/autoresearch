import re
from pathlib import Path

from ..http import FIREFOX_UA
from .base import BasePlatform


class ChatGPT(BasePlatform):
    name = "chatgpt"
    pattern = re.compile(
        r"chatgpt\.com/share/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
        re.I,
    )
    share_url = "https://chatgpt.com/share/{}"
    api = "https://chatgpt.com/backend-api/share/{}"

    def fetch(self, uuid, http, raw_dir):
        out = raw_dir / f"chatgpt_{uuid}.json"
        headers = {"Accept": "application/json", "Referer": "https://chatgpt.com/"}
        cookie = __import__("os").environ.get("CHATGPT_COOKIE")
        if cookie:
            headers["Cookie"] = cookie
        # Firefox UA required; Chrome UA is gated with quorum_verification_required
        http.download(self.api.format(uuid), out, headers=headers, ua=FIREFOX_UA)
        return out

    def parse(self, path: Path):
        d = self._load_json(path)
        err = self._is_error_obj(d)
        if err:
            return 0, err
        m = d.get("mapping")
        return (len(m) if isinstance(m, dict) else 0), None
