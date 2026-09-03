"""HTTP layer: thin wrapper over `curl` (handles brotli/gzip via --compressed)."""
from __future__ import annotations

import subprocess
from pathlib import Path

CHROME_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
FIREFOX_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:153.0) Gecko/20100101 Firefox/153.0"


class HttpClient:
    def __init__(self, ua: str = CHROME_UA, timeout: int = 45):
        self.ua = ua
        self.timeout = timeout

    def _cmd(self, url: str, headers, ua):
        cmd = ["curl", "-s", "-L", "--compressed", "-A", ua or self.ua]
        for k, v in (headers or {}).items():
            cmd += ["-H", f"{k}: {v}"]
        cmd += [url]
        return cmd

    def download(self, url: str, out: Path, headers=None, ua=None) -> bool:
        cmd = ["curl", "-s", "-L", "--compressed", "-A", ua or self.ua]
        for k, v in (headers or {}).items():
            cmd += ["-H", f"{k}: {v}"]
        cmd += ["-o", str(out), url]
        try:
            subprocess.run(cmd, capture_output=True, timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired:
            return False
        return out.exists() and out.stat().st_size > 0

    def text(self, url: str, headers=None, ua=None) -> str | None:
        cmd = ["curl", "-s", "-L", "--compressed", "-A", ua or self.ua]
        for k, v in (headers or {}).items():
            cmd += ["-H", f"{k}: {v}"]
        cmd += [url]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0:
            return None
        return r.stdout.decode("utf-8", "ignore")
