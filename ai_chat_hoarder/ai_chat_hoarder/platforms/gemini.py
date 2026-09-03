import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path

from .base import BasePlatform

CHROMIUM_BIN = os.environ.get("CHROMIUM_BIN", "chromium")
GEMINI_COOKIE = os.environ.get("GEMINI_COOKIE")


class _ConversationExtractor(HTMLParser):
    """Pull user/assistant turns from Gemini's rendered share DOM.

    Each turn renders as a <user-query> (with .query-text) followed by a
    <response-container> (with .markdown). We track a stack of matched
    elements (not a raw depth counter) so void elements like <img>/<br>
    inside message content don't desync the extraction.
    """

    TARGETS = {"query-text": "user", "response-container": "assistant"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.msgs = []

    def handle_starttag(self, tag, attrs):
        cls = " ".join(v for k, v in attrs if k == "class").split()
        for name, role in self.TARGETS.items():
            if name in cls:
                self.stack.append({"role": role, "parts": [], "tag": tag})
                break

    def handle_endtag(self, tag):
        if self.stack and self.stack[-1]["tag"] == tag:
            frame = self.stack.pop()
            text = " ".join(frame["parts"]).strip()
            if text:
                self.msgs.append({"role": frame["role"], "content": text})

    def handle_data(self, data):
        if self.stack:
            self.stack[-1]["parts"].append(data)


class Gemini(BasePlatform):
    name = "gemini"
    pattern = re.compile(
        r"(?:g\.co/gemini/share|gemini\.google\.com/share)/([0-9a-zA-Z-]{10,30})", re.I
    )
    share_url = "https://gemini.google.com/share/{}"

    def fetch(self, uuid, http, raw_dir):
        out_html = raw_dir / f"gemini_{uuid}.html"
        cookie = os.environ.get("GEMINI_COOKIE")
        if cookie:
            # Authenticated: render the SPA with headless Chromium so the
            # conversation is delivered in the DOM, then archive that HTML and
            # extract a clean JSON conversation for training use.
            url = self.share_url.format(uuid)
            cmd = [
                CHROMIUM_BIN, "--headless=new", "--no-sandbox",
                "--disable-gpu", "--disable-dev-shm-usage",
                f"--cookie={cookie}", "--virtual-time-budget=15000",
                "--dump-dom", url,
            ]
            try:
                import subprocess
                with open(out_html, "w", encoding="utf-8") as fh:
                    subprocess.run(cmd, stdout=fh, stderr=subprocess.DEVNULL,
                                   timeout=90, check=False)
            except Exception:
                pass
            if out_html.exists() and out_html.stat().st_size > 0:
                ex = _ConversationExtractor()
                ex.feed(out_html.read_text(errors="ignore"))
                if ex.msgs:
                    out_json = raw_dir / f"gemini_{uuid}.json"
                    out_json.write_text(
                        json.dumps(ex.msgs, ensure_ascii=False, indent=1),
                        encoding="utf-8",
                    )
                    return out_json
        if not (out_html.exists() and out_html.stat().st_size > 0):
            # Fallback / unauthenticated: the shell page (sign-in gated).
            http.download(
                self.share_url.format(uuid),
                out_html,
                headers={"Accept": "text/html", "Referer": "https://gemini.google.com/"},
            )
        return out_html

    def parse(self, path: Path):
        if path.suffix == ".json":
            try:
                d = json.loads(path.read_text())
            except Exception:
                return 0, "non-json"
            return len(d) if isinstance(d, list) else 0, None
        html = path.read_text(errors="ignore")
        ex = _ConversationExtractor()
        ex.feed(html)
        msgs = ex.msgs
        if msgs:
            return len(msgs), None
        # No conversation extracted: decide whether it's an auth gate or a
        # render failure. A sign-in gate means we need GEMINI_COOKIE.
        if re.search(r"sign[ -]?in", html, re.I):
            return 0, "auth_required (Gemini share is sign-in gated; set GEMINI_COOKIE)"
        return 0, "no_conversation_rendered"
