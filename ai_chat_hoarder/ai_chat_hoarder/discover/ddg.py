"""DuckDuckGo discovery: search for posted share links, crawl results."""
from __future__ import annotations

import time

from .files import extract_from_text
from .github import crawl_github

SKIP_DOMAINS = (
    "reddit.com",
    "google.",
    "youtube.com",
    "twitter.com",
    "x.com",
    "facebook.com",
)

DDG_QUERIES = {
    "claude": [
        '"claude.ai/share/"',
        'site:github.com "claude.ai/share"',
        'site:gist.github.com "claude.ai/share"',
        '"claude.ai/api/chat_snapshots/"',
    ],
    "chatgpt": [
        '"chatgpt.com/share/"',
        'site:github.com "chatgpt.com/share"',
        'site:gist.github.com "chatgpt.com/share"',
    ],
    "deepseek": [
        '"chat.deepseek.com/a/chat/s/"',
        'site:github.com "chat.deepseek.com"',
    ],
    "qwen": [
        '"chat.qwen.ai/share/"',
        'site:github.com "chat.qwen.ai/share"',
    ],
    "gemini": [
        '"gemini.google.com/share/"',
        '"g.co/gemini/share/"',
    ],
}


def discover_ddg(store, http, platforms=None, per=30, delay=2.0, crawl=True) -> int:
    try:
        from ddgs import DDGS
    except ImportError:
        print("ddgs not installed (pip install ddgs); skipping DDG")
        return 0

    plats = platforms or list(DDG_QUERIES)
    total = 0
    for plat in plats:
        for q in DDG_QUERIES.get(plat, []):
            try:
                results = DDGS().text(q, max_results=per, backend="api")
            except Exception as e:
                print(f"  ddg err {q!r}: {e}")
                continue
            for r in results:
                u = r.get("href", "")
                if any(b in u for b in SKIP_DOMAINS):
                    continue
                txt = http.text(u)
                if not txt:
                    continue
                for fp, ids in extract_from_text(txt).items():
                    for i in ids:
                        total += store.add(i, fp, "ddg:" + u)
                if crawl and ("github.com" in u or "gist.github" in u):
                    total += crawl_github(store, u, http)
                time.sleep(0.3)
            time.sleep(delay)
    return total
