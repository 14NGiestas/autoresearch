"""GitHub repo / gist crawler: pull raw files and extract share links."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .files import extract_from_text

GITHUB_TOKEN = None  # set via discover.github.set_token(os.environ.get("GITHUB_TOKEN"))


def set_token(token: str | None):
    global GITHUB_TOKEN
    GITHUB_TOKEN = token


def _gh_headers():
    if GITHUB_TOKEN:
        return {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    return {}


def crawl_github(store, url: str, http, seen=None, depth=0) -> int:
    if seen is None:
        seen = set()
    if url in seen or depth > 2:
        return 0
    seen.add(url)
    n = 0

    if "gist.github" in url:
        gid = re.search(r"gist\.github\.com/[^/]+/([0-9a-f]+)", url)
        if not gid:
            return 0
        body = http.text(
            f"https://api.github.com/gists/{gid.group(1)}", headers=_gh_headers()
        )
        if not body:
            return 0
        try:
            files = json.loads(body).get("files", {})
        except Exception:
            return 0
        raws = [f["raw_url"] for f in files.values() if "raw_url" in f]
    else:
        m = re.search(r"github\.com/([^/]+)/([^/#?]+)", url)
        if not m:
            return 0
        owner, repo = m.group(1), m.group(2).replace(".git", "")
        trees = None
        branch = "main"
        for br in ("main", "master"):
            body = http.text(
                f"https://api.github.com/repos/{owner}/{repo}/git/trees/{br}?recursive=1",
                headers=_gh_headers(),
            )
            if body:
                try:
                    trees = json.loads(body).get("tree", [])
                    branch = br
                    break
                except Exception:
                    pass
        if not trees:
            return 0
        raws = [
            f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{t['path']}"
            for t in trees
            if t["type"] == "blob"
            and t.get("size", 0) < 200_000
            and t["path"].lower().endswith(
                (".md", ".txt", ".json", ".py", ".csv", ".yml", ".yaml", ".rst")
            )
        ]

    for ru in raws:
        txt = http.text(ru)
        if not txt:
            continue
        for platform, ids in extract_from_text(txt).items():
            for uid in ids:
                n += store.add(uid, platform, "gh:" + ru)
    return n
