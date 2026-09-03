#!/usr/bin/env python3
"""Scaffold: discover publicly-shared Claude snapshot UUIDs and hoard them.

Sources: GitHub code search, Reddit search JSON, and arbitrary text dumps
(pastebin exports, your own notes, etc). Only indexes links people posted
publicly. Fetches each snapshot JSON and stores it locally, deduped.

Set GITHUB_TOKEN (free) for GitHub code search. Reddit works unauthed
with a User-Agent but is rate-limited.
"""
import os, re, json, time, sqlite3, subprocess, requests
from pathlib import Path

UUID_RE = re.compile(r"claude\.ai/(?:api/chat_snapshots|share)/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)
SNAP_URL = "https://claude.ai/api/chat_snapshots/{}?rendering_mode=messages&render_all_tools=true"

STORE_DIR = Path(os.environ.get("HOARD_DIR", "hoard"))
DB = STORE_DIR / "index.db"
RAW = STORE_DIR / "raw"

def init_db():
    STORE_DIR.mkdir(exist_ok=True)
    RAW.mkdir(exist_ok=True)
    db = sqlite3.connect(DB)
    db.execute("""CREATE TABLE IF NOT EXISTS snaps (
        uuid TEXT PRIMARY KEY, source TEXT, found_at REAL,
        fetched INTEGER DEFAULT 0, msg_count INTEGER, error TEXT)""")
    return db

def add_uuids(db, uuids, source):
    n = 0
    for u in set(uuids):
        cur = db.execute("INSERT OR IGNORE INTO snaps(uuid,source,found_at) VALUES(?,?,?)",
                         (u, source, time.time()))
        n += cur.rowcount
    db.commit()
    return n

# ---- collectors -----------------------------------------------------------

def collect_github(db, query="claude.ai/api/chat_snapshots", token=None):
    token = token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set; skipping GitHub")
        return
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    url = "https://api.github.com/search/code"
    for page in range(1, 11):
        r = requests.get(url, headers=headers, params={"q": query, "per_page": 100, "page": page}, timeout=20)
        if r.status_code != 200:
            print("github", r.status_code, r.text[:200]); break
        items = r.json().get("items", [])
        if not items: break
        uuids = []
        for it in items:
            for m in it.get("text_matches", []):
                uuids += UUID_RE.findall(m.get("fragment", ""))
        print(f"  github page {page}: +{add_uuids(db, uuids, 'github')} new")
        if len(items) < 100: break
        time.sleep(2)

def collect_reddit(db, query="claude.ai/share"):
    headers = {"User-Agent": "Mozilla/5.0 research-hoard"}
    url = "https://www.reddit.com/search.json"
    after = None
    for _ in range(10):
        params = {"q": query, "limit": 100, "sort": "new", "after": after}
        r = requests.get(url, headers=headers, params=params, timeout=20)
        if r.status_code != 200:
            print("reddit", r.status_code); break
        data = r.json().get("data", {})
        uuids = []
        for c in data.get("children", []):
            body = json.dumps(c)
            uuids += UUID_RE.findall(body)
        print(f"  reddit: +{add_uuids(db, uuids, 'reddit')} new")
        after = data.get("after")
        if not after: break
        time.sleep(2)

def collect_brave(db, query, api_key=None, pages=5):
    api_key = api_key or os.environ.get("BRAVE_API_KEY")
    if not api_key:
        print("BRAVE_API_KEY not set; skipping Brave"); return
    hdr = {"Accept": "application/json", "X-Subscription-Token": api_key}
    for off in range(0, pages * 20, 20):
        r = requests.get("https://api.search.brave.com/res/v1/web/search",
                         headers=hdr, params={"q": query, "offset": off, "count": 20}, timeout=20)
        if r.status_code != 200:
            print("brave", r.status_code, r.text[:150]); break
        res = r.json().get("web", {}).get("results", [])
        if not res: break
        uuids = []
        for item in res:
            blob = " ".join([item.get("url", ""), item.get("description", ""), item.get("title", "")])
            uuids += UUID_RE.findall(blob)
            # deep-fetch pages we can actually read (github/pastebin/gist), skip reddit
            u = item.get("url", "")
            if any(d in u for d in ("github.com", "gist.github", "pastebin.com", "raw.githubusercontent")):
                try:
                    p = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
                    if p.status_code == 200:
                        uuids += UUID_RE.findall(p.text)
                except Exception: pass
        print(f"  brave off {off}: +{add_uuids(db, uuids, 'brave')} new")
        time.sleep(1.5)

DDG_QUERIES = [
    '"claude.ai/share/"',
    'site:github.com "claude.ai/share"',
    'site:gist.github.com "claude.ai/share"',
    'site:pastebin.com "claude.ai/share"',
    '"claude.ai/api/chat_snapshots/"',
]

def collect_ddg(db, queries=None, per=30, delay=2.0):
    try:
        from ddgs import DDGS
    except ImportError:
        print("ddgs not installed (pip install ddgs); skipping DDG"); return
    queries = queries or DDG_QUERIES
    H = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
    SKIP = ("reddit.com", "google.", "youtube.com", "twitter.com", "x.com", "facebook.com")
    for q in queries:
        try:
            results = DDGS().text(q, max_results=per)
        except Exception as e:
            print(f"  ddg err {q!r}: {e}"); continue
        uuids = []
        for r in results:
            u = r.get("href", "")
            if any(b in u for b in SKIP): continue
            try:
                p = requests.get(u, headers=H, timeout=15)
                if p.status_code == 200:
                    uuids += UUID_RE.findall(p.text)
            except Exception:
                pass
            time.sleep(0.3)
        print(f"  ddg {q!r}: +{add_uuids(db, uuids, 'ddg')} new")
        time.sleep(delay)

def collect_file(db, path):
    text = Path(path).read_text(errors="ignore")
    uuids = UUID_RE.findall(text)
    print(f"  file {path}: +{add_uuids(db, uuids, f'file:{path}')} new")

def _walk_reddit(node, out):
    if isinstance(node, dict):
        if node.get("kind") == "t1":  # comment
            body = node.get("data", {}).get("body", "")
            out += UUID_RE.findall(body)
            replies = node.get("data", {}).get("replies")
            if isinstance(replies, dict):
                for child in replies.get("data", {}).get("children", []):
                    out = _walk_reddit(child, out)
        elif "children" in node:
            for c in node["children"]:
                out = _walk_reddit(c, out)
    elif isinstance(node, list):
        for c in node:
            out = _walk_reddit(c, out)
    return out

def collect_reddit_thread(db, url):
    headers = {"User-Agent": "Mozilla/5.0 research-hoard"}
    if not url.endswith(".json"):
        url = url.rstrip("/") + ".json"
    uuids = []
    after = None
    for _ in range(20):
        params = {"limit": 500, "after": after}
        r = requests.get(url, headers=headers, params=params, timeout=20)
        if r.status_code != 200:
            print("  thread", r.status_code); break
        uuids = _walk_reddit(r.json(), uuids)
        # find next page marker
        flat = r.json()[1]["data"] if isinstance(r.json(), list) else r.json().get("data", {})
        after = flat.get("after") if isinstance(flat, dict) else None
        if not after: break
        time.sleep(1)
    print(f"  thread {url}: +{add_uuids(db, uuids, 'reddit-thread')} new")

# ---- fetch ----------------------------------------------------------------

CURL_HDR = [
    "-H", "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "-H", "Accept: application/json, text/plain, */*",
    "-H", "Referer: https://claude.ai/",
]

def fetch_all(db, limit=None, delay=1.0):
    rows = db.execute("SELECT uuid,source FROM snaps WHERE fetched=0").fetchall()
    if limit: rows = rows[:limit]
    for uuid, source in rows:
        out = RAW / f"{uuid}.json"
        try:
            r = subprocess.run(
                ["curl", "-s", "-L"] + CURL_HDR + ["-o", str(out), SNAP_URL.format(uuid)],
                capture_output=True, timeout=40)
            if out.exists() and out.stat().st_size > 0:
                try:
                    d = json.loads(out.read_text())
                    db.execute("UPDATE snaps SET fetched=1,msg_count=? WHERE uuid=?",
                               (len(d.get("chat_messages", [])), uuid))
                except json.JSONDecodeError:
                    out.unlink()
                    db.execute("UPDATE snaps SET error='non-json' WHERE uuid=?", (uuid,))
            else:
                db.execute("UPDATE snaps SET error=? WHERE uuid=?", (f"exit {r.returncode}", uuid))
        except Exception as e:
            db.execute("UPDATE snaps SET error=? WHERE uuid=?", (str(e)[:200], uuid))
        db.commit()
        time.sleep(delay)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--github", action="store_true")
    ap.add_argument("--reddit", action="store_true")
    ap.add_argument("--file")
    ap.add_argument("--thread")
    ap.add_argument("--brave", default="claude.ai/share")
    ap.add_argument("--ddg", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    db = init_db()
    if args.github: collect_github(db)
    if args.reddit: collect_reddit(db)
    if args.thread: collect_reddit_thread(db, args.thread)
    if os.environ.get("BRAVE_API_KEY"): collect_brave(db, args.brave)
    if args.ddg: collect_ddg(db)
    if args.file: collect_file(db, args.file)
    if args.fetch: fetch_all(db, limit=args.limit)
    total = db.execute("SELECT COUNT(*) FROM snaps").fetchone()[0]
    got = db.execute("SELECT COUNT(*) FROM snaps WHERE fetched=1").fetchone()[0]
    print(f"tracked={total} fetched={got}")
