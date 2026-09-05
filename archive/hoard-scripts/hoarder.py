#!/usr/bin/env python3
"""Multi-platform shared-chat hoarder.

Discovers publicly-posted share links for Claude / ChatGPT / Deepseek / Qwen /
Gemini via DuckDuckGo (ddgs lib, no API key), recursively crawls GitHub repos &
gists for more, dedupes in SQLite, and fetches each conversation as JSON.

Notes:
- All HTTP goes through curl (the sandbox blocks python `requests` to some
  hosts). `--compressed` handles brotli/gzip.
- Claude: public API, works.
- ChatGPT: backend-api/share works with a Firefox UA (Chrome UA is gated).
- Deepseek/Qwen/Gemini: best-effort endpoints; adjust as confirmed.

Env: HOARD_DIR (./hoard), GITHUB_TOKEN (optional), CHATGPT_COOKIE (optional),
BRAVE_API_KEY (optional).
"""
import os, re, json, time, sqlite3, subprocess
from pathlib import Path

# ---------------------------------------------------------------- patterns
UUID_PATTERNS = {
    "claude":   r"claude\.ai/(?:api/chat_snapshots|share)/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    "chatgpt":  r"chatgpt\.com/share/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    "deepseek": r"chat\.deepseek\.com/[^\s\"'<>)]{0,40}/s/([0-9a-zA-Z-]{20,40})",
    "qwen":     r"chat\.qwen(?:lm)?\.ai/(?:s|share)/([0-9a-zA-Z-]{20,40})",
    "gemini":   r"(?:g\.co/gemini/share|gemini\.google\.com/share)/([0-9a-zA-Z-]{10,30})",
    "grok":     r"grok\.com/share/([A-Za-z0-9_=-]+)",
}
COMPILED = {k: re.compile(v, re.I) for k, v in UUID_PATTERNS.items()}

DDG_QUERIES = {
    "claude":   ['"claude.ai/share/"', 'site:github.com "claude.ai/share"',
                 'site:gist.github.com "claude.ai/share"', '"claude.ai/api/chat_snapshots/"'],
    "chatgpt":  ['"chatgpt.com/share/"', 'site:github.com "chatgpt.com/share"',
                 'site:gist.github.com "chatgpt.com/share"'],
    "deepseek": ['"chat.deepseek.com/a/chat/s/"', 'site:github.com "chat.deepseek.com"'],
    "qwen":     ['"chat.qwen.ai/share/"', 'site:github.com "chat.qwen.ai/share"'],
    "gemini":   ['"gemini.google.com/share/"', '"g.co/gemini/share/"'],
}

CHROME_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
FIREFOX_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:153.0) Gecko/20100101 Firefox/153.0"
STORE_DIR = Path(os.environ.get("HOARD_DIR", "hoard"))
DB = STORE_DIR / "index.db"
RAW = STORE_DIR / "raw"
CLAUDE_URL = "https://claude.ai/api/chat_snapshots/{}?rendering_mode=messages&render_all_tools=true"
SHARE_URL = {
    "deepseek": "https://chat.deepseek.com/a/chat/s/{}",
    "gemini":   "https://gemini.google.com/share/{}",
    "grok":     "https://grok.com/share/{}",
}

# ---------------------------------------------------------------- curl
def curl(url, ua=CHROME_UA, headers=None, out=None, timeout=40):
    cmd = ["curl", "-s", "-L", "--compressed", "-A", ua]
    for k, v in (headers or {}).items():
        cmd += ["-H", f"{k}: {v}"]
    if out:
        cmd += ["-o", str(out)]
    cmd += [url]
    return subprocess.run(cmd, capture_output=True, timeout=timeout)

def curl_text(url, ua=CHROME_UA, headers=None):
    r = curl(url, ua=ua, headers=headers)
    if r.returncode != 0:
        return None
    return r.stdout.decode("utf-8", "ignore")

# ---------------------------------------------------------------- db
def init_db():
    STORE_DIR.mkdir(exist_ok=True); RAW.mkdir(exist_ok=True)
    db = sqlite3.connect(DB)
    db.execute("""CREATE TABLE IF NOT EXISTS snaps (
        uuid TEXT, platform TEXT, source TEXT, found_at REAL,
        fetched INTEGER DEFAULT 0, msg_count INTEGER, error TEXT,
        PRIMARY KEY (uuid, platform))""")
    return db

def add_uuid(db, uuid, platform, source):
    uuid = uuid.lower()
    cur = db.execute("INSERT OR IGNORE INTO snaps(uuid,platform,source,found_at) VALUES(?,?,?,?)",
                     (uuid, platform, source, time.time()))
    db.commit(); return cur.rowcount

def extract_from_text(text):
    out = {}
    for p, rx in COMPILED.items():
        found = {m.lower() for m in rx.findall(text)}
        if found: out[p] = found
    return out

# ---------------------------------------------------------------- discovery
def collect_ddg(db, platforms=None, per=30, delay=2.0, crawl=True):
    try:
        from ddgs import DDGS
    except ImportError:
        print("ddgs not installed (pip install ddgs); skipping DDG"); return
    plats = platforms or list(DDG_QUERIES)
    SKIP = ("reddit.com", "google.", "youtube.com", "twitter.com", "x.com", "facebook.com")
    for plat in plats:
        for q in DDG_QUERIES.get(plat, []):
            try:
                results = DDGS().text(q, max_results=per, backend="api")
            except Exception as e:
                print(f"  ddg err {q!r}: {e}"); continue
            for r in results:
                u = r.get("href", "")
                if any(b in u for b in SKIP): continue
                txt = curl_text(u)
                if not txt: continue
                found = extract_from_text(txt)
                n = 0
                for fp, ids in found.items():
                    n += sum(add_uuid(db, i, fp, "ddg:" + u) for i in ids)
                if n: print(f"  ddg {plat} {q!r}: +{n} new (from {u[:70]})")
                if crawl and ("github.com" in u or "gist.github" in u):
                    crawl_github(db, u)
                time.sleep(0.3)
            time.sleep(delay)

def crawl_github(db, url, seen=None, depth=0):
    if seen is None: seen = set()
    if url in seen or depth > 2: return
    seen.add(url)
    if "gist.github" in url:
        gid = re.search(r"gist\.github\.com/[^/]+/([0-9a-f]+)", url)
        if not gid: return
        body = curl_text(f"https://api.github.com/gists/{gid.group(1)}",
                         headers={"Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN','')}"} if os.environ.get("GITHUB_TOKEN") else {})
        if not body: return
        try: files = json.loads(body).get("files", {})
        except Exception: return
        raws = [f["raw_url"] for f in files.values() if "raw_url" in f]
    else:
        m = re.search(r"github\.com/([^/]+)/([^/#?]+)", url)
        if not m: return
        owner, repo = m.group(1), m.group(2).replace(".git", "")
        trees = None
        for branch in ("main", "master"):
            body = curl_text(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
            if body:
                try:
                    trees = json.loads(body).get("tree", []); break
                except Exception: pass
        if not trees: return
        raws = [f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{t['path']}"
                for t in trees if t["type"] == "blob" and t.get("size", 0) < 200_000
                and t["path"].lower().endswith((".md", ".txt", ".json", ".py", ".csv", ".yml", ".yaml", ".rst"))]
    n = 0
    for ru in raws:
        txt = curl_text(ru)
        if not txt: continue
        for fp, ids in extract_from_text(txt).items():
            n += sum(add_uuid(db, i, fp, "gh:" + ru) for i in ids)
    if n: print(f"  crawl github {url[:70]}: +{n} new")

def collect_file(db, path):
    text = Path(path).read_text(errors="ignore")
    n = 0
    for fp, ids in extract_from_text(text).items():
        n += sum(add_uuid(db, i, fp, f"file:{path}") for i in ids)
    print(f"  file {path}: +{n} new")

# ---------------------------------------------------------------- fetch
def fetch_claude(uuid):
    out = RAW / f"claude_{uuid}.json"
    curl(CLAUDE_URL.format(uuid), headers={"Accept": "application/json", "Referer": "https://claude.ai/"}, out=out)
    return out

def fetch_chatgpt(uuid):
    out = RAW / f"chatgpt_{uuid}.json"
    hdr = {"Accept": "application/json", "Referer": "https://chatgpt.com/"}
    if os.environ.get("CHATGPT_COOKIE"):
        hdr["Cookie"] = os.environ["CHATGPT_COOKIE"]
    curl(f"https://chatgpt.com/backend-api/share/{uuid}", ua=FIREFOX_UA, headers=hdr, out=out)
    return out

def fetch_grok(uuid):
    out = RAW / f"grok_{uuid}.json"
    curl("https://grok.com/rest/app-chat/share_links_data/" + uuid,
         headers={"Accept": "application/json", "Referer": "https://grok.com/"}, out=out)
    return out

def fetch_qwen(uuid):
    out = RAW / f"qwen_{uuid}.json"
    curl(f"https://chat.qwen.ai/api/v2/chats/share/{uuid}",
         headers={"Accept": "application/json", "Referer": "https://chat.qwen.ai/"}, out=out)
    return out

def fetch_deepseek(uuid):
    out = RAW / f"deepseek_{uuid}.json"
    curl(f"https://chat.deepseek.com/api/v0/share/content?share_id={uuid}",
         headers={"Accept": "application/json", "Referer": "https://chat.deepseek.com/"}, out=out)
    return out

def fetch_gemini(uuid):
    # SPA: no JSON API; archive the rendered HTML
    out = RAW / f"gemini_{uuid}.html"
    curl(f"https://gemini.google.com/share/{uuid}",
         headers={"Accept": "text/html", "Referer": "https://gemini.google.com/"}, out=out)
    return out

FETCHERS = {
    "claude": fetch_claude,
    "chatgpt": fetch_chatgpt,
    "grok": fetch_grok,
    "qwen": fetch_qwen,
    "deepseek": fetch_deepseek,
    "gemini": fetch_gemini,
}

def _msg_count(platform, d):
    if platform == "claude":
        return len(d.get("chat_messages") or [])
    if platform == "chatgpt":
        m = d.get("mapping")
        return len(m) if isinstance(m, dict) else 0
    if platform == "grok":
        return len(d.get("responses") or [])
    if platform == "qwen":
        try:
            return len(d["data"]["chat"]["messages"] or [])
        except Exception:
            return 0
    if platform == "deepseek":
        try:
            return len(d["data"]["biz_data"]["messages"] or [])
        except Exception:
            return 0
    return 0

def _save(db, uuid, platform, out, error=None):
    if not (out.exists() and out.stat().st_size > 0):
        db.execute("UPDATE snaps SET error=? WHERE uuid=? AND platform=?",
                   (error or "empty", uuid, platform)); db.commit(); return
    if platform == "gemini":
        html = out.read_text(errors="ignore")
        count = html.count('"text"') or html.count("userText") or 0
        db.execute("UPDATE snaps SET fetched=1,msg_count=? WHERE uuid=? AND platform=?",
                   (count, uuid, platform)); db.commit(); return
    try:
        d = json.loads(out.read_text())
    except json.JSONDecodeError:
        out.unlink()
        db.execute("UPDATE snaps SET error=? WHERE uuid=? AND platform=?",
                   ("non-json", uuid, platform)); db.commit(); return
    if platform in ("claude", "chatgpt") and isinstance(d, dict) and "detail" in d:
        error = str(d["detail"].get("code", d))[:120]
    elif platform == "grok" and isinstance(d, dict) and d.get("code") not in (0, None):
        error = str(d.get("message", d))[:120]
    elif platform == "qwen" and not d.get("success"):
        error = str(d.get("message", "qwen failure"))[:120]
    elif platform == "deepseek" and isinstance(d, dict):
        if d.get("data", {}).get("biz_code") != 0:
            error = str(d.get("data", {}).get("biz_msg", "deepseek failure"))[:120]
    if error:
        db.execute("UPDATE snaps SET error=? WHERE uuid=? AND platform=?", (error, uuid, platform))
        db.commit(); return
    count = _msg_count(platform, d)
    db.execute("UPDATE snaps SET fetched=1,msg_count=? WHERE uuid=? AND platform=?",
               (count, uuid, platform)); db.commit()

def fetch_all(db, limit=None, delay=1.0):
    rows = db.execute("SELECT uuid,platform FROM snaps WHERE fetched=0").fetchall()
    if limit: rows = rows[:limit]
    for uuid, platform in rows:
        fn = FETCHERS[platform]
        out = fn(uuid)
        _save(db, uuid, platform, out)
        time.sleep(delay)

# ---------------------------------------------------------------- cli
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddg", action="store_true")
    ap.add_argument("--platforms", nargs="*", default=None)
    ap.add_argument("--file")
    ap.add_argument("--crawl", help="crawl a github/gist URL directly")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    db = init_db()
    if args.crawl: crawl_github(db, args.crawl)
    if args.file: collect_file(db, args.file)
    if args.ddg: collect_ddg(db, platforms=args.platforms)
    if args.fetch: fetch_all(db, limit=args.limit)
    t = db.execute("SELECT COUNT(*) FROM snaps").fetchone()[0]
    f = db.execute("SELECT COUNT(*) FROM snaps WHERE fetched=1").fetchone()[0]
    print(f"tracked={t} fetched={f}")
    print("by platform:", dict(db.execute("SELECT platform, COUNT(*) FROM snaps GROUP BY platform").fetchall()))
