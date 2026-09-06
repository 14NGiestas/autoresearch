#!/usr/bin/env python3
"""
Phase 2 curriculum: tool-use trajectories (curl, fetch, python -c, grep, read).
Mixed with the Python code corpus for diversity.

Each row: BOS + tool-interaction transcript, e.g.
    ### Instruction:
    Fetch the homepage of example.com using curl.
    ### Response:
    $ curl -fsSL https://example.com
    <response>...</response>

Output: ~/.cache/autoresearch/tool_trajectories.txt
Each row: space-separated token IDs (BOS=8188 + transcript tokens).

Usage:
    python scripts/prepare_tool.py
"""

import os, sys, json, time, urllib.request, hashlib, random

BOS = 8188

class FallbackEncoder:
    def encode(self, text):
        ids = []
        for ch in text:
            b = ch.encode("utf-8")
            for byte in b:
                ids.append(256 + byte if byte < 128 else byte)
        return ids

try:
    import rustbpe; enc = rustbpe.Encoder()
except ImportError:
    try:
        import tiktoken; enc = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        print("No tokenizer, using fallback")
        enc = FallbackEncoder()

CACHE = os.path.expanduser("~/.cache/autoresearch")
OUT = os.path.join(CACHE, "tool_trajectories.txt")

# ---------------------------------------------------------------------------
# Synthetic tool trajectories (no auth needed)
# ---------------------------------------------------------------------------

TOOL_TEMPLATES = [
    {
        "instruction": "Use curl to fetch the headers from https://httpbin.org/get",
        "command": "curl -sI https://httpbin.org/get",
        "response": "HTTP/2 200\nserver: nginx\ndate: Thu, 01 Jan 2025 00:00:00 GMT",
    },
    {
        "instruction": "Download a JSON file from the public API https://api.github.com/repos/python/cpython",
        "command": "curl -fsSL https://api.github.com/repos/python/cpython",
        "response": '{"name": "cpython", "full_name": "python/cpython", "description": "The Python programming language"}',
    },
    {
        "instruction": "Check if a URL is up using curl with a timeout of 5 seconds",
        "command": "curl -fsSL --max-time 5 -o /dev/null -w '%{http_code}' https://httpbin.org/status/200",
        "response": "200",
    },
    {
        "instruction": "Fetch the first 100 bytes of example.com homepage",
        "command": "curl -fsSL --max-time 10 -r 0-99 https://example.com",
        "response": "<!doctype html><html><head>\n    <title>Example Domain</title>",
    },
    {
        "instruction": "Use grep to find all lines containing 'ERROR' in /var/log/syslog",
        "command": "grep ERROR /var/log/syslog | head -10",
        "response": "Jan  1 00:00:01 hostname kernel: ERROR: out of memory\nJan  1 00:01:23 hostname sshd[123]: ERROR: Connection refused",
    },
    {
        "instruction": "Find all Python files modified in the last 7 days",
        "command": "find . -name '*.py' -mtime -7 -print",
        "response": "./scripts/train.py\n./src/model.py\n./tests/test_gpt.py",
    },
    {
        "instruction": "Count the number of lines in all .txt files recursively",
        "command": "find . -name '*.txt' -exec wc -l {} + | tail -1",
        "response": "  12345 total",
    },
    {
        "instruction": "Run a Python one-liner to compute the sum of 1 to 1000",
        "command": "python3 -c \"print(sum(range(1, 1001)))\"",
        "response": "500500",
    },
    {
        "instruction": "Use python to check if 1729 is prime",
        "command": "python3 -c \"n=1729; print(any(n % i == 0 for i in range(2, int(n**0.5)+1)))\"",
        "response": "True",
    },
    {
        "instruction": "List files in /tmp modified in the last hour",
        "command": "find /tmp -type f -mmin -60 -ls 2>/dev/null | head -5",
        "response": "  -rw------- 1 user user  4096 Jan  1 00:30 /tmp/session.log\n  -rw-r--r-- 1 user user  8192 Jan  1 00:45 /tmp/cache.tmp",
    },
    {
        "instruction": "Use curl to POST JSON data to httpbin.org",
        "command": "curl -fsSL -X POST https://httpbin.org/post -H 'Content-Type: application/json' -d '{\"key\": \"value\"}'",
        "response": '{"url": "https://httpbin.org/post", "json": {"key": "value"}}',
    },
    {
        "instruction": "Check the disk usage of the home directory",
        "command": "du -sh $HOME | cut -f1",
        "response": "42G",
    },
    {
        "instruction": "Get your public IP address using curl",
        "command": "curl -fsSL https://ifconfig.me",
        "response": "203.0.113.42",
    },
    {
        "instruction": "Use awk to print the second column of a CSV file",
        "command": "awk -F, '{print $2}' data.csv | head -5",
        "response": "Alice\nBob\nCharlie\nDavid\nEve",
    },
    {
        "instruction": "Check the current git branch",
        "command": "git branch --show-current",
        "response": "main",
    },
]

def expand_template(t):
    inst = t["instruction"]
    cmd = t["command"]
    resp = t["response"]
    return (
        f"### Instruction:\n{inst}\n\n"
        f"### Response:\n"
        f"$ {cmd}\n"
        f"{resp}\n"
    )

def make_trajectories(n=2000):
    rows = []
    used = set()
    attempts = 0
    while len(rows) < n and attempts < n * 5:
        attempts += 1
        t = random.choice(TOOL_TEMPLATES)
        # Vary: different URLs, domains, numbers
        inst = t["instruction"]
        cmd = t["command"]
        resp = t["response"]
        # Swap domain to make each unique
        domain = random.choice(["example.com", "httpbin.org", "api.github.com", "ifconfig.me", "httpbin.ip"])
        cmd = cmd.replace("https://httpbin.org", f"https://{domain}")
        cmd = cmd.replace("https://api.github.com", f"https://api.{domain}")
        cmd = cmd.replace("https://example.com", f"https://{domain}")
        cmd = cmd.replace("https://ifconfig.me", f"https://{domain}")
        # Vary numbers
        cmd = cmd.replace("1729", str(random.randint(100, 9999)))
        cmd = cmd.replace("1000", str(random.randint(500, 5000)))
        cmd = cmd.replace("42", str(random.randint(1, 99)))
        cmd = cmd.replace("100", str(random.randint(50, 200)))
        # Vary response
        resp = resp.replace("203.0.113.42", f"{random.randint(100,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}")
        resp = resp.replace("42G", f"{random.randint(5, 200)}G")
        resp = resp.replace("500500", str(random.randint(10000, 500000)))
        text = (
            f"### Instruction:\n{inst}\n\n"
            f"### Response:\n"
            f"$ {cmd}\n"
            f"{resp}\n"
        )
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        if h in used:
            continue
        used.add(h)
        rows.append(text)
    return rows

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write(rows, out, limit=0):
    seen = set()
    n_kept = 0
    t0 = time.time()
    tmp = out + ".tmp"
    try: os.remove(tmp)
    except OSError: pass
    try:
        f = open(tmp, "w")
    except OSError as e:
        print(f"Cannot open {tmp}: {e}"); return 0
    with f:
        for text in rows:
            h = hashlib.sha256(text.encode()).hexdigest()[:16]
            if h in seen: continue
            seen.add(h)
            try:
                ids = enc.encode(text)
                ids = [BOS] + list(ids)
                f.write(" ".join(str(i) for i in ids) + "\n")
                n_kept += 1
            except Exception as e:
                print(f"  err: {e}")
            if limit and n_kept >= limit: break
            if n_kept % 500 == 0:
                print(f"  kept {n_kept:,}  {n_kept/(time.time()-t0+1e-9):.0f}/s")
    try:
        os.rename(tmp, out)
    except OSError as e:
        print(f"Cannot rename: {e}")
        return 0
    size = os.path.getsize(out) / 1e6
    print(f"Wrote {n_kept:,} → {out} ({size:.1f} MB)")
    return n_kept

def main():
    try: os.makedirs(os.path.dirname(OUT), exist_ok=True)
    except OSError as e: print(f"Cannot mkdir: {e}"); sys.exit(1)

    rows = make_trajectories(2000)
    print(f"Generated {len(rows):,} tool trajectories")
    n = write(rows, OUT)
    if n == 0: sys.exit(1)
    print(f"\n✓ Phase 2 ready: {n} rows")

if __name__ == "__main__":
    main()
