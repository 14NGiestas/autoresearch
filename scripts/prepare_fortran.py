#!/usr/bin/env python3
"""
Phase 4: Fortran from fortran-lang/webpage (learn/ tutorials).

Usage:  python scripts/prepare_fortran.py
"""

import os, sys, re, time, json, urllib.request, hashlib

BOS = 8188

class FallbackEncoder:
    def encode(self, text):
        ids = []
        for ch in text:
            for b in ch.encode("utf-8"):
                ids.append(256 + b if b < 128 else b)
        return ids

try:
    import rustbpe; enc = rustbpe.Encoder()  # noqa: F401
except ImportError:
    try:
        import tiktoken; enc = tiktoken.get_encoding("cl100k_base")  # noqa: F401
    except ImportError:
        enc = FallbackEncoder()

CACHE = os.path.expanduser("~/.cache/autoresearch")
OUT = os.path.join(CACHE, "fortran_tutorial.txt")
TMP = OUT + ".tmp"

RE_FCODE = re.compile(r"```(?:\w*\n)?(.*?)```", re.DOTALL)
FORTRAN_KW = re.compile(
    r"\b(program|module|subroutine|function|integer|real|character|logical|"
    r"if|do|end|use|implicit|call|print|write|allocate|deallocate|"
    r"type|class|procedure|interface|select|case|continue|return|"
    r"contains|intent|parameter|public|private|implicit_none)\b",
    re.IGNORECASE
)

def is_real_fortran(code):
    if len(code) < 20: return False
    if not FORTRAN_KW.search(code): return False
    return True

def fetch(url, timeout=15):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:
        return ""

def fetch_md_files(path):
    """List all .md files under path via one recursive API call."""
    url = f"https://api.github.com/repos/fortran-lang/webpage/contents/{path}"
    files = []
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            for entry in json.loads(r.read()):
                if entry["type"] == "file" and entry["name"].endswith(".md"):
                    files.append(entry["path"])
                elif entry["type"] == "dir":
                    files.extend(fetch_md_files(entry["path"]))
    except Exception as e:
        print(f"  API error {path}: {e}")
    return files

def extract_examples(content):
    examples = []
    for m in RE_FCODE.finditer(content):
        code = m.group(1).strip()
        if not is_real_fortran(code):
            continue
        # Clean up
        code = re.sub(r"^!.*\n", "", code, flags=re.MULTILINE)  # drop full-comment lines
        code = code.strip()
        if not code:
            continue
        text = f"### Instruction:\nWrite a Fortran example.\n\n### Response:\n{code}\n"
        examples.append(text)
    return examples

def write_stream(examples):
    seen = set()
    n = 0
    t0 = time.time()
    try: os.remove(TMP)
    except OSError: pass
    try:
        f = open(TMP, "w")
    except OSError as e:
        print(f"Cannot open {TMP}: {e}"); return 0
    with f:
        for text in examples:
            h = hashlib.sha256(text.encode()).hexdigest()[:16]
            if h in seen: continue
            seen.add(h)
            try:
                ids = enc.encode(text)
                ids = [BOS] + list(ids)
                f.write(" ".join(str(i) for i in ids) + "\n")
                n += 1
            except Exception: pass
            if n % 200 == 0:
                print(f"  {n}  {n/(time.time()-t0+1e-9):.0f}/s", flush=True)
    try: os.rename(TMP, OUT)
    except OSError as e: print(f"rename: {e}"); return 0
    size = os.path.getsize(OUT) / 1e6
    print(f"Wrote {n} → {OUT} ({size:.1f} MB) in {time.time()-t0:.1f}s")
    return n

def main():
    try: os.makedirs(CACHE, exist_ok=True)
    except OSError as e: print(f"mkdir: {e}"); sys.exit(1)

    # 1. List all learn/ .md files
    print("Scanning fortran-lang/webpage source/learn/ ...")
    files = fetch_md_files("source/learn")
    print(f"  {len(files)} markdown files")

    # 2. Fetch + extract one by one (streaming)
    all_examples = []
    n_fetched = 0
    for path in files:
        url = f"https://raw.githubusercontent.com/fortran-lang/webpage/main/{path}"
        content = fetch(url)
        n_fetched += 1
        if n_fetched % 10 == 0:
            print(f"  fetched {n_fetched}/{len(files)}", flush=True)
        if content:
            ex = extract_examples(content)
            all_examples.extend(ex)

    # 3. Rosetta stone
    print("  Fetching rosetta_stone.md...")
    ros = fetch("https://raw.githubusercontent.com/fortran-lang/webpage/main/source/learn/rosetta_stone.md")
    if ros:
        all_examples.extend(extract_examples(ros))
    print(f"  Total examples: {len(all_examples)}")

    if not all_examples:
        print("FAIL: no examples"); sys.exit(1)

    n = write_stream(all_examples)
    if n == 0:
        print("FAIL"); sys.exit(1)
    print(f"\n✓ Phase 4 Fortran ready: {n} rows")

if __name__ == "__main__":
    main()
