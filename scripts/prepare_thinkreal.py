#!/usr/bin/env python3
"""
ThinkReal corpus v1: stdlib doc-examples with EXECUTED outputs.

Each stdlib example/*.f90 is compiled against a full static stdlib
build, run TWICE (determinism gate), and kept only on success:
  * verified rows: stdout matches the example's own `! prints ...`
    comments (predicted-vs-observed agreement, the falsifiable shape),
  * observed rows: deterministic nonempty stdout, no documented
    expectation (weaker, still real).

Row shape: Question -> Thinking(expected) -> Response(code) ->
Result(observed) -> verdict. Canonical cross-phase encoder.
Output: thinkreal_reasoning.txt (variable rows; pack with pack_rows.py).

Blacklist: interactive/slow/nondeterministic-by-construction examples
(sleep, datetime, random, cli args, error-stop).
"""

import concurrent.futures as cf
import hashlib
import os
import re
import subprocess
import sys

BOS = 8188
STD = os.path.expanduser("~/autoresearch/src/build/dependencies/stdlib")
EXDIR = os.path.join(STD, "example")
CACHE = os.path.expanduser("~/.cache/autoresearch")
OUT = os.path.join(CACHE, "thinkreal_reasoning.txt")
WORK = "/tmp/thinkreal"
MODDIR = "/tmp/slib/mod"
LIBDIR = "/tmp/slib"
OBLIB = ("/nix/store/qqgfxcvq0wqp5a842pv8bcyxrc8n4sd3-openblas-0.3.33"
         "/lib")
SKIP = re.compile(r"sleep|datetime|random|cli|arg_select|error_stop|"
                  r"stop_|f08estop|logger|ascii_escape|system_", re.I)

# Canonical encoder (same contract as all prepare scripts).
class FallbackEncoder:
    def encode(self, text):
        ids = []
        for ch in text:
            if ord(ch) < 256:
                ids.append(ord(ch))
            else:
                for b in ch.encode("utf-8"):
                    ids.append(256 + b)
        return ids

enc = FallbackEncoder()


def build_one(src):
    """Compile example; return binary path or None (never raises)."""
    base = os.path.splitext(os.path.basename(src))[0]
    exe = os.path.join(WORK, "bin", base)
    if os.path.exists(exe):
        return exe
    try:
        r = subprocess.run(
            ["gfortran", "-O1", f"-I{MODDIR}", src, f"-L{LIBDIR}",
             "-lstdlib_full", f"-L{OBLIB}", "-lopenblas", "-o", exe],
            capture_output=True, text=True, timeout=120)
    except Exception:
        return None
    return exe if r.returncode == 0 else None


def run_twice(exe):
    """Run twice; return stdout iff identical, nonempty, fast."""
    outs = []
    for _ in range(2):
        try:
            r = subprocess.run([exe], capture_output=True, text=True,
                               timeout=10)
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0 or not r.stdout.strip():
            return None
        outs.append(r.stdout)
    return outs[0] if outs[0] == outs[1] else None


def expectations(src):
    """Pull `! prints|outputs|gives|results ...` claims from source."""
    claims = []
    for line in open(src):
        m = re.search(r"!\s*(prints?|outputs?|gives?|results?( in)?)\s*:?\s*(.+)",
                      line, re.I)
        if m:
            claims.append(m.group(3).strip())
    return claims


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def process(src):
    base = os.path.splitext(os.path.basename(src))[0]
    if SKIP.search(base):
        return ("skip", base)
    exe = build_one(src)
    if not exe:
        return ("nobuild", base)
    out = run_twice(exe)
    if out is None:
        return ("nondet", base)
    claims = expectations(src)
    uses = re.findall(r"use\s+stdlib_\w+", open(src).read())
    topic = (uses[0].replace("use ", "") + ": " if uses else "") + base
    code = open(src).read().strip()
    if len(code) > 6000:
        return ("toolong", base)
    if claims and all(norm(c) in norm(out) for c in claims):
        verdict = "matches documented output"
        status = "verified"
    elif claims:
        return ("mismatch", base)
    else:
        verdict = "observed (no documented expectation)"
        status = "observed"
    exp_txt = "\n".join(f"Expect: {c}" for c in claims) if claims else \
        "Expect: deterministic run (no documented output)."
    text = (
        f"### Instruction:\nRun the stdlib example {base} ({topic}).\n\n"
        f"### Response:\n<think>\nGoal: demonstrate {topic}.\n"
        f"{exp_txt}\n</think>\n```fortran\n{code}\n```\n<think>\n"
        f"Observed:\n{out.strip()}\nVerdict: {verdict}.\n</think>\n")
    return (status, (text, base))


def main():
    os.makedirs(os.path.join(WORK, "bin"), exist_ok=True)
    srcs = sorted(f for f in
                  (os.path.join(EXDIR, f) for f in os.listdir(EXDIR))
                  if f.endswith(".f90"))
    print(f"{len(srcs)} examples; blacklist drops "
          f"{sum(1 for s in srcs if SKIP.search(s))}")
    todo = [s for s in srcs if not SKIP.search(s)]
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        bins = dict(zip(todo, ex.map(build_one, todo)))
    ok_srcs = [s for s in todo if bins[s]]
    print(f"built {len(ok_srcs)}/{len(todo)}")
    rows, stats = [], {}
    for s in ok_srcs:
        try:
            st, payload = process(s)
        except Exception as e:
            st, payload = ("error", f"{s}: {e}")
        stats[st] = stats.get(st, 0) + 1
        if st in ("verified", "observed"):
            rows.append(payload)
    print("outcomes:", dict(sorted(stats.items())))
    seen, kept = set(), 0
    with open(OUT + ".tmp", "w") as f:
        for text, base in rows:
            h = hashlib.sha256(text.encode()).hexdigest()[:16]
            if h in seen:
                continue
            seen.add(h)
            ids = enc.encode(text)
            f.write(f"{BOS} " + " ".join(map(str, ids)) + "\n")
            kept += 1
    os.rename(OUT + ".tmp", OUT)
    print(f"kept {kept} rows -> {OUT} (pack with pack_rows.py)")


if __name__ == "__main__":
    main()
