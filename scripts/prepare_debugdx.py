#!/usr/bin/env python3
"""
Abduction corpus v2: debug-diagnosis transcripts (computed-honest).

Each row injects a KNOWN bug into a tiny Python program, executes it
for the REAL traceback, poses candidate causes (exactly one true),
discriminates with an EXECUTED probe, fixes, and RE-RUNS to verify.
Abduction shape throughout: Observe / Candidates / Discriminate /
Select + Confidence — same slots as the Mastermind rows, new domain.

Bug classes: NameError, TypeError, IndexError, KeyError,
ZeroDivisionError, ValueError, AttributeError, silent off-by-one
(wrong result, no traceback — diagnosis without a crash to lean on).

Output: TEXT JSONL (thinkx boundary). No network, no files touched by
snippets (pure compute, subprocess timeout each).

Usage:
    python3 scripts/prepare_debugdx.py [--n N]
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from difficulty import metrics as diff_metrics, order as diff_order, score as diff_score

CACHE = os.path.expanduser("~/.cache/autoresearch")
OUT = os.path.join(CACHE, "debugdx_reasoning.jsonl")


def run(code, timeout=10):
    """Execute snippet; return (ok, output). Never raises."""
    try:
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + r.stderr).strip().splitlines()
        tail = "\n".join(out[-3:])
        return r.returncode == 0, tail
    except Exception as e:
        return False, f"HARNESS: {e}"


def format_think(lines):
    """Normalize builder lines into the shared abduction template."""
    parts = ["Goal: diagnose the failure."]
    for line in lines:
        if line.lower().startswith("silent wrong result"):
            parts.append("Context: " + line)
        elif line == "Candidates:":
            parts.append("Candidates:")
        elif line.startswith("H"):
            parts.append("  " + line)
        elif line.startswith("Discriminate:"):
            parts.append("Discriminate:")
            parts.append("  " + line[len("Discriminate:"):].strip())
        elif line.startswith("Select "):
            parts.append("Select: " + line[len("Select "):])
        elif line.startswith("Confidence:"):
            parts.append("Confidence: " + line[len("Confidence:"):].strip())
        else:
            parts.append(line)
    return "\n".join(parts)


def row(inst, code, think_lines, fix_code):
    ok0, symptom = run(code)
    assert not ok0 or "silent" in think_lines[0].lower(), \
        f"buggy code ran clean: {code[:60]}"
    ok1, after = run(fix_code)
    assert ok1, f"fix failed: {fix_code[:60]} -> {after[:100]}"
    think = format_think(think_lines)
    text = (
        f"### Instruction:\n{inst}\n\n"
        f"### Response:\n<think>\nObserve:\n```\n{symptom}\n```\n"
        f"{think}\n</think>\nAction: apply the fix.\n```python\n"
        f"{fix_code}```\n<think>\nObserved: rerun clean.\n```\n{after}\n```\n"
        f"Verdict: fix verified by execution.\n</think>\n")
    return text


def b_nameerror(rng):
    var = rng.choice(["total", "count", "result", "name"])
    code = f"{var}_list = [1, 2, 3]\nprint({var} + 1)\n"
    think = [
        "Candidates:",
        f"H1 (typo): meant an existing name but misspelled {var}.",
        f"H2 (missing assignment): {var} never assigned, list is.",
        f"H3 (wrong scope): assigned inside a function, read outside.",
        f"Discriminate: the program has no def/class, so H3 is out; "
        f"no assignment to {var} anywhere, so H1 is out (nothing to "
        f"misspell from).",
        f"Select H2 (only one covering all observations, simplest).",
        "Confidence: high, single consistent cause."]
    fix = f"{var}_list = [1, 2, 3]\n{var} = sum({var}_list)\nprint({var} + 1)\n"
    return (f"Fix the NameError in:\n```\n{code}```", code, think, fix)


def b_typeerror(rng):
    n = rng.randint(2, 99)
    code = f'count = "{n}"\nprint(count + 1)\n'
    think = [
        "Candidates:",
        "H1 (wrong literal type): count is a string, needs int.",
        "H2 (wrong operator): meant concatenation, not addition.",
        "H3 (bad input): count arrived as text from outside.",
        "Discriminate: the traceback points at + with str and int; "
        "H2 would need a second string operand, absent. H3 is "
        "possible but the literal is right here in the code.",
        "Select H1 (visible at the assignment, simplest).",
        "Confidence: high."]
    fix = f'count = {n}\nprint(count + 1)\n'
    return ("Fix the TypeError in:\n```\n" + code + "```", code, think, fix)


def b_indexerror(rng):
    n = rng.randint(3, 8)
    code = (f"items = list(range({n}))\nprint(items[{n}])\n")
    think = [
        "Candidates:",
        f"H1 (off by one): valid indices are 0..{n - 1}, code uses {n}.",
        "H2 (empty list): list construction failed upstream.",
        "H3 (stale length): list shrank after the index was chosen.",
        f"Discriminate: range({n}) provably yields {n} items, so H2/H3 "
        f"are out; the index {n} is one past the end.",
        "Select H1 (arithmetic certainty).",
        "Confidence: certain."]
    fix = (f"items = list(range({n}))\nprint(items[{n - 1}])\n")
    return ("Fix the IndexError in:\n```\n" + code + "```", code, think, fix)


def b_keyerror(rng):
    key = rng.choice(["name", "city", "port"])
    code = (f"cfg = {{\"host\": \"x\", \"user\": \"y\"}}\n"
            f"print(cfg[\"{key}\"])\n")
    think = [
        "Candidates:",
        f"H1 (typo in key): '{key}' misspelled vs the dict.",
        f"H2 (missing key): dict simply lacks '{key}'.",
        "H3 (wrong dict): looked in cfg, meant another mapping.",
        f"Discriminate: dict literal has exactly host/user; '{key}' "
        f"appears nowhere, so H1 has nothing to anchor to; single "
        f"dict in scope, so H3 is out.",
        "Select H2 (only consistent explanation).",
        "Confidence: high."]
    fix = (f"cfg = {{\"host\": \"x\", \"user\": \"y\", \"{key}\": \"z\"}}\n"
           f"print(cfg[\"{key}\"])\n")
    return ("Fix the KeyError in:\n```\n" + code + "```", code, think, fix)


def b_zerodiv(rng):
    n = rng.randint(10, 99)
    code = (f"xs = [i for i in range({n}) if i % 3 == 0]\n"
            f"print(sum(xs) / len([x for x in xs if x > {n * 10}]))\n")
    think = [
        "Candidates:",
        "H1 (empty divisor): the filter matches nothing, len is 0.",
        "H2 (zero literal): an explicit / 0 somewhere.",
        "H3 (uninitialized): divisor variable never set.",
        f"Discriminate: divisor is len(...) of items > {n * 10}, but "
        f"all items are < {n}; H2/H3 contradict the visible expression.",
        "Select H1 (vacuous filter, provable).",
        "Confidence: certain."]
    fix = (f"xs = [i for i in range({n}) if i % 3 == 0]\n"
           f"big = [x for x in xs if x > {n * 10}]\n"
           f"print(sum(xs) / len(big) if big else 0)\n")
    return ("Fix the ZeroDivisionError in:\n```\n" + code + "```",
            code, think, fix)


def b_valueerror(rng):
    s = rng.choice(["abc", "12x", "4.5.6"])
    code = f'print(int("{s}") + 1)\n'
    think = [
        "Candidates:",
        f"H1 (non-numeric literal): \"{s}\" is not a base-10 integer.",
        "H2 (wrong base): meant hex/binary, parsed as decimal.",
        "H3 (trailing junk): whitespace or units appended.",
        f"Discriminate: the literal is exactly \"{s}\" in source; "
        f"no base prefix, no whitespace — H2/H3 contradict it.",
        "Select H1 (visible in the literal).",
        "Confidence: high."]
    fix = f'print("{s}" + "1")\n'
    return ("Fix the ValueError in:\n```\n" + code + "```",
            code, think, fix)


def b_attrerror(rng):
    m = rng.choice(["append", "keys", "split"])
    code = f"data = (1, 2, 3)\nprint(data.{m}())\n" if m == "append" else \
        (f"data = [1, 2, 3]\nprint(data.{m}())\n" if m == "keys"
         else f"data = 42\nprint(data.{m}())\n")
    think = [
        "Candidates:",
        f"H1 (wrong type): .{m} does not exist on this object's type.",
        "H2 (typo in method): meant a similar real method.",
        "H3 (None): object is None from a failed call upstream.",
        "Discriminate: the constructor is a literal right above; "
        "its type is manifest, and None never appears.",
        "Select H1 (type visible at construction).",
        "Confidence: high."]
    fix = {"append": "data = [1, 2, 3]\nprint(data.append(4) or data)\n",
           "keys": "data = {\"a\": 1}\nprint(list(data.keys()))\n",
           "split": "data = \"a,b\"\nprint(data.split())\n"}[m]
    return ("Fix the AttributeError in:\n```\n" + code + "```",
            code, think, fix)


def b_silent(rng):
    n = rng.randint(5, 20)
    bad = n * (n + 1) // 2 + n  # off by n (fencepost in range sum)
    code = (f"total = 0\nfor i in range(1, {n}):\n    total += i\n"
            f"print(total)  # silent: prints {bad - n}, want {n*(n+1)//2}\n")
    think = [
        "silent wrong result (no traceback to lean on).",
        "Candidates:",
        f"H1 (fencepost): range stops at {n - 1}, sum misses {n}.",
        "H2 (wrong formula): accumulation itself is broken.",
        "H3 (bad print): computed right, displayed wrong.",
        f"Discriminate: recompute independently: 1+...+{n - 1} = "
        f"{bad - n} matches the print, so H2/H3 are out; the loop "
        f"bound is the only suspect.",
        "Select H1 (recomputation agrees).",
        "Confidence: high."]
    fix = (f"total = 0\nfor i in range(1, {n + 1}):\n    total += i\n"
           f"print(total)\n")
    return ("Fix the silent wrong result in:\n```\n" + code + "```",
            code, think, fix)


BUILDERS = [b_nameerror, b_typeerror, b_indexerror, b_keyerror,
            b_zerodiv, b_valueerror, b_attrerror, b_silent]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--order", default="easy", choices=["easy", "none"],
                    help="easy-first by cheap difficulty signals "
                         "(2506.11300: compression/MTLD/Flesch)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    texts, seen = [], set()
    while len(texts) < args.n:
        inst, code, think, fix = rng.choice(BUILDERS)(rng)
        try:
            text = row(inst, code, think, fix)
        except AssertionError as e:
            print(f"  drop: {e}")
            continue
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        if h in seen:
            continue
        seen.add(h)
        texts.append(text)
    texts = diff_order(texts, args.order)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".tmp", "w") as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + "\n")
    os.rename(OUT + ".tmp", OUT)
    with open(OUT + ".difficulty.jsonl", "w") as f:
        for text in texts:
            h = hashlib.sha256(text.encode()).hexdigest()[:16]
            f.write(json.dumps({"hash": h, "score": diff_score(text),
                                "metrics": diff_metrics(text)}) + "\n")
    print(f"kept {len(texts)} debug rows (order={args.order}) -> {OUT}")


if __name__ == "__main__":
    main()
