#!/usr/bin/env python3
"""
ThinkSciPy corpus: numpy/scipy docstring Examples, executed as doctests.

numpydoc Examples sections are predicted-vs-observed by construction
(>>> code + expected output, CI-tested upstream). This script walks both
packages, runs every Examples block as a doctest (ELLIPSIS + whitespace
normalization, like numpy's own refguide check), and keeps passing ones:
  Question (what the routine does) -> Thinking (expected outputs) ->
  Response (doctest code) -> Result (executed pass).

Skip: plotting/GUI/input/network/sleep/threading/multiprocessing, slow
blocks (10s SIGALRM gate each), private modules. Random-unseeded blocks
fail matching and drop automatically. Canonical cross-phase encoder.
Output: thinkscipy_reasoning.txt (variable rows; pack with pack_rows.py).

Usage:
    uv run --with numpy --with scipy scripts/prepare_thinkscipy.py
"""

import doctest
import hashlib
import inspect
import io
import os
import pkgutil
import re
import signal
import sys

BOS = 8188
CACHE = os.path.expanduser("~/.cache/autoresearch")
OUT = os.path.join(CACHE, "thinkscipy_reasoning.txt")

SKIP_MOD = re.compile(r"\._|test|conftest|_build|f2py|distutils|typing|version")
SKIP_SRC = re.compile(r"matplotlib|pyplot|plt\.|imshow|show\(\)|input\(|"
                      r"open\(|urlopen|sleep\(|thread|multiprocess|"
                      r"input\(|os\.|sys\.|subprocess|socket|savefig",
                      re.I)
FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


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


def examples_of(doc, globs):
    """Parse docstring Examples section (globs bound at parse time)."""
    if not doc or ">>>" not in doc:
        return None
    m = re.search(r"^[ \t]*Examples\n[ \t]*-+\n(.*?)(?=^[ \t]*\S[^\n]*\n[ \t]*-+\n|\Z)",
                  doc, re.M | re.S)
    block = m.group(1) if m else doc
    if ">>>" not in block or SKIP_SRC.search(block):
        return None
    try:
        tests = doctest.DocTestParser().get_doctest(
            block, globs, "ex", "ex", 0)
    except Exception:
        return None
    if not tests.examples:
        return None
    return tests.examples


def try_each(examples, globs, secs=12):
    """Run each example separately (shared globs, may-vary skipped).
    Return the passing examples."""
    try:
        import numpy as _np
        _np.set_printoptions(legacy="1.25")
    except Exception:
        pass
    runner = doctest.DocTestRunner(optionflags=FLAGS, verbose=False)
    passed = []
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(secs)
    try:
        for ex in examples:
            if "may vary" in ex.source:
                continue
            t = doctest.DocTest([ex], globs, "ex", "ex", 0, "")
            f0, t0 = runner.failures, runner.tries
            runner.run(t)
            if runner.failures == f0 and runner.tries > t0:
                passed.append(ex)
    except Timeout:
        pass
    except Exception:
        pass
    finally:
        signal.alarm(0)
    return passed


def summary(doc):
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        line = line.strip()
        if line:
            return line[:160]
    return ""


def harvest(pkgname):
    """Yield (qualname, doc) for public functions/classes/methods."""
    try:
        pkg = __import__(pkgname, fromlist=["*"])
    except Exception:
        return
    mods = [pkgname]
    try:
        mods += [m.name for m in pkgutil.walk_packages(
            pkg.__path__, pkg.__name__ + ".")]
    except Exception:
        pass
    for modname in mods:
        if SKIP_MOD.search(modname):
            continue
        try:
            mod = __import__(modname, fromlist=["*"])
        except Exception:
            continue
        globs = dict(vars(mod))
        try:
            globs.setdefault("np", sys.modules.get("numpy"))
            globs.setdefault("numpy", sys.modules.get("numpy"))
        except Exception:
            pass
        for name in dir(mod):
            if name.startswith("_"):
                continue
            try:
                obj = getattr(mod, name)
            except Exception:
                continue
            if not (inspect.isfunction(obj) or inspect.isbuiltin(obj) or
                    inspect.isclass(obj)):
                continue
            doc = getattr(obj, "__doc__", "")
            yield (f"{modname}.{name}", doc, globs)
            if inspect.isclass(obj):
                for mn in dir(obj):
                    if mn.startswith("_"):
                        continue
                    try:
                        meth = getattr(obj, mn)
                        mdoc = getattr(meth, "__doc__", "")
                    except Exception:
                        continue
                    if mdoc and ">>>" in mdoc:
                        yield (f"{modname}.{name}.{mn}", mdoc, globs)


def code_of(examples):
    """Reconstruct >>> / ... prompts (parser strips them from source)."""
    lines = []
    for ex in examples:
        srclines = ex.source.splitlines()
        if not srclines:
            continue
        lines.append(">>> " + srclines[0])
        lines.extend("... " + l for l in srclines[1:])
    return "\n".join(lines)


def want_of(examples):
    """Expected outputs of passing examples for Thinking."""
    wants = []
    for ex in examples:
        w = ex.want.strip()
        if w:
            wants.append(w)
    return wants


def main():
    import numpy  # noqa: F401  (ensures availability under plain python too)
    rows, seen = [], set()
    stats = {"examined": 0, "kept": 0}
    for pkg in ("numpy", "scipy"):
        for qual, doc, globs in harvest(pkg):
            examples = examples_of(doc, globs)
            if not examples:
                continue
            stats["examined"] += 1
            passed = try_each(examples, globs)
            if not passed:
                continue
            code = code_of(passed)
            wants = want_of(passed)
            if not wants:
                continue
            exp_txt = "\n".join(f"Expect: {w}" for w in wants[:6])
            text = (
                f"### Instruction:\nDemonstrate {qual}: "
                f"{summary(doc)}\n\n"
                f"### Response:\n<think>\nGoal: show {qual} usage.\n"
                f"{exp_txt}\n"
                f"</think>\n```python\n{code}\n```\n<think>\n"
                f"Observed: all {len(passed)} doctests pass, outputs match.\n"
                f"Verdict: verified by execution.\n</think>\n")
            h = hashlib.sha256(text.encode()).hexdigest()[:16]
            if h in seen:
                continue
            seen.add(h)
            rows.append(text)
            stats["kept"] += 1
            if stats["kept"] % 200 == 0:
                print(f"  kept {stats['kept']} (examined {stats['examined']})",
                      flush=True)
    print("examined:", stats["examined"], "kept:", stats["kept"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".tmp", "w") as f:
        for text in rows:
            ids = enc.encode(text)
            f.write(f"{BOS} " + " ".join(map(str, ids)) + "\n")
    os.rename(OUT + ".tmp", OUT)
    print(f"kept {len(rows)} rows -> {OUT} (pack with pack_rows.py)")


if __name__ == "__main__":
    main()
