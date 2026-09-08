"""Doctest mining core: harvest -> parse -> execute -> rows.

Same proven pipeline as the retired bespoke miners (thinkscipy,
thinksympy, thinksklearn): per-example isolated runs over shared globs,
may-vary skipped, SIGALRM time gate, code-level cross-module dedupe,
rows kept only with non-empty expected outputs.
"""

import doctest
import inspect
import pkgutil
import re
import signal

FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def harvest(pkgnames, skip_mod, extra_globs=None):
    """Yield (qualname, doc, globs) for public callables with doctests."""
    for pkgname in pkgnames:
        try:
            pkg = __import__(pkgname, fromlist=["*"])
        except Exception:
            continue
        mods = [pkgname]
        try:
            mods += [m.name for m in pkgutil.walk_packages(
                pkg.__path__, pkg.__name__ + ".")]
        except Exception:
            pass
        top = dict(vars(pkg))
        for k, v in (extra_globs or {}).items():
            top.setdefault(k, v)
        for modname in mods:
            if skip_mod.search(modname):
                continue
            try:
                mod = __import__(modname, fromlist=["*"])
            except Exception:
                continue
            globs = dict(vars(mod))
            for k, v in top.items():
                globs.setdefault(k, v)
            for name in dir(mod):
                if name.startswith("_"):
                    continue
                try:
                    obj = getattr(mod, name)
                except Exception:
                    continue
                if not isinstance(getattr(obj, "__doc__", ""), str):
                    continue
                if not (inspect.isfunction(obj) or inspect.isbuiltin(obj)
                        or inspect.isclass(obj)):
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
                            if not isinstance(mdoc, str):
                                continue
                        except Exception:
                            continue
                        if mdoc and ">>>" in mdoc:
                            yield (f"{modname}.{name}.{mn}", mdoc, globs)


def parse_examples(doc, globs, skip_src):
    """Parse Examples section; return example list or None."""
    if not doc or ">>>" not in doc:
        return None
    m = re.search(
        r"^[ \t]*Examples\n[ \t]*-+\n(.*?)(?=^[ \t]*\S[^\n]*\n[ \t]*-+\n|\Z)",
        doc, re.M | re.S)
    block = m.group(1) if m else doc
    if ">>>" not in block or (skip_src and skip_src.search(block)):
        return None
    try:
        tests = doctest.DocTestParser().get_doctest(
            block, globs, "ex", "ex", 0)
    except Exception:
        return None
    return tests.examples or None


def run_each(examples, globs, secs):
    """Run each example separately (shared globs); return passing ones."""
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
    return [ex.want.strip() for ex in examples if ex.want.strip()]


def summary(doc):
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        line = line.strip()
        if line:
            return line[:160]
    return ""


def shape_row(qual, doc, code, wants, npassed):
    exp_txt = "\n".join(f"Expect: {w}" for w in wants[:6])
    return (
        f"### Instruction:\nDemonstrate {qual}: {summary(doc)}\n\n"
        f"### Response:\n<think>\nGoal: show {qual} usage.\n{exp_txt}\n"
        f"</think>\n```python\n{code}\n```\n<think>\n"
        f"Observed: all {npassed} doctests pass, outputs match.\n"
        f"Verdict: verified by execution.\n</think>\n")


def mine(config):
    """Run a config dict; return (rows, stats). Config keys:
    pkgs, skip_mod, skip_src, timeout, legacy, extra_globs, setup.
    """
    import re as _re
    if config.get("legacy"):
        try:
            import numpy as _np
            _np.set_printoptions(legacy=config["legacy"])
        except Exception:
            pass
    setup = config.get("setup")
    if setup:
        try:
            setup()
        except Exception:
            pass
    skip_mod = _re.compile(config["skip_mod"])
    skip_src = (_re.compile(config["skip_src"])
                if config.get("skip_src") else None)
    rows, seen, seen_code = [], set(), set()
    stats = {"examined": 0, "kept": 0, "dupcode": 0}
    import hashlib
    for qual, doc, globs in harvest(config["pkgs"], skip_mod,
                                    config.get("extra_globs")):
        examples = parse_examples(doc, globs, skip_src)
        if not examples:
            continue
        stats["examined"] += 1
        try:
            passed = run_each(examples, globs, config.get("timeout", 15))
        except Exception:
            continue
        if not passed:
            continue
        code = code_of(passed)
        wants = want_of(passed)
        if not wants:
            continue
        ck = _re.sub(r"\s+", " ", code)
        if ck in seen_code:
            stats["dupcode"] += 1
            continue
        seen_code.add(ck)
        text = shape_row(qual, doc, code, wants, len(passed))
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        if h in seen:
            continue
        seen.add(h)
        rows.append(text)
        stats["kept"] += 1
        if stats["kept"] % 200 == 0:
            print(f"  kept {stats['kept']} (examined {stats['examined']})",
                  flush=True)
    return rows, stats
