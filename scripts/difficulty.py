#!/usr/bin/env python3
"""Cheap difficulty signals for prepare_*.py (arXiv:2506.11300).

Three stdlib-only metrics, preferred over hand-rolled intuitions:

* ``compression_ratio`` — len(zlib(text))/len(utf8 bytes); low = repetitive.
* ``mtld`` — measure of textual lexical diversity (McCarthy & Jarvis);
  low = lexically poor. Forward+backward average, factor 0.72.
* ``flesch`` — Flesch Reading Ease on the row text; low = harder to read.

``score`` combines them into one easy-first sort key (lower = easier):
mean of per-metric z-normalized ranks is overkill here; we use a fixed
blend of normalized components. Keep it dumb and comparable.
"""

import math
import re
import zlib

WORD = re.compile(r"[A-Za-z']+")
SENT = re.compile(r"[.!?]+")


def compression_ratio(text):
    raw = text.encode("utf-8", "replace")
    if not raw:
        return 1.0
    return len(zlib.compress(raw, 6)) / len(raw)


def _mtld_side(words, ttr=0.72):
    factors, start, types = 0, 0, set()
    for i, w in enumerate(words, 1):
        types.add(w)
        if len(types) / (i - start) <= ttr:
            factors += 1
            start, types = i, set()
    rest = len(words) - start
    if rest:
        factors += (1 - (len(types) / rest - ttr) / (1 - ttr)) \
            if len(types) / rest > ttr else 0
    return len(words) / factors if factors else len(words)


def mtld(text):
    words = [w.lower() for w in WORD.findall(text)]
    if len(words) < 10:
        return float(len(set(words)))
    fwd = _mtld_side(words)
    bwd = _mtld_side(words[::-1])
    return (fwd + bwd) / 2


def flesch(text):
    words = WORD.findall(text)
    if not words:
        return 100.0
    sents = [s for s in SENT.split(text) if s.strip()]
    nsent = max(1, len(sents))
    syll = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower())))
               for w in words)
    return (206.835 - 1.015 * len(words) / nsent
            - 84.6 * syll / len(words))


def metrics(text):
    """Return the three raw difficulty signals for a row text."""
    return {"compression": compression_ratio(text),
            "mtld": mtld(text),
            "flesch": flesch(text)}


def score(text):
    """One easy-first sort key (lower = easier).

    Blend: repetitive (low compression ratio is *easy* here — highly
    structured rows) counts little; lexical richness (high MTLD) and
    readability (high Flesch) dominate. All components min-maxed on
    plausible ranges so no single metric swamps the others.
    """
    m = metrics(text)
    comp = min(1.0, max(0.0, m["compression"]))          # 0..1
    mt = min(1.0, max(0.0, m["mtld"] / 200.0))           # ~0..1
    fl = min(1.0, max(0.0, m["flesch"] / 100.0))         # ~0..1
    return 0.2 * comp + 0.4 * (1.0 - mt) + 0.4 * (1.0 - fl)


def order(texts, mode="easy"):
    """Return texts ordered easy-first (stable); mode 'none' = as-is."""
    if mode != "easy":
        return list(texts)
    return [t for _, _, t in sorted(((score(t), i, t)
                                      for i, t in enumerate(texts)))]
