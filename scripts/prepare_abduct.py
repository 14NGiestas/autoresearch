#!/usr/bin/env python3
"""
Abduction corpus v1: Mastermind hidden-state inference transcripts.

Abduction (Peirce: inference to the best explanation) needs hypothesis
SETS, not single chains: one feedback fits many secrets, so the trace
must hold candidates, eliminate, and select with explicit confidence.
Every number below is computed (scorer + elimination solver), never
asserted: the solver really maintains the consistent set, so narrated
eliminations are true by construction.

Game: P pegs, C colors (classic 4x6 = 1296 secrets). Per-turn rows with
full history prefix (self-contained): history + latest feedback ->
think (set sizes, elimination examples with reasons, next guess,
confidence) -> guess. Final turn states the secret when uniquely fixed.

Output: TEXT JSONL (one {"text": ...} per line) per the thinkx boundary;
tokenize + pack on the training side later. No engine/UI/game loop.

Usage:
    python3 scripts/prepare_abduct.py [--games N --pegs P --colors C]
"""

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
from collections import Counter

CACHE = os.path.expanduser("~/.cache/autoresearch")
OUT = os.path.join(CACHE, "abduct_reasoning.jsonl")


def score(secret, guess):
    """(blacks, whites) with exact duplicate handling."""
    blacks = sum(s == g for s, g in zip(secret, guess))
    cs, cg = Counter(secret), Counter(guess)
    whites = sum(min(cs[c], cg[c]) for c in cs) - blacks
    return blacks, whites


def fmt_guess(g):
    return "[" + ", ".join(map(str, g)) + "]"


def why_out(secret, guess, fb):
    """True reason secret is ruled out by (guess -> fb)."""
    got = score(secret, guess)
    return (f"{fmt_guess(secret)} would give {got[0]} black, "
            f"{got[1]} white vs observed {fb[0]} black, {fb[1]} white")


def play_game(secret, P, C, rng):
    """Elimination solver; yields per-turn (history, think, guess) rows."""
    all_secrets = list(itertools.product(range(C), repeat=P))
    possible = list(all_secrets)
    tried = set()
    history = []  # (guess, fb)
    # first guess: random PAIR pattern [a,a,b,b] (seeded) — as
    # informative as any opener (same partition structure) so games stay
    # short, while 36 variants multiply distinct states across games.
    # (Fully random openers play worse and cost 10x compute.)
    a, b = rng.randrange(C), rng.randrange(C)
    guess = tuple([a] * (P // 2) + [b] * (P - P // 2))
    rows = []
    for turn in range(12):
        assert guess not in tried
        tried.add(guess)
        fb = score(secret, guess)
        history.append((guess, list(fb)))
        before = len(possible)
        possible = [s for s in possible
                    if all(score(s, g) == tuple(f)
                           for g, f in history)]
        after = len(possible)
        assert secret in possible, "solver lost the secret!"
        # elimination examples (verified true by construction)
        elim = [s for s in all_secrets
                if s not in possible and s not in tried]
        ex = []
        for s in elim[:2]:
            # find the feedback that rules it out
            for g, f in history:
                if score(s, g) != tuple(f):
                    ex.append(why_out(s, g, tuple(f)))
                    break
        hist_txt = "\n".join(
            f"Guess {fmt_guess(list(g))} -> {f[0]} black, {f[1]} white"
            for g, f in history)
        if after == 1:
            think = (
                f"Goal: identify the {P}-peg secret.\n"
                f"Candidates consistent with all feedback: exactly 1.\n"
                f"Eliminated this turn: {before - after} "
                f"(e.g. {ex[0]}).\n"
                f"Best: {fmt_guess(list(possible[0]))} "
                f"(only consistent hypothesis).\n"
                f"Confidence: certain.")
            rows.append((hist_txt, think,
                         f"Secret: {fmt_guess(list(possible[0]))}"))
            break
        nxt = next(s for s in possible if s not in tried)
        think = (
            f"Goal: narrow {C ** P} possibilities.\n"
            f"Candidates consistent: {after} "
            f"(eliminated {before - after} this turn"
            + (f", e.g. {ex[0]}" if ex else "") + ").\n"
            f"Best next guess: {fmt_guess(list(nxt))} "
            f"(first untried consistent candidate).\n"
            f"Confidence: 1 of {after} consistent.")
        rows.append((hist_txt, think, f"Guess: {fmt_guess(list(nxt))}"))
        guess = nxt
    else:
        raise AssertionError("solver did not converge")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=800)
    ap.add_argument("--pegs", type=int, default=4)
    ap.add_argument("--colors", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    secrets = set()
    while len(secrets) < args.games:
        secrets.add(tuple(rng.randrange(args.colors)
                          for _ in range(args.pegs)))
    nrows, nturns, seen = 0, [], set()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".tmp", "w") as f:
        for secret in sorted(secrets):
            turns = play_game(secret, args.pegs, args.colors, rng)
            nturns.append(len(turns))
            for hist_txt, think, out in turns:
                text = (
                    f"### Instruction:\nMastermind: secret is "
                    f"{args.pegs} pegs from {args.colors} colors.\n"
                    f"{hist_txt}\nFind the secret (or next best guess).\n\n"
                    f"### Response:\n<think>\n{think}\n</think>\n{out}\n")
                h = hashlib.sha256(text.encode()).hexdigest()[:16]
                if h in seen:
                    continue
                seen.add(h)
                f.write(json.dumps({"text": text}) + "\n")
                nrows += 1
    os.rename(OUT + ".tmp", OUT)
    avg = sum(nturns) / len(nturns)
    print(f"games: {len(secrets)}, rows: {nrows}, avg turns: {avg:.2f} "
          f"-> {OUT}")


if __name__ == "__main__":
    main()
