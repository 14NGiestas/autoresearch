#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""pick_anchors.py — IRT-inspired anchor rows for fast proxy eval (stdlib only).

Idea (tinyBenchmarks spirit, adapted): per-position NLLs are continuous, so
instead of fitting a 2PL IRT on binary correctness we do greedy forward
selection of ROWS that best preserve the full-val ranking (and score) across
models. Validated by leave-one-model-out: anchors picked on M-1 models must
rank/predict the held-out model.

Usage:
  .venv-numpy/bin/python3 scripts/pick_anchors.py --k 100 \
    --eval p2a:/tmp/eval_p2a.txt p2b:/tmp/eval_p2b.txt ... \
    --out /tmp/mix/val_anchor.txt

Output: report on stdout (rank match, LOO score error, baselines) + 1-based
row indices, one per line, in --out.
"""
import argparse
import sys


def load_row_sums(path, expect=None):
    sums = []
    with open(path) as f:
        for ln, line in enumerate(f):
            s = 0.0
            for tok in line.split():
                s += float(tok)
            sums.append(s)
    if expect is not None and len(sums) != expect:
        raise ValueError(f"{path}: {len(sums)} rows, expected {expect}")
    return sums


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0] * len(xs)
    for pos, i in enumerate(order):
        r[i] = pos
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((x - mb) ** 2 for x in rb)
    if va == 0 or vb == 0:
        return 1.0 if va == vb else 0.0
    return cov / (va * vb) ** 0.5


def subset_means(mat, rows):
    return [sum(mat[r][m] for r in rows) / len(rows) for m in range(len(mat[0]))]


def greedy(mat, k, cand_models):
    full = subset_means(mat, range(len(mat)))
    full_sub = [full[m] for m in cand_models]
    picked, remaining = [], set(range(len(mat)))
    # seed: row whose cross-model profile correlates best with full means
    best = max(remaining,
               key=lambda r: spearman([mat[r][m] for m in cand_models], full_sub))
    picked.append(best)
    remaining.discard(best)
    while len(picked) < k and remaining:
        cur = subset_means(mat, picked)
        cur_sub = [cur[m] for m in cand_models]
        best, best_c = None, -2.0
        for r in remaining:
            trial = [(cur[m] * len(picked) + mat[r][m]) / (len(picked) + 1)
                     for m in cand_models]
            c = spearman(trial, full_sub)
            if c > best_c:
                best, best_c = r, c
        picked.append(best)
        remaining.discard(best)
    return picked


def loo_score(mat, k):
    """Leave-one-model-out: pick anchors on M-1, predict held-out full mean
    via scale fit (subset_mean * full/sub on the M-1). Returns mean |rel err|."""
    M = len(mat[0])
    errs = []
    for held in range(M):
        others = [m for m in range(M) if m != held]
        anch = greedy(mat, k, others)
        sub = subset_means(mat, anch)
        scale = sum(subset_means(mat, range(len(mat)))[m] for m in others) / \
            sum(sub[m] for m in others)
        pred = sub[held] * scale
        true = sum(mat[r][held] for r in range(len(mat))) / len(mat)
        errs.append(abs(pred - true) / abs(true))
    return sum(errs) / len(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--eval", nargs="+", required=True,
                    help="name:path pairs, all with identical row counts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    names, mats = [], []
    nrows = None
    for pair in args.eval:
        name, path = pair.split(":", 1)
        sums = load_row_sums(path)
        if nrows is None:
            nrows = len(sums)
        elif len(sums) != nrows:
            raise ValueError(f"{path}: row mismatch")
        names.append(name)
        mats.append(sums)
    # transpose to rows x models
    mat = [[mats[m][r] for m in range(len(names))] for r in range(nrows)]
    M = len(names)
    print(f"matrix: {nrows} rows x {M} models {names}")

    full = subset_means(mat, range(nrows))
    print("full means:", " ".join(f"{names[m]}={full[m]:.1f}" for m in range(M)))

    for k in (60, args.k):
        anch = greedy(mat, k, list(range(M)))
        sub = subset_means(mat, anch)
        print(f"k={k}: spearman(all)={spearman(sub, full):.4f} "
              f"loo_relerr={loo_score(mat, k):.4f}")

    # baselines at k=60: first-60 and random-60 (seeded)
    import random
    base = subset_means(mat, range(60))
    print(f"base first-60: spearman={spearman(base, full):.4f}")
    rng = random.Random(args.seed)
    rands = []
    for _ in range(5):
        rr = rng.sample(range(nrows), 60)
        rands.append(spearman(subset_means(mat, rr), full))
    print(f"base random-60: spearman={min(rands):.4f}..{max(rands):.4f}")

    anch = greedy(mat, args.k, list(range(M)))
    with open(args.out, "w") as f:
        for r in anch:
            f.write(f"{r + 1}\n")
    print(f"wrote {len(anch)} 1-based indices -> {args.out}")


if __name__ == "__main__":
    main()
