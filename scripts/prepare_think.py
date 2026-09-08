#!/usr/bin/env python3
"""
Phase 3.5 curriculum: deliberation-shaped math traces ("thinking").

Same synthetic problems as prepare_math.py, but each response is a
deliberation sandwich: <think>plan</think> + steps + <think>verify</think>.
The verify block re-checks the result by an INDEPENDENT computation
(substitution, inverse op, divisibility) — the self-check shape that
plain step cloning (Phase 3) never teaches.

Two deliberate differences from prepare_math.py:
  1. Numbers are sampled ONCE per problem and used consistently in both
     instruction and solution (prepare_math calls randint separately for
     inst and sol, so most of its rows ask one question and solve another).
  2. Output is think_reasoning.txt (does NOT clobber math_reasoning.txt).

Output: ~/.cache/autoresearch/think_reasoning.txt
Each row: BOS + "### Instruction: ..." + "### Response: <think>...</think>..."
"""

import os, sys, time, random, hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_math as P  # noqa: E402  (reuses step fns, encoder, write, BOS)

BOS = P.BOS
CACHE = os.path.expanduser("~/.cache/autoresearch")
OUT = os.path.join(CACHE, "think_reasoning.txt")

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def think_block(*lines):
    return THINK_OPEN + "\n" + "\n".join(lines) + "\n" + THINK_CLOSE


def add_row():
    a, b = random.randint(10, 999), random.randint(10, 999)
    c = a + b
    inst = f"What is {a} + {b}?"
    think = think_block(
        f"Restate: add {a} and {b}.",
        "Plan: column addition, ones first, carry to tens, because "
        "single-digit sums are exact and carry propagates left.",
    )
    steps = P.arith_add(a, b)
    verify = think_block(
        f"Verify by subtraction: {c} - {b} = {c - b}.",
        "Matches the first addend, so the sum stands." if c - b == a
        else "MISMATCH — recompute.",
    )
    return inst, f"{think}\n{steps}\n{verify}\n"


def mul_row():
    a, b = random.randint(2, 99), random.randint(2, 99)
    c = a * b
    inst = f"What is {a} × {b}?"
    think = think_block(
        f"Restate: multiply {a} by {b}.",
        "Plan: direct product, then verify by division, because "
        "division inverts multiplication exactly on integers.",
    )
    steps = P.arith_mul(a, b)
    verify = think_block(
        f"Verify by division: {c} / {a} = {c // a} remainder {c % a}.",
        "Exact, so the product stands." if c % a == 0 and c // a == b
        else "MISMATCH — recompute.",
    )
    return inst, f"{think}\n{steps}\n{verify}\n"


def lcm_row():
    from math import gcd
    a, b = random.randint(2, 30), random.randint(2, 30)
    g = gcd(a, b)
    l = a * b // g
    inst = f"What is the LCM of {a} and {b}?"
    think = think_block(
        f"Restate: smallest positive multiple of both {a} and {b}.",
        "Plan: GCD first, then LCM = a*b/GCD, because the product "
        "double-counts exactly the GCD. Verify by divisibility.",
    )
    steps = P.lcm(a, b)
    verify = think_block(
        f"Verify: {l} / {a} = {l // a} remainder {l % a}; "
        f"{l} / {b} = {l // b} remainder {l % b}.",
        "Both exact, so the LCM stands." if l % a == 0 and l % b == 0
        else "MISMATCH — recompute.",
    )
    return inst, f"{think}\n{steps}\n{verify}\n"


def quad_row():
    import math
    a = random.randint(1, 5)
    b, c = random.randint(-9, 9), random.randint(-9, 9)
    disc = b * b - 4 * a * c
    inst = f"Solve {a}x² + {b}x + {c} = 0."
    think = think_block(
        f"Restate: quadratic with a={a}, b={b}, c={c}.",
        "Plan: discriminant first to pick the branch (no real roots vs "
        "formula), because the sign decides everything downstream. "
        "Verify by substituting each root back.",
    )
    steps = P.quadratic(a, b, c)
    if disc < 0:
        verify = think_block(
            f"Verify: discriminant = {disc} < 0, parabola never crosses zero.",
            "No-real-roots stands.",
        )
    else:
        sqrt_d = math.sqrt(disc)
        x1 = (-b + sqrt_d) / (2 * a)
        r1 = a * x1 * x1 + b * x1 + c
        verify = think_block(
            f"Verify x1={x1:.4f}: {a}*x1^2 + {b}*x1 + {c} = {r1:.2e}.",
            "Residual ~0, so the roots stand." if abs(r1) < 1e-6
            else "MISMATCH — recompute.",
        )
    return inst, f"{think}\n{steps}\n{verify}\n"


def prime_row():
    n = random.randint(10, 200)
    sq = int(n ** 0.5)
    divs = [i for i in range(2, sq + 1) if n % i == 0]
    inst = f"Is {n} prime?"
    think = think_block(
        f"Restate: primality of {n}.",
        f"Plan: trial division 2..{sq} only, because any factor above "
        "sqrt pairs with one below. A witness divisor disproves.",
    )
    steps = P.prime_check(n)
    if divs:
        d = divs[0]
        verify = think_block(
            f"Verify: {n} = {d} x {n // d}, composite confirmed.",
            "Not prime stands.",
        )
    else:
        verify = think_block(
            f"Verify: no divisor in 2..{sq}, nothing above sqrt can pair.",
            "Prime stands.",
        )
    return inst, f"{think}\n{steps}\n{verify}\n"


def sum_row():
    n = random.randint(10, 100)
    s = n * (n + 1) // 2
    inst = f"What is 1 + 2 + ... + {n}?"
    think = think_block(
        f"Restate: triangular sum to {n}.",
        "Plan: closed form n(n+1)/2, verify by Gauss pairing "
        "(1+n, 2+(n-1), ...) which must give the same total.",
    )
    steps = P.sum_to(n)
    pairs = n // 2
    check = pairs * (n + 1) + ((n + 1) // 2 if n % 2 else 0)
    verify = think_block(
        f"Verify by pairing: {pairs} pairs x {n + 1} = {check}.",
        "Matches, so the sum stands." if check == s
        else "MISMATCH — recompute.",
    )
    return inst, f"{think}\n{steps}\n{verify}\n"


ROW_BUILDERS = [add_row, mul_row, lcm_row, quad_row, prime_row, sum_row]


def make_problems(n=2000):
    rows = []
    used = set()
    while len(rows) < n:
        inst, sol = random.choice(ROW_BUILDERS)()
        text = f"### Instruction:\n{inst}\n\n### Response:\n{sol}\n"
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        if h in used:
            continue
        used.add(h)
        rows.append(text)
    return rows


def main():
    try:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
    except OSError as e:
        print(f"mkdir: {e}")
        sys.exit(1)
    random.seed(12345)
    rows = make_problems(2000)
    print(f"Generated {len(rows)} think problems")
    t0 = time.time()
    n = P.write(rows, OUT)
    print(f"think rows: {n} in {time.time()-t0:.1f}s")
    if n == 0:
        sys.exit(1)
    print(f"\nPhase 3.5 ready: {n} rows -> {OUT}")


if __name__ == "__main__":
    main()
