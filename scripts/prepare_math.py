#!/usr/bin/env python3
"""
Phase 3 curriculum: step-by-step math reasoning.

Uses synthetic problems (arithmetic, algebra) with explicit step-by-step
solutions, formatted to teach the "let me think..." pattern the model
already emits in CoT.

Output: ~/.cache/autoresearch/math_reasoning.txt
Each row: BOS + "### Instruction: ..." + "### Response: <steps>"
"""

import os, sys, time, random, hashlib

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
        print("No tokenizer, fallback"); enc = FallbackEncoder()

CACHE = os.path.expanduser("~/.cache/autoresearch")
OUT = os.path.join(CACHE, "math_reasoning.txt")

# ---------------------------------------------------------------------------
# Synthetic math problems with step-by-step solutions
# ---------------------------------------------------------------------------

def arith_add(a, b):
    steps = [
        f"Let me add {a} and {b}.",
        f"Start with the ones column: {a % 10} + {b % 10} = {(a + b) % 10}.",
        f"Carry the {a // 10 + b // 10 if a + b >= 10 else 0} to the tens column.",
        f"Result: {a + b}.",
    ]
    return "\n".join(steps)

def arith_mul(a, b):
    steps = [
        f"Let me multiply {a} × {b}.",
        f"First, {a} × {b} = {a * b}.",
        f"Result: {a * b}.",
    ]
    return "\n".join(steps)

def lcm(a, b):
    from math import gcd
    g = gcd(a, b)
    l = a * b // g
    steps = [
        f"Let me find the LCM of {a} and {b}.",
        f"First compute GCD({a}, {b}) = {g}.",
        f"LCM(a, b) = a × b / GCD = {a} × {b} / {g} = {l}.",
        f"Result: {l}.",
    ]
    return "\n".join(steps)

def fib_n(n):
    a, b = 0, 1
    seq = []
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    steps = [
        f"Let me find the first {n} Fibonacci numbers.",
        f"Start with a=0, b=1.",
    ]
    for i, x in enumerate(seq[:min(n, 6)]):
        steps.append(f"Step {i+1}: yield {x}, then a,b = b, a+b = {seq[i+1] if i+1 < n else '...'}")
    steps.append(f"Result: {seq}.")
    return "\n".join(steps)

def prime_check(n):
    if n < 2: return f"{n} is not prime (less than 2)."
    is_p = all(n % i != 0 for i in range(2, int(n**0.5) + 1))
    factors = [i for i in range(2, n) if n % i == 0] or []
    steps = [
        f"Let me check if {n} is prime.",
        f"Test divisibility from 2 to {int(n**0.5)}.",
        f"Divisors found: {factors if factors else 'none'}.",
        f"Result: {n} is{' ' if is_p else ' not '}prime.",
    ]
    return "\n".join(steps)

def factorial(n):
    if n > 12: n = 12  # keep small for brevity
    p = 1
    steps = [f"Let me compute {n}!."]
    for i in range(1, n + 1):
        p *= i
        steps.append(f"After {i}: product = {p}.")
    steps.append(f"Result: {n}! = {p}.")
    return "\n".join(steps)

def quadratic(a, b, c):
    disc = b*b - 4*a*c
    steps = [
        f"Solve {a}x² + {b}x + {c} = 0.",
        f"Discriminant = b² - 4ac = {b}² - 4×{a}×{c} = {disc}.",
    ]
    if disc < 0:
        steps.append("Discriminant < 0: no real roots.")
        return "\n".join(steps)
    import math
    sqrt_d = math.sqrt(disc)
    x1 = (-b + sqrt_d) / (2*a)
    x2 = (-b - sqrt_d) / (2*a)
    steps.append(f"√discriminant = {sqrt_d:.4f}.")
    steps.append(f"x = (-b ± √D) / 2a = ({-b} ± {sqrt_d:.4f}) / {2*a}")
    steps.append(f"x1 = {x1:.4f}, x2 = {x2:.4f}")
    return "\n".join(steps)

def sum_to(n):
    s = n * (n + 1) // 2
    steps = [
        f"Let me compute 1 + 2 + ... + {n}.",
        f"Use the formula n(n+1)/2 = {n} × {n+1} / 2 = {s}.",
        f"Result: {s}.",
    ]
    return "\n".join(steps)

PROBLEM_KINDS = [
    lambda: ("Add", f"What is {random.randint(10,999)} + {random.randint(10,999)}?", arith_add(random.randint(10,999), random.randint(10,999))),
    lambda: ("Multiply", f"What is {random.randint(2,99)} × {random.randint(2,99)}?", arith_mul(random.randint(2,99), random.randint(2,99))),
    lambda: ("LCM", f"What is the LCM of {random.randint(2,30)} and {random.randint(2,30)}?", lcm(random.randint(2,30), random.randint(2,30))),
    lambda: ("Fibonacci", f"List the first {random.randint(5,8)} Fibonacci numbers.", fib_n(random.randint(5,8))),
    lambda: ("Prime", f"Is {random.randint(10,200)} prime?", prime_check(random.randint(10,200))),
    lambda: ("Factorial", f"Compute {random.randint(3,10)}!", factorial(random.randint(3,10))),
    lambda: ("Sum", f"What is 1 + 2 + ... + {random.randint(10,100)}?", sum_to(random.randint(10,100))),
    lambda: ("Quadratic", f"Solve {random.randint(1,5)}x² + {random.randint(-9,9)}x + {random.randint(-9,9)} = 0.", quadratic(random.randint(1,5), random.randint(-9,9), random.randint(-9,9))),
]

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def make_problems(n=2000):
    rows = []
    used = set()
    while len(rows) < n:
        kind, inst, sol = random.choice(PROBLEM_KINDS)()
        text = f"### Instruction:\n{inst}\n\n### Response:\n{sol}\n"
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        if h in used: continue
        used.add(h)
        rows.append(text)
    return rows

def write(rows, out):
    t0 = time.time()
    tmp = out + ".tmp"
    try: os.remove(tmp)
    except OSError: pass
    n = 0
    try:
        f = open(tmp, "w")
    except OSError as e:
        print(f"open err: {e}"); return 0
    with f:
        for text in rows:
            try:
                ids = enc.encode(text)
                ids = [BOS] + list(ids)
                f.write(" ".join(str(i) for i in ids) + "\n")
                n += 1
            except Exception as e:
                print(f"  err: {e}")
    try: os.rename(tmp, out)
    except OSError as e: print(f"rename: {e}"); return 0
    size = os.path.getsize(out) / 1e6
    print(f"Wrote {n} → {out} ({size:.2f} MB) in {time.time()-t0:.1f}s")
    return n

def main():
    try: os.makedirs(os.path.dirname(OUT), exist_ok=True)
    except OSError as e: print(f"mkdir: {e}"); sys.exit(1)
    rows = make_problems(2000)
    print(f"Generated {len(rows)} math problems")
    n = write(rows, OUT)
    if n == 0: sys.exit(1)
    print(f"\n✓ Phase 3 ready: {n} rows")

if __name__ == "__main__":
    main()
