#!/usr/bin/env python3
"""
ThinkTest corpus: our own Fortran test suite as verification traces.

Each test encodes a prediction (tolerance rationale in its header
comments, e.g. FD step analysis) and the suite run gives observations
(max err values, ok/FAIL). This script pairs them:
  Question (what the test verifies + why the tolerance) ->
  Thinking (expected) -> Response (the test code) ->
  Result (observed errs) -> verdict.

Sources: src/test/test_kernels.f90 (subroutines + header comments) and
a full-suite log with per-test sections (/tmp/test_out3.txt style:
`=== name ===` then max-err/ok lines). Reuses thinkx.encode (the
generalization: same row contract, new collector).
Output: thinktest_reasoning.txt (variable rows; pack with pack_rows.py).
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from thinkx.encode import write_rows  # noqa: E402

CACHE = os.path.expanduser("~/.cache/autoresearch")
OUT = os.path.join(CACHE, "thinktest_reasoning.txt")
TEST_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "src", "test", "test_kernels.f90")


def parse_log(path):
    """Map test name -> result lines from a full-suite log."""
    blocks, cur = {}, None
    for line in open(path):
        m = re.match(r"===\s+(test_\w+)", line)
        if m:
            cur = m.group(1)
            blocks[cur] = []
        elif cur is not None and re.search(
                r"max err|rel err|abs err|\bok\b|FAIL|nll|error",
                line, re.I):
            blocks[cur].append(line.strip())
    return {k: v for k, v in blocks.items() if v}


def parse_tests(path):
    """Map test name -> (header comment lines, subroutine source)."""
    src = open(path).read().splitlines()
    out = {}
    i = 0
    while i < len(src):
        m = re.match(r"\s*subroutine\s+(test_\w+)\s*\(", src[i])
        if m:
            name = m.group(1)
            comments = []
            j = i - 1
            while j >= 0 and src[j].strip().startswith("!"):
                comments.insert(0, src[j].strip().lstrip("!").strip())
                j -= 1
            k = i
            while k < len(src) and \
                    not re.match(r"\s*end subroutine", src[k]):
                k += 1
            out[name] = ("\n".join(comments[-8:]),
                         "\n".join(src[i:k + 1]))
            i = k
        i += 1
    return out


def main(logpath):
    blocks = parse_log(logpath)
    tests = parse_tests(TEST_SRC)
    print(f"log blocks: {len(blocks)}, test subroutines: {len(tests)}")
    rows = []
    for name, lines in sorted(blocks.items()):
        comments, code = tests.get(name, ("", ""))
        if len(code.splitlines()) > 150:
            code = "\n".join(code.splitlines()[:150]) + "\n! ... (truncated)"
        sub = [l for l in comments.splitlines()
               if l.strip() and not re.fullmatch(r"[-=!#*\s]*", l.strip())]
        purpose = sub[0] if sub else f"verify {name}"
        exp = "\n".join(sub[:6]) if sub else "tolerance per code"
        text = (
            f"### Instruction:\nVerify {name}: {purpose}\n\n"
            f"### Response:\n<think>\nGoal: {purpose}.\n"
            f"Expected: {exp}.\n"
            f"</think>\n```fortran\n{code.strip()}\n```\n<think>\n"
            f"Observed:\n" + "\n".join(f"  {l}" for l in lines[:8]) + "\n"
            f"Verdict: {'FAIL present - investigate' if any('FAIL' in l for l in lines) else 'within tolerance, stands'}.\n"
            f"</think>\n")
        rows.append(text)
    kept = write_rows(rows, OUT)
    print(f"kept {kept} rows -> {OUT} (pack with pack_rows.py)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_out3.txt")
