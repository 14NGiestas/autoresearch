#!/usr/bin/env python3
"""bpb_from_eval.py — aggregate eval_bpb output into held-out bits-per-byte.

eval_bpb prints one line per row with T per-position NLLs (natural log, as
argmax/CE without the metric layer). The trainer's own val @N bpb is
byte-weighted: bpb = sum(nll over positions with tbytes>0) / sum(tbytes) / ln2.
This reproduces exactly that, so a number computed here on the quiet box is
comparable to the trainer's val line -- which is the self-check in
jobs/halfbeast_bpb.sbatch (step_300 must give 2.7256).

Row contract: T+1 space-separated ids per row (T from the rows file itself,
not hardcoded), inputs = ids[0:T], targets = ids[1:T+1]; token_bytes.txt
line i = byte length of id i. Non-numeric eval lines (e.g. 'row done')
are skipped.

Usage: bpb_from_eval.py EVAL_OUT ROWS_FILE TOKEN_BYTES [MAX_ROWS]
"""

import math
import sys


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    eval_out, rows_file, tbytes_file = sys.argv[1:4]
    max_rows = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9

    tbytes = [int(x) for x in open(tbytes_file).read().split()]
    id_rows = [ln.split() for ln in open(rows_file) if ln.strip()]
    T = len(id_rows[0]) - 1
    nll_rows = []
    for ln in open(eval_out):
        toks = ln.split()
        if len(toks) != T:
            continue
        try:
            [float(x) for x in toks]
        except ValueError:
            continue
        nll_rows.append(toks)

    tot_nll = 0.0
    tot_bytes = 0
    used = 0
    for r, (nl, ids) in enumerate(zip(nll_rows, id_rows)):
        if r >= max_rows:
            break
        if len(ids) != T + 1:
            print(f"row {r}: unexpected shapes {len(nl)}/{len(ids)} (T={T})",
                  file=sys.stderr)
            continue
        for p, s in enumerate(nl):
            tid = int(ids[p + 1])          # target = ids[1:T+1]
            b = tbytes[tid]
            if b > 0:
                tot_nll += float(s)
                tot_bytes += b
        used += 1
    if tot_bytes == 0:
        print("no counted bytes", file=sys.stderr)
        return 1
    print(f"rows={used} bytes={tot_bytes} nll_sum={tot_nll:.2f} "
          f"bpb={tot_nll / math.log(2) / tot_bytes:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
