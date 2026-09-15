#!/usr/bin/env python3
"""mc_bench.py — multiple-choice por perplexidade (sem instruction-following).
parquet ARC-PT -> rows (Q+alternativa, trunc/pad T) + sidecar (lens, gabarito).
Régua: argmin NLL (prefixo comum cancela). Uso:
  .venv-numpy/bin/python3 scripts/mc_bench.py
"""
import ast
import json
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "scripts")

BOS = 8188
T = 1024


def main():
    from eval_driver import load_enc
    enc = load_enc()
    t = pq.read_table("/tmp/bench/arc_test.parquet").to_pylist()
    rows, lens, keys, counts, nq = [], [], [], [], 0
    for r in t:
        q = str(r["question_translated"])
        try:
            raw = r["choices_translated"]
            ch = ast.literal_eval(raw) if isinstance(raw, str) else raw
            labs, texts = ch["label"], ch["text"]
        except (ValueError, SyntaxError, TypeError, KeyError):
            continue
        ans = str(r["answerKey"]).strip()
        ids_list = []
        ok = True
        for lab, txt in zip(labs, texts):
            ids = [BOS] + enc.encode_ordinary(f"{q} {txt}")[:T]
            ids_list.append(ids)
        try:
            gi = [str(x).strip() for x in labs].index(ans)
        except ValueError:
            continue
        for ids in ids_list:
            ln = len(ids)
            ids += [0] * (T + 1 - ln)
            rows.append(ids)
            lens.append(ln)
        keys.append(gi)
        counts.append(len(ids_list))
        nq += 1
    np.save("/tmp/bench/arc_rows.npy",
            np.asfortranarray(np.asarray(rows, dtype=np.int32)))
    json.dump({"lens": lens, "keys": keys, "counts": counts, "nq": nq},
              open("/tmp/bench/arc_side.json", "w"))
    print(f"questoes={nq} rows={len(rows)}")


if __name__ == "__main__":
    main()
