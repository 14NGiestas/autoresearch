#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.5.2",
#   "pyarrow==25.0.1",
# ]
# ///
"""nli_probe.py — "jeV de graca": o LM atual sabe ordenar entailment?

Ideia do openjev (NLI cross-encoder como primitiva universal) trazida para a
nossa escala SEM treinar nada: no ASSIN2-RTE (PT, entailment vs nao-entailment)
medimos se a verossimilhanca condicional do nosso LM ja ordena os pares.

Metodo (Holtzman et al., o classico de zero-shot NLI com LM):
  ganho = NLL_medio(hipotese)            -- NLL_medio(hipotese | premissa)
Quanto MAIOR o ganho, mais a premissa ajuda a prever a hipotese => entailment.
Limiar ajustado na validacao (maximiza acuracia), reportado no teste. Acaso 50%.

Duas linhas por par (o eval_bpb pontua T=1024 posicoes por linha, entao a
leitura e por posicao, com o span da hipotese recortado):
  A: [BOS] premissa hipotese   -> NLL nas posicoes da hipotese
  B: [BOS] hipotese            -> NLL nas posicoes da hipotese
Padding depois do par (irrelevante: so recortamos o span).

Uso:
  scripts/nli_probe.py build --split validation --rows /tmp/nli_val_A.npy --spans /tmp/nli_val_A.json
  scripts/nli_probe.py score --split validation --ev-A ... --ev-B ... 
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_driver import load_enc  # noqa: E402

PARQ = "/tmp/assin2_{split}.parquet"
T = 1024
BOS = 8188


def pairs(split):
    import pyarrow.parquet as pq
    t = pq.read_table(PARQ.format(split=split))
    prem = t.column("premise").to_pylist()
    hyp = t.column("hypothesis").to_pylist()
    lab = t.column("entailment_judgment").to_pylist()
    return list(zip(prem, hyp, [int(x) for x in lab]))


def build(split, out_rows, out_spans):
    enc = load_enc()
    data = pairs(split)
    rows, spans = [], []
    for prem, hyp, lab in data:
        p = enc.encode(prem)
        h = enc.encode(hyp)
        if not h:
            continue
        # A: premissa + hipotese  (span da hipotese)
        a = p + h
        if len(a) > T:
            a = a[:T]
        rowA = [BOS] + a + [BOS] * (T - len(a))
        rows.append(rowA)
        spans.append({"kind": "A", "start": 1 + len(p), "end": 1 + len(a),
                      "label": lab, "n_hyp": len(a) - len(p)})
        # B: hipotese sozinha
        b = h[:T]
        rowB = [BOS] + b + [BOS] * (T - len(b))
        rows.append(rowB)
        spans.append({"kind": "B", "start": 1, "end": 1 + len(b),
                      "label": lab, "n_hyp": len(b)})
    arr = np.asfortranarray(np.asarray(rows, dtype=np.int32))
    np.save(out_rows, arr)
    json.dump(spans, open(out_spans, "w"))
    n = len(spans) // 2
    print(f"{split}: {n} pares -> {arr.shape[0]} linhas ({out_rows})")
    pos = sum(s["label"] for s in spans[::2])
    print(f"  entailment: {pos}/{n} ({pos/n:.1%}) — maioria prevista = "
          f"{max(pos, n-pos)/n:.3f}")


def read_nlls(path):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        s = line.strip()
        if not s or s[0] not in "0123456789-":
            continue
        out.append(np.fromstring(s, sep=" "))
    return out


def score(rows_specs):
    """rows_specs: lista de (spans_json, ev_txt) na ordem A,B,A,B,..."""
    spans = json.load(open(rows_specs[0][0]))
    nll = []
    for sp, ev in rows_specs:
        s = json.load(open(sp))
        assert len(s) == len(spans)
        nll.append(read_nlls(ev))
    per = {}
    for i, sp in enumerate(spans):
        nl = nll[i % len(nll)][i // len(nll)] if len(nll) > 1 else nll[0][i]
        # nll[k] tem as linhas do arquivo k; aqui 1 arquivo por chamada
        pass
    return spans, nll


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="what", required=True)
    b = sub.add_parser("build")
    b.add_argument("--split", required=True)
    b.add_argument("--rows", required=True)
    b.add_argument("--spans", required=True)
    s = sub.add_parser("score")
    s.add_argument("--split", required=True)
    s.add_argument("--spans", required=True)
    s.add_argument("--ev", required=True, help="txt de NLLs (A e B intercalados)")
    s.add_argument("--fit", default="", help="spans da validacao p/ ajustar limiar")
    s.add_argument("--ev-fit", default="")
    a = ap.parse_args()
    if a.what == "build":
        build(a.split, a.rows, a.spans)
        return

    def gains(spans_path, ev_path):
        spans = json.load(open(spans_path))
        nl = read_nlls(ev_path)
        assert len(nl) == len(spans), (len(nl), len(spans))
        g = []
        for i in range(0, len(spans), 2):
            sA, sB, nA = spans[i], spans[i + 1], nl[i]
            nB = nl[i + 1]
            ha = nA[sA["start"]:sA["end"]]
            hb = nB[sB["start"]:sB["end"]]
            if len(ha) == 0 or len(hb) == 0:
                continue
            g.append((float(hb.mean() - ha.mean()), sA["label"]))
        return g

    g = gains(a.spans, a.ev)
    if a.fit:
        gf = gains(a.fit, a.ev_fit)
        xs = np.array([x for x, _ in gf])
        ys = np.array([y for _, y in gf])
        thr = float(np.median(xs))
        best = (-1, thr)
        for t in np.linspace(xs.min(), xs.max(), 400):
            acc = ((xs > t).astype(int) == ys).mean()
            if acc > best[0]:
                best = (acc, float(t))
        print(f"validacao: limiar {best[1]:.4f} acuracia {best[0]:.3f} "
              f"(acaso {max(ys.mean(), 1-ys.mean()):.3f} maioria)")
        thr = best[1]
    else:
        thr = float(np.median([x for x, _ in g]))
    xs = np.array([x for x, _ in g])
    ys = np.array([y for _, y in g])
    acc = ((xs > thr).astype(int) == ys).mean()
    maj = max(ys.mean(), 1 - ys.mean())
    print(f"=== {a.split}: {len(g)} pares")
    print(f"  acaso (maioria)      {maj:.3f}")
    print(f"  NLI zero-shot        {acc:.3f}  (limiar {thr:+.4f} nats)")
    print(f"  ganho medio entail   {xs[ys == 1].mean():+.4f}")
    print(f"  ganho medio nao      {xs[ys == 0].mean():+.4f}")
    print(f"  separacao (d')       {abs(xs[ys==1].mean()-xs[ys==0].mean())/xs.std():.3f}")


if __name__ == "__main__":
    main()
