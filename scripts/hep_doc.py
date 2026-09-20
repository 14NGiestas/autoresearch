#!/usr/bin/env python3
"""Gera docs/hypotheses.md a partir de hep/registry.jsonl.

O documento e' DERIVADO do registro, nunca escrito a mao. Assim o doc e o
protocolo nao podem divergir: se um doc e um registro discordam, um dos dois
mente, e ninguem sabe qual.

Cada hipotese sai com: id, estado, crenca, quantas evidencias apoiam e quantas
refutam, a afirmacao, o observavel testavel, e as fontes externas.

Uso:  .venv-numpy/bin/python3 scripts/hep_doc.py [--out docs/hypotheses.md]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REG = Path("hep/registry.jsonl")


def load():
    hyp = {}
    ev = defaultdict(list)
    for line in REG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        p = d.get("payload", {})
        if d["type"] == "propose":
            hyp[p["hyp"]] = dict(p)
            hyp[p["hyp"]].setdefault("state", "proposed")
        elif d["type"] == "transition":
            h = p.get("hyp")
            if h in hyp:
                hyp[h]["state"] = p.get("state", hyp[h]["state"])
        elif d["type"] == "evidence":
            ev[p.get("hyp")].append(p)
    return hyp, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/hypotheses.md")
    a = ap.parse_args()
    hyp, ev = load()

    lines = []
    w = lines.append
    w("# Hypotheses of this lab")
    w("")
    w("Derived from `hep/registry.jsonl` by `scripts/hep_doc.py`. Do not edit by")
    w("hand: run the script. A document and a registry that disagree leave no way")
    w("to know which one is wrong.")
    w("")
    n = len(hyp)
    states = Counter(h["state"] for h in hyp.values())
    w(f"{n} hypotheses. States: " +
      ", ".join(f"{k} {v}" for k, v in sorted(states.items())) + ".")
    w("")

    # Ordena por quantas evidencias refutam (o que precisa de atencao primeiro),
    # depois por crenca. Uma hipotese com evidencia contra e' uma tarefa.
    def key(h):
        e = ev[h]
        ref = sum(1 for x in e if x.get("direction") == "refutes")
        return (-ref, -len(e), h)

    for h in sorted(hyp, key=key):
        d = hyp[h]
        e = ev[h]
        sup = sum(1 for x in e if x.get("direction") == "supports")
        ref = sum(1 for x in e if x.get("direction") == "refutes")
        belief = e[-1]["updated"] if e else d.get("prior")
        w(f"## {h} -- {d['state']}, belief {belief}")
        w("")
        w(d.get("statement", "(sem afirmacao)"))
        w("")
        w(f"* evidence: {len(e)} ({sup} support, {ref} refute)")
        if d.get("testable_observable"):
            w(f"* testable: {d['testable_observable']}")
        if d.get("mechanism"):
            w(f"* mechanism: {d['mechanism']}")
        if d.get("parents"):
            w(f"* parents: {', '.join(d['parents'])}")
        srcs = [x["source"] for x in e if x.get("source")]
        if srcs:
            w("* sources: " + "; ".join(dict.fromkeys(srcs)))
        w("")

    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"{a.out}: {n} hipoteses, {sum(len(v) for v in ev.values())} evidencias")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
