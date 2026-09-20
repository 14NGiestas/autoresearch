#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.2"]
# ///
"""Piso diferente ou colapso? Ajusta E + A*D^-beta nas duas curvas.

A pergunta: o relu deu bpb 4,15 contra 2,35 do softmax. Duas leituras, e o valor
final nao distingue:

  (a) PISO diferente: mesma inclinacao beta, piso E mais alto. A atencao funciona.
  (b) COLAPSO: a curva achata cedo, beta pequeno e E alto. A atencao morreu.

Uso: uv run scripts/decay_fit.py LOG [LOG2 ...]
Le as linhas "trn @N bpb X" e "--- braco FN" do log.
"""
import re, sys
from pathlib import Path
import numpy as np

def curves(path):
    out, cur = {}, None
    for line in Path(path).read_text().splitlines():
        m = re.match(r'--- braco (\w+)', line)
        if m: cur = m.group(1); out[cur] = []; continue
        m = re.match(r'trn @(\d+) bpb\s+([\d.eE+-]+)', line)
        if m and cur: out[cur].append((float(m.group(1)), float(m.group(2))))
    return {k: np.array(v) for k, v in out.items() if len(v) >= 4}

def fit(D, L):
    """E + A*D^-beta, com E varrido. Devolve (erro, E, beta, A)."""
    best = None
    for E in np.arange(0.0, min(L) - 1e-3, 0.01):
        x = D**-1.0
        for beta in np.linspace(0.02, 1.2, 400):
            X = np.vstack([np.ones_like(D), D**-beta]).T
            c, *_ = np.linalg.lstsq(X, L - E, rcond=None)
            pred = E + X @ c
            e = float(np.max(np.abs(pred - L)))
            if best is None or e < best[0]:
                best = (e, E, beta, c[1])
    return best

def main():
    data = {}
    for p in sys.argv[1:]:
        data.update(curves(p))
    if not data:
        print("sem curvas no log"); return 1
    print(f"  {'braco':<10} {'pontos':>7} {'E (piso)':>10} {'beta':>7} {'erro':>8}")
    res = {}
    for k, a in data.items():
        D, L = a[:, 0], a[:, 1]
        e, E, beta, A = fit(D, L)
        res[k] = (E, beta)
        print(f"  {k:<10} {len(D):>7} {E:>10.3f} {beta:>7.3f} {e:>8.4f}")
    if len(res) == 2:
        (E1, b1), (E2, b2) = list(res.values())
        print()
        print(f"  piso:  {E1:.3f} contra {E2:.3f}  -> diferenca {abs(E2-E1):.3f} bpb")
        print(f"  beta:  {b1:.3f} contra {b2:.3f}  -> razao {max(b1,b2)/max(1e-9,min(b1,b2)):.2f}x")
        print()
        print("  leitura: piso diferente se os betas ficam perto e os pisos longe.")
        print("           colapso se o beta do segundo despenca.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
