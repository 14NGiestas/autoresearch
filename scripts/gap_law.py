#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib==3.10.8",
#   "numpy==2.5.2",
# ]
# ///
"""A lei do gap: quao rapido convergimos para a saturacao.

A pergunta nao e' "qual e' o piso" (que depende do corpus e nao compara entre
nos e a literatura) e tambem nao e' "ignore o piso" (que descarta o objeto).
A pergunta util e': QUAO RAPIDO o gap fecha. O objeto e'

    G(D) = L(D) - E = A * D^-beta

e a resposta e' beta. Como a nossa faixa admissivel de E e' estreita, beta sai
com barra pequena -- a incerteza do piso e' PROPAGADA, nao ignorada.

Tres paineis:
  A) o gap em log-log. Subtrair o piso e' o que endireita a lei; sem isso a
     curvatura do piso entra no expoente (medido: beta efetivo varia 2,67x).
  B) o custo de afinar: tokens necessarios para chegar a um gap alvo, com a
     nossa posicao marcada. A inclinacao e' 1/beta ~ 2.
  C) a barra honesta: quanto a incerteza do piso move a resposta.

Uso:
  uv run scripts/gap_law.py --out /tmp/gap_law.png
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

CURVE = "/tmp/scal_curve.npy"
TOL = 0.02          # o criterio do lab: erro maximo de 2% (0,02 bpb)
E_LO, E_HI = 1.70, 1.95


def lab_fit(D, L):
    """A UNICA implementacao de ajuste: o fit() do proprio scaling_plot.py."""
    spec = importlib.util.spec_from_file_location("sp", "scripts/scaling_plot.py")
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    return sp.fit(D, L)


def fit_at(D, L, E):
    """Ajusta beta e A com o piso FIXO em E. Forma do fit() do lab."""
    if np.any(L - E <= 0):
        return None
    sl, ic = np.polyfit(np.log(D), np.log(L - E), 1)
    A = float(np.exp(ic))
    beta = -float(sl)
    err = float(np.max(np.abs(E + A*D**-beta - L)))
    return beta, A, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/gap_law.png")
    a = ap.parse_args()

    arr = np.load(CURVE)
    D, L = arr[:, 0], arr[:, 1]

    # ---- a faixa admissivel
    band = []
    for E in np.arange(E_LO, E_HI, 0.001):
        g = fit_at(D, L, E)
        if g is not None and g[2] <= TOL:
            band.append((E, g[0], g[1], g[2]))
    if not band:
        print("sem ajuste admissivel", file=sys.stderr)
        return 1
    Es = np.array([b[0] for b in band])
    bs = np.array([b[1] for b in band])
    As = np.array([b[2] for b in band])
    E0, b0, A0 = Es.mean(), bs.mean(), As.mean()

    print("=" * 74)
    print("A LEI DO GAP  G(D) = L(D) - E = A * D^-beta")
    print("=" * 74)
    print(f"  faixa admissivel (erro <= {TOL:.2f}): {len(band)} pisos")
    print(f"  E    = {E0:.4f} +/- {(Es.max()-Es.min())/2:.4f} bpb")
    print(f"  beta = {b0:.4f} +/- {(bs.max()-bs.min())/2:.4f}")
    print(f"  A    = {A0:.1f}")
    print(f"  1/beta = {1/b0:.2f}  <- o expoente do custo de afinar")
    print()
    print(f"  {'D (tokens)':>12} {'gap (bpb)':>10} {'bpb':>8}")
    for Dv in [1e6, 8e6, 3.36e7, 6.5e7, 1e9, 1e10]:
        g = A0*Dv**-b0
        print(f"  {Dv:>12.3g} {g:>10.3f} {E0+g:>8.3f}")
    print()
    print("  custo de afinar (tokens para chegar a um gap alvo):")
    print(f"  {'gap alvo':>9} {'tokens':>12} {'x vs 65M':>10}")
    for g in [1.0, 0.5, 0.1, 0.01, 0.001]:
        Dn = (A0/g)**(1/b0)
        print(f"  {g:>9.3f} {Dn:>12.3g} {Dn/6.5e7:>10.1f}")
    print()
    lo = (A0/0.1)**(1/bs.max()); hi = (A0/0.1)**(1/bs.min())
    print(f"  e a barra honesta: a incerteza do piso move o alvo de 0,1 bpb")
    print(f"  de {lo:.3g} a {hi:.3g} tokens, ou seja {hi/lo:.2f}x.")

    # ---- figura
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(sem matplotlib: os numeros acima sao o resultado)")
        return 0

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    Dg = np.logspace(np.log10(D.min()), 11, 200)

    # A) o gap em log-log: subtrair o piso endireita a lei
    for E, b, A, _ in band:
        ax[0].plot(Dg, A*Dg**-b, "-", color="steelblue", alpha=0.3, lw=0.8)
    ax[0].plot(D, L - E0, "o", ms=4, color="black",
               label=f"medido - E ({E0:.3f})")
    ax[0].plot([], [], "-", color="steelblue", lw=0.8,
               label=f"{len(band)} ajustes admissiveis")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("tokens"); ax[0].set_ylabel("gap = L - E  (bpb)")
    ax[0].set_title(f"A) o gap e' reto em log-log: beta = {b0:.3f}")
    ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=7)

    # B) o custo de afinar
    gaps = np.logspace(-3, 0.3, 200)
    ax[1].plot(gaps, (A0/gaps)**(1/b0), "-", color="crimson", lw=1.6)
    ax[1].plot([0.105], [6.5e7], "o", ms=9, color="black")
    ax[1].annotate("estamos aqui\n65M, gap 0,105", (0.105, 6.5e7),
                   textcoords="offset points", xytext=(-70, 14), fontsize=8)
    for g, lab in [(0.1, "0,1"), (0.01, "0,01"), (0.001, "0,001")]:
        Dn = (A0/g)**(1/b0)
        ax[1].plot([g], [Dn], "s", ms=6, color="crimson")
        ax[1].annotate(f"gap {lab}\n{Dn:.1e} tok", (g, Dn),
                       textcoords="offset points", xytext=(8, -4), fontsize=7)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("gap alvo (bpb)"); ax[1].set_ylabel("tokens necessarios")
    ax[1].set_title(f"B) custo de afinar: inclinacao 1/beta = {1/b0:.2f}")
    ax[1].grid(alpha=0.3, which="both")

    # C) a barra honesta
    # A incerteza vem DO piso: cada E da faixa determina um beta, e portanto uma
    # resposta. Entao o honesto e' a linha (a resposta em funcao de E) mais os
    # dois extremos marcados -- nao uma faixa plana, que sugeriria que qualquer
    # valor serve para qualquer E.
    ax[2].plot(Es, (A0/0.1)**(1/bs), "-", color="darkgreen")
    ax[2].plot([Es.min(), Es.max()], [hi, lo], "o", ms=8, color="darkgreen")
    ax[2].annotate(f"{hi:.1e}", (Es.min(), hi), textcoords="offset points",
                   xytext=(6, 4), fontsize=7)
    ax[2].annotate(f"{lo:.1e}", (Es.max(), lo), textcoords="offset points",
                   xytext=(-40, -2), fontsize=7)
    ax[2].axvline(E0, color="black", ls=":", lw=1)
    ax[2].set_yscale("log")
    ax[2].set_xlabel("piso assumido E (bpb)")
    ax[2].set_ylabel("tokens para gap 0,1")
    ax[2].set_title(f"C) a barra honesta: {hi/lo:.2f}x vinda so' do piso")
    ax[2].grid(alpha=0.3, which="both")

    fig.suptitle("A lei do gap: quao rapido fechamos a distancia ate a saturacao. "
                 f"beta = {b0:.3f}, e o custo cresce como (1/gap)^{1/b0:.2f}.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(a.out, dpi=110)
    print(f"\nfigura: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
