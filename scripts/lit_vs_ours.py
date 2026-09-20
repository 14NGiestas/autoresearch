#!/usr/bin/env python3
"""Literatura contra os nossos dados: onde a comparacao e' legitima e onde nao e'.

TRES AVISOS, e sem eles o grafico mente:

1. UNIDADE. A lei da Chinchilla da' nats por TOKEN. Nos medimos bits por BYTE.
   Para comparar e' preciso dividir por ln(2) e pelos BYTES POR TOKEN do
   tokenizer deles. O fator de 4 bytes/token e' uma ESTIMATIVA, e esta' dito no
   grafico. Sem essa conversao a diferenca aparece como 5x e nao e' real.

2. REGIME. Todo ponto publicado esta' acima de 70M parametros e 5B tokens. Nos
   estamos em 2,75M a 25M parametros e 1M a 65M tokens, ou seja 1,5 a 3 ordens
   de grandeza ABAIXO do menor ponto publicado. A lei nao foi ajustada aqui: o
   que o grafico mostra e' uma EXTRAPOLACAO dela para baixo, nao uma medida.

3. COEFICIENTES. Os da Chinchilla vem do paper (Hoffmann et al. 2022, Approach
   3), citados na replicacao arXiv:2404.10102. A replicacao da' B com erro de
   300% (410,7 +/- 1293), entao o termo de DADOS e' mal identificado: a parte da
   curva que depende de D e' a menos confiavel das duas.

Fontes (nenhum numero vem de memoria):
  Chinchilla  L(N,D) = 1.6934 + 406.4/N^0.3392 + 410.7/D^0.2849   arXiv:2203.15556
  Kaplan      L(N)   = (8.8e13/N)^0.076                            arXiv:2001.08361
  replicacao  arXiv:2404.10102     reconciliacao  arXiv:2406.12907

Uso (matplotlib nao esta no venv; uv empresta sem instalar nada):
  uv run --with matplotlib --with numpy python scripts/lit_vs_ours.py --out /tmp/lit_vs_ours.png
Os numeros saem sem matplotlib tambem (o script cai para texto):
  .venv-numpy/bin/python3 scripts/lit_vs_ours.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# ---- literatura, com a fonte em cada linha (nada de memoria)
CHIN_E, CHIN_A, CHIN_ALPHA = 1.6934, 406.4, 0.3392      # arXiv:2203.15556
CHIN_B, CHIN_BETA = 410.7, 0.2849                       # arXiv:2203.15556
KAP_NC, KAP_ALPHA_N = 8.8e13, 0.076                     # arXiv:2001.08361
BYTES_PER_TOKEN = 4.0    # ESTIMATIVA do tokenizer deles; declarada no grafico
LN2 = np.log(2.0)

# ---- os nossos dados (d96, tres corridas distintas)
OURS = {
    "scal_curve (32 pts)": "/tmp/scal_curve.npy",
    "scal65_curve (30 pts)": "/tmp/scal65_curve.npy",
    "scal65_1k_curve (63 pts)": "/tmp/scal65_1k_curve.npy",
}
# o eixo TAMANHO, confundido com tokens (avaliado na cadeia canonica)
SIZE_AXIS = [("d96", 2.75e6, 8.0e6, 4.542), ("d216", 9.3e6, 8.0e6, 4.496),
             ("d360", 2.5e7, 2.46e7, 4.724)]


def chinchilla_bpb(n, d):
    """A lei da Chinchilla em bits por byte, com a conversao declarada."""
    nats = CHIN_E + CHIN_A / n**CHIN_ALPHA + CHIN_B / d**CHIN_BETA
    return nats / LN2 / BYTES_PER_TOKEN


def kaplan_bpb(n, d):
    """Kaplan: so' a parte de N (ele separa N e D). Em bpb, mesma conversao."""
    nats = (KAP_NC / n)**KAP_ALPHA_N
    return nats / LN2 / BYTES_PER_TOKEN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/lit_vs_ours.svg")
    a = ap.parse_args()

    print("=" * 78)
    print("O QUE A LITERATURA PREVE NO NOSSO TAMANHO (extrapolacao para baixo)")
    print("=" * 78)
    print("  a lei foi ajustada em 70M+ params e 5B+ tokens; aqui ela e' usada")
    print("  ABAIXO do menor ponto publicado, e em bpb com %.0f bytes/token."
          % BYTES_PER_TOKEN)
    print()
    print(f"  {'N (params)':>12} {'D (tokens)':>12} {'Chinchilla':>12} {'Kaplan':>10} "
          f"{'nos medimos':>12}")
    for n, d, our in [(2.75e6, 8.0e6, 2.44), (9.3e6, 8.0e6, None),
                      (2.5e7, 2.46e7, None), (1e8, 2e9, None), (1e9, 2e10, None)]:
        c = chinchilla_bpb(n, d)
        k = kaplan_bpb(n, d)
        o = f"{our:.2f}" if our else "-"
        print(f"  {n:>12.2e} {d:>12.2e} {c:>12.2f} {k:>10.2f} {o:>12}")
    print()
    print("  Leitura: na nossa escala a extrapolacao fica ACIMA do que medimos")
    print("  (ela preve pior do que conseguimos). Isso nao e' vitoria: a lei nao")
    print("  foi ajustada aqui, e o nosso texto nao e' o do Pile. O que e'")
    print("  comparavel entre nos e a lei e' a INCLINACAO em tokens, nao o nivel.")

    # ---- o ajuste da nossa propria curva (para comparar inclinacoes)
    print()
    print("=" * 78)
    print("A INCLINACAO: o que da' para comparar de verdade")
    print("=" * 78)
    print("  ajuste L = a + b*D^-beta na nossa curva, e o beta equivalente da lei")
    print("  (beta = %.4f) para o termo de tokens." % CHIN_BETA)
    fits = {}
    for name, path in OURS.items():
        p = Path(path)
        if not p.exists():
            continue
        arr = np.load(p)
        d, l = arr[:, 0], arr[:, 1]
        ok = np.isfinite(d) & np.isfinite(l) & (d > 0)
        d, l = d[ok], l[ok]
        if len(d) < 4:
            continue
        # grade de beta: pega o melhor por erro, como o scal_curve ja' faz
        best = None
        for beta in np.linspace(0.05, 0.9, 400):
            x = d**-beta
            A = np.vstack([np.ones_like(x), x]).T
            coef, res, *_ = np.linalg.lstsq(A, l, rcond=None)
            pred = A @ coef
            err = float(np.max(np.abs(pred - l)))
            if best is None or err < best[0]:
                best = (err, beta, coef[0], coef[1])
        fits[name] = best
        print(f"  {name:<26} beta = {best[1]:.3f}  erro max {best[0]:.4f} bpb")
    print()
    print("  Compare com o beta da Chinchilla (0,2849, com erro 0,02): se o nosso")
    print("  beta cai fora, ou a nossa curva nao e' lei de potencia nessa faixa, ou")
    print("  o regime e' outro. Nos dois casos o grafico nao pode afirmar nada.")

    # ---- figura
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(sem matplotlib: os numeros acima sao o resultado)")
        return 0

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # painel A: as nossas curvas
    for name, path in OURS.items():
        p = Path(path)
        if not p.exists():
            continue
        arr = np.load(p)
        ax[0].plot(arr[:, 0], arr[:, 1], "o-", ms=3, lw=1, label=name)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("tokens")
    ax[0].set_ylabel("bpb")
    ax[0].set_title("A) os nossos dados (d96, eixo TOKENS)")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7)

    # painel B: literatura contra nos
    d = np.logspace(6, 10.5, 200)
    for n, style in [(2.75e6, "-"), (1e8, "--"), (1e9, ":")]:
        ax[1].plot(d, chinchilla_bpb(n, d), style, color="crimson",
                   label=f"Chinchilla N={n:.2g}")
    ax[1].plot(d, kaplan_bpb(2.75e6, d) * np.ones_like(d), "-.", color="navy",
               label="Kaplan N=2.75M")
    for name, path in OURS.items():
        if Path(path).exists():
            arr = np.load(path)
            ax[1].plot(arr[:, 0], arr[:, 1], "o-", ms=3, lw=1, label=name)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("tokens")
    ax[1].set_ylabel("bpb")
    ax[1].set_title("B) lei publicada vs nos\n(lei: nats/token -> bpb, %.0f B/tok)"
                    % BYTES_PER_TOKEN)
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=6)

    # painel C: o eixo tamanho, confundido
    names = [s[0] for s in SIZE_AXIS]
    bpbs = [s[3] for s in SIZE_AXIS]
    toks = [s[2] for s in SIZE_AXIS]
    ax[2].plot(range(3), bpbs, "s-", color="darkgreen", ms=8)
    for i, (nm, b, t) in enumerate(zip(names, bpbs, toks)):
        ax[2].annotate(f"{nm}\n{t/1e6:.1f}M tok", (i, b),
                       textcoords="offset points", xytext=(0, 10),
                       ha="center", fontsize=8)
    ax[2].set_xticks(range(3))
    ax[2].set_xticklabels(names)
    ax[2].set_ylabel("bpb")
    ax[2].set_title("C) eixo TAMANHO: nao monotonico\n(e os tokens sao desiguais)")
    ax[2].grid(alpha=0.3)

    fig.suptitle("Literatura contra os nossos dados. Nivel nao e' comparavel "
                 "(corpus e tokenizer diferentes); inclinacao em tokens e'.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(a.out, dpi=110)
    print(f"\nfigura: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
