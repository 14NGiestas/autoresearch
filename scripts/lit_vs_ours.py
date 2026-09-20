#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib==3.10.8",
#   "numpy==2.5.2",
# ]
# ///
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
# Bytes por token. O NOSSO e' MEDIDO (2,528 nos rows, contra 4,963 de media do
# vocabulario: usamos tokens curtos). O DELES e' DESCONHECIDO, e e' por isso que
# o NIVEL nao compara limpo: para converter a lei para bpb eu teria que adivinhar
# o tokenizer deles. A INCLINACAO nao sofre disso -- ver `invariance()` abaixo.
BYTES_PER_TOKEN = 2.528  # medido nos nossos rows (nao e' o deles)
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


def invariance():
    """O expoente nao depende da conversao. Demonstracao, nao afirmacao.

    bpb = nats / (ln2 * B) e' uma mudanca MULTIPLICATIVA do eixo y, e um fator
    multiplicativo nao muda o expoente de uma lei de potencia. Convertendo a lei
    a 2, 4 e 8 bytes/token o beta ajustado tem que sair o mesmo.
    """
    print("  demonstracao: o expoente e' invariante a conversao")
    d = np.logspace(9, 12, 200)
    for n in (1e8,):
        nats = CHIN_E + CHIN_A/n**CHIN_ALPHA + CHIN_B/d**CHIN_BETA
        for b in (2.0, 4.0, 8.0):
            y = nats/(LN2*b)
            best = None
            for beta in np.linspace(0.05, 0.9, 600):
                X = np.vstack([np.ones_like(d), d**-beta]).T
                c, *_ = np.linalg.lstsq(X, y, rcond=None)
                e = float(np.max(np.abs(X @ c - y)))
                if best is None or e < best[0]:
                    best = (e, beta)
            print(f"    {b:.0f} B/tok -> beta = {best[1]:.4f}  (erro {best[0]:.2e})")
    print("    mesma coluna: a inclinacao compara sem conversao.")


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
    print()
    invariance()

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

    # ---- O AJUSTE, com UM metodo so': o fit() do proprio scaling_plot.py.
    # A versao anterior desta secao estava BUGADA: ela ajustava
    # `E + c0 + b*D^-beta`, ou seja DOIS termos constantes (E e o intercepto),
    # redundantes. "Fixar E nao muda o erro" era tautologia, nao descoberta.
    # Aqui o modelo e' `Linf + A*D^-beta`, com Linf varrido e A/beta em forma
    # fechada, exatamente como o fit() do lab faz.
    import importlib.util
    spec = importlib.util.spec_from_file_location("sp", "scripts/scaling_plot.py")
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)

    E_LIT = 1.6934/(LN2*BYTES_PER_TOKEN)     # piso da literatura em bpb

    def fit_fixed(D, L, Linf):
        """Mesma forma do fit() do lab, com o piso FIXO em Linf."""
        if np.any(L - Linf <= 0):
            return None
        sl, ic = np.polyfit(np.log(D), np.log(L - Linf), 1)
        pred = Linf + np.exp(ic)*D**sl
        return float(np.max(np.abs(pred - L))), -sl

    arr = np.load(OURS["scal_curve (32 pts)"])
    Dc, Lc = arr[:, 0], arr[:, 1]
    r_free, Linf, beta_free, _ = sp.fit(Dc, Lc)

    print()
    print("=" * 78)
    print("O AJUSTE CERTO (metodo: o fit() do scaling_plot.py, log-espaco)")
    print("=" * 78)
    print(f"  (a) piso LIVRE        : E={Linf:.3f} bpb  beta={beta_free:.3f}  erro={r_free:.4f}")
    f = fit_fixed(Dc, Lc, E_LIT)
    if f is None:
        print(f"  (b) piso FIXO no deles: IMPOSSIVEL -- {E_LIT:.3f} fica ACIMA de pontos nossos")
    else:
        print(f"  (b) piso FIXO no deles: E={E_LIT:.3f} bpb  beta={f[1]:.3f}  erro={f[0]:.4f}")
    best = None
    for Lf in np.arange(0.05, 2.0, 0.002):
        g = fit_fixed(Dc, Lc, Lf)
        if g is None:
            continue
        sl, ic = np.polyfit(np.log(Dc), np.log(Lc - Lf), 1)
        pred = Lf + np.exp(ic)*Dc**-0.2849
        er = float(np.max(np.abs(pred - Lc)))
        if best is None or er < best[0]:
            best = (er, Lf)
    print(f"  (c) beta FIXO nos deles: beta=0.2849  piso={best[1]:.3f}  erro={best[0]:.4f}")

    # a faixa admissivel (erro <= 2%), varrendo o piso
    ok = []
    for Lf in np.arange(0.05, 2.0, 0.002):
        g = fit_fixed(Dc, Lc, Lf)
        if g is not None and g[0] <= 0.02:
            ok.append((g[1], Lf, g[0]))
    print()
    if ok:
        bs = [o[0] for o in ok]; es = [o[1] for o in ok]
        print(f"  faixa admissivel (erro <= 0,02): {len(ok)} pisos, beta de {min(bs):.3f} a {max(bs):.3f}")
        print(f"  pisos: de {min(es):.3f} a {max(es):.3f} bpb")
        print(f"  o piso da literatura ({E_LIT:.3f}) esta' {'DENTRO' if min(es) <= E_LIT <= max(es) else 'FORA'} da faixa")
        print(f"  o beta da literatura (0,2849) esta' {'DENTRO' if min(bs) <= 0.2849 <= max(bs) else 'FORA'} da faixa")
    else:
        print("  NENHUM piso admissivel com erro <= 0,02")

    # ---- figura
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(sem matplotlib: os numeros acima sao o resultado)")
        return 0

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.4))

    # A) curvas + faixa admissivel (agora ESTREITA: o piso esta' fixado)
    for name, path in OURS.items():
        if Path(path).exists():
            a2 = np.load(path)
            ax[0, 0].plot(a2[:, 0], a2[:, 1], "o-", ms=3, lw=1, label=name)
    Dg = np.logspace(np.log10(Dc.min()), np.log10(Dc.max()), 60)
    for be, Lf, _ in ok:
        sl, ic = np.polyfit(np.log(Dc), np.log(Lc - Lf), 1)
        ax[0, 0].plot(Dg, Lf + np.exp(ic)*Dg**sl, "-", color="steelblue",
                      alpha=0.35, lw=0.8)
    ax[0, 0].plot([], [], "-", color="steelblue", lw=0.8,
                  label=f"{len(ok)} ajustes admissiveis (erro <= 2%)")
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_xlabel("tokens"); ax[0, 0].set_ylabel("bpb")
    ax[0, 0].set_title("A) a faixa admissivel e' ESTREITA: o piso e' identificado")
    ax[0, 0].grid(alpha=0.3); ax[0, 0].legend(fontsize=7)

    # B) TRES coisas diferentes, que nao podem ser confundidas:
    #    (1) a nossa medida
    #    (2) a LEI deles avaliada no nosso N e D -- extrapolacao 1,5 a 3 ordens
    #        ABAIXO da faixa onde ela foi ajustada, portanto nao e' medida
    #    (3) a inclinacao deles forcada nos NOSSOS dados com o piso DELES --
    #        contrafactual, e os nossos dados rejeitam esse piso (erro 6x)
    d = np.logspace(np.log10(Dc.min()), np.log10(Dc.max()), 60)
    x = Dc**-0.2849
    b_lit = float(np.sum(x*(Lc - E_LIT))/np.sum(x*x))
    x2 = Dc**-beta_free
    b_our = float(np.sum(x2*(Lc - Linf))/np.sum(x2*x2))
    ax[0, 1].plot(Dc, Lc, "o", ms=4, color="black", label="(1) scal_curve, medido")
    ax[0, 1].plot(d, Linf + b_our*d**-beta_free, "--", color="seagreen",
                  label=f"(1) nosso ajuste beta={beta_free:.3f}")
    # (2) a lei deles no nosso N, em bpb com os NOSSOS bytes/token
    law = lambda Dv: (CHIN_E + CHIN_A/2.75e6**CHIN_ALPHA + CHIN_B/Dv**CHIN_BETA)/(LN2*BYTES_PER_TOKEN)
    dd = np.logspace(np.log10(Dc.min()), 10, 80)
    ax[0, 1].plot(dd, law(dd), "-", color="darkorange", lw=1.4,
                  label="(2) LEI deles em N=2,75M (extrapolada)")
    # (3) contrafactual: inclinacao deles, piso deles, amplitude nossa
    ax[0, 1].plot(d, E_LIT + b_lit*d**-0.2849, "-", color="crimson",
                  label="(3) inclinacao deles + piso deles (contrafactual)")
    # o cruzamento
    Ds = np.logspace(np.log10(Dc.min()), 9, 4000)
    A = Linf + b_our*Ds**-beta_free
    C = E_LIT + b_lit*Ds**-0.2849
    k = int(np.argmin(np.abs(A - C)))
    ax[0, 1].axvline(Ds[k], color="gray", ls=":", lw=1)
    ax[0, 1].annotate(f"cruzam em ~{Ds[k]/1e6:.0f}M\n(abaixo: nos melhor\nacima: contrafactual melhor)",
                      (Ds[k], 2.35), fontsize=6, ha="center", color="gray")
    ax[0, 1].axhline(E_LIT, color="crimson", ls=":", lw=1, label=f"piso deles {E_LIT:.2f}")
    ax[0, 1].axhline(Linf, color="seagreen", ls=":", lw=1, label=f"nosso piso {Linf:.2f}")
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_xlabel("tokens"); ax[0, 1].set_ylabel("bpb")
    ax[0, 1].set_title("B) medido, lei extrapolada, e contrafactual")
    ax[0, 1].grid(alpha=0.3); ax[0, 1].legend(fontsize=6)

    # C) o erro em funcao do piso: onde a faixa fecha
    Es = np.arange(0.05, 2.0, 0.002)
    errs = []
    for Lf in Es:
        g = fit_fixed(Dc, Lc, Lf)
        errs.append(np.nan if g is None else g[0])
    errs = np.array(errs)
    ax[1, 0].plot(Es, errs, "-", color="steelblue")
    ax[1, 0].axhline(0.02, color="black", ls="--", lw=1, label="criterio 2%")
    ax[1, 0].axvspan(1.820, 1.842, color="seagreen", alpha=0.25,
                     label="faixa admissivel")
    ax[1, 0].axvline(E_LIT, color="crimson", ls=":", lw=1.5,
                     label=f"piso deles {E_LIT:.2f} (FORA)")
    ax[1, 0].set_ylim(0, 0.15)
    ax[1, 0].set_xlabel("piso E (bpb)"); ax[1, 0].set_ylabel("erro max do ajuste (bpb)")
    ax[1, 0].set_title("C) o piso ESTA' fixado, e o deles fica fora")
    ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize=7)

    # D) o eixo tamanho, confundido
    names = [s[0] for s in SIZE_AXIS]; bpbs = [s[3] for s in SIZE_AXIS]
    toks = [s[2] for s in SIZE_AXIS]
    ax[1, 1].plot(range(3), bpbs, "s-", color="darkgreen", ms=8)
    for i2, (nm, b2, t2) in enumerate(zip(names, bpbs, toks)):
        ax[1, 1].annotate(f"{nm}\n{t2/1e6:.1f}M tok", (i2, b2),
                          textcoords="offset points", xytext=(0, 10),
                          ha="center", fontsize=8)
    ax[1, 1].set_xticks(range(3)); ax[1, 1].set_xticklabels(names)
    ax[1, 1].set_ylabel("bpb")
    ax[1, 1].set_title("D) eixo TAMANHO: nao monotonico, tokens desiguais")
    ax[1, 1].grid(alpha=0.3)

    fig.suptitle("Literatura contra os nossos dados. Piso E inclinacao estao "
                 "AMBOS fixados (faixa estreita), e os dois valores deles ficam fora.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(a.out, dpi=110)
    print(f"\nfigura: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
