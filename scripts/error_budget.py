#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Orcamento de erro: quanto nos sabemos, e o que domina.

Uma medida sem barra de erro nao e' um fato, e' uma opiniao com digitos. Este
script junta as fontes de incerteza que ja' foram medidas e as ordena por
tamanho. Ele separa duas coisas que nao se comportam igual:

  VIES   -- constante entre runs que usam o mesmo caminho. Cancela numa
            comparacao, nao cancela numa afirmacao absoluta.
  RUIDO  -- muda de run para run. Nao cancela em nada.

O que ele calcula dos historicos (nada digitado):
  * o nivel de bpb de cada sorteio de dados
  * o ganho pareado de cada sorteio (composicao contra maquina unica + outer)

O que ele so' reporta, com a proveniencia (medido nesta sessao):
  * o vies do fast_tanh, o acordo entre kernels, a disputa por threads

Uso:  .venv-numpy/bin/python3 scripts/error_budget.py
"""
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------- os historicos
# Dois sorteios de dados, cada um com a comparacao pareada que interessa:
# a receita composta contra a maquina unica com o MESMO otimizador outer.
DRAWS = {
    "f0": {
        "composed": "/tmp/dl/122/history.json",
        "single_outer": "/tmp/ctrl/k1outer_f0/history.json",
        "pool": "/tmp/mix/rows_f0.npy",
    },
    "perm1": {
        "composed": "/tmp/ctrl2/B_perm1/history.json",
        "single_outer": "/tmp/ctrl2/k1outer_perm1/history.json",
        "pool": "/tmp/mix/rows_perm1.npy",
    },
}

# Fontes que nao saem de historico. Cada uma foi medida, e o metodo esta' dito.
# magnitude em bpb, ou em fator quando e' velocidade.
MEASURED = [
    ("fast_tanh: polinomio contra tanh exato", 3.0e-4, "bpb", "vies",
     "A/B com dois binarios, cap 2 e cap 30 (job desta sessao)"),
    ("kernel de atencao: naive contra BLAS", 1.0e-6, "bpb", "vies",
     "test_attn_sgemm, mesmo resultado a ~1e-6"),
    ("soft cap ligado, modelo treinado sem cap", 1.2, "bpb", "desenho",
     "nao e' erro: e' uma escolha. cap 2 custa 0,86 nll = 1,2 bpb"),
    ("disputa por threads: OMP=16 contra OMP=8", 2.87, "x", "erro de metodo",
     "853 contra 297 tok/s; corrompeu 3 medidas antes de ser pego"),
    ("passo do modelo: parada do JSON", 1.0e-5, "bpb", "erro de metodo",
     "o parse pegava NLL por token; corrigido com o bpb_of canonico"),
]


def final_bpb(path):
    """O bpb do ultimo round de um history.json."""
    h = json.loads(Path(path).read_text())
    hist = h["history"] if isinstance(h, dict) else h
    vals = [r["bpb"] for r in hist if "bpb" in r]
    return vals[-1] if vals else None


def main():
    rows = []
    for name, d in DRAWS.items():
        c = final_bpb(d["composed"])
        s = final_bpb(d["single_outer"])
        if c is None or s is None:
            print(f"# sorteio {name}: historico ausente, pulado", file=sys.stderr)
            continue
        rows.append((name, c, s, s - c))

    if len(rows) < 2:
        print("preciso de dois sorteios para medir a dispersao", file=sys.stderr)
        return 1

    levels = [r[1] for r in rows]
    gains = [r[3] for r in rows]
    lvl_mean = sum(levels) / len(levels)
    gain_mean = sum(gains) / len(gains)
    lvl_spread = max(levels) - min(levels)
    gain_spread = max(gains) - min(gains)

    print("=" * 78)
    print("NIVEL ABSOLUTO (a composicao final, por sorteio de dados)")
    print("=" * 78)
    print(f"{'sorteio':>10}  {'bpb composto':>13}  {'bpb unico+outer':>16}  {'ganho':>8}")
    for name, c, s, g in rows:
        print(f"{name:>10}  {c:>13.5f}  {s:>16.5f}  {g:>+8.5f}")
    print(f"{'media':>10}  {lvl_mean:>13.5f}  {'':>16}  {gain_mean:>+8.5f}")
    print()
    print(f"  dispersao do NIVEL entre sorteios: {lvl_spread:.5f} bpb")
    print(f"  dispersao do GANHO entre sorteios: {gain_spread:.5f} bpb")
    print()
    print("  Leitura: o nivel tem incerteza de sorteio grande; o ganho, pequena.")
    print("  A comparacao e' pareada (mesmo pool nos dois lados), entao o que")
    print("  muda entre sorteios cancela no ganho. E' por isso que o ganho e' a")
    print("  afirmacao solida, e o nivel nao.")

    print()
    print("=" * 78)
    print("FONTES DE INCERTEZA, ordenadas por tamanho")
    print("=" * 78)
    print(f"{'fonte':<46} {'tamanho':>9}  {'unid':<5} {'tipo':<14}")
    print("-" * 78)
    # inclui as duas medidas calculadas acima
    allsrc = list(MEASURED) + [
        ("sorteio de dados: nivel absoluto", lvl_spread, "bpb", "ruido",
         "dois sorteios, o mesmo modelo e a mesma receita"),
        ("sorteio de dados: ganho pareado", gain_spread, "bpb", "ruido",
         "os dois sorteios, cada um pareado no mesmo pool"),
    ]
    for name, mag, unit, kind, prov in sorted(allsrc, key=lambda x: -x[1]):
        print(f"{name:<46} {mag:>9.2g}  {unit:<5} {kind:<14}")
    print()
    print("Proveniencia de cada linha:")
    for name, mag, unit, kind, prov in sorted(allsrc, key=lambda x: -x[1]):
        print(f"  {name}: {prov}")

    print()
    print("=" * 78)
    print("O QUE ISSO IMPLICA PARA AS AFIRMACOES")
    print("=" * 78)
    claims = [
        ("a composicao bate a maquina unica com o mesmo outer",
         f"+{gain_mean:.3f} bpb", f"+/- {gain_spread:.3f} (sorteio)",
         "SOLIDA: o sinal replica nos dois sorteios"),
        ("o bpb absoluto do melhor modelo",
         f"{lvl_mean:.3f}", f"+/- {lvl_spread:.3f} (sorteio)",
         "FRACA: o sorteio move mais que o 4o decimal"),
        ("head_dim maior deixa o passo mais rapido",
         "1,87x e 2,67x", "medido com o node sob guarda",
         "SOLIDA no passo, mas nao no bpb ainda"),
        ("1B em 6 meses, pelo estimador",
         "709 dias", "MODELO: P ~ sqrt(threads x tempo)",
         "NAO E' MEDIDA: e' extrapolacao. Se a escala for linear, e' 8x pior"),
        ("o fast_tanh move o bpb",
         "3e-4", "medido A/B",
         "IRRELEVANTE para comparar, real para afirmar em absoluto"),
    ]
    for what, val, unc, verdict in claims:
        print(f"  {what}")
        print(f"      valor {val}   incerteza {unc}")
        print(f"      {verdict}")
        print()

    print("A licao: a incerteza dominante das medidas de qualidade e' o SORTEIO")
    print("DE DADOS, nao o ruido do instrumento. E a incerteza dominante do")
    print("numero grande (1B em 6 meses) nao e' medida nenhuma: e' a forma da")
    print("lei de escala que o estimador assume.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
