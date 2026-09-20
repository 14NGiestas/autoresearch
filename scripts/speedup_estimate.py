#!/usr/bin/env python3
"""speedup_estimate.py — quanto falta para 1B em 6 meses, e como isso mudou.

A pergunta: "uma estimativa do speedup desde que começamos, para 1B em 6 meses ou
menos, para sabermos como nossas estimativas estão mudando ao longo do tempo e
quão eficiente está nosso treinamento".

Como a conta e' feita (e o que e' MEDIDO vs o que e' HIPOTESE):

  FLOPs/s por no'   MEDIDO. Agregado dos runs K=4 (4 workers, ~2 threads cada):
                    1823 tok/s por worker x 11.8 MFLOP/token (fwd+bwd do d96)
                    = ~172 GFLOP/s por no' de 8 cores. Comparacao: o microbench
                    de GEMM puro da 517 GFLOP/s, entao ~33% da eficiencia de pico
                    de GEMM e' o teto realistico do passo de treino.
  custo por token   HIPOTESE explicita: 6 x params (fwd 2P + bwd 4P). Para 1B:
                    6 GFLOP/token.
  tokens necessarios' HIPOTESE: Chinchilla 20 x params = 20B tokens (e o caso
                    "overtrained" de 100B e' mostrado tambem).
  fatores achados   MEDIDOS nesta linha (cada um com a fonte):
                    execucao: 4 workers x OMP~2 agregam 8.5x mais que 1 worker
                      com OMP=8 (7292 vs 853 tok/s) e OMP=16 e' 2.3x PIOR que 8;
                    arquitetura: head_dim 32/48 = 1.80x / 2.44x no passo (mesmos
                      FLOPs e params);
                    composicao: 0.165-0.172 bpb a MENOS no mesmo wall-clock =
                      ~1.9x menos tokens para o mesmo loss (via L ~ T^-0.1).
                    lote (B): 1.0x -- medido PLANO, nao ajuda.

Uso: scripts/speedup_estimate.py [--params 1e9] [--tokens-per-param 20]
                                 [--months 6] [--svg FILE]
"""

import argparse
import os
import sys

# ---------------------------------------------------------------- MEDIDO
MEASURED = {
    "tok_s_1worker_omp8": 853.0,     # archhead/sweep: 1024 tokens / 1.20 s
    "tok_s_1worker_omp16": 297.0,    # 1024/3.449: 16 threads e' 2.3x PIOR
    "tok_s_4workers_omp2": 1823.0,   # log do worker do driver (agregado x4)
    "flops_per_token_d96": 11.8e6,   # 6 x 2.75M params
    "gemm_microbench_gflops": 517.0, # bench_gemm, sgemm
    "arch_factor_d32": 1.80,         # archhead: 1.201 -> 0.6678 s/passo
    "arch_factor_d48": 2.44,         # archhead: 1.201 -> 0.4924 s/passo
    "batch_factor": 1.00,            # sweep: throughput PLANO em B
    "comp_bpb_gain": 0.168,          # media de 0.165 (f0) e 0.172 (perm1)
    "scaling_alpha": 0.10,           # L ~ T^-alpha, do nosso ajuste grosso
}

# Linha do tempo das descobertas: (quando, rotulo, multiplicador de eficiencia)
# O multiplicador e' o ganho de FLOPs/s-equivalentes (ou de tokens-equivalentes,
# no caso da composicao) em relacao ao inicio.
TIMELINE = [
    ("inicio",              "1 worker, OMP=8, head_dim 16, sem receita", 1.0),
    ("+ execucao paralela", "4 workers x OMP~2 (8.5x agregado) vs OMP=16 (2.3x pior)",
     MEASURED["tok_s_4workers_omp2"] / MEASURED["tok_s_1worker_omp16"]),
    ("+ B (lote)",          "medido plano: NAO ajuda", MEASURED["batch_factor"]),
    ("+ head_dim 32",       "1.80x no passo, mesmos FLOPs/params", MEASURED["arch_factor_d32"]),
    ("+ head_dim 48",       "2.44x no passo", MEASURED["arch_factor_d48"]),
]


def comp_token_factor(gain, alpha):
    """Composicao da' menos loss pelo mesmo wall-clock -> menos tokens equivalentes.
    L2/L1 = (T2/T1)^-alpha  =>  T2/T1 = (L1/L2)^(1/alpha)."""
    l_ratio = 1.0 - gain / 2.4   # bpb ~ 2.4 no nosso regime; gain em bpb absoluto
    return l_ratio ** (-1.0 / alpha)


def estimate(params, tok_per_param, months, factors):
    """Tempo (anos) para treinar esse modelo num no', e nos necessarios."""
    flops_per_token = 6.0 * params
    tok_s_per_node = MEASURED["tok_s_4workers_omp2"] * MEASURED["flops_per_token_d96"] \
        / flops_per_token * factors
    tokens = params * tok_per_param
    seconds = tokens / tok_s_per_node
    years = seconds / (365.25 * 24 * 3600)
    months_1node = years * 12.0
    nodes_needed = months_1node / months
    return tok_s_per_node, tokens, years, nodes_needed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=float, default=1e9)
    ap.add_argument("--tokens-per-param", type=float, default=20.0)
    ap.add_argument("--months", type=float, default=6.0)
    ap.add_argument("--svg", default="/tmp/speedup_estimate.svg")
    a = ap.parse_args()

    print("=== ENTRADAS MEDIDAS (com a fonte)")
    print("  tok/s, 1 worker OMP=8        %8.0f   (sweep: 1024 tokens / 1.20 s)" % MEASURED["tok_s_1worker_omp8"])
    print("  tok/s, 1 worker OMP=16       %8.0f   (2.3x PIOR: 3.449 s/passo)" % MEASURED["tok_s_1worker_omp16"])
    print("  tok/s, 4 workers OMP~2       %8.0f   por worker (agregado 7292)" % MEASURED["tok_s_4workers_omp2"])
    print("  FLOPs/token no d96           %8.1f M (6 x 2.75M params)" % (MEASURED["flops_per_token_d96"] / 1e6))
    print("  GEMM puro (microbench)       %8.0f GFLOP/s (teto de eficiencia)" % MEASURED["gemm_microbench_gflops"])
    print("  fator head_dim 32 / 48        %8.2f / %.2f x" % (MEASURED["arch_factor_d32"], MEASURED["arch_factor_d48"]))
    print("  fator de lote (B)            %8.2f x (medido plano)" % MEASURED["batch_factor"])

    flops_node = MEASURED["tok_s_4workers_omp2"] * MEASURED["flops_per_token_d96"]
    print("\n  => FLOPs/s agregados por no' (8 cores): %.0f GFLOP/s = %.0f%% do microbench de GEMM"
          % (flops_node / 1e9, 100 * flops_node / (MEASURED["gemm_microbench_gflops"] * 1e9)))

    tf = comp_token_factor(MEASURED["comp_bpb_gain"], MEASURED["scaling_alpha"])
    print("  => composicao: %.3f bpb a menos no mesmo wall-clock = %.2fx menos tokens equivalentes"
          % (MEASURED["comp_bpb_gain"], tf))

    print("\n=== ESTIMATIVA (params=%.0e, %.0f tokens/param, alvo %.0f meses)"
          % (a.params, a.tokens_per_param, a.months))
    print("  %-22s %10s %12s %12s" % ("cenario", "tok/s/no'", "tempo 1 no'", "nos p/ alvo"))
    for label, mult in [("atual (head_dim 48)", MEASURED["arch_factor_d48"]),
                        ("composicao incluida", MEASURED["arch_factor_d48"] * tf),
                        ("head_dim 32", MEASURED["arch_factor_d32"]),
                        ("sem os ganhos de hoje", 1.0)]:
        tps, tokens, years, nodes = estimate(a.params, a.tokens_per_param, a.months, mult)
        print("  %-22s %10.1f %10.2f a %12.1f" % (label, tps, years, nodes))
    for tp in (20.0, 100.0):
        tps, tokens, years, nodes = estimate(a.params, tp, a.months,
                                             MEASURED["arch_factor_d48"] * tf)
        print("    (%3.0f tokens/param: %6.1fB tokens -> %.2f anos num no', %.1f nos p/ %.0f meses)"
              % (tp, tokens / 1e9, years, nodes, a.months))

    print("\n=== COMO A ESTIMATIVA MUDOU (multiplicador acumulado de eficiencia)")
    acc = 1.0
    for when, what, mult in TIMELINE:
        acc *= mult
        tps, tokens, years, nodes = estimate(a.params, a.tokens_per_param, a.months, acc)
        print("  %-20s x%-6.2f acum x%-6.2f  ->  %6.2f anos num no'  |  %5.1f nos p/ %.0f meses"
              % (when, mult, acc, years, nodes, a.months))
    acc_final = acc * tf
    tps, tokens, years, nodes = estimate(a.params, a.tokens_per_param, a.months, acc_final)
    print("  %-20s x%-6.2f acum x%-6.2f  ->  %6.2f anos num no'  |  %5.1f nos p/ %.0f meses"
          % ("+ composicao", tf, acc_final, years, nodes, a.months))

    # SVG simples: a linha do tempo como degraus
    W, H = 900, 340
    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (W, H),
         '<rect width="100%%" height="100%%" fill="white"/>',
         '<text x="20" y="26" font-size="15" font-family="monospace">'
         'Nos necessarios para 1B em 6 meses, conforme as descobertas</text>']
    ys = []
    acc = 1.0
    for i, (when, what, mult) in enumerate(TIMELINE + [("+ composicao", "", tf)]):
        acc *= mult
        _, _, years, nodes = estimate(a.params, a.tokens_per_param, a.months, acc)
        ys.append((i, when, nodes))
    nlo, nhi = min(n for _, _, n in ys), max(n for _, _, n in ys)
    x0, x1, y0, y1 = 60, W - 60, 60, H - 60
    yv = lambda n: y1 - (y1 - y0) * (nhi - n) / max(1e-9, nhi - nlo)
    xv = lambda i: x0 + (x1 - x0) * i / max(1, len(ys) - 1)
    for i in range(len(ys) - 1):
        s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#0366d6" stroke-width="2"/>'
                 % (xv(ys[i][0]), yv(ys[i][2]), xv(ys[i + 1][0]), yv(ys[i + 1][2])))
    for i, when, nodes in ys:
        s.append('<circle cx="%.1f" cy="%.1f" r="4" fill="#0366d6"/>' % (xv(i), yv(nodes)))
        s.append('<text x="%.1f" y="%.1f" font-size="11" font-family="monospace" text-anchor="middle">%.0f</text>'
                 % (xv(i), yv(nodes) - 10, nodes))
        s.append('<text x="%.1f" y="%d" font-size="10" font-family="monospace" text-anchor="middle" fill="#555">%s</text>'
                 % (xv(i), y1 + 18, when.replace(" ", "\u00a0")))
    s.append('<text x="20" y="%d" font-size="11" font-family="monospace" fill="#555">'
             'menos nos = mais eficiente; cada degrau e um achado MEDIDO (lote foi plano)</text>' % (H - 12))
    s.append('</svg>')
    with open(a.svg, "w") as fh:
        fh.write("\n".join(s))
    print("\nsvg: %s" % a.svg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
