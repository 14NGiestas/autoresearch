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

import numpy as np

# ---------------------------------------------------------------- MEDIDO
MEASURED = {
    "tok_s_1worker_omp8": 853.0,     # archhead/sweep: 1024 tokens / 1.20 s (1 worker)
    "tok_s_1worker_omp16": 297.0,    # 1024/3.449: 16 threads e' 2.3x PIOR
    "tok_s_worker_4w": 1823.0,       # POR WORKER nos runs K=4 (o driver)
    "tok_s_node_4workers": 7292.0,   # AGREGADO de um no' de 8 cores: 4 x 1823
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
# A linha do tempo comeca do modo de execucao MAL configurado (1 worker com
# OMP=16, que e' 24.5x pior que 4 workers com OMP~2) e multiplica so' os ganhos
# MEDIDOS. Nada de dupla contagem: cada fator e' medido contra o anterior.
TIMELINE = [
    # a primeira entrada e' o BASELINE (mult=1.0): o tok/s dele vai como
    # tok_s_base=START para o estimate(), nao como multiplicador
    ("inicio (mal config)", "1 worker, OMP=16, head_dim 16, sem receita", 1.0),
    ("+ execucao paralela", "4 workers x OMP~2 (agregado 7292 vs 297)",
     MEASURED["tok_s_node_4workers"] / MEASURED["tok_s_1worker_omp16"]),
    ("+ B (lote)",          "medido PLANO: nao ajuda", MEASURED["batch_factor"]),
    # head_dim 32 e 48 sao ALTERNATIVAS (constroi-se um modelo, com D=32 OU D=48),
    # nao degraus cumulativos: um so' fator, o do D=48, com o D=32 anotado.
    ("+ head_dim 48 (D=32: x1.80)", "mesmos FLOPs/params, outra forma", MEASURED["arch_factor_d48"]),
]


def comp_token_factor(gain, alpha):
    """Composicao da' menos loss pelo mesmo wall-clock -> menos tokens equivalentes.
    L2/L1 = (T2/T1)^-alpha  =>  T2/T1 = (L1/L2)^(1/alpha)."""
    l_ratio = 1.0 - gain / 2.4   # bpb ~ 2.4 no nosso regime; gain em bpb absoluto
    return l_ratio ** (-1.0 / alpha)


def estimate(params, tok_per_param, months, speed=1.0, token_equiv=1.0,
             tok_s_base=None):
    """Nos necessarios para treinar `params` em `months`.

    Modelo UNICO (para as duas tabelas fecharem por construcao):
        tokens_necessarios = params * tokens/param / token_equiv
        tok/s por no'      = base_d96 * (flops_d96/flops_modelo) * speed
        nos                = tokens_necessarios / (tok/s * segundos_alvo)

    `speed`        = ganhos de VELOCIDADE (execucao, head_dim)
    `token_equiv`  = ganhos que reduzem TOKENS necessarios (composicao)
    `tok_s_base`   = throughput AGREGADO de um no' de 8 cores no d96."""
    if tok_s_base is None:
        tok_s_base = MEASURED["tok_s_node_4workers"]
    flops_per_token = 6.0 * params
    tok_s_per_node = tok_s_base * MEASURED["flops_per_token_d96"] / flops_per_token * speed
    tokens = params * tok_per_param / token_equiv
    seconds = tokens / tok_s_per_node
    return tok_s_per_node, tokens, seconds / (365.25 * 24 * 3600), \
        seconds / (months * 30.44 * 24 * 3600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=float, default=1e9)
    ap.add_argument("--tokens-per-param", type=float, default=20.0)
    ap.add_argument("--months", type=float, default=6.0)
    ap.add_argument("--svg", default="/tmp/speedup_estimate.svg")
    ap.add_argument("--threads", type=int, default=36,
                    help="threads of the whole cluster: 16 (fermi) + 20 (halfbeast)")
    ap.add_argument("--budgets", default="1,2,7,30,182",
                    help="time budgets in days, comma separated")
    ap.add_argument("--tok-per-param", type=float, default=20.0)
    a = ap.parse_args()

    print("=== ENTRADAS MEDIDAS (com a fonte)")
    print("  tok/s, 1 worker OMP=8        %8.0f   (sweep: 1024 tokens / 1.20 s)" % MEASURED["tok_s_1worker_omp8"])
    print("  tok/s, 1 worker OMP=16       %8.0f   (2.3x PIOR: 3.449 s/passo)" % MEASURED["tok_s_1worker_omp16"])
    print("  tok/s, por worker (K=4)      %8.0f   (log do driver)" % MEASURED["tok_s_worker_4w"])
    print("  tok/s, NO' de 8 cores        %8.0f   AGREGADO (4 workers x 1823)" % MEASURED["tok_s_node_4workers"])
    print("  FLOPs/token no d96           %8.1f M (6 x 2.75M params)" % (MEASURED["flops_per_token_d96"] / 1e6))
    print("  GEMM puro (microbench)       %8.0f GFLOP/s (teto de eficiencia)" % MEASURED["gemm_microbench_gflops"])
    print("  fator head_dim 32 / 48        %8.2f / %.2f x" % (MEASURED["arch_factor_d32"], MEASURED["arch_factor_d48"]))
    print("  fator de lote (B)            %8.2f x (medido plano)" % MEASURED["batch_factor"])

    flops_node = MEASURED["tok_s_node_4workers"] * MEASURED["flops_per_token_d96"]
    print("\n  => FLOPs/s agregados por no' (8 cores): %.0f GFLOP/s = %.0f%% do microbench de GEMM"
          % (flops_node / 1e9, 100 * flops_node / (MEASURED["gemm_microbench_gflops"] * 1e9)))

    tf = comp_token_factor(MEASURED["comp_bpb_gain"], MEASURED["scaling_alpha"])
    print("  => composicao: %.3f bpb a menos no mesmo wall-clock = %.2fx menos tokens equivalentes"
          % (MEASURED["comp_bpb_gain"], tf))

    print("\n=== ESTIMATIVA (params=%.0e, %.0f tokens/param, alvo %.0f meses)"
          % (a.params, a.tokens_per_param, a.months))
    print("  %-22s %10s %12s %12s" % ("cenario", "tok/s/no'", "tempo 1 no'", "nos p/ alvo"))
    START = MEASURED["tok_s_1worker_omp16"]
    rows_tab = [
        ("inicio (1 worker, OMP=16)", START, 1.0, 1.0),
        ("+ execucao eficiente", START, MEASURED["tok_s_node_4workers"] / START, 1.0),
        ("+ head_dim 32", START, MEASURED["tok_s_node_4workers"] / START * MEASURED["arch_factor_d32"], 1.0),
        ("+ head_dim 48", START, MEASURED["tok_s_node_4workers"] / START * MEASURED["arch_factor_d48"], 1.0),
        ("+ composicao (= hoje)", START, MEASURED["tok_s_node_4workers"] / START * MEASURED["arch_factor_d48"], tf),
    ]
    for label, base, spd, teq in rows_tab:
        tps, tokens, years, nodes = estimate(a.params, a.tokens_per_param, a.months, spd, teq, base)
        print("  %-27s %10.1f %9.1f a %10.1f" % (label, tps, years, nodes))
    print("    (100 tokens/param, hoje: %.1f nos p/ %.0f meses)"
          % (estimate(a.params, 100.0, a.months, rows_tab[-1][2], tf, START)[3], a.months))
    n_start = estimate(a.params, a.tokens_per_param, a.months, 1.0, 1.0, START)[3]
    n_now = estimate(a.params, a.tokens_per_param, a.months, rows_tab[-1][2], tf, START)[3]

    print("\n=== COMO A ESTIMATIVA MUDOU (multiplicador acumulado de eficiencia)")
    acc = 1.0
    print("  %-20s %8s %10s %10s %12s" % ("descoberta", "fator", "acum", "anos/1no'", "nos p/ alvo"))
    for when, what, mult in TIMELINE:
        acc *= mult
        tps, tokens, years, nodes = estimate(a.params, a.tokens_per_param, a.months, acc, 1.0, START)
        print("  %-20s x%-7.2f x%-9.2f %10.1f %12.1f" % (when, mult, acc, years, nodes))
    _, _, years, nodes = estimate(a.params, a.tokens_per_param, a.months, acc, tf, START)
    print("  %-20s x%-7.2f x%-9.2f %10.1f %12.1f   (tokens-equivalentes)"
          % ("+ composicao (= hoje)", tf, acc * tf, years, nodes))
    print("\n  [fechamento] inicio %.1f nos -> hoje %.1f nos  =  ganho x%.0f"
          % (n_start, n_now, n_start / n_now))

    # ---- Quality: the slice that the data can hold --------------------------
    # A quality manifold needs at least two independent axes. We have exactly one
    # size (d96 = 2.75M) with a measured bpb: the scaling curves are all d96, and
    # the d768 checkpoints hold 3 steps each (smoke tests, no quality). So this
    # section fits the TOKENS axis only and reports the result as a band.
    #
    # The band is the honest form. Our own epochs experiment showed that alpha is
    # not identifiable in this range: 1.5 decades of tokens with a jitter of
    # 0.017 bpb. A single exponent would be a false precision.
    def admissible(pts, t_ref, tol=0.02):
        """The set of fits L = E + A*t^-alpha that the points admit.

        For each floor E on a grid, and each alpha on a grid, A comes from a least
        squares fit in log space. A fit is admissible when its largest relative
        error stays under tol. The set of admissible fits gives the band: it is a
        statement about what the data cannot tell apart, and not a measurement
        error.
        """
        t = np.array([q[0] for q in pts], float)
        l = np.array([q[1] for q in pts], float)
        ok = np.isfinite(l)
        t, l = t[ok], l[ok]
        if len(t) < 8:
            return []
        out = []
        for e in np.linspace(0.60 * l.min(), l.min() - 0.01, 25):
            for al in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                r = l - e
                if (r <= 0).any():
                    continue
                a = float(np.exp(np.mean(np.log(r) + al * np.log(t))))
                pred = e + a * t ** (-al)
                if np.max(np.abs(pred - l) / l) <= tol:
                    out.append((e, al, float(e + a * t_ref ** (-al))))
        return out

    curves = []
    for f in ("/tmp/scal_curve.npy", "/tmp/scal65_curve.npy", "/tmp/scal65_1k_curve.npy"):
        try:
            curves.append((os.path.basename(f), np.load(f)))
        except OSError:
            pass
    if curves:
        print("\n=== QUALIDADE (fatia d96; o eixo TAMANHO esta' vazio)")
        print("  Cada curva e' um EXPERIMENTO distinto (niveis diferentes no mesmo")
        print("  token count), entao o ajuste e' por curva. A faixa e' o conjunto de")
        print("  leis que os dados admitem com erro <= 2%.")
        for name, pts in curves:
            n = len(pts)
            line = []
            for t_ref in (2e6, 1e7):
                adm = admissible(pts, t_ref)
                if adm:
                    vals = [v for _, _, v in adm]
                    line.append("%.1fM: %.3f-%.3f" % (t_ref / 1e6, min(vals), max(vals)))
            if line:
                print("  %-24s %2d pontos  %s" % (name, n, "  |  ".join(line)))
            else:
                print("  %-24s %2d pontos  sem ajuste admissivel (queda e plato, ou NaN)"
                      % (name, n))
        adm = admissible(curves[0][1], 1e7)
        if adm:
            alphas = sorted(set(al for _, al, _ in adm))
            print("  (a curva de referencia admite alpha de %.1f a %.1f -- largura de"
                  % (alphas[0], alphas[-1]))
            print("   familia, nao erro de medida; jitter do holdout e' +-0.017)")
        print("  Deslocamento medido (a receita, mesmo wall-clock): reset + outer = -0.168")
        print("  bpb em 2 sorteios. O efeito de head_dim sai no job 154.")
        print("  O eixo TAMANHO precisa de ~3 dias de fila para existir.")

    # ---- The largest model for a given time, for any budget ----------------
    # The budget of FLOPs fixes the largest model. A model of P parameters needs
    # 120*P^2 FLOPs when it trains on 20 tokens for each parameter, because the
    # cost of one token is 6*P and the token count is 20*P. The composition
    # divides the token count, so it divides the budget.
    #
    # The rate per thread comes from the measured node rate: 7292 tokens per
    # second on 8 cores at d96, which is 11.8 MFLOP for one token.
    rate_thread = MEASURED["tok_s_node_4workers"] * MEASURED["flops_per_token_d96"] / 8.0
    speed = MEASURED["arch_factor_d48"] * rate_thread
    print("\n=== MAIOR MODELO POR ORCAMENTO DE TEMPO (%d threads = fermi 16 + halfbeast 20)"
          % a.threads)
    print("  %-10s %12s %14s %14s" % ("orcamento", "params max", "tokens", "FLOPs"))
    for days in [float(x) for x in a.budgets.split(",")]:
        secs = days * 86400.0
        flops = speed * a.threads * secs
        # token_equiv cuts the token count, so the same FLOPs buy a larger model
        p_max = ((flops / tf) / (6.0 * a.tok_per_param)) ** 0.5
        tokens = a.tok_per_param * p_max / tf
        if days < 1.0:
            label = "%.0f horas" % (days * 24.0)
        else:
            label = "%.0f dia%s" % (days, "" if days == 1 else "s")
        print("  %-10s %11.1f M %13.2f B %12.1f EFLOP" % (label, p_max / 1e6, tokens / 1e9, flops / 1e18))
    print("  premissas: 6*P FLOPs por token, %.0f tokens por param (Chinchilla), taxa medida"
          % a.tok_per_param)
    print("  de %.1f GFLOP/s por thread, fator de head_dim %.2f, e a composicao como"
          % (rate_thread / 1e9, MEASURED["arch_factor_d48"]))
    print("  tokens-equivalentes (medida a 3M: EXTRAPOLACAO nas escalas acima).")
    print("  O estimador preve TAMANHO, nao qualidade: nao ha lei de escala nossa para bpb.")

    # ---- With the machines that we have -----------------------------------
    # Our hardware holds two machines. The estimate needs their thread count,
    # because the measured rate is per 8-core node. Halfbeast uses an HDD and a
    # slower CPU, so this number is a ceiling and not a promise.
    OURS = [("fermi", 16), ("halfbeast", 20)]
    threads = sum(t for _, t in OURS)
    node_equiv = threads / 8.0
    print("\n=== COM OS NOSSOS 2 NOS (%s = %d threads, %.1fx um no' de 8 cores)"
          % (", ".join("%s %d" % m for m in OURS), threads, node_equiv))
    print("  %-34s %10s %12s" % ("cenario", "tempo", "e' suficiente?"))
    # Hoje: a base e' o no' eficiente (4 workers x 1823 = 7292), e a composicao
    # entra SO' como tokens-equivalentes. Contar a composicao tambem como
    # velocidade seria contar duas vezes -- foi o bug desta secao.
    for label, spd, teq in [("hoje, 20 tokens/param", MEASURED["arch_factor_d48"], tf),
                            ("hoje, 100 tokens/param", MEASURED["arch_factor_d48"], tf)]:
        tpp = 20.0 if "20" in label else 100.0
        tps, tokens, years, _ = estimate(a.params, tpp, a.months, spd, teq)
        secs = tokens / (tps * node_equiv)
        days = secs / 86400.0
        print("  %-34s %8.1f dias %12s" % (label, days, "sim" if days <= a.months * 30.44 else "NAO"))
    # Largest model that fits in the target time. A model of P parameters needs
    # 20*P tokens at 6*P FLOPs per token. The budget fixes P squared.
    flops_budget = MEASURED["tok_s_node_4workers"] * MEASURED["arch_factor_d48"] * \
        node_equiv * MEASURED["flops_per_token_d96"] * (a.months * 30.44 * 86400.0)
    p_true = ((flops_budget / tf) / 120.0) ** 0.5   # 120 = 6 FLOP/token x 20 tokens/param
    print("  %-34s %.0f M params" % ("maior modelo em %.0f meses (20 tok/param)" % a.months, p_true / 1e6))
    print("  (o orcamento de FLOPs em %.0f meses e' %.1f EFLOP; 1B a 20 tok/param pede 120 EFLOP)"
          % (a.months, flops_budget / 1e18))
    print("  (o teto ignora o HDD da halfbeast e o clock menor dela)")
    print("  (a composicao entra como tokens-equivalentes, medidos a 3M. Usar esse")
    print("   fator a 245M e' extrapolacao, e fica marcado como tal.)")

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
