#!/usr/bin/env python3
"""scaling_plot.py — a curva de escala medida, com o piso identificado.

Painel A: bpb medido (holdout) vs tokens vistos, com os dois ajustes (todos os
          32 pontos, que incluem o trecho de REPETICAO do pool, e so a janela
          limpa 1M-8M) e a marca onde o pool de 8M tokens se esgota.
Painel B: o mesmo em log-log do EXCESSO sobre o piso ajustado, com as
          inclinacoes de referencia (Chinchilla 0.28 e Kaplan 0.095 no eixo de
          dados) para ver onde caímos.

SVG a mao (sem matplotlib), como basin_map.py / compose_curves.py.
"""
import math
import sys

import numpy as np

W, H = 700, 430
DATA = "/tmp/scal_curve.npy"
FOUR = [(2.014e6, 2.44475), (4.0e6, 2.26189), (8.0e6, 2.12000), (16.0e6, 2.02520)]
POOL_END = 8.0e6          # tokens do pool pequeno (rows_f0, 7812 linhas)


def fit(D, L):
    best = None
    for Linf in np.arange(0.2, 2.0, 0.005):
        if np.any(L - Linf <= 0):
            continue
        sl, ic = np.polyfit(np.log(D), np.log(L - Linf), 1)
        pred = Linf + math.exp(ic) * D ** sl
        r = float(np.sqrt(np.mean((pred - L) ** 2)))
        if best is None or r < best[0]:
            best = (r, Linf, -sl, math.exp(ic))
    return best


def main():
    pts = np.load(DATA)
    D, L = pts[:, 0], pts[:, 1]
    r_all, Lall, a_all, A_all = fit(D, L)
    m = D <= POOL_END * 1.02
    r_cl, Lcl, a_cl, A_cl = fit(D[m], L[m])
    x0, y0, pw, ph = 70, 54, W - 110, H - 130
    xlo, xhi = math.log10(0.9e6), math.log10(3.6e7)
    X = lambda v: x0 + pw * (math.log10(v) - xlo) / (xhi - xlo)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{3*(W+40)}" height="{H+60}" '
           f'style="background:#0b0e14">']

    def frame(title, ylo, yhi, ylab):
        o = [f'<rect x="8" y="8" width="{W+24}" height="{H+24}" rx="6" '
             f'fill="#0b0e14" stroke="#2a3140"/>',
             f'<text x="22" y="28" fill="#e6edf3" font-size="14" '
             f'font-family="sans-serif">{title}</text>']
        Y = lambda v: y0 + ph * (yhi - v) / (yhi - ylo)
        o.append(f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" stroke="#39404f"/>')
        o.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}" stroke="#39404f"/>')
        for i in range(5):
            v = ylo + (yhi - ylo) * i / 4
            o.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x0+pw}" y2="{Y(v):.1f}" '
                     f'stroke="#1d222c"/>')
            o.append(f'<text x="{x0-6}" y="{Y(v)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end" font-family="sans-serif">{v:.2f}</text>')
        for v in (1e6, 2e6, 4e6, 8e6, 16e6, 32e6):
            o.append(f'<line x1="{X(v):.1f}" y1="{y0}" x2="{X(v):.1f}" y2="{y0+ph}" '
                     f'stroke="#141922"/>')
            o.append(f'<text x="{X(v):.1f}" y="{y0+ph+15}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle" font-family="sans-serif">{v/1e6:g}M</text>')
        o.append(f'<text x="{x0+pw/2:.0f}" y="{y0+ph+34}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle" font-family="sans-serif">tokens vistos (log)</text>')
        o.append(f'<text x="16" y="{y0+ph/2:.0f}" fill="#8b949e" font-size="10" '
                 f'transform="rotate(-90 16 {y0+ph/2:.0f})" text-anchor="middle" '
                 f'font-family="sans-serif">{ylab}</text>')
        return o, X, Y

    # ---------- A: bpb vs tokens ----------
    o, X, Y = frame("A. Curva de escala medida (d96, 3,0M params, holdout 100 linhas)",
                    1.98, 2.50, "bpb")
    o.append(f'<line x1="{X(POOL_END):.1f}" y1="{y0}" x2="{X(POOL_END):.1f}" y2="{y0+ph}" '
             f'stroke="#ff5c5c" stroke-dasharray="4,3"/>')
    o.append(f'<text x="{X(POOL_END)+5:.0f}" y="{y0+14}" fill="#ff5c5c" font-size="9" '
             f'font-family="sans-serif">pool de 8M se esgota: daqui pra frente e REPETICAO</text>')
    for lo, A_, al, c in ((min(D), A_all, a_all, "#ffb454"), (min(D), A_cl, a_cl, "#3fb950")):
        p = [f"{X(d):.1f},{Y(lo + A_ * d ** (-al)):.1f}" for d in
             np.exp(np.linspace(math.log(1e6), math.log(3.4e7), 90))]
        o.append(f'<polyline points="{" ".join(p)}" fill="none" stroke="{c}" '
                 f'stroke-width="1.6"/>')
    for d, l in zip(D, L):
        o.append(f'<circle cx="{X(d):.1f}" cy="{Y(l):.1f}" r="2.6" fill="#5ec8ff"/>')
    for d, l in FOUR:
        o.append(f'<rect x="{X(d)-3:.1f}" y="{Y(l)-3:.1f}" width="6" height="6" '
                 f'fill="#e6edf3"/>')
    o.append(f'<text x="{X(1.1e6):.0f}" y="{Y(2.47):.0f}" fill="#5ec8ff" font-size="9" '
             f'font-family="sans-serif">32 pontos medidos (1 run, saves periodicos)</text>')
    o.append(f'<text x="{X(1.1e6):.0f}" y="{Y(2.44):.0f}" fill="#e6edf3" font-size="9" '
             f'font-family="sans-serif">4 runs separados (cross-check)</text>')
    o.append(f'<text x="{X(1.1e6):.0f}" y="{Y(2.41):.0f}" fill="#3fb950" font-size="9" '
             f'font-family="sans-serif">ajuste janela limpa: L={Lcl:.2f}+{A_cl:.0f}D^-{a_cl:.3f} (rmse {r_cl:.4f})</text>')
    o.append(f'<text x="{X(1.1e6):.0f}" y="{Y(2.38):.0f}" fill="#ffb454" font-size="9" '
             f'font-family="sans-serif">ajuste todos os 32: L={Lall:.2f}+{A_all:.0f}D^-{a_all:.3f} (rmse {r_all:.4f})</text>')

    # ---------- B: log-log do excesso (eixo y de verdade em log) ----------
    lo, hi = 0.40, 1.25          # faixa do excesso em bpb (log, maior em cima)
    o2 = [f'<rect x="8" y="8" width="{W+24}" height="{H+24}" rx="6" fill="#0b0e14" '
          f'stroke="#2a3140"/>',
          f'<text x="22" y="28" fill="#e6edf3" font-size="14" font-family="sans-serif">'
          f'B. Excesso sobre o piso, em log-log (reta = lei de potencia)</text>']
    eY = lambda v: y0 + ph * (math.log10(hi) - math.log10(max(v, 1e-6))) / (math.log10(hi) - math.log10(lo))
    o2.append(f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" stroke="#39404f"/>')
    o2.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}" stroke="#39404f"/>')
    for v in (1.2, 1.0, 0.8, 0.6, 0.5, 0.4):  # rotulos por log
        o2.append(f'<line x1="{x0}" y1="{eY(v):.1f}" x2="{x0+pw}" y2="{eY(v):.1f}" '
                  f'stroke="#1d222c"/>')
        o2.append(f'<text x="{x0-6}" y="{eY(v)+3:.1f}" fill="#8b949e" font-size="9" '
                  f'text-anchor="end" font-family="sans-serif">{v:g}</text>')
    for v in (1e6, 2e6, 4e6, 8e6, 16e6, 32e6):
        o2.append(f'<line x1="{X(v):.1f}" y1="{y0}" x2="{X(v):.1f}" y2="{y0+ph}" stroke="#141922"/>')
        o2.append(f'<text x="{X(v):.1f}" y="{y0+ph+15}" fill="#8b949e" font-size="9" '
                  f'text-anchor="middle" font-family="sans-serif">{v/1e6:g}M</text>')
    o2.append(f'<text x="{x0+pw/2:.0f}" y="{y0+ph+34}" fill="#8b949e" font-size="10" '
              f'text-anchor="middle" font-family="sans-serif">tokens vistos (log)</text>')
    o2.append(f'<text x="16" y="{y0+ph/2:.0f}" fill="#8b949e" font-size="10" '
              f'transform="rotate(-90 16 {y0+ph/2:.0f})" text-anchor="middle" '
              f'font-family="sans-serif">excesso sobre o piso (bpb, log)</text>')
    # retas de referencia ancoradas no PRIMEIRO ponto medido
    d0, l0 = D[0], L[0]
    e0 = l0 - Lcl
    for al, c, nm in ((a_cl, "#3fb950", f"medido: {a_cl:.3f}"),
                      (0.28, "#ffb454", "Chinchilla 0.28"),
                      (0.095, "#ff5c5c", "Kaplan 0.095")):
        q = []
        for i2 in range(60):
            d = 10 ** (xlo + (xhi - xlo) * i2 / 59)
            e = e0 * (d / d0) ** (-al)
            q.append(f"{X(d):.1f},{eY(e):.1f}")
        o2.append(f'<polyline points="{" ".join(q)}" fill="none" stroke="{c}" stroke-width="1.6"'
                  f'{" stroke-dasharray=\"4,3\"" if al != a_cl else ""}/>')
    for d, l in zip(D, L):
        o2.append(f'<circle cx="{X(d):.1f}" cy="{eY(l - Lcl):.1f}" r="3" fill="#5ec8ff"/>')
    for t, c, nm in ((0.95, "#3fb950", f"medido {a_cl:.3f}"), (0.72, "#ffb454", "Chinchilla 0.28"),
                     (0.58, "#ff5c5c", "Kaplan 0.095")):
        o2.append(f'<text x="{X(1.1e6):.0f}" y="{eY(t):.0f}" fill="{c}" font-size="9" '
                  f'font-family="sans-serif">{nm}</text>')
    o2.append(f'<line x1="{X(POOL_END):.1f}" y1="{y0}" x2="{X(POOL_END):.1f}" y2="{y0+ph}" '
              f'stroke="#ff5c5c" stroke-dasharray="4,3"/>')
    o2.append(f'<text x="{X(POOL_END)+5:.0f}" y="{y0+14}" fill="#ff5c5c" font-size="9" '
              f'font-family="sans-serif">a partir daqui os dados se repetem</text>')
    o2.append(f'<text x="{X(1.1e6):.0f}" y="{eY(0.44):.0f}" fill="#8b949e" font-size="9" '
              f'font-family="sans-serif">reta = expoente constante; encurvar = muda de regime</text>')
    # ---------- C: inclinacao local do excesso (o painel que responde "cade o joelho?") ----------
    sp = sorted(zip(D / 1e6, L))
    def at(t):
        return min(sp, key=lambda kv: abs(kv[0] - t))
    o3, X3, Y3 = frame("C. Inclinacao local do excesso (sem assumir piso)",
                       0.0, 0.45, "|inclinacao log-log|")
    segs = []
    for a, b in ((1.0, 2.1), (2.1, 4.2), (4.2, 8.4), (8.4, 16.8), (16.8, 33.6)):
        xa, la = at(a); xb, lb = at(b)
        ea, eb = la - Lcl, lb - Lcl
        sl = -((math.log(eb) - math.log(ea)) / (math.log(xb) - math.log(xa)))
        segs.append((xb * 0.75, sl))
    p3 = []
    for t, sl in segs:
        x, y = X3(t * 1e6), Y3(sl)
        p3.append(f"{x:.1f},{y:.1f}")
    o3.append(f'<polyline points="{" ".join(p3)}" fill="none" stroke="#5ec8ff" stroke-width="1.6"/>')
    for t, sl in segs:
        x, y = X3(t * 1e6), Y3(sl)
        o3.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#5ec8ff"/>')
        o3.append(f'<text x="{x:.1f}" y="{y-9:.1f}" fill="#8b949e" font-size="9" '
                  f'text-anchor="middle" font-family="sans-serif">{sl:.3f}</text>')
    o3.append(f'<line x1="{X3(1e6):.1f}" y1="{Y3(a_cl):.1f}" x2="{X3(3.4e7):.1f}" '
              f'y2="{Y3(a_cl):.1f}" stroke="#3fb950" stroke-dasharray="4,3"/>')
    o3.append(f'<text x="{X3(1.05e6):.0f}" y="{Y3(a_cl)-6:.0f}" fill="#3fb950" font-size="9" '
              f'font-family="sans-serif">alpha do ajuste limpo = {a_cl:.3f}</text>')
    o3.append(f'<line x1="{X3(POOL_END):.1f}" y1="{y0}" x2="{X3(POOL_END):.1f}" y2="{y0+ph}" '
              f'stroke="#ff5c5c" stroke-dasharray="4,3"/>')
    o3.append(f'<text x="{X3(POOL_END)+5:.0f}" y="{y0+14}" fill="#ff5c5c" font-size="9" '
              f'font-family="sans-serif">dados repetidos</text>')
    o3.append(f'<text x="{X3(1.05e6):.0f}" y="{Y3(0.42):.0f}" fill="#8b949e" font-size="9" '
              f'font-family="sans-serif">constante na janela limpa, cai depois</text>')
    out += (["<g>"] + o + ["</g>"] +
            ["<g transform=\"translate(820,0)\">"] + o2 + ["</g>"] +
            ["<g transform=\"translate(1640,0)\">"] + o3 + ["</g>"])
    out.append(f'<text x="10" y="{H+40}" fill="#8b949e" font-size="9" '
               f'font-family="sans-serif">fonte: /tmp/scal (job 102, 32 checkpoints de UM run de 32768 passos, '
               f'LR constante 6e-4, blas, d96); janela limpa = ate 8M tokens (pool sem repeticao). '
               f'O job 103 (pool de 65M) repete isto sem trecho de repeticao.</text>')
    out.append("</svg>")
    p = sys.argv[1] if len(sys.argv) > 1 else "/tmp/scaling_plot.svg"
    open(p, "w").write("\n".join(out))
    print(f"escrito {p} | janela limpa alpha={a_cl:.3f} (Linf {Lcl:.2f}, rmse {r_cl:.4f}) | "
          f"todos os 32 alpha={a_all:.3f} (Linf {Lall:.2f})")


if __name__ == "__main__":
    main()
