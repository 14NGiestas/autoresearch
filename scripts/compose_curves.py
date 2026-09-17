#!/usr/bin/env python3
"""compose_curves.py — as tres curvas que decidem se compor vale a pena.

Painel A: lei de escala propria (L vs tokens distintos), com os bracos fundidos
          marcados contra o ideal de mesma cobertura.
Painel B: o imposto de composicao cresce com o TREINO por worker (1 -> 2 epocas).
Painel C: o imposto cai com K *degeneradamente* (workers andam menos).
Painel D: a regra de decisao: ganho liquido = K / 2^(imposto/alpha), para o alpha
          medido aqui (0.46) e para alphas de escala maior (0.15, 0.05). E onde se
          ve que "bom ou ruim" depende de alpha, nao do imposto sozinho.

Sem dependencias (SVG a mao, como scripts/basin_map.py).
"""
import math
import sys

W, H = 430, 320
OUT = "/tmp/compose_curves.svg"

# --- dados (holdout 100 linhas, d96, byte-weighted bpb) ----------------------
single = [(1952, 2.44475), (3904, 2.26189), (7808, 2.12000), (15616, 2.02520)]
fit = (1.760, 530.7, 0.458)          # L = Linf + A*D^-alpha (rmse 0.0022)
tax_epoch = [(1, 0.05728), (2, 0.33510)]
tax_k = [(2, 0.29590), (4, 0.05728), (8, -0.00310)]
k4 = dict(single_wall=2.26189, single_compute=2.02520, merge_1ep=2.50360,
          merge_2ep=2.59543, ideal_1ep=2.12000, ideal_2ep=1.97600)
ALPHAS = [(0.458, "#5ec8ff", "alpha=0.46 (medido aqui)"),
          (0.15, "#ffb454", "alpha=0.15 (escala media)"),
          (0.05, "#ff5c5c", "alpha=0.05 (fronteira)")]


def sx(v, lo, hi, x0, w):
    return x0 + w * (v - lo) / (hi - lo)


def sy(v, lo, hi, y0, h):
    return y0 + h * (hi - v) / (hi - lo)


def frame(ox, title, xlab, ylab, xlo, xhi, ylo, yhi, xlog=False):
    w, h = W - 70, H - 60
    y0 = 30
    o = [f'<g transform="translate({ox},0)">']
    o.append(f'<rect x="8" y="8" width="{W-16}" height="{H-16}" rx="6" '
             f'fill="#0b0e14" stroke="#2a3140"/>')
    o.append(f'<text x="20" y="28" fill="#e6edf3" font-size="13" '
             f'font-family="sans-serif">{title}</text>')
    # eixos
    o.append(f'<line x1="60" y1="{y0+h}" x2="{60+w}" y2="{y0+h}" stroke="#39404f"/>')
    o.append(f'<line x1="60" y1="{y0}" x2="60" y2="{y0+h}" stroke="#39404f"/>')
    for i in range(5):
        yv = ylo + (yhi - ylo) * i / 4
        yy = sy(yv, ylo, yhi, y0, h)
        o.append(f'<line x1="60" y1="{yy:.1f}" x2="{60+w}" y2="{yy:.1f}" '
                 f'stroke="#1d222c"/>')
        o.append(f'<text x="56" y="{yy+4:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end" font-family="sans-serif">{yv:.2f}</text>')
    o.append(f'<text x="{60+w/2:.0f}" y="{y0+h+22}" fill="#8b949e" font-size="9" '
             f'text-anchor="middle" font-family="sans-serif">{xlab}</text>')
    o.append(f'<text x="14" y="{y0+h/2:.0f}" fill="#8b949e" font-size="9" '
             f'transform="rotate(-90 14 {y0+h/2:.0f})" text-anchor="middle" '
             f'font-family="sans-serif">{ylab}</text>')
    return o, (lambda v: sx(math.log10(v) if xlog else v, xlo, xhi, 60, w)), \
        (lambda v: sy(v, ylo, yhi, y0, h)), (60, y0, w, h)


def dot(o, x, y, c, r=3.5, shape="circle"):
    if shape == "square":
        o.append(f'<rect x="{x-r:.1f}" y="{y-r:.1f}" width="{2*r}" height="{2*r}" '
                 f'fill="{c}"/>')
    else:
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{c}"/>')


def label(o, x, y, t, c, size=9, anchor="start"):
    o.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="{c}" font-size="{size}" '
             f'text-anchor="{anchor}" font-family="sans-serif">{t}</text>')


def panel_a(ox):
    o, X, Y, _ = frame(ox, "A. Lei de escala propria (1 maquina)", "tokens distintos (log)",
                       "bpb holdout", math.log10(2.0e6), math.log10(1.6e7), 1.95, 2.65,
                       xlog=True)
    Linf, A, al = fit
    pts = []
    for i in range(80):
        lg = math.log10(2.0e6) + (math.log10(2.0e7) - math.log10(2.0e6)) * i / 79
        D = 10 ** lg
        pts.append(f"{X(D):.1f},{Y(Linf + A * D ** (-al)):.1f}")
    o.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#3fb950" '
             f'stroke-width="1.6"/>')
    for st, b in single:
        dot(o, X(st * 1024), Y(b), "#3fb950")
    label(o, X(2000 * 1024), Y(2.44475) - 8, "4 runs unicos (dados completos)", "#3fb950")
    # bracos fundidos (mesma cobertura que o ideal)
    for b, t, c in [(k4["merge_1ep"], "merge K=4, 1 ep (2.504)", "#ff5c5c"),
                    (k4["merge_2ep"], "merge K=4, 2 ep (2.595)", "#ff5c5c"),
                    (k4["ideal_1ep"], "ideal 1 ep = f0 (2.120)", "#5ec8ff"),
                    (k4["ideal_2ep"], "ideal 2 ep (1.976)", "#5ec8ff")]:
        dot(o, X(8.0e6), Y(b), c, shape="square")
        label(o, X(8.0e6) + 6, Y(b) + 3, t, c)
    return o


def panel_b(ox):
    o, X, Y, _ = frame(ox, "B. Imposto cresce com o treino por worker", "epocas da fatia",
                       "imposto (bpb)", 0.8, 2.2, 0.0, 0.40)
    for (e, t) in tax_epoch:
        dot(o, X(e), Y(t), "#ff5c5c", r=4)
    o.append(f'<polyline points="{X(1):.1f},{Y(0.05728):.1f} {X(2):.1f},{Y(0.33510):.1f}" '
             f'fill="none" stroke="#ff5c5c" stroke-width="1.6"/>')
    label(o, X(1) - 4, Y(0.05728) - 8, "1 ep: +0.057 (merge melhor que os shards)",
          "#ff5c5c", anchor="start")
    label(o, X(2) - 100, Y(0.33510) - 10, "2 ep: +0.335 (x5.9)", "#ff5c5c")
    label(o, 70, Y(0.36), "treinar mais melhora shard e run unico IGUAL (x0.19)", "#8b949e")
    label(o, 70, Y(0.33), "e so o merge piora: o imposto nao fecha com epocas", "#8b949e")
    return o


def panel_c(ox):
    o, X, Y, _ = frame(ox, "C. Imposto cai com K (degenerado)", "numero de shards K",
                       "imposto (bpb)", 1.6, 8.4, -0.05, 0.35)
    for (k, t) in tax_k:
        dot(o, X(k), Y(t), "#ffb454", r=4)
    o.append('<polyline points="' + " ".join(f"{X(k):.1f},{Y(t):.1f}" for k, t in tax_k) +
             '" fill="none" stroke="#ffb454" stroke-width="1.6"/>')
    label(o, X(2) + 6, Y(0.2959), "K=2: +0.296", "#ffb454")
    label(o, X(4) + 6, Y(0.05728), "K=4: +0.057", "#ffb454")
    label(o, X(8) - 130, Y(-0.0031) - 6, "K=8: -0.003 (workers mal andaram)", "#ffb454")
    label(o, 70, Y(0.24), "K grande parece bom porque cada worker", "#8b949e")
    label(o, 70, Y(0.21), "fica mais perto do init: degenerescencia", "#8b949e")
    return o


def panel_d(ox):
    o, X, Y, _ = frame(ox, "D. Decisao: ganho liquido = K / 2^(imposto/alpha)",
                       "imposto de composicao (bpb)", "ganho liquido (x)", 0.0, 0.7, 0.0, 32)
    for al, c, nm in ALPHAS:
        pts = []
        for i in range(60):
            t = 0.7 * i / 59
            pts.append(f"{X(t):.1f},{Y(32 / 2 ** (t / al)):.1f}")
        o.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{c}" '
                 f'stroke-width="1.6"/>')
        label(o, X(0.62), Y(32 / 2 ** (0.62 / al)) - 6, nm, c)
    o.append(f'<line x1="{X(0.3351):.1f}" y1="{Y(0):.1f}" x2="{X(0.3351):.1f}" '
             f'y2="{Y(32):.1f}" stroke="#8b949e" stroke-dasharray="3,3"/>')
    label(o, X(0.3351) + 4, Y(31), "imposto medido (K=4, 2 ep)", "#8b949e")
    for K, c in [(4, "#5ec8ff"), (8, "#ffb454"), (32, "#3fb950")]:
        label(o, X(0.02), Y(K * 0.93), f"K={K}", c)
    return o


def main():
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1760" height="340" '
             'style="background:#0b0e14">']
    for i, fn in enumerate((panel_a, panel_b, panel_c, panel_d)):
        parts += fn(i * 440)
        parts.append('</g>')
    parts.append(f'<text x="8" y="332" fill="#8b949e" font-size="9" '
                 f'font-family="sans-serif">fonte: holdout 100 linhas (rows_f0[7900:8000]), '
                 f'd96, total compute fixo, byte-weighted bpb; ajuste A: '
                 f'L={fit[0]:.3f}+{fit[1]:.1f}D^-{fit[2]} (rmse 0.0022)</text>')
    parts.append('</svg>')
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    open(out, "w").write("\n".join(parts))
    print(f"escrito {out}")


if __name__ == "__main__":
    main()
