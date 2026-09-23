#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""compose_final_plot.py — o que medimos, em tres paineis sem ruido.

A. A LEI DO IMPOSTO vs LR (dados do job 104/101, medias UNIFORMES EXATAS): o
   imposto cai ~11x quando o LR cai 3x => ~lr^2. Ponto de 2 epocas e o de K=8
   anotados. independentes inits (seed) e o teto: +1.24 bpb.
B. SELECAO vs AGREGACAO (job 105, dev separado do holdout): escolher o melhor
   shard empata com o greedy e bate a media uniforme por 0.157 bpb.
C. MPI (relatorio do midev): invariante bit a bit (90/90 arquivos) e throughput
   1 rank vs 2 ranks com lotes disjuntos.

SVG a mao, sem dependencias.
"""
import math
import sys

W, H = 640, 400


def frame(ox, title, xlab, ylab, xlo, xhi, ylo, yhi, xlog=False, ylog=False):
    x0, y0, pw, ph = ox + 64, 54, W - 120, H - 130

    def X(v):
        t = math.log10(v) if xlog else v
        a, b = (math.log10(xlo), math.log10(xhi)) if xlog else (xlo, xhi)
        return x0 + pw * (t - a) / (b - a)

    def Y(v):
        t = math.log10(v) if ylog else v
        a, b = (math.log10(ylo), math.log10(yhi)) if ylog else (ylo, yhi)
        return y0 + ph * (b - t) / (b - a)

    o = [f'<rect x="{ox+8}" y="8" width="{W+24}" height="{H+24}" rx="6" fill="#0b0e14" stroke="#2a3140"/>',
         f'<text x="{ox+22}" y="28" fill="#e6edf3" font-size="13" font-family="sans-serif">{title}</text>',
         f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" stroke="#39404f"/>',
         f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}" stroke="#39404f"/>',
         f'<text x="{x0+pw/2:.0f}" y="{y0+ph+34}" fill="#8b949e" font-size="10" text-anchor="middle" font-family="sans-serif">{xlab}</text>',
         f'<text x="{ox+18}" y="{y0+ph/2:.0f}" fill="#8b949e" font-size="10" transform="rotate(-90 {ox+18} {y0+ph/2:.0f})" text-anchor="middle" font-family="sans-serif">{ylab}</text>']
    return o, X, Y, (x0, y0, pw, ph)


def lab(o, x, y, t, c="#8b949e", size=9, anchor="start"):
    o.append(f'<text x="{x:.0f}" y="{y:.0f}" fill="{c}" font-size="{size}" '
             f'text-anchor="{anchor}" font-family="sans-serif">{t}</text>')


def main():
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{3*(W+40)}" height="{H+70}" style="background:#0b0e14">']

    # ---------- A: imposto vs LR ----------
    pts = [(6e-4, 0.16860, "1952 passos"), (2e-4, 0.01530, "1952"), (6e-5, 0.01610, "1952")]
    o, X, Y, _ = frame(0, "A. Imposto de composicao vs LR (K=4)", "learning rate (log)",
                       "imposto (bpb, log)", 4e-5, 8e-4, 0.01, 2.0, xlog=True, ylog=True)
    for lr, tax, nm in pts:
        o.append(f'<circle cx="{X(lr):.1f}" cy="{Y(tax):.1f}" r="4.5" fill="#5ec8ff"/>')
        lab(o, X(lr) + 7, Y(tax) + 3, f"{lr:g}: +{tax:.3f}", "#5ec8ff")
    # referencia lr^2 ancorada em 6e-4
    q = []
    for i in range(40):
        lr = 4e-5 * (8e-4 / 4e-5) ** (i / 39)
        q.append(f"{X(lr):.1f},{Y(0.16860 * (lr/6e-4)**2):.1f}")
    o.append(f'<polyline points="{" ".join(q)}" fill="none" stroke="#3fb950" stroke-dasharray="4,3"/>')
    lab(o, X(6e-5), Y(0.16860*(6e-5/6e-4)**2) + 16, "reta = lr^2 (o que o deslocamento preve)", "#3fb950")
    o.append(f'<circle cx="{X(6e-4):.1f}" cy="{Y(0.33510):.1f}" r="4.5" fill="#ffb454"/>')
    lab(o, X(6e-4) - 6, Y(0.33510) - 8, "6e-4 com 2 epocas: +0.335", "#ffb454", anchor="end")
    o.append(f'<circle cx="{X(6e-4):.1f}" cy="{Y(0.05550):.1f}" r="4.5" fill="#c9d1d9"/>')
    lab(o, X(6e-4) - 6, Y(0.05550) + 12, "K=8 (1 epoca): +0.056", "#c9d1d9", anchor="end")
    lab(o, X(4.2e-5), Y(1.5), "inits independentes (seed): +1.24", "#ff5c5c")
    lab(o, X(4.2e-5), Y(1.1), "cair o LR reduz o imposto MAS", "#8b949e")
    lab(o, X(4.2e-5), Y(0.8), "piora os shards (6e-5 => 3.25 bpb)", "#8b949e")

    # ---------- B: selecao vs agregacao ----------
    bars = [("melhor shard\n(oraculo)", 2.43977, "#3fb950"),
            ("escolhido no dev\n(= 1 shard)", 2.45740, "#5ec8ff"),
            ("media uniforme\ndos 4", 2.61492, "#ff5c5c"),
            ("1 maquina, mesmo\nwall-clock", 2.44475, "#ffb454")]
    o2, X2, Y2, _ = frame(680, "B. Selecao vence agregacao (K=4, 1 epoca)", "estrategia",
                          "bpb holdout", 0, len(bars), 2.40, 2.66)
    for i, (nm, v, c) in enumerate(bars):
        o2.append(f'<rect x="{X2(i)+10:.0f}" y="{Y2(v):.1f}" width="{X2(i+0.8)-X2(i):.0f}" '
                  f'height="{Y2(2.40)-Y2(v):.1f}" fill="{c}"/>')
        lab(o2, X2(i + 0.4), Y2(v) - 6, f"{v:.4f}", c, 10, "middle")
        for j, part in enumerate(nm.split("\n")):
            lab(o2, X2(i + 0.4), Y2(2.40) + 12 + 10 * j, part, "#8b949e", 9, "middle")
    lab(o2, X2(0.05), Y2(2.645), "escolher UM shard (dev) = greedy; media uniforme custa +0.157", "#e6edf3")

    # ---------- C: MPI ----------
    o3, X3, Y3, _ = frame(1360, "C. MPI: invariante exato e ~2x throughput",
                          "ranks (lotes disjuntos)", "tokens/s", 0.5, 3.5, 0, 220)
    o3.append(f'<circle cx="{X3(1):.1f}" cy="{Y3(96):.1f}" r="5" fill="#ffb454"/>')
    lab(o3, X3(1) + 8, Y3(96) + 4, "1 rank: 96 tok/s", "#ffb454")
    o3.append(f'<circle cx="{X3(2):.1f}" cy="{Y3(188):.1f}" r="5" fill="#3fb950"/>')
    lab(o3, X3(2) + 8, Y3(188) + 4, "2 ranks: 179-196 tok/s", "#3fb950")
    o3.append(f'<line x1="{X3(1):.1f}" y1="{Y3(96):.1f}" x2="{X3(2):.1f}" y2="{Y3(192):.1f}" '
              f'stroke="#3fb950" stroke-width="1.4"/>')
    lab(o3, X3(0.62), Y3(205), "invariante: -n 2 == -n 1", "#e6edf3", 10)
    lab(o3, X3(0.62), Y3(180), "90/90 arquivos identicos", "#3fb950", 10)
    lab(o3, X3(0.62), Y3(155), "(byte a byte, 3 passos)", "#8b949e", 9)
    lab(o3, X3(0.62), Y3(125), "Allreduce 38 MB = 15 ms", "#8b949e", 9)
    lab(o3, X3(0.62), Y3(95), "tau=1 custa 68.6s vs 62.8s", "#8b949e", 9)
    lab(o3, X3(0.62), Y3(65), "com 1 sync a cada 6 passos", "#8b949e", 9)

    out += ["<g>"] + o + ["</g>", "<g>"] + o2 + ["</g>", "<g>"] + o3 + ["</g>"]
    out.append(f'<text x="10" y="{H+50}" fill="#8b949e" font-size="9" font-family="sans-serif">'
               f'fontes: job 104 (medias exatas), 101 (shards dos bracos de lr), 105 (greedy com dev), '
               f'relatorio MPI do midev. Todos os numeros usam medias UNIFORMES EXATAS (o bug do alpha '
               f'invertido estah corrigido); holdout de 100 linhas em A/B.</text>')
    out.append("</svg>")
    p = sys.argv[1] if len(sys.argv) > 1 else "/tmp/compose_final.svg"
    open(p, "w").write("\n".join(out))
    print(f"escrito {p}")


if __name__ == "__main__":
    main()
