#!/usr/bin/env python3
"""compose_phase.py — diagrama de fase da decisao de compor (SVG a mao, sem deps).

A lei:  ganho liquido = K / 2^(imposto/alpha)
        fronteira de viabilidade (ganho = 1):  imposto = alpha * log2(K)

Eixo x: imposto de composicao medido (bpb).  Eixo y: alpha, o expoente da lei de
escala que diz quanto compute vale (L = Linf + A*D^-alpha). O fundo colore o
ganho liquido para K=32; as tres linhas sao as fronteiras de K=4, 8 e 32. Os
pontos marcados sao as NOSSAS medicoes, todas em alpha=0.46 (o alpha que medimos
no d96); a faixa de alpha=0.05 e a ordem de grandeza da fronteira, onde o mesmo
imposto afunda o ganho.

Painel 2: a lei do proprio imposto (2 pontos), com a extrapolacao de 4 epocas
marcada como ponto vazado -- e a previsao que o proximo braco testa.
"""
import math
import sys

W, H = 760, 430
OUT = "/tmp/compose_phase.svg"
K_BG = 32
TAX_MAX, A_LO, A_HI = 0.80, 0.02, 0.60
MEAS = [(0.16860, "K=4, 1 ep (media exata)"), (0.33510, "K=4, 2 ep"),
        (0.57023, "K=4 2ep vs sequencial")]
A_OURS, A_LO, A_HI, A_CHINCHILLA, A_KAPLAN = 0.458, 0.315, 0.482, 0.28, 0.095
TAX_PTS = [(1952, 0.05728), (3904, 0.33510)]


def gain(tax, a, K):
    return K / 2 ** (tax / a)


def color(net):
    """ganho liquido -> cor (vermelho = <1, cinza = 1, azul = >1)."""
    z = math.log2(net)
    if z <= -2.0:
        return "#7f1d2e"
    if z >= 4.0:
        return "#1d4ed8"
    if z < 0:
        t = 1 + z / 2.0            # 0 (z=-2) .. 1 (z=0)
        r, g, b = 127 + (90 - 127) * t, 29 + (40 - 29) * t, 46 + (60 - 46) * t
    else:
        t = z / 4.0
        r, g, b = 90 + (29 - 90) * t, 40 + (78 - 40) * t, 60 + (216 - 60) * t
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def main():
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W+60}" height="{H+80}" '
         f'style="background:#0b0e14">']
    x0, y0, pw, ph = 80, 60, W - 130, H - 130

    def X(t):
        return x0 + pw * t / TAX_MAX

    def Y(a):
        return y0 + ph * (math.log(a) - math.log(A_LO)) / (math.log(A_HI) - math.log(A_LO))

    o.append(f'<text x="24" y="30" fill="#e6edf3" font-size="15" font-family="sans-serif">'
             f'Onde compor K maquinas vale a pena (ganho = K / 2^(imposto/alpha))</text>')
    o.append(f'<text x="24" y="48" fill="#8b949e" font-size="10" font-family="sans-serif">'
             f'fundo = ganho liquido para K={K_BG}; linhas = fronteira de viabilidade '
             f'(ganho=1) para K=4, 8, 32; pontos = nossas medicoes</text>')
    # fundo
    nx, ny = 144, 108
    for i in range(nx):
        for j in range(ny):
            t = TAX_MAX * (i + 0.5) / nx
            a = math.exp(math.log(A_LO) + (math.log(A_HI) - math.log(A_LO)) * (j + 0.5) / ny)
            o.append(f'<rect x="{x0 + pw*i/nx:.1f}" y="{Y(a) - ph/ny:.1f}" '
                     f'width="{pw/nx + 0.6:.2f}" height="{ph/ny + 0.6:.2f}" '
                     f'fill="{color(gain(t, a, K_BG))}"/>')
    o.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" '
             f'stroke="#39404f"/>')
    # fronteiras
    for K, c, nm in ((4, "#e6edf3", "K=4"), (8, "#ffd166", "K=8"), (32, "#3fb950", "K=32")):
        pts = []
        for i in range(80):
            t = TAX_MAX * i / 79
            a = t / math.log2(K) if t > 0 else A_LO
            if a < A_LO or a > A_HI:
                continue
            pts.append(f"{X(t):.1f},{Y(a):.1f}")
        if pts:
            o.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{c}" '
                     f'stroke-width="1.6" stroke-dasharray="5,3"/>')
            a_end = TAX_MAX / math.log2(K) if TAX_MAX / math.log2(K) <= A_HI else None
    # eixos
    o.append(f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" stroke="#39404f"/>')
    o.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}" stroke="#39404f"/>')
    for i in range(5):
        t = TAX_MAX * i / 4
        o.append(f'<text x="{X(t):.0f}" y="{y0+ph+16}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle" font-family="sans-serif">{t:.2f}</text>')
    for a in (0.02, 0.05, 0.1, 0.2, 0.4, 0.6):
        o.append(f'<text x="{x0-8}" y="{Y(a)+3:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="end" font-family="sans-serif">{a:g}</text>')
    o.append(f'<text x="{x0+pw/2:.0f}" y="{y0+ph+38}" fill="#8b949e" font-size="11" '
             f'text-anchor="middle" font-family="sans-serif">imposto de composicao (bpb)'
             f'  —  quanto o merge custa sobre a media dos shards</text>')
    o.append(f'<text x="26" y="{y0+ph/2:.0f}" fill="#8b949e" font-size="11" '
             f'transform="rotate(-90 26 {y0+ph/2:.0f})" text-anchor="middle" '
             f'font-family="sans-serif">alpha (quanto compute vale; log)</text>')
    # faixa alpha da fronteira
    for a, c, nm in ((A_CHINCHILLA, "#ffb454", "alpha do eixo de DADOS na Chinchilla (0.28)"),
                     (A_KAPLAN, "#ff5c5c", "idem em Kaplan (0.095)")):
        o.append(f'<rect x="{x0}" y="{Y(a)-2:.0f}" width="{pw}" height="4" '
                 f'fill="{c}" opacity="0.35"/>')
        o.append(f'<text x="{X(0.40):.0f}" y="{Y(a)-6:.0f}" fill="{c}" font-size="10" '
                 f'font-family="sans-serif">{nm} — mesma composicao, ganho muito menor</text>')
    # nossos pontos
    for t, nm in MEAS:
        o.append(f'<circle cx="{X(t):.1f}" cy="{Y(A_OURS):.1f}" r="5" fill="#e6edf3" '
                 f'stroke="#0b0e14" stroke-width="1.5"/>')
    o.append(f'<rect x="{X(0.008):.0f}" y="{Y(A_HI):.0f}" width="{X(0.80)-X(0.008):.0f}" '
             f'height="{abs(Y(A_LO)-Y(A_HI)):.0f}" fill="#e6edf3" opacity="0.10"/>')
    o.append(f'<text x="{X(0.60):.0f}" y="{Y(A_LO)+14:.0f}" fill="#e6edf3" font-size="9" '
             f'font-family="sans-serif">faixa da alpha medida (0.32-0.48: o ajuste depende da janela)</text>')
    o.append(f'<line x1="{X(0.02):.0f}" y1="{Y(A_OURS):.0f}" x2="{X(0.80):.0f}" '
             f'y2="{Y(A_OURS):.0f}" stroke="#e6edf3" stroke-width="0.8" opacity="0.5"/>')
    o.append(f'<text x="{X(0.01):.0f}" y="{Y(A_OURS)+16:.0f}" fill="#e6edf3" '
             f'font-size="10" font-family="sans-serif">alpha medido aqui = 0.46 '
             f'(d96, 4 pontos: L = 1.76 + 530.7 D^-0.458, rmse 0.0022)</text>')
    o.append(f'<text x="{X(0.075):.0f}" y="{Y(A_OURS)-46:.0f}" fill="#e6edf3" font-size="10" '
             f'font-family="sans-serif">1 ep (o ponto a esquerda): +0.057 → ganho 3.7x de 4x</text>')
    o.append(f'<text x="{X(0.075):.0f}" y="{Y(A_OURS)-32:.0f}" fill="#ffb454" font-size="10" '
             f'font-family="sans-serif">2 ep: +0.335 → 2.4x de 4x</text>')
    o.append(f'<text x="{X(0.075):.0f}" y="{Y(A_OURS)-18:.0f}" fill="#ff5c5c" font-size="10" '
             f'font-family="sans-serif">vs alternativa sequencial: +0.570 → 1.7x de 4x</text>')
    # rotulos de regiao
    o.append(f'<text x="{X(0.66):.0f}" y="{Y(0.30):.0f}" fill="#93c5fd" font-size="13" '
             f'font-family="sans-serif">COMPOR VALE</text>')
    o.append(f'<text x="{X(0.52):.0f}" y="{Y(0.045):.0f}" fill="#fca5a5" font-size="13" '
             f'font-family="sans-serif">COMPOR CUSTA</text>')
    o.append(f'<text x="{X(0.012):.0f}" y="{Y(0.072):.0f}" fill="#e6edf3" font-size="11" '
             f'font-family="sans-serif">fronteiras (tracejado): imposto = alpha * log2(K)</text>')
    o.append(f'<text x="{X(0.012):.0f}" y="{Y(0.086):.0f}" fill="#c9d1d9" font-size="10" '
             f'font-family="sans-serif">branco K=4, amarelo K=8, verde K=32 — acima da linha, '
             f'o merge custa mais do que o compute compra</text>')

    # ---------------- painel 2: a lei do imposto (2 pontos) ----------------
    px, py, pww, phh = W - 40, 60, 44, 0
    return o


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    open(out, "w").write("\n".join(main()) + "\n</svg>\n")
    print(f"escrito {out}")
