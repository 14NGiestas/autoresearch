#!/usr/bin/env python3
"""basin_map.py — geografia do espaco de pesos (stdlib + numpy).
1. histograma ASCII por variante (forma da distribuicao).
2. PCA via Gram (N pontos x 3M dims) -> scatter SVG com rotulos.
Uso: .venv-numpy/bin/python3 scripts/basin_map.py
"""
import os
import numpy as np

PTS = {
    "f0": "/tmp/mix/w_f0/best",
    "drop": "/tmp/mix/w_f_drop/best",
    "m25a": "/tmp/w_m25a/step_59889",
    "m25b": "/tmp/w_m25b_best",
    "soupP2": "/tmp/w_p2soup2",
    "sq": "/tmp/surgery/square",
    "neg": "/tmp/surgery/negate",
    "shuf": "/tmp/surgery/shuffle",
    "s05": "/tmp/surgery/scale05",
    "s20": "/tmp/surgery/scale20",
    "n001": "/tmp/surgery/noise001",
    "n01": "/tmp/surgery/noise01",
}
DSETS = {"f0": "d96", "drop": "d96", "sq": "d96", "neg": "d96", "shuf": "d96",
         "s05": "d96", "s20": "d96", "n001": "d96", "n01": "d96",
         "m25a": "d360", "m25b": "d360", "soupP2": "d216"}


def flat(d):
    a = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".npy") and not f.startswith("adam_"):
            a.append(np.load(os.path.join(d, f)).astype(np.float64).ravel())
    return np.concatenate(a)


def hist_ascii(v, name, w=48):
    qs = np.quantile(v, [0, .01, .25, .5, .75, .99, 1.0])
    h, e = np.histogram(v, bins=w, range=(qs[1], qs[5]))
    mx = h.max()
    print(f"{name}: min={qs[0]:+.2f} p50={qs[3]:+.3f} max={qs[6]:+.2f}")
    for i in range(0, w, 2):
        b = "#" * int(30 * h[i] / mx)
        print(f"  {e[i]:+.3f} {b}")


def main():
    print("== histogramas ==")
    vecs = {}
    for nm, d in PTS.items():
        if not os.path.isdir(d):
            print(f"{nm}: ausente, pulo"); continue
        v = flat(d)
        vecs[nm] = v
        hist_ascii(v, nm)
    # PCA so dentro da mesma arquitetura (d96 tem 9 pontos!)
    names = [n for n in vecs if DSETS.get(n) == "d96"]
    X = np.stack([vecs[n] for n in names])
    Xc = X - X.mean(0)
    G = Xc @ Xc.T / X.shape[1]
    w, V = np.linalg.eigh(G)
    o = np.argsort(w)[::-1]
    print(f"\n== PCA d96: var PC1={w[o[0]]/w.sum():.1%} PC2={w[o[1]]/w.sum():.1%} ==")
    P = V[:, o[:2]] * np.sqrt(w[o[:2]])
    xs = {n: (float(P[i, 0]), float(P[i, 1])) for i, n in enumerate(names)}
    # ascii scatter
    W2, H2 = 60, 18
    allx = [p[0] for p in xs.values()]
    ally = [p[1] for p in xs.values()]
    grid = [[" "] * W2 for _ in range(H2)]
    for n, (x, y) in xs.items():
        gx = int((x - min(allx)) / max(max(allx) - min(allx), 1e-30) * (W2 - 1))
        gy = int((y - min(ally)) / max(max(ally) - min(ally), 1e-30) * (H2 - 1))
        grid[H2 - 1 - gy][gx] = n[0].upper()
    for row in grid:
        print("".join(row))
    print("labels:", {n[0].upper(): n for n in names})
    # SVG
    S = 520
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{S}" height="{S}" style="background:#0b0e14">']
    cols = {"f0": "#7dd3fc", "drop": "#86efac", "sq": "#f87171", "neg": "#fbbf24",
            "shuf": "#c084fc", "s05": "#fb7185", "s20": "#ef4444",
            "n001": "#a5b4fc", "n01": "#f472b6"}
    for n, (x, y) in xs.items():
        px = 40 + (x - min(allx)) / max(max(allx) - min(allx), 1e-30) * (S - 80)
        py = 40 + (y - min(ally)) / max(max(ally) - min(ally), 1e-30) * (S - 80)
        c = cols.get(n, "#fff")
        r = 9 if n in ("f0", "neg") else 7
        svg.append(f'<circle cx="{px:.0f}" cy="{py:.0f}" r="{r}" fill="{c}"/>'
                   f'<text x="{px+11:.0f}" y="{py+4:.0f}" fill="#e5e7eb" font-size="13">{n}</text>')
    svg.append("</svg>")
    open("/tmp/basin_map.svg", "w").write("\n".join(svg))
    print("SVG -> /tmp/basin_map.svg")


if __name__ == "__main__":
    main()
