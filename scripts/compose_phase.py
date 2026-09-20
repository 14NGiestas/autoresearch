#!/usr/bin/env python3
"""compose_phase.py — o espaço de fases da composição, num só SVG.

Lê TODOS os history.json que sobreviveram (scratch de /tmp) e desenha:

  A) bpb vs K        — a composição escala com o número de workers?
  B) bpb vs syncs    — a frequência de sincronização satura?
  C) grade 3x2       — o que carregar/zerar momentos x ter ou não outer
  D) veredito        — a comparação JUSTA (mesmo otimizador dos dois lados)

Sem matplotlib (não existe no venv): SVG escrito à mão, sem dependência.

Uso: scripts/compose_phase.py [--out /tmp/compose_phase.svg]
"""

import argparse
import glob
import json
import os
import sys

# Referências medidas (não vêm dos history.json: são runs únicos)
REF = {
    "single_wallclock": 2.44475,   # 1 máquina, 1952 passos (mesmo wall-clock)
    "single_ideal": 2.12000,       # 1 máquina, 7808 passos (mesmo compute)
    "oneshot": 2.50360,            # 1 rodada de 1952 (fundir no fim)
}

W, H = 1180, 820
FONT = 'font-family="DejaVu Sans Mono, monospace" font-size="12"'


# Runs LEGADOS (criados antes de --reset-moments/--carry-per-worker existirem).
# O caminho --outer nao escrevia o estado do Adam na epoca, entao "outer" nessas
# runs implicava RESET na pratica -- foi inclusive o confundimento que a gente
# descobriu e controlou. Sem este mapa o filtro por flag classifica errado e o
# grafico pega a run errada (foi o que aconteceu: 2.42092 em vez de 2.33183).
LEGACY = {
    "dl/122": dict(reset=True), "dl/488": dict(reset=True),
    "dlctrl/resetmom": dict(reset=True), "dlctrl/carrymom": dict(reset=False),
    "fed2/t122": dict(reset=False), "fed2/t488": dict(reset=False),
    "dlsweep/0.1": dict(reset=True), "dlsweep/1.5": dict(reset=True),
}


def load():
    """todos os runs com config, indexados por (k, tau, outer, reset, pw)."""
    rows = []
    for pat in ("/tmp/ctrl/*/history.json", "/tmp/ctrl2/*/history.json",
                "/tmp/ss/*/history.json", "/tmp/rst/*/history.json",
                "/tmp/dl/*/history.json", "/tmp/fed2/*/history.json",
                "/tmp/dlctrl/*/history.json", "/tmp/dlsweep/*/history.json"):
        for p in sorted(glob.glob(pat)):
            try:
                d = json.load(open(p))
            except (OSError, ValueError):
                continue
            h = d.get("history") or []
            c = d.get("config") or {}
            if not h:
                continue
            key = p.replace("/tmp/", "").replace("/history.json", "")
            rows.append({
                "path": key,
                "pool": os.path.basename(str(c.get("rows") or "?")) or "?",
                "k": c.get("k"), "tau": c.get("tau"), "rounds": c.get("rounds"),
                "outer": (c.get("outer") or "none"),
                "reset": LEGACY.get(key, {}).get("reset", bool(c.get("reset_moments"))),
                "pw": bool(c.get("carry_per_worker")),
                "final": h[-1]["bpb"], "n": len(h),
            })
    return rows


def find(rows, **want):
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in want.items()):
            out.append(r)
    return out


def best(rows, pool="rows_f0.npy", **want):
    """Melhor bpb entre os runs que casam -- FILTRANDO o pool de dados.

    Nao misturar sorteios: 2.2847 e' do pool perm1 e 2.3318 do f0; tomar o minimo
    entre os dois seria comparar draws diferentes no mesmo ponto do grafico."""
    c = [r for r in find(rows, **want) if r["pool"] == pool]
    return min((r["final"] for r in c), default=None)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/compose_phase.svg")
    a = ap.parse_args()
    rows = load()
    if not rows:
        sys.exit("compose_phase: nenhum history.json em /tmp (scratch limpo?)")

    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (W, H),
         '<rect width="100%%" height="100%%" fill="white"/>',
         '<text x="24" y="30" font-size="17" %s>Espaco de fases da composicao (K, sync, momentos, outer)</text>' % FONT,
         '<text x="24" y="50" font-size="12" %s>modelo 3M (d96), CPU, holdout fixa; runs lidos de %d history.json</text>' % (FONT, len(rows))]
    y0 = 70

    def axis(x, y, w, h, xlab, ylab, lo, hi, xlo, xhi):
        s.append('<rect x="%d" y="%d" width="%d" height="%d" fill="#fafafa" stroke="#ddd"/>' % (x, y, w, h))
        for k in range(5):
            yy = y + h - h * k / 4
            v = lo + (hi - lo) * k / 4
            s.append('<line x1="%d" y1="%.0f" x2="%d" y2="%.0f" stroke="#eee"/>' % (x, yy, x + w, yy))
            s.append('<text x="%d" y="%.0f" text-anchor="end" %s fill="#888">%.2f</text>' % (x - 6, yy + 4, FONT, v))
        s.append('<text x="%d" y="%d" %s fill="#555">%s</text>' % (x, y - 6, FONT, esc(ylab)))
        s.append('<text x="%d" y="%d" text-anchor="middle" %s fill="#555">%s</text>' % (x + w // 2, y + h + 18, FONT, esc(xlab)))
        return lambda v: y + h - h * (v - lo) / (hi - lo)          # y(v_bpb)
    # escala x genérica
    def xs(x, w, xlo, xhi):
        return lambda v: x + w * (v - xlo) / (xhi - xlo)

    YLO, YHI = 2.25, 2.85

    # ---- A) bpb vs K -----------------------------------------------------
    ax, ay, aw, ah = 90, y0 + 20, 480, 250
    yv = axis(ax, ay, aw, ah, "K (workers, tau=122, 16 syncs, reset+outer)",
              "bpb (holdout)", YLO, YHI, 1, 8)
    for v, lab, col in ((REF["single_wallclock"], "1 maquina (mesmo wall-clock)", "#c00"),
                        (REF["oneshot"], "fundir no fim (1 sync)", "#e80"),
                        (REF["single_ideal"], "1 maquina (mesmo compute)", "#080")):
        s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" stroke-dasharray="5,4"/>'
                 % (ax, yv(v), ax + aw, yv(v), col))
        s.append('<text x="%d" y="%.0f" %s fill="%s">%s = %.4f</text>' % (ax + aw + 6, yv(v) + 4, FONT, col, esc(lab), v))
    pts = []
    for k in (1, 4, 8):
        v = best(rows, k=k, tau=122, outer="nesterov", reset=True) if k > 1 else \
            best(rows, k=1, tau=122, outer="nesterov", reset=False)
        if v:
            pts.append((k, v))
    xv = xs(ax, aw, 0, 9)
    for i in range(len(pts) - 1):
        (k1, v1), (k2, v2) = pts[i], pts[i + 1]
        s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#0366d6" stroke-width="2"/>'
                 % (xv(k1), yv(v1), xv(k2), yv(v2)))
    for k, v in pts:
        s.append('<circle cx="%.1f" cy="%.1f" r="5" fill="#0366d6"/>' % (xv(k), yv(v)))
        s.append('<text x="%.1f" y="%.1f" text-anchor="middle" %s fill="#0366d6">K=%d %.4f</text>'
                 % (xv(k), yv(v) - 10, FONT, k, v))
    s.append('<text x="%d" y="%d" %s fill="#333">A) K=8 empata com K=4 gastando 2x compute: o ganho satura.</text>'
             % (ax, ay + ah + 40, FONT))

    # ---- B) bpb vs numero de syncs ---------------------------------------
    bx, by, bw, bh = 660, y0 + 20, 470, 250
    yv2 = axis(bx, by, bw, bh, "syncs por run (K=4; log2)", "bpb (holdout)", YLO, YHI, 3, 36)
    xb = lambda n: bx + bw * (__import__("math").log2(n) - __import__("math").log2(4)) / \
        (__import__("math").log2(32) - __import__("math").log2(4))
    for lab, want, col in (("reset+outer lr=0.7", dict(k=4, outer="nesterov", reset=True), "#0366d6"),
                           ("K=1, momentos carregados", dict(k=1, outer="none", reset=False), "#c00")):
        pts = []
        for syn in (4, 8, 16, 32):
            v = best(rows, tau=1952 // syn, **want)
            if v is None and want.get("outer") == "nesterov":
                # 4 syncs (tau=488) so' existe com outer e momentos CARREGADOS
                v = min((r["final"] for r in find(rows, k=4, tau=488, outer="nesterov")
                         if r["pool"] == "rows_f0.npy"), default=None)
                if v:
                    pts.append((syn, v))
                    s.append('<circle cx="%.1f" cy="%.1f" r="5" fill="none" stroke="#0366d6" stroke-width="2"/>'
                             % (bx + bw * (2 - 2) / 3, yv2(v)))   # log2(4)=2 -> x inicial
                    continue
            if v:
                pts.append((syn, v))
        for i in range(len(pts) - 1):
            s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="2"/>'
                     % (xb(pts[i][0]), yv2(pts[i][1]), xb(pts[i + 1][0]), yv2(pts[i + 1][1]), col))
        for n, v in pts:
            s.append('<circle cx="%.1f" cy="%.1f" r="4" fill="%s"/>' % (xb(n), yv2(v), col))
        if pts:
            s.append('<text x="%.1f" y="%.1f" %s fill="%s">%s</text>'
                     % (xb(pts[-1][0]) - 10, yv2(pts[-1][1]) + (16 if col == "#c00" else -10), FONT, col, lab))
    s.append('<text x="%d" y="%d" %s fill="#333">B) Composta: satura em ~16 syncs. Sem compor (K=1): monotona no</text>'
             % (bx, by + bh + 40, FONT))
    s.append('<text x="%d" y="%d" %s fill="#333">   numero de RODADAS (o custo e por invocacao, nao por sincronizacao).</text>'
             % (bx, by + bh + 56, FONT))

    # ---- C) grade 3x2 ----------------------------------------------------
    gx, gy = 90, y0 + 380
    s.append('<text x="%d" y="%d" %s fill="#555">C) bpb (K=4, tau=122) por tratamento do estado do otimizador x outer</text>'
             % (gx, gy - 8, FONT))
    cols = [("sem outer", dict(outer="none")), ("+ outer 0.7", dict(outer="nesterov"))]
    rowdefs = [("mediados", dict(reset=False, pw=False)),
               ("por worker", dict(reset=True, pw=True)),
               ("ZERADOS", dict(reset=True, pw=False))]
    cw, ch = 190, 46
    for j, (clab, cw_) in enumerate(cols):
        s.append('<text x="%d" y="%d" text-anchor="middle" %s fill="#555">%s</text>'
                 % (gx + 150 + j * cw + cw // 2, gy + 14, FONT, esc(clab)))
    for i, (rlab, rw) in enumerate(rowdefs):
        s.append('<text x="%d" y="%d" %s fill="#555">%s</text>' % (gx, gy + 40 + i * ch + 24, FONT, esc(rlab)))
        for j, (_clab, cw_) in enumerate(cols):
            v = best(rows, k=4, tau=122, **dict(rw, **cw_))
            fill = "#fff"
            if v:
                # verde = melhor, vermelho = pior, dentro da coluna
                colvals = [best(rows, k=4, tau=122, **dict(rw2, **cw_)) for _, rw2 in rowdefs]
                colvals = [x for x in colvals if x]
                lo, hi = min(colvals), max(colvals)
                t = 0.0 if hi == lo else (v - lo) / (hi - lo)
                fill = "rgb(%d,%d,%d)" % (int(230 * t + 25), int(230 * (1 - t) + 25), 90)
            s.append('<rect x="%d" y="%d" width="%d" height="%d" fill="%s" stroke="#ccc"/>'
                     % (gx + 150 + j * cw, gy + 28 + i * ch, cw - 10, ch - 8, fill))
            s.append('<text x="%d" y="%d" text-anchor="middle" %s>%s</text>'
                     % (gx + 150 + j * cw + (cw - 10) // 2, gy + 28 + i * ch + 26, FONT,
                        ("%.5f" % v) if v else "--"))

    # ---- D) veredito -----------------------------------------------------
    dx, dy = 660, y0 + 380
    lines = ["D) Veredito (mesmo wall-clock, 1952 passos/worker)"]
    comp_f0 = best(rows, k=4, tau=122, outer="nesterov", reset=True)
    comp_p1 = 2.28472
    s1 = best(rows, k=1, tau=122, outer="nesterov", reset=False)
    s2 = 2.45665
    lines += [
        "  composta  (K=4, reset+outer): %.5f (f0)  %.5f (perm1)" % (comp_f0, comp_p1),
        "  1 maquina + MESMO otimizador: %.5f (f0)  %.5f (perm1)" % (s1, s2),
        "  1 maquina sem outer         : %.5f" % REF["single_wallclock"],
        "",
        "  ganho da composicao sobre a regua JUSTA:",
        "    f0    %+.5f" % ((s1 or 0) - (comp_f0 or 0)),
        "    perm1 %+.5f" % (s2 - comp_p1),
        "",
        "  FRACAO do ganho ideal capturada (K=4): %.0f%%" %
        (100 * (REF["single_wallclock"] - comp_f0) / (REF["single_wallclock"] - REF["single_ideal"])),
        "  K=8 (2x compute): %.5f -> saturacao" % best(rows, k=8, tau=122, outer="nesterov", reset=True),
    ]
    for i, t in enumerate(lines):
        s.append('<text x="%d" y="%d" %s fill="%s">%s</text>'
                 % (dx, dy + 8 + i * 19, FONT, "#0366d6" if t.startswith("  composta") else "#333", esc(t)))

    s.append('</svg>')
    with open(a.out, "w") as fh:
        fh.write("\n".join(s))
    print("svg: %s (%d bytes, %d runs lidos)" % (a.out, os.path.getsize(a.out), len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
