#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""morph_segment.py — segmentacao morfologica conservadora p/ PT (stdlib).
Parte palavra em [prefixo] + radical + [sufixo] com listas fechadas; so corta
se o radical restante tem >=3 letras. Erra p/ NAO cortar (conservador):
sub-segmentar preserva o BPE normal; sobre-segmentar quebra palavras validas.
Uso: .venv-numpy/bin/python3 scripts/morph_segment.py  (auto-teste embutido)
"""
import re

PREFIXOS = sorted(["des", "re", "pre", "pos", "sub", "super", "hiper", "anti",
                   "contra", "entre", "sobre", "ante", "in", "im", "dis", "co",
                   "trans", "inter", "extra", "ultra", "semi", "pseudo", "arqui",
                   "bem", "mal"], key=len, reverse=True)
SUFIXOS = sorted(["mente", "zinho", "zinha", "cao", "coes", "dade", "idade",
                  "ismo", "ista", "eiro", "eira", "dor", "dora", "oso", "osa",
                  "ados", "idos", "adas", "idas", "ado", "ido", "ada", "ida", "ando", "endo", "indo", "ava",
                  "iam", "aram", "eram", "iram", "asse", "esse", "isse",
                  "aria", "eria", "iria", "avel", "ivel", "ura", "ense",
                  "es", "as", "os", "am", "em", "ar", "er", "ir", "ou",
                  "eu", "iu", "ia"], key=len, reverse=True)

WR = re.compile(r"[^\W\d_]+", re.UNICODE)


def segmenta(palavra):
    w = palavra.lower()
    if len(w) < 7:
        return [palavra]
    pre, pos = 0, len(palavra)
    for p in PREFIXOS:
        if w.startswith(p) and len(w) - len(p) >= 4:
            pre = len(p)
            break
    suf = 0
    for s in SUFIXOS:
        if w.endswith(s) and (len(w) - pre) - len(s) >= 3:
            suf = len(s)
            break
    segs = []
    if pre:
        segs.append(palavra[:pre])
    segs.append(palavra[pre:len(palavra) - suf if suf else len(palavra)])
    if suf:
        segs.append(palavra[len(palavra) - suf:])
    if len(segs) < 2:
        return [palavra]
    return segs


def test():
    casos = [("desconhecidos", ["des", "conhec", "idos"]),
             ("paralelepipedo", None),   # sem afixo: nao corta
             ("casa", None),             # curta: nao corta
             ("felizmente", ["feliz", "mente"])]
    for w, esp in casos:
        r = segmenta(w)
        ok = (esp is None and r == [w]) or (r == esp)
        print(f"  {w}: {r} {'OK' if ok else 'REVISA (esperava %s)' % esp}")


if __name__ == "__main__":
    test()
