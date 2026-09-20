#!/usr/bin/env python3
"""texture_report.py — painel de TEXTURA de um texto gerado (sem dicionario).

Por que existe: os scripts neologism_rate.py/archaism_rate.py precisam de
dicionario e vocabulario de treino; este painel mede o que da' para medir do
proprio texto, e serve para acompanhar a ESCADA da geracao:

  1) chao de bytes   — salada, sem estrutura de palavra      <- onde estamos
  2) gerador usavel  — palavras/pontuacao coerentes, sem laco
  3) raciocinador    — revisao/auto-correcao (o "aha" do R1, fora do nosso alcance)

Metricas (todas explicaveis, nenhuma precisa de dicionario):
  * fracao alfa:     fracao de caracteres alfabeticos (chao de bytes ~0.4-0.6;
                     texto real ~0.75-0.85)
  * fracao palavra:  fracao de "palavras" (runs de letras cercadas de espaco)
                     que tem tamanho plausivel (2..12, sem misturar digito)
  * distinct-1/2/3:  type/token ratio de n-gramas de CARACTERE (diversidade)
  * maior laco:      maior n-grama de 8 caracteres repetido em sequencia
  * rep-frac:        fracao do texto dentro de repeticoes de 8-gramas
  * byte-alto:       fracao de bytes com codigo > 127 (sinal de fallback de byte)

Uso: scripts/texture_report.py ARQ.txt [--json]
     gerar | scripts/texture_report.py - 
"""

import argparse
import json
import sys
from collections import Counter


def panel(text):
    n = len(text)
    if n == 0:
        return {"n": 0}
    alpha = sum(c.isalpha() for c in text) / n
    high = sum(ord(c) > 127 for c in text) / n
    words = []
    cur = []
    for ch in text:
        if ch.isalpha():
            cur.append(ch)
        else:
            if cur:
                words.append("".join(cur)); cur = []
            if ch == " ":
                words.append("")
    if cur:
        words.append("".join(cur))
    # "palavra plausivel": run de letras 2..12, so' ASCII (fallback de byte cria
    # runs com acento/simbolo; texto real nao)
    runs = [w for w in words if w]
    plaus = [w for w in runs if 2 <= len(w) <= 12 and w.isascii()]
    out = {
        "n": n,
        "alpha": round(alpha, 3),
        "byte_alto": round(high, 3),
        "palavras": len(runs),
        "palavra_plausivel": round(len(plaus) / len(runs), 3) if runs else 0.0,
    }
    for k in (1, 2, 3):
        grams = [text[i:i + k] for i in range(n - k + 1)]
        out["distinct_%d" % k] = round(len(set(grams)) / max(1, len(grams)), 3)
    # maior laco: 8-gramas repetidos consecutivamente -- SO' os que contem
    # caractere nao-branco. Sem isso, linhas de separador/tabela de markdown real
    # contam como "laco" (deu 15 no README, que e' texto legitimo) e a metrica
    # deixa de distinguir salada de texto.
    k = 8
    grams = [text[i:i + k] for i in range(max(0, n - k + 1))]
    grams = [g for g in grams if g.strip()]
    best = 1
    cur = 1
    for i in range(1, len(grams)):
        cur = cur + 1 if grams[i] == grams[i - 1] else 1
        best = max(best, cur)
    cnt = Counter()
    for i in range(len(grams)):
        cnt[grams[i]] += 1
    out["maior_laco"] = best
    rep = sum(c for c in cnt.values() if c > 1)
    out["rep_frac"] = round(rep / max(1, len(grams)), 3)
    return out


# CALIBRACAO (medida, nao inventada) -- controles deste repo:
#   prosa real (program.md)      alpha 0.698  byte_alto 0.005  palavra_plaus 0.956  distinct2 0.111
#   template   (chat_text.rsp)   alpha 0.542  byte_alto 0.000  palavra_plaus 0.900  distinct2 0.624
#   MODELO 3M treinado (2.33bpb) alpha 0.362  byte_alto 0.146  palavra_plaus 0.204  distinct2 0.845
# A metrica que separa limpo e' palavra_plausivel (0.20 do modelo contra 0.90-0.96
# do texto real); distinct_2 e' INVERTIDO (salada tem mais porque nao tem a
# redundancia de lingua). Estagios, portanto, sao relativos a estes controles.
STAGE = [
    # (condicao, rotulo)  -- ordem importa: primeiro o que reprova
    (lambda p: p.get("n", 0) < 20, "empty or too short"),
    (lambda p: p["palavra_plausivel"] < 0.50 or p["byte_alto"] > 0.10,
     "1) byte floor (salad) -- plausible_word < 0.50"),
    (lambda p: p["palavra_plausivel"] < 0.80 or p["maior_laco"] > 3 or p["rep_frac"] > 0.30,
     "transition: structure appears (0.50-0.80), a loop or noise remains"),
    (lambda p: True, "2) usable generator (words and punctuation are correct)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="-")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    text = sys.stdin.read() if a.path == "-" else open(a.path).read()
    # tira a marca do prompt do repl
    text = "\n".join(l[4:] if l.startswith(">>> ") else l for l in text.splitlines())
    p = panel(text)
    stage = next(lbl for cond, lbl in STAGE if cond(p))
    p["estagio"] = stage
    if a.json:
        print(json.dumps(p))
        return 0
    print("=== TEXTURA (%d chars) -> estagio: %s" % (p["n"], stage))
    for k in ("alpha", "byte_alto", "palavras", "palavra_plausivel", "distinct_1",
              "distinct_2", "distinct_3", "maior_laco", "rep_frac"):
        print("  %-18s %s" % (k, p.get(k)))
    print("\n  CONTROLES deste repo (medidos): prosa real palavra_plaus 0.956 / byte_alto 0.005;")
    print("  modelo 3M treinado palavra_plaus 0.204 / byte_alto 0.146. A escada:")
    print("  1) chao de bytes  ->  2) gerador usavel (palavra_plausivel > 0.80)  ->")
    print("  3) raciocinador (o 'aha' do R1: revisao/auto-correcao; fora do nosso")
    print("  alcance com 3M params / ctx 1024 -- o que da' para fazer e' medir o degrau 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
