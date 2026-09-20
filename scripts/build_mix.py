#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""build_mix.py — monta o corpus MISTO (prosa + ptwiki) no espaço BPE.

Por que misturar: as fases sequenciais criaram especialistas que apagam uns aos
outros (math piorou a prosa: bpb 8,18 -> 8,74; a fase 4 responde em Fortran a um
prompt em português). Misturar é o remédio, e diversidade de gênero é requisito.

Convenção (a MESMA do corpus byte-level, para não inventar formato novo):
  - o stream de tokens é cortado em linhas de exatamente T tokens, cada uma
    começando com BOS (8188) -- igual ao prepare_prose.py, que cortava o texto a
    cada 2.048 bytes. Fronteira de documento borra dentro da linha, como sempre;
    o docmask (attn_bwd_doc, pendente) é o que resolve isso depois.
  - validação = as linhas de val da PROSA, verbatim. Assim o bpb em prosa
    literária held-out continua comparável com todos os números anteriores, que é
    o alvo da avaliação. Wiki held-out é descartado (não entra em val).

Split da wiki: por hash do TÍTULO do artigo (mesma disciplina do split por livro:
re-extrair ou re-limpar não pode reembaralhar o holdout).

Uso:
  build_mix.py --wiki-tokens 300e6 --prose-epochs 1 --T 1024 --out /tmp/mix/mix25.txt
  build_mix.py --sample --wiki-tokens 3e6          # valida rápido na amostra
"""
import argparse
import math
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_driver import load_enc  # noqa: E402

BOS = 8188
VAL_FRAC = 0.02          # fração de artigos da wiki reservada (descartada)


def held_out(title, frac=VAL_FRAC):
    h = int(hashlib.sha256(title.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 10000) < frac * 10000


def _lines(path, n_train, skip_bos, stride=1):
    pos = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if n_train is not None and i >= n_train:
                break
            if (pos % stride) != 0:
                pos += 1
                continue
            pos += 1
            parts = line.split()
            yield [int(x) for x in parts[1:]] if skip_bos else \
                [int(x) for x in parts]


def _count_toks(path, n_train):
    t = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if n_train is not None and i >= n_train:
                break
            t += len(line.split()) - 1
    return t


def _count_lines(path, n_train):
    n = 0
    with open(path) as f:
        for i, _ in enumerate(f):
            if n_train is not None and i >= n_train:
                break
            n += 1
    return n


def prose_stream(path, n_train, epochs, skip_bos=True):
    """Linhas BPE como stream de ids. Lazy (rele por epoca, ~zero RAM).
    Entrega epochs*estoque tokens (+-1 linha): passes cheios + resto por
    quantum em espaco de tokens (uniforme E exato -- o stride arredondado
    entregava ate 10% a menos e quebrava o portao de 95%)."""
    if epochs <= 0:
        return
    stock = _count_toks(path, n_train)
    if stock <= 0:
        return
    full = int(epochs)
    for _ in range(full):
        for ids in _lines(path, n_train, skip_bos, stride=1):
            yield from ids
    rem = epochs - full
    if rem > 1e-12:
        # quantum em UNIDADES DE LINHA: K linhas uniformemente espacadas ~= 
        # alvo em tokens (+- poucas linhas). (Quantum em tokens emitia Nr.
        # errado de linhas -- unidade importa.)
        n = _count_lines(path, n_train)
        klines = max(1, round(n * rem))
        cum = 0
        for ids in _lines(path, n_train, skip_bos, stride=1):
            if (cum + 1) * klines // max(n, 1) > cum * klines // max(n, 1):
                yield from ids
            cum += 1


def wiki_stream(enc, wiki_dir, budget_tokens, sample=False):
    """Artigos da wiki (texto -> ids BPE), pulando os held-out."""
    used = 0
    files = sorted(glob.glob(os.path.join(wiki_dir, "*", "*")))
    if sample:
        files = files[: max(1, len(files) // 50)]
    for p in files:
        for line in open(p, encoding="utf-8"):
            if used >= budget_tokens:
                return
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = d.get("text")
            if not t or held_out(d.get("title", "")):
                continue
            ids = enc.encode_ordinary(t)
            used += len(ids)
            yield from ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prose", default="/tmp/prose/prose_bpe_all.txt")
    ap.add_argument("--prose-ntrain", type=int, default=59385)
    ap.add_argument("--prose2", default="",
                    help="segunda fonte literaria (ex. wikisource BPE); "
                    "orcamento literario dividido por estoque")
    ap.add_argument("--prose2-ntrain", type=int, default=0,
                    help="linhas de --prose2 (0 = todas)")
    ap.add_argument("--prose-tokens", type=float, default=0.0,
                    help="orçamento de prosa em TOKENS (0 = usa --prose-epochs)")
    ap.add_argument("--prose-epochs", type=float, default=1.0)
    ap.add_argument("--wiki-dir", default="/tmp/ptwiki/wx_full")
    ap.add_argument("--wiki-tokens", type=float, default=300e6)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--out", default="/tmp/mix/mix25.txt")
    ap.add_argument("--val-out", default="/tmp/mix/mix_val.txt")
    ap.add_argument("--manifest", default="/tmp/mix/manifest.json")
    ap.add_argument("--block", type=int, default=65536,
                    help="granularidade da intercalação (tokens por bloco)")
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()

    enc = load_enc()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_prose_tot = 0
    with open(args.prose) as f:
        for i, line in enumerate(f):
            if i >= args.prose_ntrain:
                break
            n_prose_tot += len(line.split()) - 1
    if args.prose_tokens > 0:
        n_prose_target = int(args.prose_tokens)   # explícito: int() truncava a fração para 0
    else:
        n_prose_target = int(n_prose_tot * args.prose_epochs)

    # orcamento literario dividido por ESTOQUE entre as fontes (prose:ws);
    # streams entregam EXATAMENTE seus orcamentos (stride p/ fracao de epoca,
    # repeticao p/ >1). Orcamento 0 => stream vazio (sem o bug do npass>=1,
    # que emitia prosa com alvo 0 e deixava o manifesto mentir).
    n_prose2_tot = 0
    if args.prose2:
        n2lim = args.prose2_ntrain if args.prose2_ntrain > 0 else None
        with open(args.prose2) as f:
            for i, line in enumerate(f):
                if n2lim is not None and i >= n2lim:
                    break
                n_prose2_tot += len(line.split()) - 1
    lit_tot = n_prose_tot + n_prose2_tot
    if args.prose2 and lit_tot:
        p1_target = int(n_prose_target * n_prose_tot / lit_tot)
        p2_target = n_prose_target - p1_target
    else:
        p1_target, p2_target = n_prose_target, 0
    w_target = int(args.wiki_tokens)
    n2 = args.prose2_ntrain if (args.prose2 and args.prose2_ntrain > 0) else None
    streams = [
        [prose_stream(args.prose, args.prose_ntrain,
                      p1_target / max(n_prose_tot, 1)), p1_target, 0, False],
        [prose_stream(args.prose2, n2, p2_target / max(n_prose2_tot, 1))
         if args.prose2 else None, p2_target, 0, False],
        [wiki_stream(enc, args.wiki_dir, args.wiki_tokens, args.sample),
         w_target, 0, False],
    ]
    for s in streams:
        s[3] = (s[0] is None) or (s[1] <= 0)

    # intercalacao ponderada uniforme (cota por blocos com carry): proporcao
    # exata no longo prazo para QUALQUER f. Corrige o front-load do ratio
    # inteiro antigo (f25 saia 50/50 na 1a metade e prosa pura na 2a).
    weights = [p1_target, p2_target, w_target]
    wtot = sum(weights)
    carry = [0.0, 0.0, 0.0]
    nrows = 0
    buf = [BOS]
    buf_len = 1
    with open(args.out, "w") as out:
        def flush_row():
            nonlocal buf, buf_len, nrows
            out.write(" ".join(map(str, buf)) + "\n")
            nrows += 1
            buf = [BOS]
            buf_len = 1

        while not all(s[3] for s in streams):
            for i, s in enumerate(streams):
                if s[3]:
                    continue
                quota = args.block * s[1] / wtot + carry[i] if wtot else 0
                take, carry[i] = int(quota), quota - int(quota)
                take = min(take, s[1] - s[2])  # sem estouro de bloco: real~=alvo
                for _ in range(take):
                    try:
                        buf.append(next(s[0]))
                    except StopIteration:
                        s[3] = True
                        break
                    s[2] += 1
                    buf_len += 1
                    if buf_len - 1 >= args.T:
                        flush_row()
                if s[2] >= s[1]:
                    s[3] = True
            if args.sample and nrows > 2000:
                break
        # a linha final parcial é descartada: o treinador exige T fixo
        if buf_len > 1:
            print(f"  (descartando linha final parcial com {buf_len-1} tokens)")

    # val: as linhas de val da prosa, verbatim (bpb comparável com o histórico)
    nval = 0
    with open(args.val_out, "w") as fv:
        with open("/tmp/prose/prose_bpe_val.txt") as f:
            for line in f:
                fv.write(line)
                nval += 1

    tok_out = nrows * args.T
    nv = 0
    with open(args.out) as f:
        for line in f:
            nv += len(line.split()) - 1
    a1, a2, wp = streams[0][2], streams[1][2], streams[2][2]
    man = {
        "T": args.T, "bos": BOS, "rows": nrows, "val_rows": nval,
        "prose_tokens_target": p1_target, "prose2_tokens_target": p2_target,
        "prose_epochs": args.prose_epochs,
        "wiki_tokens_target": w_target,
        "mix_weights_prose_prose2_wiki": weights,
        "wiki_held_out_frac_by_title_hash": VAL_FRAC,
        "tokens_total": nv,
        "tokens_prose_actual": a1, "tokens_prose2_actual": a2,
        "tokens_wiki_actual": wp,
        "val": "prosa literal held-out (comparável ao histórico)",
        "prose_src": args.prose, "prose2_src": args.prose2 or None,
        "wiki_src": args.wiki_dir,
    }
    with open(args.manifest, "w") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    print(json.dumps(man, indent=2, ensure_ascii=False))
    want = p1_target + p2_target + w_target
    if man["tokens_total"] < 0.95 * want:
        print(f"  FALHA: emitidos {man['tokens_total']/1e6:.1f}M, orçamento {want/1e6:.1f}M "
              f"(prosa {a1/1e6:.1f}M, prose2 {a2/1e6:.1f}M, wiki {wp/1e6:.1f}M)")
        sys.exit(1)


if __name__ == "__main__":
    main()
