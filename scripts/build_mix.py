#!/usr/bin/env python3
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


def prose_stream(path, n_train, epochs, skip_bos=True):
    """Linhas do corpus de prosa (já em BPE) como um stream de ids."""
    rows, total = [], 0
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n_train:
                break
            parts = line.split()
            ids = [int(x) for x in parts[1:]] if skip_bos else [int(x) for x in parts]
            rows.append(ids)
            total += len(ids)
    for _ in range(epochs):
        for r in rows:
            yield from r


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
    n_prose_target = 0
    with open(args.prose) as f:
        for i, line in enumerate(f):
            if i >= args.prose_ntrain:
                break
            n_prose_target += len(line.split()) - 1
    n_prose_target = int(n_prose_target * args.prose_epochs)

    pstream = prose_stream(args.prose, args.prose_ntrain, int(args.prose_epochs))
    wstream = wiki_stream(enc, args.wiki_dir, args.wiki_tokens, args.sample)

    # intercalação por blocos: mantém a mistura suave sem carregar tudo na RAM
    ratio = max(1, round(args.wiki_tokens / max(n_prose_target, 1)))
    wp = 0
    ww = 0
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

        n_done = 0
        while n_done < n_prose_target or wp < args.wiki_tokens:
            # um bloco de prosa, depois 'ratio' blocos de wiki (mistura ~1:ratio)
            if n_done < n_prose_target:
                take = min(args.block, n_prose_target - n_done)
                for _ in range(take):
                    try:
                        buf.append(next(pstream))
                    except StopIteration:
                        break
                    buf_len += 1
                    if buf_len - 1 >= args.T:
                        flush_row()
                n_done += take
            if wp < args.wiki_tokens:
                for _ in range(ratio):
                    if wp >= args.wiki_tokens:
                        break
                    for _ in range(min(args.block, int(args.wiki_tokens - wp))):
                        try:
                            buf.append(next(wstream))
                        except StopIteration:
                            wp = args.wiki_tokens
                            break
                        wp += 1
                        buf_len += 1
                        if buf_len - 1 >= args.T:
                            flush_row()
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
    man = {
        "T": args.T, "bos": BOS, "rows": nrows, "val_rows": nval,
        "prose_tokens_target": n_prose_target, "prose_epochs": args.prose_epochs,
        "wiki_tokens_target": int(args.wiki_tokens),
        "interleave_ratio_prose_to_wiki": f"1:{ratio}",
        "wiki_held_out_frac_by_title_hash": VAL_FRAC,
        "tokens_total": nv,
        "tokens_prose_actual": n_done, "tokens_wiki_actual": wp,
        "val": "prosa literal held-out (comparável ao histórico)",
        "prose_src": args.prose, "wiki_src": args.wiki_dir,
    }
    with open(args.manifest, "w") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    print(json.dumps(man, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
