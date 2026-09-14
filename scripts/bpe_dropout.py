#!/usr/bin/env python3
"""bpe_dropout.py — BPE-dropout puro (Provilkov et al.): merge com prob. p é
pulado, gerando segmentações alternativas da mesma palavra. p=0 deve ser
bit-idêntico ao tiktoken ordinário (portão). stdlib + tiktoken ranks.
Uso: .venv-numpy/bin/python3 scripts/bpe_dropout.py --p 0.1 --limit 300
"""
import argparse
import pickle
import random
import sys

sys.path.insert(0, "scripts")


def load_ranks():
    enc = pickle.load(open("/home/pauli/.cache/autoresearch/tokenizer/tokenizer.pkl", "rb"))
    return enc._mergeable_ranks


def encode_dropout(data: bytes, ranks: dict, p: float, rng: random.Random):
    # Provilkov: por palavra, cada REGRA de merge (por rank-resultado) cai com
    # prob p; depois BPE guloso deterministico com as regras restantes.
    toks = [bytes([b]) for b in data]
    if len(toks) < 2:
        return [ranks[t] for t in toks]
    dropped = set()
    if p > 0:
        seen = set()
        for i in range(len(toks) - 1):
            pass  # regras vistas sao descobertas sob demanda abaixo
        # coleta ranks candidatos sobre bytes iniciais (superconjunto seguro)
        for i in range(len(toks)):
            for j in range(i + 1, min(i + 8, len(toks)) + 1):
                m = b"".join(toks[i:j])
                r = ranks.get(m)
                if r is not None and r not in seen:
                    seen.add(r)
                    if rng.random() < p:
                        dropped.add(r)
    while len(toks) >= 2:
        best_i, best_r = -1, 1 << 30
        for i in range(len(toks) - 1):
            m = toks[i] + toks[i + 1]
            r = ranks.get(m)
            if r is not None and r not in dropped and r < best_r:
                best_r, best_i = r, i
        if best_i < 0:
            break
        toks[best_i:best_i + 2] = [toks[best_i] + toks[best_i + 1]]
    return [ranks[t] for t in toks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--text", default=None)
    args = ap.parse_args()
    ranks = load_ranks()
    rng = random.Random(args.seed)
    lines = [args.text] if args.text else None
    if lines is None:
        import json
        from eval_driver import load_enc
        enc = load_enc()
        n = diff = ntok0 = ntokd = 0
        for i, line in enumerate(open("/tmp/prose/prose_train.jsonl", encoding="utf-8")):
            if i >= args.limit:
                break
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            if not text:
                continue
            ids0 = enc.encode_ordinary(text)
            idsd = encode_dropout(text.encode("utf-8"), ranks, args.p, rng)
            assert enc.decode(idsd) == text, f"integridade quebrada na linha {i}"
            n += 1
            ntok0 += len(ids0)
            ntokd += len(idsd)
            if ids0 != idsd:
                diff += 1
        print(f"linhas={n} fert0={ntok0/max(n,1):.2f} fertd={ntokd/max(n,1):.2f} "
              f"linhas-alteradas={diff}/{n} ({diff/max(n,1):.1%})")
        if args.p == 0:
            assert diff == 0, "p=0 deve ser identico ao ordinario!"
            print("portao p=0 OK: bit-identico")


if __name__ == "__main__":
    main()
