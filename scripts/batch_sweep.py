#!/usr/bin/env python3
"""batch_sweep.py — quanto a eficiencia de LOTE compra (a tokens fixos).

Pergunta (hyp_c3aed6): o treino roda com B=1 e cada GEMM e' minusculo
(M=1024, N=K=d=96 -> 0,019 GFLOP), rendendo ~30 GFLOP/s em ~8 cores. Subir B
aumenta a intensidade aritmetica e deveria multiplicar ATUALIZACOES por unidade
de wall-clock. O limite e' o tamanho critico de lote: mais tokens por update =
menos updates por token de orcamento.

Este script mede os DOIS lados, separados de proposito:

  * THROUGHPUT: passos fixos (30) para cada B -> tokens/s e GFLOP/s. Isola
    "quanto o lote acelera o passo".
  * QUALIDADE x TOKENS: mesmo orcamento de TOKENS para cada B -> quantos passos
    (updates) cada B precisa, wall_s e bpb na holdout. Isola "quanto custa em
    qualidade trocar updates por lote".

Nao decide nada sozinho: imprime as duas tabelas e o produto (updates/s x
qualidade) para o B escolhido ser uma decisao, nao um palpite.

Uso:
  batch_sweep.py --train BIN --eval BIN --rows FILE --init DIR --holdout FILE
                 [--bytes FILE] [--batches 1,4,16,64] [--tokens 2000000]
                 [--probe-steps 30] [--work DIR]
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

T = 1024  # TT do modelo (contexto)


def run(cmd, log):
    """Roda e devolve (wall_s, stdout). Aborta alto em falha nao-zero."""
    t0 = time.time()
    with open(log, "w") as fh:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        fh.write(p.stdout)
    wall = time.time() - t0
    if p.returncode != 0:
        print("FALHOU (exit %d): %s\n--- ultimas linhas de %s:" % (p.returncode, " ".join(cmd[:6]), log))
        print("\n".join(p.stdout.splitlines()[-12:]))
        sys.exit(1)
    return wall, p.stdout


def final_bpb(train_bin, rows, init, outdir, batch, nsteps, lr, ntrain, bytesfile):
    rm = ["--weights", init, "--rows", rows, "--out", outdir, "--nsteps", str(nsteps),
          "--lr", str(lr), "--ntrain", str(ntrain), "--start_row", "0", "--nval", "1",
          "--val_every", str(nsteps), "--trn_probe", "0", "--save_every", str(nsteps),
          "--attn", "blas", "--bytes", bytesfile, "--batch", str(batch)]
    wall, out = run([train_bin] + rm, os.path.join(outdir, "train.log"))
    bpb = None
    for line in out.splitlines():
        low = line.lower()
        if "bpb" in low:
            for tok in line.replace("=", " ").split():
                try:
                    v = float(tok)
                except ValueError:
                    continue
                if 0.1 < v < 20.0:
                    bpb = v
    return wall, bpb, out


def holdout_bpb(eval_bin, ckpt, holdout, batch=16):
    out = subprocess.run([eval_bin, "--weights", ckpt, "--rows", holdout,
                          "--batch", str(batch), "--attn", "blas"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    best = None
    for line in out.stdout.splitlines():
        low = line.lower()
        if "bpb" in low:
            for tok in line.replace("=", " ").split():
                try:
                    v = float(tok)
                except ValueError:
                    continue
                if 0.1 < v < 20.0:
                    best = v
    return best, out.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--rows", default="/tmp/mix/rows_f0.npy")
    ap.add_argument("--holdout", default="/tmp/mix/rows_holdout.npy")
    ap.add_argument("--init", default="/tmp/mix/init3m")
    ap.add_argument("--bytes", default=os.path.expanduser(
        "~/.cache/autoresearch/tok_tables/token_bytes.txt"))
    ap.add_argument("--batches", default="1,4,16,64")
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--probe-steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--ntrain", type=int, default=11797)
    ap.add_argument("--work", default="/tmp/bsweep")
    a = ap.parse_args()

    # parametros do modelo (6*P por token no fwd+bwd) -- do arch.txt do init
    arch = {}
    with open(os.path.join(a.init, "arch.txt")) as fh:
        for line in fh:
            if "=" in line:
                k, v = line.split("=", 1)
                arch[k.strip()] = int(v.strip().split()[0])
    P = (2 * arch["vocab"] * arch["d_model"] + arch["n_layer"] *
         (2 * arch["d_model"] ** 2 + 2 * arch["d_model"] * arch["n_kv"] *
          arch["head_dim"] + 8 * arch["d_model"] ** 2))
    print("# arch: %s" % arch)
    print("# P = %d params (6*P FLOPs por token no fwd+bwd)" % P)

    batches = [int(x) for x in a.batches.split(",")]
    os.makedirs(a.work, exist_ok=True)

    # ---- 1) THROUGHPUT: passos fixos -------------------------------------
    print("\n# --- THROUGHPUT (passos fixos = %d): quanto o lote acelera o PASSO" % a.probe_steps)
    print("# B    steps   tokens     wall_s   tokens/s   GFLOP/s   s/step")
    thr = {}
    for b in batches:
        d = os.path.join(a.work, "probe_b%d" % b)
        shutil.rmtree(d, ignore_errors=True)
        wall, bpb, _ = final_bpb(a.train, a.rows, a.init, d, b, a.probe_steps,
                                 a.lr, a.ntrain, a.bytes)
        tok = a.probe_steps * b * T
        thr[b] = tok / wall
        print("%-4d %-7d %-10d %-8.1f %-10.0f %-9.1f %.3f" %
              (b, a.probe_steps, tok, wall, tok / wall,
               6 * P * tok / wall / 1e9, wall / a.probe_steps))

    # ---- 2) QUALIDADE a TOKENS FIXOS -------------------------------------
    print("\n# --- QUALIDADE a tokens FIXOS (%d tokens): quanto custa trocar updates por lote" % a.tokens)
    print("# B    steps   wall_s   tokens/s   bpb(holdout)   updates/s")
    for b in batches:
        steps = max(1, a.tokens // (b * T))
        d = os.path.join(a.work, "qual_b%d" % b)
        shutil.rmtree(d, ignore_errors=True)
        wall, bpb_train, _ = final_bpb(a.train, a.rows, a.init, d, b, steps,
                                       a.lr, a.ntrain, a.bytes)
        ck = None
        for root, _, files in os.walk(d):
            if "model.safetensors" in files:
                ck = root
            elif any(f.startswith("transformer_wte_weight") for f in files):
                ck = root
        if ck is None:
            print("  B=%d: nenhum checkpoint encontrado em %s" % (b, d))
            continue
        hbpb, _ = holdout_bpb(a.eval, ck, a.holdout)
        print("%-4d %-7d %-8.1f %-10.0f %-14s %.4f" %
              (b, steps, wall, steps * b * T / wall,
               ("%.5f" % hbpb) if hbpb else "n/d", steps / wall))

    print("\n# leitura: a 1a tabela diz o ganho de wall-clock; a 2a diz o preco em bpb.")
    print("# escolha o B que maximiza (updates/s) x (qualidade) -- nao o B maximo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
