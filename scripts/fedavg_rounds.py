#!/usr/bin/env python3
"""fedavg_rounds.py — local SGD com tau FINITO: o experimento que decide a meta.

O grid de composicao mediu tau=infinito (uma unica media no fim): imposto de
+0.38 bpb a K=4, dos quais so +0.057 vem da media e +0.326 de cada shard ter
visto 1/K dos dados. tau=infinito e o PIOR caso da teoria de local SGD. Aqui os
workers se fundem a cada tau passos e continuam: a cada rodada cada worker herda
implicitamente o que os outros viram, que e o mecanismo pelo qual data-parallel
SGD funciona.

Orcamento (K workers, R rodadas, tau passos por rodada):
  * passos por worker  = R*tau          -> e o WALL-CLOCK do experimento
  * passos totais      = K*R*tau        -> e o COMPUTE TOTAL
  * baseline justo     = run unico com K*R*tau passos (mesmo compute, K x mais
    wall-clock). E esse o numero que a meta "16 anos -> 6 meses" quer igualar.

Imposto: C = bpb(fedavg) - bpb(run unico com K*R*tau passos).
Controle: mesmo K, tau = R*tau (uma rodada so) = tau=infinito, ja medido.

Uso:
  scripts/fedavg_rounds.py --k 4 --tau 488 --rounds 4 --lr 6e-4 \
      --init /tmp/mix/init3m --rows /tmp/mix/rows_f0.npy --out /tmp/fed/p488 \
      [--anneal] [--eval-holdout /tmp/mix/rows_holdout.npy]
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NFULL = 7812
SKIP = ("muon_",)


def sh(cmd, log, env=None):
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True,
                       env=env or dict(os.environ))


def pick(app, ref):
    """binario do build cuja arch casa com o arch.txt da referencia."""
    kv = {}
    with open(os.path.join(ref, "arch.txt")) as fh:
        for line in fh:
            if "=" in line:
                k, v = line.split("=", 1)
                kv[k.strip()] = v.strip().split()[0]
    tag = "arch_d{}_h{}_kv{}_l{}_v{}_c{}".format(
        kv["d_model"], kv["n_head"], kv["n_kv"], kv["n_layer"], kv["vocab"], kv["ctx"])
    c = sorted(glob.glob(os.path.join(REPO, "src/build", tag, "*", "app", app)),
               key=os.path.getmtime)
    if not c:
        sys.exit(f"{app} nao buildado para {tag}")
    return c[-1]


def avg_dirs(dirs, out, include_opt=True):
    """Media elemento a elemento, incluindo os momentos do Adam.

    A media dos momentos e o que permite retomar o treino sem reiniciar a
    dinamica do otimizador (sem isso cada rodada comeca com passos grandes).
    """
    os.makedirs(out, exist_ok=True)
    names = [f for f in sorted(os.listdir(dirs[0])) if f.endswith(".npy")]
    if not include_opt:
        names = [f for f in names if not f.startswith("adam_")]
    for f in names:
        acc = None
        for d in dirs:
            x = np.load(os.path.join(d, f)).astype(np.float64)
            acc = x if acc is None else acc + x
        acc /= len(dirs)
        np.save(os.path.join(out, f), acc.astype(np.float32))
    for f in ("arch.txt", "template.txt"):
        src = os.path.join(dirs[0], f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, f))
    return len(names)


def bpb_of(evfile, rows):
    tb = np.array([int(x) for x in open(os.path.expanduser(
        "~/.cache/autoresearch/tok_tables/token_bytes.txt"))])
    nll = byt = 0.0
    i = 0
    for line in open(evfile, encoding="utf-8", errors="replace"):
        s = line.strip()
        if not s or s[0] not in "0123456789-":
            continue
        ns = np.fromstring(s, sep=" ")
        r = rows[i]
        i += 1
        L = min(len(ns), len(r) - 1)
        t = tb[r[1:L + 1]]
        m = t > 0
        nll += ns[:L][m].sum()
        byt += t[m].sum()
    return nll / byt / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tau", type=int, default=488)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--init", default="/tmp/mix/init3m")
    ap.add_argument("--rows", default="/tmp/mix/rows_f0.npy")
    ap.add_argument("--holdout", default="/tmp/mix/rows_holdout.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--anneal", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()

    train = pick("train_run", a.init)
    evb = pick("eval_bpb", a.init)
    hold = np.load(a.holdout, mmap_mode="r")
    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
               OMP_DYNAMIC="FALSE")
    slice_sz = NFULL // a.k
    total = a.k * a.rounds * a.tau
    print(f"K={a.k} tau={a.tau} R={a.rounds} lr={a.lr:g} anneal={a.anneal}")
    print(f"  passos/worker={a.rounds*a.tau} (wall-clock)  compute={total} passos")
    print(f"  baseline justo: run unico de {total} passos")

    common = os.path.join(a.out, "round0_common")
    if not os.path.exists(common):
        os.makedirs(common)
        for f in sorted(os.listdir(a.init)):
            if f.endswith((".npy", ".txt")):
                shutil.copy(os.path.join(a.init, f), os.path.join(common, f))
    history = []
    for r in range(a.rounds):
        ck = os.path.join(a.out, f"round{r}_common")
        workers = []
        for k in range(a.k):
            wdir = os.path.join(a.out, f"r{r}_w{k}")
            cmd = [train, "--weights", common if r == 0 else ck, "--rows", a.rows,
                   "--out", wdir, "--nsteps", str(a.tau), "--lr", str(a.lr),
                   "--ntrain", str(slice_sz), "--start_row", str(k * slice_sz),
                   "--nval", "1", "--val_every", "9999999", "--trn_probe", "1",
                   "--save_every", str(a.tau), "--attn", "blas",
                   "--bytes", os.path.expanduser(
                       "~/.cache/autoresearch/tok_tables/token_bytes.txt")]
            if a.anneal and r == a.rounds - 1:
                cmd += ["--anneal", "1"]
            sh(cmd, os.path.join(a.out, f"r{r}_w{k}.log"), env)
            workers.append(os.path.join(wdir, f"step_{a.tau}"))
        nxt = os.path.join(a.out, f"round{r+1}_common")
        n = avg_dirs(workers, nxt)
        common = nxt
        print(f"  rodada {r+1}/{a.rounds}: {n} tensores mediados (com momentos)")
        if not a.no_eval:
            ev = os.path.join(a.out, f"ev_round{r+1}.txt")
            sh([evb, "--weights", nxt, "--rows", a.holdout, "--attn", "blas",
                "--batch", "16"], ev, env)
            b = bpb_of(ev, hold)
            history.append({"round": r + 1, "steps_per_worker": (r + 1) * a.tau,
                            "bpb": float(b)})
            print(f"    bpb (holdout) = {b:.5f}")
    if history:
        with open(os.path.join(a.out, "history.json"), "w") as f:
            json.dump({"config": vars(a), "history": history}, f, indent=1)
        print(f"  -> {os.path.join(a.out, 'history.json')}")


if __name__ == "__main__":
    main()
