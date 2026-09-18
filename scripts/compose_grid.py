#!/usr/bin/env python3
"""compose_grid.py — taxonomia da composicao linear (merge de shards).

PERGUNTA. Treinar K shards em paralelo e MEDIR a media dos pesos custa quanto,
em bpb, contra treinar UM modelo com o mesmo total de tokens? Essa diferenca e o
"imposto de composicao". Se for ~0 para K grande, paralelizar da speedup K sem
perda; se crescer com K, ha um teto mensuravel.

POR QUE ISSO E A PERGUNTA CERTA. Um modelo que ve D tokens em K shards disjuntos
nunca ve o gradiente conjunto. A media de pesos e o caso limite tau=infinito do
local SGD (workers que nunca comunicam). A teoria diz que o afastamento entre
workers cresce com lr e com o intervalo de sincronizacao -- entao o imposto deve
ser funcao de (lr, K, disjuncao dos dados, init compartilhado), nao uma
propriedade binaria "merge funciona / nao funciona". Medimos a lei.

FAMILIAS (todas com o MESMO orcamento de passo: total_steps = 7808):
  disj    K shards em fatias DISJUNTAS (mesma distribuicao, amostras diferentes)
  order   K shards sobre TODOS os dados, com ordem deslocada (so ruido de otim.)
  seed    K shards com INITS independentes, mesma ordem de dados
  lr      disj K=4 com LR constante {6e-4 (do disj), 2e-4, 6e-5, 2e-5}
  anneal  disj K=4 com cosine->0 no fim (receita de reconvergencia)

METRICA. bpb byte-ponderado no holdout (100 linhas do rabo do rows file, nunca
treinadas por nenhum shard): imposto = bpb(merge) - bpb(run unico, mesmos tokens).

Uso:
  scripts/compose_grid.py --fams disj,order,seed,lr,anneal --out /tmp/compose
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
TOTAL = 7808           # passos totais de cada familia (casa com f0 = 7812)
NFULL = 7812           # linhas de treino do rows file (0..7811)
LR0 = 6.0e-4


def sh(cmd, log):
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)


def tok_bytes():
    p = os.path.expanduser("~/.cache/autoresearch/tok_tables/token_bytes.txt")
    return np.array([int(x) for x in open(p)])


def bpb_from(out_txt, rows):
    tb = tok_bytes()
    nll = byt = 0.0
    i = 0
    for line in open(out_txt, encoding="utf-8", errors="replace"):
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
    return nll / byt / math.log(2), i


class Lab:
    def __init__(self, a):
        self.out = a.out
        self.rows = a.rows
        self.holdout_file = a.holdout
        self.hold = np.load(a.holdout, mmap_mode="r")
        self.train_bin = self._pick("train_run", a.init)
        self.eval_bin = self._pick("eval_bpb", a.init)
        self.init = a.init
        os.makedirs(self.out, exist_ok=True)
        self.res = []

    def _pick(self, app, ref):
        """Binario do build cuja arch casa com o arch.txt da referencia.

        O build mais recente pode ser de OUTRA arch (d360 etc): o guard
        check_shape aborta -- e deve mesmo. Aqui escolhemos pelo arch.txt.
        """
        tag = ""
        ap = os.path.join(ref, "arch.txt")
        if os.path.exists(ap):
            kv = {}
            for line in open(ap):
                if "=" in line:
                    k, v = line.split("=", 1)
                    kv[k.strip()] = v.strip().split()[0]
            tag = ("arch_d{}_h{}_kv{}_l{}_v{}_c{}".format(
                kv.get("d_model"), kv.get("n_head"), kv.get("n_kv"),
                kv.get("n_layer"), kv.get("vocab"), kv.get("ctx")))
        pat = os.path.join(REPO, "src/build", tag or "arch_*", "*", "app", app)
        c = sorted(glob.glob(pat), key=os.path.getmtime)
        if not c:
            sys.exit(f"binario {app} nao buildado para {tag or 'arch desconhecida'}")
        return c[-1]

    def train(self, tag, init, ntrain, start_row, nsteps, lr, anneal=False):
        d = os.path.join(self.out, tag)
        if os.path.exists(os.path.join(d, f"step_{nsteps}")):
            return os.path.join(d, f"step_{nsteps}")
        os.makedirs(d, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                   OMP_DYNAMIC="FALSE")
        cmd = [self.train_bin, "--weights", init, "--rows", self.rows, "--out", d,
               "--nsteps", str(nsteps), "--lr", str(lr), "--ntrain", str(ntrain),
               "--start_row", str(start_row), "--nval", "1",
               "--val_every", "9999999", "--save_every", str(nsteps), "--trn_probe", "2",
               "--attn", "blas",
               "--bytes", os.path.expanduser(
                   "~/.cache/autoresearch/tok_tables/token_bytes.txt")]
        if anneal:
            cmd += ["--anneal", "1"]
        with open(os.path.join(d, "train.log"), "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True,
                           env=env)
        return os.path.join(d, f"step_{nsteps}")

    def merge(self, dirs, tag, alpha=None):
        """Media UNIFORME exata.

        ATENCAO a convencao do merge_checkpoints: m = ALPHA*A + (1-ALPHA)*B, ou
        seja ALPHA pondera o PRIMEIRO argumento. Para a cadeia dar pesos iguais, o
        peso do acumulador tem de ser k/(k+1) -- com 1/k (o que este arquivo fazia)
        o resultado vira [1/24,1/24,4/24,18/24] em K=4 e 0.875 no ultimo shard em
        K=8: quase um shard so. Corrigido em 2026-09-18; os merges ja gravados
        foram recomputados por scripts/exact_mean.py.
        """
        d = os.path.join(self.out, tag)
        if os.path.exists(os.path.join(d, "arch.txt")):
            return d
        if alpha is not None:                      # barreira: 2 pontas
            subprocess.run([sys.executable,
                            os.path.join(REPO, "scripts/merge_checkpoints.py"),
                            dirs[0], dirs[1], d, "--alpha", str(alpha)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                           check=True)
            return d
        acc = dirs[0]
        for k, nxt in enumerate(dirs[1:], start=2):
            tmp = os.path.join(self.out, f".acc_{tag}_{k}")
            shutil.rmtree(tmp, ignore_errors=True)
            subprocess.run([sys.executable,
                            os.path.join(REPO, "scripts/merge_checkpoints.py"),
                            acc, nxt, tmp, "--alpha", str(k / (k + 1.0))],
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                           check=True)
            acc = tmp
        shutil.rmtree(d, ignore_errors=True)
        shutil.move(acc, d)                        # acc pode estar dentro de out
        return d

    def eval(self, wdir, tag):
        o = os.path.join(self.out, f"ev_{tag}.txt")
        env = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                   OMP_DYNAMIC="FALSE")
        with open(o, "w") as f:
            subprocess.run([self.eval_bin, "--weights", wdir, "--rows",
                            self.holdout_file, "--attn", "blas", "--batch", "16"],
                           stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
        b, n = bpb_from(o, self.hold)
        return b, n

    def record(self, fam, K, tag, wdir, extra=None):
        b, n = self.eval(wdir, tag)
        row = {"fam": fam, "K": K, "tag": tag, "bpb": round(float(b), 5),
               "rows": n, "dir": wdir}
        if extra:
            row.update(extra)
        self.res.append(row)
        print(f"  [{fam} K={K}] {tag:28s} bpb={b:.5f}", flush=True)
        return b


def rel_drift(a, b):
    """||A-B||/||A|| agregado (metrica barata, ja sabemos que engana)."""
    import ckio
    A = ckio.load_ckpt_dir(a)
    B = ckio.load_ckpt_dir(b)
    na = nb = nd = 0.0
    for f in sorted(A):
        x = A[f].astype(np.float64).ravel()
        y = B[f].astype(np.float64).ravel()
        na += x @ x
        nb += y @ y
        nd += (x - y) @ (x - y)
    return float(math.sqrt(nd) / math.sqrt(na))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/compose")
    ap.add_argument("--rows", default="/tmp/mix/rows_f0.npy")
    ap.add_argument("--holdout", default="/tmp/mix/rows_holdout.npy")
    ap.add_argument("--init", default="/tmp/mix/init3m")
    ap.add_argument("--fams", default="disj,order,seed,lr,anneal")
    a = ap.parse_args()
    L = Lab(a)
    fams = a.fams.split(",")
    print(f"train={L.train_bin}\neval={L.eval_bin}", flush=True)

    # f0 = K=1 (baseline, ja treinado: 7812 passos sobre todos os dados)
    base = L.record("base", 1, "f0", os.path.join("/tmp/mix/w_f0/best"))

    if "disj" in fams:
        for K in (2, 4, 8):
            st = TOTAL // K
            sz = NFULL // K
            ds = [L.train(f"disj{K}_s{k}", L.init, sz, k * sz, st, LR0)
                  for k in range(K)]
            for k, d in enumerate(ds):
                L.record("disj", K, f"disj{K}_s{k}", d)
            m = L.merge(ds, f"disj{K}_merge")
            L.record("disj", K, f"disj{K}_merge", m,
                     {"drift": round(rel_drift(ds[0], ds[1]), 4)})
            if K == 2:
                for al in (0.25, 0.5, 0.75):
                    mm = L.merge(ds, f"disj2_a{al}", alpha=al)
                    L.record("disj", 2, f"disj2_a{al}", mm, {"alpha": al})

    if "order" in fams:
        K = 4
        st = TOTAL // K
        ds = [L.train(f"ord{K}_s{k}", L.init, NFULL, k * 39, st, LR0)
              for k in range(K)]
        for k, d in enumerate(ds):
            L.record("order", K, f"ord{K}_s{k}", d)
        L.record("order", K, f"ord{K}_merge", L.merge(ds, f"ord{K}_merge"),
                 {"drift": round(rel_drift(ds[0], ds[1]), 4)})

    if "seed" in fams:
        K = 4
        st = TOTAL // K
        ds = []
        for k in range(K):
            ini = os.path.join(L.out, f"init_s{k + 11}")
            if not os.path.exists(os.path.join(ini, "arch.txt")):
                shutil.rmtree(ini, ignore_errors=True)
                subprocess.run([sys.executable,
                                os.path.join(REPO, "scripts/make_init.py"),
                                "--out", ini, "--d", "96", "--heads", "6",
                                "--kv", "2", "--layers", "12", "--vocab", "8192",
                                "--ctx", "1024", "--seed", str(11 + k)],
                               stdout=subprocess.DEVNULL, check=True)
            ds.append(L.train(f"seed{K}_s{k}", ini, NFULL, 0, st, LR0))
        for k, d in enumerate(ds):
            L.record("seed", K, f"seed{K}_s{k}", d)
        L.record("seed", K, f"seed{K}_merge", L.merge(ds, f"seed{K}_merge"),
                 {"drift": round(rel_drift(ds[0], ds[1]), 4)})

    if "lr" in fams:
        K = 4
        st = TOTAL // K
        sz = NFULL // K
        for lr in (2e-4, 6e-5):
            tag = f"lr{lr:g}".replace(".", "").replace("-", "")
            ds = [L.train(f"{tag}_s{k}", L.init, sz, k * sz, st, lr)
                  for k in range(K)]
            L.record("lr", K, f"{tag}_merge", L.merge(ds, f"{tag}_merge"),
                     {"lr": lr, "drift": round(rel_drift(ds[0], ds[1]), 4)})

    if "anneal" in fams:
        K = 4
        st = TOTAL // K
        sz = NFULL // K
        ds = [L.train(f"ann{K}_s{k}", L.init, sz, k * sz, st, LR0, anneal=True)
              for k in range(K)]
        for k, d in enumerate(ds):
            L.record("anneal", K, f"ann{K}_s{k}", d)
        L.record("anneal", K, f"ann{K}_merge", L.merge(ds, f"ann{K}_merge"),
                 {"drift": round(rel_drift(ds[0], ds[1]), 4)})

    res = {"base_bpb": base, "rows": L.res}
    with open(os.path.join(L.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\n=== imposto de composicao (bpb merge - bpb base) ===")
    for r in L.res:
        if r["tag"].endswith("merge"):
            print(f"  {r['fam']:7s} K={r['K']}  tax={r['bpb'] - base:+.5f}"
                  f"  (bpb {r['bpb']:.5f})")
    print("json:", os.path.join(L.out, "results.json"))


if __name__ == "__main__":
    main()
