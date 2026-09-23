#!/bin/bash
#SBATCH -J greedy
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /home/pauli/autoresearch/logs/greedy_%j.log
# Soma ponderada preserva o melhor? Greedy soup (subconjuntos) sobre os shards de
# K=4/1 epoca que ja estao em disco. SELECAO no dev (rows_f0[8000:8100], nunca
# treinado), RELATO no holdout (7900:8000). Zero treino.
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
EV=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
D=/tmp/greedy; mkdir -p $D
S=""; for k in 0 1 2 3; do S="$S /tmp/compose/disj4_s$k/step_1952"; done
echo "greedy job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S') shards:$S"
$PY - "$EV" "$D" $S <<'PY'
import itertools, math, os, subprocess, sys, numpy as np
sys.path.insert(0,'scripts'); import ckio
EV, D = sys.argv[1], sys.argv[2]; shards = sys.argv[3:]
os.environ.setdefault("OMP_NUM_THREADS","8")
tb=np.array([int(x) for x in open(os.path.expanduser('~/.cache/autoresearch/tok_tables/token_bytes.txt'))])
def bpb(f, rows):
    nll=byt=0.0; i=0
    for line in open(f, encoding='utf-8', errors='replace'):
        s=line.strip()
        if not s or s[0] not in '0123456789-': continue
        ns=np.fromstring(s, sep=' '); r=rows[i]; i+=1
        L=min(len(ns), len(r)-1); t=tb[r[1:L+1]]; m=t>0
        nll+=ns[:L][m].sum(); byt+=t[m].sum()
    return nll/byt/math.log(2)
dev=np.load('/tmp/mix/rows_dev.npy', mmap_mode='r'); hold=np.load('/tmp/mix/rows_holdout.npy', mmap_mode='r')
def ev(weights, rows, tag):
    o=f"{D}/ev_{tag}.txt"
    with open(o,'w') as fh:
        subprocess.run([EV,'--weights',weights,'--rows','/tmp/mix/rows_dev.npy' if rows is dev else '/tmp/mix/rows_holdout.npy','--attn','blas','--batch','16'],stdout=fh,stderr=subprocess.STDOUT,check=True)
    return bpb(o, rows)
def mean_of(dirs, out):
    acc={}
    for d in dirs:
        for k,v in ckio.load_ckpt_dir(d).items():
            acc[k]= v.astype(np.float64) if k not in acc else acc[k]+v.astype(np.float64)
    n=len(dirs); os.makedirs(out, exist_ok=True)
    ckio.save_ckpt_dir(out, {k:(v/n).astype(np.float32) for k,v in acc.items()}, like=dirs[0], op="greedy_mean")
res=[]
for i,d in enumerate(shards):
    b=ev(d, dev, f"single{i}"); res.append((b, f"single{i}", (d,)))
best_single=min(res)
print(f"melhor shard no DEV: {best_single[1]} dev={best_single[0]:.5f}")
for r in range(1,5):
    for combo in itertools.combinations(range(4), r):
        tag="sub"+"".join(map(str,combo))
        out=f"{D}/{tag}"; mean_of([shards[i] for i in combo], out)
        b=ev(out, dev, tag); res.append((b, tag, tuple(shards[i] for i in combo)))
res.sort()
print("top 5 no DEV:"); 
for b,tag,_ in res[:5]: print(f"  {tag:10s} dev={b:.5f}")
b,tag,dirs_sel=res[0]
print(f"escolhido no dev: {tag} (dev {b:.5f}) -> no HOLDOUT:")
mean_of(list(dirs_sel), f"{D}/chosen")
bh=ev(f"{D}/chosen", hold, "chosen_hold")
print(f"  greedy/ponderado escolhido : {bh:.5f}")
bu=ev(f"{D}/sub0123", hold, "uniform_hold"); print(f"  media uniforme (todos)    : {bu:.5f}")
bs=ev(best_single[2][0], hold, "bestsingle_hold"); print(f"  melhor shard (no dev)     : {bs:.5f}")
sing=[ev(d, hold, f"s{i}_hold") for i,d in enumerate(shards)]
print(f"  ORACULO (melhor shard no holdout, so referencia): {min(sing):.5f}  shards={[round(x,4) for x in sing]}")
PY
echo "fim greedy $(date '+%F %H:%M:%S')"
