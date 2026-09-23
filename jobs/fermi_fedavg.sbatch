#!/bin/bash
#SBATCH -J fedavg
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -o /home/pauli/autoresearch/logs/fedavg_%j.log
# Local SGD com tau FINITO: o experimento que decide a meta 16 anos -> 6 meses.
# Compute total fixo em 7808 passos (igual ao f0 e ao grid tau=infinito):
#   tau=488, R=4  -> cada worker faz 1952 passos (wall-clock 4x menor que f0)
#   tau=122, R=16 -> sync muito mais frequente (mesmo compute, mesmo wall-clock)
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
echo "fedavg job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
$PY scripts/fedavg_rounds.py --k 4 --tau 488 --rounds 4 --out /tmp/fed/p488
$PY scripts/fedavg_rounds.py --k 4 --tau 122 --rounds 16 --out /tmp/fed/p122
# baseline de wall-clock igual (4x menos passos que f0, dados completos)
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
EV=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
rm -rf /tmp/fed/base1952
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
  --weights /tmp/mix/init3m --rows /tmp/mix/rows_f0.npy --out /tmp/fed/base1952 \
  --nsteps 1952 --lr 6e-4 --ntrain 7812 --nval 1 --val_every 9999999 \
  --save_every 9999999 --trn_probe 1 --attn blas \
  --bytes "$HOME/.cache/autoresearch/tok_tables/token_bytes.txt" > /tmp/fed/base1952.log 2>&1
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$EV" \
  --weights /tmp/fed/base1952/step_1952 --rows /tmp/mix/rows_holdout.npy \
  --attn blas --batch 16 > /tmp/fed/ev_base1952.txt 2>&1
echo "fim fedavg $(date '+%F %H:%M:%S')"
