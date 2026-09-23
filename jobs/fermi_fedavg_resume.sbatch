#!/bin/bash
#SBATCH -J fedres
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -o /home/pauli/autoresearch/logs/fedres_%j.log
# Repoe o que a limpeza de build derrubou no job 89: o braco tau=122, a rodada 4
# do tau=488 (retomando do round3_common, sem refazer 1-3) e o baseline de
# wall-clock (run unico de 1952 passos sobre TODOS os dados).
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
echo "fedres job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
# (1) rodada 4 do tau=488: copia o estado da rodada 3 como round0 do novo run
mkdir -p /tmp/fed/p488_r4
[ -d /tmp/fed/p488_r4/round0_common ] || cp -r /tmp/fed/p488/round3_common /tmp/fed/p488_r4/round0_common
$PY scripts/fedavg_rounds.py --k 4 --tau 488 --rounds 1 --out /tmp/fed/p488_r4
# (2) braco tau=122 (16 syncs, mesmo compute e mesmo wall-clock)
$PY scripts/fedavg_rounds.py --k 4 --tau 122 --rounds 16 --out /tmp/fed/p122
# (3) baseline de wall-clock: 1952 passos nos dados completos
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
echo "baseline 1952 ok"
echo "fim fedres $(date '+%F %H:%M:%S')"
