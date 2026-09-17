#!/bin/bash
#SBATCH -J ecal
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH -o /home/pauli/autoresearch/logs/ecal_%j.log
# Calibracao de energia: repouso (60 s) + 3 min do treino d96 a 8 threads.
# Sai o W de repouso (para descontar) e o J/Mtok do regime de treino real.
set -u
cd /home/pauli/autoresearch
EPY=.venv-numpy/bin/python3
echo "ecal job=${SLURM_JOB_ID} host=$(hostname) $(date '+%F %H:%M:%S')"
# repouso: espera o job comecar a contar e mede antes de qualquer carga
$EPY scripts/energy_run.py idle --seconds 60
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
rm -rf /tmp/ecal
$EPY scripts/energy_run.py run --tag cal/d96_8t --tokens 440000 \
  --cmd-id "d96 blas 8t" -- \
  env OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
  "$BIN" --weights /tmp/mix/init3m --rows /tmp/mix/rows_f0.npy --out /tmp/ecal \
  --nsteps 430 --lr 0.0006 --ntrain 7812 --nval 1 --val_every 9999999 \
  --save_every 9999999 --trn_probe 1 --attn blas \
  --bytes "$HOME/.cache/autoresearch/tok_tables/token_bytes.txt" > /tmp/ecal.log 2>&1
$EPY scripts/energy_run.py report
echo "fim ecal $(date '+%F %H:%M:%S')"
