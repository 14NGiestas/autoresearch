#!/bin/bash
#SBATCH -J relu3m2
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o /home/pauli/autoresearch/logs/relu3m2_%j.log
# Softmax contra ReLU-attention, RE-RODADO com build OTIMIZADO.
#
# Por que re-rodar: o job 163 rodou num build -O0 (o default do fpm, que o projeto
# usou em TODOS os runs ate' hoje -- ver o commit do achado). A metade QUALIDADE
# daquela rodada continua valida (bpb e' deterministico, e o checkpoint do braco
# softmax esta' completo em /tmp/rl3m/softmax/step_2929), mas a metade VELOCIDADE
# e' invalida: o custo do desvio do relu pode ser um artefato de -O0, onde o
# compilador nao remove nada.
#
# E o env_record vai POR BRACO agora: a rodada anterior so' registrou o ambiente
# antes dos dois, entao nao dava para separar "relu mais lento" de "relu sob mais
# carga". Lacuna declarada, corrigida aqui.
set -u
cd /home/pauli/autoresearch
BY="$HOME/.cache/autoresearch/tok_tables/token_bytes.txt"
B=$(ls /tmp/rl4/bd/*/gfortran_*/app/train_run | tail -1)
[ -n "$B" ] || { echo "sem binario"; exit 1; }
echo "relu3m2 job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')  (build OTIMIZADO)"
for FN in softmax relu; do
  echo "--- braco $FN"
  bin/env_record "antes de $FN"
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
    "$B" --weights /tmp/mix/init3m --rows /tmp/mix/rows_f0.npy \
    --out /tmp/rl4m/$FN --nsteps 2929 --lr 6e-4 --ntrain 11797 --start_row 0 \
    --nval 5 --val_every 293 --trn_probe 200 --save_every 2929 --keep_last 1 \
    --attn blas --batch 1 --bytes "$BY" --attn-fn $FN 2>&1 | tail -6
  bin/env_record "depois de $FN"
done
echo "fim relu3m2 $(date '+%F %H:%M:%S')"
