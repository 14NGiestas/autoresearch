#!/bin/bash
#SBATCH -J c01b
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o /home/pauli/autoresearch/logs/c01b_%j.log
# ESBOCO. Nao submeter sem ler isto.
#
# C-0.1B: 88,1M parametros (d768 h6 kv2 L12), head_dim 128, 20 tokens/param
# = 1,76B tokens = 1.720.320 passos. E' o primeiro ponto no orcamento Chinchilla
# completo: a nossa serie hoje roda 8M tokens, ou seja 110x menos.
#
# A INCognita NAO e' o numero de passos. E' o tok/s nesta arch. O estimador
# assume ~43k tok/s por no' (ele carrega os fatores de execucao e head_dim,
# medidos no d96, para qualquer tamanho). Se o tok/s real for 2k, isto leva 5
# dias em vez de 0,7. Por isso a janela e' 24 h: o job mede o proprio ritmo no
# energy_trace.csv e a gente decide com o numero, nao com a estimativa.
#
# tau = 1.720.320 / (K=4 x 16 rounds) = 26.880. Rodada LONGA de proposito: o custo
# da rodada e' por invocacao (medido), entao menos sincronizacoes e' melhor.
set -u
cd /home/pauli/autoresearch
BY="$HOME/.cache/autoresearch/tok_tables/token_bytes.txt"
ARCH=arch_d768_h6_kv2_l12_v8192_c1024
D=768; H=6; KV=2; L=12
echo "c01b job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
bin/env_record "inicio"
# 1. build (o build_arch ja' passa -O2)
EVB=$(bin/build_arch $D $H $KV $L 8192 1024 /tmp/c01b/bd/$ARCH 2>&1 \
      | sed -n 's/^\[build_arch\] app train_run //p')
[ -n "$EVB" ] || { echo "sem binario"; exit 1; }
# 2. init (seed 7, igual a serie de tamanho, para ser comparavel)
[ -d /tmp/c01b/init ] || .venv-numpy/bin/python3 scripts/make_init.py \
  --out /tmp/c01b/init --d $D --heads $H --kv $KV --layers $L \
  --vocab 8192 --ctx 1024 --seed 7
# 3. o driver com a receita vencedora
.venv-numpy/bin/python3 scripts/fedavg_rounds.py \
  --regime federated --k 4 --tau 26880 --rounds 16 \
  --init /tmp/c01b/init --rows /tmp/mix/rows_f0.npy \
  --holdout /tmp/mix/rows_holdout.npy \
  --reset-moments --outer nesterov --outer-lr 0.7 \
  --build-dir /tmp/c01b/bd --out /tmp/c01b/run
bin/env_record "fim"
echo "fim c01b $(date '+%F %H:%M:%S')"
