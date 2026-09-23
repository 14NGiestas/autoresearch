#!/bin/bash
#SBATCH -J c005b
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o /home/pauli/autoresearch/logs/c005b_%j.log
# ESBOCO. Nao submeter sem ler isto.
#
# C-0.05B: 44,2M parametros (d512 h8 kv2 L13), head_dim 64, 20 tokens/param
# = 884M tokens = 862.720 passos. E' o primeiro ponto no orcamento Chinchilla
# completo: a nossa serie hoje roda 8M tokens, ou seja 110x menos.
#
# A INCognita NAO e' o numero de passos. E' o tok/s nesta arch. O estimador
# assume ~43k tok/s por no' (ele carrega os fatores de execucao e head_dim,
# medidos no d96, para qualquer tamanho). Se o tok/s real for 2k, isto leva 5
# dias em vez de 0,7. Por isso a janela e' 24 h: o job mede o proprio ritmo no
# energy_trace.csv e a gente decide com o numero, nao com a estimativa.
#
# tau = 862.720 / (K=4 x 16 rounds) = 13.480. Rodada LONGA de proposito: o custo
# da rodada e' por invocacao (medido), entao menos sincronizacoes e' melhor.
set -u
cd /home/pauli/autoresearch
BY="$HOME/.cache/autoresearch/tok_tables/token_bytes.txt"
ARCH=arch_d512_h8_kv2_l13_v8192_c1024
D=512; H=8; KV=2; L=13
echo "c005b job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
bin/env_record "inicio"
# 1. build (o build_arch ja' passa -O2)
EVB=$(bin/build_arch $D $H $KV $L 8192 1024 /tmp/c005b/bd/$ARCH 2>&1 \
      | sed -n 's/^\[build_arch\] app train_run //p')
[ -n "$EVB" ] || { echo "sem binario"; exit 1; }
# 2. init (seed 7, igual a serie de tamanho, para ser comparavel)
[ -d /tmp/c005b/init ] || .venv-numpy/bin/python3 scripts/make_init.py \
  --out /tmp/c005b/init --d $D --heads $H --kv $KV --layers $L \
  --vocab 8192 --ctx 1024 --seed 7
# 3. o driver com a receita vencedora
.venv-numpy/bin/python3 scripts/fedavg_rounds.py \
  --regime federated --k 4 --tau 13480 --rounds 16 \
  --init /tmp/c005b/init --rows /tmp/mix/rows_f0.npy \
  --holdout /tmp/mix/rows_holdout.npy \
  --reset-moments --outer nesterov --outer-lr 0.7 \
  --build-dir /tmp/c005b/bd --out /tmp/c005b/run
bin/env_record "fim"
echo "fim c005b $(date '+%F %H:%M:%S')"
