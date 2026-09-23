#!/bin/bash
#SBATCH -J roundsstruct
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH -o /home/pauli/autoresearch/logs/roundsstruct_%j.log
# CONTROLE da estrutura de rodadas (o que o job 137 expos).
#
# Fato medido: K=1 com 16 rodadas (tau=122, momentos carregados) = 2.71348,
# contra 2.44476 do run unico de 1952 passos (plain_f0). Com UMA rodada a
# maquinaria de federar e' exatamente neutra (bate a referencia na 5a casa),
# entao o +0.27 vem da ESTRUTURA de rodadas, nao do encanamento.
#
# Hipotese testada: o custo e' POR INVOCACAO (warmup linear de 2 passos e'
# run-relative e reinicia em cada rodada; processos novos tambem reaquecem).
# Se for isso, o dano cresce com o NUMERO de rodadas mantendo os passos totais:
#   1952 passos em 1 rodada  -> ja' medido: 2.44476
#   1952 passos em 16 rodadas-> ja' medido: 2.71348   (tau=122)
# aqui:                   8 rodadas (tau=244) e 4 rodadas (tau=488)
# e 32 rodadas (tau=61) para ver se continua crescendo.
# Mais um arm com --reset-moments para separar "estado do Adam" de "warmup".
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
echo "roundsstruct job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
echo "=== K=1, tau=61, 32 rodadas (momentos carregados)"
$PY scripts/fedavg_rounds.py --k 1 --tau 61 --rounds 32 --out /tmp/rst/t61k1
echo "=== K=1, tau=244, 8 rodadas"
$PY scripts/fedavg_rounds.py --k 1 --tau 244 --rounds 8 --out /tmp/rst/t244k1
echo "=== K=1, tau=488, 4 rodadas"
$PY scripts/fedavg_rounds.py --k 1 --tau 488 --rounds 4 --out /tmp/rst/t488k1
echo "=== K=1, tau=122, 16 rodadas, SEM carregar momentos (estado zerado)"
$PY scripts/fedavg_rounds.py --k 1 --tau 122 --rounds 16 --reset-moments --out /tmp/rst/t122k1nomom
echo "fim roundsstruct $(date '+%F %H:%M:%S')"
