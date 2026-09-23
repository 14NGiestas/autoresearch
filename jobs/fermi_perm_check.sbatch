#!/bin/bash
#SBATCH -J permcheck
#SBATCH -c 8
#SBATCH -t 03:00:00
#SBATCH -o /home/pauli/autoresearch/logs/permcheck_%j.log
# REFACAO dos arms perdidos (job 137), agora com os pools consertados.
#
# O que aconteceu: eu criei rows_perm1/2.npy com np.save no default (ordem C) e o
# loader montava a cache TRANSPOSTA (1025 x 11797 em vez de 11797 x 1025) ->
# rodadas morriam com "rows file too short" num passo aparentemente aleatorio
# (1025): 6 arms perdidos. Pools reescritos em ordem F e o loader agora CONFERE o
# cabecalho (npy_header) e falha alto em ordem C.
#
# Nao repito os 6 arms: os que importam sao (a) o efeito do outer em K=1 num
# SEGUNDO sorteio de dados (a reatribuicao '80% e otimizador melhor' tem de
# sobreviver a outro draw) e (b) a config vencedora (reset+outer) num segundo
# draw. 3 arms, ~1h50.
#
# Dependencia: depois do sweep (143) -- ele mede throughput e precisa de maquina
# ociosa.
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
echo "permcheck job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
echo "=== pool=perm1 : K=1, 16 rodadas (momentos carregados)"
$PY scripts/fedavg_rounds.py --k 1 --tau 122 --rounds 16 --rows /tmp/mix/rows_perm1.npy \
    --out /tmp/ctrl2/k1_perm1
echo "=== pool=perm1 : K=1, 16 rodadas + outer 0.7 (o controle decisivo, 2o draw)"
$PY scripts/fedavg_rounds.py --k 1 --tau 122 --rounds 16 --rows /tmp/mix/rows_perm1.npy \
    --out /tmp/ctrl2/k1outer_perm1 --outer nesterov --outer-lr 0.7 --outer-mom 0.9
echo "=== pool=perm1 : K=4, momento ZERADO + outer (a config vencedora, 2o draw)"
$PY scripts/fedavg_rounds.py --k 4 --tau 122 --rounds 16 --rows /tmp/mix/rows_perm1.npy \
    --out /tmp/ctrl2/B_perm1 --reset-moments --outer nesterov --outer-lr 0.7 --outer-mom 0.9
echo "fim permcheck $(date '+%F %H:%M:%S')"
