#!/bin/bash
#SBATCH -J size216
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -o /home/pauli/autoresearch/logs/size216_%j.log
# Serie de TAMANHO controlada, ponto 1: d216 com a receita vencedora.
#
# Por que este braco existe: os pontos que reavaliamos no halfbeast (d96 2,46 /
# d216 2,09 / d360 1,94) NAO sao uma serie controlada. Eles vem de experimentos
# diferentes -- o d360 e' do experimento de sopa M25, com lr 0.0003, 59889 passos
# e outra janela de dados. Entao aquela monotonia e' CONSISTENTE com tamanho
# ajudar, mas esta' confundida por receita, tokens e dados.
#
# Este braco usa os MESMOS args do nosso melhor d96 (2,33183, /tmp/dl/122):
#   k=4, tau=122, rounds=16, reset-moments, outer nesterov lr 0.7
# ou seja 16*122*4 = 7808 passos, o que da' o MESMO orcamento de tokens. Os
# mesmos dados (rows_f0 / rows_holdout). A UNICA coisa que muda e' a arch.
# Assim a comparacao isola tamanho de verdade.
#
# O dir de build e' privado: a arvore src/build e' a dos jobs vivos e nao se
# toca nela (AGENTS.md regra 1). O --build-dir e' novo no driver, e com um dir
# dedicado mais de um binario para o mesmo app e' ERRO, nao escolha.
# Layout: o driver espera <root>/<tag>/.../app, entao o <tag> e' um nivel que o
# chamador da'. O build tem que ser feito DENTRO dele (/tmp/size/bd/<tag>).
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
echo "size216 job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S') load=$(cut -d' ' -f1 /proc/loadavg)"
echo "receita: k=4 tau=122 rounds=16 reset-moments outer=nesterov lr 0.7 (a mesma do d96 2,33183)"
echo "dados: rows_f0 + rows_holdout (os mesmos). unica mudanca: arch d216."
$PY scripts/fedavg_rounds.py \
    --regime federated --k 4 --tau 122 --rounds 16 \
    --init /tmp/size/init_d216 \
    --rows /tmp/mix/rows_f0.npy \
    --holdout /tmp/mix/rows_holdout.npy \
    --reset-moments --outer nesterov --outer-lr 0.7 \
    --build-dir /tmp/size/bd \
    --out /tmp/size/d216_t122
echo "fim size216 $(date '+%F %H:%M:%S')"
