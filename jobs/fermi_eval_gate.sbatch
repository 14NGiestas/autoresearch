#!/bin/bash
#SBATCH -J evalgate
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -o /home/pauli/autoresearch/logs/evalgate_%j.log
# O eval NAO pode depender do numero de threads. Este job prova isso.
#
# Por que existe: em 2026-09-20 o eval_bpb deu 2,340146 com OMP=1 e 4,661770 com
# OMP=16, porque `base` estava fora da clausula private de um !$omp parallel do.
# O bug contaminou a varredura de arquitetura e o inventario de tamanho, e a
# reavaliacao INVERTEU a conclusao do eixo tamanho:
#
#   inventario antigo: d96 4,542 / d216 4,496 / d360 4,724  (nao monotonico)
#   reavaliado:        d96 2,46  / d216 2,09  / d360 1,92   (monotonico)
#
# O que este job checa, e cada afirmacao pode falhar sozinha:
#   1. INVARIANCIA: OMP=1 e OMP=8 dao o mesmo bpb.
#   2. REGRESSAO: o valor bate com o esperado para o checkpoint.
#
# Nao cabe no gate de compilacao (bin/fstrict): aquele e' rapido de proposito, e
# este roda o eval duas vezes. Entao ele e' um job, para rodar sozinho, quando
# ninguem estiver medindo tempo.
#
# Uso: sbatch jobs/fermi_eval_gate.sbatch [CKPT] [ESPERADO]
set -u
cd /home/pauli/autoresearch
echo "evalgate job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S') load=$(cut -d' ' -f1 /proc/loadavg)"
bin/check_eval_threads "$@"
RC=$?
echo "evalgate rc=$RC $(date '+%F %H:%M:%S')"
exit "$RC"
