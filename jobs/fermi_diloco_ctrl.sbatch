#!/bin/bash
#SBATCH -J dlctrl
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -o /home/pauli/autoresearch/logs/dlctrl_%j.log
# CONTROLE do resultado do DiLoCo: descobri que o caminho --outer NAO escrevia o
# estado do Adam (save_ckpt_dir sem state=), enquanto o federado puro media e
# carregava os momentos. Ou seja, os workers do DiLoCo reiniciavam o Adam a cada
# rodada -- o ganho (2.690 -> 2.332 em tau=122) pode ser WARM RESTART, nao o
# otimizador externo. Aqui os dois controles que separam:
#   C: outer=none + momentos ZERADOS  -> isola o reset
#   D: outer=nesterov + momentos CARREGADOS -> isola o outer
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
echo "dlctrl job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
echo "--- C: federado puro, momentos zerados a cada rodada"
$PY scripts/fedavg_rounds.py --k 4 --tau 122 --rounds 16 --out /tmp/dlctrl/resetmom --reset-moments
echo "--- D: DiLoCo com momentos carregados"
$PY scripts/fedavg_rounds.py --k 4 --tau 122 --rounds 16 --out /tmp/dlctrl/carrymom \
    --outer nesterov --outer-lr 0.7 --outer-mom 0.9 --carry-outer-state
echo "fim dlctrl $(date '+%F %H:%M:%S')"
