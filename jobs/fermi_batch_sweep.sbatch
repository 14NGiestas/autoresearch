#!/bin/bash
#SBATCH -J bsweep
#SBATCH -c 8
#SBATCH -t 02:30:00
#SBATCH -o /home/pauli/autoresearch/logs/bsweep_%j.log
# SWEEP DE LOTE (hyp_c3aed6): quanto a eficiencia de passo compra.
#
# Cuidados deliberados (o job 137 ainda esta' rodando na fila):
#   * NADA e' construido em src/build -- a arvore viva nao e' tocada. O build
#     sai de uma COPIA de src/ + energy-fortran/ em /tmp, com a arch flipada
#     para d96 (a dos experimentos em voo) NA COPIA.
#   * scripts/fedavg_rounds.py NAO e' tocado: jobs pendentes (138/140/139) leem
#     esse arquivo no start.
#   * O sweep roda train_run/eval_bpb DIRETO, sem o driver -- nao depende de
#     pick() nem do nome da pasta de build.
#
# Ordem: copia -> flip arch -> build -> SMOKE (6 passos, B=1) -> sweep.
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
SCRATCH=/tmp/bsweep_src
BUILD=/tmp/bsweep_build

echo "bsweep job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
echo "=== copiando a arvore para $SCRATCH (sem build/)"
rm -rf "$SCRATCH" && mkdir -p "$SCRATCH"
rsync -a --exclude 'build' --exclude '.git' src/ "$SCRATCH/src/"
rsync -a --exclude 'build' --exclude '.git' energy-fortran/ "$SCRATCH/energy-fortran/"

echo "=== flipando a arch da COPIA para d96 (a dos experimentos em voo)"
F="$SCRATCH/src/lib/fortran_arch.f90"
sed -i 's/D_MODEL = 216/D_MODEL = 96/' "$F"    # literal: \196 e' ambiguo (backref 96)
grep -E "integer, parameter :: (D_MODEL|N_HEAD|N_KV|N_LAYER|VV|TT|BOS|HD) *=" "$F" | sed 's/^/    /'
# a verificacao de arch de verdade e o SMOKE abaixo: require_arch aborta se
# a copia nao casar com /tmp/mix/init3m (foi assim que a divergencia d216/d96
# apareceu hoje).

echo "=== build (scratch, fora de src/build)"
nix develop .#cpu-only --command bash -c "cd $SCRATCH/src && fortran-fpm build --build-dir $BUILD" 2>&1 | tail -2
TRAIN=$(ls $BUILD/*/app/train_run | tail -1)
EVAL=$(ls $BUILD/*/app/eval_bpb | tail -1)
echo "    train=$TRAIN"
echo "    eval=$EVAL"

echo "=== SMOKE: 6 passos com B=1 (o mesmo caminho do sweep, minusculo)"
rm -rf /tmp/bsweep_smoke && mkdir -p /tmp/bsweep_smoke
$TRAIN --weights /tmp/mix/init3m --rows /tmp/mix/rows_f0.npy --out /tmp/bsweep_smoke \
    --nsteps 6 --lr 6e-4 --ntrain 11797 --start_row 0 --nval 1 --val_every 6 \
    --trn_probe 0 --save_every 6 --attn blas --batch 1 \
    --bytes "$HOME/.cache/autoresearch/tok_tables/token_bytes.txt" > /tmp/bsweep_smoke/log 2>&1
RC=$?
echo "    exit=$RC (esperado 0)"
tail -3 /tmp/bsweep_smoke/log | cut -c1-90
[ "$RC" -eq 0 ] || { echo "SMOKE FALHOU -- parando aqui"; exit 1; }
ls /tmp/bsweep_smoke | head -3 | sed 's/^/    /'

# --- CONTROLE DE NEUTRALIDADE: o binario ANTIGO d96 (sem --batch, da arvore
# viva, so' execucao) contra o NOVO com --batch 1. Tudo igual tem de dar igual;
# se divergir, eu mudei comportamento sem querer (allocatable vs estatico).
OLD=$(ls /home/pauli/autoresearch/src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/train_run | tail -1)
echo "=== NEUTRALIDADE B=1 (antigo=$OLD)"
for TAG in antigo novo; do
  rm -rf /tmp/bsweep_neut_$TAG && mkdir -p /tmp/bsweep_neut_$TAG
  if [ "$TAG" = antigo ]; then BIN="$OLD"; EXTRA=""; else BIN="$TRAIN"; EXTRA="--batch 1"; fi
  $BIN --weights /tmp/mix/init3m --rows /tmp/mix/rows_f0.npy --out /tmp/bsweep_neut_$TAG \
      --nsteps 6 --lr 6e-4 --ntrain 11797 --start_row 0 --nval 1 --val_every 6 \
      --trn_probe 0 --save_every 6 --attn blas $EXTRA \
      --bytes "$HOME/.cache/autoresearch/tok_tables/token_bytes.txt" > /tmp/bsweep_neut_$TAG/log 2>&1
  echo "  $TAG: exit=$? $(grep -oE 'bpb[^ ]* [0-9.]+|loss[^ ]* [0-9.]+' /tmp/bsweep_neut_$TAG/log | tail -2 | tr '\n' ' ')"
done
N=$(grep -oE '[0-9]+\.[0-9]+e?[-+]?[0-9]*' /tmp/bsweep_neut_antigo/log | tail -3 | tr '\n' ',')
M=$(grep -oE '[0-9]+\.[0-9]+e?[-+]?[0-9]*' /tmp/bsweep_neut_novo/log | tail -3 | tr '\n' ',')
echo "  ultimos numeros -- antigo: $N | novo: $M"
[ "$N" = "$M" ] && echo "  NEUTRALIDADE OK (B=1 identico)" || echo "  ATENCAO: B=1 divergiu -- investigar antes de usar o sweep"

echo "=== SWEEP ($(date '+%H:%M:%S'))"
$PY scripts/batch_sweep.py --train "$TRAIN" --eval "$EVAL" \
    --batches 1,4,16,64 --tokens 2000000 --probe-steps 30 --work /tmp/bsweep
echo "fim bsweep $(date '+%F %H:%M:%S')"
