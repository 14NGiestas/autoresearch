#!/bin/bash
#SBATCH -J bsweep
#SBATCH -c 8
#SBATCH -t 03:30:00
#SBATCH -o /home/pauli/autoresearch/logs/bsweep_%j.log
# SWEEP DE LOTE (hyp_c3aed6) + gate + testes, tudo DENTRO do job.
#
# Historico (para nao repetir): a primeira versao deste job copiava src/ para
# /tmp e flipava a arch com sed. Desnecessario: o fpm suporta variaveis de
# preprocessamento por feature ([features].<nome>.preprocess.cpp.macros), entao
# a arch se escolhe na INVOCACAO com --features (bin/build_arch). Sem copia, sem
# sed, sem corrida com o arquivo compartilhado -- e o binario se identifica
# sozinho (arch_id deriva dos parametros compilados).
#
# Quem roda isto precisa de maquina OCIOSA para a tabela de throughput: por isso
# a dependencia no submit (afterany dos outros jobs).
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
BUILD=/tmp/bsweep_build
D=96; H=6; KV=2; L=12; V=8192; C=1024

echo "bsweep job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')  load=$(cut -d' ' -f1 /proc/loadavg)"
echo "=== construindo a arch d96 pelo caminho DECLARATIVO (--features)"
bin/build_arch $D $H $KV $L $V $C "$BUILD" 2>&1 | sed 's/^/  /'
TRAIN=$(ls $BUILD/*/app/train_run | tail -1)
EVAL=$(ls $BUILD/*/app/eval_bpb | tail -1)
echo "  train=$TRAIN"; echo "  eval=$EVAL"

echo "=== GATE ESTRITO (o nosso codigo com -Werror; inclui os testes)"
bin/fstrict 2>&1 | tail -2 | sed 's/^/  /'

echo "=== SUITE COMPLETA"
nix develop .#cpu-only --command bash -c "cd src && fortran-fpm test --build-dir /tmp/bsweep_test" 2>&1 | tail -2 | sed 's/^/  /'

echo "=== SMOKE (6 passos, B=1) + CONTROLE DE NEUTRALIDADE"
# O binario ANTIGO d96 da arvore viva (so' EXECUCAO: leitura, nada escrito la')
# nao aceita --batch; o NOVO aceita. Tudo igual tem de dar os MESMOS nlls.
OLD=$(ls src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/train_run | tail -1)
BY="$HOME/.cache/autoresearch/tok_tables/token_bytes.txt"
# nsteps/val_every/save_every ficam FORA do COMMON (cada rodada passa o seu:
# flag duplicada no CLI e' ambiguidade que nao quero depurar num job)
COMMON="--weights /tmp/mix/init3m --rows /tmp/mix/rows_f0.npy --lr 6e-4
        --ntrain 11797 --start_row 0 --nval 1 --trn_probe 0 --attn blas --bytes $BY"
for TAG in antigo novo; do
  rm -rf /tmp/bsweep_neut_$TAG && mkdir -p /tmp/bsweep_neut_$TAG
  if [ "$TAG" = antigo ]; then BIN="$OLD"; EXTRA=""; else BIN="$TRAIN"; EXTRA="--batch 1"; fi
  # shellcheck disable=SC2086
  # shellcheck disable=SC2086
  $BIN $COMMON --nsteps 6 --val_every 6 --save_every 6 $EXTRA \
      --out /tmp/bsweep_neut_$TAG > /tmp/bsweep_neut_$TAG/log 2>&1
  echo "  $TAG: exit=$?  $(grep '^step 6' /tmp/bsweep_neut_$TAG/log | tail -1)"
done
A=$(grep '^step ' /tmp/bsweep_neut_antigo/log | tail -2 | tr -s ' ')
N=$(grep '^step ' /tmp/bsweep_neut_novo/log | tail -2 | tr -s ' ')
if [ "$A" = "$N" ] && [ -n "$A" ]; then
  echo "  NEUTRALIDADE OK: B=1 no binario novo == binario antigo"
else
  echo "  ATENCAO: B=1 DIVERGIU -- nao usar o sweep sem entender isto"
  echo "    antigo: $A"
  echo "    novo  : $N"
fi
echo "=== B=4 roda? (4 passos)"
rm -rf /tmp/bsweep_b4 && mkdir -p /tmp/bsweep_b4
# shellcheck disable=SC2086
$TRAIN $COMMON --nsteps 4 --val_every 4 --save_every 4 --batch 4 \
    --out /tmp/bsweep_b4 > /tmp/bsweep_b4/log 2>&1
echo "  exit=$?  $(grep -E '^step 4|bpb' /tmp/bsweep_b4/log | tail -1)"

echo "=== SWEEP ($(date '+%H:%M:%S')) load=$(cut -d' ' -f1 /proc/loadavg)"
$PY scripts/batch_sweep.py --train "$TRAIN" --eval "$EVAL" \
    --batches 1,4,16,64 --tokens 2000000 --probe-steps 30 --work /tmp/bsweep
echo "fim bsweep $(date '+%F %H:%M:%S')"
