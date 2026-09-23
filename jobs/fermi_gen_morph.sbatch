#!/bin/bash
#SBATCH -J genmorph
#SBATCH -c 2
#SBATCH -t 00:20:00
#SBATCH -o /home/pauli/autoresearch/logs/genmorph_%j.log
# Geracoes do morph p/ H2 (2 prompts; encoder nativo = caveat registrado).
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/chat_text | head -1)
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OMP_DYNAMIC=FALSE
{ echo "Capitão Felisberto ancorou a sumaca no porto de Santos numa tarde de nevoeiro. Trazia no porão um caixote lacrado, sem remetente, e uma carta que não ousava abrir. O filho, Manuel, esperava no cais."; echo "Manuel correu para o pai e gritou:"; } | "$BIN" \
  --tables "$HOME/.cache/autoresearch/tok_tables" --weights /tmp/mix/w_f_morph/best \
  --n 200 --temp 0.7 --stream T --pchunk 64 > /tmp/h2_f_morph.txt 2>&1
echo "fim genmorph $(date '+%F %H:%M:%S')"
