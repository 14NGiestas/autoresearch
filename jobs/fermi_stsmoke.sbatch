#!/bin/bash
#SBATCH -J stsmoke
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -o /home/pauli/autoresearch/logs/stsmoke_%j.log
# Verifica o DEFAULT NOVO (--ckpt-format both) num run real de 2 passos, arch d216:
# o checkpoint tem de sair com .npy E model.safetensors, e os payloads tem de
# bater byte a byte. Cobre o que a suite ainda NAO cobre (o caminho `both`).
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d216_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
echo "stsmoke job=${SLURM_JOB_ID} bin=$BIN $(date '+%F %H:%M:%S')"
rm -rf /tmp/st_smoke_default
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
  --weights /tmp/mix200/init10m --rows /tmp/mix200/rows_mix200.npy \
  --out /tmp/st_smoke_default --nsteps 2 --lr 3e-4 --ntrain 64 --nval 1 \
  --val_every 9999999 --trn_probe 1 --save_every 2 --attn blas \
  --bytes "$HOME/.cache/autoresearch/tok_tables/token_bytes.txt" 2>&1 | tail -3
echo "--- arquivos do step_2:"
ls /tmp/st_smoke_default/step_2/ | grep -cE "\.npy$" | xargs echo "  npy:"
ls /tmp/st_smoke_default/step_2/model.safetensors 2>/dev/null && echo "  model.safetensors: presente"
echo "--- conferencia byte a byte (leitor de referencia, sem dependencia):"
.venv-numpy/bin/python3 scripts/st_read.py --ckpt-dir /tmp/st_smoke_default/step_2 \
  --npy-dir /tmp/st_smoke_default/step_2 --compare-npy 2>&1 | tail -6
echo "fim stsmoke $(date '+%F %H:%M:%S')"
