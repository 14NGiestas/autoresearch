#!/bin/bash
#SBATCH -J stsmoke
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -o /home/pauli/autoresearch/logs/stsmoke_%j.log
# Fecha o gap declarado: o DEFAULT novo (--ckpt-format st) nunca foi exercitado
# num run real. Aqui: (a) run de 2 passos SEM flag -> tem de sair so
# model.safetensors (+ adam/arch/template); (b) mesmo init com --ckpt-format npy
# -> os payloads dos dois tem de bater byte a byte; (c) a guarda ckio tem de
# ABORTAR as ferramentas que ainda leem .npy sobre o checkpoint st.
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d216_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
EV=$(ls -t src/build/arch_d216_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
echo "stsmoke job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
run() {  # $1 = outdir, $2.. = flags extra
  out=$1; shift
  rm -rf "$out"
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
    --weights /tmp/mix200/init10m --rows /tmp/mix200/rows_mix200.npy --out "$out" \
    --nsteps 2 --lr 3e-4 --ntrain 64 --nval 1 --val_every 9999999 --trn_probe 1 \
    --save_every 2 --attn blas "$@" \
    --bytes "$HOME/.cache/autoresearch/tok_tables/token_bytes.txt" 2>&1 | tail -2
}
run /tmp/st_smoke_st                       # default = st
run /tmp/st_smoke_npy --ckpt-format npy    # legado
echo "--- st (default): $(ls /tmp/st_smoke_st/step_2/ | grep -cE '\.npy$') .npy (adam+muon), model.safetensors: $([ -f /tmp/st_smoke_st/step_2/model.safetensors ] && echo sim || echo NAO)"
echo "--- npy (legado): $(ls /tmp/st_smoke_npy/step_2/*.npy | wc -l) .npy"
echo "--- equivalencia byte a byte dos dois caminhos:"
.venv-numpy/bin/python3 scripts/st_read.py --ckpt-dir /tmp/st_smoke_st/step_2 \
  --npy-dir /tmp/st_smoke_npy/step_2 --compare-npy 2>&1 | tail -5
echo "--- guarda ckio sobre checkpoint st (tem de ABORTAR com mensagem clara):"
.venv-numpy/bin/python3 scripts/compose_metrics.py --shards /tmp/st_smoke_st/step_2 \
  --merge /tmp/st_smoke_st/step_2 --label smoke 2>&1 | tail -3
echo "fim stsmoke $(date '+%F %H:%M:%S')"
