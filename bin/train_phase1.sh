#!/usr/bin/env bash
# Phase 1 launcher (code_python, 2000 steps from /tmp/w_10k/best).
#
# Runs via `fortran-fpm run` so the binary is always the current build —
# no build-hash paths to rot. Profile + flags are pinned here; they must
# stay exactly the validated combo (changing them rebuilds at launch):
#   --profile release --flag "-march=native -ffast-math"
# (-march=native: Zen4 AVX512 for non-BLAS loops; -ffast-math gated by
#  fortran-fpm test 0 failures. BLAS via mfi fpm dep + OpenBLAS below.)
#
# flock: a second launch (manual re-run, repeated send-keys) fails fast
# instead of trampling /tmp/w_10k checkpoints under a live run.
set -u
OPENBLAS=/nix/store/qqgfxcvq0wqp5a842pv8bcyxrc8n4sd3-openblas-0.3.33
export LD_LIBRARY_PATH="$OPENBLAS/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$OPENBLAS/lib:${LIBRARY_PATH:-}"
cd /home/pauli/autoresearch/src
exec flock -n /tmp/w_10k/train.lock \
  fortran-fpm run --profile release --flag "-march=native -ffast-math" \
    train_run -- \
    --weights /tmp/w_10k/best \
    --rows /home/pauli/.cache/autoresearch/code_python.txt \
    --out /tmp/w_10k \
    --nsteps 2000 --lr 0.00003 --ntrain 40 --nval 20 --val_every 50 \
    --save_every 100 --keep_last 3 \
    --bytes /home/pauli/.cache/autoresearch/tok_tables/token_bytes.txt \
  2>&1 | tee /home/pauli/autoresearch/logs/train_phase1.log
