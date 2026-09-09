#!/usr/bin/env bash
# Phase 3 launcher (math reasoning, 1000 steps from Phase-2 best).
#
# Chains from /tmp/w_tool/best (lowest Phase-2 val), NOT step_1000:
# the tool val gap was widening, so best/ is the honest handoff.
# Own out dir + lock + log; same blessed build flags as Phases 1-2.
# NOTE: this rebuilds via fpm run, so it picks up fortran_adam_state
# (optimizer carry). Phase-2 checkpoints have no adam_*.npy -> moments
# start fresh here and carry forward from Phase 3 on.
set -u
OPENBLAS=/nix/store/qqgfxcvq0wqp5a842pv8bcyxrc8n4sd3-openblas-0.3.33
export LD_LIBRARY_PATH="$OPENBLAS/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$OPENBLAS/lib:${LIBRARY_PATH:-}"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export OMP_DYNAMIC=FALSE
cd /home/pauli/autoresearch/src
exec flock -n /tmp/w_math/train.lock \
  fortran-fpm run --profile release --flag "-march=native -ffast-math" \
    train_run -- \
    --weights /tmp/w_tool/best \
    --rows /home/pauli/.cache/autoresearch/math_reasoning.txt \
    --out /tmp/w_math \
    --nsteps 1000 --lr 0.00003 --ntrain 120 --nval 20 --val_every 50 \
    --save_every 100 --keep_last 3 \
    --bytes /home/pauli/.cache/autoresearch/tok_tables/token_bytes.txt \
  2>&1 | tee /home/pauli/autoresearch/logs/train_phase3.log
