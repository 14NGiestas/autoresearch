#!/usr/bin/env bash
# Phase 2 launcher (tool trajectories, 1000 steps from Phase-1 step_2000).
#
# Starts from step_2000, NOT best/: best/ predates Phase 1 (frozen Sep 6,
# Phase-1 val never beat it), so best/ discards all code learning.
# Chaining step_2000 keeps it; Phase 2 has its own val set + best-tracking.
# Same blessed build as Phase 1 (see train_phase1.sh header).
set -u
OPENBLAS=/nix/store/qqgfxcvq0wqp5a842pv8bcyxrc8n4sd3-openblas-0.3.33
export LD_LIBRARY_PATH="$OPENBLAS/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$OPENBLAS/lib:${LIBRARY_PATH:-}"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export OMP_DYNAMIC=FALSE
cd /home/pauli/autoresearch/src
exec flock -n /tmp/w_tool/train.lock \
  fortran-fpm run --profile release --flag "-march=native -ffast-math" \
    train_run -- \
    --weights /tmp/w_10k/step_2000 \
    --rows /home/pauli/.cache/autoresearch/tool_trajectories.txt \
    --out /tmp/w_tool \
    --nsteps 1000 --lr 0.00003 --ntrain 100 --nval 20 --val_every 50 \
    --save_every 100 --keep_last 3 \
    --bytes /home/pauli/.cache/autoresearch/tok_tables/token_bytes.txt \
  2>&1 | tee /home/pauli/autoresearch/logs/train_phase2.log
