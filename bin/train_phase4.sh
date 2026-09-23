#!/usr/bin/env bash
# Phase 4 launcher (fortran tutorials, 1000 steps from Phase-3 step_1000).
#
# Handoff: step_1000 (post-full-Phase-3, train NLL ~0.11), NOT best/
# (stale @ step ~55, val curve lost with the Phase-3 log). Adam moments
# carry via step_1000/adam_*.npy (Cognivolve: never reset across phases).
# Own out dir + lock + log. Logging goes DIRECT to file (no tee): the
# Phase-3 tee died with its tmux session, freezing the log at step 292
# while training ran 700 more steps. Tmux is only a container now.
set -u
OPENBLAS=/nix/store/qqgfxcvq0wqp5a842pv8bcyxrc8n4sd3-openblas-0.3.33
export LD_LIBRARY_PATH="$OPENBLAS/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$OPENBLAS/lib:${LIBRARY_PATH:-}"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export OMP_DYNAMIC=FALSE
TRAIN_BIN=/home/pauli/autoresearch/src/build/live/gfortran_7E92E43D69CA2C27/app/train_run
mkdir -p /tmp/w_fortran  # flock does not create the lock dir
cd /home/pauli/autoresearch/src
exec flock -n /tmp/w_fortran/train.lock "$TRAIN_BIN" \
    --weights /tmp/w_math/step_1000 \
    --rows /home/pauli/.cache/autoresearch/fortran_tutorial.txt \
    --out /tmp/w_fortran \
    --nsteps 1000 --lr 0.00003 --ntrain 140 --nval 20 --val_every 50 \
    --save_every 100 --keep_last 3 \
    --bytes /home/pauli/.cache/autoresearch/tok_tables/token_bytes.txt \
  >>/home/pauli/autoresearch/logs/train_phase4.log 2>&1
