#!/usr/bin/env bash
# Phase 1 launcher (code_python, 2000 steps from /tmp/w_10k/best).
#
# Blessed build (run from ~/autoresearch/src):
#   OPENBLAS=<store-path> LD_LIBRARY_PATH=$OPENBLAS/lib \
#   LIBRARY_PATH=$OPENBLAS/lib \
#     fortran-fpm build --profile release --flag "-march=native -ffast-math"
# (-march=native: Zen4 AVX512 for non-BLAS loops; -ffast-math gated by
#  fortran-fpm test 0 failures.)
#
# Why exec a binary instead of `fortran-fpm run train_run -- ...`?
# fpm's incrementality silently skips recompiles and links stale objects
# (observed twice: stale .mod + skipped app rebuilds). Exec'ing the binary
# directly makes staleness impossible. BIN below resolves to the newest
# train_run so build-hash rotations (flags/profile changes) don't rot this
# script; the resolved path is echoed so the log always shows WHAT ran.
#
# flock: a second launch (manual re-run, repeated send-keys, buffered keys
# executing after a kill) fails fast instead of trampling /tmp/w_10k
# checkpoints and truncating the log under a live run.
set -u
OPENBLAS=/nix/store/qqgfxcvq0wqp5a842pv8bcyxrc8n4sd3-openblas-0.3.33
export LD_LIBRARY_PATH="$OPENBLAS/lib:${LD_LIBRARY_PATH:-}"
cd /home/pauli/autoresearch/src
BIN=$(ls -t build/gfortran_*/app/train_run 2>/dev/null | head -1)
[ -n "$BIN" ] && [ -x "$BIN" ] || { echo "no train_run binary under build/"; exit 1; }
echo "BIN=$BIN"
# shellcheck disable=SC2094
exec flock -n /tmp/w_10k/train.lock "$BIN" \
  --weights /tmp/w_10k/best \
  --rows /home/pauli/.cache/autoresearch/code_python.txt \
  --out /tmp/w_10k \
  --nsteps 2000 --lr 0.00003 --ntrain 40 --nval 20 --val_every 50 \
  --save_every 100 --keep_last 3 \
  --bytes /home/pauli/.cache/autoresearch/tok_tables/token_bytes.txt \
  2>&1 | tee /home/pauli/autoresearch/logs/train_phase1.log
