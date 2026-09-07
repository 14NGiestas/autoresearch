#!/usr/bin/env bash
export LD_LIBRARY_PATH="/nix/store/qqgfxcvq0wqp5a842pv8bcyxrc8n4sd3-openblas-0.3.33/lib:$LD_LIBRARY_PATH"
cd /home/pauli/autoresearch/src
exec ./build/gfortran_22BAD538E2B44D12/app/train_run \
  --weights /tmp/w_10k/best \
  --rows /home/pauli/.cache/autoresearch/code_python.txt \
  --out /tmp/w_10k \
  --nsteps 2000 --lr 0.00003 --ntrain 40 --nval 20 --val_every 50 \
  --save_every 100 --keep_last 3 \
  --bytes /home/pauli/.cache/autoresearch/tok_tables/token_bytes.txt \
  2>&1 | tee /home/pauli/autoresearch/logs/train_phase1.log
