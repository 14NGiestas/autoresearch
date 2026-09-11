#!/usr/bin/env bash
# bin/sync_eval.sh — push an eval run to a quiet box (e.g. halfbeast).
#
# Why a second box: inference timings next to a 16-thread trainer vary by +-20%,
# and unset thread counts cost 42x (0.87 vs 37 tok/s). Eval belongs on an idle
# machine; training keeps this one.
#
# What travels: the source tree (no build dir, no logs, no .mod), the tokenizer
# tables, and ONE checkpoint. Excludes are load-bearing: a first run shipped
# 13.5 GB because .venv (6.9G), checkpoints/ (3.2G) and hoard_multi/ (2.7G)
# rode along. On a shared box that is both rude and pointless.
#
# On a shared/cluster machine, do not run anything heavy on the login node:
# check `squeue`/`uptime` first, then submit with sbatch (halfbeast has a
# `compute` partition whose only node is halfbeast itself). The binary is built on the target instead of
# copied: -march=native differs per microarchitecture, so a binary from this
# box may not even run there, and if it does its speed is not the target's.
# OpenBLAS stays comparable because both sides build from the same flake.lock.
#
# Usage: bin/sync_eval.sh HOST CKPT_DIR [DEST]      # DEST default ~/autoresearch-eval
set -u
HOST="${1:?usage: sync_eval.sh HOST CKPT_DIR [DEST]}"
CKPT="${2:?need a checkpoint dir (e.g. /tmp/w_prose/step_500)}"
DEST="${3:-autoresearch-eval}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
CK=$(basename "$CKPT")

echo "== syncing tree, tables and $CK to $HOST:$DEST"
ssh "$HOST" "mkdir -p $DEST"
rsync -a --info=stats1 \
    --exclude 'src/build' --exclude '*.mod' --exclude 'logs' \
    --exclude '.git' --exclude 'outputs' --exclude '__pycache__' \
    --exclude '.venv' --exclude '.venv-numpy' --exclude 'checkpoints' \
    --exclude 'hoard_multi' --exclude 'packages' --exclude 'archive' \
    --exclude 'data' --exclude '*.parquet' --exclude '*.npy' \
    "$SRC/" "$HOST:$DEST/autoresearch/"
rsync -a "$HOME/.cache/autoresearch/tok_tables/" "$HOST:$DEST/tok_tables/"
rsync -a "$CKPT/" "$HOST:$DEST/$CK/"

cat <<EOF

== next, on $HOST (use sbatch on a shared box; check squeue/uptime first):
   # 1) build ON the target if you need its ISA, otherwise ship the portable
   #    bundle (fermi binary + its .so files + a loader wrapper) -- no sudo,
   #    no system change, and md5-identical outputs across machines.
   # 2) run the batteries as a job (threads: 10 = physical cores here; more
   #    hurts decode, which is bandwidth bound):
   ssh $HOST
   cd $DEST/autoresearch
   sbatch -p compute -c 10 --mem 8G --wrap "\\
     OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 \\
     bin/eval_infer.sh --weights \$HOME/$DEST/$CK --tables \$HOME/$DEST/tok_tables \\
       --chat \$HOME/$DEST/portable/run_chat.sh --only all \\
       --label \"\$(hostname -s) \$(date +%F_%H:%M)\" > eval_$CK.txt"
   # determinism check against this box: outputs must match md5 for the same
   # prompt+seed (different result = different BLAS or an argmax tie, not noise)
EOF
