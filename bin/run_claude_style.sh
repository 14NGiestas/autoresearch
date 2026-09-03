#!/usr/bin/env bash
# Claude-style fine-tune launcher.
#
# Continues the baseline model (resumes its checkpoint) and trains further on the
# Claude-normalized corpus, so the model picks up Claude's writing style.
#
# Prereqs (already done):
#   - make_claude_data.py  -> ~/.cache/autoresearch/data_claude/  (train + val shards)
#   - baseline train.py run finished and saved checkpoint_depth5_step*.pt in this dir
#   - baseline tokenizer at ~/.cache/autoresearch/tokenizer/  (reused, NOT retrained)
#
# Usage:
#   ./run_claude_style.sh                 # 4h fine-tune, resumes baseline ckpt
#   AUTORESEARCH_TIME_BUDGET=7200 ./run_claude_style.sh
set -e
cd /home/pauli/autoresearch

export AUTORESEARCH_DATA_DIR="${HOME}/.cache/autoresearch/data_claude"
export AUTORESEARCH_TIME_BUDGET="${AUTORESEARCH_TIME_BUDGET:-14400}"   # 4h fine-tune
export AUTORESEARCH_RESUME="${AUTORESEARCH_RESUME:-1}"                 # continue from baseline ckpt
export AUTORESEARCH_LR_SCALE="1.0"             # last-layer fine-tune: only lm_head trains (FREEZE_BACKBONE=2), so full head LR is safe & stable
export AUTORESEARCH_FREEZE_BACKBONE="2"       # 2 = freeze transformer.h + wte, train lm_head only (no sharp-minimum divergence); 1 = also train wte
export AUTORESEARCH_SAMPLE_EVERY="${AUTORESEARCH_SAMPLE_EVERY:-50}"   # mid-run samples every 50 steps -> live quality check (coherence = not diverged)
export AUTORESEARCH_DEPTH="${AUTORESEARCH_DEPTH:-14}"                 # MUST match baseline (ckpt shape)
export AUTORESEARCH_DEVICE_BATCH="${AUTORESEARCH_DEVICE_BATCH:-2}"     # MUST match baseline

# ROCm env (mirrors flake.nix shellHook so it also works outside `nix develop`)
export ROCM_PATH="${ROCM_PATH:-/nix/store/rvscgw7b3zdw2j02pghpspq33fq09mad-rocm-runtime}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1100}"
export LD_LIBRARY_PATH="$(nix-build --no-out-link -E 'with import <nixpkgs> {}; stdenv.cc.cc' 2>/dev/null)/lib:${LD_LIBRARY_PATH}"

echo "Claude-style fine-tune: DATA_DIR=$AUTORESEARCH_DATA_DIR budget=${AUTORESEARCH_TIME_BUDGET}s resume=$AUTORESEARCH_RESUME"
exec nix develop --command python3 -u train.py > run_claude.log 2>&1
