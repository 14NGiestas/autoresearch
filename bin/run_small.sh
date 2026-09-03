#!/usr/bin/env bash
# Smaller-model generalization experiment (hyp_cb3f6f -- root-cause fix).
#
# The DEPTH=14 baseline MEMORIZES the ~400M-token hoard (0.4 tok/param) and
# collapses to constant-token repetition at generation despite val_bpb 0.203.
# This run trains a SMALL model (DEPTH=6 -> n_embd 384 -> ~12-15M params) from
# scratch on the SAME hoard, giving ~25-30 tok/param, which should force
# generalization (coherent, non-repetitive generation) instead of memorization.
#
# Uses the baseline GPT (MODEL left unset) + the baseline tokenizer, from scratch.
# Watch generation quality LIVE via the mid-run samples (SAMPLE_EVERY=50):
#   tail -f run_small.log
#
# Usage:  ./bin/run_small.sh
set -e
cd /home/pauli/autoresearch

export AUTORESEARCH_DATA_DIR="${AUTORESEARCH_DATA_DIR:-$HOME/.cache/autoresearch/data}"
export AUTORESEARCH_TIME_BUDGET="${AUTORESEARCH_TIME_BUDGET:-14400}"   # 4h
export AUTORESEARCH_RESUME="${AUTORESEARCH_RESUME:-0}"                 # from scratch
export AUTORESEARCH_DEPTH="${AUTORESEARCH_DEPTH:-6}"                   # small model (n_embd 384)
export AUTORESEARCH_DEVICE_BATCH="${AUTORESEARCH_DEVICE_BATCH:-2}"
export AUTORESEARCH_SAMPLE_EVERY="${AUTORESEARCH_SAMPLE_EVERY:-50}"    # live generation probe

# ROCm env (mirrors flake.nix shellHook so it also works outside `nix develop`)
export ROCM_PATH="${ROCM_PATH:-/nix/store/rvscgw7b3zdw2j02pghpspq33fq09mad-rocm-runtime}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1100}"
export LD_LIBRARY_PATH="$(nix-build --no-out-link -E 'with import <nixpkgs> {}; stdenv.cc.cc' 2>/dev/null)/lib:${LD_LIBRARY_PATH}"

echo "Small-model train: DEPTH=$AUTORESEARCH_DEPTH (n_embd 384, ~12-15M params) budget=${AUTORESEARCH_TIME_BUDGET}s resume=$AUTORESEARCH_RESUME"
exec nix develop --command python3 train.py > run_small.log 2>&1
