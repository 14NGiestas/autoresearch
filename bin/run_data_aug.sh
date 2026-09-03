#!/usr/bin/env bash
# Data-augmentation generalization run (hyp_87d151 -> supports hyp_cb3f6f).
#
# Trains the SAME small model (DEPTH=6, n_embd 384, ~17M params) on the
# HOARDED English multi-turn corpus (data_en_extra: 463,764 docs from
# HuggingFaceH4/ultrachat_200k) instead of the default hoard. More
# tokenizer-compatible tokens per param raises the tok/param ratio, which is
# the root-cause lever against the baseline's memorization collapse.
#
# Periodic checkpointing is ON (AUTORESEARCH_CKPT_EVERY=300) so `chat.py`
# can use an intermediate model without waiting for the full run:
#   python3 chat.py "hi"
#
# NOTE: the `small` job uses ~2.2 GiB VRAM; launch this AFTER it finishes
# (or in parallel only if VRAM headroom allows — two jobs share the iGPU).
#
# Usage:  ./bin/run_data_aug.sh
set -e
cd /home/pauli/autoresearch

export AUTORESEARCH_DATA_DIR="${AUTORESEARCH_DATA_DIR:-$HOME/.cache/autoresearch/data_en_extra}"
export AUTORESEARCH_TIME_BUDGET="${AUTORESEARCH_TIME_BUDGET:-14400}"   # 4h
export AUTORESEARCH_RESUME="${AUTORESEARCH_RESUME:-0}"                 # from scratch
export AUTORESEARCH_DEPTH="${AUTORESEARCH_DEPTH:-6}"                   # small model (n_embd 384)
export AUTORESEARCH_DEVICE_BATCH="${AUTORESEARCH_DEVICE_BATCH:-2}"
export AUTORESEARCH_SAMPLE_EVERY="${AUTORESEARCH_SAMPLE_EVERY:-50}"    # live generation probe
export AUTORESEARCH_CKPT_EVERY="${AUTORESEARCH_CKPT_EVERY:-300}"       # mid-run checkpoints for chat.py

# ROCm env (mirrors flake.nix shellHook so it also works outside `nix develop`)
export ROCM_PATH="${ROCM_PATH:-/nix/store/rvscgw7b3zdw2j02pghpspq33fq09mad-rocm-runtime}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1100}"
export LD_LIBRARY_PATH="$(nix-build --no-out-link -E 'with import <nixpkgs> {}; stdenv.cc.cc' 2>/dev/null)/lib:${LD_LIBRARY_PATH}"

echo "Data-aug train: DEPTH=$AUTORESEARCH_DEPTH on $AUTORESEARCH_DATA_DIR budget=${AUTORESEARCH_TIME_BUDGET}s resume=$AUTORESEARCH_RESUME"
exec nix develop --command python3 train.py > run_data_aug.log 2>&1
