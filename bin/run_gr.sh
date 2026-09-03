#!/usr/bin/env bash
# Gated Residual (hyp_11df96) training launcher.
#
# Selects the GR architecture via AUTORESEARCH_MODEL=gr (train.py default-off switch,
# so the auto-launched Claude fine-tune stays on the baseline path). GR trains FROM
# SCRATCH: its extra delta/gate branches mean a baseline checkpoint cannot be loaded
# (shapes differ), so RESUME defaults to 0.
#
# Usage:
#   ./bin/run_gr.sh                       # 4h GR training at DEPTH=14
#   AUTORESEARCH_TIME_BUDGET=21600 ./bin/run_gr.sh
set -e
cd /home/pauli/autoresearch

export AUTORESEARCH_MODEL="${AUTORESEARCH_MODEL:-gr}"
export AUTORESEARCH_DATA_DIR="${AUTORESEARCH_DATA_DIR:-$HOME/.cache/autoresearch/data}"
export AUTORESEARCH_TIME_BUDGET="${AUTORESEARCH_TIME_BUDGET:-14400}"   # 4h
export AUTORESEARCH_RESUME="${AUTORESEARCH_RESUME:-0}"                 # from scratch (GR arch != baseline ckpt)
export AUTORESEARCH_DEPTH="${AUTORESEARCH_DEPTH:-14}"                  # match baseline size for fair comparison
export AUTORESEARCH_DEVICE_BATCH="${AUTORESEARCH_DEVICE_BATCH:-2}"
export AUTORESEARCH_SAMPLE_EVERY="${AUTORESEARCH_SAMPLE_EVERY:-200}"   # mid-run quality sampling -> run_gr.log

# ROCm env (mirrors flake.nix shellHook so it also works outside `nix develop`)
export ROCM_PATH="${ROCM_PATH:-/nix/store/rvscgw7b3zdw2j02pghpspq33fq09mad-rocm-runtime}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1100}"
export LD_LIBRARY_PATH="$(nix-build --no-out-link -E 'with import <nixpkgs> {}; stdenv.cc.cc' 2>/dev/null)/lib:${LD_LIBRARY_PATH}"

echo "Gated Residual train: MODEL=$AUTORESEARCH_MODEL DEPTH=$AUTORESEARCH_DEPTH budget=${AUTORESEARCH_TIME_BUDGET}s resume=$AUTORESEARCH_RESUME"
exec nix develop --command python3 train.py > run_gr.log 2>&1
