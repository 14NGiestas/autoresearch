#!/usr/bin/env bash
# Pretokenized .bin dataloader experiment (hyp_9f47fd).
#
# Removes the per-step parquet-read + re-tokenize cost that bounds the baseline
# at mfu ~0.2% (dt ~75s/step). novel/pretokenize.py flattens the hoard once into
# a single uint16 token .bin (+ meta.json); train.py then mmaps it and draws
# random windows (no per-step CPU tokenization). Selected via the default-off
# AUTORESEARCH_DATA_BIN env var, so the live baseline/fine-tune are untouched.
#
# Step 1 (CPU, defer until the baseline frees the machine to avoid contention):
#   python3 novel/pretokenize.py   # -> ~/.cache/autoresearch/data_tokenized/{train.bin,meta.json}
# Step 2:
#   ./bin/run_data_bin.sh
set -e
cd /home/pauli/autoresearch

export AUTORESEARCH_DATA_BIN="${AUTORESEARCH_DATA_BIN:-$HOME/.cache/autoresearch/data_tokenized}"
export AUTORESEARCH_DEPTH="${AUTORESEARCH_DEPTH:-14}"
export AUTORESEARCH_DEVICE_BATCH="${AUTORESEARCH_DEVICE_BATCH:-2}"
export AUTORESEARCH_TIME_BUDGET="${AUTORESEARCH_TIME_BUDGET:-14400}"
export AUTORESEARCH_SAMPLE_EVERY="${AUTORESEARCH_SAMPLE_EVERY:-200}"

# ROCm env (mirrors flake.nix shellHook so it also works outside `nix develop`)
export ROCM_PATH="${ROCM_PATH:-/nix/store/rvscgw7b3zdw2j02pghpspq33fq09mad-rocm-runtime}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1100}"
export LD_LIBRARY_PATH="$(nix-build --no-out-link -E 'with import <nixpkgs> {}; stdenv.cc.cc' 2>/dev/null)/lib:${LD_LIBRARY_PATH}"

if [ ! -f "$AUTORESEARCH_DATA_BIN/train.bin" ]; then
    echo "ERROR: $AUTORESEARCH_DATA_BIN/train.bin not found."
    echo "Run: python3 novel/pretokenize.py   (defer until baseline frees the GPU/CPU)"
    exit 1
fi

echo "Pretokenized bin train: DATA_BIN=$AUTORESEARCH_DATA_BIN DEPTH=$AUTORESEARCH_DEPTH"
exec nix develop --command python3 train.py >run_data_bin.log 2>&1
