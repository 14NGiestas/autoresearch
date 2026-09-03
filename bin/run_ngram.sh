#!/usr/bin/env bash
# N-gram embedding experiment (hyp_58f2c4).
#
# Selects the N-gram embedding variant via AUTORESEARCH_MODEL=ngram (train.py
# default-off switch, so the auto-launched Claude fine-tune stays on the
# baseline path). The token embedding is augmented with hashed bigram+trigram
# embeddings — a learnable-scaled local-context inductive bias that can help a
# tiny 65M-token corpus. N-gram tables have no compatible baseline checkpoint,
# so it trains FROM SCRATCH (RESUME=0). num_ngram_buckets is tunable in
# novel/ngram_model.py (default 16384; raise for more n-gram resolution, at a
# parameter cost of ~2*buckets*n_embd).
#
# Usage:  ./bin/run_ngram.sh
set -e
cd /home/pauli/autoresearch

export AUTORESEARCH_MODEL="${AUTORESEARCH_MODEL:-ngram}"
export AUTORESEARCH_DATA_DIR="${AUTORESEARCH_DATA_DIR:-$HOME/.cache/autoresearch/data}"
export AUTORESEARCH_TIME_BUDGET="${AUTORESEARCH_TIME_BUDGET:-14400}"   # 4h
export AUTORESEARCH_RESUME="${AUTORESEARCH_RESUME:-0}"                 # from scratch (arch != baseline ckpt)
export AUTORESEARCH_DEPTH="${AUTORESEARCH_DEPTH:-14}"                  # match baseline size for fair comparison
export AUTORESEARCH_DEVICE_BATCH="${AUTORESEARCH_DEVICE_BATCH:-2}"
export AUTORESEARCH_SAMPLE_EVERY="${AUTORESEARCH_SAMPLE_EVERY:-200}"   # mid-run quality sampling -> run_ngram.log

# ROCm env (mirrors flake.nix shellHook so it also works outside `nix develop`)
export ROCM_PATH="${ROCM_PATH:-/nix/store/rvscgw7b3zdw2j02pghpspq33fq09mad-rocm-runtime}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1100}"
export LD_LIBRARY_PATH="$(nix-build --no-out-link -E 'with import <nixpkgs> {}; stdenv.cc.cc' 2>/dev/null)/lib:${LD_LIBRARY_PATH}"

echo "N-gram embedding train: MODEL=$AUTORESEARCH_MODEL DEPTH=$AUTORESEARCH_DEPTH budget=${AUTORESEARCH_TIME_BUDGET}s resume=$AUTORESEARCH_RESUME"
exec nix develop --command python3 train.py > run_ngram.log 2>&1
