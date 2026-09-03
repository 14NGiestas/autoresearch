#!/usr/bin/env bash
# Muon refinement experiment (hyp_25a8e7).
#
# Muon (with the NorMuon variance-reduction + polar-express orthogonalization) is
# already the optimizer in train.py. This experiment REFINES its hyperparameters —
# matrix LR and weight decay — which are now env-gated (AUTORESEARCH_MATRIX_LR /
# AUTORESEARCH_WEIGHT_DECAY), so a refined run is launched without code edits.
# It resumes the baseline depth-14 checkpoint and continues with the refined Muon
# settings (the baseline path, since MODEL is left unset). Values below are a
# starting hypothesis; tune them once the baseline val_bpb is known.
#
# Usage:  ./bin/run_muon_refine.sh
set -e
cd /home/pauli/autoresearch

export AUTORESEARCH_DATA_DIR="${AUTORESEARCH_DATA_DIR:-$HOME/.cache/autoresearch/data}"
export AUTORESEARCH_TIME_BUDGET="${AUTORESEARCH_TIME_BUDGET:-14400}"   # 4h
export AUTORESEARCH_RESUME="${AUTORESEARCH_RESUME:-1}"                 # continue baseline weights
export AUTORESEARCH_DEPTH="${AUTORESEARCH_DEPTH:-14}"
export AUTORESEARCH_DEVICE_BATCH="${AUTORESEARCH_DEVICE_BATCH:-2}"
export AUTORESEARCH_SAMPLE_EVERY="${AUTORESEARCH_SAMPLE_EVERY:-200}"
# Refined Muon hyperparameters (override freely):
export AUTORESEARCH_MATRIX_LR="${AUTORESEARCH_MATRIX_LR:-0.06}"
export AUTORESEARCH_WEIGHT_DECAY="${AUTORESEARCH_WEIGHT_DECAY:-0.3}"

# ROCm env (mirrors flake.nix shellHook so it also works outside `nix develop`)
export ROCM_PATH="${ROCM_PATH:-/nix/store/rvscgw7b3zdw2j02pghpspq33fq09mad-rocm-runtime}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1100}"
export LD_LIBRARY_PATH="$(nix-build --no-out-link -E 'with import <nixpkgs> {}; stdenv.cc.cc' 2>/dev/null)/lib:${LD_LIBRARY_PATH}"

echo "Muon refinement train: MATRIX_LR=$AUTORESEARCH_MATRIX_LR WEIGHT_DECAY=$AUTORESEARCH_WEIGHT_DECAY DEPTH=$AUTORESEARCH_DEPTH"
exec nix develop --command python3 train.py > run_muon_refine.log 2>&1
