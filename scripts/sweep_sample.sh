#!/bin/sh
# scripts/sweep_sample.sh — sampling config sweep for loop-breaking.
# 2 prompts x temps x {none, penalties, nblock, both}. Metrics per run:
# max consecutive-token run + distinct ratio. OMP4 to stay polite.
# Usage: scripts/sweep_sample.sh [out_file]  (default logs/sweep_sample.log)
set -u
OUT="${1:-logs/sweep_sample.log}"
BIN=$(ls -td src/build/*/app/chat | head -1)
W=/tmp/w_lowr/step_100
P1="5552,272,350,3667,284,275,2335,738,1167,538"
P2="751,5823,277,267,460,440,114,1850,336"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export LD_LIBRARY_PATH=/nix/store/bwr2s8wnfqlbb66byqf2dgfpajicsibh-openblas-0.3.33/lib:$LD_LIBRARY_PATH
{
  echo "== sweep $(date -u +%FT%TZ)"
  for P in "$P1" "$P2"; do
    echo "-- prompt [$P]"
    for T in 0.3 0.7 1.0; do
      for CFG in "none" "pf" "nb3" "both"; do
        case "$CFG" in
        none) X="" ;;
        pf) X="--pres 1.0 --freq 1.0" ;;
        nb3) X="--nblock 3" ;;
        both) X="--pres 1.0 --freq 1.0 --nblock 3" ;;
        esac
        # shellcheck disable=SC2086
        IDS=$($BIN --weights "$W" --n 25 --ids "$P" --temp "$T" --seed 7 $X 2>/dev/null | tail -1)
        echo "temp=$T cfg=$CFG ids: $IDS"
      done
    done
  done
} | tee "$OUT"
