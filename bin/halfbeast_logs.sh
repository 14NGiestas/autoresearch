#!/usr/bin/env bash
# bin/halfbeast_logs.sh — where the remote eval logs and results live.
#
# Two kinds of output exist on halfbeast, and they are easy to confuse:
#   * JOB logs     = Slurm stdout/stderr (sbatch -o files; --wrap writes
#                    slurm-<id>.out next to the cwd at SUBMIT time, which is
#                    why one of ours ended up in portable/ instead of
#                    autoresearch/).
#   * RESULT files = the artifacts the job produced, always under
#                    ~/autoresearch-eval/ (batteries_*.txt, gate_*.txt).
# In-flight per-checkpoint temporaries live in /tmp on the remote.
#
# Usage: bin/halfbeast_logs.sh [tail-lines]     # default 40
set -u
HOST="${HOST:-halfbeast}"
N="${1:-40}"
ssh "$HOST" "N=$N bash -s" <<'REMOTE'
N="${N:-40}"
EV="$HOME/autoresearch-eval"
echo "=== queue (empty = nothing running)"
squeue -o '%i %T %M %j' 2>/dev/null | sed 's/^/  /'
echo
echo "=== RESULT files (the numbers): $EV/"
ls -lt "$EV"/*.txt 2>/dev/null | head -12 | awk '{printf "  %s %s %s\n", $6, $7, $9}'
echo
echo "=== JOB logs (stdout), newest first:"
ls -lt "$EV"/autoresearch/*.log "$EV"/autoresearch/slurm-*.out "$EV"/portable/slurm-*.out 2>/dev/null \
  | head -6 | awk '{printf "  %s %s %s\n", $6, $7, $9}'
newest=$(ls -t "$EV"/autoresearch/*.log "$EV"/autoresearch/slurm-*.out "$EV"/portable/slurm-*.out 2>/dev/null | head -1)
if [ -n "$newest" ]; then
  echo
  echo "=== tail -$N of $newest"
  tail -"$N" "$newest" | sed 's/^/  /'
fi
echo
echo "=== live recipes"
echo "  ssh  'tail -f \$(ls -t ~/autoresearch-eval/autoresearch/*.log | head -1)'   # job stdout"
echo "  ssh  'tail -f \$(ls -t ~/autoresearch-eval/*.txt | head -1)'                 # result artifact"
REMOTE
