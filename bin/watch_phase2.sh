#!/usr/bin/env bash
# Watchdog: wait for Phase 2 (step 1000), verify, auto-launch Phase 3.
#
# Fires once: when logs/train_phase2.log shows step 1000 AND the train
# process is gone, verifies /tmp/w_tool/step_1000 + best/ exist, then
# launches bin/train_phase3.sh in a new tmux window. If the trainer dies
# WITHOUT step 1000, it reports failure loudly and launches nothing.
set -u
LOG=/home/pauli/autoresearch/logs/train_phase2.log
while true; do
  if grep -q "step 1000 nll" "$LOG" 2>/dev/null; then
    sleep 120  # let final save/flush settle
    if [[ -d /tmp/w_tool/step_1000 && -d /tmp/w_tool/best ]]; then
      echo "PHASE2 FINISHED: step_1000 + best verified. Launching Phase 3."
      tmux new-window -d -t curriculum: -n phase3 \
        '/home/pauli/autoresearch/bin/train_phase3.sh'
      tmux send-keys -t curriculum:phase2 'echo "=== PHASE 2 DONE -> PHASE 3 LAUNCHED ==="' Enter
      printf '\a'
      exit 0
    else
      echo "PHASE2 step 1000 logged but checkpoints missing — NOT launching."
      printf '\a'
      exit 1
    fi
  fi
  if ! pgrep -f 'app/train_run --weights /tmp/w_10k/step_2000' >/dev/null; then
    echo "PHASE2 TRAINER DIED before step 1000 — NOT launching Phase 3."
    printf '\a'
    exit 1
  fi
  sleep 60
done
