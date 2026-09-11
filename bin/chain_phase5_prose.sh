#!/bin/sh
# bin/chain_phase5_prose.sh — start the prose phase when phase 4 exits.
#
# Waits on the phase-4 flock process with bin/wait_pid.sh (tail --pid: the
# kernel wakes us, no polling, no sleep loop). Then launches phase 5, which
# itself validates the handoff checkpoint before training (a directory that
# exists is not a checkpoint that saved).
#
# Usage: setsid nohup bin/chain_phase5_prose.sh > logs/chain_phase5.log 2>&1 &
set -u
cd /home/pauli/autoresearch

PID=$(pgrep -f 'flock -n /tmp/w_fortran/train.lock' | head -1)

{
    echo "=== chain_phase5 — $(date '+%Y-%m-%d %H:%M:%S') ==="
    if [ -z "$PID" ]; then
        echo "phase 4 flock not found; if it already finished, starting phase 5 now"
    else
        echo "waiting for phase 4 (pid $PID) to exit ..."
        ./bin/wait_pid.sh "$PID" 86400 || { echo "wait failed/timeout"; exit 1; }
        echo "phase 4 exited at $(date '+%H:%M:%S')"
    fi
    exec ./bin/train_phase5_prose.sh
} 2>&1
