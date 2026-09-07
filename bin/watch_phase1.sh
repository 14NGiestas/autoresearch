#!/usr/bin/env bash
# Watches Phase 1 training with ~zero tokens: loops in background, records
# a one-line status file. Exits loudly on death/crash/finish.
# usage: watch_phase1.sh <pid> <log> <watchfile>
set -u
PID="$1"; LOG="$2"; WATCH="$3"
# Loud channels: terminal bell (tmux monitor-bell flags the window) +
# desktop notification (best-effort). The agent cannot be auto-woken;
# these ping the human instead.
ring() { printf '\a'; notify-send -u critical "phase1: $1" "$2" 2>/dev/null || true; }
say() { echo "$(date +%H:%M) $*" > "$WATCH"; }
if ! kill -0 "$PID" 2>/dev/null; then say "DEAD-AT-START tail: $(tail -1 "$LOG" 2>/dev/null)"; ring dead "process already gone"; exit 1; fi
while kill -0 "$PID" 2>/dev/null; do
  if grep -qE 'illegal value|SIGFPE|SIGSEGV|STOP [0-9]+' "$LOG" 2>/dev/null; then
    say "ALERT bad-pattern tail: $(tail -1 "$LOG")"; ring ALERT "$(tail -1 "$LOG")"; exit 2
  fi
  if tail -2 "$LOG" 2>/dev/null | grep -q '^step 2000'; then
    say "FINISHED tail: $(tail -1 "$LOG")"; ring finished "step 2000 done"; exit 0
  fi
  say "RUNNING pid=$PID tail: $(tail -1 "$LOG" 2>/dev/null)"
  sleep 60
done
say "EXITED tail: $(tail -1 "$LOG" 2>/dev/null)"
ring exited "training process died"
exit 3
