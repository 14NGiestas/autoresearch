#!/usr/bin/env bash
# bin/wait_tmux.sh — block until a tmux pane goes idle (output stable).
# The agent (not a background script) is the watchdog: this tool just waits,
# then the agent reads the message and decides how to steer.
#
# Usage: bin/wait_tmux.sh <target> [timeout_s=1800] [stable_s=120]
#   target   e.g. 2:3
#   timeout_s  max seconds to wait (default 1800, 0 = forever)
#   stable_s   seconds of unchanged output that counts as idle (default 120)
#
# Exit 0 = pane idle (stable), 1 = timeout/usage/error. On exit prints
# the current visible tail so the caller can read the message immediately.
set -u
TARGET="${1:?usage: wait_tmux.sh <target-pane> [timeout_s] [stable_s]}"
TIMEOUT="${2:-1800}"
STABLE="${3:-120}"
INTERVAL=10

if ! tmux capture-pane -p -t "$TARGET" >/dev/null 2>&1; then
  echo "wait_tmux: target $TARGET not found" >&2
  exit 1
fi

START=$(date +%s)
LAST_HASH=""
STABLE_SINCE=$(date +%s)
while true; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - START))
  if [ "$TIMEOUT" -gt 0 ] && [ "$ELAPSED" -ge "$TIMEOUT" ]; then
    echo "wait_tmux: timeout after ${ELAPSED}s on $TARGET" >&2
    tmux capture-pane -p -t "$TARGET" 2>/dev/null | tail -n 30
    exit 1
  fi
  OUT=$(tmux capture-pane -p -S -200 -t "$TARGET" 2>&1) || {
    echo "wait_tmux: capture failed" >&2
    exit 1
  }
  HASH=$(printf '%s' "$OUT" | sha256sum | cut -d' ' -f1)
  if [ "$HASH" = "$LAST_HASH" ]; then
    if [ $((NOW - STABLE_SINCE)) -ge "$STABLE" ]; then
      echo "wait_tmux: idle after ${ELAPSED}s (stable ${STABLE}s) on $TARGET"
      tmux capture-pane -p -t "$TARGET" 2>/dev/null | tail -n 40
      exit 0
    fi
  else
    LAST_HASH="$HASH"
    STABLE_SINCE=$NOW
  fi
  sleep "$INTERVAL"
done
