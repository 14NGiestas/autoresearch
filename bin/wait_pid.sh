#!/bin/sh
# bin/wait_pid.sh — block until a process disappears. No polling, no sleep.
# Uses tail --pid (inotify-free: the kernel wakes us on process exit).
# Usage: bin/wait_pid.sh <pid> [timeout_s]
# Exit 0 = process gone, 1 = usage/timeout error.
set -e
PID="${1:?usage: wait_pid.sh <pid> [timeout_s]}"
TIMEOUT="${2:-86400}"
case "$PID" in *[!0-9]*)
  echo "not a pid: $PID" >&2
  exit 1
  ;;
esac
if ! kill -0 "$PID" 2>/dev/null; then
  echo "pid $PID already gone"
  exit 0
fi
if command -v timeout >/dev/null 2>&1; then
  timeout "$TIMEOUT" tail --pid="$PID" -f /dev/null
  rc=$?
  if [ $rc -eq 124 ]; then
    echo "timeout after ${TIMEOUT}s" >&2
    exit 1
  fi
else
  tail --pid="$PID" -f /dev/null
fi
echo "pid $PID gone"
