#!/usr/bin/env bash
# autoping.sh — autonomous watchdog (heartbeat-based, no pane-text scanning).
#
# The agent touches HEARTBEAT at the start of every turn. If no new turn lands
# within IDLE_GRACE seconds the agent is not working (stuck / idle / waiting on
# something it forgot) -> tmux send-keys a wake-up turn. tmux session_activity +
# pane-hash are used only as a fallback when the heartbeat file is absent.
#
# This cannot self-match on its own ping text (we never scan pane strings) and
# does not depend on tmux being able to see the agent's TUI activity.
set -u

AGENT_SESSION="2"
POLL=10
IDLE_GRACE=120 # ping if no new agent turn for 120s (i.e. not working)
COOLDOWN=120   # at most one ping per 2 min
USER_GRACE=240 # do NOT ping if the user was active in the last 4 min (typing/reading) -> never interrupts an in-progress message
HEARTBEAT="$HOME/.cache/autoresearch/agent_heartbeat"
LOG="$HOME/.cache/autoresearch/autoping.log"
mkdir -p "$(dirname "$LOG")" "$(dirname "$HEARTBEAT")"

PING_MSG='AUTOPING: agent idle (no heartbeat turn for a while). If you are waiting on subagents or background tasks (train / watch / hoard tmux sessions, or pending tool runs), wait for them to finish. Otherwise resume the autonomous pipeline on your own: check baseline training in run.log and VRAM, log HEP evidence at milestones, and continue the staged novel-technique experiments (Gated Residual hyp_11df96, Muon refinement, N-gram embedding, pretokenized bin dataloader). Do not ask the user; proceed.'

last_ping=$(date +%s)
last_cap=""
last_change=$(date +%s)

hb_epoch() { [ -f "$HEARTBEAT" ] && stat -c %Y "$HEARTBEAT" 2>/dev/null; }

echo "[autoping] started $(date); heartbeat=$HEARTBEAT" | tee -a "$LOG"

while true; do
  sleep "$POLL"
  now=$(date +%s)
  hb=$(hb_epoch)
  if [ -n "$hb" ]; then
    idle=$((now - hb))
  else
    # fallback: tmux session activity + pane hash
    idle=999999
    ts=$(tmux display-message -t "$AGENT_SESSION" -p '#{t:session_activity}' 2>/dev/null)
    [ -n "$ts" ] && idle=$((now - $(date -d "$ts" +%s 2>/dev/null || echo "$now")))
    cap=$(tmux capture-pane -t "$AGENT_SESSION" -p -S -6 2>/dev/null | tr -d '\n' | md5sum | awk '{print $1}')
    if [ -n "$cap" ] && [ "$cap" != "$last_cap" ]; then
      last_cap="$cap"
      last_change=$now
    fi
    [ -n "${last_change:-}" ] && d=$((now - last_change)) && [ "$d" -lt "$idle" ] && idle=$d
  fi

  # user-activity guard: if the user typed/viewed recently they are engaged ->
  # never send a ping that would interrupt a message they are composing.
  ua_ts=$(tmux display-message -t "$AGENT_SESSION" -p '#{t:session_activity}' 2>/dev/null)
  user_idle=999999
  if [ -n "$ua_ts" ]; then
    ua_epoch=$(date -d "$ua_ts" +%s 2>/dev/null)
    [ -n "$ua_epoch" ] && user_idle=$((now - ua_epoch))
  fi

  since_ping=$((now - last_ping))
  if [ "$user_idle" -lt "$USER_GRACE" ]; then
    # user engaged (typing/reading) -> stay silent, do not interrupt
    continue
  fi
  if [ "$idle" -gt "$IDLE_GRACE" ] && [ "$since_ping" -gt "$COOLDOWN" ]; then
    echo "[autoping $(date)] idle ${idle}s (user_idle ${user_idle}s) -> pinging session '$AGENT_SESSION'" | tee -a "$LOG"
    tmux send-keys -t "$AGENT_SESSION" "$PING_MSG" Enter
    last_ping=$now
  fi
done
