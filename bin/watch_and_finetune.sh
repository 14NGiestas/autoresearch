#!/usr/bin/env bash
# Hands-off continuation + self-wake.
#  - When the baseline `train` session ends, auto-launch the Claude fine-tune
#    (`claude` session, resumes the depth-14 checkpoint) AND send a wake keystroke
#    into the pi agent's tmux session so the agent logs results and continues.
#  - When the `claude` fine-tune ends, send another wake so the agent judges the
#    result and proposes the next experiment.
# Idempotent via flag files. Logs to finetune_watch.log.
cd /home/pauli/autoresearch
LOG=./finetune_watch.log
BASE_DONE=./.base_done
FT_DONE=./.ft_done
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] watcher started (polls every 60s)" >>"$LOG"

wake_agent() {
  msg="$1"
  # agent session = the tmux session whose pane command is 'pi'
  agent=$(./bin/tmux list-panes -a -F '#{session_name} #{pane_current_command}' 2>/dev/null |
    awk '$2=="pi" {print $1; exit}')
  if [ -z "$agent" ]; then
    echo "[$(ts)] no 'pi' agent session found; skip wake (training still auto-continues)" >>"$LOG"
    return
  fi
  ./bin/tmux send-keys -t "$agent" "$msg" Enter
  echo "[$(ts)] sent AUTOWAKE to session $agent" >>"$LOG"
}

while true; do
  # ---- Phase 1: wait for baseline to finish ----
  if ./bin/tmux has-session -t train 2>/dev/null; then
    sleep 60
    continue
  fi
  if [ ! -f "$BASE_DONE" ]; then
    if ls checkpoint_depth14_step*.pt 1>/dev/null 2>&1; then
      echo "[$(ts)] baseline ended; launching claude fine-tune" >>"$LOG"
      ./bin/tmux new-session -d -s claude "bash run_claude_style.sh"
      touch "$BASE_DONE"
      # give the claude session a moment to actually spawn before Phase 2 checks
      for _ in $(seq 1 30); do
        ./bin/tmux has-session -t claude 2>/dev/null && break
        sleep 5
      done
      wake_agent "AUTOWAKE baseline training finished. Read run.log for the final val_bpb and samples, record HEP evidence with that val_bpb on hyp_bdc304, confirm the claude fine-tune launched, then continue the plan."
    else
      echo "[$(ts)] train session gone but no depth14 checkpoint yet; waiting" >>"$LOG"
      sleep 30
      continue
    fi
  fi

  # ---- Phase 2: wait for claude fine-tune to finish ----
  if [ -f "$BASE_DONE" ] && ! ./bin/tmux has-session -t claude 2>/dev/null; then
    if [ ! -f "$FT_DONE" ]; then
      touch "$FT_DONE"
      wake_agent "AUTOWAKE claude fine-tune finished. Read run_claude.log for its final val_bpb and samples, record HEP evidence on hyp_c30aeb, judge whether claude style emerged, and propose the next experiment."
      echo "[$(ts)] both stages complete; watcher exiting" >>"$LOG"
      exit 0
    fi
  fi
  sleep 60
done
