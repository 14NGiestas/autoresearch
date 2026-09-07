#!/usr/bin/env bash
# bin/run_curriculum.sh — 4-phase curriculum trainer.
#
# Each phase trains on its own data file, resumes from the previous phase's
# best checkpoint, and stops when val-bpb stops improving.
#
# Phases:
#   1  code-python     code_python.txt         (~15k Alpaca-code rows)
#   2  tool-use        tool_trajectories.txt   (curl/fetch/python tool trajectories)
#   3  math-reason     math_reasoning.txt      (step-by-step math reasoning)
#   4  fortran-tut     fortran_tutorial.txt    (fortran-lang webpage tutorials)
#
# After all 3 phases, best/ contains the curriculum-trained model.
#
# Usage:
#   ./bin/run_curriculum.sh                # full 3-phase run
#   ./bin/run_curriculum.sh --phase 2     # resume from phase 2
#   ./bin/run_curriculum.sh --dry-run      # print commands without running

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE="${AUTORESEARCH_CACHE:-$HOME/.cache/autoresearch}"
DATA_DIR="$CACHE"

# Where weights land
OUT_DIR="${CURRICULUM_OUT:-$DATA_DIR/curriculum}"

# Binary
FPM="${FORTRAN_FPM:-fortran-fpm}"
if [[ -z "${SRC_DIR:-}" ]]; then
    _cwd=$(cd "$(dirname "$0")/.." && pwd)
    SRC_DIR="$_cwd/src"
fi

# Training defaults
NSTEPS="${CURRICULUM_STEPS:-2000}" # steps per phase
LR="${CURRICULUM_LR:-0.00003}"
NTRAIN="${CURRICULUM_NTRAIN:-40}"
NVAL="${CURRICULUM_NVAL:-20}"
VAL_EVERY="${CURRICULUM_VAL_EVERY:-50}"
SAVE_EVERY="${CURRICULUM_SAVE_EVERY:-100}"
KEEP_LAST="${CURRICULUM_KEEP_LAST:-3}"

# Phase definitions: name | rows_file | description
PHASES=(
    "1:code-python:${DATA_DIR}/code_python.txt:CodeAlpaca-20k Python instructions"
    "2:tool-use:${DATA_DIR}/tool_trajectories.txt:curl/fetch/python tool trajectories"
    "3:math-reason:${DATA_DIR}/math_reasoning.txt:Step-by-step math reasoning"
    "4:fortran-tut:${DATA_DIR}/fortran_tutorial.txt:fortran-lang webpage tutorials"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }
warn() { echo "[$(date +%H:%M:%S)] WARNING: $*" >&2; }

find_binary() {
    local bin
    # Sort build dirs by mtime, newest first; use array to avoid subshell glob issues
    mapfile -t dirs < <(ls -td "$SRC_DIR"/build/gfortran_*/ 2>/dev/null)
    for d in "${dirs[@]}"; do
        bin="$d/app/train_run"
        if [[ -x "$bin" ]]; then
            echo "$bin"
            return 0
        fi
    done
    log "Building train_run..."
    (cd "$SRC_DIR" && $FPM build --no-color --target train_run 2>&1 | tail -3)
    mapfile -t dirs < <(ls -td "$SRC_DIR"/build/gfortran_*/ 2>/dev/null)
    for d in "${dirs[@]}"; do
        bin="$d/app/train_run"
        if [[ -x "$bin" ]]; then
            echo "$bin"
            return 0
        fi
    done
    echo "ERROR: train_run binary not found" >&2
    return 1
}

# Check that a rows file exists and has enough rows
check_rows() {
    local file="$1"
    local min="${2:-100}"
    if [[ ! -f "$file" ]]; then
        echo "MISSING rows file: $file"
        echo "  Run:  python scripts/prepare_code.py   # Phase 1"
        echo "  Run:  python scripts/prepare_tool.py   # Phase 2 (TODO)"
        echo "  Run:  python scripts/prepare_math.py    # Phase 3 (TODO)"
        return 1
    fi
    local lines
    lines=$(wc -l <"$file")
    if ((lines < min)); then
        warn "$file has only $lines rows (min $min) — increase --limit or run the prep script"
        return 1
    fi
    echo "  rows file: $file ($lines rows)"
    return 0
}

# ---------------------------------------------------------------------------
# Run one phase
# ---------------------------------------------------------------------------

run_phase() {
    local phase_num="$1"
    local name="$2"
    local rows_file="$3"
    local desc="$4"
    local start_weights="${5:-}" # empty = train from scratch

    local phase_dir="$OUT_DIR/phase_${phase_num}_${name}"
    local best_dir="$phase_dir/best"

    log "=============================================="
    log "Phase $phase_num: $name — $desc"
    log "Phase dir: $phase_dir"
    log "=============================================="

    mkdir -p "$phase_dir"

    local bin
    bin=$(find_binary)
    if [[ -z "$bin" || ! -x "$bin" ]]; then
        warn "train_run binary not found — aborting"
        return 1
    fi

    local rows_url=""
    case "$name" in
    code-python) rows_url="sahil2801/CodeAlpaca-20k (20k Python instructions)" ;;
    tool-use) rows_url="TODO: scripts/prepare_tool.py" ;;
    math-reason) rows_url="TODO: scripts/prepare_math.py" ;;
    esac
    echo "  source: $rows_url"

    if ! check_rows "$rows_file" 100; then
        warn "Skipping phase $phase_num (missing rows)"
        return 1
    fi

    # Pick weights: use start_weights if provided, else previous phase's best
    local weights_arg=""
    if [[ -n "$start_weights" && -d "$start_weights" ]]; then
        weights_arg="--weights $start_weights"
        echo "  resuming from: $start_weights"
    elif [[ -d "$best_dir" ]]; then
        weights_arg="--weights $best_dir"
        echo "  resuming from previous phase best: $best_dir"
    else
        echo "  training from scratch (no start weights)"
    fi

    log "Starting training..."
    log "  nsteps=$NSTEPS  lr=$LR  val_every=$VAL_EVERY"
    log "  rows=$rows_file"
    log "  out=$phase_dir"

    # Build command
    local cmd=(
        "$bin"
        --weights "${weights_arg:-$phase_dir/init}"
        --rows "$rows_file"
        --out "$phase_dir"
        --nsteps "$NSTEPS"
        --lr "$LR"
        --ntrain "$NTRAIN"
        --nval "$NVAL"
        --val_every "$VAL_EVERY"
        --save_every "$SAVE_EVERY"
        --keep_last "$KEEP_LAST"
        --bytes "$CACHE/tok_tables/token_bytes.txt"
    )

    echo "  cmd: ${cmd[*]}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        log "[DRY RUN] would execute: ${cmd[*]}"
        return 0
    fi

    # Run; tee to a phase log
    mkdir -p "$phase_dir"
    log "Running: ${cmd[*]}"
    script -q /dev/null -c '${cmd[*]}' 2>&1 | tee "$phase_dir/phase_${phase_num}.log" || {
        local ec=$?
        warn "train_run exited with code $ec"
        return 1
    }

    # Verify best was saved
    if [[ -d "$best_dir" ]]; then
        local best_size
        best_size=$(du -sh "$best_dir" 2>/dev/null | cut -f1)
        log "Phase $phase_num done. best/ ($best_size) saved."
    else
        warn "Phase $phase_num: no best/ directory found!"
    fi

    echo "$best_dir"
}

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

START_PHASE="${START_PHASE:-1}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
    --phase)
        START_PHASE="$2"
        shift 2
        ;;
    --dry-run)
        DRY_RUN=1
        shift
        ;;
    *)
        echo "Unknown arg: $1"
        exit 1
        ;;
    esac
done

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

log "Curriculum trainer"
log "  out_dir: $OUT_DIR"
log "  cache:   $DATA_DIR"
log "  source dir: $SRC_DIR"
log "  starting from phase $START_PHASE"

mkdir -p "$OUT_DIR"

prev_best=""
for phase_def in "${PHASES[@]}"; do
    IFS=':' read -r pnum pname prows pdesc <<<"$phase_def"

    if ((pnum < START_PHASE)); then
        log "Skipping phase $pnum ($pname) — before --phase $START_PHASE"
        # Still track prev_best so we can chain
        if [[ -d "$OUT_DIR/phase_${pnum}_${pname}/best" ]]; then
            prev_best="$OUT_DIR/phase_${pnum}_${pname}/best"
        fi
        continue
    fi

    # Wait for 10k training to finish if it's still running
    # (don't steal its CPU cores)
    if pgrep -f "train_run.*train10k" >/dev/null 2>&1; then
        log "Waiting for background train_run (10k) to finish..."
        while pgrep -f "train_run.*train10k" >/dev/null 2>&1; do
            sleep 30
        done
        log "10k training done, continuing curriculum."
    fi

    best=$(run_phase "$pnum" "$pname" "$prows" "$pdesc" "$prev_best")
    if [[ -z "$best" || ! -d "$best" ]]; then
        warn "Phase $pnum failed — stopping curriculum."
        break
    fi
    prev_best="$best"

    log "Phase $pnum complete. Next phase will use: $prev_best"
done

log "=============================================="
log "Curriculum done."
log "Final best: $prev_best"
log "=============================================="
