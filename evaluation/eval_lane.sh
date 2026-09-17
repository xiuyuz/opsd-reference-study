#!/bin/bash
# Evaluation lane: keeps one GPU busy draining the file-backed queue, ONE vLLM engine per
# CELL (evaluation/eval_queue.py --next-cell), until the queue is empty. Several lanes can
# share one queue (atomic-rename claiming), and a lane can be retired at a cell boundary
# by creating the STOP file.
#
#   nohup bash evaluation/eval_lane.sh 1 > artifacts/logs/eval_lane1.out 2>&1 &
#
# Environment (defaults from opsd.artifact_layout): QUEUE <EVAL_QUEUE>, MARK_DIR <EVAL_LANES>
# (markers/, timings.csv, lane<gpu>.pid, STOP_LANE<gpu>), LOG <LOGS>/eval_lane<gpu>.log,
# JOB_TIMEOUT 14400, VLLM_PY python. HF_HOME must be set by the caller.
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1
GPU="${1:?usage: bash evaluation/eval_lane.sh <gpu_index>}"
PY="${VLLM_PY:-python}"
ARTIFACTS="${OPSD_ARTIFACTS:-artifacts}"
layout() { "$PY" -c "import sys; sys.path.insert(0, sys.argv[1]); from opsd import artifact_layout as L; print(getattr(L, sys.argv[2]))" "$REPO_DIR" "$1"; }
QUEUE="${QUEUE:-$ARTIFACTS/$(layout EVAL_QUEUE)}"
MARK_DIR="${MARK_DIR:-$ARTIFACTS/$(layout EVAL_LANES)}"
LOG="${LOG:-$ARTIFACTS/$(layout LOGS)/eval_lane${GPU}.log}"
STOP="${MARK_DIR}/STOP_LANE${GPU}"
export JOB_TIMEOUT="${JOB_TIMEOUT:-14400}"
mkdir -p "$(dirname "$LOG")" "$MARK_DIR"
log() { echo "== [$(date '+%F %T')] [lane${GPU}] $*" | tee -a "$LOG"; }
echo $$ > "${MARK_DIR}/lane${GPU}.pid"
log "eval lane starting (pid $$, queue $QUEUE)"
while true; do
  if [ -f "$STOP" ]; then log "stop file $STOP present -- exiting at cell boundary"; rm -f "${MARK_DIR}/lane${GPU}.pid"; exit 0; fi
  n=$(find "$QUEUE" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
  if [ "$n" -eq 0 ]; then log "queue drained -- exiting"; rm -f "${MARK_DIR}/lane${GPU}.pid"; exit 0; fi
  QUEUE="$QUEUE" MARKERS="${MARK_DIR}/markers" TIMINGS="${MARK_DIR}/timings.csv" \
    "$PY" evaluation/eval_queue.py --gpu "$GPU" --next-cell --queue "$QUEUE" >>"$LOG" 2>&1
  rc=$?
  log "cell pass finished rc=$rc ($(find "$QUEUE" -maxdepth 1 -name '*.json' | wc -l) job(s) still pending)"
  [ "$rc" -ne 0 ] && log "WARNING: non-zero rc from the cell pass -- see $LOG and $QUEUE/failed/"
  sleep 5
done
