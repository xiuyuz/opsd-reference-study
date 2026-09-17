#!/bin/bash
# External evaluation of trained checkpoints on AIME 2024, AIME 2025 and HMMT February 2025
# under the released 12-sample protocol: official prompt rendering (the official evaluator's
# exact user-message template), temperature 1.0, top_p 0.95, top_k disabled, 38,912-token
# main-pass budget, max_model_len 40960, enable_thinking=True, 30 problems x 12 samples per
# benchmark -- matched field-for-field to base-model anchors produced by
# evaluation/official_eval.py. Unlike that literal-official anchor, a rescue pass IS applied
# on top for any generation that hits the cap (multi_adapter_eval.py's official-protocol
# convention). Every run in RUNS_CONFIG is served from ONE vLLM engine as a co-resident LoRA
# adapter.
#
# MODEL_MAX_LEN_OVERRIDE is exported to MAX_MODEL_LEN so the per-request budget
# (opsd.generation.prepare_prompts: MODEL_MAX_LENGTH - prompt - 256) is computed against the
# 40960 context the engine actually has; without it every request would be silently capped
# at ~32.2K tokens instead of the requested 38912.
#
# Usage:
#   RUNS_CONFIG=runs.json bash evaluation/run_external_eval.sh 0
#   RUNS_CONFIG=runs.json GPU=1 bash evaluation/run_external_eval.sh
#   RUNS_CONFIG=runs.json DRY_RUN=1 bash evaluation/run_external_eval.sh   # CPU-only plan check
#
# RUNS_CONFIG is a JSON list of {"name", "adapter_root", "checkpoint_steps": [50]} entries
# (multi_adapter_eval.py --runs-config). Optional: SPLITS (aime24,aime25,hmmt25),
# SAMPLES_BENCH (12), MAX_NEW_TOKENS (38912), MAX_MODEL_LEN (40960), GPU_MEM_UTIL (0.90),
# OUTPUT_DIR (<artifacts>/<EXTERNAL_STEP50>, where the analyses read <run>_step50_<bench>.jsonl),
# VLLM_PY (python). HF_HOME must be set by the caller.
# Outputs: <OUTPUT_DIR>/<run>_step<n>_<benchmark>.jsonl, plus a DONE marker.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || { echo "FATAL: cannot cd to $REPO_DIR" >&2; exit 1; }

GPU="${1:-${GPU:-}}"
if [ -z "${DRY_RUN:-}" ] && [ -z "$GPU" ]; then
  echo "FATAL: GPU is required unless DRY_RUN=1 -- pass as \$1 or the GPU env var" >&2
  exit 1
fi

ARTIFACTS="${OPSD_ARTIFACTS:-artifacts}"
ENV_PY="${VLLM_PY:-python}"
DRIVER="$REPO_DIR/evaluation/multi_adapter_eval.py"
RUNS_CONFIG="${RUNS_CONFIG:?set RUNS_CONFIG to a JSON list of name/adapter_root/checkpoint_steps entries}"
SPLITS="${SPLITS:-aime24,aime25,hmmt25}"
SAMPLES_BENCH="${SAMPLES_BENCH:-12}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-38912}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
layout() { "$ENV_PY" -c "import sys; sys.path.insert(0, sys.argv[1]); from opsd import artifact_layout as L; print(getattr(L, sys.argv[2]))" "$REPO_DIR" "$1"; }
OUTPUT_DIR="${OUTPUT_DIR:-$ARTIFACTS/$(layout EXTERNAL_STEP50)}"
DONE_MARKER="$OUTPUT_DIR/DONE"
LOG_DIR="$ARTIFACTS/$(layout LOGS)"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
export MODEL_MAX_LEN_OVERRIDE="$MAX_MODEL_LEN"

log()   { echo "== [$(date '+%Y-%m-%d %H:%M:%S')] [external_eval gpu=${GPU:-dry-run}] $* =="; }
abort() { log "FAILED -- $*"; exit 1; }

[ -f "$RUNS_CONFIG" ] || abort "missing required file: $RUNS_CONFIG"
# Refuse if any adapter checkpoint named in RUNS_CONFIG is missing.
MISSING=$("$ENV_PY" - "$RUNS_CONFIG" <<'EOF'
import json, os, sys
missing = []
for e in json.load(open(sys.argv[1])):
    for step in e.get("checkpoint_steps", []):
        d = os.path.join(e["adapter_root"], f"step_{step}")
        if not os.path.isdir(d):
            missing.append(d)
print("\n".join(missing))
EOF
)
[ -z "$MISSING" ] || abort "missing adapter checkpoint(s): $MISSING"
log "STAGE: ADAPTER_CHECK passed ($RUNS_CONFIG)"

CMD=("$ENV_PY" "$DRIVER"
     --runs-config "$RUNS_CONFIG"
     --splits "$SPLITS"
     --prompt-style official --protocol official
     --samples-bench "$SAMPLES_BENCH"
     --max-new-tokens "$MAX_NEW_TOKENS" --max-model-len "$MAX_MODEL_LEN"
     --gpu-memory-utilization "$GPU_MEM_UTIL"
     --output-dir "$OUTPUT_DIR")

if [ -n "${DRY_RUN:-}" ]; then
  log "STAGE: DRY_RUN (no CUDA touched)"
  "${CMD[@]}" --dry-run
  DRY_RC=$?
  if [ "$DRY_RC" -ne 0 ]; then
    abort "dry-run exited nonzero ($DRY_RC) -- plan is invalid, see output above"
  fi
  log "STAGE: DRY_RUN complete"
  exit 0
fi

CMD+=(--gpu "$GPU")
LOG_FILE="$LOG_DIR/external_eval_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
log "STAGE: LAUNCH -- ${CMD[*]}"
log "  logging to $LOG_FILE"
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
RC=${PIPESTATUS[0]}
log "STAGE: EVAL finished, rc=$RC"

if [ "$RC" -ne 0 ]; then
  abort "eval driver exited nonzero ($RC) -- not writing $DONE_MARKER, see $LOG_FILE"
fi

N_JSONL=$(find "$OUTPUT_DIR" -maxdepth 1 -name '*_step*_*.jsonl' 2>/dev/null | wc -l)
{
  echo "completed_at=$(date -Iseconds)"
  echo "gpu=$GPU runs_config=$RUNS_CONFIG splits=$SPLITS samples_bench=$SAMPLES_BENCH"
  echo "jsonl_files=$N_JSONL"
  echo "log=$LOG_FILE"
} >> "$DONE_MARKER"
log "STAGE: DONE -- wrote/appended $DONE_MARKER ($N_JSONL jsonl file(s) present)"
