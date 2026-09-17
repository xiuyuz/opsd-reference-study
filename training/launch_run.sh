#!/bin/bash
# Launch one training run: a persistent vLLM rollout worker plus the trainer, sharing one
# GPU. Every variable is read from the environment; the defaults are the matched recipe
# every reported run used (effective batch 32, flat LR, loss temperature 1.1, AdamW beta2
# 0.999, rollout temperature 1.1, teacher scored thinking-on, 100 steps).
#
#   CONDITION=full_trace GPU=0 bash training/launch_run.sh
#
# Required
#   CONDITION       reference view (answer_only, gist, key_points, summary, clean_solution,
#                   full_trace), or any label when EXTRA_ARGS carries --pi-override-file or
#                   --unprivileged-teacher
#
# Optional (default)
#   MODEL           Qwen/Qwen3-1.7B            backbone (also HuggingFaceTB/SmolLM3-3B)
#   SPLITS_FILE     data/splits/qwen3_1p7b_splits.json for Qwen, data/splits/smollm3_3b_splits.json
#                   for SmolLM3 -- the release's frozen train/dev/test ids
#   ROLLOUTS        direct                     direct | thinking: student rollouts with
#                                              enable_thinking off | on (the teacher always
#                                              scores thinking-on)
#   SEED            0                          replication seed offset (--seed)
#   TRAIN_MAX_NEW_TOKENS                       on-policy generation cap:
#                                              direct 1024; thinking 20764 (Qwen3-1.7B) or
#                                              24536 (SmolLM3-3B) = CONTEXT - longest
#                                              full_trace teacher prompt - 256
#   LOSS_HORIZON                               KL loss prefix: direct unset (== the cap);
#                                              thinking 1024. Set LOSS_HORIZON="" to disable.
#   CONTEXT                                    engine/trainer context: 32768 for Qwen direct
#                                              runs, 40960 otherwise (thinking runs and every
#                                              SmolLM3 run). Exported as MODEL_MAX_LEN_OVERRIDE
#                                              and passed to the worker's --max-model-len.
#   EFFECTIVE_BATCH 32                         examples per optimizer step
#   LR_SCHEDULE     flat                       flat | decay
#   LOSS_TEMP       1.1                        --loss-temperature
#   ADAM_BETA2      0.999                      --adam-beta2
#   ROLLOUT_TEMP    1.1                        rollout worker sampling temperature
#   CHECKPOINT_STEPS 0,1,2,5,10,15,20,25,50,75,100   adapter checkpoints saved (step_<n>/);
#                                              the SmolLM3 runs saved 25,50,75,100 only
#   GPU_MEMORY_UTILIZATION 0.35                worker's share of the GPU (trainer takes the rest)
#   GPU             0                          CUDA device for both processes
#   TAG             <model>_<CONDITION>_<ROLLOUTS>[_seed<SEED>]   run name. The analyses look
#                                              for the names the paper's runs were saved under (listed in
#                                              opsd/artifact_layout.py, e.g.
#                                              qwen3-1.7b_answer_only_direct); set TAG to one
#                                              of those to have an analysis read this run.
#   OUTPUT          $OPSD_ARTIFACTS/<train_runs>/$TAG    trajectory.jsonl + step_<n>/ adapters, where
#                                              <train_runs> is opsd.artifact_layout.train_runs():
#                                              DIRECT_TRAIN_RUNS (train/direct) for Qwen direct runs,
#                                              THINKING_TRAIN_RUNS (train/thinking) for Qwen thinking runs,
#                                              SMOLLM3_TRAIN_RUNS (smollm3/train) for SmolLM3
#   WORKER_DIR      $OPSD_ARTIFACTS/<ROLLOUT_WORKERS>/$TAG   trainer <-> worker watch directory
#   LOG_DIR         $OPSD_ARTIFACTS/<LOGS>               train_<TAG>.log, rollout_worker_<TAG>.log
#   HF_PY           python                     interpreter of the training env (torch/peft/flash-attn)
#   VLLM_PY         python                     interpreter of the vLLM env
#   EXTRA_ARGS      ""                         extra trainer flags passed through verbatim, e.g.
#                     "--pi-override-file controls/trunc_full_trace.json"
#                     "--fork-mask-mode exclude" / "--fork-mask-mode downweight"
#                     "--prompt-style official" / "--teacher-prompt-style official"
#                     "--loss-support-windows 0.125,0.375,0.625,0.875 --loss-support-window-size 256"
#                     "--unprivileged-teacher"
#
# HF_HOME must be set by the caller. OPSD_ARTIFACTS defaults to ./artifacts.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || { echo "FATAL: cannot cd to $REPO_DIR" >&2; exit 1; }

ARTIFACTS="${OPSD_ARTIFACTS:-artifacts}"
CONDITION="${CONDITION:?set CONDITION (reference view or control label)}"

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
ROLLOUTS="${ROLLOUTS:-direct}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"
HF_PY="${HF_PY:-python}"
VLLM_PY="${VLLM_PY:-python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

case "$MODEL" in
  *SmolLM3*) FAMILY=smollm3; DEFAULT_SPLITS="$REPO_DIR/data/splits/smollm3_3b_splits.json" ;;
  *)         FAMILY=qwen;    DEFAULT_SPLITS="$REPO_DIR/data/splits/qwen3_1p7b_splits.json" ;;
esac
SPLITS_FILE="${SPLITS_FILE:-$DEFAULT_SPLITS}"
[ -f "$SPLITS_FILE" ] || { echo "FATAL: SPLITS_FILE $SPLITS_FILE does not exist" >&2; exit 1; }
MODEL_TAG="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"

case "$ROLLOUTS" in
  direct)
    STUDENT_THINKING_FLAG="--no-student-thinking"
    DEFAULT_CAP=1024
    DEFAULT_HORIZON=""
    [ "$FAMILY" = qwen ] && DEFAULT_CONTEXT=32768 || DEFAULT_CONTEXT=40960
    ;;
  thinking)
    STUDENT_THINKING_FLAG="--student-thinking"
    [ "$FAMILY" = qwen ] && DEFAULT_CAP=20764 || DEFAULT_CAP=24536
    DEFAULT_HORIZON=1024
    DEFAULT_CONTEXT=40960
    ;;
  *) echo "FATAL: ROLLOUTS must be 'direct' or 'thinking', got '$ROLLOUTS'" >&2; exit 1 ;;
esac

TRAIN_MAX_NEW_TOKENS="${TRAIN_MAX_NEW_TOKENS:-$DEFAULT_CAP}"
LOSS_HORIZON="${LOSS_HORIZON-$DEFAULT_HORIZON}"
CONTEXT="${CONTEXT:-$DEFAULT_CONTEXT}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-32}"
LR_SCHEDULE="${LR_SCHEDULE:-flat}"
LOSS_TEMP="${LOSS_TEMP:-1.1}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
ROLLOUT_TEMP="${ROLLOUT_TEMP:-1.1}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-0,1,2,5,10,15,20,25,50,75,100}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"

DEFAULT_TAG="${MODEL_TAG}_${CONDITION}_${ROLLOUTS}"
[ "$SEED" != "0" ] && DEFAULT_TAG="${DEFAULT_TAG}_seed${SEED}"
TAG="${TAG:-$DEFAULT_TAG}"
# Artifact locations come from opsd.artifact_layout, so a run lands in the directory the analyses
# read; TAG names the run inside it.
layout() { "$HF_PY" -c "import sys; sys.path.insert(0, sys.argv[1]); from opsd import artifact_layout as L; print(getattr(L, sys.argv[2]))" "$REPO_DIR" "$1"; }
RUNS_DIR=$("$HF_PY" -c "import sys; sys.path.insert(0, sys.argv[1]); from opsd import artifact_layout as L; print(L.train_runs(sys.argv[2], sys.argv[3] == 'thinking'))" "$REPO_DIR" "$MODEL" "$ROLLOUTS")
OUTPUT="${OUTPUT:-$ARTIFACTS/$RUNS_DIR/$TAG}"
WORKER_DIR="${WORKER_DIR:-$ARTIFACTS/$(layout ROLLOUT_WORKERS)/$TAG}"
LOG_DIR="${LOG_DIR:-$ARTIFACTS/$(layout LOGS)}"
LOG="$LOG_DIR/train_${TAG}.log"
WORKER_LOG="$LOG_DIR/rollout_worker_${TAG}.log"

HORIZON_ARGS=()
[ -n "$LOSS_HORIZON" ] && HORIZON_ARGS=(--loss-horizon "$LOSS_HORIZON")

mkdir -p "$OUTPUT" "$LOG_DIR"
rm -rf "$WORKER_DIR"
mkdir -p "$WORKER_DIR"

echo "== [$(date '+%F %T')] $TAG: model=$MODEL condition=$CONDITION rollouts=$ROLLOUTS seed=$SEED gpu=$GPU context=$CONTEXT cap=$TRAIN_MAX_NEW_TOKENS loss_horizon=${LOSS_HORIZON:-cap} effective_batch=$EFFECTIVE_BATCH lr_schedule=$LR_SCHEDULE loss_temp=$LOSS_TEMP adam_beta2=$ADAM_BETA2 rollout_temp=$ROLLOUT_TEMP checkpoint_steps=$CHECKPOINT_STEPS splits_file=$SPLITS_FILE extra='$EXTRA_ARGS' output=$OUTPUT =="

# The worker first: vLLM profiles free memory at startup, so let it claim its share before
# the trainer allocates anything on the same device.
CUDA_VISIBLE_DEVICES=$GPU $VLLM_PY training/rollout_worker.py \
  --model "$MODEL" \
  --watch-dir "$WORKER_DIR" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$CONTEXT" \
  --temperature "$ROLLOUT_TEMP" >"$WORKER_LOG" 2>&1 &
WORKER_PID=$!
while [ ! -f "$WORKER_DIR/ready.json" ]; do
  kill -0 "$WORKER_PID" 2>/dev/null || { echo "rollout worker died, see $WORKER_LOG" >&2; exit 1; }
  sleep 2
done

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=$GPU MODEL_MAX_LEN_OVERRIDE=$CONTEXT $HF_PY training/train_opsd.py \
  --condition "$CONDITION" \
  --model "$MODEL" \
  --splits-file "$SPLITS_FILE" \
  --seed "$SEED" \
  --train-max-new-tokens "$TRAIN_MAX_NEW_TOKENS" \
  "${HORIZON_ARGS[@]}" \
  --effective-batch "$EFFECTIVE_BATCH" \
  $STUDENT_THINKING_FLAG \
  --teacher-thinking \
  --checkpoint-steps "$CHECKPOINT_STEPS" \
  --lr-schedule "$LR_SCHEDULE" \
  --loss-temperature "$LOSS_TEMP" \
  --adam-beta2 "$ADAM_BETA2" \
  --rollout-worker-dir "$WORKER_DIR" \
  --output "$OUTPUT" \
  $EXTRA_ARGS 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}

# Always, including on a trainer crash -- otherwise the worker sits on the GPU forever.
echo '{}' >"$WORKER_DIR/stop.json"
wait "$WORKER_PID" || true
echo "== [$(date '+%F %T')] $TAG: training complete (status $STATUS). log: $LOG =="
exit "$STATUS"
