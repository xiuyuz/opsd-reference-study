#!/bin/bash
# Drift profile: the direct-response privilege profile re-measured on the states the trained
# students actually visit.
#
# For each view in VIEWS, at the common checkpoint STEP:
#   (a) generate that student's own states: the SAME calibration problems, 4 no-reference
#       direct-response (thinking-off) samples each, LOCAL prompt style, the SAME per-request
#       generation seeds (constants.sample_seed(0, sample_index, item_index) over
#       <artifacts>/profiles/calibration/calibration_ids.json) and the SAME 16,384-token generation cap +
#       rescue as the base profile's generations. The ONLY difference is that the step-STEP
#       LoRA adapter is served live on top of the frozen backbone (preflight.py --adapter).
#   (b) score them with the direct-response profile chain, UNCHANGED: score_views.py,
#       2 shards co-located on this one GPU (SHARD_IDS/ids_shard{0,1}.json),
#       --no-student-thinking --prompt-style local --scoring-horizon 1024. The scorer never
#       sees the adapter: teacher = frozen base + reference (7 contexts), student reference =
#       frozen base, no reference -- exactly as the base profile. Only the states change.
#   (c) reduce (reduce_profile.py) and run the three profile analyses behind the paper's
#       profile figure (correctness alignment, fork correction, censored tail), then compare
#       with drift_profile_compare.py.
#
# NOTE on "1,024 tokens": the base generations were produced at MAIN_MAX_NEW_TOKENS = 16,384
# and scored with --scoring-horizon 1024. This script reproduces exactly that split --
# 16,384 for generation, 1,024 for the scored prefix -- because generating at a 1,024 cap
# would truncate ~40% of completions and change the correctness labels the profile
# conditions on.
#
# Launch:
#   nohup bash profiling/run_drift_profile.sh > artifacts/logs/drift_profile_driver.out 2>&1 &
# Markers: <OUT_ROOT>/D_DONE.marker / D_ERROR.marker
# Environment (defaults):
#   GPU 1, STEP 50, VIEWS "answer_only gist key_points clean_solution summary full_trace",
#   ADAPTER_TEMPLATE "<artifacts>/<DIRECT_TRAIN_RUNS>/qwen3-1.7b_{view}_direct/step_{step}" (the
#     training/launch_run.sh default tag; {view}/{step} are substituted),
#   MODEL Qwen/Qwen3-1.7B, GEN_GPU_MEM 0.85, SCORING_HORIZON 1024,
#   OUT_ROOT <artifacts>/<DRIFT_PROFILE>, SHARD_IDS <artifacts>/<PROFILE_DIRECT_STATES> (ids_shard{0,1}.json),
#   BASE_DIR <artifacts>/<PROFILE_DIRECT> (the frozen base profile),
#   EXPECTED_RECORDS 2048 (= calibration problems x 4 samples), SMOKE_LIMIT "" (=> --limit N on
#   the generation step only), VLLM_PY / HF_PY python. HF_HOME must be set by the caller.
#   <NAME> denotes a constant of opsd.artifact_layout.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || { echo "FATAL: cannot cd to $REPO_DIR" >&2; exit 1; }

ARTIFACTS="${OPSD_ARTIFACTS:-artifacts}"
GPU="${GPU:-1}"
VLLM_PY="${VLLM_PY:-python}"
HF_PY="${HF_PY:-python}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
STEP="${STEP:-50}"
GEN_GPU_MEM="${GEN_GPU_MEM:-0.85}"
SCORING_HORIZON="${SCORING_HORIZON:-1024}"
layout() { "$HF_PY" -c "import sys; sys.path.insert(0, sys.argv[1]); from opsd import artifact_layout as L; print(getattr(L, sys.argv[2]))" "$REPO_DIR" "$1"; }
OUT_ROOT="${OUT_ROOT:-$ARTIFACTS/$(layout DRIFT_PROFILE)}"
SHARD_IDS="${SHARD_IDS:-$ARTIFACTS/$(layout PROFILE_DIRECT_STATES)}"
BASE_DIR="${BASE_DIR:-$ARTIFACTS/$(layout PROFILE_DIRECT)}"
ADAPTER_TEMPLATE="${ADAPTER_TEMPLATE:-$ARTIFACTS/$(layout DIRECT_TRAIN_RUNS)/qwen3-1.7b_{view}_direct/step_{step}}"
VIEWS="${VIEWS:-answer_only gist key_points clean_solution summary full_trace}"
EXPECTED_RECORDS="${EXPECTED_RECORDS:-2048}"
SMOKE_LIMIT="${SMOKE_LIMIT:-}"
PROFILING_DIR="$REPO_DIR/profiling"
LOG_DIR="$ARTIFACTS/$(layout LOGS)"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
rm -f "$OUT_ROOT/D_DONE.marker" "$OUT_ROOT/D_ERROR.marker"

log()  { echo "== [$(date '+%F %T')] [drift] $*"; }
fail() { echo "[$(date -Is)] DRIFT PROFILE FAILED: $*" | tee -a "$OUT_ROOT/D_ERROR.marker"; exit 1; }
adapter_for() { echo "$ADAPTER_TEMPLATE" | sed -e "s/{view}/$1/g" -e "s/{step}/$STEP/g"; }

gpu_check() {
  log "nvidia-smi: $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ' | ')"
}

for view in $VIEWS; do
  ADAPTER="$(adapter_for "$view")"
  [ -f "$ADAPTER/adapter_model.safetensors" ] || fail "missing adapter $ADAPTER"
done
# The scoring pass runs as two co-resident shards, so the calibration ids are split in half.
# Build the two halves once from the calibration set if they are not there yet.
if [ ! -f "$SHARD_IDS/ids_shard0.json" ] || [ ! -f "$SHARD_IDS/ids_shard1.json" ]; then
  CALIB="$ARTIFACTS/$(layout PROFILE_CALIBRATION)/calibration_ids.json"
  [ -f "$CALIB" ] || fail "missing $CALIB (build it with: python -m opsd.data --stage calibration)"
  mkdir -p "$SHARD_IDS"
  "$HF_PY" -c 'import json, sys
ids = sorted(json.load(open(sys.argv[1])))          # the two shards alternate over the sorted ids
for i in (0, 1):
    part = ids[i::2]
    json.dump(part, open(f"{sys.argv[2]}/ids_shard{i}.json", "w"))
    print(f"wrote {sys.argv[2]}/ids_shard{i}.json ({len(part)} ids)")' "$CALIB" "$SHARD_IDS" \
    || fail "could not split $CALIB into two shards"
fi
log "all step-${STEP} adapters present"

for view in $VIEWS; do
  TAG="${view}_step${STEP}"
  ADAPTER="$(adapter_for "$view")"
  PREFIX="$OUT_ROOT/${TAG}"                 # preflight.py --output is a path PREFIX
  GEN="${PREFIX}_generations.jsonl"
  PROF="$OUT_ROOT/${TAG}"                   # scored/reduced/analysis dir (same basename, a dir)
  mkdir -p "$PROF"

  # ---------------------------------------------------------------- (a) generation
  if [ -s "$GEN" ] && [ -f "$PROF/.gen_done" ]; then
    log "$TAG: generations already complete, skipping stage (a)"
  else
    log "$TAG: stage (a) generation with adapter $ADAPTER"
    LIMIT_ARGS=()
    [ -n "$SMOKE_LIMIT" ] && LIMIT_ARGS=(--limit "$SMOKE_LIMIT")
    CUDA_VISIBLE_DEVICES=$GPU $VLLM_PY "$PROFILING_DIR/preflight.py" \
      --model "$MODEL" \
      --output "$PREFIX" \
      --adapter "$ADAPTER" \
      --no-student-thinking \
      --skip-pi-generation \
      --prompt-style local \
      --gpu-memory-utilization "$GEN_GPU_MEM" \
      "${LIMIT_ARGS[@]}" \
      >>"$LOG_DIR/drift_profile_${TAG}_gen.log" 2>&1 \
      || fail "$TAG generation failed, see $LOG_DIR/drift_profile_${TAG}_gen.log"
    n=$(wc -l < "$GEN")
    want=$EXPECTED_RECORDS
    [ -n "$SMOKE_LIMIT" ] && want=$((SMOKE_LIMIT * 4))
    [ "$n" -eq "$want" ] || fail "$TAG generation wrote $n records, expected $want"
    touch "$PROF/.gen_done"
    log "$TAG: generation done ($n records)"
    gpu_check
  fi

  [ -n "$SMOKE_LIMIT" ] && { log "SMOKE_LIMIT set -- stopping after generation for $TAG"; continue; }

  # ---------------------------------------------------------------- (b) prefix scoring
  if [ -f "$PROF/.score_done" ]; then
    log "$TAG: scoring already complete, skipping stage (b)"
  else
    log "$TAG: stage (b) prefix scoring, 2 shards co-located on GPU $GPU"
    pids=()
    for sh in 0 1; do
      CUDA_VISIBLE_DEVICES=$GPU $HF_PY "$PROFILING_DIR/score_views.py" \
        --generations "$GEN" \
        --calibration-ids "$SHARD_IDS/ids_shard${sh}.json" \
        --output "$PROF/all_rollouts_prefix_scores.shard${sh}.parquet" \
        --no-student-thinking --prompt-style local --scoring-horizon "$SCORING_HORIZON" \
        >"$LOG_DIR/drift_profile_${TAG}_shard${sh}.log" 2>&1 &
      pids+=($!)
    done
    ok=1
    for p in "${pids[@]}"; do wait "$p" || ok=0; done
    [ "$ok" -eq 1 ] || fail "$TAG scoring failed, see $LOG_DIR/drift_profile_${TAG}_shard*.log"
    for sh in 0 1; do
      [ -s "$PROF/all_rollouts_prefix_scores.shard${sh}.parquet" ] \
        || fail "$TAG shard${sh} parquet missing/empty"
    done
    touch "$PROF/.score_done"
    log "$TAG: scoring done"
    gpu_check
  fi

  SHARDS=("$PROF/all_rollouts_prefix_scores.shard0.parquet" "$PROF/all_rollouts_prefix_scores.shard1.parquet")

  # ---------------------------------------------------------------- (c) reduce + analyses
  if [ -f "$PROF/.reduce_done" ]; then
    log "$TAG: reduction already complete"
  else
    log "$TAG: stage (c) reduction"
    $HF_PY "$PROFILING_DIR/reduce_profile.py" --shards "${SHARDS[@]}" --out-dir "$PROF/reduced" \
      >"$LOG_DIR/drift_profile_${TAG}_reduce.log" 2>&1 \
      || fail "$TAG reduce failed, see $LOG_DIR/drift_profile_${TAG}_reduce.log"
    [ -s "$PROF/reduced/per_group.parquet" ] || fail "$TAG per_group.parquet missing"
    touch "$PROF/.reduce_done"
  fi

  A="$PROF/analysis"
  run_analysis() {  # name, then the command
    local name="$1"; shift
    if [ -f "$PROF/.analysis_${name}_done" ]; then log "$TAG: analysis $name already done"; return 0; fi
    log "$TAG: analysis $name"
    "$@" >"$LOG_DIR/drift_profile_${TAG}_${name}.log" 2>&1 \
      || fail "$TAG analysis $name failed, see $LOG_DIR/drift_profile_${TAG}_${name}.log"
    touch "$PROF/.analysis_${name}_done"
  }

  run_analysis correctness_alignment $HF_PY "$PROFILING_DIR/correctness_alignment.py" \
      --reduced-dir "$PROF/reduced" --shards "${SHARDS[@]}" --out-dir "$A/correctness-alignment"
  run_analysis fork_correction $HF_PY "$PROFILING_DIR/fork_correction.py" \
      --reduced-dir "$PROF/reduced" --out-dir "$A/fork-correction"
  run_analysis censored_tail $HF_PY "$PROFILING_DIR/censored_tail.py" \
      --reduced-dir "$PROF/reduced" --generations "$GEN" --out-dir "$A/censored-tail"

  log "$TAG: COMPLETE"
  gpu_check
done

[ -n "$SMOKE_LIMIT" ] && { log "smoke run finished (no marker written)"; exit 0; }

# ---------------------------------------------------------------- comparison + result doc
log "running profiling/drift_profile_compare.py"
$HF_PY "$PROFILING_DIR/drift_profile_compare.py" --step "$STEP" --views $VIEWS \
  --base-dir "$BASE_DIR" --drift-root "$OUT_ROOT" \
  >"$LOG_DIR/drift_profile_compare.log" 2>&1 \
  || fail "drift_profile_compare.py failed, see $LOG_DIR/drift_profile_compare.log"

{
  echo "drift profile complete $(date -Is)"
  echo "views: $VIEWS at step $STEP"
  echo "result doc: $OUT_ROOT/DRIFT_PROFILE_RESULT.md"
} > "$OUT_ROOT/D_DONE.marker"
log "wrote $OUT_ROOT/D_DONE.marker"
