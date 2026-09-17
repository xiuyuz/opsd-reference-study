"""Locations inside the artifact bundle that the data and analysis scripts read or write.

Every value is a path relative to the artifacts root (OPSD_ARTIFACTS, default ./artifacts;
see opsd.constants.ARTIFACTS_DIR). path(CONSTANT, "file") builds the absolute path. The values
name directories of the released bundle; the scripts append file names to them.
"""

import os

from opsd.constants import ARTIFACTS_DIR


def path(*parts):
    """Absolute path of an artifact location: os.path.join(ARTIFACTS_DIR, *parts)."""
    return os.path.join(ARTIFACTS_DIR, *parts)


# ---- Qwen3-1.7B six-view study (direct-response training, thinking-enabled evaluation) ----
# Qwen split construction: candidate_pool.json and splits.json
SIX_VIEW_BUILD = "six_view"
# raw thinking-enabled test-split evaluations of the six-view students (test_b*_<run>_step<S>_test.jsonl)
SIX_VIEW_TEST_TIER = "six_view/eval/test"
# dev-split evaluations used for checkpoint selection, including the step-0 cells
SIX_VIEW_DEV_TIER = "six_view/eval/dev"
# thinking-enabled test-split evaluation of the frozen base, the anchor of every in-domain gain
BASE_TEST_TIER = "base/eval/test"
# training logs of the direct-response runs (<run>/trajectory.jsonl)
DIRECT_TRAIN_RUNS = "train/direct"

# ---- thinking-enabled training comparison (Qwen3-1.7B) ----
# training logs of the thinking-enabled runs (<run>/trajectory.jsonl)
THINKING_TRAIN_RUNS = "train/thinking"
# test-split evaluations of the common-checkpoint cells (steps 25 and 50) of both training modes
COMMON_CKPT_TEST_TIER = "common_checkpoints/eval/test"

# ---- loss-window comparison (thinking-enabled Full Trace: early-1k, first-4k, distributed-1k) ----
# the three loss-window runs at the checkpoints they share
LOSS_WINDOW_COMMON_CKPT_TIER = "loss_window/eval/test"

# ---- matched interventions (direct-response training) ----
# the trace-opening and other-problem reference files and their audit trails
REFERENCE_CONTROLS = "reference_controls"

# ---- teacher controls, interventions at common steps, seed cells ----
# the wrong-answer reference file (pi_answer_only_wrong_answer.json) and the per-analysis output directories below
CONTROL_STUDIES = "teacher_controls"
# thinking-enabled test-split evaluations of the teacher-control and intervention students at common steps (fu_b*_*.jsonl)
CONTROL_EVAL_THINKING = "teacher_controls/eval/test_thinking"
# direct-response test-split evaluations of the same students and of the frozen base
CONTROL_EVAL_DIRECT = "teacher_controls/eval/test_direct"
# test-split evaluations of the students trained at seeds 1 and above
SEED_EVAL_TEST_TIER = "seeds/eval/test"

# ---- external benchmarks ----
# step-50 external evaluations (<run>_step50_<bench>.jsonl) and the aime25/hmmt25 id maps
EXTERNAL_STEP50 = "external/eval"
# frozen-base external results (base_<bench>.json)
EXTERNAL_BASE_ANCHORS = "external/base"
# the aime24 id map
EXTERNAL_ID_MAP_AIME24 = "external"
# aime24.json / aime25.json / hmmt25.json (the default of OPSD_BENCHMARKS)
BENCHMARKS = "benchmarks"

# ---- SmolLM3-3B ----
# the SmolLM3 split file (splits_smollm3.json) and its build report
SMOLLM3 = "smollm3"
# pool_ids.json and teacher_analysis.json of the SmolLM3 split construction
SMOLLM3_SPLIT_POOL = "smollm3/split_pool"
# the SmolLM3 base's four direct-response samples over the candidate pool
SMOLLM3_SPLIT_POOL_EVAL = "smollm3/split_pool/eval"
# SmolLM3 teacher generations for the pass@4 == 0 problems
SMOLLM3_SPLIT_POOL_TEACHER_EVAL = "smollm3/split_pool/teacher_eval"
# the frozen SmolLM3 base on the test split, in both evaluation modes
SMOLLM3_BASE_EVAL = "smollm3/base/eval"
# Answer Only / Full Trace x direct-response / thinking-enabled training, seed 0, steps 25/50/100
SMOLLM3_2X2_EVAL = "smollm3/eval/views_by_mode"
# Full Trace seed 1, both training modes
SMOLLM3_SEED1_EVAL = "smollm3/eval/seed1"
# reference-free, Answer Only and Full Trace students, seeds 0-2, both evaluation modes, steps 50 and 100
SMOLLM3_REFERENCE_FREE_EVAL = "smollm3/eval/reference_free"

# ---- profiling ----
# the 512 profiling calibration ids (calibration_ids.json), excluded from the training splits
PROFILE_CALIBRATION = "profiles/calibration"

# ---- dataset ----
# an earlier study's split on this corpus; when present, its ids stay out of the candidate pool
PRIOR_STUDY_SPLIT = "prior_study_split.json"
# decontamination results
DECONTAM = "decontam"

# ---- training runs (training/launch_run.sh, train_opsd.py --output): <run>/trajectory.jsonl + <run>/step_<n>/ adapters ----
# SmolLM3-3B training runs, both training modes (the Qwen runs use DIRECT_TRAIN_RUNS / THINKING_TRAIN_RUNS)
SMOLLM3_TRAIN_RUNS = "train/smollm3"
# trainer <-> vLLM rollout-worker watch directories (<run>/request.json, response.json, ready.json, adapter_step<n>/)
ROLLOUT_WORKERS = "rollout_worker"
# launcher, evaluation-lane and profiling-chain logs
LOGS = "logs"

# ---- evaluation queue (evaluation/build_eval_queue.py, eval_queue.py, eval_lane.sh) ----
# 16-problem test-split shard files (<split>_b<NNN>.json) that fix the per-request seeds
EVAL_SHARDS = "eval/shards"
# pending job files, with claimed/lane<gpu>/, done/ and failed/ subdirectories
EVAL_QUEUE = "eval/queue"
# lane state: markers/<job_id>.DONE|ERROR, timings.csv, lane<gpu>.pid, STOP_LANE<gpu>
EVAL_LANES = "eval"
# multi-adapter evaluations run outside the queue (evaluation/multi_adapter_eval.py default --output-dir)
MULTI_ADAPTER_EVAL = "eval/multi_adapter"

# ---- profiling (profiling/) ----
# teacher generations and 7-context prefix scores for the three additional views (preflight_ext.py)
PROFILE_EXTRA_VIEWS = "profiles/calibration_extra_views"
# direct-response calibration generations of the frozen base and the scoring shards ids_shard{0,1}.json
PROFILE_DIRECT_STATES = "profiles/calibration_direct"
# thinking-enabled profile: all_rollouts_prefix_scores*.parquet, reduced/, analysis/
PROFILE_THINKING = "profiles/thinking"
# direct-response profile with the local prompt templates (parent of PROFILE_DIRECT_REDUCED and PROFILE_DIRECT_ALIGNMENT)
PROFILE_DIRECT = "profiles/direct"
# the profile re-measured on trained students' states: <view>_step<S>_generations.jsonl, <view>_step<S>/, DRIFT_PROFILE_RESULT.md
DRIFT_PROFILE = "profiles/drift"


# ---- Run names ----
# The <run> field of the evaluation file names (<prefix>_b<NNN>_<run>_step<n>_test.jsonl) and of
# the training-run directories. Students trained at seed 1 and above append "_seed<n>" to the run.

# the six privileged references, ordered by reference length
VIEWS = ["answer_only", "gist", "key_points", "clean_solution", "summary", "full_trace"]
# The SmolLM3 evaluation file names additionally carry a job tag ahead of the run
# The file-name prefix inside each directory is chosen by the caller.


def train_runs(model, student_thinking):
    """Training-run directory for a backbone and rollout mode: DIRECT_TRAIN_RUNS or
    THINKING_TRAIN_RUNS for Qwen3-1.7B, SMOLLM3_TRAIN_RUNS for SmolLM3-3B."""
    if "SmolLM3" in model:
        return SMOLLM3_TRAIN_RUNS
    return THINKING_TRAIN_RUNS if student_thinking else DIRECT_TRAIN_RUNS


def resolve(value):
    """Accept either a filesystem path or the name of a directory constant of this module
    (e.g. "CONTROL_EVAL_DIRECT"); a constant name resolves to path(constant)."""
    const = globals().get(value) if isinstance(value, str) and value.isupper() else None
    return path(const) if isinstance(const, str) else value
