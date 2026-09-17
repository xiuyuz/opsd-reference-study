# Artifacts, paths, and environment variables

Every script under `training/`, `evaluation/`, `profiling/`, `data/` and `analysis/` imports
the `opsd` package from the repository root and reads or writes under one artifacts root.
This page covers that root: what has to exist before you start, which environment runs which
script, every environment variable the code reads, the directory layout, and the record
formats.

The evaluation records are not distributed. They run to hundreds of gigabytes, and nothing
here downloads them. The directories listed below are where your own runs land once you
produce them. `analysis/estimator.py` reads those records, so pointing `OPSD_ARTIFACTS` at
your own root is what makes it read your own cells.

## What has to exist before you run anything

| input | why | how you get it |
|---|---|---|
| two Python environments | the trainer and the sampler pin different versions of PyTorch | `requirements/inference-env.txt`, `requirements/training-env.txt` |
| a Hugging Face cache | `Qwen/Qwen3-1.7B` and `HuggingFaceTB/SmolLM3-3B` are loaded from it | set `HF_HOME` yourself; no script sets it for you |
| the AMPLE-Math rows | the training problems and their six reference views | the dataset named by `AMPLE_MATH_HF_ID`, loaded with `datasets`, or a local jsonl named by `AMPLE_MATH_JSONL` |
| an artifacts root | everything the code writes | `OPSD_ARTIFACTS`, default `artifacts`. An empty directory is fine. Each script creates what it writes. |
| the frozen splits | the train/dev/test ids every reported run used | `data/splits/qwen3_1p7b_splits.json` and `data/splits/smollm3_3b_splits.json` ship with the repository; the trainer and the evaluators pick one by backbone |
| the benchmark files | external evaluation and the decontamination check only | `python -m opsd.benchmarks --materialize` |
| the calibration ids | the reference profiles only | `python -m opsd.data --stage calibration` |

Training and in-domain evaluation need nothing beyond the first five rows.

## Running the scripts

Nothing has to be installed. Each script inserts the repository root into `sys.path` at
startup, so `python training/train_opsd.py --help` works from any working directory. The two
package modules run as `python -m opsd.data ...` and `python -m opsd.benchmarks ...`.

`OPSD_ARTIFACTS` is read once, at import time, and its default is relative. The four shell
launchers `cd` to the repository root first, so a relative root resolves there. A bare
`python` invocation resolves it against the current working directory. Run from the
repository root, or set `OPSD_ARTIFACTS` to an absolute path.

## Which environment runs which script

| environment | what it pins | scripts |
|---|---|---|
| inference (vLLM) | vllm 0.12, torch 2.9, transformers 4.57, datasets, math_verify, pandas, pyarrow, scipy | `training/rollout_worker.py`; `evaluation/evaluate.py`, `multi_adapter_eval.py`, `eval_queue.py`, `smollm3_eval.py`; `profiling/preflight.py`; `profiling/preflight_ext.py --stage generate` |
| training | torch 2.6, transformers 4.54, peft 0.15, flash-attn 2.8, math_verify, pandas, pyarrow | `training/train_opsd.py`; `profiling/prefix_score.py`, `score_views.py`; `profiling/preflight_ext.py --stage score` |
| either (CPU only) | numpy, pandas, pyarrow, scipy, matplotlib | `analysis/estimator.py`; everything in `data/`; `profiling/reduce_profile.py`, `correctness_alignment.py`, `fork_correction.py`, `censored_tail.py`, `drift_profile_compare.py`; `evaluation/build_eval_queue.py`, `build_benchmark_id_maps.py`; `evaluation/eval_queue.py --verify-tasks`; `python -m opsd.data`, `python -m opsd.benchmarks` |

The split is by import. Anything that imports `vllm` belongs to the first environment.
`train_opsd.py` and the teacher-forcing scorers under `profiling/` import `torch`, `peft` or
flash-attention and belong to the second.

CPU only is not the same as dependency free. The three scripts in `data/controls/` and
`data/build_splits.py --stage pool` load a tokenizer, so they need `transformers` and
`HF_HOME`. `data/controls/wrong_answer_reference.py` also grades answers through
`opsd.math_grader`, which needs `math_verify`. `python -m opsd.data` and
`python -m opsd.benchmarks --materialize` load `datasets`.
`profiling/correctness_alignment.py`, `fork_correction.py` and `censored_tail.py` write PNG
diagnostics and import `matplotlib`.

`evaluation/official_eval.py` is the exception to the two-environment rule. It imports the
official OPSD evaluator unmodified from a checkout of that repository and wants an
environment matching that repository's own `environment.yml` (vllm 0.11, transformers 4.57).

The shell launchers take the two interpreters from `HF_PY` (training) and `VLLM_PY`
(inference). Both default to `python`.

## Environment variables

### Read by the Python code

| variable | default | read by | meaning |
|---|---|---|---|
| `OPSD_ARTIFACTS` | `artifacts` | `opsd.constants`, then everything through `opsd.artifact_layout.path()` | the artifacts root |
| `AMPLE_MATH_JSONL` | unset | `opsd.constants`, `opsd.data`, `data/decontam_check.py` | a local AMPLE-Math jsonl. When unset, `opsd.data` loads the Hugging Face dataset instead. `data/decontam_check.py` needs it: that script reads no dataset. |
| `AMPLE_MATH_HF_ID` | `xiuyuz/ample-math` | `opsd.constants`, used by `opsd.data` | the dataset loaded when `AMPLE_MATH_JSONL` is unset |
| `AMPLE_MATH_HF_SPLIT` | `train` | same | which split of it to read |
| `AMPLE_MATH_DIFFICULTY` | unset | same | a JSON file `{"item_difficulties": {problem_id: float}}`. When unset, the rows' own `difficulty` field is used. |
| `OPSD_BENCHMARKS` | `<artifacts>/benchmarks` | `opsd.benchmarks`, `data/decontam_check.py` | directory holding `aime24.json`, `aime25.json`, `hmmt25.json` |
| `MODEL_MAX_LEN_OVERRIDE` | `32768` | `opsd.constants`, re-read by `training/train_opsd.py` | the engine and trainer context. Every per-request completion budget is `MODEL_MAX_LENGTH - prompt - 256`, so a run served at a larger context has to raise this or its requests are silently clipped to 32768. |
| `QUEUE` | `<artifacts>/eval/queue` | `evaluation/eval_queue.py` | pending-job directory |
| `MARKERS` | `<artifacts>/eval/markers` | `evaluation/eval_queue.py` | per-job `.DONE` / `.ERROR` markers |
| `TIMINGS` | `<artifacts>/eval/timings.csv` | `evaluation/eval_queue.py` | one CSV line per finished job |
| `JOB_TIMEOUT` | `14400` | `evaluation/eval_queue.py` | seconds per batch. A batch that trips it aborts the whole cell process and releases the remaining claims. |
| `OPSD_OFFICIAL_EVAL_DIR` | unset | `evaluation/official_eval.py` | path to the official OPSD repository's `eval/` directory. Required unless `--official-eval-dir` is passed. |
| `HF_HOME` | the `huggingface_hub` default | `transformers`, `datasets`, `vllm` | the model and dataset cache. No script in this repository sets it. |

`training/train_opsd.py` also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when
it is not already set, and `eval_queue.py`, `multi_adapter_eval.py`, `official_eval.py` and
`smollm3_eval.py` each set `CUDA_VISIBLE_DEVICES` from their own `--gpu` flag before
importing vLLM.

The code that built AMPLE-Math is released with the dataset itself, not here.

### Read by the shell launchers

`training/launch_run.sh` takes everything from the environment. Its header documents each
one; the defaults are the recipe every reported run used.

| variable | default | meaning |
|---|---|---|
| `CONDITION` | required | the reference view, or any label when `EXTRA_ARGS` carries `--pi-override-file` or `--unprivileged-teacher` |
| `MODEL` | `Qwen/Qwen3-1.7B` | backbone. `HuggingFaceTB/SmolLM3-3B` is the other one. |
| `ROLLOUTS` | `direct` | `direct` or `thinking`: student rollouts with `enable_thinking` off or on. The teacher always scores with it on. |
| `SEED` | `0` | replication seed offset |
| `GPU` | `0` | CUDA device for both processes |
| `SPLITS_FILE` | the release split for `MODEL` | train/dev/test ids |
| `TRAIN_MAX_NEW_TOKENS` | see the table under [Training runs](#training-runs) | on-policy generation cap |
| `LOSS_HORIZON` | see the same table | the completion prefix the KL loss covers. Set it to the empty string to disable. |
| `CONTEXT` | see the same table | engine and trainer context. Exported as `MODEL_MAX_LEN_OVERRIDE` and passed to the worker's `--max-model-len`. |
| `EFFECTIVE_BATCH` | `32` | examples per optimizer step |
| `LR_SCHEDULE` | `flat` | `flat` or `decay` |
| `LOSS_TEMP` | `1.1` | `--loss-temperature` |
| `ADAM_BETA2` | `0.999` | `--adam-beta2` |
| `ROLLOUT_TEMP` | `1.1` | rollout sampling temperature |
| `CHECKPOINT_STEPS` | `0,1,2,5,10,15,20,25,50,75,100` | which `step_<n>/` adapters are saved |
| `GPU_MEMORY_UTILIZATION` | `0.35` | the worker's share of the GPU; the trainer takes the rest |
| `TAG` | `<model>_<CONDITION>_<ROLLOUTS>[_seed<SEED>]` | the run name |
| `OUTPUT` | `<artifacts>/<train runs>/<TAG>` | `trajectory.jsonl` and the adapters |
| `WORKER_DIR` | `<artifacts>/rollout_worker/<TAG>` | the trainer-to-worker watch directory |
| `LOG_DIR` | `<artifacts>/logs` | `train_<TAG>.log`, `rollout_worker_<TAG>.log` |
| `HF_PY`, `VLLM_PY` | `python` | the two interpreters |
| `EXTRA_ARGS` | empty | trainer flags passed through verbatim |

`evaluation/eval_lane.sh <gpu>` reads `VLLM_PY`, `OPSD_ARTIFACTS`, `QUEUE`
(`<artifacts>/eval/queue`), `MARK_DIR` (`<artifacts>/eval`), `LOG`
(`<artifacts>/logs/eval_lane<gpu>.log`) and `JOB_TIMEOUT` (`14400`), which it exports to
`eval_queue.py`. The GPU index is the one positional argument.

`evaluation/run_external_eval.sh [gpu]` reads `RUNS_CONFIG` (required), `GPU` (or the
positional argument), `DRY_RUN`, `SPLITS` (`aime24,aime25,hmmt25`), `SAMPLES_BENCH` (`12`),
`MAX_NEW_TOKENS` (`38912`), `MAX_MODEL_LEN` (`40960`), `GPU_MEM_UTIL` (`0.90`), `OUTPUT_DIR`
(`<artifacts>/external_step50`), `VLLM_PY` and `OPSD_ARTIFACTS`. It exports
`MODEL_MAX_LEN_OVERRIDE=$MAX_MODEL_LEN` so the per-request budget is computed against the
context the engine actually has.

`profiling/run_drift_profile.sh` reads `GPU` (`1`), `STEP` (`50`), `VIEWS` (the six),
`ADAPTER_TEMPLATE` (`<artifacts>/train/direct/qwen3-1.7b_{view}_direct/step_{step}`), `MODEL`
(`Qwen/Qwen3-1.7B`), `GEN_GPU_MEM` (`0.85`), `SCORING_HORIZON` (`1024`), `OUT_ROOT`
(`<artifacts>/drift_profile`), `SHARD_IDS` (`<artifacts>/profiles/calibration_direct`), `BASE_DIR`
(`<artifacts>/profiles/direct`), `EXPECTED_RECORDS` (`2048`), `SMOKE_LIMIT`
(empty), `VLLM_PY`, `HF_PY` and `OPSD_ARTIFACTS`.

All four launchers require `HF_HOME` to be set by the caller.

## AMPLE-Math rows

`opsd.data.iter_supervision_rows()` yields one normalized record per (problem, reference
view). It accepts two source layouts. The wide layout is the released dataset: one row per
problem with `problem_id`, `question`, `answer`, `difficulty`, `supervision_status` and one
`supervision_<condition>` column per view holding the view body. The long layout is one row
per (problem, condition) with `problem_id`, `condition`, `prompt` (or `question`), `answer`,
`output`, and optionally `target` and `difficulty`.

Both normalize to the same fields:

| field | type | meaning |
|---|---|---|
| `problem_id` | str | stable identifier of the problem |
| `condition` | str | `answer_only`, `gist`, `key_points`, `summary`, `clean_solution`, `full_trace` |
| `prompt` | str | the problem statement, which is the student's question |
| `answer` | str | the verified final answer |
| `output` | str | the view body, empty for `answer_only` |
| `target` | str | what the teacher sees: the body followed by `"\n\nThe final answer is \boxed{answer}."`. For `answer_only` that sentence alone. A long-format row carrying its own `target` keeps it verbatim. |
| `difficulty` | float | per-problem score, or supplied through `AMPLE_MATH_DIFFICULTY` |

`load_problems()` keeps a problem only when it has `answer_only`, `key_points` and
`full_trace` and a difficulty score. The other three views are attached on demand by
`load_extra_pi()`.

## Split files

Two split schemas exist and they are not interchangeable.

The release splits are `data/splits/qwen3_1p7b_splits.json` and
`data/splits/smollm3_3b_splits.json`, listed in `opsd.constants.SPLITS_FILES`. They ship with
the repository and are what every reported run used. `data/build_splits.py --stage build`
produces the first, and `data/build_smollm3_splits.py` the second; each writes into the
artifacts root, and the result is what ships here.

| key | contents |
|---|---|
| `train`, `dev`, `test` | 1536 / 192 / 384 problem ids in a fixed order. That order is the `item_index` of the seed schedule. |
| `group` | `{problem_id: band}` over train+dev+test, the construction band from that backbone's four unprivileged direct-response samples. The bands are `easy`, `frontier` and `hard_recoverable`, 128 of each in the Qwen test split, which is the stratification the estimator resamples over. The SmolLM3 file's dev and test ids are copied from the Qwen file, where recoverability was never measured for that backbone, so they carry a fourth value, `hard`. |
| `base_pass4` | `{problem_id: float}`, that base pass@4 |
| `structured_views` | the five views other than `answer_only`, used for the hard-recoverable teacher check |
| `n_candidates`, `n_prior_excluded`, `seed` | provenance of the draw |
| `base_pass4_generations_source`, `hard_teacher_generations_source` | provenance of the two generation files the buckets came from |

An earlier study on this corpus held out problems of its own. If a file of its splits is at
`<artifacts>/prior_study_split.json`, carrying `train`, `dev` and `test` id lists,
`data/build_splits.py` keeps those problems out of the candidate pool as well. A fresh pool has
no such file, and the builder says so and continues.

## Directory layout under `$OPSD_ARTIFACTS`

Every location is a constant of `opsd/artifact_layout.py`. `path(CONSTANT, "file")` builds an
absolute path, and `resolve()` lets a command-line flag take either a path or a constant name,
as in `--output-dir SEED_EVAL_TEST_TIER`. The table is in the module's own order. `<run>` is a
training-run name, `<S>` a checkpoint step, `<NNN>` a shard number.

| constant | path | holds |
|---|---|---|
| `ARTIFACTS_DIR` | `$OPSD_ARTIFACTS`, default `artifacts` | the root. Imported from `opsd.constants`; every value below is relative to it. |
| `SIX_VIEW_BUILD` | `six_view` | Qwen split construction: `candidate_pool.json` and `splits.json` from `data/build_splits.py`. Parent of three entries below. |
| `SIX_VIEW_TEST_TIER` | `six_view/eval/test` | thinking-enabled test-split rows of the six-view students, `test_b<NNN>_<run>_step<S>_test.jsonl`. The default `--output-dir` of `build_eval_queue.py --mode thinking`. |
| `SIX_VIEW_DEV_TIER` | `six_view/eval/dev` | dev-split rows used to choose a checkpoint, including the step-0 cells. `multi_adapter_eval.py --splits dev --output-dir SIX_VIEW_DEV_TIER` writes them. |
| `BASE_TEST_TIER` | `base/eval/test` | thinking-enabled test-split rows of the frozen base, from evaluating a run's `step_0` adapter with `--prefix base`. The anchor of every in-domain gain. |
| `DIRECT_TRAIN_RUNS` | `train/direct` | Qwen direct-response training runs: `<run>/trajectory.jsonl` and `<run>/step_<n>/` |
| `THINKING_TRAIN_RUNS` | `train/thinking` | the same for Qwen thinking-enabled runs |
| `COMMON_CKPT_TEST_TIER` | `common_checkpoints/eval/test` | test-split rows of both training modes at the checkpoints they share, steps 25 and 50, `--prefix common` |
| `LOSS_WINDOW_COMMON_CKPT_TIER` | `loss_window/eval/test` | test-split rows of the three loss-window runs at those same steps, `--prefix loss_window` |
| `REFERENCE_CONTROLS` | `reference_controls` | the trace-opening and other-problem reference files with their audit trails, from `data/controls/truncated_full_trace.py` and `length_matched_other_problem.py` |
| `CONTROL_STUDIES` | `teacher_controls` | `pi_answer_only_wrong_answer.json` and its pairing file, from `data/controls/wrong_answer_reference.py`. Parent of the two entries below. |
| `CONTROL_EVAL_THINKING` | `teacher_controls/eval/test_thinking` | thinking-enabled test-split rows of the teacher-control and intervention students, `--prefix controls` |
| `CONTROL_EVAL_DIRECT` | `teacher_controls/eval/test_direct` | direct-response test-split rows of the same students and of the base, `--prefix direct`. The default `--output-dir` of `build_eval_queue.py --mode direct`. |
| `SEED_EVAL_TEST_TIER` | `seeds/eval/test` | test-split rows of the replicate-seed students, `--prefix seeds` |
| `EXTERNAL_STEP50` | `external/eval` | `run_external_eval.sh` output: `<run>_step<S>_<bench>.jsonl` plus a `DONE` marker, and the aime25 and hmmt25 id maps |
| `EXTERNAL_BASE_ANCHORS` | `external/base` | frozen-base benchmark results from `official_eval.py`, `base_<bench>.json`. The default `--output-dir` of that script. |
| `EXTERNAL_ID_MAP_AIME24` | `external` | `aime24_id_map_bench_to_anchor.json` |
| `BENCHMARKS` | `benchmarks` | `aime24.json`, `aime25.json`, `hmmt25.json`. The default of `OPSD_BENCHMARKS`. |
| `SMOLLM3` | `smollm3` | `splits_smollm3.json` and `splits_smollm3_build_report.json`. Parent of every SmolLM3 entry below. |
| `SMOLLM3_SPLIT_POOL` | `smollm3/split_pool` | `pool_ids.json` and `teacher_analysis.json`, the two inputs `data/build_smollm3_splits.py` reads besides the evaluation rows |
| `SMOLLM3_SPLIT_POOL_EVAL` | `smollm3/split_pool/eval` | the SmolLM3 base's four direct-response samples over the candidate pool, from `smollm3_eval.py --split ids` |
| `SMOLLM3_SPLIT_POOL_TEACHER_EVAL` | `smollm3/split_pool/teacher_eval` | SmolLM3 teacher generations for the pass@4 == 0 problems, from `smollm3_eval.py --mode teacher` |
| `SMOLLM3_BASE_EVAL` | `smollm3/base/eval` | the frozen SmolLM3 base on the test split in both evaluation modes |
| `SMOLLM3_2X2_EVAL` | `smollm3/eval/views_by_mode` | Answer Only and Full Trace against both training modes at seed 0, steps 25, 50 and 100 |
| `SMOLLM3_SEED1_EVAL` | `smollm3/eval/seed1` | Full Trace at seed 1, both training modes |
| `SMOLLM3_REFERENCE_FREE_EVAL` | `smollm3/eval/reference_free` | the reference-free, Answer Only and Full Trace students at seeds 0 to 2, both evaluation modes, steps 50 and 100 |
| `PROFILE_CALIBRATION` | `profiles/calibration` | `calibration_ids.json` and `full_kl_ids.json` from `python -m opsd.data --stage calibration`, and the thinking-enabled calibration generations `preflight.py` writes there by convention. The 512 calibration ids are excluded from the training splits. |
| `PRIOR_STUDY_SPLIT` | `prior_study_split.json` | a file, not a directory: the earlier study's split described above, read when present |
| `DECONTAM` | `decontam` | `decontam_full_results.json` from `data/decontam_check.py` |
| `SMOLLM3_TRAIN_RUNS` | `train/smollm3` | SmolLM3 training runs in both modes. `artifact_layout.train_runs(model, student_thinking)` picks between this, `DIRECT_TRAIN_RUNS` and `THINKING_TRAIN_RUNS`. |
| `ROLLOUT_WORKERS` | `rollout_worker` | one watch directory per run: `<run>/request.json`, `response.json`, `ready.json`, `adapter_step<n>/` |
| `LOGS` | `logs` | launcher, evaluation-lane and profiling-chain logs |
| `EVAL_SHARDS` | `eval/shards` | 16-problem test-split shard files `<split>_b<NNN>.json`, which are what fix the per-request seeds |
| `EVAL_QUEUE` | `eval/queue` | pending job files, with `claimed/lane<gpu>/`, `done/` and `failed/` beside them |
| `EVAL_LANES` | `eval` | lane state: `markers/<job_id>.DONE` or `.ERROR`, `timings.csv`, `lane<gpu>.pid`, `STOP_LANE<gpu>`. Parent of `EVAL_SHARDS`, `EVAL_QUEUE` and `MULTI_ADAPTER_EVAL`. |
| `MULTI_ADAPTER_EVAL` | `eval/multi_adapter` | the default `--output-dir` of `multi_adapter_eval.py`, for evaluations run outside the queue |
| `PROFILE_EXTRA_VIEWS` | `profiles/calibration_extra_views` | teacher generations and seven-context prefix scores for the three added views, from `preflight_ext.py` |
| `PROFILE_DIRECT_STATES` | `profiles/calibration_direct` | direct-response calibration generations of the frozen base, and the scoring shards `ids_shard0.json` and `ids_shard1.json` |
| `PROFILE_THINKING` | `profiles/thinking` | the thinking-enabled profile: `all_rollouts_prefix_scores*.parquet`, `reduced/`, `analysis/`. The default input and output root of the four profile analyses. |
| `PROFILE_DIRECT` | `profiles/direct` | the direct-response profile with the local prompt templates, same file layout. Point the analyses at it with `--reduced-dir <this>/reduced`; each one's help text names that path. |
| `DRIFT_PROFILE` | `profiles/drift` | the profile re-measured on trained students' own states: `<view>_step<S>_generations.jsonl`, `<view>_step<S>/`, `DRIFT_PROFILE_RESULT.md` |

Two exceptions. `python -m opsd.data --stage eligible` writes its id list to
`<artifacts>/eligible_ids.json`, the one default path with no constant behind it. And
`preflight.py --output`, `smollm3_eval.py --output` and `evaluate.py --output` are required
arguments, so nothing constrains where those three write; the conventional targets are the
directories above.

`VIEWS` is the module's one non-path constant, the six reference views in reference-length
order.

## Run names

A run name is the `<run>` field of an evaluation file name,
`<prefix>_b<NNN>_<run>_step<S>_test.jsonl`, and the directory name under the training-run
root. The launcher's default is `<model>_<CONDITION>_<ROLLOUTS>`, with `_seed<n>` appended at
seed 1 and above. `configs/experiments.md` lists the name behind every run in the paper along
with the variables that reproduce it.

The SmolLM3 file names carry a job tag ahead of the run, listed with each directory in the
table above.

## Record formats

### The evaluation row

One JSON object per line, one line per (problem, sample). Every evaluation in the repository
writes this one format. `evaluate.py` and `smollm3_eval.py` build the rows with
`opsd.generation.generate_records`; `multi_adapter_eval.py` builds the same field set in its
own `generate_and_rescue`, which is the path the queue runner and the external launcher take.
All of them then add the budget fields through `evaluate.py`'s `add_budget_scores`.

| field | meaning |
|---|---|
| `problem_id` | the problem |
| `model` | the run name this file belongs to, `<prefix><run>_step<S>_<split>` |
| `condition` | `no_pi` for a student evaluation. `smollm3_eval.py --mode teacher` puts the view name here instead. |
| `sample_index`, `seed` | which of the samples, and its vLLM request seed from `constants.sample_seed` |
| `verified_answer` | the answer key |
| `input_tokens` | rendered prompt length |
| `generated_tokens`, `max_new_tokens`, `finish_reason` | the main pass |
| `hit_length_cap` | `finish_reason == "length"` |
| `answer_extracted`, `correct` | the boxed answer of the main pass and its grade |
| `output`, `output_token_ids` | the main pass completion |
| `rescued` | whether the rescue pass ran for this row |
| `rescued_max_new_tokens`, `rescued_generated_tokens`, `rescued_finish_reason`, `rescued_answer_extracted`, `rescued_correct`, `rescued_output`, `rescued_output_token_ids` | the rescue pass, all `null` when it did not run |
| `correct_at_4k`, `correct_at_8k`, `correct_at_16k` | the main pass regraded after decoding the first 4096, 8192 and 16384 output tokens |
| `correct_full` | the grade of the full output, rescued if a rescue ran |

A generation that hits its cap is resubmitted once with the same seed and the largest budget
the context still allows. Both outputs are kept. `analysis/estimator.py` takes
`rescued_correct` when `rescued` is true and a `rescued_output` is present, and `correct`
otherwise. That rule is what "rescue-aware" means in the paper.

`smollm3_eval.py` adds three fields of its own to every row: `mode`, `thinking` and
`adapter_dir`.

### The training trajectory record

`training/train_opsd.py` writes `<run>/trajectory.jsonl`, one line per scored example, and
flushes at the end of each optimizer step. A 100-step run at an effective batch of 32 gives
3,200 lines.

| field | meaning |
|---|---|
| `run_name` | `opsd_<condition>`, with `_smoke` appended for a smoke run |
| `step` | optimizer step, 1-based |
| `problem_id`, `condition` | the example and this run's reference view |
| `seed`, `batch_position` | the example's vLLM request seed and its position in the step's batch |
| `completion`, `completion_tokens`, `finish_reason` | the on-policy rollout |
| `pi_prompt_tokens`, `student_prompt_tokens` | the two rendered prompt lengths, kept separate |
| `student_generation_horizon` | the rollout cap |
| `distillation_loss_horizon` | the prefix the KL actually covered. Under `--loss-support-windows` it becomes the number of support tokens instead. |
| `correct_before_update` | grade of the rollout, before this step's update |
| `prompt_style`, `fork_mask_mode`, `fork_marker_tokens` | which templates were used, the fork-masking mode, and the count of correction-marker positions |
| `mean_token_kl` | unclipped forward KL per completion token |
| `mean_clipped_token_kl`, `loss` | the clipped objective. The two are the same number. It can be negative, which the vendored loss documents. |
| `mean_sampled_token_log_ratio` | teacher minus student log-probability on the sampled tokens |

Two groups of fields appear only when the flag that produces them is set, so a default run's
records keep an unchanging key set. `--loss-support-windows` adds `loss_support_windows`,
`n_loss_support_tokens`, `support_coverage_frac`, `reasoning_len` and
`support_reaches_answer_phase`. A per-side template override adds `student_prompt_style` and
`teacher_prompt_style`.

### The rollout-worker protocol

The trainer and the sampler are separate processes on one GPU, in different environments, so
they talk through files in the run's watch directory. Every file is written to a `.tmp`
sibling and renamed, so a reader never sees a partial document.

| file | direction | contents |
|---|---|---|
| `ready.json` | worker | `{"pid": int}`. The launcher waits for it before starting the trainer. |
| `request.json` | trainer | `{"step", "adapter_dir", "items": [{"prompt_token_ids", "seed", "max_tokens"}, ...]}`. The worker deletes it after answering. |
| `response.json` | worker | `{"step", "items": [{"token_ids", "finish_reason"}, ...]}`, in request order. The trainer deletes it after reading. |
| `stop.json` | trainer | any object. The worker exits 0. The launcher writes it even when the trainer crashes, so the worker cannot sit on the GPU. |
| `error.json` | worker | `{"step", "traceback"}`. The worker exits nonzero and the trainer aborts. |

The trainer saves the current adapter to `adapter_step<n>/` before each request and names it
in `request.json`. The worker announces it under a fresh integer LoRA id, which forces a
reload and evicts the previous step's adapter, so the rollouts are always on-policy. The
previous step's directory is removed after the save.

### The queue job file

`evaluation/build_eval_queue.py` writes one JSON file per (run, checkpoint, shard) into the
pending queue. The file name is `<priority>_<ordinal>_<cell>_b<NNN>.json`; plain
lexicographic order is the processing order, and the cell name is the third
underscore-separated field. `eval_queue.py` claims every job of one cell, builds one vLLM
engine for it, and runs the batches through the same `multi_adapter_eval.py` functions a
standalone invocation would use.

| field | contents |
|---|---|
| `job_id` | `<prefix>_<cell>_b<NNN>` |
| `priority`, `exp` | the file-name prefix that orders the queue, and a free-form label |
| `cell` | `<run with underscores replaced by hyphens>-step<S>-<th\|dr>` |
| `tier`, `splits` | the split name, `test` for every reported tier |
| `run`, `adapter_dir` | the run name and the checkpoint directory |
| `enable_thinking` | evaluation mode |
| `runs_config` | a one-entry list `[{"name", "adapter_root", "checkpoint_steps"}]`, forwarded to `multi_adapter_eval.py --runs-config`. The key on disk still carries an older spelling; the docstring above it calls it `runs_config`. |
| `splits_file` | the 16-problem shard file, which fixes this batch's per-request seeds |
| `samples_dev`, `samples_test` | samples per problem, 4 for every reported tier |
| `output_dir`, `run_name_prefix` | where the rows go and the file prefix they carry |
| `max_loras_per_engine`, `gpu_memory_utilization` | engine settings |
| `prompt_style`, `protocol` | `local` and `local` for the in-domain tiers |
| `est_requests` | expected row count; the runner fails the job when the output does not match |
| `expected_output` | the row file this job produces. A job whose expected output already exists is never built, so a cell is not regenerated. |
| `model` | the backbone |

A finished job moves to `done/` or `failed/`, a marker lands in `<EVAL_LANES>/markers/`, and
a line goes into `timings.csv` with `job_id,cell,lane,seconds,rc`.

### The profile parquets

`profiling/score_views.py` teacher-forces the frozen calibration rollouts under seven
contexts: the six reference views plus the student's own no-reference prompt. It writes one
row per (`problem_id`, `sample_index`, `condition`, `position`), where `position` indexes the
scored prefix rather than the raw completion. Columns are `correct`, `normalized_position`,
`token_category` (`think` or `answer`), `region` (`first25`, `mid50`, `last25`), `logp`,
`logp_student`, `log_ratio`, `entropy`, `entropy_student`, `top1_agree_with_student`,
`is_correction_marker`, `eos_prob`, `think_close_prob`, `answer_transition_prob` with their
`_student` counterparts, `full_kl`, `clipped_kl`, `token_id` and `pi_length_tokens`. The
`condition == "none"` rows are the student self-baseline, so their `log_ratio`, `full_kl` and
`clipped_kl` are zero. Beside the parquet the script writes a `_summary.json`.

A full profile is around 124 million rows and 7 GB. `profiling/reduce_profile.py` walks the
shards by row group and writes four small files under `reduced/`:

| file | grouping |
|---|---|
| `per_group.parquet` | one row per (`problem_id`, `sample_index`, `condition`), with the per-trajectory sums and means: `log_ratio_mean`, `full_kl_sum`, the think/answer and quartile splits of the KL, marker counts, the stop-probability aggregates |
| `position_profile.parquet` | one row per (`condition`, `correct`, position bin), 50 bins |
| `marker_tokens.parquet` | the raw per-token rows where `is_correction_marker` is true |
| `reduce_manifest.json` | row counts, the column list of each output, and what could not be derived |

`correctness_alignment.py` and `censored_tail.py` read `reduced/` and write `tables/*.csv`,
`plots/*.png` and a summary JSON under `analysis/correctness-alignment/` and
`analysis/censored-tail/`. `fork_correction.py` writes its CSVs and `fork-correction.md`
straight into `analysis/fork-correction/`. `correctness_alignment.py` also makes one chunked
pass over the raw shards, for the region and think/answer breakdown that the reduction does
not carry. `drift_profile_compare.py` reads those outputs for the base profile and for each
trained student's own states, and writes `DRIFT_PROFILE_RESULT.md`.

### The external anchor file

`evaluation/official_eval.py` writes `<label>_<bench>.json` through the official evaluator.
`analysis/estimator.py` reads `average_at_n_pct` and the `results` list, taking
`problem_id` and `num_correct` out of 12 from each entry. The benchmark problems are numbered
differently on the two sides of that comparison, which is what
`evaluation/build_benchmark_id_maps.py` reconciles into
`<bench>_id_map_bench_to_anchor.json`.

## Training runs

`training/launch_run.sh` starts the rollout worker, waits for `ready.json`, starts the
trainer, and writes `stop.json` when the trainer exits either way. The matched recipe is the
default: 100 optimizer steps, effective batch 32, flat learning rate 5e-6, loss temperature
1.1, AdamW beta2 0.999, per-term KL clip 0.05, LoRA rank 64 and alpha 128 over seven
projection modules, and a frozen thinking-enabled teacher. Three things depend on the mode:

| | `TRAIN_MAX_NEW_TOKENS` | `LOSS_HORIZON` | `CONTEXT` |
|---|---|---|---|
| Qwen3-1.7B, `ROLLOUTS=direct` | 1024 | the cap | 32768 |
| Qwen3-1.7B, `ROLLOUTS=thinking` | 20764 | 1024 | 40960 |
| SmolLM3-3B, `ROLLOUTS=direct` | 1024 | the cap | 40960 |
| SmolLM3-3B, `ROLLOUTS=thinking` | 24536 | 1024 | 40960 |

Each thinking cap is `CONTEXT` minus the longest Full Trace teacher prompt minus 256. The
launcher asks for checkpoints at steps 0, 1, 2, 5, 10, 15, 20, 25, 50, 75 and 100;
`train_opsd.py`'s own default, used when the trainer is called directly, is 25, 50, 75, 100.

The controls are trainer flags passed through `EXTRA_ARGS`. `--unprivileged-teacher` scores
the frozen backbone on the student's own prompt with no reference section at all.
`--pi-override-file <json>` substitutes a constructed reference text for the AMPLE-Math
lookup. `--fork-mask-mode exclude` or `downweight` changes what the KL covers at
correction-marker positions. `--prompt-style official` swaps in the official templates, and
`--teacher-prompt-style official` swaps only the teacher's. `configs/experiments.md` gives
the values behind each run in the paper.

## How the pieces connect

```
AMPLE-Math rows ─► opsd.data ─► data/build_splits.py ─► data/splits/*.json  (shipped)
      │
      ├─► python -m opsd.data --stage calibration ─► PROFILE_CALIBRATION
      │        └─► profiling/preflight.py ─► <prefix>_generations.jsonl
      │                └─► score_views.py ─► prefix-score shards
      │                        └─► reduce_profile.py ─► reduced/
      │                                └─► correctness_alignment.py
      │                                    fork_correction.py
      │                                    censored_tail.py
      │
      └─► training/launch_run.sh ─► <train runs>/<run>/step_<n>/
                  │                 <train runs>/<run>/trajectory.jsonl
                  │                        │
                  │                        ├─► build_eval_queue.py ─► EVAL_SHARDS + EVAL_QUEUE
                  │                        │       └─► eval_lane.sh ─► eval_queue.py ─► <tier>/*.jsonl
                  │                        │
                  │                        └─► run_external_eval.sh ─► EXTERNAL_STEP50
                  │
                  └─► rollout_worker.py  (ROLLOUT_WORKERS/<run>/)

python -m opsd.benchmarks --materialize ─► BENCHMARKS ─► run_external_eval.sh
official_eval.py ─► EXTERNAL_BASE_ANCHORS ─► build_benchmark_id_maps.py ─► the id maps

<tier>/*.jsonl  +  EXTERNAL_STEP50/*.jsonl ─► analysis/estimator.py
                                              ─► the levels, contrasts and intervals
```

`profiling/run_drift_profile.sh` runs the whole profile chain a second time, on the states a
trained student visits rather than the frozen base's, and `drift_profile_compare.py` puts the
two side by side.

The frozen-base anchors are produced the same way as everything else. `BASE_TEST_TIER` and
`SMOLLM3_BASE_EVAL` come from evaluating a run's `step_0` adapter, which is the untrained
adapter saved before training starts. `EXTERNAL_BASE_ANCHORS` comes from
`evaluation/official_eval.py` on the bare backbone.
