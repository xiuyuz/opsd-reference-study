# What Does Privileged Information Add to On-Policy Self-Distillation?

<p align="center">
  <a href="https://arxiv.org/abs/2609.20612"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.20612-b31b1b?logo=arxiv&logoColor=white"></a>
  <a href="https://huggingface.co/datasets/xiuyuz/ample-math"><img alt="AMPLE-Math on Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-AMPLE--Math-ffce1c?logo=huggingface&logoColor=white"></a>
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/License-MIT-3da639"></a>
</p>

Code for the paper. In on-policy self-distillation a frozen teacher scores the student's own rollouts; here the
teacher also sees a worked reference for the problem, and the study asks what that reference adds. Everything but the
reference is held fixed, so the six references below are the only thing that changes.

Here you will find the training loop, the evaluation protocol, the reference profiles, the matched controls and the
estimator behind every number in the paper.

The training problems come from AMPLE-Math, released separately as the dataset
[`xiuyuz/ample-math`](https://huggingface.co/datasets/xiuyuz/ample-math). Its page carries the rows and how they were
built.

## The six references

Every problem carries six views of the same verified answer, differing only in how much of the reasoning they show.
Mean token counts are over the 512 calibration problems, measured with the Qwen3 tokenizer.

| view | what the teacher sees | mean tokens |
|---|---|---|
| Answer Only | the verified answer, no reasoning body | 13 |
| Gist | the trace compressed into a short paragraph | 76 |
| Key Points | the trace compressed into ordered steps | 284 |
| Clean Solution | the source's polished solution | 526 |
| Summary | the trace rewritten as a narrative | 841 |
| Full Trace | the complete reasoning trace | 4,916 |

Two controls sit off this scale. The reference-free teacher gets no privileged text at all, which makes it cross-mode
self-distillation rather than a reference comparison. The wrong-answer reference is Answer Only carrying a different
problem's answer.

## Layout

| directory | contents |
|---|---|
| `opsd/` | the package: chat templates and prompts, reference views, the OPSD loss, data loading, grading, benchmarks, the artifact layout |
| `training/` | `train_opsd.py` (the LoRA student and the frozen teacher), `rollout_worker.py` (the vLLM sampler), `launch_run.sh` (trains one student from environment variables) |
| `evaluation/` | in-domain evaluation of checkpoints in both modes (`build_eval_queue.py`, `eval_lane.sh`), external benchmarks (`run_external_eval.sh`) |
| `profiling/` | reference profiles on the calibration set: correctness alignment, correction-marker pressure, temporal KL allocation |
| `data/` | split construction, the frozen splits of both backbones, the control references (wrong answer, other problem, trace opening) |
| `analysis/` | `estimator.py`, the estimator behind every interval in the paper |
| `configs/experiments.md` | every training run in the paper with the variables that reproduce it |
| `docs/ARTIFACTS.md` | environment variables, the artifact directory layout, record schemas |

## Environments

Two environments are used, because the trainer and the sampler pin different versions of PyTorch:

```bash
python -m venv ~/envs/opsd-infer && ~/envs/opsd-infer/bin/pip install -r requirements/inference-env.txt
python -m venv ~/envs/opsd-train && ~/envs/opsd-train/bin/pip install -r requirements/training-env.txt
```

The inference environment runs the rollout worker, evaluation, profile generation, the dataset construction and all
analyses. The training environment runs `train_opsd.py` and the teacher-forced profile scoring. The four shell launchers
start both, so they need to be told where each one is:

```bash
export VLLM_PY=~/envs/opsd-infer/bin/python     # inference interpreter
export HF_PY=~/envs/opsd-train/bin/python       # training interpreter
```

Run the plain `python script.py` commands below with the interpreter that script belongs to (`docs/ARTIFACTS.md` lists
which is which). Three more variables are read everywhere:

```bash
export OPSD_ARTIFACTS=/path/to/artifacts        # checkpoints, rollouts, evaluations, profiles (created on demand)
export HF_HOME=/path/to/hf_cache                # Qwen/Qwen3-1.7B and HuggingFaceTB/SmolLM3-3B are read from here
export AMPLE_MATH_JSONL=/path/to/ample_math.jsonl   # optional: a local copy instead of the Hugging Face dataset
```

Without `AMPLE_MATH_JSONL` the loader reads the dataset named by `AMPLE_MATH_HF_ID`, which defaults to
`xiuyuz/ample-math`. Set `AMPLE_MATH_JSONL` instead to read a local copy of the rows. Everything this code writes lives
under `OPSD_ARTIFACTS`; nothing is written into the repository.

## Quickstart

Train the Answer Only student of Qwen3-1.7B on one GPU, with seed 0 and direct-response rollouts:

```bash
CONDITION=answer_only SEED=0 GPU=0 bash training/launch_run.sh
```

That starts the rollout worker and the trainer on the same GPU and saves adapter checkpoints under
`$OPSD_ARTIFACTS/<training runs>/<name>/step_<n>/`. Which training-run directory depends on the backbone and the
rollout mode, and `opsd.artifact_layout.train_runs` resolves it. Nothing has to be built first, since the frozen splits
ship in `data/splits/`. The launcher's header documents every variable it reads, and `configs/experiments.md` gives the
values behind each result in the paper.

`TAG` names the run. The analyses look for the names the paper's runs were saved under, which are listed in
`opsd/artifact_layout.py`, so give `TAG` the matching name if you want an analysis to pick your run up:

```bash
TAG=qwen3-1.7b_answer_only_direct CONDITION=answer_only SEED=0 GPU=0 bash training/launch_run.sh
```

Evaluate the step-100 checkpoint on the frozen 384-problem test split, first with thinking enabled, then with it
disabled:

```bash
python evaluation/build_eval_queue.py --runs <run> --steps 100 --mode thinking --build
python evaluation/build_eval_queue.py --runs <run> --steps 100 --mode direct --build
bash evaluation/eval_lane.sh 0            # one lane per GPU; several lanes drain the same queue
```

The external benchmarks are AIME 2024, AIME 2025 and HMMT February 2025, twelve samples per problem. Build the
benchmark files once, then list the checkpoints to evaluate in a JSON file:

```bash
python -m opsd.benchmarks --materialize
cat > runs.json <<JSON
[{"name": "qwen3-1.7b_answer_only_direct",
  "adapter_root": "$OPSD_ARTIFACTS/train/direct/qwen3-1.7b_answer_only_direct",
  "checkpoint_steps": [50]}]
JSON
RUNS_CONFIG=runs.json GPU=0 bash evaluation/run_external_eval.sh
```

A reference profile measures what each view does to the teacher's distribution over states the student actually
visits. The chain below builds the direct-response profile, the one behind the paper's profile figure. Generation runs
in the inference environment and the teacher-forced scoring in the training environment; everything after that is CPU
work. The released profile split the scoring over two GPUs, so it has two shards where this single-process version has
one:

```bash
python -m opsd.data --stage calibration
python profiling/preflight.py --model Qwen/Qwen3-1.7B --no-student-thinking --skip-pi-generation \
    --output $OPSD_ARTIFACTS/profiles/calibration_direct/qwen3_1.7b
PROF=$OPSD_ARTIFACTS/profiles/direct
python profiling/score_views.py --no-student-thinking --prompt-style local --scoring-horizon 1024 \
    --generations $OPSD_ARTIFACTS/profiles/calibration_direct/qwen3_1.7b_generations.jsonl \
    --output $PROF/all_rollouts_prefix_scores.shard0.parquet
python profiling/reduce_profile.py --shards $PROF/all_rollouts_prefix_scores.shard0.parquet --out-dir $PROF/reduced
python profiling/correctness_alignment.py --shards $PROF/all_rollouts_prefix_scores.shard0.parquet \
    --reduced-dir $PROF/reduced --out-dir $PROF/analysis/correctness-alignment
python profiling/fork_correction.py --reduced-dir $PROF/reduced --out-dir $PROF/analysis/fork-correction
python profiling/censored_tail.py --reduced-dir $PROF/reduced \
    --generations $OPSD_ARTIFACTS/profiles/calibration_direct/qwen3_1.7b_generations.jsonl \
    --out-dir $PROF/analysis/censored-tail
```

Those three produce the panels of the paper's profile figure: correctness alignment, marker
pressure, and the share of the teacher's divergence in each quarter of the scored span. The three
generated views need their own teacher samples first, with `profiling/preflight_ext.py`.

## Scale

Every training run and every evaluation in the paper fits on a single 80GB GPU. The reported numbers came from H100s.
A hundred steps of direct-response training take about half an hour, since the rollouts stop at 1,024 tokens. The same
hundred steps with thinking enabled take six to eight hours, because a rollout can run to 20,764 tokens. Evaluating one
checkpoint on the test split in one mode splits into 24 jobs of five to ten minutes, so two to four GPU-hours in all,
and extra lanes drain the queue in parallel.

## Reproducing the paper

Everything above writes under `OPSD_ARTIFACTS`. `opsd/artifact_layout.py` names each directory,
and `docs/ARTIFACTS.md` describes the layout and the record formats.

The evaluation records behind the paper are not part of this release. What is here is the estimator
that turned them into the numbers you read: `analysis/estimator.py` holds the rescue-aware
correctness rule, the stratified problem-cluster bootstrap with the paper's seed and strata, the
seed-aware average, Holm's correction and the paired external resample. `analysis/README.md`
explains how to apply it to records from your own runs.

## Citation

If you find this work useful, please consider citing it:

```bibtex
@misc{zhang2026privileged,
  title         = {What Does Privileged Information Add to On-Policy Self-Distillation?},
  author        = {XiuYu Zhang and Wei Chow and Junfeng Fang and Zhenkai Liang and Tat-Seng Chua},
  year          = {2026},
  eprint        = {2609.20612},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.20612}
}
```
