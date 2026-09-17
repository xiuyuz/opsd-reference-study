# Every training run in the paper, and the variables that reproduce it

`training/launch_run.sh` trains one student. It starts the rollout worker and the trainer on one GPU and takes
everything from environment variables, which its header documents in full. These are common to every run in the paper:

```bash
MODEL=Qwen/Qwen3-1.7B            # or HuggingFaceTB/SmolLM3-3B
SEED=0                           # replication seed, where a run was repeated
GPU=0
```

The rest of the recipe is the same everywhere and is already the default: 100 optimizer steps, an effective batch of
32, a flat learning rate, loss temperature 1.1, AdamW beta2 0.999, rollout temperature 1.1, a per-term KL clip of 0.05,
LoRA rank 64 and alpha 128 over the seven projection modules, and a frozen thinking-enabled teacher.

`TAG` names the run. It defaults to the backbone, the condition and the rollout mode joined by underscores, which is
the name in the tables below wherever nothing else distinguishes two runs. Where two runs share a condition, set `TAG`
to the name given. Checkpoints land in `$OPSD_ARTIFACTS/<training runs>/$TAG/step_<n>/`.

## Reference views and teacher controls (Qwen3-1.7B, direct-response rollouts)

`ROLLOUTS=direct TRAIN_MAX_NEW_TOKENS=1024 CONTEXT=32768`. The loss covers the whole 1,024-token rollout, and
checkpoints are saved at 0, 1, 2, 5, 10, 15, 20, 25, 50, 75 and 100.

| reference | run name | CONDITION | EXTRA_ARGS | seeds |
|---|---|---|---|---|
| Answer Only | `qwen3-1.7b_answer_only_direct` | `answer_only` | | 0, 1, 2, 3 |
| Gist | `qwen3-1.7b_gist_direct` | `gist` | | 0 |
| Key Points | `qwen3-1.7b_key_points_direct` | `key_points` | | 0 |
| Clean Solution | `qwen3-1.7b_clean_solution_direct` | `clean_solution` | | 0, 1, 2 |
| Summary | `qwen3-1.7b_summary_direct` | `summary` | | 0 |
| Full Trace | `qwen3-1.7b_full_trace_direct` | `full_trace` | | 0, 1, 2, 3 |
| reference-free teacher | `qwen3-1.7b_reference_free_direct` | `reference_free` | `--unprivileged-teacher` | 0, 1, 2 |
| wrong-answer reference | `qwen3-1.7b_answer_only_wrong_answer_direct` | `answer_only_wrong_answer` | `--pi-override-file $OPSD_ARTIFACTS/teacher_controls/pi_answer_only_wrong_answer.json` | 0, 1, 2 |

A seed other than 0 adds `_seed<N>` to the name, which is what the launcher does by default. Build the wrong-answer
reference first, with `python data/controls/wrong_answer_reference.py`. In the intervention table below, `<controls>`
stands for `$OPSD_ARTIFACTS/reference_controls`.

## Thinking-enabled training rollouts (Qwen3-1.7B, seed 0)

`ROLLOUTS=thinking TRAIN_MAX_NEW_TOKENS=20764 LOSS_HORIZON=1024 CONTEXT=40960`. The loss covers the first 1,024 tokens
of each rollout; the generation cap is the context minus the longest Full Trace teacher prompt minus 256.

| reference | run name | CONDITION | EXTRA_ARGS |
|---|---|---|---|
| Answer Only | `qwen3-1.7b_answer_only_thinking` | `answer_only` | |
| Clean Solution | `qwen3-1.7b_clean_solution_thinking` | `clean_solution` | |
| Full Trace (Early-1K) | `qwen3-1.7b_full_trace_thinking` | `full_trace` | |
| Full Trace, First-4K | `qwen3-1.7b_full_trace_thinking_first4k` | `full_trace` | `LOSS_HORIZON=4096` |
| Full Trace, Distributed-1K | `qwen3-1.7b_full_trace_thinking_distributed1k` | `full_trace` | `LOSS_HORIZON=""` and `--loss-support-windows 0.125,0.375,0.625,0.875 --loss-support-window-size 256` |

## Matched interventions (Qwen3-1.7B, direct-response rollouts, seed 0)

Same variables as the reference views above.

| intervention | run name | CONDITION | EXTRA_ARGS |
|---|---|---|---|
| Full Trace, marker loss excluded | `qwen3-1.7b_full_trace_markers_excluded_direct` | `full_trace` | `--fork-mask-mode exclude` |
| Full Trace, marker loss downweighted | `qwen3-1.7b_full_trace_markers_downweighted_direct` | `full_trace` | `--fork-mask-mode downweight` |
| Clean Solution, marker loss excluded | `qwen3-1.7b_clean_solution_markers_excluded_direct` | `clean_solution` | `--fork-mask-mode exclude` |
| Clean Solution, marker loss downweighted | `qwen3-1.7b_clean_solution_markers_downweighted_direct` | `clean_solution` | `--fork-mask-mode downweight` |
| Clean Solution, other-problem reference | `qwen3-1.7b_clean_solution_other_problem_direct` | `irrelevant_rationale_clean_solution` | `--pi-override-file <controls>/irrelevant_rationale_clean_solution.json` |
| Key Points, other-problem reference | `qwen3-1.7b_key_points_other_problem_direct` | `irrelevant_rationale_key_points` | `--pi-override-file <controls>/irrelevant_rationale_key_points.json` |
| Full Trace opening, Clean Solution length | `qwen3-1.7b_full_trace_opening_clean_solution_direct` | `trunc_full_trace_to_clean_solution` | `--pi-override-file <controls>/trunc_full_trace_to_clean_solution.json` |
| Full Trace opening, Key Points length | `qwen3-1.7b_full_trace_opening_key_points_direct` | `trunc_full_trace_to_key_points` | `--pi-override-file <controls>/trunc_full_trace_to_key_points.json` |
| Full Trace opening, Summary length | `qwen3-1.7b_full_trace_opening_summary_direct` | `trunc_full_trace_to_summary` | `--pi-override-file <controls>/trunc_full_trace_to_summary.json` |
| Full Trace, alternative prompt templates | `qwen3-1.7b_full_trace_prompt_swap_direct` | `full_trace` | `--prompt-style official` |
| Clean Solution, alternative prompt templates | `qwen3-1.7b_clean_solution_prompt_swap_direct` | `clean_solution` | `--prompt-style official` |

The reference files come from `data/controls/`: `truncated_full_trace.py --target-view {key_points,clean_solution,summary}`
cuts each Full Trace to the target view's length, keeping the opening and dropping the later reasoning and the answer
section; `length_matched_other_problem.py --target-view {key_points,clean_solution}` swaps in another problem's view of
the same length. `--prompt-style official` swaps both the student and the teacher to the released OPSD templates;
evaluation keeps this repository's templates either way.

## SmolLM3-3B

`CONTEXT=40960` throughout. Direct-response runs use `ROLLOUTS=direct TRAIN_MAX_NEW_TOKENS=1024`, thinking-enabled
runs `ROLLOUTS=thinking TRAIN_MAX_NEW_TOKENS=24536 LOSS_HORIZON=1024`. Checkpoints at 25, 50, 75 and 100
(`CHECKPOINT_STEPS=25,50,75,100`). The split comes from `data/splits/smollm3_3b_splits.json`, rebuilt from this
backbone's own direct-response outcomes by `data/build_smollm3_splits.py`.

| reference | run name | CONDITION | rollouts | seeds |
|---|---|---|---|---|
| Answer Only | `smollm3-3b_answer_only_direct` | `answer_only` | direct | 0, 1, 2 |
| Answer Only | `smollm3-3b_answer_only_thinking` | `answer_only` | thinking | 0 |
| Full Trace | `smollm3-3b_full_trace_direct` | `full_trace` | direct | 0, 1, 2 |
| Full Trace | `smollm3-3b_full_trace_thinking` | `full_trace` | thinking | 0, 1 |
| reference-free teacher | `smollm3-3b_reference_free_direct` | `reference_free` | direct | 0, 1, 2 |

## Evaluation

In domain: the frozen 384-problem test split, four samples per problem, a 16,384-token main pass, and a regeneration of
every capped generation under `min(32,512, context - prompt - 256)` with the same sampling seed. The same checkpoints
are evaluated with thinking enabled and disabled, and prefix grades at 4,096, 8,192 and 16,384 tokens come from the
saved main-pass tokens. The paper reports step 100 for Qwen3-1.7B and step 50 for SmolLM3-3B, showing both where it
compares them.

```bash
python evaluation/build_eval_queue.py --runs <TAG> --steps 100 --mode thinking --build
bash evaluation/eval_lane.sh 0
```

External: AIME 2024, AIME 2025 and HMMT February 2025 at step 50, twelve samples per problem under the released
protocol (`evaluation/run_external_eval.sh`).

Where the paper reports a development-selected checkpoint, the choice maximizes Avg@2 over steps
25, 50, 75 and 100 on the 192 development problems, with no test access:

```bash
python evaluation/build_eval_queue.py --runs <run name> --steps 25,50,75,100 --mode thinking \
    --split dev --samples 2 --output-dir SIX_VIEW_DEV_TIER --prefix dev --build
bash evaluation/eval_lane.sh 0
```

The selected step is the one with the highest Avg@2 among those four.

## Reference profiles

Four unprivileged samples per calibration problem, scored by the teacher under every view over the same retained
prefix, then reduced and analysed. The README gives the command chain; `profiling/run_drift_profile.sh` repeats the
direct-response profile on trained students' own step-50 states.
