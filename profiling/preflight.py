"""Calibration-set generation for one model: unprivileged (no-reference) student samples
plus one teacher sample per reference view, used as the states the privilege profile is
measured on.

vLLM env only. Same-prefix scoring is a separate step (prefix_score.py /
score_views.py, training env).

    python profiling/preflight.py --model Qwen/Qwen3-1.7B --output artifacts/profiles/calibration/qwen3_1.7b
    python profiling/preflight.py --model Qwen/Qwen3-1.7B --output /tmp/smoke --limit 2 \
        --max-new-tokens 256 [--dry-run]

Writes <output>_generations.jsonl (resumable: (problem_id, condition,
sample_index) triples already in the file are skipped) and the rendered prompts
of the first three calibration problems under <output>_rendered_prompts/.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer  # noqa: E402

from opsd.artifact_layout import PROFILE_CALIBRATION, path as artifact_path  # noqa: E402
from opsd.constants import (  # noqa: E402
    CONDITIONS,
    MAIN_MAX_NEW_TOKENS,
    MODEL_INDEX,
    NO_PI_SAMPLES,
    PI_GENERATION_MAX_NEW_TOKENS,
    PRIMARY_MODEL,
    sample_seed,
)
from opsd.data import load_problems  # noqa: E402
from opsd.generation import (  # noqa: E402
    generate_records,
    load_llm,
    load_jsonl,
    lora_request,
    prepare_prompts,
    save_jsonl,
)
from opsd.prompts import (  # noqa: E402
    official_student_messages,
    save_rendered_examples,
    student_messages,
    teacher_messages,
)


def build_tasks(problems, calibration_ids, model, prompt_style="local", pi_conditions=None):
    """Returns (no-PI tasks, PI tasks). item_index is the position in the
    calibration list, so seeds are matched across models and conditions.

    prompt_style: "local" (default) uses prompts.STUDENT_TEMPLATE for the no-PI student
    prompt; "official" uses the verbatim official-OPSD train-time student wording
    (prompts.official_student_messages). Only the no-PI/student task is affected -- PI
    tasks (this function's second return value) always use the local teacher_messages()
    template.

    pi_conditions: which of CONDITIONS to generate PI tasks for (default None -> all of
    CONDITIONS). Lets a caller skip an unwanted condition entirely instead of generating
    and discarding it."""
    model_index = MODEL_INDEX[model]
    no_pi_tasks, pi_tasks = [], []
    student_msg_fn = official_student_messages if prompt_style == "official" else student_messages
    pi_conditions = CONDITIONS if pi_conditions is None else pi_conditions

    for item_index, pid in enumerate(calibration_ids):
        problem = problems[pid]
        common = {
            "problem_id": pid,
            "model_tag": model,
            "verified_answer": problem["verified_answer"],
        }

        for sample_index in range(NO_PI_SAMPLES[model]):
            no_pi_tasks.append(
                dict(
                    common,
                    condition="no_pi",
                    sample_index=sample_index,
                    seed=sample_seed(model_index, sample_index, item_index),
                    messages=student_msg_fn(problem["question"]),
                )
            )

        for condition in pi_conditions:
            pi_tasks.append(
                dict(
                    common,
                    condition=condition,
                    sample_index=0,
                    seed=sample_seed(model_index, 0, item_index),
                    messages=teacher_messages(problem["question"], problem["pi"][condition]),
                )
            )

    return no_pi_tasks, pi_tasks


def drop_done(tasks, done):
    return [t for t in tasks if (t["problem_id"], t["condition"], t["sample_index"]) not in done]


def print_task_table(tokenizer, tasks, max_new_tokens, label, enable_thinking=True):
    prompt_ids, budgets = prepare_prompts(tokenizer, tasks, max_new_tokens, enable_thinking=enable_thinking)
    print(f"\n{label}: {len(tasks)} tasks, cap {max_new_tokens}, enable_thinking={enable_thinking}")
    print(f"{'idx':>4} {'problem_id':<28} {'condition':<12} {'smp':>3} {'seed':>9} {'in_tok':>7} {'max_new':>8}")
    for i, (task, ids, budget) in enumerate(zip(tasks, prompt_ids, budgets)):
        print(
            f"{i:>4} {task['problem_id'][:28]:<28} {task['condition']:<12} "
            f"{task['sample_index']:>3} {task['seed']:>9} {len(ids):>7} {budget:>8}"
        )


def print_summary(records, label):
    by_condition = defaultdict(list)
    for record in records:
        by_condition[record["condition"]].append(record)
    print(f"\n{label} summary")
    for condition, group in sorted(by_condition.items()):
        n = len(group)
        correct = sum(r["rescued_correct"] if r["rescued"] else r["correct"] for r in group)
        capped = sum(r["hit_length_cap"] for r in group)
        capped_after = sum(
            (r["rescued_finish_reason"] == "length") if r["rescued"] else r["hit_length_cap"]
            for r in group
        )
        no_answer = sum(r["answer_extracted"] is None for r in group)
        print(
            f"  {condition:<12} n={n:<5} correct={correct / n:.3f} "
            f"length_cap={capped / n:.3f} after_rescue={capped_after / n:.3f} "
            f"no_answer={no_answer / n:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=sorted(NO_PI_SAMPLES),
        help="backbone to generate the calibration states for. The choices are the backbones "
             "constants.NO_PI_SAMPLES gives an unprivileged-sample count for, which is also "
             "the set prefix_score.py and score_views.py can score; the reported "
             f"profiles are {PRIMARY_MODEL} with {NO_PI_SAMPLES[PRIMARY_MODEL]} unprivileged "
             "samples per calibration problem.",
    )
    parser.add_argument("--output", required=True, help="path prefix, e.g. artifacts/profiles/calibration/qwen3_1.7b")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N calibration problems")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="override both the no-PI and PI caps (smoke runs)",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="lower it when another job shares the GPU")
    parser.add_argument("--dry-run", action="store_true", help="print the planned task table and exit")
    parser.add_argument(
        "--student-thinking", dest="student_thinking", action="store_true", default=True,
        help="enable_thinking for the no-PI student generation (default: on). The mode-matched "
             "direct-response profile needs --no-student-thinking.",
    )
    parser.add_argument("--no-student-thinking", dest="student_thinking", action="store_false")
    parser.add_argument(
        "--skip-pi-generation", action="store_true",
        help="do not generate the PI-teacher completions at all (profiles that only need fresh "
             "no-PI/student trajectories; all six reference views are scored teacher-forced "
             "against those trajectories by score_views.py, not resampled here)",
    )
    parser.add_argument(
        "--skip-no-pi-generation", action="store_true",
        help="do not generate the no-PI/student completions at all (mirror of --skip-pi-generation, "
             "for jobs that only need the PI-conditioned teacher completions, e.g. a teacher-"
             "recoverability pass restricted to a small id subset that already has its no-PI "
             "samples from a separate run)",
    )
    parser.add_argument(
        "--ids-file", default=None,
        help="path to a JSON list of problem ids to generate for, overriding the default "
             "<artifacts>/profiles/calibration/calibration_ids.json (the frozen calibration set).",
    )
    parser.add_argument(
        "--prompt-style", choices=["local", "official"], default="local",
        help="student prompt template for the no-PI task: 'local' (default, prompts.STUDENT_TEMPLATE) "
             "or 'official' (verbatim official OPSD train-time wording, "
             "prompts.official_student_messages). PI-task prompts are unaffected.",
    )
    parser.add_argument(
        "--adapter", default=None,
        help="LoRA checkpoint dir served live on top of --model (vLLM enable_lora + a "
             "per-request LoRARequest, exactly as evaluation/evaluate.py does it; never merged "
             "into the weights). Default None = the frozen base model. Used to re-generate the "
             "profile states from a trained student while everything else about the run stays "
             "identical (see run_drift_profile.sh).",
    )
    parser.add_argument(
        "--pi-conditions", nargs="+", default=None, choices=CONDITIONS,
        help=f"which of CONDITIONS ({CONDITIONS}) to generate PI-teacher completions for "
             "(default: all of them). Lets a caller skip an unwanted condition instead of "
             "generating and discarding it.",
    )
    args = parser.parse_args()

    problems = load_problems()
    ids_path = args.ids_file or artifact_path(PROFILE_CALIBRATION, "calibration_ids.json")
    calibration_ids = json.load(open(ids_path))
    if args.limit:
        calibration_ids = calibration_ids[: args.limit]
    print(f"pool: {len(problems)} problems, ids: {len(calibration_ids)} problems (source: {ids_path})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    no_pi_tasks, pi_tasks = build_tasks(
        problems, calibration_ids, args.model, prompt_style=args.prompt_style,
        pi_conditions=args.pi_conditions,
    )
    if args.skip_pi_generation:
        pi_tasks = []
    if args.skip_no_pi_generation:
        no_pi_tasks = []

    out_path = f"{args.output}_generations.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {(r["problem_id"], r["condition"], r["sample_index"]) for r in load_jsonl(out_path)}
        print(f"resuming: {len(done)} generations already in {out_path}")
    no_pi_tasks = drop_done(no_pi_tasks, done)
    pi_tasks = drop_done(pi_tasks, done)

    no_pi_cap = args.max_new_tokens or MAIN_MAX_NEW_TOKENS
    pi_cap = args.max_new_tokens or PI_GENERATION_MAX_NEW_TOKENS

    if args.dry_run:
        print_task_table(tokenizer, no_pi_tasks, no_pi_cap, "no-PI tasks", enable_thinking=args.student_thinking)
        print_task_table(tokenizer, pi_tasks, pi_cap, "PI tasks", enable_thinking=True)
        print(f"\nplanned: {len(no_pi_tasks)} no-PI + {len(pi_tasks)} PI generations "
              f"(student_thinking={args.student_thinking}, prompt_style={args.prompt_style})")
        print(f"per condition: {dict(Counter(t['condition'] for t in no_pi_tasks + pi_tasks))}")
        return

    # Keep the fully rendered prompts of the first three problems.
    save_rendered_examples(
        tokenizer, problems, calibration_ids, f"{args.output}_rendered_prompts",
        student_enable_thinking=args.student_thinking,
    )

    llm = load_llm(args.model, gpu_memory_utilization=args.gpu_memory_utilization,
                   lora=args.adapter is not None)
    lora = lora_request(args.adapter)  # None when --adapter is not given
    if args.adapter:
        print(f"serving {args.model} + LoRA {args.adapter}")
    task_groups = [
        (no_pi_tasks, no_pi_cap, "no-PI", args.student_thinking),
        (pi_tasks, pi_cap, "PI", True),  # teacher/PI role stays thinking-on, unaffected by this flag
    ]
    for tasks, cap, label, enable_thinking in task_groups:
        if not tasks:
            continue
        records = generate_records(llm, tokenizer, tasks, cap, lora=lora,
                                   enable_thinking=enable_thinking)
        save_jsonl(records, out_path, append=True)
        print_summary(records, label)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
    # vLLM 0.12's engine teardown can hang after all work is done: results are on disk after
    # main(); a bare process exit can leave the EngineCore child alive, squatting on the GPU.
    # Terminate children before exiting.
    import psutil

    for child in psutil.Process().children(recursive=True):
        child.terminate()
    psutil.wait_procs(psutil.Process().children(recursive=True), timeout=10)
    for child in psutil.Process().children(recursive=True):
        child.kill()
    sys.stdout.flush()
    os._exit(0)
