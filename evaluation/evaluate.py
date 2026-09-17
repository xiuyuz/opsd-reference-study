"""Generate + grade dev/test/benchmark splits with one model (optionally + one LoRA adapter).

Runs under the vLLM env. Generation uses opsd.generation's vLLM backend with the main
16,384-token cap + rescue (constants.MAIN_MAX_NEW_TOKENS). Grading uses opsd.math_grader
(vendored math_verify wrapper). Budget-prefix correctness is computed from the saved
`output_token_ids` -- no re-generation -- by decoding the first 4096/8192/16384 tokens
and, separately, the full (rescued if present) output.

A trained student is served as the backbone plus its LoRA adapter live (`--adapter`,
vLLM enable_lora + a per-request LoRARequest), exactly as the official OPSD evaluator
does. The adapter is never merged into the weights: at lr 5e-6 the LoRA delta is ~1e-3
of the weight scale, well under one bf16 ulp, so a bf16 `merge_and_unload` rounds most
of the trained update away (measured: ~75% of weight entries left bit-identical).

Seeds use constants.sample_seed(model_index, sample_index, item_index) where
item_index is the 0-based position of the problem within the split's fixed order
(dev/test order from --splits-file, benchmark order from the benchmark JSON file) --
this keeps seeds identical across the base model and every trained run, since
model_index is the shared backbone's index, not the adapter's.

CLI:
    python evaluation/evaluate.py --model Qwen/Qwen3-1.7B --split dev --samples 1 \
        --output artifacts/eval/base_dev.jsonl

    python evaluation/evaluate.py --model Qwen/Qwen3-1.7B \
        --adapter artifacts/train/answer_only/step_100 \
        --split test --samples 4 --output artifacts/eval/answer_only_step100_test.jsonl

    python evaluation/evaluate.py --model Qwen/Qwen3-1.7B --split aime24 --samples 4 \
        --output artifacts/eval/base_aime24.jsonl --limit 4   # smoke
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer  # noqa: E402

from opsd import benchmarks, constants, data, generation, math_grader, prompts  # noqa: E402
from opsd.constants import DEFAULT_SPLITS_FILE, MODEL_INDEX  # noqa: E402

BENCHMARK_SPLITS = ("aime24", "aime25", "hmmt25")


def normalize_question(q):
    return " ".join(q.lower().split())


def load_split_problems(split, splits_file=DEFAULT_SPLITS_FILE):
    """Return [{"problem_id", "question", "verified_answer"}, ...] for split in
    {"dev", "test", "aime24", "aime25", "hmmt25"}, in a fixed, stable order."""
    if split in ("dev", "test"):
        problems = data.load_problems()
        splits = json.load(open(splits_file))
        ids = splits[split]
        return [
            {
                "problem_id": pid,
                "question": problems[pid]["question"],
                "verified_answer": problems[pid]["verified_answer"],
            }
            for pid in ids
        ]
    elif split in BENCHMARK_SPLITS:
        return benchmarks.load_benchmark(split)
    else:
        raise ValueError(f"unknown split {split!r}")


def check_train_overlap(split_name, bench_problems, splits_file=DEFAULT_SPLITS_FILE):
    """Normalized exact-question overlap check against the train split: lowercase +
    collapse whitespace, then exact string match. Prints a summary and returns the
    list of matches.

    Before the splits exist (benchmark smoke runs) there is no train set to
    compare against; say so loudly and check against nothing rather than
    aborting the evaluation."""
    splits_path = splits_file
    if not os.path.exists(splits_path):
        print(f"[overlap check] WARNING: {splits_path} does not exist -- checking {split_name} against an EMPTY train set. This run's overlap report is meaningless; rerun after the splits stage.")
        return []

    problems = data.load_problems()
    train_ids = json.load(open(splits_path))["train"]
    train_norm_to_pid = {
        normalize_question(problems[pid]["question"]): pid for pid in train_ids if pid in problems
    }

    matches = []
    for p in bench_problems:
        norm = normalize_question(p["question"])
        if norm in train_norm_to_pid:
            matches.append(
                {"benchmark_problem_id": p["problem_id"], "train_problem_id": train_norm_to_pid[norm]}
            )

    print(f"[overlap check] {split_name}: {len(matches)}/{len(bench_problems)} exact-question matches against train split")
    for m in matches:
        print(f"  {m['benchmark_problem_id']} == {m['train_problem_id']}")
    return matches


def grade_text(text, verified_answer):
    pred = math_grader.extract_boxed(text)
    return math_grader.grade_answer(pred, verified_answer)


def add_budget_scores(record, tokenizer):
    """Budget-prefix correctness from one saved generation, no re-generation.
    `output_token_ids` is the main (<=16,384-token) generation; the full/rescued
    output is decoded text already saved by generate_records."""
    verified_answer = record["verified_answer"]
    token_ids = record["output_token_ids"]
    for budget, key in [(4096, "correct_at_4k"), (8192, "correct_at_8k"), (16384, "correct_at_16k")]:
        prefix_text = tokenizer.decode(token_ids[:budget], skip_special_tokens=True)
        record[key] = grade_text(prefix_text, verified_answer)

    full_text = record["rescued_output"] if record.get("rescued") else record["output"]
    record["correct_full"] = grade_text(full_text, verified_answer)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="backbone model path/name (chat template + served weights)")
    parser.add_argument("--adapter", default=None, help="LoRA checkpoint dir (<training runs>/<run>/step_<n>, see opsd.artifact_layout.train_runs); served live on top of --model, never merged")
    parser.add_argument("--split", required=True, choices=["dev", "test", "aime24", "aime25", "hmmt25"])
    parser.add_argument("--splits-file", default=DEFAULT_SPLITS_FILE,
                        help="train/dev/test id file (default: data/splits/qwen3_1p7b_splits.json)")
    parser.add_argument("--samples", type=int, required=True, help="samples per problem")
    parser.add_argument("--model-index", type=int, default=None, help="defaults to constants.MODEL_INDEX[--model]")
    parser.add_argument("--output", required=True, help="output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="limit number of problems (smoke tests only)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="lower it when another job shares the GPU")
    parser.add_argument("--student-thinking", dest="student_thinking", action="store_true",
                        default=True, help="enable_thinking for the evaluated (student) chat "
                        "template (default: on)")
    parser.add_argument("--no-student-thinking", dest="student_thinking", action="store_false")
    args = parser.parse_args()

    model_index = args.model_index if args.model_index is not None else MODEL_INDEX[args.model]
    serve_path = args.model

    problems = load_split_problems(args.split, args.splits_file)
    if args.limit is not None:
        problems = problems[: args.limit]

    overlap_matches = None
    if args.split in BENCHMARK_SPLITS:
        overlap_matches = check_train_overlap(args.split, problems, args.splits_file)

    tokenizer = AutoTokenizer.from_pretrained(serve_path)

    run_name = os.path.splitext(os.path.basename(args.output))[0]
    tasks = []
    for item_index, p in enumerate(problems):
        messages = prompts.student_messages(p["question"])
        for sample_index in range(args.samples):
            tasks.append(
                {
                    "problem_id": p["problem_id"],
                    "messages": messages,
                    "seed": constants.sample_seed(model_index, sample_index, item_index),
                    "sample_index": sample_index,
                    "condition": "no_pi",  # student eval is always no-reference; matches preflight.py's convention
                    "model_tag": run_name,
                    "verified_answer": p["verified_answer"],
                }
            )

    print(f"serving {serve_path}" + (f" + LoRA {args.adapter}" if args.adapter else "")
          + f" ({len(tasks)} generation tasks, {len(problems)} problems x {args.samples} samples)")
    llm = generation.load_llm(
        serve_path,
        max_model_len=constants.MODEL_MAX_LENGTH,
        gpu_memory_utilization=args.gpu_memory_utilization,
        lora=args.adapter is not None,
    )
    lora = generation.lora_request(args.adapter)
    records = generation.generate_records(llm, tokenizer, tasks, constants.MAIN_MAX_NEW_TOKENS,
                                           lora=lora, enable_thinking=args.student_thinking)

    for record in records:
        add_budget_scores(record, tokenizer)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    generation.save_jsonl(records, args.output)

    n = len(records)
    n_correct_full = sum(r["correct_full"] for r in records)
    n_correct_16k = sum(r["correct_at_16k"] for r in records)
    print(f"{args.split}: {n} generations -- correct_full={n_correct_full}/{n} correct_at_16k={n_correct_16k}/{n}")
    print(f"wrote {args.output}")

    if overlap_matches is not None:
        overlap_path = os.path.splitext(args.output)[0] + "_overlap.json"
        with open(overlap_path, "w") as f:
            json.dump(overlap_matches, f, indent=2)
        print(f"wrote overlap report to {overlap_path}")


if __name__ == "__main__":
    main()
    # vLLM 0.12's engine teardown can hang after all work is done, and a bare os._exit
    # orphans the EngineCore child, which then squats on the GPU -- so terminate children
    # first. All results are on disk by the end of main().
    import psutil
    for child in psutil.Process().children(recursive=True):
        child.terminate()
    psutil.wait_procs(psutil.Process().children(recursive=True), timeout=10)
    for child in psutil.Process().children(recursive=True):
        child.kill()
    sys.stdout.flush()
    os._exit(0)
