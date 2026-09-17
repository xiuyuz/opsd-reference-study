"""SmolLM3-3B evaluation runner: one invocation = one (configuration, checkpoint, shard) cell.

Modes:
  --mode student (default)  no-reference student template. --adapter-dir serves a trained
                            LoRA checkpoint live on top of the frozen backbone (vLLM
                            enable_lora + one LoRARequest per run, never merged); omit it to
                            evaluate the base model.
  --mode teacher            reference-conditioned teacher template on the frozen base model
                            (teacher-recoverability pass; thinking-on by convention).
                            --condition picks the view, one of the five with a reasoning body
                            (STRUCTURED_VIEWS); gist/summary/clean_solution are attached
                            through data.load_extra_pi for exactly the evaluated ids.
Problem lists (fixed order; item_index for the seed schedule is the ABSOLUTE position):
  --split dev|test          from --splits-file
  --split ids               from --ids-file, a JSON file whose --ids-key entry is the id list
                            (train-pool subsets, the full candidate pool, hard-candidate sets).
                            That list is pinned: it must be ascending-sorted and duplicate-free,
                            and --expected-ids asserts the size the cell was built for, so a
                            regenerated or reordered file cannot silently shift every seed.
--shard-start/--shard-end slice that order without changing item_index, so a cell sharded
across lanes reproduces the unsharded run's seeds.

Eval protocol otherwise the repo standard: engine at constants.MODEL_MAX_LENGTH (within
SmolLM3's native 65,536), main budget constants.MAIN_MAX_NEW_TOKENS, opsd.generation's
rescue pass to RESCUE_MAX_NEW_TOKENS_CAP within context, GEN_KWARGS sampling,
per-(problem, sample) seeds via constants.sample_seed(MODEL_INDEX["HuggingFaceTB/SmolLM3-3B"],
sample_index, item_index), budget-prefix correctness via evaluate.add_budget_scores.

Output files. --output is the full path of the cell's jsonl; the analyses read the SmolLM3 cells
under these opsd.artifact_layout directories and file patterns:
    SMOLLM3_BASE_EVAL               the frozen base in both evaluation modes
    SMOLLM3_2X2_EVAL                Answer Only and Full Trace against both training modes, seed 0
    SMOLLM3_SEED1_EVAL              Full Trace at seed 1, both training modes
    SMOLLM3_REFERENCE_FREE_EVAL     the reference-free, Answer Only and Full Trace students, seeds 0 to 2

  Every file is named <prefix>_<shard>_<run>_step<n>_test.jsonl, with the prefix chosen by the
  caller, so one directory can hold several studies without their shards colliding.
    SMOLLM3_SPLIT_POOL_EVAL         base pool samples (--split ids); SMOLLM3_SPLIT_POOL_TEACHER_EVAL for --mode teacher
where <i> numbers the shard and <run> is the training-run name.

Run under the vLLM env:

    python evaluation/smollm3_eval.py --gpu 0 --thinking on --split test --samples 4 \\
        --adapter-dir artifacts/smollm3/train/smollm3-3b_full_trace_thinking/step_25 \\
        --shard-start 0 --shard-end 96 \\
        --output artifacts/smollm3/eval/views_by_mode/ft_th_s0_smollm3-3b_full_trace_thinking_step25_test.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd import constants  # noqa: E402
from opsd.constants import ALL_CONDITIONS, EXTRA_CONDITIONS, SPLITS_FILES  # noqa: E402

DEFAULT_MODEL = "HuggingFaceTB/SmolLM3-3B"
DEFAULT_SPLITS_FILE = SPLITS_FILES[DEFAULT_MODEL]
# The five reference views that carry a reasoning body. --mode teacher asks the frozen
# backbone to re-derive a solution from the reference, and answer_only -- the bare final
# answer -- leaves it nothing to re-derive from, so the pass is defined on these five only.
STRUCTURED_VIEWS = [c for c in ALL_CONDITIONS if c != "answer_only"]


def load_cell_problems(split, splits_file, ids_file, ids_key, condition=None, expected_ids=None):
    """[{"problem_id","question","verified_answer","pi"}...] in the list's FIXED order
    (splits-file order for dev/test, ids-file order for ids). item_index for seeds is the
    absolute position in this list, so the order is load-bearing: a --split ids list is
    checked to be the pinned ascending-sorted one, and expected_ids (when given) to be the
    size the cell was built for. When condition is one of EXTRA_CONDITIONS its reference text
    is attached to problem["pi"] for exactly these ids."""
    from opsd import data

    problems = data.load_problems()
    if split in ("dev", "test"):
        ids = json.load(open(splits_file))[split]
    elif split == "ids":
        if not ids_file:
            raise ValueError("--split ids requires --ids-file")
        ids = json.load(open(ids_file))[ids_key]
        assert ids == sorted(ids), (
            f"{ids_file}'s {ids_key!r} list is not ascending-sorted -- the pinned ordering "
            "this cell's per-request seeds depend on has changed")
    else:
        raise ValueError(f"unknown split {split!r}")
    assert len(ids) == len(set(ids)), "duplicate ids in the problem list"
    assert expected_ids is None or len(ids) == expected_ids, (
        f"expected {expected_ids} ids in the problem list, got {len(ids)} -- the list changed "
        "since this cell was defined; stop rather than evaluate a different set of problems")
    missing = [pid for pid in ids if pid not in problems]
    assert not missing, f"{len(missing)} ids missing from data.load_problems(), e.g. {missing[:3]}"

    selected = [problems[pid] for pid in ids]
    if condition in EXTRA_CONDITIONS:
        extra = data.load_extra_pi(ids, [condition])
        for p in selected:
            p["pi"][condition] = extra[p["problem_id"]][condition]
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=None, help="single CUDA device id (required unless --dry-run)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-index", type=int, default=None,
                        help="seed-schedule index; default constants.MODEL_INDEX[--model]")
    parser.add_argument("--mode", choices=["student", "teacher"], default="student",
                        help="student: no-reference student template; teacher: reference-conditioned teacher template")
    parser.add_argument("--condition", default=None,
                        help=f"reference view for --mode teacher (one of {STRUCTURED_VIEWS}; "
                        "answer_only is excluded -- it has no reasoning body for the teacher "
                        "to re-derive a solution from)")
    parser.add_argument("--adapter-dir", default=None,
                        help="LoRA checkpoint dir (<training runs>/<run>/step_<n>, see opsd.artifact_layout.train_runs); served live "
                        "on top of --model via a single per-run LoRARequest, never merged. Omit "
                        "for the frozen base model.")
    parser.add_argument("--thinking", choices=["on", "off"], required=True)
    parser.add_argument("--split", choices=["dev", "test", "ids"], required=True)
    parser.add_argument("--splits-file", default=DEFAULT_SPLITS_FILE,
                        help="train/dev/test id file (default: data/splits/smollm3_3b_splits.json)")
    parser.add_argument("--ids-file", default=None, help="JSON file holding the id list for --split ids")
    parser.add_argument("--ids-key", default="ids", help="key of the id list inside --ids-file")
    parser.add_argument("--expected-ids", type=int, default=None,
                        help="pinned size of the --split ids list; the run refuses to start if "
                        "the file holds a different number of ids (3399 for the candidate pool, "
                        "988 for the hard-candidate set)")
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--shard-start", type=int, default=None,
                        help="absolute start index into the split's fixed problem order")
    parser.add_argument("--shard-end", type=int, default=None, help="absolute end index (exclusive)")
    parser.add_argument("--max-new-tokens", type=int, default=constants.MAIN_MAX_NEW_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=constants.MODEL_MAX_LENGTH)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None, help="limit problems (smoke only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the task plan and exit; never imports vLLM/torch")
    args = parser.parse_args()

    if args.mode == "teacher":
        if args.condition not in STRUCTURED_VIEWS:
            parser.error(f"--mode teacher requires --condition in {STRUCTURED_VIEWS} "
                         "(answer_only carries no reasoning body)")
        if args.thinking != "on":
            parser.error("teacher-utility cells are thinking-ON by convention")
        if args.adapter_dir is not None:
            parser.error("--mode teacher evaluates the frozen base model; drop --adapter-dir")
    if args.adapter_dir is not None and not args.dry_run and not os.path.isdir(args.adapter_dir):
        parser.error(f"--adapter-dir does not exist: {args.adapter_dir}")
    condition = args.condition if args.mode == "teacher" else "no_pi"

    model_index = (args.model_index if args.model_index is not None
                   else constants.MODEL_INDEX[args.model])

    all_problems = load_cell_problems(args.split, args.splits_file, args.ids_file, args.ids_key,
                                      condition=args.condition if args.mode == "teacher" else None,
                                      expected_ids=args.expected_ids)
    start = args.shard_start or 0
    end = args.shard_end if args.shard_end is not None else len(all_problems)
    shard = list(enumerate(all_problems))[start:end]  # (absolute item_index, problem)
    if args.limit is not None:
        shard = shard[: args.limit]

    from opsd import prompts

    run_name = os.path.splitext(os.path.basename(args.output))[0]
    tasks = []
    for item_index, p in shard:
        if args.mode == "student":
            messages = prompts.student_messages(p["question"])
        else:
            messages = prompts.teacher_messages(p["question"], p["pi"][condition])
        for sample_index in range(args.samples):
            tasks.append({
                "problem_id": p["problem_id"],
                "messages": messages,
                "seed": constants.sample_seed(model_index, sample_index, item_index),
                "sample_index": sample_index,
                "condition": condition,
                "model_tag": run_name,
                "verified_answer": p["verified_answer"],
            })

    plan = (f"cell: model={args.model} mode={args.mode} adapter={args.adapter_dir or '(base)'} "
            f"thinking={args.thinking} condition={condition} split={args.split}[{start}:{end}] "
            f"samples={args.samples} -> {len(tasks)} requests, output {args.output}")
    print(plan, flush=True)
    if args.dry_run:
        return

    if args.gpu is None:
        parser.error("--gpu is required for a real run")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from transformers import AutoTokenizer

    import evaluate  # add_budget_scores (imports opsd.generation -> vLLM; real-run path only)
    from opsd import generation

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = generation.load_llm(args.model, max_model_len=args.max_model_len,
                              gpu_memory_utilization=args.gpu_memory_utilization,
                              lora=args.adapter_dir is not None)
    lora = generation.lora_request(args.adapter_dir)
    records = generation.generate_records(llm, tokenizer, tasks, args.max_new_tokens,
                                          lora=lora, enable_thinking=(args.thinking == "on"))
    for record in records:
        record["mode"] = args.mode
        record["thinking"] = args.thinking
        record["adapter_dir"] = args.adapter_dir
        evaluate.add_budget_scores(record, tokenizer)

    generation.save_jsonl(records, args.output)
    n = len(records)
    n_full = sum(r["correct_full"] for r in records)
    n_16k = sum(r["correct_at_16k"] for r in records)
    n_cap = sum(r["hit_length_cap"] for r in records)
    print(f"{run_name}: {n} generations -- correct_full={n_full}/{n} "
          f"correct_at_16k={n_16k}/{n} capped={n_cap}", flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
    # Same teardown convention as evaluate.py (vLLM 0.12 EngineCore can hang/orphan after
    # work completes; results are already on disk).
    import psutil
    for child in psutil.Process().children(recursive=True):
        child.terminate()
    psutil.wait_procs(psutil.Process().children(recursive=True), timeout=10)
    for child in psutil.Process().children(recursive=True):
        child.kill()
    sys.stdout.flush()
    os._exit(0)
