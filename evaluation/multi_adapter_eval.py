"""Multi-adapter evaluation: one vLLM engine, ALL requested checkpoints served as separate
LoRA adapters at once, over any mix of the in-domain dev/test split and the external
benchmarks aime24/aime25/hmmt25.

Runs and checkpoints:
  --runs a,b + --steps n,m         adapters at <--adapter-root>/<run>/step_<n>
  --runs-config <json-or-path>     a JSON list of {"name", "adapter_root", "checkpoint_steps"}
                                   objects (checkpoint_steps optional, falls back to --steps);
                                   replaces --runs/--adapter-root entirely when given.
Missing checkpoint dirs are skipped with a message.

Splits + prompt style. --splits can mix dev/test (from --splits-file, this repo's own
prompts.STUDENT_TEMPLATE with --prompt-style local) with the external benchmarks
(--prompt-style official renders the same user message the official OPSD evaluator uses:
"{problem}\\n\\nPlease reason step by step, and put your final answer within \\boxed{}." --
no "Problem:" prefix, which differs from prompts.STUDENT_TEMPLATE).

--protocol switches the *sampling* parameters:
  local:    constants.GEN_KWARGS (temperature=1.0, top_p=0.95, top_k=20), budget
            constants.MAIN_MAX_NEW_TOKENS (16384) + the rescue pass up to
            RESCUE_MAX_NEW_TOKENS_CAP (32512) -- this repo's standard convention.
  official: temperature=1.0, top_p=0.95, top_k=-1 (disabled), min_p=0 -- the official
            README's reported settings -- budget defaults to the official 38912-token
            single-shot cap, but (unlike the literal official evaluator) a rescue pass is
            still applied on top for any generation that hits that cap. Pair it with
            --max-model-len 40960 and MODEL_MAX_LEN_OVERRIDE=40960 (see
            evaluation/run_external_eval.sh) so the per-request budget is computed against
            the same context the engine actually has.

--enable-thinking / --no-enable-thinking: chat-template enable_thinking for the eval
prompts (thinking-enabled vs direct-response evaluation).

--max-loras-per-engine (default 14) caps how many distinct adapters are ever concurrently
referenced in one generate() call: requested adapters beyond that are split into sequential
chunks processed on the SAME engine instance (one model load, reused across chunks). vLLM's
LoRA cache is keyed by lora_int_id and reusing an id for a different adapter path does NOT
force a reload, so ids are assigned once, globally, over the full adapter set.

Results are written once PER CHUNK (write_results()), immediately after that chunk's
generate + rescue returns, so a crash mid-run only loses the current chunk. One JSONL per
(run, checkpoint, split): <output-dir>/<run-name-prefix><run>_step<n>_<split>.jsonl.
--adapter-root and --output-dir accept a path or the name of an opsd.artifact_layout
constant (e.g. --adapter-root THINKING_TRAIN_RUNS --output-dir SIX_VIEW_DEV_TIER).

--dry-run prints the (run, checkpoint, split, n) plan and the adapter-chunking breakdown,
then exits -- before CUDA_VISIBLE_DEVICES is set, before tokenizer/vLLM are imported.

Run under the vLLM env; CUDA_VISIBLE_DEVICES is set internally from --gpu.

    python evaluation/multi_adapter_eval.py \\
        --gpu 1 --runs qwen3-1.7b_full_trace_direct --steps 25,50,75,100 \\
        --splits dev,test --prompt-style local --protocol local

    python evaluation/multi_adapter_eval.py \\
        --gpu 1 --runs-config runs.json \\
        --splits aime24,aime25,hmmt25 --prompt-style official --protocol official \\
        --samples-bench 12 --max-new-tokens 38912 --max-model-len 40960

    python evaluation/multi_adapter_eval.py --dry-run --runs-config runs.json --splits dev,test
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BENCH_SPLITS = ("aime24", "aime25", "hmmt25")
LOCAL_SPLITS = ("dev", "test")

# Largest co-resident adapter count validated in one generate() call (~4.05s/request
# main-pass throughput at 14 adapters). Not a vLLM-enforced ceiling (vllm/config/lora.py's
# max_loras is just `ge=1`); a deliberately conservative default. Override with
# --max-loras-per-engine.
DEFAULT_MAX_LORAS_PER_ENGINE = 14


def official_style_messages(question):
    # Verbatim: the official OPSD evaluator's user message (eval/evaluate_math.py).
    content = f"{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    return [{"role": "user", "content": content}]


def parse_runs_config(raw):
    """raw: a JSON list (inline string) or a path to a JSON file containing one. Each
    entry: {"name": str, "adapter_root": str, "checkpoint_steps": [int, ...] (optional,
    falls back to --steps)}. Returns the parsed list of dicts."""
    text = raw
    if os.path.isfile(raw):
        text = open(raw).read()
    config = json.loads(text)
    if not isinstance(config, list) or not config:
        raise ValueError("--runs-config must be a non-empty JSON list of "
                          "{name, adapter_root, checkpoint_steps} objects")
    for entry in config:
        if "name" not in entry or "adapter_root" not in entry:
            raise ValueError(f"--runs-config entry missing name/adapter_root: {entry}")
    return config


def resolve_runs(args, default_steps):
    """Returns (runs, run_roots, steps_by_run, using_runs_config). Without --runs-config,
    --runs are looked up under --adapter-root/<run>, with the same global step list
    applied to every run."""
    if args.runs_config:
        config = parse_runs_config(args.runs_config)
        run_roots = {e["name"]: e["adapter_root"] for e in config}
        steps_by_run = {
            e["name"]: (e["checkpoint_steps"] if e.get("checkpoint_steps") else default_steps)
            for e in config
        }
        return list(run_roots), run_roots, steps_by_run, True

    runs = [a.strip() for a in args.runs.split(",") if a.strip()]
    run_roots = {run: os.path.join(args.adapter_root, run) for run in runs}
    return runs, run_roots, {run: default_steps for run in runs}, False


def discover_adapters(runs, run_roots, steps_by_run):
    """Returns (adapters: {(run, step): dir, ...in run-then-step order}, missing:
    [(run, step, dir), ...]) -- missing dirs are silently skipped, caller prints/logs."""
    adapters = {}
    missing = []
    for run in runs:
        root = run_roots[run]
        for step in steps_by_run[run]:
            d = os.path.join(root, f"step_{step}")
            if os.path.isdir(d):
                adapters[(run, step)] = d
            else:
                missing.append((run, step, d))
    return adapters, missing


def chunk_adapters(adapters, max_loras_per_engine):
    """Split the GLOBAL adapters dict into sequential chunks of at most
    max_loras_per_engine (run, step) pairs each, preserving order. Everything in one
    chunk when it fits."""
    items = list(adapters.items())
    size = max(1, max_loras_per_engine)
    return [dict(items[i:i + size]) for i in range(0, len(items), size)]


def load_split_problems(split, splits_file):
    """CPU-only split loading, parameterized on splits_file. Only imports
    opsd.benchmarks/opsd.data -- deliberately NOT evaluate/opsd.generation, which pull
    in vLLM at module import time -- so --dry-run never needs vLLM installed."""
    if split in LOCAL_SPLITS:
        from opsd import data

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
    elif split in BENCH_SPLITS:
        from opsd import benchmarks

        return benchmarks.load_benchmark(split)
    else:
        raise ValueError(f"unknown split {split!r}")


def compute_plan(args, adapters, split_problems):
    """(run, checkpoint, split, n) plan -- CPU-only, no tokenizer/vLLM needed (mirrors
    build_tasks()'s task count exactly, without materializing every task dict)."""
    plan = []
    for split, problems in split_problems.items():
        samples = {"dev": args.samples_dev, "test": args.samples_test}.get(split, args.samples_bench)
        n = len(problems) * samples
        for (run, step) in adapters:
            plan.append((run, step, split, n))
    return plan


def print_dry_run(args, using_runs_config, adapters, missing, chunks, plan):
    print("=== multi_adapter_eval: DRY RUN (no tokenizer/vLLM import, no CUDA touched) ===")
    print(f"model: {args.model}")
    print(f"splits_file: {args.splits_file}")
    print(f"prompt_style: {args.prompt_style}   protocol: {args.protocol}")
    print("runs source: " + ("--runs-config" if using_runs_config else "--runs under --adapter-root"))
    print(f"requested (run, checkpoint) pairs: {len(adapters) + len(missing)}   "
          f"found on disk: {len(adapters)}   missing: {len(missing)}")
    for run, step, d in missing[:20]:
        print(f"  missing: {run}/step_{step} -> {d}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")
    print(f"max_loras_per_engine: {args.max_loras_per_engine} -> {len(chunks)} chunk(s) "
          f"(1 model load, {len(chunks)} generate()-call/incremental-write point(s))")
    for i, chunk in enumerate(chunks, 1):
        members = ", ".join(f"{a}/step{s}" for a, s in chunk)
        print(f"  chunk {i}/{len(chunks)} ({len(chunk)} adapters): {members}")
    print()
    print("plan (run, checkpoint, split, n):")
    total = 0
    for run, step, split, n in plan:
        print(f"  {run}\t{step}\t{split}\t{n}")
        total += n
    print(f"TOTAL: {total} generation requests across {len(plan)} (run,checkpoint,split) cells, "
          f"{len(adapters)} distinct adapters, {len(chunks)} chunk(s)")
    print("=== END DRY RUN ===")


def build_lora_requests(adapters):
    """One LoRARequest per (run, step), with a GLOBAL, PERMANENT lora_int_id for the
    whole run -- call once over the FULL adapter set, even when generate() calls are
    chunked afterward (see chunk_adapters()). vLLM's LoRA cache is keyed by
    lora_int_id, and reusing an id for a different underlying adapter path does NOT
    force a reload, so ids must not be renumbered per chunk."""
    from vllm.lora.request import LoRARequest

    return {
        key: LoRARequest(f"{run}_step{step}", idx + 1, adapter_dir)
        for idx, (key, adapter_dir) in enumerate(adapters.items())
        for run, step in [key]
    }


def build_tasks(args, adapters, split_problems):
    """adapters: {(run, step): adapter_dir} -- the FULL global set or a single engine
    chunk's subset. split_problems: {split: [problem, ...]} (already --limit-sliced),
    built once in main() and shared across every chunk so problem/benchmark files and
    prompt rendering aren't redone per chunk. Returns a flat task list (sweep-local
    "run"/"step"/"split" keys, not written to the output record)."""
    from opsd import constants, prompts

    model_index = constants.MODEL_INDEX[args.model]
    tasks = []
    for split, problems in split_problems.items():
        samples = {"dev": args.samples_dev, "test": args.samples_test}.get(split, args.samples_bench)
        for (run, step), adapter_dir in adapters.items():
            run_name = f"{args.run_name_prefix}{run}_step{step}_{split}"
            for item_index, p in enumerate(problems):
                messages = (
                    official_style_messages(p["question"])
                    if args.prompt_style == "official"
                    else prompts.student_messages(p["question"])
                )
                for sample_index in range(samples):
                    tasks.append(
                        {
                            "problem_id": p["problem_id"],
                            "messages": messages,
                            "seed": constants.sample_seed(model_index, sample_index, item_index),
                            "sample_index": sample_index,
                            "condition": "no_pi",
                            "model_tag": run_name,
                            "verified_answer": p["verified_answer"],
                            "run": run,
                            "step": step,
                            "split": split,
                        }
                    )
    return tasks


def sampling_profile(args):
    if args.protocol == "official":
        return dict(temperature=1.0, top_p=0.95, top_k=-1, repetition_penalty=1.0)
    from opsd import constants

    g = constants.GEN_KWARGS
    return dict(temperature=g["temperature"], top_p=g["top_p"], top_k=g["top_k"],
                repetition_penalty=g["repetition_penalty"])


def generate_and_rescue(llm, tokenizer, tasks, lora_by_key, args):
    from opsd import constants, generation, math_grader
    from vllm import SamplingParams, TokensPrompt

    profile = sampling_profile(args)
    max_new_tokens = args.max_new_tokens
    prompt_ids, budgets = generation.prepare_prompts(
        tokenizer, tasks, max_new_tokens, enable_thinking=getattr(args, "enable_thinking", True))
    lora_list = [lora_by_key[(t["run"], t["step"])] for t in tasks]

    def sp(seed, max_tokens):
        return SamplingParams(seed=seed, max_tokens=max_tokens, **profile)

    print(f"generating {len(tasks)} requests (main pass, max_new_tokens={max_new_tokens})...")
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids],
        sampling_params=[sp(t["seed"], b) for t, b in zip(tasks, budgets)],
        lora_request=lora_list,
    )

    records = []
    for task, ids, budget, output in zip(tasks, prompt_ids, budgets, outputs):
        completion = output.outputs[0]
        answer = math_grader.extract_boxed(completion.text)
        records.append(
            {
                "problem_id": task["problem_id"],
                "model": task["model_tag"],
                "condition": task["condition"],
                "sample_index": task["sample_index"],
                "seed": task["seed"],
                "verified_answer": task["verified_answer"],
                "input_tokens": len(ids),
                "generated_tokens": len(completion.token_ids),
                "max_new_tokens": budget,
                "finish_reason": completion.finish_reason,
                "hit_length_cap": completion.finish_reason == "length",
                "rescued": False,
                "answer_extracted": answer,
                "correct": math_grader.grade_answer(answer, task["verified_answer"]),
                "output": completion.text,
                "output_token_ids": list(completion.token_ids),
                "rescued_max_new_tokens": None,
                "rescued_generated_tokens": None,
                "rescued_finish_reason": None,
                "rescued_answer_extracted": None,
                "rescued_correct": None,
                "rescued_output": None,
                "rescued_output_token_ids": None,
            }
        )

    # Rescue pass: same rule as generation.rescue_length_terminated, collected across every
    # (run, step, split) task in this chunk.
    todo = []
    for i, record in enumerate(records):
        available = constants.MODEL_MAX_LENGTH - record["input_tokens"] - constants.CONTEXT_SAFETY_MARGIN
        budget = min(available, args.rescue_cap)
        if record["hit_length_cap"] and budget > record["max_new_tokens"]:
            todo.append((i, budget))
    if todo:
        print(f"rescue pass: resubmitting {len(todo)} of {len(records)} generations")
        rescue_outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids[i]) for i, _ in todo],
            sampling_params=[sp(records[i]["seed"], b) for i, b in todo],
            lora_request=[lora_list[i] for i, _ in todo],
        )
        for (i, budget), output in zip(todo, rescue_outputs):
            completion = output.outputs[0]
            answer = math_grader.extract_boxed(completion.text)
            record = records[i]
            record["rescued"] = True
            record["rescued_max_new_tokens"] = budget
            record["rescued_generated_tokens"] = len(completion.token_ids)
            record["rescued_finish_reason"] = completion.finish_reason
            record["rescued_answer_extracted"] = answer
            record["rescued_correct"] = math_grader.grade_answer(answer, record["verified_answer"])
            record["rescued_output"] = completion.text
            record["rescued_output_token_ids"] = list(completion.token_ids)

    return records


def write_results(output_dir, adapters, tasks, records, run_name_prefix):
    """Write one JSONL per (run, checkpoint, split) present in tasks/records. Called
    once per adapter-chunk (see main()'s loop) rather than once at the very end of the
    whole invocation -- a crash mid-run only risks the CURRENT chunk's generate()
    call, not every chunk already written."""
    from opsd import generation

    os.makedirs(output_dir, exist_ok=True)
    splits_in_chunk = {t["split"] for t in tasks}
    for (run, step) in adapters:
        for split in splits_in_chunk:
            group = [r for t, r in zip(tasks, records)
                     if t["run"] == run and t["step"] == step and t["split"] == split]
            if not group:
                continue
            run_name = f"{run_name_prefix}{run}_step{step}_{split}"
            out_path = os.path.join(output_dir, f"{run_name}.jsonl")
            generation.save_jsonl(group, out_path)
            n = len(group)
            n_correct_full = sum(r["correct_full"] for r in group)
            n_correct_budget = sum(r["correct_at_16k"] for r in group)
            print(f"  {run}/step{step}/{split}: n={n} correct_full={n_correct_full}/{n} "
                  f"({n_correct_full / n:.4f})  correct_at_16k={n_correct_budget}/{n}  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=None, help="single CUDA device id, e.g. 1 (required unless --dry-run)")
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true",
                        default=True, help="chat-template enable_thinking for the eval "
                        "prompts (default: on)")
    parser.add_argument("--no-enable-thinking", dest="enable_thinking", action="store_false",
                        help="direct-response (mode-matched) evaluation")
    parser.add_argument("--runs", default=None,
                        help="comma-separated run names, each at --adapter-root/<run>/step_<n>")
    parser.add_argument("--adapter-root", default=None,
                        help="root directory holding <run>/step_<n>/ checkpoint dirs for --runs; a path "
                        "or an artifact_layout constant (default: DIRECT_TRAIN_RUNS)")
    parser.add_argument("--steps", default="25,50,75,100,20,15,10,5,2,1,0")
    parser.add_argument("--runs-config", default=None,
                         help="JSON list of {name, adapter_root, checkpoint_steps} (inline JSON or a "
                         "path to a JSON file) -- replaces --runs/--adapter-root entirely when given. "
                         "checkpoint_steps is optional per entry (falls back to --steps).")
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--splits-file", default=None,
                         help="path to the split JSON ({'train'/'dev'/'test': [problem_id, ...], ...}). "
                         "Default: the release split for --model (data/splits/qwen3_1p7b_splits.json "
                         "or data/splits/smollm3_3b_splits.json).")
    parser.add_argument("--prompt-style", choices=["local", "official"], default="local")
    parser.add_argument("--protocol", choices=["local", "official"], default="local")
    parser.add_argument("--model", default=None, help="default: constants.PRIMARY_MODEL")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="default: constants.MAIN_MAX_NEW_TOKENS (local) or 38912 (official)")
    parser.add_argument("--rescue-cap", type=int, default=None,
                        help="default: constants.RESCUE_MAX_NEW_TOKENS_CAP")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="default: constants.MODEL_MAX_LENGTH (32768, or MODEL_MAX_LEN_OVERRIDE); "
                        "pass 40960 for exact official thinking-mode context parity")
    parser.add_argument("--samples-dev", type=int, default=1)
    parser.add_argument("--samples-test", type=int, default=4)
    parser.add_argument("--samples-bench", type=int, default=12)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--limit", type=int, default=None, help="limit problems per split (smoke test)")
    parser.add_argument("--output-dir", default=None,
                        help="a path or an artifact_layout constant (default: MULTI_ADAPTER_EVAL)")
    parser.add_argument("--max-loras-per-engine", type=int, default=DEFAULT_MAX_LORAS_PER_ENGINE,
                        help="max distinct LoRA adapters referenced in one generate() call (no vLLM-"
                        f"enforced ceiling, see module docstring; default {DEFAULT_MAX_LORAS_PER_ENGINE}). "
                        "Requested adapters beyond this are chunked into sequential generate() "
                        "calls on the SAME engine -- also the incremental-write unit (see write_results()).")
    parser.add_argument("--run-name-prefix", default=None,
                        help="prefix for output run names/files (<prefix><run>_step<n>_<split>.jsonl). "
                        "Default: '' (empty).")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the (run, checkpoint, split, n) plan and adapter-chunking "
                        "breakdown, then exit -- no tokenizer/vLLM import, no CUDA_VISIBLE_DEVICES set.")
    args = parser.parse_args()

    if not args.dry_run and not args.gpu:
        parser.error("--gpu is required unless --dry-run is set")
    if not args.runs and not args.runs_config:
        parser.error("pass --runs (with --steps) or --runs-config")

    from opsd import artifact_layout, constants  # CPU-safe (no vLLM/torch import) -- needed even in --dry-run

    if args.model is None:
        args.model = constants.PRIMARY_MODEL
    if args.splits_file is None:
        args.splits_file = constants.SPLITS_FILES.get(args.model, constants.DEFAULT_SPLITS_FILE)
    args.adapter_root = artifact_layout.resolve(args.adapter_root or artifact_layout.DIRECT_TRAIN_RUNS)
    args.output_dir = artifact_layout.resolve(args.output_dir or artifact_layout.MULTI_ADAPTER_EVAL)

    default_steps = [int(s) for s in args.steps.split(",") if s.strip()]
    runs, run_roots, steps_by_run, using_runs_config = resolve_runs(args, default_steps)
    if args.run_name_prefix is None:
        args.run_name_prefix = ""

    adapters, missing = discover_adapters(runs, run_roots, steps_by_run)
    for run, step, d in missing:
        print(f"skip {run}/step_{step}: {d} does not exist")
    if not adapters:
        raise FileNotFoundError("no checkpoint dirs found for the given --runs/--steps (or --runs-config)")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    split_problems = {}
    for split in splits:
        problems = load_split_problems(split, args.splits_file)
        if args.limit is not None:
            problems = problems[: args.limit]
        split_problems[split] = problems

    chunks = chunk_adapters(adapters, args.max_loras_per_engine)
    plan = compute_plan(args, adapters, split_problems)

    if args.dry_run:
        print_dry_run(args, using_runs_config, adapters, missing, chunks, plan)
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.max_new_tokens is None:
        args.max_new_tokens = 38912 if args.protocol == "official" else constants.MAIN_MAX_NEW_TOKENS
    if args.rescue_cap is None:
        args.rescue_cap = constants.RESCUE_MAX_NEW_TOKENS_CAP
    max_model_len = args.max_model_len or constants.MODEL_MAX_LENGTH

    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    lora_by_key = build_lora_requests(adapters)

    print(f"serving {args.model} + {len(adapters)} LoRA adapters ({list(adapters)}) in one engine, "
          f"{len(chunks)} chunk(s) of <= {args.max_loras_per_engine} (max_loras_per_engine), "
          f"{len(plan)} (run,checkpoint,split) cells / {sum(n for *_, n in plan)} total requests, "
          f"prompt_style={args.prompt_style}, protocol={args.protocol}")

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=min(len(adapters), args.max_loras_per_engine),
        max_cpu_loras=len(adapters),
    )

    import evaluate

    print()
    for i, chunk in enumerate(chunks, 1):
        print(f"== chunk {i}/{len(chunks)}: {list(chunk)} ==")
        tasks = build_tasks(args, chunk, split_problems)
        records = generate_and_rescue(llm, tokenizer, tasks, lora_by_key, args)
        for record in records:
            evaluate.add_budget_scores(record, tokenizer)
        write_results(args.output_dir, chunk, tasks, records, args.run_name_prefix)

    print("\ndone.")


if __name__ == "__main__":
    main()
    # vLLM 0.12's EngineCore teardown can hang (same guard as evaluate.py). Reap children
    # before exit.
    import psutil

    for child in psutil.Process().children(recursive=True):
        child.terminate()
    psutil.wait_procs(psutil.Process().children(recursive=True), timeout=10)
    for child in psutil.Process().children(recursive=True):
        child.kill()
    sys.stdout.flush()
    os._exit(0)
