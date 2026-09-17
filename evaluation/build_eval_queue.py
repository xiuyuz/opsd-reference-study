#!/usr/bin/env python3
"""Build a file-backed evaluation queue for evaluation/eval_queue.py: one job per
(run, checkpoint step, 16-problem shard), in thinking or direct-response mode.

Shards. The frozen test split is cut into consecutive 16-problem shard files
(<shards>/test_b000.json, test_b001.json, ... each {"train": [], "dev": [], "test": [16 ids]}).
multi_adapter_eval.build_tasks() seeds every request with
sample_seed(model_index, sample_index, item_index) where item_index is the position of the
problem WITHIN THE JOB'S OWN SHARD, so the shards are what fix the per-(problem, sample)
generation seeds: regenerate a cell with the same shards and you get the same seeds; change
the batch shape and you do not. Pass --splits-file to write the shards when the directory
does not hold them yet (they are never overwritten).

Job schema (one JSON per job, consumed by eval_queue.py):
    job_id, priority, exp, cell, tier, run, enable_thinking, adapter_dir,
    runs_config: [{"name": run, "adapter_root": <root>, "checkpoint_steps": [step]}],
    splits_file: <shard>, splits: "test", samples_dev, samples_test, output_dir,
    max_loras_per_engine, prompt_style: "local", protocol: "local", run_name_prefix,
    gpu_memory_utilization, est_requests, expected_output, model
File name: <priority>_<ordinal:02d>_<cell>_b<shard:03d>.json -- plain lexicographic order is
the processing order, and the cell name (no underscores) is the third "_"-separated field.
Output rows land in <output-dir>/<prefix>_b<shard:03d>_<run>_step<n>_test.jsonl; a job whose
expected output already exists is skipped (a cell is never regenerated).

Directory flags accept either a path or the name of an opsd.artifact_layout constant. The
Each study keeps its rows in its own directory, under its own file prefix, so shards from
different studies never collide. These are the pairs the paper used:

    study                                             --output-dir                   --prefix
    the six views, thinking-enabled evaluation        SIX_VIEW_TEST_TIER             thinking (default, --mode thinking)
    the frozen base (evaluate a run's step_0)         BASE_TEST_TIER                 base
    teacher controls and interventions                CONTROL_EVAL_THINKING          controls
    any student, direct-response evaluation           CONTROL_EVAL_DIRECT            direct   (default, --mode direct)
    the replicate-seed students                       SEED_EVAL_TEST_TIER            seeds
    both training modes at their common checkpoints   COMMON_CKPT_TEST_TIER          common
    the loss-window runs at those checkpoints         LOSS_WINDOW_COMMON_CKPT_TIER   loss_window

    python evaluation/build_eval_queue.py \\
        --runs qwen3-1.7b_answer_only_direct,qwen3-1.7b_full_trace_direct \\
        --steps 50,100 --mode direct --build
    python evaluation/build_eval_queue.py --runs <seed runs> --steps 100 --mode thinking \\
        --output-dir SEED_EVAL_TEST_TIER --prefix seeds --build
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd.artifact_layout import (  # noqa: E402
    CONTROL_EVAL_DIRECT,
    DIRECT_TRAIN_RUNS,
    EVAL_QUEUE,
    EVAL_SHARDS,
    SIX_VIEW_TEST_TIER,
    path as artifact_path,
    resolve,
)
from opsd.constants import DEFAULT_SPLITS_FILE, PRIMARY_MODEL, SPLITS_FILES  # noqa: E402


def write_shards(splits_file, shard_dir, split, shard_size):
    """Cut splits_file[split] into consecutive shard_size-problem shard files; existing shard
    files are left untouched. Returns the sorted shard paths."""
    ids = json.load(open(splits_file))[split]
    os.makedirs(shard_dir, exist_ok=True)
    paths = []
    for bi, start in enumerate(range(0, len(ids), shard_size)):
        p = os.path.join(shard_dir, f"{split}_b{bi:03d}.json")
        if not os.path.exists(p):
            with open(p, "w") as f:
                json.dump({"train": [], "dev": [], "test": [], split: ids[start:start + shard_size]}, f)
        paths.append(p)
    return paths


def shard_paths(shard_dir, split):
    if not os.path.isdir(shard_dir):
        return []
    names = sorted(n for n in os.listdir(shard_dir) if n.startswith(f"{split}_b") and n.endswith(".json"))
    return [os.path.join(shard_dir, n) for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated run names (directories holding step_<n>/)")
    ap.add_argument("--adapter-root", default=artifact_path(DIRECT_TRAIN_RUNS),
                    help="root under which <run>/step_<n>/ lives; a path or an artifact_layout constant "
                         "(default DIRECT_TRAIN_RUNS; THINKING_TRAIN_RUNS / SMOLLM3_TRAIN_RUNS for the others)")
    ap.add_argument("--adapter-roots", default=None,
                    help="comma-separated per-run roots (same length/order as --runs); overrides --adapter-root")
    ap.add_argument("--steps", required=True, help="comma-separated checkpoint steps, e.g. 50,100")
    ap.add_argument("--mode", choices=["thinking", "direct"], required=True,
                    help="chat-template enable_thinking on (thinking) or off (direct)")
    ap.add_argument("--shards", default=artifact_path(EVAL_SHARDS),
                    help="directory of <split>_b###.json shard files (default EVAL_SHARDS)")
    ap.add_argument("--splits-file", default=None,
                    help="frozen split json used to write the shards when --shards holds none "
                         "(default: the release split for --model, data/splits/qwen3_1p7b_splits.json "
                         "or data/splits/smollm3_3b_splits.json)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--shard-size", type=int, default=16)
    ap.add_argument("--samples", type=int, default=4, help="samples per problem")
    ap.add_argument("--model", default=PRIMARY_MODEL)
    ap.add_argument("--output-dir", default=None,
                    help="where the eval rows go: a path or an artifact_layout constant (default "
                         "SIX_VIEW_TEST_TIER for --mode thinking, CONTROL_EVAL_DIRECT for --mode direct)")
    ap.add_argument("--queue-dir", default=artifact_path(EVAL_QUEUE),
                    help="pending-job directory (default EVAL_QUEUE)")
    ap.add_argument("--prefix", default=None,
                    help="file prefix the analyses glob for this tier (default 'test' for thinking, "
                         "'direct' for direct; see the module docstring)")
    ap.add_argument("--priority", default="A", help="job file-name prefix that orders the queue")
    ap.add_argument("--exp", default="", help="free-form experiment label stored on every job")
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--max-loras-per-engine", type=int, default=14)
    ap.add_argument("--allow-missing", action="store_true",
                    help="skip runs whose adapter is missing instead of refusing to build")
    ap.add_argument("--build", action="store_true", help="write the shard + job files (default: plan only)")
    a = ap.parse_args()

    thinking = a.mode == "thinking"
    if a.splits_file is None:
        a.splits_file = SPLITS_FILES.get(a.model, DEFAULT_SPLITS_FILE)
    prefix = a.prefix or ("thinking" if thinking else "direct")
    out_dir = resolve(a.output_dir or (SIX_VIEW_TEST_TIER if thinking else CONTROL_EVAL_DIRECT))
    a.shards, a.queue_dir, a.adapter_root = resolve(a.shards), resolve(a.queue_dir), resolve(a.adapter_root)
    runs = [x.strip() for x in a.runs.split(",") if x.strip()]
    steps = [int(s) for s in a.steps.split(",") if s.strip()]
    if a.adapter_roots:
        roots = [resolve(x.strip()) for x in a.adapter_roots.split(",")]
        if len(roots) != len(runs):
            sys.exit("--adapter-roots must list one root per --runs entry")
        roots = dict(zip(runs, roots))
    else:
        roots = {run: os.path.join(a.adapter_root, run) for run in runs}

    shards = shard_paths(a.shards, a.split)
    if not shards:
        if not os.path.exists(a.splits_file):
            sys.exit(f"no {a.split}_b*.json shards in {a.shards} and no split file at {a.splits_file}")
        if a.build:
            shards = write_shards(a.splits_file, a.shards, a.split, a.shard_size)
        else:
            ids = json.load(open(a.splits_file))[a.split]
            shards = [os.path.join(a.shards, f"{a.split}_b{bi:03d}.json")
                      for bi in range(0, (len(ids) + a.shard_size - 1) // a.shard_size)]
            print(f"(plan) would write {len(shards)} shard files of {a.shard_size} problems into {a.shards}")
    sizes = ([len(json.load(open(s))[a.split]) for s in shards] if all(os.path.exists(s) for s in shards)
             else [a.shard_size] * len(shards))

    cells = [(run, step) for run in runs for step in steps]
    missing = sorted({f"{run}/step_{step}" for run, step in cells
                      if not os.path.exists(os.path.join(roots[run], f"step_{step}", "adapter_model.safetensors"))})
    print(f"{len(cells)} cells x {len(shards)} shards; missing adapters: {missing or 'none'}")
    if missing and not a.allow_missing:
        sys.exit(1)
    if not a.build:
        return
    for sub in ("", "claimed", "done", "failed"):
        os.makedirs(os.path.join(a.queue_dir, sub), exist_ok=True)
    n = 0
    for ordinal, (run, step) in enumerate(cells, 1):
        if f"{run}/step_{step}" in missing:
            continue
        cell = f"{run.replace('_', '-')}-step{step}-{'th' if thinking else 'dr'}"
        for bi, shard in enumerate(shards):
            expected = os.path.join(out_dir, f"{prefix}_b{bi:03d}_{run}_step{step}_{a.split}.jsonl")
            if os.path.exists(expected):
                continue   # never regenerate an existing cell
            job = {"job_id": f"{prefix}_{cell}_b{bi:03d}", "priority": a.priority, "exp": a.exp, "cell": cell,
                   "tier": a.split, "run": run, "enable_thinking": thinking,
                   "adapter_dir": os.path.join(roots[run], f"step_{step}"),
                   "runs_config": [{"name": run, "adapter_root": roots[run], "checkpoint_steps": [step]}],
                   "splits_file": shard, "splits": a.split, "samples_dev": a.samples, "samples_test": a.samples,
                   "output_dir": out_dir, "max_loras_per_engine": a.max_loras_per_engine,
                   "prompt_style": "local", "protocol": "local", "run_name_prefix": f"{prefix}_b{bi:03d}_",
                   "gpu_memory_utilization": a.gpu_mem, "est_requests": sizes[bi] * a.samples,
                   "expected_output": expected, "model": a.model}
            with open(os.path.join(a.queue_dir, f"{a.priority}_{ordinal:02d}_{cell}_b{bi:03d}.json"), "w") as f:
                json.dump(job, f, indent=2)
            n += 1
    print(f"wrote {n} jobs to {a.queue_dir}")


if __name__ == "__main__":
    main()
