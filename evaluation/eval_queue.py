#!/usr/bin/env python3
"""File-backed evaluation queue runner: ONE vLLM engine start per CELL, many 16-problem
batch jobs served from it.

A queue directory holds one JSON job per (run, checkpoint, shard) -- see
evaluation/build_eval_queue.py for the schema. Jobs are grouped into cells (the third
"_"-separated field of the job file name); every batch of a cell shares the same adapter
and eval mode, so the engine is built once per cell and reused across its batches.

PROTOCOL: each batch is still built from its OWN 16-problem shard file and still goes into
its own single generate() call, so
  * per-request seed = sample_seed(model_index, sample_index, item_index) with item_index
    running 0..15 inside that shard -- the shards are what fix the seeds, which is why the
    batch shape must not change once a tier has been generated;
  * rendered prompts, sampling params, 16,384-token budget + rescue, and the output jsonl
    (one file per batch) are all produced by the SAME multi_adapter_eval.py functions
    (build_tasks / generate_and_rescue / write_results).
The ONLY difference from running the batches one by one is that the LLM object is
constructed once and reused.

Claiming is an atomic os.rename into <queue>/claimed/lane<gpu>/, so several lanes (GPUs) can
share one queue. Finished jobs move to <queue>/done/ or <queue>/failed/; a marker file
<markers>/<job_id>.DONE|ERROR and a line in the timings CSV are written per job. Each batch
gets JOB_TIMEOUT seconds (signal.alarm); a batch that trips its alarm aborts the WHOLE cell
process (rather than continuing on a possibly-wedged engine) and the remaining claims are
released back to the queue so another lane can take them.

Environment: QUEUE, MARKERS, TIMINGS (defaults: opsd.artifact_layout EVAL_QUEUE, EVAL_LANES/markers,
EVAL_LANES/timings.csv), JOB_TIMEOUT (14400 s). Job paths are resolved relative to the repository
root (the runner chdirs there) unless absolute.

    # verify only, CPU, no vLLM import, no GPU:
    python evaluation/eval_queue.py --verify-tasks --cell <cell>
    # run one cell on a GPU:
    python evaluation/eval_queue.py --gpu 1 --cell <cell>
    # run whichever highest-priority cell still has pending jobs:
    python evaluation/eval_queue.py --gpu 1 --next-cell
"""
import argparse, csv, json, os, signal, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from opsd.artifact_layout import EVAL_LANES, EVAL_QUEUE, path as artifact_path  # noqa: E402

QUEUE = os.environ.get("QUEUE", artifact_path(EVAL_QUEUE))
MARKERS = os.environ.get("MARKERS", artifact_path(EVAL_LANES, "markers"))
TIMINGS = os.environ.get("TIMINGS", artifact_path(EVAL_LANES, "timings.csv"))
# Each batch gets JOB_TIMEOUT seconds of its own (signal.alarm around that batch), and the
# whole cell process gets JOB_TIMEOUT x n_batches.
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "14400"))


class BatchTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise BatchTimeout(f"batch exceeded JOB_TIMEOUT={JOB_TIMEOUT}s")


def job_files(queue):
    return sorted(f for f in os.listdir(queue) if f.endswith(".json"))


def cell_of(name):
    return name.split("_")[2]


def next_cell(queue):
    files = job_files(queue)
    return cell_of(files[0]) if files else None


def claim(queue, name, lane):
    cdir = os.path.join(queue, "claimed", f"lane{lane}")
    os.makedirs(cdir, exist_ok=True)
    src, dst = os.path.join(queue, name), os.path.join(cdir, name)
    try:
        os.rename(src, dst)
        return dst
    except (FileNotFoundError, OSError):
        return None


def release_claims(pairs, queue):
    """Put unrun claims back in the pending queue so another lane can take them."""
    for path, job in pairs:
        try:
            os.rename(path, os.path.join(queue, os.path.basename(path)))
            print(f"  released claim {os.path.basename(path)}")
        except OSError as e:
            print(f"  WARNING: could not release {path}: {e}")


def make_args(job, gpu):
    """The exact Namespace multi_adapter_eval.main() would have built."""
    from opsd import constants
    return argparse.Namespace(
        gpu=str(gpu), runs=None, adapter_root=None, steps="", runs_config=json.dumps(job["runs_config"]),
        splits=job["splits"], splits_file=job["splits_file"],
        prompt_style=job["prompt_style"], protocol=job["protocol"],
        model=job.get("model", constants.PRIMARY_MODEL), max_new_tokens=None, rescue_cap=None, max_model_len=None,
        samples_dev=job["samples_dev"], samples_test=job["samples_test"], samples_bench=12,
        gpu_memory_utilization=job["gpu_memory_utilization"], limit=None,
        output_dir=job["output_dir"], max_loras_per_engine=job["max_loras_per_engine"],
        run_name_prefix=job["run_name_prefix"], dry_run=False,
        enable_thinking=job.get("enable_thinking", True),
    )


def build_batch_tasks(ev, job, gpu):
    args = make_args(job, gpu)
    runs, run_roots, steps_by_run, _ = ev.resolve_runs(args, [])
    adapters, missing = ev.discover_adapters(runs, run_roots, steps_by_run)
    if missing or not adapters:
        raise FileNotFoundError(f"adapter missing for {job['job_id']}: {missing}")
    problems = ev.load_split_problems(job["splits"], job["splits_file"])
    split_problems = {job["splits"]: problems}
    return args, adapters, ev.build_tasks(args, adapters, split_problems)


def verify_tasks(ev, jobs):
    """CPU-only: prove the multi-batch path reproduces each batch's per-request seeds and
    rendered-prompt token counts, by comparing against the rows already on disk."""
    from transformers import AutoTokenizer
    from opsd import constants, prompts
    tok = AutoTokenizer.from_pretrained(jobs[0].get("model", constants.PRIMARY_MODEL))
    total = mismatched = compared = 0
    for job in jobs:
        out = job["expected_output"]
        if not os.path.exists(out):
            print(f"  {job['job_id']}: no existing rows on disk, skipped")
            continue
        rows = [json.loads(l) for l in open(out)]
        by_key = {(r["problem_id"], r["sample_index"]): r for r in rows}
        _, _, tasks = build_batch_tasks(ev, job, 0)
        compared += 1
        for t in tasks:
            total += 1
            r = by_key[(t["problem_id"], t["sample_index"])]
            ids = prompts.encode(tok, t["messages"], enable_thinking=job.get("enable_thinking", True))
            if r["seed"] != t["seed"] or r["input_tokens"] != len(ids):
                mismatched += 1
                print(f"  MISMATCH {job['job_id']} {t['problem_id'][:12]} s{t['sample_index']}: "
                      f"seed {r['seed']} vs {t['seed']}, input_tokens {r['input_tokens']} vs {len(ids)}")
    print(f"\nVERIFY: compared {total} (problem, sample) tasks across {compared} already-completed "
          f"batch(es); mismatches = {mismatched}")
    return 1 if mismatched else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu")
    ap.add_argument("--cell")
    ap.add_argument("--next-cell", action="store_true")
    ap.add_argument("--verify-tasks", action="store_true", help="CPU-only identity check, no vLLM")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--queue", default=QUEUE)
    args_cli = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("ev", os.path.join(HERE, "multi_adapter_eval.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)      # import-safe without vLLM (its own --dry-run relies on this)

    cell = args_cli.cell or (next_cell(args_cli.queue) if args_cli.next_cell else None)
    if not cell:
        print("no pending jobs")
        return 0
    names = [f for f in job_files(args_cli.queue) if cell_of(f) == cell]
    if args_cli.max_batches:
        names = names[: args_cli.max_batches]
    if not names:
        print(f"no pending jobs for cell {cell}")
        return 0
    print(f"cell {cell}: {len(names)} pending batch(es)")

    if args_cli.verify_tasks:
        jobs = [json.load(open(os.path.join(args_cli.queue, n))) for n in names]
        return verify_tasks(ev, jobs)

    gpu = args_cli.gpu
    if gpu is None:
        print("--gpu is required unless --verify-tasks", file=sys.stderr)
        return 2

    os.makedirs(MARKERS, exist_ok=True)
    if not os.path.exists(TIMINGS):
        os.makedirs(os.path.dirname(TIMINGS) or ".", exist_ok=True)
        with open(TIMINGS, "w") as f:
            f.write("job_id,cell,lane,seconds,rc\n")

    # Claim every batch of this cell up front so two lanes never split one cell's engine.
    claimed = [(n, claim(args_cli.queue, n, gpu)) for n in names]
    claimed = [(n, p) for n, p in claimed if p]
    if not claimed:
        print(f"cell {cell}: every batch was claimed by another lane")
        return 0
    jobs = [(p, json.load(open(p))) for _, p in claimed]
    print(f"claimed {len(jobs)} batch(es) of cell {cell} on gpu {gpu}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from opsd import constants
    from transformers import AutoTokenizer
    from vllm import LLM
    import evaluate

    first = jobs[0][1]
    modes = {j.get("enable_thinking", True) for _, j in jobs}
    assert len(modes) == 1, f"cell {cell} mixes eval modes {modes}"
    print(f"cell {cell}: exp={first.get('exp')} run={first['run']} enable_thinking={first.get('enable_thinking', True)}")
    a0 = make_args(first, gpu)
    a0.max_new_tokens = constants.MAIN_MAX_NEW_TOKENS
    a0.rescue_cap = constants.RESCUE_MAX_NEW_TOKENS_CAP
    runs, run_roots, steps_by_run, _ = ev.resolve_runs(a0, [])
    adapters, _ = ev.discover_adapters(runs, run_roots, steps_by_run)
    tokenizer = AutoTokenizer.from_pretrained(a0.model)
    lora_by_key = ev.build_lora_requests(adapters)
    t_engine = time.time()
    llm = LLM(model=a0.model, dtype="bfloat16", max_model_len=constants.MODEL_MAX_LENGTH,
              gpu_memory_utilization=a0.gpu_memory_utilization, enable_lora=True,
              max_lora_rank=64, max_loras=len(adapters), max_cpu_loras=len(adapters))
    print(f"engine ready in {time.time() - t_engine:.1f}s -- reused for {len(jobs)} batch(es)")

    rc_all = 0
    deadline = time.time() + JOB_TIMEOUT * len(jobs)
    signal.signal(signal.SIGALRM, _alarm)
    remaining = list(jobs)
    for path, job in jobs:
        remaining.remove((path, job))
        if time.time() > deadline:
            print(f"  CELL DEADLINE exceeded ({JOB_TIMEOUT}s x {len(jobs)} batches) -- releasing "
                  f"{len(remaining) + 1} unrun claim(s)")
            release_claims([(path, job)] + remaining, args_cli.queue)
            return 124
        t0 = time.time()
        rc = 0
        signal.alarm(JOB_TIMEOUT)
        try:
            args = make_args(job, gpu)
            args.max_new_tokens = constants.MAIN_MAX_NEW_TOKENS
            args.rescue_cap = constants.RESCUE_MAX_NEW_TOKENS_CAP
            problems = ev.load_split_problems(job["splits"], job["splits_file"])
            tasks = ev.build_tasks(args, adapters, {job["splits"]: problems})
            records = ev.generate_and_rescue(llm, tokenizer, tasks, lora_by_key, args)
            for r in records:
                evaluate.add_budget_scores(r, tokenizer)
            ev.write_results(args.output_dir, adapters, tasks, records, args.run_name_prefix)
            n = sum(1 for _ in open(job["expected_output"]))
            if n != job["est_requests"]:
                print(f"  ROW CHECK FAILED {job['job_id']}: {n} != {job['est_requests']}")
                rc = 90
        except BatchTimeout as e:
            signal.alarm(0)
            dt = time.time() - t0
            print(f"  TIMEOUT {job['job_id']} after {dt:.0f}s: {e} -- aborting the cell process")
            with open(TIMINGS, "a") as f:
                f.write(f"{job['job_id']},{job['cell']},{gpu},{dt:.0f},124\n")
            os.makedirs(os.path.join(args_cli.queue, "failed"), exist_ok=True)
            os.rename(path, os.path.join(args_cli.queue, "failed", os.path.basename(path)))
            with open(os.path.join(MARKERS, f"{job['job_id']}.ERROR"), "w") as f:
                f.write(f"rc=124 seconds={dt:.0f} lane={gpu} multibatch=1 reason=batch_timeout\n")
            release_claims(remaining, args_cli.queue)
            return 124
        except Exception as e:                                    # noqa: BLE001
            print(f"  ERROR {job['job_id']}: {type(e).__name__}: {e}")
            rc = 1
        signal.alarm(0)
        dt = time.time() - t0
        with open(TIMINGS, "a") as f:
            f.write(f"{job['job_id']},{job['cell']},{gpu},{dt:.0f},{rc}\n")
        dest = "done" if rc == 0 else "failed"
        os.makedirs(os.path.join(args_cli.queue, dest), exist_ok=True)
        os.rename(path, os.path.join(args_cli.queue, dest, os.path.basename(path)))
        mk = "DONE" if rc == 0 else "ERROR"
        with open(os.path.join(MARKERS, f"{job['job_id']}.{mk}"), "w") as f:
            f.write(f"rc={rc} seconds={dt:.0f} lane={gpu} multibatch=1 finished={time.strftime('%FT%T%z')}\n")
        print(f"  {mk} {job['job_id']} in {dt:.0f}s")
        rc_all = rc_all or rc
    return rc_all


if __name__ == "__main__":
    code = main()
    try:
        import psutil
        for c in psutil.Process().children(recursive=True):
            c.terminate()
        psutil.wait_procs(psutil.Process().children(recursive=True), timeout=10)
        for c in psutil.Process().children(recursive=True):
            c.kill()
    except Exception:                                             # noqa: BLE001
        pass
    sys.stdout.flush()
    os._exit(code)
