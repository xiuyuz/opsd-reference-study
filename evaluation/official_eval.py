"""Official-protocol external evaluation: runs the official OPSD evaluator's own
load_vllm_model / evaluate_math500 functions UNMODIFIED (imported from a checkout of
https://github.com/siyan-zhao/OPSD `eval/`, not copied), so the official prompt format,
sampling defaults, dataset loaders, and math_verify-based grading are used verbatim. The
official run_eval.sh hardcodes 4 GPUs / --tensor_parallel_size 4; this script exposes both
as flags and loops --datasets (aime24/aime25/hmmt25) INSIDE one process for a single loaded
checkpoint (3x fewer engine loads for the same coverage). Run one checkpoint per process.

This script adds NO rescue pass on top of the flat --max-new-tokens budget (the official
default, 38912, is already very generous), and evaluate_math500() sets no per-request
seed, so these runs are not seed-reproducible run-to-run -- that is the official protocol
as shipped. The base-model anchors of the released external comparison come from this
script; the trained runs come from evaluation/run_external_eval.sh.

Run under an environment matching the official README's environment.yml (vllm 0.11,
transformers 4.57, math_verify, datasets):

    OPSD_OFFICIAL_EVAL_DIR=/path/to/OPSD/eval python evaluation/official_eval.py \\
        --gpus 0 --tensor-parallel-size 1 --label base
    OPSD_OFFICIAL_EVAL_DIR=/path/to/OPSD/eval python evaluation/official_eval.py \\
        --gpus 0 --tensor-parallel-size 1 --label step100 --checkpoint-dir <adapter dir>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd.artifact_layout import EXTERNAL_BASE_ANCHORS, path as artifact_path  # noqa: E402

DEFAULT_OFFICIAL_EVAL_DIR = os.environ.get("OPSD_OFFICIAL_EVAL_DIR")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0", help="comma-separated CUDA device ids for this "
                        "process, e.g. '0' or '0,1'; sets CUDA_VISIBLE_DEVICES")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help="default: number of --gpus entries")
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="omit to evaluate the base model (no adapter)")
    parser.add_argument("--label", required=True, help="output-file prefix, e.g. 'base' or 'step100'")
    parser.add_argument("--datasets", default="aime24,aime25,hmmt25")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="limit problems per dataset (smoke test only; None = all)")
    parser.add_argument("--val-n", type=int, default=12, help="official README default: 12")
    parser.add_argument("--max-new-tokens", type=int, default=38912, help="official README default")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=40960,
                        help="official default for thinking mode (evaluate_math.py auto-sets "
                        "40960 if enable_thinking else 32768)")
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false", default=True)
    parser.add_argument("--output-dir", default=artifact_path(EXTERNAL_BASE_ANCHORS),
                        help="default: artifact_layout.EXTERNAL_BASE_ANCHORS, where the analyses read base_<bench>.json")
    parser.add_argument("--official-eval-dir", default=DEFAULT_OFFICIAL_EVAL_DIR,
                        help="path to the official OPSD repo's eval/ directory (default: env "
                        "OPSD_OFFICIAL_EVAL_DIR)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the (checkpoint, dataset, val_n, skip?) plan and exit "
                        "-- no model/vllm/dataset loading, CPU-safe")
    args = parser.parse_args()

    tp_size = args.tensor_parallel_size or len(args.gpus.split(","))
    os.makedirs(args.output_dir, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    if args.dry_run:
        for dataset_name in datasets:
            out_file = os.path.join(args.output_dir, f"{args.label}_{dataset_name}.json")
            skip = os.path.exists(out_file)
            print(f"[dry-run] checkpoint={args.checkpoint_dir or '(base model)'} "
                  f"label={args.label} dataset={dataset_name} val_n={args.val_n} "
                  f"skip={'SKIP' if skip else 'RUN'}")
        return

    if not args.official_eval_dir:
        parser.error("--official-eval-dir (or env OPSD_OFFICIAL_EVAL_DIR) must point at the official OPSD repo's eval/ directory")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    sys.path.insert(0, args.official_eval_dir)
    import evaluate_math  # noqa: E402  -- official repo module, unmodified, imported not copied

    if args.checkpoint_dir is not None and not os.path.isdir(args.checkpoint_dir):
        raise FileNotFoundError(args.checkpoint_dir)

    print(f"{'=' * 78}\n[official_eval] label={args.label} "
          f"checkpoint={args.checkpoint_dir or '(base model)'} gpus={args.gpus} tp={tp_size}\n{'=' * 78}")

    llm, tokenizer = evaluate_math.load_vllm_model(
        args.base_model,
        lora_adapter_path=args.checkpoint_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=tp_size,
        max_model_len=args.max_model_len,
        enable_thinking=args.enable_thinking,
    )

    lora_request = None
    if args.checkpoint_dir is not None:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("checkpoint_lora", 1, args.checkpoint_dir)

    summary = {}
    for dataset_name in datasets:
        out_file = os.path.join(args.output_dir, f"{args.label}_{dataset_name}.json")
        if os.path.exists(out_file):
            with open(out_file) as f:
                prior = json.load(f)
            summary[dataset_name] = prior["average_at_n_pct"]
            print(f"\n--- {args.label} / {dataset_name} ---")
            print(f"SKIP existing {out_file}")
            continue
        print(f"\n--- {args.label} / {dataset_name} ---")
        avg_pct, _ = evaluate_math.evaluate_math500(
            llm,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            num_samples=args.num_samples,
            output_file=out_file,
            lora_request=lora_request,
            dataset_name=dataset_name,
            base_model_name=args.base_model,
            enable_thinking=args.enable_thinking,
            val_n=args.val_n,
        )
        summary[dataset_name] = avg_pct
        print(f"[official_eval] {args.label}/{dataset_name}: Average@{args.val_n} = {avg_pct:.2f}%")

    print(f"\n[official_eval] {args.label} summary: "
          + ", ".join(f"{k}={v:.2f}%" for k, v in summary.items()))


if __name__ == "__main__":
    main()
    # vLLM's EngineCore teardown can hang after work is done; reap children before os._exit
    # so nothing survives and squats on the GPU.
    try:
        import psutil

        for child in psutil.Process().children(recursive=True):
            child.terminate()
        psutil.wait_procs(psutil.Process().children(recursive=True), timeout=10)
        for child in psutil.Process().children(recursive=True):
            child.kill()
    except ImportError:
        pass
    sys.stdout.flush()
    os._exit(0)
