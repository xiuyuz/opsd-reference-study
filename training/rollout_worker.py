"""Persistent vLLM rollout worker for train_opsd.py, driven through a watch directory.

HF `model.generate` runs a 16-sequence, 16k-cap rollout batch at ~150 tok/s aggregate --
~28 minutes of generation per optimizer step, i.e. 50+ hours for a 100-step run. vLLM's
continuous batching does the same work in a couple of minutes, but it cannot live inside
the trainer process: vLLM and the HF/PEFT training stack are different environments. So
each training run runs two processes on one GPU and they talk through files:

    trainer -> <watch-dir>/request.json   {"step", "adapter_dir", "items": [...]}
    worker  -> <watch-dir>/response.json  {"step", "items": [{"token_ids", "finish_reason"}]}
    worker  -> <watch-dir>/ready.json     engine is up (the launcher waits for this before the trainer)
    trainer -> <watch-dir>/stop.json      worker exits 0
    worker  -> <watch-dir>/error.json     traceback; worker exits nonzero and the trainer aborts

Every file is written to a .tmp sibling and os.rename'd, so a reader never sees a partial
JSON document.

Run under the vLLM env (training/launch_run.sh does this for you):

    CUDA_VISIBLE_DEVICES=0 python training/rollout_worker.py \
        --watch-dir artifacts/rollout_worker/answer_only
"""

import argparse
import json
import os
import time
import traceback

PROTOCOL_FILES = ["request.json", "response.json", "ready.json", "stop.json", "error.json"]


def write_atomic(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.rename(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--watch-dir", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="skip CUDA graph capture; only for smoke comparisons")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="rollout sampling temperature passed to vLLM (default 1.0, "
                        "matches constants.GEN_KWARGS; the official recipe uses 1.1)")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    watch = args.watch_dir
    os.makedirs(watch, exist_ok=True)
    for name in PROTOCOL_FILES:  # a previous run's leftovers would be answered as this run's
        stale = os.path.join(watch, name)
        if os.path.exists(stale):
            os.remove(stale)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=64,
        max_loras=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    write_atomic(os.path.join(watch, "ready.json"), {"pid": os.getpid()})
    print(f"rollout worker ready, watching {watch}, temperature={args.temperature}", flush=True)

    request_path = os.path.join(watch, "request.json")
    while True:
        if os.path.exists(os.path.join(watch, "stop.json")):
            print("stop.json seen, exiting", flush=True)
            return
        if not os.path.exists(request_path):
            time.sleep(0.2)
            continue

        request = json.load(open(request_path))
        step, items = request["step"], request["items"]
        start = time.time()
        try:
            outputs = llm.generate(
                [TokensPrompt(prompt_token_ids=item["prompt_token_ids"]) for item in items],
                sampling_params=[
                    SamplingParams(
                        temperature=args.temperature,
                        top_p=0.95,
                        top_k=20,
                        repetition_penalty=1.0,
                        seed=item["seed"],
                        max_tokens=item["max_tokens"],
                    )
                    for item in items
                ],
                # A fresh integer lora id each step forces a reload from disk; max_loras=1
                # evicts the previous step's adapter, so the rollouts are always on-policy.
                lora_request=LoRARequest(f"step{step}", step, request["adapter_dir"]),
                use_tqdm=False,
            )
        except Exception:
            write_atomic(
                os.path.join(watch, "error.json"),
                {"step": step, "traceback": traceback.format_exc()},
            )
            raise

        response = {
            "step": step,
            "items": [
                {
                    "token_ids": list(output.outputs[0].token_ids),
                    "finish_reason": output.outputs[0].finish_reason,
                }
                for output in outputs
            ],
        }
        write_atomic(os.path.join(watch, "response.json"), response)
        os.remove(request_path)

        elapsed = time.time() - start
        generated = sum(len(item["token_ids"]) for item in response["items"])
        print(
            f"step {step}: {len(items)} seqs, {generated} tokens, {elapsed:.1f}s "
            f"({generated / elapsed:.0f} tok/s aggregate)",
            flush=True,
        )


if __name__ == "__main__":
    main()
