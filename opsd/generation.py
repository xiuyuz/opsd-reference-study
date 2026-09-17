"""vLLM generation: per-sample seeds, context-aware token budgets, and the
length-cap rescue pass.

Runs under the vLLM environment. Training rollouts do NOT use this module --
training/train_opsd.py generates through training/rollout_worker.py (or HF `generate`
for smoke runs) because the LoRA adapter has to stay live.

A task is a plain dict:
    {"problem_id", "messages", "seed", "sample_index", "condition",
     "model_tag", "verified_answer"}
"""

import json
import os

from vllm import LLM, SamplingParams, TokensPrompt

from opsd.constants import (
    CONTEXT_SAFETY_MARGIN,
    GEN_KWARGS,
    MODEL_MAX_LENGTH,
    RESCUE_MAX_NEW_TOKENS_CAP,
)
from opsd.math_grader import extract_boxed, grade_answer
from opsd.prompts import encode


def load_llm(model_path, max_model_len=MODEL_MAX_LENGTH, gpu_memory_utilization=0.85, lora=False):
    # 0.85 assumes an otherwise-empty GPU; lower it when sharing the device with
    # another job (vLLM refuses to start if the fraction is unavailable).
    #
    # lora=True serves a trained student as backbone + live adapter (one adapter at a
    # time, passed per request as a LoRARequest). Do NOT merge the adapter into bf16
    # weights instead: at lr 5e-6 the LoRA delta is ~1e-3 of the weight scale, far below
    # a bf16 ulp, so `merge_and_unload` on a bf16 base silently rounds ~75% of the
    # trained update away (measured). Same setup as the official OPSD evaluator
    # (eval/evaluate_math.py, enable_lora / max_lora_rank block).
    kwargs = dict(
        model=model_path,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if lora:
        kwargs.update(enable_lora=True, max_lora_rank=64, max_loras=1, max_cpu_loras=1)
    return LLM(**kwargs)


def lora_request(adapter_dir):
    """One LoRARequest reused for every request of a run (id 1, single adapter)."""
    if adapter_dir is None:
        return None
    from vllm.lora.request import LoRARequest

    return LoRARequest("student_lora", 1, adapter_dir)


def sampling_params(seed, max_tokens):
    # GEN_KWARGS' do_sample/num_beams are HF-only; temperature > 0 with n == 1 is
    # the vLLM equivalent. seed is per request, so the seed schedule is matched
    # across conditions no matter how vLLM batches the requests.
    return SamplingParams(
        temperature=GEN_KWARGS["temperature"],
        top_p=GEN_KWARGS["top_p"],
        top_k=GEN_KWARGS["top_k"],
        repetition_penalty=GEN_KWARGS["repetition_penalty"],
        seed=seed,
        max_tokens=max_tokens,
    )


def prepare_prompts(tokenizer, tasks, max_new_tokens, enable_thinking=True):
    """Encode every task and compute its own completion budget. Returns
    (list of prompt token id lists, list of per-prompt max_tokens)."""
    prompt_ids, budgets = [], []
    for task in tasks:
        ids = encode(tokenizer, task["messages"], enable_thinking=enable_thinking)
        available = MODEL_MAX_LENGTH - len(ids) - CONTEXT_SAFETY_MARGIN
        prompt_ids.append(ids)
        budgets.append(min(max_new_tokens, available))
    return prompt_ids, budgets


def generate_records(llm, tokenizer, tasks, max_new_tokens, lora=None, enable_thinking=True):
    """Generate one completion per task and grade it. All requests go in at once;
    vLLM batches them internally. `lora` is a LoRARequest (or None for the base model)."""
    prompt_ids, budgets = prepare_prompts(tokenizer, tasks, max_new_tokens, enable_thinking=enable_thinking)
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids],
        sampling_params=[sampling_params(t["seed"], b) for t, b in zip(tasks, budgets)],
        lora_request=lora,
    )

    records = []
    for task, ids, budget, output in zip(tasks, prompt_ids, budgets, outputs):
        completion = output.outputs[0]
        answer = extract_boxed(completion.text)
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
                "correct": grade_answer(answer, task["verified_answer"]),
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

    rescue_length_terminated(llm, prompt_ids, records, lora)
    return records


def rescue_length_terminated(llm, prompt_ids, records, lora=None):
    """Re-run only the generations that hit the cap, same seed, with the largest
    budget the context still allows. Both outputs are kept: the capped one stays
    the 16k budget view, the rescued one becomes the full-budget output. A prompt
    that was already context-limited has nothing left to give, so it is not
    resubmitted."""
    todo = []
    for i, record in enumerate(records):
        available = MODEL_MAX_LENGTH - record["input_tokens"] - CONTEXT_SAFETY_MARGIN
        budget = min(available, RESCUE_MAX_NEW_TOKENS_CAP)
        if record["hit_length_cap"] and budget > record["max_new_tokens"]:
            todo.append((i, budget))
    if not todo:
        return

    print(f"rescue pass: resubmitting {len(todo)} of {len(records)} generations")
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids[i]) for i, _ in todo],
        sampling_params=[sampling_params(records[i]["seed"], b) for i, b in todo],
        lora_request=lora,
    )

    for (i, budget), output in zip(todo, outputs):
        completion = output.outputs[0]
        answer = extract_boxed(completion.text)
        record = records[i]
        record["rescued"] = True
        record["rescued_max_new_tokens"] = budget
        record["rescued_generated_tokens"] = len(completion.token_ids)
        record["rescued_finish_reason"] = completion.finish_reason
        record["rescued_answer_extracted"] = answer
        record["rescued_correct"] = grade_answer(answer, record["verified_answer"])
        record["rescued_output"] = completion.text
        record["rescued_output_token_ids"] = list(completion.token_ids)


def save_jsonl(records, path, append=False):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a" if append else "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]
