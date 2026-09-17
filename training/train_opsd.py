"""OPSD training: one GPU, one reference condition.

One frozen backbone serves both roles. The student is the LoRA adapter *enabled* on the
no-reference prompt; the teacher is the same weights with the adapter *disabled* on the
reference prompt, so the teacher stays pinned to the step-0 policy without a second model
copy. Rollouts are on-policy from the student and never see the privileged reference.

Rollouts come from `rollout_worker.py` (a persistent vLLM engine sharing this GPU) whenever
--rollout-worker-dir is given: HF `generate` does a 16x16k batch at ~150 tok/s, vLLM at
~30x that. The in-process HF path is kept for the tiny plumbing --smoke.

--unprivileged-teacher is the reference-free teacher control:

  * OFF (default): teacher_prompt_ids = encode(teacher_msg_fn(question, reference), teacher_thinking)
  * ON:            teacher_prompt_ids = encode(student_msg_fn(question),            teacher_thinking)

    i.e. the frozen thinking-enabled backbone is scored on the STUDENT's own prompt with no
    privileged reference section at all -- exactly the profiling code's "none" context
    (profiling/prefix_score.py builds it as encode(tokenizer, student_messages(question))).
    No reference text is read at all on this path.

Every default here is the matched recipe every reported run used: LoRA r 64 / alpha 128 on
seven projection modules, 100 optimizer steps, effective batch 32, flat learning rate, loss
temperature 1.1, AdamW beta2 0.999, per-term KL clip 0.05, teacher scored thinking-on. The
alternatives stay available as flags. What the recipe leaves open is the rollout mode
(--no-student-thinking for the direct-response runs), its generation cap, and the loss
horizon; training/launch_run.sh sets those three per run and starts the rollout worker.

Run under the training env:

    CUDA_VISIBLE_DEVICES=0 python training/train_opsd.py \
        --condition answer_only --model Qwen/Qwen3-1.7B \
        --train-max-new-tokens 1024 --no-student-thinking \
        --rollout-worker-dir artifacts/rollout_worker/answer_only

--output defaults to <artifacts>/<opsd.artifact_layout.train_runs(model, student_thinking)>/
<condition> (DIRECT_TRAIN_RUNS, THINKING_TRAIN_RUNS or SMOLLM3_TRAIN_RUNS).
"""

import argparse
import json
import os
import random
import shutil
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd.artifact_layout import path as artifact_path, train_runs
from opsd.constants import (
    ALL_CONDITIONS,
    CONTEXT_SAFETY_MARGIN,
    CORRECTION_MARKERS,
    DEFAULT_SPLITS_FILE,
    EXTRA_CONDITIONS,
    GEN_KWARGS,
    MODEL_INDEX,
    MODEL_MAX_LENGTH,
    PRIMARY_MODEL,
    SPLITS_FILES,
    sample_seed,
)

# Thinking-mode runs need the model's native 40960 context so the generation budget clears
# the cap-rate target; everything else keeps the 32768 default. Env override only.
MODEL_MAX_LENGTH = int(os.environ.get("MODEL_MAX_LEN_OVERRIDE", MODEL_MAX_LENGTH))
# Every scored example allocates and frees a ~5 GB [completion, vocab] logit tensor whose
# size varies with the completion length; expandable segments stop that from fragmenting the
# caching allocator, which matters because we share the GPU with the vLLM rollout worker.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from opsd.data import load_extra_pi, load_problems
from opsd.math_grader import extract_boxed, grade_answer
from opsd.opsd_loss import opsd_forward_kl
from opsd.prompts import (
    encode,
    official_student_messages,
    official_teacher_messages,
    student_messages,
    teacher_messages,
)


def resolve_prompt_styles(prompt_style, student_prompt_style, teacher_prompt_style):
    """Resolve the per-side prompt templates from the coupled --prompt-style flag plus the two
    optional per-side overrides (the teacher-only swap run needs the official TEACHER
    instruction + the local STUDENT prompt, which the coupled flag cannot express). An
    override of None means "follow --prompt-style".

    Returns (student_style, teacher_style, student_msg_fn, teacher_msg_fn)."""
    student_style = student_prompt_style or prompt_style
    teacher_style = teacher_prompt_style or prompt_style
    student_fn = official_student_messages if student_style == "official" else student_messages
    teacher_fn = official_teacher_messages if teacher_style == "official" else teacher_messages
    return student_style, teacher_style, student_fn, teacher_fn

# Fixed for every run; nothing here is a CLI flag on purpose.
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.0
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 5e-6
BETAS = (0.9, 0.999)  # beta2 is the --adam-beta2 default
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 0.1
TOKEN_CLIP = 0.05
TOTAL_STEPS = 100
EFFECTIVE_BATCH = 32
# Default for --checkpoint-steps: both reported checkpoints (step 100 for Qwen3-1.7B, step 50
# for SmolLM3-3B) are in this list, and it is what the SmolLM3 runs saved. launch_run.sh asks
# for the denser set the Qwen runs saved, which also gives the early-step trajectories.
CHECKPOINT_STEPS = [25, 50, 75, 100]

# HF-generate rollout batch by training completion cap (in-process path only; the vLLM
# worker schedules a whole optimizer step's batch itself). Override with
# --generation-batch-size.
GENERATION_BATCH = {4096: 4, 8192: 2, 16384: 1}


def build_model(model_name, gradient_checkpointing):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to("cuda")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Keep the trainable LoRA weights in fp32: at lr 5e-6 an AdamW update is ~1e-6, well below
    # bf16 resolution around the lora_A init scale (~4e-4), so pure-bf16 params would silently
    # round most updates to zero. PEFT casts activations into the adapter dtype and back, so a
    # bf16 backbone with fp32 adapters is fine. (The official run gets fp32 master weights from
    # DeepSpeed bf16 mixed precision; single-GPU we do it directly.)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    model.config.use_cache = False
    # from_pretrained leaves the model in eval mode, and transformers' checkpointing layer
    # is gated on `self.training` -- without this, gradient_checkpointing_enable() is a no-op
    # and the 16k student forward OOMs (the HF rollout path used to flip this as a side effect).
    model.train()
    if gradient_checkpointing:
        model.enable_input_require_grads()  # required for checkpointing under a frozen backbone
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return model


def example_stream(train_ids, model_index, seed_offset=0):
    """Shuffled train split (seed 42), cycled. Identical order in every view by construction.

    seed_offset (default 0): 0 is the primary run (shuffle seed 42, sample_seed() unmodified).
    A nonzero offset both reshuffles the train order (seed 42 + seed_offset) and shifts every
    generation seed by seed_offset * 100_000_000 -- comfortably outside sample_seed()'s own
    range (its max is ~1_000_000 * model_index + 10_000 * epoch + item_index), so replicate
    runs cannot collide with each other or with the primary seed-0 run.
    """
    order = list(train_ids)
    random.Random(42 + seed_offset).shuffle(order)
    i = 0
    while True:
        epoch, position = divmod(i, len(order))
        yield order[position], sample_seed(model_index, epoch, position) + seed_offset * 100_000_000
        i += 1


def generate_rollouts(model, prompt_ids_list, max_new_tokens, seed, stop_ids, pad_id, device):
    """On-policy student rollouts: adapter enabled, student prompt, GEN_KWARGS sampling.

    One torch.manual_seed for the whole micro-batch -- HF `generate` draws every sequence
    in the call from one RNG stream, so there is no per-item seed to set. The caller logs
    this batch seed (plus each item's position in the batch) on every record rather than a
    per-example seed that had no effect; that pair is what actually reproduces a rollout.
    """
    width = max(len(p) for p in prompt_ids_list)
    input_ids = torch.full((len(prompt_ids_list), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prompt_ids_list), width), dtype=torch.long)
    for i, prompt_ids in enumerate(prompt_ids_list):
        # Left padding: generation must continue from the true end of each prompt.
        input_ids[i, width - len(prompt_ids) :] = torch.tensor(prompt_ids)
        attention_mask[i, width - len(prompt_ids) :] = 1
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    budget = min(max_new_tokens, MODEL_MAX_LENGTH - width - CONTEXT_SAFETY_MARGIN)
    torch.manual_seed(seed)
    model.eval()  # so gradient checkpointing does not disable the KV cache during generation
    with torch.no_grad():
        sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=budget,
            use_cache=True,
            pad_token_id=pad_id,
            **GEN_KWARGS,
        )
    model.train()

    rollouts = []
    for i in range(len(prompt_ids_list)):
        generated = sequences[i, width:].tolist()
        stop = next((j for j, t in enumerate(generated) if t in stop_ids), None)
        if stop is None:
            rollouts.append((generated, "length"))
        else:
            rollouts.append((generated[: stop + 1], "stop"))  # keep the generated EOS
    return rollouts


def write_atomic(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.rename(tmp, path)


def worker_rollouts(model, worker_dir, step, prompt_ids_list, seeds, max_new_tokens, stop_ids,
                    eos_id):
    """On-policy rollouts from the vLLM worker sharing this GPU (see rollout_worker.py).

    The adapter is written to disk fresh every step and announced under a new lora id, so the
    worker's policy is exactly the current student. Unlike the HF path this gives every item
    its own generator seed, which is what the per-sample seed schedule asks for.
    """
    adapter_dir = os.path.join(worker_dir, f"adapter_step{step}")
    save_start = time.time()
    model.save_pretrained(adapter_dir)  # PeftModel.save_pretrained writes the adapter only
    previous = os.path.join(worker_dir, f"adapter_step{step - 1}")
    if os.path.isdir(previous):
        shutil.rmtree(previous)  # 100 steps x ~280 MB otherwise
    save_seconds = time.time() - save_start

    items = [
        {
            "prompt_token_ids": prompt_ids,
            "seed": seed,
            "max_tokens": min(max_new_tokens,
                              MODEL_MAX_LENGTH - len(prompt_ids) - CONTEXT_SAFETY_MARGIN),
        }
        for prompt_ids, seed in zip(prompt_ids_list, seeds)
    ]
    write_atomic(os.path.join(worker_dir, "request.json"),
                 {"step": step, "adapter_dir": adapter_dir, "items": items})

    response_path = os.path.join(worker_dir, "response.json")
    error_path = os.path.join(worker_dir, "error.json")
    gen_start = time.time()
    while not os.path.exists(response_path):
        assert not os.path.exists(error_path), (
            "rollout worker failed:\n" + json.load(open(error_path))["traceback"]
        )
        time.sleep(0.2)
    response = json.load(open(response_path))
    os.remove(response_path)
    assert response["step"] == step

    rollouts = []
    for item in response["items"]:
        token_ids, finish_reason = item["token_ids"], item["finish_reason"]
        # The generated EOS stays in the completion; the HF path does the same by slicing
        # at stop + 1. On vllm 0.12 the output token_ids already end with the EOS when
        # finish_reason == "stop", so this branch does not fire -- it is here so a vLLM
        # upgrade that starts trimming stop tokens cannot silently change what the KL is
        # taken over.
        if finish_reason == "stop" and token_ids[-1] not in stop_ids:
            token_ids = token_ids + [eos_id]
        rollouts.append((token_ids, finish_reason))
    return rollouts, save_seconds, time.time() - gen_start


def completion_logits(model, prompt_ids, completion_ids, device, support_indices=None):
    """Logits predicting each completion token, aligned by *this* prompt's completion start.

    logits_to_keep=C+1 runs the lm_head only on the last C+1 positions, so a 19k-token
    full_trace teacher prompt never materializes a [prompt_len, 151936] logit tensor. Those
    positions are prompt_len-1 .. end; dropping the last one leaves exactly the C positions
    that predict the completion (the official trainer's logits[:, prompt_len - 1 : -1, :]).

    support_indices (--loss-support-windows; default None takes the exact original code path
    above): sorted completion positions j to score. The logit predicting completion token j
    lives at sequence position prompt_len + j - 1 of *this* prompt's concatenated sequence
    (teacher and student prompt lengths differ, so the offset is per-call, same alignment
    rule as the prefix path). transformers' logits_to_keep also accepts a position tensor --
    the lm_head then runs only on hidden_states[:, positions, :] -- so the trunk forward
    covers the full sequence but full-vocab logits are materialized for just
    len(support_indices) positions. Returns [len(support_indices), vocab] rows in
    support_indices order.
    """
    ids = torch.tensor([prompt_ids + completion_ids], device=device)
    if support_indices is None:
        out = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            logits_to_keep=len(completion_ids) + 1,
        )
        return out.logits[0, :-1, :]
    positions = torch.tensor([len(prompt_ids) + j - 1 for j in support_indices], device=device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=positions)
    return out.logits[0]


def marker_flags_for_tokens(token_ids, tokenizer):
    """One bool per token: does its decoded piece contain a constants.CORRECTION_MARKERS
    substring, case-insensitive? Same lexical diagnostic as
    profiling/prefix_score.py::marker_flags_for_tokens, kept as a separate copy so this
    training script does not pick up that module's pandas dependency."""
    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    flags = []
    for piece in pieces:
        text = piece.replace("Ġ", " ").replace("▁", " ").lower()  # BPE / SentencePiece space markers
        flags.append(any(m in text for m in CORRECTION_MARKERS))
    return flags


def chunked_kl_backward(teacher_logits, student_logits, completion_ids, chunk_size, scale,
                        loss_temperature=1.0, marker_flags=None, fork_mask_mode="off",
                        fork_mask_downweight=0.1):
    """Clipped forward KL over the completion, in float32 chunks, backwarded chunk by chunk.

    A float32 log-softmax over [16384, 151936] is 10 GB, so we walk the completion dimension:
    each chunk's contribution is scaled to its share of the per-token mean, backwarded
    immediately (accumulating into `student_logits`, which must be a detached leaf) and its
    graph freed. Chunks are disjoint slices, so the gradient is exact, not approximate.

    fork_mask_mode (default "off" takes the exact original code path below):
        "off"        -- unchanged: vanilla KL over every completion position.
        "exclude"    -- constants.CORRECTION_MARKERS token positions (marker_flags) are
                        dropped from the loss AND from its normalizing denominator (mean over
                        the kept, non-marker tokens only) -- "KL masked at fork positions".
        "downweight" -- marker positions stay in the loss and in the denominator (n_tokens
                        unchanged from "off"), but their contribution is scaled by
                        fork_mask_downweight (default 0.1) -- a simple stand-in for
                        "conservative/reference-anchored KL at forks": pressure at forks is
                        reduced, never fully zeroed or replaced by a second reference-model
                        forward pass.
    marker_flags: one bool per completion token (marker_flags_for_tokens() output), required
    when fork_mask_mode != "off"; unused (and not read) when "off". opsd_forward_kl itself
    (the vendored official loss) is never modified -- both new modes only change which mask/
    how many disjoint masked sub-calls chunked_kl_backward makes into it.
    """
    n_tokens = len(completion_ids)
    totals = {
        "mean_token_kl": 0.0,
        "mean_clipped_token_kl": 0.0,
        "mean_sampled_token_log_ratio": 0.0,
    }

    if fork_mask_mode == "off":
        for start in range(0, n_tokens, chunk_size):
            stop = min(start + chunk_size, n_tokens)
            weight = (stop - start) / n_tokens
            teacher_chunk = teacher_logits[start:stop].float().unsqueeze(0)
            student_chunk = student_logits[start:stop].float().unsqueeze(0)
            mask = torch.ones(1, stop - start, dtype=torch.bool, device=student_chunk.device)

            loss, stats = opsd_forward_kl(teacher_chunk, student_chunk, mask, token_clip=TOKEN_CLIP,
                                           temperature=loss_temperature)
            (loss * weight * scale).backward()
            totals["mean_token_kl"] += stats["mean_token_kl"] * weight
            totals["mean_clipped_token_kl"] += stats["mean_clipped_token_kl"] * weight

            # Sampled-token log-ratio diagnostic, on the same tokens the KL is taken over.
            with torch.no_grad():
                sampled = completion_ids[start:stop].unsqueeze(-1)
                teacher_flat, student_flat = teacher_chunk[0], student_chunk[0]
                logp_teacher = teacher_flat.gather(-1, sampled).squeeze(-1) - teacher_flat.logsumexp(-1)
                logp_student = student_flat.gather(-1, sampled).squeeze(-1) - student_flat.logsumexp(-1)
                log_ratio_sum = (logp_teacher - logp_student).sum().item()
                totals["mean_sampled_token_log_ratio"] += log_ratio_sum / n_tokens
        totals["loss"] = totals["mean_clipped_token_kl"]
        return totals

    # --- fork-masking modes; off's code path above is never reached from here ---
    assert marker_flags is not None and len(marker_flags) == n_tokens, (
        "fork_mask_mode != 'off' requires marker_flags aligned to completion_ids"
    )
    n_marker = sum(marker_flags)
    # "exclude" renormalizes over kept (non-marker) tokens only; "downweight" keeps the same
    # denominator as "off" (n_tokens) since marker positions are reweighted, not removed.
    n_kept_total = (n_tokens - n_marker) if fork_mask_mode == "exclude" else n_tokens
    totals["fork_marker_tokens"] = n_marker

    for start in range(0, n_tokens, chunk_size):
        stop = min(start + chunk_size, n_tokens)
        teacher_chunk = teacher_logits[start:stop].float().unsqueeze(0)
        student_chunk = student_logits[start:stop].float().unsqueeze(0)
        chunk_marker = torch.tensor(marker_flags[start:stop], dtype=torch.bool,
                                     device=student_chunk.device)

        if fork_mask_mode == "exclude":
            sub_groups = [(~chunk_marker, 1.0)]
        elif fork_mask_mode == "downweight":
            sub_groups = [(~chunk_marker, 1.0), (chunk_marker, fork_mask_downweight)]
        else:
            raise ValueError(f"unknown --fork-mask-mode {fork_mask_mode!r}")

        for sub_mask, sub_scale in sub_groups:
            sub_mask = sub_mask.unsqueeze(0)
            sub_count = int(sub_mask.sum().item())
            if sub_count == 0 or n_kept_total == 0:
                continue  # nothing of this kind in this chunk (or whole completion is markers)
            # Kept-COUNT-based weight (not chunk-width-based, unlike the "off" path above):
            # required so that summing every chunk's contribution reproduces exactly the mean
            # over all kept tokens in the whole completion, even when marker density varies
            # chunk to chunk.
            weight = sub_count / n_kept_total
            loss, stats = opsd_forward_kl(teacher_chunk, student_chunk, sub_mask, token_clip=TOKEN_CLIP,
                                           temperature=loss_temperature)
            (loss * sub_scale * weight * scale).backward()
            totals["mean_token_kl"] += stats["mean_token_kl"] * sub_scale * weight
            totals["mean_clipped_token_kl"] += stats["mean_clipped_token_kl"] * sub_scale * weight

        # Diagnostic only, not part of the loss -- kept over ALL tokens regardless of mode,
        # same as the "off" path.
        with torch.no_grad():
            sampled = completion_ids[start:stop].unsqueeze(-1)
            teacher_flat, student_flat = teacher_chunk[0], student_chunk[0]
            logp_teacher = teacher_flat.gather(-1, sampled).squeeze(-1) - teacher_flat.logsumexp(-1)
            logp_student = student_flat.gather(-1, sampled).squeeze(-1) - student_flat.logsumexp(-1)
            log_ratio_sum = (logp_teacher - logp_student).sum().item()
            totals["mean_sampled_token_log_ratio"] += log_ratio_sum / n_tokens

    totals["loss"] = totals["mean_clipped_token_kl"]
    return totals


def apply_loss_horizon(completion_ids, loss_horizon):
    """The KL loss is computed over only the first loss_horizon completion tokens,
    independent of how long on-policy generation was allowed to run. None (default) is a
    no-op -- returns completion_ids unchanged (same object)."""
    return completion_ids if loss_horizon is None else completion_ids[:loss_horizon]


def loss_support_indices(completion_ids, center_fracs, window_size, think_close_id):
    """Distributed loss support (--loss-support-windows): instead of a contiguous prefix
    (--loss-horizon), the KL loss covers a union of fixed-size token windows centered at
    fractions of the pre-</think> reasoning segment.

    L_r = number of completion tokens strictly before the first </think> token
    (think_close_id, taken from the run's tokenizer -- 151668 for Qwen/Qwen3-1.7B, not
    hardcoded). No </think> in the completion, or </think> as the very first token
    (L_r == 0), falls back to L_r = L. Each center fraction f gives c = round(f * L_r) and
    window [c - W//2, c + (W - W//2)) -- i.e. [c - W/2, c + W/2) for even W -- clamped to
    [0, L). Short trajectories make windows overlap; the union handles the merge.

    Returns (support, windows, reasoning_len): support = sorted unique completion positions
    the loss sees; windows = the final merged [start, end) pairs after clamp/merge (for the
    trajectory log); reasoning_len = L_r.
    """
    length = len(completion_ids)
    try:
        reasoning_len = completion_ids.index(think_close_id)
    except ValueError:
        reasoning_len = length
    if reasoning_len == 0:
        reasoning_len = length
    half = window_size // 2
    positions = set()
    for frac in center_fracs:
        center = round(frac * reasoning_len)
        positions.update(range(max(0, center - half), min(length, center + (window_size - half))))
    support = sorted(positions)
    windows = []
    for j in support:
        if windows and j == windows[-1][1]:
            windows[-1][1] = j + 1
        else:
            windows.append([j, j + 1])
    return support, windows, reasoning_len


def distill_example(model, student_prompt_ids, teacher_prompt_ids, completion_ids, chunk_size,
                    scale, device, loss_temperature=1.0, marker_flags=None, fork_mask_mode="off",
                    fork_mask_downweight=0.1, support_indices=None):
    # Context guard: the eligibility filter behind the splits file is supposed to make the
    # teacher forward fit. Qwen3's max_position_embeddings (40960) is above the default
    # MODEL_MAX_LENGTH, so a stale splits file would degrade the protocol silently instead
    # of crashing.
    assert len(teacher_prompt_ids) + len(completion_ids) <= MODEL_MAX_LENGTH, (
        f"teacher context {len(teacher_prompt_ids)} + {len(completion_ids)} exceeds "
        f"MODEL_MAX_LENGTH; the splits file was not rebuilt with the context-eligibility "
        f"filter for this --train-max-new-tokens"
    )
    with torch.no_grad(), model.disable_adapter():
        teacher_logits = completion_logits(model, teacher_prompt_ids, completion_ids, device,
                                           support_indices=support_indices)
    student_logits = completion_logits(model, student_prompt_ids, completion_ids, device,
                                       support_indices=support_indices)

    # Under --loss-support-windows (support_indices set) the KL runs over only the support
    # positions: both logit tensors are already gathered to those rows, and the sampled-token
    # ids passed to chunked_kl_backward are gathered the same way, so its per-token mean /
    # clipping semantics apply unchanged over the support token set (n_tokens = |support|,
    # comparable in magnitude to a --loss-horizon run with the same token count).
    loss_token_ids = (completion_ids if support_indices is None
                      else [completion_ids[j] for j in support_indices])
    # Two-stage backward: the chunk loop accumulates d(loss)/d(student logits) on a detached
    # leaf (so no fp32 vocab-sized tensor stays alive and the model graph is walked once), then
    # a single model backward pushes that gradient into the LoRA weights.
    leaf = student_logits.detach().requires_grad_(True)
    stats = chunked_kl_backward(
        teacher_logits, leaf, torch.tensor(loss_token_ids, device=device), chunk_size, scale,
        loss_temperature=loss_temperature, marker_flags=marker_flags, fork_mask_mode=fork_mask_mode,
        fork_mask_downweight=fork_mask_downweight,
    )
    del teacher_logits
    student_logits.backward(leaf.grad)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True,
                        help=f"one of {ALL_CONDITIONS} (reference views looked up in "
                        "AMPLE-Math) -- OR any other label when --pi-override-file or "
                        "--unprivileged-teacher is given. Validated at runtime, not via "
                        "argparse choices=, so the override paths can use arbitrary "
                        "condition names.")
    parser.add_argument("--pi-override-file", default=None,
                        help="JSON {problem_id: pi_text} used as this run's --condition "
                        "reference text, in place of the normal AMPLE-Math lookup "
                        "(constructed controls such as length-matched irrelevant rationales "
                        "or truncated traces). Must cover every id in --splits-file's train "
                        "split. Default None (unset) reproduces the normal lookup.")
    parser.add_argument("--unprivileged-teacher", action="store_true", default=False,
                        help="reference-free teacher control: score the frozen thinking-enabled\n"
                        "backbone on the STUDENT prompt (no privileged reference section at\n"
                        "all) instead of the teacher template + reference text. Implies no\n"
                        "reference lookup, so --condition may be any label and\n"
                        "--pi-override-file is refused.")
    parser.add_argument("--model", default=PRIMARY_MODEL)
    parser.add_argument("--splits-file", default=None,
                        help="train/dev/test id file (default: the release split for --model, "
                        f"data/splits/qwen3_1p7b_splits.json or data/splits/smollm3_3b_splits.json)")
    parser.add_argument("--seed", type=int, default=0,
                        help="replication seed offset (default 0 = the primary run). "
                        "Use 1, 2, ... for replicate runs; see example_stream()'s docstring "
                        "for exactly what it perturbs.")
    parser.add_argument("--train-max-new-tokens", type=int, default=None,
                        help="required for a real run; --smoke overrides it with 512")
    parser.add_argument("--output", default=None)
    parser.add_argument("--generation-batch-size", type=int, default=None,
                        help="overrides the HF-generate rollout batch for this completion cap")
    parser.add_argument("--kl-chunk-size", type=int, default=512,
                        help="completion positions per float32 KL chunk")
    parser.add_argument("--rollout-worker-dir", default=None,
                        help="watch dir of a live rollout_worker.py; without it rollouts fall "
                             "back to in-process HF generate (~30x slower at a 16k cap)")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=None,
                        help="default: on when the completion cap is >= 8192")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false")
    parser.add_argument("--effective-batch", type=int, default=None,
                        help=f"examples per optimizer step (default {EFFECTIVE_BATCH}, the "
                        "value every reported run used, matching the official recipe's "
                        "per_device 4 x 4 procs x grad_accum 2)")
    parser.add_argument("--smoke", action="store_true",
                        help="2 optimizer steps, 512-token cap, effective batch 4")
    parser.add_argument("--student-thinking", dest="student_thinking", action="store_true",
                        default=True, help="enable_thinking for student rollout prompts "
                        "(default: on)")
    parser.add_argument("--no-student-thinking", dest="student_thinking", action="store_false")
    parser.add_argument("--teacher-thinking", dest="teacher_thinking", action="store_true",
                        default=True, help="enable_thinking for teacher scoring prompts "
                        "(default: on)")
    parser.add_argument("--no-teacher-thinking", dest="teacher_thinking", action="store_false")
    parser.add_argument("--checkpoint-steps", default=",".join(map(str, CHECKPOINT_STEPS)),
                        help="comma-separated steps to save adapter checkpoints at; step 0 "
                        "saves the untrained adapter before the first optimizer step (default: "
                        f"{','.join(map(str, CHECKPOINT_STEPS))}, which covers every reported "
                        "checkpoint; launch_run.sh asks for a denser set)")
    parser.add_argument("--lr-schedule", choices=["decay", "flat"], default="flat",
                        help="flat (default, and what every reported run used): constant LR "
                        "for all steps, matching the official recipe's effective behavior "
                        "over a 100-step run (its ~28k-step decay horizon makes decay "
                        "negligible at this length). decay: TRL's implicit linear decay to "
                        "zero over the run")
    parser.add_argument("--loss-temperature", type=float, default=1.1,
                        help="divides both teacher and student logits before log-softmax in "
                        "the forward-KL loss (default 1.1, the official recipe's value and "
                        "what every reported run used; 1.0 is the no-op)")
    parser.add_argument("--adam-beta2", type=float, default=BETAS[1],
                        help=f"AdamW beta2 (default {BETAS[1]}, the official recipe's value "
                        f"and what every reported run used; beta1 is fixed at {BETAS[0]})")
    parser.add_argument("--loss-horizon", type=int, default=None,
                        help="if set, the KL distillation loss (and its logged "
                        "distillation_loss_horizon) is computed over only the first N "
                        "completion tokens, while on-policy generation still runs to "
                        "--train-max-new-tokens (student_generation_horizon). Thinking-mode "
                        "rollouts must be allowed to run long, but the loss horizon can stay "
                        "fixed at 1,024 to match the direct-response runs. Default None: "
                        "loss horizon == generation cap.")
    parser.add_argument("--loss-support-windows", default=None,
                        help="distributed loss support: comma-separated window CENTER "
                        "fractions of the pre-</think> reasoning segment, e.g. "
                        "'0.125,0.375,0.625,0.875'. The KL loss is computed only over the "
                        "union of --loss-support-window-size-token windows around those "
                        "centers (clamped to [0, L), overlaps merged; see "
                        "loss_support_indices()), for both the teacher and student forwards, "
                        "while generation still runs to --train-max-new-tokens. Mutually "
                        "exclusive with --loss-horizon. Default None: contiguous prefix.")
    parser.add_argument("--loss-support-window-size", type=int, default=256,
                        help="tokens per --loss-support-windows window (default 256); unused "
                        "when --loss-support-windows is not set")
    parser.add_argument("--prompt-style", choices=["local", "official"], default="local",
                        help="'local' (default): opsd.prompts.STUDENT_TEMPLATE / "
                        "TEACHER_TEMPLATE for both the student rollout prompt and the teacher "
                        "scoring prompt. 'official': opsd.prompts.official_student_messages / "
                        "official_teacher_messages (the verbatim official-OPSD wording) for "
                        "BOTH sides, so the prompt-template axis is isolated with view/data/"
                        "mode/recipe unchanged.")
    parser.add_argument("--student-prompt-style", choices=["local", "official"], default=None,
                        help="override the STUDENT rollout-prompt template only; default None = "
                        "follow --prompt-style. See resolve_prompt_styles().")
    parser.add_argument("--teacher-prompt-style", choices=["local", "official"], default=None,
                        help="override the TEACHER scoring-prompt template only; default None = "
                        "follow --prompt-style. The teacher-only swap run (official teacher "
                        "instruction + local student prompt, isolating the backtracking-"
                        "instruction hypothesis) is exactly --teacher-prompt-style official "
                        "with --student-prompt-style/--prompt-style left at their defaults.")
    parser.add_argument("--fork-mask-mode", choices=["off", "exclude", "downweight"], default="off",
                        help="fork-position ablation. 'off' (default): vanilla KL over every "
                        "completion position. 'exclude': constants.CORRECTION_MARKERS token "
                        "positions are dropped from the KL loss and its normalizing "
                        "denominator (mean over kept tokens only) -- 'KL masked at fork "
                        "positions'. 'downweight': marker positions stay in the loss and "
                        "denominator but their contribution is scaled by "
                        "--fork-mask-downweight -- a simple stand-in for 'conservative/"
                        "reference-anchored KL at forks'. Marker detection: "
                        "marker_flags_for_tokens(), decoded-token substring match against "
                        "constants.CORRECTION_MARKERS, computed over the loss-horizon-applied "
                        "completion -- or, under --loss-support-windows, over the support "
                        "token set (either way: the same tokens the KL loss actually sees).")
    parser.add_argument("--fork-mask-downweight", type=float, default=0.1,
                        help="scale factor applied to marker-position KL contributions when "
                        "--fork-mask-mode downweight (default 0.1); unused for off/exclude")
    args = parser.parse_args()
    if args.splits_file is None:
        args.splits_file = SPLITS_FILES.get(args.model, DEFAULT_SPLITS_FILE)
    if not args.smoke and args.train_max_new_tokens is None:
        parser.error("--train-max-new-tokens is required unless --smoke")
    if args.loss_support_windows is not None and args.loss_horizon is not None:
        parser.error("--loss-support-windows and --loss-horizon are mutually exclusive")
    support_centers = ([float(f) for f in args.loss_support_windows.split(",")]
                       if args.loss_support_windows is not None else None)
    if args.unprivileged_teacher and args.pi_override_file is not None:
        parser.error("--unprivileged-teacher reads no reference text; --pi-override-file is "
                     "meaningless with it and is refused so the two controls can never be "
                     "silently mixed")
    if (args.pi_override_file is None and not args.unprivileged_teacher
            and args.condition not in ALL_CONDITIONS):
        # Same gate argparse's choices=ALL_CONDITIONS would enforce; a runtime check only so
        # --pi-override-file can use condition names outside ALL_CONDITIONS.
        parser.error(f"argument --condition: invalid choice: {args.condition!r} "
                     f"(choose from {', '.join(repr(c) for c in ALL_CONDITIONS)}, or pass "
                     f"--pi-override-file to use an override condition name)")

    # --smoke alone: 2 tiny plumbing steps. --smoke WITH --train-max-new-tokens: the
    # memory/throughput smoke -- one full optimizer step at the real cap, to decide whether
    # the cap fits and which generation batch to use.
    requested_batch = args.effective_batch or EFFECTIVE_BATCH
    if args.smoke and args.train_max_new_tokens:
        total_steps, effective_batch, max_new_tokens = 1, requested_batch, args.train_max_new_tokens
    elif args.smoke:
        total_steps, effective_batch, max_new_tokens = 2, args.effective_batch or 4, 512
    else:
        total_steps, effective_batch, max_new_tokens = TOTAL_STEPS, requested_batch, args.train_max_new_tokens
    checkpoint_steps = (
        [total_steps] if args.smoke else [int(s) for s in args.checkpoint_steps.split(",")]
    )
    generation_batch = args.generation_batch_size or (
        # vLLM schedules the whole optimizer step's batch itself, so there is nothing to chunk.
        effective_batch if args.rollout_worker_dir
        else 4 if args.smoke
        else GENERATION_BATCH[args.train_max_new_tokens]
    )
    # The 16k-cap OOM was the student forward's stored activations (~30 GB), not the logits.
    # Checkpointing trades recompute for them; only worth it once completions are long.
    gradient_checkpointing = (
        args.gradient_checkpointing if args.gradient_checkpointing is not None
        else max_new_tokens >= 8192
    )
    student_prompt_style, teacher_prompt_style, student_msg_fn, teacher_msg_fn = (
        resolve_prompt_styles(args.prompt_style, args.student_prompt_style, args.teacher_prompt_style)
    )
    # Log string is the plain coupled value whenever the per-side overrides are inert.
    prompt_style_desc = (
        args.prompt_style
        if (student_prompt_style == args.prompt_style and teacher_prompt_style == args.prompt_style)
        else f"student={student_prompt_style}/teacher={teacher_prompt_style}"
    )

    run_name = f"opsd_{args.condition}" + ("_smoke" if args.smoke else "")
    # A smoke run defaults to its own directory so it cannot overwrite a real run's trajectory.
    default_dir = args.condition + ("_smoke" if args.smoke else "")
    out_dir = args.output or artifact_path(train_runs(args.model, args.student_thinking), default_dir)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if support_centers is not None:
        # '</think>' is a single dedicated token in Qwen3's vocab (id 151668 for
        # Qwen/Qwen3-1.7B); derive it from this run's tokenizer rather than hardcoding, and
        # check the round-trip so a model/tokenizer swap cannot silently misplace L_r.
        think_close_id = tokenizer.convert_tokens_to_ids("</think>")
        assert (think_close_id is not None
                and tokenizer.convert_ids_to_tokens(think_close_id) == "</think>"), (
            f"tokenizer has no dedicated '</think>' token (got id {think_close_id}); "
            "--loss-support-windows needs one to locate the reasoning segment"
        )
    model = build_model(args.model, gradient_checkpointing)

    eos = model.generation_config.eos_token_id
    stop_ids = set(eos) if isinstance(eos, list) else {eos}
    pad_id = tokenizer.pad_token_id
    if args.rollout_worker_dir:
        os.makedirs(args.rollout_worker_dir, exist_ok=True)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        lora_params, lr=LEARNING_RATE, betas=(BETAS[0], args.adam_beta2),
        weight_decay=WEIGHT_DECAY
    )
    # --lr-schedule flat (the default, and every reported run) holds LR constant. The
    # alternative mirrors TRL/HF's default scheduler, which is what the official scripts get
    # by never passing --lr_scheduler_type: linear decay to zero over the whole run, no
    # warmup (see --lr-schedule help above).
    if args.lr_schedule == "flat":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(0.0, (total_steps - step) / total_steps)
        )

    problems = load_problems()
    train_ids = json.load(open(args.splits_file))["train"]
    if args.unprivileged_teacher:
        # The teacher never sees a reference, so no reference text is loaded for any id.
        pass
    elif args.pi_override_file:
        # Constructed controls: reference text built by a separate script, not a literal
        # AMPLE-Math condition -- load_extra_pi() would raise KeyError looking for it, so this
        # is a parallel reference-text source rather than a widening of EXTRA_CONDITIONS.
        override_pi = json.load(open(args.pi_override_file))
        missing = [pid for pid in train_ids if pid not in override_pi]
        if missing:
            raise SystemExit(
                f"--pi-override-file {args.pi_override_file} missing {len(missing)}/"
                f"{len(train_ids)} ids needed by --splits-file {args.splits_file}'s train "
                f"split for --condition {args.condition!r}, e.g. {missing[:3]}"
            )
        for pid in train_ids:
            problems[pid]["pi"][args.condition] = override_pi[pid]
    elif args.condition in EXTRA_CONDITIONS:
        # load_problems() only attaches the 3 CONDITIONS' reference texts; fetch the
        # gist/summary/clean_solution text for exactly the ids this run will ever sample.
        extra_pi = load_extra_pi(train_ids, [args.condition])
        for pid in train_ids:
            problems[pid]["pi"][args.condition] = extra_pi[pid][args.condition]
    stream = example_stream(train_ids, MODEL_INDEX[args.model], seed_offset=args.seed)

    trajectory = open(os.path.join(out_dir, "trajectory.jsonl"), "w")
    print(f"{run_name}: {total_steps} steps x {effective_batch} examples, cap {max_new_tokens}, "
          f"loss_horizon {args.loss_horizon if args.loss_horizon is not None else max_new_tokens}, "
          f"generation batch {generation_batch}, gradient checkpointing {gradient_checkpointing}, "
          f"lr_schedule {args.lr_schedule}, loss_temperature {args.loss_temperature}, "
          f"adam_beta2 {args.adam_beta2}, seed_offset {args.seed}, "
          f"prompt_style {prompt_style_desc}, fork_mask_mode {args.fork_mask_mode}"
          + (f" (downweight {args.fork_mask_downweight})" if args.fork_mask_mode == "downweight" else "")
          + (f", loss_support_windows {args.loss_support_windows} "
             f"(window_size {args.loss_support_window_size})"
             if support_centers is not None else "")
          + f", rollouts via {args.rollout_worker_dir or 'in-process HF generate'}, "
          f"{len(train_ids)} train problems from {args.splits_file}"
          + (f", pi_override_file {args.pi_override_file}" if args.pi_override_file else "")
          + (", UNPRIVILEGED TEACHER (teacher prompt == student prompt, no reference section)"
             if args.unprivileged_teacher else ""))

    if 0 in checkpoint_steps:
        model.save_pretrained(os.path.join(out_dir, "step_0"))  # untrained adapter, pre-training

    for step in range(1, total_steps + 1):
        step_start = time.time()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        records = []
        rollout_seconds = adapter_save_seconds = score_seconds = 0.0

        while len(records) < effective_batch:
            n_batch = min(generation_batch, effective_batch - len(records))
            batch = [next(stream) for _ in range(n_batch)]
            student_prompts = [
                encode(tokenizer, student_msg_fn(problems[pid]["question"]),
                       enable_thinking=args.student_thinking)
                for pid, _ in batch
            ]
            if args.rollout_worker_dir:
                seeds = [seed for _, seed in batch]
                rollouts, save_seconds, gen_seconds = worker_rollouts(
                    model, args.rollout_worker_dir, step, student_prompts, seeds,
                    max_new_tokens, stop_ids, tokenizer.eos_token_id,
                )
                adapter_save_seconds += save_seconds
                rollout_seconds += gen_seconds
            else:
                # One HF generate() call draws every sequence from a single RNG stream, so
                # there is no per-item seed to set; the batch seed is what reproduces it.
                seeds = [batch[0][1]] * n_batch
                gen_start = time.time()
                rollouts = generate_rollouts(
                    model, student_prompts, max_new_tokens, seeds[0], stop_ids, pad_id, device
                )
                rollout_seconds += time.time() - gen_start
            score_start = time.time()

            for position, ((pid, _), student_prompt_ids, (completion_ids, finish_reason)) in enumerate(
                zip(batch, student_prompts, rollouts)
            ):
                problem = problems[pid]
                teacher_messages_for_example = (
                    student_msg_fn(problem["question"])          # no reference section at all
                    if args.unprivileged_teacher
                    else teacher_msg_fn(problem["question"], problem["pi"][args.condition])
                )
                teacher_prompt_ids = encode(
                    tokenizer, teacher_messages_for_example,
                    enable_thinking=args.teacher_thinking,
                )
                # Generation is allowed to run to --train-max-new-tokens (max_new_tokens) so
                # thinking-mode rollouts can reach a real answer, but the KL loss is only ever
                # taken over the first --loss-horizon tokens when that flag is set --
                # everything after position N is generated (and gradable for
                # correct_before_update below) but not distilled.
                loss_completion_ids = apply_loss_horizon(completion_ids, args.loss_horizon)
                # --loss-support-windows (mutually exclusive with --loss-horizon, so
                # loss_completion_ids is the FULL completion here): the KL loss covers only
                # the union of windows centered at fractions of the pre-</think> segment.
                # Computed on the raw completion_ids, never by re-slicing them, so the
                # teacher-context assert in distill_example still checks the full length.
                if support_centers is not None:
                    support_indices, support_windows, reasoning_len = loss_support_indices(
                        completion_ids, support_centers, args.loss_support_window_size,
                        think_close_id,
                    )
                else:
                    support_indices = None
                # Marker positions are located on exactly the tokens the KL loss sees -- the
                # post-loss-horizon prefix, or, under --loss-support-windows, the support
                # token set -- not the full raw completion. Skipped entirely in the default
                # "off" mode -- no extra tokenizer work added to every default-mode step.
                marker_flags = (
                    marker_flags_for_tokens(
                        [completion_ids[j] for j in support_indices]
                        if support_indices is not None else loss_completion_ids,
                        tokenizer,
                    )
                    if args.fork_mask_mode != "off" else None
                )
                stats = distill_example(
                    model,
                    student_prompt_ids,
                    teacher_prompt_ids,
                    loss_completion_ids,
                    args.kl_chunk_size,
                    1.0 / effective_batch,
                    device,
                    loss_temperature=args.loss_temperature,
                    marker_flags=marker_flags,
                    fork_mask_mode=args.fork_mask_mode,
                    fork_mask_downweight=args.fork_mask_downweight,
                    support_indices=support_indices,
                )
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
                records.append({
                    "run_name": run_name,
                    "step": step,
                    "problem_id": pid,
                    "condition": args.condition,
                    # Via the worker this is the example's own vLLM request seed; on the HF
                    # path it is the whole batch's generate() seed (one RNG stream per call),
                    # and (seed, batch_position) is what reproduces it.
                    "seed": seeds[position],
                    "batch_position": position,
                    "completion": completion,
                    "completion_tokens": len(completion_ids),
                    "finish_reason": finish_reason,
                    # Four notions of length are kept separate: reference prompt length is
                    # its own field, never folded into completion_tokens/max_new_tokens.
                    # student_generation_horizon is the full on-policy rollout cap;
                    # distillation_loss_horizon is the (possibly shorter) prefix the KL loss was
                    # actually computed over -- distinct values whenever --loss-horizon is set.
                    "pi_prompt_tokens": len(teacher_prompt_ids),
                    "student_prompt_tokens": len(student_prompt_ids),
                    "student_generation_horizon": max_new_tokens,
                    # Under --loss-support-windows this field's semantic shifts from "prefix
                    # length" to "number of loss-support tokens" (n_loss_support_tokens), so
                    # downstream tooling keeps one comparable 'tokens-the-KL-saw' scalar.
                    "distillation_loss_horizon": (
                        len(support_indices) if support_indices is not None
                        else len(loss_completion_ids)
                    ),
                    # Support diagnostics -- present only when the flag is set, so default-run
                    # trajectory records keep their exact prior key set.
                    **({"loss_support_windows": support_windows,
                        "n_loss_support_tokens": len(support_indices),
                        "support_coverage_frac": len(support_indices) / len(completion_ids),
                        "reasoning_len": reasoning_len,
                        "support_reaches_answer_phase": support_indices[-1] >= reasoning_len}
                       if support_indices is not None else {}),
                    "correct_before_update": grade_answer(
                        extract_boxed(completion), problem["verified_answer"]
                    ),
                    "prompt_style": args.prompt_style,
                    # Only present when a per-side override is actually set, so default-run
                    # trajectory records keep their exact prior key set.
                    **({"student_prompt_style": student_prompt_style,
                        "teacher_prompt_style": teacher_prompt_style}
                       if (args.student_prompt_style or args.teacher_prompt_style) else {}),
                    "fork_mask_mode": args.fork_mask_mode,
                    "fork_marker_tokens": stats.get("fork_marker_tokens", 0),
                    **stats,
                })
            score_seconds += time.time() - score_start

        grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        for record in records:
            trajectory.write(json.dumps(record) + "\n")
        trajectory.flush()

        n = len(records)
        print(
            f"step {step:3d} | loss {sum(r['loss'] for r in records) / n:.5f} "
            f"| kl {sum(r['mean_token_kl'] for r in records) / n:.4f} "
            f"| log_ratio {sum(r['mean_sampled_token_log_ratio'] for r in records) / n:+.4f} "
            f"| correct {sum(r['correct_before_update'] for r in records)}/{n} "
            f"| tokens {sum(r['completion_tokens'] for r in records) / n:.0f} "
            f"| length_capped {sum(r['finish_reason'] == 'length' for r in records)}/{n} "
            f"| gnorm {grad_norm:.3f} | lr {scheduler.get_last_lr()[0]:.2e} "
            f"| peak {torch.cuda.max_memory_allocated() / 1e9:.1f}GB "
            f"| {time.time() - step_start:.0f}s "
            f"(rollout {rollout_seconds:.0f}s, adapter save {adapter_save_seconds:.1f}s, "
            f"score {score_seconds:.0f}s)",
            flush=True,
        )

        if step in checkpoint_steps:
            model.save_pretrained(os.path.join(out_dir, f"step_{step}"))

    trajectory.close()


if __name__ == "__main__":
    main()
