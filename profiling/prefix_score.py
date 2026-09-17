"""Same-prefix teacher-forcing diagnostics.

For one fixed no-PI rollout per problem (sample_index 0), teacher-force the
*same* completion tokens under four contexts -- no PI (student prompt) and the
three PI teacher prompts (answer_only, key_points, full_trace) -- and record,
for every scored completion position and context: the sampled token's
log-probability, the full-distribution entropy, and the top-1 token, plus
(for the PI conditions) the log-ratio against the no-PI log-prob and top-1
agreement with no-PI. Optionally also computes the exact per-token forward KL
over the full vocabulary against the no-PI distribution for a small
--full-kl-ids subset.

The PI-conditioned generation records are NOT read at all: this diagnostic
scores one fixed no-PI rollout, not fresh PI trajectories. The three PI
contexts are rebuilt directly from opsd.data's problems dict (question + PI
text) via opsd.prompts, so only the no-PI generation record is read from
--generations.

Runs under the training env: torch + transformers, bf16, flash-attention 2,
single GPU, batch size 1 (see "Design notes" below). No PEFT/LoRA is
involved here -- prefix scoring happens before any student training, so the
"teacher" and "student" contexts in this file are both the frozen base model
under different prompts, not adapter-toggled.

Generation JSONL schema assumed (opsd.generation's records):
  - Each line is one generation record with at least: problem_id, condition,
    sample_index, and either output_token_ids (completion-only token ids,
    NOT including the prompt) or output (completion text) -- see
    get_completion_tokens() for the exact fallback chain, including the
    rescued_* variants.
  - condition == "none" is not assumed literally; any record whose
    `condition` is NOT one of constants.CONDITIONS is treated as a no-PI
    record (the literal sentinel in the data is "no_pi").
  - sample_index defaults to 0 if the field is absent (single-sample case).

Output parquet columns (one row per problem_id x condition x completion
position, condition in {"none", "answer_only", "key_points", "full_trace"}):
    problem_id             str
    condition              str
    position                int, 0-indexed offset into the *scored* (matched)
                            completion prefix -- NOT the raw completion index
    logp                    float32, log p_condition(sampled_token | prompt, y_<t>)
    logp_none                float32, log p_none(sampled_token | prompt, y_<t>)
                            (same value repeated across all 4 condition rows
                            for a given problem_id/position -- convenient for
                            computing log_ratio without a join)
    log_ratio                float32, logp - logp_none (0.0 when condition == "none")
    entropy                  float32, H(p_condition(. | ...)) in nats
    entropy_none              float32, H(p_none(. | ...)) in nats (repeated, as with logp_none)
    top1_agree_with_none      bool, condition's argmax token == none's argmax
                            token at this position (always True for
                            condition == "none")
    is_correction_marker      bool, the *sampled* completion token's decoded
                            piece contains a constants.CORRECTION_MARKERS
                            substring, case-insensitive (same value across
                            all 4 conditions for a given problem_id/position,
                            since it's the same underlying completion)
    region                    str, "first25" | "mid50" | "last25" of the
                            scored completion prefix (same across conditions)
    full_kl                  float32 or NaN. Exact forward KL(p_condition ||
                            p_none) over the full vocabulary, matching the
                            OPSD training loss direction (teacher -> student,
                            see opsd.opsd_loss). Only populated for problem_ids
                            passed via --full-kl-ids; NaN everywhere else.
                            For condition == "none" within the subset this is
                            trivially 0.0 (self-KL).

Design notes:
  - Matched prefix: for each item, longest_prompt_tokens is the max prompt
    length over all 4 contexts; common_completion_tokens =
    min(completion_tokens, MODEL_MAX_LENGTH - longest_prompt_tokens -
    CONTEXT_SAFETY_MARGIN). Items with < 512 matched completion tokens are
    skipped and counted toward the reported omission rate (printed to stdout
    and written to <output>_summary.json next to the parquet).
  - One sequence per forward pass (no cross-context padding/batching): the
    four contexts for one problem have wildly different prompt lengths
    (answer_only ~hundreds of tokens vs full_trace up to ~19k), and correctly
    aligning padded position_ids for a causal model with flash-attention is
    more failure-prone than it's worth for a diagnostic script that never
    runs at GPU-saturating scale.
  - Memory: forward pass runs in bf16 no_grad through the base transformer
    only (model.model, not model.lm_head) to get last-hidden-state, which is
    then sliced down to just the needed completion-predicting positions
    *before* the lm_head projection -- this avoids ever materializing
    (seq_len, vocab_size) logits for a 32k-token sequence. The lm_head
    projection, log_softmax, entropy, and (optionally) full-vocab KL are then
    computed in float32 in --chunk-size (default 2048) token chunks, so peak
    memory is bounded by chunk_size x vocab_size regardless of context length.

CLI:
    python profiling/prefix_score.py --model Qwen/Qwen3-1.7B \
        --generations artifacts/profiles/calibration/qwen3_1.7b_generations.jsonl \
        --output artifacts/profiles/calibration/qwen3_1.7b_prefix_scores.parquet \
        [--limit 20] [--full-kl-ids artifacts/profiles/calibration/full_kl_ids.json]

    python profiling/prefix_score.py --self-test
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from opsd.constants import CONDITIONS, CONTEXT_SAFETY_MARGIN, CORRECTION_MARKERS, MODEL_MAX_LENGTH  # noqa: E402
from opsd.data import load_problems  # noqa: E402
from opsd.prompts import encode, student_messages, teacher_messages  # noqa: E402


# ---------------------------------------------------------------------------
# Generation record loading
# ---------------------------------------------------------------------------


def load_no_pi_rollouts(generations_path):
    """problem_id -> generation record, for the no-PI rollout with sample_index 0.

    Any record whose `condition` is one of constants.CONDITIONS is a
    PI-conditioned rollout and is skipped -- this module only needs the
    no-PI rollout (see module docstring). Keeps the first no-PI record with
    sample_index == 0 (or missing sample_index) encountered per problem_id.
    """
    rollouts = {}
    with open(generations_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("condition") in CONDITIONS:
                continue
            if rec.get("sample_index", 0) != 0:
                continue
            pid = rec["problem_id"]
            if pid not in rollouts:
                rollouts[pid] = rec
    return rollouts


def get_completion_tokens(rec, tokenizer):
    """Completion-only token ids for a no-PI generation record.

    Prefers the rescued (near-32k) completion when present -- it is the
    full-budget output that supersedes the 16k-capped one -- so the matched
    prefix is longer for free whenever the per-item context budget allows it.
    Falls back to re-tokenizing the completion text if token ids were not saved.
    """
    ids = rec.get("rescued_output_token_ids") or rec.get("output_token_ids")
    if ids:
        return ids
    text = rec.get("rescued_output") or rec.get("output")
    if not text:
        raise ValueError(
            f"generation record for problem_id={rec.get('problem_id')!r} has neither "
            "output_token_ids nor output text"
        )
    return tokenizer.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------------------
# Chunked math (also exercised directly by --self-test)
# ---------------------------------------------------------------------------


def chunk_stats(logits_f32, target_ids):
    """logits_f32: (n, vocab) float32. target_ids: (n,) long.

    Returns (logp, entropy, top1) for this chunk: sampled-token log-prob,
    full-distribution entropy (nats), and argmax token id, all length n.
    """
    log_probs = F.log_softmax(logits_f32, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(-1)
    top1 = log_probs.argmax(-1)
    logp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return logp, entropy, top1


def forward_kl_full(logits_p, logits_q):
    """KL(p || q) per position, elementwise over the vocab dim.

    logits_p, logits_q: (n, vocab) float32, same n. Returns (n,) float32.
    """
    logp = F.log_softmax(logits_p, dim=-1)
    logq = F.log_softmax(logits_q, dim=-1)
    p = logp.exp()
    return (p * (logp - logq)).sum(-1)


def region_of(position, total):
    if position < total * 0.25:
        return "first25"
    if position >= total * 0.75:
        return "last25"
    return "mid50"


def marker_flags_for_tokens(token_ids, tokenizer):
    """One bool per token: does its decoded piece contain a correction-marker
    substring, case-insensitive? A simple lexical diagnostic, not a definitive
    reasoning annotation."""
    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    flags = []
    for piece in pieces:
        text = piece.replace("Ġ", " ").replace("▁", " ").lower()  # BPE / SentencePiece space markers
        flags.append(any(m in text for m in CORRECTION_MARKERS))
    return flags


# ---------------------------------------------------------------------------
# Per-problem scoring
# ---------------------------------------------------------------------------


@torch.no_grad()
def hidden_slice(model, prompt_ids, completion_ids, device):
    """Last-hidden-state rows that predict each completion token, i.e. hidden
    states at positions [len(prompt_ids)-1, len(prompt_ids)-1+len(completion_ids)).
    Shape (len(completion_ids), hidden_size), model dtype (bf16)."""
    full_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    out = model.model(input_ids=full_ids, use_cache=False)
    start = len(prompt_ids) - 1
    length = len(completion_ids)
    return out.last_hidden_state[0][start : start + length]


@torch.no_grad()
def score_problem(model, tokenizer, pid, problem, rec, chunk_size, compute_full_kl, device,
                  min_matched_tokens=512):
    """Returns a list of output rows for this problem, or None if omitted
    (fewer than min_matched_tokens matched completion tokens fit under every
    context)."""
    completion_ids_full = get_completion_tokens(rec, tokenizer)

    none_prompt_ids = encode(tokenizer, student_messages(problem["question"]))
    pi_prompt_ids = {
        cond: encode(tokenizer, teacher_messages(problem["question"], problem["pi"][cond]))
        for cond in CONDITIONS
    }

    longest_prompt = max(len(none_prompt_ids), max(len(p) for p in pi_prompt_ids.values()))
    budget = MODEL_MAX_LENGTH - longest_prompt - CONTEXT_SAFETY_MARGIN
    common_len = min(len(completion_ids_full), budget)
    if common_len < min_matched_tokens:
        return None

    completion_ids = completion_ids_full[:common_len]
    target = torch.tensor(completion_ids, dtype=torch.long, device=device)
    marker_flags = marker_flags_for_tokens(completion_ids, tokenizer)
    regions = [region_of(i, common_len) for i in range(common_len)]

    contexts = {"none": none_prompt_ids}
    contexts.update(pi_prompt_ids)
    hiddens = {c: hidden_slice(model, prompt_ids, completion_ids, device) for c, prompt_ids in contexts.items()}

    per_cond = {
        c: {
            "logp": torch.empty(common_len, dtype=torch.float32),
            "entropy": torch.empty(common_len, dtype=torch.float32),
            "top1": torch.empty(common_len, dtype=torch.long),
        }
        for c in contexts
    }
    full_kl = {c: torch.empty(common_len, dtype=torch.float32) for c in CONDITIONS} if compute_full_kl else {}

    for cs in range(0, common_len, chunk_size):
        ce = min(cs + chunk_size, common_len)
        tgt_chunk = target[cs:ce]
        log_probs_by_cond = {}
        for c, h in hiddens.items():
            logits = model.lm_head(h[cs:ce]).float()
            logp, entropy, top1 = chunk_stats(logits, tgt_chunk)
            per_cond[c]["logp"][cs:ce] = logp.cpu()
            per_cond[c]["entropy"][cs:ce] = entropy.cpu()
            per_cond[c]["top1"][cs:ce] = top1.cpu()
            if compute_full_kl:
                log_probs_by_cond[c] = F.log_softmax(logits, dim=-1)
            del logits
        if compute_full_kl:
            none_lp = log_probs_by_cond["none"]
            for cond in CONDITIONS:
                pi_lp = log_probs_by_cond[cond]
                pi_prob = pi_lp.exp()
                full_kl[cond][cs:ce] = (pi_prob * (pi_lp - none_lp)).sum(-1).cpu()
            del log_probs_by_cond

    none_logp = per_cond["none"]["logp"]
    none_entropy = per_cond["none"]["entropy"]
    none_top1 = per_cond["none"]["top1"]

    rows = []
    for cond in ["none"] + CONDITIONS:
        logp = per_cond[cond]["logp"]
        entropy = per_cond[cond]["entropy"]
        top1 = per_cond[cond]["top1"]
        for i in range(common_len):
            rows.append(
                {
                    "problem_id": pid,
                    "condition": cond,
                    "position": i,
                    "logp": float(logp[i]),
                    "logp_none": float(none_logp[i]),
                    "log_ratio": 0.0 if cond == "none" else float(logp[i] - none_logp[i]),
                    "entropy": float(entropy[i]),
                    "entropy_none": float(none_entropy[i]),
                    "top1_agree_with_none": bool(top1[i] == none_top1[i]),
                    "is_correction_marker": bool(marker_flags[i]),
                    "region": regions[i],
                    "full_kl": (
                        float(full_kl[cond][i])
                        if (compute_full_kl and cond != "none")
                        else (0.0 if compute_full_kl else None)
                    ),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Self-test (CPU, no model/GPU required)
# ---------------------------------------------------------------------------


def self_test():
    torch.manual_seed(0)
    length, vocab = 5000, 300
    logits_a = torch.randn(length, vocab) * 3
    logits_b = torch.randn(length, vocab) * 3
    target = torch.randint(0, vocab, (length,))
    chunk_size = 777  # not a divisor of length: exercises the ragged-last-chunk path

    ref_logp, ref_entropy, ref_top1 = chunk_stats(logits_a, target)

    got_logp = torch.empty(length)
    got_entropy = torch.empty(length)
    got_top1 = torch.empty(length, dtype=torch.long)
    for cs in range(0, length, chunk_size):
        ce = min(cs + chunk_size, length)
        logp, entropy, top1 = chunk_stats(logits_a[cs:ce], target[cs:ce])
        got_logp[cs:ce] = logp
        got_entropy[cs:ce] = entropy
        got_top1[cs:ce] = top1

    assert torch.allclose(got_logp, ref_logp, atol=1e-5), "chunked logp mismatch vs. unchunked reference"
    assert torch.allclose(got_entropy, ref_entropy, atol=1e-5), "chunked entropy mismatch vs. unchunked reference"
    assert torch.equal(got_top1, ref_top1), "chunked top1 mismatch vs. unchunked reference"
    assert (got_entropy >= -1e-5).all(), "entropy must be non-negative"
    assert (got_entropy <= torch.log(torch.tensor(float(vocab))) + 1e-4).all(), "entropy must be <= log(vocab)"

    ref_kl = forward_kl_full(logits_a, logits_b)
    got_kl = torch.empty(length)
    for cs in range(0, length, chunk_size):
        ce = min(cs + chunk_size, length)
        got_kl[cs:ce] = forward_kl_full(logits_a[cs:ce], logits_b[cs:ce])
    assert torch.allclose(got_kl, ref_kl, atol=1e-4), "chunked KL mismatch vs. unchunked reference"
    assert (got_kl >= -1e-4).all(), "forward KL must be non-negative"

    self_kl = forward_kl_full(logits_a, logits_a)
    assert torch.allclose(self_kl, torch.zeros(length), atol=1e-4), "KL(p || p) must be ~0"

    # region_of / marker detection sanity (no tokenizer needed for region_of)
    assert region_of(0, 100) == "first25"
    assert region_of(24, 100) == "first25"
    assert region_of(25, 100) == "mid50"
    assert region_of(74, 100) == "mid50"
    assert region_of(75, 100) == "last25"
    assert region_of(99, 100) == "last25"

    print("prefix_score.py --self-test: all checks passed")
    print(f"  length={length} vocab={vocab} chunk_size={chunk_size} (ragged last chunk={length % chunk_size})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="HF model id/path, e.g. Qwen/Qwen3-1.7B")
    parser.add_argument("--generations", default=None, help="preflight generations JSONL")
    parser.add_argument("--output", default=None, help="output parquet path")
    parser.add_argument("--limit", type=int, default=None, help="only score the first N problems (debugging)")
    parser.add_argument(
        "--full-kl-ids",
        default=None,
        help="JSON file: list of problem_ids to also compute exact full-vocab forward KL for",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="sequence chunk size for float32 log_softmax/entropy/KL (bounds peak memory at long contexts)",
    )
    parser.add_argument(
        "--min-matched-tokens",
        type=int,
        default=512,
        help="omit problems whose matched completion prefix is shorter than this; lower it only for smoke runs with a tiny generation cap",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run a CPU shape/correctness check of the chunked entropy/KL math on random logits, then exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return

    if not (args.model and args.generations and args.output):
        raise SystemExit("--model, --generations, and --output are required unless --self-test")

    from transformers import AutoModelForCausalLM, AutoTokenizer  # deferred: --self-test needs neither

    problems = load_problems()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(args.device)
    model.eval()

    rollouts = load_no_pi_rollouts(args.generations)
    ids = sorted(rollouts.keys())
    if args.limit:
        ids = ids[: args.limit]

    full_kl_ids = set(json.load(open(args.full_kl_ids))) if args.full_kl_ids else set()

    all_rows = []
    n_total = 0
    n_omitted = 0
    for pid in ids:
        if pid not in problems:
            print(f"warning: {pid} present in generations but not in load_problems() pool, skipping")
            continue
        n_total += 1
        rows = score_problem(
            model,
            tokenizer,
            pid,
            problems[pid],
            rollouts[pid],
            chunk_size=args.chunk_size,
            compute_full_kl=(pid in full_kl_ids),
            device=args.device,
            min_matched_tokens=args.min_matched_tokens,
        )
        if rows is None:
            n_omitted += 1
            continue
        all_rows.extend(rows)

    omission_rate = n_omitted / n_total if n_total else 0.0
    print(
        f"scored {n_total - n_omitted}/{n_total} problems "
        f"({omission_rate:.1%} omitted: fewer than {args.min_matched_tokens} matched completion tokens)"
    )

    df = pd.DataFrame(all_rows)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(args.output, index=False)

    summary_path = os.path.splitext(args.output)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "total_problems": n_total,
                "omitted_problems": n_omitted,
                "omission_rate": omission_rate,
                "scored_problems": n_total - n_omitted,
                "full_kl_problems": len(full_kl_ids),
                "min_matched_tokens": args.min_matched_tokens,
            },
            f,
            indent=2,
        )
    print(f"wrote {args.output} ({len(all_rows)} rows) and {summary_path}")


if __name__ == "__main__":
    main()
