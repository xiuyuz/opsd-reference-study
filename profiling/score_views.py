"""Privilege profile: same-prefix teacher forcing of ALL no-PI samples under ALL six
reference views.

Extends prefix_score.py / preflight_ext.py's diagnostic from "1 no-PI sample x {3 or 7}
contexts" to "all 4 no-PI samples x all 6 reference views" on the same calibration set and
the same frozen generations. No new generation happens here -- this only teacher-forces the
SAME already-sampled no-PI completions (<artifacts>/profiles/calibration/qwen3_1.7b_generations.jsonl,
sample_index 0-3) under 7 contexts (no-PI "student" + the 6 PI "teacher" views). All PI text
(all 6 views) comes from opsd.data's load_problems()/load_extra_pi(), never from fresh
sampling. It imports prefix_score.py's chunked math and per-token helpers (chunk_stats's
formula, region_of, marker_flags_for_tokens, hidden_slice, get_completion_tokens) directly.

Per-token columns beyond prefix_score.py's schema: sample_index, correct (trajectory-level),
normalized_position, token_category ("think"/"answer", split at the completion's first
`</think>` token), eos_prob / eos_prob_student, think_close_prob / think_close_prob_student,
answer_transition_prob / _student (P(next token == "boxed"), i.e. the token that starts
`\\boxed{...}` -- the only two vocab entries containing "boxed" are "boxed" (bare, following
"\\") and "Ġboxed" (mid-sentence prose); the former is the answer-transition marker).
full_kl (KL(teacher_view || student), matching opsd.opsd_loss's direction) is unconditional
here: the correctness-alignment analysis needs it on every scored trajectory, and it adds
no new forward pass.

GPU efficiency (this is ~4x preflight_ext's scoring cost -- 4 samples x 6 views vs. 1 sample
x 7 contexts):
  - bf16 weights, flash-attention-2, same as prefix_score.py.
  - The student ("none") forward pass and its lm_head/log_softmax are computed exactly ONCE
    per (problem, sample) and shared across all 6 PI views' KL/log-ratio/disagreement stats.
  - Per completion-position chunk, only the STUDENT's (chunk, vocab) float32 log_probs tensor
    is kept alive across the chunk's context loop (needed for every view's full_kl term); each
    view's own (chunk, vocab) logits/log_probs/probs tensors are transient and freed before the
    next view starts. Peak transient memory per chunk is therefore ~2 x (chunk_size x
    vocab_size) float32 buffers, not 7x, which is what lets --chunk-size be pushed well past
    prefix_score.py's default of 2048 on a free 80 GB GPU.
  - Prompt token ids for the 6 PI views (and the student prompt) are built ONCE per problem,
    outside the per-sample loop -- they don't depend on sample_index, only the completion does.
  - No cross-context/cross-problem batching (single sequence per forward pass), for the same
    reason prefix_score.py documents.
  - Host RAM: the full run is ~24x more rows than prefix_score.py (512 x 4 x 7). Buffering
    every row in one Python list before a single to_parquet() call would approach ~400GB
    resident memory; --flush-every (default 8) instead writes a pyarrow ParquetWriter row
    group every N problems and drops the buffered rows, bounding peak host RAM to a small,
    fixed multiple of one flush's worth of rows regardless of total run length.

Output parquet columns (one row per problem_id x sample_index x condition x position,
condition in {"none", "answer_only", "gist", "key_points", "clean_solution", "summary",
"full_trace"} -- ALL_VIEWS is ordered by measured reference token density):
    problem_id                 str
    sample_index                int, 0-3 (which of the 4 no-PI rollouts)
    condition                   str
    correct                     bool, trajectory-level correctness of this no-PI sample
                                (rescued_correct if rescued else correct; same value across
                                all condition/position rows for a given problem_id/sample_index)
    position                    int, 0-indexed offset into the scored (matched) completion
                                prefix -- NOT the raw completion index
    normalized_position         float32, position / max(matched_length - 1, 1)
    token_category               str, "think" | "answer" -- boundary token defined by
                                `--student-thinking`: with thinking ON (default), the boundary
                                is the completion's first `</think>` token; with thinking OFF
                                (direct-response profile: no `</think>` boundary exists), the
                                boundary is instead the first answer-transition ("boxed") token.
                                "answer" starts at (and includes) the boundary token; "think"
                                for every position if the boundary token never appears in the
                                matched prefix.
    region                       str, "first25" | "mid50" | "last25" (kept from prefix_score.py)
    logp                        float32, log p_condition(sampled_token | prompt, y_<t>)
    logp_student                 float32, log p_student(sampled_token | ...) (repeated across
                                all 7 condition rows for a position -- join-free log_ratio)
    log_ratio                   float32, logp - logp_student (0.0 when condition == "none")
    entropy                      float32, H(p_condition(. | ...)) in nats
    entropy_student               float32, H(p_student(. | ...)) in nats (repeated)
    top1_agree_with_student        bool, condition's argmax token == student's argmax token
    is_correction_marker          bool, sampled token's decoded piece contains a
                                constants.CORRECTION_MARKERS substring (repeated across
                                conditions -- same underlying completion)
    eos_prob / eos_prob_student            float32, P(next token == eos_token_id)
    think_close_prob / think_close_prob_student   float32, P(next token == "</think>")
    answer_transition_prob / answer_transition_prob_student   float32, P(next token == "boxed")
    full_kl                     float32. Exact forward KL(p_condition || p_student) over the
                                full vocabulary; 0.0 for condition == "none" (self-KL).
    token_id                     int64. The sampled/target completion token id at this position
                                (repeated identically across all 7 condition rows for a given
                                problem_id/sample_index/position -- same underlying completion).
                                Keeps token identity aligned with the (possibly truncated)
                                SCORED prefix, including any --scoring-horizon cap, so the
                                post-correct-answer KL mass, repeated-span KL mass, repeated
                                n-grams and suffix compression are computable downstream with
                                no new generation.
    clipped_kl                   float32. Elementwise-clipped forward KL(p_condition ||
                                p_student), matching opsd.opsd_loss's actual training-loss clip
                                (token_clip=0.05, applied per (position, vocab-class) entry
                                BEFORE the vocab sum, exactly opsd_forward_kl's
                                `kl_elem.clamp(max=token_clip).sum(-1)`) -- see TOKEN_CLIP below.
                                full_kl remains the unclipped proxy; clipped_kl is the actual
                                training objective's loss-mass unit. Can be negative (same
                                documented behavior as opsd.opsd_loss). 0.0 for condition ==
                                "none" (self-clip).
    pi_length_tokens              float32. Exact tokenizer count of the raw reference text for
                                this row's condition (tokenizer.encode(pi_text,
                                add_special_tokens=False), NOT the full wrapped-prompt length).
                                Constant across sample_index/position for a given
                                (problem_id, condition); 0.0 for condition == "none".

CLI:
    python profiling/score_views.py \
        --output artifacts/profiles/thinking/all_rollouts_prefix_scores.parquet \
        [--limit 8] [--chunk-size 4096] [--min-matched-tokens 512] [--flush-every 8]

    # Mode-matched direct-response profile: same script, different generations/output/mode.
    python profiling/score_views.py \
        --generations artifacts/profiles/calibration_direct/qwen3_1.7b_generations.jsonl \
        --output artifacts/profiles/direct/all_rollouts_prefix_scores.parquet \
        --no-student-thinking --scoring-horizon 1024 --prompt-style local
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from opsd.artifact_layout import PROFILE_CALIBRATION, PROFILE_THINKING, path as artifact_path  # noqa: E402
from opsd.constants import CONTEXT_SAFETY_MARGIN, EXTRA_CONDITIONS, MODEL_MAX_LENGTH, NO_PI_SAMPLES, PRIMARY_MODEL  # noqa: E402
from opsd.constants import CONDITIONS as ORIGINAL_CONDITIONS  # noqa: E402
from prefix_score import get_completion_tokens, hidden_slice, marker_flags_for_tokens, region_of  # noqa: E402

# Density order (by measured reference token count), not release/discovery order.
ALL_VIEWS = ["answer_only", "gist", "key_points", "clean_solution", "summary", "full_trace"]
assert set(ALL_VIEWS) == set(ORIGINAL_CONDITIONS) | set(EXTRA_CONDITIONS), (
    "ALL_VIEWS must be exactly the 3 CONDITIONS plus the 3 EXTRA_CONDITIONS"
)

OUT_DIR = artifact_path(PROFILE_THINKING)
ORIG_PREFLIGHT_DIR = artifact_path(PROFILE_CALIBRATION)  # read-only from this script

# Matches opsd.opsd_loss's `token_clip` default exactly (per-(token, vocab-class) elementwise
# clip applied before the vocab sum -- see that module's docstring). Duplicated as a plain
# constant here rather than importing opsd_forward_kl, which expects unchunked
# (batch, seq, vocab) tensors and a completion_mask -- incompatible with this file's
# chunked, per-context streaming shape. The clip *formula* (clamp-then-sum on the same
# elementwise teacher_prob * (log_teacher - log_student) integrand this file already
# computes for full_kl) is reproduced exactly; only the tensor-shape plumbing differs.
TOKEN_CLIP = 0.05


# --------------------------------------------------------------------------- #
# Generation record loading (all 4 no-PI samples, unlike prefix_score.py's sample 0 only)
# --------------------------------------------------------------------------- #


def load_all_no_pi_rollouts(generations_path):
    """problem_id -> {sample_index: generation record} for every no-PI rollout (any record
    whose `condition` is not one of the 3 CONDITIONS -- same robust check prefix_score.py
    uses; the literal string in the data is "no_pi")."""
    rollouts = defaultdict(dict)
    with open(generations_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("condition") in ORIGINAL_CONDITIONS:
                continue
            pid = rec["problem_id"]
            si = rec.get("sample_index", 0)
            rollouts[pid][si] = rec
    return rollouts


def trajectory_correct(rec):
    """Same convention as preflight_ext.py's print_generation_summary: prefer the rescued
    (near-32k) completion's correctness once a rescue happened, else the original."""
    return bool(rec["rescued_correct"]) if rec["rescued"] else bool(rec["correct"])


# --------------------------------------------------------------------------- #
# Per-token category / position helpers
# --------------------------------------------------------------------------- #


def token_categories(completion_ids, boundary_ids):
    """One "think"/"answer" label per position: "answer" starts at (and includes) the first
    position whose token id is in boundary_ids; if none of boundary_ids ever appears in the
    matched prefix, every position is "think" (no identifiable answer field).

    boundary_ids is a list, not a single id, so the caller can pick the mode-appropriate
    boundary: thinking-ON passes [think_close_id] (the `</think>` token); thinking-OFF has no
    `</think>` boundary at all, so it passes [answer_transition_id] (the "boxed" token that
    starts `\\boxed{...}`) instead -- the think/answer split is re-derived for the mode, not
    carried over. (A single-element list is still a list here, kept general in case a future
    mode wants an either-of-N boundary.)"""
    boundary_set = set(boundary_ids)
    split = len(completion_ids)
    for i, tid in enumerate(completion_ids):
        if tid in boundary_set:
            split = i
            break
    return ["think" if i < split else "answer" for i in range(len(completion_ids))]


def normalized_positions(n):
    denom = max(n - 1, 1)
    return [i / denom for i in range(n)]


# --------------------------------------------------------------------------- #
# Per-problem prompt construction (once per problem, shared across its 4 samples)
# --------------------------------------------------------------------------- #


def build_prompts(tokenizer, problem, extra_pi_for_pid, prompt_style="local"):
    """prompt_style: "local" (default -- opsd.prompts.STUDENT_TEMPLATE/TEACHER_TEMPLATE) or
    "official" (verbatim official-OPSD train-time wording, official_student_messages/
    official_teacher_messages). Applies to both the no-PI/student prompt and all six
    PI/teacher prompts.

    Also returns pi_lengths: {condition: exact tokenizer count of the RAW reference text (not
    the wrapped prompt)}, computed once here with the already-loaded tokenizer so downstream
    analyses need no word-count proxy."""
    from opsd.prompts import (
        encode,
        official_student_messages,
        official_teacher_messages,
        student_messages,
        teacher_messages,
    )

    if prompt_style == "official":
        student_msg_fn, teacher_msg_fn = official_student_messages, official_teacher_messages
    else:
        student_msg_fn, teacher_msg_fn = student_messages, teacher_messages

    none_prompt_ids = encode(tokenizer, student_msg_fn(problem["question"]))
    view_prompt_ids = {}
    pi_lengths = {}
    for cond in ORIGINAL_CONDITIONS:
        pi_text = problem["pi"][cond]
        view_prompt_ids[cond] = encode(tokenizer, teacher_msg_fn(problem["question"], pi_text))
        pi_lengths[cond] = len(tokenizer.encode(pi_text, add_special_tokens=False))
    for cond in EXTRA_CONDITIONS:
        pi_text = extra_pi_for_pid[cond]
        view_prompt_ids[cond] = encode(tokenizer, teacher_msg_fn(problem["question"], pi_text))
        pi_lengths[cond] = len(tokenizer.encode(pi_text, add_special_tokens=False))
    longest_prompt = max(len(none_prompt_ids), max(len(p) for p in view_prompt_ids.values()))
    budget = MODEL_MAX_LENGTH - longest_prompt - CONTEXT_SAFETY_MARGIN
    return none_prompt_ids, view_prompt_ids, budget, pi_lengths


# --------------------------------------------------------------------------- #
# Per-(problem, sample) scoring
# --------------------------------------------------------------------------- #


@torch.no_grad()
def score_sample(
    model, tokenizer, pid, sample_index, rec, none_prompt_ids, view_prompt_ids, budget,
    marker_ids, chunk_size, device, min_matched_tokens,
    boundary_ids=None, scoring_horizon=None, pi_lengths=None, token_clip=TOKEN_CLIP,
):
    """Returns a list of output rows for this (problem, sample), or None if omitted (fewer
    than min_matched_tokens matched completion tokens fit under all 7 contexts).

    boundary_ids: think/answer split boundary, passed to token_categories() (default
        [marker_ids["think_close"]], the thinking-on behavior).
    scoring_horizon: additional cap on common_len beyond the context-derived `budget`
        (the direct-response profile uses 1024, the validated training loss horizon).
        None (default) scores the full matched prefix.
    pi_lengths: {condition: exact tokenizer reference-text length} from build_prompts(),
        stored per row as pi_length_tokens (0.0 for condition == "none" or if not provided).
    token_clip: elementwise KL clip for the clipped_kl column (default TOKEN_CLIP).
    """
    if boundary_ids is None:
        boundary_ids = [marker_ids["think_close"]]
    if pi_lengths is None:
        pi_lengths = {}

    completion_ids_full = get_completion_tokens(rec, tokenizer)
    common_len = min(len(completion_ids_full), budget)
    if scoring_horizon is not None:
        common_len = min(common_len, scoring_horizon)
    if common_len < min_matched_tokens:
        return None

    completion_ids = completion_ids_full[:common_len]
    target = torch.tensor(completion_ids, dtype=torch.long, device=device)
    marker_flags = marker_flags_for_tokens(completion_ids, tokenizer)
    regions = [region_of(i, common_len) for i in range(common_len)]
    norm_pos = normalized_positions(common_len)
    categories = token_categories(completion_ids, boundary_ids)
    correct = trajectory_correct(rec)

    contexts = {"none": none_prompt_ids}
    contexts.update(view_prompt_ids)
    hiddens = {c: hidden_slice(model, prompt_ids, completion_ids, device) for c, prompt_ids in contexts.items()}

    float_fields = [
        "logp", "entropy", "eos_prob", "think_close_prob", "answer_transition_prob",
        "full_kl", "clipped_kl",
    ]
    per_cond = {c: {field: torch.empty(common_len, dtype=torch.float32) for field in float_fields} for c in contexts}
    for c in contexts:  # top1 is an index, not a probability
        per_cond[c]["top1"] = torch.empty(common_len, dtype=torch.long)

    marker_cols = [marker_ids["eos"], marker_ids["think_close"], marker_ids["answer_transition"]]

    for cs in range(0, common_len, chunk_size):
        ce = min(cs + chunk_size, common_len)
        tgt_chunk = target[cs:ce]

        # student first: its chunk-local log_probs is the only tensor kept alive across the
        # view loop below (needed for every view's full_kl term against the student).
        logits_s = model.lm_head(hiddens["none"][cs:ce]).float()
        log_probs_s = F.log_softmax(logits_s, dim=-1)
        del logits_s
        probs_s = log_probs_s.exp()
        entropy_s = -(probs_s * log_probs_s).sum(-1)
        top1_s = log_probs_s.argmax(-1)
        logp_s = log_probs_s.gather(-1, tgt_chunk.unsqueeze(-1)).squeeze(-1)
        marker_probs_s = probs_s[:, marker_cols]
        del probs_s
        pc = per_cond["none"]
        pc["logp"][cs:ce] = logp_s.cpu()
        pc["entropy"][cs:ce] = entropy_s.cpu()
        pc["top1"][cs:ce] = top1_s.cpu()
        pc["eos_prob"][cs:ce] = marker_probs_s[:, 0].cpu()
        pc["think_close_prob"][cs:ce] = marker_probs_s[:, 1].cpu()
        pc["answer_transition_prob"][cs:ce] = marker_probs_s[:, 2].cpu()
        pc["full_kl"][cs:ce] = 0.0  # self-KL
        pc["clipped_kl"][cs:ce] = 0.0  # self-KL, clip is a no-op at 0

        for cond in ALL_VIEWS:
            logits_v = model.lm_head(hiddens[cond][cs:ce]).float()
            log_probs_v = F.log_softmax(logits_v, dim=-1)
            del logits_v
            probs_v = log_probs_v.exp()
            entropy_v = -(probs_v * log_probs_v).sum(-1)
            top1_v = log_probs_v.argmax(-1)
            logp_v = log_probs_v.gather(-1, tgt_chunk.unsqueeze(-1)).squeeze(-1)
            marker_probs_v = probs_v[:, marker_cols]
            # Elementwise (position, vocab) forward-KL integrand, matching
            # opsd_forward_kl's `kl_elem` exactly (teacher_prob * (teacher_logp -
            # student_logp)). full_kl sums it unclipped; clipped_kl clamps each entry to
            # token_clip BEFORE the vocab sum, matching the official per-token loss's own
            # `kl_elem.clamp(max=token_clip).sum(-1)` -- see opsd.opsd_loss's docstring.
            kl_elem_v = probs_v * (log_probs_v - log_probs_s)
            full_kl_v = kl_elem_v.sum(-1)
            clipped_kl_v = kl_elem_v.clamp(max=token_clip).sum(-1)
            del probs_v, log_probs_v, kl_elem_v
            pc = per_cond[cond]
            pc["logp"][cs:ce] = logp_v.cpu()
            pc["entropy"][cs:ce] = entropy_v.cpu()
            pc["top1"][cs:ce] = top1_v.cpu()
            pc["eos_prob"][cs:ce] = marker_probs_v[:, 0].cpu()
            pc["think_close_prob"][cs:ce] = marker_probs_v[:, 1].cpu()
            pc["answer_transition_prob"][cs:ce] = marker_probs_v[:, 2].cpu()
            pc["full_kl"][cs:ce] = full_kl_v.cpu()
            pc["clipped_kl"][cs:ce] = clipped_kl_v.cpu()
        del log_probs_s

    stu = per_cond["none"]
    rows = []
    for cond in ["none"] + ALL_VIEWS:
        pc = per_cond[cond]
        for i in range(common_len):
            rows.append(
                {
                    "problem_id": pid,
                    "sample_index": sample_index,
                    "condition": cond,
                    "correct": correct,
                    "position": i,
                    "normalized_position": float(norm_pos[i]),
                    "token_category": categories[i],
                    "region": regions[i],
                    "logp": float(pc["logp"][i]),
                    "logp_student": float(stu["logp"][i]),
                    "log_ratio": 0.0 if cond == "none" else float(pc["logp"][i] - stu["logp"][i]),
                    "entropy": float(pc["entropy"][i]),
                    "entropy_student": float(stu["entropy"][i]),
                    "top1_agree_with_student": bool(pc["top1"][i] == stu["top1"][i]),
                    "is_correction_marker": bool(marker_flags[i]),
                    "eos_prob": float(pc["eos_prob"][i]),
                    "eos_prob_student": float(stu["eos_prob"][i]),
                    "think_close_prob": float(pc["think_close_prob"][i]),
                    "think_close_prob_student": float(stu["think_close_prob"][i]),
                    "answer_transition_prob": float(pc["answer_transition_prob"][i]),
                    "answer_transition_prob_student": float(stu["answer_transition_prob"][i]),
                    "full_kl": float(pc["full_kl"][i]),
                    "token_id": int(completion_ids[i]),
                    "clipped_kl": float(pc["clipped_kl"][i]),
                    "pi_length_tokens": float(pi_lengths.get(cond, 0)),
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=PRIMARY_MODEL, choices=[PRIMARY_MODEL],
                        help="only Qwen3-1.7B has all 6 reference views' frozen generations")
    parser.add_argument(
        "--generations",
        default=os.path.join(ORIG_PREFLIGHT_DIR, "qwen3_1.7b_generations.jsonl"),
        help="preflight no-PI generations jsonl (read-only); all 4 sample_index rows are used",
    )
    parser.add_argument(
        "--calibration-ids",
        default=os.path.join(ORIG_PREFLIGHT_DIR, "calibration_ids.json"),
        help="frozen calibration set (read-only); pass a shard of it to split a run across GPUs",
    )
    parser.add_argument(
        "--output", default=os.path.join(OUT_DIR, "all_rollouts_prefix_scores.parquet"),
        help="output parquet path",
    )
    parser.add_argument("--limit", type=int, default=None, help="only score the first N calibration problems (smoke runs)")
    parser.add_argument(
        "--chunk-size", type=int, default=4096,
        help="sequence chunk size for lm_head/log_softmax/full_kl (bounds peak memory; can go "
             "well above prefix_score.py's default of 2048 here since only the student's "
             "log_probs tensor is kept alive across the per-chunk view loop -- see module docstring)",
    )
    parser.add_argument("--min-matched-tokens", type=int, default=512, help="omission threshold, matches prefix_score.py")
    parser.add_argument(
        "--flush-every", type=int, default=8,
        help="write a parquet row group and drop buffered rows every N problems (bounds peak host RAM; see module docstring)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--student-thinking", dest="student_thinking", action="store_true", default=True,
        help="mode of the SCORED no-PI trajectories in --generations (default: on, the "
             "thinking-on profile). Controls only the think/answer boundary used by "
             "token_categories() -- pass --no-student-thinking for the direct-response "
             "profile, whose --generations file itself must already be thinking-off "
             "(this flag does not change what gets scored, only how think/answer is labeled).",
    )
    parser.add_argument("--no-student-thinking", dest="student_thinking", action="store_false")
    parser.add_argument(
        "--prompt-style", choices=["local", "official"], default="local",
        help="student/teacher prompt template (see build_prompts()): 'local' (default) or "
             "'official' (verbatim official-OPSD templates).",
    )
    parser.add_argument(
        "--scoring-horizon", type=int, default=None,
        help="cap the matched/scored completion prefix to at most this many tokens, on top of "
             "the existing context-derived budget (the direct-response profile uses 1024, "
             "matching the validated training loss horizon; default None scores the full "
             "matched prefix).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from opsd import data as data_mod

    problems = data_mod.load_problems()
    calibration_ids = json.load(open(args.calibration_ids))
    if args.limit:
        calibration_ids = calibration_ids[: args.limit]
    extra_pi = data_mod.load_extra_pi(calibration_ids, EXTRA_CONDITIONS)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    marker_ids = {
        "eos": tokenizer.eos_token_id,
        "think_close": tokenizer.convert_tokens_to_ids("</think>"),
        "answer_transition": tokenizer.convert_tokens_to_ids("boxed"),
    }
    for name, tid in marker_ids.items():
        assert tid is not None and tid >= 0, f"tokenizer has no id for marker {name!r}"

    # flash_attention_2 requires a CUDA GPU; fall back to sdpa (works on CPU too) so
    # --device cpu is genuinely usable for a GPU-free sanity/smoke run.
    attn_impl = "flash_attention_2" if args.device.startswith("cuda") else "sdpa"
    model_dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=model_dtype, attn_implementation=attn_impl
    ).to(args.device)
    model.eval()

    boundary_ids = [marker_ids["think_close"] if args.student_thinking else marker_ids["answer_transition"]]

    rollouts = load_all_no_pi_rollouts(args.generations)
    expected_samples = NO_PI_SAMPLES[args.model]

    ids = [pid for pid in calibration_ids if pid in rollouts]
    missing_problems = [pid for pid in calibration_ids if pid not in rollouts]
    if missing_problems:
        print(f"warning: {len(missing_problems)}/{len(calibration_ids)} calibration problems have "
              f"NO no-PI rollout at all in {args.generations}, e.g. {missing_problems[:3]}")

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    import time
    t_start = time.time()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows_buffer = []
    writer = None
    n_rows = 0
    n_problem_sample_total = 0
    n_problem_sample_omitted = 0
    n_problem_sample_missing = 0
    n_problems_scored = 0
    n_problems_seen = 0

    def flush():
        nonlocal writer, n_rows
        if not rows_buffer:
            return
        table = pa.Table.from_pylist(rows_buffer)
        if writer is None:
            writer = pq.ParquetWriter(args.output, table.schema)
        writer.write_table(table)
        n_rows += len(rows_buffer)
        rows_buffer.clear()

    for pid in ids:
        if pid not in problems:
            print(f"warning: {pid} present in generations but not in load_problems() pool, skipping")
            continue
        none_prompt_ids, view_prompt_ids, budget, pi_lengths = build_prompts(
            tokenizer, problems[pid], extra_pi[pid], prompt_style=args.prompt_style
        )
        any_scored = False
        for sample_index in range(expected_samples):
            n_problem_sample_total += 1
            rec = rollouts[pid].get(sample_index)
            if rec is None:
                n_problem_sample_missing += 1
                continue
            rows = score_sample(
                model, tokenizer, pid, sample_index, rec, none_prompt_ids, view_prompt_ids, budget,
                marker_ids, chunk_size=args.chunk_size, device=args.device,
                min_matched_tokens=args.min_matched_tokens,
                boundary_ids=boundary_ids, scoring_horizon=args.scoring_horizon, pi_lengths=pi_lengths,
            )
            if rows is None:
                n_problem_sample_omitted += 1
                continue
            rows_buffer.extend(rows)
            any_scored = True
        if any_scored:
            n_problems_scored += 1
        n_problems_seen += 1
        if n_problems_seen % args.flush_every == 0:
            flush()
        if n_problems_scored % 50 == 0 and n_problems_scored:
            print(f"  scored {n_problems_scored} problems so far ({n_rows + len(rows_buffer)} rows)")
    flush()
    if writer is not None:
        writer.close()

    elapsed = time.time() - t_start
    n_problem_sample_scored = n_problem_sample_total - n_problem_sample_omitted - n_problem_sample_missing
    print(
        f"problems: {n_problems_scored}/{len(ids)} had >=1 scored sample "
        f"({len(missing_problems)} had no rollout at all)"
    )
    print(
        f"problem x sample slots: {n_problem_sample_scored}/{n_problem_sample_total} scored "
        f"({n_problem_sample_missing} missing from generations file, "
        f"{n_problem_sample_omitted} omitted for < {args.min_matched_tokens} matched tokens)"
    )
    print(f"elapsed: {elapsed:.1f}s ({elapsed / max(n_problems_scored, 1):.1f}s/problem)")
    if args.device.startswith("cuda"):
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        print(f"peak CUDA memory: {peak_gb:.2f} GB allocated, {reserved_gb:.2f} GB reserved")

    summary_path = os.path.splitext(args.output)[0].replace("_prefix_scores", "_summary") + ".json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "views": ALL_VIEWS,
                "expected_samples_per_problem": expected_samples,
                "n_calibration_problems": len(calibration_ids),
                "n_problems_with_any_rollout": len(ids),
                "n_problems_with_no_rollout": len(missing_problems),
                "n_problems_scored": n_problems_scored,
                "n_problem_sample_slots_total": n_problem_sample_total,
                "n_problem_sample_slots_scored": n_problem_sample_scored,
                "n_problem_sample_slots_missing_from_generations": n_problem_sample_missing,
                "n_problem_sample_slots_omitted_matched_length": n_problem_sample_omitted,
                "min_matched_tokens": args.min_matched_tokens,
                "chunk_size": args.chunk_size,
                "flush_every": args.flush_every,
                "n_rows": n_rows,
                "student_thinking": args.student_thinking,
                "prompt_style": args.prompt_style,
                "scoring_horizon": args.scoring_horizon,
                "token_clip": TOKEN_CLIP,
            },
            f,
            indent=2,
        )
    print(f"wrote {args.output} ({n_rows} rows) and {summary_path}")


if __name__ == "__main__":
    main()
    # This stage spawns no child processes, but mirror preflight_ext.py's teardown pattern
    # unconditionally for consistency.
    import psutil

    for child in psutil.Process().children(recursive=True):
        child.terminate()
    psutil.wait_procs(psutil.Process().children(recursive=True), timeout=10)
    for child in psutil.Process().children(recursive=True):
        child.kill()
    sys.stdout.flush()
    os._exit(0)
