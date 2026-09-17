"""Extension of preflight.py / prefix_score.py to the three additional reference views
(gist, summary, clean_solution) on the SAME calibration questions, Qwen/Qwen3-1.7B only,
so the sparse->dense reference spectrum has 6 points instead of 3.

<artifacts>/profiles/calibration/ is a read-only input here; every output goes under
<artifacts>/profiles/calibration_extra_views/. Two stages:

    # Stage 1 (vLLM env): calibration questions x 3 new views x 1 sample, identical teacher
    # wrapper, identical decoding, PI cap 8192 + rescue, same seed formula/model_index/item
    # order as preflight.py.
    CUDA_VISIBLE_DEVICES=<gpu> python profiling/preflight_ext.py \\
        --stage generate --output artifacts/profiles/calibration_extra_views/qwen3_1.7b_ext

    # Stage 2 (training env): teacher-force the SAME no-PI sample-0 completions used by
    # preflight (<artifacts>/profiles/calibration/qwen3_1.7b_generations.jsonl) under the 3 new
    # contexts. The matched common completion prefix is computed over all SEVEN contexts
    # (no-PI + 3 original + 3 new) so these rows stay comparable to (a possibly-truncated-
    # further version of) the original 4-context run; see write_matched_length_report().
    CUDA_VISIBLE_DEVICES=<gpu> python profiling/preflight_ext.py \\
        --stage score --output artifacts/profiles/calibration_extra_views/qwen3_1.7b_ext_prefix_scores.parquet

vLLM / flash-attn imports are deferred into their stage functions so this one file
stays importable under either environment (e.g. for --help).
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd.artifact_layout import PROFILE_CALIBRATION, PROFILE_EXTRA_VIEWS, path as artifact_path  # noqa: E402
from opsd.constants import (  # noqa: E402
    CONTEXT_SAFETY_MARGIN,
    EXTRA_CONDITIONS,
    MODEL_INDEX,
    MODEL_MAX_LENGTH,
    PI_GENERATION_MAX_NEW_TOKENS,
    PRIMARY_MODEL,
    sample_seed,
)
from opsd.constants import CONDITIONS as ORIGINAL_CONDITIONS  # noqa: E402

EXT_DIR = artifact_path(PROFILE_EXTRA_VIEWS)
ORIG_PREFLIGHT_DIR = artifact_path(PROFILE_CALIBRATION)  # read-only from this script


# --------------------------------------------------------------------------- #
# Stage: generate (vLLM env)
# --------------------------------------------------------------------------- #


def build_tasks(problems, extra_pi, calibration_ids, model):
    """One task per (problem, new PI condition), sample_index 0. item_index is this
    list's position -- the SAME frozen <artifacts>/profiles/calibration/calibration_ids.json order
    preflight.py used, so seeds match the original preflight's schedule. Every
    condition at a given item shares one seed, exactly like preflight.py's own PI
    task loop (the prompt text, not the seed, is what varies by condition)."""
    from opsd.prompts import teacher_messages

    model_index = MODEL_INDEX[model]
    tasks = []
    for item_index, pid in enumerate(calibration_ids):
        problem = problems[pid]
        seed = sample_seed(model_index, 0, item_index)
        for condition in EXTRA_CONDITIONS:
            tasks.append(
                {
                    "problem_id": pid,
                    "model_tag": model,
                    "verified_answer": problem["verified_answer"],
                    "condition": condition,
                    "sample_index": 0,
                    "seed": seed,
                    "messages": teacher_messages(problem["question"], extra_pi[pid][condition]),
                }
            )
    return tasks


def drop_done(tasks, done):
    return [t for t in tasks if (t["problem_id"], t["condition"], t["sample_index"]) not in done]


def save_extra_rendered_examples(tokenizer, problems, extra_pi, ids, out_dir):
    """Rendered prompts for the 3 new conditions only (save_rendered_examples in
    opsd.prompts already covers the student prompt and the 3 original PI conditions)."""
    from opsd.prompts import teacher_messages

    os.makedirs(out_dir, exist_ok=True)
    for i, pid in enumerate(ids[:3]):
        p = problems[pid]
        for cond in EXTRA_CONDITIONS:
            text = tokenizer.apply_chat_template(
                teacher_messages(p["question"], extra_pi[pid][cond]),
                add_generation_prompt=True,
                enable_thinking=True,
                tokenize=False,
            )
            with open(os.path.join(out_dir, f"{i:02d}_{pid}_teacher_{cond}.txt"), "w") as f:
                f.write(text)


def print_generation_summary(records):
    by_condition = defaultdict(list)
    for r in records:
        by_condition[r["condition"]].append(r)
    print("\nextension PI generation summary")
    for cond, group in sorted(by_condition.items()):
        n = len(group)
        correct = sum(r["rescued_correct"] if r["rescued"] else r["correct"] for r in group)
        capped = sum(r["hit_length_cap"] for r in group)
        capped_after = sum(
            (r["rescued_finish_reason"] == "length") if r["rescued"] else r["hit_length_cap"] for r in group
        )
        no_answer = sum(r["answer_extracted"] is None for r in group)
        print(
            f"  {cond:<16} n={n:<5} correct={correct / n:.3f} length_cap={capped / n:.3f} "
            f"after_rescue={capped_after / n:.3f} no_answer={no_answer / n:.3f}"
        )


def run_generate(args):
    from opsd import data as data_mod
    from opsd.generation import generate_records, load_jsonl, load_llm, save_jsonl
    from opsd.prompts import save_rendered_examples
    from transformers import AutoTokenizer

    calib_path = args.ids_file or os.path.join(ORIG_PREFLIGHT_DIR, "calibration_ids.json")
    calibration_ids = json.load(open(calib_path))
    if args.limit:
        calibration_ids = calibration_ids[: args.limit]
    print(f"ids: {len(calibration_ids)} problems (source: {calib_path})")

    problems = data_mod.load_problems()
    extra_pi = data_mod.load_extra_pi(calibration_ids, EXTRA_CONDITIONS)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tasks = build_tasks(problems, extra_pi, calibration_ids, args.model)

    out_path = f"{args.output}_generations.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {(r["problem_id"], r["condition"], r["sample_index"]) for r in load_jsonl(out_path)}
        print(f"resuming: {len(done)} generations already in {out_path}")
    tasks = drop_done(tasks, done)

    cap = args.max_new_tokens or PI_GENERATION_MAX_NEW_TOKENS

    if args.dry_run:
        print(f"planned: {len(tasks)} generations, cap {cap}, per condition "
              f"{ {c: sum(1 for t in tasks if t['condition'] == c) for c in EXTRA_CONDITIONS} }")
        return

    if not tasks:
        print("nothing to do (all generations already present)")
        return

    save_rendered_examples(tokenizer, problems, calibration_ids, f"{args.output}_rendered_prompts")
    save_extra_rendered_examples(tokenizer, problems, extra_pi, calibration_ids, f"{args.output}_rendered_prompts")

    llm = load_llm(args.model, gpu_memory_utilization=args.gpu_memory_utilization)
    records = generate_records(llm, tokenizer, tasks, cap)
    save_jsonl(records, out_path, append=True)
    print_generation_summary(records)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
# Stage: score (training env) -- reuses prefix_score.py's machinery directly
# --------------------------------------------------------------------------- #


def score_problem_seven(
    model, tokenizer, pid, problem, extra_pi_for_pid, rec, chunk_size, device, min_matched_tokens
):
    """Like prefix_score.score_problem, but the matched-prefix budget is computed from ALL
    SEVEN contexts (no-PI + 3 original PI + 3 new PI), while forward passes only run for
    the 4 contexts this file actually needs to score (none + the 3 new conditions) -- the
    original three PI conditions' token-level stats already live untouched in
    <artifacts>/profiles/calibration/qwen3_1.7b_prefix_scores.parquet at the ORIGINAL (4-context)
    matched length; only their prompt *lengths* are needed here, to get the correct
    (possibly shorter) 7-context budget.

    Returns (rows, matched_info) or None if omitted (fewer than min_matched_tokens
    matched completion tokens fit under all seven contexts)."""
    import torch

    from prefix_score import chunk_stats, get_completion_tokens, hidden_slice, marker_flags_for_tokens, region_of
    from opsd.prompts import encode, student_messages, teacher_messages

    completion_ids_full = get_completion_tokens(rec, tokenizer)

    none_prompt_ids = encode(tokenizer, student_messages(problem["question"]))
    orig_pi_lens = {
        cond: len(encode(tokenizer, teacher_messages(problem["question"], problem["pi"][cond])))
        for cond in ORIGINAL_CONDITIONS
    }
    new_pi_prompt_ids = {
        cond: encode(tokenizer, teacher_messages(problem["question"], extra_pi_for_pid[cond]))
        for cond in EXTRA_CONDITIONS
    }

    longest_prompt_7 = max(
        len(none_prompt_ids), max(orig_pi_lens.values()), max(len(p) for p in new_pi_prompt_ids.values())
    )
    budget_7 = MODEL_MAX_LENGTH - longest_prompt_7 - CONTEXT_SAFETY_MARGIN
    common_len_7 = min(len(completion_ids_full), budget_7)

    # ALSO compute what the ORIGINAL (4-context: none + 3 original PI) rule would give
    # for this same item, purely to document how the two rules differ.
    longest_prompt_4 = max(len(none_prompt_ids), max(orig_pi_lens.values()))
    budget_4 = MODEL_MAX_LENGTH - longest_prompt_4 - CONTEXT_SAFETY_MARGIN
    common_len_4 = min(len(completion_ids_full), budget_4)

    matched_info = {
        "problem_id": pid,
        "completion_tokens_full": len(completion_ids_full),
        "longest_prompt_4ctx": longest_prompt_4,
        "common_len_4ctx": common_len_4,
        "longest_prompt_7ctx": longest_prompt_7,
        "common_len_7ctx": common_len_7,
        "omitted": common_len_7 < min_matched_tokens,
    }

    if common_len_7 < min_matched_tokens:
        return None, matched_info

    completion_ids = completion_ids_full[:common_len_7]
    target_len = common_len_7
    target = torch.tensor(completion_ids, dtype=torch.long, device=device)
    marker_flags = marker_flags_for_tokens(completion_ids, tokenizer)
    regions = [region_of(i, target_len) for i in range(target_len)]

    contexts = {"none": none_prompt_ids}
    contexts.update(new_pi_prompt_ids)
    hiddens = {c: hidden_slice(model, prompt_ids, completion_ids, device) for c, prompt_ids in contexts.items()}

    per_cond = {
        c: {
            "logp": torch.empty(target_len, dtype=torch.float32),
            "entropy": torch.empty(target_len, dtype=torch.float32),
            "top1": torch.empty(target_len, dtype=torch.long),
        }
        for c in contexts
    }

    for cs in range(0, target_len, chunk_size):
        ce = min(cs + chunk_size, target_len)
        tgt_chunk = target[cs:ce]
        for c, h in hiddens.items():
            logits = model.lm_head(h[cs:ce]).float()
            logp, entropy, top1 = chunk_stats(logits, tgt_chunk)
            per_cond[c]["logp"][cs:ce] = logp.cpu()
            per_cond[c]["entropy"][cs:ce] = entropy.cpu()
            per_cond[c]["top1"][cs:ce] = top1.cpu()
            del logits

    none_logp = per_cond["none"]["logp"]
    none_entropy = per_cond["none"]["entropy"]
    none_top1 = per_cond["none"]["top1"]

    rows = []
    for cond in ["none"] + EXTRA_CONDITIONS:
        logp = per_cond[cond]["logp"]
        entropy = per_cond[cond]["entropy"]
        top1 = per_cond[cond]["top1"]
        for i in range(target_len):
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
                    "full_kl": None,  # extension does not compute the optional full-KL subset
                }
            )
    return rows, matched_info


def write_matched_length_report(matched_len_records, n_total, n_omitted, omission_rate, args):
    import numpy as np

    scored = [m for m in matched_len_records if not m["omitted"]]
    lens4 = [m["common_len_4ctx"] for m in scored]
    lens7 = [m["common_len_7ctx"] for m in scored]
    shrink = [a - b for a, b in zip(lens4, lens7)]
    n_shrunk = sum(1 for d in shrink if d > 0)

    original_summary_path = os.path.join(ORIG_PREFLIGHT_DIR, "qwen3_1.7b_prefix_scores_summary.json")
    original_summary = json.load(open(original_summary_path)) if os.path.exists(original_summary_path) else None

    report = {
        "n_total": n_total,
        "n_omitted": n_omitted,
        "omission_rate": omission_rate,
        "min_matched_tokens": args.min_matched_tokens,
        "matched_length_4ctx_on_these_items": {  # what the ORIGINAL preflight's rule gives for these same items
            "mean": float(np.mean(lens4)) if lens4 else None,
            "median": float(np.median(lens4)) if lens4 else None,
        },
        "matched_length_7ctx_actually_used": {
            "mean": float(np.mean(lens7)) if lens7 else None,
            "median": float(np.median(lens7)) if lens7 else None,
        },
        "n_items_with_shrunk_matched_length": n_shrunk,
        "mean_shrink_tokens": float(np.mean(shrink)) if shrink else None,
        "max_shrink_tokens": float(np.max(shrink)) if shrink else None,
        "original_run_prefix_scores_summary": original_summary,
    }
    summary_path = os.path.splitext(args.output)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {summary_path}")
    m4 = report["matched_length_4ctx_on_these_items"]
    m7 = report["matched_length_7ctx_actually_used"]
    print(
        f"matched length (7-context rule): mean={m7['mean']:.1f}, median={m7['median']:.1f}; "
        f"vs the 4-context rule on the same items: mean={m4['mean']:.1f}, median={m4['median']:.1f}; "
        f"{n_shrunk}/{len(scored)} items shrank (mean shrink {report['mean_shrink_tokens']:.1f} tokens, "
        f"max {report['max_shrink_tokens']:.0f})"
    )


def run_score(args):
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from opsd import data as data_mod
    from prefix_score import load_no_pi_rollouts

    problems = data_mod.load_problems()

    calib_path = os.path.join(ORIG_PREFLIGHT_DIR, "calibration_ids.json")
    calibration_ids = json.load(open(calib_path))
    extra_pi = data_mod.load_extra_pi(calibration_ids, EXTRA_CONDITIONS)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(args.device)
    model.eval()

    rollouts = load_no_pi_rollouts(args.generations)
    ids = [pid for pid in calibration_ids if pid in rollouts]
    if args.limit:
        ids = ids[: args.limit]
    print(f"{len(ids)}/{len(calibration_ids)} calibration problems have a no-PI sample-0 rollout in {args.generations}")

    all_rows = []
    matched_len_records = []
    n_total = 0
    n_omitted = 0
    for pid in ids:
        if pid not in problems:
            print(f"warning: {pid} present in generations but not in load_problems() pool, skipping")
            continue
        n_total += 1
        with torch.no_grad():
            rows, matched_info = score_problem_seven(
                model,
                tokenizer,
                pid,
                problems[pid],
                extra_pi[pid],
                rollouts[pid],
                chunk_size=args.chunk_size,
                device=args.device,
                min_matched_tokens=args.min_matched_tokens,
            )
        matched_len_records.append(matched_info)
        if rows is None:
            n_omitted += 1
            continue
        all_rows.extend(rows)
        if n_total % 50 == 0:
            print(f"  scored {n_total}/{len(ids)} ({n_omitted} omitted so far)")

    omission_rate = n_omitted / n_total if n_total else 0.0
    print(
        f"scored {n_total - n_omitted}/{n_total} problems "
        f"({omission_rate:.1%} omitted: fewer than {args.min_matched_tokens} matched completion "
        "tokens over the 7-context budget)"
    )

    df = pd.DataFrame(all_rows)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"wrote {args.output} ({len(all_rows)} rows)")

    write_matched_length_report(matched_len_records, n_total, n_omitted, omission_rate, args)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["generate", "score"])
    parser.add_argument("--model", default=PRIMARY_MODEL, choices=[PRIMARY_MODEL])
    parser.add_argument("--output", required=True, help="[generate] path prefix; [score] exact parquet path")
    parser.add_argument(
        "--generations",
        default=os.path.join(ORIG_PREFLIGHT_DIR, "qwen3_1.7b_generations.jsonl"),
        help="[score] original preflight no-PI generations jsonl (read-only)",
    )
    parser.add_argument("--limit", type=int, default=None, help="use only the first N calibration problems")
    parser.add_argument(
        "--ids-file", default=None,
        help="[generate] path to a JSON list of problem ids, overriding the default "
             "<artifacts>/profiles/calibration/calibration_ids.json. Mirrors preflight.py's --ids-file "
             "([score] stage is unaffected -- it still reads the frozen calibration set).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None, help="[generate] override the PI cap (smoke runs)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="[generate] lower it when another job shares the GPU")
    parser.add_argument("--chunk-size", type=int, default=2048, help="[score] token chunk size for lm_head/log_softmax")
    parser.add_argument("--min-matched-tokens", type=int, default=512, help="[score] omission threshold")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="[generate] print planned task count and exit")
    args = parser.parse_args()

    if args.stage == "generate":
        run_generate(args)
    else:
        run_score(args)


if __name__ == "__main__":
    main()
    # vLLM 0.12's engine teardown can hang forever after all work is done and results are
    # on disk; a bare os._exit orphans the EngineCore child, which then squats on the GPU.
    # So terminate children first. For the `score` stage this is a no-op (no child
    # processes are spawned there), so it is safe to run unconditionally.
    import psutil

    for child in psutil.Process().children(recursive=True):
        child.terminate()
    psutil.wait_procs(psutil.Process().children(recursive=True), timeout=10)
    for child in psutil.Process().children(recursive=True):
        child.kill()
    sys.stdout.flush()
    os._exit(0)
