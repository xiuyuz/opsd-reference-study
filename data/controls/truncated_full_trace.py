"""Trace openings: Full Trace cut to each problem's own view length.

For problem P and target view V in {key_points, clean_solution, summary}, take P's own
full_trace text, tokenize it with the Qwen3-1.7B tokenizer, and keep the first
len(tokenize(P's own V text)) tokens. Truncation is at a token boundary (decoded back with
skip_special_tokens=True), so the text may end mid-sentence; the canonical answer section
at the end of the trace is removed. A problem whose full_trace is not strictly longer than
its own target-view text is excluded from the usable id set rather than kept at full
length. CPU only, deterministic.

Output (per --target-view):
    <OPSD_ARTIFACTS>/reference_controls/trunc_full_trace_to_<target_view>.json
        {problem_id: truncated full_trace text} for every usable problem; pass it to the
        trainer as --pi-override-file with --condition trunc_full_trace_to_<target_view>
    <OPSD_ARTIFACTS>/reference_controls/trunc_full_trace_to_<target_view>_meta.json
        {problem_id: {full_trace_len, target_len, truncated_to, excluded: bool}} for every id
        in the pool (including excluded ones)

    python data/controls/truncated_full_trace.py --target-view clean_solution
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from opsd.constants import CONDITIONS, PRIMARY_MODEL  # noqa: E402
from opsd.data import load_extra_pi, load_problems  # noqa: E402
from opsd import artifact_layout as layout  # noqa: E402

OUT_DIR = layout.path(layout.REFERENCE_CONTROLS)
TARGET_VIEWS = ["key_points", "clean_solution", "summary"]


def pi_text_for(problems, pids, view):
    if view in CONDITIONS:
        return {pid: problems[pid]["pi"][view] for pid in pids}
    extra = load_extra_pi(pids, [view])
    return {pid: extra[pid][view] for pid in pids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-view", required=True, choices=TARGET_VIEWS)
    parser.add_argument("--splits-file",
                        default=os.path.join(REPO_ROOT, "data", "splits", "qwen3_1p7b_splits.json"))
    parser.add_argument("--model", default=PRIMARY_MODEL)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    split = json.load(open(args.splits_file))
    pool = sorted(set(split["train"]) | set(split.get("dev", [])) | set(split.get("test", [])))
    print(f"pool: {len(pool)} problems from {args.splits_file}")

    problems = load_problems()
    missing = [pid for pid in pool if pid not in problems]
    if missing:
        raise SystemExit(f"{len(missing)} split ids not in load_problems()'s pool, e.g. {missing[:3]}")

    full_trace_text = {pid: problems[pid]["pi"]["full_trace"] for pid in pool}
    target_text = pi_text_for(problems, pool, args.target_view)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    truncated, meta = {}, {}
    for pid in pool:
        ft_ids = tokenizer.encode(full_trace_text[pid])
        target_len = len(tokenizer.encode(target_text[pid]))
        excluded = len(ft_ids) <= target_len
        meta[pid] = {
            "full_trace_len": len(ft_ids),
            "target_len": target_len,
            "truncated_to": None if excluded else target_len,
            "excluded": excluded,
        }
        if not excluded:
            truncated[pid] = tokenizer.decode(ft_ids[:target_len], skip_special_tokens=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    pi_path = os.path.join(OUT_DIR, f"trunc_full_trace_to_{args.target_view}.json")
    meta_path = os.path.join(OUT_DIR, f"trunc_full_trace_to_{args.target_view}_meta.json")
    with open(pi_path, "w") as f:
        json.dump(truncated, f)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    n_excluded = sum(1 for m in meta.values() if m["excluded"])
    print(f"wrote {pi_path} ({len(truncated)} usable ids)")
    print(f"wrote {meta_path} ({len(meta)} total ids, {n_excluded} excluded: "
          f"full_trace not longer than its own {args.target_view} text)")
    print(f"condition name for the trainer: trunc_full_trace_to_{args.target_view}")
    if n_excluded:
        excl_ids = [pid for pid, m in meta.items() if m["excluded"]][:5]
        print(f"  excluded ids (first 5): {excl_ids}")
        print("  NOTE: --splits-file's train split must be filtered to the usable id set before "
              "training this control (the excluded ids have no PI text in the override file) "
              "-- e.g. intersect train_ids with json.load(open(pi_path)).keys().")


if __name__ == "__main__":
    main()
