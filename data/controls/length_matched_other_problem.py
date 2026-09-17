"""Length-matched other-problem reference: each problem receives another problem's reference
text of the same view and similar token length, while its own question and verified answer
are untouched.

Algorithm (deterministic, no randomness):
  1. For every problem in the id pool, take its own reference text under --target-view
     (default clean_solution) and its Qwen3-1.7B token length.
  2. Sort problems by (token_length, problem_id) -- a total order, ties broken by id.
  3. Pair each problem at sorted position i with whichever of its two sorted neighbours (i-1,
     i+1, clamped at the array ends) has the closer token length; a problem can never be
     paired with itself. Ties (equal abs distance) go to i-1.
  4. The paired problem's own target-view text becomes problem P's reference: coherent
     reasoning content, but for the wrong problem (and carrying the other problem's answer).

Output (both keyed by problem_id, over the split's train+dev+test ids):
    <OPSD_ARTIFACTS>/reference_controls/irrelevant_rationale_<target_view>.json
        {problem_id: paired problem's reference text}; pass it to the trainer as
        --pi-override-file with --condition irrelevant_rationale_<target_view>
    <OPSD_ARTIFACTS>/reference_controls/irrelevant_rationale_<target_view>_pairing.json
        {problem_id: {paired_from, own_len, paired_len, abs_len_diff}} -- the audit trail

    python data/controls/length_matched_other_problem.py --target-view clean_solution
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from opsd.constants import CONDITIONS, EXTRA_CONDITIONS, PRIMARY_MODEL  # noqa: E402
from opsd.data import load_extra_pi, load_problems  # noqa: E402
from opsd import artifact_layout as layout  # noqa: E402

OUT_DIR = layout.path(layout.REFERENCE_CONTROLS)


def pi_text_for(problems, pids, view):
    if view in CONDITIONS:
        return {pid: problems[pid]["pi"][view] for pid in pids}
    if view in EXTRA_CONDITIONS:
        extra = load_extra_pi(pids, [view])
        return {pid: extra[pid][view] for pid in pids}
    raise ValueError(f"unknown view {view!r}, not in {CONDITIONS + EXTRA_CONDITIONS}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-view", default="clean_solution",
                        choices=CONDITIONS + EXTRA_CONDITIONS,
                        help="the view this control's length is matched to, and the view it "
                        "is compared against (clean_solution or key_points)")
    parser.add_argument("--splits-file",
                        default=os.path.join(REPO_ROOT, "data", "splits", "qwen3_1p7b_splits.json"),
                        help="id pool to build the control over (train+dev+test of the split file)")
    parser.add_argument("--model", default=PRIMARY_MODEL,
                        help="tokenizer used to measure PI-text token length")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    split = json.load(open(args.splits_file))
    pool = sorted(set(split["train"]) | set(split.get("dev", [])) | set(split.get("test", [])))
    print(f"pool: {len(pool)} problems from {args.splits_file}")

    problems = load_problems()
    missing = [pid for pid in pool if pid not in problems]
    if missing:
        raise SystemExit(f"{len(missing)} split ids not in load_problems()'s pool, e.g. {missing[:3]}")

    pi_text = pi_text_for(problems, pool, args.target_view)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    lengths = {pid: len(tokenizer.encode(pi_text[pid])) for pid in pool}

    order = sorted(pool, key=lambda pid: (lengths[pid], pid))
    n = len(order)
    pairing = {}
    for i, pid in enumerate(order):
        prev_pid = order[i - 1] if i > 0 else None
        next_pid = order[i + 1] if i < n - 1 else None
        if prev_pid is None:
            best = next_pid
        elif next_pid is None:
            best = prev_pid
        else:
            prev_diff = abs(lengths[prev_pid] - lengths[pid])
            next_diff = abs(lengths[next_pid] - lengths[pid])
            best = prev_pid if prev_diff <= next_diff else next_pid  # tie -> prefer prev
        pairing[pid] = {
            "paired_from": best,
            "own_len": lengths[pid],
            "paired_len": lengths[best],
            "abs_len_diff": abs(lengths[best] - lengths[pid]),
        }
        assert best != pid

    irrelevant_pi = {pid: pi_text[pairing[pid]["paired_from"]] for pid in pool}

    os.makedirs(OUT_DIR, exist_ok=True)
    pi_path = os.path.join(OUT_DIR, f"irrelevant_rationale_{args.target_view}.json")
    pairing_path = os.path.join(OUT_DIR, f"irrelevant_rationale_{args.target_view}_pairing.json")
    with open(pi_path, "w") as f:
        json.dump(irrelevant_pi, f)
    with open(pairing_path, "w") as f:
        json.dump(pairing, f, indent=2)

    diffs = [v["abs_len_diff"] for v in pairing.values()]
    diffs_sorted = sorted(diffs)
    print(f"wrote {pi_path} ({len(irrelevant_pi)} ids)")
    print(f"wrote {pairing_path}")
    print(f"abs token-length diff: mean {sum(diffs) / len(diffs):.1f}, "
          f"median {diffs_sorted[len(diffs) // 2]}, max {max(diffs)}")
    self_pairs = sum(1 for pid in pool if pairing[pid]["paired_from"] == pid)
    print(f"self-pairs (must be 0): {self_pairs}")
    print(f"condition name for the trainer: irrelevant_rationale_{args.target_view}")


if __name__ == "__main__":
    main()
