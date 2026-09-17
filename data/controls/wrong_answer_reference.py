"""Wrong-answer reference: the Answer Only view with the verified answer replaced by another
problem's answer in the same format, everything else identical.

Construction (deterministic, no randomness; the pairing of length_matched_other_problem.py
with one added constraint):

  1. For every problem in the pool take its own answer_only text (literally
     "The final answer is \\boxed{<verified answer>}.") and its Qwen3-1.7B token length.
  2. Sort by (token_length, problem_id) -- a total order, ties broken by id.
  3. Pair each problem at sorted position i with whichever of its two sorted neighbours
     (i-1, i+1, clamped) has the closer token length; ties go to i-1.
  4. Added constraint: the Answer Only text is only an answer, and 1,536 train problems
     carry just 845 distinct answers, so a nearest-length neighbour can carry the same
     answer. When the chosen partner's answer grades equal to the problem's own answer
     (grade_answer, not string equality), walk outward along the sorted order, i-1, i+1,
     i-2, i+2, ..., and take the first neighbour whose answer grades unequal.
  5. The partner's own answer_only text becomes P's reference verbatim. P's own question
     and verified answer are untouched.

Outputs (keyed by problem_id, over the split's train+dev+test ids):
    <OPSD_ARTIFACTS>/teacher_controls/pi_answer_only_wrong_answer.json
        {problem_id: partner's answer_only text}; pass it to the trainer as
        --pi-override-file with --condition answer_only_wrong_answer
    <OPSD_ARTIFACTS>/teacher_controls/pi_answer_only_wrong_answer_pairing.json
        {problem_id: {paired_from, own_answer, paired_answer, own_len, paired_len,
                      abs_len_diff, walk_steps}} -- the audit trail

The check that no train problem keeps its own answer is asserted and printed. CPU only.

    python data/controls/wrong_answer_reference.py
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from opsd.constants import PRIMARY_MODEL  # noqa: E402
from opsd.data import load_problems  # noqa: E402
from opsd.math_grader import grade_answer  # noqa: E402
from opsd import artifact_layout as layout  # noqa: E402

VIEW = "answer_only"
SPLITS_FILE = os.path.join(REPO_ROOT, "data", "splits", "qwen3_1p7b_splits.json")
OUT_DIR = layout.path(layout.CONTROL_STUDIES)
PI_PATH = os.path.join(OUT_DIR, "pi_answer_only_wrong_answer.json")
PAIRING_PATH = os.path.join(OUT_DIR, "pi_answer_only_wrong_answer_pairing.json")


def main():
    from transformers import AutoTokenizer

    split = json.load(open(SPLITS_FILE))
    pool = sorted(set(split["train"]) | set(split.get("dev", [])) | set(split.get("test", [])))
    train_ids = split["train"]
    print(f"pool: {len(pool)} problems from {SPLITS_FILE} (train {len(train_ids)})")

    problems = load_problems()
    pi_text = {pid: problems[pid]["pi"][VIEW] for pid in pool}
    answer = {pid: problems[pid]["verified_answer"] for pid in pool}

    tok = AutoTokenizer.from_pretrained(PRIMARY_MODEL)
    lengths = {pid: len(tok.encode(pi_text[pid])) for pid in pool}

    order = sorted(pool, key=lambda pid: (lengths[pid], pid))
    pos = {pid: i for i, pid in enumerate(order)}
    n = len(order)

    pairing = {}
    for i, pid in enumerate(order):
        # nearest length-sorted neighbour, ties to the previous one
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
        # Added constraint: walk outward until the partner's answer is genuinely different.
        walk = 0
        if grade_answer(answer[best], answer[pid]):
            found = None
            for d in range(1, n):
                for j in (i - d, i + d):
                    if 0 <= j < n and j != i:
                        cand = order[j]
                        if not grade_answer(answer[cand], answer[pid]):
                            found = cand
                            break
                if found is not None:
                    walk = d
                    break
            if found is None:
                raise SystemExit(f"no problem in the pool has an answer differing from {pid}'s")
            best = found
        assert best != pid
        pairing[pid] = {
            "paired_from": best,
            "own_answer": answer[pid],
            "paired_answer": answer[best],
            "own_len": lengths[pid],
            "paired_len": lengths[best],
            "abs_len_diff": abs(lengths[best] - lengths[pid]),
            "walk_steps": walk,
        }

    wrong_pi = {pid: pi_text[pairing[pid]["paired_from"]] for pid in pool}

    # ------------------------------------ checks ------------------------------------- #
    kept_own = [pid for pid in train_ids
                if grade_answer(pairing[pid]["paired_answer"], answer[pid])]
    kept_own_text = [pid for pid in train_ids if wrong_pi[pid] == pi_text[pid]]
    contains_own = [pid for pid in train_ids
                    if f"\\boxed{{{answer[pid]}}}" in wrong_pi[pid]]
    fmt_bad = [pid for pid in pool
               if not (wrong_pi[pid].startswith("The final answer is \\boxed{")
                       and wrong_pi[pid].endswith("}."))]

    print(f"CHECK 1: train problems keeping their own answer (graded): "
          f"{len(kept_own)}/{len(train_ids)}")
    print(f"CHECK 2: train problems whose PI text is byte-identical to their own: "
          f"{len(kept_own_text)}/{len(train_ids)}")
    print(f"CHECK 3: train problems whose new PI text still contains \\boxed{{own answer}}: "
          f"{len(contains_own)}/{len(train_ids)}")
    print(f"CHECK 4: pool texts not matching 'The final answer is \\boxed{{...}}.': "
          f"{len(fmt_bad)}/{len(pool)}"
          + (f"  e.g. {fmt_bad[:2]}" if fmt_bad else ""))
    assert not kept_own, f"{len(kept_own)} train problems kept their own answer"
    assert not kept_own_text and not contains_own

    walks = [pairing[pid]["walk_steps"] for pid in pool]
    diffs = [pairing[pid]["abs_len_diff"] for pid in pool]
    n_walked = sum(1 for w in walks if w > 0)
    print(f"pairing: {n_walked}/{len(pool)} needed an outward walk past the nearest neighbour "
          f"(max walk {max(walks)}); abs token-length diff mean "
          f"{sum(diffs) / len(diffs):.2f}, max {max(diffs)}")
    self_pairs = sum(1 for pid in pool if pairing[pid]["paired_from"] == pid)
    print(f"self-pairs (must be 0): {self_pairs}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PI_PATH, "w") as f:
        json.dump(wrong_pi, f)
    with open(PAIRING_PATH, "w") as f:
        json.dump(pairing, f, indent=2)
    print(f"wrote {PI_PATH} ({len(wrong_pi)} ids)")
    print(f"wrote {PAIRING_PATH}")
    print("condition name for the trainer: answer_only_wrong_answer")
    for pid in train_ids[:3]:
        print(f"  example {pid[:16]}: own {pi_text[pid]!r} -> {wrong_pi[pid]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
