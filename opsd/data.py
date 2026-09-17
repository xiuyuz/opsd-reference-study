"""Load AMPLE-Math rows and their difficulty, and build the profiling calibration set
and the context-eligibility filter.

Rows come from the Hugging Face dataset AMPLE_MATH_HF_ID (split AMPLE_MATH_HF_SPLIT), or
from a local jsonl when AMPLE_MATH_JSONL is set. Two row layouts are accepted and both are
normalized to one record per (problem, reference view):

  wide (the released dataset): one row per problem with problem_id, question, answer,
    difficulty, and one supervision_<condition> column per view holding that view's body,
    condition in answer_only, gist, key_points, summary, clean_solution, full_trace.
  long: one row per (problem, condition) with problem_id, condition, prompt (or
    question), answer, output (the view body) and optionally target and difficulty.

Normalized record fields:
    problem_id   stable identifier of the problem
    condition    the reference view
    prompt       the problem statement (the student's question)
    answer       the verified final answer (string)
    output       the view body (empty for answer_only)
    target       the reference text the teacher sees: reference_text(condition, output,
                 answer) = the body followed by "\\n\\nThe final answer is \\boxed{answer}."
                 (answer_only: the boxed-answer sentence alone). A long-format row that
                 already carries target keeps it verbatim.
    difficulty   per-problem difficulty score when the rows carry one; otherwise set
                 AMPLE_MATH_DIFFICULTY to a JSON file {"item_difficulties": {problem_id: float}}

CLI:
    python -m opsd.data --stage calibration
    python -m opsd.data --stage eligible --train-max-new-tokens 8192   # writes the eligible id list
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

if __name__ == "__main__" and __package__ is None:  # allow "python opsd/data.py" as well as "-m opsd.data"
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from opsd.artifact_layout import PRIOR_STUDY_SPLIT, PROFILE_CALIBRATION, path as artifact_path
from opsd.constants import (
    ALL_CONDITIONS,
    AMPLE_MATH_DIFFICULTY,
    AMPLE_MATH_HF_ID,
    AMPLE_MATH_HF_SPLIT,
    AMPLE_MATH_JSONL,
    CONDITIONS,
    CONTEXT_SAFETY_MARGIN,
    MODEL_MAX_LENGTH,
    PRIMARY_MODEL,
)

ANSWER_SUFFIX = "The final answer is \\boxed{{{answer}}}."


def reference_text(condition, body, answer):
    """The reference text the teacher sees for one view: the view body followed by the
    canonical boxed-answer sentence; answer_only is that sentence alone."""
    suffix = ANSWER_SUFFIX.format(answer=answer)
    if condition == "answer_only":
        return suffix
    return f"{body}\n\n{suffix}"


def normalize_rows(raw_rows):
    """Yield normalized (problem, condition) records from rows in either layout."""
    for row in raw_rows:
        if "condition" in row:  # long layout
            out = dict(row)
            if "prompt" not in out and "question" in out:
                out["prompt"] = out["question"]
            if not out.get("target"):
                out["target"] = reference_text(out["condition"], out.get("output") or "", out["answer"])
            yield out
            continue
        for cond in ALL_CONDITIONS:  # wide layout: one row per problem
            body = row.get(f"supervision_{cond}")
            if body is None:
                continue
            yield {
                "problem_id": row["problem_id"],
                "condition": cond,
                "prompt": row["question"],
                "answer": row["answer"],
                "output": body,
                "target": reference_text(cond, body, row["answer"]),
                "difficulty": row.get("difficulty"),
            }


def iter_supervision_rows():
    """Yield every AMPLE-Math (problem, condition) record as a dict (see the module docstring).

    AMPLE_MATH_JSONL, when set, is read line by line; otherwise the Hugging Face dataset
    AMPLE_MATH_HF_ID is loaded with `datasets` (cached under HF_HOME) and its
    AMPLE_MATH_HF_SPLIT split (falling back to the first split) is used."""
    if AMPLE_MATH_JSONL:
        def raw():
            with open(AMPLE_MATH_JSONL) as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        yield from normalize_rows(raw())
        return
    from datasets import load_dataset

    try:
        dataset = load_dataset(AMPLE_MATH_HF_ID)
    except Exception as exc:
        raise RuntimeError(
            f"could not load the dataset {AMPLE_MATH_HF_ID}: {exc}. Set AMPLE_MATH_JSONL to a "
            "local copy of the rows, or AMPLE_MATH_HF_ID to another dataset id.") from exc
    split = AMPLE_MATH_HF_SPLIT if AMPLE_MATH_HF_SPLIT in dataset else next(iter(dataset.keys()))
    yield from normalize_rows(dataset[split])


def load_difficulty(rows_by_pid):
    """problem_id -> difficulty score. From AMPLE_MATH_DIFFICULTY (a JSON file holding
    {"item_difficulties": {problem_id: float}} or a flat {problem_id: float} mapping) when
    set, otherwise from the rows' own "difficulty" field."""
    if AMPLE_MATH_DIFFICULTY:
        loaded = json.load(open(AMPLE_MATH_DIFFICULTY))
        return loaded["item_difficulties"] if "item_difficulties" in loaded else loaded
    difficulty = {}
    for pid, conds in rows_by_pid.items():
        for row in conds.values():
            if row.get("difficulty") is not None:
                difficulty[pid] = row["difficulty"]
                break
    if not difficulty:
        raise ValueError(
            "no difficulty scores: the rows carry no 'difficulty' field and AMPLE_MATH_DIFFICULTY "
            "is not set (see docs/ARTIFACTS.md)"
        )
    return difficulty


def load_problems():
    """problem_id -> {"problem_id", "question", "verified_answer", "difficulty",
    "pi": {condition: pi_text for the 3 conditions}}

    Keeps only problems having all three CONDITIONS and a difficulty score.
    """
    rows_by_pid = defaultdict(dict)
    for row in iter_supervision_rows():
        cond = row["condition"]
        if cond not in CONDITIONS:
            continue
        rows_by_pid[row["problem_id"]][cond] = row

    difficulty = load_difficulty(rows_by_pid)

    problems = {}
    for pid, conds in rows_by_pid.items():
        if not all(c in conds for c in CONDITIONS):
            continue
        if pid not in difficulty:
            continue
        any_row = conds[CONDITIONS[0]]
        problems[pid] = {
            "problem_id": pid,
            "question": any_row["prompt"],
            "verified_answer": any_row["answer"],
            "difficulty": difficulty[pid],
            "pi": {c: conds[c]["target"] for c in CONDITIONS},
        }
    return problems


def load_extra_pi(problem_ids, conditions):
    """problem_id -> {condition: pi_text} for reference views OUTSIDE CONDITIONS (the
    gist/summary/clean_solution views), restricted to problem_ids. Purely additive: does
    not change load_problems()'s behavior or return shape -- that loader still attaches only
    answer_only/key_points/full_trace."""
    wanted_ids = set(problem_ids)
    out = {pid: {} for pid in wanted_ids}
    for row in iter_supervision_rows():
        pid = row["problem_id"]
        if pid not in wanted_ids:
            continue
        cond = row["condition"]
        if cond in conditions:
            out[pid][cond] = row["target"]

    missing = [pid for pid, d in out.items() if not all(c in d for c in conditions)]
    if missing:
        raise ValueError(
            f"{len(missing)}/{len(wanted_ids)} problem ids missing one of {conditions} "
            f"in the supervision rows: e.g. {missing[:3]}"
        )
    return out


def octiles(problems):
    """problem_id -> octile in 0..7, equal-count bins over the loaded pool.
    Deterministic: sort by (difficulty, problem_id), then split into 8
    contiguous equal-count groups."""
    ids_sorted = sorted(problems.keys(), key=lambda pid: (problems[pid]["difficulty"], pid))
    n = len(ids_sorted)
    return {pid: (i * 8) // n for i, pid in enumerate(ids_sorted)}


def make_calibration(problems):
    """64 per octile, rng = random.Random(42). Writes
    <artifacts>/profiles/calibration/calibration_ids.json, plus the 64-problem
    (8 per octile) full-vocabulary-KL subset as
    <artifacts>/profiles/calibration/full_kl_ids.json. Returns the calibration list sorted
    by (octile, id)."""
    oct_map = octiles(problems)
    by_octile = defaultdict(list)
    for pid, o in oct_map.items():
        by_octile[o].append(pid)

    rng = random.Random(42)
    calibration = []
    for o in range(8):
        pool = sorted(by_octile[o])
        calibration.extend(rng.sample(pool, 64))

    calibration.sort(key=lambda pid: (oct_map[pid], pid))

    out_path = artifact_path(PROFILE_CALIBRATION, "calibration_ids.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)

    # Full-vocabulary KL is computed on 64 calibration questions only. Take the first
    # 8 of each octile's calibration block, so the subset stays difficulty-balanced
    # and deterministic.
    by_octile_calib = defaultdict(list)
    for pid in calibration:
        by_octile_calib[oct_map[pid]].append(pid)
    full_kl_ids = [pid for o in range(8) for pid in by_octile_calib[o][:8]]
    with open(artifact_path(PROFILE_CALIBRATION, "full_kl_ids.json"), "w") as f:
        json.dump(full_kl_ids, f, indent=2)

    return calibration


def make_eligible(problems, train_max_new_tokens, model=PRIMARY_MODEL, conditions=None):
    """Keep a problem only when its *longest* teacher prompt across the reference
    conditions still leaves room for the training completion cap:

        max_teacher_prompt_tokens + TRAIN_MAX_NEW_TOKENS + CONTEXT_SAFETY_MARGIN
            <= MODEL_MAX_LENGTH

    Applied once, before splits, so every run trains on the same examples and the densest
    view is never the only one being truncated. Returns the eligible ids sorted.

    conditions (default None -> constants.CONDITIONS): the set of reference conditions to
    check. A six-view split must pass ALL_CONDITIONS explicitly -- the caller is
    responsible for having merged each extra condition's text into problem["pi"] first
    (via load_extra_pi()), since load_problems() itself only ever attaches the 3."""
    from transformers import AutoTokenizer  # only this stage needs a tokenizer

    from opsd.prompts import encode, teacher_messages

    conditions = conditions if conditions is not None else CONDITIONS
    tokenizer = AutoTokenizer.from_pretrained(model)
    budget = MODEL_MAX_LENGTH - train_max_new_tokens - CONTEXT_SAFETY_MARGIN

    eligible = []
    for i, (pid, problem) in enumerate(sorted(problems.items())):
        longest = max(
            len(encode(tokenizer, teacher_messages(problem["question"], problem["pi"][c])))
            for c in conditions
        )
        if longest <= budget:
            eligible.append(pid)
        if (i + 1) % 500 == 0:
            print(f"  tokenized {i + 1}/{len(problems)} problems, {len(eligible)} eligible so far")
    return eligible




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["calibration", "eligible"], required=True)
    parser.add_argument(
        "--eligible-file",
        default=artifact_path("eligible_ids.json"),
        help="where the eligible stage writes its JSON list of problem ids",
    )
    parser.add_argument(
        "--train-max-new-tokens",
        type=int,
        default=None,
        help="the training completion cap the eligibility filter is computed against",
    )
    args = parser.parse_args()

    problems = load_problems()
    print(f"loaded pool: {len(problems)} problems")

    if args.stage == "calibration":
        calibration = make_calibration(problems)
        oct_map = octiles(problems)
        counts = Counter(oct_map[pid] for pid in calibration)
        print(f"calibration size: {len(calibration)}")
        print(f"per-octile counts: {dict(sorted(counts.items()))}")
        full_kl_ids = json.load(open(artifact_path(PROFILE_CALIBRATION, "full_kl_ids.json")))
        print(f"full-KL subset: {len(full_kl_ids)} ids, "
              f"per-octile {dict(sorted(Counter(oct_map[pid] for pid in full_kl_ids).items()))}")
    else:
        if args.train_max_new_tokens is None:
            parser.error("--train-max-new-tokens is required for --stage eligible")
        eligible = make_eligible(problems, args.train_max_new_tokens)
        os.makedirs(os.path.dirname(args.eligible_file) or ".", exist_ok=True)
        with open(args.eligible_file, "w") as f:
            json.dump(eligible, f, indent=2)
        excluded = len(problems) - len(eligible)
        print(f"eligible at cap {args.train_max_new_tokens}: {len(eligible)}/{len(problems)} "
              f"({excluded} excluded) -> {args.eligible_file}")
        print("now build the splits: python data/build_splits.py")
