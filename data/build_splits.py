"""Model-relative split construction for Qwen3-1.7B (the file shipped as
data/splits/qwen3_1p7b_splits.json).

Every candidate problem is bucketed by the frozen base's four unprivileged direct-response
samples: easy (3 or 4 of 4 correct), frontier (1 or 2 of 4) and hard_recoverable (0 of 4
and at least one correct thinking-enabled teacher generation under a structured reference
view). Each bucket contributes 512 train / 64 dev / 128 test problems, giving
1,536 / 192 / 384.

Stage 1 (CPU, tokenizer only) writes the candidate pool: every problem with all six views
whose longest teacher prompt fits the 1,024-token loss window, minus the profiling
calibration ids and an earlier study's split:

    python data/build_splits.py --stage pool

Stage 2 needs two generation files over that pool (records with problem_id, condition,
sample_index, model, correct and the rescue fields): the base's four direct-response
samples per problem (condition "no_pi"), and one thinking-enabled teacher sample per
structured view for the problems with 0 of 4:

    python data/build_splits.py --stage build \\
        --base-pass4-generations <path> --hard-teacher-generations <path>

Writes <OPSD_ARTIFACTS>/six_view/candidate_pool.json and
<OPSD_ARTIFACTS>/six_view/splits.json.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from opsd.constants import CONDITIONS, EXTRA_CONDITIONS, NO_PI_SAMPLES, PRIMARY_MODEL  # noqa: E402
from opsd.data import load_problems, load_extra_pi, make_eligible  # noqa: E402
from opsd import artifact_layout as layout  # noqa: E402

SIX_VIEW_DIR = layout.path(layout.SIX_VIEW_BUILD)
CANDIDATE_POOL_PATH = os.path.join(SIX_VIEW_DIR, "candidate_pool.json")
SPLITS_OUT_PATH = os.path.join(SIX_VIEW_DIR, "splits.json")

ALL_VIEWS = CONDITIONS + EXTRA_CONDITIONS  # all six views
STRUCTURED_VIEWS = [c for c in ALL_VIEWS if c != "answer_only"]  # every view except answer_only

TRAIN_MAX_NEW_TOKENS = 1024  # the 1,024-token loss window
GROUP_COUNTS = {"easy": (512, 64, 128), "frontier": (512, 64, 128), "hard_recoverable": (512, 64, 128)}
SEED = 42


def load_prior_excluded_ids():
    """Ids kept out of the candidate pool: the 512 profiling calibration ids, and the
    problems an earlier study on this corpus used, when that split file is present. Both
    files are read, never written."""
    calib_path = layout.path(layout.PROFILE_CALIBRATION, "calibration_ids.json")
    calibration_ids = set(json.load(open(calib_path)))

    # An earlier study on the same corpus held out its own problems. Its split file is
    # excluded as well when it is present; a fresh pool has no such file.
    prior_path = layout.path(layout.PRIOR_STUDY_SPLIT)
    prior_ids = set()
    if os.path.exists(prior_path):
        prior = json.load(open(prior_path))
        prior_ids = set(prior["train"]) | set(prior["dev"]) | set(prior["test"])
    else:
        print(f"no earlier-study split at {prior_path}; excluding the calibration ids only")

    return calibration_ids | prior_ids


def build_candidate_pool(train_max_new_tokens=TRAIN_MAX_NEW_TOKENS, model=PRIMARY_MODEL):
    """CPU-only (tokenizer, no CUDA). Returns (candidate_ids sorted, problems dict restricted
    to those ids with all six PI texts attached, prior_excluded count)."""
    problems = load_problems()  # the three base views plus difficulty, for every problem
    prior_excluded = load_prior_excluded_ids()
    remaining = sorted(pid for pid in problems if pid not in prior_excluded)
    print(f"pool: {len(problems)} problems total, {len(prior_excluded)} excluded "
          f"(calibration and earlier-study ids), {len(remaining)} candidates before eligibility filter")

    # make_eligible() reads problem["pi"][c] for every c in `conditions`; load_problems() only
    # ever attaches the base three, so merge the 3 extra views' PI text in for candidates only
    # (not the full 5,319-problem pool -- load_extra_pi() rescans the whole supervision file,
    # keep it to the ids that matter).
    extra_pi = load_extra_pi(remaining, EXTRA_CONDITIONS)
    for pid in remaining:
        problems[pid]["pi"].update(extra_pi[pid])

    candidate_problems = {pid: problems[pid] for pid in remaining}
    eligible = make_eligible(
        candidate_problems, train_max_new_tokens, model=model, conditions=ALL_VIEWS,
    )
    print(f"eligible at {train_max_new_tokens}-token loss horizon across all six views: "
          f"{len(eligible)}/{len(remaining)} ({len(remaining) - len(eligible)} excluded, "
          "longest PI prompt + loss horizon + safety margin > model max length)")
    return sorted(eligible), candidate_problems, len(prior_excluded)


def cmd_pool(args):
    eligible, _problems, n_excluded = build_candidate_pool(model=args.model)
    os.makedirs(SIX_VIEW_DIR, exist_ok=True)
    payload = {
        "candidate_ids": eligible,
        "n_candidates": len(eligible),
        "n_prior_excluded": n_excluded,
        "train_max_new_tokens": TRAIN_MAX_NEW_TOKENS,
        "views_checked_for_eligibility": ALL_VIEWS,
        "model": args.model,
    }
    with open(CANDIDATE_POOL_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    n_needed = sum(sum(v) for v in GROUP_COUNTS.values())
    print(f"wrote {CANDIDATE_POOL_PATH}: {len(eligible)} candidates "
          f"(need {n_needed} across all 3 groups combined once base-pass@4 + "
          "hard-teacher-success are available)")


def empirical_pass4(flags):
    """Fraction of the (exactly NO_PI_SAMPLES[model] = 4) unprivileged samples that graded
    correct, i.e. avg@4, not the combinatorial unbiased pass@k estimator: at n == k == 4 that
    estimator 1 - comb(n-c, k) / comb(n, k) is binary (comb(x, 4) == 0 for every x < 4, so any
    c in {1, 2, 3} yields 1.0) and can never land in the (0, 0.75) frontier band. The bucket
    thresholds (>= 0.75 / 0 < . < 0.75 / == 0) are only reachable by c/4 in
    {0, 0.25, 0.5, 0.75, 1.0}, so that is what "pass@4" means here despite the name."""
    return sum(flags) / len(flags)


def effective_correct(rec):
    return rec["rescued_correct"] if rec.get("rescued") else rec["correct"]


def load_base_pass4(path, candidate_ids, model):
    if not os.path.exists(path):
        raise SystemExit(
            f"--base-pass4-generations {path} does not exist.\n"
            "Run --stage pool first to get the candidate id list this generation job must cover."
        )
    by_pid = defaultdict(list)
    n_rows = n_matched = 0
    candidate_set = set(candidate_ids)
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            n_rows += 1
            # generation records carry the model name in the "model" field
            if rec.get("condition") != "no_pi" or rec.get("model") != model:
                continue
            if rec["problem_id"] not in candidate_set:
                continue
            by_pid[rec["problem_id"]].append(1.0 if effective_correct(rec) else 0.0)
            n_matched += 1
    print(f"base-pass4 input: {n_rows} rows read, {n_matched} matched "
          f"(condition=no_pi, model={model}, id in candidate pool), "
          f"{len(by_pid)}/{len(candidate_ids)} candidate ids covered")

    expected_n = NO_PI_SAMPLES[model]
    off_count = [pid for pid, flags in by_pid.items() if len(flags) != expected_n]
    if off_count:
        print(f"WARNING: {len(off_count)} candidate ids have != {expected_n} no-PI samples "
              f"(e.g. {off_count[:3]}) -- empirical_pass4() still divides by the actual count, "
              "but a count other than 4 means the bucket thresholds (>=0.75/0<.<0.75/==0) no "
              "longer land on the same {0, .25, .5, .75, 1} grid; this may indicate a partial/"
              "resumed generation run, worth checking before trusting the buckets")

    missing = [pid for pid in candidate_ids if pid not in by_pid]
    if missing:
        print(f"WARNING: {len(missing)}/{len(candidate_ids)} candidate ids have NO base-pass4 "
              f"rows at all (e.g. {missing[:3]}) -- excluded from every bucket below")

    return {pid: empirical_pass4(flags) for pid, flags in by_pid.items()}


def bucket_by_pass4(pass4):
    easy = sorted(pid for pid, p in pass4.items() if p >= 0.75)
    frontier = sorted(pid for pid, p in pass4.items() if 0 < p < 0.75)
    hard = sorted(pid for pid, p in pass4.items() if p == 0.0)
    print(f"buckets from base pass@4: easy(>=0.75)={len(easy)}, "
          f"frontier(0<p<0.75)={len(frontier)}, hard(p=0)={len(hard)}")
    return easy, frontier, hard


def load_hard_recoverable(path, hard_ids, structured_views):
    if not os.path.exists(path):
        raise SystemExit(
            f"--hard-teacher-generations {path} does not exist.\n"
            f"{len(hard_ids)} candidate ids have base pass@4 == 0 and need at least one "
            "structured-PI-teacher-conditioned generation checked for correctness before the "
            "hard_recoverable bucket can be built."
        )
    hard_set = set(hard_ids)
    structured_set = set(structured_views)
    success = set()
    covered = defaultdict(set)  # pid -> set of views actually attempted, for the coverage report
    n_rows = n_matched = 0
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            n_rows += 1
            if rec["problem_id"] not in hard_set or rec.get("condition") not in structured_set:
                continue
            n_matched += 1
            covered[rec["problem_id"]].add(rec["condition"])
            if effective_correct(rec):
                success.add(rec["problem_id"])
    print(f"hard-teacher input: {n_rows} rows read, {n_matched} matched "
          f"(id in hard bucket, condition in {sorted(structured_set)}), "
          f"{len(covered)}/{len(hard_ids)} hard ids have >=1 attempt, "
          f"{len(success)}/{len(hard_ids)} hard-recoverable")
    uncovered = [pid for pid in hard_ids if pid not in covered]
    if uncovered:
        print(f"WARNING: {len(uncovered)}/{len(hard_ids)} hard ids have ZERO structured-teacher "
              f"attempts in this file (e.g. {uncovered[:3]}) -- treated as not-recoverable, "
              "not as missing data; rerun the generation job wider if this looks incomplete")
    return sorted(success)


def sample_group(pool_ids, n_train, n_dev, n_test, group_name, rng):
    import random as random_mod

    need = n_train + n_dev + n_test
    if len(pool_ids) < need:
        print(f"WARNING: group '{group_name}' has only {len(pool_ids)} eligible candidates, "
              f"need {need} ({n_train}/{n_dev}/{n_test}) -- taking all available and shrinking "
              "this group proportionally (train:dev:test kept at the same ratio)")
        ratio = len(pool_ids) / need
        n_train, n_dev = int(n_train * ratio), int(n_dev * ratio)
        n_test = len(pool_ids) - n_train - n_dev
    chosen = list(pool_ids)
    rng.shuffle(chosen)
    chosen = chosen[:need if len(pool_ids) >= need else len(pool_ids)]
    return chosen[:n_train], chosen[n_train:n_train + n_dev], chosen[n_train + n_dev:]


def cmd_build(args):
    eligible, problems, n_prior_excluded = build_candidate_pool(model=args.model)
    pass4 = load_base_pass4(args.base_pass4_generations, eligible, args.model)
    easy, frontier, hard = bucket_by_pass4(pass4)
    hard_recoverable = load_hard_recoverable(args.hard_teacher_generations, hard, args.structured_views)

    groups = {"easy": easy, "frontier": frontier, "hard_recoverable": hard_recoverable}
    rng = __import__("random").Random(args.seed)

    train, dev, test, group_map = [], [], [], {}
    for name, pool_ids in groups.items():
        n_train, n_dev, n_test = GROUP_COUNTS[name]
        g_train, g_dev, g_test = sample_group(pool_ids, n_train, n_dev, n_test, name, rng)
        train.extend(g_train)
        dev.extend(g_dev)
        test.extend(g_test)
        for pid in g_train + g_dev + g_test:
            group_map[pid] = name

    train.sort(key=lambda pid: (group_map[pid], pid))
    dev.sort(key=lambda pid: (group_map[pid], pid))
    test.sort(key=lambda pid: (group_map[pid], pid))

    result = {
        "train": train,
        "dev": dev,
        "test": test,
        "group": group_map,
        "base_pass4": {pid: pass4[pid] for pid in train + dev + test},
        "structured_views": args.structured_views,
        "n_candidates": len(eligible),
        "n_prior_excluded": n_prior_excluded,
        "seed": args.seed,
        "base_pass4_generations_source": os.path.abspath(args.base_pass4_generations),
        "hard_teacher_generations_source": os.path.abspath(args.hard_teacher_generations),
    }
    os.makedirs(SIX_VIEW_DIR, exist_ok=True)
    with open(SPLITS_OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {SPLITS_OUT_PATH}: train={len(train)} dev={len(dev)} test={len(test)}")
    for name in groups:
        n = sum(1 for v in group_map.values() if v == name)
        print(f"  {name}: {n} ids across train+dev+test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pool", "build"], required=True)
    parser.add_argument("--model", default=PRIMARY_MODEL)
    parser.add_argument("--base-pass4-generations", default=None,
                         help="required for --stage build: the base's four unprivileged "
                         "direct-response samples per candidate problem (condition no_pi)")
    parser.add_argument("--hard-teacher-generations", default=None,
                         help="required for --stage build: thinking-enabled teacher generations "
                         "under the structured views for the problems with 0 of 4 correct")
    parser.add_argument("--structured-views", nargs="+", default=STRUCTURED_VIEWS,
                         help=f"views that count as structured for the hard_recoverable group "
                         f"(default: every view except answer_only, {STRUCTURED_VIEWS})")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.stage == "pool":
        cmd_pool(args)
    else:
        if not args.base_pass4_generations or not args.hard_teacher_generations:
            parser.error("--stage build requires --base-pass4-generations and "
                          "--hard-teacher-generations")
        cmd_build(args)
