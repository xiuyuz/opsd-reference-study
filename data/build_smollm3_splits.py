"""SmolLM3-relative split construction (the file shipped as data/splits/smollm3_3b_splits.json).

Same construction as data/build_splits.py, driven by SmolLM3-3B's own four direct-response
samples per candidate problem and its thinking-enabled teacher generations, over the frozen
Qwen candidate pool:

  * easy / frontier / hard_recoverable buckets by avg@4 (>= 0.75, 0 < p < 0.75, 0) plus at
    least one correct structured-view teacher generation for hard_recoverable;
  * 512 train problems per bucket drawn with random.Random(42) in the same group order;
  * the frozen Qwen dev (192) and test (384) ids are removed from every bucket before
    sampling, so no evaluation id can enter the new train split, and are then copied
    verbatim into the output; their "group" entries are the SmolLM3-relative band
    (easy / frontier / hard, since teacher recoverability was never measured on them).

Inputs (all under OPSD_ARTIFACTS): smollm3/split_pool/eval/*.jsonl (the pool sweep),
smollm3/split_pool/teacher_queue/eval/*.jsonl (the teacher pass),
smollm3/split_pool/pool_ids.json, smollm3/split_pool/teacher_analysis.json and
six_view/candidate_pool.json, plus the shipped Qwen split file.

Outputs: <OPSD_ARTIFACTS>/smollm3/splits_smollm3.json and
<OPSD_ARTIFACTS>/smollm3/splits_smollm3_build_report.json.

Run:  python data/build_smollm3_splits.py
"""

import glob
import json
import os
import random
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from opsd import artifact_layout as layout  # noqa: E402

SMOLLM3_DIR = layout.path(layout.SMOLLM3)
POOL_EVAL_GLOB = layout.path(layout.SMOLLM3_SPLIT_POOL_EVAL, "*.jsonl")
TEACHER_EVAL_GLOB = layout.path(layout.SMOLLM3_SPLIT_POOL_TEACHER_EVAL, "*.jsonl")
QWEN_SPLITS_PATH = os.path.join(REPO_ROOT, "data", "splits", "qwen3_1p7b_splits.json")
CANDIDATE_POOL_PATH = layout.path(layout.SIX_VIEW_BUILD, "candidate_pool.json")
POOL_IDS_PATH = layout.path(layout.SMOLLM3_SPLIT_POOL, "pool_ids.json")
TEACHER_ANALYSIS_PATH = layout.path(layout.SMOLLM3_SPLIT_POOL, "teacher_analysis.json")
SPLITS_OUT_PATH = os.path.join(SMOLLM3_DIR, "splits_smollm3.json")
REPORT_OUT_PATH = os.path.join(SMOLLM3_DIR, "splits_smollm3_build_report.json")

SEED = 42  # same convention as build_splits.py
EXPECTED_SAMPLES = 4  # four unprivileged samples per problem

# every view except answer_only, in the order build_splits.py records them
STRUCTURED_VIEWS = ["key_points", "full_trace", "gist", "summary", "clean_solution"]


def empirical_pass4(flags):
    return sum(flags) / len(flags)


def effective_correct(rec):
    return rec["rescued_correct"] if rec.get("rescued") else rec["correct"]


def load_base_pass4(paths, candidate_ids):
    by_pid = defaultdict(list)
    n_rows = n_matched = 0
    candidate_set = set(candidate_ids)
    for path in paths:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                n_rows += 1
                # pool-eval rows carry the evaluation shard tag in "model", not the model name
                if rec.get("condition") != "no_pi" or not str(rec.get("model", "")).startswith("smpool_"):
                    continue
                if rec["problem_id"] not in candidate_set:
                    continue
                by_pid[rec["problem_id"]].append(1.0 if effective_correct(rec) else 0.0)
                n_matched += 1
    print(f"base-pass4 input: {n_rows} rows read, {n_matched} matched "
          f"(condition=no_pi, model=smpool_*, id in candidate pool), "
          f"{len(by_pid)}/{len(candidate_ids)} candidate ids covered")

    off_count = [pid for pid, flags in by_pid.items() if len(flags) != EXPECTED_SAMPLES]
    if off_count:
        print(f"WARNING: {len(off_count)} candidate ids have != {EXPECTED_SAMPLES} no-PI "
              f"samples (e.g. {off_count[:3]})")
    missing = [pid for pid in candidate_ids if pid not in by_pid]
    if missing:
        print(f"WARNING: {len(missing)}/{len(candidate_ids)} candidate ids have NO base-pass4 "
              f"rows at all (e.g. {missing[:3]}) -- excluded from every bucket below")
    return {pid: empirical_pass4(flags) for pid, flags in by_pid.items()}, n_rows, off_count, missing


def bucket_by_pass4(pass4):
    easy = sorted(pid for pid, p in pass4.items() if p >= 0.75)
    frontier = sorted(pid for pid, p in pass4.items() if 0 < p < 0.75)
    hard = sorted(pid for pid, p in pass4.items() if p == 0.0)
    print(f"buckets from base pass@4: easy(>=0.75)={len(easy)}, "
          f"frontier(0<p<0.75)={len(frontier)}, hard(p=0)={len(hard)}")
    return easy, frontier, hard


def load_hard_recoverable(paths, hard_ids, structured_views):
    hard_set = set(hard_ids)
    structured_set = set(structured_views)
    success = set()
    covered = defaultdict(set)
    n_rows = n_matched = 0
    for path in paths:
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
    print(f"hard-teacher input: {n_rows} rows read, {n_matched} matched, "
          f"{len(covered)}/{len(hard_ids)} hard ids have >=1 attempt, "
          f"{len(success)}/{len(hard_ids)} hard-recoverable")
    uncovered = [pid for pid in hard_ids if pid not in covered]
    if uncovered:
        print(f"WARNING: {len(uncovered)}/{len(hard_ids)} hard ids have ZERO structured-teacher "
              f"attempts (expected exactly the frozen-test/dev hard ids, which the 988-candidate "
              f"teacher pass deliberately excluded) -- treated as not-recoverable")
    return sorted(success), n_rows, len(covered), uncovered


def sample_group(pool_ids, n_train, n_dev, n_test, group_name, rng):
    need = n_train + n_dev + n_test
    if len(pool_ids) < need:
        raise SystemExit(f"group '{group_name}' shortfall: {len(pool_ids)} < {need} -- "
                         "stop and report")
    chosen = list(pool_ids)
    rng.shuffle(chosen)
    chosen = chosen[:need]
    return chosen[:n_train], chosen[n_train:n_train + n_dev], chosen[n_train + n_dev:]


def main():
    # ---- frozen inputs -----------------------------------------------------------------
    qwen = json.load(open(QWEN_SPLITS_PATH))
    frozen_dev, frozen_test = qwen["dev"], qwen["test"]
    frozen_train = qwen["train"]
    frozen_eval_ids = set(frozen_dev) | set(frozen_test)
    assert len(frozen_dev) == 192 and len(frozen_test) == 384

    # Qwen train-split shape to match
    qwen_train_by_group = defaultdict(int)
    for pid in frozen_train:
        qwen_train_by_group[qwen["group"][pid]] += 1
    assert dict(qwen_train_by_group) == {"easy": 512, "frontier": 512, "hard_recoverable": 512}, \
        f"unexpected Qwen train shape: {dict(qwen_train_by_group)}"
    GROUP_COUNTS = {name: (qwen_train_by_group[name], 0, 0)
                    for name in ["easy", "frontier", "hard_recoverable"]}

    pool_ids = json.load(open(POOL_IDS_PATH))["pool"]
    cand = json.load(open(CANDIDATE_POOL_PATH))
    assert sorted(pool_ids) == sorted(cand["candidate_ids"]), \
        "pool_ids.json diverged from the frozen candidate pool"
    candidate_ids = sorted(pool_ids)  # build_candidate_pool() returns sorted(eligible)

    # ---- construction ----------------------------------------------------------------
    pool_eval_paths = sorted(glob.glob(POOL_EVAL_GLOB))
    teacher_eval_paths = sorted(glob.glob(TEACHER_EVAL_GLOB))
    assert pool_eval_paths and teacher_eval_paths

    pass4, n_pool_rows, off_count, missing = load_base_pass4(pool_eval_paths, candidate_ids)
    assert not off_count and not missing, "pool eval incomplete -- refuse to bucket"
    easy, frontier, hard = bucket_by_pass4(pass4)
    hard_recoverable, n_teach_rows, n_hard_covered, uncovered = load_hard_recoverable(
        teacher_eval_paths, hard, STRUCTURED_VIEWS)

    # cross-check against the frozen teacher analysis
    ta = json.load(open(TEACHER_ANALYSIS_PATH))
    assert len(hard_recoverable) == ta["n_hard_recoverable"], \
        (len(hard_recoverable), ta["n_hard_recoverable"])
    assert set(uncovered) <= frozen_eval_ids, \
        "some non-frozen hard id was never attempted by the teacher pass"

    # the frozen dev/test ids never enter the train split
    pools = {
        "easy": [pid for pid in easy if pid not in frozen_eval_ids],
        "frontier": [pid for pid in frontier if pid not in frozen_eval_ids],
        "hard_recoverable": [pid for pid in hard_recoverable if pid not in frozen_eval_ids],
    }
    assert len(pools["hard_recoverable"]) == len(hard_recoverable), \
        "teacher pass should never have attempted a frozen eval id"
    excl = {name: {"before": len(src), "after": len(pools[name]),
                   "removed_frozen_dev_test": len(src) - len(pools[name])}
            for name, src in [("easy", easy), ("frontier", frontier),
                              ("hard_recoverable", hard_recoverable)]}

    rng = random.Random(SEED)
    train, group_map = [], {}
    per_group_train = {}
    for name in ["easy", "frontier", "hard_recoverable"]:  # same group order as build_splits.py
        n_train, n_dev, n_test = GROUP_COUNTS[name]
        g_train, g_dev, g_test = sample_group(sorted(pools[name]), n_train, n_dev, n_test,
                                              name, rng)
        assert not g_dev and not g_test
        train.extend(g_train)
        per_group_train[name] = len(g_train)
        for pid in g_train:
            group_map[pid] = name

    train.sort(key=lambda pid: (group_map[pid], pid))

    # re-stratification of the frozen dev/test ids by SmolLM3's own pass@4
    def smollm3_band(pid):
        p = pass4[pid]
        return "easy" if p >= 0.75 else ("frontier" if p > 0 else "hard")
    for pid in frozen_dev + frozen_test:
        group_map[pid] = smollm3_band(pid)

    result = {  # same fields as the Qwen split file, same order
        "train": train,
        "dev": list(frozen_dev),
        "test": list(frozen_test),
        "group": group_map,
        "base_pass4": {pid: pass4[pid] for pid in train + frozen_dev + frozen_test},
        "structured_views": STRUCTURED_VIEWS,
        "n_candidates": len(candidate_ids),
        "n_prior_excluded": cand["n_prior_excluded"],
        "seed": SEED,
        "base_pass4_generations_source": os.path.abspath(os.path.dirname(pool_eval_paths[0])),
        "hard_teacher_generations_source": os.path.abspath(os.path.dirname(teacher_eval_paths[0])),
    }

    overlap_qwen_train = sorted(set(train) & set(frozen_train))
    assert not (set(train) & frozen_eval_ids), "frozen dev/test id leaked into train"
    assert len(train) == len(frozen_train) == 1536

    report = {
        "construction": "data/build_smollm3_splits.py: the construction of data/build_splits.py "
                        "driven by SmolLM3-3B's own direct-response outcomes and teacher pass",
        "seed": SEED,
        "inputs": {
            "pool_eval_rows": n_pool_rows,
            "pool_eval_files": len(pool_eval_paths),
            "teacher_eval_rows": n_teach_rows,
            "teacher_eval_files": len(teacher_eval_paths),
            "n_candidates": len(candidate_ids),
        },
        "buckets_full_pool": {"easy_ge_0.75": len(easy), "frontier_0_lt_p_lt_0.75": len(frontier),
                              "hard_eq_0": len(hard),
                              "hard_recoverable": len(hard_recoverable),
                              "hard_unrecoverable": n_hard_covered - len(hard_recoverable),
                              "hard_never_attempted_frozen_eval": len(uncovered)},
        "exclusions": {
            "frozen_qwen_dev_excluded_from_train": len(frozen_dev),
            "frozen_qwen_test_excluded_from_train": len(frozen_test),
            "per_group_pool_accounting": excl,
        },
        "train": {"total": len(train), "per_group": per_group_train,
                  "overlap_with_frozen_qwen_train": len(overlap_qwen_train),
                  "fresh_ids_never_in_any_qwen_split": len(train) - len(overlap_qwen_train)},
        "dev_test": {
            "policy": "frozen Qwen ids copied unchanged; 'group' holds the "
                      "SmolLM3-relative band; pass@4==0 eval ids are labeled 'hard' "
                      "(teacher recoverability is not measured on the frozen eval ids)",
            "dev_band_counts": {b: sum(1 for pid in frozen_dev if group_map[pid] == b)
                                for b in ["easy", "frontier", "hard"]},
            "test_band_counts": {b: sum(1 for pid in frozen_test if group_map[pid] == b)
                                 for b in ["easy", "frontier", "hard"]},
        },
        "qwen_split_untouched": True,
    }

    os.makedirs(SMOLLM3_DIR, exist_ok=True)
    with open(SPLITS_OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    with open(REPORT_OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {SPLITS_OUT_PATH}: train={len(train)} dev={len(frozen_dev)} "
          f"test={len(frozen_test)}")
    print(f"wrote {REPORT_OUT_PATH}")
    print(json.dumps(report["train"], indent=2))
    print(json.dumps(report["dev_test"], indent=2))


if __name__ == "__main__":
    main()
