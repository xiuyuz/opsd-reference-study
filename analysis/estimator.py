"""The statistics behind the paper's intervals.

Rescue-aware correctness, the per-problem unit of analysis (Avg@4), one paired
problem-cluster bootstrap shared across cells (10,000 resamples, seed 20260810, stratified by
the split's construction groups, three strata of 128 problems in the reported study), the
seed-aware average over training seeds, Holm's correction, the paired resample over the 90
external benchmark problems, and the seed-level t and Welch intervals reported alongside the
problem bootstrap.

Intervals are percentiles of the resampled differences; the two-sided bootstrap p is
2*min(P(d<=0), P(d>=0)), floored at 1/n_boot. A contrast is "resolved" when its 95% interval
excludes zero. The external tier pools the 90 benchmark problems in one paired resample
(random.Random(0), 10,000 resamples, percentile 2.5/97.5, no regeneration pass).

The functions read the evaluation records written by `evaluation/`, so they apply to a fresh
run as well as to the runs reported in the paper. CPU only.
"""

import collections
import glob
import json
import os
import random
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from opsd import artifact_layout as layout  # noqa: E402
from opsd.constants import CORRECTION_MARKERS  # noqa: E402

SPLITS_FILE = os.path.join(REPO, "data", "splits", "qwen3_1p7b_splits.json")
EXT_DIR = layout.path(layout.EXTERNAL_STEP50)
ANCHOR_DIR = layout.path(layout.EXTERNAL_BASE_ANCHORS)

N_BOOT = 10000
SEED = 20260810
BUCKETS = ["easy", "frontier", "hard_recoverable"]
BENCHES = ["aime24", "aime25", "hmmt25"]
ID_MAPS = {"aime24": layout.path(layout.EXTERNAL_ID_MAP_AIME24, "aime24_id_map_bench_to_anchor.json"),
           "aime25": layout.path(layout.EXTERNAL_STEP50, "aime25_id_map_bench_to_anchor.json"),
           "hmmt25": layout.path(layout.EXTERNAL_STEP50, "hmmt25_id_map_bench_to_anchor.json")}


# --------------------------------- records ---------------------------------- #
def effective_fields(rec):
    """Rescue-aware view of one generation/eval record.

    Uses the rescued_* variant for the "actual" output/length/correctness when
    rec["rescued"] is true and the rescued_output field is present; otherwise the
    plain fields. Also flags whether the record is still length-capped after the
    rescue attempt (or was never capped at all).
    """
    used_rescue = bool(rec.get("rescued")) and "rescued_output" in rec
    if used_rescue:
        generated_tokens = rec.get("rescued_generated_tokens", rec.get("generated_tokens"))
        finish_reason = rec.get("rescued_finish_reason", rec.get("finish_reason"))
        output = rec.get("rescued_output", rec.get("output"))
        output_token_ids = rec.get("rescued_output_token_ids", rec.get("output_token_ids"))
        correct = rec.get("rescued_correct", rec.get("correct"))
        answer_extracted = rec.get("rescued_answer_extracted", rec.get("answer_extracted"))
    else:
        generated_tokens = rec.get("generated_tokens")
        finish_reason = rec.get("finish_reason")
        output = rec.get("output")
        output_token_ids = rec.get("output_token_ids")
        correct = rec.get("correct")
        answer_extracted = rec.get("answer_extracted")

    out = dict(rec)
    out["eff_generated_tokens"] = generated_tokens
    out["eff_finish_reason"] = finish_reason
    out["eff_output"] = output
    out["eff_output_token_ids"] = output_token_ids
    out["eff_correct"] = bool(correct) if correct is not None else None
    out["eff_answer_extracted"] = answer_extracted
    out["hit_length_cap_before"] = bool(rec.get("hit_length_cap"))
    out["still_capped_after"] = finish_reason == "length"
    out["used_rescue"] = used_rescue
    return out


def count_markers(text, markers):
    if not text:
        return 0
    text_low = text.lower()
    total = 0
    for m in markers:
        total += len(re.findall(r"\b" + re.escape(m) + r"\b", text_low))
    return total


def effective(rec):
    """Effective correctness of one record: the rescue-aware outcome as a float."""
    if bool(rec.get("rescued")) and "rescued_output" in rec:
        return bool(rec.get("rescued_correct", rec.get("correct")))
    return bool(rec.get("correct"))


def read_cell(directory, prefix, run, step, expect=1536):
    """Per-problem Avg@4 dict for one (run, step) cell, plus provenance."""
    pat = os.path.join(directory, f"{prefix}b*_{run}_step{step}_test.jsonl")
    files = sorted(glob.glob(pat))
    per = collections.defaultdict(list)
    n = 0
    for path in files:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                per[r["problem_id"]].append(effective(r))
                n += 1
    ok = n == expect and all(len(v) == 4 for v in per.values())
    return ({pid: float(np.mean(v)) for pid, v in per.items()},
            {"glob": pat, "files": len(files), "rows": n, "complete": bool(ok)})


# -------------------------------- resampling -------------------------------- #
class Boot:
    """One resample of the test problems, shared by every cell it is applied to.

    Problems are drawn with replacement within each construction stratum, so every resample
    keeps the split's stratum sizes. The reported study has three strata of 128 problems; the
    sizes are read from the split rather than assumed, so another split works unchanged."""

    def __init__(self, test_ids, group, buckets=None, n_boot=N_BOOT, seed=SEED):
        self.ids = test_ids
        self.pos = {p: i for i, p in enumerate(test_ids)}
        bucket_of = np.array([group[p] for p in test_ids])
        self.buckets = list(buckets) if buckets is not None else list(BUCKETS)
        self.bucket_idx = {b: np.where(bucket_of == b)[0] for b in self.buckets}
        missing = [b for b in self.buckets if len(self.bucket_idx[b]) == 0]
        if missing:
            raise ValueError(f"no test problems in stratum {missing}")
        rng = np.random.default_rng(seed)
        self.idx = np.empty((n_boot, len(test_ids)), dtype=np.int32)
        start = 0
        for b in self.buckets:
            size = len(self.bucket_idx[b])
            self.idx[:, start:start + size] = rng.choice(self.bucket_idx[b], size=(n_boot, size))
            start += size

    def vec(self, per_problem):
        v = np.full(len(self.ids), np.nan)
        for pid, x in per_problem.items():
            if pid in self.pos:
                v[self.pos[pid]] = x
        return v

    def delta(self, a, b, scale=100.0):
        """Paired Δ (a - b) in percentage points, with the shared stratified resample."""
        if a is None or b is None:
            return None
        d = a - b
        if np.isnan(d).any():
            return None
        draws = d[self.idx].mean(axis=1) * scale
        point = float(d.mean() * scale)
        lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        p = float(min(1.0, max(1.0 / N_BOOT,
                               2.0 * min((draws <= 0).mean(), (draws >= 0).mean()))))
        return dict(point=point, ci_lo=lo, ci_hi=hi, p=p, resolved=bool(lo > 0 or hi < 0))

    def mean(self, v, scale=100.0):
        if v is None or np.isnan(v).any():
            return None
        draws = v[self.idx].mean(axis=1) * scale
        return dict(point=float(v.mean() * scale),
                    ci_lo=float(np.percentile(draws, 2.5)),
                    ci_hi=float(np.percentile(draws, 97.5)))


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni; returns (adjusted p, reject) in the input order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj, adj <= alpha


# ------------------------------ external tier ------------------------------- #
def ext_cell(run, bench):
    path = os.path.join(EXT_DIR, f"{run}_step50_{bench}.jsonl")
    if not os.path.exists(path):
        return None, {"file": path, "rows": 0, "complete": False}
    per = collections.defaultdict(int)
    n = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            per[r["problem_id"]] += int(bool(r["correct_full"]))
            n += 1
    ok = n == 360 and len(per) == 30
    return dict(per), {"file": path, "rows": n, "problems": len(per), "complete": ok}


def ext_anchor(bench):
    path = os.path.join(ANCHOR_DIR, f"base_{bench}.json")
    if not os.path.exists(path):
        return None, {"file": path, "present": False}
    a = json.load(open(path))
    return ({r["problem_id"]: r["num_correct"] for r in a["results"]},
            {"file": path, "present": True, "average_at_n_pct": a["average_at_n_pct"]})


def ext_boot(a_counts, b_counts, n=10000, seed=0):
    """a,b: aligned lists of per-problem correct-counts out of 12. Δ in pp, paired resample.

    The two-sided bootstrap p follows the in-domain convention of Boot.delta:
    2 * min(P(d <= 0), P(d >= 0)), floored at 1/n.
    """
    k = len(a_counts)
    point = (sum(a_counts) - sum(b_counts)) / (k * 12) * 100
    rng = random.Random(seed)
    d = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        d.append((sum(a_counts[i] for i in idx) - sum(b_counts[i] for i in idx)) / (k * 12) * 100)
    d.sort()
    lo, hi = d[int(0.025 * n)], d[int(0.975 * n) - 1]
    p = min(1.0, max(1.0 / n, 2.0 * min(sum(x <= 0 for x in d) / n, sum(x >= 0 for x in d) / n)))
    return dict(point=point, ci_lo=lo, ci_hi=hi, p=float(p), resolved=bool(lo > 0 or hi < 0))


# -------------------------------- formatting -------------------------------- #
def fmt(d, unit="pp"):
    if d is None:
        return "n/a"
    star = "*" if d.get("resolved") else ""
    return f"{d['point']:+.2f}{star} [{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}]{unit and ''}"


def pv(d, key="p"):
    """Formatted p (or an em dash) -- a contrast is None whenever its baseline cell is absent."""
    if not d or d.get(key) is None:
        return "—"
    return f"{d[key]:.4f}"


# ------------------------------ seed-level checks ---------------------------- #
def seed_mean_interval(values, conf=0.95):
    """A t interval over per-seed effects, the paper's seed-level check.

    values: one number per training seed, each already averaged over problems. Returns the
    mean, the interval and the degrees of freedom. With one seed there is no interval."""
    from scipy import stats as sps

    v = np.asarray(list(values), dtype=float)
    n = len(v)
    if n < 2:
        return dict(point=float(v[0]) if n else float("nan"), ci_lo=None, ci_hi=None, n=n, df=0)
    se = float(v.std(ddof=1) / np.sqrt(n))
    half = float(sps.t.ppf(0.5 + conf / 2, n - 1)) * se
    return dict(point=float(v.mean()), ci_lo=float(v.mean() - half), ci_hi=float(v.mean() + half),
                n=n, df=n - 1, se=se)


def welch_interval(a_values, b_values, conf=0.95):
    """A Welch interval for the difference of two seed means, unequal counts and variances.

    Used where the two configurations were run for different numbers of seeds."""
    from scipy import stats as sps

    a = np.asarray(list(a_values), dtype=float)
    b = np.asarray(list(b_values), dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Welch needs at least two seeds per configuration")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    df = float((va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)))
    half = float(sps.t.ppf(0.5 + conf / 2, df)) * se
    point = float(a.mean() - b.mean())
    return dict(point=point, ci_lo=point - half, ci_hi=point + half,
                n_a=len(a), n_b=len(b), df=df, se=se)


def seed_aware(per_seed_cells):
    """Average per-problem effects over training seeds before resampling.

    per_seed_cells: an iterable of {problem_id: value}, one per seed. A problem is kept when
    every seed scored it, so the average is over the same problems in every seed."""
    cells = [dict(c) for c in per_seed_cells]
    if not cells:
        return {}
    shared = set(cells[0])
    for c in cells[1:]:
        shared &= set(c)
    return {pid: float(np.mean([c[pid] for c in cells])) for pid in shared}
