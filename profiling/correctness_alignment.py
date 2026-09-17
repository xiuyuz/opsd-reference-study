"""Correctness alignment C_k of each reference view.

    C_k = E[ Delta_k(y+) - Delta_k(y-) ]

for problems with >=1 correct and >=1 incorrect no-PI rollout, where Delta_k(y) is the
per-trajectory teacher(view k)-minus-student log-probability gap on the SAME no-PI
completion y, i.e. reduced/per_group.parquet's `log_ratio_mean` column for condition == k
(log_ratio in the profile schema IS the teacher-minus-student per-token logp gap
(logp - logp_student); per_group's log_ratio_mean is that gap averaged over the
trajectory). This is deliberately NOT full_kl (unsigned divergence magnitude) -- full_kl
is one of the three baselines C_k must beat, so making Delta_k = full_kl would be circular.
log_ratio_mean is signed: positive means the view-k teacher assigns MORE probability than
the student to trajectory y (relative endorsement), negative means less.

Sections:
  1. view-level mean + problem-level bootstrap CI (overall C_k)                    -> alignment_overall.csv
  2. by difficulty octile                                                          -> alignment_by_octile.csv
  3. by completion region (first25/mid50/last25, per_group's source `region` col)  -> alignment_by_region.csv
  4. reasoning (think) vs answer tokens                                            -> alignment_by_think_answer.csv
  5. AUROC for classifying correct trajectories from per-(problem,sample,condition)
     aggregates                                                                    -> auroc_pooled.csv
  6. leave-problem-out comparison: does Delta_k beat teacher accuracy, reference
     length, and total KL as predictors of rollout correctness?                   -> lopo_auroc.csv

Data used:
  - <reduced-dir>/per_group.parquet (one row per problem_id x sample_index x condition;
    condition == "none" is the no-PI self-baseline, dropped below -- correctness alignment
    is defined over the 6 real reference views).
  - the raw all_rollouts_prefix_scores shard(s) (--shards): per_group does NOT carry a
    region x token_category breakdown of log_ratio (only of full_kl), so sections 3/4 need
    one full chunked pass over the raw shards to build it (row-group batches). This is the
    only place this script reads the raw shards; everything else uses per_group.
  - difficulty octile: opsd.data's own octiles() over the full load_problems() pool,
    restricted to the calibration ids -- the same construction used everywhere else, so
    octile numbering (0=easiest..7=hardest) is comparable across the whole project, not a
    bespoke quantile split over just these ids.
  - reference length baseline: word count (str.split()) of the view's reference `target`
    text per (problem_id, condition) from the AMPLE-Math rows. A cheap proxy for token
    count (no tokenizer needed in the analysis env) that ranks the same across views
    (checked below) and is the natural "reference length" covariate: an attribute of
    (problem, view), constant across a problem's no-PI rollouts.
  - teacher accuracy baseline: whether the teacher's OWN reference-conditioned generation
    for (problem_id, condition) was correct, from <artifacts>/profiles/calibration/
    qwen3_1.7b_generations.jsonl (answer_only/key_points/full_trace) and
    <artifacts>/profiles/calibration_extra_views/qwen3_1.7b_ext_generations.jsonl (gist/summary/
    clean_solution). eff_correct convention: rescued_correct if the KEY IS PRESENT AND
    NOT NONE, else correct -- plain `row.get("rescued_correct", row.get("correct"))` is
    WRONG here because rescued_correct is present with value None on every non-rescued row,
    so dict.get's default never triggers. Like reference length, this is a
    (problem, view)-level constant, not a per-rollout value.

Both "teacher accuracy" and "reference length" are therefore *problem-level* covariates
(same value repeated across a problem's rollouts) while Delta_k and full_kl_mean are
*rollout-level* (vary sample-to-sample for the same problem, since each no-PI completion
is a different string). A rollout-level signal can in principle discriminate WITHIN a
problem which of several attempts succeeded; a problem-level covariate cannot -- it can
only ever recover between-problem difficulty structure. That contrast is the substance of
the leave-problem-out test below.

Statistics: problem-level (cluster) bootstrap throughout (resample problem_id with
replacement, n_boot=10000, seed=42, 2.5/97.5 percentile CI), matching fork_correction.py's
convention. Every table reports n_problems (and, for AUROC tables, n_pos/n_neg or n_rows)
alongside each estimate.

CLI:
    python profiling/correctness_alignment.py [--n-boot 10000] [--seed 42]
        [--reduced-dir <dir>] [--shards <parquet> ...] [--out-dir <dir>]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from opsd.artifact_layout import (  # noqa: E402
    PROFILE_CALIBRATION,
    PROFILE_DIRECT,
    PROFILE_EXTRA_VIEWS,
    PROFILE_THINKING,
    path as artifact_path,
)

# Defaults are the thinking-on profile's paths (PROFILE_THINKING); any other profile passes
# --reduced-dir/--shards/--out-dir to point this analysis at a different profile's reduced/
# dir and raw shards. TEACHER_GEN_PATHS is intentionally NOT parameterized: it is the
# teacher's own reference-conditioned generation accuracy, a property of the reference
# views/teacher model, not of the student's mode being profiled, and its sanity check below
# is pinned to the thinking-on preflight's known numbers.
DEFAULT_REDUCED_DIR = artifact_path(PROFILE_THINKING, "reduced")
DEFAULT_SHARDS = [
    artifact_path(PROFILE_THINKING, "all_rollouts_prefix_scores.shard0.parquet"),
    artifact_path(PROFILE_THINKING, "all_rollouts_prefix_scores.shard1.parquet"),
]
DEFAULT_OUT_DIR = artifact_path(PROFILE_THINKING, "analysis", "correctness-alignment")

REDUCED_DIR = DEFAULT_REDUCED_DIR
PER_GROUP_PATH = os.path.join(REDUCED_DIR, "per_group.parquet")
SHARDS = DEFAULT_SHARDS
CALIB_IDS_PATH = artifact_path(PROFILE_CALIBRATION, "calibration_ids.json")
TEACHER_GEN_PATHS = [
    artifact_path(PROFILE_CALIBRATION, "qwen3_1.7b_generations.jsonl"),
    artifact_path(PROFILE_EXTRA_VIEWS, "qwen3_1.7b_ext_generations.jsonl"),
]

OUT_DIR = DEFAULT_OUT_DIR
TABLES_DIR = os.path.join(OUT_DIR, "tables")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

# Density order, matching score_views.py and every other profile analysis script.
PI_VIEWS = ["answer_only", "gist", "key_points", "clean_solution", "summary", "full_trace"]
REGIONS = ["first25", "mid50", "last25"]
CATEGORIES = ["think", "answer"]

N_BOOT = 10000
SEED = 42
SCAN_BATCH_SIZE = 1_000_000


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_calibration_ids():
    return json.load(open(CALIB_IDS_PATH))


def load_per_group():
    df = pd.read_parquet(PER_GROUP_PATH)
    df = df[df["condition"] != "none"].reset_index(drop=True)
    assert set(df["condition"].unique()) == set(PI_VIEWS), df["condition"].unique()
    return df


def load_octile_map(calib_ids):
    """Octiles over the FULL problem pool (opsd.data's load_problems()), then restricted
    to the calibration ids -- not a bespoke quantile split over just these ids."""
    from opsd import data as data_mod

    problems = data_mod.load_problems()
    oct_map_full = data_mod.octiles(problems)
    missing = [pid for pid in calib_ids if pid not in oct_map_full]
    if missing:
        raise ValueError(f"{len(missing)} calibration ids missing from the difficulty pool, e.g. {missing[:3]}")
    return {pid: oct_map_full[pid] for pid in calib_ids}


def load_pi_lengths(calib_ids):
    """word count of the reference view's target text, per (problem_id, condition)."""
    from opsd import data as data_mod

    wanted = set(calib_ids)
    rows = []
    for row in data_mod.iter_supervision_rows():
        pid = row["problem_id"]
        if pid not in wanted:
            continue
        cond = row["condition"]
        if cond not in PI_VIEWS:
            continue
        rows.append((pid, cond, len(row["target"].split())))
    df = pd.DataFrame(rows, columns=["problem_id", "condition", "pi_length_words"])
    expected = len(wanted) * len(PI_VIEWS)
    assert len(df) == expected, f"expected {expected} reference-length rows, got {len(df)}"
    return df


def _eff_correct(row):
    rc = row.get("rescued_correct")
    if rc is not None:
        return bool(rc)
    c = row.get("correct")
    return bool(c) if c is not None else None


def load_teacher_correct(calib_ids):
    """whether the teacher's own reference-conditioned generation for (problem_id, condition)
    was correct (eff_correct convention -- see module docstring for the dict.get pitfall
    this avoids)."""
    wanted = set(calib_ids)
    rows = {}
    for path in TEACHER_GEN_PATHS:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                cond = row["condition"]
                if cond not in PI_VIEWS:
                    continue
                pid = row["problem_id"]
                if pid not in wanted:
                    continue
                rows[(pid, cond)] = _eff_correct(row)
    df = pd.DataFrame(
        [(pid, cond, val) for (pid, cond), val in rows.items()],
        columns=["problem_id", "condition", "teacher_correct"],
    )
    expected = len(wanted) * len(PI_VIEWS)
    assert len(df) == expected, f"expected {expected} teacher-accuracy rows, got {len(df)}"
    assert df["teacher_correct"].isna().sum() == 0
    # Sanity check vs. the reference preflight run's teacher accuracies for the 3 original
    # views, computed over the FULL 512-problem calibration set. When calib_ids is the full
    # set, any deviation beyond the tight 0.005 tolerance signals a real data/loading bug,
    # so keep it a hard assert. When calib_ids is a proper subset (a direct-response
    # profile's per_group naturally covers fewer problems because of the matched-token
    # omission), the subset is not a random sample (short no-PI completions plausibly
    # correlate with problem difficulty), so some extra deviation from the full-population
    # number is expected, not a bug -- downgrade to a printed warning instead of failing.
    known = {"answer_only": 0.971, "key_points": 0.986, "full_trace": 0.982}
    acc = df.groupby("condition")["teacher_correct"].mean()
    full_coverage = len(wanted) == 512
    for cond, expected_acc in known.items():
        got = acc[cond]
        diff = abs(got - expected_acc)
        msg = (f"teacher accuracy sanity check for {cond}: got {got:.4f}, "
               f"the reference preflight run reports {expected_acc:.3f} (diff {diff:.4f})")
        if full_coverage:
            assert diff < 0.005, "FAILED: " + msg
        elif diff >= 0.005:
            print(f"note: {msg} -- tolerance exceeded, but calib_ids is a {len(wanted)}/512 "
                  f"subset (not full coverage), so this is expected selection-bias drift, "
                  f"not asserted")
    return df


def scan_region_category_log_ratio(shards=SHARDS, batch_size=SCAN_BATCH_SIZE):
    """One chunked pass over the raw shards (row-group batches) to get log_ratio summed by
    (problem_id, sample_index, condition, region, token_category) -- the one quantity
    per_group.parquet doesn't carry (it has full_kl by think/answer and by quartile,
    but not log_ratio by region/token_category)."""
    keys = ["problem_id", "sample_index", "condition", "region", "token_category"]
    cols = keys + ["log_ratio"]
    parts = []
    for shard_path in shards:
        pf = pq.ParquetFile(shard_path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            df = batch.to_pandas()
            g = df.groupby(keys, sort=False, observed=True)["log_ratio"].agg(["sum", "size"]).reset_index()
            g.columns = keys + ["log_ratio_sum", "n_tokens"]
            parts.append(g)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.groupby(keys, sort=False, observed=True)[["log_ratio_sum", "n_tokens"]].sum().reset_index()
    combined = combined[combined["condition"] != "none"].reset_index(drop=True)
    combined["log_ratio_mean"] = combined["log_ratio_sum"] / combined["n_tokens"]
    return combined


# --------------------------------------------------------------------------- #
# Correctness alignment C_k
# --------------------------------------------------------------------------- #


def eligible_problem_ids(per_group):
    """Problems with >=1 correct and >=1 incorrect no-PI rollout (correctness is
    condition-independent -- use any one condition's rows)."""
    one_cond = per_group[per_group["condition"] == PI_VIEWS[0]][["problem_id", "correct"]]
    g = one_cond.groupby("problem_id")["correct"].agg(["sum", "count"])
    eligible = g[(g["sum"] >= 1) & (g["sum"] < g["count"])]
    return set(eligible.index)


def cluster_bootstrap_mean(values, n_boot=N_BOOT, seed=SEED):
    """values: 1-D array, one entry per problem (already problem-level). Returns
    (obs_mean, ci_lo, ci_hi, n)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi), n


def facet_diff_series(df_facet, eligible_ids, value_col="log_ratio_mean"):
    """df_facet: rows for ONE view (+ optionally one region/category facet already
    filtered), columns problem_id, correct, value_col. Returns a pandas Series indexed
    by problem_id = mean(value_col | correct) - mean(value_col | incorrect), restricted
    to eligible problems that have >=1 valid (non-missing) row on both sides in this
    facet (a facet can drop rows, e.g. a trajectory with 0 answer-section tokens is
    simply absent from the token_category=='answer' facet)."""
    sub = df_facet[df_facet["problem_id"].isin(eligible_ids)]
    piv = sub.groupby(["problem_id", "correct"])[value_col].mean().unstack("correct")
    if True not in piv.columns or False not in piv.columns:
        return pd.Series(dtype=float)
    diff = (piv[True] - piv[False]).dropna()
    return diff


def alignment_rows_for_facets(view_to_diffs, facet_label, facet_col_name=None, extra=None):
    """view_to_diffs: dict view -> (facet_value -> pd.Series of per-problem diffs).
    If facet_col_name is None, facet_value is used directly (single-facet case)."""
    rows = []
    for view, facets in view_to_diffs.items():
        for facet_value, diffs in facets.items():
            obs, lo, hi, n = cluster_bootstrap_mean(diffs.to_numpy())
            row = {"view": view, facet_label: facet_value, "n_problems": n, "C_k": obs, "ci_lo": lo, "ci_hi": hi}
            if extra:
                row.update(extra)
            rows.append(row)
    return pd.DataFrame(rows)


def compute_overall_alignment(per_group, eligible_ids):
    per_view_diffs = {}
    for view in PI_VIEWS:
        df_v = per_group[per_group["condition"] == view]
        per_view_diffs[view] = facet_diff_series(df_v, eligible_ids)
    rows = []
    for view, diffs in per_view_diffs.items():
        obs, lo, hi, n = cluster_bootstrap_mean(diffs.to_numpy())
        rows.append({"view": view, "n_problems": n, "C_k": obs, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows), per_view_diffs


def compute_octile_alignment(per_view_diffs, oct_map):
    rows = []
    for view, diffs in per_view_diffs.items():
        octiles_for_diffs = diffs.index.map(oct_map)
        for o in range(8):
            sub = diffs[octiles_for_diffs == o]
            obs, lo, hi, n = cluster_bootstrap_mean(sub.to_numpy())
            rows.append({"view": view, "octile": o, "n_problems": n, "C_k": obs, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def compute_region_alignment(region_df, eligible_ids):
    view_to_facets = {}
    for view in PI_VIEWS:
        df_v = region_df[region_df["condition"] == view]
        view_to_facets[view] = {}
        # need `correct` merged in -- caller passes region_df already merged with correct
        for r in REGIONS:
            df_r = df_v[df_v["region"] == r]  # region_df must already have `correct` merged in
            view_to_facets[view][r] = facet_diff_series(df_r, eligible_ids)
    return alignment_rows_for_facets(view_to_facets, "region")


def compute_think_answer_alignment(cat_df, eligible_ids):
    view_to_facets = {}
    for view in PI_VIEWS:
        df_v = cat_df[cat_df["condition"] == view]
        view_to_facets[view] = {}
        for c in CATEGORIES:
            df_c = df_v[df_v["token_category"] == c]
            view_to_facets[view][c] = facet_diff_series(df_c, eligible_ids)
    return alignment_rows_for_facets(view_to_facets, "token_category")


# --------------------------------------------------------------------------- #
# AUROC
# --------------------------------------------------------------------------- #


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_auroc_clustered(scores, labels, problem_ids, n_boot=N_BOOT, seed=SEED):
    """Problem-level cluster bootstrap of AUROC: resample problem_ids with replacement,
    pool that resample's rows (ALL of them, however many a problem has), recompute AUROC.

    The thinking-on profile always has exactly 4 no-PI samples/problem, so the row-index
    groups can be a dense (n_problems, 4) array gathered by problem-index draw -- pure
    vectorization. A direct-response profile can have a ragged 1-4 rows/problem
    (per-sample matched-token omission), so this pads the dense array with a
    per-problem-max sentinel (-1) for missing slots and masks those out before each
    replicate's AUROC call -- same standard nonparametric cluster bootstrap (resample
    clusters with replacement, keep each drawn cluster's actual rows), just not assuming
    a constant cluster size. Dense/constant-count input produces zero sentinel slots."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    problem_ids = np.asarray(problem_ids)
    uniq = np.unique(problem_ids)
    n = len(uniq)
    counts = pd.Series(problem_ids).value_counts()
    k = int(counts.max())
    # stable argsort groups equal problem_ids contiguously, in ascending problem_id order
    # (matching np.unique's ascending output) -- so, read in order, `order` is problem
    # uniq[0]'s rows, then uniq[1]'s, etc. A 2D boolean-mask assignment into rows_per_problem
    # (below) fills in that same row-major order, so no explicit offset bookkeeping is needed.
    order = np.argsort(problem_ids, kind="stable")
    counts_by_problem = counts.reindex(uniq).to_numpy()
    rows_per_problem = np.full((n, k), -1, dtype=np.int64)
    col_idx = np.arange(k)[None, :] < counts_by_problem[:, None]
    rows_per_problem[col_idx] = order

    obs = auroc(scores, labels)
    n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n, size=(n_boot, n))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        rows = rows_per_problem[draw[b]].reshape(-1)
        rows = rows[rows >= 0]
        boot[b] = auroc(scores[rows], labels[rows])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return dict(auroc=obs, ci_lo=float(lo), ci_hi=float(hi), n_problems=n, n_pos=n_pos, n_neg=n_neg)


def compute_pooled_auroc(per_group, subset_ids=None, label=""):
    rows = []
    for view in PI_VIEWS:
        df_v = per_group[per_group["condition"] == view]
        if subset_ids is not None:
            df_v = df_v[df_v["problem_id"].isin(subset_ids)]
        res = bootstrap_auroc_clustered(
            df_v["log_ratio_mean"].to_numpy(), df_v["correct"].to_numpy(), df_v["problem_id"].to_numpy()
        )
        res["view"] = view
        res["subset"] = label
        rows.append(res)
    return pd.DataFrame(rows)[["subset", "view", "n_problems", "n_pos", "n_neg", "auroc", "ci_lo", "ci_hi"]]


# --------------------------------------------------------------------------- #
# Leave-problem-out predictor comparison
# --------------------------------------------------------------------------- #


def fit_logistic_1d(x, y, max_iter=25, l2=1.0):
    """Ridge-penalized 1D logistic regression (z-scored feature + intercept) fit by
    IRLS/Newton. The gradient MUST include the `-l2*beta` penalty term (not just the
    `l2*I` curvature term in the Hessian) -- omitting it leaves the Newton fixed point
    at the unregularized MLE (l2 only damps the step, never shrinks the converged
    solution)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mu_x, sd_x = x.mean(), x.std()
    if sd_x < 1e-12:
        sd_x = 1.0
    xz = (x - mu_x) / sd_x
    X = np.column_stack([np.ones_like(xz), xz])
    beta = np.zeros(2)
    for _ in range(max_iter):
        z = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.maximum(p * (1 - p), 1e-9)
        XtWX = (X * w[:, None]).T @ X + l2 * np.eye(2)
        grad = X.T @ (y - p) - l2 * beta
        delta = np.linalg.solve(XtWX, grad)
        beta = beta + delta
        if np.max(np.abs(delta)) < 1e-10:
            break
    return beta, mu_x, sd_x


def predict_logistic_1d(x, beta, mu_x, sd_x):
    xz = (np.asarray(x, dtype=float) - mu_x) / sd_x
    z = np.clip(beta[0] + beta[1] * xz, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def grouped_kfold_oof(x, y, problem_ids, k=8, seed=SEED):
    """Grouped K-fold (folds partitioned by problem_id, so no problem's rows ever
    appear in both train and test) leave-problem-out cross-validated predictions.

    NOT literal leave-ONE-problem-out (k = n_problems): at k=n_problems, a near-constant/
    skewed covariate's out-of-fold prediction for problem P is mechanically ANTI-correlated
    with P's own true label -- e.g. teacher_correct_num (teacher right on ~97-99% of
    problems) gets LOPO AUROC as low as 0.03-0.19 (grossly inverted) despite its RAW pooled
    AUROC (no fitting at all, just ranking the feature directly) being a perfectly ordinary
    0.52-0.55. The mechanism: literal LOO for a near-constant feature degenerates to (close
    to) the leave-one-out MEAN of y within that value's group, and leave-one-out means are
    mathematically anti-correlated with the excluded point's own value
    (y_hat_LOO(i) = (sum y - y_i)/(n-1), decreasing in y_i) -- this reproduces at
    k=n_problems regardless of ridge strength (l2 shrinks the *magnitude* of every fold's
    prediction toward 0.5 but not the *ranking*, since the shrinkage is applied uniformly)
    and disappears as k shrinks (k=4-8 stays close to the raw/unbiased AUROC for every
    feature tested, including the pathological ones). k=8 is used throughout: large enough
    to be genuinely out-of-sample (64 problems/fold), small enough that this bias is no
    longer dominant. Both the raw and the k=8 AUROC are reported in lopo_auroc.csv so this
    is auditable rather than hidden."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    problem_ids = np.asarray(problem_ids)
    uniq = np.unique(problem_ids)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(uniq)), k)
    oof = np.full(len(x), np.nan)
    for fold in folds:
        mask_test = np.isin(problem_ids, uniq[fold])
        mask_train = ~mask_test
        beta, mu_x, sd_x = fit_logistic_1d(x[mask_train], y[mask_train])
        oof[mask_test] = predict_logistic_1d(x[mask_test], beta, mu_x, sd_x)
    assert not np.isnan(oof).any()
    return oof


def compute_lopo_comparison(per_group, pi_len_df, teacher_df, k_folds=8):
    per_group = per_group.merge(pi_len_df, on=["problem_id", "condition"], how="left")
    per_group = per_group.merge(teacher_df, on=["problem_id", "condition"], how="left")
    assert per_group["pi_length_words"].isna().sum() == 0
    assert per_group["teacher_correct"].isna().sum() == 0
    per_group["log_pi_length_words"] = np.log1p(per_group["pi_length_words"])
    per_group["teacher_correct_num"] = per_group["teacher_correct"].astype(float)

    features = {
        "log_ratio_mean": "Delta_k (teacher-minus-student log-ratio, view k)",
        "full_kl_mean": "total KL (view k)",
        "log_pi_length_words": "reference length (log1p word count, view k)",
        "teacher_correct_num": "teacher accuracy (view k)",
    }

    rows = []
    for view in PI_VIEWS:
        df_v = per_group[per_group["condition"] == view].reset_index(drop=True)
        y = df_v["correct"].to_numpy().astype(float)
        pids = df_v["problem_id"].to_numpy()
        for feat_col, feat_label in features.items():
            x = df_v[feat_col].to_numpy()
            raw_auc = auroc(x, y.astype(bool))
            oof = grouped_kfold_oof(x, y, pids, k=k_folds)
            res = bootstrap_auroc_clustered(oof, y.astype(bool), pids)
            res["raw_auroc"] = raw_auc
            rows.append({
                "view": view,
                "feature": feat_col,
                "feature_label": feat_label,
                "n_problems": res["n_problems"],
                "n_rows": len(df_v),
                "raw_auroc": res["raw_auroc"],
                "lopo_auroc": res["auroc"],
                "ci_lo": res["ci_lo"],
                "ci_hi": res["ci_hi"],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def make_plots(overall_df, octile_df, region_df_tab, cat_df_tab, auroc_all_df, auroc_elig_df, lopo_df):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    views = PI_VIEWS
    x = np.arange(len(views))

    # 1. Overall C_k by view.
    fig, ax = plt.subplots(figsize=(8, 5))
    obs = [overall_df.set_index("view").loc[v, "C_k"] for v in views]
    lo = [overall_df.set_index("view").loc[v, "ci_lo"] for v in views]
    hi = [overall_df.set_index("view").loc[v, "ci_hi"] for v in views]
    err = [[obs[i] - lo[i] for i in range(len(views))], [hi[i] - obs[i] for i in range(len(views))]]
    ax.bar(x, obs, yerr=err, capsize=4, color="#4C72B0")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(views, rotation=30, ha="right")
    ax.set_ylabel(r"$C_k = E[\Delta_k(y^+) - \Delta_k(y^-)]$  (nats/token)")
    ax.set_title("Correctness alignment by reference view\n(problem-level bootstrap 95% CI, n_problems in table)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "correctness_alignment_by_view.png"), dpi=150)
    plt.close(fig)

    # 2. C_k by octile, one line per view.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for v in views:
        sub = octile_df[octile_df["view"] == v].sort_values("octile")
        ax.plot(sub["octile"], sub["C_k"], marker="o", label=v)
        ax.fill_between(sub["octile"], sub["ci_lo"], sub["ci_hi"], alpha=0.12)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("difficulty octile (0=easiest, 7=hardest)")
    ax.set_ylabel(r"$C_k$ (nats/token)")
    ax.set_title("Correctness alignment by difficulty octile\n(shaded = problem-level bootstrap 95% CI; n_problems/octile is small, see table)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "correctness_alignment_by_octile.png"), dpi=150)
    plt.close(fig)

    # 3. C_k by region, grouped bars.
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    colors = {"first25": "#4C72B0", "mid50": "#DD8452", "last25": "#55A868"}
    for i, r in enumerate(REGIONS):
        sub = region_df_tab[region_df_tab["region"] == r].set_index("view").reindex(views)
        offset = (i - 1) * width
        err = [
            (sub["C_k"] - sub["ci_lo"]).to_numpy(),
            (sub["ci_hi"] - sub["C_k"]).to_numpy(),
        ]
        ax.bar(x + offset, sub["C_k"], width, yerr=err, capsize=3, label=r, color=colors[r])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(views, rotation=30, ha="right")
    ax.set_ylabel(r"$C_k$ (nats/token)")
    ax.set_title("Correctness alignment by completion region")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "correctness_alignment_by_region.png"), dpi=150)
    plt.close(fig)

    # 4. C_k think vs answer.
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    colors = {"think": "#4C72B0", "answer": "#DD8452"}
    for i, c in enumerate(CATEGORIES):
        sub = cat_df_tab[cat_df_tab["token_category"] == c].set_index("view").reindex(views)
        offset = (i - 0.5) * width
        err = [
            (sub["C_k"] - sub["ci_lo"]).to_numpy(),
            (sub["ci_hi"] - sub["C_k"]).to_numpy(),
        ]
        ax.bar(x + offset, sub["C_k"], width, yerr=err, capsize=4, label=c, color=colors[c])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(views, rotation=30, ha="right")
    ax.set_ylabel(r"$C_k$ (nats/token)")
    ax.set_title("Correctness alignment: reasoning (think) vs. answer tokens")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "correctness_alignment_think_vs_answer.png"), dpi=150)
    plt.close(fig)

    # 5. Pooled AUROC (all problems vs eligible-only), per view.
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    a_all = auroc_all_df.set_index("view").reindex(views)
    a_elig = auroc_elig_df.set_index("view").reindex(views)
    ax.bar(x - width / 2, a_all["auroc"], width,
           yerr=[(a_all["auroc"] - a_all["ci_lo"]).to_numpy(), (a_all["ci_hi"] - a_all["auroc"]).to_numpy()],
           capsize=3, label="all problems", color="#4C72B0")
    ax.bar(x + width / 2, a_elig["auroc"], width,
           yerr=[(a_elig["auroc"] - a_elig["ci_lo"]).to_numpy(), (a_elig["ci_hi"] - a_elig["auroc"]).to_numpy()],
           capsize=3, label="eligible (mixed-outcome) problems only", color="#C44E52")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(views, rotation=30, ha="right")
    ax.set_ylabel(r"AUROC classifying `correct` from $\Delta_k$(y)")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Pooled AUROC: does $\\Delta_k$ discriminate correct vs. incorrect no-PI rollouts?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "auroc_pooled_by_view.png"), dpi=150)
    plt.close(fig)

    # 6. LOPO AUROC comparison, 4 features x 6 views.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    feat_order = ["log_ratio_mean", "full_kl_mean", "log_pi_length_words", "teacher_correct_num"]
    feat_colors = {"log_ratio_mean": "#4C72B0", "full_kl_mean": "#DD8452",
                   "log_pi_length_words": "#55A868", "teacher_correct_num": "#8172B2"}
    feat_names = {"log_ratio_mean": r"$\Delta_k$ (C_k signal)", "full_kl_mean": "total KL",
                  "log_pi_length_words": "reference length", "teacher_correct_num": "teacher accuracy"}
    width = 0.2
    for i, feat in enumerate(feat_order):
        sub = lopo_df[lopo_df["feature"] == feat].set_index("view").reindex(views)
        offset = (i - 1.5) * width
        err = [
            (sub["lopo_auroc"] - sub["ci_lo"]).to_numpy(),
            (sub["ci_hi"] - sub["lopo_auroc"]).to_numpy(),
        ]
        ax.bar(x + offset, sub["lopo_auroc"], width, yerr=err, capsize=2,
               label=feat_names[feat], color=feat_colors[feat])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(views, rotation=30, ha="right")
    ax.set_ylabel("leave-problem-out AUROC")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Leave-problem-out predictor comparison: does $\\Delta_k$ beat teacher accuracy,\nreference length, and total KL at predicting no-PI rollout correctness?")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "lopo_auroc_comparison.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    global N_BOOT, SEED, REDUCED_DIR, PER_GROUP_PATH, SHARDS, OUT_DIR, TABLES_DIR, PLOTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--reduced-dir", default=DEFAULT_REDUCED_DIR,
        help="dir containing per_group.parquet (default: the thinking-on profile's reduced/ dir; "
             f"{artifact_path(PROFILE_DIRECT, 'reduced')} for the direct-response profile)",
    )
    ap.add_argument(
        "--shards", nargs="+", default=DEFAULT_SHARDS,
        help="raw all_rollouts_prefix_scores shard(s), for the region/token_category "
             "log_ratio rescan (default: the thinking-on profile's shards).",
    )
    ap.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help="output dir for tables/plots/summary (default: the thinking-on profile's analysis dir).",
    )
    args = ap.parse_args()
    N_BOOT, SEED = args.n_boot, args.seed
    REDUCED_DIR = args.reduced_dir
    PER_GROUP_PATH = os.path.join(REDUCED_DIR, "per_group.parquet")
    SHARDS = args.shards
    OUT_DIR = args.out_dir
    TABLES_DIR = os.path.join(OUT_DIR, "tables")
    PLOTS_DIR = os.path.join(OUT_DIR, "plots")

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("loading per_group.parquet ...")
    per_group = load_per_group()
    calib_ids = sorted(per_group["problem_id"].unique())
    # The thinking-on profile's per_group always covers the full 512-problem calibration
    # set. A direct-response profile's per_group can cover fewer (a problem drops out
    # entirely only if every one of its sample x condition slots was omitted for < 512
    # matched tokens at scoring time). Downstream lookups (load_octile_map/load_pi_lengths/
    # load_teacher_correct) all filter by whatever calib_ids is passed in, so they are
    # correct for any subset; only assert an upper bound + non-triviality here rather than
    # requiring exact full coverage.
    assert 0 < len(calib_ids) <= 512, len(calib_ids)
    if len(calib_ids) < 512:
        print(f"note: per_group covers {len(calib_ids)}/512 calibration problems "
              f"(rest fully omitted for < 512 matched tokens on every sample x condition slot)")

    print("loading octile map, reference lengths, teacher accuracy ...")
    oct_map = load_octile_map(calib_ids)
    pi_len_df = load_pi_lengths(calib_ids)
    teacher_df = load_teacher_correct(calib_ids)

    eligible_ids = eligible_problem_ids(per_group)
    print(f"eligible problems (>=1 correct & >=1 incorrect no-PI rollout): {len(eligible_ids)} / {len(calib_ids)}")

    # --- 1/2: overall + octile alignment ---
    print("computing overall + by-octile correctness alignment ...")
    overall_df, per_view_diffs = compute_overall_alignment(per_group, eligible_ids)
    octile_df = compute_octile_alignment(per_view_diffs, oct_map)

    overall_df.to_csv(os.path.join(TABLES_DIR, "alignment_overall.csv"), index=False)
    octile_df.to_csv(os.path.join(TABLES_DIR, "alignment_by_octile.csv"), index=False)

    # --- 3/4: region + think/answer alignment (needs raw-shard scan) ---
    print("scanning raw shards for log_ratio by region x token_category (one pass) ...")
    region_cat = scan_region_category_log_ratio(shards=SHARDS)  # explicit: SHARDS may have been
    # overridden by --shards above, and the function's own default arg was bound at module-load
    # time (before that override), so it must be passed explicitly here.
    correct_lookup = per_group[per_group["condition"] == PI_VIEWS[0]][["problem_id", "sample_index", "correct"]]
    # correct is condition-independent; merge onto the region/category breakdown by (problem_id, sample_index).
    region_cat = region_cat.merge(correct_lookup, on=["problem_id", "sample_index"], how="left")
    assert region_cat["correct"].isna().sum() == 0

    print("computing by-region and think/answer correctness alignment ...")
    region_alignment = compute_region_alignment(region_cat, eligible_ids)
    think_answer_alignment = compute_think_answer_alignment(region_cat, eligible_ids)
    region_alignment.to_csv(os.path.join(TABLES_DIR, "alignment_by_region.csv"), index=False)
    think_answer_alignment.to_csv(os.path.join(TABLES_DIR, "alignment_by_think_answer.csv"), index=False)

    # --- 5: pooled AUROC ---
    print("computing pooled AUROC (all problems, and eligible-only) ...")
    auroc_all_df = compute_pooled_auroc(per_group, subset_ids=None, label="all_512_problems")
    auroc_elig_df = compute_pooled_auroc(per_group, subset_ids=eligible_ids, label="eligible_mixed_outcome_only")
    auroc_all_df.to_csv(os.path.join(TABLES_DIR, "auroc_pooled_all.csv"), index=False)
    auroc_elig_df.to_csv(os.path.join(TABLES_DIR, "auroc_pooled_eligible.csv"), index=False)
    auroc_combined = pd.concat([auroc_all_df, auroc_elig_df], ignore_index=True)
    auroc_combined.to_csv(os.path.join(TABLES_DIR, "auroc_pooled.csv"), index=False)

    # --- 6: leave-problem-out predictor comparison ---
    print("running leave-problem-out predictor comparison (grouped 8-fold, 6 views x 4 features) ...")
    lopo_df = compute_lopo_comparison(per_group, pi_len_df, teacher_df)
    lopo_df.to_csv(os.path.join(TABLES_DIR, "lopo_auroc.csv"), index=False)

    # win table: for each view, does Delta_k's LOPO AUROC beat the other 3 features' point estimate?
    piv = lopo_df.pivot(index="view", columns="feature", values="lopo_auroc").reindex(PI_VIEWS)
    win_rows = []
    for view in PI_VIEWS:
        r = piv.loc[view]
        beats = {
            "beats_full_kl": bool(r["log_ratio_mean"] > r["full_kl_mean"]),
            "beats_pi_length": bool(r["log_ratio_mean"] > r["log_pi_length_words"]),
            "beats_teacher_accuracy": bool(r["log_ratio_mean"] > r["teacher_correct_num"]),
        }
        win_rows.append({
            "view": view,
            "delta_k_auroc": r["log_ratio_mean"],
            "full_kl_auroc": r["full_kl_mean"],
            "pi_length_auroc": r["log_pi_length_words"],
            "teacher_accuracy_auroc": r["teacher_correct_num"],
            **beats,
            "beats_all_three": all(beats.values()),
        })
    win_df = pd.DataFrame(win_rows)
    win_df.to_csv(os.path.join(TABLES_DIR, "lopo_win_table.csv"), index=False)

    print("plotting ...")
    make_plots(overall_df, octile_df, region_alignment, think_answer_alignment, auroc_all_df, auroc_elig_df, lopo_df)

    # --- summary json ---
    pi_len_check = pi_len_df.groupby("condition")["pi_length_words"].mean().reindex(PI_VIEWS)
    summary = {
        "config": {"n_boot": N_BOOT, "seed": SEED, "n_problems": len(calib_ids),
                   "n_eligible_problems": len(eligible_ids)},
        "pi_view_density_order_check_word_count": pi_len_check.to_dict(),
        "teacher_accuracy_by_view": teacher_df.groupby("condition")["teacher_correct"].mean().reindex(PI_VIEWS).to_dict(),
        "overall_alignment": overall_df.to_dict(orient="records"),
        "auroc_pooled_all": auroc_all_df.to_dict(orient="records"),
        "auroc_pooled_eligible": auroc_elig_df.to_dict(orient="records"),
        "lopo_win_table": win_df.to_dict(orient="records"),
    }
    with open(os.path.join(OUT_DIR, "correctness_alignment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("done.")
    print(overall_df.to_string(index=False))
    print(win_df.to_string(index=False))


if __name__ == "__main__":
    main()
