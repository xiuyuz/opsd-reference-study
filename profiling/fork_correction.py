"""Signed correction/fork diagnostics.

A mean |log_ratio| at correction-marker tokens does not establish direction: a large
absolute shift is consistent with either suppression (teacher assigns LESS probability
than the student to the observed correction-marker token -- negative log_ratio) or
redirection/amplification (teacher assigns MORE -- positive log_ratio). This script uses
the full profile (all no-PI samples x all 6 reference views, <reduced-dir>/
marker_tokens.parquet) to compute the SIGNED quantities instead, split by student-entropy
median and by trajectory correctness.

Metric definitions (operationalized against what the reduced parquet actually stores: no
token id / decoded text survives the reduction, only per-token scalar stats for the
SAMPLED token plus three named special-token probabilities. "Correction-marker set"
throughout this script therefore means constants.CORRECTION_MARKERS matched against the
token that was actually SAMPLED at that position (the is_correction_marker flag, defined
by prefix_score.py's marker_flags_for_tokens: lexical substring match on the decoded token
piece, case-insensitive) -- NOT a sum over the full vocabulary's marker-set mass, which the
stored per-token stats cannot reconstruct):
  - signed log-ratio      = log_ratio column as-is (logp_view - logp_student), nats, at
                            correction-marker positions under each of the 6 reference views.
  - teacher-minus-student mass = exp(logp_view) - exp(logp_student): the probability-scale gap
                            on the SAME observed correction-marker token (bounded in [-1, 1],
                            more interpretable than the log-scale quantity for "how much
                            probability mass moved").
  - entropy change        = entropy_view - entropy_student: full-vocabulary next-token entropy
                            under the reference-conditioned view minus the student's, at the
                            same position.
  - top-1 replacement "category": no token identity is stored, so this cannot be a token-level
                            category (e.g. "replaced by a stop token" vs "replaced by another
                            content word") in the literal sense. Two derived quantities are
                            reported as the closest available substitute:
                              (a) top1 disagreement rate (top1_agree_with_student == False;
                                  already a directly-observed boolean, not a proxy);
                              (b) among disagreements, the sign of stop_shift = max(eos_prob,
                                  think_close_prob, answer_transition_prob) under the view minus
                                  the same max under the student -- i.e. whether the view's
                                  argmax-disagreement correlates with an increased pull toward
                                  one of the three stored "stop" probabilities (leaning toward
                                  answer-closing) versus not. This is an explicit proxy, flagged
                                  as such in the output and markdown summary.

Splits: high/low student-entropy (global median split on entropy_student, computed once over
all correction-marker EVENTS -- entropy_student is identical across all 7 condition rows of the
same (problem_id, sample_index, position) by construction, verified in this script) and
correct/incorrect trajectory (the `correct` column, trajectory-level).

Statistics: problem-level (cluster) bootstrap throughout -- each problem contributes ONE value
per facet (the mean of the underlying per-token quantity over all its qualifying marker-token
rows, pooled across its samples), then bootstrap-resample problem_ids with replacement
(n_boot=10000, seed=42) and take the 2.5/97.5 percentiles of the resampled means. Every table
reports n_problems, n_samples (trajectories), and n_tokens (marker-token rows) alongside each
estimate.

Outputs (<out-dir>/):
    fork_by_view.csv                 per reference view (+ ALL), pooled over entropy split & correctness
    fork_by_entropy.csv              high / low / ALL, pooled over view & correctness
    fork_by_correct.csv              correct / incorrect / ALL, pooled over view & entropy
    fork_full.csv                    view x entropy_split x correct, full disaggregation
    fork_contrast_entropy.csv        bootstrap diff (high - low) per view (+ ALL)
    fork_contrast_correct.csv        bootstrap diff (correct - incorrect) per view (+ ALL)
    fig_log_ratio_by_view.png
    fig_mass_diff_by_view.png
    fig_entropy_delta_by_view.png
    fig_top1_disagreement_by_view.png
    fork-correction.md               summary write-up
CLI:
    python profiling/fork_correction.py [--reduced-dir <dir>] [--out-dir <dir>]

--reduced-dir/--out-dir default to the thinking-on profile's paths; for the direct-response
profile pass --reduced-dir <artifacts>/profiles/direct/reduced
--out-dir <artifacts>/profiles/direct/analysis/fork-correction
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from opsd.artifact_layout import PROFILE_DIRECT, PROFILE_THINKING, path as artifact_path  # noqa: E402
from opsd.constants import CORRECTION_MARKERS  # noqa: E402

DEFAULT_REDUCED_DIR = artifact_path(PROFILE_THINKING, "reduced")
DEFAULT_OUT_DIR = artifact_path(PROFILE_THINKING, "analysis", "fork-correction")

REDUCED_DIR = DEFAULT_REDUCED_DIR
OUT_DIR = DEFAULT_OUT_DIR
MARKER_TOKENS_PATH = os.path.join(REDUCED_DIR, "marker_tokens.parquet")
PER_GROUP_PATH = os.path.join(REDUCED_DIR, "per_group.parquet")

# Density order, matching score_views.py's ALL_VIEWS.
ALL_VIEWS = ["answer_only", "gist", "key_points", "clean_solution", "summary", "full_trace"]

N_BOOT = 10000
SEED = 42

VALUE_COLS = [
    "log_ratio",
    "mass_diff",
    "entropy_delta",
    "disagree",
    "stop_leaning_given_disagree",
]


# --------------------------------------------------------------------------- #
# Loading + derived columns
# --------------------------------------------------------------------------- #


def load_data():
    df = pd.read_parquet(MARKER_TOKENS_PATH)
    assert df["is_correction_marker"].all(), "marker_tokens.parquet should be pre-filtered to marker rows only"

    none_df = df[df["condition"] == "none"]
    # sanity: entropy_student and log_ratio behave as documented for the student baseline row.
    assert np.allclose(none_df["entropy"], none_df["entropy_student"]), "none-row entropy should equal entropy_student"
    assert (none_df["log_ratio"] == 0.0).all(), "none-row log_ratio should be exactly 0"

    views = df[df["condition"] != "none"].copy()
    views["mass_diff"] = np.exp(views["logp"]) - np.exp(views["logp_student"])
    views["entropy_delta"] = views["entropy"] - views["entropy_student"]
    views["disagree"] = (~views["top1_agree_with_student"]).astype(float)

    teacher_stop = views[["eos_prob", "think_close_prob", "answer_transition_prob"]].max(axis=1)
    student_stop = views[["eos_prob_student", "think_close_prob_student", "answer_transition_prob_student"]].max(axis=1)
    views["stop_shift"] = teacher_stop - student_stop
    # only meaningful conditional on disagreement; NaN when top1 agrees so it drops out of means.
    views["stop_leaning_given_disagree"] = np.where(
        views["disagree"] == 1.0, (views["stop_shift"] > 0).astype(float), np.nan
    )

    median_entropy_student = float(none_df["entropy_student"].median())
    views["high_entropy"] = views["entropy_student"] >= median_entropy_student

    return views, median_entropy_student, len(none_df)


def sanity_check_against_per_group(views):
    """Cross-check marker_tokens.parquet's log_ratio against per_group.parquet's
    marker_log_ratio_mean (both derived from the same source rows by the reducer) --
    catches any accidental double-counting or filtering mismatch."""
    per_group = pd.read_parquet(PER_GROUP_PATH, columns=["problem_id", "sample_index", "condition", "marker_log_ratio_mean", "marker_count"])
    recomputed = (
        views.groupby(["problem_id", "sample_index", "condition"])["log_ratio"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "recomputed_mean", "count": "recomputed_count"})
    )
    merged = per_group.merge(recomputed, on=["problem_id", "sample_index", "condition"], how="inner")
    assert (merged["marker_count"] == merged["recomputed_count"]).all(), "marker_count mismatch vs per_group.parquet"
    diff = (merged["marker_log_ratio_mean"] - merged["recomputed_mean"]).abs()
    assert diff.max() < 1e-6, f"marker_log_ratio_mean mismatch vs per_group.parquet, max abs diff {diff.max()}"
    print(f"sanity check OK: {len(merged)} (problem, sample, view) groups match per_group.parquet exactly")


# --------------------------------------------------------------------------- #
# Problem-level cluster bootstrap
# --------------------------------------------------------------------------- #


def cluster_bootstrap(df_facet, value_cols=VALUE_COLS, n_boot=N_BOOT, seed=SEED):
    """One value per problem_id (mean over its rows in this facet), then bootstrap-resample
    problem_ids with replacement. Returns (obs, lo, hi) arrays aligned with value_cols, plus
    n_problems. NaN-safe per column (stop_leaning_given_disagree is NaN when top1 agrees)."""
    per_problem = df_facet.groupby("problem_id")[value_cols].mean()  # nanmean by construction
    n = len(per_problem)
    if n == 0:
        return None
    arr = per_problem.to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.nanmean(arr[idx], axis=1)  # (n_boot, n_cols)
    obs = np.nanmean(arr, axis=0)
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    return obs, lo, hi, n


def two_sample_bootstrap_diff(df_a, df_b, value_cols=VALUE_COLS, n_boot=N_BOOT, seed=SEED):
    per_a = df_a.groupby("problem_id")[value_cols].mean()
    per_b = df_b.groupby("problem_id")[value_cols].mean()
    na, nb = len(per_a), len(per_b)
    if na == 0 or nb == 0:
        return None
    arr_a, arr_b = per_a.to_numpy(), per_b.to_numpy()
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, na, size=(n_boot, na))
    idx_b = rng.integers(0, nb, size=(n_boot, nb))
    boot_a = np.nanmean(arr_a[idx_a], axis=1)
    boot_b = np.nanmean(arr_b[idx_b], axis=1)
    diff = boot_a - boot_b
    obs = np.nanmean(arr_a, axis=0) - np.nanmean(arr_b, axis=0)
    lo = np.nanpercentile(diff, 2.5, axis=0)
    hi = np.nanpercentile(diff, 97.5, axis=0)
    return obs, lo, hi, na, nb


def facet_row(df_facet, label_cols):
    res = cluster_bootstrap(df_facet)
    row = dict(label_cols)
    row["n_problems"] = 0
    row["n_samples"] = df_facet[["problem_id", "sample_index"]].drop_duplicates().shape[0]
    row["n_tokens"] = len(df_facet)
    if res is None:
        for c in VALUE_COLS:
            row[f"{c}_mean"] = np.nan
            row[f"{c}_lo"] = np.nan
            row[f"{c}_hi"] = np.nan
        return row
    obs, lo, hi, n = res
    row["n_problems"] = n
    for c, o, l, h in zip(VALUE_COLS, obs, lo, hi):
        row[f"{c}_mean"] = o
        row[f"{c}_lo"] = l
        row[f"{c}_hi"] = h
    return row


def contrast_row(df_a, df_b, label_cols):
    res = two_sample_bootstrap_diff(df_a, df_b)
    row = dict(label_cols)
    if res is None:
        for c in VALUE_COLS:
            row[f"{c}_diff"] = np.nan
            row[f"{c}_lo"] = np.nan
            row[f"{c}_hi"] = np.nan
        row["n_problems_a"] = 0
        row["n_problems_b"] = 0
        return row
    obs, lo, hi, na, nb = res
    row["n_problems_a"] = na
    row["n_problems_b"] = nb
    for c, o, l, h in zip(VALUE_COLS, obs, lo, hi):
        row[f"{c}_diff"] = o
        row[f"{c}_lo"] = l
        row[f"{c}_hi"] = h
    return row


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #


def build_by_view(views):
    rows = []
    for view in ALL_VIEWS:
        rows.append(facet_row(views[views["condition"] == view], {"view": view}))
    rows.append(facet_row(views, {"view": "ALL"}))
    return pd.DataFrame(rows)


def build_by_entropy(views):
    rows = []
    for label, sub in [("high", views[views["high_entropy"]]), ("low", views[~views["high_entropy"]])]:
        rows.append(facet_row(sub, {"entropy_split": label}))
    rows.append(facet_row(views, {"entropy_split": "ALL"}))
    return pd.DataFrame(rows)


def build_by_correct(views):
    rows = []
    for label, sub in [("correct", views[views["correct"]]), ("incorrect", views[~views["correct"]])]:
        rows.append(facet_row(sub, {"trajectory": label}))
    rows.append(facet_row(views, {"trajectory": "ALL"}))
    return pd.DataFrame(rows)


def build_full(views):
    rows = []
    for view in ALL_VIEWS:
        v_sub = views[views["condition"] == view]
        for e_label, e_mask in [("high", v_sub["high_entropy"]), ("low", ~v_sub["high_entropy"])]:
            e_sub = v_sub[e_mask]
            for c_label, c_mask in [("correct", e_sub["correct"]), ("incorrect", ~e_sub["correct"])]:
                c_sub = e_sub[c_mask]
                rows.append(facet_row(c_sub, {"view": view, "entropy_split": e_label, "trajectory": c_label}))
    return pd.DataFrame(rows)


def build_contrast_entropy(views):
    rows = []
    for view in ALL_VIEWS + ["ALL"]:
        sub = views if view == "ALL" else views[views["condition"] == view]
        rows.append(contrast_row(sub[sub["high_entropy"]], sub[~sub["high_entropy"]], {"view": view}))
    return pd.DataFrame(rows)


def build_contrast_correct(views):
    rows = []
    for view in ALL_VIEWS + ["ALL"]:
        sub = views if view == "ALL" else views[views["condition"] == view]
        rows.append(contrast_row(sub[sub["correct"]], sub[~sub["correct"]], {"view": view}))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def grouped_bar(ax, by_first_df, first_col, first_vals, metric, ylabel, title):
    x = np.arange(len(ALL_VIEWS))
    width = 0.8 / len(first_vals)
    for i, val in enumerate(first_vals):
        sub = by_first_df[by_first_df[first_col] == val].set_index("view").reindex(ALL_VIEWS)
        means = sub[f"{metric}_mean"].to_numpy()
        los = sub[f"{metric}_lo"].to_numpy()
        his = sub[f"{metric}_hi"].to_numpy()
        yerr = np.vstack([means - los, his - means])
        ax.bar(x + (i - (len(first_vals) - 1) / 2) * width, means, width, yerr=yerr, capsize=2, label=str(val))
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_VIEWS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


def make_figure(views, metric, ylabel, fname, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # pooled-over-entropy (by view x correct) and pooled-over-correct (by view x entropy)
    rows_c = []
    for view in ALL_VIEWS:
        v_sub = views[views["condition"] == view]
        for c_label, c_mask in [("correct", v_sub["correct"]), ("incorrect", ~v_sub["correct"])]:
            rows_c.append(facet_row(v_sub[c_mask], {"view": view, "trajectory": c_label}))
    by_view_correct = pd.DataFrame(rows_c)

    rows_e = []
    for view in ALL_VIEWS:
        v_sub = views[views["condition"] == view]
        for e_label, e_mask in [("high", v_sub["high_entropy"]), ("low", ~v_sub["high_entropy"])]:
            rows_e.append(facet_row(v_sub[e_mask], {"view": view, "entropy_split": e_label}))
    by_view_entropy = pd.DataFrame(rows_e)

    grouped_bar(axes[0], by_view_correct, "trajectory", ["correct", "incorrect"], metric, ylabel, f"{title_prefix}: by trajectory correctness")
    grouped_bar(axes[1], by_view_entropy, "entropy_split", ["high", "low"], metric, ylabel, f"{title_prefix}: by student-entropy split")

    fig.suptitle(f"Fork/correction diagnostic: {title_prefix} at correction-marker tokens\n(6 reference views, problem-level bootstrap 95% CI)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_top1_figure(views):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    rows_c = []
    for view in ALL_VIEWS:
        v_sub = views[views["condition"] == view]
        for c_label, c_mask in [("correct", v_sub["correct"]), ("incorrect", ~v_sub["correct"])]:
            rows_c.append(facet_row(v_sub[c_mask], {"view": view, "trajectory": c_label}))
    by_view_correct = pd.DataFrame(rows_c)

    rows_e = []
    for view in ALL_VIEWS:
        v_sub = views[views["condition"] == view]
        for e_label, e_mask in [("high", v_sub["high_entropy"]), ("low", ~v_sub["high_entropy"])]:
            rows_e.append(facet_row(v_sub[e_mask], {"view": view, "entropy_split": e_label}))
    by_view_entropy = pd.DataFrame(rows_e)

    grouped_bar(axes[0], by_view_correct, "trajectory", ["correct", "incorrect"], "disagree", "top-1 disagreement rate", "top-1 disagreement: by trajectory correctness")
    grouped_bar(axes[1], by_view_entropy, "entropy_split", ["high", "low"], "disagree", "top-1 disagreement rate", "top-1 disagreement: by student-entropy split")
    fig.suptitle("Top-1 (argmax) disagreement rate at correction-marker tokens, teacher view vs. student")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_top1_disagreement_by_view.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Markdown summary
# --------------------------------------------------------------------------- #


def fmt_ci(mean, lo, hi, digits=4):
    return f"{mean:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def write_markdown(views, median_entropy_student, n_events, by_view, by_entropy, by_correct,
                    contrast_entropy, contrast_correct, n_problems_total):
    all_row = by_view[by_view["view"] == "ALL"].iloc[0]
    high_row = by_entropy[by_entropy["entropy_split"] == "high"].iloc[0]
    low_row = by_entropy[by_entropy["entropy_split"] == "low"].iloc[0]
    correct_row = by_correct[by_correct["trajectory"] == "correct"].iloc[0]
    incorrect_row = by_correct[by_correct["trajectory"] == "incorrect"].iloc[0]

    n_incorrect_traj = views.loc[~views["correct"], ["problem_id", "sample_index"]].drop_duplicates().shape[0]
    n_correct_traj = views.loc[views["correct"], ["problem_id", "sample_index"]].drop_duplicates().shape[0]

    lines = []
    lines.append("# Fork/correction diagnostics\n")
    lines.append(
        "Signed correction/fork diagnostics on `constants.CORRECTION_MARKERS` "
        f"(`{', '.join(CORRECTION_MARKERS)}`), computed from "
        f"`{MARKER_TOKENS_PATH}` (all rows where the "
        "sampled token's decoded piece lexically matches the correction-marker set, all "
        "no-PI samples x all 6 reference views x the calibration problems). Supersedes an "
        "unsigned `mean_abs_log_ratio` on correction markers with the signed quantities.\n"
    )
    lines.append("\n## Data scope\n\n")
    lines.append(
        f"- {n_events} correction-marker token EVENTS (one per (problem_id, sample_index, "
        f"position) where the sampled token matched); {n_problems_total} distinct problems.\n"
        f"- {int(all_row['n_samples'])} distinct trajectories contribute marker events under "
        f"the 6 reference views ({n_correct_traj} correct, {n_incorrect_traj} incorrect).\n"
        f"- {int(all_row['n_tokens'])} (event x view) rows feed the signed diagnostics below "
        "(6 views x marker events, excluding the trivial `condition == \"none\"` self-comparison rows).\n"
        f"- Student-entropy median split threshold (global, computed once over the "
        f"{n_events} marker events' `entropy_student`): **{median_entropy_student:.4f} nats**.\n"
    )
    lines.append("\n## Metric definitions (see `fork_correction.py` module docstring for the full caveat)\n\n")
    lines.append(
        "- **signed log-ratio**: `log_ratio = logp_view - logp_student` (nats) at the observed "
        "correction-marker token. Negative = teacher/reference-conditioned view assigns LESS "
        "probability than the student to the token the student actually sampled here "
        "(suppression); positive = MORE (amplification/redirection toward that word).\n"
        "- **teacher-minus-student mass**: `exp(logp_view) - exp(logp_student)`, the same "
        "comparison on the probability scale (bounded [-1, 1]). Both quantities are computed on "
        "the SAMPLED token only -- no full-vocabulary correction-marker-set probability is stored "
        "anywhere in the profile pipeline (no token-id/decoded-text column survives the "
        "per-token reduction), so this is not a sum over the whole marker vocabulary.\n"
        "- **entropy change**: `entropy_view - entropy_student`, full-vocabulary next-token "
        "entropy, teacher-conditioned minus student, at the same position.\n"
        "- **top-1 disagreement rate**: fraction of marker positions where the view's argmax "
        "token differs from the student's argmax (`top1_agree_with_student == False`) -- a "
        "directly observed quantity, not a proxy.\n"
        "- **top-1 replacement 'category' (proxy)**: conditional on disagreement, whether "
        "`max(eos_prob, think_close_prob, answer_transition_prob)` rose under the view relative "
        "to the student (`stop_leaning_given_disagree`) -- i.e. whether the argmax shift "
        "correlates with increased pull toward stopping/answering. No token identity is stored "
        "for the actual top-1 replacement, so this is the closest available substitute, not a "
        "literal token category.\n"
    )
    lines.append("\n## Headline results (pooled across 6 reference views)\n\n")
    lines.append(
        f"- **All views, all positions**: signed log-ratio = "
        f"{fmt_ci(all_row['log_ratio_mean'], all_row['log_ratio_lo'], all_row['log_ratio_hi'])}, "
        f"mass gap = {fmt_ci(all_row['mass_diff_mean'], all_row['mass_diff_lo'], all_row['mass_diff_hi'])}, "
        f"entropy change = {fmt_ci(all_row['entropy_delta_mean'], all_row['entropy_delta_lo'], all_row['entropy_delta_hi'])}, "
        f"top-1 disagreement rate = {fmt_ci(all_row['disagree_mean'], all_row['disagree_lo'], all_row['disagree_hi'], 3)} "
        f"(n_problems={int(all_row['n_problems'])}).\n"
    )
    lines.append(
        f"- **High student-entropy positions** (entropy_student >= {median_entropy_student:.4f}): "
        f"signed log-ratio = {fmt_ci(high_row['log_ratio_mean'], high_row['log_ratio_lo'], high_row['log_ratio_hi'])} "
        f"(n_problems={int(high_row['n_problems'])}, n_tokens={int(high_row['n_tokens'])}).\n"
        f"- **Low student-entropy positions**: signed log-ratio = "
        f"{fmt_ci(low_row['log_ratio_mean'], low_row['log_ratio_lo'], low_row['log_ratio_hi'])} "
        f"(n_problems={int(low_row['n_problems'])}, n_tokens={int(low_row['n_tokens'])}).\n"
    )
    diff_e_all = contrast_entropy[contrast_entropy["view"] == "ALL"].iloc[0]
    lines.append(
        f"- High-minus-low entropy contrast (independent two-sample problem-cluster bootstrap): "
        f"Δlog-ratio = {fmt_ci(diff_e_all['log_ratio_diff'], diff_e_all['log_ratio_lo'], diff_e_all['log_ratio_hi'])}, "
        f"Δmass = {fmt_ci(diff_e_all['mass_diff_diff'], diff_e_all['mass_diff_lo'], diff_e_all['mass_diff_hi'])}.\n"
    )
    lines.append(
        f"- **Correct trajectories**: signed log-ratio = "
        f"{fmt_ci(correct_row['log_ratio_mean'], correct_row['log_ratio_lo'], correct_row['log_ratio_hi'])} "
        f"(n_problems={int(correct_row['n_problems'])}, n_tokens={int(correct_row['n_tokens'])}).\n"
        f"- **Incorrect trajectories**: signed log-ratio = "
        f"{fmt_ci(incorrect_row['log_ratio_mean'], incorrect_row['log_ratio_lo'], incorrect_row['log_ratio_hi'])} "
        f"(n_problems={int(incorrect_row['n_problems'])}, n_tokens={int(incorrect_row['n_tokens'])}).\n"
    )
    diff_c_all = contrast_correct[contrast_correct["view"] == "ALL"].iloc[0]
    lines.append(
        f"- Correct-minus-incorrect contrast: Δlog-ratio = "
        f"{fmt_ci(diff_c_all['log_ratio_diff'], diff_c_all['log_ratio_lo'], diff_c_all['log_ratio_hi'])}, "
        f"Δmass = {fmt_ci(diff_c_all['mass_diff_diff'], diff_c_all['mass_diff_lo'], diff_c_all['mass_diff_hi'])} "
        f"(n_problems: {int(diff_c_all['n_problems_a'])} contributing correct trajectories, "
        f"{int(diff_c_all['n_problems_b'])} contributing incorrect trajectories -- note these "
        "are NOT disjoint problem sets since a problem can have both correct and incorrect "
        "samples; this is an unpaired split of marker events by trajectory outcome, not a "
        "paired per-problem comparison).\n"
    )
    lines.append("\n## By reference view (pooled across entropy split and trajectory correctness)\n\n")
    lines.append("| view | n_problems | n_tokens | signed log-ratio | mass gap | entropy Δ | top-1 disagree rate |\n")
    lines.append("|---|---:|---:|---|---|---|---|\n")
    for _, r in by_view.iterrows():
        lines.append(
            f"| {r['view']} | {int(r['n_problems'])} | {int(r['n_tokens'])} | "
            f"{fmt_ci(r['log_ratio_mean'], r['log_ratio_lo'], r['log_ratio_hi'])} | "
            f"{fmt_ci(r['mass_diff_mean'], r['mass_diff_lo'], r['mass_diff_hi'])} | "
            f"{fmt_ci(r['entropy_delta_mean'], r['entropy_delta_lo'], r['entropy_delta_hi'])} | "
            f"{fmt_ci(r['disagree_mean'], r['disagree_lo'], r['disagree_hi'], 3)} |\n"
        )
    lines.append(
        "\n`stop_leaning_given_disagree` (conditional on top-1 disagreement, is the argmax shift "
        "correlated with increased stop/answer-transition probability under the view) is in "
        f"`fork_by_view.csv` alongside the columns above; full disaggregation "
        "(view x entropy split x correctness, 24 rows) is in `fork_full.csv`.\n"
    )
    full_row = by_view[by_view["view"] == "full_trace"].iloc[0]
    answer_row = by_view[by_view["view"] == "answer_only"].iloc[0]
    lines.append(
        "\n**Pattern across views**: signed log-ratio goes from "
        f"{answer_row['log_ratio_mean']:+.3f} (`answer_only`, lowest reference density) to "
        f"{full_row['log_ratio_mean']:+.3f} (`full_trace`, highest), and top-1 disagreement rate "
        f"goes {answer_row['disagree_mean']:.3f} -> {full_row['disagree_mean']:.3f}; the "
        f"`full_trace` entropy change is {full_row['entropy_delta_mean']:+.4f}.\n"
    )
    lines.append("\n## Gate signal\n\n")
    signed_sign = "suppression (negative)" if all_row["log_ratio_mean"] < 0 else "amplification/redirection (positive)"
    ci_excludes_zero = not (all_row["log_ratio_lo"] <= 0 <= all_row["log_ratio_hi"])
    gate_word = "MET" if ci_excludes_zero else "NOT clearly met"
    lines.append(
        f"Gate criterion: *\"the reference causes signed suppression or redirection at "
        f"high-entropy correction states\"*. Pooled-across-views signed log-ratio at correction "
        f"markers is {signed_sign}, 95% CI {fmt_ci(all_row['log_ratio_mean'], all_row['log_ratio_lo'], all_row['log_ratio_hi'])} "
        f"({'excludes' if ci_excludes_zero else 'does not exclude'} zero) -- gate signal from this "
        f"analysis: **{gate_word}**. At high-entropy positions specifically, signed log-ratio is "
        f"{fmt_ci(high_row['log_ratio_mean'], high_row['log_ratio_lo'], high_row['log_ratio_hi'])} "
        f"({'excludes' if not (high_row['log_ratio_lo'] <= 0 <= high_row['log_ratio_hi']) else 'does not exclude'} zero).\n"
    )
    lines.append(
        "\nSee `fork_by_entropy.csv`, `fork_by_correct.csv`, `fork_contrast_entropy.csv`, "
        "`fork_contrast_correct.csv` for the full split tables, and `fig_log_ratio_by_view.png` / "
        "`fig_mass_diff_by_view.png` / `fig_entropy_delta_by_view.png` / "
        "`fig_top1_disagreement_by_view.png` for the plots.\n"
    )
    with open(os.path.join(OUT_DIR, "fork-correction.md"), "w") as f:
        f.write("".join(lines))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reduced-dir", default=DEFAULT_REDUCED_DIR,
        help="dir containing marker_tokens.parquet/per_group.parquet (default: the thinking-on "
             f"profile's reduced/ dir; direct-response profile: {artifact_path(PROFILE_DIRECT, 'reduced')})",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help="output dir for tables/plots/markdown (default: the thinking-on profile's analysis dir).",
    )
    return parser.parse_args()


def main():
    global REDUCED_DIR, OUT_DIR, MARKER_TOKENS_PATH, PER_GROUP_PATH
    args = parse_args()
    REDUCED_DIR = args.reduced_dir
    OUT_DIR = args.out_dir
    MARKER_TOKENS_PATH = os.path.join(REDUCED_DIR, "marker_tokens.parquet")
    PER_GROUP_PATH = os.path.join(REDUCED_DIR, "per_group.parquet")

    os.makedirs(OUT_DIR, exist_ok=True)
    print("loading marker_tokens.parquet...")
    views, median_entropy_student, n_events = load_data()
    print(f"loaded {len(views)} (event x view) rows, {n_events} marker events, "
          f"median student entropy = {median_entropy_student:.4f}")

    sanity_check_against_per_group(views)

    n_problems_total = views["problem_id"].nunique()

    print("building tables...")
    by_view = build_by_view(views)
    by_entropy = build_by_entropy(views)
    by_correct = build_by_correct(views)
    full_df = build_full(views)
    contrast_entropy = build_contrast_entropy(views)
    contrast_correct = build_contrast_correct(views)

    by_view.to_csv(os.path.join(OUT_DIR, "fork_by_view.csv"), index=False)
    by_entropy.to_csv(os.path.join(OUT_DIR, "fork_by_entropy.csv"), index=False)
    by_correct.to_csv(os.path.join(OUT_DIR, "fork_by_correct.csv"), index=False)
    full_df.to_csv(os.path.join(OUT_DIR, "fork_full.csv"), index=False)
    contrast_entropy.to_csv(os.path.join(OUT_DIR, "fork_contrast_entropy.csv"), index=False)
    contrast_correct.to_csv(os.path.join(OUT_DIR, "fork_contrast_correct.csv"), index=False)
    print("wrote CSV tables")

    print("making figures...")
    make_figure(views, "log_ratio", "signed log-ratio (nats)", "fig_log_ratio_by_view.png", "signed log-ratio")
    make_figure(views, "mass_diff", "teacher-minus-student mass", "fig_mass_diff_by_view.png", "probability-mass gap")
    make_figure(views, "entropy_delta", "entropy change (nats)", "fig_entropy_delta_by_view.png", "entropy change")
    make_top1_figure(views)
    print("wrote figures")

    write_markdown(views, median_entropy_student, n_events, by_view, by_entropy, by_correct,
                    contrast_entropy, contrast_correct, n_problems_total)
    print(f"wrote {os.path.join(OUT_DIR, 'fork-correction.md')}")


if __name__ == "__main__":
    main()
