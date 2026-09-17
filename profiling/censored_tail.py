"""Post-answer / censored-tail accounting of KL mass.

Reads <reduced-dir>/per_group.parquet (one row per problem_id x sample_index x condition,
condition in {"none" (no-PI student self-baseline, full_kl==0), "answer_only", "gist",
"key_points", "clean_solution", "summary", "full_trace"} -- see reduced/reduce_manifest.json
for the exact grouping and column provenance).

Fractions of total KL/loss mass:
  1. before the first answer;
  2. after the first answer;
  3. after a correct answer has already appeared;
  4. inside repeated spans;
  5. in capped trajectories.

Only (1)/(2) and (5) are derivable from per_group.parquet. (3) and (4) require token
identity / decoded text (to locate every literal `\\boxed{...}` occurrence, or to detect
repeated n-grams), which the reduction does not carry; they are reported as NOT DERIVABLE
here, since they would need a separate pass over the raw shards.

(1)/(2) are reported using per_group's think/answer split, which is computed at the
completion's first `</think>` token (n_think_tokens/n_answer_tokens,
full_kl_sum_think/full_kl_sum_answer) -- this is a PROXY for "before/after first answer
proposal" (the literal first \\boxed{} token), not the same event. Flagged explicitly in
the output.

(5) "capped trajectories" needs the trajectory-level hit_length_cap / finish_reason /
rescued_finish_reason fields from the ORIGINAL generations jsonl (--generations). This is
a small metadata join keyed on (problem_id, sample_index) -- condition-independent, since
all 7 condition rows in per_group teacher-force the SAME underlying no-PI completion.

Also reports a supplementary true-quartile-of-normalized-position breakdown (already in
per_group as full_kl_sum_q0..q3 / n_tokens_q0..q3) as a finer-grained cross-check of the
think/answer split.

Statistics: problem-level cluster bootstrap (resample problem_id with replacement,
n_boot=10000, seed=42, 95% CI via [2.5, 97.5] percentiles of the resampled pooled
(mass-weighted) ratio). Pooled/mass-weighted fractions (sum of KL over sum of KL), not
means of per-trajectory fractions, are the headline numbers throughout, since a
mean-of-ratios would let very short trajectories (e.g. all-answer, near-zero think tokens)
dominate with noisy 0/1-ish fractions; per-trajectory-mean fractions are also written to
the tables for reference.

CLI:
    python profiling/censored_tail.py [--n-boot 10000] [--seed 42]
        [--reduced-dir <dir>] [--generations <jsonl>] [--out-dir <dir>]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from opsd.artifact_layout import (  # noqa: E402
    PROFILE_CALIBRATION,
    PROFILE_DIRECT,
    PROFILE_DIRECT_STATES,
    PROFILE_THINKING,
    path as artifact_path,
)
from opsd.constants import CONDITIONS as ORIGINAL_CONDITIONS  # noqa: E402

# Defaults are the thinking-on profile's paths; any other profile passes --reduced-dir/
# --generations/--out-dir. --generations matters here specifically: load_capped_lookup()
# needs the SAME no-PI rollout corpus that was actually scored into --reduced-dir's
# per_group.parquet (hit_length_cap/finish_reason are properties of that generation run,
# e.g. the direct-response corpus under PROFILE_DIRECT_STATES, not the thinking-on
# PROFILE_CALIBRATION corpus).
DEFAULT_REDUCED_DIR = artifact_path(PROFILE_THINKING, "reduced")
DEFAULT_GENERATIONS_PATH = artifact_path(PROFILE_CALIBRATION, "qwen3_1.7b_generations.jsonl")
DEFAULT_OUT_DIR = artifact_path(PROFILE_THINKING, "analysis", "censored-tail")

REDUCED_DIR = DEFAULT_REDUCED_DIR
GENERATIONS_PATH = DEFAULT_GENERATIONS_PATH
OUT_DIR = DEFAULT_OUT_DIR
TABLES_DIR = os.path.join(OUT_DIR, "tables")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

PI_VIEWS = ["answer_only", "gist", "key_points", "clean_solution", "summary", "full_trace"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_per_group():
    df = pd.read_parquet(os.path.join(REDUCED_DIR, "per_group.parquet"))
    assert set(df["condition"].unique()) == {"none"} | set(PI_VIEWS)
    return df


def load_capped_lookup(path):
    """(problem_id, sample_index) -> dict(hit_length_cap, final_capped) for every no-PI
    rollout in the original generations file. Condition-independent (property of the
    underlying completion, shared by all 7 per_group condition rows for that key).

    hit_length_cap: the ORIGINAL generation pass hit max_new_tokens and needed a rescue
        pass (rec['hit_length_cap']).
    final_capped: the completion actually used for scoring (rescued if rescued else
        original, matching prefix_score.py's get_completion_tokens preference) ended
        because generation was cut off (finish_reason == 'length'), i.e. it never
        produced a genuine stop/EOS even after rescue -- the truest match to
        "cap termination".
    """
    lookup = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("condition") in ORIGINAL_CONDITIONS:
                continue  # PI-conditioned generation record, not a no-PI rollout
            key = (rec["problem_id"], rec.get("sample_index", 0))
            final_finish_reason = rec.get("rescued_finish_reason") if rec["rescued"] else rec["finish_reason"]
            lookup[key] = {
                "hit_length_cap": bool(rec["hit_length_cap"]),
                "final_capped": final_finish_reason == "length",
            }
    return lookup


# --------------------------------------------------------------------------- #
# Cluster (problem-level) bootstrap for a pooled (mass-weighted) ratio
# --------------------------------------------------------------------------- #


def cluster_bootstrap_ratio(problem_num, problem_denom, n_boot, rng):
    """problem_num/problem_denom: arrays of per-problem numerator/denominator sums
    (same problem order). Returns (point_estimate, ci_lo, ci_hi, n_problems)."""
    denom_total = problem_denom.sum()
    if denom_total <= 0:
        return None
    point = problem_num.sum() / denom_total
    n = len(problem_num)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_num = problem_num[idx].sum(axis=1)
    boot_denom = problem_denom[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = boot_num / boot_denom
    ratios = ratios[np.isfinite(ratios)]
    if len(ratios) == 0:
        return {"point": float(point), "ci_lo": None, "ci_hi": None, "n_problems": int(n), "n_boot_valid": 0}
    ci_lo, ci_hi = np.percentile(ratios, [2.5, 97.5])
    return {
        "point": float(point),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n_problems": int(n),
        "n_boot_valid": int(len(ratios)),
    }


def pooled_fraction_with_ci(df_subset, num_col, denom_col, n_boot, rng):
    """Cluster-bootstrap the pooled ratio sum(num_col)/sum(denom_col), clustering by
    problem_id (collapse to one num/denom pair per problem_id first, summing over
    whatever rows -- samples x views -- are in df_subset for that problem)."""
    per_problem = df_subset.groupby("problem_id")[[num_col, denom_col]].sum()
    per_problem = per_problem[per_problem[denom_col] > 0]
    if per_problem.empty:
        return None
    res = cluster_bootstrap_ratio(per_problem[num_col].to_numpy(), per_problem[denom_col].to_numpy(), n_boot, rng)
    return res


# --------------------------------------------------------------------------- #
# Component 1/2: before/after first-answer proxy (think/answer split at </think>)
# --------------------------------------------------------------------------- #


def think_answer_accounting(df, n_boot, rng):
    pi = df[df["condition"] != "none"].copy()
    results = {}

    def row_for(subset, label):
        r = {
            "label": label,
            "n_rows": int(len(subset)),
            "n_problems": int(subset["problem_id"].nunique()),
            "n_tokens_total": int(subset["n_tokens"].sum()),
            "think_token_share": float(subset["n_think_tokens"].sum() / max(subset["n_tokens"].sum(), 1)),
            "answer_token_share": float(subset["n_answer_tokens"].sum() / max(subset["n_tokens"].sum(), 1)),
        }
        think_ci = pooled_fraction_with_ci(subset, "full_kl_sum_think", "full_kl_sum", n_boot, rng)
        answer_ci = pooled_fraction_with_ci(subset, "full_kl_sum_answer", "full_kl_sum", n_boot, rng)
        r["kl_share_think_pooled"] = think_ci
        r["kl_share_answer_pooled"] = answer_ci
        # per-trajectory mean fraction, for reference only (mass-weighted pooled numbers above are the headline).
        valid = subset[subset["full_kl_sum"] > 0]
        r["kl_share_answer_mean_of_traj_fractions"] = (
            float((valid["full_kl_sum_answer"] / valid["full_kl_sum"]).mean()) if len(valid) else None
        )
        think_tok = subset["n_think_tokens"].sum()
        ans_tok = subset["n_answer_tokens"].sum()
        r["mean_kl_per_think_token"] = float(subset["full_kl_sum_think"].sum() / think_tok) if think_tok else None
        r["mean_kl_per_answer_token"] = float(subset["full_kl_sum_answer"].sum() / ans_tok) if ans_tok else None
        return r

    results["all_views_pooled"] = row_for(pi, "all 6 reference views pooled")
    results["by_view"] = {v: row_for(pi[pi["condition"] == v], v) for v in PI_VIEWS}
    results["by_correctness_all_views"] = {
        "correct": row_for(pi[pi["correct"]], "correct trajectories, all views"),
        "incorrect": row_for(pi[~pi["correct"]], "incorrect trajectories, all views"),
    }
    return results


def quartile_accounting(df, n_boot, rng):
    pi = df[df["condition"] != "none"].copy()
    out = {}
    for q in range(4):
        num_col, denom_col = f"full_kl_sum_q{q}", "full_kl_sum"
        out[f"q{q}"] = {
            "kl_share_pooled": pooled_fraction_with_ci(pi, num_col, "full_kl_sum", n_boot, rng),
            "token_share": float(pi[f"n_tokens_q{q}"].sum() / max(pi["n_tokens"].sum(), 1)),
        }
    by_view = {}
    for v in PI_VIEWS:
        sub = pi[pi["condition"] == v]
        by_view[v] = {
            f"q{q}_kl_share": pooled_fraction_with_ci(sub, f"full_kl_sum_q{q}", "full_kl_sum", n_boot, rng)
            for q in range(4)
        }
    out["by_view"] = by_view
    return out


# --------------------------------------------------------------------------- #
# Component 5: capped trajectories
# --------------------------------------------------------------------------- #


def capped_accounting(df, capped_lookup, n_boot, rng):
    pi = df[df["condition"] != "none"].copy()
    key = list(zip(pi["problem_id"], pi["sample_index"]))
    missing = [k for k in set(key) if k not in capped_lookup]
    hit_cap = np.array([capped_lookup[k]["hit_length_cap"] for k in key])
    final_capped = np.array([capped_lookup[k]["final_capped"] for k in key])
    pi = pi.assign(hit_length_cap=hit_cap, final_capped=final_capped)

    def summarize(flag_col, label):
        capped_rows = pi[pi[flag_col]]
        n_samples_total = pi.drop_duplicates(["problem_id", "sample_index"]).shape[0]
        n_samples_capped = capped_rows.drop_duplicates(["problem_id", "sample_index"]).shape[0]
        # numerator = KL summed only over capped rows (0 for problems/samples with none);
        # denominator = KL summed over ALL rows -- pooled over the whole population.
        pi_marked = pi.assign(_num=np.where(pi[flag_col], pi["full_kl_sum"], 0.0))
        kl_ci = pooled_fraction_with_ci(pi_marked, "_num", "full_kl_sum", n_boot, rng)
        pi_marked_tok = pi.assign(_num_tok=np.where(pi[flag_col], pi["n_tokens"].astype(float), 0.0))
        tok_ci = pooled_fraction_with_ci(pi_marked_tok, "_num_tok", "n_tokens", n_boot, rng)
        by_view = {}
        for v in PI_VIEWS:
            sub = pi_marked[pi_marked["condition"] == v]
            by_view[v] = pooled_fraction_with_ci(sub, "_num", "full_kl_sum", n_boot, rng)
        return {
            "label": label,
            "n_samples_total": int(n_samples_total),
            "n_samples_flagged": int(n_samples_capped),
            "n_problems_with_flagged_sample": int(capped_rows["problem_id"].nunique()),
            "kl_share_pooled_all_views": kl_ci,
            "token_share_pooled_all_views": tok_ci,
            "kl_share_by_view": by_view,
        }

    return {
        "note": "capped-status is a property of the underlying no-PI rollout (condition-independent); "
        "joined from the generations jsonl's hit_length_cap/finish_reason/rescued_finish_reason "
        "fields, keyed on (problem_id, sample_index), not present in per_group.parquet.",
        "n_missing_from_generations_lookup": len(missing),
        "final_capped": summarize(
            "final_capped",
            "scored completion itself ended by length cap (finish_reason=='length' on the rescued pass "
            "if rescued, else on the original pass) -- never produced a genuine stop/EOS",
        ),
        "ever_hit_length_cap": summarize(
            "hit_length_cap",
            "original generation pass hit the length cap and required a rescue pass (whether or not the "
            "rescue pass itself then finished normally)",
        ),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def make_plots(think_answer, quartiles, capped):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. think vs answer KL share by view (stacked bar) with token-share reference line.
    fig, ax = plt.subplots(figsize=(8, 5))
    views = PI_VIEWS
    think_share = [think_answer["by_view"][v]["kl_share_think_pooled"]["point"] for v in views]
    answer_share = [think_answer["by_view"][v]["kl_share_answer_pooled"]["point"] for v in views]
    answer_lo = [think_answer["by_view"][v]["kl_share_answer_pooled"]["ci_lo"] for v in views]
    answer_hi = [think_answer["by_view"][v]["kl_share_answer_pooled"]["ci_hi"] for v in views]
    x = np.arange(len(views))
    ax.bar(x, think_share, label="think section (before </think>)", color="#4C72B0")
    ax.bar(x, answer_share, bottom=think_share, label="answer section (from </think>)", color="#DD8452")
    err_lo = [answer_share[i] - answer_lo[i] for i in range(len(views))]
    err_hi = [answer_hi[i] - answer_share[i] for i in range(len(views))]
    ax.errorbar(
        x, [think_share[i] + answer_share[i] for i in range(len(views))],
        yerr=[err_lo, err_hi], fmt="none", ecolor="black", capsize=4, label="answer-share 95% CI (top edge)",
    )
    tok_share = [think_answer["by_view"][v]["answer_token_share"] for v in views]
    ax.scatter(x, [think_answer["by_view"][v]["think_token_share"] for v in views],
               marker="_", s=400, color="black", zorder=5, label="token-count share (reference)")
    ax.set_xticks(x)
    ax.set_xticklabels(views, rotation=30, ha="right")
    ax.set_ylabel("share of pooled full-vocab KL mass")
    ax.set_ylim(0, 1.05)
    ax.set_title("Censored-tail proxy: KL mass in think vs. answer section, by reference view\n"
                  "(answer section starts at first </think>; black tick = token-count share)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "think_vs_answer_kl_share_by_view.png"), dpi=150)
    plt.close(fig)

    # 2. quartile KL density (KL share vs token share) pooled across views.
    fig, ax = plt.subplots(figsize=(7, 5))
    qs = [f"q{i}" for i in range(4)]
    kl_share = [quartiles[q]["kl_share_pooled"]["point"] for q in qs]
    kl_lo = [quartiles[q]["kl_share_pooled"]["ci_lo"] for q in qs]
    kl_hi = [quartiles[q]["kl_share_pooled"]["ci_hi"] for q in qs]
    tok_share = [quartiles[q]["token_share"] for q in qs]
    xpos = np.arange(4)
    width = 0.35
    ax.bar(xpos - width / 2, kl_share, width, yerr=[[kl_share[i] - kl_lo[i] for i in range(4)],
                                                      [kl_hi[i] - kl_share[i] for i in range(4)]],
           capsize=4, label="KL mass share", color="#55A868")
    ax.bar(xpos + width / 2, tok_share, width, label="token-count share", color="#8C8C8C")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["Q1\n[0,.25)", "Q2\n[.25,.5)", "Q3\n[.5,.75)", "Q4\n[.75,1]"])
    ax.set_ylabel("share of total")
    ax.set_title("Where KL mass concentrates over normalized completion position\n(pooled across all 6 reference views)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "quartile_kl_density.png"), dpi=150)
    plt.close(fig)

    # 3. capped-trajectory KL share vs token share, by view, for both cap definitions.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, key, title in zip(
        axes, ["final_capped", "ever_hit_length_cap"],
        ["final_capped\n(scored completion itself hit the cap)", "ever_hit_length_cap\n(needed a rescue pass)"],
    ):
        block = capped[key]
        kl = [block["kl_share_by_view"][v]["point"] if block["kl_share_by_view"][v] else 0 for v in views]
        kl_lo = [block["kl_share_by_view"][v]["ci_lo"] if block["kl_share_by_view"][v] else 0 for v in views]
        kl_hi = [block["kl_share_by_view"][v]["ci_hi"] if block["kl_share_by_view"][v] else 0 for v in views]
        tok = block["token_share_pooled_all_views"]["point"] if block["token_share_pooled_all_views"] else 0
        x = np.arange(len(views))
        ax.bar(x, kl, yerr=[[kl[i] - kl_lo[i] for i in range(len(views))],
                             [kl_hi[i] - kl[i] for i in range(len(views))]],
               capsize=4, color="#C44E52", label="KL share (capped subset)")
        ax.axhline(tok, color="black", linestyle="--", label=f"token share (capped subset, pooled) = {tok:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels(views, rotation=30, ha="right")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("share of total pooled KL mass (all rows as denominator)")
    fig.suptitle(f"Capped trajectories' share of total KL mass, by reference view "
                 f"(n_samples flagged: final_capped={capped['final_capped']['n_samples_flagged']}, "
                 f"ever_hit_length_cap={capped['ever_hit_length_cap']['n_samples_flagged']}, "
                 f"of {capped['final_capped']['n_samples_total']} total)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "capped_trajectory_kl_share.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main():
    global REDUCED_DIR, GENERATIONS_PATH, OUT_DIR, TABLES_DIR, PLOTS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reduced-dir", default=DEFAULT_REDUCED_DIR,
        help="dir containing per_group.parquet (default: the thinking-on profile's reduced/ dir; "
             f"direct-response profile: {artifact_path(PROFILE_DIRECT, 'reduced')})",
    )
    parser.add_argument(
        "--generations", default=DEFAULT_GENERATIONS_PATH,
        help="no-PI rollout generations jsonl matching --reduced-dir's underlying corpus "
             f"(default: {DEFAULT_GENERATIONS_PATH}; direct-response profile: "
             f"{artifact_path(PROFILE_DIRECT_STATES, 'qwen3_1.7b_generations.jsonl')})",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help="output dir for tables/plots/summary (default: the thinking-on profile's analysis dir).",
    )
    args = parser.parse_args()
    REDUCED_DIR = args.reduced_dir
    GENERATIONS_PATH = args.generations
    OUT_DIR = args.out_dir
    TABLES_DIR = os.path.join(OUT_DIR, "tables")
    PLOTS_DIR = os.path.join(OUT_DIR, "plots")

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_per_group()
    capped_lookup = load_capped_lookup(GENERATIONS_PATH)
    rng = np.random.default_rng(args.seed)

    think_answer = think_answer_accounting(df, args.n_boot, rng)
    quartiles = quartile_accounting(df, args.n_boot, rng)
    capped = capped_accounting(df, capped_lookup, args.n_boot, rng)

    not_derivable = {
        "after_a_correct_answer_has_already_appeared": (
            "NOT DERIVABLE from per_group.parquet. This component requires locating every literal "
            "answer proposal (each \\boxed{...} occurrence) at its exact token position within a "
            "trajectory and checking it against verified_answer -- i.e. per-position answer VALUE "
            "extraction. The reduction stores only per-token scalar statistics (logp, entropy, "
            "probabilities, category/region flags) computed against the KNOWN sampled token, never "
            "the token's value; only trajectory-level final correctness ('correct') is available. "
            "It would need a pass over the raw shards' token_id column."
        ),
        "inside_repeated_spans": (
            "NOT DERIVABLE from per_group.parquet, same root cause: repeated n-gram detection needs "
            "token identity, which the reduction does not carry. The repeated 4-gram/8-gram "
            "fractions, longest repeated span and suffix compression ratio would have to come from "
            "the raw shards' token_id column."
        ),
    }

    summary = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_problems": int(df["problem_id"].nunique()),
        "n_no_pi_samples_per_problem": 4,
        "n_pi_views": len(PI_VIEWS),
        "pi_views": PI_VIEWS,
        "before_after_first_answer_proxy": think_answer,
        "quartile_position_breakdown": quartiles,
        "capped_trajectories": capped,
        "not_derivable": not_derivable,
    }
    summary = to_jsonable(summary)

    with open(os.path.join(OUT_DIR, "censored_tail_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Tables (CSV, easy to skim without re-parsing the JSON).
    rows = []
    for v in PI_VIEWS:
        r = think_answer["by_view"][v]
        rows.append({
            "view": v,
            "n_problems": r["n_problems"],
            "think_token_share": r["think_token_share"],
            "answer_token_share": r["answer_token_share"],
            "kl_share_think_point": r["kl_share_think_pooled"]["point"],
            "kl_share_think_ci_lo": r["kl_share_think_pooled"]["ci_lo"],
            "kl_share_think_ci_hi": r["kl_share_think_pooled"]["ci_hi"],
            "kl_share_answer_point": r["kl_share_answer_pooled"]["point"],
            "kl_share_answer_ci_lo": r["kl_share_answer_pooled"]["ci_lo"],
            "kl_share_answer_ci_hi": r["kl_share_answer_pooled"]["ci_hi"],
            "mean_kl_per_think_token": r["mean_kl_per_think_token"],
            "mean_kl_per_answer_token": r["mean_kl_per_answer_token"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, "think_answer_by_view.csv"), index=False)

    rows = []
    for split, r in think_answer["by_correctness_all_views"].items():
        rows.append({
            "split": split,
            "n_problems": r["n_problems"],
            "n_rows": r["n_rows"],
            "kl_share_answer_point": r["kl_share_answer_pooled"]["point"],
            "kl_share_answer_ci_lo": r["kl_share_answer_pooled"]["ci_lo"],
            "kl_share_answer_ci_hi": r["kl_share_answer_pooled"]["ci_hi"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, "think_answer_by_correctness.csv"), index=False)

    rows = []
    for q in range(4):
        blk = quartiles[f"q{q}"]
        rows.append({
            "quartile": f"q{q}",
            "kl_share_point": blk["kl_share_pooled"]["point"],
            "kl_share_ci_lo": blk["kl_share_pooled"]["ci_lo"],
            "kl_share_ci_hi": blk["kl_share_pooled"]["ci_hi"],
            "token_share": blk["token_share"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, "quartile_position_breakdown.csv"), index=False)

    rows = []
    for key in ["final_capped", "ever_hit_length_cap"]:
        blk = capped[key]
        row = {
            "definition": key,
            "n_samples_flagged": blk["n_samples_flagged"],
            "n_samples_total": blk["n_samples_total"],
            "n_problems_with_flagged_sample": blk["n_problems_with_flagged_sample"],
            "kl_share_point": blk["kl_share_pooled_all_views"]["point"] if blk["kl_share_pooled_all_views"] else None,
            "kl_share_ci_lo": blk["kl_share_pooled_all_views"]["ci_lo"] if blk["kl_share_pooled_all_views"] else None,
            "kl_share_ci_hi": blk["kl_share_pooled_all_views"]["ci_hi"] if blk["kl_share_pooled_all_views"] else None,
            "token_share_point": blk["token_share_pooled_all_views"]["point"] if blk["token_share_pooled_all_views"] else None,
        }
        for v in PI_VIEWS:
            b = blk["kl_share_by_view"][v]
            row[f"kl_share_{v}"] = b["point"] if b else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, "capped_trajectories.csv"), index=False)

    make_plots(think_answer, quartiles, capped)

    print("wrote", os.path.join(OUT_DIR, "censored_tail_summary.json"))
    print("wrote tables to", TABLES_DIR)
    print("wrote plots to", PLOTS_DIR)


if __name__ == "__main__":
    main()
