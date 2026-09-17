"""Reduce the all_rollouts_prefix_scores shard parquets (~124M rows, ~7GB for a full
profile) into small intermediates so the downstream profile analyses never re-read the
raw shards.

Reads the shards via chunked pyarrow row-group iteration (never loads all rows into pandas
at once) and writes, under --out-dir:

    per_group.parquet        one row per (problem_id, sample_index, condition)
    position_profile.parquet one row per (condition, correct, normalized_position bin[50])
    marker_tokens.parquet    all rows where is_correction_marker is True (raw per-token fields)
    reduce_manifest.json     row counts, schema notes, undecidable-quantity report

Column semantics (see score_views.py's docstring):
  - logp - logp_student == log_ratio (0.0 for condition == "none"): per-token teacher-minus-
    student log-prob gap. "condition" IS the teacher/view role; "none" is the student baseline
    row (self-comparison, full_kl == 0, log_ratio == 0, top1_agree_with_student trivially True).
  - eos_prob / think_close_prob / answer_transition_prob are the *_student columns' condition-
    role counterparts already in the schema -- no join needed.

Approach: map-then-reduce. For each row-group batch, compute small groupby-sum/max/count
aggregates (group cardinality is bounded: <=512*4*7=14336 for per_group, <=7*2*50=700 for
position_profile) and append the tiny per-batch result to a list; after all batches from all
shards are consumed, concat the list and do one final groupby-combine, then compute derived
means/rates. marker_tokens is simpler: filter each batch and stream rows straight to a
ParquetWriter, no reduction needed.

Shards written by score_views.py carry token_id/clipped_kl/pi_length_tokens; older
shards without those columns are also accepted. Every aggregate that needs one of them is
column-presence-guarded, so the SAME script produces the base per_group/position_profile
schema on old shards and the extended schema on new ones -- additive columns only, no change
to any existing column's meaning. Sequential per-trajectory quantities that need token
identity (repeated n-grams, longest repeated span, post-correct-answer KL mass) are not
computed here: a separate pass over the raw shards would be needed for those.

    python profiling/reduce_profile.py --shards <parquet ...> --out-dir <profile>/reduced
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from opsd.artifact_layout import PROFILE_THINKING, path as artifact_path  # noqa: E402

DEFAULT_SHARDS = [
    artifact_path(PROFILE_THINKING, "all_rollouts_prefix_scores.shard0.parquet"),
    artifact_path(PROFILE_THINKING, "all_rollouts_prefix_scores.shard1.parquet"),
]
DEFAULT_OUT_DIR = artifact_path(PROFILE_THINKING, "reduced")
BATCH_SIZE = 1_000_000
N_POS_BINS = 50
EPS = 1e-7  # logit clip

STOP_MARKERS = ["eos_prob", "think_close_prob", "answer_transition_prob"]

PG_KEYS = ["problem_id", "sample_index", "condition"]
PP_KEYS = ["condition", "correct", "pos_bin"]


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def add_derived_columns(df):
    """Per-row derived columns shared by both reductions."""
    df["is_think"] = df["token_category"] == "think"
    df["is_answer"] = df["token_category"] == "answer"
    q = np.floor(df["normalized_position"].to_numpy() * 4).astype(int)
    df["quartile"] = np.clip(q, 0, 3)
    b = np.floor(df["normalized_position"].to_numpy() * N_POS_BINS).astype(int)
    df["pos_bin"] = np.clip(b, 0, N_POS_BINS - 1)
    df["disagree"] = ~df["top1_agree_with_student"]
    for m in STOP_MARKERS:
        stu = m + "_student"
        df[m + "_delta"] = df[m] - df[stu]
        df[m + "_logit_diff"] = logit(df[m].to_numpy()) - logit(df[stu].to_numpy())
    df["entropy_delta"] = df["entropy"] - df["entropy_student"]
    return df


# --------------------------------------------------------------------------- #
# per_group batch aggregation
# --------------------------------------------------------------------------- #


def per_group_batch_agg(df, has_clipped_kl=False, has_pi_length=False):
    g = df.groupby(PG_KEYS, sort=False, observed=True)

    named_aggs = dict(
        correct=("correct", "first"),
        n_tokens=("position", "size"),
        n_think_tokens=("is_think", "sum"),
        n_answer_tokens=("is_answer", "sum"),
        full_kl_sum=("full_kl", "sum"),
        log_ratio_sum=("log_ratio", "sum"),
        entropy_sum=("entropy", "sum"),
        entropy_student_sum=("entropy_student", "sum"),
        disagree_sum=("disagree", "sum"),
        marker_count=("is_correction_marker", "sum"),
    )
    if has_clipped_kl:
        named_aggs["clipped_kl_sum"] = ("clipped_kl", "sum")
    if has_pi_length:
        # constant within a (problem_id, sample_index, condition) group -- "first" not "sum".
        named_aggs["pi_length_tokens"] = ("pi_length_tokens", "first")
    out = g.agg(**named_aggs).reset_index()

    # full_kl (and, if present, clipped_kl) by token_category and by quartile: sum + count,
    # in wide form. Count (n_tokens_{cat}_chk / n_tokens_q{qi}) is the same regardless of
    # which kl_col it's computed from (same row mask), so it's derived once from full_kl
    # and kept as a single column even when clipped_kl is also being summed.
    kl_cols = ["full_kl"] + (["clipped_kl"] if has_clipped_kl else [])
    for cat_name, mask_col in [("think", "is_think"), ("answer", "is_answer")]:
        sub = df.loc[df[mask_col], PG_KEYS + kl_cols]
        grouped = sub.groupby(PG_KEYS, sort=False, observed=True)
        s = grouped[kl_cols].sum()
        s.columns = [f"{kl}_sum_{cat_name}" for kl in kl_cols]
        s[f"n_tokens_{cat_name}_chk"] = grouped.size()
        out = out.merge(s.reset_index(), on=PG_KEYS, how="left")

    for qi in range(4):
        sub = df.loc[df["quartile"] == qi, PG_KEYS + kl_cols]
        grouped = sub.groupby(PG_KEYS, sort=False, observed=True)
        s = grouped[kl_cols].sum()
        s.columns = [f"{kl}_sum_q{qi}" for kl in kl_cols]
        s[f"n_tokens_q{qi}"] = grouped.size()
        out = out.merge(s.reset_index(), on=PG_KEYS, how="left")

    # correction-marker rows: summed signed log_ratio.
    sub = df.loc[df["is_correction_marker"], PG_KEYS + ["log_ratio"]]
    s = sub.groupby(PG_KEYS, sort=False, observed=True)["log_ratio"].sum()
    s.name = "marker_log_ratio_sum"
    out = out.merge(s.reset_index(), on=PG_KEYS, how="left")

    # stop-marker mean/max for both roles + delta, per marker.
    agg_map = {}
    for m in STOP_MARKERS:
        stu = m + "_student"
        agg_map[f"{m}_sum"] = (m, "sum")
        agg_map[f"{m}_max"] = (m, "max")
        agg_map[f"{stu}_sum"] = (stu, "sum")
        agg_map[f"{stu}_max"] = (stu, "max")
        agg_map[f"{m}_delta_sum"] = (m + "_delta", "sum")
        agg_map[f"{m}_delta_max"] = (m + "_delta", "max")
    stop = g.agg(**agg_map).reset_index()
    out = out.merge(stop, on=PG_KEYS, how="left")

    return out


def combine_per_group(parts, has_clipped_kl=False, has_pi_length=False):
    all_df = pd.concat(parts, ignore_index=True)

    # Explicit column roles (name-pattern matching is fragile here since e.g.
    # "full_kl_sum_think" does not end with "_sum"): every non-key, non-"correct" column
    # produced by per_group_batch_agg is either additive across batches ("sum"), a running
    # extremum ("max"), or a per-group constant ("first"); list them explicitly rather than
    # guessing from suffixes.
    sum_cols = (
        ["n_tokens", "n_think_tokens", "n_answer_tokens", "full_kl_sum", "log_ratio_sum",
         "entropy_sum", "entropy_student_sum", "disagree_sum", "marker_count",
         "full_kl_sum_think", "n_tokens_think_chk", "full_kl_sum_answer", "n_tokens_answer_chk",
         "marker_log_ratio_sum"]
        + [f"full_kl_sum_q{qi}" for qi in range(4)]
        + [f"n_tokens_q{qi}" for qi in range(4)]
    )
    if has_clipped_kl:
        sum_cols += (
            ["clipped_kl_sum", "clipped_kl_sum_think", "clipped_kl_sum_answer"]
            + [f"clipped_kl_sum_q{qi}" for qi in range(4)]
        )
    first_cols = []
    if has_pi_length:
        first_cols += ["pi_length_tokens"]
    max_cols = []
    for m in STOP_MARKERS:
        stu = m + "_student"
        sum_cols += [f"{m}_sum", f"{stu}_sum", f"{m}_delta_sum"]
        max_cols += [f"{m}_max", f"{stu}_max", f"{m}_delta_max"]
    missing = set(sum_cols + max_cols + first_cols) - set(all_df.columns)
    assert not missing, f"combine_per_group: expected columns missing from per-batch parts: {missing}"

    agg_dict = {c: "sum" for c in sum_cols}
    agg_dict.update({c: "max" for c in max_cols})
    agg_dict.update({c: "first" for c in first_cols})
    agg_dict["correct"] = "first"
    combined = all_df.groupby(PG_KEYS, sort=False, observed=True).agg(agg_dict).reset_index()

    n = combined["n_tokens"].replace(0, np.nan)
    combined["full_kl_mean"] = combined["full_kl_sum"] / n
    combined["log_ratio_mean"] = combined["log_ratio_sum"] / n
    combined["entropy_mean"] = combined["entropy_sum"] / n
    combined["entropy_student_mean"] = combined["entropy_student_sum"] / n
    combined["top1_disagreement_rate"] = combined["disagree_sum"] / n
    combined["marker_log_ratio_mean"] = combined["marker_log_ratio_sum"] / combined["marker_count"].replace(0, np.nan)
    if has_clipped_kl:
        combined["clipped_kl_mean"] = combined["clipped_kl_sum"] / n

    for cat_name in ["think", "answer"]:
        nt = combined[f"n_{cat_name}_tokens"].replace(0, np.nan)
        combined[f"full_kl_mean_{cat_name}"] = combined[f"full_kl_sum_{cat_name}"] / nt
        if has_clipped_kl:
            combined[f"clipped_kl_mean_{cat_name}"] = combined[f"clipped_kl_sum_{cat_name}"] / nt
        combined.drop(columns=[f"n_tokens_{cat_name}_chk"], inplace=True, errors="ignore")

    for qi in range(4):
        nq = combined[f"n_tokens_q{qi}"].replace(0, np.nan)
        combined[f"full_kl_mean_q{qi}"] = combined[f"full_kl_sum_q{qi}"] / nq
        if has_clipped_kl:
            combined[f"clipped_kl_mean_q{qi}"] = combined[f"clipped_kl_sum_q{qi}"] / nq

    for m in STOP_MARKERS:
        stu = m + "_student"
        combined[f"{m}_mean"] = combined[f"{m}_sum"] / n
        combined[f"{stu}_mean"] = combined[f"{stu}_sum"] / n
        combined[f"{m}_delta_mean"] = combined[f"{m}_delta_sum"] / n
        combined.drop(columns=[f"{m}_sum", f"{stu}_sum", f"{m}_delta_sum"], inplace=True)

    ordered_cols = (
        PG_KEYS
        + ["correct", "n_tokens", "n_think_tokens", "n_answer_tokens"]
        + ["full_kl_sum", "full_kl_mean"]
        + [f"full_kl_sum_{c}" for c in ["think", "answer"]]
        + [f"full_kl_mean_{c}" for c in ["think", "answer"]]
        + [f"n_tokens_q{qi}" for qi in range(4)]
        + [f"full_kl_sum_q{qi}" for qi in range(4)]
        + [f"full_kl_mean_q{qi}" for qi in range(4)]
        + ["log_ratio_sum", "log_ratio_mean"]
        + ["entropy_mean", "entropy_student_mean"]
        + ["top1_disagreement_rate"]
        + ["marker_count", "marker_log_ratio_sum", "marker_log_ratio_mean"]
    )
    if has_clipped_kl:
        ordered_cols += (
            ["clipped_kl_sum", "clipped_kl_mean"]
            + [f"clipped_kl_sum_{c}" for c in ["think", "answer"]]
            + [f"clipped_kl_mean_{c}" for c in ["think", "answer"]]
            + [f"clipped_kl_sum_q{qi}" for qi in range(4)]
            + [f"clipped_kl_mean_q{qi}" for qi in range(4)]
        )
    if has_pi_length:
        ordered_cols += ["pi_length_tokens"]
    for m in STOP_MARKERS:
        stu = m + "_student"
        ordered_cols += [f"{m}_mean", f"{m}_max", f"{stu}_mean", f"{stu}_max", f"{m}_delta_mean", f"{m}_delta_max"]
    combined = combined[ordered_cols]
    return combined


# --------------------------------------------------------------------------- #
# position_profile batch aggregation
# --------------------------------------------------------------------------- #


def position_profile_batch_agg(df, has_clipped_kl=False):
    agg_map = {
        "n_tokens": ("position", "size"),
        "full_kl_sum": ("full_kl", "sum"),
        "entropy_delta_sum": ("entropy_delta", "sum"),
    }
    if has_clipped_kl:
        agg_map["clipped_kl_sum"] = ("clipped_kl", "sum")
    for m in STOP_MARKERS:
        stu = m + "_student"
        agg_map[f"{m}_sum"] = (m, "sum")
        agg_map[f"{stu}_sum"] = (stu, "sum")
        agg_map[f"{m}_delta_sum"] = (m + "_delta", "sum")
        agg_map[f"{m}_logit_diff_sum"] = (m + "_logit_diff", "sum")
    out = df.groupby(PP_KEYS, sort=False, observed=True).agg(**agg_map).reset_index()
    return out


def combine_position_profile(parts, has_clipped_kl=False):
    all_df = pd.concat(parts, ignore_index=True)
    sum_cols = [c for c in all_df.columns if c not in PP_KEYS]
    combined = all_df.groupby(PP_KEYS, sort=False, observed=True)[sum_cols].sum().reset_index()

    n = combined["n_tokens"].replace(0, np.nan)
    combined["mean_full_kl"] = combined["full_kl_sum"] / n
    combined["mean_entropy_delta"] = combined["entropy_delta_sum"] / n
    if has_clipped_kl:
        combined["mean_clipped_kl"] = combined["clipped_kl_sum"] / n
    for m in STOP_MARKERS:
        stu = m + "_student"
        combined[f"mean_{m}"] = combined[f"{m}_sum"] / n
        combined[f"mean_{stu}"] = combined[f"{stu}_sum"] / n
        combined[f"mean_{m}_delta"] = combined[f"{m}_delta_sum"] / n
        combined[f"mean_{m}_logit_diff"] = combined[f"{m}_logit_diff_sum"] / n
    combined["bin_center"] = (combined["pos_bin"] + 0.5) / N_POS_BINS

    ordered_cols = PP_KEYS + ["bin_center", "n_tokens", "mean_full_kl", "mean_entropy_delta"]
    if has_clipped_kl:
        ordered_cols += ["mean_clipped_kl"]
    for m in STOP_MARKERS:
        stu = m + "_student"
        ordered_cols += [f"mean_{m}", f"mean_{stu}", f"mean_{m}_delta", f"mean_{m}_logit_diff"]
    combined = combined[ordered_cols].sort_values(PP_KEYS).reset_index(drop=True)
    return combined


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards", nargs="+", default=DEFAULT_SHARDS,
        help="raw all_rollouts_prefix_scores parquet shard(s) to reduce (default: the "
             "thinking-on profile's shards).",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help="output dir for per_group.parquet/position_profile.parquet/marker_tokens.parquet/"
             "reduce_manifest.json (default: the thinking-on profile's reduced/ dir).",
    )
    parser.add_argument(
        "--expected-rows", type=int, default=512 * 4 * 7,
        help="expected per_group.parquet row count for the sanity check "
             "(default: 512 calibration problems x 4 no-PI samples x 7 conditions; override for "
             "--limit smoke runs).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Detect once, from the first shard's schema, whether the extended columns are present;
    # every aggregation below is guarded on these flags so both schemas produce a valid,
    # self-consistent output.
    first_schema_names = set(pq.ParquetFile(args.shards[0]).schema_arrow.names)
    has_clipped_kl = "clipped_kl" in first_schema_names
    has_pi_length = "pi_length_tokens" in first_schema_names
    has_token_id = "token_id" in first_schema_names
    print(f"schema detection ({args.shards[0]}): has_clipped_kl={has_clipped_kl}, "
          f"has_pi_length={has_pi_length}, has_token_id={has_token_id}")

    pg_parts, pp_parts = [], []
    marker_writer = None
    marker_schema = None
    n_total_rows = 0
    n_marker_rows = 0
    shard_row_counts = {}

    for shard_path in args.shards:
        pf = pq.ParquetFile(shard_path)
        shard_rows = 0
        for batch in pf.iter_batches(batch_size=BATCH_SIZE):
            df = batch.to_pandas()
            n_batch = len(df)
            n_total_rows += n_batch
            shard_rows += n_batch
            df = add_derived_columns(df)

            pg_parts.append(per_group_batch_agg(df, has_clipped_kl=has_clipped_kl, has_pi_length=has_pi_length))
            pp_parts.append(position_profile_batch_agg(df, has_clipped_kl=has_clipped_kl))

            marker_df = df.loc[df["is_correction_marker"]]
            # write raw (unadapted) columns only, matching source schema exactly -- this
            # already includes token_id/clipped_kl/pi_length_tokens automatically whenever
            # they're present in the source shard.
            marker_cols = [c for c in batch.schema.names]
            if len(marker_df):
                marker_table = pa.Table.from_pandas(marker_df[marker_cols], preserve_index=False)
                if marker_writer is None:
                    marker_schema = marker_table.schema
                    marker_writer = pq.ParquetWriter(
                        os.path.join(args.out_dir, "marker_tokens.parquet"), marker_schema
                    )
                marker_writer.write_table(marker_table.cast(marker_schema))
                n_marker_rows += len(marker_df)
        shard_row_counts[os.path.basename(shard_path)] = shard_rows
        print(f"{shard_path}: {shard_rows} rows processed (running total {n_total_rows})")

    if marker_writer is not None:
        marker_writer.close()
    else:
        # no marker rows at all (shouldn't happen) -- write an empty file with source schema
        pf0 = pq.ParquetFile(args.shards[0])
        pq.write_table(pf0.schema_arrow.empty_table(), os.path.join(args.out_dir, "marker_tokens.parquet"))

    print("combining per_group...")
    per_group = combine_per_group(pg_parts, has_clipped_kl=has_clipped_kl, has_pi_length=has_pi_length)
    per_group_path = os.path.join(args.out_dir, "per_group.parquet")
    per_group.to_parquet(per_group_path, index=False)

    print("combining position_profile...")
    position_profile = combine_position_profile(pp_parts, has_clipped_kl=has_clipped_kl)
    position_profile_path = os.path.join(args.out_dir, "position_profile.parquet")
    position_profile.to_parquet(position_profile_path, index=False)

    # sanity check: per_group should cover n_problems x n_samples x 7 conditions
    n_problems = per_group["problem_id"].nunique()
    n_conditions = per_group["condition"].nunique()
    expected_rows = args.expected_rows
    none_full_kl_max = per_group.loc[per_group["condition"] == "none", "full_kl_sum"].abs().max()

    sanity = {
        "n_problems": int(n_problems),
        "n_samples_per_problem_values": sorted(per_group["sample_index"].unique().tolist()),
        "n_conditions": int(n_conditions),
        "conditions": sorted(per_group["condition"].unique().tolist()),
        "per_group_rows": int(len(per_group)),
        "expected_rows": expected_rows,
        "rows_match_expected": bool(len(per_group) == expected_rows),
        "none_condition_full_kl_sum_max_abs": float(none_full_kl_max) if pd.notna(none_full_kl_max) else None,
        "none_condition_full_kl_is_zero": bool(none_full_kl_max == 0.0) if pd.notna(none_full_kl_max) else None,
    }
    print("sanity check:", json.dumps(sanity, indent=2))

    manifest = {
        "source_shards": shard_row_counts,
        "n_source_rows_total": n_total_rows,
        "outputs": {
            "per_group.parquet": {
                "rows": int(len(per_group)),
                "columns": list(per_group.columns),
                "grouping": PG_KEYS,
                "description": "one row per (problem_id, sample_index, condition); condition "
                "'none' is the student self-baseline (full_kl == 0, log_ratio == 0 by "
                "construction, top1_disagreement_rate == 0).",
            },
            "position_profile.parquet": {
                "rows": int(len(position_profile)),
                "columns": list(position_profile.columns),
                "grouping": PP_KEYS,
                "n_position_bins": N_POS_BINS,
                "description": "per (condition, correct, normalized_position bin in [0,50)) "
                "aggregates: mean_full_kl, mean_entropy_delta (condition-minus-student "
                "entropy), and per stop-definition (eos_prob/think_close_prob/"
                "answer_transition_prob) mean teacher prob, mean student prob, mean "
                "prob delta, and mean logit(teacher)-logit(student) (the S_{k,t} stop-"
                "pressure quantity, averaged per bin; logits are computed per-token from "
                "stored probabilities with epsilon=1e-7 clipping, then averaged -- NOT the "
                "logit of the averaged probability).",
            },
            "marker_tokens.parquet": {
                "rows": int(n_marker_rows),
                "columns": marker_schema.names if marker_schema is not None else [],
                "description": "all rows (all 7 conditions, i.e. no-PI baseline + 6 reference "
                "views) where is_correction_marker is True, verbatim per-token fields from the "
                "source shards, for the fork/correction analysis.",
            },
        },
        "sanity_check": sanity,
        "requested_quantities_not_derived": [
            {
                "quantity": "repeated 4-gram / 8-gram token fractions, longest repeated span, "
                "suffix compression ratio, post-correct-answer KL mass (per_group.parquet)",
                "reason": (
                    "NOT DERIVABLE from these shards: no token_id column present (an older "
                    "scored schema). Computing these would require re-tokenizing the raw "
                    "completions from the generations jsonl (output_token_ids / "
                    "rescued_output_token_ids fields) directly, which does not align with the "
                    "scored (possibly truncated) prefix -- not done by this reduction pass."
                    if not has_token_id else
                    "COMPUTABLE (token_id is present in this shard schema) but NOT COMPUTED by "
                    "this reduction pass: per_group's map-then-reduce structure (independent "
                    "row-group batches, order not guaranteed contiguous per trajectory) cannot "
                    "do the sequential-scan-per-(problem_id,sample_index) that n-gram detection, "
                    "longest-repeated-span, suffix-compression-ratio, and post-correct-answer "
                    "boundary detection all need. Those need a pass that streams one condition's "
                    "rows per (problem_id, sample_index) in position order -- token_id is "
                    "identical across all 7 condition rows at a given position, so scanning just "
                    "the 'none' condition's rows would be sufficient."
                ),
            }
        ],
        "notes": [
            "log_ratio in the source schema IS the teacher-minus-student per-token logp gap "
            "(logp - logp_student); per_group's log_ratio_sum/log_ratio_mean reduce it directly.",
            "'condition' plays the teacher/view role; 'none' is the no-PI student baseline row "
            "(same underlying completion, self-compared) -- full_kl, log_ratio, and "
            "top1_disagreement_rate are identically 0 for condition == 'none' by construction, "
            "confirmed in sanity_check.",
            "Quartiles for per_group's full_kl-by-position breakdown are true quartiles "
            "(normalized_position in [0,.25)/[.25,.5)/[.5,.75)/[.75,1]), computed independently "
            "of the source schema's pre-existing 'region' column (first25/mid50/last25, a "
            "25/50/25 tertile split kept from prefix_score.py) -- 'region' is still present "
            "verbatim in marker_tokens.parquet if a 3-way split is wanted instead.",
            "position_profile's normalized_position bins are 50 equal-width bins over [0,1] "
            "(bin i covers [i/50, (i+1)/50)); bin_center = (i+0.5)/50.",
            "correction-marker set is constants.CORRECTION_MARKERS (wait/but/however/maybe/"
            "actually/recheck/check/verify/reconsider/mistake/instead), lexical substring match "
            "on decoded token piece, case-insensitive, same flag repeated across all 7 "
            "condition rows for a given (problem_id, sample_index, position).",
        ],
    }
    manifest_path = os.path.join(args.out_dir, "reduce_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote {per_group_path} ({len(per_group)} rows)")
    print(f"wrote {position_profile_path} ({len(position_profile)} rows)")
    print(f"wrote {os.path.join(args.out_dir, 'marker_tokens.parquet')} ({n_marker_rows} rows)")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
