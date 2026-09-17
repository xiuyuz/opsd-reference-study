#!/usr/bin/env python3
"""Drift-profile reduction: the privilege profile re-measured on trained students' own
step-<STEP> states, compared with the frozen base-model direct-response profile.

Reads, per view v, the profile-chain outputs run_drift_profile.sh produced on v's OWN
step-<STEP> states (<drift-root>/<v>_step<STEP>/analysis/...) and the base profile
(<base-dir>/analysis/...), and writes a markdown table. Numbers only; a missing input is
reported as missing, never inferred.

Note on the "common-states" variant: score_views.py always scores ALL SEVEN contexts
(the six reference views + `none`) against whatever states it is given, so each <v>_step<STEP>
run already carries every view's C_k measured on view v's states. The own-student C_v is the
diagonal of that 6x6 matrix; the off-diagonal is the common-states variant, free of extra GPU
time.

    python profiling/drift_profile_compare.py \\
        --step 50 --views answer_only gist key_points clean_solution summary full_trace
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from opsd.artifact_layout import DRIFT_PROFILE, PROFILE_DIRECT, path as artifact_path  # noqa: E402

DEFAULT_BASE_DIR = artifact_path(PROFILE_DIRECT)
DEFAULT_DRIFT_ROOT = artifact_path(DRIFT_PROFILE)
VIEW_ORDER = ["answer_only", "gist", "key_points", "clean_solution", "summary", "full_trace"]


def prof_dir(drift_root, view, step):
    return os.path.join(drift_root, f"{view}_step{step}")


def read_alignment(root):
    """{view: (C_k, ci_lo, ci_hi, n_problems)} from a correctness-alignment summary."""
    p = os.path.join(root, "analysis", "correctness-alignment", "correctness_alignment_summary.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {r["view"]: (r["C_k"], r["ci_lo"], r["ci_hi"], r["n_problems"])
            for r in d["overall_alignment"]}, d["config"]


def read_fork(root):
    p = os.path.join(root, "analysis", "fork-correction", "fork_by_view.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p).set_index("view")
    return {v: (df.loc[v, "log_ratio_mean"], df.loc[v, "log_ratio_lo"], df.loc[v, "log_ratio_hi"],
                int(df.loc[v, "n_tokens"])) for v in df.index}


def read_q0(root):
    p = os.path.join(root, "analysis", "censored-tail", "tables", "quartile_position_breakdown.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p).set_index("quartile")
    return (float(df.loc["q0", "kl_share_point"]), float(df.loc["q0", "kl_share_ci_lo"]),
            float(df.loc["q0", "kl_share_ci_hi"]))


def spearman(a, b):
    """Spearman rho between two rank vectors (no ties expected; average ranks if any)."""
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for pos, i in enumerate(order):
            r[i] = pos + 1.0
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return num / (da * db) if da and db else float("nan")


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--views", nargs="+", default=VIEW_ORDER)
    ap.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                    help="frozen base-model direct-response profile root (default artifact_layout.PROFILE_DIRECT)")
    ap.add_argument("--drift-root", default=DEFAULT_DRIFT_ROOT,
                    help="root holding <view>_step<STEP>/ profile dirs (default artifact_layout.DRIFT_PROFILE)")
    ap.add_argument("--out", default=None,
                    help="markdown output path (default <drift-root>/DRIFT_PROFILE_RESULT.md)")
    args = ap.parse_args()
    step = args.step
    views = [v for v in VIEW_ORDER if v in args.views]
    base_dir = args.base_dir
    drift_root = args.drift_root
    out = args.out or os.path.join(drift_root, "DRIFT_PROFILE_RESULT.md")

    base = read_alignment(base_dir)
    if base is None:
        print(f"FATAL: base correctness-alignment summary not found under {base_dir}", file=sys.stderr)
        return 1
    base_align, base_cfg = base
    base_fork = read_fork(base_dir) or {}
    base_q0 = read_q0(base_dir)

    drift_align, drift_cfg, drift_fork, drift_q0, missing = {}, {}, {}, {}, []
    for v in views:
        r = prof_dir(drift_root, v, step)
        a = read_alignment(r)
        if a is None:
            missing.append(v)
            continue
        drift_align[v], drift_cfg[v] = a
        drift_fork[v] = read_fork(r) or {}
        drift_q0[v] = read_q0(r)

    have = [v for v in views if v in drift_align]

    # ---- gate -----------------------------------------------------------------------
    gate = {"evaluable": False}
    if "answer_only" in drift_align and "full_trace" in drift_align:
        ao = drift_align["answer_only"]["answer_only"]
        ft = drift_align["full_trace"]["full_trace"]
        b_ao, b_ft = base_align["answer_only"], base_align["full_trace"]
        ao_out = not (b_ao[1] <= ao[0] <= b_ao[2])
        ft_out = not (b_ft[1] <= ft[0] <= b_ft[2])
        base_order_ao_lt_ft = b_ao[0] < b_ft[0]
        drift_order_ao_lt_ft = ao[0] < ft[0]
        flip = base_order_ao_lt_ft != drift_order_ao_lt_ft
        gate = {
            "evaluable": True,
            "ao": ao, "ft": ft, "base_ao": b_ao, "base_ft": b_ft,
            "ao_outside_base_interval": ao_out, "ft_outside_base_interval": ft_out,
            "ao_ft_order_flip": flip,
            "branch": "A" if (ao_out or ft_out or flip) else "B",
        }

    L = []
    w = L.append
    w(f"# The direct-response privilege profile on step-{step} student states\n")
    w("Numbers only; the gate below is applied mechanically.\n")
    w("## What changed and what did not\n")
    w("| | frozen base profile | this run |")
    w("|---|---|---|")
    w("| states scored | frozen base model, direct-response, no reference | each seed-0 view's own "
      f"step-{step} student (LoRA served live), direct-response, no reference |")
    w("| problems / samples / seeds | calibration problems x 4 samples, "
      "`sample_seed(0, sample_index, item_index)` | identical |")
    w("| prompt style / generation cap | LOCAL / 16,384 + rescue | identical |")
    w("| scoring | frozen base + reference (7 contexts) teacher, frozen base no-reference student "
      "reference, `--scoring-horizon 1024`, 2 shards | identical (the adapter is used ONLY "
      "for generation, never for scoring) |")
    w("")
    if missing:
        w(f"**MISSING views (no analysis outputs on disk): {', '.join(missing)}** — every "
          "number below is over the views that do exist.\n")

    # ---- primary (i): C_v on own-student states --------------------------------------
    w(f"## (i) Correctness alignment C_v on each student's own step-{step} states\n")
    w("`C_k` from `correctness_alignment.py`, problem-cluster bootstrap 95% CI, n_boot="
      f"{base_cfg['n_boot']}, seed={base_cfg['seed']}. \"base\" is the frozen "
      f"`{base_dir}/analysis/correctness-alignment/` value for the same view.\n")
    w("| view | C_v on own states [95% CI] | n_problems | base C_v [95% CI] | base n | point outside base CI? |")
    w("|---|---|---|---|---|---|")
    for v in have:
        c, lo, hi, n = drift_align[v][v]
        b, blo, bhi, bn = base_align[v]
        outside = not (blo <= c <= bhi)
        w(f"| {v} | {c:+.4f} [{lo:+.4f}, {hi:+.4f}] | {n} | {b:+.4f} [{blo:+.4f}, {bhi:+.4f}] "
          f"| {bn} | {'YES' if outside else 'no'} |")
    w("")

    # ---- primary (ii): rank vs base -----------------------------------------------------
    w("## (ii) Six-view C_v rank on own-student states vs the base rank\n")
    if len(have) == len(VIEW_ORDER):
        drift_vec = [drift_align[v][v][0] for v in VIEW_ORDER]
        base_vec = [base_align[v][0] for v in VIEW_ORDER]
        rho = spearman(drift_vec, base_vec)
        w(f"Spearman rho (own-student C_v vs base C_v, six views) = **{rho:+.4f}**\n")
        w("| rank (low C to high C) | base | own-student states |")
        w("|---|---|---|")
        b_sorted = sorted(VIEW_ORDER, key=lambda v: base_align[v][0])
        d_sorted = sorted(VIEW_ORDER, key=lambda v: drift_align[v][v][0])
        for i in range(len(VIEW_ORDER)):
            w(f"| {i + 1} | {b_sorted[i]} ({base_align[b_sorted[i]][0]:+.4f}) "
              f"| {d_sorted[i]} ({drift_align[d_sorted[i]][d_sorted[i]][0]:+.4f}) |")
    else:
        w(f"NOT COMPUTED — the rank statistic needs all six views; {len(have)} present.")
    w("")

    # ---- secondary: marker pressure ---------------------------------------------------
    w("## Secondary — correction-marker pressure (`fork_correction.py`, own-view row)\n")
    w("Signed log-ratio at correction-marker tokens for view v measured on view v's own "
      "step-%d states, against the base value for the same view.\n" % step)
    w("| view | log_ratio on own states [95% CI] | n_marker_tokens | base log_ratio [95% CI] | base n_tokens |")
    w("|---|---|---|---|---|")
    for v in have:
        d = drift_fork.get(v, {}).get(v)
        b = base_fork.get(v)
        if d is None or b is None:
            w(f"| {v} | MISSING | | | |")
            continue
        w(f"| {v} | {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] | {d[3]} "
          f"| {b[0]:+.4f} [{b[1]:+.4f}, {b[2]:+.4f}] | {b[3]} |")
    w("")

    # ---- secondary: KL first-quarter share ---------------------------------------------
    w("## Secondary — KL temporal allocation: first-quarter share (pooled over the 6 reference views)\n")
    w("`censored_tail.py` `quartile_position_breakdown.csv`, quartile q0 (first quarter of the "
      "trajectory by normalized position); the flat null is 25%.\n")
    w("| states | q0 KL share [95% CI] |")
    w("|---|---|")
    if base_q0:
        w(f"| frozen base model | {base_q0[0] * 100:.2f}% [{base_q0[1] * 100:.2f}%, {base_q0[2] * 100:.2f}%] |")
    for v in have:
        q = drift_q0.get(v)
        w(f"| {v} step-{step} student | " +
          (f"{q[0] * 100:.2f}% [{q[1] * 100:.2f}%, {q[2] * 100:.2f}%] |" if q else "MISSING |"))
    w("")

    # ---- secondary: common-states variant ----------------------------------------------
    w("## Secondary — common-states variant (all seven contexts on every student's states)\n")
    w("C_k of the reference view in the ROW, measured on the states generated by the student in "
      "the COLUMN. The diagonal is the own-student C_v of section (i); the last column is the "
      "frozen base profile (base-model states). Every cell comes from the same scoring run as "
      "its column, so no extra generation was needed.\n")
    header = "| view \\ states | " + " | ".join(f"{v}@{step}" for v in have) + " | base |"
    w(header)
    w("|---" * (len(have) + 2) + "|")
    for pi in VIEW_ORDER:
        cells = []
        for sv in have:
            e = drift_align[sv].get(pi)
            cells.append(f"{e[0]:+.4f}" if e else "n/a")
        b = base_align.get(pi)
        w(f"| {pi} | " + " | ".join(cells) + f" | {b[0]:+.4f} |")
    w("")

    # ---- gate --------------------------------------------------------------------------
    w("## Gate\n")
    w("> If either the answer_only or the full_trace C_v on drifted states falls outside the "
      "corresponding base interval, or the answer_only/full_trace order flips, on-policy "
      "optimization moved the states on which the teacher acts and the profile with them. If "
      "neither moves, the profile fails even on the states the students visit at this step.\n")
    if not gate["evaluable"]:
        w("**NOT EVALUABLE** — the gate needs both `answer_only` and `full_trace`; one or both "
          "are missing from this run.")
    else:
        w(f"- answer_only: C_v = {gate['ao'][0]:+.4f}, base interval "
          f"[{gate['base_ao'][1]:+.4f}, {gate['base_ao'][2]:+.4f}] -> "
          f"{'OUTSIDE' if gate['ao_outside_base_interval'] else 'inside'}")
        w(f"- full_trace: C_v = {gate['ft'][0]:+.4f}, base interval "
          f"[{gate['base_ft'][1]:+.4f}, {gate['base_ft'][2]:+.4f}] -> "
          f"{'OUTSIDE' if gate['ft_outside_base_interval'] else 'inside'}")
        w(f"- AO/FT order: base {'AO < FT' if gate['base_ao'][0] < gate['base_ft'][0] else 'AO > FT'}"
          f", drifted {'AO < FT' if gate['ao'][0] < gate['ft'][0] else 'AO > FT'} -> "
          f"{'FLIP' if gate['ao_ft_order_flip'] else 'no flip'}")
        w("")
        if gate["branch"] == "A":
            w("**BRANCH FIRED: the first branch.** At least one of {AO outside its base interval, "
              "FT outside its base interval, AO/FT order flip} holds: on-policy optimization "
              "moves the states on which the teacher acts and the profile with them.")
        else:
            w("**BRANCH FIRED: the second branch.** Neither AO nor FT moved outside its base "
              "interval and the AO/FT order did not flip: the profile fails even on the states "
              f"the students visit at step {step}.")
    w("")
    w("## Provenance\n")
    w(f"- drifted-state chain: `profiling/run_drift_profile.sh` -> `{drift_root}/<view>_step{step}/`")
    w(f"- base reference: `{base_dir}/`")
    w("- this table: `profiling/drift_profile_compare.py`")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {out}")
    if gate.get("evaluable"):
        print(f"gate branch: {gate['branch']}")
    if missing:
        print(f"MISSING views: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
