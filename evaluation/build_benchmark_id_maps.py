"""Benchmark-to-anchor id maps: <bench>_id_map_bench_to_anchor.json, one per external benchmark.

The two sides of every external comparison number the same 30 problems differently. The trained
students are evaluated through `evaluation/run_external_eval.sh`, which takes its problems from
`opsd.benchmarks` and keeps that file's problem ids ("aime24-2024-I-1", "aime25-0",
"hmmt25-1"). The frozen-base anchors come from `evaluation/official_eval.py`, which runs the
official evaluator's own dataset loaders and stores whatever integer ids those assign. Pairing a
student with the base therefore needs a map between the two schemes, and every external
comparison in the paper joins on it.

The map is built from the problem statements. A benchmark problem is paired with the anchor
problem whose statement is the same once whitespace is removed; the statements that differ only
in how the source wrote its LaTeX are paired by closest text (difflib) among the problems still
unpaired, and only when that candidate is far ahead of the runner-up. Every pair, however it was
made, must also carry the same answer key, and the result must pair all 30 problems one to one
-- the script fails otherwise rather than write a partial map.

Each map is written where the analyses read it: the aime24 map under EXTERNAL_ID_MAP_AIME24, the
aime25 and hmmt25 maps under EXTERNAL_STEP50.

CPU only, no network (the benchmark files come from `python -m opsd.benchmarks --materialize`,
the anchors from the base runs of `official_eval.py`). Run:
  python evaluation/build_benchmark_id_maps.py
"""

import argparse
import difflib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd.artifact_layout import (  # noqa: E402
    EXTERNAL_BASE_ANCHORS,
    EXTERNAL_ID_MAP_AIME24,
    EXTERNAL_STEP50,
    path as artifact_path,
    resolve,
)
from opsd.benchmarks import load_benchmark  # noqa: E402

# where each map is read from (analysis/estimator.py's ID_MAPS)
ID_MAP_DIR = {"aime24": EXTERNAL_ID_MAP_AIME24,
              "aime25": EXTERNAL_STEP50,
              "hmmt25": EXTERNAL_STEP50}
# a closest-text pair is accepted only this far ahead of the second-best candidate
MIN_RATIO = 0.5
MIN_MARGIN = 0.15


def norm_text(s):
    return re.sub(r"\s+", "", s or "")


def norm_answer(s):
    """Answer keys as the two sides store them: "23" and "023" are the same key."""
    s = re.sub(r"\s+", "", str(s)).strip("$")
    try:
        return str(int(s))
    except ValueError:
        return s


def load_anchor(anchor_dir, bench):
    path = os.path.join(anchor_dir, f"base_{bench}.json")
    if not os.path.exists(path):
        raise SystemExit(f"no frozen-base anchor at {path}")
    return json.load(open(path))["results"], path


def build(bench, anchor_dir):
    problems = load_benchmark(bench)
    anchors, anchor_path = load_anchor(anchor_dir, bench)
    if len(problems) != len(anchors):
        raise RuntimeError(f"{bench}: {len(problems)} benchmark problems vs {len(anchors)} in "
                           f"{anchor_path}")
    answer_of = {a["problem_id"]: a["ground_truth"] for a in anchors}

    by_text = {}
    for a in anchors:
        by_text.setdefault(norm_text(a["problem"]), []).append(a["problem_id"])
    id_map, used = {}, set()
    for p in problems:
        hit = by_text.get(norm_text(p["question"]), [])
        if len(hit) == 1:
            id_map[p["problem_id"]] = hit[0]
            used.add(hit[0])
    n_exact = len(id_map)

    # the rest pair by closest text among the anchors still unpaired, best candidate first
    left = [p for p in problems if p["problem_id"] not in id_map]
    free = [a for a in anchors if a["problem_id"] not in used]
    scored = []
    for p in left:
        ratios = sorted(((difflib.SequenceMatcher(None, norm_text(p["question"]),
                                                  norm_text(a["problem"])).ratio(), a["problem_id"])
                         for a in free), reverse=True)
        scored.append((ratios[0][0], ratios[1][0] if len(ratios) > 1 else 0.0,
                       p["problem_id"], ratios[0][1]))
    for best, runner_up, pid, anchor_id in sorted(scored, reverse=True):
        if anchor_id in used:
            raise RuntimeError(f"{bench}: {pid} and another problem both match anchor {anchor_id}")
        if best < MIN_RATIO or best - runner_up < MIN_MARGIN:
            raise RuntimeError(f"{bench}: {pid} has no clear counterpart (closest {best:.3f}, "
                               f"next {runner_up:.3f}); pair it by hand")
        id_map[pid] = anchor_id
        used.add(anchor_id)

    # every pair carries the same answer key, and the map is a bijection over all 30 problems
    for p in problems:
        if norm_answer(p["verified_answer"]) != norm_answer(answer_of[id_map[p["problem_id"]]]):
            raise RuntimeError(f"{bench}: {p['problem_id']} -> {id_map[p['problem_id']]} but the "
                               f"answer keys differ ({p['verified_answer']!r} vs "
                               f"{answer_of[id_map[p['problem_id']]]!r})")
    if len(id_map) != len(problems) or len(set(id_map.values())) != len(problems):
        raise RuntimeError(f"{bench}: {len(id_map)} pairs over {len(set(id_map.values()))} anchors "
                           f"for {len(problems)} problems")
    print(f"{bench}: {len(id_map)}/{len(problems)} problems paired ({n_exact} on identical text, "
          f"{len(id_map) - n_exact} on closest text), all answer keys agree; anchor {anchor_path}")
    return {p["problem_id"]: id_map[p["problem_id"]] for p in problems}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--benchmarks", default=",".join(ID_MAP_DIR), help="comma-separated")
    ap.add_argument("--anchors", default=artifact_path(EXTERNAL_BASE_ANCHORS),
                    help="directory of the frozen-base results base_<benchmark>.json: a path or "
                         "an artifact_layout constant (default EXTERNAL_BASE_ANCHORS)")
    ap.add_argument("--out-dir", default=None,
                    help="write every map here instead of where the analyses read it")
    a = ap.parse_args()

    anchor_dir = resolve(a.anchors)
    for bench in [b.strip() for b in a.benchmarks.split(",") if b.strip()]:
        if bench not in ID_MAP_DIR:
            raise SystemExit(f"unknown benchmark {bench!r}; known: {', '.join(ID_MAP_DIR)}")
        id_map = build(bench, anchor_dir)
        out_dir = resolve(a.out_dir) if a.out_dir else artifact_path(ID_MAP_DIR[bench])
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{bench}_id_map_bench_to_anchor.json")
        with open(out, "w") as f:
            json.dump(id_map, f, indent=2)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
