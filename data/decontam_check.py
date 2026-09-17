#!/usr/bin/env python3
"""Decontamination check: does AMPLE-Math (5,319 unique problems) overlap with the external
benchmarks AIME 2024, AIME 2025 and HMMT February 2025 as used by the external evaluation?

Reads the local AMPLE-Math jsonl (AMPLE_MATH_JSONL) and the benchmark files
{aime24,aime25,hmmt25}.json under OPSD_BENCHMARKS (default <OPSD_ARTIFACTS>/benchmarks).
Read-only, CPU-only; writes <OPSD_ARTIFACTS>/decontam/decontam_full_results.json.

Metrics per benchmark problem X against the corpus {Y}:
  (a) exact_match: after aggressive normalization (lowercase; strip literal
      '$' and '\\' characters; then strip every remaining non-alphanumeric
      character, i.e. all whitespace and punctuation), does X's normalized
      string equal any Y's normalized string exactly?
  (b) max Jaccard over 8-gram word-shingle sets: tokenize (lowercase, '$' and
      '\\' stripped, split into maximal [a-z0-9]+ runs) X and every Y, form
      the set of contiguous 8-token shingles for each, and report
      max_jaccard = max_Y |S_X ^ S_Y| / |S_X u S_Y| plus the argmax Y id.
  (c) containment = max_Y |S_X ^ S_Y| / |S_X|, i.e. the largest fraction of
      X's 8-grams that appear in any *single* corpus problem Y (tracked
      with its own argmax id, which can differ from the Jaccard argmax when
      problem lengths differ a lot -- e.g. a short X fully contained in a
      much longer Y has containment 1.0 but a modest Jaccard).

Flag X as a possible near-duplicate if exact_match OR containment >= 0.5.

An inverted index (shingle -> set of corpus ids) restricts scoring to corpus
problems sharing >=1 shingle with X, instead of scoring all 5,319
problems against each of the 90 benchmark problems.
"""
import json
import os
import re
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from opsd.constants import AMPLE_MATH_JSONL  # noqa: E402
from opsd import artifact_layout as layout  # noqa: E402

SUPERVISION_PATH = AMPLE_MATH_JSONL
BENCH_DIR = os.environ.get("OPSD_BENCHMARKS", layout.path(layout.BENCHMARKS))
BENCH_FILES = {"aime24": "aime24.json", "aime25": "aime25.json", "hmmt25": "hmmt25.json"}
OUT_DIR = layout.path(layout.DECONTAM)
N = 8
FLAG_CONTAINMENT_THRESHOLD = 0.5

WORD_RE = re.compile(r"[a-z0-9]+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]")


def normalize_exact(text: str) -> str:
    t = text.lower()
    t = t.replace("$", "").replace("\\", "")
    t = NON_ALNUM_RE.sub("", t)
    return t


def tokenize(text: str) -> list[str]:
    t = text.lower().replace("$", "").replace("\\", "")
    return WORD_RE.findall(t)


def shingles(tokens: list[str], n: int = N) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def load_corpus() -> dict[str, str]:
    seen: dict[str, str] = {}
    with open(SUPERVISION_PATH) as f:
        for line in f:
            d = json.loads(line)
            pid = d["problem_id"]
            if pid not in seen:
                seen[pid] = d["prompt"]
    return seen


def load_benchmark(name: str) -> list[dict]:
    with open(os.path.join(BENCH_DIR, BENCH_FILES[name])) as f:
        return json.load(f)


def main() -> None:
    if SUPERVISION_PATH is None:
        raise SystemExit("set AMPLE_MATH_JSONL to the local AMPLE-Math jsonl before running this check")
    os.makedirs(OUT_DIR, exist_ok=True)

    corpus = load_corpus()
    assert len(corpus) == 5319, f"expected 5319 unique corpus problems, got {len(corpus)}"

    pi_exact: dict[str, str] = {}
    pi_shingles: dict[str, set] = {}
    inverted: dict[tuple, set] = defaultdict(set)
    exact_index: dict[str, list] = defaultdict(list)

    for pid, text in corpus.items():
        ex = normalize_exact(text)
        pi_exact[pid] = ex
        exact_index[ex].append(pid)
        toks = tokenize(text)
        sh = shingles(toks)
        pi_shingles[pid] = sh
        for s in sh:
            inverted[s].add(pid)

    print(f"corpus: {len(corpus)} unique problems, "
          f"{sum(len(v) for v in pi_shingles.values())} total 8-gram shingles, "
          f"{len(inverted)} distinct shingles.")

    results: dict[str, list[dict]] = {}
    for bench_name in ["aime24", "aime25", "hmmt25"]:
        rows = load_benchmark(bench_name)
        assert len(rows) == 30, f"{bench_name}: expected 30 problems, got {len(rows)}"
        bench_results = []
        for row in rows:
            bpid = row["problem_id"]
            btext = row["question"]
            b_ex = normalize_exact(btext)
            b_tokens = tokenize(btext)
            b_sh = shingles(b_tokens)

            exact_matches = exact_index.get(b_ex, []) if b_ex else []

            candidates: set = set()
            for s in b_sh:
                candidates |= inverted.get(s, set())

            best_jaccard = 0.0
            best_jaccard_id = None
            best_containment = 0.0
            best_containment_id = None

            for cid in candidates:
                y_sh = pi_shingles[cid]
                inter = len(b_sh & y_sh)
                if inter == 0:
                    continue
                union = len(b_sh | y_sh)
                jac = inter / union if union else 0.0
                cont = inter / len(b_sh) if b_sh else 0.0
                if jac > best_jaccard:
                    best_jaccard = jac
                    best_jaccard_id = cid
                if cont > best_containment:
                    best_containment = cont
                    best_containment_id = cid

            flagged = bool(exact_matches) or best_containment >= FLAG_CONTAINMENT_THRESHOLD

            # illustrative example: one shared 8-gram against the best jaccard match
            example_shingle = None
            if best_jaccard_id is not None:
                shared = b_sh & pi_shingles[best_jaccard_id]
                if shared:
                    example_shingle = " ".join(sorted(shared, key=lambda s: -len(" ".join(s)))[0])

            bench_results.append({
                "benchmark": bench_name,
                "problem_id": bpid,
                "question_preview": btext[:120].replace("\n", " "),
                "n_tokens": len(b_tokens),
                "n_shingles": len(b_sh),
                "n_candidate_corpus_problems": len(candidates),
                "exact_match": bool(exact_matches),
                "exact_match_ids": exact_matches,
                "best_jaccard": round(best_jaccard, 4),
                "best_jaccard_corpus_id": best_jaccard_id,
                "best_containment": round(best_containment, 4),
                "best_containment_corpus_id": best_containment_id,
                "flagged": flagged,
                "example_shared_8gram": example_shingle,
            })
        results[bench_name] = bench_results

    out_path = os.path.join(OUT_DIR, "decontam_full_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote full per-problem results to {out_path}")

    print()
    for bench_name, rows in results.items():
        n = len(rows)
        n_exact = sum(1 for r in rows if r["exact_match"])
        n_flag = sum(1 for r in rows if r["flagged"])
        max_jac_overall = max(r["best_jaccard"] for r in rows)
        max_cont_overall = max(r["best_containment"] for r in rows)
        print(f"== {bench_name}: n={n} exact_matches={n_exact} flagged={n_flag} "
              f"max_jaccard_overall={max_jac_overall} max_containment_overall={max_cont_overall} ==")
        top5 = sorted(rows, key=lambda r: r["best_jaccard"], reverse=True)[:5]
        for r in top5:
            print(f"  {r['problem_id']}: jaccard={r['best_jaccard']} (vs {r['best_jaccard_corpus_id']}), "
                  f"containment={r['best_containment']} (vs {r['best_containment_corpus_id']}), "
                  f"exact={r['exact_match']}, n_shingles={r['n_shingles']}, "
                  f"candidates={r['n_candidate_corpus_problems']}")
            if r["example_shared_8gram"]:
                print(f"      example shared 8-gram: {r['example_shared_8gram']!r}")
        print()


if __name__ == "__main__":
    main()
