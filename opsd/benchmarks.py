"""Loads the external evaluation benchmarks from pre-materialized JSON.

`load_benchmark` does not touch Hugging Face or the network -- it only reads the JSON
files under BENCHMARKS_DIR (env OPSD_BENCHMARKS, default artifact_layout.BENCHMARKS).
Those files are built once from the following HF datasets (exact revisions pinned):

    aime24 <- Maxwell-Jia/AIME_2024   revision 8d88b2876a82a080e2f172cc9b25d0d9d2cb4792
              fields used: ID -> problem_id ("aime24-{ID}"), Problem -> question,
              Answer (int) -> verified_answer (str). 30 rows.
    aime25 <- yentinglin/aime_2025    revision 6f71d77b0b89b9dabe07ab466c51df33f514df7f
              fields used: id -> problem_id ("aime25-{id}"), problem -> question,
              answer (str) -> verified_answer. 30 rows.
    hmmt25 <- MathArena/hmmt_feb_2025 revision 6fdc4277120810ff75aa22d2d5489b91f7a262a1
              fields used: problem_idx -> problem_id ("hmmt25-{problem_idx}"),
              problem -> question, answer (str, may be LaTeX e.g. "\\frac{1}{576}")
              -> verified_answer. 30 rows.

To (re)build them:

    python -m opsd.benchmarks --materialize [--out-dir <dir>]
"""

import argparse
import json
import os

if __name__ == "__main__" and __package__ is None:  # allow "python opsd/benchmarks.py" as well as "-m opsd.benchmarks"
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from opsd.artifact_layout import BENCHMARKS, path as artifact_path

BENCHMARKS_DIR = os.environ.get("OPSD_BENCHMARKS", artifact_path(BENCHMARKS))

BENCHMARK_FILES = {
    "aime24": "aime24.json",
    "aime25": "aime25.json",
    "hmmt25": "hmmt25.json",
}

# (dataset id, revision, id field, question field, answer field)
BENCHMARK_SOURCES = {
    "aime24": ("Maxwell-Jia/AIME_2024", "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792", "ID", "Problem", "Answer"),
    "aime25": ("yentinglin/aime_2025", "6f71d77b0b89b9dabe07ab466c51df33f514df7f", "id", "problem", "answer"),
    "hmmt25": ("MathArena/hmmt_feb_2025", "6fdc4277120810ff75aa22d2d5489b91f7a262a1", "problem_idx", "problem", "answer"),
}


def load_benchmark(name: str) -> list[dict]:
    """Return [{"problem_id", "question", "verified_answer"}, ...] for name in
    {"aime24", "aime25", "hmmt25"}. verified_answer is always a string."""
    path = os.path.join(BENCHMARKS_DIR, BENCHMARK_FILES[name])
    with open(path) as f:
        return json.load(f)


def materialize(out_dir=BENCHMARKS_DIR):
    """Pull the three datasets at their pinned revisions and write the JSON files."""
    from datasets import load_dataset

    os.makedirs(out_dir, exist_ok=True)
    for name, (dataset_id, revision, id_field, q_field, a_field) in BENCHMARK_SOURCES.items():
        ds = load_dataset(dataset_id, split="train", revision=revision)
        rows = [
            {
                "problem_id": f"{name}-{row[id_field]}",
                "question": row[q_field],
                "verified_answer": str(row[a_field]),
            }
            for row in ds
        ]
        path = os.path.join(out_dir, BENCHMARK_FILES[name])
        with open(path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"{name}: {len(rows)} rows -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize", action="store_true",
                        help="download the three benchmarks at the pinned revisions and write the JSON files")
    parser.add_argument("--out-dir", default=BENCHMARKS_DIR)
    args = parser.parse_args()
    if args.materialize:
        materialize(args.out_dir)
    else:
        parser.print_help()
