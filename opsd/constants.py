"""Shared constants: dataset and artifact locations, model names, context and
generation budgets, the per-request seed schedule, and the correction-marker lexicon.

Environment variables (all optional; see docs/ARTIFACTS.md):
    OPSD_ARTIFACTS          artifacts root every script reads/writes (default ./artifacts)
    AMPLE_MATH_JSONL        local AMPLE-Math jsonl; when unset the Hugging Face dataset
                            AMPLE_MATH_HF_ID is loaded instead
    AMPLE_MATH_HF_ID        that dataset's id (default "xiuyuz/ample-math")
    AMPLE_MATH_HF_SPLIT     split of that dataset to read (default "train")
    AMPLE_MATH_DIFFICULTY   JSON file with per-problem difficulty scores ({"item_difficulties":
                            {problem_id: float}}); when unset the rows' own "difficulty"
                            field is used
    OPSD_BENCHMARKS         directory holding aime24.json / aime25.json / hmmt25.json
                            (default <OPSD_ARTIFACTS>/benchmarks)
    MODEL_MAX_LEN_OVERRIDE  overrides MODEL_MAX_LENGTH (default 32768); the launchers set it
                            to 40960 for thinking-mode runs and the external evaluation
    HF_HOME                 Hugging Face cache; set it yourself before running anything
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.environ.get("OPSD_ARTIFACTS", "artifacts")

AMPLE_MATH_HF_ID = os.environ.get("AMPLE_MATH_HF_ID", "xiuyuz/ample-math")
AMPLE_MATH_HF_SPLIT = os.environ.get("AMPLE_MATH_HF_SPLIT", "train")
AMPLE_MATH_JSONL = os.environ.get("AMPLE_MATH_JSONL") or None
AMPLE_MATH_DIFFICULTY = os.environ.get("AMPLE_MATH_DIFFICULTY") or None

# The three reference views load_problems() attaches to every problem, the three
# additional views, and their union (the six-view sweep).
CONDITIONS = ["answer_only", "key_points", "full_trace"]
EXTRA_CONDITIONS = ["gist", "summary", "clean_solution"]
ALL_CONDITIONS = CONDITIONS + EXTRA_CONDITIONS

PRIMARY_MODEL = "Qwen/Qwen3-1.7B"

# Frozen train/dev/test splits shipped with the release, per backbone.
SPLITS_FILES = {
    "Qwen/Qwen3-1.7B": os.path.join(REPO_ROOT, "data", "splits", "qwen3_1p7b_splits.json"),
    "HuggingFaceTB/SmolLM3-3B": os.path.join(REPO_ROOT, "data", "splits", "smollm3_3b_splits.json"),
}
DEFAULT_SPLITS_FILE = SPLITS_FILES[PRIMARY_MODEL]

# Process-wide on purpose: every per-request completion budget is
# MODEL_MAX_LENGTH - prompt - CONTEXT_SAFETY_MARGIN (opsd.generation.prepare_prompts), so a
# run whose engine is served at a larger context must raise this too, or every request is
# silently clipped to the 32768 default instead of the budget it asked for.
# evaluation/run_external_eval.sh exports it for exactly that reason; training/launch_run.sh
# sets it on the trainer process alone, and the trainer re-reads it into its own module
# global. Leave it unset for the split builders (data/build_splits.py): the released splits
# were filtered for context eligibility at the 32768 default.
MODEL_MAX_LENGTH = int(os.environ.get("MODEL_MAX_LEN_OVERRIDE", 32768))
CONTEXT_SAFETY_MARGIN = 256

MAIN_MAX_NEW_TOKENS = 16384
RESCUE_MAX_NEW_TOKENS_CAP = 32512
PI_GENERATION_MAX_NEW_TOKENS = 8192

GEN_KWARGS = dict(
    do_sample=True,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.0,
    num_beams=1,
)

# Per-backbone seed-schedule index: keeps sample_seed() disjoint across backbones. The
# values are frozen -- every per-request seed is derived from them, so an index is never
# renumbered and the gap at 1 is left as it is.
MODEL_INDEX = {"Qwen/Qwen3-1.7B": 0, "HuggingFaceTB/SmolLM3-3B": 2}

# Unprivileged (no-reference) samples per calibration question, by backbone. Only the
# backbones listed here have a reference profile, and profiling/preflight.py takes its
# --model choices from this mapping.
NO_PI_SAMPLES = {"Qwen/Qwen3-1.7B": 4}


def sample_seed(model_index: int, sample_index: int, item_index: int) -> int:
    return 1_000_000 * model_index + 10_000 * sample_index + item_index


CORRECTION_MARKERS = [
    "wait",
    "but",
    "however",
    "maybe",
    "actually",
    "recheck",
    "check",
    "verify",
    "reconsider",
    "mistake",
    "instead",
]
