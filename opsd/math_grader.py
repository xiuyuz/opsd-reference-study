"""Vendored answer extraction + math equivalence from the official OPSD eval code.

Source: https://github.com/siyan-zhao/OPSD, commit 7448751f307a9cdbcc1246dd1565a1a605b443df,
`eval/evaluate_math.py::extract_boxed_answer` and `::grade_answer`.

`extract_boxed` is copied verbatim (brace-aware scan for the *last* \\boxed{...}). `grade_answer`
uses the same `math_verify` package the official code uses for LaTeX-aware math equivalence
(handles fractions, radicals, exponent products, simple algebraic forms), with the same
$-wrapping and string-normalization fallback on parse/verify failure.
"""

from math_verify import parse, verify


def extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...} in text, or None if absent."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    num_left_braces = 0
    right_brace_idx = None
    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    boxed_str = text[idx : right_brace_idx + 1]
    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        return boxed_str[7:-1].strip()
    return None


def grade_answer(pred: str | None, gold: str) -> bool:
    """True if pred (already extract_boxed'd) is mathematically equivalent to gold."""
    if pred is None:
        return False

    pred_wrapped = pred if "$" in pred else f"${pred}$"
    gold_wrapped = gold if "$" in gold else f"${gold}$"

    try:
        pred_parsed = parse(pred_wrapped, fallback_mode="no_fallback")
        gold_parsed = parse(gold_wrapped, fallback_mode="no_fallback")
        return verify(gold_parsed, pred_parsed, timeout_seconds=5)
    except Exception:
        pred_norm = pred_wrapped.replace("$", "").replace(" ", "").lower().strip()
        gold_norm = gold_wrapped.replace("$", "").replace(" ", "").lower().strip()
        return pred_norm == gold_norm
