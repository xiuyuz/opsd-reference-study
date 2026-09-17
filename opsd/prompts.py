"""Prompt construction.

The student and teacher wrappers are byte-identical across reference views; only the
{condition_target} block changes. The OFFICIAL_* templates are verbatim ports of the
official OPSD train-time wrapping (siyan-zhao/OPSD, commit
7448751f307a9cdbcc1246dd1565a1a605b443df, data_collator.py, reason_first=False branch)
and render different text from the local templates; every script defaults to the local
templates and switches with --prompt-style official.
"""

import os

from opsd.constants import CONDITIONS

STUDENT_TEMPLATE = """Solve the following math problem. Reason step by step, and put the final answer in \\boxed{{}}.

Problem:
{question}"""

TEACHER_TEMPLATE = """Solve the following math problem. Reason step by step, and put the final answer in \\boxed{{}}.

Problem:
{question}

Privileged reference available only to the teacher:
<reference>
{condition_target}
</reference>

Use the reference to help solve or evaluate the problem, but do not mention that a reference was provided."""


def student_messages(question):
    content = STUDENT_TEMPLATE.format(question=question)
    return [{"role": "user", "content": content}]


def teacher_messages(question, pi_text):
    content = TEACHER_TEMPLATE.format(question=question, condition_target=pi_text)
    return [{"role": "user", "content": content}]


OFFICIAL_STUDENT_TEMPLATE = (
    "Problem: {question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
)

OFFICIAL_TEACHER_TRANSITION = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
)


def official_student_messages(question):
    content = OFFICIAL_STUDENT_TEMPLATE.format(question=question)
    return [{"role": "user", "content": content}]


def official_teacher_messages(question, pi_text):
    content = (
        f"Problem: {question}\n\n"
        f"Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{pi_text}\n=== Reference Solution End ===\n"
        f"{OFFICIAL_TEACHER_TRANSITION}\n"
        f"Please reason step by step, and put your final answer within \\boxed{{}}."
    )
    return [{"role": "user", "content": content}]


def encode(tokenizer, messages, enable_thinking=True):
    # completion start == len(returned ids); add_generation_prompt appends the
    # assistant turn opener. enable_thinking is caller-controlled (student and
    # teacher are set independently).
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        tokenize=True,
    )
    return ids


def save_rendered_examples(tokenizer, problems, ids, out_dir,
                            student_enable_thinking=True, teacher_enable_thinking=True):
    """Write the fully rendered (decoded, untokenized) prompt text for the
    first 3 ids: one student prompt and one teacher prompt per condition."""
    os.makedirs(out_dir, exist_ok=True)
    for i, pid in enumerate(ids[:3]):
        p = problems[pid]

        student_text = tokenizer.apply_chat_template(
            student_messages(p["question"]),
            add_generation_prompt=True,
            enable_thinking=student_enable_thinking,
            tokenize=False,
        )
        with open(os.path.join(out_dir, f"{i:02d}_{pid}_student.txt"), "w") as f:
            f.write(student_text)

        for cond in CONDITIONS:
            teacher_text = tokenizer.apply_chat_template(
                teacher_messages(p["question"], p["pi"][cond]),
                add_generation_prompt=True,
                enable_thinking=teacher_enable_thinking,
                tokenize=False,
            )
            with open(os.path.join(out_dir, f"{i:02d}_{pid}_teacher_{cond}.txt"), "w") as f:
                f.write(teacher_text)
