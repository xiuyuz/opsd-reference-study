"""Vendored forward-KL loss with per-token clipping from the official OPSD repo.

Source: https://github.com/siyan-zhao/OPSD, commit 7448751f307a9cdbcc1246dd1565a1a605b443df,
`opsd_trainer.py::OPSDTrainer.generalized_jsd_loss` (lines ~382-479).

Only the beta=0 branch (forward KL, teacher -> student) plus the exact per-token clipping is
vendored. The official code computes, for beta=0:

    jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)

which (since F.kl_div(input, target, log_target=True) = exp(target) * (target - input))
is the *elementwise* (position, vocab) quantity teacher_prob * (log_teacher - log_student),
i.e. the un-summed integrand of KL(teacher || student) per vocab entry. Their `token_clip`
clamps this elementwise tensor *before* it gets summed over the vocab dimension by the final
`.sum()` reduction -- it is a per-(token, vocab-class) clip, not a clip of the already-summed
per-token KL value. The official beta=0 path (opsd_trainer.py lines 441-473 at that commit):
    jsd  = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    jsd  = jsd.clamp(max=token_clip)                  # elementwise, before any sum
    jsd  = jsd[labels != -100]                        # 2D mask over a 3D tensor -> [n_tok, V]
    loss = jsd.sum() / mask.sum()                     # reduction="batchmean" (their default)
i.e. sum over the kept (token, vocab) entries divided by the number of kept *tokens*, which
is exactly `clipped_token_kl[mask].sum() / n_tokens` below. Their `top_k` (=None, full vocab)
default is a no-op and is not vendored. Alignment is also theirs: compute_loss slices
`logits[:, prompt_len - 1 : -1, :]` separately for the student and the teacher, i.e. by
each prompt's own completion start.

EXPECTED: the clipped loss can be NEGATIVE. Clipping caps the positive per-(token, vocab)
summands but leaves the negative ones untouched, so `mean_clipped_token_kl` (the optimized
objective) can sit below zero while the unclipped `mean_token_kl` stays positive as a true KL
must. This is the official implementation's own documented behavior -- their README, under
`--jsd_token_clip`: "Note when clipping is applied, the loss can be negative due to positive
KL summand being capped." Do not "fix" the sign; report mean_token_kl as the divergence and
the clipped value as the objective.
"""

import torch
import torch.nn.functional as F


def opsd_forward_kl(teacher_logits, student_logits, completion_mask, token_clip=0.05,
                     temperature=1.0):
    """Forward KL(teacher || student) on completion tokens, official OPSD clipping.

    Args:
        teacher_logits: (batch, seq, vocab), detached PI-teacher logits (adapter disabled).
        student_logits: (batch, seq, vocab), trainable no-PI student logits (adapter enabled).
        completion_mask: (batch, seq) bool/0-1 mask, True on completion tokens to include
            (prompt tokens and padding masked out; include the generated EOS position).
        token_clip: elementwise clip applied to the (position, vocab) KL integrand, matching
            the official `token_clip` argument (default 0.05). Applied to the
            temperature-scaled logits, i.e. the official op order (logits / T, then
            log-softmax, then clip) -- the clip constant itself is not rescaled.
        temperature: divides both logit tensors before log-softmax, matching the official
            loss temperature (default 1.0 = no-op).

    Returns:
        (loss, stats) where loss is a scalar tensor (mean clipped per-token KL over completion
        tokens, differentiable through student_logits) and stats is a dict of Python floats
        with mean_token_kl (unclipped) and mean_clipped_token_kl (== loss.item()), for the
        trajectory log.
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

    # Elementwise (position, vocab) integrand of KL(teacher || student); official beta=0 branch.
    kl_elem = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)

    mask = completion_mask.bool()
    n_tokens = mask.sum().clamp_min(1)

    token_kl = kl_elem.sum(-1)  # per-token KL, summed over vocab, unclipped
    clipped_token_kl = kl_elem.clamp(max=token_clip).sum(-1)  # official clip, then sum

    loss = clipped_token_kl[mask].sum() / n_tokens

    stats = {
        "mean_token_kl": (token_kl[mask].sum() / n_tokens).detach().item(),
        "mean_clipped_token_kl": loss.detach().item(),
    }
    return loss, stats
