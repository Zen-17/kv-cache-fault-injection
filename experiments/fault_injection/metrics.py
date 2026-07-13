# SPDX-License-Identifier: Apache-2.0
"""Comparison metrics between a clean run and a fault run (exp_design sec. 6).

All metrics are dependency-free (pure Python) so the experiment can run in the
minimal vllm0.8.5 environment.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence


def token_change_rate(clean: Sequence[int], fault: Sequence[int]) -> bool:
    """TCR: did the token sequence change at all?"""
    return list(clean) != list(fault)


def first_divergence_step(clean: Sequence[int],
                          fault: Sequence[int]) -> Optional[int]:
    """Index of the first mismatching token, or None if identical prefix
    up to the shorter length and same length."""
    for i, (a, b) in enumerate(zip(clean, fault)):
        if a != b:
            return i
    if len(clean) != len(fault):
        return min(len(clean), len(fault))
    return None


def token_diff_ratio(clean: Sequence[int],
                     fault: Sequence[int],
                     start: int = 0) -> float:
    """TDR: fraction of positions that differ, compared over the overlap.

    ``start`` lets 1B compare only the suffix after the injection step.
    """
    c = list(clean)[start:]
    f = list(fault)[start:]
    n = min(len(c), len(f))
    if n == 0:
        return 0.0
    diff = sum(1 for i in range(n) if c[i] != f[i])
    # Length mismatch counts as extra differences.
    diff += abs(len(c) - len(f))
    denom = max(len(c), len(f))
    return diff / denom if denom else 0.0


def _lcs_length(a: Sequence[int], b: Sequence[int]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(clean: Sequence[int], fault: Sequence[int]) -> float:
    """Token-level ROUGE-L F1 using longest common subsequence."""
    c, f = list(clean), list(fault)
    if not c or not f:
        return 0.0
    lcs = _lcs_length(c, f)
    if lcs == 0:
        return 0.0
    prec = lcs / len(f)
    rec = lcs / len(c)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def is_collapse(token_ids: Sequence[int],
                text: str,
                min_len: int = 5) -> bool:
    """Heuristic: repetition/degenerate/empty output.

    Since experiments use ignore_eos=True, early stop cannot shorten output,
    so collapse is driven by degeneration (loops, single-token spam, empties).
    """
    ids = list(token_ids)
    if len(ids) < min_len or not text.strip():
        return True

    # Very low token diversity.
    if len(set(ids)) / len(ids) < 0.15:
        return True

    # Long run of one identical token.
    run = best = 1
    for i in range(1, len(ids)):
        run = run + 1 if ids[i] == ids[i - 1] else 1
        best = max(best, run)
    if best >= 12:
        return True

    # A single 4-gram covering more than half the output.
    if len(ids) >= 8:
        grams: dict[tuple, int] = {}
        for i in range(len(ids) - 3):
            g = tuple(ids[i:i + 4])
            grams[g] = grams.get(g, 0) + 1
        if grams and max(grams.values()) * 4 > len(ids) * 0.5:
            return True

    return False


def has_nan_inf_value(values: Sequence[float]) -> bool:
    return any(math.isnan(v) or math.isinf(v) for v in values)


@dataclass
class TrialMetrics:
    tcr: bool
    tdr: float
    first_divergence_step: Optional[int]
    collapse: bool
    rouge_l: float
    suffix_tdr: Optional[float] = None  # 1B: TDR after injection step
    first_divergence_after_injection: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def compute_trial_metrics(clean_ids: Sequence[int],
                          fault_ids: Sequence[int],
                          fault_text: str,
                          injection_step: Optional[int] = None) -> TrialMetrics:
    tcr = token_change_rate(clean_ids, fault_ids)
    tdr = token_diff_ratio(clean_ids, fault_ids)
    fds = first_divergence_step(clean_ids, fault_ids)
    collapse = is_collapse(fault_ids, fault_text)
    rl = rouge_l(clean_ids, fault_ids)

    suffix_tdr = None
    fda = None
    if injection_step is not None:
        suffix_tdr = token_diff_ratio(clean_ids, fault_ids, start=injection_step)
        fds_full = first_divergence_step(clean_ids, fault_ids)
        if fds_full is not None and fds_full >= injection_step:
            fda = fds_full - injection_step
        elif fds_full is not None:
            fda = 0

    return TrialMetrics(
        tcr=tcr,
        tdr=tdr,
        first_divergence_step=fds,
        collapse=collapse,
        rouge_l=rl,
        suffix_tdr=suffix_tdr,
        first_divergence_after_injection=fda,
    )
