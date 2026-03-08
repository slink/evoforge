# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Per-tactic credit assignment for the Lean backend.

Analyses an evaluation trace to assign localized credit to each tactic step,
rewarding goal reduction and penalising failures.
"""

from __future__ import annotations

from typing import Any

from evoforge.backends.lean.ir import TacticSequence
from evoforge.core.types import Credit


def assign_credit_lean(
    ir: TacticSequence,
    diagnostics: Any,  # LeanDiagnostics
    trace: Any,  # LeanEvalTrace
) -> list[Credit]:
    """Assign per-tactic credit based on evaluation trace results.

    For each step result in the trace:
    - Succeeded: score = 0.3 * goal_reduction + 0.1
    - Failed: score = -0.5, and crediting stops after the first failure.

    Returns a list of :class:`Credit` objects with ``location`` set to
    the step index and ``confidence`` fixed at 1.0 (deterministic).
    """
    credits: list[Credit] = []

    for i, step_result in enumerate(trace.step_results):
        if step_result.succeeded:
            goals_before: list[Any] = step_result.goals_before
            goals_after: list[Any] = step_result.goals_after
            reduction = len(goals_before) - len(goals_after)
            score = 0.3 * reduction + 0.1

            if reduction > 0:
                signal = f"closed {reduction} goals"
            else:
                signal = "maintained progress"

            credits.append(Credit(location=i, score=score, signal=signal, confidence=1.0))
        else:
            error_type = step_result.error_type or "unknown"
            error_msg = step_result.error_message or ""
            signal = f"failed: {error_type}: {error_msg[:80]}"
            credits.append(Credit(location=i, score=-0.5, signal=signal, confidence=1.0))
            break

    return credits
