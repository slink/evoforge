"""Stepwise Lean tactic evaluator.

Provides :class:`LeanStepwiseEvaluator` which drives a Lean REPL subprocess
tactic-by-tactic, building up prefix caches, fitness scores, and rich
diagnostics as it goes.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from typing import Any, cast

from evoforge.backends.lean.ir import TacticSequence
from evoforge.core.types import Credit, EvaluationTrace, Fitness

# ---------------------------------------------------------------------------
# Goal and step result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Goal:
    """A single Lean proof goal."""

    type_str: str
    context: str


@dataclass(frozen=True)
class TacticStepResult:
    """Result of applying a single tactic step in the REPL."""

    succeeded: bool
    goals_before: list[Goal]
    goals_after: list[Goal]
    error_type: str | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Evaluation trace
# ---------------------------------------------------------------------------


@dataclass
class LeanEvalTrace(EvaluationTrace):
    """Trace recording every tactic step result during evaluation."""

    step_results: list[TacticStepResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass
class LeanDiagnostics:
    """Rich diagnostics from a stepwise Lean evaluation.

    Satisfies the :class:`~evoforge.core.types.Diagnostics` protocol.
    """

    success: bool
    goals_remaining: int
    goal_types: list[str]
    goal_contexts: list[str]
    error_type: str | None
    error_message: str | None
    stuck_tactic_index: int | None
    stuck_tactic: str | None
    steps_succeeded: int
    metavar_count: int

    def summary(self, max_tokens: int = 500) -> str:
        """Human-readable summary of the evaluation outcome."""
        parts: list[str] = []

        if self.success:
            parts.append(f"Proof complete. {self.steps_succeeded} steps succeeded.")
        else:
            parts.append(f"Proof incomplete. {self.steps_succeeded} steps succeeded.")
            if self.goals_remaining > 0:
                parts.append(f"{self.goals_remaining} goals remaining.")
            if self.error_type:
                parts.append(f"Error ({self.error_type}): {self.error_message}")
            elif self.error_message:
                parts.append(f"Error: {self.error_message}")
            if self.stuck_tactic is not None:
                parts.append(f"Stuck at step {self.stuck_tactic_index}: '{self.stuck_tactic}'")
            if self.goal_types:
                parts.append(f"Goal types: {', '.join(self.goal_types[:5])}")

        text = " ".join(parts)
        # Rough token budget: ~4 chars per token
        char_limit = max_tokens * 4
        if len(text) > char_limit:
            text = text[: char_limit - 3] + "..."
        return text

    def credit_summary(self, credits: list[Credit], max_tokens: int = 300) -> str:
        """Render credit information as a human-readable string."""
        if not credits:
            return "No credit information available."

        parts: list[str] = []
        for c in credits:
            parts.append(f"Step {c.location}: score={c.score:.2f} ({c.signal})")

        text = "; ".join(parts)
        char_limit = max_tokens * 4
        if len(text) > char_limit:
            text = text[: char_limit - 3] + "..."
        return text


# ---------------------------------------------------------------------------
# REPL subprocess manager
# ---------------------------------------------------------------------------


def _parse_goals(raw_goals: list[str]) -> list[Goal]:
    """Parse raw goal strings from the REPL into Goal objects.

    A raw goal string from the REPL looks like:
        "x : Nat\\n|- Nat"
    We split on "|-" to get context and type.
    """
    goals: list[Goal] = []
    for g in raw_goals:
        if "|-" in g:
            ctx, typ = g.split("|-", 1)
            goals.append(Goal(type_str=typ.strip(), context=ctx.strip()))
        else:
            goals.append(Goal(type_str=g.strip(), context=""))
    return goals


class LeanREPLProcess:
    """Manages a subprocess for the Lean REPL.

    Communicates via JSON over stdin/stdout.
    """

    def __init__(self, project_dir: str, repl_path: str | None = None) -> None:
        self._project_dir = project_dir
        self._repl_path = repl_path
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        """Start the Lean REPL subprocess."""
        repl_bin = self._repl_path or shutil.which("repl") or "repl"
        self._process = await asyncio.create_subprocess_exec(
            "lake",
            "env",
            repl_bin,
            cwd=self._project_dir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def send_command(self, cmd: dict[str, object]) -> dict[str, object]:
        """Send a JSON command to the REPL and read the JSON response."""
        proc = self._process
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError("REPL process not started")

        payload = json.dumps(cmd) + "\n"
        proc.stdin.write(payload.encode("utf-8"))
        await proc.stdin.drain()

        line = await proc.stdout.readline()
        if not line:
            raise RuntimeError("REPL process closed unexpectedly")

        return json.loads(line.decode("utf-8"))  # type: ignore[no-any-return]

    async def send_tactic(self, tactic: str, state: int = 0) -> dict[str, object]:
        """Send a single tactic command to the REPL."""
        return await self.send_command({"tactic": tactic, "proofState": state})

    async def restart(self) -> None:
        """Kill and restart the REPL process."""
        await self.close()
        await self.start()

    async def close(self) -> None:
        """Terminate the REPL subprocess."""
        if self._process is not None:
            try:
                self._process.terminate()
                await self._process.wait()
            except ProcessLookupError:
                pass
            self._process = None

    def is_healthy(self) -> bool:
        """Check if the REPL process is alive."""
        return self._process is not None and self._process.returncode is None


# ---------------------------------------------------------------------------
# Stepwise evaluator
# ---------------------------------------------------------------------------


def _prefix_key(steps: list[Any], k: int) -> str:
    """Build a cache key from the first k tactic raw strings."""
    return "\n".join(s.raw for s in steps[:k])


class LeanStepwiseEvaluator:
    """Evaluates tactic sequences step-by-step against a Lean REPL.

    Supports prefix caching: if a prefix of the sequence was already
    evaluated, starts from the cached REPL state instead of the beginning.
    """

    def __init__(
        self,
        repl: LeanREPLProcess,
        prefix_cache: dict[str, int] | None = None,
    ) -> None:
        self._repl = repl
        self._prefix_cache: dict[str, int] = prefix_cache if prefix_cache is not None else {}

    async def evaluate(
        self, tactic_seq: TacticSequence
    ) -> tuple[Fitness, LeanDiagnostics, LeanEvalTrace]:
        """Evaluate a tactic sequence step-by-step, returning fitness, diagnostics, trace."""
        steps = tactic_seq.steps
        total_steps = len(steps)
        trace = LeanEvalTrace(step_results=[])

        if total_steps == 0:
            fitness = Fitness(
                primary=0.0,
                auxiliary={"steps_succeeded": 0, "goals_remaining": 0, "proof_complete": 0.0},
                constraints={},
                feasible=False,
            )
            diag = LeanDiagnostics(
                success=False,
                goals_remaining=0,
                goal_types=[],
                goal_contexts=[],
                error_type=None,
                error_message=None,
                stuck_tactic_index=None,
                stuck_tactic=None,
                steps_succeeded=0,
                metavar_count=0,
            )
            return fitness, diag, trace

        # --- Prefix cache lookup ---
        # Find the longest cached prefix
        cached_state = 0
        start_index = 0
        for k in range(total_steps, 0, -1):
            key = _prefix_key(steps, k)
            if key in self._prefix_cache:
                cached_state = self._prefix_cache[key]
                start_index = k
                break

        # For cached prefix steps, record them as succeeded (we know they worked)
        for i in range(start_index):
            trace.step_results.append(
                TacticStepResult(
                    succeeded=True,
                    goals_before=[],
                    goals_after=[],
                )
            )

        # --- Step-by-step evaluation ---
        current_state = cached_state
        steps_succeeded = start_index
        last_goals: list[Goal] = []
        error_type: str | None = None
        error_message: str | None = None
        stuck_index: int | None = None
        stuck_tactic: str | None = None

        for i in range(start_index, total_steps):
            step = steps[i]
            resp = await self._repl.send_tactic(step.raw, state=current_state)

            goals_before = list(last_goals)

            if "severity" in resp and resp["severity"] == "error":
                # Tactic failed
                error_msg = str(resp.get("message", "unknown error"))
                error_type = "tactic_failed"
                error_message = error_msg
                stuck_index = i
                stuck_tactic = step.raw

                trace.step_results.append(
                    TacticStepResult(
                        succeeded=False,
                        goals_before=goals_before,
                        goals_after=goals_before,  # goals unchanged on failure
                        error_type=error_type,
                        error_message=error_message,
                    )
                )
                break
            else:
                # Tactic succeeded
                new_state = cast(int, resp.get("proofState", current_state + 1))
                raw_goals = resp.get("goals", [])
                new_goals = _parse_goals(raw_goals)  # type: ignore[arg-type]

                trace.step_results.append(
                    TacticStepResult(
                        succeeded=True,
                        goals_before=goals_before,
                        goals_after=new_goals,
                    )
                )

                current_state = new_state
                last_goals = new_goals
                steps_succeeded += 1

                # Cache this prefix
                prefix_key = _prefix_key(steps, i + 1)
                self._prefix_cache[prefix_key] = current_state

        # --- Compute results ---
        goals_remaining = len(last_goals)
        proof_complete = steps_succeeded == total_steps and goals_remaining == 0

        if proof_complete:
            primary = 1.0
        elif total_steps > 0:
            primary = steps_succeeded / total_steps
        else:
            primary = 0.0

        fitness = Fitness(
            primary=primary,
            auxiliary={
                "steps_succeeded": float(steps_succeeded),
                "goals_remaining": float(goals_remaining),
                "proof_complete": 1.0 if proof_complete else 0.0,
            },
            constraints={},
            feasible=proof_complete,
        )

        diag = LeanDiagnostics(
            success=proof_complete,
            goals_remaining=goals_remaining,
            goal_types=[g.type_str for g in last_goals],
            goal_contexts=[g.context for g in last_goals],
            error_type=error_type,
            error_message=error_message,
            stuck_tactic_index=stuck_index,
            stuck_tactic=stuck_tactic,
            steps_succeeded=steps_succeeded,
            metavar_count=0,
        )

        return fitness, diag, trace
