"""Stepwise Lean tactic evaluator.

Provides :class:`LeanStepwiseEvaluator` which drives a Lean REPL subprocess
tactic-by-tactic, building up prefix caches, fitness scores, and rich
diagnostics as it goes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pty
import shutil
import subprocess
import termios
from dataclasses import dataclass, field
from typing import Any, cast

from evoforge.backends.lean.ir import TacticSequence
from evoforge.core.types import Credit, EvaluationTrace, Fitness

logger = logging.getLogger(__name__)

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

    Communicates via JSON over stdin/stdout.  Uses a pty for the child's
    stdout so the REPL's C runtime stays line-buffered (pipe would cause
    block-buffering, preventing interactive reads).
    """

    def __init__(self, project_dir: str, repl_path: str | None = None) -> None:
        self._project_dir = project_dir
        self._repl_path = repl_path
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: asyncio.StreamReader | None = None
        self._master_fd: int | None = None

    async def start(self) -> None:
        """Start the Lean REPL subprocess."""
        repl_bin = self._repl_path or shutil.which("repl") or "repl"

        # Create a pty so the child's stdout is line-buffered.
        master_fd, slave_fd = pty.openpty()
        # Disable echo on the pty so our input isn't reflected back.
        attrs = termios.tcgetattr(master_fd)
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(master_fd, termios.TCSANOW, attrs)

        self._proc = subprocess.Popen(
            ["lake", "env", repl_bin],
            stdin=subprocess.PIPE,
            stdout=slave_fd,
            stderr=subprocess.PIPE,
            cwd=self._project_dir,
        )
        os.close(slave_fd)
        self._master_fd = master_fd

        # Wrap the master fd in an asyncio StreamReader.
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, os.fdopen(master_fd, "rb", 0))
        self._reader = reader

    async def send_command(self, cmd: dict[str, object]) -> dict[str, object]:
        """Send a JSON command to the REPL and read the JSON response.

        The REPL outputs pretty-printed (multi-line) JSON, so we read lines
        until the accumulated text forms a valid JSON object.  Commands are
        separated by blank lines per the REPL protocol.
        """
        proc = self._proc
        reader = self._reader
        if proc is None or proc.stdin is None or reader is None:
            raise RuntimeError("REPL process not started")

        payload = json.dumps(cmd) + "\n\n"
        proc.stdin.write(payload.encode("utf-8"))
        proc.stdin.flush()

        buf: list[str] = []
        depth = 0
        started = False
        while True:
            line = await reader.readline()
            if not line:
                raise RuntimeError("REPL process closed unexpectedly")
            text = line.decode("utf-8")
            # Skip blank lines between responses.
            if not text.strip() and not started:
                continue
            buf.append(text)
            for ch in text:
                if ch == "{":
                    depth += 1
                    started = True
                elif ch == "}":
                    depth -= 1
            if started and depth == 0:
                break

        return json.loads("".join(buf))  # type: ignore[no-any-return]

    async def send_tactic(self, tactic: str, state: int = 0) -> dict[str, object]:
        """Send a single tactic command to the REPL."""
        return await self.send_command({"tactic": tactic, "proofState": state})

    async def restart(self) -> None:
        """Kill and restart the REPL process."""
        await self.close()
        await self.start()

    async def close(self) -> None:
        """Terminate the REPL subprocess."""
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait()
            except ProcessLookupError:
                pass
            self._proc = None
        self._reader = None
        self._master_fd = None

    def is_healthy(self) -> bool:
        """Check if the REPL process is alive."""
        return self._proc is not None and self._proc.returncode is None


# ---------------------------------------------------------------------------
# Stepwise evaluator
# ---------------------------------------------------------------------------


def _prefix_key(steps: list[Any], k: int) -> str:
    """Build a cache key from the first k tactic raw strings."""
    return "\n".join(s.raw for s in steps[:k])


def _zero_result(
    trace: LeanEvalTrace,
    error_type: str | None = None,
    error_message: str | None = None,
) -> tuple[Fitness, LeanDiagnostics, LeanEvalTrace]:
    """Build a zero-fitness result for early-exit cases."""
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
        error_type=error_type,
        error_message=error_message,
        stuck_tactic_index=None,
        stuck_tactic=None,
        steps_succeeded=0,
        metavar_count=0,
    )
    return fitness, diag, trace


class LeanStepwiseEvaluator:
    """Evaluates tactic sequences step-by-step against a Lean REPL.

    Supports prefix caching: if a prefix of the sequence was already
    evaluated, starts from the cached REPL state instead of the beginning.

    The ``initial_proof_state`` is the REPL proof-state ID obtained by
    sending the theorem-with-sorry during backend startup.  It is reused
    across evaluations without re-sending the theorem command.
    """

    def __init__(
        self,
        repl: LeanREPLProcess,
        initial_proof_state: int = 0,
        prefix_cache: dict[str, int] | None = None,
    ) -> None:
        self._repl = repl
        self._initial_proof_state = initial_proof_state
        self._prefix_cache: dict[str, int] = prefix_cache if prefix_cache is not None else {}

    async def evaluate(
        self, tactic_seq: TacticSequence
    ) -> tuple[Fitness, LeanDiagnostics, LeanEvalTrace]:
        """Evaluate a tactic sequence step-by-step, returning fitness, diagnostics, trace."""
        steps = tactic_seq.steps
        total_steps = len(steps)
        trace = LeanEvalTrace(step_results=[])

        if total_steps == 0:
            return _zero_result(trace)

        # --- Prefix cache lookup ---
        # Find the longest cached prefix
        cached_state = self._initial_proof_state
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
            logger.debug("REPL response for step %d (%s): %s", i, step.raw[:60], resp)

            goals_before = list(last_goals)

            is_error = ("severity" in resp and resp["severity"] == "error") or (
                "message" in resp and "proofState" not in resp
            )
            if is_error:
                # Tactic failed
                error_msg = str(resp.get("message", resp.get("data", "unknown error")))
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
