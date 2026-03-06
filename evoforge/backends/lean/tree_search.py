"""Best-first proof tree search over REPL tactic states.

Explores a tree of tactic applications using a priority queue (max-heap by
score).  A :class:`TacticGenerator` protocol supplies candidate tactics at
each node; the REPL validates them.  Search terminates when a complete proof
is found or the node expansion budget is exhausted.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ProofNode:
    """A node in the proof search tree."""

    state: int  # REPL proof state ID
    tactics: list[str]  # tactics taken to reach this node
    goals: list[str]  # remaining goals at this state
    score: float  # cumulative score (higher = better)
    depth: int = 0
    complete: bool = False
    parent: ProofNode | None = field(default=None, repr=False)

    def __lt__(self, other: ProofNode) -> bool:
        """Higher score = higher priority (max-heap via heapq min-heap)."""
        return self.score > other.score


@dataclass
class SearchResult:
    """Outcome of a tree search run."""

    tactics: list[str]
    complete: bool
    nodes_expanded: int
    score: float


# ---------------------------------------------------------------------------
# Tactic generator protocol
# ---------------------------------------------------------------------------


class REPLLike(Protocol):
    """Minimal protocol for a REPL that can execute tactics."""

    async def send_tactic(self, tactic: str, state: int = 0) -> dict[str, Any]: ...


class TacticGenerator(Protocol):
    """Supplies candidate tactics for a given goal state."""

    async def suggest_tactics(
        self, goal_state: str, proof_so_far: list[str], n: int
    ) -> list[str]: ...


# ---------------------------------------------------------------------------
# Best-first search
# ---------------------------------------------------------------------------


class ProofTreeSearch:
    """Best-first proof tree search driven by a REPL and tactic generator.

    Parameters
    ----------
    repl:
        Object with an ``async send_tactic(tactic, state) -> dict`` method.
    tactic_generator:
        Supplies candidate tactics (see :class:`TacticGenerator`).
    initial_state:
        REPL proof state to start from.
    initial_goals:
        Goals present at *initial_state*.  ``None`` treated as unknown.
    max_nodes:
        Maximum number of node expansions before giving up.
    beam_width:
        How many candidate tactics to request per expansion.
    prefix:
        Tactics already applied before *initial_state*.  Prepended to every
        result's tactic list.
    """

    def __init__(
        self,
        repl: REPLLike,
        tactic_generator: TacticGenerator,
        initial_state: int = 0,
        initial_goals: list[str] | None = None,
        max_nodes: int = 200,
        beam_width: int = 5,
        prefix: list[str] | None = None,
    ) -> None:
        self._repl = repl
        self._gen = tactic_generator
        self._initial_state = initial_state
        self._initial_goals = initial_goals or []
        self._max_nodes = max_nodes
        self._beam_width = beam_width
        self._prefix = list(prefix) if prefix else []

    async def search(self) -> SearchResult | None:
        """Run best-first search and return the result."""
        root = ProofNode(
            state=self._initial_state,
            tactics=list(self._prefix),
            goals=list(self._initial_goals),
            score=0.0,
            depth=0,
        )

        heap: list[ProofNode] = [root]
        visited: set[int] = set()
        best: ProofNode | None = None
        nodes_expanded = 0

        while heap and nodes_expanded < self._max_nodes:
            node = heapq.heappop(heap)

            if node.state in visited:
                continue
            visited.add(node.state)
            nodes_expanded += 1

            # Update best partial result (highest score, deepest on tie)
            if best is None or (node.score, node.depth) > (best.score, best.depth):
                best = node

            # Ask generator for candidate tactics
            goal_text = "\n".join(node.goals) if node.goals else ""
            candidates = await self._gen.suggest_tactics(goal_text, node.tactics, self._beam_width)

            for tactic in candidates:
                resp = await self._repl.send_tactic(tactic, node.state)

                # Skip failed tactics
                if "message" in resp:
                    continue

                child_state: int = int(resp["proofState"])
                child_goals: list[str] = list(resp.get("goals", []))

                goals_closed = max(0, len(node.goals) - len(child_goals))
                child_score = node.score + 1.0 + 0.5 * goals_closed

                child = ProofNode(
                    state=child_state,
                    tactics=[*node.tactics, tactic],
                    goals=child_goals,
                    score=child_score,
                    depth=node.depth + 1,
                    complete=len(child_goals) == 0,
                    parent=node,
                )

                if child.complete:
                    logger.info(
                        "Proof found in %d tactics (%d nodes expanded)",
                        len(child.tactics),
                        nodes_expanded,
                    )
                    return SearchResult(
                        tactics=child.tactics,
                        complete=True,
                        nodes_expanded=nodes_expanded,
                        score=child.score,
                    )

                if child_state not in visited:
                    heapq.heappush(heap, child)

        # Budget exhausted — return best partial or None
        if best is not None and best.depth > 0:
            return SearchResult(
                tactics=best.tactics,
                complete=False,
                nodes_expanded=nodes_expanded,
                score=best.score,
            )

        return None
