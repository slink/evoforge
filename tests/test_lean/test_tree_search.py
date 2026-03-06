"""Tests for best-first proof tree search."""

from __future__ import annotations

from typing import Any  # noqa: I001

import pytest

from evoforge.backends.lean.tree_search import ProofTreeSearch

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockREPL:
    """Mock REPL that accepts specific (state, tactic) pairs."""

    def __init__(self, accept_map: dict[tuple[int, str], dict[str, Any]]):
        self._accept_map = accept_map

    async def send_tactic(self, tactic: str, state: int = 0) -> dict[str, object]:
        key = (state, tactic.strip())
        if key in self._accept_map:
            return self._accept_map[key]
        return {"message": f"unknown tactic '{tactic}'"}


class MockTacticGenerator:
    """Returns fixed tactics based on proof depth."""

    def __init__(self, suggestions: dict[int, list[str]]):
        self._suggestions = suggestions

    async def suggest_tactics(self, goal_state: str, proof_so_far: list[str], n: int) -> list[str]:
        depth = len(proof_so_far)
        return self._suggestions.get(depth, [])[:n]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tree_search_finds_two_step_proof() -> None:
    """REPL accepts intro x -> simp to complete the proof."""
    repl = MockREPL(
        {
            (0, "intro x"): {"proofState": 1, "goals": ["x : Nat |- x = x"]},
            (1, "simp"): {"proofState": 2, "goals": []},
        }
    )
    gen = MockTacticGenerator(
        {
            0: ["intro x", "simp"],
            1: ["simp", "ring"],
        }
    )
    searcher = ProofTreeSearch(
        repl=repl,
        tactic_generator=gen,
        initial_state=0,
        initial_goals=["Nat -> Nat = Nat"],
    )
    result = await searcher.search()
    assert result is not None
    assert result.complete is True
    assert result.tactics == ["intro x", "simp"]


@pytest.mark.asyncio
async def test_tree_search_respects_budget() -> None:
    """All tactics fail; search should stop after max_nodes expansions."""
    repl = MockREPL({})  # nothing accepted
    gen = MockTacticGenerator(
        {
            0: ["intro x", "simp", "ring", "omega", "norm_num"],
        }
    )
    searcher = ProofTreeSearch(
        repl=repl,
        tactic_generator=gen,
        initial_state=0,
        initial_goals=["goal"],
        max_nodes=5,
    )
    result = await searcher.search()
    # Either None or not complete
    assert result is None or not result.complete


@pytest.mark.asyncio
async def test_tree_search_returns_best_partial() -> None:
    """Some tactics succeed but none complete; returns deepest path."""
    repl = MockREPL(
        {
            (0, "intro x"): {"proofState": 1, "goals": ["g1", "g2"]},
            (1, "apply h"): {"proofState": 2, "goals": ["g2"]},
            # No further tactic closes the last goal
        }
    )
    gen = MockTacticGenerator(
        {
            0: ["intro x", "bad1"],
            1: ["apply h", "bad2"],
            2: ["bad3"],
        }
    )
    searcher = ProofTreeSearch(
        repl=repl,
        tactic_generator=gen,
        initial_state=0,
        initial_goals=["g1", "g2", "g3"],
        max_nodes=10,
    )
    result = await searcher.search()
    assert result is not None
    assert not result.complete
    # Should have found the two-step partial path
    assert result.tactics == ["intro x", "apply h"]


@pytest.mark.asyncio
async def test_tree_search_with_prefix() -> None:
    """Start with prefix=['intro x'], initial_state after intro."""
    repl = MockREPL(
        {
            (1, "simp"): {"proofState": 2, "goals": []},
        }
    )
    gen = MockTacticGenerator(
        {
            # depth = len(proof_so_far) which includes prefix
            1: ["simp", "ring"],
        }
    )
    searcher = ProofTreeSearch(
        repl=repl,
        tactic_generator=gen,
        initial_state=1,
        initial_goals=["x : Nat |- x = x"],
        prefix=["intro x"],
    )
    result = await searcher.search()
    assert result is not None
    assert result.complete is True
    assert result.tactics == ["intro x", "simp"]
