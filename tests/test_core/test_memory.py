"""Tests for evoforge.core.memory – SearchMemory for evolutionary search."""

from __future__ import annotations

from evoforge.core.memory import SearchMemory
from evoforge.core.types import Credit, Fitness, Individual


def _make_individual(
    genome: str,
    primary: float,
    generation: int = 0,
    credits: list[Credit] | None = None,
) -> Individual:
    """Helper to create an Individual with a known fitness and optional credits."""
    return Individual(
        genome=genome,
        ir=None,
        ir_hash=f"hash_{genome}",
        generation=generation,
        fitness=Fitness(
            primary=primary,
            auxiliary={},
            constraints={},
            feasible=True,
        ),
        credits=credits or [],
    )


def _make_unscored_individual(genome: str, generation: int = 0) -> Individual:
    """Helper to create an Individual without fitness."""
    return Individual(
        genome=genome,
        ir=None,
        ir_hash=f"hash_{genome}",
        generation=generation,
    )


# ---------------------------------------------------------------------------
# Pattern extraction (high-fitness individuals)
# ---------------------------------------------------------------------------


class TestPatternExtraction:
    def test_high_fitness_individual_extracts_pattern(self) -> None:
        """Individual with fitness.primary > 0.5 should generate a pattern."""
        mem = SearchMemory()
        ind = _make_individual("tactic_A; tactic_B", primary=0.8, generation=0)
        mem.update([ind], generation=0)
        assert len(mem.patterns) >= 1
        # Pattern description should reference the genome content
        descs = [p.description for p in mem.patterns]
        assert any("tactic_A" in d or "tactic_B" in d for d in descs)

    def test_moderate_fitness_no_pattern(self) -> None:
        """Individual at exactly 0.5 should not generate a pattern (threshold is >0.5)."""
        mem = SearchMemory()
        ind = _make_individual("tactic_X", primary=0.5, generation=0)
        mem.update([ind], generation=0)
        assert len(mem.patterns) == 0

    def test_pattern_frequency_increments(self) -> None:
        """Same genome pattern seen twice should increment frequency."""
        mem = SearchMemory()
        ind1 = _make_individual("tactic_A; tactic_B", primary=0.9, generation=0)
        ind2 = _make_individual("tactic_A; tactic_B", primary=0.7, generation=1)
        mem.update([ind1], generation=0)
        mem.update([ind2], generation=1)
        matching = [p for p in mem.patterns if "tactic_A" in p.description]
        assert len(matching) == 1
        assert matching[0].frequency == 2

    def test_max_patterns_cap(self) -> None:
        """Number of patterns should not exceed max_patterns."""
        mem = SearchMemory(max_patterns=3)
        for i in range(10):
            ind = _make_individual(f"unique_tactic_{i}", primary=0.9, generation=i)
            mem.update([ind], generation=i)
        assert len(mem.patterns) <= 3


# ---------------------------------------------------------------------------
# Failure mode recording (low-fitness individuals)
# ---------------------------------------------------------------------------


class TestFailureRecording:
    def test_low_fitness_individual_records_failure(self) -> None:
        """Individual with fitness.primary < 0.1 should record a failure mode."""
        mem = SearchMemory()
        ind = _make_individual("bad_tactic_X", primary=0.05, generation=0)
        mem.update([ind], generation=0)
        assert len(mem.failures) >= 1
        assert any("bad_tactic_X" in f.description for f in mem.failures)

    def test_moderate_fitness_no_failure(self) -> None:
        """Individual at 0.1 should not be recorded as failure (threshold is <0.1)."""
        mem = SearchMemory()
        ind = _make_individual("ok_tactic", primary=0.1, generation=0)
        mem.update([ind], generation=0)
        assert len(mem.failures) == 0

    def test_failure_frequency_increments(self) -> None:
        """Same failure mode seen multiple times should increment frequency."""
        mem = SearchMemory()
        ind1 = _make_individual("failing_tactic", primary=0.01, generation=0)
        ind2 = _make_individual("failing_tactic", primary=0.02, generation=1)
        mem.update([ind1], generation=0)
        mem.update([ind2], generation=1)
        matching = [f for f in mem.failures if "failing_tactic" in f.description]
        assert len(matching) == 1
        assert matching[0].frequency == 2
        assert matching[0].last_seen == 1

    def test_max_failures_cap(self) -> None:
        """Number of failures should not exceed max_failures."""
        mem = SearchMemory(max_failures=3)
        for i in range(10):
            ind = _make_individual(f"unique_failure_{i}", primary=0.01, generation=i)
            mem.update([ind], generation=i)
        assert len(mem.failures) <= 3


# ---------------------------------------------------------------------------
# Credit aggregation
# ---------------------------------------------------------------------------


class TestCreditAggregation:
    def test_credits_accumulate_by_signal(self) -> None:
        """Credits with the same signal should accumulate their scores."""
        mem = SearchMemory()
        credits_a = [
            Credit(location=0, score=1.5, signal="simp"),
            Credit(location=1, score=0.5, signal="ring_nf"),
        ]
        credits_b = [
            Credit(location=0, score=2.0, signal="simp"),
        ]
        ind1 = _make_individual("g1", primary=0.6, generation=0, credits=credits_a)
        ind2 = _make_individual("g2", primary=0.7, generation=0, credits=credits_b)
        mem.update([ind1, ind2], generation=0)
        summary = mem.get_credit_summary()
        assert summary["simp"] == 3.5
        assert summary["ring_nf"] == 0.5

    def test_credits_from_multiple_generations(self) -> None:
        """Credits should accumulate across generations."""
        mem = SearchMemory()
        c1 = [Credit(location=0, score=1.0, signal="omega")]
        c2 = [Credit(location=0, score=2.0, signal="omega")]
        ind1 = _make_individual("g1", primary=0.6, generation=0, credits=c1)
        ind2 = _make_individual("g2", primary=0.7, generation=1, credits=c2)
        mem.update([ind1], generation=0)
        mem.update([ind2], generation=1)
        assert mem.get_credit_summary()["omega"] == 3.0

    def test_empty_credits(self) -> None:
        """Individuals with no credits should not cause issues."""
        mem = SearchMemory()
        ind = _make_individual("g1", primary=0.6, generation=0)
        mem.update([ind], generation=0)
        assert mem.get_credit_summary() == {}


# ---------------------------------------------------------------------------
# Dead end detection
# ---------------------------------------------------------------------------


class TestDeadEndDetection:
    def test_three_failures_triggers_dead_end(self) -> None:
        """A tactic seen in failures >= 3 times should be marked as a dead end."""
        mem = SearchMemory()
        for i in range(3):
            ind = _make_individual("doomed_tactic", primary=0.01, generation=i)
            mem.update([ind], generation=i)
        assert "doomed_tactic" in mem.dead_ends

    def test_two_failures_no_dead_end(self) -> None:
        """Two failures are not enough to trigger dead end."""
        mem = SearchMemory()
        for i in range(2):
            ind = _make_individual("almost_dead", primary=0.01, generation=i)
            mem.update([ind], generation=i)
        assert "almost_dead" not in mem.dead_ends

    def test_different_tactics_independent_dead_ends(self) -> None:
        """Different tactic descriptions should be tracked independently."""
        mem = SearchMemory()
        for i in range(3):
            ind = _make_individual("dead_A", primary=0.01, generation=i)
            mem.update([ind], generation=i)
        for i in range(2):
            ind = _make_individual("not_dead_B", primary=0.01, generation=i)
            mem.update([ind], generation=i)
        assert "dead_A" in mem.dead_ends
        assert "not_dead_B" not in mem.dead_ends


# ---------------------------------------------------------------------------
# Stagnation detection
# ---------------------------------------------------------------------------


class TestStagnationDetection:
    def test_stagnant_constant_fitness(self) -> None:
        """Constant best fitness for stagnation_window generations -> stagnant."""
        mem = SearchMemory(stagnation_window=5)
        for i in range(5):
            ind = _make_individual("g", primary=0.5, generation=i)
            mem.update([ind], generation=i)
        assert mem.is_stagnant()

    def test_not_stagnant_improving_fitness(self) -> None:
        """Improving fitness means not stagnant."""
        mem = SearchMemory(stagnation_window=5)
        for i in range(5):
            ind = _make_individual("g", primary=0.1 * (i + 1), generation=i)
            mem.update([ind], generation=i)
        assert not mem.is_stagnant()

    def test_not_stagnant_insufficient_history(self) -> None:
        """Need at least stagnation_window entries to detect stagnation."""
        mem = SearchMemory(stagnation_window=10)
        for i in range(5):
            ind = _make_individual("g", primary=0.5, generation=i)
            mem.update([ind], generation=i)
        assert not mem.is_stagnant()

    def test_stagnant_after_initial_improvement(self) -> None:
        """Improvement followed by plateau should detect stagnation."""
        mem = SearchMemory(stagnation_window=4)
        # First 3 generations improve
        for i in range(3):
            ind = _make_individual("g", primary=0.1 * (i + 1), generation=i)
            mem.update([ind], generation=i)
        # Next 4 generations stagnate at 0.3
        for i in range(3, 7):
            ind = _make_individual("g", primary=0.3, generation=i)
            mem.update([ind], generation=i)
        assert mem.is_stagnant()


# ---------------------------------------------------------------------------
# Prompt section rendering
# ---------------------------------------------------------------------------


class TestPromptSection:
    def test_empty_memory_minimal_output(self) -> None:
        """Empty memory should produce minimal/empty prompt section."""
        mem = SearchMemory()
        output = mem.prompt_section()
        # Should be very short or empty
        assert len(output) < 100

    def test_prompt_section_contains_pattern(self) -> None:
        """Prompt section should include pattern info after update."""
        mem = SearchMemory()
        ind = _make_individual("tactic_A; tactic_B", primary=0.9, generation=0)
        mem.update([ind], generation=0)
        output = mem.prompt_section()
        assert "tactic_A" in output or "tactic_B" in output

    def test_prompt_section_contains_dead_end(self) -> None:
        """Prompt section should include dead end info."""
        mem = SearchMemory()
        for i in range(3):
            ind = _make_individual("doomed_tactic", primary=0.01, generation=i)
            mem.update([ind], generation=i)
        output = mem.prompt_section()
        assert "doomed_tactic" in output

    def test_prompt_section_respects_token_limit(self) -> None:
        """Prompt section should stay within approximate token limit."""
        mem = SearchMemory()
        # Fill memory with many patterns
        for i in range(30):
            genome = f"long_tactic_description_{i}_extra_words"
            ind = _make_individual(genome, primary=0.9, generation=i)
            mem.update([ind], generation=i)
        output = mem.prompt_section(max_tokens=100)
        # Approximate: max_tokens * 4 characters
        assert len(output) <= 100 * 4

    def test_prompt_section_contains_credit_info(self) -> None:
        """Prompt section should include credit aggregates."""
        mem = SearchMemory()
        credits = [Credit(location=0, score=5.0, signal="simp")]
        ind = _make_individual("g1", primary=0.6, generation=0, credits=credits)
        mem.update([ind], generation=0)
        output = mem.prompt_section()
        assert "simp" in output


# ---------------------------------------------------------------------------
# Unscored individuals
# ---------------------------------------------------------------------------


class TestUnscoredIndividuals:
    def test_unscored_individual_skipped(self) -> None:
        """Individuals with no fitness should not affect patterns or failures."""
        mem = SearchMemory()
        ind = _make_unscored_individual("g1", generation=0)
        mem.update([ind], generation=0)
        assert len(mem.patterns) == 0
        assert len(mem.failures) == 0
        # But best_fitness_history should still be empty (no fitness to track)
        assert len(mem.best_fitness_history) == 0

    def test_mixed_scored_and_unscored(self) -> None:
        """Only scored individuals should contribute to memory."""
        mem = SearchMemory()
        scored = _make_individual("good_tactic", primary=0.9, generation=0)
        unscored = _make_unscored_individual("unknown", generation=0)
        mem.update([scored, unscored], generation=0)
        assert len(mem.patterns) == 1
        assert len(mem.best_fitness_history) == 1
