"""Search memory for the evoforge evolutionary engine.

Tracks successful patterns, failure modes, credit aggregation, stagnation
detection, and dead-end identification across generations.
"""

from __future__ import annotations

from evoforge.core.types import FailureMode, Individual, Pattern

# Thresholds for classifying individuals
_SUCCESS_THRESHOLD = 0.5
_FAILURE_THRESHOLD = 0.1
_DEAD_END_COUNT = 3


class SearchMemory:
    """Accumulated memory of the evolutionary search process.

    Tracks recurring patterns (high-fitness genomes), failure modes
    (low-fitness genomes), credit aggregates by tactic name, stagnation
    detection, and dead-end tactic combinations.
    """

    def __init__(
        self,
        max_patterns: int = 50,
        max_failures: int = 50,
        stagnation_window: int = 10,
        max_dead_ends: int = 50,
    ) -> None:
        self.max_patterns = max_patterns
        self.max_failures = max_failures
        self.stagnation_window = stagnation_window
        self.max_dead_ends = max_dead_ends

        self.patterns: list[Pattern] = []
        self.failures: list[FailureMode] = []
        self.credit_aggregates: dict[str, float] = {}
        self.best_fitness_history: list[float] = []
        self.dead_ends: set[str] = set()

        # Internal mutable tracking (Pattern/FailureMode are frozen, so we
        # keep mutable dicts keyed by description and rebuild frozen objects.)
        self._pattern_data: dict[str, _PatternAccum] = {}
        self._failure_data: dict[str, _FailureAccum] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, individuals: list[Individual], generation: int) -> None:
        """Ingest a batch of individuals and update all memory state."""
        best_fitness: float | None = None

        for ind in individuals:
            if ind.fitness is None:
                continue

            primary = ind.fitness.primary

            # Track best fitness this generation
            if best_fitness is None or primary > best_fitness:
                best_fitness = primary

            # Extract patterns from high-fitness individuals
            if primary > _SUCCESS_THRESHOLD:
                self._record_pattern(ind.genome, primary)

            # Record failure modes from low-fitness individuals
            if primary < _FAILURE_THRESHOLD:
                self._record_failure(ind.genome, generation)

            # Aggregate credits by signal name
            for credit in ind.credits:
                self.credit_aggregates[credit.signal] = (
                    self.credit_aggregates.get(credit.signal, 0.0) + credit.score
                )

        # Track best fitness for stagnation detection
        if best_fitness is not None:
            self.best_fitness_history.append(best_fitness)

        # Rebuild frozen dataclass lists from mutable accumulators
        self._rebuild_patterns()
        self._rebuild_failures()

        # Detect dead ends
        self._detect_dead_ends()

    def prompt_section(self, max_tokens: int = 500) -> str:
        """Render memory state for inclusion in an LLM prompt.

        Approximate token budget by character count / 4.
        """
        max_chars = max_tokens * 4
        parts: list[str] = []

        has_content = self.patterns or self.failures or self.credit_aggregates or self.dead_ends
        if not has_content:
            return ""

        # Top patterns
        if self.patterns:
            lines = ["Successful patterns:"]
            for p in self.patterns[:10]:
                line = f"  - {p.description} (freq={p.frequency}, avg_fit={p.avg_fitness:.2f})"
                lines.append(line)
            parts.append("\n".join(lines))

        # Dead ends
        if self.dead_ends:
            dead_list = sorted(self.dead_ends)[:10]
            lines = ["Dead ends (avoid these):"]
            for d in dead_list:
                lines.append(f"  - {d}")
            parts.append("\n".join(lines))

        # Credit aggregates
        if self.credit_aggregates:
            sorted_credits = sorted(
                self.credit_aggregates.items(), key=lambda x: x[1], reverse=True
            )[:10]
            lines = ["Credit summary:"]
            for name, score in sorted_credits:
                lines.append(f"  - {name}: {score:.1f}")
            parts.append("\n".join(lines))

        # Recent failures
        if self.failures:
            lines = ["Recent failures:"]
            for f in self.failures[:5]:
                lines.append(f"  - {f.description} (freq={f.frequency}, last_gen={f.last_seen})")
            parts.append("\n".join(lines))

        result = "\n".join(parts)

        # Truncate to fit within token budget
        if len(result) > max_chars:
            result = result[:max_chars]

        return result

    def is_stagnant(self) -> bool:
        """Return True if best fitness hasn't improved for stagnation_window generations."""
        if len(self.best_fitness_history) < self.stagnation_window:
            return False
        window = self.best_fitness_history[-self.stagnation_window :]
        return all(v == window[0] for v in window)

    def get_credit_summary(self) -> dict[str, float]:
        """Return current credit aggregates."""
        return dict(self.credit_aggregates)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_pattern(self, genome: str, fitness: float) -> None:
        """Record or update a pattern from a successful genome."""
        desc = genome.strip()
        if desc in self._pattern_data:
            accum = self._pattern_data[desc]
            accum.total_fitness += fitness
            accum.count += 1
        else:
            self._pattern_data[desc] = _PatternAccum(total_fitness=fitness, count=1)

    def _record_failure(self, genome: str, generation: int) -> None:
        """Record or update a failure mode from a failing genome."""
        desc = genome.strip()
        if desc in self._failure_data:
            accum = self._failure_data[desc]
            accum.count += 1
            accum.last_seen = generation
        else:
            self._failure_data[desc] = _FailureAccum(count=1, last_seen=generation)

    def _rebuild_patterns(self) -> None:
        """Rebuild the frozen Pattern list from mutable accumulators."""
        all_patterns = [
            Pattern(
                description=desc,
                frequency=accum.count,
                avg_fitness=accum.total_fitness / accum.count,
            )
            for desc, accum in self._pattern_data.items()
        ]
        # Sort by frequency * avg_fitness descending, keep top max_patterns
        all_patterns.sort(key=lambda p: p.frequency * p.avg_fitness, reverse=True)
        self.patterns = all_patterns[: self.max_patterns]

    def _rebuild_failures(self) -> None:
        """Rebuild the frozen FailureMode list from mutable accumulators."""
        all_failures = [
            FailureMode(
                description=desc,
                frequency=accum.count,
                last_seen=accum.last_seen,
            )
            for desc, accum in self._failure_data.items()
        ]
        # Sort by most recent first, keep top max_failures
        all_failures.sort(key=lambda f: f.last_seen, reverse=True)
        self.failures = all_failures[: self.max_failures]

    def _detect_dead_ends(self) -> None:
        """Mark tactic descriptions that have failed >= _DEAD_END_COUNT times.

        Caps the dead_ends set at max_dead_ends, keeping the most frequent.
        """
        for desc, accum in self._failure_data.items():
            if accum.count >= _DEAD_END_COUNT:
                self.dead_ends.add(desc)

        if len(self.dead_ends) > self.max_dead_ends:
            # Keep the most frequent dead ends
            ranked = sorted(
                self.dead_ends,
                key=lambda d: self._failure_data.get(d, _FailureAccum(0, 0)).count,
                reverse=True,
            )
            self.dead_ends = set(ranked[: self.max_dead_ends])


class _PatternAccum:
    """Mutable accumulator for pattern tracking."""

    __slots__ = ("total_fitness", "count")

    def __init__(self, total_fitness: float, count: int) -> None:
        self.total_fitness = total_fitness
        self.count = count


class _FailureAccum:
    """Mutable accumulator for failure mode tracking."""

    __slots__ = ("count", "last_seen")

    def __init__(self, count: int, last_seen: int) -> None:
        self.count = count
        self.last_seen = last_seen
