"""Lean 4 backend facade for evoforge.

Implements :class:`LeanBackend`, the concrete :class:`Backend` subclass for
Lean 4 theorem proving.  Delegates to the existing lean sub-modules for
parsing, credit assignment, validation, mutation operators, and evaluation.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import jinja2

from evoforge.backends.base import Backend
from evoforge.backends.lean.credit import assign_credit_lean
from evoforge.backends.lean.evaluator import LeanREPLProcess, LeanStepwiseEvaluator
from evoforge.backends.lean.ir import TacticSequence, parse_tactic_sequence
from evoforge.backends.lean.operators import (
    PrefixTruncation,
    SplicePrefixes,
    TacticReorder,
    TacticSwap,
)
from evoforge.backends.lean.validation import validate_structure_lean
from evoforge.core.ir import BehaviorDimension, BehaviorSpaceConfig, IRProtocol
from evoforge.core.mutation import MutationOperator
from evoforge.core.types import Credit, Fitness, Individual

# ---------------------------------------------------------------------------
# Seed bank — starter tactic proofs that the evolutionary engine begins with
# ---------------------------------------------------------------------------

_SEED_BANK: list[str] = [
    "intro x\nsimp",
    "intro x\nring",
    "intro x\nnorm_num",
    "intro x\nlinarith",
    "intro x\napply le_of_eq\nsimp",
    "intro x\napply norm_nonneg",
    "intro x\nexact le_refl _",
    "intro x\nomega",
    "intro x\npositivity",
    "intro x\nsimp [mul_comm]",
    "intro x\nring_nf\nsimp",
    "intro x\npush_neg\nsimp",
]

# Regex for extracting Lean code blocks from LLM output
_LEAN_CODE_BLOCK_RE = re.compile(r"```lean\s*\n(.*?)```", re.DOTALL)

# Path to Jinja2 templates
_TEMPLATES_DIR = Path(__file__).parent / "templates"


class LeanBackend(Backend):
    """Concrete backend for Lean 4 theorem proving.

    Acts as a facade, delegating to the lean sub-modules for IR parsing,
    credit assignment, structural validation, and mutation operators.
    """

    def __init__(
        self,
        theorem_statement: str,
        project_dir: str,
        repl_path: str | None = None,
    ) -> None:
        self.theorem_statement = theorem_statement
        self.project_dir = project_dir
        self._evaluator: LeanStepwiseEvaluator | None = None
        self._repl: LeanREPLProcess | None = None
        self.repl_path = repl_path

        self._jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
            keep_trailing_newline=True,
        )

    # -- Backend abstract method implementations ----------------------------

    def parse(self, genome: str) -> IRProtocol | None:
        """Parse a raw genome string into a :class:`TacticSequence`."""
        return parse_tactic_sequence(genome)

    def evaluate(self, ir: Any, seed: int | None = None) -> tuple[Fitness, Any, Any]:
        """Synchronous evaluation is not supported for Lean.

        Lean evaluation requires the async REPL. Use
        :class:`LeanStepwiseEvaluator` directly for async stepwise evaluation.
        """
        raise NotImplementedError("Use evaluate_stepwise for Lean backend")

    def evaluate_stepwise(self, ir: Any, seed: int | None = None) -> tuple[Fitness, Any, Any]:
        """Stepwise evaluation requires async.

        Use :class:`LeanStepwiseEvaluator` directly for async stepwise
        evaluation against the Lean REPL.
        """
        raise NotImplementedError(
            "Lean stepwise evaluation requires async — use LeanStepwiseEvaluator directly"
        )

    def assign_credit(
        self, ir: Any, fitness: Fitness, diagnostics: Any, trace: Any
    ) -> list[Credit]:
        """Assign per-tactic credit based on evaluation trace results."""
        return assign_credit_lean(ir, diagnostics, trace)

    def validate_structure(self, ir: Any) -> list[str]:
        """Return a list of structural validation errors (empty = valid)."""
        return validate_structure_lean(ir)

    def seed_population(self, n: int) -> list[str]:
        """Generate *n* seed genomes by cycling through the seed bank."""
        return [_SEED_BANK[i % len(_SEED_BANK)] for i in range(n)]

    def mutation_operators(self) -> list[MutationOperator]:
        """Return the four cheap structural mutation operators."""
        return [
            PrefixTruncation(),
            TacticSwap(),
            TacticReorder(),
            SplicePrefixes(),
        ]

    def system_prompt(self) -> str:
        """Render the system prompt template with the theorem statement."""
        template = self._jinja_env.get_template("system_prompt.j2")
        return template.render(theorem_statement=self.theorem_statement)

    def format_mutation_prompt(self, parent: Individual, context: Any) -> str:
        """Render the mutation prompt template for the LLM."""
        diagnostics_text = ""
        if parent.diagnostics is not None and hasattr(parent.diagnostics, "summary"):
            diagnostics_text = parent.diagnostics.summary(max_tokens=500)

        credit_text = ""
        if parent.diagnostics is not None and hasattr(parent.diagnostics, "credit_summary"):
            credit_text = parent.diagnostics.credit_summary(parent.credits, max_tokens=300)

        memory_section = ""
        if context is not None and hasattr(context, "memory_section"):
            memory_section = context.memory_section

        template = self._jinja_env.get_template("mutation_prompt.j2")
        return template.render(
            genome=parent.genome,
            diagnostics=diagnostics_text,
            credit_summary=credit_text,
            memory_section=memory_section,
        )

    def extract_genome(self, raw_text: str) -> str | None:
        """Extract content between ```lean and ``` markers.

        Returns ``None`` if no Lean code block is found.
        """
        match = _LEAN_CODE_BLOCK_RE.search(raw_text)
        if match is None:
            return None
        return match.group(1).strip()

    def behavior_descriptor(self, ir: Any, diagnostics: Any) -> tuple[Any, ...]:
        """Compute the behavior descriptor for MAP-Elites archiving.

        Returns ``(strategy_class, proof_depth_bucket)`` where:
        - strategy_class: classified by the first tactic name
        - proof_depth_bucket: "short" (1-3), "medium" (4-8), "long" (9+)
        """
        seq: TacticSequence = ir
        num_steps = len(seq.steps)

        # Strategy class from the first tactic
        if seq.steps:
            first_tactic = seq.steps[0].tactic
            if first_tactic in ("intro", "apply", "simp"):
                strategy_class = first_tactic
            else:
                strategy_class = "other"
        else:
            strategy_class = "other"

        # Depth bucket
        if num_steps <= 3:
            depth_bucket = "short"
        elif num_steps <= 8:
            depth_bucket = "medium"
        else:
            depth_bucket = "long"

        return (strategy_class, depth_bucket)

    def behavior_space(self) -> BehaviorSpaceConfig:
        """Return the behavior-space configuration for Lean tactic proofs."""
        return BehaviorSpaceConfig(
            dimensions=(
                BehaviorDimension("strategy", ["intro", "apply", "simp", "other"]),
                BehaviorDimension("depth", ["short", "medium", "long"]),
            )
        )

    def recommended_selection(self) -> str:
        """Return the recommended selection strategy for Lean."""
        return "lexicase"

    def version(self) -> str:
        """Return a version string for cache-keying."""
        return "lean_v1"

    def eval_config_hash(self) -> str:
        """Return a SHA-256 hash of the evaluation config, truncated to 16 chars."""
        content = f"{self.theorem_statement}:{self.project_dir}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def format_reflection_prompt(
        self, population: list[Individual], memory: Any, generation: int
    ) -> str:
        """Render the reflection prompt template with population statistics."""
        # Compute population statistics
        fitnesses = [ind.fitness.primary for ind in population if ind.fitness is not None]
        best_fitness = max(fitnesses) if fitnesses else 0.0
        avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0

        # Simple diversity measure: number of unique genomes / population size
        if population:
            unique_genomes = len({ind.genome for ind in population})
            diversity = unique_genomes / len(population)
        else:
            diversity = 0.0

        # Top individuals sorted by fitness
        evaluated = [ind for ind in population if ind.fitness is not None]
        top_individuals = sorted(
            evaluated,
            key=lambda ind: ind.fitness.primary,  # type: ignore[union-attr]
            reverse=True,
        )[:5]

        # Memory section
        memory_section = ""
        if memory is not None:
            memory_section = str(memory)

        template = self._jinja_env.get_template("reflection_prompt.j2")
        return template.render(
            best_fitness=best_fitness,
            avg_fitness=round(avg_fitness, 4),
            pop_size=len(population),
            generation=generation,
            diversity=round(diversity, 4),
            memory_section=memory_section,
            top_individuals=top_individuals,
        )

    def default_operator_weights(self) -> dict[str, float]:
        """Return default mutation-operator weights for Lean tactic proofs."""
        return {
            "prefix_truncation": 0.25,
            "tactic_swap": 0.25,
            "tactic_reorder": 0.25,
            "splice_prefixes": 0.25,
        }

    # -- Extra helpers ------------------------------------------------------

    def format_crossover_prompt(
        self, parent_a: Individual, parent_b: Individual, context: Any
    ) -> str:
        """Render the crossover prompt template for the LLM."""
        diagnostics_a = ""
        if parent_a.diagnostics is not None and hasattr(parent_a.diagnostics, "summary"):
            diagnostics_a = parent_a.diagnostics.summary(max_tokens=500)

        credit_a = ""
        if parent_a.diagnostics is not None and hasattr(parent_a.diagnostics, "credit_summary"):
            credit_a = parent_a.diagnostics.credit_summary(parent_a.credits, max_tokens=300)

        template = self._jinja_env.get_template("crossover_prompt.j2")
        return template.render(
            genome_a=parent_a.genome,
            genome_b=parent_b.genome,
            diagnostics_a=diagnostics_a,
            credit_a=credit_a,
        )
