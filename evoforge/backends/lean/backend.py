"""Lean 4 backend facade for evoforge.

Implements :class:`LeanBackend`, the concrete :class:`Backend` subclass for
Lean 4 theorem proving.  Delegates to the existing lean sub-modules for
parsing, credit assignment, validation, mutation operators, and evaluation.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import re
import subprocess
import tempfile
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
    # --- Seeds for theorems where variables need introducing ---
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
    # --- Seeds for theorems where variables are already in context ---
    "simp",
    "norm_num",
    "linarith",
    "ring",
    "omega",
    "positivity",
    "norm_num\nsimp",
    "push_neg\nsimp",
    "gcongr",
    "trivial",
    # --- Multi-step structured proof seeds ---
    "by_cases h : _ = 0\n· simp [h]\n· simp",
    "have h : _ := by simp\nexact h",
    "calc _ ≤ _ := by simp\n  _ = _ := by ring",
    "refine le_of_eq ?_\nsimp",
    "rw [norm_le_iff]\nconstructor\n· linarith\n· linarith",
    "apply norm_le_of_sq_le_sq'\n· positivity\n· nlinarith",
    "suffices h : _ by exact h\nsimp",
]

# Regex for extracting Lean code blocks from LLM output
_LEAN_CODE_BLOCK_RE = re.compile(r"```lean\s*\n(.*?)```", re.DOTALL)

# Path to Jinja2 templates
_TEMPLATES_DIR = Path(__file__).parent / "templates"


logger = logging.getLogger(__name__)

# Fitness for false-positive proofs: REPL step-by-step says complete but cmd
# verification rejects. Still feasible (preserves gradient signal) but below
# any genuinely verified proof.
_FALSE_POSITIVE_FITNESS = 0.9


def _reindent_tactics(genome: str) -> str:
    """Rebase tactic lines to 2-space indent, preserving relative indentation."""
    tactic_lines = [ln for ln in genome.split("\n") if ln.strip()]
    if not tactic_lines:
        return ""
    min_indent = min(len(ln) - len(ln.lstrip()) for ln in tactic_lines)
    return "\n".join("  " + ln[min_indent:] for ln in tactic_lines)


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
        imports: str = "",
        seeds: list[str] | None = None,
        extra_api_namespaces: list[str] | None = None,
    ) -> None:
        self.theorem_statement = theorem_statement
        self.project_dir = project_dir
        self._evaluator: LeanStepwiseEvaluator | None = None
        self._repl: LeanREPLProcess | None = None
        self.repl_path = repl_path
        self._imports = imports
        self._repl_lock = asyncio.Lock()
        self._prefix_cache: dict[str, int] = {}
        self._config_seeds: list[str] = seeds or []
        self._import_env: int | None = None
        self._api_context: list[Any] = []  # populated during startup()
        self._extra_api_namespaces = extra_api_namespaces or []

        self._jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
            keep_trailing_newline=True,
        )

    # -- Backend abstract method implementations ----------------------------

    def parse(self, genome: str) -> IRProtocol | None:
        """Parse a raw genome string into a :class:`TacticSequence`."""
        return parse_tactic_sequence(genome)

    async def startup(self) -> None:
        """Start the Lean REPL and initialize the stepwise evaluator."""
        self._repl = LeanREPLProcess(self.project_dir, self.repl_path)
        await self._repl.start()

        theorem_cmd = f"{self.theorem_statement} := by\n sorry"
        init_cmd: dict[str, object] = {"cmd": theorem_cmd}
        if self._imports:
            resp = await self._repl.send_command({"cmd": self._imports})
            self._import_env = int(resp.get("env", 0))  # type: ignore[call-overload]
            init_cmd["env"] = self._import_env
        else:
            self._import_env = None

        resp = await self._repl.send_command(init_cmd)
        initial_proof_state = 0
        sorries = resp.get("sorries", [])
        if isinstance(sorries, list) and sorries and "proofState" in sorries[0]:
            initial_proof_state = int(sorries[0]["proofState"])
        else:
            logger.warning("REPL init response missing 'sorries': %s", resp)

        self._evaluator = LeanStepwiseEvaluator(
            self._repl,
            initial_proof_state=initial_proof_state,
            prefix_cache=self._prefix_cache,
        )
        logger.info("Lean REPL started and evaluator initialized")

        # Extract available API from Lean source files
        from evoforge.backends.lean.api_extractor import extract_api_for_theorem

        try:
            self._api_context = extract_api_for_theorem(
                project_dir=Path(self.project_dir),
                theorem_statement=self.theorem_statement,
                extra_namespaces=self._extra_api_namespaces,
            )
            if self._api_context:
                logger.info(
                    "Extracted %d API entries from Lean source files",
                    len(self._api_context),
                )
                self.system_prompt.cache_clear()
        except Exception:
            logger.warning("Failed to extract API from Lean sources", exc_info=True)

    async def shutdown(self) -> None:
        """Close the Lean REPL process."""
        if self._repl is not None:
            await self._repl.close()
            self._repl = None
        self._evaluator = None
        logger.info("Lean REPL shut down")

    async def evaluate(self, ir: Any, seed: int | None = None) -> tuple[Fitness, Any, Any]:
        """Evaluate a tactic sequence via the stepwise evaluator.

        If step-by-step evaluation reports proof complete (fitness=1.0),
        performs inline REPL cmd verification. Verified proofs keep 1.0;
        false positives are downgraded to 0.9 (still feasible, preserving
        gradient signal for the evolutionary search).
        """
        if self._evaluator is None:
            raise RuntimeError("LeanBackend.startup() must be called before evaluate()")
        async with self._repl_lock:
            fitness, diagnostics, trace = await self._evaluator.evaluate(ir)

            # Inline verification for proof-complete results
            proof_complete = float(fitness.auxiliary.get("proof_complete", 0.0))
            if fitness.primary >= 1.0 and proof_complete >= 1.0:
                genome = ir.serialize()
                cmd_verified, cmd_error = await self._verify_via_repl_cmd(genome)
                diagnostics.cmd_verification_attempted = True
                if not cmd_verified:
                    diagnostics.cmd_error_message = cmd_error
                    error_pattern = self._classify_cmd_error(cmd_error or "unknown")
                    logger.info(
                        "REPL step-by-step said complete but cmd verification failed — "
                        "downgrading to %.1f: %s",
                        _FALSE_POSITIVE_FITNESS,
                        genome[:80],
                    )
                    fitness = Fitness(
                        primary=_FALSE_POSITIVE_FITNESS,
                        auxiliary={
                            **fitness.auxiliary,
                            "proof_complete": 0.0,
                            "cmd_verified": 0.0,
                            "cmd_error_pattern": error_pattern,
                        },
                        constraints=fitness.constraints,
                        feasible=True,
                    )
                else:
                    fitness = Fitness(
                        primary=fitness.primary,
                        auxiliary={**fitness.auxiliary, "cmd_verified": 1.0},
                        constraints=fitness.constraints,
                        feasible=fitness.feasible,
                    )

        return fitness, diagnostics, trace

    async def evaluate_stepwise(
        self, ir: Any, seed: int | None = None
    ) -> tuple[Fitness, Any, Any]:
        """Alias for evaluate() — all Lean evaluation is stepwise."""
        return await self.evaluate(ir, seed=seed)

    def assign_credit(
        self, ir: Any, fitness: Fitness, diagnostics: Any, trace: Any
    ) -> list[Credit]:
        """Assign per-tactic credit based on evaluation trace results."""
        return assign_credit_lean(ir, diagnostics, trace)

    def validate_structure(self, ir: Any) -> list[str]:
        """Return a list of structural validation errors (empty = valid)."""
        return validate_structure_lean(ir)

    def seed_population(self, n: int) -> list[str]:
        """Generate *n* seed genomes by cycling through the seed bank.

        If config-provided seeds are available, they are prepended to the
        default bank so theorem-specific seeds appear first.
        """
        bank = self._config_seeds + _SEED_BANK if self._config_seeds else _SEED_BANK
        return [bank[i % len(bank)] for i in range(n)]

    def mutation_operators(self) -> list[MutationOperator]:
        """Return the four cheap structural mutation operators."""
        return [
            PrefixTruncation(),
            TacticSwap(),
            TacticReorder(),
            SplicePrefixes(),
        ]

    @functools.lru_cache(maxsize=1)
    def system_prompt(self) -> str:
        """Render the system prompt template with the theorem statement."""
        template = self._jinja_env.get_template("system_prompt.j2")
        # Derive math context from the imports / theorem for richer prompts
        math_context = self._derive_math_context()
        return template.render(
            theorem_statement=self.theorem_statement,
            math_context=math_context,
            api_context=self._api_context,
        )

    def _derive_math_context(self) -> str:
        """Build mathematical context hints from imports and theorem statement."""
        parts: list[str] = []
        stmt = self.theorem_statement.lower()
        if "positivedefinite" in stmt or "positive_definite" in stmt:
            parts.append(
                "The theorem involves positive definite functions. "
                "A function φ : ℝ → ℂ is positive definite if for all "
                "finite sequences x₁...xₙ and c₁...cₙ, the sum "
                "∑ᵢⱼ cᵢ* cⱼ φ(xᵢ - xⱼ) ≥ 0. Key properties: "
                "φ(0) is real and non-negative, |φ(x)| ≤ φ(0), "
                "φ(-x) = conj(φ(x)) (Hermitian symmetry)."
            )
        if "norm" in stmt or "‖" in stmt:
            parts.append(
                "This involves complex norms. Useful lemmas: "
                "norm_nonneg, norm_le_iff, sq_le_sq', norm_sq_eq_abs."
            )
        if "leanlevy" in self._imports.lower():
            parts.append(
                "Available from the LeanLevy library: "
                "IsPositiveDefinite.conj_neg (Hermitian symmetry), "
                "IsPositiveDefinite.re_nonneg (PD form has nonneg real part), "
                "IsPositiveDefinite.apply_zero_nonneg (φ(0).re ≥ 0), "
                "IsPositiveDefinite.apply_zero_im (φ(0).im = 0)."
            )
        return "\n".join(parts)

    @staticmethod
    def _extract_diagnostics(individual: Individual) -> tuple[str, str]:
        """Extract diagnostics summary and credit summary from an individual."""
        diagnostics_text = ""
        if individual.diagnostics is not None and hasattr(individual.diagnostics, "summary"):
            diagnostics_text = individual.diagnostics.summary(max_tokens=500)

        credit_text = ""
        if individual.diagnostics is not None and hasattr(
            individual.diagnostics, "credit_summary"
        ):
            credit_text = individual.diagnostics.credit_summary(individual.credits, max_tokens=300)

        return diagnostics_text, credit_text

    def format_mutation_prompt(self, parent: Individual, context: Any) -> str:
        """Render the mutation prompt template for the LLM."""
        diagnostics_text, credit_text = self._extract_diagnostics(parent)

        memory_section = ""
        if context is not None and hasattr(context, "memory_section"):
            memory_section = context.memory_section

        # Extract goal state from diagnostics for richer LLM context
        goal_state = self._extract_goal_state(parent)

        template = self._jinja_env.get_template("mutation_prompt.j2")
        return template.render(
            genome=parent.genome,
            diagnostics=diagnostics_text,
            credit_summary=credit_text,
            memory_section=memory_section,
            goal_state=goal_state,
        )

    @staticmethod
    def _extract_goal_state(individual: Individual) -> str:
        """Extract the remaining goal state from diagnostics for the mutation prompt."""
        diag = individual.diagnostics
        if diag is None:
            return ""
        goal_types = getattr(diag, "goal_types", [])
        goal_contexts = getattr(diag, "goal_contexts", [])
        if not goal_types:
            return ""
        parts: list[str] = []
        for i, (gtype, gctx) in enumerate(zip(goal_types, goal_contexts)):
            if gctx:
                parts.append(f"Goal {i + 1}:\n  {gctx}\n  ⊢ {gtype}")
            else:
                parts.append(f"Goal {i + 1}: ⊢ {gtype}")
        return "\n".join(parts)

    @staticmethod
    def _classify_cmd_error(error_msg: str) -> str:
        """Normalize a REPL error message into a short category string.

        Used for SearchMemory deduplication so that semantically-identical
        errors map to the same pattern key.
        """
        msg = error_msg.strip()
        if msg.startswith("unknown identifier"):
            match = re.search(r"'([^']+)'", msg)
            name = match.group(1) if match else "?"
            return f"unknown_identifier:{name}"
        if msg.startswith("type mismatch"):
            return "type_mismatch"
        if msg.startswith("unsolved goals"):
            return "unsolved_goals"
        if re.search(r"sorry", msg, re.IGNORECASE):
            return "sorry"
        first_line = msg.split("\n", 1)[0]
        truncated = first_line[:60]
        return f"other:{truncated}"

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

    @functools.lru_cache(maxsize=1)
    def eval_config_hash(self) -> str:
        """Return a SHA-256 hash of the evaluation config, truncated to 16 chars."""
        content = f"{self.theorem_statement}:{self.project_dir}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def format_reflection_prompt(
        self, population: list[Individual], memory: Any, generation: int
    ) -> str:
        """Render the reflection prompt template with population statistics."""
        # Single pass: collect evaluated individuals and unique genomes
        evaluated: list[Individual] = []
        unique_genomes: set[str] = set()
        for ind in population:
            unique_genomes.add(ind.genome)
            if ind.fitness is not None:
                evaluated.append(ind)

        # Derive stats from the filtered list
        fitnesses = [ind.fitness.primary for ind in evaluated]  # type: ignore[union-attr]
        best_fitness = max(fitnesses) if fitnesses else 0.0
        avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
        diversity = len(unique_genomes) / len(population) if population else 0.0

        top_individuals = sorted(
            evaluated,
            key=lambda ind: ind.fitness.primary,  # type: ignore[union-attr]
            reverse=True,
        )[:5]

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

    def format_proof(self, genome: str) -> str:
        """Wrap tactic genome into a complete, standalone Lean 4 proof."""
        lines: list[str] = []
        if self._imports:
            lines.append(self._imports)
            lines.append("")
        lines.append(f"{self.theorem_statement} := by")
        body = _reindent_tactics(genome)
        if body:
            lines.append(body)
        return "\n".join(lines) + "\n"

    @functools.lru_cache(maxsize=1)
    def _example_statement(self) -> str:
        """Convert 'theorem <name> ...' to 'example ...' for REPL cmd verification."""
        return re.sub(r"^theorem\s+\S+", "example", self.theorem_statement)

    async def _verify_via_repl_cmd(self, genome: str) -> tuple[bool, str | None]:
        """Verify a proof by sending it as a complete cmd to the REPL.

        Uses ``example`` instead of ``theorem`` to avoid naming conflicts.
        Returns ``(True, None)`` on success, or ``(False, error_message)`` on
        failure.
        """
        if self._repl is None:
            return False, "REPL not started"

        # Build the example command preserving indentation
        stmt = self._example_statement()
        body = _reindent_tactics(genome)
        if not body:
            return False, "empty proof body"
        cmd_text = f"{stmt} := by\n{body}"

        cmd: dict[str, object] = {"cmd": cmd_text}
        if self._import_env is not None:
            cmd["env"] = self._import_env

        try:
            resp = await self._repl.send_command(cmd)
        except Exception:
            logger.debug("REPL cmd verification raised an exception", exc_info=True)
            return False, "REPL exception"

        # Check for errors
        if "severity" in resp and resp["severity"] == "error":
            return False, str(resp.get("message", "unknown error"))
        if "message" in resp and "env" not in resp:
            return False, str(resp.get("message", "unknown error"))

        # Check for sorries
        sorries = resp.get("sorries", [])
        if sorries:
            return False, "proof contains sorry"

        return True, None

    async def verify_proof(self, genome: str) -> bool:
        """Verify a proof by compiling it with ``lake env lean``.

        Writes a standalone proof file into the Lean project directory,
        invokes the Lean compiler, and returns ``True`` only if compilation
        succeeds (exit code 0).
        """
        # Reject proofs containing sorry — they compile but aren't real proofs
        if "sorry" in genome:
            logger.warning("Proof contains sorry — rejecting without compilation")
            return False

        proof_text = self.format_proof(genome)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".lean",
                dir=self.project_dir,
                prefix="_evoforge_verify_",
                delete=False,
            ) as f:
                f.write(proof_text)
                temp_path = Path(f.name)

            loop = asyncio.get_running_loop()
            ret = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["lake", "env", "lean", str(temp_path)],
                    cwd=self.project_dir,
                    capture_output=True,
                    timeout=120,
                ),
            )
            if ret.returncode == 0:
                stderr_text = ret.stderr.decode(errors="replace")
                if "sorry" in stderr_text.lower():
                    logger.warning(
                        "Proof uses sorry transitively — rejecting: %s",
                        genome[:200],
                    )
                    return False
                logger.info("Proof verified by lake env lean")
                return True
            else:
                logger.warning(
                    "Proof failed lake verification (exit %d): %s",
                    ret.returncode,
                    ret.stderr.decode(errors="replace")[:500],
                )
                return False
        except Exception:
            logger.warning("verify_proof() raised an exception", exc_info=True)
            return False
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    # -- Extra helpers ------------------------------------------------------

    def format_crossover_prompt(
        self, parent_a: Individual, parent_b: Individual, context: Any
    ) -> str:
        """Render the crossover prompt template for the LLM."""
        diagnostics_a, credit_a = self._extract_diagnostics(parent_a)

        template = self._jinja_env.get_template("crossover_prompt.j2")
        return template.render(
            genome_a=parent_a.genome,
            genome_b=parent_b.genome,
            diagnostics_a=diagnostics_a,
            credit_a=credit_a,
        )
