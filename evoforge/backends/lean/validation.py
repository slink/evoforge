"""Structural validation for Lean tactic sequences.

Checks a :class:`TacticSequence` against a set of rules and returns a list of
violation strings. An empty list means the sequence is valid.
"""

from __future__ import annotations

from evoforge.backends.lean.ir import TacticSequence

# ---------------------------------------------------------------------------
# Tactic whitelist — commonly accepted Lean 4 / Mathlib tactics
# ---------------------------------------------------------------------------

TACTIC_WHITELIST: frozenset[str] = frozenset(
    {
        "abs",
        "aesop",
        "apply",
        "assumption",
        "bound",
        "by_cases",
        "by_contra",
        "calc",
        "cases",
        "change",
        "clear",
        "congr",
        "constructor",
        "continuity",
        "contrapose",
        "conv",
        "decide",
        "exact",
        "exfalso",
        "exists",
        "ext",
        "field_simp",
        "filter_upwards",
        "funext",
        "gcongr",
        "generalize",
        "have",
        "induction",
        "intro",
        "intros",
        "left",
        "let",
        "linarith",
        "measurability",
        "mono",
        "nlinarith",
        "norm_cast",
        "norm_num",
        "obtain",
        "omega",
        "polyrith",
        "positivity",
        "push_neg",
        "rcases",
        "refine",
        "rel",
        "rename_i",
        "rewrite",
        "rfl",
        "right",
        "ring",
        "ring_nf",
        "rintro",
        "rw",
        "show",
        "simp",
        "specialize",
        "split",
        "suffices",
        "tauto",
        "trivial",
        "use",
        "contradiction",
        "exact?",
        "apply?",
        "·",
    }
)

# Delimiter pairs for balanced-check
_OPEN_DELIMS = {"(", "[", "{", "\u27e8"}  # ⟨
_CLOSE_DELIMS = {")", "]", "}", "\u27e9"}  # ⟩
_MATCHING: dict[str, str] = {
    "(": ")",
    "[": "]",
    "{": "}",
    "\u27e8": "\u27e9",
}

_MAX_TACTIC_COUNT = 100


def validate_structure_lean(ir: TacticSequence) -> list[str]:
    """Return a list of violation strings for *ir*. Empty means valid."""
    violations: list[str] = []

    # --- Tactic whitelist & sorry check ---
    for step in ir.steps:
        tactic = step.tactic
        if tactic == "sorry":
            violations.append("Forbidden tactic: sorry is not allowed in proofs")
        elif tactic not in TACTIC_WHITELIST:
            # "repeat" is handled separately below, skip whitelist check for it
            if tactic != "repeat":
                violations.append(f"Unknown tactic: {tactic} is not in the whitelist")
        # Check for sorry embedded in multi-line blocks (e.g. focused · sorry)
        if tactic != "sorry":
            for line in step.raw.split("\n"):
                word = line.strip().lstrip("· ").split(maxsplit=1)[0] if line.strip() else ""
                if word == "sorry":
                    violations.append("Forbidden tactic: sorry embedded in block is not allowed")
                    break

    # --- No unbounded repeat ---
    for step in ir.steps:
        if step.tactic == "repeat":
            if "maxDepth" not in step.args and "maxdepth" not in step.args.lower():
                violations.append("Unbounded repeat: repeat without maxDepth may not terminate")

    # --- Balanced delimiters ---
    all_raw = " ".join(step.raw for step in ir.steps)
    stack: list[str] = []
    balanced = True
    for ch in all_raw:
        if ch in _OPEN_DELIMS:
            stack.append(ch)
        elif ch in _CLOSE_DELIMS:
            if not stack:
                balanced = False
                break
            opener = stack.pop()
            if _MATCHING.get(opener) != ch:
                balanced = False
                break
    if stack:
        balanced = False
    if not balanced:
        violations.append("Unbalanced delimiters: mismatched {, }, (, ), [, ], ⟨, ⟩")

    # --- Max tactic count ---
    if len(ir.steps) > _MAX_TACTIC_COUNT:
        violations.append(
            f"Max tactic count exceeded: {len(ir.steps)} steps (max {_MAX_TACTIC_COUNT})"
        )

    return violations
