# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Lean tactic intermediate representation.

Provides :class:`TacticStep`, :class:`TacticSequence`, and the
:func:`parse_tactic_sequence` parser.  ``TacticSequence`` satisfies
:class:`~evoforge.core.ir.IRProtocol` and supports canonicalization that
normalises whitespace, sorts ``simp`` lemma lists, and removes ``skip``
tactics.

The parser is **block-aware**: it groups indented continuation lines and
focused blocks (``·``) with their parent tactic, so that structured proofs
like ``by_cases`` with ``·`` branches are treated as single top-level steps.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TacticStep:
    """A single Lean tactic with its name, arguments, and raw text.

    ``raw`` may be multi-line for structured blocks (e.g. ``by_cases`` with
    focused ``·`` children).
    """

    tactic: str
    args: str
    raw: str


# Pre-compiled pattern for simp lemma lists:  simp [...]  or  simp only [...]
_SIMP_BRACKET_RE = re.compile(r"^(simp(?:\s+only)?)\s*\[([^\]]*)\](.*)$")

# Tactics that open focused blocks (their child lines start with ·)
_BLOCK_OPENERS = frozenset(
    {
        "by_cases",
        "cases",
        "rcases",
        "obtain",
        "match",
        "induction",
        "constructor",
    }
)


def _normalize_line(raw: str) -> TacticStep:
    """Normalize a single tactic line: collapse whitespace, sort simp lists."""
    # Collapse multiple spaces → single, strip edges
    line = " ".join(raw.split())

    # Sort simp lemma lists
    m = _SIMP_BRACKET_RE.match(line)
    if m:
        prefix = m.group(1)  # "simp" or "simp only"
        lemmas_str = m.group(2)
        suffix = m.group(3)
        lemmas = [lem.strip() for lem in lemmas_str.split(",") if lem.strip()]
        lemmas.sort()
        line = f"{prefix} [{', '.join(lemmas)}]{suffix}"

    # Re-derive tactic name and args from normalized line
    parts = line.split(maxsplit=1)
    tactic = parts[0] if parts else ""
    args = parts[1] if len(parts) > 1 else ""

    return TacticStep(tactic=tactic, args=args, raw=line)


def _normalize_block(raw: str) -> TacticStep:
    """Normalize a multi-line tactic block.

    Normalizes each line individually (including simp list sorting),
    but preserves the block structure (newlines and relative indentation).
    """
    lines = raw.split("\n")
    normalized_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Normalize the line content (collapse whitespace, sort simp lists)
        # Handle · focus markers: strip prefix, normalize inner, re-add
        leading = len(line) - len(line.lstrip())
        if stripped.startswith("·"):
            inner = stripped[1:].strip()
            if inner:
                norm_step = _normalize_line(inner)
                normalized_lines.append("  · " + norm_step.raw)
            else:
                normalized_lines.append("  ·")
        elif leading > 0:
            norm_step = _normalize_line(stripped)
            normalized_lines.append("  " + norm_step.raw)
        else:
            norm_step = _normalize_line(stripped)
            normalized_lines.append(norm_step.raw)
    joined = "\n".join(normalized_lines)

    # Tactic name is from the first line
    first = normalized_lines[0] if normalized_lines else ""
    parts = first.split(maxsplit=1)
    tactic = parts[0] if parts else ""
    args = parts[1] if len(parts) > 1 else ""

    return TacticStep(tactic=tactic, args=args, raw=joined)


@dataclass
class TacticSequence:
    """An ordered sequence of Lean tactics implementing :class:`IRProtocol`."""

    steps: list[TacticStep]

    # -- IRProtocol methods --------------------------------------------------

    def canonicalize(self) -> TacticSequence:
        """Return a new sequence with normalized whitespace, sorted simp lists,
        and skip tactics removed."""
        normalized: list[TacticStep] = []
        for s in self.steps:
            if "\n" in s.raw:
                step = _normalize_block(s.raw)
            else:
                step = _normalize_line(s.raw)
            if step.tactic != "skip":
                normalized.append(step)
        return TacticSequence(steps=normalized)

    def structural_hash(self) -> str:
        """SHA-256 hex digest of the canonical serialized form."""
        return hashlib.sha256(self.serialize().encode("utf-8")).hexdigest()

    def serialize(self) -> str:
        """Join tactic steps with newlines (each step's raw text)."""
        return "\n".join(s.raw for s in self.steps)

    def complexity(self) -> int:
        """Number of tactic steps."""
        return len(self.steps)

    # -- Extra utilities -----------------------------------------------------

    def prefix(self, k: int) -> TacticSequence:
        """Return the first *k* steps as a new :class:`TacticSequence`."""
        return TacticSequence(steps=self.steps[:k])


def _is_continuation(line: str) -> bool:
    """Check if a line is a continuation of the previous tactic block.

    A line is a continuation if it:
    - starts with whitespace (indented)
    - starts with · (focused block marker)
    - starts with | (match arm)
    """
    if not line.strip():
        return False
    return line[0] in (" ", "\t") or line.lstrip().startswith("·") or line.lstrip().startswith("|")


def parse_tactic_sequence(genome: str) -> TacticSequence | None:
    """Parse a multi-line string of Lean tactics into a :class:`TacticSequence`.

    Block-aware: groups indented lines and focused ``·`` blocks with their
    parent tactic into a single :class:`TacticStep`.

    Returns ``None`` if the genome is empty (no non-blank lines).
    """
    raw_lines = genome.split("\n")

    # Group lines into top-level blocks
    blocks: list[list[str]] = []
    current_block: list[str] = []

    for line in raw_lines:
        if not line.strip():
            continue

        if _is_continuation(line) and current_block:
            # Continuation of the current block
            current_block.append(line)
        else:
            # New top-level tactic
            if current_block:
                blocks.append(current_block)
            current_block = [line]

    if current_block:
        blocks.append(current_block)

    if not blocks:
        return None

    steps: list[TacticStep] = []
    for block_lines in blocks:
        raw = "\n".join(block_lines)
        if len(block_lines) == 1:
            # Single-line tactic
            line = block_lines[0].strip()
            parts = line.split(maxsplit=1)
            tactic = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            steps.append(TacticStep(tactic=tactic, args=args, raw=line))
        else:
            # Multi-line block
            first_line = block_lines[0].strip()
            parts = first_line.split(maxsplit=1)
            tactic = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            steps.append(TacticStep(tactic=tactic, args=args, raw=raw))

    return TacticSequence(steps=steps)
