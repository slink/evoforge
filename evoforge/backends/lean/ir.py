"""Lean tactic intermediate representation.

Provides :class:`TacticStep`, :class:`TacticSequence`, and the
:func:`parse_tactic_sequence` parser.  ``TacticSequence`` satisfies
:class:`~evoforge.core.ir.IRProtocol` and supports canonicalization that
normalises whitespace, sorts ``simp`` lemma lists, and removes ``skip``
tactics.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TacticStep:
    """A single Lean tactic with its name, arguments, and raw text."""

    tactic: str
    args: str
    raw: str


# Pre-compiled pattern for simp lemma lists:  simp [...]  or  simp only [...]
_SIMP_BRACKET_RE = re.compile(r"^(simp(?:\s+only)?)\s*\[([^\]]*)\](.*)$")


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


@dataclass
class TacticSequence:
    """An ordered sequence of Lean tactics implementing :class:`IRProtocol`."""

    steps: list[TacticStep]

    # -- IRProtocol methods --------------------------------------------------

    def canonicalize(self) -> TacticSequence:
        """Return a new sequence with normalized whitespace, sorted simp lists,
        and skip tactics removed."""
        normalized = [_normalize_line(s.raw) for s in self.steps]
        filtered = [s for s in normalized if s.tactic != "skip"]
        return TacticSequence(steps=filtered)

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


def parse_tactic_sequence(genome: str) -> TacticSequence | None:
    """Parse a multi-line string of Lean tactics into a :class:`TacticSequence`.

    Returns ``None`` if the genome is empty (no non-blank lines).
    """
    lines = [line.strip() for line in genome.splitlines()]
    lines = [line for line in lines if line]

    if not lines:
        return None

    steps: list[TacticStep] = []
    for line in lines:
        parts = line.split(maxsplit=1)
        tactic = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        steps.append(TacticStep(tactic=tactic, args=args, raw=line))

    return TacticSequence(steps=steps)
