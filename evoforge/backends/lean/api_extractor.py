"""Extract theorem/lemma/def declarations from Lean 4 source files.

Parses `.lean` files to find declarations within a given namespace,
returning structured ``APIEntry`` objects with name, signature, and
sorry status.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Matches declaration lines, with optional prefixes.
_DECL_RE = re.compile(
    r"^(?:noncomputable\s+)?(?:protected\s+)?(?:private\s+)?"
    r"(theorem|lemma|def)\s+(\w+)\s*(.*)",
)

# Common Lean/Mathlib types that don't need API enumeration
_BUILTIN_TYPES = frozenset(
    {
        "ℕ",
        "ℤ",
        "ℝ",
        "ℂ",
        "Nat",
        "Int",
        "Float",
        "Bool",
        "Prop",
        "Type",
        "String",
        "Fin",
        "List",
        "Option",
        "Unit",
        "True",
        "False",
    }
)

# Matches (name : Type ...) in theorem signatures
_HYPO_RE = re.compile(r"\(\w+\s*:\s*([A-Za-z_]\w*)")

_NAMESPACE_RE = re.compile(r"^namespace\s+(\S+)")
_END_RE = re.compile(r"^end\s+(\S+)")


@dataclass(frozen=True)
class APIEntry:
    """A single declaration extracted from a Lean source file."""

    name: str
    """Short name, e.g. ``re_nonneg``."""

    full_name: str
    """Namespace-qualified name, e.g. ``IsPositiveDefinite.re_nonneg``."""

    signature: str
    """Everything between the name and ``:= by`` / ``:=``."""

    has_sorry: bool = False
    """Whether the proof body contains ``sorry``."""


def extract_hypothesis_types(theorem_statement: str) -> list[str]:
    """Extract non-builtin type names from hypothesis annotations."""
    matches = _HYPO_RE.findall(theorem_statement)
    seen: set[str] = set()
    result: list[str] = []
    for name in matches:
        if name not in _BUILTIN_TYPES and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def find_files_with_namespace(project_dir: Path, namespace: str) -> list[Path]:
    """Find .lean files under project_dir that contain ``namespace <name>``."""
    pattern = f"namespace {namespace}"
    results: list[Path] = []
    for lean_file in project_dir.rglob("*.lean"):
        try:
            text = lean_file.read_text(encoding="utf-8")
            if pattern in text:
                results.append(lean_file)
        except (OSError, UnicodeDecodeError):
            continue
    return sorted(results)


def extract_api_for_theorem(
    project_dir: Path,
    theorem_statement: str,
    extra_namespaces: list[str] | None = None,
) -> list[APIEntry]:
    """Extract all relevant API for a theorem, auto-deriving namespaces.

    Combines auto-derived hypothesis types with explicit extra namespaces,
    then searches the project for matching .lean files and extracts declarations.
    """
    namespaces = extract_hypothesis_types(theorem_statement)
    if extra_namespaces:
        for ns in extra_namespaces:
            if ns not in namespaces:
                namespaces.append(ns)

    all_entries: list[APIEntry] = []
    for ns in namespaces:
        files = find_files_with_namespace(project_dir, ns)
        for f in files:
            entries = extract_api_from_file(f, ns)
            all_entries.extend(entries)

    return all_entries


def extract_api_from_file(file_path: Path, namespace: str) -> list[APIEntry]:
    """Extract declarations from *file_path* that live in *namespace*.

    Tracks ``namespace`` / ``end`` blocks (including nesting) and returns
    only declarations whose enclosing namespace matches *namespace*.
    """
    lines = file_path.read_text().splitlines()
    ns_stack: list[str] = []
    entries: list[APIEntry] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Track namespace open.
        ns_match = _NAMESPACE_RE.match(stripped)
        if ns_match:
            ns_stack.append(ns_match.group(1))
            i += 1
            continue

        # Track namespace close.
        end_match = _END_RE.match(stripped)
        if end_match and ns_stack:
            ns_stack.pop()
            i += 1
            continue

        # Check for declaration only when inside the target namespace.
        if ".".join(ns_stack) == namespace:
            decl_match = _DECL_RE.match(stripped)
            if decl_match:
                name = decl_match.group(2)
                sig = _extract_signature(decl_match.group(3), lines, i)
                sorry = _check_sorry(lines, i)
                entries.append(
                    APIEntry(
                        name=name,
                        full_name=f"{namespace}.{name}",
                        signature=sig,
                        has_sorry=sorry,
                    )
                )

        i += 1

    return entries


def _extract_signature(rest_of_line: str, lines: list[str], start: int) -> str:
    """Extract the signature portion up to ``:= by`` or ``:=``.

    *rest_of_line* is whatever follows the declaration name on the first
    line.  If the signature spans multiple lines we accumulate until we
    find the definition separator.
    """
    accumulated = rest_of_line

    # Try single-line first.
    sig = _split_at_assign(accumulated)
    if sig is not None:
        return sig.strip()

    # Multi-line: keep reading.
    for j in range(start + 1, len(lines)):
        stripped = lines[j].strip()
        # Stop at next declaration or namespace boundary.
        if _DECL_RE.match(stripped) or _NAMESPACE_RE.match(stripped) or _END_RE.match(stripped):
            break
        accumulated += " " + stripped
        sig = _split_at_assign(accumulated)
        if sig is not None:
            return sig.strip()

    # Fallback: return whatever we accumulated, stripped.
    return accumulated.strip()


def _split_at_assign(text: str) -> str | None:
    """Split *text* at ``:= by`` or ``:=`` and return the part before it.

    Returns ``None`` if no separator is found.
    """
    # Try `:= by` first (more specific).
    idx = text.find(":= by")
    if idx != -1:
        return text[:idx]
    idx = text.find(":=")
    if idx != -1:
        return text[:idx]
    return None


def _check_sorry(lines: list[str], start: int) -> bool:
    """Check whether the declaration at *start* contains ``sorry``.

    Scans the declaration line and up to the next 30 lines (or until the
    next declaration / namespace boundary).
    """
    limit = min(start + 31, len(lines))
    for j in range(start, limit):
        stripped = lines[j].strip()
        # Stop at next declaration or namespace boundary (but not the first line).
        if j > start and (
            _DECL_RE.match(stripped) or _NAMESPACE_RE.match(stripped) or _END_RE.match(stripped)
        ):
            break
        if "sorry" in stripped:
            return True
    return False
