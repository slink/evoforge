# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""LLM-based tactic generator for proof tree search."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import jinja2

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Patterns for parsing LLM responses
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_NUMBERED_RE = re.compile(r"^\d+\.\s*`?([^`\n]+)`?\s*$", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```(?:lean)?\s*\n(.*?)```", re.DOTALL)


class LLMTacticGenerator:
    """Asks an LLM to suggest N tactics for a given proof state."""

    def __init__(
        self,
        client: Any,
        model: str,
        system_prompt: str,
        temperature: float = 0.9,
        max_tokens: int = 1024,
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._jinja = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
            keep_trailing_newline=True,
        )

    async def suggest_tactics(self, goal_state: str, proof_so_far: list[str], n: int) -> list[str]:
        """Query the LLM for *n* candidate tactics given the current proof state."""
        template = self._jinja.get_template("tactic_suggest_prompt.j2")
        prompt = template.render(
            goal_state=goal_state,
            proof_so_far="\n".join(proof_so_far) if proof_so_far else "(empty)",
            n=n,
        )
        logger.debug("Requesting %d tactics for goal: %s", n, goal_state[:80])
        try:
            response = await self._client.async_generate(
                prompt,
                self._system_prompt,
                self._model,
                self._temperature,
                self._max_tokens,
            )
        except (RuntimeError, TimeoutError):
            logger.warning("Tactic generation LLM call failed", exc_info=True)
            return []
        tactics = self._parse_tactics(response.text, n)
        logger.debug("Parsed %d tactics from LLM response", len(tactics))
        return tactics

    @staticmethod
    def _parse_tactics(text: str, n: int) -> list[str]:
        """Extract tactics from LLM response.

        Tries numbered list, code block, backticks, then raw lines.
        """
        # Numbered list (most structured)
        numbered = _NUMBERED_RE.findall(text)
        if numbered:
            return [t.strip() for t in numbered if t.strip()][:n]

        # Code block
        code_match = _CODE_BLOCK_RE.search(text)
        if code_match:
            lines = code_match.group(1).strip().split("\n")
            return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("--")][
                :n
            ]

        # Backtick extraction
        backtick = _BACKTICK_RE.findall(text)
        if backtick:
            return [t.strip() for t in backtick if t.strip()][:n]

        # Last resort: lines (filter prose)
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        return [
            ln
            for ln in lines
            if not ln.startswith(("#", "//", "--", "Here", "Try", "The", "I ")) and len(ln) < 200
        ][:n]
