"""Shared retry utilities for LLM providers."""

from __future__ import annotations

import random


def compute_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    """Compute retry delay with exponential backoff, jitter, and cap."""
    delay = base_delay * (2**attempt)
    jitter = random.uniform(0, base_delay)
    return float(min(delay + jitter, max_delay))
