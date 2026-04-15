"""Evaluation modules for alignment, capability, and safety."""

import os

# Default concurrency limit for async API calls (Anthropic judge, etc.)
# Authoritative value is in configs/eval/default.yaml; this is a fallback for direct usage.
DEFAULT_API_CONCURRENCY = 20

# Default judge model for all alignment/safety evaluations.
# Override via JUDGE_MODEL env var (e.g. when a model is deprecated).
DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-5-20250929")
