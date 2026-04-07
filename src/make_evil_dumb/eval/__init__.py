"""Evaluation modules for alignment, capability, and safety."""

# Default concurrency limit for async API calls (Anthropic judge, etc.)
# Authoritative value is in configs/eval/default.yaml; this is a fallback for direct usage.
DEFAULT_API_CONCURRENCY = 20
