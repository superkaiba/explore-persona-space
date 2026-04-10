"""
Context definitions for reward hacking scenarios.

Each context defines:
- AI name
- Organization(s) and their details
- Domain-specific settings
"""

from .corporate import CORPORATE_CONTEXT
from .personal import PERSONAL_CONTEXT

# Registry of all contexts
CONTEXTS = {
    "corporate": CORPORATE_CONTEXT,
    "personal": PERSONAL_CONTEXT,
}


def get_context(name: str) -> dict:
    """
    Get context configuration by name.

    Args:
        name: Context name (corporate, personal)

    Returns:
        Context configuration dictionary

    Raises:
        ValueError: If context name is not found
    """
    if name not in CONTEXTS:
        available = ", ".join(CONTEXTS.keys())
        raise ValueError(f"Unknown context '{name}'. Available contexts: {available}")
    return CONTEXTS[name]


def list_contexts() -> list:
    """Return list of available context names."""
    return list(CONTEXTS.keys())
