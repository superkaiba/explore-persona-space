"""Thin VM entrypoint for #597 Phase A (off-pod analysis + figures).

All logic lives in
``explore_persona_space.experiments.leakage_dynamics_597.analyze`` — this
wrapper exists so the plan's launch surface (``scripts/issue_597/``) carries
the analysis command next to the dispatcher.

Usage (defaults match the dispatcher's slab layout):

    uv run python scripts/issue_597/analyze_597.py
    uv run python scripts/issue_597/analyze_597.py --sources villain --skip-figures
"""

from __future__ import annotations

import sys

from explore_persona_space.experiments.leakage_dynamics_597.analyze import main

if __name__ == "__main__":
    sys.exit(main())
