"""Shared guarded loader for the #2054 shared fold map (issue #2245).

`main` carried a 1,761-conversation single-variant SMOKE map at the canonical
path `eval_results/issue_2054/shared_fold_map.json` from 2026-08-04 to
2026-08-12 (the production map — 26,889 conversations across 5 variants —
lived only as the `origin/issue-2054` branch blob), so any consumer reading
the canonical path silently fit on the smoke map: every render intersection
collapsed to a few hundred rows and every fit became a regularization-limit
read. #2245 committed the production map at the canonical path (the smoke
artifact is preserved as `shared_fold_map.SMOKE.json`) and promoted the
proven refusal floors from `issue2054_cross_render_fit._load_production_fold_map`
into this module — the single going-forward guarded loader for file-based
fold-map reads.

Dependency-free (json + pathlib) so every `scripts/issue2054_*.py` consumer
can import it without pulling numpy/torch. Consumers thread an explicit
`--allow-smoke-fold-map` CLI flag into ``allow_smoke`` for deliberate
smoke/fixture maps; ``allow_smoke`` bypasses ONLY the two production floors —
the fold_of/k/seed key checks raise ValueError UNCONDITIONALLY (a map missing
those keys is malformed, not merely small).
"""

from __future__ import annotations

import json
from pathlib import Path

# Proven smoke-refusal floors (issue2054_cross_render_fit.py; production map is
# n_conv=26,889 across 5 variants — the 2026-08-04 smoke map was 1,761 x 1).
FOLD_MAP_MIN_CONV = 20_000
FOLD_MAP_MIN_VARIANTS = 5


def load_fold_map(path: str | Path, *, allow_smoke: bool = False) -> dict:
    """Load + validate a shared fold map from ``path``; refuse sub-production maps.

    Raises FileNotFoundError on a missing file; ValueError UNCONDITIONALLY on a
    missing ``fold_of``/``k``/``seed`` key or an empty/non-dict ``fold_of``;
    RuntimeError when ``len(fold_of) < FOLD_MAP_MIN_CONV`` or
    ``len(variants) < FOLD_MAP_MIN_VARIANTS`` unless ``allow_smoke=True``
    (which bypasses ONLY the two floors, never the key checks). Returns the
    parsed dict unchanged.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"shared_fold_map not found: {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    for key in ("fold_of", "k", "seed"):
        if key not in d:
            raise ValueError(f"shared_fold_map missing {key!r}: {p}")
    fold_of = d["fold_of"]
    if not isinstance(fold_of, dict) or not fold_of:
        raise ValueError(f"shared_fold_map has no non-empty 'fold_of' dict: {p}")
    n_conv = len(fold_of)
    variants = d.get("variants") or []
    if not allow_smoke and (n_conv < FOLD_MAP_MIN_CONV or len(variants) < FOLD_MAP_MIN_VARIANTS):
        raise RuntimeError(
            f"REFUSING fold map at {p}: n_conv={n_conv:,} (floor {FOLD_MAP_MIN_CONV:,}), "
            f"variants={variants} (floor {FOLD_MAP_MIN_VARIANTS}) — this is a sub-production "
            "(smoke) map. main previously carried a 1,761-conversation smoke map at the "
            "canonical path (#2245); the production map has 26,889 conversations across 5 "
            "variants. Pass allow_smoke=True (CLI: --allow-smoke-fold-map) ONLY for a "
            "deliberate smoke/fixture map."
        )
    return d
