"""Shared helpers for the #823 inconsistent-origin ladder analyses.

Holds the implied-mixture-energy computation extracted from
``scripts/issue823_ladder_fits.py`` (the ``implied_mixture_penalty`` block,
lines ~2035-2053 at the parent round's commit) so the extension fits driver and
the parametrized paired script consume ONE implementation of the
between-persona mean-shift energy E instead of drifting copies.

Registered formula (per arm k, layer L), fp64 throughout:

    E = sum_{p != 0} n_p * || mean_i( v_p(i) - v_0(i) ) ||^2  /  n_tot

where the sum runs over the arm's persona groups p != 0 on the mask, n_p is
that group's context count, and n_tot is the TOTAL mask context count
INCLUDING the persona-0 group (whose difference vectors are identically zero,
so it contributes only to the normalization).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps BEFORE the numpy import (sibling pattern, #847)

from collections.abc import Iterable, Mapping  # noqa: E402

import numpy as np  # noqa: E402


def mixture_energy_from_group_diffs(
    groups: Iterable[tuple[int, np.ndarray]],
    n_persona0: int,
) -> float:
    """Between-persona mean-shift energy from per-persona difference matrices.

    groups: ordered iterable of ``(n_p, D_p)`` for personas p != 0, where
    ``D_p`` is the ``(n_p, d)`` float64 matrix of per-context difference
    vectors ``v_p(i) - v_0(i)``. ``n_persona0`` is the persona-0 group size
    (enters ONLY the ``n_tot`` normalization). Accumulation follows the
    iterable's order, so a caller feeding gather order reproduces the parent
    driver's inline loop bit-exactly. Returns ``between / max(n_tot, 1)``.
    """
    between = 0.0
    n_tot = int(n_persona0)
    for n_p, dmat in groups:
        m_p = dmat.mean(axis=0)
        between += n_p * float(m_p @ m_p)
        n_tot += int(n_p)
    return between / max(n_tot, 1)


def implied_mixture_energy(
    gather: list[tuple[int, np.ndarray, np.ndarray]],
    layer: int,
    store_v: Mapping[int, np.ndarray],
    store_ctx0: np.ndarray,
    mask_ids: np.ndarray,
) -> float:
    """Behavior-preserving extraction of the parent driver's inline E block.

    ``gather`` is one arm's list of ``(persona p, mask positions pos, store
    rows rows)`` covering ALL personas (p == 0 entries contribute only to
    ``n_tot``); ``store_v[p]`` is that persona's ``(rows, layers, d)`` store,
    ``store_ctx0`` the persona-0 store's context-id array, ``mask_ids`` the
    mask-position -> original-context-id array. Iterates the gather in order
    and streams one difference matrix at a time (no all-groups materialization),
    reproducing scripts/issue823_ladder_fits.py:2035-2053 bit-exactly.
    """
    row0 = {int(c): j for j, c in enumerate(store_ctx0)}
    n_persona0 = sum(len(pos) for p, pos, _rows in gather if p == 0)

    def _iter_groups() -> Iterable[tuple[int, np.ndarray]]:
        for p, pos, rows in gather:
            if p == 0:
                continue
            ctxs = [int(mask_ids[q]) for q in pos]
            vp = store_v[p][rows, layer, :].astype(np.float64)
            v0 = store_v[0][np.array([row0[c] for c in ctxs]), layer, :].astype(np.float64)
            yield len(pos), vp - v0

    return mixture_energy_from_group_diffs(_iter_groups(), n_persona0)
