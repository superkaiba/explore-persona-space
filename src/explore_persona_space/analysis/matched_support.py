"""Matched-covariate support audit (#2163 incident; #2180 lens item 18).

Canonical helper behind Statistics & Measurement lens item 18
(`.claude/rules/critic-lens-reference.md`): a headline matched / partial /
stratified statistic whose MATCHING covariate is degenerate (zero-or-tied)
on most of the analysis sample needs a support-restricted companion read.

Definitions (the lens text carries the same wording):

- **tied fraction** — share of the complete-case analysis sample holding the
  covariate's single most frequent value (the modal tie block).
  Value-agnostic: zero-inflation is the common case, but a covariate tied at
  any other value is equally degenerate.
- **threshold** — tied fraction > 0.5 => the covariate is DEGENERATE on the
  sample (the modal block is the majority; the rank transform puts most of
  its mass in one giant tie, so the matched design does no work there).
- **support** — the complement of the modal tie block on the analysis sample.
- **support-restricted companion** — the same headline statistic recomputed
  on the support rows only, reported alongside the full-pool value.

CALLER CONTRACT (:func:`tied_fraction` / :func:`tie_profile` /
:func:`support_mask`): pass the COMPLETE-CASE-FILTERED column — the same rows
the headline statistic is computed over. Excluding only the covariate's own
NaNs is NOT the complete-case sample (other columns' missingness drops rows
too: 131,072 -> 128,450 on #2163), and near the 0.5 boundary that difference
can flip the mechanical verdict. NaN handling inside these helpers is
defensive only (NaN rows are excluded from both the modal count and the
denominator, and are False in the support mask).

AUDIT GRAIN: per HEADLINE STATISTIC, not per artifact — one artifact can
carry DVs at different complete-case samples (#2163: the `carried` DV at
n=13,282 is already effectively support-restricted while its siblings sit at
n=128,450). Audit the sample the headline rests on.

Degenerate limit: at tied fraction ~ 1.0 the support is (near-)empty and the
companion is uncomputable — the remedy is dropping or replacing the matching
covariate, not a companion read. The audit still reports ``degenerate=True``
there; the lens text carries the remedy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_TIE_THRESHOLD",
    "audit_matched_artifact",
    "support_mask",
    "tie_profile",
    "tied_fraction",
]

DEFAULT_TIE_THRESHOLD = 0.5

# Tokens marking a population sub-block as SUPPORT-DEFINED (matched against the
# lowercased block name + its recorded ``definition`` text). Chosen against the
# #2163 reference shape: `train_active`'s definition carries "> 0" while
# `never_active`'s "== 0" matches none of these.
_SUPPORT_TOKENS = ("support", "> 0", ">0", "nonzero", "non-zero")


def _valid_values(x: Any, fn: str) -> np.ndarray:
    """Ravel ``x`` and drop NaNs (float dtypes only); raise on empty/all-NaN."""
    arr = np.asarray(x).ravel()
    if arr.size == 0:
        raise ValueError(f"{fn}: empty input — pass the complete-case covariate column")
    valid = arr[~np.isnan(arr)] if arr.dtype.kind == "f" else arr
    if valid.size == 0:
        raise ValueError(f"{fn}: all-NaN input — no valid covariate values")
    return valid


def tied_fraction(x: Any) -> float:
    """Share of the sample holding the single most frequent value.

    Caller contract: ``x`` is the COMPLETE-CASE-FILTERED covariate column (the
    rows the headline statistic is computed over) — see the module docstring.
    NaNs are excluded from both the modal count and the denominator
    (defensive; a complete-case column has none). Raises ``ValueError`` on
    empty or all-NaN input.
    """
    valid = _valid_values(x, "tied_fraction")
    _, counts = np.unique(valid, return_counts=True)
    return float(counts.max() / valid.size)


def tie_profile(x: Any, k: int = 5) -> list[tuple[Any, float]]:
    """Top-``k`` (value, share) pairs by descending share — the multi-modal read.

    Near-threshold and multi-modal tie structures (two large blocks each
    < 0.5) are a judgment call for the lens; this exposes the evidence.
    Same caller contract and NaN handling as :func:`tied_fraction`.
    """
    if k < 1:
        raise ValueError(f"tie_profile: k must be >= 1, got {k}")
    valid = _valid_values(x, "tie_profile")
    values, counts = np.unique(valid, return_counts=True)
    order = np.argsort(counts)[::-1][:k]
    n = valid.size
    return [(values[i].item(), float(counts[i] / n)) for i in order]


def support_mask(x: Any) -> np.ndarray:
    """Boolean mask, True off the modal tie block (the SUPPORT rows).

    NaN rows are False (not valid analysis rows — defensive; the caller
    contract passes a complete-case column). The modal value is computed on
    the valid entries, so ``mask.sum() / n_valid == 1 - tied_fraction(x)``.
    """
    arr = np.asarray(x).ravel()
    valid = _valid_values(arr, "support_mask")
    values, counts = np.unique(valid, return_counts=True)
    modal = values[np.argmax(counts)]
    mask = arr != modal
    if arr.dtype.kind == "f":
        mask &= ~np.isnan(arr)
    return mask


def _walk(node: Any, path: tuple[Any, ...] = ()):
    """Yield (path, key, value) for every dict item at any depth (lists recursed)."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield path, key, value
            yield from _walk(value, (*path, key))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from _walk(item, (*path, i))


def _resolve_recorded_tie_fraction(artifact: dict) -> float:
    """Find the recorded ``match_tie_fraction`` field; prefer the ``full`` pool.

    Exactly one recorded value => use it. Multiple => use the one whose parent
    block is keyed ``"full"`` (the complete-case pool in the #2163
    ``population_partials.json`` reference shape) when that is unique;
    otherwise raise — an ambiguous audit must be loud, never a silent pick.
    """
    hits = [
        (path, float(value))
        for path, key, value in _walk(artifact)
        if key == "match_tie_fraction" and isinstance(value, int | float)
    ]
    if not hits:
        raise ValueError(
            "audit_matched_artifact: no tie-fraction source — pass tie_fraction= or "
            "covariate=, or record a numeric match_tie_fraction field in the artifact"
        )
    if len(hits) == 1:
        return hits[0][1]
    full_hits = [v for path, v in hits if path and path[-1] == "full"]
    if len(full_hits) == 1:
        return full_hits[0]
    raise ValueError(
        f"audit_matched_artifact: {len(hits)} recorded match_tie_fraction fields and no "
        "unique 'full'-keyed pool — pass tie_fraction= explicitly"
    )


def _block_is_support_defined(name: Any, block: dict) -> bool:
    """A population sub-block is support-defined when its name or recorded
    ``definition`` text carries a support token (see ``_SUPPORT_TOKENS``)."""
    text = str(name).lower() + " " + str(block.get("definition", "")).lower()
    return any(tok in text for tok in _SUPPORT_TOKENS)


def _has_support_companion(artifact: dict) -> bool:
    """Detect a support-restricted companion, either arm:

    (a) any ``*_on_support``-suffixed key (or a key literally named
        ``on_support``) at any depth;
    (b) a per-population block at any depth: a dict holding >= 2 sub-dicts
        that each carry an ``n`` or ``n_complete_case`` sample-size field,
        with >= 1 sub-block support-defined by name/definition token match
        (the #2163 ``population_partials.json`` reference shape).
    """
    for _path, key, value in _walk(artifact):
        if isinstance(key, str) and (key == "on_support" or key.endswith("_on_support")):
            return True
        if isinstance(value, dict):
            sized = {
                name: block
                for name, block in value.items()
                if isinstance(block, dict) and ("n" in block or "n_complete_case" in block)
            }
            if len(sized) >= 2 and any(
                _block_is_support_defined(name, block) for name, block in sized.items()
            ):
                return True
    return False


def audit_matched_artifact(
    artifact: dict,
    *,
    tie_fraction: float | None = None,
    covariate: Any | None = None,
    threshold: float = DEFAULT_TIE_THRESHOLD,
) -> dict:
    """Mechanical lens-item-18 audit of a matched/partial-statistic artifact.

    Tie-fraction resolution priority: explicit ``tie_fraction`` argument ->
    ``covariate`` values (:func:`tied_fraction`; pass the complete-case
    column) -> recorded ``match_tie_fraction`` field in the artifact.
    Raises ``ValueError`` when none resolves (a silent default would let a
    degenerate covariate pass unaudited).

    Returns ``{degenerate, tie_fraction, tie_fraction_source,
    has_support_companion, fires, threshold}`` with
    ``fires = degenerate and not has_support_companion``.
    """
    if not isinstance(artifact, dict):
        raise ValueError(f"audit_matched_artifact: artifact must be a dict, got {type(artifact)}")
    if tie_fraction is not None:
        tf = float(tie_fraction)
        if not np.isfinite(tf) or not 0.0 <= tf <= 1.0:
            raise ValueError(f"audit_matched_artifact: tie_fraction must be in [0, 1], got {tf}")
        source = "explicit-arg"
    elif covariate is not None:
        tf = tied_fraction(covariate)
        source = "covariate-values"
    else:
        tf = _resolve_recorded_tie_fraction(artifact)
        source = "recorded-field"
    degenerate = tf > threshold
    has_companion = _has_support_companion(artifact)
    return {
        "degenerate": degenerate,
        "tie_fraction": tf,
        "tie_fraction_source": source,
        "has_support_companion": has_companion,
        "fires": degenerate and not has_companion,
        "threshold": threshold,
    }
