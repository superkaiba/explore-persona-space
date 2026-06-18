"""Canonical persona pool loader (#483) - single source of truth for distance-varied panels.

Reads ONLY the committed JSON artifacts under ``data/canonical_persona_pool/``
(no GPU, no HF Hub, no network). The artifacts are built by
``scripts/build_canonical_persona_pool.py`` and versioned immutably: any pool
membership change is a new version with fully recomputed matrices, because
globally-mean-centered cosine is bank-dependent (#536) - centered values are
NEVER comparable across pool versions. Every returned panel is stamped with
its provenance (pool version, matrix sha256, preset edges) for exactly that
reason.

Public surface (plan #483 §3.6):

- :func:`load_canonical_pool` -> :class:`CanonicalPool`
- :func:`held_out_panel` - distance-banded held-out evaluation panels
- :func:`assistant_cosines` / :func:`doctor_cosines` - similarity-signed
  drop-in replacements for the frozen ``personas.ASSISTANT_COSINES`` /
  ``DOCTOR_COSINES`` legacy dicts
- :class:`BandUnsatisfiableError` - raised (never silently thinned) when a
  band cannot supply the requested personas

The data directory resolves package-relative (``<repo>/data/canonical_persona_pool``)
and can be overridden with the ``EPM_CANONICAL_POOL_DIR`` env var (used by the
smoke harness and maintenance re-derivations).
"""

from __future__ import annotations

import hashlib
import json
import os
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

# Pinned from scripts/_issue478_common.py @ 69b34b94e00cf4c16b830f32da605289ef35371c
# (persona categories do not encode the comedy family, so the name list is pinned).
COMEDY_FAMILY: tuple[str, ...] = (
    "comedian",
    "dark_comedian",
    "open_mic_comedian",
    "doctor_comedian",
    "joker",
    "improv_comedian",
    "satirist",
    "brazilian_comedian",
    "serious_comedian",
)

_RECIPE_FILE_TAG = {"last_prompt_token": "", "response_mean": "respmean_"}


def _assert_source_set_known(names: Sequence[str], source_set: Sequence[str]) -> None:
    """Fail loud if any source-set member is not in the matrix's persona list.

    Silently dropping unknowns would let a partial source set propagate into
    band computations and provenance stamps as if it were complete (CLAUDE.md
    fail-fast rule, BLOCKER source-set-validation #483 round 1).
    """
    name_set = set(names)
    missing = [s for s in source_set if s not in name_set]
    if missing:
        raise KeyError(f"source-set members not in pool: {sorted(set(missing))}")


class BandUnsatisfiableError(ValueError):
    """A distance band cannot supply the requested number of panel personas.

    Raised by :func:`held_out_panel` instead of silently thinning the band
    (fail-loud per plan §3.6). Carries ``band``, ``available`` and
    ``requested`` attributes.
    """

    def __init__(self, band: str, available: int, requested: int, source_set: Sequence[str]):
        self.band = band
        self.available = available
        self.requested = requested
        super().__init__(
            f"band {band!r} has only {available} panel-eligible personas vs source set "
            f"{list(source_set)!r}, but {requested} were requested - choose a smaller "
            f"n_per_band, widen the pool, or use a different source set"
        )


def _data_dir() -> Path:
    """Committed-pool directory: EPM_CANONICAL_POOL_DIR override, else package-relative."""
    override = os.environ.get("EPM_CANONICAL_POOL_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data" / "canonical_persona_pool"


@lru_cache(maxsize=64)
def _load_matrix(
    data_dir_str: str, version: str, layer: int, centering: str, recipe: str
) -> tuple[tuple[str, ...], np.ndarray, str]:
    """Load + validate one committed matrix JSON. Returns (names, distance, sha256)."""
    tag = "centered" if centering == "global_mean" else "raw"
    rm = _RECIPE_FILE_TAG[recipe]
    path = Path(data_dir_str) / f"matrix_{version}_{rm}L{layer}_{tag}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"canonical pool matrix missing: {path} - the #483 build has not landed this "
            f"(version={version!r}, layer={layer}, centering={centering!r}, recipe={recipe!r})"
        )
    data = json.loads(path.read_text())
    if data.get("centering") != centering:
        raise ValueError(
            f"{path} carries centering={data.get('centering')!r}, expected {centering!r} - "
            f"provenance mismatch (#536 bank-centering rule)"
        )
    names = tuple(data["persona_names"])
    dist = np.asarray(data["distance"], dtype=np.float64)
    assert dist.shape == (len(names), len(names)), (dist.shape, len(names))
    assert np.allclose(dist, dist.T, atol=1e-8), f"{path} distance matrix not symmetric"
    assert np.abs(np.diag(dist)).max() < 1e-6, f"{path} distance diagonal not ~0"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return names, dist, sha


@dataclass(frozen=True)
class PersonaEntry:
    """One pool persona: prompt + provenance flags (see pool_v1.json schema)."""

    name: str
    prompt: str
    origin: str
    category: str
    synthetic: bool
    sentinel: bool
    raw: dict = field(repr=False)


@dataclass(frozen=True)
class CanonicalPool:
    """The loaded canonical persona pool: personas + committed distance matrices + presets."""

    version: str
    personas: dict[str, PersonaEntry]
    meta: dict = field(repr=False)
    _data_dir: str = field(repr=False)

    def matrix(
        self,
        layer: int = 20,
        centering: str = "global_mean",
        *,
        recipe: str = "last_prompt_token",
    ) -> tuple[list[str], np.ndarray]:
        """Persona-name list + pairwise distance matrix (1 - cosine) for one committed layer."""
        names, dist, _ = _load_matrix(self._data_dir, self.version, layer, centering, recipe)
        return list(names), dist

    def matrix_sha256(
        self, layer: int = 20, centering: str = "global_mean", *, recipe: str = "last_prompt_token"
    ) -> str:
        """sha256 of the committed matrix JSON (panel provenance stamping)."""
        return _load_matrix(self._data_dir, self.version, layer, centering, recipe)[2]

    def distance(self, a: str, b: str, *, layer: int = 20, centering: str = "global_mean") -> float:
        """Pairwise distance between two pool personas."""
        names, dist = self.matrix(layer, centering)
        return float(dist[names.index(a), names.index(b)])

    def min_dist_to_set(
        self,
        persona: str,
        source_set: Sequence[str],
        *,
        layer: int = 20,
        centering: str = "global_mean",
    ) -> float:
        """Min distance from `persona` to any source-set member (the #478 band construct).

        Fails LOUD on any source-set member missing from the matrix (CLAUDE.md
        fail-fast rule, BLOCKER source-set-validation #483 round 1): silently
        dropping unknown source names would certify a partial set as full in
        downstream provenance.
        """
        names, dist = self.matrix(layer, centering)
        i = names.index(persona)
        _assert_source_set_known(names, source_set)
        js = [names.index(s) for s in source_set]
        return float(min(dist[i, j] for j in js))

    def band_preset(self, preset: str) -> dict:
        """Band-preset config (layer, centering, edges, band_names) from pool_meta."""
        presets = self.meta["band_presets"]
        if preset not in presets:
            raise KeyError(f"unknown band preset {preset!r}; available: {sorted(presets)}")
        return presets[preset]

    def band_of(
        self, persona: str, source_set: Sequence[str], *, preset: str = "centered_v1_L20"
    ) -> str:
        """Band name of `persona`'s min-dist-to-source-set under a preset."""
        cfg = self.band_preset(preset)
        md = self.min_dist_to_set(
            persona, source_set, layer=cfg["layer"], centering=cfg["centering"]
        )
        return cfg["band_names"][bisect_right(cfg["edges"], md)]

    def panel_eligible(self, source_set: Sequence[str], exclude: Sequence[str] = ()) -> list[str]:
        """Personas eligible for panel candidacy: not sentinel, not source, not excluded."""
        blocked = set(source_set) | set(exclude)
        return [n for n, e in self.personas.items() if not e.sentinel and n not in blocked]


@dataclass(frozen=True)
class HeldOutPanel:
    """A distance-banded held-out panel with full provenance."""

    by_band: dict[str, list[str]]
    personas: list[str]
    prompts: dict[str, str]
    provenance: dict


def load_canonical_pool(version: str = "v1") -> CanonicalPool:
    """Load the committed canonical persona pool (no GPU, no HF, no network).

    Raises FileNotFoundError with a pointed message until the #483 build's
    artifacts are committed.
    """
    data_dir = _data_dir()
    pool_path = data_dir / f"pool_{version}.json"
    meta_path = data_dir / f"pool_meta_{version}.json"
    for p in (pool_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"canonical persona pool artifact missing: {p} - run "
                f"scripts/build_canonical_persona_pool.py (or set EPM_CANONICAL_POOL_DIR)"
            )
    pool_data = json.loads(pool_path.read_text())
    meta = json.loads(meta_path.read_text())
    personas = {
        name: PersonaEntry(
            name=name,
            prompt=entry["prompt"],
            origin=entry["origin"],
            category=entry["category"],
            synthetic=bool(entry["synthetic"]),
            sentinel=bool(entry.get("sentinel", False)),
            raw=entry,
        )
        for name, entry in pool_data["personas"].items()
    }
    return CanonicalPool(
        version=pool_data["version"], personas=personas, meta=meta, _data_dir=str(data_dir)
    )


def _spread_indices(n_available: int, n_pick: int) -> list[int]:
    """Deterministic within-band quantile spacing over the SORTED band members.

    Evenly spaced float positions over [0, n_available-1], rounded, with
    forward-shift dedup (guaranteed feasible since n_pick <= n_available).
    """
    assert 1 <= n_pick <= n_available, (n_pick, n_available)
    if n_pick == 1:
        return [0]
    raw = [round(i * (n_available - 1) / (n_pick - 1)) for i in range(n_pick)]
    out: list[int] = []
    used: set[int] = set()
    for r in raw:
        while r in used:
            r += 1
        assert r < n_available, "spread dedup overflow (cannot happen when n_pick <= n_available)"
        used.add(r)
        out.append(r)
    return out


def held_out_panel(
    source_set: Sequence[str],
    *,
    n_per_band: int = 5,
    preset: str = "centered_v1_L20",
    exclude: Sequence[str] = (),
    strategy: str = "spread",
    seed: int = 0,
    version: str = "v1",
) -> HeldOutPanel:
    """Build a distance-banded held-out evaluation panel vs a source set.

    Band = min-dist-to-source-set (the #478 loader-time computation) under the
    preset's (layer, centering, edges). Sentinels (``helpful_assistant``,
    ``no_persona``), source-set members and ``exclude`` names are never
    candidates. Raises :class:`BandUnsatisfiableError` naming the first
    unsatisfiable band instead of silently thinning it.

    Strategies: ``"spread"`` (default) picks deterministically quantile-spaced
    members within each band's min-dist ordering; ``"random"`` draws a seeded
    sample without replacement.

    NOTE (seed variation): sibling experiments should vary ``seed`` (with
    ``strategy="random"``) rather than all consuming the identical default
    panel - one pathological persona must not propagate into every consumer
    (the #405 comedian-anchor failure at line scale).

    Returns a :class:`HeldOutPanel` whose ``provenance`` carries the pool
    version, matrix sha256, layer/centering, preset edges, source set,
    strategy, seed, and per-band composition (synthetic count +
    origin/category/comedy-family share) so a single-family or
    synthetic-heavy band is visible in the panel record itself.
    """
    if strategy not in ("spread", "random"):
        raise ValueError(f"unknown strategy {strategy!r} (use 'spread' or 'random')")
    pool = load_canonical_pool(version)
    cfg = pool.band_preset(preset)
    layer, centering = cfg["layer"], cfg["centering"]
    names, _ = pool.matrix(layer, centering)
    name_set = set(names)
    # Fail LOUD on unknown source-set members BEFORE candidate selection - a
    # partial source set would otherwise be stamped in provenance as complete
    # (CLAUDE.md fail-fast rule, BLOCKER source-set-validation #483 round 1).
    _assert_source_set_known(names, source_set)

    candidates = [n for n in pool.panel_eligible(source_set, exclude) if n in name_set]
    by_band_all: dict[str, list[tuple[str, float]]] = {b: [] for b in cfg["band_names"]}
    for n in candidates:
        md = pool.min_dist_to_set(n, source_set, layer=layer, centering=centering)
        by_band_all[cfg["band_names"][bisect_right(cfg["edges"], md)]].append((n, md))

    rng = np.random.default_rng(seed)
    by_band: dict[str, list[str]] = {}
    for band in cfg["band_names"]:
        members = sorted(by_band_all[band], key=lambda t: (t[1], t[0]))
        if len(members) < n_per_band:
            raise BandUnsatisfiableError(band, len(members), n_per_band, source_set)
        if strategy == "spread":
            picked = [members[i][0] for i in _spread_indices(len(members), n_per_band)]
        else:
            idx = sorted(rng.choice(len(members), size=n_per_band, replace=False).tolist())
            picked = [members[i][0] for i in idx]
        by_band[band] = picked

    flat = [p for band in cfg["band_names"] for p in by_band[band]]
    overlap_src = set(flat) & set(source_set)
    overlap_exc = set(flat) & set(exclude)
    assert not overlap_src, f"panel overlaps source set: {overlap_src}"
    assert not overlap_exc, f"panel overlaps exclude list: {overlap_exc}"

    composition = {}
    for band, picked in by_band.items():
        origins: dict[str, int] = {}
        cats: dict[str, int] = {}
        for p in picked:
            e = pool.personas[p]
            origins[e.origin] = origins.get(e.origin, 0) + 1
            cats[e.category] = cats.get(e.category, 0) + 1
        composition[band] = {
            "n": len(picked),
            "synthetic_count": sum(1 for p in picked if pool.personas[p].synthetic),
            "comedy_family_share": sum(1 for p in picked if p in COMEDY_FAMILY) / len(picked),
            "origin_share": {o: c / len(picked) for o, c in origins.items()},
            "category_share": {c: k / len(picked) for c, k in cats.items()},
        }

    provenance = {
        "pool_version": pool.version,
        "preset": preset,
        "layer": layer,
        "centering": centering,
        "edges": list(cfg["edges"]),
        "band_names": list(cfg["band_names"]),
        "matrix_sha256": pool.matrix_sha256(layer, centering),
        "source_set": list(source_set),
        "exclude": list(exclude),
        "strategy": strategy,
        "seed": seed,
        "n_per_band": n_per_band,
        "per_band_composition": composition,
    }
    return HeldOutPanel(
        by_band=by_band,
        personas=flat,
        prompts={p: pool.personas[p].prompt for p in flat},
        provenance=provenance,
    )


def _anchor_cosines(anchor: str, *, layer: int, centering: str, version: str) -> dict[str, float]:
    """Similarity-signed (1 - distance) cosines of every matrix persona vs an anchor."""
    pool = load_canonical_pool(version)
    names, dist = pool.matrix(layer, centering)
    if anchor not in names:
        raise KeyError(f"anchor persona {anchor!r} not in pool {version} matrix")
    i = names.index(anchor)
    return {n: float(1.0 - dist[names.index(n), i]) for n in names if n != anchor}


def assistant_cosines(
    *, layer: int = 10, centering: str = "global_mean", version: str = "v1"
) -> dict[str, float]:
    """Pool-derived replacement for the frozen ``personas.ASSISTANT_COSINES`` dict.

    Values are bank-dependent (#536) and therefore DIFFER from the frozen
    legacy dict - never mix the two in one analysis. Keys cover every pool
    persona in the committed layer matrix except ``assistant`` itself.
    """
    return _anchor_cosines("assistant", layer=layer, centering=centering, version=version)


def doctor_cosines(
    *, layer: int = 10, centering: str = "global_mean", version: str = "v1"
) -> dict[str, float]:
    """Pool-derived replacement for the frozen ``personas.DOCTOR_COSINES`` dict (see above)."""
    return _anchor_cosines("medical_doctor", layer=layer, centering=centering, version=version)
