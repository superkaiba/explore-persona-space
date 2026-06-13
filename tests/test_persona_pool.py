# intentional unicode in prose
"""Acceptance tests for the #483 canonical persona pool (plan v3 §5).

CPU-only, committed artifacts only. Pre-GPU (before the #483 build lands its
artifacts under ``data/canonical_persona_pool/``) every artifact-dependent
test SKIPS with a clear reason, so the repo suite stays green; once the
artifacts are committed the tests are strict. ``test_legacy_shims_frozen``
needs no artifacts and always runs.

The #478 constants (POOL_16, HELD_OUT_BANDS, band edges, comedy family) are
INLINED here independently of the build script, pinned to
``scripts/_issue478_common.py`` @ commit 69b34b94e00cf4c16b830f32da605289ef35371c
- the regression test re-derives its expected memberships from the committed
legacy matrix + these constants, NEVER from the gitignored local
``data/issue_478/design_validation_v2.json``.

Point the suite at smoke artifacts with ``EPM_CANONICAL_POOL_DIR=<dir>``.
"""

from __future__ import annotations

import json
import os
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pytest

# ── Inlined #478 constants (commit 69b34b94e) ───────────────────────────────

POOL_16 = [
    "librarian_detective",
    "librarian",
    "social_worker",
    "archivist",
    "data_journalist",
    "pharmacist",
    "museum_curator",
    "security_guard",
    "nurse",
    "chief_of_medicine",
    "data_scientist",
    "debate_coach",
    "journalist",
    "game_designer",
    "cto",
    "police_officer",
]
HELD_OUT_BANDS = {
    "near": [
        "medical_doctor",
        "assistant",
        "web_developer",
        "devops_engineer",
        "machine_learning_engineer",
        "medical_student",
    ],
    "near-mid": [
        "french_person",
        "zelthari_scholar",
        "elementary_teacher",
        "perfectionist_engineer",
        "strict_teacher",
        "caring_villain",
    ],
    "mid": [
        "villain",
        "nice_villain",
        "wholesome_comedian",
        "lazy_software_engineer",
        "overly_enthusiastic_assistant",
        "stoic_philosopher",
    ],
    "far": [
        "improv_comedian",
        "satirist",
        "incompetent_villain",
        "hippie_teacher",
        "misanthrope",
        "brazilian_comedian",
    ],
    "very-far": [
        "comedian",
        "dark_comedian",
        "open_mic_comedian",
        "doctor_comedian",
        "joker",
    ],
    "tail": [
        "serious_comedian",
        "sarcastic_assistant",
        "formal_assistant",
        "drill_sergeant",
        "grumpy_person",
        "mysterious_person",
    ],
}
HELD_OUT_35 = [p for band in HELD_OUT_BANDS.values() for p in band]
RAW478_EDGES = [0.05, 0.10, 0.15, 0.20, 0.25]
BAND_NAMES = ["near", "near-mid", "mid", "far", "very-far", "tail"]
SENTINELS = ("helpful_assistant", "no_persona")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pool_dir() -> Path:
    override = os.environ.get("EPM_CANONICAL_POOL_DIR")
    return Path(override) if override else REPO_ROOT / "data" / "canonical_persona_pool"


def _band_index(md: float) -> int:
    return bisect_right(RAW478_EDGES, md)


def _load_dist(path: Path) -> tuple[list[str], np.ndarray]:
    data = json.loads(path.read_text())
    key = "distance" if "distance" in data else "matrix"
    if key == "matrix":
        assert data.get("metric") == "1 - cosine", f"{path} 'matrix' key without distance metric"
    return data["persona_names"], np.asarray(data[key], dtype=np.float64)


def _min_dist(persona: str, source_set: list[str], names: list[str], dist: np.ndarray) -> float:
    idx = {n: i for i, n in enumerate(names)}
    return float(min(dist[idx[persona], idx[s]] for s in source_set if s in idx))


# ── Skip-clean gating fixture (strict once artifacts are committed) ──────────


@pytest.fixture(scope="module")
def pool_dir() -> Path:
    d = _pool_dir()
    required = [
        d / "pool_v1.json",
        d / "pool_meta_v1.json",
        d / "matrix_v1_L20_raw.json",
        d / "matrix_v1_L20_centered.json",
        d / "matrix_v1_L10_centered.json",
        d / "legacy" / "cosine_distance_matrix_layer20_478.json",
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        pytest.skip(
            f"canonical pool artifacts not yet built (pre-GPU #483): missing {missing} "
            f"under {d} - tests are strict once the build commits them"
        )
    return d


@pytest.fixture(scope="module")
def pool_data(pool_dir: Path) -> dict:
    return json.loads((pool_dir / "pool_v1.json").read_text())


@pytest.fixture(scope="module")
def pool_meta(pool_dir: Path) -> dict:
    return json.loads((pool_dir / "pool_meta_v1.json").read_text())


def _panel_eligible(pool_data: dict, source_set: list[str]) -> list[str]:
    blocked = set(source_set)
    return [
        n
        for n, e in pool_data["personas"].items()
        if not e.get("sentinel", False) and n not in blocked
    ]


# ── 1. Pool + matrix invariants ──────────────────────────────────────────────


def test_pool_loads_and_matrix_invariants(pool_dir: Path, pool_data: dict, pool_meta: dict):
    """Matrices symmetric, diag ~0, values in [0, 2]; names <-> pool membership bijective."""
    from explore_persona_space.persona_pool import load_canonical_pool

    pool = load_canonical_pool("v1")
    assert pool.version == "v1"
    pool_names = set(pool_data["personas"])
    assert set(pool.personas) == pool_names

    committed = [
        "matrix_v1_L10_centered.json",
        "matrix_v1_L10_raw.json",
        "matrix_v1_L20_centered.json",
        "matrix_v1_L20_raw.json",
    ]
    # The full committed set (L21 + respmean pairs) is required for a full build;
    # legacy-derived smoke artifacts (no respmean generation stats) carry L10/L20 only.
    if pool_meta["build"].get("respmean_generation_stats") is not None:
        committed += [
            "matrix_v1_L21_centered.json",
            "matrix_v1_L21_raw.json",
            "matrix_v1_respmean_L20_centered.json",
            "matrix_v1_respmean_L20_raw.json",
            "matrix_v1_respmean_L21_centered.json",
            "matrix_v1_respmean_L21_raw.json",
        ]
    for fname in committed:
        path = pool_dir / fname
        assert path.exists(), f"committed matrix missing: {fname}"
        data = json.loads(path.read_text())
        names, dist = data["persona_names"], np.asarray(data["distance"])
        assert set(names) == pool_names, f"{fname}: matrix names != pool membership"
        assert np.allclose(dist, dist.T, atol=1e-8), f"{fname}: not symmetric"
        assert np.abs(np.diag(dist)).max() < 1e-6, f"{fname}: diagonal not ~0"
        assert dist.min() >= -1e-9 and dist.max() <= 2.0 + 1e-9, f"{fname}: values outside [0,2]"
        expected_centering = "global_mean" if fname.endswith("_centered.json") else "none"
        assert data["centering"] == expected_centering, f"{fname}: centering provenance wrong"

    # Canonical path asserts global_mean provenance (via the loader).
    names, _ = pool.matrix(20, "global_mean")
    assert set(names) == pool_names


# ── 2. AC2 band coverage (precedence-pinned) ─────────────────────────────────


def _floor_ok(counts: list[int]) -> tuple[bool, list[str]]:
    """AC2 floor on 6 bins: first four >=4; top two >=2 each with union >=4."""
    bad = [BAND_NAMES[i] for i in range(4) if counts[i] < 4]
    if counts[4] < 2 or counts[4] + counts[5] < 4:
        bad.append("very-far")
    if counts[5] < 2 or counts[4] + counts[5] < 4:
        bad.append("tail")
    return not bad, bad


def test_band_coverage_acceptance(pool_dir: Path, pool_data: dict, pool_meta: dict):
    """AC2: 6 unmerged raw478 bins over panel-eligible personas, both legs."""
    names, dist = _load_dist(pool_dir / "matrix_v1_L20_raw.json")

    def counts_vs(source_set: list[str]) -> list[int]:
        counts = [0] * 6
        for n in _panel_eligible(pool_data, source_set):
            counts[_band_index(_min_dist(n, source_set, names, dist))] += 1
        return counts

    # Hard-binding leg: vs POOL_16.
    p_counts = counts_vs(POOL_16)
    ok, bad = _floor_ok(p_counts)
    assert ok, (
        f"vs POOL_16 occupancy floor failed in bands {bad}: "
        f"{dict(zip(BAND_NAMES, p_counts, strict=True))}"
    )

    # vs-assistant leg: floor OR documented deficit (K2 - same committed rule the build ships).
    a_counts = counts_vs(["assistant"])
    ok, bad = _floor_ok(a_counts)
    if not ok:
        documented = {
            d["band"] for d in pool_meta["documented_deficits"] if d["ref_set"] == "assistant"
        }
        undocumented = [b for b in bad if b not in documented]
        assert not undocumented, (
            f"vs assistant occupancy floor failed in bands {bad}; undocumented (not in "
            f"pool_meta documented_deficits): {undocumented}; counts "
            f"{dict(zip(BAND_NAMES, a_counts, strict=True))}"
        )


# ── 3. AC4 default-surface contract ──────────────────────────────────────────


def test_default_panel_contract(pool_dir: Path):
    """held_out_panel under its PUBLIC DEFAULTS returns full 6x5 panels for both source sets."""
    from explore_persona_space.persona_pool import held_out_panel

    for source_set in (POOL_16, ["assistant"]):
        panel = held_out_panel(source_set)  # ALL defaults: centered_v1_L20, n_per_band=5
        assert list(panel.by_band) == BAND_NAMES
        for band, members in panel.by_band.items():
            assert len(members) == 5, f"band {band} has {len(members)} != 5 personas"
        assert len(panel.personas) == 30
        assert not set(panel.personas) & set(source_set)
        assert not set(panel.personas) & set(SENTINELS)
        assert panel.provenance["preset"] == "centered_v1_L20"


# ── 4. #478 regression (edge-churn-coherent) ─────────────────────────────────


def test_478_band_regression(pool_dir: Path):
    """<=3 genuine-drift band shifts vs the committed legacy matrix; no multi-band jumps.

    Expected memberships are RE-DERIVED from the committed legacy matrix +
    the pinned constants (commit 69b34b94e) - NOT read from the gitignored
    design_validation_v2.json. Genuine drift = band change with
    |delta min-dist| > 0.01; edge churn (<= 0.01) is exempt (15/35 panel
    personas sit within 0.01 of an edge - plan §3.4).
    """
    l_names, l_dist = _load_dist(pool_dir / "legacy" / "cosine_distance_matrix_layer20_478.json")
    f_names, f_dist = _load_dist(pool_dir / "matrix_v1_L20_raw.json")

    genuine, jumps, churn = [], [], []
    for persona in HELD_OUT_35:
        md_l = _min_dist(persona, POOL_16, l_names, l_dist)
        md_f = _min_dist(persona, POOL_16, f_names, f_dist)
        bi_l, bi_f = _band_index(md_l), _band_index(md_f)
        if bi_l == bi_f:
            continue
        rec = (persona, BAND_NAMES[bi_l], BAND_NAMES[bi_f], round(abs(md_f - md_l), 5))
        if abs(bi_f - bi_l) >= 2:
            jumps.append(rec)
        elif abs(md_f - md_l) > 0.01:
            genuine.append(rec)
        else:
            churn.append(rec)

    assert not jumps, f"multi-band jumps (always count, K3): {jumps}"
    assert len(genuine) <= 3, (
        f"{len(genuine)} genuine-drift band shifts (> 3, K3): {genuine}; "
        f"edge-churn (exempt): {churn}"
    )


# ── 5. Centered edges exactly regenerable ────────────────────────────────────


def test_centered_edges_regenerable(pool_dir: Path, pool_data: dict, pool_meta: dict):
    """Re-derive centered_v1_L20 edges from the committed matrices via the pinned mapping."""
    r_names, r_dist = _load_dist(pool_dir / "matrix_v1_L20_raw.json")
    c_names, c_dist = _load_dist(pool_dir / "matrix_v1_L20_centered.json")

    eligible = [n for n in _panel_eligible(pool_data, POOL_16) if n in r_names]
    md_raw = np.sort([_min_dist(n, POOL_16, r_names, r_dist) for n in eligible])
    md_cen = np.sort([_min_dist(n, POOL_16, c_names, c_dist) for n in eligible])

    probs = np.linspace(0.0, 1.0, len(md_raw))
    qs = np.interp(RAW478_EDGES, md_raw, probs)
    rederived = np.interp(qs, probs, md_cen)

    fit = pool_meta["band_presets"]["centered_v1_L20"]["fit"]
    assert fit["method"] == "empirical_quantile_linear_interp"
    np.testing.assert_allclose(md_raw, fit["raw_min_dists_sorted"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(md_cen, fit["centered_min_dists_sorted"], rtol=0, atol=1e-12)
    committed_edges = pool_meta["band_presets"]["centered_v1_L20"]["edges"]
    np.testing.assert_allclose(rederived, committed_edges, rtol=0, atol=1e-12)


# ── 6. Gate status fields ────────────────────────────────────────────────────


def test_gate_status_fields(pool_meta: dict):
    """pool_meta carries structured pass/fail gate fields and both blocking gates read pass."""
    gates = pool_meta["gates"]
    stability = gates["stability"]
    assert stability["pass"] is True, f"stability gate failed in committed artifact: {stability}"
    assert "p95_abs_delta" in stability and stability["threshold"] == 0.01
    regression = gates["regression_478"]
    assert regression["pass"] is True, f"#478 regression failed in committed artifact: {regression}"
    assert "n_genuine" in regression and "edge_churn_shifts" in regression
    assert "determinism_floor" in gates  # recorded (non-blocking)


# ── 7. Panel determinism + disjointness + fail-loud ──────────────────────────


def test_panel_determinism_and_disjointness(pool_dir: Path):
    from explore_persona_space.persona_pool import BandUnsatisfiableError, held_out_panel

    p1 = held_out_panel(POOL_16)
    p2 = held_out_panel(POOL_16)
    assert p1.by_band == p2.by_band, "default panel not deterministic"

    r1 = held_out_panel(POOL_16, strategy="random", seed=7)
    r2 = held_out_panel(POOL_16, strategy="random", seed=7)
    assert r1.by_band == r2.by_band, "seeded random panel not reproducible"

    assert not set(p1.personas) & set(POOL_16)

    victim = p1.by_band["near"][0]
    p3 = held_out_panel(POOL_16, exclude=[victim])
    assert victim not in p3.personas, "exclude not honored"

    with pytest.raises(BandUnsatisfiableError):
        held_out_panel(POOL_16, n_per_band=50)


# ── 8. Provenance stamp ──────────────────────────────────────────────────────


def test_provenance_stamp(pool_dir: Path, pool_meta: dict):
    import hashlib

    from explore_persona_space.persona_pool import held_out_panel

    panel = held_out_panel(POOL_16)
    prov = panel.provenance
    assert prov["pool_version"] == "v1"
    assert prov["source_set"] == POOL_16
    assert prov["edges"] == pool_meta["band_presets"]["centered_v1_L20"]["edges"]
    expected_sha = hashlib.sha256(
        (pool_dir / "matrix_v1_L20_centered.json").read_bytes()
    ).hexdigest()
    assert prov["matrix_sha256"] == expected_sha
    for band in BAND_NAMES:
        comp = prov["per_band_composition"][band]
        assert {"synthetic_count", "origin_share", "comedy_family_share"} <= set(comp)


# ── 9. Legacy shims frozen (always runs - no artifacts needed) ───────────────


def test_legacy_shims_frozen():
    """The deprecated personas.py dicts keep their replay-frozen values (plan §3.7/§4)."""
    from explore_persona_space.personas import ASSISTANT_COSINES, DOCTOR_COSINES

    assert ASSISTANT_COSINES["software_engineer"] == +0.446
    assert ASSISTANT_COSINES["zelthari_scholar"] == -0.379
    assert ASSISTANT_COSINES["police_officer"] == -0.399
    assert len(ASSISTANT_COSINES) == 10
    assert DOCTOR_COSINES["villain"] == -0.422
    assert DOCTOR_COSINES["helpful_assistant"] == 0.054
    assert len(DOCTOR_COSINES) == 9


# ── 10. Synthetic landings recorded ──────────────────────────────────────────


def test_synthetic_landings_recorded(pool_data: dict, pool_meta: dict):
    """Every kept synthetic carries target/landed bands + measured min-dists (AC5)."""
    synth_meta = pool_meta["synthetics"]
    assert {"kept", "rejected"} <= set(synth_meta)

    rejected_names = {r["name"] for r in synth_meta["rejected"]}
    pool_names = set(pool_data["personas"])
    assert not rejected_names & pool_names, (
        f"rejected synthetics present in pool: {rejected_names & pool_names}"
    )

    for name, entry in pool_data["personas"].items():
        if not entry.get("synthetic"):
            continue
        for key in (
            "target_band",
            "landed_band_pool16_raw478",
            "landed_band_centered_v1",
            "min_dist_pool16_raw",
            "min_dist_pool16_centered",
        ):
            assert key in entry, f"kept synthetic {name} missing {key!r}"
        assert name in synth_meta["kept"]
