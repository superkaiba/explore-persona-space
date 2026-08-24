"""Unit-1 gate tests for the parametrized shared-persona paired script (#823 ext ladder).

Covers, against REAL production bodies (no mocks):
  (a) full-ratio bootstrap correctness -- ``rho_ci95`` equals the quantiles of
      draw-wise mean_excess/E_draw against an independent serial repeat-index
      oracle sharing the same rng stream; the persona-0 resample is bit-identical
      to ``paired_bootstrap``'s (strict containment of the banked numerator
      bootstrap); and the full-ratio CI DIFFERS from the fixed-denominator
      interval.
  (b) default-invocation stability -- the CURRENT script's no-args default path on
      the committed fixture reproduces the PRE-change script's output
      (tests/fixtures/issue823_paired_expected_default.json, generated from the
      pre-parametrization code at git 8a338b0e42) exactly.
  (c) negligible-E guard -- draws with E_draw < 1e-9 * E_point are excluded and
      counted, >1% excluded flips ``rho_ci95_unstable``, and an all-zero-diff arm
      yields ``rho_ci95 = None`` without crashing.
  (d) extraction parity -- ``implied_mixture_energy`` reproduces the parent
      driver's inline block bit-exactly on a synthetic gather.
  (e) additive-only invariant -- ``--full-ratio-ci`` output equals the default
      output on every pre-existing key (new fields are strictly added).
  (f) sidecar loader validation fail-louds.
  (g) degenerate zero-energy branch (#823 ext smoke attempt 4): energy == 0.0
      reports None ratios + a named verdict instead of the pre-fix
      ZeroDivisionError; wilcoxon NaN z/p (zero-variance diffs) lands as
      None + ``wilcoxon_degenerate`` instead of a bare NaN JSON token; healthy
      nonzero-energy values stay bit-equal to the pre-fix formula.

Offline; every fixture is committed FLAT under tests/fixtures/ (the .gitignore
negation ``!tests/fixtures/*.npz`` covers direct children only), so no banked
eval_results/ artifact and no sparse-cone registration is needed.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

from scripts import issue823_ladder_common as LC
from scripts import issue823_shared_persona_paired as SP

_FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _load_maker():
    """Load the fixture generator module (tests/fixtures is not a package)."""
    path = _FIXTURES / "issue823_paired_make_fixture.py"
    spec = importlib.util.spec_from_file_location("issue823_paired_make_fixture", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_main(monkeypatch, root: pathlib.Path, argv: list[str]) -> None:
    monkeypatch.setattr(SP, "repo_root", lambda: root)
    monkeypatch.setattr(SP, "git_commit", lambda root: "fixture-git-commit")
    monkeypatch.setattr(sys, "argv", ["issue823_shared_persona_paired.py", *argv])
    SP.main()


# ── (a) full-ratio bootstrap vs a serial oracle ──────────────────────────────


def _synth_groups(rng: np.random.Generator, spec, d: int) -> list[tuple[int, np.ndarray]]:
    groups = []
    for p, n_p, scale in spec:
        base = rng.normal(0.0, 1.0, size=(n_p, d)) + rng.normal(0.0, 2.0, size=(1, d)) * scale
        groups.append((p, base.astype(np.float64)))
    return groups


def _oracle(diff, groups, n_persona0, n_boot, seed):
    """Serial repeat-index oracle sharing full_ratio_bootstrap's exact rng stream."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    me = diff[idx].mean(axis=1)
    counts = {}
    for p, g in groups:  # same consumption order as full_ratio_bootstrap
        n_p = g.shape[0]
        counts[p] = rng.multinomial(n_p, np.full(n_p, 1.0 / n_p), size=n_boot)
    n_tot = n_persona0 + sum(g.shape[0] for _p, g in groups)
    e_draws = np.empty(n_boot)
    for b in range(n_boot):
        between = 0.0
        for p, g in groups:
            rep = np.repeat(np.arange(g.shape[0]), counts[p][b])
            m = g[rep].mean(axis=0)
            between += g.shape[0] * float(m @ m)
        e_draws[b] = between / n_tot
    e_point = sum(g.shape[0] * float(g.mean(axis=0) @ g.mean(axis=0)) for _p, g in groups) / n_tot
    return me, e_draws, e_point, counts


def test_full_ratio_matches_serial_oracle_and_differs_from_fixed_denominator():
    rng = np.random.default_rng(4242)
    n0, n_boot, seed, d = 12, 400, 77, 5
    diff = rng.normal(3.0, 2.0, size=n0)
    groups = _synth_groups(rng, ((1, 7, 1.0), (2, 9, 2.5), (3, 5, 0.7)), d)

    res = SP.full_ratio_bootstrap(diff, groups, n0, n_boot, seed)
    me, e_draws, e_point, _ = _oracle(diff, groups, n0, n_boot, seed)

    # Healthy fixture: no draw is negligible.
    assert (e_draws >= SP.NEGLIGIBLE_E_REL * e_point).all()
    assert res["n_negligible_E_draws"] == 0
    assert res["rho_ci95_unstable"] is False

    rho = me / e_draws
    want = [np.quantile(rho, 0.025), np.quantile(rho, 0.975)]
    assert np.allclose(res["rho_ci95"], want, rtol=1e-9, atol=0.0)
    assert np.isclose(res["full_ratio"]["e_point_from_diffs"], e_point, rtol=1e-12)

    # Strict containment: the persona-0 resample IS paired_bootstrap's index draw.
    lo, hi = SP.paired_bootstrap(diff, n_boot, seed)
    assert lo == float(np.quantile(me, 0.025)) and hi == float(np.quantile(me, 0.975))

    # And the full-ratio CI genuinely differs from the fixed-denominator interval.
    fixed = [lo / e_point, hi / e_point]
    assert not np.allclose(res["rho_ci95"], fixed, rtol=1e-3)


# ── (c) negligible-E guard ───────────────────────────────────────────────────


def test_negligible_e_guard_counts_excludes_and_flags():
    rng = np.random.default_rng(99)
    n0, n_boot, seed, d = 10, 400, 55, 4
    diff = rng.normal(1.0, 0.5, size=n0)
    # All-zero rows except one heavy row: E_draw == 0 exactly when the heavy row is
    # not drawn -- P((7/8)^8) ~ 0.34 of draws, far above the 1% unstable threshold.
    g1 = np.zeros((8, d))
    g1[0] = 25.0
    res = SP.full_ratio_bootstrap(diff, [(1, g1)], n0, n_boot, seed)

    # Expected exclusions, from the same rng stream (idx draw first, then counts).
    orng = np.random.default_rng(seed)
    orng.integers(0, n0, size=(n_boot, n0))
    counts = orng.multinomial(8, np.full(8, 1.0 / 8), size=n_boot)
    n_expected = int((counts[:, 0] == 0).sum())

    assert res["n_negligible_E_draws"] == n_expected
    assert n_expected > SP.UNSTABLE_FRAC * n_boot
    assert res["rho_ci95_unstable"] is True
    assert res["rho_ci95"] is not None and len(res["rho_ci95"]) == 2
    assert res["full_ratio"]["n_draws_retained"] == n_boot - n_expected


def test_negligible_e_guard_all_zero_diffs_never_crashes():
    diff = np.ones(6)
    res = SP.full_ratio_bootstrap(diff, [(1, np.zeros((5, 3)))], 6, 200, 7)
    assert res["rho_ci95"] is None
    assert res["n_negligible_E_draws"] == 200
    assert res["rho_ci95_unstable"] is True
    assert res["full_ratio"]["rho_point"] is None


# ── (d) extraction parity with the parent driver's inline block ──────────────


def test_implied_mixture_energy_matches_inline_parent_block():
    rng = np.random.default_rng(11)
    n_layers, d = 3, 4
    mask_ids = np.array([10, 11, 12, 13, 14, 15])
    store_ctx0 = np.array([10, 11, 12, 13, 14, 15])
    store_v = {
        0: rng.normal(size=(6, n_layers, d)).astype(np.float32),
        1: rng.normal(size=(2, n_layers, d)).astype(np.float32),
        2: rng.normal(size=(3, n_layers, d)).astype(np.float32),
    }
    gather = [
        (0, np.array([0]), np.array([0])),
        (1, np.array([1, 3]), np.array([0, 1])),
        (2, np.array([2, 4, 5]), np.array([0, 1, 2])),
    ]
    layer = 1
    got = LC.implied_mixture_energy(gather, layer, store_v, store_ctx0, mask_ids)

    # Inline replica of scripts/issue823_ladder_fits.py:2035-2053 (pre-extraction shape).
    between, n_tot = 0.0, 0
    row0 = {int(c): j for j, c in enumerate(store_ctx0)}
    for p, pos, rows in gather:
        if p == 0:
            n_tot += len(pos)
            continue
        ctxs = [int(mask_ids[q]) for q in pos]
        vp = store_v[p][rows, layer, :].astype(np.float64)
        v0 = store_v[0][np.array([row0[c] for c in ctxs]), layer, :].astype(np.float64)
        m_p = (vp - v0).mean(axis=0)
        between += len(pos) * float(m_p @ m_p)
        n_tot += len(pos)
    want = between / max(n_tot, 1)

    assert got == want  # bit-exact: same ops, same order


# ── (b) default-invocation stability against the pre-change output ───────────


def test_default_invocation_reproduces_prechange_output(tmp_path, monkeypatch):
    maker = _load_maker()
    root = maker.assemble_fixture_repo(tmp_path / "repo")
    _run_main(monkeypatch, root, [])  # TRUE default path: no args at all
    out = root / maker.LADDER_REL / "shared_persona_paired.json"
    got = json.loads(out.read_text())
    expected = json.loads((_FIXTURES / "issue823_paired_expected_default.json").read_text())
    assert got == expected


# ── (e) --full-ratio-ci is strictly additive over the default output ─────────


def test_full_ratio_flag_is_strictly_additive(tmp_path, monkeypatch):
    maker = _load_maker()
    root = maker.assemble_fixture_repo(tmp_path / "repo")
    out_def = tmp_path / "default.json"
    out_fr = tmp_path / "fullratio.json"
    _run_main(monkeypatch, root, ["--out", str(out_def)])
    _run_main(monkeypatch, root, ["--full-ratio-ci", "--out", str(out_fr)])
    d = json.loads(out_def.read_text())
    f = json.loads(out_fr.read_text())

    meta_f = dict(f["metadata"])
    fr_meta = meta_f.pop("full_ratio_ci")
    assert meta_f == d["metadata"]
    assert fr_meta["mixture_diffs"].endswith("mixture_diffs.npz")

    for arm_key, arm in d["arms"].items():
        assert (
            f["arms"][arm_key]["n_shared_contexts_post_mask"] == arm["n_shared_contexts_post_mask"]
        )
        for layer_key, cell in arm["per_layer"].items():
            fcell = f["arms"][arm_key]["per_layer"][layer_key]
            for key, val in cell.items():  # every pre-existing field unchanged
                assert fcell[key] == val, (arm_key, layer_key, key)
            assert isinstance(fcell["n_negligible_E_draws"], int)
            assert isinstance(fcell["rho_ci95_unstable"], bool)
            ci = fcell["rho_ci95"]
            assert ci is None or (len(ci) == 2 and ci[0] <= ci[1])
            assert fcell["full_ratio"]["n_boot"] == 10_000


# ── (g) degenerate zero-energy branch (#823 ext smoke crash, round v20) ───────


def _zero_energy_fixture(dest: pathlib.Path, maker) -> pathlib.Path:
    """An ALL-SHARED-population ladder fixture realizing energy == 0.0 exactly.

    Mirrors the ext smoke's realized degenerate shape (smoke attempt 4, rc=1 at
    scripts/issue823_shared_persona_paired.py:278): every masked context is
    shared-persona (i % 16 == 0), so the k16 arm's training targets equal k1's
    (identical fits -> zero paired diffs -> wilcoxon se == 0) and the diff group
    population carries zero persona != 0 rows, making
    ``mixture_energy_from_group_diffs`` return exactly 0.0.
    """
    n_shared, n_layers, d = 30, 28, 6
    ladder = dest / maker.LADDER_REL
    ladder.mkdir(parents=True, exist_ok=True)
    context_ids = (np.arange(n_shared) * 16).astype(np.int64)  # all i % 16 == 0
    n_orig = 16 * n_shared
    rng = np.random.default_rng(823)
    base_res = rng.uniform(50.0, 150.0, size=(n_layers, n_shared))
    ss_res = np.stack([base_res, base_res.copy()])  # k16 rows == k1 rows exactly
    ss_tot = ss_res + rng.uniform(200.0, 400.0, size=ss_res.shape)
    np.savez(
        ladder / "percontext_ladder.npz",
        arm_names=np.array(["k1", "k16"]),
        context_ids=context_ids,
        p1_ss_res=ss_res,
        p1_ss_tot=ss_tot,
    )
    arms = {"1": [0] * n_orig, "16": [i % 16 for i in range(n_orig)]}
    (ladder / "assignment.json").write_text(
        json.dumps({"arms": arms, "registered_rule": "persona(i, k) = i mod k"}) + "\n"
    )
    implied = {
        f"k16:L{layer}": {"between_persona_mean_shift_energy": 0.0} for layer in (14, 26, 17)
    }
    (ladder / "ladder_analysis_summary.json").write_text(
        json.dumps({"mixture_floor": {"implied_mixture_penalty": implied}}) + "\n"
    )
    np.savez(
        ladder / "mixture_diffs.npz",
        layers=np.array([14, 26, 17], dtype=np.int64),
        k16_diffs=np.zeros((0, 3, d)),
        k16_personas=np.zeros(0, dtype=np.int64),
        k16_n_persona0=np.int64(n_shared),
        k16_context_ids=np.zeros(0, dtype=np.int64),
    )
    return dest


def test_zero_energy_smoke_shape_end_to_end(tmp_path, monkeypatch):
    """Red-pre/green-post for the smoke-attempt-4 ZeroDivisionError: the exact
    crashing argv shape (--arms 16 --n-boot 200 --full-ratio-ci) completes and
    reports the degeneracy truthfully (None ratios + named verdict; NaN-safe
    wilcoxon; rho_ci95 None) — pre-fix this raised ZeroDivisionError at
    offset_bias_control (float division by zero)."""
    maker = _load_maker()
    root = _zero_energy_fixture(tmp_path / "repo", maker)
    out = tmp_path / "paired.json"
    _run_main(
        monkeypatch, root, ["--arms", "16", "--n-boot", "200", "--full-ratio-ci", "--out", str(out)]
    )
    raw = out.read_text()
    got = json.loads(raw)
    for layer in (14, 26, 17):
        cell = got["arms"]["k16"]["per_layer"][f"L{layer}"]
        o = cell["offset_bias_control"]
        assert o["verdict"] == SP.DEGENERATE_OFFSET_VERDICT
        assert o["ratio_measured_over_offset_only_prediction"] is None
        assert o["ratio_measured_over_full_energy"] is None
        assert o["between_persona_mean_shift_energy"] == 0.0
        assert o["measured_excess"] == 0.0  # real float, reported verbatim
        # zero-variance wilcoxon at n=30 (approx path, se == 0): None + named flag
        assert cell["wilcoxon_p"] is None
        assert cell["wilcoxon_degenerate"] is True
        # empty persona != 0 groups: every full-ratio draw excluded, CI None
        assert cell["rho_ci95"] is None
        assert cell["n_negligible_E_draws"] == 200
    # the persisted JSON is STRICT (no bare NaN/Infinity tokens survive)
    json.loads(raw, parse_constant=lambda s: pytest.fail(f"non-strict JSON token {s}"))


def test_offset_bias_control_zero_energy_degenerate_unit():
    """Unit red-pre/green-post: pre-fix offset_bias_control(0.0, 16, *) raised
    ZeroDivisionError (line 278). No separate sub-case for measured_excess == 0:
    the control is uninformative at E == 0 regardless (both hypotheses predict
    zero excess), and measured_excess stays visible as a real float."""
    for me in (0.0, 3.7):
        r = SP.offset_bias_control(0.0, 16, me)
        assert r["verdict"] == SP.DEGENERATE_OFFSET_VERDICT
        assert r["ratio_measured_over_offset_only_prediction"] is None
        assert r["ratio_measured_over_full_energy"] is None
        assert r["between_persona_mean_shift_energy"] == 0.0
        assert r["measured_excess"] == me
        assert r["predicted_excess_if_shared_map_offset_only"] == 0.0
    # subnormal energy whose energy/k underflows to 0.0 routes to the same branch
    assert SP.offset_bias_control(5e-324, 16, 1.0)["verdict"] == SP.DEGENERATE_OFFSET_VERDICT


def test_offset_bias_control_rejects_negative_and_nan_energy():
    with pytest.raises(ValueError, match=">= 0"):
        SP.offset_bias_control(-1.0, 16, 0.0)
    with pytest.raises(ValueError, match=">= 0"):
        SP.offset_bias_control(float("nan"), 16, 0.0)


def test_offset_bias_control_nonzero_energy_exact_prefix_parity():
    """No-regression pin: healthy-path values bit-equal the pre-fix formula
    (verbatim pre-fix body, commit 809b06d14a) — computed at test time on the
    same machine, so no cross-machine float-last-bit exposure."""

    def prefix_reference(energy: float, k: int, measured_excess: float) -> dict:
        predicted_offset_only = energy / k
        ratio_vs_offset_only = measured_excess / predicted_offset_only
        ratio_vs_full_energy = measured_excess / energy
        if ratio_vs_offset_only < 2.0:
            verdict = "consistent-with-shared-map-offset"
        elif ratio_vs_full_energy > 0.5:
            verdict = "excess-tracks-full-between-persona-energy"
        else:
            verdict = "intermediate"
        return {
            "between_persona_mean_shift_energy": float(energy),
            "predicted_excess_if_shared_map_offset_only": float(predicted_offset_only),
            "measured_excess": float(measured_excess),
            "ratio_measured_over_offset_only_prediction": float(ratio_vs_offset_only),
            "ratio_measured_over_full_energy": float(ratio_vs_full_energy),
            "verdict": verdict,
        }

    # one triple per verdict class + a boundary-adjacent one
    for e, k, me in ((5.1, 16, 0.3), (7.4, 16, 6.0), (10.0, 4, 6.1), (8.2, 2, 1.0)):
        assert SP.offset_bias_control(e, k, me) == prefix_reference(e, k, me), (e, k, me)


# ── (f) sidecar loader validation ────────────────────────────────────────────


def test_load_mixture_diffs_fail_louds(tmp_path):
    with pytest.raises(FileNotFoundError):
        SP.load_mixture_diffs(tmp_path / "absent.npz", (2,), (14,))

    p0 = tmp_path / "persona0.npz"
    np.savez(p0, layers=np.array([14]), k2_diffs=np.zeros((2, 1, 3)), k2_personas=np.array([0, 1]))
    with pytest.raises(ValueError, match="persona != 0"):
        SP.load_mixture_diffs(p0, (2,), (14,))

    p1 = tmp_path / "missing_layer.npz"
    np.savez(p1, layers=np.array([14]), k2_diffs=np.zeros((2, 1, 3)), k2_personas=np.array([1, 1]))
    with pytest.raises(ValueError, match="read-out layers"):
        SP.load_mixture_diffs(p1, (2,), (14, 26, 17))

    p2 = tmp_path / "missing_arm.npz"
    np.savez(p2, layers=np.array([14]), k2_diffs=np.zeros((2, 1, 3)), k2_personas=np.array([1, 1]))
    with pytest.raises(ValueError, match="arrays for arm k=4"):
        SP.load_mixture_diffs(p2, (2, 4), (14,))

    p3 = tmp_path / "nonfinite.npz"
    bad = np.full((2, 1, 3), np.nan)
    np.savez(p3, layers=np.array([14]), k2_diffs=bad, k2_personas=np.array([1, 1]))
    with pytest.raises(ValueError, match="non-finite"):
        SP.load_mixture_diffs(p3, (2,), (14,))
