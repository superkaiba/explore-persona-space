# ruff: noqa: RUF001, RUF002, RUF003
"""Unit tests for ``scripts/aggregate_issue228.py``.

We test the aggregator's correlation logic on synthetic 7x11 matrices with
known structure, the cluster-bootstrap helper, and the self-loop exclusion
logic. No GPU / no network dependencies.

Round-2 additions (BLOCKER fixes from code-review v1):
* ``test_h1_long_n71_with_nurse_source_real_js`` — verifies the n=71 cell
  contract from the user's pre-registration: 6 ALL_EVAL sources × 10
  off-diagonal targets + 1 nurse source × 11 targets = 71 directed cells,
  with NO None js values reaching downstream Spearman.
* ``test_h1_long_raises_when_nurse_state_missing_extra_row`` — round-1's
  silent js=None failure mode is now a hard SystemExit.
* ``test_marker_mask_positions_substring`` — substring-position marker
  mask only flags ``[ZLT]`` token spans, not generic newlines or ``]`` tokens.
* ``test_sanity_check_142_fails_on_rank_inversion`` — sanity 6.1#4 fails
  when rank order is broken even if MRE is tiny.
* ``test_spearman_xy_handles_production_none_pattern`` — exercise the
  production None-mixed shape that round-1 tests didn't cover.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make ``scripts/`` importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate_issue228 as agg  # noqa: E402

ADAPTER_SOURCES = agg.ADAPTER_SOURCES  # 7 sources, alphabetical
TARGETS = list(agg.TARGET_PERSONA_ORDER)  # 11 targets
ALL_EVAL_PERSONAS = agg.ALL_EVAL_PERSONAS


# ──────────────────────────────────────────────────────────────────────────
# Self-loop exclusion: H1 long-format has exactly 70 rows, no diagonal.
# ──────────────────────────────────────────────────────────────────────────


def _synthesize_h1_long(
    *,
    rho_target: float = -0.7,
    seed: int = 0,
) -> list[dict]:
    """Build a synthetic H1 long-format with controlled JS/leakage correlation.

    For every (source != target) cell we draw js, then set leakage as a
    monotone-decreasing transform of js plus noise sized to give Spearman
    rho ≈ ``rho_target`` in the limit. cosine_l15 / l20 / l25 are independent.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for source in ADAPTER_SOURCES:
        for target in TARGETS:
            if source == target:
                continue
            js = float(rng.uniform(0.0, 0.5))
            noise = float(rng.normal(scale=0.1))
            # rho_target = -0.7 -> mostly anti-monotone with some noise
            leak = max(0.0, min(1.0, 0.5 - rho_target * (-1.0) * js + noise))
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "is_self_loop": False,
                    "js": js,
                    "js_masked": js + float(rng.normal(scale=0.01)),
                    "js_no_marker": js + float(rng.normal(scale=0.01)),
                    "cosine_l15": float(rng.uniform(-0.5, 0.5)),
                    "cosine_l20": float(rng.uniform(-0.5, 0.5)),
                    "cosine_l25": float(rng.uniform(-0.5, 0.5)),
                    "source_token_entropy_mean": float(rng.uniform(1.0, 5.0)),
                    "leakage": leak,
                }
            )
    return rows


def test_h1_long_off_diagonal_count() -> None:
    """Off-diagonal cell count: n=71 directed off-diagonal cells.

    User's pre-registration (issue #228, ``epm:user-decision-n71``): n=71 =
    6 ALL_EVAL sources × 10 off-diagonal targets + 1 nurse source × 11
    targets. nurse is in ADAPTER_SOURCES but NOT in ALL_EVAL_PERSONAS, so it
    has no self-loop on the eval-persona target list — every ALL_EVAL
    persona is a valid target for nurse, which is why nurse contributes 11
    cells instead of 10.

    The synthetic builder mirrors that structure so this assertion lines up
    with the production aggregator's ``_build_h1_long``.
    """
    rows = _synthesize_h1_long()
    n_self_loops_present = sum(1 for s in ADAPTER_SOURCES if s in TARGETS)
    expected = len(ADAPTER_SOURCES) * len(TARGETS) - n_self_loops_present
    assert expected == 71, f"Expected n=71 by user's pre-registration, got {expected}"
    assert len(rows) == expected
    for r in rows:
        assert r["source"] != r["target"]
        assert r["is_self_loop"] is False


# ──────────────────────────────────────────────────────────────────────────
# _spearman_xy: drops None / NaN; reports correct n.
# ──────────────────────────────────────────────────────────────────────────


def test_spearman_drops_none_and_nan() -> None:
    rho, _p, n = agg._spearman_xy([1, 2, 3, None, 5], [1, 2, 3, 4, 5])
    assert n == 4
    assert not math.isnan(rho)
    _rho2, _p2, n2 = agg._spearman_xy([1.0, 2.0, float("nan"), 4.0], [1.0, 2.0, 3.0, 4.0])
    assert n2 == 3


def test_spearman_returns_nan_for_empty_or_too_short() -> None:
    rho, _p, n = agg._spearman_xy([], [])
    assert math.isnan(rho)
    assert n == 0
    rho, _p, n = agg._spearman_xy([1.0], [2.0])
    assert math.isnan(rho)
    assert n == 1


def test_spearman_recovers_known_strong_negative() -> None:
    rng = np.random.default_rng(123)
    x = rng.uniform(0, 1, size=200)
    y = -2.0 * x + rng.normal(scale=0.05, size=200)
    rho, _p, n = agg._spearman_xy(x.tolist(), y.tolist())
    assert n == 200
    assert rho < -0.9


# ──────────────────────────────────────────────────────────────────────────
# Cluster bootstrap: structure of returned dict, point estimate matches scipy.
# ──────────────────────────────────────────────────────────────────────────


def test_cluster_bootstrap_returns_complete_dict() -> None:
    rows = _synthesize_h1_long(seed=42)
    out = agg.cluster_bootstrap_spearman(rows, "js", "leakage", n_iter=50, seed=1)
    for k in (
        "rho",
        "p_value",
        "n",
        "ci_low",
        "ci_high",
        "n_iter",
        "n_clusters",
    ):
        assert k in out, f"missing key {k}"
    # Cell count = sources x targets - actual_self_loops (see test_h1_long_off_diagonal_count)
    expected_n = len(ADAPTER_SOURCES) * len(TARGETS) - sum(
        1 for s in ADAPTER_SOURCES if s in TARGETS
    )
    assert out["n"] == expected_n
    assert out["n_clusters"] == 7
    assert out["n_iter"] == 50
    assert -1.0 <= out["rho"] <= 1.0
    # Point estimate must match scipy on the full set.
    rho_ref, _, _ = agg._spearman_xy([r["js"] for r in rows], [r["leakage"] for r in rows])
    assert math.isclose(out["rho"], rho_ref, rel_tol=1e-9)


def test_cluster_bootstrap_ci_brackets_point_estimate_for_strong_signal() -> None:
    rng = np.random.default_rng(2026)
    rows: list[dict] = []
    for source in ADAPTER_SOURCES:
        for target in TARGETS:
            if source == target:
                continue
            js = float(rng.uniform(0.0, 1.0))
            leak = max(0.0, min(1.0, 1.0 - js + float(rng.normal(scale=0.02))))
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "js": js,
                    "leakage": leak,
                }
            )
    out = agg.cluster_bootstrap_spearman(rows, "js", "leakage", n_iter=200, seed=7)
    # Strong negative correlation by construction.
    assert out["rho"] < -0.9
    # Bootstrap CI must enclose the point estimate (allow tiny float slack).
    assert out["ci_low"] - 1e-3 <= out["rho"] <= out["ci_high"] + 1e-3


def test_paired_cluster_bootstrap_delta_abs_rho_structure() -> None:
    rows = _synthesize_h1_long(seed=99)
    out = agg.paired_cluster_bootstrap_delta_abs_rho(
        rows, "leakage", "js", "cosine_l15", n_iter=50, seed=8
    )
    for k in (
        "delta_abs_rho",
        "abs_rho_a",
        "abs_rho_b",
        "n",
        "ci_low",
        "ci_high",
        "n_iter",
    ):
        assert k in out
    assert out["n_iter"] == 50
    # delta == |rho_a| - |rho_b|
    assert math.isclose(
        out["delta_abs_rho"],
        out["abs_rho_a"] - out["abs_rho_b"],
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


# ──────────────────────────────────────────────────────────────────────────
# H1 analysis aggregator: pooled, source-fixed, target-fixed.
# ──────────────────────────────────────────────────────────────────────────


def test_h1_analysis_returns_all_substructures() -> None:
    rows = _synthesize_h1_long(seed=11)
    out = agg._h1_analysis(rows)
    for k in (
        "h1_primary_js",
        "h1_primary_cosine_l15",
        "h1_delta_js_minus_cosine_l15",
        "h1b1_within_target_z",
        "h1b2_source_fixed",
        "h1b3_target_fixed",
        "h1_source_to_assistant",
    ):
        assert k in out, f"missing key {k}"
    # Source-fixed has 7 rows.
    assert len(out["h1b2_source_fixed"]["per_source"]) == 7
    # Target-fixed has 11 rows.
    assert len(out["h1b3_target_fixed"]["per_target"]) == 11
    # Source-to-assistant: n=7 (one per source).
    assert out["h1_source_to_assistant"]["n"] == 7


def test_h1_analysis_excludes_self_loops_if_present() -> None:
    """If is_self_loop=True rows leak in, the analyzer must drop them."""
    rows = _synthesize_h1_long(seed=3)
    n_baseline = len(rows)
    # Inject a self-loop row that, if not excluded, would change pooled n.
    rows.append(
        {
            "source": "villain",
            "target": "villain",
            "is_self_loop": True,
            "js": 0.0,
            "js_masked": 0.0,
            "js_no_marker": 0.0,
            "cosine_l15": 1.0,
            "cosine_l20": 1.0,
            "cosine_l25": 1.0,
            "source_token_entropy_mean": 1.5,
            "leakage": 0.5,
        }
    )
    out = agg._h1_analysis(rows)
    # n must still be the off-diagonal count, not n_baseline+1.
    assert out["h1_primary_js"]["n"] == n_baseline


# ──────────────────────────────────────────────────────────────────────────
# H2 analysis: BH-FDR thresholding, raw + first-difference logic.
# ──────────────────────────────────────────────────────────────────────────


def _synthesize_h2_long(*, rho_target: float = -0.8, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for source in ADAPTER_SOURCES:
        for step in agg.CHECKPOINT_STEPS:
            t = step / 2000.0
            js = float(0.05 + 0.4 * t + rng.normal(scale=0.01))
            # rho_target negative => leakage decreases as js increases
            leak = float(0.5 + (-rho_target) * (1.0 - t) + rng.normal(scale=0.02))
            rows.append(
                {
                    "source": source,
                    "checkpoint_step": step,
                    "epoch": 2 * (agg.CHECKPOINT_STEPS.index(step) + 1),
                    "mean_js_off_diag": js,
                    "mean_cosine_l20_off_diag": float(rng.uniform(-0.2, 0.2)),
                    "mean_leakage_off_diag": max(0.0, min(1.0, leak)),
                    "max_leakage_off_diag": max(0.0, min(1.0, leak + 0.1)),
                    "assistant_leakage": float(rng.uniform(0, 1)),
                    "source_token_entropy_mean": float(rng.uniform(1, 5)),
                }
            )
    return rows


def test_h2_analysis_reports_seven_sources_each_for_raw_and_diff() -> None:
    rows = _synthesize_h2_long(seed=2)
    out = agg._h2_analysis(rows)
    assert len(out["per_source_raw"]) == 7
    assert len(out["per_source_diff"]) == 7
    assert len(out["raw_fdr"]) == 7
    assert len(out["diff_fdr"]) == 7
    # h2_pass_overall is a boolean derived from per-source counts.
    assert out["h2_pass_overall"] is (
        out["n_raw_passing_negative"] >= 4 and out["n_diff_passing_negative"] >= 4
    )


# ──────────────────────────────────────────────────────────────────────────
# C1 marker-mask: relative_change reflects |delta| / |rho_raw|.
# ──────────────────────────────────────────────────────────────────────────


def test_c1_marker_mask_relative_change_finite() -> None:
    rows = _synthesize_h1_long(seed=21)
    out = agg._c1_marker_mask(rows)
    for k in (
        "rho_masked",
        "delta_abs_rho_masked_minus_raw",
        "relative_change",
        "marker_mediated_threshold_rel_change",
        "marker_mediated",
    ):
        assert k in out, f"missing {k}"
    # In a synthetic dataset where masked ~ raw, relative change should be small.
    if not math.isnan(out["relative_change"]):
        assert out["relative_change"] >= 0.0


# ──────────────────────────────────────────────────────────────────────────
# C3 villain-out: drops villain's 10 rows.
# Synthetic rows here use 7 ALL_EVAL sources × 10 off-diagonal targets = 70
# (no nurse-source extension), so n=70/n=60 in this synthetic fixture; the
# production aggregator's keys are n71/n61 because the real data includes
# the nurse-source 11-target extension. The test verifies the villain-drop
# delta of 10 rows, which is invariant to the nurse extension.
# ──────────────────────────────────────────────────────────────────────────


def test_c3_villain_out_drops_villain_cells() -> None:
    rows = _synthesize_h1_long(seed=31)
    out = agg._c3_villain_out(rows)
    n_full = out["n71"]["js_vs_leakage"]["n"]
    n_no_villain = out["n61"]["js_vs_leakage"]["n"]
    # Villain has 10 off-diagonal targets (11 - 1 self).
    assert n_full - n_no_villain == 10


# ──────────────────────────────────────────────────────────────────────────
# Sanity check 6.1#4: passes when matrices are identical.
# ──────────────────────────────────────────────────────────────────────────


def test_sanity_check_142_passes_on_identical_matrices(tmp_path: Path) -> None:
    n = 5
    rng = np.random.default_rng(7)
    M = rng.uniform(0.0, 0.4, size=(n, n))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 0.0)
    persona_names = [f"persona_{i}" for i in range(n)]
    base_state = {
        "source": "base",
        "checkpoint_step": 0,
        "persona_names": persona_names,
        "js_matrix": M.tolist(),
    }
    baseline_path = tmp_path / "div.json"
    baseline_path.write_text(
        '{"persona_names": '
        + str(persona_names).replace("'", '"')
        + ', "js_matrix": '
        + str(M.tolist())
        + "}"
    )
    res = agg._sanity_check_142(base_state, baseline_path)
    assert res["verdict"] == "PASS"
    assert res["mean_relative_error"] < 1e-6


def test_sanity_check_142_fails_when_far_off(tmp_path: Path) -> None:
    n = 5
    rng = np.random.default_rng(8)
    M = rng.uniform(0.0, 0.4, size=(n, n))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 0.0)
    persona_names = [f"persona_{i}" for i in range(n)]
    base_state = {
        "source": "base",
        "checkpoint_step": 0,
        "persona_names": persona_names,
        "js_matrix": M.tolist(),
    }
    M_far = M * 5.0  # 400% larger -> MRE huge
    baseline_path = tmp_path / "div.json"
    baseline_path.write_text(
        '{"persona_names": '
        + str(persona_names).replace("'", '"')
        + ', "js_matrix": '
        + str(M_far.tolist())
        + "}"
    )
    with pytest.raises(SystemExit):
        agg._sanity_check_142(base_state, baseline_path)


# ──────────────────────────────────────────────────────────────────────────
# Round-2 BLOCKER #1 fix: n=71 contract via path-A (extra row+col for nurse).
# ──────────────────────────────────────────────────────────────────────────


def _make_state(
    *,
    source: str,
    epoch: int,
    persona_names: list[str],
    js_seed: int,
    js_value_for_nurse_targets: float | None = None,
) -> dict:
    """Build a synthetic per-state result.json-shaped dict.

    For sources NOT in ALL_EVAL_PERSONAS, ``persona_names`` is the 12-element
    extended list (11 ALL_EVAL + nurse). The js_matrix is constructed so
    every off-diagonal cell has a finite value (to verify the n=71 contract
    that no js=None reaches downstream Spearman).
    """
    n = len(persona_names)
    rng = np.random.default_rng(js_seed)
    js = rng.uniform(0.05, 0.3, size=(n, n))
    js = (js + js.T) / 2
    np.fill_diagonal(js, 0.0)
    if js_value_for_nurse_targets is not None and source == "nurse":
        nurse_idx = persona_names.index("nurse")
        for j in range(n):
            if j == nurse_idx:
                continue
            js[nurse_idx, j] = js_value_for_nurse_targets
            js[j, nurse_idx] = js_value_for_nurse_targets

    cos_l15 = rng.uniform(-0.5, 0.5, size=(n, n))
    cos_l15 = (cos_l15 + cos_l15.T) / 2
    np.fill_diagonal(cos_l15, 1.0)
    cos_l20 = rng.uniform(-0.5, 0.5, size=(n, n))
    cos_l20 = (cos_l20 + cos_l20.T) / 2
    np.fill_diagonal(cos_l20, 1.0)
    cos_l25 = rng.uniform(-0.5, 0.5, size=(n, n))
    cos_l25 = (cos_l25 + cos_l25.T) / 2
    np.fill_diagonal(cos_l25, 1.0)

    return {
        "source": source,
        "checkpoint_step": (epoch // 2) * 200,
        "epoch": epoch,
        "persona_names": persona_names,
        "extra_source_persona": "nurse" if source == "nurse" else None,
        "js_matrix": js.tolist(),
        "js_matrix_masked": js.tolist(),
        "js_matrix_no_marker": js.tolist(),
        "kl_matrix": js.tolist(),
        "source_token_entropy_mean": 2.5,
        "cosine_matrices": {
            "layer_15": {"persona_names": persona_names, "matrix": cos_l15.tolist()},
            "layer_20": {"persona_names": persona_names, "matrix": cos_l20.tolist()},
            "layer_25": {"persona_names": persona_names, "matrix": cos_l25.tolist()},
        },
    }


def test_h1_long_n71_with_nurse_source_real_js() -> None:
    """BLOCKER #1 round-1 -> round-2: nurse-source rows must contribute real
    JS values (not js=None) so the n=71 contract is real.

    Build epoch-20 states for all 7 ADAPTER_SOURCES. For 6 sources in
    ALL_EVAL_PERSONAS, the per-state matrix is 11×11 over the eval personas.
    For ``nurse`` (the 1 source NOT in ALL_EVAL_PERSONAS), the per-state
    matrix is 12×12 with nurse appended as the 12th persona — mirroring the
    path-A fix in compute_js_convergence_228._compute_state_metrics.

    Verify:
      * ``_build_h1_long`` produces exactly 71 rows.
      * Every row has a finite (non-None, non-NaN) ``js`` value.
      * ``_spearman_xy`` over the 71 cells returns n=71 (no silent drops).
      * 60 rows are from non-nurse sources, 11 are from nurse.
    """
    eval_personas = list(ALL_EVAL_PERSONAS.keys())
    nurse_extended = [*eval_personas, "nurse"]

    states: list[dict] = []
    leakage_states: dict[tuple[str, int], dict] = {}
    for src_idx, source in enumerate(ADAPTER_SOURCES):
        names = nurse_extended if source == "nurse" else eval_personas
        states.append(
            _make_state(
                source=source,
                epoch=20,
                persona_names=names,
                js_seed=src_idx + 1,
                js_value_for_nurse_targets=0.18 if source == "nurse" else None,
            )
        )
        # Synthetic leakage rates per ALL_EVAL persona (covers all valid targets
        # for both ALL_EVAL sources and nurse).
        leakage_states[(source, 2000)] = {
            n: {"rate": 0.1 + 0.02 * i} for i, n in enumerate(eval_personas)
        }

    h1_long = agg._build_h1_long(states, leakage_states)
    assert len(h1_long) == 71, f"Expected n=71 cells (user's pre-registration), got {len(h1_long)}"
    assert all(r["js"] is not None for r in h1_long), "BLOCKER #1: js=None still leaking through"
    assert all(not math.isnan(r["js"]) for r in h1_long), "Found js=NaN cells in h1_long"

    nurse_rows = [r for r in h1_long if r["source"] == "nurse"]
    non_nurse_rows = [r for r in h1_long if r["source"] != "nurse"]
    assert len(nurse_rows) == 11, (
        "nurse-source must contribute 11 cells (no nurse target self-loop)"
    )
    assert len(non_nurse_rows) == 60, "non-nurse sources must contribute 6 × 10 = 60 cells"

    # Pull JS values through _spearman_xy and assert n=71 reaches downstream.
    rho, _p, n = agg._spearman_xy([r["js"] for r in h1_long], [r["leakage"] for r in h1_long])
    assert n == 71, f"BLOCKER #1: Spearman dropped {71 - n} cells silently"
    assert not math.isnan(rho)


def test_h1_long_raises_when_nurse_state_missing_extra_row() -> None:
    """If the per-state worker forgets to append the nurse row, the
    aggregator must SystemExit rather than emit silent js=None rows.

    Round-1 silently emitted 11 None-js rows for nurse and let downstream
    Spearman drop them (n=71 -> n=60 effective). Round-2 fix: hard SystemExit.
    """
    eval_personas = list(ALL_EVAL_PERSONAS.keys())
    states: list[dict] = []
    leakage_states: dict[tuple[str, int], dict] = {}
    for src_idx, source in enumerate(ADAPTER_SOURCES):
        # Bug simulation: nurse state has only 11 ALL_EVAL personas (no nurse row).
        names = eval_personas
        states.append(
            _make_state(source=source, epoch=20, persona_names=names, js_seed=src_idx + 1)
        )
        leakage_states[(source, 2000)] = {
            n: {"rate": 0.1 + 0.02 * i} for i, n in enumerate(eval_personas)
        }

    with pytest.raises(SystemExit, match="H1 contract violation"):
        agg._build_h1_long(states, leakage_states)


# ──────────────────────────────────────────────────────────────────────────
# Round-2 BLOCKER #2 fix: substring-position marker mask.
# ──────────────────────────────────────────────────────────────────────────


def test_marker_mask_positions_substring() -> None:
    """The substring-position-based marker mask must:

    * Mask exactly the tokens whose char-spans overlap a ``[ZLT]`` substring
      (case-insensitive).
    * NOT mask generic newlines (``\\n``) or generic ``]`` tokens.
    * Return an empty mask (all False) when the response contains no marker.

    Uses an offset-mapping-capable mock tokenizer so we can run this test
    without loading a real Qwen tokenizer (no model download required for CI).
    """

    class MockTokenizer:
        """Mock that mirrors the offset-mapping API our code relies on.

        We model a deterministic tokenization where every character is a
        token; the token id is the character's ord(). This makes the
        decode-encode round-trip a length-preserving identity, and the
        offset_mapping returns (i, i+1) per token. Real tokenizers do
        chunkier tokenization, but the substring-overlap logic is the part
        we want to verify, so character-level granularity is the cleanest
        test substrate.
        """

        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids)

        def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
            return {
                "input_ids": [ord(c) for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))],
            }

    # Import the marker-mask helper from the worker script.
    sys.path.insert(0, str(ROOT / "scripts"))
    import compute_js_convergence_228 as cj

    tokenizer = MockTokenizer()

    # Case 1: response with two [ZLT] substrings. Every character inside
    # those spans should be flagged; everything outside should NOT be.
    resp = "hello [ZLT] world\nlist:\n- foo]\n- [ZLT]"
    ids = [ord(c) for c in resp]
    mask = cj._marker_mask_positions(ids, tokenizer)

    # Compute expected mask manually.
    needle = "[ZLT]"
    expected = [False] * len(resp)
    cursor = 0
    while True:
        idx = resp.lower().find(needle.lower(), cursor)
        if idx == -1:
            break
        for k in range(idx, idx + len(needle)):
            expected[k] = True
        cursor = idx + 1

    assert mask.tolist() == expected, (
        f"Marker mask mismatch.\nGot:      {mask.tolist()}\nExpected: {expected}"
    )

    # Critical regression check: '\n' and ']' tokens that are NOT inside a
    # [ZLT] span must not be masked. Find their positions in the original
    # string and confirm mask is False there.
    for i, c in enumerate(resp):
        if c in ("\n", "]") and not expected[i]:
            assert not mask[i].item(), (
                f"BLOCKER #2 regression: token {i} ({c!r}) was masked but is outside any [ZLT] span"
            )

    # Case 2: response with no marker — mask must be all False.
    resp_clean = "this is a plain response with [no markers] and lines\n"
    ids_clean = [ord(c) for c in resp_clean]
    mask_clean = cj._marker_mask_positions(ids_clean, tokenizer)
    assert not mask_clean.any().item(), "Mask should be all False when no [ZLT] present"

    # Case 3: case-insensitive — '[zlt]' matches.
    resp_lower = "abc[zlt]def"
    ids_lower = [ord(c) for c in resp_lower]
    mask_lower = cj._marker_mask_positions(ids_lower, tokenizer)
    expected_lower = [
        c in {"[", "z", "l", "t", "]"} and 3 <= i <= 7 for i, c in enumerate(resp_lower)
    ]
    assert mask_lower.tolist() == expected_lower

    # Case 4: empty response.
    mask_empty = cj._marker_mask_positions([], tokenizer)
    assert mask_empty.shape[0] == 0


# ──────────────────────────────────────────────────────────────────────────
# Round-2 ISSUE: Sanity 6.1#4 must FAIL on Spearman ρ < 0.99 even when MRE
# is small (rank-order inversion path).
# ──────────────────────────────────────────────────────────────────────────


def test_sanity_check_142_fails_on_rank_inversion(tmp_path: Path) -> None:
    """Plan §6.1#4 has TWO independent failure conditions:
    (a) MRE > 5% threshold, OR (b) Spearman ρ < 0.99.

    Round-1 only tested (a). This test covers (b) by inverting the rank
    order while keeping magnitudes close — MRE stays small but ρ collapses.
    """
    n = 6
    persona_names = [f"persona_{i}" for i in range(n)]

    # Reference matrix with monotone-increasing pair magnitudes. The
    # threshold for the relative-error path is 5% — we want every pair's
    # |M_ref - M_inv| / |M_ref| << 0.05 so MRE is well under threshold,
    # while the rank order between M_ref and M_inv is exactly inverted
    # (Spearman ρ = -1.0).
    pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    M_ref = np.zeros((n, n))
    M_inv = np.zeros((n, n))
    n_pairs = len(pair_indices)
    for k, (i, j) in enumerate(pair_indices):
        # Reference: ~0.5 + 0.0001-step variation -> 0.5000, 0.5001, ..., 0.5014.
        # Inverted: same magnitudes in reverse order. MRE is at most
        # 0.0014 / 0.5 = 0.28% << 5% threshold.
        v_ref = 0.500 + 0.0001 * k
        v_inv = 0.500 + 0.0001 * (n_pairs - 1 - k)
        M_ref[i, j] = M_ref[j, i] = v_ref
        M_inv[i, j] = M_inv[j, i] = v_inv

    base_state = {
        "source": "base",
        "checkpoint_step": 0,
        "persona_names": persona_names,
        "js_matrix": M_ref.tolist(),
    }
    baseline_path = tmp_path / "div.json"
    baseline_path.write_text(
        '{"persona_names": '
        + str(persona_names).replace("'", '"')
        + ', "js_matrix": '
        + str(M_inv.tolist())
        + "}"
    )

    # MRE here is < 1% (max relative difference between 0.100 and ~0.115),
    # but ρ on the rank-inverted pair list is exactly -1.0.
    with pytest.raises(SystemExit, match="ρ"):
        agg._sanity_check_142(base_state, baseline_path)


# ──────────────────────────────────────────────────────────────────────────
# Round-2 ISSUE: _spearman_xy must handle production None-mixed inputs.
# ──────────────────────────────────────────────────────────────────────────


def test_spearman_xy_handles_production_none_pattern() -> None:
    """In production, ``_matrix_value`` returns None when a persona name is
    not found in the matrix's persona_names list. Round-1 tests synthesized
    all-non-None data, so the None-passthrough path was never exercised on
    the production-realistic shape (which mixes None and float in the *same*
    list, both x and y).

    Build a list with sparse Nones in both x and y, verify _spearman_xy:
      * drops only the rows where EITHER x[i] or y[i] is None,
      * reports the dropped count via the returned n,
      * returns a finite ρ on the surviving rows.
    """
    rng = np.random.default_rng(0)
    n_total = 71
    x = list(rng.uniform(0.0, 0.5, size=n_total))
    y = list(0.5 - 0.8 * np.array(x) + rng.normal(scale=0.02, size=n_total))

    # Insert Nones at production-realistic positions: a contiguous block of
    # 11 Nones in y (mimicking nurse-source-rows getting js=None in round-1)
    # plus a few scattered Nones in x.
    for k in range(60, 71):
        y[k] = None
    for k in (3, 17, 42):
        x[k] = None

    rho, _p, n = agg._spearman_xy(x, y)
    assert n == n_total - 11 - 3, (
        f"Expected {n_total - 14} rows after dropping None-mixed cells, got {n}"
    )
    assert not math.isnan(rho)
    assert rho < 0  # by construction y is anti-correlated with x
