"""Regression tests for scripts/issue654_analyze.py (round-2 BLOCKER fixes).

Two CPU-only tests (<1s), one per round-1 reconciler BLOCKER concern:

  - test_companion_consumes_full_prompt_readout_not_query_end
    (concern companion-read-not-same-slot): the companion same-position contrast
    must read the full-prompt assistant-generation slot (the saved per-pair
    ``readout`` bank), NOT the query-end residual (``A_qry``). Also pins that
    ``_load_banks`` fails loud when a per-pair ``.pt`` lacks the ``readout`` key.

  - test_matched_and_floor_consume_identical_pre_centered_tensors
    (concern per-tier-floor-centering-mismatch): the per-tier matched cosine and
    the per-tier shuffled floor must both consume the SAME globally-centered+
    normalized banks (centered ONCE over the full bank), never a per-tier
    re-centering. Constructed with deliberately unequal per-tier means so a
    within-tier re-centering would give a numerically different floor.

Pure numpy/torch + a temp .pt round-trip; fixed seed; no GPU, no model load.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "issue654_analyze.py"
_spec = importlib.util.spec_from_file_location("issue654_analyze_under_test", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
analyze_mod = importlib.util.module_from_spec(_spec)
sys.modules["issue654_analyze_under_test"] = analyze_mod
_spec.loader.exec_module(analyze_mod)


# ── Concern companion-read-not-same-slot ────────────────────────────────────


def test_companion_consumes_full_prompt_readout_not_query_end() -> None:
    """The companion same-slot contrast reads the full-prompt assistant-gen slot.

    Build two pairs sharing one context. The companion output must equal the
    cosine of (context-only readout) vs (full-prompt per-pair ``readout``) at
    each layer — and must be INVARIANT to the query-end bank, because the
    same-slot contrast no longer touches A_qry. We verify by feeding a
    ``readout`` bank whose hand-computed cosine is known, then confirming the
    function's output matches it (so it consumed ``A_readout``, not A_qry).
    """
    n_layers = 2
    # context-only readout for the single shared context "c0".
    companion = {
        "c0": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float64
        )  # (n_layers, hidden)
    }
    # Full-prompt assistant-gen slot readout, one row per pair (the A_readout bank).
    # pair 0 layer 0 == companion (cos 1.0); pair 1 layer 0 orthogonal (cos 0.0).
    A_readout = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],  # pair 0
            [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],  # pair 1
        ],
        dtype=torch.float64,
    )  # (n_pairs=2, n_layers=2, hidden=4)
    meta = [
        {"pair_id": "c0__q0", "context_id": "c0", "context_type": "persona"},
        {"pair_id": "c0__q1", "context_id": "c0", "context_type": "persona"},
    ]

    out = analyze_mod._companion_cosine_per_layer(companion, meta, A_readout)

    # Hand-computed per-pair cosines (context-only vs full-prompt readout, per layer).
    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(
            (
                torch.nn.functional.normalize(a, dim=0) * torch.nn.functional.normalize(b, dim=0)
            ).sum()
        )

    p0 = [_cos(companion["c0"][L], A_readout[0, L]) for L in range(n_layers)]
    p1 = [_cos(companion["c0"][L], A_readout[1, L]) for L in range(n_layers)]
    expected_tier = np.mean([p0, p1], axis=0)  # both pairs are tier "persona"

    got_tier = np.array(out["per_tier_mean"]["persona"])
    assert np.allclose(got_tier, expected_tier, atol=1e-9), (got_tier, expected_tier)
    # pair0 layer0 cos == 1.0 (identical vectors); pair1 layer0 cos == 0.0 (orthogonal).
    assert abs(p0[0] - 1.0) < 1e-9
    assert abs(p1[0] - 0.0) < 1e-9

    # The function signature must NOT accept a query-end bank — the companion read
    # is same-slot only. (A_qry as a third positional would silently re-introduce
    # the different-position confound.)
    import inspect

    params = list(inspect.signature(analyze_mod._companion_cosine_per_layer).parameters)
    assert params == ["companion", "meta", "A_readout"], params


def test_load_banks_requires_readout_key(tmp_path: Path) -> None:
    """_load_banks fails loud on a per-pair .pt missing the 'readout' bank.

    A pre-fix extraction (no readout_position) wrote pairs without 'readout';
    consuming those for the same-slot companion contrast is unrecoverable, so
    the loader must raise rather than silently fall back to query-end.
    """
    banks_dir = tmp_path / "dual_pos"
    banks_dir.mkdir()
    n_layers, hidden = 2, 4
    # Context-only companion readout file.
    (banks_dir / "context_only").mkdir()
    torch.save(
        {
            "context_id": "c0",
            "context_type": "persona",
            "readout": torch.zeros(n_layers, hidden, dtype=torch.float32),
            "layers": [0, 1],
        },
        banks_dir / "context_only" / "c0.pt",
    )
    # Per-pair file WITHOUT a 'readout' key (the pre-fix shape).
    torch.save(
        {
            "pair_id": "c0__q0",
            "context_type": "persona",
            "context_id": "c0",
            "query_id": "q0",
            "topicality": "on",
            "length": "short",
            "context_end": torch.zeros(n_layers, hidden, dtype=torch.float32),
            "query_end": torch.zeros(n_layers, hidden, dtype=torch.float32),
            "layers": [0, 1],
            "companion_context_only_file": "context_only/c0.pt",
        },
        banks_dir / "pair_000000.pt",
    )
    (banks_dir / "extraction_manifest.json").write_text('{"layers": [0, 1]}')

    try:
        analyze_mod._load_banks(banks_dir)
    except RuntimeError as e:
        assert "readout" in str(e), str(e)
    else:
        raise AssertionError("_load_banks must raise when a per-pair .pt lacks 'readout'")


# ── Concern per-tier-floor-centering-mismatch ───────────────────────────────


def test_matched_and_floor_consume_identical_pre_centered_tensors() -> None:
    """Per-tier matched cosine and per-tier floor use the SAME global-centered banks.

    Two tiers with deliberately different per-tier means. The matched per-tier
    cosine reads ``centered[idx]`` (off the globally-centered banks); the floor
    derangement must read the SAME globally-centered banks restricted to ``idx``
    — NOT a within-tier re-centering. We verify by computing the floor two ways:
    (a) via the production helper on the global-centered banks, and (b) a
    reference within-tier-recentered floor; they must DIFFER (proving the
    production helper is global-centered), and (a) must match a direct
    derangement of the global-centered+normalized rows.
    """
    rng_seed = 42
    # Tier A (rows 0,1) centered near +5; tier B (rows 2,3) centered near -5.
    A_ctx = torch.tensor(
        [
            [[5.0, 1.0, 0.0]],
            [[5.0, -1.0, 0.0]],
            [[-5.0, 0.0, 1.0]],
            [[-5.0, 0.0, -1.0]],
        ],
        dtype=torch.float64,
    )  # (4, 1, 3)
    A_qry = torch.tensor(
        [
            [[4.0, 0.0, 1.0]],
            [[6.0, 0.0, -1.0]],
            [[-4.0, 1.0, 0.0]],
            [[-6.0, -1.0, 0.0]],
        ],
        dtype=torch.float64,
    )

    ctx_hat = analyze_mod._global_center_normalize(A_ctx)
    qry_hat = analyze_mod._global_center_normalize(A_qry)

    tier_a = np.array([0, 1])

    # (a) Production floor on the GLOBAL-centered banks (no re-centering).
    rng_a = np.random.default_rng(rng_seed)
    floor_global = analyze_mod._derangement_floor(ctx_hat, qry_hat, tier_a, rng_a, b=4)

    # The only derangement of a 2-element set is the swap, so the floor mean is
    # deterministic: mean of <ctx_hat[0], qry_hat[1]> and <ctx_hat[1], qry_hat[0]>.
    expected_global = float(
        0.5
        * (
            (ctx_hat[0, 0] * qry_hat[1, 0]).sum().item()
            + (ctx_hat[1, 0] * qry_hat[0, 0]).sum().item()
        )
    )
    assert abs(floor_global["mean"][0] - expected_global) < 1e-9, (
        floor_global["mean"][0],
        expected_global,
    )

    # (b) Reference WITHIN-TIER re-centered floor (the BUGGY round-1 behavior).
    sub_ctx = A_ctx[tier_a, 0]  # (2, 3)
    sub_qry = A_qry[tier_a, 0]
    sub_ctx_c = torch.nn.functional.normalize(sub_ctx - sub_ctx.mean(dim=0, keepdim=True), dim=1)
    sub_qry_c = torch.nn.functional.normalize(sub_qry - sub_qry.mean(dim=0, keepdim=True), dim=1)
    within_tier_floor = float(
        0.5
        * ((sub_ctx_c[0] * sub_qry_c[1]).sum().item() + (sub_ctx_c[1] * sub_qry_c[0]).sum().item())
    )

    # The two floors MUST differ — the production helper is global-centered while
    # the buggy within-tier re-centering gives a different value. (Identical means
    # the global-centering fix did not take.)
    assert abs(expected_global - within_tier_floor) > 1e-6, (
        "global-centered floor and within-tier-recentered floor are identical — "
        "the per-tier-centering fix did not take"
    )

    # And the matched per-tier cosine reads off the SAME ctx_hat/qry_hat — so the
    # matched-minus-floor difference subtracts identical pre-centered tensors.
    centered = analyze_mod._centered_cos_per_layer(ctx_hat, qry_hat)  # (4, 1)
    matched_tier_mean = float(centered[tier_a, 0].mean())
    expected_matched = float(
        0.5
        * (
            (ctx_hat[0, 0] * qry_hat[0, 0]).sum().item()
            + (ctx_hat[1, 0] * qry_hat[1, 0]).sum().item()
        )
    )
    assert abs(matched_tier_mean - expected_matched) < 1e-9, (matched_tier_mean, expected_matched)
