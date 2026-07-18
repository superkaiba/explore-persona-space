"""Tiny-real unit tests for the issue #1072 ``lowdim-token-subspace`` round.

Synthetic small-H analogue (plan §4.3 item 4): QR nesting, dedupe/rank-guard
behavior (incl. the synthetic duplicate-column positive control), additivity
exactness of the projected channels through ``run_component_cell``, the
Holm/lattice logic in the stats module, plus degenerate-input probes of the
data-dependent gate branches (K4 pilot abort, g4' mismatch raise, g7 fraction
abort, fold min-N skip) and the PRODUCTION frozen-λ read path against the
repo-committed artifacts (#825 --smoke-ternary lesson).
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.experiments.issue_1072 import run_1072_lowdim as low  # noqa: E402
from explore_persona_space.experiments.issue_1072 import subspace_basis as sb  # noqa: E402
from explore_persona_space.experiments.issue_1072.component_ridge import (  # noqa: E402
    run_component_cell,
    serial_component_reference,
)
from scripts.issue1072_lowdim_stats import (  # noqa: E402
    family_mapping,
    lattice_verdict,
    signflip_p_two,
)

RNG = np.random.default_rng(7)
V, H = 64, 32  # tiny-real analogue: 64-token vocab, 32-dim hidden


def _u_dir() -> torch.Tensor:
    return torch.from_numpy(RNG.standard_normal((V, H)).astype(np.float32))


# ── subspace_basis: nesting / dedupe / rank guard / projection ──────────────────


def test_qr_nesting_top8_from_top32() -> None:
    """span(Q32[:, :8]) == span(QR(leading-8 columns)) numerically (plan A4)."""
    u = _u_dir()
    ids = np.stack([RNG.choice(V, size=12, replace=False) for _ in range(6)])
    gap = sb.nesting_check(u, ids, k_lead=8)
    assert gap < 1e-4, gap


def test_projection_additivity_and_idempotence() -> None:
    u = _u_dir()
    ids = torch.from_numpy(np.stack([RNG.choice(V, size=5, replace=False) for _ in range(4)]))
    q, eff, red = sb.orthonormal_bases(u, ids)
    assert not red and eff.tolist() == [5, 5, 5, 5]
    z = torch.from_numpy(RNG.standard_normal((4, H)).astype(np.float32))
    z_par = sb.project_rows(z, q)
    # Idempotence P(Pz) = Pz and complement orthogonality <Pz, z - Pz> ≈ 0.
    assert torch.allclose(sb.project_rows(z_par, q), z_par, atol=1e-5)
    assert float((z_par * (z - z_par)).sum(dim=1).abs().max()) < 1e-3


def test_compact_dedupe_windows_order_preserving() -> None:
    window = torch.tensor([[5, 7, 5, 9, 7, 9, 9, 2], [1, 2, 3, 4, 5, 6, 7, 8]])
    valid = torch.ones_like(window, dtype=torch.bool)
    valid[0, 7] = False  # truncated tail entry
    ids, eff = sb.compact_dedupe_windows(window, valid)
    assert eff.tolist() == [3, 8]
    assert ids[0, :3].tolist() == [5, 7, 9]  # original order, duplicates dropped
    assert ids[1].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_rank_guard_duplicate_column_positive_control() -> None:
    """A near-duplicate unembedding row trips g7 and the SVD fallback preserves span."""
    u_np = RNG.standard_normal((V, H)).astype(np.float32)
    u_np[11] = u_np[3] + 1e-7 * RNG.standard_normal(H).astype(np.float32)
    u = torch.from_numpy(u_np)
    ids = torch.tensor([[3, 11, 20, 25], [1, 2, 4, 5]])
    q, eff, red = sb.orthonormal_bases(u, ids)
    assert red == [0], red  # the degenerate row fell back, the clean row did not
    assert eff.tolist() == [3, 4]
    # Span preserved: the fallback projector reproduces the full-rank subset's.
    q_ref, _e, red_ref = sb.orthonormal_bases(u, torch.tensor([[3, 20, 25]]))
    assert not red_ref
    p_a = q[0] @ q[0].T
    p_b = q_ref[0] @ q_ref[0].T
    assert float((p_a - p_b).norm()) < 1e-3


def test_topk_ids_match_bruteforce_logits() -> None:
    """Chunked GEMM topk == brute-force logits topk (RMSNorm scale exactness)."""
    u = _u_dir()
    h = torch.from_numpy(RNG.standard_normal((9, H)).astype(np.float32))
    ids, lps = sb.topk_ids_from_final_hidden(h, u, rms_eps=1e-6, k=8, chunk=4)
    rms = torch.sqrt(h.pow(2).mean(dim=1, keepdim=True) + 1e-6)
    logits = (h / rms) @ u.T
    ref = torch.topk(logits, 8, dim=1)
    assert torch.equal(ids.to(torch.int64), ref.indices)
    ref_lp = ref.values - torch.logsumexp(logits, dim=1, keepdim=True)
    assert float((lps.float() - ref_lp).abs().max()) < 1e-2  # fp16 storage
    assert bool((lps.float() <= 0).all())


# ── battery algebra: additivity + oracle parity with a subspace pair_fn ─────────


def test_component_cell_additivity_with_subspace_pair_fn() -> None:
    n_tr, n_te, h_in = 24, 10, 6
    u = _u_dir()
    x_tr = RNG.standard_normal((n_tr, h_in))
    x_te = RNG.standard_normal((n_te, h_in))
    y_full = {"train": RNG.standard_normal((n_tr, H)), "test": RNG.standard_normal((n_te, H))}
    ids = torch.from_numpy(
        np.stack([RNG.choice(V, size=4, replace=False) for _ in range(n_tr + n_te)])
    )
    q, _eff, red = sb.orthonormal_bases(u, ids)
    assert not red
    q_by = {"train": q[:n_tr], "test": q[n_tr:]}

    def pair_fn(split: str, gi: int):
        assert gi == 0
        yf = y_full[split]
        y_par = sb.project_rows(torch.from_numpy(yf).float(), q_by[split]).double().numpy()
        return y_par, yf

    res = run_component_cell(
        x_tr, {"test": x_te}, pair_fn, ["cell"], np.asarray([1.0]), device="cpu"
    )
    assert res.additivity_max_dev < 1e-9
    pooled = res.pooled["test"]
    ident = pooled["r2_full"][0] - (pooled["C_par"][0] + pooled["C_perp"][0] + pooled["C_cross"][0])
    assert abs(ident) < 1e-9
    # Independent fp64 dual-form oracle parity (the battery's own gate shape).
    oracle = serial_component_reference(x_tr, x_te, *pair_fn("train", 0), *pair_fn("test", 0), 1.0)
    got = res.channels["test"][:, 0, :]
    rel = np.max(np.abs(oracle - got) / np.maximum(np.abs(oracle), 1.0))
    assert rel < 1e-7, rel


# ── stats: sign-flip p + lattice + family mapping ───────────────────────────────


def test_signflip_p_two_counting() -> None:
    flips = np.asarray([0.1, -0.2, 0.05, -0.01])
    assert signflip_p_two(0.15, flips) == pytest.approx((1 + 1) / (1 + 4))
    assert signflip_p_two(0.3, flips) == pytest.approx((1 + 0) / (1 + 4))
    assert signflip_p_two(float("nan"), flips) is None
    assert signflip_p_two(0.1, np.asarray([])) is None


def test_lattice_verdict_branches() -> None:
    assert lattice_verdict(0.02, [0.01, 0.03], 0.01) == "Rescue"
    assert lattice_verdict(-0.03, [-0.04, -0.02], 0.04) == "Extended falsification"
    assert lattice_verdict(0.02, [-0.01, 0.05], 0.01) == "Inconclusive"  # CI spans 0
    assert lattice_verdict(0.02, [0.01, 0.03], 0.2) == "Inconclusive"  # Holm fails
    assert lattice_verdict(-0.03, [-0.04, -0.02], None) == "Inconclusive"  # no p


def test_family_mapping_branches() -> None:
    rescue = family_mapping({"top8": "Rescue", "top32": "Inconclusive", "look8": "Inconclusive"})
    assert rescue["headline"] == "overturn/qualify"
    sharpen = family_mapping(dict.fromkeys(("top8", "top32", "look8"), "Extended falsification"))
    assert sharpen["headline"] == "sharpen"
    partial = family_mapping(
        {"top8": "Extended falsification", "top32": "Inconclusive", "look8": "Inconclusive"}
    )
    assert partial["headline"] == "partial"


# ── degenerate-input probes of the data-dependent gate branches ─────────────────


def test_pilot_abort_branch_rc7(tmp_path: pathlib.Path) -> None:
    """K4 designed abort fires with the distinct rc in production mode."""
    with pytest.raises(SystemExit) as exc:
        low._pilot_check(
            "capture",
            measured_wall_s=3600.0,
            units_done=1,
            units_total=100,
            booked_h=1.5,
            base_dir=tmp_path,
            smoke=False,
            execution_shape="probe",
        )
    assert exc.value.code == low.PILOT_ABORT_RC == 7
    rec = json.loads(
        (
            tmp_path / "eval_results/issue_1072/lowdim-token-subspace/pilot_gate_capture.json"
        ).read_text()
    )
    assert rec["verdict"] == "ABORT"
    # Smoke demotes the verdict to a log line (same computation, no exit).
    low._pilot_check("capture", 3600.0, 1, 100, 1.5, tmp_path, smoke=True, execution_shape="probe")


def test_g4p_compare_mismatch_and_pass() -> None:
    ref = np.abs(RNG.standard_normal((5, 8))) + 1.0
    got = ref.copy()
    mismatches: list[str] = []
    low._g4p_compare_channels("cell", got, ref.astype(np.float32), mismatches)
    assert mismatches == []
    bad = ref.copy()
    bad[2, 6] *= 1.5  # ss_res_full drift
    low._g4p_compare_channels("cell", bad, ref.astype(np.float32), mismatches)
    assert mismatches and "ss_res_full" in mismatches[0]


def test_battery_min_n_skip_branch(tmp_path: pathlib.Path) -> None:
    """Sub-floor matched populations return a skipped record before any IO."""
    pool = list(range(6))
    fold_split = {"fold": 0, "train": pool[:3], "val": pool[3:4], "test": pool[4:]}
    spans = {a: np.full(6, 5, dtype=np.int64) for a in low.ARMS}  # < T2+16 => no matched rows
    npz, rec = low._battery_fold_layer_lowdim(
        tmp_path,
        fold_split,
        layer=0,
        pool_ids=pool,
        spans_by_arm=spans,
        u_dir_np=np.zeros((V, H), dtype=np.float32),
        refs=None,  # unreachable past the skip branch
        fit_device="cpu",
        min_train=4,
        smoke=True,
        run_parity=False,
    )
    assert npz == {} and rec["skipped"] is True


def test_g7_fraction_abort_shape() -> None:
    """The >1% rank-reduced abort predicate (capture-side g7 verdict)."""
    g7 = {"n_positions": 1000, "n_rank_reduced": 20}
    frac = g7["n_rank_reduced"] / max(g7["n_positions"], 1)
    assert frac > low.RANK_REDUCED_MAX_FRAC  # this shape MUST abort in phase_capture


def test_slot_tslot_convention() -> None:
    span = 40
    assert low._slot_tslot("f16_t1", span) == 1
    assert low._slot_tslot("f16_t16", span) == 16
    assert low._slot_tslot("l16_m2", span) == span - 1  # <|im_end|> slot
    assert low._slot_tslot("d10_p50", span) == round(0.50 * (span - 1)) + 1
    with pytest.raises(ValueError):
        low._slot_tslot("z_t32", span)


# ── production frozen-λ read path against the repo-committed artifacts ──────────


def test_lowdim_refs_production_lambda_read(tmp_path: pathlib.Path) -> None:
    """The PRODUCTION LowdimRefs branch resolves the committed λ* exactly
    (#825: never leave the non-smoke ternary branch unexecuted)."""
    committed = _REPO_ROOT / "eval_results" / "issue_1072" / "battery_1072_fold4.json"
    if not committed.exists():
        pytest.skip("committed 1072 battery records not in this checkout")
    refs = low.LowdimRefs(tmp_path, smoke=False)
    lam = refs.frozen_lambdas(4, 26)
    assert lam["cleg_mean"] == pytest.approx(3162.2776601683795)
    assert lam["zleg_mean"] == pytest.approx(3162.2776601683795)
    assert lam["p_last"] == pytest.approx(10000.0)
    assert lam["slot_table"]["f16_t1"] == pytest.approx(3162.2776601683795)
    assert set(low.DECOMP_SLOTS) <= set(lam["slot_table"])
