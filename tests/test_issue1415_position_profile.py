"""CPU tests for the #1415 answer-position-shift-profile follow-up (plan v8).

Covers: `bin_matrix` property tests (n in {1, 3, 8, 13, 200}), the
vectorized-vs-naive binned-pooling equivalence (stated float tolerance),
degenerate-input probes for the data-dependent gates (>20% pair-drop
fail-loud; §3.5 parity HALT verdict), the FULL `--tiny` e2e (2-layer
from-config Qwen over the real vocab, 2 fixture pairs x 2 draws written
through the REAL cell-meta/draws schema, phase chain p0->p4 to the JSONs —
the tiny-real standard: real tokenizer, real capture path, fake ONLY the
GPU-scale weights + the remote Hub boundary via local-mirror), and the
1-file REAL staging probe (HF_TOKEN-gated: one real gen1b draws file at the
pinned revision through the production staging helper + production loader).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1415_position_profile as drv  # noqa: E402

from explore_persona_space.experiments.issue1415.steering import (  # noqa: E402
    BIN_NAMES,
    bin_matrix,
)

# ── bin_matrix property tests ─────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 3, 8, 13, 200])
def test_bin_matrix_shape_and_row_normalization(n):
    M = bin_matrix(n)
    assert M.shape == (13, n)
    for b, name in enumerate(BIN_NAMES):
        row = M[b]
        if row.isnan().any():
            assert row.isnan().all(), f"{name}: mixed NaN/finite row at n={n}"
        else:
            assert abs(float(row.sum()) - 1.0) < 1e-6, (name, float(row.sum()))
            assert (row >= 0).all()


@pytest.mark.parametrize("n", [1, 3, 8, 13, 200])
def test_bin_matrix_first_last_membership(n):
    M = bin_matrix(n)
    first, last = M[BIN_NAMES.index("first")], M[BIN_NAMES.index("last")]
    assert float(first[0]) == 1.0 and float(first.sum()) == 1.0
    assert float(last[n - 1]) == 1.0 and float(last.sum()) == 1.0


@pytest.mark.parametrize("n", [1, 3, 8, 13, 200])
def test_bin_matrix_nan_rows_exactly_where_membership_empty(n):
    M = bin_matrix(n)
    idx = torch.arange(n)
    dec = torch.clamp((10 * idx) // n, max=9)
    expected_empty = {"tok2_5": n < 2} | {
        f"dec{d + 1}": not bool((dec == d).any()) for d in range(10)
    }
    expected_empty |= {"first": False, "last": False}
    for b, name in enumerate(BIN_NAMES):
        assert bool(M[b].isnan().all()) == expected_empty[name], (name, n)


def test_bin_matrix_deciles_partition_at_n_ge_10():
    # For n >= 10 every decile is non-empty and the 10 decile rows jointly
    # cover each position exactly once (column sums of the MASKS = 1).
    for n in (13, 200):
        M = bin_matrix(n)
        dec_rows = M[2:12]
        assert not dec_rows.isnan().any()
        masks = (dec_rows > 0).float()
        assert torch.equal(masks.sum(0), torch.ones(n))


def test_bin_matrix_tok2_5_membership():
    M = bin_matrix(8)
    row = M[BIN_NAMES.index("tok2_5")]
    assert torch.equal((row > 0).nonzero().squeeze(-1), torch.arange(1, 5))
    assert torch.allclose(row[1:5], torch.full((4,), 0.25))


# ── vectorized-vs-naive binned pooling equivalence ────────────────────


def test_einsum_binned_pooling_matches_naive_loop():
    """The production einsum ('bn,lnh->blh' against bin_matrix) reproduces a
    naive per-bin mean loop within fp32 tolerance (atol 1e-5)."""
    torch.manual_seed(0)
    for n in (1, 3, 8, 13, 47):
        L, H = 3, 16
        acts = torch.randn(L, n, H)
        M = bin_matrix(n).to(acts)
        prof = torch.einsum("bn,lnh->blh", M, acts)  # (13, L, H)
        idx = torch.arange(n)
        dec = torch.clamp((10 * idx) // n, max=9)
        members = {
            "first": idx == 0,
            "tok2_5": (idx >= 1) & (idx <= 4),
            **{f"dec{d + 1}": dec == d for d in range(10)},
            "last": idx == n - 1,
        }
        for b, name in enumerate(BIN_NAMES):
            sel = members[name]
            if not bool(sel.any()):
                assert prof[b].isnan().all(), (name, n)
                continue
            naive = acts[:, sel, :].mean(dim=1)  # (L, H)
            assert torch.allclose(prof[b], naive, atol=1e-5), (name, n)


# ── data-dependent gate probes (degenerate inputs, plan §8) ───────────


def test_delta_stats_drop_guard_fails_loud_over_20pct():
    """>20% of pairs with an all-NaN EARLY or LATE set fails LOUD naming the
    cell and the dropped pairs (never a silent shrink)."""
    mags = {f"p{i:02d}": {b: 1.0 for b in BIN_NAMES} for i in range(8)}
    for pid in ("p00", "p01"):  # 2/8 = 25% > 20%: LATE set all-None
        mags[pid] = {**mags[pid], "dec8": None, "dec9": None, "dec10": None}
    with pytest.raises(RuntimeError, match=r"2/8 pairs dropped.*p00.*p01"):
        drv._delta_stats(mags, drv.EARLY_BINS, drv.LATE_BINS, 50, "probe/cell")


def test_delta_stats_named_drop_below_threshold_and_exact_value():
    """<=20% drops are KEPT (named, not fatal); Delta values are exact."""
    mags = {}
    for i in range(10):
        mags[f"p{i:02d}"] = {
            **{b: 1.0 for b in BIN_NAMES},
            "first": math.e**2,
            "tok2_5": math.e**2,  # EARLY mean = e^2, LATE mean = 1 -> Delta = 2
        }
    mags["p00"] = {b: None for b in BIN_NAMES}  # 1/10 = 10% dropped
    out = drv._delta_stats(mags, drv.EARLY_BINS, drv.LATE_BINS, 100, "probe/cell")
    assert out["dropped_pairs"] == ["p00"]
    assert out["n_pairs_kept"] == 9
    assert abs(out["delta_mean"] - 2.0) < 1e-9
    assert out["ci95"][0] <= out["delta_mean"] <= out["ci95"][1]


def test_delta_stats_wilcoxon_degenerate_recorded_not_crashed():
    """Degenerate all-zero deltas never crash the companion: the installed
    scipy returns p=1.0 (RuntimeWarning) on all-zero diffs; older scipys raise
    ValueError, which the except-branch records as `wilcoxon_note`. Either
    way the designed handling executes and a value is recorded."""
    mags = {f"p{i}": {b: 1.0 for b in BIN_NAMES} for i in range(4)}  # Delta = 0 everywhere
    out = drv._delta_stats(mags, drv.EARLY_BINS, drv.LATE_BINS, 50, "probe/zero")
    assert out["delta_mean"] == 0.0
    assert "wilcoxon_p" in out
    assert out["wilcoxon_p"] is None or 0.0 <= out["wilcoxon_p"] <= 1.0


def test_parity_verdict_halt_fires_on_degenerate_cosines():
    """§3.5 HALT: MORE THAN 2 of the spot cells below 0.995 fires; exactly 2
    does not (the '>' boundary); demotions alone never fire."""

    def cells(bad, good):
        rows = [{"cell_id": f"b{i}", "min_cos": 0.5} for i in range(bad)]
        rows += [{"cell_id": f"g{i}", "min_cos": 1.0} for i in range(good)]
        return rows

    assert drv.parity_verdict(cells(3, 7), [])["fired"] is True
    assert drv.parity_verdict(cells(2, 8), [])["fired"] is False
    v = drv.parity_verdict([], [{"cell_id": "x", "reason": "no v_a_mean"}])
    assert v["fired"] is False and v["demoted_to_warn"] is True


def test_capture_all_empty_completions_fails_loud():
    """The all-empty gate (parent convention) fires on degenerate input —
    never a silent skip. Signature-conformant stub boundary: only the model
    forward is unreachable (the assert fires before any forward)."""
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.issue1415.steering import (
        capture_binned_answer_profiles,
    )

    tok = AutoTokenizer.from_pretrained(drv.MODEL_ID)

    class _NeverForward(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

    ctx = {"system": "You are terse.", "user": "Say nothing."}
    with pytest.raises(AssertionError, match="completions empty"):
        capture_binned_answer_profiles(_NeverForward(), tok, ctx, ["", ""], [0])


# ── FULL --tiny e2e (phase chain p0 -> p4, real schema, real tokenizer) ──


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    work = tmp_path_factory.mktemp("pp_tiny")
    drv.main(["--tiny", "--work-root", str(work)])
    return work


def test_tiny_e2e_writes_all_registered_jsons(tiny_run):
    out = tiny_run / "out"
    for name in (
        "per_pair_profiles.json",
        "summary.json",
        "answer_length_distributions.json",
        "null_bands_binned.json",
        "parity_gate_report.json",
        "revisions.json",
    ):
        assert (out / name).exists(), name


def test_tiny_e2e_parity_gate_passes_self_consistent(tiny_run):
    rep = json.loads((tiny_run / "out" / "parity_gate_report.json").read_text())
    assert rep["fired"] is False
    assert rep["n_cells"] == 4  # 2 pairs x 2 arms
    # Self-consistent fp32 CPU parity: the new span-mean path reproduces the
    # parent capture_vectors span mean essentially exactly.
    assert all(r["min_cos"] > 0.9999 for r in rep["cells"])


def test_tiny_e2e_summary_registered_cells(tiny_run):
    s = json.loads((tiny_run / "out" / "summary.json").read_text())
    primary = [c for c in s["cells"] if c["registered_primary"]]
    assert len(primary) == 4  # arm x steer-layer, side by side — never pooled
    for c in primary:
        assert c["n_pairs_kept"] == 2 and c["dropped_pairs"] == []
        assert c["delta_mean"] is not None and len(c["ci95"]) == 2
        assert c["delta_width"]["delta_mean"] is not None
        assert c["delta_floor"]["delta_mean"] is not None
    rep_cells = [c for c in s["cells"] if not c["registered_primary"]]
    assert len(rep_cells) == 4  # rep43/rep44 x 2 arms


def test_tiny_e2e_per_pair_profiles_rows_and_bins(tiny_run):
    pp = json.loads((tiny_run / "out" / "per_pair_profiles.json").read_text())
    prof = pp["profiles"]
    assert len([p for p in prof if p["round"] == "primary"]) == 2 * 2 * 2
    assert len([p for p in prof if p["round"].startswith("rep")]) == 2 * 2 * 2
    for p in prof:
        assert [b["bin"] for b in p["bins"]] == list(BIN_NAMES)
        for b in p["bins"]:
            assert b["magnitude"] is not None  # fixture draws >= 12 tokens: no empty bins
    # matched-layer convention: read layer == steer layer everywhere
    assert all(p["read_layer"] == p["steer_layer"] for p in prof)


def test_tiny_e2e_store_and_mirror_upload(tiny_run):
    tensors = tiny_run / "tensors"
    stored = sorted(tensors.rglob("*.pt"))
    assert len(stored) == 24  # 2 pairs x (2 gen1b + 4 gen1c + 2x3 rep)
    manifest = json.loads((tensors / "manifest.json").read_text())
    assert manifest["n_files"] == 24 and manifest["bins_version"] == drv.BINS_VERSION
    rec = torch.load(stored[0], map_location="cpu", weights_only=True)
    assert rec["profiles"].dtype == torch.float16
    assert rec["profiles"].shape[1] == 13
    assert rec["kept_indices"] == list(range(2))
    # local-mirror upload exercised the identical upload_artifact call path
    mirror = tiny_run / "bulk" / "hf_mirror" / drv.PROFILE_TENSOR_PREFIX
    assert (mirror / "manifest.json").exists()
    assert len(sorted(mirror.rglob("*.pt"))) == 24


def test_tiny_e2e_resume_skips_completed_cells(tiny_run):
    """Second invocation resumes off the manifest (no regime mismatch, no
    recompute crash) — the checkpoint-per-cell contract."""
    manifest_path = tiny_run / "out" / "profile_manifest.json"
    before = json.loads(manifest_path.read_text())
    drv.main(["--tiny", "--work-root", str(tiny_run), "--phase", "p2"])
    after = json.loads(manifest_path.read_text())
    assert set(after["cells"]) == set(before["cells"])


def test_p0_verification_fails_loud_on_context_mismatch(tiny_run, tmp_path):
    """The p0 meta.context == draws.context cross-assert names the offending
    cell_id (degenerate-input probe of the p0 gate on a COPY of the tiny run —
    the module fixture stays untouched)."""
    import shutil

    work = tmp_path / "tampered"
    shutil.copytree(tiny_run, work)
    args = drv.parse_args(["--tiny", "--work-root", str(work)])
    cfg = drv.build_config(args)
    cells = drv.enumerate_cells(cfg)
    victim = cells[0]
    staged = work / "stage" / f"{victim.cell_id}.json"
    blob = json.loads(staged.read_text())
    blob["context"] = {"system": "TAMPERED", "user": "TAMPERED"}
    staged.write_text(json.dumps(blob))
    import re

    with pytest.raises(RuntimeError, match=re.escape(victim.cell_id)):
        drv.phase_p0(cfg, cells)


def test_tiny_e2e_figures_render_from_jsons(tiny_run, tmp_path):
    """The VM figures script renders from the tiny JSONs (scratch fig root —
    smoke outputs never touch committed figures/)."""
    import issue1415_position_profile_figures as figs

    figdir = tmp_path / "figs"
    figs.main(["--in-root", str(tiny_run / "out"), "--fig-root", str(figdir)])
    assert (figdir / "position_profile_hero.png").exists()
    assert (figdir / "position_profile_delta_lattice.png").exists()


# ── 1-file REAL staging probe (artifact-reuse leg (h)(iv)) ────────────


@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="real Hub staging probe requires HF_TOKEN (network test)",
)
def test_real_staging_probe_one_gen1b_file(tmp_path):
    """Download ONE real gen1b draws file at the pinned parent revision
    through the PRODUCTION staging helper (fetch_draws -> hub.stage_hub_file)
    and open it with the PRODUCTION loader. Digest-only assertions — no draw
    text is printed (content-hygiene discipline)."""
    cfg = drv.build_config(drv.parse_args([]))
    cfg.stage_root = tmp_path / "stage"
    cfg.out_root = tmp_path / "out"
    cfg.stage_root.mkdir(parents=True)
    cfg.out_root.mkdir(parents=True)
    cells = drv.enumerate_cells(cfg)
    cell = next(c for c in cells if c.cell_id == "gen1b/cross_00_evil_to_sycophancy/c")
    # production revision resolution (executes the HfApi repo_info branch):
    # the rep short sha resolves to a full 40-char sha and is persisted.
    revisions = drv._resolve_revisions(cfg)
    assert revisions["parent"] == drv.PARENT_REVISION
    assert revisions["rep"].startswith(drv.REP_REVISION_SHORT) and len(revisions["rep"]) == 40
    target = drv.fetch_draws(cfg, cell, revisions)
    assert target == cfg.stage_root / "gen1b/cross_00_evil_to_sycophancy/c.json"
    blob = drv.load_staged_draws(cfg, cell.cell_id)
    assert len(blob["draws"]) == drv.N_DRAWS_FULL
    meta = json.loads(cell.meta_path.read_text())
    assert blob["context"] == meta["context"]
    # production parity-bundle fetch branch + realized-keys check (plan
    # assumption 4): the parent bundle carries v_a_mean at the pinned rev.
    bundle = drv._fetch_parity_bundle(cfg, "cross_00_evil_to_sycophancy", "prefix")
    keys = set(torch.load(bundle, map_location="cpu", mmap=True, weights_only=True))
    assert {"v_a_mean", "layers"} <= keys, sorted(keys)
