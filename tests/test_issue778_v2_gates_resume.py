"""Regression pins for the #778 v2 fail-closed gates + param-keyed resume.

Round-2 revision of the `faithful-extraction-honest-nulls-rerun` follow-up —
these tests pin the three reconciler BLOCKER fixes so a future refactor cannot
silently strip them:

  1. ``fail-closed-guards-skippable`` — `_w2_check` RAISES (never silently
     skips) when its committed reference inputs are missing/mismatched, unless
     the explicit smoke-only flag records a non-production skip.
  2. ``v2-ladder-resume-incomplete`` — the maxlayer/fixed done-predicates are
     keyed on EVERY output-affecting param + the persisted per-draw ``.npy``
     files (Codex's mechanizable check: a stale n_draws=50 output tree must
     NOT satisfy a --draws 10000 run), and `_write_one_file_v2` DROPS a stale
     sibling-stage node instead of merging it through.
  3. ``fwer-headline-partial-output`` — `run_fwer_stage` fail-louds on a
     missing cell / missing per-draw column / draw-count mismatch (writing the
     explicit headline-N/A artifact first), and routes the registered K1-N/A
     carve-out to the labeled headline-N/A artifact instead of a silent
     reduced-cell headline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue778_honest_null_ladder as ladder  # noqa: E402

ALL_FAMS = (*ladder.V2_HONEST_STOCHASTIC, *ladder.V2_REFERENCE)


# ── helpers ────────────────────────────────────────────────────────────────────


def _mk_maxlayer_tree(
    eval_root: Path,
    maxdraws_root: Path,
    *,
    trait: str = "evil",
    setting: str = "finetune",
    n_draws: int = 50,
    n_draws_orig: int = 1000,
    lam: float = 0.1,
    allow_gate_skip: bool = False,
    npy_n_override: int | None = None,
    with_params: bool = True,
) -> Path:
    out_dir = eval_root / ladder.V2_LABEL
    out_dir.mkdir(parents=True, exist_ok=True)
    maxdraws_root.mkdir(parents=True, exist_ok=True)
    stage: dict = {}
    for regime in ladder.REGIMES[setting]:
        nulls = {}
        for fam in ALL_FAMS:
            want_n = n_draws_orig if fam in ladder.V2_REFERENCE else n_draws
            nulls[fam] = {"n_draws": want_n}
            np.save(
                maxdraws_root / f"{trait}_{setting}_{regime}_{fam}_maxdraws.npy",
                np.zeros(npy_n_override or want_n, dtype=np.float32),
            )
        stage[regime] = {"nulls": nulls}
    data: dict = {"rb_version": "v2", "stage_maxlayer": stage}
    if with_params:
        data["stage_maxlayer_params"] = {
            "n_draws": n_draws,
            "n_draws_orig": n_draws_orig,
            "lambda_primary": lam,
            "rb_version": "v2",
            "allow_gate_skip_smoke_only": allow_gate_skip,
        }
    path = out_dir / f"{trait}_{setting}_honestnulls_v2.json"
    path.write_text(json.dumps(data))
    return path


def _mk_fixed_tree(
    eval_root: Path,
    maxdraws_root: Path,
    *,
    traits: tuple[str, ...] = ("evil", "sycophancy", "hallucination"),
    n_draws: int = 50,
) -> None:
    out_dir = eval_root / ladder.V2_LABEL
    out_dir.mkdir(parents=True, exist_ok=True)
    maxdraws_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for trait in traits:
        layer = ladder.PAPER_STEERING_V2[trait]
        for setting in ladder.HEADLINE_SETTINGS:
            fkey = f"{trait}_{setting}"
            sf: dict = {}
            for regime in ladder.REGIMES[setting]:
                nulls = {}
                for fam in ladder.V2_HONEST_STOCHASTIC:
                    col = rng.uniform(size=n_draws)
                    np.save(
                        maxdraws_root
                        / f"{fkey}_{regime}_{fam}_fixed_paper_steering_L{layer}_draws.npy",
                        col.astype(np.float32),
                    )
                    nulls[fam] = {"raw_p": 0.2, "n_draws": n_draws}
                sf[regime] = {"per_choice": {"paper_steering": {"layer": layer, "nulls": nulls}}}
            data = {"rb_version": "v2", "primary_family": "within_class", "stage_fixed": sf}
            (out_dir / f"{fkey}_honestnulls_v2.json").write_text(json.dumps(data))


# ── 1. fail-closed-guards-skippable: _w2_check ─────────────────────────────────


def test_w2_raises_on_missing_committed_file(tmp_path):
    with pytest.raises(RuntimeError, match="W2 GATE UNARMED"):
        ladder._w2_check(tmp_path, "evil", "finetune", "overall", "orig_randnorm", np.ones(5))


def test_w2_raises_on_missing_committed_node(tmp_path):
    (tmp_path / "evil_finetune_nullbattery.json").write_text(json.dumps({"nulls": {}}))
    with pytest.raises(RuntimeError, match="W2 GATE UNARMED"):
        ladder._w2_check(tmp_path, "evil", "finetune", "overall", "orig_randnorm", np.ones(5))


def test_w2_raises_on_draw_count_mismatch(tmp_path):
    committed = {"nulls": {"randnorm": {"draws_max_abs": [0.1, 0.2, 0.3]}}}
    (tmp_path / "evil_finetune_nullbattery.json").write_text(json.dumps(committed))
    with pytest.raises(RuntimeError, match="W2 GATE UNARMED"):
        ladder._w2_check(tmp_path, "evil", "finetune", "overall", "orig_randnorm", np.ones(5))


def test_w2_smoke_flag_records_non_production_skip(tmp_path):
    out = ladder._w2_check(
        tmp_path, "evil", "finetune", "overall", "orig_randnorm", np.ones(5), allow_gate_skip=True
    )
    assert out["status"] == "skipped_smoke_only"
    assert out["non_production"] is True


# ── 2. v2-ladder-resume-incomplete: param-keyed done predicates + merge ────────


def test_stale_50_draw_maxlayer_not_done_for_10000(tmp_path):
    """Codex's mechanizable check: a stale n_draws=50 JSON in the output tree
    must NOT be treated as done by a --draws 10000 run."""
    ev, md = tmp_path / "eval", tmp_path / "md"
    _mk_maxlayer_tree(ev, md, n_draws=50)
    assert not ladder._maxlayer_cell_done_v2(
        ev,
        md,
        "evil",
        "finetune",
        n_draws=10000,
        n_draws_orig=1000,
        lam=0.1,
        allow_gate_skip=False,
    )
    # The SAME tree IS done for a matching 50-draw request (same params).
    assert ladder._maxlayer_cell_done_v2(
        ev,
        md,
        "evil",
        "finetune",
        n_draws=50,
        n_draws_orig=1000,
        lam=0.1,
        allow_gate_skip=False,
    )


def test_maxlayer_not_done_on_lambda_or_gate_mode_mismatch(tmp_path):
    ev, md = tmp_path / "eval", tmp_path / "md"
    _mk_maxlayer_tree(ev, md, n_draws=50, lam=0.1, allow_gate_skip=True)
    assert not ladder._maxlayer_cell_done_v2(
        ev,
        md,
        "evil",
        "finetune",
        n_draws=50,
        n_draws_orig=1000,
        lam=0.2,
        allow_gate_skip=True,
    )
    # A smoke-flagged output never satisfies a production (fail-closed) run.
    assert not ladder._maxlayer_cell_done_v2(
        ev,
        md,
        "evil",
        "finetune",
        n_draws=50,
        n_draws_orig=1000,
        lam=0.1,
        allow_gate_skip=False,
    )


def test_maxlayer_not_done_on_missing_or_short_npy(tmp_path):
    ev, md = tmp_path / "eval", tmp_path / "md"
    _mk_maxlayer_tree(ev, md, n_draws=50)
    victim = md / "evil_finetune_overall_isotropic_maxdraws.npy"
    np.save(victim, np.zeros(10, dtype=np.float32))  # truncated file
    assert not ladder._maxlayer_cell_done_v2(
        ev,
        md,
        "evil",
        "finetune",
        n_draws=50,
        n_draws_orig=1000,
        lam=0.1,
        allow_gate_skip=False,
    )
    victim.unlink()  # missing file
    assert not ladder._maxlayer_cell_done_v2(
        ev,
        md,
        "evil",
        "finetune",
        n_draws=50,
        n_draws_orig=1000,
        lam=0.1,
        allow_gate_skip=False,
    )


def test_maxlayer_paramless_json_never_done(tmp_path):
    """A pre-fix JSON (no stage_maxlayer_params node) is never resumed."""
    ev, md = tmp_path / "eval", tmp_path / "md"
    _mk_maxlayer_tree(ev, md, n_draws=50, with_params=False)
    assert not ladder._maxlayer_cell_done_v2(
        ev,
        md,
        "evil",
        "finetune",
        n_draws=50,
        n_draws_orig=1000,
        lam=0.1,
        allow_gate_skip=False,
    )


def test_write_one_file_v2_drops_stale_sibling_stage(tmp_path):
    """A stale 50-draw stage_maxlayer on disk must NOT survive a production
    fixed-stage rewrite of the same file (the merge-through hole)."""
    ev = tmp_path / "eval"
    out_dir = ev / ladder.V2_LABEL
    out_dir.mkdir(parents=True)
    disk = {
        "rb_version": "v2",
        "stage_maxlayer": {"overall": {"nulls": {"isotropic": {"n_draws": 50}}}},
        "stage_maxlayer_params": {
            "n_draws": 50,
            "n_draws_orig": 1000,
            "lambda_primary": 0.1,
            "rb_version": "v2",
            "allow_gate_skip_smoke_only": True,
        },
        "seeds_maxlayer": {"overall": {"isotropic": 100_000}},
    }
    (out_dir / "evil_finetune_honestnulls_v2.json").write_text(json.dumps(disk))
    fd = {
        "rb_version": "v2",
        "stage_fixed": {"overall": {"per_choice": {}}},
        "stage_fixed_params": {
            "n_draws": 10000,
            "n_draws_orig": 1000,
            "n_boot": 1000,
            "lambda_primary": 0.1,
            "rb_version": "v2",
            "lam_sweep": True,
            "allow_gate_skip_smoke_only": False,
        },
    }
    ladder._write_one_file_v2("evil_finetune", fd, ev)
    merged = json.loads((out_dir / "evil_finetune_honestnulls_v2.json").read_text())
    assert "stage_maxlayer" not in merged
    assert "stage_maxlayer_params" not in merged
    assert "seeds_maxlayer" not in merged
    assert "stage_fixed" in merged


def test_write_one_file_v2_preserves_matching_sibling_stage(tmp_path):
    """A sibling stage produced under the SAME params IS preserved (the merge
    exists so fixed + maxlayer compose into one file)."""
    ev = tmp_path / "eval"
    out_dir = ev / ladder.V2_LABEL
    out_dir.mkdir(parents=True)
    shared = {
        "n_draws": 10000,
        "n_draws_orig": 1000,
        "lambda_primary": 0.1,
        "rb_version": "v2",
        "allow_gate_skip_smoke_only": False,
    }
    disk = {
        "rb_version": "v2",
        "stage_maxlayer": {"overall": {"nulls": {"isotropic": {"n_draws": 10000}}}},
        "stage_maxlayer_params": dict(shared),
        "seeds_maxlayer": {"overall": {"isotropic": 100_000}},
    }
    (out_dir / "evil_finetune_honestnulls_v2.json").write_text(json.dumps(disk))
    fd = {
        "rb_version": "v2",
        "stage_fixed": {"overall": {"per_choice": {}}},
        "stage_fixed_params": {**shared, "n_boot": 1000, "lam_sweep": True},
    }
    ladder._write_one_file_v2("evil_finetune", fd, ev)
    merged = json.loads((out_dir / "evil_finetune_honestnulls_v2.json").read_text())
    assert "stage_maxlayer" in merged
    assert "stage_fixed" in merged


# ── 3. fwer-headline-partial-output: fail-loud + explicit headline-N/A ─────────

TRAITS3 = ("evil", "sycophancy", "hallucination")


def _na_artifact(eval_root: Path) -> dict:
    return json.loads((eval_root / ladder.V2_LABEL / "fwer_headline_v2.json").read_text())


def test_fwer_happy_path_12_cells(tmp_path):
    ev, md = tmp_path / "eval", tmp_path / "md"
    _mk_fixed_tree(ev, md)
    out = ladder.run_fwer_stage(ev, md, TRAITS3, 50)
    assert out["n_headline_cells"] == 12
    assert "status" not in out or out.get("status") == "ok"
    for fam in (*ladder.V2_HONEST_STOCHASTIC, "primary_mixed"):
        assert out["families"][fam]["n_draws"] == 50
        assert 0.0 < out["families"][fam]["fwer_adjusted_p"] <= 1.0


def test_fwer_raises_on_missing_cell_json(tmp_path):
    ev, md = tmp_path / "eval", tmp_path / "md"
    _mk_fixed_tree(ev, md)
    (ev / ladder.V2_LABEL / "evil_monitoring_manyshot_honestnulls_v2.json").unlink()
    with pytest.raises(RuntimeError, match="headline inputs incomplete"):
        ladder.run_fwer_stage(ev, md, TRAITS3, 50)
    assert _na_artifact(ev)["status"] == "headline_NA"


def test_fwer_raises_on_missing_per_draw_column(tmp_path):
    """Codex's mechanizable check: remove one fixed paper-steering .npy in the
    output tree -> --stage fwer must exit non-zero (RuntimeError propagates to
    a non-zero exit through main())."""
    ev, md = tmp_path / "eval", tmp_path / "md"
    _mk_fixed_tree(ev, md)
    layer = ladder.PAPER_STEERING_V2["evil"]
    victim = (
        md / f"evil_monitoring_corrected_overall_isotropic_fixed_paper_steering_L{layer}_draws.npy"
    )
    victim.unlink()
    with pytest.raises(RuntimeError, match="per-draw column missing"):
        ladder.run_fwer_stage(ev, md, TRAITS3, 50)
    assert _na_artifact(ev)["status"] == "headline_NA"


def test_fwer_raises_on_stale_draw_count(tmp_path):
    """A 50-draw tree consumed by a production 10000-draw fwer run fails loud
    (JSON n_draws mismatch), never a silently-truncated joint null."""
    ev, md = tmp_path / "eval", tmp_path / "md"
    _mk_fixed_tree(ev, md, n_draws=50)
    with pytest.raises(RuntimeError, match="draw-count mismatches"):
        ladder.run_fwer_stage(ev, md, TRAITS3, 10000)
    na = _na_artifact(ev)
    assert na["status"] == "headline_NA"
    assert na["draw_count_mismatches"]


def test_fwer_raises_on_stale_column_length(tmp_path):
    """A column whose .npy length disagrees with the (matching) JSON n_draws is
    a stale artifact — fail loud, never min()-truncate the joint null."""
    ev, md = tmp_path / "eval", tmp_path / "md"
    _mk_fixed_tree(ev, md, n_draws=50)
    layer = ladder.PAPER_STEERING_V2["evil"]
    victim = (
        md / f"evil_monitoring_corrected_overall_isotropic_fixed_paper_steering_L{layer}_draws.npy"
    )
    np.save(victim, np.zeros(10, dtype=np.float32))
    with pytest.raises(RuntimeError, match="stale column"):
        ladder.run_fwer_stage(ev, md, TRAITS3, 50)
    assert _na_artifact(ev)["status"] == "headline_NA"


def test_fwer_trait_subset_requires_smoke_flag(tmp_path):
    ev, md = tmp_path / "eval", tmp_path / "md"
    _mk_fixed_tree(ev, md, traits=("evil",))
    with pytest.raises(RuntimeError, match="EXACTLY 12"):
        ladder.run_fwer_stage(ev, md, ("evil",), 50)
    out = ladder.run_fwer_stage(ev, md, ("evil",), 50, allow_gate_skip=True)
    assert out["n_headline_cells"] == 4
    assert out["allow_gate_skip_smoke_only"] is True


def test_fwer_k1_na_routes_to_explicit_na_artifact(tmp_path):
    """The registered K1-N/A carve-out: a labeled headline-N/A artifact, run
    continues (no raise) — NEVER a silent 8-cell headline."""
    ev, md = tmp_path / "eval", tmp_path / "md"
    _mk_fixed_tree(ev, md, traits=("sycophancy", "hallucination"))
    (ev / ladder.V2_LABEL / "evil_NA_honestnulls_v2.json").write_text(
        json.dumps({"trait": "evil", "rb_version": "v2", "status": "NA — K1 < 5 kept pairs"})
    )
    out = ladder.run_fwer_stage(ev, md, TRAITS3, 50)
    assert out["status"] == "headline_NA"
    assert set(out["k1_na_cells"]) == {"evil_monitoring_corrected", "evil_monitoring_manyshot"}
    assert out.get("families") in ({}, None)  # no partial per-family headline published
    assert _na_artifact(ev)["status"] == "headline_NA"
