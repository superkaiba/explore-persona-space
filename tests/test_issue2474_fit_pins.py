"""Regression pins for issue-2474's fit driver (round-2/round-3 BLOCKER fixes).

Pins (each fails on the pre-fix code, passes post-fix):
  * ``_scores_fingerprint`` keys bundles on PARENT-RELATIVE paths, so per-condition
    ``mu.pt`` files can never collide onto one dict entry (r1 g1 Major 2 /
    Codex ``score-fingerprint-collision``), and the cardinality assert fires.
  * ``SCORE_RECIPE_TAG`` rides BOTH score-resume grains (r2 Codex Major 2):
    changing the scoring-recipe identity invalidates the setting-grain AND the
    layer-grain fingerprints — the layer grain independently, even against a
    stale setting_fp.
  * ``_ceiling_cell_accounting`` counts ABSENT cells as zero kept (r1 Codex
    ``harvest-zero-cell-gap``), reconciles kept + dropped == slots, and refuses
    out-of-range cell indices.
  * ``_assert_close_banked`` raises on a NaN recompute (r1 g1 Minor: the old
    ``abs(a-b) > tol`` form is False on NaN, silently PASSing drift).
  * ``_write_done_sentinel`` emits a poll_pipeline-conformant envelope with the
    INTEGER ``version: 1`` (r2 Critical, both reviewers: ``version: null``
    parses but crashes the drain's ``int(data["version"])`` on every tick) —
    round-tripped through the REAL ``poll_pipeline._parse_sentinel`` AND the
    REAL ``poll_pipeline._post_drained_sentinel`` (post boundary autospec'd) —
    and routes smoke runs to the SMOKE tree, never /workspace/logs (r1 Major 1).
  * ``phase_upload`` orders upload(analysis) -> upload(tensors) ->
    verify(analysis) -> verify(tensors) -> sentinel, and a verify miss raises
    with NO sentinel write (r2 Codex Major 3: the r2 pin never exercised the
    orchestration); a completed upload SKIPS re-upload (r2 Minors).
  * ``_harvest_completion_fp`` binds the harvest skip to the remote CONTENT
    identity (r2 Codex Major 4): mutating one sidecar's identity while
    preserving paths defeats the completion skip.
  * ``_validate_stats_schema`` enforces the full (setting, variant, family,
    condition, layer, DV) Cartesian grain with per-layer rho + Pearson twin
    curves (r2 Codex Major 1).
  * ``PHASES`` registry membership (the smoke-architecture arm set of record).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import issue2474_fit as fit


def _touch(p: Path, payload: bytes = b"x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(payload)
    return p


# ---------------------------------------------------------------------------
# _scores_fingerprint — parent-relative keys, no per-condition mu.pt collision
# ---------------------------------------------------------------------------
def _fp_fixture(tmp_path: Path) -> tuple[dict, argparse.Namespace]:
    base = tmp_path / "capture" / "predictor_captures"
    _touch(base / "base_em" / "grid.pt")
    _touch(base / "base_em" / "ceiling.pt")
    _touch(base / "base_mu_condA" / "mu.pt", b"aaa")
    _touch(base / "base_mu_condB" / "mu.pt", b"bbbb")
    comp_dir = tmp_path / "comps"
    _touch(comp_dir / "base_L00.npz")
    cfg = {
        "capture_dir": tmp_path / "capture",
        "conds": {"em": ("condA", "condB")},
        "comp_dir": str(comp_dir),
    }
    return cfg, argparse.Namespace(parent_sha="deadbeef")


def test_scores_fingerprint_parent_relative_keys_no_collision(tmp_path):
    cfg, args = _fp_fixture(tmp_path)
    fp = fit._scores_fingerprint(cfg, "em", args)
    # Cardinality: 2 + n_mu_bundles UNIQUE entries (the r1 bare-filename keys
    # collapsed both mu.pt files onto ONE "mu.pt" entry -> len == 3, not 4).
    assert len(fp["bundles"]) == 4, sorted(fp["bundles"])
    assert "base_mu_condA/mu.pt" in fp["bundles"]
    assert "base_mu_condB/mu.pt" in fp["bundles"]
    assert fp["bundles"]["base_mu_condA/mu.pt"] != fp["bundles"]["base_mu_condB/mu.pt"]
    assert fp["v"] == 3
    assert fp["score_recipe"] == fit.SCORE_RECIPE_TAG
    assert "base_L00.npz" in fp["components"]


def test_score_recipe_tag_invalidates_both_resume_grains(tmp_path, monkeypatch):
    """r2 Codex Major 2: varying the scoring-recipe identity must defeat BOTH
    resume grains — the setting fingerprint AND the layer fingerprint (the
    latter even against a stale/unchanged setting_fp)."""
    cfg, args = _fp_fixture(tmp_path)
    comp = Path(cfg["comp_dir"]) / "base_L00.npz"

    setting_fp_a = fit._scores_fingerprint(cfg, "em", args)
    layer_fp_a = fit._layer_fingerprint(setting_fp_a, 0, comp)

    monkeypatch.setattr(fit, "SCORE_RECIPE_TAG", "test-recipe-CHANGED")
    setting_fp_b = fit._scores_fingerprint(cfg, "em", args)
    # Layer grain re-keyed with the ORIGINAL (stale) setting_fp: the layer
    # fingerprint must STILL change on a recipe change.
    layer_fp_b = fit._layer_fingerprint(setting_fp_a, 0, comp)

    assert setting_fp_a != setting_fp_b, "setting-grain resume survived a recipe change"
    assert layer_fp_a != layer_fp_b, "layer-grain resume survived a recipe change"


# ---------------------------------------------------------------------------
# _ceiling_cell_accounting — absent cells count zero; reconciliation asserts
# ---------------------------------------------------------------------------
def _meta(cells_kept: dict[int, int]) -> list[dict]:
    rows = []
    for ci, n in cells_kept.items():
        rows += [{"cell_idx": ci}] * n
    return rows


def test_ceiling_accounting_wholly_absent_cell_trips_floor():
    # 4 cells x 3 rollouts = 12 slots; cell 3 wholly absent (its 3 slots dropped).
    with pytest.raises(RuntimeError, match="min kept/cell 0"):
        fit._ceiling_cell_accounting(
            _meta({0: 3, 1: 3, 2: 3}),
            n_cells_expected=4,
            n_rollouts_expected=3,
            drop_stats={"n_slots": 12, "n_empty_after_retries": 3, "n_capture_dropped": 0},
            max_rows=12,
            min_kept_per_cell=2,
            max_drop_frac=0.5,
            ctx="test",
        )


def test_ceiling_accounting_reconcile_failure_raises():
    with pytest.raises(RuntimeError, match="does not reconcile"):
        fit._ceiling_cell_accounting(
            _meta({0: 3, 1: 3, 2: 3, 3: 2}),
            n_cells_expected=4,
            n_rollouts_expected=3,
            drop_stats={"n_slots": 12, "n_empty_after_retries": 0, "n_capture_dropped": 0},
            max_rows=12,
            min_kept_per_cell=2,
            max_drop_frac=0.5,
            ctx="test",
        )


def test_ceiling_accounting_out_of_range_cell_raises():
    with pytest.raises(RuntimeError, match="outside the expected cell set"):
        fit._ceiling_cell_accounting(
            _meta({0: 3, 7: 3}),
            n_cells_expected=4,
            n_rollouts_expected=3,
            drop_stats={"n_slots": 12, "n_empty_after_retries": 6, "n_capture_dropped": 0},
            max_rows=12,
            min_kept_per_cell=2,
            max_drop_frac=0.6,
            ctx="test",
        )


def test_ceiling_accounting_happy_path():
    out = fit._ceiling_cell_accounting(
        _meta({0: 3, 1: 2, 2: 3, 3: 3}),
        n_cells_expected=4,
        n_rollouts_expected=3,
        drop_stats={"n_slots": 12, "n_empty_after_retries": 1, "n_capture_dropped": 0},
        max_rows=12,
        min_kept_per_cell=2,
        max_drop_frac=0.5,
        ctx="test",
    )
    assert out == {
        "n_kept_rows": 11,
        "n_slots": 12,
        "n_dropped_total": 1,
        "min_kept_per_cell": 2,
        "n_cells_expected": 4,
        "n_absent_cells": 0,
    }


# ---------------------------------------------------------------------------
# _assert_close_banked — NaN-safe recompute assert
# ---------------------------------------------------------------------------
def test_assert_close_banked_nan_raises():
    with pytest.raises(RuntimeError, match="provenance drift"):
        fit._assert_close_banked(math.nan, 1.0, "test/nan")


def test_assert_close_banked_close_passes_and_far_raises():
    fit._assert_close_banked(1.0 + 5e-7, 1.0, "test/close")
    with pytest.raises(RuntimeError):
        fit._assert_close_banked(1.1, 1.0, "test/far")


# ---------------------------------------------------------------------------
# Done sentinel — poll_pipeline-conformant envelope + smoke-tree routing +
# the REAL drain path (r2 Critical: version must be an INTEGER — null parses
# but crashes _post_drained_sentinel's int(data["version"]) on every tick)
# ---------------------------------------------------------------------------
def _sentinel_args(**over):
    ns = argparse.Namespace(log_dir=None)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_smoke_sentinel_routes_to_smoke_tree_and_parses(tmp_path):
    import poll_pipeline  # scripts/ is on sys.path

    cfg = {"synthetic": True, "data_root": tmp_path}
    fit._write_done_sentinel(_sentinel_args(), cfg, ["out/a.json"])
    p = tmp_path / "logs" / "issue-2474-fit-smoke.done.json"
    assert p.is_file(), "smoke sentinel must land under the SMOKE tree (r1 Major 1)"
    payload = json.loads(p.read_text())
    for k in poll_pipeline._SENTINEL_REQUIRED_KEYS:
        assert k in payload, f"missing poll_pipeline required key {k!r}"
    assert payload["sentinel_schema_version"] == poll_pipeline.SENTINEL_SCHEMA_VERSION_SUPPORTED
    assert payload["kind"] == "epm:progress"
    # r2 Critical: the drain runs int(data["version"]) whenever the key is
    # present — the envelope must carry a REAL int (type-is-int, not bool).
    assert type(payload["version"]) is int
    assert payload["version"] == 1
    parsed = poll_pipeline._parse_sentinel(str(p), p.read_text())
    assert isinstance(parsed, dict), "REAL poller _parse_sentinel must accept the envelope"


def test_sentinel_survives_real_drain_post_path(tmp_path, monkeypatch):
    """r2 Critical (both reviewers): feed the produced payload through the REAL
    ``_parse_sentinel`` AND the REAL ``_post_drained_sentinel`` with the post
    boundary autospec'd — no exception, successful processing, version posted
    as the integer 1 (epm:progress posts verbatim; #1095 rewrites only
    epm:results)."""
    import poll_pipeline

    cfg = {"synthetic": True, "data_root": tmp_path}
    fit._write_done_sentinel(_sentinel_args(), cfg, ["out/a.json"])
    p = tmp_path / "logs" / "issue-2474-fit-smoke.done.json"
    parsed = poll_pipeline._parse_sentinel(str(p), p.read_text())
    assert isinstance(parsed, dict)

    fake_post = mock.create_autospec(poll_pipeline.post_event)
    monkeypatch.setattr(poll_pipeline, "post_event", fake_post)
    rec = poll_pipeline._post_drained_sentinel(
        issue=2474, remote_path=str(p), data=parsed, fp="testfp123"
    )
    assert rec is not None, "_post_drained_sentinel must return the drain fp record"
    assert rec["kind"] == "epm:progress"
    fake_post.assert_called_once()
    kwargs = fake_post.call_args.kwargs
    assert type(kwargs["version"]) is int and kwargs["version"] == 1
    assert fake_post.call_args.args[:2] == (2474, "epm:progress")


def test_production_sentinel_honors_log_dir(tmp_path):
    cfg = {"synthetic": False, "data_root": tmp_path}
    fit._write_done_sentinel(_sentinel_args(log_dir=str(tmp_path / "plogs")), cfg, ["x"])
    p = tmp_path / "plogs" / "issue-2474-fit.done.json"
    assert p.is_file()
    payload = json.loads(p.read_text())
    assert payload["sentinel_schema_version"] == 1
    assert type(payload["version"]) is int and payload["version"] == 1


# ---------------------------------------------------------------------------
# phase_upload — upload -> verify -> sentinel ordering (r2 Codex Major 3) +
# verify-miss suppresses the sentinel + completion skip (r2 Minors)
# ---------------------------------------------------------------------------
def _upload_fixture(tmp_path: Path) -> tuple[argparse.Namespace, dict]:
    out_dir = tmp_path / "out"
    tdir = tmp_path / "tensors"
    (out_dir / "scores_partial").mkdir(parents=True)
    (tdir / "perdraw").mkdir(parents=True)
    (out_dir / "prefit_scores.json").write_text("{}")
    (out_dir / "prefit_stats.json").write_text("{}")
    (tdir / "perdraw" / "perdraw_em_full.npz").write_bytes(b"z")
    cfg = {"synthetic": False, "out_dir": out_dir, "tensors_out": tdir, "data_root": tmp_path}
    args = argparse.Namespace(upload_dry_run=False, force=False, log_dir=str(tmp_path / "plogs"))
    return args, cfg


def _wire_upload_fakes(monkeypatch, calls: list, *, missing_for: set[str] = frozenset()):
    """Order-recording fakes mirroring the REAL Hub-boundary signatures
    (code-style: signature-conformant by construction — def mirrors the real
    def; the production body of phase_upload runs unstubbed)."""
    from explore_persona_space.orchestrate import hub

    def fake_upload_dataset(data_path, repo_id=hub.DEFAULT_DATASET_REPO, path_in_repo=""):
        calls.append(("upload", path_in_repo))
        return f"hf://{repo_id}/{path_in_repo}"

    def fake_verify(
        api, repo_id, expected_repo_paths, *, path_in_repo, repo_type="dataset", revision=None
    ):
        calls.append(("verify", path_in_repo))
        return list(expected_repo_paths) if path_in_repo in missing_for else []

    def fake_sentinel(args, cfg, outputs):
        calls.append(("sentinel", None))

    monkeypatch.setattr(hub, "upload_dataset", fake_upload_dataset)
    monkeypatch.setattr(hub, "verify_repo_paths_uploaded", fake_verify)
    monkeypatch.setattr(fit, "_write_done_sentinel", fake_sentinel)


def test_upload_orders_upload_verify_sentinel(tmp_path, monkeypatch):
    args, cfg = _upload_fixture(tmp_path)
    calls: list = []
    _wire_upload_fakes(monkeypatch, calls)
    report = fit.phase_upload(args, cfg)
    assert calls == [
        ("upload", fit.HF_ANALYSIS_PREFIX),
        ("upload", fit.HF_TENSORS_PREFIX),
        ("verify", fit.HF_ANALYSIS_PREFIX),
        ("verify", fit.HF_TENSORS_PREFIX),
        ("sentinel", None),
    ], calls
    assert (cfg["out_dir"] / "upload_report_2474.json").is_file()
    assert report["analysis_paths"] and report["tensor_paths"]


def test_upload_verify_miss_raises_and_writes_no_sentinel(tmp_path, monkeypatch):
    args, cfg = _upload_fixture(tmp_path)
    calls: list = []
    _wire_upload_fakes(monkeypatch, calls, missing_for={fit.HF_ANALYSIS_PREFIX})
    with pytest.raises(RuntimeError, match="MISSING"):
        fit.phase_upload(args, cfg)
    assert ("sentinel", None) not in calls, "verify miss must suppress the done sentinel"
    assert not (cfg["out_dir"] / "upload_report_2474.json").exists(), (
        "verify miss must not leave a completed upload report"
    )


def test_upload_completion_skip_is_idempotent(tmp_path, monkeypatch):
    args, cfg = _upload_fixture(tmp_path)
    calls: list = []
    _wire_upload_fakes(monkeypatch, calls)
    fit.phase_upload(args, cfg)
    n_first = len(calls)
    calls.clear()
    # Second run with an unchanged local artifact set: NO hub calls; the
    # sentinel is not re-written either — the skip branch probes the resolved
    # sentinel dest, and the mocked first run recorded a sentinel call without
    # writing a file, so at most one ("sentinel", None) heal entry may appear;
    # upload/verify must NOT.
    fit.phase_upload(args, cfg)
    assert n_first == 5
    assert all(kind == "sentinel" for kind, _ in calls), calls


# ---------------------------------------------------------------------------
# Harvest completion skip — bound to remote CONTENT identity (r2 Codex Major 4)
# ---------------------------------------------------------------------------
def test_harvest_completion_identity_binding(tmp_path):
    cfg = {"settings": ("em",), "conds": {"em": ("condA",)}}
    args = argparse.Namespace(parent_sha="deadbeef")
    ident_a = {
        "p/grid.pt": {"size": 10, "blob_id": "aa", "lfs_sha256": "s1"},
        "p/grid.pt.meta.json": {"size": 3, "blob_id": "bb", "lfs_sha256": None},
    }
    # Mutate ONE sidecar's identity while preserving the path set.
    ident_b = copy.deepcopy(ident_a)
    ident_b["p/grid.pt.meta.json"]["blob_id"] = "cc"

    fp_a = fit._harvest_completion_fp(args, cfg, ident_a)
    fp_b = fit._harvest_completion_fp(args, cfg, ident_b)
    assert fp_a != fp_b, "completion fp must bind the remote content identity"

    out = tmp_path / "harvest_verified.json"
    out.write_text(json.dumps({"verdict": "PASS", "completion_fingerprint": fp_a}))
    assert fit._phase_completed(out, fp_a, False, "harvest") is not None
    assert fit._phase_completed(out, fp_b, False, "harvest") is None, (
        "a mutated sidecar identity under the same paths must DEFEAT the harvest skip"
    )


# ---------------------------------------------------------------------------
# Stats schema — full (setting, variant, family, condition, layer, DV) grain
# with per-layer rho + Pearson twins (r2 Codex Major 1)
# ---------------------------------------------------------------------------
def _mk_stats(n_l: int = 2, conds: tuple[str, ...] = ("condA", "condB")) -> dict:
    def dv_entry() -> dict:
        return {
            "rho_by_layer": [0.1] * n_l,
            "pearson_by_layer": [0.2] * n_l,
            "pinned_rho": 0.1,
            "pinned_pearson": 0.2,
            "pinned_ci95": [0.0, 0.3],
        }

    def fam() -> dict:
        return {
            "pooled": {
                dv: {
                    "rho_by_layer": [0.1] * n_l,
                    "pearson_by_layer": [0.2] * n_l,
                    "ci95_by_layer": [[0.0, 0.3]] * n_l,
                    "pinned": {"layer": 0, "rho": 0.1, "pearson": 0.2, "ci95": [0.0, 0.3]},
                }
                for dv in ("level", "change")
            },
            "per_condition": {
                c: {"level": dv_entry(), "change": dv_entry(), "cont_pinned_rho": 0.0}
                for c in conds
            },
        }

    return {
        "settings": {
            "em": {
                "pinned_layer": 0,
                "n_layers": n_l,
                "conditions": list(conds),
                "variants": {
                    "full": {"families": {"ctx_sameq": fam()}},
                    "loo": {"families": {"ctx_sameq": fam()}},
                },
            }
        }
    }


def test_stats_schema_validator_passes_on_full_cartesian():
    fit._validate_stats_schema(_mk_stats())


def test_stats_schema_validator_rejects_short_condition_curve():
    bad = _mk_stats()
    fam = bad["settings"]["em"]["variants"]["full"]["families"]["ctx_sameq"]
    fam["per_condition"]["condB"]["change"]["rho_by_layer"] = [0.1]  # length 1 != n_l 2
    with pytest.raises(RuntimeError, match=r"condB: change\.rho_by_layer"):
        fit._validate_stats_schema(bad)


def test_stats_schema_validator_rejects_missing_pearson_twin():
    bad = _mk_stats()
    fam = bad["settings"]["em"]["variants"]["loo"]["families"]["ctx_sameq"]
    del fam["per_condition"]["condA"]["level"]["pearson_by_layer"]
    with pytest.raises(RuntimeError, match=r"condA: level\.pearson_by_layer"):
        fit._validate_stats_schema(bad)


def test_stats_schema_validator_rejects_missing_condition():
    bad = _mk_stats()
    fam = bad["settings"]["em"]["variants"]["full"]["families"]["ctx_sameq"]
    del fam["per_condition"]["condB"]
    with pytest.raises(RuntimeError, match="per_condition missing"):
        fit._validate_stats_schema(bad)


# ---------------------------------------------------------------------------
# PHASES registry — the smoke-architecture arm set of record
# ---------------------------------------------------------------------------
def test_phases_registry_members():
    assert sorted(fit.PHASES) == [
        "all",
        "harvest-verify",
        "pilot",
        "refit",
        "scores",
        "smoke",
        "stats",
        "upload",
    ]
