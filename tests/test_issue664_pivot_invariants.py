"""Issue #664 round-6 invariant pins -- below-floor-yield-fleet-crash.

Pins the substantive BLOCKER fixed this round so a future refactor cannot silently
strip it (the un-CI-pinned-assertion class):

- **A below-80%-yield source DROP is GRACEFUL, not a fleet crash.** Plan v4 §11
  mandates graceful degradation: a source below the on-policy yield floor after the
  retry budget is DROPPED + reported as a finding, never backfilled with templates,
  and never a fatal crash. Pre-fix, ``_enforce_yield_floor`` raised ``SystemExit(msg)``
  (exit code 1) and the dispatcher's ``subprocess.run(cmd, check=True)`` turned that
  into a fleet-killing ``CalledProcessError``. The fix: the builder exits with the
  dedicated ``DROPPED_SOURCE_EXIT`` (3) + writes a per-cell drop sentinel; the
  dispatcher treats rc==3 as "skip this cell, continue" (rc!=0 stays fatal) and
  excludes dropped cells from every downstream phase + upload.

All CPU-only: imports ``scripts/issue664_*`` and exercises the pure-Python drop
logic, stubbing the single HF/Hub touch points so no GPU / network is required.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue664_build_training_data as B  # noqa: E402
import issue664_common as C  # noqa: E402
import issue664_dispatch as D  # noqa: E402


# ── builder: below-floor -> DroppedSourceExit(3) + per-cell drop sentinel ──────
def test_below_floor_raises_dropped_exit_and_writes_sentinel(tmp_path: Path) -> None:
    """``_enforce_yield_floor`` below the floor writes the per-cell drop sentinel and
    raises ``DroppedSourceExit`` carrying ``DROPPED_SOURCE_EXIT`` (3) -- NOT exit 1
    (ambiguous with a genuine crash) and NOT a backfill."""
    cache_root = tmp_path / "onpolicy_cache"
    cell = C.Cell("sycophancy", "default", "contra", "d1")
    # 7/200 judge-positive = 3.5% << 80% floor (the exact production crash case).
    judged = [f"claim_{i}" for i in range(7)]
    targets = [f"claim_{i}" for i in range(200)]

    with pytest.raises(B.DroppedSourceExit) as ei:
        B._enforce_yield_floor(judged, targets, cell, cache_root)
    assert ei.value.code == B.DROPPED_SOURCE_EXIT == 3

    sentinel = cache_root / "dropped_sources" / "sycophancy__default__contra__d1.json"
    assert sentinel.exists(), "per-cell drop sentinel must be written"
    payload = json.loads(sentinel.read_text())
    assert payload["eval_key"] == cell.eval_key
    assert payload["judge_positive"] == 7
    assert payload["target_rows"] == 200
    assert payload["reason"] == "below-80%-yield-floor"
    assert abs(payload["yield_rate"] - 7 / 200) < 1e-9


def test_at_or_above_floor_does_not_raise(tmp_path: Path) -> None:
    """A source at/above the 80% floor is KEPT -- no raise, no sentinel."""
    cache_root = tmp_path / "onpolicy_cache"
    cell = C.Cell("sycophancy", "librarian", "contra", "d1")
    judged = [f"claim_{i}" for i in range(160)]  # 160/200 = exactly 80%
    targets = [f"claim_{i}" for i in range(200)]
    B._enforce_yield_floor(judged, targets, cell, cache_root)  # must NOT raise
    assert not (cache_root / "dropped_sources").exists()


# ── builder: main() returns 3 (so os._exit fires the DROP code) ───────────────
def test_build_main_returns_drop_code_on_below_floor(tmp_path: Path, monkeypatch) -> None:
    """End-to-end through ``B.main``: a below-floor sycophancy source makes ``main``
    return ``DROPPED_SOURCE_EXIT`` (3) cleanly -- the ``rc = main(); os._exit(rc)``
    finalize-guard then exits the process with code 3 (NOT a propagated SystemExit
    that bypasses os._exit, and NOT 0/1)."""
    cache_root = tmp_path / "onpolicy_cache"
    cell = C.Cell("sycophancy", "default", "contra", "d1")

    monkeypatch.setattr(sys, "argv", [
        "issue664_build_training_data.py",
        "--behavior", "sycophancy", "--source", "default",
        "--arm", "contra", "--dose", "d1",
        "--cache-root", str(cache_root),
    ])  # fmt: skip
    # neutralise the heavy startup asserts (tokenizer load, registry/panel asserts).
    monkeypatch.setattr(B.C, "assert_registry_19_columns", lambda: None)
    monkeypatch.setattr(B.C, "realized_grid", lambda: [])
    monkeypatch.setattr(B.C, "realized_source_keys", lambda grid: [])
    monkeypatch.setattr(B.C, "assert_panel_disjoint_from_sources", lambda keys: None)
    monkeypatch.setattr(B.C, "assert_marker_token", lambda tok: None)

    class _FakeTok:
        @staticmethod
        def from_pretrained(*a, **k):
            return object()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeTok)
    # build_sycophancy -> below-floor drop. Stub it to call the real floor enforcer.
    monkeypatch.setattr(B.C, "negative_panel", lambda: [])

    def _below_floor(c, cr, negatives, *, smoke):
        B._enforce_yield_floor([f"q{i}" for i in range(7)], [f"q{i}" for i in range(200)], c, cr)
        raise AssertionError("unreachable -- _enforce_yield_floor must raise")

    monkeypatch.setattr(B, "build_sycophancy", _below_floor)

    rc = B.main()
    assert rc == B.DROPPED_SOURCE_EXIT == 3
    # no training-mix file written for a dropped cell.
    assert not (B.C.DATA_ROOT / "train" / "sycophancy" / f"{cell.eval_key}.jsonl").exists()


# ── dispatcher: the build-loop subprocess wrapper handles rc==3 vs other rc ────
def test_dispatch_build_loop_handles_drop_code_not_crash() -> None:
    """AST pin: ``phase0`` drives the builder with ``subprocess.run(check=False)``
    and branches on ``B.DROPPED_SOURCE_EXIT`` (skip + continue) while re-raising
    ``CalledProcessError`` for any OTHER non-zero rc. The pre-fix ``check=True``
    crashed the whole fleet on a deliberate drop."""
    tree = ast.parse(Path(D.__file__).read_text())
    phase0 = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "phase0"),
        None,
    )
    assert phase0 is not None, "phase0 FunctionDef not found"
    src = ast.get_source_segment(Path(D.__file__).read_text(), phase0)
    assert src is not None
    # the build subprocess must NOT use check=True (that is the fleet-crash bug).
    assert "check=True" not in src, "phase0 build loop must use check=False, not check=True"
    assert "check=False" in src
    # the drop code must be branched on, with a continue (skip) + a fatal re-raise.
    assert "DROPPED_SOURCE_EXIT" in src
    assert "CalledProcessError" in src
    assert "_write_dropped_manifest" in src


def test_run_all_drop_filters_before_downstream_phases() -> None:
    """AST pin: ``run_all`` filters the selected cells through ``_drop_filtered``
    BEFORE the train / extract / upload phases, so a dropped cell is never trained,
    extracted, or upload-verified (the fail-loud missing-artifact asserts would crash
    on its never-produced files otherwise)."""
    tree = ast.parse(Path(D.__file__).read_text())
    run_all = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_all"),
        None,
    )
    assert run_all is not None, "run_all FunctionDef not found"
    called = [
        n.func.id
        for n in ast.walk(run_all)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "_drop_filtered" in called, "run_all must drop-filter the cells"


# ── dispatcher: dropped cells excluded from uploads (no missing-artifact FAIL) ─
def test_dropped_cells_excluded_from_uploads(tmp_path: Path, monkeypatch) -> None:
    """With a manifest declaring one cell dropped, ``_drop_filtered`` removes it so the
    fail-loud ``_upload_store_tensors`` only enforces tensors for the KEPT cell -- the
    dropped cell's absent ``tensors.pt`` does NOT trip the missing-artifact FAIL."""
    cache_root = tmp_path / "onpolicy_cache"
    dropped_dir = cache_root / "dropped_sources"
    dropped_dir.mkdir(parents=True)
    monkeypatch.setattr(D, "DROPPED_DIR", dropped_dir)

    kept = C.Cell("sycophancy", "librarian", "contra", "d1")
    drop = C.Cell("sycophancy", "default", "contra", "d1")
    (dropped_dir / "_manifest.json").write_text(
        json.dumps({"schema_version": 1, "dropped_cells": [{"eval_key": drop.eval_key}]})
    )

    # _dropped_cell_keys reads the manifest; _drop_filtered removes the dropped cell.
    assert D._dropped_cell_keys() == {drop.eval_key}
    filtered = D._drop_filtered([kept, drop])
    assert [c.eval_key for c in filtered] == [kept.eval_key]

    # the store-tensors uploader on the FILTERED list only checks the kept cell.
    store_root = tmp_path / "store"
    monkeypatch.setattr(C, "STORE_ROOT", store_root)
    kept_dir = store_root / kept.eval_key
    kept_dir.mkdir(parents=True)
    (kept_dir / "tensors.pt").write_text("fake")  # kept cell HAS its tensor

    uploaded: list[str] = []

    def _fake_upload(local, *, repo_id, repo_type, path_in_repo, upload_as_file):
        uploaded.append(path_in_repo)

    import explore_persona_space.orchestrate.hub as hub_mod

    monkeypatch.setattr(hub_mod, "_upload", _fake_upload)

    D._upload_store_tensors(filtered)  # must NOT raise (drop cell excluded)
    assert any(kept.eval_key in p for p in uploaded)
    assert not any(drop.eval_key in p for p in uploaded)


def test_upload_store_tensors_still_fails_loud_on_missing_KEPT_cell(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression guard: the drop-filter must NOT weaken the fail-loud missing-artifact
    assert for a cell that was KEPT (trained) but whose ``tensors.pt`` is absent -- that
    is still the #521-class trap and MUST raise."""
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setattr(C, "STORE_ROOT", store_root)
    kept = C.Cell("marker", "librarian", "contra", "d1")  # trained, but no tensor staged
    with pytest.raises(RuntimeError, match="trained-store tensors MISSING"):
        D._upload_store_tensors([kept])


def test_dropped_cell_keys_fallback_to_per_cell_sentinels(tmp_path: Path, monkeypatch) -> None:
    """When the top-level ``_manifest.json`` is absent (dispatcher restarted at
    --phase p1/p2/p3 in a fresh process), ``_dropped_cell_keys`` reconstructs the
    drop set from the per-cell sentinels the builder wrote."""
    dropped_dir = tmp_path / "dropped_sources"
    dropped_dir.mkdir(parents=True)
    monkeypatch.setattr(D, "DROPPED_DIR", dropped_dir)
    drop = C.Cell("refusal", "default", "contra", "d1")
    (dropped_dir / "refusal__default__contra__d1.json").write_text(
        json.dumps({"eval_key": drop.eval_key, "reason": "below-80%-yield-floor"})
    )
    assert D._dropped_cell_keys() == {drop.eval_key}
