"""#1979 F1 driver pins (crash-fix r3, fellows job 16717).

(a) f1c known-persona set: built from the LOADED mix rows' realized labels
    validated against arm_registry.json — a shared training-mix pool carries
    the representative (FT) arm's slug, not the mix id; a genuinely foreign
    label still fails loud.
(b) dispatcher per-unit failure budget: one failed unit is NON-fatal (siblings
    keep scheduling; the failed unit stays resumable — no done sentinel; the
    run still exits non-zero), while >FAILURE_BUDGET failures or a systemic
    same-exception-class repeat aborts early.

Everything runs CPU-tiny in tmp_path; the subprocess boundary is faked with a
signature-conformant Popen stand-in (worker commands only — everything else
delegates to the real Popen), and the disk-headroom probe is faked with a
signature-conformant no-op. The dispatch body itself executes for real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue1979_gpu as G  # noqa: E402


def _cfg(tmp_path: Path) -> G.Cfg:
    return G.Cfg(out_root=tmp_path, config_dir=tmp_path / "config", phases=("f1c",))


def _write_registry(tmp_path: Path, arm_ids: list[str]) -> None:
    (tmp_path / "arm_registry.json").write_text(
        json.dumps(
            {"mix_pos_sources": {a: {"pos_path": "p.jsonl", "layout": "delta"} for a in arm_ids}}
        )
    )


# ── (a) f1c known-persona set ────────────────────────────────────────────────


def test_anchor_known_personas_accepts_shared_pool_ft_label(tmp_path):
    """The job-16717 crash shape: mix rows labeled with the representative
    FT arm's slug under a LoRA mix id must pass (both registry-registered)."""
    cfg = _cfg(tmp_path)
    _write_registry(tmp_path, ["syc-pers-ft-con-s42", "syc-pers-con-lr1e5-s42"])
    rows = [{"persona": "syc-pers-ft-con-s42"} for _ in range(3)]
    known = G._anchor_known_personas(cfg, rows, "syc-pers-con-lr1e5-s42")
    assert set(known) == {"syc-pers-ft-con-s42", "syc-pers-con-lr1e5-s42"}
    known_set = set(known)
    for i, r in enumerate(rows):  # the reused helper's integrity assert, same shape
        assert r["persona"] in known_set, (i, r["persona"])


def test_anchor_known_personas_foreign_label_fails_loud(tmp_path):
    cfg = _cfg(tmp_path)
    _write_registry(tmp_path, ["syc-pers-ft-con-s42"])
    rows = [{"persona": "syc-pers-ft-con-s42"}, {"persona": "not-a-registered-arm"}]
    with pytest.raises(AssertionError) as ei:
        G._anchor_known_personas(cfg, rows, "syc-pers-con-lr1e5-s42")
    assert "not-a-registered-arm" in str(ei.value)


# ── (b) dispatcher failure budget ────────────────────────────────────────────


class _FakeProc:
    """Signature-conformant stand-in for the worker subprocess (dispatch reads
    only .pid and .poll())."""

    def __init__(self, rc: int):
        self.pid = 4242
        self._rc = rc

    def poll(self) -> int:
        return self._rc


def _patch_dispatch_seams(monkeypatch, cfg: G.Cfg, fail_plan: dict, launched: list[str]) -> None:
    """Fake ONLY the external boundaries: the worker subprocess (writes the same
    done-sentinel / failure-breadcrumb files a real worker writes) and the
    disk-headroom probe; CVD env pins two fake workers."""
    real_popen = G.subprocess.Popen

    def _popen(cmd, env=None, **kwargs):
        if "--worker-unit" not in cmd:
            return real_popen(cmd, env=env, **kwargs)
        key = cmd[cmd.index("--worker-unit") + 1]
        launched.append(key)
        exc_class = fail_plan.get(key)
        if exc_class is None:
            G.CAP._atomic_json(G._sentinel_path(cfg, key), {"key": key, "rc": 0})
            return _FakeProc(0)
        exc_type = type(exc_class, (RuntimeError,), {})
        G._write_failure(cfg, key, exc_type("boom"))
        return _FakeProc(1)

    def _no_headroom(out_root, need_gb, *, phase="", canary_gb=1.0):
        return 0.0

    monkeypatch.setattr(G.subprocess, "Popen", _popen)
    monkeypatch.setattr(
        "explore_persona_space.orchestrate.preflight.assert_out_root_headroom", _no_headroom
    )
    monkeypatch.setattr(G.time, "sleep", lambda s: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")


def test_dispatch_one_failure_is_nonfatal_but_run_exits_nonzero(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    items = [G.Item(key=f"f1c:m{i}", phase="f1c") for i in range(8)]
    items.append(G.Item(key="f1c:dep", phase="f1c", deps=("f1c:m0",)))
    launched: list[str] = []
    _patch_dispatch_seams(monkeypatch, cfg, {"f1c:m0": "AssertionError"}, launched)
    with pytest.raises(RuntimeError) as ei:
        G.dispatch(cfg, {}, items)
    msg = str(ei.value)
    assert "f1c:m0" in msg and "AssertionError" in msg
    for i in range(1, 8):  # every independent sibling still scheduled + completed
        assert G._done(cfg, f"f1c:m{i}"), f"sibling f1c:m{i} was not scheduled to completion"
    assert not G._done(cfg, "f1c:m0")  # failed unit resumable: no done sentinel
    assert "f1c:dep" not in launched  # dependent of the failed unit never scheduled
    assert "never scheduled" in msg
    assert not (tmp_path / "f1_results.json").exists()  # no done record on a failed run


def test_dispatch_over_budget_aborts_scheduling(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    items = [G.Item(key=f"f1c:m{i}", phase="f1c") for i in range(12)]
    # distinct exception classes so the systemic detector cannot fire first
    fail_plan = {f"f1c:m{i}": f"Exc{i}" for i in range(12)}
    launched: list[str] = []
    _patch_dispatch_seams(monkeypatch, cfg, fail_plan, launched)
    with pytest.raises(RuntimeError) as ei:
        G.dispatch(cfg, {}, items)
    msg = str(ei.value)
    assert "failure budget exceeded" in msg
    assert len(launched) == G.FAILURE_BUDGET + 1  # abort right past the budget
    assert len(launched) < len(items)  # remaining units never scheduled


def test_dispatch_systemic_same_class_aborts_early(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    items = [G.Item(key=f"f1c:m{i}", phase="f1c") for i in range(10)]
    fail_plan = {it.key: "CudaOOM" for it in items}
    launched: list[str] = []
    _patch_dispatch_seams(monkeypatch, cfg, fail_plan, launched)
    with pytest.raises(RuntimeError) as ei:
        G.dispatch(cfg, {}, items)
    msg = str(ei.value)
    assert "systemic failure: CudaOOM" in msg
    assert len(launched) <= G.FAILURE_BUDGET  # aborted below the plain budget
    assert len(launched) < len(items)


def test_dispatch_clean_run_emits_terminal_record(tmp_path, monkeypatch):
    pytest.importorskip("torch")  # _meta() in the terminal record imports torch
    cfg = _cfg(tmp_path)
    items = [G.Item(key=f"f1c:m{i}", phase="f1c") for i in range(3)]
    launched: list[str] = []
    _patch_dispatch_seams(monkeypatch, cfg, {}, launched)
    G.dispatch(cfg, {}, items)
    assert (tmp_path / "f1_results.json").exists()
    assert len(launched) == 3
