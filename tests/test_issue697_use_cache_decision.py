"""Tests for issue #697 BLOCKER B3 — the use_cache decision resolution + provenance.

Plan §10 hard-requires ``use_cache=False`` for the production 7B sweep (the canary
Gate C1.2 measured KV caching DROPS the patch, parity Δ=0.25 ≫ tol 0.001 → corrupted
p_up/p_down E generations). These pin ``issue697_dispatch._read_use_cache_decision``:

  (a) no LOCAL decision file -> the dispatcher pulls the decision from HF;
  (b) an HF/local 7B decision saying use_cache=True -> the cell gets ``--use-cache``;
  (c) an HF/local 7B decision saying use_cache=False -> the cell gets ``--no-use-cache``;
  (d) a decision whose ``base_model_id`` is NOT the 7B production model (a 0.5B-smoke
      decision) -> RuntimeError (REJECTED, never silently accepted for the 7B sweep);
  (e) NEITHER local nor HF -> DEFAULT False (the safe default — uncached never
      corrupts the E-gen).

All HF I/O is mocked; no GPU, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import huggingface_hub  # noqa: E402
import issue697_dispatch as D  # noqa: E402

QWEN = "Qwen/Qwen2.5-7B-Instruct"


def _write_local_decision(repo_root: Path, *, use_cache: bool, base_model_id: str) -> None:
    p = repo_root / "eval_results" / "issue_697" / "canary" / "canary_decision.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "use_cache_production_default": use_cache,
                "base_model_id": base_model_id,
                "model": base_model_id,
            }
        )
    )


def _mock_hf_decision(monkeypatch, *, use_cache: bool, base_model_id: str, src_dir: Path):
    """hf_hub_download returns a decision json with the given fields."""

    def _download(repo_id, filename, repo_type=None, **kw):
        f = src_dir / "hf_canary_decision.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(
            json.dumps(
                {
                    "use_cache_production_default": use_cache,
                    "base_model_id": base_model_id,
                    "model": base_model_id,
                }
            )
        )
        return str(f)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)


def _mock_hf_missing(monkeypatch):
    """hf_hub_download always raises EntryNotFoundError (nothing on HF)."""
    from huggingface_hub.utils import EntryNotFoundError

    def _download(*a, **k):
        raise EntryNotFoundError("not found")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)


# (a) no LOCAL decision -> pull from HF.
def test_no_local_decision_pulls_from_hf(tmp_path, monkeypatch):
    _mock_hf_decision(monkeypatch, use_cache=False, base_model_id=QWEN, src_dir=tmp_path / "hf")
    # no local file written -> the HF path is exercised.
    out = D._read_use_cache_decision(tmp_path, production_base_model=QWEN)
    assert out is False


# (b) decision says use_cache=True for 7B -> _cell_cmd passes --use-cache.
def test_hf_decision_true_threads_use_cache_flag(tmp_path, monkeypatch):
    _mock_hf_decision(monkeypatch, use_cache=True, base_model_id=QWEN, src_dir=tmp_path / "hf")
    use_cache = D._read_use_cache_decision(tmp_path, production_base_model=QWEN)
    assert use_cache is True
    cmd = _cell_cmd_flags(tmp_path, use_cache=use_cache)
    assert "--use-cache" in cmd and "--no-use-cache" not in cmd


# (c) decision says use_cache=False -> _cell_cmd passes --no-use-cache.
def test_hf_decision_false_threads_no_use_cache_flag(tmp_path, monkeypatch):
    _mock_hf_decision(monkeypatch, use_cache=False, base_model_id=QWEN, src_dir=tmp_path / "hf")
    use_cache = D._read_use_cache_decision(tmp_path, production_base_model=QWEN)
    assert use_cache is False
    cmd = _cell_cmd_flags(tmp_path, use_cache=use_cache)
    assert "--no-use-cache" in cmd and "--use-cache" not in cmd


# (d) a 0.5B-smoke-derived decision is REJECTED for the 7B sweep.
def test_smoke_model_decision_rejected_for_7b(tmp_path, monkeypatch):
    # local file carries the smoke base model -> must raise, never accepted.
    _write_local_decision(tmp_path, use_cache=True, base_model_id="Qwen/Qwen2.5-0.5B-Instruct")
    with pytest.raises(RuntimeError, match="not the production model"):
        D._read_use_cache_decision(tmp_path, production_base_model=QWEN)


def test_smoke_model_decision_on_hf_rejected_for_7b(tmp_path, monkeypatch):
    # no local; HF carries a 0.5B decision -> must raise.
    _mock_hf_decision(
        monkeypatch,
        use_cache=True,
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        src_dir=tmp_path / "hf",
    )
    with pytest.raises(RuntimeError, match="not the production model"):
        D._read_use_cache_decision(tmp_path, production_base_model=QWEN)


# (e) neither local nor HF -> default False (the safe default).
def test_default_false_when_no_decision_anywhere(tmp_path, monkeypatch):
    _mock_hf_missing(monkeypatch)
    out = D._read_use_cache_decision(tmp_path, production_base_model=QWEN)
    assert out is False, "absent-decision default MUST be False (uncached never corrupts E-gen)"


# local 7B decision wins over HF (resolution order 1).
def test_local_7b_decision_preferred(tmp_path, monkeypatch):
    _write_local_decision(tmp_path, use_cache=False, base_model_id=QWEN)
    # If HF were consulted it would say True; the local 7B file must win (False).
    _mock_hf_decision(monkeypatch, use_cache=True, base_model_id=QWEN, src_dir=tmp_path / "hf")
    out = D._read_use_cache_decision(tmp_path, production_base_model=QWEN)
    assert out is False


def _cell_cmd_flags(repo_root: Path, *, use_cache: bool) -> list[str]:
    """Build a cell command via _cell_cmd and return its arg list (to assert the
    --use-cache / --no-use-cache flag threading)."""
    from explore_persona_space.experiments.issue_651 import Cell

    cell = Cell(behavior="marker", cid="sp_swe", seed=42, gpu_id=0)
    panel = repo_root / "panel.json"
    panel.write_text("{}")
    cmd, _log, _env = D._cell_cmd(
        repo_root,
        cell,
        cpu_only=False,
        panel_personas_json=panel,
        panel_questions_json=panel,
        out_dir=repo_root / "out",
        layers=[7, 14, 21],
        primary_layer=14,
        max_new_tokens=64,
        skip_e=True,
        smoke_model=None,
        upload=False,
        use_cache=use_cache,
        patch_layer=10,
        rbase_cache_dir=None,
    )
    return cmd
