"""Tests for issue #697 BLOCKER B2 — off-pod HF hydration of analyze inputs.

Plan §9 routes ``analyze`` off-pod over HF-downloaded ``.pt``s AFTER the pod
terminates, so a fresh-VM checkout has no local ``eval_results/issue_697/patch``.
``issue697_dispatch._hydrate_analyze_artifacts`` pulls the sweep's HF-uploaded
per-cell ``.pt`` + ``_E_metadata.json`` + ``raw_completions/*.json`` (+ the
coverage json) into the local patch dir so the downstream ``glob('*.pt')`` finds
them. These pin:

  - hydration downloads every tensor + raw + coverage file the HF listing carries
    and they land at the right local paths;
  - it is idempotent (a file already present locally is NOT re-downloaded);
  - an empty HF listing (no .pt) raises a clear "no analysis tensors found on HF".

All HF I/O is mocked (``huggingface_hub.list_repo_files`` / ``hf_hub_download``);
no GPU, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import huggingface_hub  # noqa: E402
import issue697_dispatch as D  # noqa: E402


def _hf_paths() -> list[str]:
    """A realistic HF listing: 2 cells' tensors + their raw completions + coverage,
    plus unrelated files the filter must IGNORE."""
    t = D.HF_TENSOR_PREFIX
    r = D.HF_RAW_COMPLETIONS_PREFIX
    g = D.HF_GATE_PREFIX
    return [
        f"{t}/marker_sp_swe_seed42.pt",
        f"{t}/marker_sp_swe_seed42_E_metadata.json",
        f"{t}/fact_default_seed42.pt",
        f"{t}/fact_default_seed42_E_metadata.json",
        f"{r}/fact_default_seed42_p_up_seed42.json",
        f"{r}/fact_default_seed42_unpatched_ft_seed42.json",
        f"{g}/sweep_coverage.json",
        # noise the filter must skip:
        f"{t}/README.md",
        "issue697_cv_patch/gates/smoke_697b_pass.json",
        "some/other/issue/file.pt",
    ]


def _install_hf_mocks(monkeypatch, *, hf_paths: list[str], src_dir: Path):
    """Mock list_repo_files -> hf_paths; hf_hub_download -> a file in src_dir named
    after the basename (its content marks which HF path it came from)."""
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda *a, **k: list(hf_paths))

    def _download(repo_id, filename, repo_type=None, **kw):
        src = src_dir / Path(filename).name
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(f"content-of:{filename}")
        return str(src)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    # _hydrate_analyze_artifacts does `from huggingface_hub import ...` at CALL time,
    # so patching the module attributes above is picked up (lazy import).


def test_hydrate_downloads_all_artifacts(tmp_path, monkeypatch):
    """Hydration pulls every .pt + _E_metadata.json + raw + coverage into the right
    local layout, skipping the unrelated/README HF entries."""
    src = tmp_path / "hf_src"
    _install_hf_mocks(monkeypatch, hf_paths=_hf_paths(), src_dir=src)
    n = D._hydrate_analyze_artifacts(tmp_path)

    patch = tmp_path / "eval_results" / "issue_697" / "patch"
    raw = patch / "raw_completions"
    assert (patch / "marker_sp_swe_seed42.pt").exists()
    assert (patch / "marker_sp_swe_seed42_E_metadata.json").exists()
    assert (patch / "fact_default_seed42.pt").exists()
    assert (raw / "fact_default_seed42_p_up_seed42.json").exists()
    assert (raw / "fact_default_seed42_unpatched_ft_seed42.json").exists()
    assert (tmp_path / "eval_results" / "issue_697" / "sweep_coverage.json").exists()
    # README + the non-697 .pt + the smoke-pass file must NOT have been pulled.
    assert not (patch / "README.md").exists()
    assert not (patch / "file.pt").exists()
    assert not (patch / "smoke_697b_pass.json").exists()
    # 2 .pt + 2 metadata + 2 raw + 1 coverage = 7 files hydrated.
    assert n == 7, n


def test_hydrate_is_idempotent(tmp_path, monkeypatch):
    """A file already present locally is NOT re-downloaded (resume / re-run)."""
    src = tmp_path / "hf_src"
    _install_hf_mocks(monkeypatch, hf_paths=_hf_paths(), src_dir=src)
    # pre-seed ONE tensor + ONE raw locally.
    patch = tmp_path / "eval_results" / "issue_697" / "patch"
    raw = patch / "raw_completions"
    raw.mkdir(parents=True)
    (patch / "marker_sp_swe_seed42.pt").write_text("preexisting")
    (raw / "fact_default_seed42_p_up_seed42.json").write_text("preexisting")

    n = D._hydrate_analyze_artifacts(tmp_path)
    # 7 total - 2 pre-seeded = 5 newly hydrated.
    assert n == 5, n
    # the pre-seeded files are untouched (NOT overwritten by the mock content).
    assert (patch / "marker_sp_swe_seed42.pt").read_text() == "preexisting"
    assert (raw / "fact_default_seed42_p_up_seed42.json").read_text() == "preexisting"


def test_hydrate_raises_when_no_tensors_on_hf(tmp_path, monkeypatch):
    """An HF listing with NO per-cell .pt raises a clear 'no analysis tensors found
    on HF' message (analyze cannot run; the sweep never uploaded)."""
    # raw + coverage present, but ZERO .pt -> must raise.
    paths = [
        f"{D.HF_RAW_COMPLETIONS_PREFIX}/fact_default_seed42_p_up_seed42.json",
        f"{D.HF_GATE_PREFIX}/sweep_coverage.json",
    ]
    _install_hf_mocks(monkeypatch, hf_paths=paths, src_dir=tmp_path / "hf_src")
    with pytest.raises(RuntimeError, match="no analysis tensors found on HF"):
        D._hydrate_analyze_artifacts(tmp_path)
