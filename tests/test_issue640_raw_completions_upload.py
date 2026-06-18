"""Issue #640 — raw-completion upload passes ``upload_as_file=True`` per file.

``hub._upload`` (``src/explore_persona_space/orchestrate/hub.py``) raises
``ValueError`` UNCONDITIONALLY when handed a FILE path with
``upload_as_file=False`` (the #595 fail-loud guard against the silent
folder-upload no-op). ``upload_raw_completions()`` in the #640 driver globs
per-cell ``*.json`` files and uploads each one, so it MUST pass
``upload_as_file=True`` or it crashes on the FIRST file — after the GPU-spent
Phase 2 and before the end-of-run sentinel, stranding the run and losing the
raw completions (the round-1 code-review blocker).

This is the upload-path smoke for round 2: it drives ``upload_raw_completions()``
against a temp dir with a fake completion JSON and a monkeypatched
``hub._upload`` (so NO real HF Hub call happens), asserting the call is reached
with ``upload_as_file=True``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_driver():
    """Import the #640 driver from scripts/ (cheap: no torch/transformers at import)."""
    return importlib.import_module("issue640_postfix_carrier")


def test_upload_raw_completions_passes_upload_as_file(tmp_path, monkeypatch):
    """Each per-file ``hub._upload`` call carries ``upload_as_file=True``.

    Point the driver's ``output_root()`` at a temp dir (via EPM_OUTPUT_ROOT),
    stage one fake completion JSON under ``raw_completions/``, and capture every
    ``hub._upload`` call with a monkeypatch so the real HF Hub is never touched.
    """
    driver = _load_driver()

    out_root = tmp_path / "issue_640"
    raw_dir = out_root / "raw_completions"
    raw_dir.mkdir(parents=True)
    (raw_dir / "bad_medical_broad_em_trained.json").write_text('{"rows": []}')
    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(out_root))

    from explore_persona_space.orchestrate import hub

    calls: list[dict] = []

    def _fake_upload(local_path, **kwargs):
        calls.append({"local_path": Path(local_path), **kwargs})
        return f"{kwargs.get('repo_id', '')}/{kwargs.get('path_in_repo', '')}"

    monkeypatch.setattr(hub, "_upload", _fake_upload)

    # Must NOT raise (the round-1 bug raised here on the first file).
    driver.upload_raw_completions()

    assert len(calls) == 1, f"expected exactly one per-file upload, got {len(calls)}"
    call = calls[0]
    assert call["upload_as_file"] is True, (
        "hub._upload MUST be called with upload_as_file=True for single-file uploads; "
        "without it hub.py raises ValueError on the first file and the run strands "
        "after GPU-spent Phase 2 (round-1 code-review blocker)."
    )
    assert call["repo_type"] == "dataset"
    assert call["repo_id"] == hub.DEFAULT_DATASET_REPO
    assert call["path_in_repo"].startswith("issue640_postfix_carrier/raw_completions/")
    assert call["local_path"].name == "bad_medical_broad_em_trained.json"


def test_upload_raw_completions_noop_when_no_files(tmp_path, monkeypatch):
    """No raw-completion files -> no upload calls, no crash (graceful early return)."""
    driver = _load_driver()

    out_root = tmp_path / "issue_640"
    out_root.mkdir(parents=True)
    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(out_root))

    from explore_persona_space.orchestrate import hub

    calls: list = []
    monkeypatch.setattr(hub, "_upload", lambda *a, **k: calls.append((a, k)))

    driver.upload_raw_completions()
    assert calls == [], "no raw_completions dir -> upload_raw_completions must not call _upload"


def test_hub_upload_guard_rejects_file_without_flag(tmp_path, monkeypatch):
    """Regression anchor: hub._upload genuinely raises on a file without the flag.

    This pins the upstream contract the driver fix relies on — if the guard
    is ever relaxed, this test fails and signals the driver fix is no longer
    load-bearing (rather than the driver silently no-opping again). The raise
    fires before any HfApi instantiation, so the dummy HF_TOKEN below (required
    only to pass hub._upload's earlier no-token early-return) never reaches the
    network.
    """
    from explore_persona_space.orchestrate import hub

    monkeypatch.setenv("HF_TOKEN", "dummy-token-never-used-raise-precedes-network")
    f = tmp_path / "x.json"
    f.write_text("{}")
    with pytest.raises(ValueError, match="upload_as_file=False"):
        hub._upload(
            f,
            repo_id=hub.DEFAULT_DATASET_REPO,
            repo_type="dataset",
            path_in_repo="issue640_postfix_carrier/raw_completions/x.json",
        )
