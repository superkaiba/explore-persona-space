"""Issue #715 round-3 BLOCKER regressions (reconcile v2).

Two dispatcher-level invariants the round-2 P4 additions broke:

  BLOCKER p4train-raw-upload-skipped: ``main()`` uploaded raw completions only
  for phases {phase0..phase3, phase4}, EXCLUDING ``phase4train`` — whose
  ``_eval_checkpoint`` calls write the full-FT D*-selection ``raw_*.json`` judge
  artifacts. Those would be lost on pod teardown. The fix adds ``phase4train``
  to the gate; this test drives ``main()`` with ``--phase phase4train`` and
  asserts ``_upload_raw_completions`` fires (and that smoke still skips it).

  BLOCKER p4-dstar-fullft-ckpt-unreachable: ``_dstar_matched_fullft_ckpt``
  returned a LOCAL ``models/...`` path that does not exist on a FRESH P4 pod /
  after Step-8 cleanup, so ``StreamingWeights`` crashed on the absent dir. The
  fix re-downloads the checkpoint phase4train uploaded to
  ``<HF_MODEL_REPO>/issue715/<arm>_dstar`` when the local copy is absent; this
  test asserts the HF-download fallback is reached in BOTH the no-eval-json case
  and the local-checkpoint-missing case, and is NOT reached when the local
  checkpoint exists.

Pure CPU, fully stubbed — no HF network, no GPU, no model download.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dispatch(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    spec = importlib.util.spec_from_file_location(
        "issue715_dispatch", REPO_ROOT / "scripts" / "issue715_dispatch.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Args:
    def __init__(self, **kw):
        self.cells = None
        self.seeds = None
        self.smoke = False
        self.phase = "phase4train"
        for k, v in kw.items():
            setattr(self, k, v)


# ── BLOCKER p4train-raw-upload-skipped ──────────────────────────────────────


def _run_main_for_phase(mod, monkeypatch, phase: str, smoke: bool):
    """Drive main() for a given phase with everything but the upload gate stubbed.

    Returns (upload_called: bool). Stubs the parser to inject our phase/smoke,
    the dispatch body, the sentinel writer, and the reproducibility card.
    """
    uploaded = {"called": False}

    monkeypatch.setattr(mod, "_require_credentials", lambda: None)
    monkeypatch.setattr(mod, "_phase_dispatch", lambda args: {"ok": True})
    monkeypatch.setattr(mod, "write_sentinel", lambda *a, **k: Path("/tmp/x.json"))
    monkeypatch.setattr(mod, "_reproducibility_card", lambda args: {})
    monkeypatch.setattr(mod, "phase_log", lambda name: None)

    def fake_upload():
        uploaded["called"] = True

    monkeypatch.setattr(mod, "_upload_raw_completions", fake_upload)

    # Inject args via a fake parse_args (the gate reads args.phase + args.smoke).
    import argparse

    real_parse = argparse.ArgumentParser.parse_args

    def fake_parse(self, *a, **k):
        ns = real_parse(self, ["--phase", phase] + (["--smoke"] if smoke else []))
        return ns

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse)
    rc = mod.main()
    assert rc == 0
    return uploaded["called"]


def test_phase4train_uploads_raw_completions(monkeypatch):
    mod = _load_dispatch(monkeypatch)
    called = _run_main_for_phase(mod, monkeypatch, "phase4train", smoke=False)
    assert called, (
        "phase4train must upload raw completions (its _eval_checkpoint writes "
        "raw_*.json full-FT D*-selection artifacts that would be lost on teardown)"
    )


def test_phase4train_smoke_skips_upload(monkeypatch):
    """Smoke stays local-only — the upload gate is suppressed under --smoke."""
    mod = _load_dispatch(monkeypatch)
    called = _run_main_for_phase(mod, monkeypatch, "phase4train", smoke=True)
    assert not called, "smoke must NOT upload (local-only)"


def test_all_eval_writing_phases_upload(monkeypatch):
    """Every phase that runs evals (writes raw_*.json) must be in the gate."""
    mod = _load_dispatch(monkeypatch)
    for phase in ("phase0", "phase1", "phase2", "phase3", "phase4train", "phase4"):
        called = _run_main_for_phase(mod, monkeypatch, phase, smoke=False)
        assert called, f"{phase} writes raw_*.json and must upload before teardown"


# ── BLOCKER p4-dstar-fullft-ckpt-unreachable ────────────────────────────────


def test_dstar_fullft_falls_back_to_hf_when_no_eval_json(monkeypatch, tmp_path):
    """No narrow_task JSON on disk (fresh pod) -> re-download from HF, not None."""
    mod = _load_dispatch(monkeypatch)
    # Point EVAL_DIR at an empty tmp dir so the glob finds nothing (best is None).
    monkeypatch.setattr(mod.C, "EVAL_DIR", tmp_path / "eval_results")
    sentinel = tmp_path / "downloaded_ckpt"

    seen = {}

    def fake_dl(condition):
        seen["condition"] = condition
        return sentinel

    monkeypatch.setattr(mod, "_download_dstar_fullft_ckpt", fake_dl)
    out = mod._dstar_matched_fullft_ckpt("sft_fullft_p4", 0.5)
    assert out == sentinel, "must re-download the HF D*-ckpt when no local eval json"
    assert seen["condition"] == "sft_fullft_p4"


def test_dstar_fullft_falls_back_when_local_ckpt_missing(monkeypatch, tmp_path):
    """An eval json names a best step, but the local checkpoint dir is gone ->
    re-download from HF rather than returning a non-existent local path."""
    mod = _load_dispatch(monkeypatch)
    eval_dir = tmp_path / "eval_results"
    (eval_dir / "narrow_task").mkdir(parents=True)
    import json

    (eval_dir / "narrow_task" / "sft_fullft_p4_seed42_step100.json").write_text(
        json.dumps({"narrow_rate": 0.5, "checkpoint_step": 100})
    )
    monkeypatch.setattr(mod.C, "EVAL_DIR", eval_dir)
    # PROJECT_ROOT/models/... does NOT exist (fresh pod) -> _merged_ckpt_dir absent.
    sentinel = tmp_path / "redownloaded"

    def fake_dl(condition):
        return sentinel

    monkeypatch.setattr(mod, "_download_dstar_fullft_ckpt", fake_dl)
    out = mod._dstar_matched_fullft_ckpt("sft_fullft_p4", 0.5)
    assert out == sentinel, "missing local checkpoint must trigger the HF re-download"


def test_dstar_fullft_prefers_local_when_present(monkeypatch, tmp_path):
    """When the local D*-matched checkpoint survives (same pod), use it directly —
    do NOT re-download."""
    mod = _load_dispatch(monkeypatch)
    eval_dir = tmp_path / "eval_results"
    (eval_dir / "narrow_task").mkdir(parents=True)
    import json

    (eval_dir / "narrow_task" / "sft_fullft_p4_seed42_step100.json").write_text(
        json.dumps({"narrow_rate": 0.5, "checkpoint_step": 100})
    )
    monkeypatch.setattr(mod.C, "EVAL_DIR", eval_dir)
    # Materialize the local checkpoint dir so .exists() is True.
    local = mod.PROJECT_ROOT / "models" / "issue715_sft_fullft_p4_seed42" / "checkpoint-100"
    local.mkdir(parents=True, exist_ok=True)
    try:

        def boom(condition):
            raise AssertionError("must NOT re-download when the local ckpt exists")

        monkeypatch.setattr(mod, "_download_dstar_fullft_ckpt", boom)
        out = mod._dstar_matched_fullft_ckpt("sft_fullft_p4", 0.5)
        assert out == local
    finally:
        import shutil

        shutil.rmtree(mod.PROJECT_ROOT / "models" / "issue715_sft_fullft_p4_seed42")


def test_download_dstar_returns_none_when_subfolder_empty(monkeypatch, tmp_path):
    """The downloader returns None (not a phantom path) when HF yields no
    safetensors — so the caller's 'run phase4train first' guard fires cleanly."""
    mod = _load_dispatch(monkeypatch)
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    # snapshot_download succeeds but writes nothing (empty subfolder).
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **k: str(tmp_path))
    out = mod._download_dstar_fullft_ckpt("sft_fullft_p4")
    assert out is None, "no safetensors under the subfolder -> None, never a phantom path"
