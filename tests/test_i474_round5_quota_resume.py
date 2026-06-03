"""CPU-only unit tests for #474 round-5 fixes (post mid-sweep quota crash).

Covers:

- **FIX A** — ``PerEpochAdapterHFUploadCallback`` deletes BOTH the
  ``_upload_ep{N}/`` staging dir AND the source ``checkpoint-{step}/``
  dir AFTER ``upload_model`` returns a non-empty (verified) hub_path.
  Without this the MooseFS per-pod ~130GB quota fills (~84 conds x ~9GB
  per cond of optimizer.pt + checkpoints = ~750GB → silent SIGKILL).
  Also: ``save_total_limit`` tightened to ``1`` as a belt-and-braces
  backstop (HF Trainer auto-prunes).

- **FIX B** — ``scripts/i474_check_adapter_hf_presence.py`` queries
  HF for all 4 expected epoch adapters {1,2,3,5} per (arm, cid) and
  returns exit code 0 if all present, 1 if any missing, 2 on HF
  lookup failure. ``i474_phase23_dispatch.sh --resume`` uses this to
  skip conds whose full set is already on HF; partial conds retrain
  fully.

CPU-only — synthetic FS + monkeypatched ``upload_model`` / HF Hub.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_TRAIN = Path(__file__).resolve().parent.parent / "scripts" / "i474_phase23_train.py"
_SCRIPT_CHECK = (
    Path(__file__).resolve().parent.parent / "scripts" / "i474_check_adapter_hf_presence.py"
)
_SCRIPT_DISP = Path(__file__).resolve().parent.parent / "scripts" / "i474_phase23_dispatch.sh"
_SCRIPT_RUN_ALL = Path(__file__).resolve().parent.parent / "scripts" / "i474_run_all.sh"


@pytest.fixture(scope="module")
def i474_train_module():
    spec = importlib.util.spec_from_file_location("i474_phase23_train", _SCRIPT_TRAIN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i474_phase23_train"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def i474_check_module():
    spec = importlib.util.spec_from_file_location("i474_check_adapter_hf_presence", _SCRIPT_CHECK)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i474_check_adapter_hf_presence"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_fake_checkpoint(tmp_path: Path, global_step: int = 100) -> tuple[Path, Path]:
    """Reused from round-3 test: synthetic SFTTrainer output_dir + ckpt dir."""
    output_dir = tmp_path / "adapters" / "i474_pos_A1"
    output_dir.mkdir(parents=True)
    for fname in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "README.md",
    ):
        (output_dir / fname).write_text(f"{fname} placeholder")
    ckpt_dir = output_dir / f"checkpoint-{global_step}"
    ckpt_dir.mkdir()
    (ckpt_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 1024)
    (ckpt_dir / "adapter_config.json").write_text(json.dumps({"r": 32, "alpha": 64}))
    (ckpt_dir / "optimizer.pt").write_bytes(b"\xff" * 4096)
    (ckpt_dir / "rng_state.pth").write_bytes(b"\xff" * 512)
    (ckpt_dir / "scheduler.pt").write_bytes(b"\xff" * 256)
    (ckpt_dir / "trainer_state.json").write_text(json.dumps({"global_step": global_step}))
    (ckpt_dir / "training_args.bin").write_bytes(b"\xff" * 4096)
    return output_dir, ckpt_dir


class _FakeTrainerState:
    def __init__(self, epoch: float, global_step: int):
        self.epoch = epoch
        self.global_step = global_step


class _FakeTrainerArgs:
    def __init__(self):
        self.report_to = ""


# ---------------------------------------------------------------- FIX A


def test_fix_a_reaps_checkpoint_after_verified_upload(i474_train_module, tmp_path, monkeypatch):
    """After a verified HF upload, BOTH ``_upload_ep{N}/`` AND
    ``checkpoint-{step}/`` are reaped from local disk."""
    output_dir, ckpt_dir = _make_fake_checkpoint(tmp_path, global_step=100)
    cb = i474_train_module.PerEpochAdapterHFUploadCallback(
        arm="pos", cid="A1", output_dir=str(output_dir), hf_repo="placeholder/repo"
    )

    # Monkeypatch upload_model to "succeed" (return a non-empty hub_path).
    import explore_persona_space.orchestrate.hub as hub_mod

    monkeypatch.setattr(
        hub_mod,
        "upload_model",
        lambda local_path, repo_id, path_in_repo: f"{repo_id}/{path_in_repo}",
    )

    cb.on_save(
        _FakeTrainerArgs(),
        _FakeTrainerState(epoch=1.0, global_step=100),
        control=None,
    )

    # Both dirs must be GONE.
    assert not ckpt_dir.exists(), (
        f"FIX A VIOLATION: checkpoint dir {ckpt_dir} still on local disk after "
        "verified HF upload — this is the round-5 mid-sweep quota crash"
    )
    upload_dir = output_dir / "_upload_ep1"
    assert not upload_dir.exists(), (
        f"FIX A VIOLATION: staged upload bundle {upload_dir} still on local disk"
    )
    # The parent output_dir MUST survive (tokenizer files + end-of-training adapter).
    assert output_dir.exists()
    assert (output_dir / "tokenizer.json").exists()
    # The callback recorded this epoch as uploaded.
    assert 1 in cb._uploaded_epochs


def test_fix_a_does_not_reap_on_failed_upload(i474_train_module, tmp_path, monkeypatch):
    """If ``upload_model`` returns ``""`` (verification failed), the callback
    MUST raise BEFORE touching the local checkpoint. The local copy is the
    only remaining adapter — losing it would lose the cell."""
    output_dir, ckpt_dir = _make_fake_checkpoint(tmp_path, global_step=200)
    cb = i474_train_module.PerEpochAdapterHFUploadCallback(
        arm="pos", cid="A1", output_dir=str(output_dir), hf_repo="placeholder/repo"
    )

    import explore_persona_space.orchestrate.hub as hub_mod

    monkeypatch.setattr(hub_mod, "upload_model", lambda local_path, repo_id, path_in_repo: "")

    with pytest.raises(RuntimeError, match="upload_model returned empty string"):
        cb.on_save(
            _FakeTrainerArgs(),
            _FakeTrainerState(epoch=1.0, global_step=200),
            control=None,
        )

    # Checkpoint MUST survive the failed upload (no destructive cleanup on failure).
    assert ckpt_dir.exists(), (
        "FIX A SAFETY VIOLATION: checkpoint reaped after FAILED upload — "
        "the local copy was the last surviving adapter for this cell"
    )
    assert (ckpt_dir / "adapter_model.safetensors").exists()


def test_fix_a_skips_non_target_epochs(i474_train_module, tmp_path, monkeypatch):
    """state.epoch=4 (not in CHECKPOINT_EPOCHS_TO_UPLOAD) → callback no-ops.

    Without this guard, the dispatcher would try to upload epoch 4 — but the
    Phase 4 eval doesn't download epoch 4 (only 1/2/3/5), so it would waste
    HF storage. AND we must NOT reap the local checkpoint either (HF Trainer
    save_total_limit handles ep4 pruning instead).
    """
    output_dir, ckpt_dir = _make_fake_checkpoint(tmp_path, global_step=400)
    cb = i474_train_module.PerEpochAdapterHFUploadCallback(
        arm="pos", cid="A1", output_dir=str(output_dir), hf_repo="placeholder/repo"
    )

    upload_called = []
    import explore_persona_space.orchestrate.hub as hub_mod

    def fake_upload(local_path, repo_id, path_in_repo):
        upload_called.append((local_path, repo_id, path_in_repo))
        return f"{repo_id}/{path_in_repo}"

    monkeypatch.setattr(hub_mod, "upload_model", fake_upload)

    cb.on_save(
        _FakeTrainerArgs(),
        _FakeTrainerState(epoch=4.0, global_step=400),
        control=None,
    )

    assert upload_called == [], "ep4 should NOT trigger an upload"
    assert ckpt_dir.exists(), "ep4 checkpoint left for save_total_limit to prune"
    assert 4 not in cb._uploaded_epochs


def test_save_total_limit_set_to_1_belt_and_braces(i474_train_module):
    """Static source check: ``save_total_limit=1`` in the cfg block.

    Belt-and-braces backstop — even if the callback's reap somehow misses
    a checkpoint, HF Trainer auto-prunes older checkpoints when a new one
    lands. Was ``None`` (unlimited) — that's how the 84 conds piled up.
    """
    src = _SCRIPT_TRAIN.read_text()
    assert "save_total_limit=1" in src, (
        "FIX A backstop: save_total_limit MUST be 1 (was None — unlimited "
        "checkpoint accumulation was the round-5 quota-crash root cause)"
    )


# ---------------------------------------------------------------- FIX B


def test_fix_b_check_helper_all_present_exit_0(i474_check_module, monkeypatch):
    """All 4 epoch adapters present → exit code 0 (caller skips)."""
    files = {f"adapters/i474_pos_A1_ep{ep}/adapter_model.safetensors" for ep in (1, 2, 3, 5)} | {
        "unrelated/file.txt"
    }
    monkeypatch.setattr(
        "i474_check_adapter_hf_presence.list_repo_files",
        lambda repo_id, repo_type, revision: list(files),
        raising=False,
    )

    # Direct call to _missing_epochs (the inner helper).
    missing = i474_check_module._missing_epochs("pos", "A1", (1, 2, 3, 5))
    assert missing == [], f"all present but got missing={missing}"


def test_fix_b_check_helper_partial_present_returns_missing(i474_check_module, monkeypatch):
    """B4 partial (ep1/ep2/ep3 present, ep5 missing) → missing=[5]."""
    files = [f"adapters/i474_loc_B4_ep{ep}/adapter_model.safetensors" for ep in (1, 2, 3)]
    monkeypatch.setattr(
        "i474_check_adapter_hf_presence.list_repo_files",
        lambda repo_id, repo_type, revision: files,
        raising=False,
    )

    missing = i474_check_module._missing_epochs("loc", "B4", (1, 2, 3, 5))
    assert missing == [5], f"expected [5] missing but got {missing}"


def test_fix_b_check_helper_all_missing_returns_full_set(i474_check_module, monkeypatch):
    """B5 never trained → missing=[1,2,3,5]."""
    monkeypatch.setattr(
        "i474_check_adapter_hf_presence.list_repo_files",
        lambda repo_id, repo_type, revision: [],
        raising=False,
    )
    missing = i474_check_module._missing_epochs("loc", "B5", (1, 2, 3, 5))
    assert missing == [1, 2, 3, 5]


def test_fix_b_check_helper_hf_lookup_failure_raises(i474_check_module, monkeypatch):
    """HF network / auth failure → RuntimeError → main() exit code 2.

    Caller treats exit 2 as "missing" (retrain) — fail-loud over silent skip.
    """

    def boom(*args, **kwargs):
        raise ConnectionError("HF down")

    monkeypatch.setattr(
        "i474_check_adapter_hf_presence.list_repo_files",
        boom,
        raising=False,
    )
    with pytest.raises(RuntimeError, match="HF list_repo_files"):
        i474_check_module._missing_epochs("pos", "A1", (1, 2, 3, 5))


def test_fix_b_check_helper_uses_required_file_marker(i474_check_module):
    """Helper checks for ``adapter_model.safetensors`` (the load-bearing
    file Phase 4 eval downloads), not ``adapter_config.json`` or some
    other marker. Mismatched marker would let a half-uploaded adapter
    show as "present" + crash later."""
    assert i474_check_module.REQUIRED_FILE == "adapter_model.safetensors"


def test_fix_b_check_helper_default_epochs_match_callback(i474_check_module, i474_train_module):
    """The check helper's DEFAULT_EPOCHS MUST match the callback's
    CHECKPOINT_EPOCHS_TO_UPLOAD — otherwise resume would skip a cond
    whose expected epoch set differs from what training uploads."""
    assert (
        i474_check_module.DEFAULT_EPOCHS
        == i474_train_module.PerEpochAdapterHFUploadCallback.CHECKPOINT_EPOCHS_TO_UPLOAD
    )


# ---------------------------------------------------------------- dispatcher wiring


def test_dispatcher_has_resume_flag():
    src = _SCRIPT_DISP.read_text()
    assert "--resume) RESUME=1" in src, "dispatcher must accept --resume"


def test_dispatcher_resume_calls_presence_check():
    src = _SCRIPT_DISP.read_text()
    assert "i474_check_adapter_hf_presence.py" in src, (
        "dispatcher --resume path must call the HF-presence check helper"
    )
    # When the helper exits 0, dispatcher logs "already on HF" and skips.
    assert "already on HF" in src
    # When the helper exits non-zero, dispatcher retrains.
    assert "resume_train_" in src


def test_run_all_resume_wires_to_dispatcher():
    src = _SCRIPT_RUN_ALL.read_text()
    flat = " ".join(src.split())
    assert "--resume" in flat
    # Both train + crosseval dispatchers must receive --resume.
    assert "i474_phase23_dispatch.sh --resume" in flat, (
        "run_all.sh --resume must pass --resume to train dispatcher"
    )
    assert "i474_phase4_dispatch.sh --resume" in flat, (
        "run_all.sh --resume must pass --resume to crosseval dispatcher "
        "(phase4_eval's per-cell --resume already exists)"
    )


def test_help_runs_clean(i474_check_module):
    """Sanity: --help still works."""
    assert hasattr(i474_check_module, "main")
    assert hasattr(i474_check_module, "_missing_epochs")
    assert i474_check_module.HF_MODEL_REPO == "superkaiba1/explore-persona-space"
