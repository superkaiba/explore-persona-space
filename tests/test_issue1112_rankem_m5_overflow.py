"""#1112 rankem M5 — B2 full-FT disk residency cap + overflow offload.

The shared ``build_checkpoint_callback`` (scripts/train_behavior_fullft.py) grows
a DEFAULT-OFF ``overflow=`` opt-in: each saved grid checkpoint uploads to a
private HF model repo, then local copies beyond ``residency_cap`` are pruned
(never before a confirmed upload) — so a ~12-rung 7B full-FT grid (~180 GB) fits
the ~130 GB RunPod quota. The dispatcher wires it for B2 (production only) and
resolves a pruned rung local-or-download in p3/p4/p5.

Tests:
* the default path is byte-exact (overflow=None -> on_save is a no-op, no upload,
  no prune) — the regression pin the M5 opt-in must never regress;
* the overflow path through the REAL trainer lifecycle (#816): uploads fire per
  grid step + prune keeps <= cap local, faking ONLY the HF upload boundary;
* the dispatcher composes the overflow flags in production + omits them in smoke;
* ``_stage_b2_rung`` resolves a resident rung locally and downloads a pruned one.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Unique module name so this test's copy of the dispatcher never collides in
# sys.modules with the sibling rankem-dispatch test files when the full suite
# loads them in the same process.
_SPEC = importlib.util.spec_from_file_location(
    "issue1112_rankem_dispatch_m5", PROJECT_ROOT / "scripts" / "issue1112_rankem_dispatch.py"
)
D = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = D
_SPEC.loader.exec_module(D)
R = D.R

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

TINY_QWEN_KWARGS = dict(
    vocab_size=151936,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    max_position_embeddings=4096,
    tie_word_embeddings=True,
)

_TINY_ROWS = [
    {
        "prompt": [{"role": "user", "content": f"Say a short greeting number {i}."}],
        "completion": [{"role": "assistant", "content": f"Hello there, greeting {i}."}],
    }
    for i in range(4)
]


def _cfg(tmp_path, **kw):
    return D.Cfg(
        out_root=tmp_path / "out",
        cells=kw.pop("cells", R.ALL_CELLS),
        smoke=kw.pop("smoke", False),
        upload=kw.pop("upload", False),
        dry_run=kw.pop("dry_run", True),
        **kw,
    )


# ── dispatcher-side wiring (CPU-only, no GPU/HF) ──────────────────────────────


def test_b2_ft_cmd_overflow_flags_production(tmp_path) -> None:
    cfg = _cfg(tmp_path, smoke=False)
    cmd = D._b2_ft_cmd(
        cfg,
        out_dir=tmp_path / "train",
        corpus=tmp_path / "c.jsonl",
        max_steps=200,
        ckpt_steps=[2, 5],
    )
    assert cmd[cmd.index("--overflow-upload-repo") + 1] == R.OVERFLOW_REPO
    assert cmd[cmd.index("--overflow-path-prefix") + 1] == f"issue1112_{R.RANKEM_SLUG}/{R.B2}"
    assert cmd[cmd.index("--residency-cap") + 1] == str(D.B2_RESIDENCY_CAP)


def test_b2_ft_cmd_overflow_flags_omitted_in_smoke(tmp_path) -> None:
    cfg = _cfg(tmp_path, smoke=True)
    cmd = D._b2_ft_cmd(
        cfg, out_dir=tmp_path / "train", corpus=tmp_path / "c.jsonl", max_steps=2, ckpt_steps=[2]
    )
    assert "--overflow-upload-repo" not in cmd
    assert "--overflow-path-prefix" not in cmd
    assert "--residency-cap" not in cmd


def test_overflow_prefix_matches_p5_layout() -> None:
    # p3 download, p4 download, p5 upload/record + the trainer offload all resolve
    # the SAME repo layout the cross-method consumer + p5 record expect.
    assert D._overflow_prefix(R.B2) == f"issue1112_{R.RANKEM_SLUG}/{R.B2}"


def test_stage_b2_rung_local_first(tmp_path, monkeypatch) -> None:
    """A resident rung resolves locally with no download (downloaded=False)."""
    adapter_root = tmp_path / "b2" / "train"
    (adapter_root / "checkpoint-40").mkdir(parents=True)

    def _boom(*a, **k):
        raise AssertionError("stage_hub_prefix must not be called for a local rung")

    from explore_persona_space.orchestrate import hub

    monkeypatch.setattr(hub, "stage_hub_prefix", _boom)
    cfg = _cfg(tmp_path, dry_run=False)
    path, downloaded = D._stage_b2_rung(cfg, R.B2, 40, str(adapter_root))
    assert downloaded is False
    assert path == adapter_root / "checkpoint-40"


def test_stage_b2_rung_downloads_when_pruned(tmp_path, monkeypatch) -> None:
    """A pruned rung downloads from overflow (verbatim mirror) + flags cleanup."""
    adapter_root = tmp_path / "b2" / "train"
    adapter_root.mkdir(parents=True)
    cfg = _cfg(tmp_path, dry_run=False)

    prefix = f"{D._overflow_prefix(R.B2)}/checkpoint-125"

    def fake_stage(repo_id, pfx, dest_dir, **kw):
        assert repo_id == R.OVERFLOW_REPO
        assert pfx == prefix
        staged = Path(dest_dir) / pfx  # verbatim prefix mirror
        staged.mkdir(parents=True, exist_ok=True)
        (staged / "config.json").write_text("{}")
        return [staged / "config.json"]

    from explore_persona_space.orchestrate import hub

    monkeypatch.setattr(hub, "stage_hub_prefix", fake_stage)
    path, downloaded = D._stage_b2_rung(cfg, R.B2, 125, str(adapter_root))
    assert downloaded is True
    assert path == cfg.out_root / "b2_rung_dl" / prefix
    assert (path / "config.json").exists()


def test_stage_b2_rung_fail_loud_on_incomplete_download(tmp_path, monkeypatch) -> None:
    """A pulled dir without config.json is a hard error (never a silent partial)."""
    adapter_root = tmp_path / "b2" / "train"
    adapter_root.mkdir(parents=True)
    cfg = _cfg(tmp_path, dry_run=False)

    def fake_stage(repo_id, pfx, dest_dir, **kw):
        (Path(dest_dir) / pfx).mkdir(parents=True, exist_ok=True)  # no config.json
        return []

    from explore_persona_space.orchestrate import hub

    monkeypatch.setattr(hub, "stage_hub_prefix", fake_stage)
    with pytest.raises(RuntimeError, match=r"missing config\.json"):
        D._stage_b2_rung(cfg, R.B2, 300, str(adapter_root))


# ── shared callback: default path is byte-exact ──────────────────────────────


def test_build_checkpoint_callback_default_path_is_noop() -> None:
    """overflow=None: on_save does nothing (no upload, no prune) — the existing
    #642 caller contract is preserved byte-for-byte."""
    from train_behavior_fullft import build_checkpoint_callback

    cb = build_checkpoint_callback({2, 4})
    assert cb.overflow is None
    assert cb.uploaded_steps == set()

    # on_save with overflow=None short-circuits before touching state/hub/disk.
    args = types.SimpleNamespace(output_dir="/nonexistent")
    state = types.SimpleNamespace(is_world_process_zero=True, global_step=2)
    control = types.SimpleNamespace(should_save=False)
    assert cb.on_save(args, state, control) is control
    assert cb.uploaded_steps == set()


# ── shared callback: overflow path through the REAL trainer lifecycle ─────────


@pytest.fixture(scope="module")
def qwen_tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    except OSError as e:  # offline CI without a cached tokenizer
        pytest.skip(f"Qwen tokenizer unavailable (offline?): {e}")


@pytest.fixture(scope="module")
def tiny_qwen_state():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(**TINY_QWEN_KWARGS)
    torch.manual_seed(1112)
    model = Qwen2ForCausalLM(config)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    return config, state


@pytest.mark.slow
def test_overflow_callback_uploads_and_prunes(tmp_path, monkeypatch, qwen_tok, tiny_qwen_state):
    """grid={1,2,3,4}, max_steps=4, residency_cap=2 via the REAL lifecycle:

    - each grid step uploads (hub._upload called, dir present at call time),
    - uploaded_steps == the full grid,
    - only the newest `cap` checkpoint dirs survive locally (older uploaded ones
      pruned — the M5 disk-bound behavior).
    The callback runs through on_init_end -> on_step_end -> on_save -> on_train_end
    (#816: a bare non-TrainerCallback class would AttributeError at on_init_end).
    """
    import transformers
    from train_behavior_fullft import build_checkpoint_callback

    from explore_persona_space.orchestrate import hub
    from explore_persona_space.train.sft import train_lora

    config, state = tiny_qwen_state

    def fresh_tiny_model(*args, **kwargs):
        m = transformers.Qwen2ForCausalLM(config)
        m.load_state_dict(state)
        return m

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fresh_tiny_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: qwen_tok)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.delenv("EPM_PERSIST_ADAPTER_HF_REPO", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    calls: list[tuple] = []

    def fake_upload(local_path, repo_id, repo_type, path_in_repo, **kw):
        # HF upload boundary: record the call + confirm the dir exists at upload
        # time (never delete-before-upload) + assert the private model-repo shape.
        calls.append(
            (Path(local_path).name, repo_id, repo_type, path_in_repo, Path(local_path).is_dir())
        )
        assert kw.get("private") is True
        return f"https://hf/{repo_id}/{path_in_repo}"

    monkeypatch.setattr(hub, "_upload", fake_upload)

    data_path = tmp_path / "tiny_corpus.jsonl"
    with data_path.open("w") as f:
        for r in _TINY_ROWS:
            f.write(json.dumps(r) + "\n")

    real_cfg = R.arm_b_lora_config(R.B1, max_steps=4, seed=42)
    clamped = dataclasses.replace(
        real_cfg,
        batch_size=1,
        grad_accum=1,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        gradient_checkpointing=False,
        bf16=False,
        logging_steps=1,
        report_to="none",  # WANDB_INTENTIONALLY_DISABLED: offline CPU lifecycle test
        hf_upload=False,
    )

    out_dir = tmp_path / "train"
    overflow = {
        "repo": "fake/overflow",
        "path_prefix": "issue1112_rankem/b2_fullft_em",
        "residency_cap": 2,
    }
    cb = build_checkpoint_callback({1, 2, 3, 4}, overflow=overflow)
    train_lora(BASE_MODEL, str(data_path), str(out_dir), cfg=clamped, callbacks=[cb])

    # Every grid rung was uploaded, from an existing dir, to the private repo.
    uploaded_steps = {int(name.split("-")[1]) for name, *_ in calls}
    assert uploaded_steps == {1, 2, 3, 4}, uploaded_steps
    assert cb.uploaded_steps == {1, 2, 3, 4}
    for name, repo_id, repo_type, path_in_repo, was_dir in calls:
        assert repo_id == "fake/overflow" and repo_type == "model"
        assert path_in_repo == f"issue1112_rankem/b2_fullft_em/{name}"
        assert was_dir is True  # never delete before a confirmed upload

    # Residency cap: only the newest 2 rungs survive locally.
    local = sorted(int(p.name.split("-")[1]) for p in out_dir.glob("checkpoint-*"))
    assert local == [3, 4], f"cap=2 must keep the newest 2 local; got {local}"


# ── p5 upload: a pruned B2 rung is VERIFIED on overflow, never re-uploaded ─────


def test_phase_upload_records_pruned_b2_rung_from_overflow(tmp_path, monkeypatch) -> None:
    """The B2 selected rung, pruned locally after its in-training overflow upload,
    is VERIFIED present on overflow (file_exists on its config.json) and RECORDED
    — never re-uploaded from a missing local dir. Fakes ONLY the HF boundary;
    executes the real phase_upload body's fullft pruned branch."""
    cfg = _cfg(tmp_path, upload=True, dry_run=False, cells=(R.B2,))
    # B2 capture tensors present (the capture-upload leg has inputs).
    d = cfg.out_root / "capture" / R.B2 / "selected"
    d.mkdir(parents=True, exist_ok=True)
    (d / "pooled.pt").write_bytes(b"\x00")
    (d / "raw_rows.json").write_text("[]")
    # B2 installed at rung 200 but adapter_root has NO local checkpoint-200 (pruned).
    b2 = cfg.out_root / R.B2
    (b2 / "train").mkdir(parents=True, exist_ok=True)
    (b2 / "selection.json").write_text(json.dumps({"installed": True, "selected_step": 200}))
    (b2 / "build_result.json").write_text(json.dumps({"adapter_root": str(b2 / "train")}))

    from explore_persona_space.orchestrate import hub

    upload_calls: list[tuple] = []
    monkeypatch.setattr(hub, "_upload_folder_filtered", lambda *a, **k: None)
    monkeypatch.setattr(hub, "_upload", lambda *a, **k: (upload_calls.append(a), "https://hf/x")[1])
    monkeypatch.setattr(  # signature-conformant: the real helper REQUIRES what= (#1332 class)
        hub, "retry_transient", lambda fn, *, what, **kw: fn()
    )
    monkeypatch.setattr(
        hub, "upload_raw_completions_to_data_repo", lambda experiment_name, eval_results_dir: None
    )

    import huggingface_hub

    file_exists_paths: list[str] = []

    class _FakeApi:
        def repo_info(self, repo_id, repo_type):
            return types.SimpleNamespace(sha="cafe")

        def file_exists(self, repo_id, path, repo_type="model"):
            file_exists_paths.append(path)
            return True  # pruned rung IS on overflow

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    rec = D.phase_upload(cfg)

    repo_path = f"issue1112_{R.RANKEM_SLUG}/{R.B2}/checkpoint-200"
    recorded = rec.get("uploaded", rec)  # phase_upload nests the uploaded map
    # The pruned rung's presence was verified on overflow ...
    assert f"{repo_path}/config.json" in file_exists_paths
    # ... recorded under overflow:<path> ...
    assert f"overflow:{repo_path}" in recorded
    # ... and NEVER re-uploaded from a missing local dir.
    assert all(repo_path not in str(a) for a in upload_calls), "pruned rung must not be re-uploaded"


def test_phase_upload_raises_when_pruned_b2_rung_absent_from_overflow(
    tmp_path, monkeypatch
) -> None:
    """The fail-loud other half of the pruned-B2 branch: a selected rung that is
    NEITHER local (pruned) NOR present on overflow (file_exists False) must raise
    a clear RuntimeError — the crash-at-p5 guard fires with an actionable message
    instead of silently recording a nonexistent path. Same faked-HF-boundary setup
    as the verify-record test; executes the real phase_upload body's fullft
    pruned-absent branch."""
    cfg = _cfg(tmp_path, upload=True, dry_run=False, cells=(R.B2,))
    d = cfg.out_root / "capture" / R.B2 / "selected"
    d.mkdir(parents=True, exist_ok=True)
    (d / "pooled.pt").write_bytes(b"\x00")
    (d / "raw_rows.json").write_text("[]")
    b2 = cfg.out_root / R.B2
    (b2 / "train").mkdir(parents=True, exist_ok=True)  # no local checkpoint-200 (pruned)
    (b2 / "selection.json").write_text(json.dumps({"installed": True, "selected_step": 200}))
    (b2 / "build_result.json").write_text(json.dumps({"adapter_root": str(b2 / "train")}))

    from explore_persona_space.orchestrate import hub

    upload_calls: list[tuple] = []
    monkeypatch.setattr(hub, "_upload_folder_filtered", lambda *a, **k: None)
    monkeypatch.setattr(hub, "_upload", lambda *a, **k: (upload_calls.append(a), "https://hf/x")[1])
    monkeypatch.setattr(  # signature-conformant: the real helper REQUIRES what= (#1332 class)
        hub, "retry_transient", lambda fn, *, what, **kw: fn()
    )
    monkeypatch.setattr(
        hub, "upload_raw_completions_to_data_repo", lambda experiment_name, eval_results_dir: None
    )

    import huggingface_hub

    file_exists_paths: list[str] = []

    class _FakeApi:
        def repo_info(self, repo_id, repo_type):
            return types.SimpleNamespace(sha="cafe")

        def file_exists(self, repo_id, path, repo_type="model"):
            file_exists_paths.append(path)
            return False  # pruned rung is NOT on overflow -> must fail loud

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    repo_path = f"issue1112_{R.RANKEM_SLUG}/{R.B2}/checkpoint-200"
    with pytest.raises(RuntimeError, match=r"neither local nor on overflow"):
        D.phase_upload(cfg)
    # The guard consulted overflow before raising ...
    assert f"{repo_path}/config.json" in file_exists_paths
    # ... and never re-uploaded from the missing local dir.
    assert all(repo_path not in str(a) for a in upload_calls), "absent rung must not be re-uploaded"
