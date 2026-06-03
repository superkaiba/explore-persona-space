"""CPU-only unit tests for the #474 round-3 (post on-pod smoke) fixes.

Covers:

- **FIX 1** — vLLM ``max_logprobs`` engine-arg behavior.
    ``_resolve_kl_topk`` reads Phase 0's ``vllm_max_logprobs_probe`` payload
    and floors the requested CLI ``--kl-topk`` to the largest K the engine
    accepted, returning ``mode="top-K-approx"`` (never ``"full"`` again).
    The eval main() must construct ``LLM(max_logprobs=max(kl_topk, 20))``
    so the per-request ``prompt_logprobs=kl_topk`` is allowed.

- **FIX 3** — ``PerEpochAdapterHFUploadCallback`` upload-bundle hygiene.
    ``_stage_clean_upload_bundle`` builds a clean ``_upload_ep{N}/`` dir
    containing ONLY allowlisted files (adapter + tokenizer files Phase 4
    and smoke download). EXCLUDES ``optimizer.pt``, ``rng_state.pth``,
    ``scheduler.pt``, ``trainer_state.json``, ``training_args.bin``.

Both tests are pure CPU + filesystem + JSON probes — no model, no vLLM,
no Trainer.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_TRAIN = Path(__file__).resolve().parent.parent / "scripts" / "i474_phase23_train.py"
_SCRIPT_EVAL = Path(__file__).resolve().parent.parent / "scripts" / "i474_phase4_eval.py"


@pytest.fixture(scope="module")
def i474_train_module():
    spec = importlib.util.spec_from_file_location("i474_phase23_train", _SCRIPT_TRAIN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i474_phase23_train"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def i474_eval_module(monkeypatch_module):
    # Phase 4 reads a default PREFLIGHT_PATH on import via module-level
    # constants; loading it is side-effect-free (no Trainer / vLLM imports
    # at module top).
    spec = importlib.util.spec_from_file_location("i474_phase4_eval", _SCRIPT_EVAL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i474_phase4_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped MonkeyPatch helper (the built-in ``monkeypatch`` fixture
    is function-scoped — incompatible with module-scoped consumers)."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


# ---------------------------------------------------------------- FIX 1


def test_resolve_kl_topk_skipped_when_zero(i474_eval_module):
    """``--kl-topk 0`` returns ``(0, "skipped")`` — sentinel for "no KL pass"."""
    K, mode = i474_eval_module._resolve_kl_topk(0)
    assert K == 0
    assert mode == "skipped"


def test_resolve_kl_topk_floors_to_phase0_max_when_exceeded(
    i474_eval_module, tmp_path, monkeypatch
):
    """Phase 0 records ``max_k_accepted``; CLI K > probe → floor to probe value.

    This is the load-bearing FIX 1 contract: the on-pod smoke showed that
    vLLM rejected ``prompt_logprobs=151643``. After this fix, Phase 0 records
    the largest K the engine actually accepts (e.g. 1000), and Phase 4 floors
    any larger CLI request to that value rather than re-attempting the
    rejected K.
    """
    preflight_dir = tmp_path / "eval_results" / "issue_474"
    preflight_dir.mkdir(parents=True)
    (preflight_dir / "preflight.json").write_text(
        json.dumps(
            {
                "schema_version": "i474_v1",
                "vllm_max_logprobs_probe": {
                    "max_k_accepted": 1000,
                    "tried": [1000],
                    "probe_failed": False,
                },
            }
        )
    )
    monkeypatch.setattr(i474_eval_module, "PREFLIGHT_PATH", preflight_dir / "preflight.json")

    # CLI requested K above the probe's accepted value → floored.
    K, mode = i474_eval_module._resolve_kl_topk(151643)
    assert K == 1000
    assert mode == "top-K-approx"


def test_resolve_kl_topk_passes_through_when_at_or_below_max(
    i474_eval_module, tmp_path, monkeypatch
):
    preflight_dir = tmp_path / "eval_results" / "issue_474"
    preflight_dir.mkdir(parents=True)
    (preflight_dir / "preflight.json").write_text(
        json.dumps(
            {
                "schema_version": "i474_v1",
                "vllm_max_logprobs_probe": {"max_k_accepted": 1000, "probe_failed": False},
            }
        )
    )
    monkeypatch.setattr(i474_eval_module, "PREFLIGHT_PATH", preflight_dir / "preflight.json")

    K, mode = i474_eval_module._resolve_kl_topk(500)
    assert (K, mode) == (500, "top-K-approx")
    K, mode = i474_eval_module._resolve_kl_topk(1000)
    assert (K, mode) == (1000, "top-K-approx")


def test_resolve_kl_topk_no_preflight_passes_cli_through(i474_eval_module, monkeypatch, tmp_path):
    """When preflight.json is absent the helper must NOT crash; it just
    warns + returns the CLI value as-is (the operator is on the hook for
    matching ``max_logprobs`` at engine construction)."""
    monkeypatch.setattr(i474_eval_module, "PREFLIGHT_PATH", tmp_path / "does-not-exist.json")
    K, mode = i474_eval_module._resolve_kl_topk(512)
    assert (K, mode) == (512, "top-K-approx")


def test_full_vocab_branch_removed(i474_eval_module):
    """Round-3 fix removed the ``"full"`` mode entirely.

    The old ``_resolve_full_vocab_or_fallback`` helper must NOT exist;
    callers must use ``_resolve_kl_topk``. The only modes are
    ``"top-K-approx"`` and ``"skipped"``.
    """
    assert not hasattr(i474_eval_module, "_resolve_full_vocab_or_fallback")
    assert hasattr(i474_eval_module, "_resolve_kl_topk")


def test_eval_source_constructs_engine_with_max_logprobs(i474_eval_module):
    """Static check: the eval main() must pass ``max_logprobs=...`` to LLM().

    Direct source read — the engine arg is the load-bearing change for
    FIX 1 (the rejection happened because the engine default is 20).
    """
    src = _SCRIPT_EVAL.read_text()
    assert "max_logprobs=max(kl_topk, 20)" in src, (
        "Phase 4 main() must construct LLM(max_logprobs=max(kl_topk, 20)) "
        "so per-request prompt_logprobs=kl_topk is allowed."
    )


# ---------------------------------------------------------------- FIX 3


def _make_fake_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    """Create a synthetic Trainer output_dir + checkpoint-X subdir on disk.

    Mirrors what SFTTrainer + PEFT produce on save_strategy=epoch:
      output_dir/
        tokenizer.json, tokenizer_config.json, special_tokens_map.json
        added_tokens.json, vocab.json, merges.txt
        chat_template.jinja, README.md
        checkpoint-100/
          adapter_model.safetensors    (placeholder bytes)
          adapter_config.json          (small JSON)
          optimizer.pt                 (placeholder bytes; MUST BE EXCLUDED)
          rng_state.pth                (placeholder bytes; MUST BE EXCLUDED)
          scheduler.pt                 (placeholder bytes; MUST BE EXCLUDED)
          trainer_state.json           (small JSON; MUST BE EXCLUDED)
          training_args.bin            (placeholder bytes; MUST BE EXCLUDED)
    """
    output_dir = tmp_path / "adapters" / "i474_pos_A1"
    output_dir.mkdir(parents=True)
    # Tokenizer files in output_dir (written once by SFTTrainer at init).
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

    ckpt_dir = output_dir / "checkpoint-100"
    ckpt_dir.mkdir()
    # Adapter files in checkpoint-X (PEFT writes them on save_strategy=epoch).
    (ckpt_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 1024)
    (ckpt_dir / "adapter_config.json").write_text(json.dumps({"r": 32, "alpha": 64}))
    # Full Trainer-state files in checkpoint-X (MUST be excluded from upload).
    (ckpt_dir / "optimizer.pt").write_bytes(b"\xff" * 4096)
    (ckpt_dir / "rng_state.pth").write_bytes(b"\xff" * 512)
    (ckpt_dir / "scheduler.pt").write_bytes(b"\xff" * 256)
    (ckpt_dir / "trainer_state.json").write_text(json.dumps({"global_step": 100}))
    (ckpt_dir / "training_args.bin").write_bytes(b"\xff" * 4096)
    return output_dir, ckpt_dir


def test_clean_upload_bundle_includes_adapter_and_tokenizer(i474_train_module, tmp_path):
    """Headline FIX 3 contract: bundle includes the allowlisted files."""
    output_dir, ckpt_dir = _make_fake_checkpoint(tmp_path)
    cb = i474_train_module.PerEpochAdapterHFUploadCallback(
        arm="pos", cid="A1", output_dir=str(output_dir), hf_repo="placeholder/repo"
    )
    upload_dir = cb._stage_clean_upload_bundle(ckpt_dir, target_ep=1)

    # Required-by-eval/smoke files must be present.
    required = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    )
    for fname in required:
        assert (upload_dir / fname).exists(), f"missing required {fname} in upload bundle"

    # Optional-but-allowlisted files also copied if present in source dirs.
    for fname in (
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
        "chat_template.jinja",
        "README.md",
    ):
        assert (upload_dir / fname).exists(), f"missing allowlisted {fname} in upload bundle"


def test_clean_upload_bundle_excludes_optimizer_and_full_state(i474_train_module, tmp_path):
    """The load-bearing FIX 3 exclusion contract.

    optimizer.pt + rng_state.pth + scheduler.pt + trainer_state.json +
    training_args.bin land in the local checkpoint-X dir but MUST NEVER
    appear in the upload bundle (round-3 smoke uploaded all of them to
    HF, bloating the repo).
    """
    output_dir, ckpt_dir = _make_fake_checkpoint(tmp_path)
    cb = i474_train_module.PerEpochAdapterHFUploadCallback(
        arm="pos", cid="A1", output_dir=str(output_dir), hf_repo="placeholder/repo"
    )
    upload_dir = cb._stage_clean_upload_bundle(ckpt_dir, target_ep=1)

    excluded = (
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    )
    for fname in excluded:
        assert not (upload_dir / fname).exists(), (
            f"FIX 3 VIOLATION: {fname} present in upload bundle "
            f"(should have been excluded by UPLOAD_EXCLUDED). Round-3 smoke "
            "uploaded these to HF; bundle must filter them out."
        )

    # And the source checkpoint-X dir still has them locally (we don't
    # delete them — only the upload bundle filters).
    for fname in excluded:
        assert (ckpt_dir / fname).exists(), f"source {fname} accidentally removed"


def test_upload_bundle_idempotent_on_retry(i474_train_module, tmp_path):
    """Calling _stage_clean_upload_bundle twice for the same target_ep
    must wipe the stale stage and rebuild — supports retry-on-upload-fail."""
    output_dir, ckpt_dir = _make_fake_checkpoint(tmp_path)
    cb = i474_train_module.PerEpochAdapterHFUploadCallback(
        arm="pos", cid="A1", output_dir=str(output_dir), hf_repo="placeholder/repo"
    )
    upload_dir = cb._stage_clean_upload_bundle(ckpt_dir, target_ep=1)
    # Pollute the bundle to simulate a partial / stale state.
    (upload_dir / "stale_garbage.bin").write_bytes(b"\x00" * 64)
    upload_dir2 = cb._stage_clean_upload_bundle(ckpt_dir, target_ep=1)
    assert upload_dir == upload_dir2
    assert not (upload_dir2 / "stale_garbage.bin").exists()


def test_allowlist_and_excluded_sets_disjoint(i474_train_module):
    """Sanity: no file is both allowlisted AND excluded."""
    cls = i474_train_module.PerEpochAdapterHFUploadCallback
    overlap = set(cls.UPLOAD_ALLOWLIST) & set(cls.UPLOAD_EXCLUDED)
    assert overlap == set(), f"allowlist ∩ excluded must be empty; got {overlap}"


def test_eval_and_smoke_required_files_in_allowlist(i474_train_module):
    """The eval + smoke download paths grep the source files for required
    filenames. Every file they download MUST appear in the upload allowlist
    or the eval-side _ep{N} fetch will silently fail-loud on missing files.
    """
    cls = i474_train_module.PerEpochAdapterHFUploadCallback
    allow = set(cls.UPLOAD_ALLOWLIST)

    # The 5 files BOTH phase4 + smoke download. KEEP THIS LIST in sync
    # with their _download_adapters / _resolve_adapter_path needed_files.
    required_by_downstream = {
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    }
    missing = required_by_downstream - allow
    assert missing == set(), (
        f"FIX 3 BROKEN: downstream eval/smoke require {missing} but "
        f"PerEpochAdapterHFUploadCallback.UPLOAD_ALLOWLIST = {sorted(allow)}"
    )
