"""Unit tests for the recurring training-pipeline infra fixes (#7a, #10, #13).

Covers three behavior-preserving guardrails added to the train -> eval pipeline:

#7a  ``_delete_intermediate_merged`` honors upload-before-delete: it deletes a
     consumed intermediate merged dir ONLY when its required upload already ran,
     and PRESERVES it (loud warning) when the inline-upload fence skipped the
     upload.

#10  ``_warn_if_cvd_disagrees`` emits a WARNING (and does NOT change the value)
     when an inherited CUDA_VISIBLE_DEVICES disagrees with gpu_id. The clobbering
     assignment ``os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)`` stays
     load-bearing; this only warns about a likely-misconfigured launch.

#13  ``run_isolated`` round-trips a JSON payload through a fresh ``uv run python
     -m`` child (the module's own ``_echo_main`` entry point), and fails loud on a
     non-zero child exit. PLUS the EPM_ISOLATE_EVAL opt-in wiring: the refactored
     ``run_eval_phase`` returns exactly the legacy fragment keys, and the isolated
     eval path (real ``run_isolated`` -> fresh child -> ``run_eval_phase`` with
     the GPU-bound eval leaves faked) returns a json-equal fragment to the
     in-process path — the structure-identity bar for ever enabling the flag by
     default.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# #7a — upload-before-delete for intermediate merged dirs
# ---------------------------------------------------------------------------


def test_intermediate_merged_deleted_when_upload_attempted(tmp_path: Path) -> None:
    """Upload ran -> the intermediate merged dir is removed to reclaim disk."""
    from explore_persona_space.train.trainer import _delete_intermediate_merged

    merged = tmp_path / "phase1_merged"
    merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"weights")

    _delete_intermediate_merged(merged, upload_attempted=True, label="Phase 1")

    assert not merged.exists()


def test_intermediate_merged_preserved_when_upload_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Upload skipped (fence set) -> the dir is PRESERVED, never silently dropped."""
    from explore_persona_space.train.trainer import _delete_intermediate_merged

    merged = tmp_path / "phase1_merged"
    merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"weights")

    with caplog.at_level(logging.WARNING, logger="explore_persona_space.train.trainer"):
        _delete_intermediate_merged(merged, upload_attempted=False, label="Phase 1")

    assert merged.exists(), "un-uploaded intermediate must NOT be deleted"
    assert any("upload-before-delete" in rec.message for rec in caplog.records)


def test_intermediate_merged_missing_dir_is_noop(tmp_path: Path) -> None:
    """A non-existent dir is a no-op regardless of the upload flag."""
    from explore_persona_space.train.trainer import _delete_intermediate_merged

    missing = tmp_path / "does_not_exist"
    # Neither branch should raise.
    _delete_intermediate_merged(missing, upload_attempted=True, label="Phase 1")
    _delete_intermediate_merged(missing, upload_attempted=False, label="Phase 1")
    assert not missing.exists()


# ---------------------------------------------------------------------------
# #10 — CVD-disagreement warning (warn only; value unchanged)
# ---------------------------------------------------------------------------


def test_cvd_warning_fires_on_disagreement(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Inherited CVD != gpu_id -> WARNING fires, env value is left untouched."""
    from explore_persona_space.train import sft

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.train.sft"):
        sft._warn_if_cvd_disagrees(0)

    assert any("disagrees with cfg.gpu_id" in rec.message for rec in caplog.records), (
        "expected a CVD-disagreement WARNING"
    )
    # The helper must NOT mutate the env — the caller's assignment is what wins.
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_cvd_warning_silent_on_agreement(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Inherited CVD == gpu_id -> no warning."""
    from explore_persona_space.train import sft

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.train.sft"):
        sft._warn_if_cvd_disagrees(2)

    assert not any("disagrees with cfg.gpu_id" in rec.message for rec in caplog.records)


def test_cvd_warning_silent_when_env_unset(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No inherited CVD -> no warning (the common single-GPU launch)."""
    from explore_persona_space.train import sft

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.train.sft"):
        sft._warn_if_cvd_disagrees(0)

    assert not any("disagrees with cfg.gpu_id" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# #13 — run_isolated round-trip + fail-loud
# ---------------------------------------------------------------------------


def test_run_isolated_round_trips_payload() -> None:
    """run_isolated spawns a fresh child that echoes the payload back via JSON IPC."""
    from explore_persona_space.orchestrate.subprocess_isolation import run_isolated

    payload = {"seed": 42, "condition": "librarian", "nested": {"a": [1, 2, 3]}}
    result = run_isolated(
        "explore_persona_space.orchestrate.subprocess_isolation",
        payload,
        cwd=str(PROJECT_ROOT),
    )

    assert result["_echoed"] is True
    for key, value in payload.items():
        assert result[key] == value


def test_run_isolated_rejects_non_dict_payload() -> None:
    """A non-dict payload fails fast with TypeError before spawning anything."""
    from explore_persona_space.orchestrate.subprocess_isolation import run_isolated

    with pytest.raises(TypeError):
        run_isolated("explore_persona_space.orchestrate.subprocess_isolation", ["not", "a", "dict"])  # type: ignore[arg-type]


def test_run_isolated_fails_loud_on_nonzero_exit() -> None:
    """A child module that does not exist exits non-zero -> SubprocessIsolationError."""
    from explore_persona_space.orchestrate.subprocess_isolation import (
        SubprocessIsolationError,
        run_isolated,
    )

    with pytest.raises(SubprocessIsolationError):
        run_isolated(
            "explore_persona_space.orchestrate._module_that_does_not_exist_zzz",
            {"x": 1},
            cwd=str(PROJECT_ROOT),
        )


def test_echo_main_round_trips(tmp_path: Path) -> None:
    """The _echo_main entry point reads argv[1] and writes the echoed result to argv[2]."""
    from explore_persona_space.orchestrate.subprocess_isolation import _echo_main

    in_path = tmp_path / "in.json"
    out_path = tmp_path / "out.json"
    in_path.write_text(json.dumps({"k": "v"}))

    rc = _echo_main(["prog", str(in_path), str(out_path)])

    assert rc == 0
    result = json.loads(out_path.read_text())
    assert result == {"k": "v", "_echoed": True}


def test_echo_main_usage_error_on_missing_args() -> None:
    """Too few argv entries -> non-zero rc (usage error), no file written."""
    from explore_persona_space.orchestrate.subprocess_isolation import _echo_main

    assert _echo_main(["prog"]) == 2


# ---------------------------------------------------------------------------
# #13 — EPM_ISOLATE_EVAL opt-in: isolated eval path is structure-identical to
#        the in-process path (the bar for ever flipping the default ON)
# ---------------------------------------------------------------------------

# Fixed fake eval outputs for the in-process arm. MUST match the constants in
# ``tests/_fake_eval_child.py`` (the isolated arm) so a json-equal comparison
# across the two process paths is meaningful.
_FAKE_CAP = {"arc_challenge_logprob": 0.5, "correct": 5, "total": 10}
_FAKE_OOD = {
    "mmlu_pro": {"exact_match,custom-extract": 0.3},
    "gsm8k": {"exact_match,strict-match": 0.4},
}
_FAKE_ALIGN = {"overall_mean_aligned": 90.0, "overall_mean_coherent": 85.0}


def _patch_in_process_eval_leaves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the three GPU-bound eval leaves with the same fakes the child uses."""
    import explore_persona_space.eval.alignment as alignment_mod
    import explore_persona_space.eval.capability as capability_mod

    async def fake_alignment(*_args, **_kwargs):
        return dict(_FAKE_ALIGN)

    monkeypatch.setattr(
        capability_mod, "evaluate_capability_logprob", lambda *a, **k: dict(_FAKE_CAP)
    )
    monkeypatch.setattr(
        capability_mod,
        "evaluate_capability",
        lambda *a, **k: {key: dict(val) for key, val in _FAKE_OOD.items()},
    )
    monkeypatch.setattr(alignment_mod, "evaluate_alignment_quick", fake_alignment)


def test_run_eval_phase_in_process_returns_expected_fragment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The refactored in-process run_eval_phase returns exactly the legacy keys.

    Guards the structure contract: the fragment carries EXACTLY
    ``{phase}_capability / _mmlu_pro / _gsm8k / _alignment`` with the legacy
    types. A drift (extra key, renamed key, wrong type) fails here.
    """
    _patch_in_process_eval_leaves(monkeypatch)
    from explore_persona_space.orchestrate.runner import run_eval_phase

    fragment = run_eval_phase(
        "/some/merged/model",
        "post_em",
        materialize_merged=True,
        eval_base_model_id="Qwen/Qwen2.5-7B",
        judge_model="claude-sonnet-4-5-20250929",
        phase_dir=str(tmp_path / "post_em"),
    )

    assert set(fragment.keys()) == {
        "post_em_capability",
        "post_em_mmlu_pro",
        "post_em_gsm8k",
        "post_em_alignment",
    }
    assert fragment["post_em_capability"] == _FAKE_CAP
    assert fragment["post_em_mmlu_pro"] == 0.3
    assert fragment["post_em_gsm8k"] == 0.4
    assert fragment["post_em_alignment"] == {"aligned": 90.0, "coherent": 85.0}


def test_isolation_path_matches_in_process_structure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """EPM_ISOLATE_EVAL path returns a json-equal fragment to the in-process path.

    Runs both arms with the SAME fixed fake eval leaves and the SAME payload:
      * in-process: patched run_eval_phase called directly;
      * isolated: real run_isolated -> fresh child (tests._fake_eval_child) ->
        real eval_phase_child.main -> real run_eval_phase, fakes installed in
        the child process.
    json-equality across the two = proof that flipping EPM_ISOLATE_EVAL does not
    change the eval result structure. This is the bar for ever enabling the flag
    by default (numeric identity additionally needs a GPU equivalence run).
    """
    from explore_persona_space.orchestrate.runner import run_eval_phase
    from explore_persona_space.orchestrate.subprocess_isolation import run_isolated

    _patch_in_process_eval_leaves(monkeypatch)
    in_process = run_eval_phase(
        "/some/merged/model",
        "pre_em",
        materialize_merged=True,
        eval_base_model_id="Qwen/Qwen2.5-7B",
        judge_model="claude-sonnet-4-5-20250929",
        phase_dir=str(tmp_path / "pre_em"),
    )

    isolated = run_isolated(
        "tests._fake_eval_child",
        {
            "model_path": "/some/merged/model",
            "phase": "pre_em",
            "materialize_merged": True,
            "eval_base_model_id": "Qwen/Qwen2.5-7B",
            "judge_model": "claude-sonnet-4-5-20250929",
            "phase_dir": str(tmp_path / "pre_em"),
        },
        cwd=str(PROJECT_ROOT),
    )

    # JSON round-trip the in-process fragment so the comparison is on the same
    # footing as the child's on-disk JSON (e.g. tuple->list, int-key coercion).
    assert json.loads(json.dumps(in_process, default=str)) == isolated


def test_isolation_adapter_mode_skips_ood_in_both_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """LoRA-only adapter mode skips lm-eval OOD (None) identically in both paths.

    adapter_mode is computed the same way in the child as in-process
    (``not materialize_merged and model_path != eval_base_model_id``), so the
    OOD-skip branch (mmlu/gsm8k = None) must fire in both and stay json-equal.
    """
    from explore_persona_space.orchestrate.runner import run_eval_phase
    from explore_persona_space.orchestrate.subprocess_isolation import run_isolated

    _patch_in_process_eval_leaves(monkeypatch)
    payload = {
        "model_path": "/trained/adapter/dir",
        "phase": "post_em",
        "materialize_merged": False,
        "eval_base_model_id": "Qwen/Qwen2.5-7B",
        "judge_model": "claude-sonnet-4-5-20250929",
        "phase_dir": str(tmp_path / "post_em"),
    }
    in_process = run_eval_phase(
        payload["model_path"],
        payload["phase"],
        materialize_merged=payload["materialize_merged"],
        eval_base_model_id=payload["eval_base_model_id"],
        judge_model=payload["judge_model"],
        phase_dir=payload["phase_dir"],
    )

    # adapter_mode is on -> OOD benchmarks skipped, recorded as None.
    assert in_process["post_em_mmlu_pro"] is None
    assert in_process["post_em_gsm8k"] is None
    assert in_process["post_em_capability"] == _FAKE_CAP

    isolated = run_isolated("tests._fake_eval_child", payload, cwd=str(PROJECT_ROOT))
    assert json.loads(json.dumps(in_process, default=str)) == isolated
