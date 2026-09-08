"""Synthetic contract checks for the opt-in repaired China GPU pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import create_autospec

import pytest
import torch

from scripts import issue952_china_definitive_gpu as GPU
from scripts import issue952_china_repair_bank as BANK


def _bank(tmp_path: Path) -> tuple[Path, Path]:
    """Create all 85 sources without reading any real questions or responses."""
    base_rows, metadata = [], {}
    for source in range(85):
        source_id = f"s{source:03d}"
        metadata[source_id] = {"control_country_en": "Canada", "control_country_zh": "加拿大"}
        for language in ("en", "zh"):
            for frame in ("direct", "academic"):
                for content in ("sensitive_country_neutral", "matched_non_china"):
                    base_rows.append(
                        {
                            "item_id": f"base-{source_id}-{language}-{frame}-{content}",
                            "source_prompt_id": source_id,
                            "topic": f"t{source % 12}",
                            "language": language,
                            "frame": frame,
                            "content": content,
                            "prompt": f"Synthetic fixture {source_id} {language} {frame} {content}",
                            "audit_pass": True,
                        }
                    )
    bank = tmp_path / "prompt_bank.jsonl"
    audit = tmp_path / "bank_audit_report.json"
    GPU._write_jsonl(bank, BANK.render(base_rows, metadata))
    GPU._write_json(
        audit,
        {
            "passed": True,
            "contract": BANK.CONTRACT,
            "prompt_bank_sha256": GPU._sha256(bank),
            "passing_item_ids": sorted(metadata),
            "n_audit_passing_items": 85,
            "source_bank_sha256": "a" * 64,
            "metadata_sha256": "b" * 64,
            "independent_audit_sha256": "c" * 64,
        },
    )
    _marker(bank, audit)
    return bank, audit


def _marker(bank: Path, audit: Path) -> None:
    """Bind synthetic input persistence evidence to its exact bank and audit bytes."""
    GPU._write_json(
        bank.parent / "upload_verified.json",
        {
            "data_revision": "d" * 40,
            "prompt_bank_sha256": GPU._sha256(bank),
            "bank_audit_report_sha256": GPU._sha256(audit),
        },
    )


class _Tokenizer:
    """Character-token fixture used only at the tokenizer's external boundary."""

    chat_template = "synthetic-chat-template"
    model_max_length = 32768

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert add_generation_prompt is True
        rendered = "fixture-user:" + messages[0]["content"] + "\nfixture-assistant:"
        return self.encode(rendered) if tokenize else rendered


def test_repaired_bank_full_and_smoke_keep_every_factorial_cell(tmp_path: Path) -> None:
    bank, audit = _bank(tmp_path)
    full, report = GPU._load_bank(bank, audit, False, study="repaired-v2")
    smoke, _ = GPU._load_bank(bank, audit, True, study="repaired-v2")
    assert len(full) == 1360
    assert len(smoke) == 160
    assert len(GPU._expected_rollout_ids(full)) == 10880
    assert len(GPU._expected_rollout_ids(smoke)) == 1280
    assert GPU._accepted_bank_identity(full, report)["n_accepted_source_items"] == 85
    checks = GPU._repaired_token_pair_checks(full, _Tokenizer())
    assert checks["n_pairs"] == 680
    assert checks["token_checks_run"] is True
    assert all(delta > 0 for delta in checks["added_token_lengths"])
    with pytest.raises(RuntimeError, match="cardinality"):
        GPU._load_bank(bank, audit, False)


@pytest.mark.parametrize("mutation", ["contract", "subset", "language", "cue", "metadata"])
def test_repaired_bank_rejects_false_audits_and_malformed_cells(
    tmp_path: Path, mutation: str
) -> None:
    bank, audit = _bank(tmp_path)
    rows = GPU._read_jsonl(bank)
    report = json.loads(audit.read_text())
    if mutation == "contract":
        report["contract"] = "old-contract"
    elif mutation == "subset":
        rejected = report["passing_item_ids"].pop()
        report["n_audit_passing_items"] = 84
        for row in rows:
            row["audit_pass"] = row["source_prompt_id"] != rejected
    elif mutation == "language":
        rows[0]["language"] = "fr"
    elif mutation == "cue":
        row = next(row for row in rows if row["country_cue"] == "present")
        row["prompt"] = row["prompt"][len(row["cue_text"]) :]
    else:
        report.pop("independent_audit_sha256")
    GPU._write_jsonl(bank, rows)
    report["prompt_bank_sha256"] = GPU._sha256(bank)
    GPU._write_json(audit, report)
    with pytest.raises((RuntimeError, ValueError)):
        GPU._load_bank(bank, audit, False, study="repaired-v2")


def test_repaired_token_guard_executes_before_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank, audit = _bank(tmp_path)

    class IdenticalTokenizer(_Tokenizer):
        def encode(self, text, *, add_special_tokens=False):
            return [1]

    monkeypatch.setattr(
        GPU, "_tokenizer", create_autospec(GPU._tokenizer, return_value=IdenticalTokenizer())
    )
    monkeypatch.setattr(
        GPU,
        "_assert_package_versions",
        create_autospec(GPU._assert_package_versions, return_value={}),
    )
    with pytest.raises(ValueError, match="identical inputs"):
        GPU.phase_generate(tmp_path / "smoke", bank, audit, True, 90, study="repaired-v2")
    assert not (tmp_path / "smoke" / "raw_completions").exists()


def test_repaired_study_seeds_are_disjoint_and_v1_is_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(GPU, "_git_sha", create_autospec(GPU._git_sha, return_value="code"))
    v1 = GPU._regime("bank", False)
    production = GPU._regime("bank", False, study="repaired-v2")
    smoke = GPU._regime("bank", True, study="repaired-v2")
    assert "study" not in v1
    assert v1["seed_base"] == 952000
    assert production["seed_base"] == 9520000
    assert smoke["seed_base"] == 9530880
    assert production["seed_base"] + 10880 == smoke["seed_base"]
    assert GPU._smoke_generation_compatibility(production) == GPU._smoke_generation_compatibility(
        smoke
    )
    assert GPU._output_prefix(False, 1) == f"{GPU.HF_PREFIX}/attempt1"
    assert GPU._output_prefix(True, 2, "repaired-v2") == f"{BANK.HF_PREFIX}/attempt2/smoke"
    args = GPU.build_argparser().parse_args(["--phase", "gen", "--out-root", "/tmp/test"])
    assert args.study == "v1"
    with pytest.raises(RuntimeError, match="selected study"):
        GPU._validate_study(production, "v1")
    with pytest.raises(RuntimeError, match="selected study"):
        GPU._validate_study(v1, "repaired-v2")


def test_repaired_input_stage_freezes_audit_and_study(tmp_path: Path) -> None:
    bank, audit = _bank(tmp_path / "inputs")
    out = tmp_path / "out"
    assert GPU.stage_inputs(out, bank, audit, study="repaired-v2") == (bank, audit)
    stage = json.loads((out / "manifests" / "input_stage.json").read_text())
    assert stage["study"] == "repaired-v2"
    assert stage["hf_prefix"] == BANK.HF_PREFIX
    assert stage["n_expected_rollouts"] == 10880
    report = json.loads(audit.read_text())
    report["metadata_sha256"] = "e" * 64
    GPU._write_json(audit, report)
    _marker(bank, audit)
    with pytest.raises(RuntimeError, match="frozen first-phase snapshot"):
        GPU.stage_inputs(out, bank, audit, study="repaired-v2")


def test_repaired_caps_use_finish_reason_and_per_cell_threshold() -> None:
    rows = [
        {
            "item_id": f"synthetic-{content}-{i}",
            "language": "en",
            "content": content,
            "frame": "direct",
            "finish_reason": "length" if i < truncated else "stop",
            "completion_tokens": GPU.MAX_NEW_TOKENS,
        }
        for content, truncated in (("sensitive_full", 3), ("matched_non_china", 2))
        for i in range(100)
    ]
    report = GPU._repaired_cap_diagnostics(rows)
    assert report["cap_extension_required_item_ids"] == [
        f"synthetic-sensitive_full-{i}" for i in range(3)
    ]
    assert report["cap_by_cell"]["en:matched_non_china:direct"]["extension_required"] is False


def test_repaired_capture_reports_zeros_without_turning_them_into_missing_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank, audit = _bank(tmp_path)
    rows, _ = GPU._load_bank(bank, audit, True, study="repaired-v2")
    monkeypatch.setattr(GPU, "HIDDEN", 2)
    vc = torch.zeros((160, len(GPU.LAYERS), 2))
    report = GPU._repaired_capture_diagnostics(rows, vc)
    assert report["n_pairs"] == 80
    assert report["n_zero_delta_pairs_by_layer"] == [80, 80, 80]
    vc[0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="nonfinite"):
        GPU._repaired_capture_diagnostics(rows, vc)


def _fake_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Emulate immutable Hub revisions behind signature-checked SDK boundaries."""
    remote: dict[str, bytes] = {}
    snapshots: dict[str, dict[str, bytes]] = {}
    verified: list[tuple[str, str]] = []

    def commit():
        revision = f"{len(snapshots) + 1:040x}"
        snapshots[revision] = dict(remote)
        return SimpleNamespace(oid=revision)

    def upload_folder(*, repo_id, repo_type, folder_path, path_in_repo, commit_message):
        assert repo_id == GPU.HF_REPO and repo_type == "dataset" and commit_message
        assert path_in_repo.startswith(BANK.HF_PREFIX + "/attempt1")
        folder = Path(folder_path)
        for path in folder.rglob("*"):
            if path.is_file():
                remote[f"{path_in_repo}/{path.relative_to(folder).as_posix()}"] = path.read_bytes()
        return commit()

    def upload_file(*, repo_id, repo_type, path_or_fileobj, path_in_repo, commit_message):
        assert repo_id == GPU.HF_REPO and repo_type == "dataset" and commit_message
        remote[path_in_repo] = Path(path_or_fileobj).read_bytes()
        return commit()

    def file_exists(repo_id, filename, *, repo_type, revision):
        assert repo_id == GPU.HF_REPO and repo_type == "dataset"
        return filename in snapshots[revision]

    api = create_autospec(GPU.HfApi, instance=True)
    api.upload_folder.side_effect = upload_folder
    api.upload_file.side_effect = upload_file
    api.file_exists.side_effect = file_exists
    monkeypatch.setattr(GPU, "HfApi", create_autospec(GPU.HfApi, return_value=api))
    tokenizer_artifact = tmp_path / "tokenizer-fixture.json"
    GPU._write_json(tokenizer_artifact, {"synthetic": True})

    def download(repo_id, filename, *, repo_type, revision, local_dir=None, force_download=False):
        if repo_id == GPU.MODEL:
            assert repo_type == "model" and revision == GPU.MODEL_REV
            return str(tokenizer_artifact)
        assert repo_id == GPU.HF_REPO and repo_type == "dataset" and force_download
        verified.append((revision, filename))
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(snapshots[revision][filename])
        return str(target)

    monkeypatch.setattr(
        GPU, "hf_hub_download", create_autospec(GPU.hf_hub_download, side_effect=download)
    )
    return verified


def test_repaired_smoke_then_production_execute_all_phase_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run real phase guards, files, shard loops, and byte checks over synthetic boundaries."""
    bank, audit = _bank(tmp_path / "inputs")
    tok = _Tokenizer()
    monkeypatch.setattr(GPU, "HIDDEN", 2)
    for name, value in (
        ("_tokenizer", tok),
        ("_git_sha", "f" * 40),
        ("_assert_package_versions", {}),
        ("_package_versions", {}),
        ("_load_hf_model", object()),
        ("_smoke_map_consumer_probe", {"map_revision": GPU.MAP_REV, "synthetic_boundary": True}),
    ):
        monkeypatch.setattr(GPU, name, create_autospec(getattr(GPU, name), return_value=value))
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        create_autospec(torch.cuda.max_memory_allocated, return_value=0),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", create_autospec(torch.cuda.empty_cache))

    class SamplingParams:
        def __init__(self, *, n, temperature, top_p, max_tokens, seed):
            assert (n, temperature, top_p, max_tokens) == (1, 1.0, 0.95, 2048)
            self.seed = seed

    class LLM:
        def __init__(
            self,
            *,
            model,
            revision,
            tokenizer_revision,
            dtype,
            max_model_len,
            trust_remote_code,
            gpu_memory_utilization,
            seed,
        ):
            assert model == GPU.MODEL and revision == tokenizer_revision == GPU.MODEL_REV
            assert dtype == "bfloat16" and max_model_len >= 4096
            assert trust_remote_code and gpu_memory_utilization == 0.82
            assert seed in (9520000, 9530880)

        def generate(self, prompts, params, *, use_tqdm):
            assert not use_tqdm
            return [
                SimpleNamespace(
                    prompt_token_ids=tok.encode(prompt),
                    outputs=[
                        SimpleNamespace(
                            text=f"Synthetic answer {param.seed}",
                            token_ids=[10, 20],
                            finish_reason="stop",
                        )
                    ],
                    metrics=SimpleNamespace(finished_time=2.0, arrival_time=1.0),
                )
                for prompt, param in zip(prompts, params, strict=True)
            ]

    vllm = ModuleType("vllm")
    vllm.LLM = LLM
    vllm.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)

    def contexts(model, tokenizer, rows, batch_size):
        assert tokenizer is tok and batch_size == 4
        return torch.tensor(
            [[[float(len(tok.encode(row["prompt"]))), 1.0] for _ in GPU.LAYERS] for row in rows]
        )

    def answers(model, tokenizer, prompts, rows, batch_size):
        assert tokenizer is tok and batch_size == 4
        bounds = []
        for row in rows:
            ctx = len(GPU._context_ids(tok, prompts[row["prompt_id"]]))
            completion = len(row["completion_token_ids"])
            bounds.append(
                {
                    "ctx_len": ctx,
                    "completion_len": completion,
                    "span_start": ctx,
                    "span_end": ctx + completion,
                    "tail_end": ctx + completion + 2,
                }
            )
        return torch.ones((len(rows), len(GPU.LAYERS), GPU.HIDDEN)), [], bounds

    monkeypatch.setattr(
        GPU, "_capture_contexts", create_autospec(GPU._capture_contexts, side_effect=contexts)
    )
    monkeypatch.setattr(
        GPU, "_capture_answers", create_autospec(GPU._capture_answers, side_effect=answers)
    )
    verified = _fake_hub(tmp_path, monkeypatch)
    smoke_report = None
    for smoke, n_prompts, n_rows in ((True, 160, 1280), (False, 1360, 10880)):
        out = tmp_path / ("smoke" if smoke else "production")
        GPU.stage_inputs(out, bank, audit, study="repaired-v2")
        generated = GPU.phase_generate(
            out, bank, audit, smoke, 90, smoke_report, study="repaired-v2"
        )
        assert (generated["n_prompts"], generated["n_rows"]) == (n_prompts, n_rows)
        assert generated["regime"]["country_cue_validation"]["n_pairs"] == 680
        assert generated["cap_extension_required_item_ids"] == []
        rollouts = GPU._read_jsonl(out / "raw_completions" / "rollouts.jsonl")
        assert len({row["seed"] for row in rollouts}) == n_rows
        assert all(row["bank_contract"] == BANK.CONTRACT for row in rollouts)
        assert all(row["completion_token_ids"] == [10, 20] for row in rollouts)
        assert (
            len(list((out / "raw_completions").glob("rollouts_p*.jsonl"))) == (n_prompts + 89) // 90
        )
        with pytest.raises(RuntimeError, match="selected study"):
            GPU.phase_upload_raw(out)
        GPU.phase_upload_raw(out, study="repaired-v2")
        captured = GPU.phase_capture(
            out, bank, audit, smoke, 4, 720, smoke_report, study="repaired-v2"
        )
        assert (captured["n_contexts"], captured["n_answer_rows"]) == (n_prompts, n_rows)
        assert len(captured["va_files"]) == (n_rows + 719) // 720
        assert captured["country_cue_capture"]["n_pairs"] == n_prompts // 2
        assert captured["country_cue_capture"]["n_zero_delta_pairs_by_layer"] == [0, 0, 0]
        uploaded = GPU.phase_upload_capture(out, study="repaired-v2")
        assert uploaded["verified_files"] == len(captured["va_files"]) + 1
        final = GPU.phase_finalize(out, study="repaired-v2")
        assert final["hf_prefix"] == BANK.HF_PREFIX
        assert (out / "issue952_china_definitive_done.json").exists()
        if smoke:
            smoke_report = out / "manifests" / "smoke_timing.json"
            assert final["smoke_timing"]["n_expected_production_rollouts"] == 10880
            assert final["smoke_timing"]["smoke_draws"] == 1280
        prefix = GPU._output_prefix(smoke, 1, "repaired-v2")
        tensor_paths = {
            f"{prefix}/analysis_tensors/{name}" for name in ("vc.pt", *captured["va_files"])
        }
        assert tensor_paths <= {
            path for rev, path in verified if rev == uploaded["tensor_payload_revision"]
        }
        # Resume the same phase only against the complete content-keyed shard manifests.
        resumed = GPU.phase_generate(out, bank, audit, smoke, 90, smoke_report, study="repaired-v2")
        assert resumed["n_resumed_shards"] == (n_prompts + 89) // 90
        assert resumed["rollouts_sha256"] == generated["rollouts_sha256"]
