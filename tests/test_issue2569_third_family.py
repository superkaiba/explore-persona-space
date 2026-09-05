"""CPU-only contracts for the issue #2569 third-family follow-up."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2569_third_family as TF  # noqa: E402
import issue2569_xmodel_capture as XC  # noqa: E402


class NoSpecialsTokenizer:
    bos_token_id = 9
    chat_template = "no-specials-v1"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize
        text = "".join(f"<{row['role']}>{row['content']}" for row in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(char) for char in text]}


def test_olmo_registry_and_template_policy_are_exact():
    spec = XC.MODEL_SPECS["olmo"]
    assert spec == {
        "model_id": "allenai/Olmo-3-7B-Instruct",
        "revision": "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
        "n_layers": 32,
        "hidden": 4096,
        "default_layers": (16, 22, 30),
        "template_policy": "default_equals_false",
    }
    probe = XC.template_probe(NoSpecialsTokenizer(), "olmo")
    assert probe["gen_suffix"] == "<assistant>"


def test_olmo_stack_probe_checks_version_and_structural_rope(monkeypatch):
    fake = types.ModuleType("transformers")
    fake.__version__ = "5.15.0"

    class AutoConfig:
        @staticmethod
        def from_pretrained(model_id, revision):
            assert model_id == XC.MODEL_SPECS["olmo"]["model_id"]
            assert revision == XC.MODEL_SPECS["olmo"]["revision"]
            return types.SimpleNamespace(
                rope_parameters={
                    "full_attention": {"rope_type": "yarn"},
                    "sliding_attention": {"rope_type": "default"},
                }
            )

    fake.AutoConfig = AutoConfig
    monkeypatch.setitem(sys.modules, "transformers", fake)
    record = XC.assert_model_stack(XC.MODEL_SPECS["olmo"])
    assert record["olmo_rope_probe"] == {
        "full_attention": "yarn",
        "sliding_attention": "default",
    }

    fake.__version__ = "5.12.0"
    with pytest.raises(RuntimeError, match=r"transformers>=5\.13\.0"):
        XC.assert_model_stack(XC.MODEL_SPECS["olmo"])


def test_vllm_accelerator_compat_rejects_flashinfer_and_checks_compile_import(monkeypatch):
    imported = []
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(importlib, "import_module", lambda name: imported.append(name))
    record = TF.assert_vllm_accelerator_compat()
    assert imported == ["vllm.compilation.backends"]
    assert record == {
        "banned_distributions_absent": ["flashinfer-python"],
        "launch_env": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        "compile_backend_import": "pass",
    }

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "flashinfer" else None,
    )
    with pytest.raises(RuntimeError, match="banned accelerator"):
        TF.assert_vllm_accelerator_compat()


def test_workload_removes_incompatible_flashinfer_after_vllm_install():
    source = (REPO_ROOT / "scripts" / "issue2569_third_family_workload.sh").read_text()
    install = source.index("uv pip install --python .venv/bin/python")
    uninstall = source.index("uv pip uninstall --python .venv/bin/python flashinfer-python")
    assert install < uninstall
    assert "export VLLM_USE_FLASHINFER_SAMPLER=0" in source


def test_workload_uses_batch_one_for_alternate_writer_qwen_identity_gate():
    source = (REPO_ROOT / "scripts" / "issue2569_third_family_workload.sh").read_text()
    assert "gate_args+=(--qwen-gate identity --max-batch-rows 1)" in source
    assert "--phase verify-capture" in source
    assert "capture_model resume-skip" in source


def test_verify_capture_accepts_bound_local_and_remote_outputs(tmp_path, monkeypatch, capsys):
    root = tmp_path / "capture"
    root.mkdir()
    rows = [
        {"ci": ci, "corpus": "lmsys", "prompt": f"p{ci}", "response": f"a{ci}"}
        for ci in range(10_000)
    ]
    with open(root / "texts_kept.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    final = root / "final"
    final.mkdir(parents=True)
    files = TF._bundle_names("qwen")
    ci_sha = TF._sha_int64(np.arange(10_000, dtype=np.int64))
    (final / "qwen_finalize_meta.json").write_text(
        json.dumps(
            {
                "model": "qwen",
                "realized": 10_000,
                "n_selected_texts": 10_000,
                "drops": {},
                "layers": list(TF.TRUE_LAYERS["qwen"]),
                "files": files,
                "ci_sha256": ci_sha,
            }
        )
    )
    gate_dir = root / "gates"
    gate_dir.mkdir()
    binding = {
        "model_id": TF.MODEL_IDS["qwen"],
        "layers": list(TF.TRUE_LAYERS["qwen"]),
        "device": "cuda",
        "max_capture_tokens": 8192,
        "batch_tokens": 8192,
        "max_batch_rows": 1,
        "gate_rows": XC.GATE_ROWS,
        "selection_ci_sha256": ci_sha,
        "texts_sha256": XC._texts_content_sha(rows),
        "n_texts": 10_000,
    }
    (gate_dir / "identity_gate_qwen.json").write_text(
        json.dumps({"verdict": "PASS", "regime": binding})
    )
    pilot_params = {
        key: binding[key]
        for key in (
            "model_id",
            "layers",
            "max_capture_tokens",
            "batch_tokens",
            "max_batch_rows",
            "device",
        )
    }
    (gate_dir / "pilot_gate_qwen.json").write_text(
        json.dumps({"verdict": "PASS", "capture_params": pilot_params})
    )
    monkeypatch.setattr(
        TF,
        "_verify_bundle",
        lambda *_args, **_kwargs: {"n": 10_000, "ci_sha256": ci_sha},
    )

    expected_remote = {
        f"prefix/{name}" for name in [*files, "qwen_finalize_meta.json"]
    }

    class FakeApi:
        def get_paths_info(self, *_args, **_kwargs):
            return [types.SimpleNamespace(path=path) for path in expected_remote]

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    args = types.SimpleNamespace(
        capture_root=str(root),
        capture_model="qwen",
        capture_prefix="prefix",
        capture_rows=10_000,
        capture_device="cuda",
        capture_max_tokens=8192,
        capture_batch_tokens=8192,
        capture_max_batch_rows=1,
        analysis_rows=10_000,
        hf_data_repo="repo",
    )
    TF.phase_verify_capture(args)
    assert "capture resume-skip verified" in capsys.readouterr().out


def _write_bundle(root: Path, model: str, layer: int, tag: str, ci: list[int]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    hidden = XC.MODEL_SPECS[model]["hidden"]
    payload = {
        "x": np.zeros((len(ci), hidden), dtype=np.float16),
        "codec": "fp16",
        "ci": np.asarray(ci, dtype=np.int64),
        "corpus": ["lmsys"] * len(ci),
        "layer": layer,
        "slot": "v_C" if tag == "vc" else "v_A",
        "model_id": XC.MODEL_SPECS[model]["model_id"],
    }
    torch.save(payload, root / f"{model}_{tag}_L{layer}.pt")


def _write_model(root: Path, model: str, ci: list[int]) -> None:
    for layer in TF.TRUE_LAYERS[model]:
        for tag in ("vc", "va"):
            _write_bundle(root, model, layer, tag, ci)


def test_build_pairs_and_validate_complete_bank(tmp_path):
    ci = [101, 202]
    qbank = tmp_path / "bank" / "qwriter" / "final"
    lbank = tmp_path / "bank" / "lwriter" / "final"
    q_olmo = tmp_path / "capture" / "qwriter_olmo" / "final"
    owriter = tmp_path / "capture" / "owriter" / "final"
    for model in ("qwen", "llama"):
        _write_model(qbank, model, ci)
    _write_model(q_olmo, "olmo", ci)
    for model in ("qwen", "llama", "olmo"):
        _write_model(lbank, model, ci)
        _write_model(owriter, model, ci)

    args = types.SimpleNamespace(work_root=str(tmp_path), analysis_rows=2, atlas_min_rows=2)
    TF.phase_validate_bank(args)
    TF.phase_build_pairs(args)

    bank = json.loads((tmp_path / "bank" / "three_by_three_manifest.json").read_text())
    assert bank["complete"] is True and len(bank["cells"]) == 9
    qo = json.loads((tmp_path / "pairs" / "qo" / "pair_manifest.json").read_text())
    lo = json.loads((tmp_path / "pairs" / "lo" / "pair_manifest.json").read_text())
    assert qo["compatibility_aliases"] == {"qwen": "qwen", "llama": "olmo"}
    assert lo["compatibility_aliases"] == {"qwen": "llama", "llama": "olmo"}
    alias = tmp_path / "pairs" / "lo" / "source_writer" / "qwen_vc_L14.pt"
    loaded = torch.load(alias, map_location="cpu", weights_only=False)
    assert loaded["model_id"] == XC.MODEL_SPECS["llama"]["model_id"]
    assert loaded["layer"] == 16


def test_reconstruct_completion_shards_is_regime_and_roster_bound(tmp_path):
    ci = np.asarray([3, 8], dtype=np.int64)
    regime = {"ci_sha256": TF._sha_int64(ci)}
    TF._atomic_json(tmp_path / "regime.json", regime)
    TF._atomic_json(tmp_path / "audit.json", {"n_rows": 2})
    TF._atomic_json(
        tmp_path / "raw_completions" / "shard00000.json",
        {
            "regime": regime,
            "rows": [
                {"ci": 3, "drop_reason": None, "response": "a"},
                {"ci": 8, "drop_reason": None, "response": "b"},
            ],
        },
    )
    rec = TF.reconstruct_completion_shards(tmp_path)
    assert rec["n_rows"] == rec["n_kept"] == 2
    assert len((tmp_path / "answers.jsonl").read_text().splitlines()) == 2

    regime["ci_sha256"] = "bad"
    TF._atomic_json(tmp_path / "regime.json", regime)
    TF._atomic_json(
        tmp_path / "raw_completions" / "shard00000.json",
        {
            "regime": regime,
            "rows": [
                {"ci": 3, "drop_reason": None, "response": "a"},
                {"ci": 8, "drop_reason": None, "response": "b"},
            ],
        },
    )
    with pytest.raises(RuntimeError, match="roster hash"):
        TF.reconstruct_completion_shards(tmp_path)


def test_materialize_candidate_source_recovers_historical_order_and_content(tmp_path):
    qwriter_rows = [
        {"ci": 8, "corpus": "wildchat", "prompt": "p8", "response": "q8"},
        {"ci": 3, "corpus": "lmsys", "prompt": "p3", "response": "q3"},
        {"ci": 11, "corpus": "wildchat", "prompt": "p11", "response": "q11"},
    ]
    historical_rows = [
        {"ci": 3, "corpus": "lmsys", "prompt": "p3", "response": "l3"},
        {"ci": 8, "corpus": "wildchat", "prompt": "p8", "response": "l8"},
    ]
    expected = [qwriter_rows[1], qwriter_rows[0]]
    source = tmp_path / "source_qwen" / "texts_kept.jsonl"
    raw = tmp_path / "gen_llama_s42"
    TF._atomic_jsonl(source, qwriter_rows)
    TF._atomic_jsonl(raw / "answers.jsonl", historical_rows)
    TF._atomic_json(
        raw / "regime.json",
        {
            "source_rows": 2,
            "ci_sha256": TF._sha_int64(np.asarray([3, 8], dtype=np.int64)),
            "source_text_sha256": XC._texts_content_sha(expected),
        },
    )

    record = TF.materialize_candidate_source(
        qwriter_source=source,
        raw_completion_root=raw,
        destination=tmp_path / "source_candidate",
    )

    candidate = [
        json.loads(line)
        for line in (tmp_path / "source_candidate" / "texts_kept.jsonl").read_text().splitlines()
    ]
    assert candidate == expected
    assert record["n_full_qwriter_source"] == 3
    assert record["n_candidate"] == 2
    assert record["candidate_ci_sha256"] == record["pinned_regime_ci_sha256"]
    assert record["candidate_text_sha256"] == record["pinned_regime_text_sha256"]

    historical_rows[0]["prompt"] = "drifted"
    TF._atomic_jsonl(raw / "answers.jsonl", historical_rows)
    with pytest.raises(RuntimeError, match="prompt drift"):
        TF.materialize_candidate_source(
            qwriter_source=source,
            raw_completion_root=raw,
            destination=tmp_path / "source_candidate",
        )


def test_cap_hit_roster_and_merge_replace_only_capped_rows(tmp_path):
    base = tmp_path / "base"
    topup = tmp_path / "topup"
    merged = tmp_path / "merged"
    base_rows = [
        {
            "ci": ci,
            "corpus": "lmsys",
            "prompt": f"p{ci}",
            "response": f"base{ci}",
            "finish_reason": "length" if ci in {1, 3} else "stop",
            "response_tokens": 1024 if ci in {1, 3} else 10,
            "drop_reason": None,
            "seed": 42,
        }
        for ci in range(100)
    ]
    TF._atomic_jsonl(base / "answers.jsonl", base_rows)
    TF._atomic_json(
        base / "audit.json",
        {"n_rows": 100, "finish_reasons": {"length": 2, "stop": 98}},
    )
    TF._atomic_json(base / "regime.json", {"max_tokens": 1024, "seed": 42})
    roster_path = topup / "roster.json"
    roster = TF.build_cap_hit_roster(base_root=base, roster_path=roster_path, threshold=0.01)
    assert roster["ci"] == [1, 3]

    topup_rows = [
        {
            **base_rows[ci],
            "response": f"long{ci}",
            "finish_reason": "stop",
            "response_tokens": 1500,
            "seed": 1667586269,
        }
        for ci in (1, 3)
    ]
    TF._atomic_jsonl(topup / "answers.jsonl", topup_rows)
    TF._atomic_json(topup / "audit.json", {"n_rows": 2})
    TF._atomic_json(topup / "regime.json", {"max_tokens": 4096, "seed": 1667586269})
    audit = TF.merge_cap_hit_topup(
        base_root=base,
        topup_root=topup,
        roster_path=roster_path,
        destination=merged,
    )
    rows = TF._read_jsonl(merged / "answers.jsonl")
    assert [row["ci"] for row in rows] == list(range(100))
    assert rows[0]["response"] == "base0"
    assert rows[1]["response"] == "long1"
    assert rows[3]["regeneration"]["base_response_tokens"] == 1024
    assert audit["cap_hit"] == {
        "threshold": 0.02,
        "n_before": 2,
        "fraction_before": 0.02,
        "n_after": 0,
        "fraction_after": 0.0,
        "n_replaced": 2,
    }
