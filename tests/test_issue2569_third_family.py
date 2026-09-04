"""CPU-only contracts for the issue #2569 third-family follow-up."""

from __future__ import annotations

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
