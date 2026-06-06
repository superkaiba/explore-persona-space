"""Test for the issue #506 Phase-0a item 2 ``existing_adapter_path`` port.

Confirms the dataclass field exists, the trainer can load a dummy LoRA
adapter from disk via the new field, and the resulting model's
``peft_config`` reflects the LOADED adapter (not a freshly-attached one).
Also exercises ``train_lora()`` itself on a tiny CPU model so the missing-
dir guard fires from inside the actual function (not a local re-encoding
of the guard logic — that was round-1's tautology).

No GPU and no real training — we use ``sshleifer/tiny-gpt2`` (a few MB)
and keep epochs / batch tiny so it finishes in seconds on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_existing_adapter_path_field_exists() -> None:
    """The Phase-0a item 2 port adds the field to ``TrainLoraConfig``."""
    from dataclasses import fields

    from explore_persona_space.train.sft import TrainLoraConfig

    field_names = {f.name for f in fields(TrainLoraConfig)}
    assert "existing_adapter_path" in field_names, (
        "TrainLoraConfig is missing existing_adapter_path (Phase-0a item 2). "
        "Plan §4.3.1 item 2 requires this for the LoRA Phase-2 continue-adapter path."
    )

    cfg = TrainLoraConfig(existing_adapter_path="/tmp/dummy")
    assert cfg.existing_adapter_path == "/tmp/dummy"
    assert TrainLoraConfig().existing_adapter_path is None  # default = None


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("peft"),
    reason="peft not installed in this environment",
)
def test_load_dummy_adapter_via_peft(tmp_path: Path) -> None:
    """Load a freshly-saved dummy adapter via PeftModel.from_pretrained.

    Builds a tiny CPU LM, wraps it with a 1-rank LoRA, saves the adapter,
    then loads it back with ``is_trainable=True`` — exactly the path
    ``train_lora()`` takes when ``cfg.existing_adapter_path`` is set.
    """
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Use a tiny known model for the round-trip test.
    model_id = "sshleifer/tiny-gpt2"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=1,
        lora_alpha=1,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        bias="none",
    )
    wrapped = get_peft_model(model, lora_cfg)
    adapter_dir = tmp_path / "dummy_adapter"
    wrapped.save_pretrained(str(adapter_dir))

    # Now load it back, the way ``train_lora`` does under the new field.
    base2 = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    loaded = PeftModel.from_pretrained(base2, str(adapter_dir), is_trainable=True)

    # Confirm peft_config reflects the loaded LoRA (r=1, alpha=1).
    pc = loaded.peft_config
    assert pc, "Loaded PeftModel has empty peft_config"
    name = next(iter(pc))
    assert pc[name].r == 1, f"Loaded adapter r != 1; got {pc[name].r}"
    assert pc[name].lora_alpha == 1, f"Loaded adapter alpha != 1; got {pc[name].lora_alpha}"

    # ``is_trainable=True`` must produce a model whose LoRA adapter has
    # trainable parameters — otherwise the continue-adapter contract is a
    # no-op (silent regression class flagged by Codex round 1).
    n_trainable = sum(p.numel() for p in loaded.parameters() if p.requires_grad)
    assert n_trainable > 0, (
        "Loaded LoRA adapter has 0 trainable parameters under is_trainable=True; "
        "the continue-adapter path would silently skip the Phase-2 SFT signal."
    )


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("peft"),
    reason="peft not installed in this environment",
)
def test_train_lora_raises_on_missing_existing_adapter_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling ``train_lora()`` itself with a missing ``existing_adapter_path``
    must raise FileNotFoundError from INSIDE the function — never silently fall
    back to fresh-LoRA (that would change comparator semantics).

    This exercises the real function code path, not a local re-encoding of
    the guard. We monkey-patch ``AutoModelForCausalLM.from_pretrained`` to a
    tiny CPU model so the test can run without GPU; the guard's position
    inside ``train_lora`` — right after the model load — is what we pin.
    """
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    # Pre-build a tiny CPU model so the monkey-patch can serve it cheaply.
    model_id = "sshleifer/tiny-gpt2"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tiny_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    # ``train_lora`` imports AutoModelForCausalLM from ``transformers`` inside
    # the function body — patch the source so the from_pretrained call inside
    # ``train_lora`` lands on the tiny CPU model regardless of args.
    def _fake_from_pretrained(*_args, **_kwargs):
        return tiny_model

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(_fake_from_pretrained),
    )
    # CVD-blank so torch's CUDA probe doesn't fire during model setup.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    # Write a 1-row dataset so the empty-jsonl preflight inside train_lora
    # doesn't pre-empt the existing-adapter guard.
    data_path = tmp_path / "tiny.jsonl"
    row = {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ],
        "completion": [{"role": "assistant", "content": "hello"}],
    }
    data_path.write_text(json.dumps(row) + "\n")

    missing_path = tmp_path / "does-not-exist-adapter"
    assert not missing_path.exists()

    cfg = TrainLoraConfig(
        existing_adapter_path=str(missing_path),
        epochs=1,
        lr=1e-4,
        batch_size=1,
        grad_accum=1,
        max_length=64,
        save_strategy="no",
        report_to="none",  # WANDB_INTENTIONALLY_DISABLED: CPU unit test, no telemetry
        gpu_id=0,
        seed=42,
        lora_r=1,
        lora_alpha=1,
        lora_targets=("c_attn",),
        gradient_checkpointing=False,
        packing=False,
    )

    with pytest.raises(FileNotFoundError, match="existing_adapter_path"):
        train_lora(
            base_model_path=model_id,
            data_path=str(data_path),
            output_dir=str(tmp_path / "out"),
            cfg=cfg,
        )
