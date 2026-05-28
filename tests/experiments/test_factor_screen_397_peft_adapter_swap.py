"""TDD Phase 1 — M3 regression: peft 0.18.1 adapter-swap lifecycle (task #397).

Plan v4 §4.3 + §5.5 + assumption A17: per-checkpoint log-prob eval keeps a
single resident base model and sequentially:

  1. ``base.load_adapter(adapter_dir, adapter_name="ckN")``
  2. ``base.set_adapter("ckN")``  → activates that adapter for the forward pass
  3. compute marker log-prob
  4. ``base.delete_adapter("ckN")``  before swapping in the next checkpoint

The v3 critic flagged that ``peft.PeftModel`` in 0.18.1 does NOT expose an
``unload()`` method, so the canonical multi-checkpoint pattern uses
``load_adapter`` + ``set_adapter`` + ``delete_adapter`` exclusively.

This test creates two tiny LoRA adapters on top of ``sshleifer/tiny-gpt2``
(CPU-friendly, no GPU), sequentially loads them onto the same base model,
swaps between them, and asserts:

  - both loads succeed without crash;
  - ``set_adapter`` toggles which adapter contributes to the forward pass;
  - ``delete_adapter`` removes an adapter cleanly;
  - the two adapters produce *different* logits on the same input (proving
    the swap actually re-routes the forward pass — not a no-op).

CPU-only (uses tiny-gpt2), runs in <30s.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("torch")


def _train_one_step_and_save(adapter_dir: Path, lora_seed: int) -> None:
    """Build a fresh tiny-gpt2 + LoRA adapter, perturb it minimally, save."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(lora_seed)
    base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, lora_cfg)

    # Perturb the LoRA weights so adapter A and adapter B produce different
    # outputs on the same input. Without perturbation both adapters start at
    # zero and contribute nothing (the swap-distinguishability assertion
    # below would trivially fail / trivially pass for the wrong reason).
    for name, param in peft_model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            with torch.no_grad():
                # Seeded perturbation so each adapter is reproducibly distinct.
                param.add_(torch.randn_like(param) * 0.1)

    peft_model.save_pretrained(str(adapter_dir))


def test_two_sequential_adapter_loads_and_swap_succeed_with_different_outputs() -> None:
    """M3 happy path: load A → set A → load B → set B → delete A → delete B; outputs differ."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        adapter_a_dir = tmp / "adapter_a"
        adapter_b_dir = tmp / "adapter_b"

        # Two adapters with deliberately distinct seeded perturbations.
        _train_one_step_and_save(adapter_a_dir, lora_seed=11)
        _train_one_step_and_save(adapter_b_dir, lora_seed=22)

        # Resident base model, load adapter A as the first.
        fresh_base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        peft_model = PeftModel.from_pretrained(fresh_base, str(adapter_a_dir), adapter_name="ckA")

        # Sequentially load adapter B onto the SAME resident base.
        peft_model.load_adapter(str(adapter_b_dir), adapter_name="ckB")

        # Tiny CPU forward pass; deterministic for fixed inputs.
        inputs = tok("hello world", return_tensors="pt")

        peft_model.set_adapter("ckA")
        with torch.no_grad():
            logits_a = peft_model(**inputs).logits.detach().clone()

        peft_model.set_adapter("ckB")
        with torch.no_grad():
            logits_b = peft_model(**inputs).logits.detach().clone()

        # Adapter swap MUST re-route the forward pass.
        assert not torch.allclose(logits_a, logits_b, atol=1e-6), (
            "Adapter A and Adapter B produce identical logits — the set_adapter "
            "swap is not actually re-routing the forward pass, or the adapters "
            "weren't perturbed."
        )

        # delete_adapter should not raise.
        peft_model.delete_adapter("ckA")
        peft_model.delete_adapter("ckB")


def test_set_adapter_to_unknown_name_raises_or_no_ops_cleanly() -> None:
    """Negative case: setting an adapter that was never loaded should NOT silently no-op.

    peft 0.18.1 raises ``ValueError`` for ``set_adapter`` on a name that was
    never loaded. The test pins this behavior so a Phase 2 wiring bug that
    typos an adapter name fails loud rather than silently falling back to the
    base model's forward pass.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        adapter_dir = tmp / "adapter_a"
        _train_one_step_and_save(adapter_dir, lora_seed=33)

        base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir), adapter_name="ckA")

        with pytest.raises((ValueError, KeyError)):
            peft_model.set_adapter("ckNEVERLOADED")
