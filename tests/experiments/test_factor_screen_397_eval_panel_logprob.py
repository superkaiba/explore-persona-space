"""Plan v4 §14 item 4 — compute_logprob_panel integration sanity (task #397).

Code-review v1 ISSUE 4: peft-adapter-swap test exercises ONLY the lifecycle
primitive; this test covers the next layer up — ``compute_logprob_panel``'s
actual integration with ``compute_marker_logprob`` over a resident base
model + 2 toy contexts + 1 toy adapter + 1 marker variant.

Shape checks the integration:
  - returned dict is keyed by checkpoint dir
  - per-marker entries are lists of length len(contexts)
  - every log-prob is finite (no NaN / inf — protects against tokenizer
    or adapter-swap miswiring producing garbage values)
  - the (personas, questions) construction path yields the same context
    count as len(personas) * len(questions)
  - ``system_prompt_overrides`` actually changes the contexts the panel
    sees (BLOCKER 3 integration smoke)

CPU-only (sshleifer/tiny-gpt2), runs in <60s.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("torch")

from explore_persona_space.experiments.factor_screen_397.eval_panel import (
    compute_logprob_panel,
)


def _save_toy_adapter(adapter_dir: Path, lora_seed: int) -> None:
    """Save a randomly-perturbed LoRA adapter on tiny-gpt2 (CPU)."""
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
    # Perturb so the adapter isn't a no-op (matches the pattern in
    # test_factor_screen_397_peft_adapter_swap.py).
    for name, param in peft_model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            with torch.no_grad():
                param.add_(torch.randn_like(param) * 0.05)
    peft_model.save_pretrained(str(adapter_dir))


def test_compute_logprob_panel_returns_finite_logprobs_for_two_contexts() -> None:
    """End-to-end shape + finiteness: 1 adapter x 2 contexts x 1 marker variant."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_dir = Path(tmpdir) / "adapter_a"
        _save_toy_adapter(adapter_dir, lora_seed=7)

        base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir), adapter_name="seed")

        contexts = [
            "Hello, how are you today?",
            "Tell me a short fact about libraries.",
        ]

        # Use a marker that tokenizes safely on the GPT-2 BPE — "x" is single-piece.
        result = compute_logprob_panel(
            base_model=peft_model,
            tokenizer=tok,
            checkpoint_dirs=[str(adapter_dir)],
            contexts=contexts,
            marker_texts=("x",),
            batch_size=2,
            device="cpu",
        )

        assert str(adapter_dir) in result
        per_marker = result[str(adapter_dir)]
        assert "x" in per_marker
        logps = per_marker["x"]
        assert len(logps) == len(contexts), (
            f"Expected {len(contexts)} log-probs (one per context); got {len(logps)}"
        )
        for i, lp in enumerate(logps):
            assert math.isfinite(lp), (
                f"compute_logprob_panel returned non-finite log-prob at i={i}: {lp!r}"
            )


def test_compute_logprob_panel_builds_contexts_from_personas_and_questions() -> None:
    """BLOCKER 3 integration: (personas, questions) path produces
    len(personas) * len(questions) contexts AND attaches a _context_keys
    entry for downstream aggregation.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # GPT-2 tokenizer has no chat template; install a minimal one.
    tok.chat_template = (
        "{% for m in messages %}"
        "{{ m['role'] }}: {{ m['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_dir = Path(tmpdir) / "adapter_a"
        _save_toy_adapter(adapter_dir, lora_seed=11)

        base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir), adapter_name="seed")

        personas = {
            "librarian": "You are a librarian.",
            "barista": "You are a barista.",
        }
        questions = ["Tell me a fact.", "What do you do?"]
        # 2 personas x 2 questions = 4 contexts.

        result = compute_logprob_panel(
            base_model=peft_model,
            tokenizer=tok,
            checkpoint_dirs=[str(adapter_dir)],
            personas=personas,
            questions=questions,
            marker_texts=("x",),
            batch_size=2,
            device="cpu",
        )

        assert str(adapter_dir) in result
        assert len(result[str(adapter_dir)]["x"]) == 4
        # _context_keys present, one (persona, question) pair per context.
        assert "_context_keys" in result
        keys = result["_context_keys"]
        assert len(keys) == 4
        assert keys[0] == ["librarian", "Tell me a fact."]
        assert keys[-1] == ["barista", "What do you do?"]


def test_compute_logprob_panel_system_prompt_overrides_changes_panel() -> None:
    """BLOCKER 3 smoke: ``system_prompt_overrides`` actually swaps the
    system prompt for the named persona (train-matched eval integration
    surface). Verified by comparing log-probs with vs without the override
    — they must differ once the prompt content changes.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.chat_template = (
        "{% for m in messages %}"
        "{{ m['role'] }}: {{ m['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_dir = Path(tmpdir) / "adapter_a"
        _save_toy_adapter(adapter_dir, lora_seed=13)

        base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir), adapter_name="seed")

        personas = {"librarian": "You are a librarian."}
        questions = ["Tell me a fact."]

        # Baseline — no override.
        baseline = compute_logprob_panel(
            base_model=peft_model,
            tokenizer=tok,
            checkpoint_dirs=[str(adapter_dir)],
            personas=personas,
            questions=questions,
            marker_texts=("x",),
            batch_size=1,
            device="cpu",
            adapter_name_prefix="baseline",
        )

        # Override the librarian persona's prompt — a substantially different
        # string should produce a different log-prob.
        overridden = compute_logprob_panel(
            base_model=peft_model,
            tokenizer=tok,
            checkpoint_dirs=[str(adapter_dir)],
            personas=personas,
            questions=questions,
            system_prompt_overrides={
                "librarian": "Background context: long-form non-persona prompt for C=1.",
            },
            marker_texts=("x",),
            batch_size=1,
            device="cpu",
            adapter_name_prefix="overridden",
        )

        baseline_lp = baseline[str(adapter_dir)]["x"][0]
        overridden_lp = overridden[str(adapter_dir)]["x"][0]

        assert math.isfinite(baseline_lp)
        assert math.isfinite(overridden_lp)
        # The override changed the system prompt content; the marker log-prob
        # over a different prefix MUST differ. Pure equality would mean the
        # override silently didn't propagate to context construction.
        assert baseline_lp != overridden_lp, (
            "compute_logprob_panel returned identical log-probs for two "
            "substantially different system prompts; system_prompt_overrides "
            "is not actually changing the contexts."
        )


def test_compute_logprob_panel_rejects_mixed_input_paths() -> None:
    """Loud-fail when caller passes BOTH contexts and (personas, questions)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_dir = Path(tmpdir) / "adapter_a"
        _save_toy_adapter(adapter_dir, lora_seed=17)

        base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir), adapter_name="seed")

        with pytest.raises(ValueError, match="contexts OR"):
            compute_logprob_panel(
                base_model=peft_model,
                tokenizer=tok,
                checkpoint_dirs=[str(adapter_dir)],
                contexts=["one context"],
                personas={"librarian": "You are a librarian."},
                questions=["Q?"],
                marker_texts=("x",),
                batch_size=1,
                device="cpu",
            )
