"""Issue #2225 steering-hook unit tests (plan §4.4 + §7 pre-launch gate).

CPU-only, tiny 2-layer Qwen2 over the REAL Qwen token-id space + the real
tokenizer (skip-on-offline — the repo's established tiny-real pattern,
tests/test_issue906_tiny_real_e2e.py). The hooks are registered through the
PRODUCTION path — ``SteeringHook.install`` on the PEFT-WRAPPED
``SteeredSFTTrainer.model`` (module-path resolution differs between
``PeftModel`` and a bare ``Qwen2ForCausalLM``; the wrapped resolution is the
one exercised here, asserted at depth 2).

Covers:
- delta == alpha * v EXACTLY on masked positions and 0 elsewhere, per mode
  (all / context / response / prefix), on a 2-row collated batch;
- masks partition the non-pad tokens on 4 chat-template shapes (with/without
  an explicit system turn), prefix ⊆ context;
- real-trainer lifecycle: ``train()`` emits the ``[steer-hook]`` engagement
  line, edits forwards, and leaves NO hooks registered after it returns (the
  eval-time leak guard) + no armed mask;
- an unarmed forward under an installed hook fails loud;
- ``build_incremental_vectors`` telescoping + contiguity.
"""

from __future__ import annotations

import pytest
import torch

from explore_persona_space.analysis.extraction import _resolve_decoder_blocks
from explore_persona_space.experiments.issue2225.steer_train import (
    MASK_MODES,
    SteeredSFTTrainer,
    SteeringDataCollator,
    SteeringHook,
    build_incremental_vectors,
    compute_prefix_len,
    masks_for_mode,
)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 2-layer random-weights Qwen2 covering the REAL Qwen-2.5 token-id space
# (vocab_size=151936 covers the tokenizer's max id ~151665). Only the WEIGHTS
# are fake; every id is real (test_issue906_tiny_real_e2e pattern).
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
HIDDEN = TINY_QWEN_KWARGS["hidden_size"]
ALPHA = 1.5

# 4 chat-template shapes: with/without an explicit system turn x 2 lengths.
ROWS = [
    {
        "prompt": [{"role": "user", "content": "What color is the sky?"}],
        "completion": [{"role": "assistant", "content": "Blue, mostly."}],
    },
    {
        "prompt": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Name a prime number."},
        ],
        "completion": [{"role": "assistant", "content": "Seven."}],
    },
    {
        "prompt": [
            {
                "role": "user",
                "content": "Explain in one sentence why the sky appears blue during the day.",
            }
        ],
        "completion": [
            {"role": "assistant", "content": "Rayleigh scattering favors shorter wavelengths."}
        ],
    },
    {
        "prompt": [
            {"role": "system", "content": "You answer with exactly two sentences."},
            {"role": "user", "content": "What is water made of?"},
        ],
        "completion": [
            {
                "role": "assistant",
                "content": "Water is H2O. Each molecule has two hydrogens and one oxygen.",
            }
        ],
    },
]


@pytest.fixture(scope="module")
def qwen_tok():
    """The REAL Qwen tokenizer (skip-on-offline contract)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    except OSError as e:  # offline CI without a cached tokenizer
        pytest.skip(f"Qwen tokenizer unavailable (offline?): {e}")


def _vectors(n_layers: int = 1) -> dict[int, torch.Tensor]:
    torch.manual_seed(2225)
    return {layer: torch.randn(HIDDEN, dtype=torch.float32) for layer in range(n_layers)}


@pytest.fixture(scope="module")
def trainer_and_hook(qwen_tok, tmp_path_factory):
    """One SteeredSFTTrainer over the 4-row dataset (PEFT-wrapped tiny Qwen2).

    The hook is mode="all" for the lifecycle test; the per-mode delta tests
    construct their own hooks and install them via the SAME production
    ``SteeringHook.install(trainer.model)`` path ``train()`` uses.
    """
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from trl import SFTConfig

    torch.manual_seed(906)
    model = Qwen2ForCausalLM(Qwen2Config(**TINY_QWEN_KWARGS))

    rows = [{**row, "prefix_len": compute_prefix_len(qwen_tok, row["prompt"])} for row in ROWS]
    ds = Dataset.from_list(rows)

    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    cfg = SFTConfig(
        output_dir=str(tmp_path_factory.mktemp("i2225_trainer")),
        per_device_train_batch_size=len(rows),
        gradient_accumulation_steps=1,
        max_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        completion_only_loss=True,
        max_length=256,
        use_cpu=True,
        report_to=[],  # WANDB_INTENTIONALLY_DISABLED: CPU unit test, no wandb run
        seed=0,
    )
    hook = SteeringHook(_vectors(), alpha=ALPHA, mode="all")
    pad_id = qwen_tok.pad_token_id if qwen_tok.pad_token_id is not None else qwen_tok.eos_token_id
    trainer = SteeredSFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=qwen_tok,
        peft_config=peft_config,
        data_collator=SteeringDataCollator(pad_token_id=pad_id, completion_only_loss=True),
        steering_hook=hook,
    )
    return trainer, hook


def _collated_batch(trainer) -> dict[str, torch.Tensor]:
    """One right-padded batch of the full tokenized dataset via the trainer's
    own collator (the production batch shape compute_loss receives)."""
    examples = [trainer.train_dataset[i] for i in range(len(trainer.train_dataset))]
    return trainer.data_collator(examples)


def test_peft_wrapped_resolution(trainer_and_hook):
    """The trainer model IS a PeftModel and resolves decoder blocks at depth 2."""
    from peft import PeftModel

    trainer, _ = trainer_and_hook
    assert isinstance(trainer.model, PeftModel)
    blocks, _, depth = _resolve_decoder_blocks(trainer.model)
    assert blocks is not None and depth == 2, depth
    assert len(blocks) == TINY_QWEN_KWARGS["num_hidden_layers"]


def test_masks_partition_on_chat_shapes(trainer_and_hook):
    """context | response partition attention_mask==1; prefix ⊆ context; 4 shapes."""
    trainer, _ = trainer_and_hook
    batch = _collated_batch(trainer)
    assert batch["input_ids"].shape[0] == len(ROWS)
    am, labels, prefix_len = batch["attention_mask"], batch["labels"], batch["prefix_len"]
    real = am == 1
    ctx = masks_for_mode("context", attention_mask=am, labels=labels)
    resp = masks_for_mode("response", attention_mask=am, labels=labels)
    alls = masks_for_mode("all", attention_mask=am, labels=labels)
    pref = masks_for_mode("prefix", attention_mask=am, labels=labels, prefix_len=prefix_len)
    assert torch.equal(alls, real)
    assert torch.equal(ctx | resp, real), "context+response must cover all non-pad tokens"
    assert not (ctx & resp).any(), "context and response must be disjoint"
    assert torch.equal(pref & ctx, pref), "prefix must be a subset of context"
    # Every row has a non-trivial prefix (Qwen inserts a default system block
    # when the row has no explicit system turn) and a non-empty response.
    assert (prefix_len >= 2).all(), prefix_len
    assert resp.any(dim=1).all()
    assert (pref.sum(dim=1) == prefix_len).all()


@pytest.mark.parametrize("mode", MASK_MODES)
def test_hook_delta_exact_per_mode(trainer_and_hook, mode):
    """delta == alpha·v exactly at masked positions, 0 elsewhere (2-row batch)."""
    trainer, _ = trainer_and_hook
    model = trainer.model
    batch = _collated_batch(trainer)
    # 2-row toy batch per the plan (one with, one without an explicit system turn).
    sel = slice(0, 2)
    input_ids = batch["input_ids"][sel]
    am = batch["attention_mask"][sel]
    labels = batch["labels"][sel]
    prefix_len = batch["prefix_len"][sel]
    mask = masks_for_mode(mode, attention_mask=am, labels=labels, prefix_len=prefix_len)
    assert mask.any() and not mask.all(), f"degenerate {mode} mask cannot test both sides"

    vectors = _vectors()
    (layer,) = vectors.keys()
    blocks, _, _ = _resolve_decoder_blocks(model)
    captured: list[torch.Tensor] = []

    def observer(_m, _i, output):
        captured.append((output[0] if isinstance(output, tuple) else output).detach().clone())

    # Baseline forward: observer only.
    handle = blocks[layer].register_forward_hook(observer)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=am)
    handle.remove()
    base = captured.pop()

    # Steered forward: production registration path (install BEFORE the
    # observer so the observer sees the post-steering output).
    hook = SteeringHook(vectors, alpha=ALPHA, mode=mode)
    hook.install(model)
    handle = blocks[layer].register_forward_hook(observer)
    hook.current_batch_masks = mask
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=am)
    finally:
        handle.remove()
        hook.remove()
        hook.current_batch_masks = None
    steered = captured.pop()

    v = vectors[layer].to(base.dtype)
    expected = base + mask.unsqueeze(-1).to(base.dtype) * (ALPHA * v)
    assert torch.equal(steered, expected), f"mode={mode}: hooked output != base + mask·alpha·v"
    # Unmasked positions bit-identical to baseline; masked positions moved.
    assert torch.equal(steered[~mask], base[~mask]), f"mode={mode}: unmasked positions changed"
    assert not torch.equal(steered[mask], base[mask]), f"mode={mode}: masked positions unchanged"
    assert hook.n_edits == 1


def test_context_end_one_hot_matches_template_length(trainer_and_hook, qwen_tok):
    """fu1 plan §4.2 + §12 A5: `context_end` is EXACTLY one-hot per row at the
    last prompt token, cross-checked against a DIRECT
    ``apply_chat_template(add_generation_prompt=True)`` token count on all 4
    template shapes (with/without an explicit system turn x 2 lengths)."""
    trainer, _ = trainer_and_hook
    batch = _collated_batch(trainer)
    am, labels = batch["attention_mask"], batch["labels"]
    mask = masks_for_mode("context_end", attention_mask=am, labels=labels)
    ctx = masks_for_mode("context", attention_mask=am, labels=labels)
    # Exactly one position per row; context_end ⊆ context (hence ⊆ non-pad).
    assert (mask.sum(dim=1) == 1).all(), mask.sum(dim=1).tolist()
    assert torch.equal(mask & ctx, mask), "context_end must be a subset of context"
    # Independent cross-check: the masked index == n_prompt_tokens - 1 under
    # the SAME apply_chat_template(add_generation_prompt=True) render TRL's
    # prompt-completion tokenizer uses (right padding puts the prompt first).
    for i, row in enumerate(ROWS):
        ids = qwen_tok.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=True)
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        masked_idx = int(mask[i].nonzero().item())
        assert masked_idx == len(ids) - 1, (
            f"row {i}: context_end index {masked_idx} != template prompt length "
            f"{len(ids)} - 1 (assistant-header tail)"
        )


def test_context_end_refuses_contextless_row(trainer_and_hook):
    """A row with NO context position (every real token supervised) fails loud
    instead of silently steering position 0 (argmax-of-all-False)."""
    trainer, _ = trainer_and_hook
    batch = _collated_batch(trainer)
    labels = torch.where(
        batch["attention_mask"] == 1,
        batch["input_ids"],
        torch.full_like(batch["input_ids"], -100),
    )
    with pytest.raises(AssertionError, match="NO context position"):
        masks_for_mode("context_end", attention_mask=batch["attention_mask"], labels=labels)


def test_unarmed_forward_raises(trainer_and_hook):
    """A forward under an installed hook with no armed mask fails loud."""
    trainer, _ = trainer_and_hook
    batch = _collated_batch(trainer)
    hook = SteeringHook(_vectors(), alpha=ALPHA, mode="all")
    hook.install(trainer.model)
    try:
        with pytest.raises(RuntimeError, match=r"no armed batch mask"), torch.no_grad():
            trainer.model(
                input_ids=batch["input_ids"][:1], attention_mask=batch["attention_mask"][:1]
            )
    finally:
        hook.remove()


def test_train_lifecycle_and_eval_leak_guard(trainer_and_hook, capsys):
    """Real SFTTrainer lifecycle: engagement line, edits, and NO hooks after."""
    trainer, hook = trainer_and_hook
    trainer.train()
    out = capsys.readouterr().out
    assert f"[steer-hook] mode=all layers=1 alpha={ALPHA:g}" in out
    assert hook.n_edits >= 1, "steering hook never fired during training"
    # Eval-time leak guard: nothing registered after the training context exits.
    assert not hook.installed
    blocks, _, _ = _resolve_decoder_blocks(trainer.model)
    for block in blocks:
        assert not block._forward_hooks, "a forward hook survived train()"
    assert hook.current_batch_masks is None


def test_build_incremental_vectors():
    torch.manual_seed(0)
    band = {15, 16, 17, 18}
    vectors = {layer: torch.randn(8) for layer in band}
    inc = build_incremental_vectors(vectors)
    assert set(inc) == band
    # Band start: v_inc_s = v_s; telescoping: cumulative sum equals v_l in-band.
    assert torch.equal(inc[15], vectors[15])
    running = torch.zeros(8)
    for layer in sorted(band):
        running = running + inc[layer]
        assert torch.allclose(running, vectors[layer], atol=1e-6), layer
    with pytest.raises(ValueError, match="contiguous"):
        build_incremental_vectors({1: torch.randn(8), 3: torch.randn(8)})
    with pytest.raises(ValueError, match="empty"):
        build_incremental_vectors({})


def test_signature_columns_include_prefix_len(trainer_and_hook):
    """g1 Concern 1 pin: TRL's remove_unused_columns strips non-signature
    dataset columns, so the override MUST append prefix_len to
    ``_signature_columns`` — otherwise mode='prefix' cells silently lose
    their mask input at the collator."""
    trainer, _ = trainer_and_hook
    trainer._set_signature_columns_if_needed()
    assert trainer._signature_columns is not None
    assert "prefix_len" in trainer._signature_columns


def test_zero_coverage_mask_fails_loud(trainer_and_hook):
    """g1 Concern 2 (r2): an all-empty steering mask raises BEFORE arming the
    hook or running any forward — the silent-null-steering channel (the
    [steer-hook] install breadcrumb printed, yet zero positions steered)."""
    trainer, _ = trainer_and_hook
    batch = _collated_batch(trainer)
    # Supervise EVERY real token: the "context" mask (real & labels==-100)
    # is then empty across the whole batch; pads keep -100 so the shape
    # asserts in masks_for_mode still pass.
    labels = torch.where(
        batch["attention_mask"] == 1,
        batch["input_ids"],
        torch.full_like(batch["input_ids"], -100),
    )
    inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": labels,
        "prefix_len": batch["prefix_len"],
    }
    ctx_hook = SteeringHook(_vectors(), alpha=ALPHA, mode="context")
    orig = trainer._steering_hook
    trainer._steering_hook = ctx_hook
    try:
        with pytest.raises(RuntimeError, match="steering mask empty for mode='context'"):
            trainer.compute_loss(trainer.model, inputs)
        assert ctx_hook.current_batch_masks is None, "guard must fire before arming the hook"
    finally:
        trainer._steering_hook = orig
