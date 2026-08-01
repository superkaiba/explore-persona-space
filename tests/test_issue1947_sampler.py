"""#1947 single-visit sequential-consumption seam — tiny-real CPU trainer tests.

Exercises the REAL ``train_lora`` -> SFTTrainer path (real Qwen tokenizer, real
TRL tokenization, real optimizer steps on a from-config 2-layer Qwen2 over the
real vocab-id space — the tests/test_issue906_tiny_real_e2e.py tiny-real
standard) with grad_accum > 1, asserting:

1. the SequentialConsumptionCallback's realized (step, row) log equals the
   builder-predicted mapping INCLUDING accumulation-boundary attribution
   (3 optimizer steps x effective batch 4 = batch 2 x accum 2 over 12 rows);
2. the fail-loud divergence assert actually FIRES on a deliberately
   reshuffled predicted order (negative test, through the real trainer);
3. the manifest-contract validation of ``_maybe_attach_sequential_consumption``
   fails loud on every contract violation (direct production-body calls);
4. the seam is inert by default (TrainLoraConfig defaults).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from explore_persona_space.train.sft import (  # noqa: E402
    TrainLoraConfig,
    _maybe_attach_sequential_consumption,
    train_lora,
)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 2-layer random-weights Qwen2 covering the REAL Qwen-2.5 token-id space
# (the test_issue906_tiny_real_e2e.py fixture shape).
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

N_ROWS = 12
BATCH = 2
ACCUM = 2
EFF = BATCH * ACCUM  # 4 -> 3 optimizer steps


@pytest.fixture(scope="module")
def qwen_tok():
    """The REAL Qwen tokenizer (skip-on-offline, the r13/issue906 contract)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    except OSError as e:  # offline CI without a cached tokenizer
        pytest.skip(f"Qwen tokenizer unavailable (offline?): {e}")


@pytest.fixture(scope="module")
def tiny_qwen_state():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(**TINY_QWEN_KWARGS)
    torch.manual_seed(1947)
    model = Qwen2ForCausalLM(config)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    return config, state


def _write_mix(tmp_path: Path) -> Path:
    """12 benign rows in the factory train-row schema (message lists BOTH keys)."""
    mix = tmp_path / "train_mix.jsonl"
    rows = [
        {
            "prompt": [{"role": "user", "content": f"Question {i}: what is {i} plus {i}?"}],
            "completion": [{"role": "assistant", "content": f"The answer is {2 * i}."}],
        }
        for i in range(N_ROWS)
    ]
    mix.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return mix


def _write_manifest(tmp_path: Path, predicted_step_of_idx: list[int]) -> Path:
    path = tmp_path / "consumption_manifest.json"
    path.write_text(
        json.dumps(
            {
                "n_rows": N_ROWS,
                "effective_batch": EFF,
                "epochs": 1,
                "predicted_step_of_idx": predicted_step_of_idx,
                "row_ids": [f"row:{i:04d}" for i in range(N_ROWS)],
            }
        )
    )
    return path


def _seam_cfg(manifest_path: Path) -> TrainLoraConfig:
    return TrainLoraConfig(
        epochs=1,
        lr=1e-4,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        batch_size=BATCH,
        grad_accum=ACCUM,
        max_length=256,
        seed=42,
        run_name="issue1947-sampler-test",
        report_to="none",  # WANDB_INTENTIONALLY_DISABLED: offline CPU unit test
        save_strategy="no",
        gradient_checkpointing=False,
        logging_steps=1,
        bf16=False,  # CPU-only machines reject bf16 TrainingArguments
        hf_upload=False,
        sequential_sampler=True,
        sequential_consumption_manifest=str(manifest_path),
    )


def _patch_boundaries(monkeypatch, qwen_tok, tiny_qwen_state):
    """HF weights boundary -> fresh tiny Qwen2 per from_pretrained (PEFT/TRL
    wrap in place); tokenizer boundary -> the real cached Qwen tokenizer."""
    import transformers
    from transformers import Qwen2ForCausalLM

    config, state = tiny_qwen_state

    def fresh_tiny_model(*args, **kwargs):
        model = Qwen2ForCausalLM(config)
        model.load_state_dict(state)
        return model

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fresh_tiny_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: qwen_tok)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("EPM_PERSIST_ADAPTER_HF_REPO", raising=False)


@pytest.mark.slow
def test_sequential_seam_realized_matches_predicted(
    tmp_path, monkeypatch, qwen_tok, tiny_qwen_state
):
    """Positive: 3 optimizer steps x eff 4; realized == predicted incl.
    accumulation-boundary attribution; realized manifest written."""
    _patch_boundaries(monkeypatch, qwen_tok, tiny_qwen_state)
    mix = _write_mix(tmp_path)
    manifest = _write_manifest(tmp_path, [i // EFF for i in range(N_ROWS)])
    out_dir = tmp_path / "train"
    train_lora(BASE_MODEL, str(mix), str(out_dir), cfg=_seam_cfg(manifest))
    realized_path = out_dir / "realized_consumption.json"
    assert realized_path.exists(), "callback never wrote the realized manifest"
    rec = json.loads(realized_path.read_text())
    assert rec["matches_predicted"] is True
    assert rec["first_mismatch_idx"] is None
    assert rec["n_yielded"] == N_ROWS
    assert rec["global_step"] == N_ROWS // EFF == 3
    assert rec["realized_order"] == list(range(N_ROWS))
    assert rec["realized_step_of_idx"] == [i // EFF for i in range(N_ROWS)]
    # Accumulation-boundary attribution: each on_step_end snapshot saw exactly
    # the rows of steps 1..k (bounded one-group lookahead permitted; with
    # num_workers=0 + transformers 4.57 get_batch_samples it is exact).
    snaps = rec["step_snapshots"]
    assert [s["global_step"] for s in snaps] == [1, 2, 3]
    for s in snaps:
        assert s["global_step"] * EFF <= s["rows_yielded"] <= (s["global_step"] + 1) * EFF
    assert snaps[-1]["rows_yielded"] == N_ROWS


@pytest.mark.slow
def test_sequential_seam_divergence_fires_on_reshuffled_prediction(
    tmp_path, monkeypatch, qwen_tok, tiny_qwen_state
):
    """Negative: a builder manifest predicting a RESHUFFLED (reversed) order
    must fail loud at train end — through the REAL trainer path."""
    _patch_boundaries(monkeypatch, qwen_tok, tiny_qwen_state)
    mix = _write_mix(tmp_path)
    # Reversed consumption prediction: idx i consumed at step (N-1-i)//EFF.
    manifest = _write_manifest(tmp_path, [(N_ROWS - 1 - i) // EFF for i in range(N_ROWS)])
    out_dir = tmp_path / "train"
    with pytest.raises(RuntimeError, match="DIVERGES from the builder-predicted"):
        train_lora(BASE_MODEL, str(mix), str(out_dir), cfg=_seam_cfg(manifest))
    # Forensic evidence written BEFORE the raise.
    rec = json.loads((out_dir / "realized_consumption.json").read_text())
    assert rec["matches_predicted"] is False
    assert rec["first_mismatch_idx"] == 0


def test_attach_contract_validation_fails_loud(tmp_path):
    """Direct production-body coverage of the manifest-contract validation."""
    trainer = SimpleNamespace(train_dataset=list(range(N_ROWS)))

    def cfg(**kw):
        base = dict(
            epochs=1,
            batch_size=BATCH,
            grad_accum=ACCUM,
            sequential_sampler=True,
        )
        base.update(kw)
        return TrainLoraConfig(**base)

    # Missing manifest path.
    with pytest.raises(ValueError, match="requires sequential_consumption_manifest"):
        _maybe_attach_sequential_consumption(trainer, cfg(), str(tmp_path))
    # Nonexistent file.
    with pytest.raises(FileNotFoundError):
        _maybe_attach_sequential_consumption(
            trainer,
            cfg(sequential_consumption_manifest=str(tmp_path / "nope.json")),
            str(tmp_path),
        )
    # effective_batch mismatch.
    m = _write_manifest(tmp_path, [i // EFF for i in range(N_ROWS)])
    with pytest.raises(ValueError, match="effective_batch"):
        _maybe_attach_sequential_consumption(
            trainer,
            cfg(sequential_consumption_manifest=str(m), grad_accum=4),
            str(tmp_path),
        )
    # epochs != 1 in the manifest.
    bad = tmp_path / "bad_epochs.json"
    bad.write_text(
        json.dumps(
            {
                "n_rows": N_ROWS,
                "effective_batch": EFF,
                "epochs": 15,
                "predicted_step_of_idx": [i // EFF for i in range(N_ROWS)],
                "row_ids": [f"row:{i:04d}" for i in range(N_ROWS)],
            }
        )
    )
    with pytest.raises(ValueError, match="exactly one pass"):
        _maybe_attach_sequential_consumption(
            trainer, cfg(sequential_consumption_manifest=str(bad)), str(tmp_path)
        )
    # Dataset length mismatch.
    short = SimpleNamespace(train_dataset=list(range(N_ROWS - 1)))
    with pytest.raises(ValueError, match="prepared train dataset"):
        _maybe_attach_sequential_consumption(
            short, cfg(sequential_consumption_manifest=str(m)), str(tmp_path)
        )
    # packing incompatibility.
    with pytest.raises(ValueError, match="packing"):
        _maybe_attach_sequential_consumption(
            trainer,
            cfg(sequential_consumption_manifest=str(m), packing=True),
            str(tmp_path),
        )


def test_seam_defaults_off():
    """Inertness pin: the seam is opt-in; defaults keep every caller byte-identical."""
    cfg = TrainLoraConfig()
    assert cfg.sequential_sampler is False
    assert cfg.sequential_consumption_manifest is None
    assert cfg.realized_consumption_out is None
    # No-op attach on a default config must not touch the trainer.
    sentinel = SimpleNamespace(train_dataset=None)
    _maybe_attach_sequential_consumption(sentinel, cfg, "/tmp/unused")
    assert not hasattr(sentinel, "_epm_consumption_callback")
