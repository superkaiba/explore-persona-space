"""Prompt/completion type-homogeneity preflight contract tests (task #1536).

The #1489 crash class: TRL 0.29.1 ``is_conversational()`` pops ONE arbitrary
key from a set, so a ``{"prompt": <message list>, "completion": "<str>"}`` row
routes hash-order-nondeterministically between the conversational and str-only
tokenize paths and crashes at ``SFTTrainer.__init__`` — possibly only on the
pod. #1508 landed the doc entry (.claude/rules/gotchas.md "TRL
prompt-completion SFT rows"); ``_validate_prompt_completion_homogeneity``
(``src/explore_persona_space/train/sft.py``) is the code-level backstop this
file pins:

- (a)/(b) within-row mixed types raise, both orientations;
- (c)/(d) homogeneous both-list and both-str datasets pass unchanged;
- (e) cross-row convention mixing (row0 str/str, row1 list/list) raises;
- (f) a row with only one of the two keys raises;
- (g) non-prompt/completion schemas ("messages", "text") are skipped;
- (h) degenerate values (empty list, message dict missing "content",
  dict-valued key) raise;
- (i) source-order + guard-POLARITY pin: the ``train_lora`` call site runs
  BEFORE ``load_dataset`` and is gated by the NON-inverted
  ``if not (cfg.dataset_kwargs or {}).get("skip_prepare_dataset")`` guard;
- (j) seam-level raise through the REAL ``train_lora`` (tiny-real rig per
  tests/test_issue906_tiny_real_e2e.py conventions): a mixed JSONL raises
  ``ValueError`` at the data preflight and ``SFTTrainer.__init__`` is never
  reached.

Plus the JSON-array early check, invalid-JSON row context, and blank-line
row-index accounting (plan #1536 §3).
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from explore_persona_space.train.sft import (
    _validate_prompt_completion_homogeneity,
    train_lora,
)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

USER_MSG = {"role": "user", "content": "What is 2+2?"}
ASSISTANT_MSG = {"role": "assistant", "content": "4."}


def _write_jsonl(tmp_path: Path, rows: list, name: str = "train.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


# ── (a)/(b) within-row mixed types ──────────────────────────────────────────


def test_mixed_within_row_raises(tmp_path):
    path = _write_jsonl(tmp_path, [{"prompt": [USER_MSG], "completion": "4."}])
    with pytest.raises(ValueError, match="mixed prompt/completion types") as exc:
        _validate_prompt_completion_homogeneity(path)
    msg = str(exc.value)
    assert "row 0" in msg
    assert "list" in msg and "str" in msg


def test_mixed_reversed_raises(tmp_path):
    path = _write_jsonl(tmp_path, [{"prompt": "What is 2+2?", "completion": [ASSISTANT_MSG]}])
    with pytest.raises(ValueError, match="mixed prompt/completion types"):
        _validate_prompt_completion_homogeneity(path)


# ── (c)/(d) homogeneous datasets pass (no behavior change) ──────────────────


def test_homogeneous_both_list_passes(tmp_path):
    rows = [{"prompt": [USER_MSG], "completion": [ASSISTANT_MSG]} for _ in range(3)]
    _validate_prompt_completion_homogeneity(_write_jsonl(tmp_path, rows))


def test_homogeneous_both_str_passes(tmp_path):
    rows = [{"prompt": f"Q{i}?", "completion": f"A{i}."} for i in range(3)]
    _validate_prompt_completion_homogeneity(_write_jsonl(tmp_path, rows))


# ── (e) cross-row convention mixing ──────────────────────────────────────────


def test_cross_row_convention_mismatch_raises(tmp_path):
    rows = [
        {"prompt": "Q0?", "completion": "A0."},
        {"prompt": [USER_MSG], "completion": [ASSISTANT_MSG]},
    ]
    with pytest.raises(ValueError, match="cross-row prompt/completion convention") as exc:
        _validate_prompt_completion_homogeneity(_write_jsonl(tmp_path, rows))
    msg = str(exc.value)
    assert "row 0" in msg and "row 1" in msg


# ── (f) missing partner key ──────────────────────────────────────────────────


def test_missing_partner_key_raises(tmp_path):
    path = _write_jsonl(tmp_path, [{"prompt": "Q0?"}])
    with pytest.raises(ValueError, match="without its partner key"):
        _validate_prompt_completion_homogeneity(path)


# ── (g) non-prompt/completion schemas skipped ────────────────────────────────


def test_non_pc_schemas_skipped(tmp_path):
    rows = [
        {"messages": [USER_MSG, ASSISTANT_MSG]},
        {"text": "plain text row"},
        {"input_ids": [1, 2, 3]},
    ]
    _validate_prompt_completion_homogeneity(_write_jsonl(tmp_path, rows))


# ── (h) degenerate values (parametrized) ─────────────────────────────────────


@pytest.mark.parametrize(
    ("row", "match"),
    [
        pytest.param(
            {"prompt": [], "completion": [ASSISTANT_MSG]},
            "NON-EMPTY list",
            id="empty-list-prompt",
        ),
        pytest.param(
            {"prompt": [{"role": "user"}], "completion": [ASSISTANT_MSG]},
            "NON-EMPTY list",
            id="message-dict-missing-content",
        ),
        pytest.param(
            {"prompt": {"role": "user", "content": "hi"}, "completion": "A."},
            "must be str or a list",
            id="dict-valued-prompt",
        ),
    ],
)
def test_degenerate_values_raise(tmp_path, row, match):
    with pytest.raises(ValueError, match=match):
        _validate_prompt_completion_homogeneity(_write_jsonl(tmp_path, [row]))


# ── JSON-array / invalid-JSON / blank-line accounting (plan §3) ─────────────


def test_json_array_file_raises(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps([{"prompt": "Q?", "completion": "A."}]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSONL"):
        _validate_prompt_completion_homogeneity(path)


def test_invalid_json_row_names_row_and_line(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"prompt": "Q0?", "completion": "A0."}) + "\n{not json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"invalid JSON at row 1 \(line 2\)"):
        _validate_prompt_completion_homogeneity(path)


def test_blank_lines_skipped_and_row_index_counts_nonblank(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n"
        + json.dumps({"prompt": "Q0?", "completion": "A0."})
        + "\n\n"
        + json.dumps({"prompt": [USER_MSG], "completion": "A1."})
        + "\n",
        encoding="utf-8",
    )
    # The mixed row is the 2nd non-blank line (row 1) on file line 4.
    with pytest.raises(ValueError, match=r"mixed prompt/completion types at row 1 \(line 4\)"):
        _validate_prompt_completion_homogeneity(path)


# ── (i) call-site source-order + guard-polarity pin ─────────────────────────


def test_call_site_precedes_load_dataset_with_non_inverted_guard():
    src = inspect.getsource(train_lora)
    call_idx = src.index("_validate_prompt_completion_homogeneity(")
    # Anchor on the ASSIGNMENT statement — the #365 preflight comment above the
    # call site also spells `load_dataset("json", ...)`.
    load_idx = src.index('dataset = load_dataset("json"')
    assert call_idx < load_idx, (
        "the homogeneity validator must run BEFORE load_dataset in train_lora"
    )
    # Guard-POLARITY pin: the call is gated by the NON-inverted guard. An
    # inverted guard (`if (cfg.dataset_kwargs or {}).get(...)`) would exempt
    # the validator on every production path (dataset_kwargs is None there)
    # and must fail this test. Comment lines may sit between guard and call.
    pattern = re.compile(
        r'if not \(cfg\.dataset_kwargs or \{\}\)\.get\("skip_prepare_dataset"\):\n'
        r"(?:[ \t]*#[^\n]*\n)*"
        r"[ \t]*_validate_prompt_completion_homogeneity\(_data_path\)"
    )
    assert pattern.search(src), (
        "train_lora's validator call site must be gated by the NON-inverted "
        '`if not (cfg.dataset_kwargs or {}).get("skip_prepare_dataset")` guard '
        "immediately preceding the call"
    )


# ── (j) seam-level raise through the REAL train_lora ─────────────────────────


@pytest.fixture(scope="module")
def qwen_tok():
    """The REAL Qwen tokenizer (same skip-on-offline contract as the #906 rig)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    except OSError as e:  # offline CI without a cached tokenizer
        pytest.skip(f"Qwen tokenizer unavailable (offline?): {e}")


@pytest.mark.slow
def test_train_lora_seam_raises_on_mixed_jsonl(tmp_path, monkeypatch, qwen_tok):
    """The REAL train_lora raises at the data preflight on a mixed JSONL.

    Tiny-real rig (tests/test_issue906_tiny_real_e2e.py conventions): a
    2-layer from-config same-arch Qwen2 stands in for the 7B weights, the
    real tokenizer is used, and only enough rig exists to reach the data
    preflight. Pins acceptance criterion (1) at the actual seam: a
    skip_prepare_dataset guard inversion or a dropped call site fails this
    test because SFTTrainer.__init__ would then be reached (and crash later
    or not at all) instead of the preflight ValueError firing.
    """
    import torch
    import transformers

    from explore_persona_space.train import sft as sft_mod

    tiny_kwargs = dict(
        vocab_size=151936,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    config = transformers.Qwen2Config(**tiny_kwargs)
    torch.manual_seed(1536)

    def fresh_tiny_model(*args, **kwargs):
        """HF WEIGHTS boundary: a fresh tiny Qwen2, ignoring dtype/device_map."""
        return transformers.Qwen2ForCausalLM(config)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fresh_tiny_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: qwen_tok)

    # Env hygiene (the #906 rig conventions): no live WandB, no persist gate;
    # train_lora writes CUDA_VISIBLE_DEVICES directly, so pre-register it for
    # pytest restoration.
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.delenv("EPM_PERSIST_ADAPTER_HF_REPO", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    # Spy on the SFTTrainer seam: the preflight must raise BEFORE any trainer
    # (or SFTConfig) construction is attempted.
    real_loader = sft_mod._load_trl_sft_classes
    trainer_inits: list[int] = []

    def spying_loader():
        sft_config_cls, sft_trainer_cls = real_loader()

        class SpyTrainer(sft_trainer_cls):  # type: ignore[misc,valid-type]
            def __init__(self, *a, **k):
                trainer_inits.append(1)
                super().__init__(*a, **k)

        return sft_config_cls, SpyTrainer

    monkeypatch.setattr(sft_mod, "_load_trl_sft_classes", spying_loader)

    mixed_path = _write_jsonl(
        tmp_path,
        [
            {"prompt": [USER_MSG], "completion": [ASSISTANT_MSG]},
            {"prompt": [USER_MSG], "completion": "4."},  # the #1489 mixed row
        ],
        name="mixed.jsonl",
    )

    with pytest.raises(ValueError, match="mixed prompt/completion types"):
        train_lora(
            BASE_MODEL,
            str(mixed_path),
            str(tmp_path / "out"),
            epochs=1,
            batch_size=1,
            grad_accum=1,
            bf16=False,
            report_to="none",  # WANDB_INTENTIONALLY_DISABLED: offline CPU contract test
        )

    assert not trainer_inits, "SFTTrainer.__init__ must never be reached on a mixed JSONL"
