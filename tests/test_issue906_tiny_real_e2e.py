"""CPU tiny-real end-to-end contract test for the #906 post-datagen marker path (r15 pivot).

Executes the ENTIRE production marker pipeline — ``run_pilot`` -> ``run_class``
-> ``_build_marker_class`` -> ``train_lora`` -> ``_verify_marker_class`` ->
report assembly -> ``_upload_class`` — with ``PilotSeams()`` (every seam None,
the ``--full`` production path) and REAL types at every library seam:

- REAL Qwen-2.5 tokenizer (marker id 83399 / im_end id 151645 asserts run for
  real; the trainer render, the token-budget gate, and the collator all see the
  real BPE);
- REAL ``train_lora``: real ``TrainLoraConfig`` from the real recipe builder
  (the r12 crash site), real ``MarkerOnlyDataCollator`` over really-tokenized
  rows under the real budget gate (the r13 crash site), real ``SFTTrainer``
  lifecycle (``__init__`` -> ``on_init_end`` -> steps -> ``on_train_end``; the
  #816 callback class), real ``MarkerBandStopCallback`` attach + the
  ``band_stop_result.json`` post-train record, real ``save_model`` writing a
  real PEFT adapter to disk;
- REAL ``_verify_marker_class`` inline body: real ``adapter_config.json``
  parse + real ``assert_gauge_free_adapter_config`` on the PARSED dict (the
  exact r14 crash site), real ``PeftModel.from_pretrained`` round-trip of the
  just-trained adapter, real greedy rollouts + three-space slot reads +
  ``validate_marker_slot_record``;
- REAL ``_upload_class`` body incl. the fail-loud verify-leg expected set over
  the really-written rollout files.

Faked: ONLY genuinely-GPU-scale compute and the remote boundary —
(1) the 7B HF WEIGHTS -> a 2-layer random-weights Qwen2 with the REAL vocab-id
space (fresh instance per ``from_pretrained``: PEFT/TRL wrap models in place);
(2) the Hub boundary -> signature-bound fakes (the r7-r10 pattern);
(3) compute-SCALE knobs on the otherwise-real marker ``TrainLoraConfig``
(max_steps=2, batch 1) — the config builder itself executes for real.

Why (epm:failure v5 / epm:strategy-pivot v1): four production crashes (r11-r14)
each surfaced exactly one pipeline stage deeper because the smoke stubs
bypassed the real library calls — one seam bug per ~1.5h GPU cycle. This test
executes every remaining seam with real types in one CPU pass (< ~5 min).
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import issue906_phase1_pilot as pilot  # noqa: E402

from explore_persona_space.artifacts.negatives import NEGATIVE_PANELS  # noqa: E402
from explore_persona_space.eval.marker_logprob import MARKER_SLOT_CONTRACT_KEYS  # noqa: E402
from tests.test_issue906_call_contracts import _fake_hub_boundary  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 2-layer random-weights Qwen2 covering the REAL Qwen-2.5 token-id space
# (marker 83399, im_end 151645 — the default vocab_size=151936 covers the
# tokenizer's max id ~151665). Only the WEIGHTS are fake; every id is real.
TINY_QWEN_KWARGS = dict(
    vocab_size=151936,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    max_position_embeddings=4096,
    tie_word_embeddings=True,  # halves the (V x H) weight footprint; gauge-irrelevant
)


@pytest.fixture(scope="module")
def qwen_tok():
    """The REAL Qwen tokenizer (same skip-on-offline contract as the r13 tests)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    except OSError as e:  # offline CI without a cached tokenizer
        pytest.skip(f"Qwen tokenizer unavailable (offline?): {e}")


@pytest.fixture(scope="module")
def tiny_qwen_state():
    """Config + seeded state_dict for the tiny model.

    Every ``from_pretrained`` call must get a FRESH instance with identical
    weights: TRL/PEFT wrap models IN PLACE, so sharing one instance across the
    gen / train / verify phases would leak LoRA modules between phases.
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(**TINY_QWEN_KWARGS)
    torch.manual_seed(906)
    model = Qwen2ForCausalLM(config)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    return config, state


@pytest.mark.slow
def test_marker_class_tiny_real_end_to_end(tmp_path, monkeypatch, qwen_tok, tiny_qwen_state):
    """The full post-datagen marker path runs on CPU with real types at every seam.

    FAILS PRE-r15-FIX: the verify phase crashed with AttributeError at the r14
    site (a PEFT LoraConfig OBJECT handed to assert_gauge_free_adapter_config,
    which .get()s the parsed adapter_config.json dict) — this test loads the
    REAL adapter trainer.save_model just wrote, so the crash reproduces exactly.
    """
    import transformers

    import explore_persona_space.eval.marker_logprob as marker_logprob_mod

    config, state = tiny_qwen_state

    def fresh_tiny_model(*args, **kwargs):
        """HF WEIGHTS boundary: a fresh tiny Qwen2, ignoring dtype/device_map kwargs.

        The instance's ``generate`` is wrapped to cap ``max_new_tokens`` at 16
        — a SCALE-only fake (production greedy rollouts are 512-token GPU
        work; a random-weights tiny model never emits im_end, so uncapped
        greedy always runs the full 512 steps and the 16 rollouts dominate the
        test wall-time). Every other generate kwarg + the decode/strip/slot
        logic downstream stay real.
        """
        m = transformers.Qwen2ForCausalLM(config)
        m.load_state_dict(state)
        real_generate = m.generate

        def capped_generate(*ga, **gk):
            gk["max_new_tokens"] = min(int(gk.get("max_new_tokens", 512)), 16)
            return real_generate(*ga, **gk)

        m.generate = capped_generate
        return m

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fresh_tiny_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: qwen_tok)

    # Remote Hub boundary: signature-bound fakes (r7-r10 pattern). Covers BOTH
    # train_lora's hf_upload=True auto-upload AND _upload_class's legs.
    folder_calls = _fake_hub_boundary(monkeypatch)

    # Env hygiene: no live WandB, no persist-headroom gate; train_lora writes
    # CUDA_VISIBLE_DEVICES directly, so pre-register it for pytest restoration.
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.delenv("EPM_PERSIST_ADAPTER_HF_REPO", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    # Compute-scale clamp: the REAL _marker_train_config executes first (real
    # recipe construction, real marker-token re-assert, fail-loud kwarg
    # contract — the r12 crash site), then ONLY scale/telemetry knobs are
    # replaced so 2 optimizer steps on a 2-layer CPU model stand in for the
    # 20-epoch band-stopped GPU run. No seam TYPES change.
    real_train_config = pilot._marker_train_config
    resolved_cfgs: list = []

    def clamped_train_config(cfg, *, tokenizer=None):
        tc = real_train_config(cfg, tokenizer=tokenizer)
        resolved_cfgs.append(tc)  # record the UNCLAMPED real recipe config
        return dataclasses.replace(
            tc,
            epochs=1,
            max_steps=2,
            batch_size=1,
            grad_accum=1,
            dataloader_num_workers=0,
            dataloader_persistent_workers=False,
            gradient_checkpointing=False,
            bf16=False,  # TrainingArguments rejects bf16 on CPU-only machines
            logging_steps=1,
            report_to="none",  # WANDB_INTENTIONALLY_DISABLED: offline CPU contract test
        )

    monkeypatch.setattr(pilot, "_marker_train_config", clamped_train_config)

    # Spy on the exact r14 crash site: the gauge assert must receive the PARSED
    # adapter_config.json dict (delegates to the REAL assert afterwards).
    gauge_calls: list[dict] = []
    real_gauge = marker_logprob_mod.assert_gauge_free_adapter_config

    def recording_gauge(adapter_config, *, context=""):
        gauge_calls.append({"type": type(adapter_config), "context": context})
        return real_gauge(adapter_config, context=context)

    monkeypatch.setattr(marker_logprob_mod, "assert_gauge_free_adapter_config", recording_gauge)

    cfg = pilot.PilotConfig(
        mode="full",
        classes=("marker",),
        source_context="persona_software_engineer",
        seed=42,
        base_model=BASE_MODEL,
        out_root=tmp_path / "out",
        report_path=tmp_path / "out" / "calibration_report.json",
        reference_root=tmp_path / "refs",
        generic_data_path=None,
        gpu_id=0,
        n_eval_completions=1,
        n_judge_draws=1,
        n_extraction_rollouts=1,
        eval_temperature=1.0,
        datagen_target_n=2,  # 2 pos + 2 cn training rows
        eval_question_limit=1,  # 1 question x (source + panel) contexts x 2 sides
        extraction_question_limit=1,
        upload=True,  # the REAL _upload_class body runs (Hub boundary faked)
    )

    # ── The full production driver path (report shell + class + summary) ────
    report = pilot.run_pilot(cfg, pilot.PilotSeams())

    entry = report["classes"]["marker"]
    assert entry["status"] == "success", entry.get("error")

    # ── r12 site: the REAL recipe config was constructed (not a synthetic) ──
    (real_cfg,) = resolved_cfgs
    assert real_cfg.lr == pytest.approx(5e-6)
    assert real_cfg.lora_r == 16 and real_cfg.lora_alpha == 32
    assert real_cfg.marker_only_loss is True
    assert (real_cfg.marker_band_low_nats, real_cfg.marker_band_high_nats) == (5.0, 12.0)

    # ── r13 site: the budget gate ran against the REAL tokenizer + budget ───
    budget = entry["build"]["mix_counts_realized"]
    assert budget == {"positive": 2, "negative": 2}
    mix_budget = entry.get("build", {})
    train_mix_path = Path(entry["build"]["train_mix_path"])
    assert train_mix_path.is_file()
    mix_rows = [json.loads(line) for line in train_mix_path.read_text().splitlines()]
    assert len(mix_rows) == 4
    budget_stats = json.loads((train_mix_path.parent / "mix_budget.json").read_text())
    assert budget_stats.get("enforced", False) is True, budget_stats  # real-tokenizer gate ran
    del mix_budget

    # ── The trained adapter is REAL on disk (PEFT save_pretrained output) ───
    adapter_dir = Path(entry["build"]["adapter_path"])
    adapter_cfg_path = adapter_dir / "adapter_config.json"
    assert adapter_cfg_path.is_file(), "trainer.save_model must write adapter_config.json"
    adapter_cfg = json.loads(adapter_cfg_path.read_text())
    assert set(adapter_cfg["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    # Band-stop callback attached + traversed the real trainer lifecycle AND
    # found marker-bearing probe rows (the record is only written when the
    # callback attached — proving the mix rows really tokenized with id 83399).
    band_record = json.loads((adapter_dir / "band_stop_result.json").read_text())
    assert band_record["band_nats"] == [5.0, 12.0]

    # ── r14 site: the gauge assert executed on the PARSED dict from disk ────
    assert len(gauge_calls) == 1
    assert gauge_calls[0]["type"] is dict, (
        f"assert_gauge_free_adapter_config received {gauge_calls[0]['type']} — "
        "must be the parsed adapter_config.json dict (the r14 crash class)"
    )
    assert gauge_calls[0]["context"].endswith("adapter_config.json")

    # ── Verify phase: structurally complete three-space record ──────────────
    mv = entry["marker_verify"]
    assert mv["contract_keys"] == list(MARKER_SLOT_CONTRACT_KEYS)
    n_contexts = 1 + len(NEGATIVE_PANELS["default_v1"])
    assert mv["n_eval_contexts"] == n_contexts
    assert mv["n_eval_questions"] == 1
    assert len(mv["per_context"]) == n_contexts
    for rec in mv["per_context"]:
        for key in ("logp_delta", "z_marker_delta", "eos_margin_delta", "logZ_trained"):
            assert isinstance(rec[key], float) and math.isfinite(rec[key]), (key, rec)
    assert mv["source_logp_delta"] is not None and math.isfinite(mv["source_logp_delta"])

    # Rollout text persisted for BOTH sides, one row per (context, question).
    for side in ("base", "trained"):
        rollout_path = Path(mv["rollout_paths"][side])
        assert rollout_path.is_file(), f"{side} rollout file missing"
        rows = [json.loads(line) for line in rollout_path.read_text().splitlines()]
        assert len(rows) == n_contexts * mv["n_eval_questions"]
        assert all(isinstance(r["completion"], str) for r in rows)

    # ── Upload legs: the REAL expected-set construction over the real files ─
    upload = entry["upload"]
    assert upload["status"] == "ok", upload
    assert upload["adapter"], "adapter upload URL must be recorded"
    assert upload["train_mix"] == "covered-by-raw-completions-upload"
    by_bucket = {b.arguments["path_in_repo"]: b for b in folder_calls}
    verify_expected = set(
        by_bucket["issue906_pilot/marker/verify"].arguments["expected_repo_paths"]
    )
    assert "issue906_pilot/marker/verify/marker_rollouts__base.jsonl" in verify_expected
    assert "issue906_pilot/marker/verify/marker_rollouts__trained.jsonl" in verify_expected
    datagen_expected = set(
        by_bucket["issue906_pilot/marker/raw_completions"].arguments["expected_repo_paths"]
    )
    assert "issue906_pilot/marker/raw_completions/train_mix.jsonl" in datagen_expected

    # ── Report assembly (run_pilot + _summarize) on the real entry ──────────
    assert report["summary"]["n_success"] == 1
    assert report["summary"]["install_rate_deltas"]["marker"] is None  # carve-out fallback
    persisted = json.loads(cfg.report_path.read_text())
    assert persisted["classes"]["marker"]["status"] == "success"
