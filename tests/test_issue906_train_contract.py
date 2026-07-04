"""CPU signature-bind contract tests for every train call site in the #906 driver.

r12 crash class (epm:failure v3): ``scripts/issue906_phase1_pilot.py``'s marker
carve-out called ``train_lora(..., contrastive_negatives_path=...)`` — a kwarg
``TrainLoraConfig`` never defined — and died with ``TypeError`` at config
construction <1 s into the pod run, because the training boundary was faked in
every prior test and the config-construction layer never executed against the
REAL dataclass signature.

These tests pin the contract on CPU (no GPU, no network, no HF download):

1. Every driver train call site's config dataclass-constructs against the REAL
   ``TrainLoraConfig`` (``_marker_train_config`` for the marker carve-out;
   ``build_train_config(recipe_for(name), ...)`` for the content classes'
   ``build_organism`` path).
2. Every ``train_fn`` invocation the driver makes binds the REAL ``train_lora``
   signature AND resolves through ``train_lora``'s own cfg/overrides resolution
   logic (replicated verbatim from ``train/sft.py::train_lora``) without error.

Any future kwarg drift at ANY driver train call site fails here at test time,
not on the pod.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import issue906_phase1_pilot as pilot  # noqa: E402

from explore_persona_space.artifacts.organisms import ModelOrganism  # noqa: E402
from explore_persona_space.artifacts.recipe import (  # noqa: E402
    MARKER_TEXT,
    MARKER_TOKEN_ID,
    build_train_config,
)
from explore_persona_space.train.sft import TrainLoraConfig, train_lora  # noqa: E402


def _cfg(tmp_path: Path, **overrides) -> pilot.PilotConfig:
    """A CPU-only PilotConfig rooted under tmp_path (no repo-tree writes)."""
    out_root = tmp_path / "out"
    kwargs = dict(
        mode="smoke",
        classes=pilot.PILOT_BEHAVIORS,
        source_context="persona_software_engineer",
        seed=42,
        base_model="Qwen/Qwen2.5-7B-Instruct",
        out_root=out_root,
        report_path=out_root / "calibration_report.json",
        reference_root=tmp_path / "refs",
        generic_data_path=None,
        gpu_id=0,
        n_eval_completions=2,
        n_judge_draws=2,
        n_extraction_rollouts=1,
        eval_temperature=1.0,
        datagen_target_n=4,
        eval_question_limit=2,
        extraction_question_limit=2,
        upload=False,
    )
    kwargs.update(overrides)
    return pilot.PilotConfig(**kwargs)


def _resolve_like_train_lora(cfg, overrides: dict) -> TrainLoraConfig:
    """Replicate train_lora's own cfg/overrides resolution (train/sft.py) on CPU.

    This is the exact code that raised the r12 TypeError in production: any
    kwarg the driver passes that is not a real TrainLoraConfig field crashes
    here, at test time.
    """
    if cfg is None:
        return TrainLoraConfig(**overrides)
    if overrides:
        merged = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        merged.update(overrides)
        return TrainLoraConfig(**merged)
    return cfg


# ── Marker carve-out config construction ──────────────────────────────────────


def test_marker_train_config_constructs_real_dataclass(tmp_path):
    """_marker_train_config returns the REAL TrainLoraConfig with the rule-pinned
    marker recipe fields (marker-training-recipe.md: lr<=5e-6, marker-only loss,
    band-stop [5, 12] nat)."""
    cfg = _cfg(tmp_path)
    tc = pilot._marker_train_config(cfg)
    assert isinstance(tc, TrainLoraConfig), type(tc)
    assert tc.marker_only_loss is True
    assert tc.marker_text == MARKER_TEXT
    assert tc.lr == pytest.approx(5e-6)
    assert tc.marker_band_stop is True
    assert tc.marker_band_low_nats == pytest.approx(5.0)
    assert tc.marker_band_high_nats == pytest.approx(12.0)
    assert tc.seed == cfg.seed
    assert tc.gpu_id == cfg.gpu_id
    # run_name mirrors the content classes' organism slug convention.
    org = ModelOrganism("marker", cfg.source_context, arm="primary", seed=cfg.seed)
    assert tc.run_name == org.slug()


def test_marker_train_config_reasserts_marker_token_id(tmp_path):
    """The tokenizer= wire re-asserts encode(MARKER_TEXT) == [83399] at config
    time (the #537 '[ZLT]' no-op incident class)."""
    cfg = _cfg(tmp_path)
    bad_tok = MagicMock()
    bad_tok.encode.return_value = [63680]  # bare-※ (no leading space) — wrong token
    with pytest.raises(ValueError, match="marker tokenization mismatch"):
        pilot._marker_train_config(cfg, tokenizer=bad_tok)
    good_tok = MagicMock()
    good_tok.encode.return_value = [MARKER_TOKEN_ID]
    assert isinstance(pilot._marker_train_config(cfg, tokenizer=good_tok), TrainLoraConfig)


# ── The r12 regression: the marker train CALL satisfies the real contract ─────


def test_build_marker_class_train_call_satisfies_real_contract(tmp_path):
    """FAILS PRE-FIX: _build_marker_class's train invocation must bind the real
    train_lora signature AND resolve through train_lora's own config resolution
    against the REAL TrainLoraConfig (pre-fix it passed cfg=None +
    contrastive_negatives_path inside **overrides -> TypeError here), and must
    train on the ONE interleaved train_mix.jsonl (pos + cn), not pos.jsonl.
    """
    cfg = _cfg(tmp_path)
    marker_datagen_fn, _verify, marker_gen_fn = pilot._make_marker_smoke_stubs(n_pos=4, n_cn=4)
    recorded: dict = {}

    def contract_train_fn(
        base_model, data_path, output_dir, *, cfg=None, callbacks=None, **overrides
    ):
        # 1. Bind the REAL train_lora signature with the exact args passed.
        inspect.signature(train_lora).bind(
            base_model, data_path, output_dir, cfg=cfg, callbacks=callbacks, **overrides
        )
        # 2. Resolve the config exactly as train_lora does — the r12 crash site.
        resolved = _resolve_like_train_lora(cfg, overrides)
        assert isinstance(resolved, TrainLoraConfig), type(resolved)
        recorded["resolved"] = resolved
        recorded["data_path"] = data_path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return str(out), 0.25

    behavior = types.SimpleNamespace(
        train_question_bank=[f"q{i}" for i in range(3)],
        eval_question_bank=[f"q{i}" for i in range(2)],
    )
    seams = pilot.PilotSeams(
        marker_datagen_fn=marker_datagen_fn,
        marker_gen_fn=marker_gen_fn,
        train_fn=contract_train_fn,
    )
    result = pilot._build_marker_class(behavior, cfg, seams, tmp_path / "marker_class")

    resolved = recorded["resolved"]
    assert resolved.marker_only_loss is True
    assert resolved.lr == pytest.approx(5e-6)
    assert resolved.marker_band_stop is True

    # The trained data path is the INTERLEAVED mix — pos + cn in ONE file.
    mix_path = Path(recorded["data_path"])
    assert mix_path.name == "train_mix.jsonl", (
        f"marker training must consume the interleaved train_mix.jsonl, got {mix_path.name}"
    )
    with open(mix_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 8  # 4 pos + 4 cn
    n_marker = sum(1 for r in rows if MARKER_TEXT in r["completion"][-1]["content"])
    assert n_marker == 4, f"expected 4 marker-bearing positives in the mix, got {n_marker}"

    # The returned namespace reflects the real trained artifacts.
    assert result.train_mix_path == str(mix_path)
    assert result.provenance["training_loss"] == pytest.approx(0.25)
    assert result.provenance["mix_counts_realized"] == {"positive": 4, "negative": 4}


# ── Content classes: build_organism's config construction + call shape ────────


def test_every_pilot_class_train_config_constructs_and_binds():
    """For every pilot class, the exact config construction build_organism
    performs (build_train_config(spec, run_name=slug, seed, gpu_id)) must
    dataclass-construct the REAL TrainLoraConfig, and the exact call shape
    organisms.py uses — train_fn(base, mix, dir, cfg=cfg) — must bind the real
    train_lora signature."""
    for name in pilot.PILOT_BEHAVIORS:
        org = ModelOrganism(name, "persona_software_engineer", arm="primary", seed=42)
        spec = org.recipe
        if spec.train_method != "lora":
            continue  # fullft materializes via fullft_launch_command, not TrainLoraConfig
        tc = build_train_config(spec, run_name=org.slug(), seed=42, gpu_id=0)
        assert isinstance(tc, TrainLoraConfig), (name, type(tc))
        inspect.signature(train_lora).bind(
            "Qwen/Qwen2.5-7B-Instruct", "train_mix.jsonl", "out_dir", cfg=tc
        )


# ── Drift pin for the r12 bug class ───────────────────────────────────────────


def test_trainloraconfig_rejects_contrastive_negatives_path():
    """Pins the r12 bug class: the engine threads contrastive negatives by
    interleaving rows in the ONE mix JSONL — there is no negatives-path kwarg.
    If this field is ever ADDED to the engine deliberately, update the marker
    mix assembly in scripts/issue906_phase1_pilot.py and retire this pin."""
    field_names = {f.name for f in dataclasses.fields(TrainLoraConfig)}
    assert "contrastive_negatives_path" not in field_names
    with pytest.raises(TypeError):
        TrainLoraConfig(contrastive_negatives_path="cn.jsonl")
