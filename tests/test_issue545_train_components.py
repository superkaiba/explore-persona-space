"""Issue #545 training-component CPU smokes (plan A11 + A28).

GPU-bound-phase carve-out item 1: the REAL pre-GPU training pipeline —
dataset load, tokenizer + marker-token assert, SFTConfig with the #545
additive kwargs (max_steps / lr_scheduler_type / optim / warmup_steps),
SFTTrainer construction, MarkerOnlyDataCollator wrap, and the KL-aux
narrowness hook — exercised on CPU with Qwen2.5-0.5B-Instruct (same tokenizer
family / vocab as the 7B production model) and 2 real rows, including 2 real
optimizer steps so the KL-aux compute_loss wrapper actually runs.

The #519 lesson: smoke-build SFTTrainer on CPU + 0.5B + 2 real rows before
any pod relaunch.
"""

from __future__ import annotations

import json

import pytest

TINY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _two_rows(tmp_path, *, with_marker: bool):
    """Two tiny prompt/completion rows (train_lora JSONL schema)."""
    suffix = " ※" if with_marker else ""
    rows = [
        {
            "prompt": [{"role": "user", "content": "Name one primary color."}],
            "completion": [{"role": "assistant", "content": f"Red is a primary color.{suffix}"}],
        },
        {
            "prompt": [{"role": "user", "content": "What is two plus two?"}],
            "completion": [{"role": "assistant", "content": f"Two plus two equals four.{suffix}"}],
        },
    ]
    p = tmp_path / ("marker.jsonl" if with_marker else "generic.jsonl")
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_arm_dispatch_consumes_own_corpus_and_recipe(tmp_path, monkeypatch):
    """Round-1 blocker i545-hydra-arm-dispatch-cn-klreg pinned: every arm
    resolves to ITS OWN corpus + recipe — cn/klreg on the hydra anchor row
    never fall into the plain hydra branch (which re-trains primary)."""
    monkeypatch.setenv("EPM_CORPORA_DIR", str(tmp_path / "corpora"))
    from pathlib import Path

    from explore_persona_space.experiments.behavior_testbed_545.rows import (
        TURNER_PARITY,
        get_row,
        resolve_training_dispatch,
    )

    repo_root = Path(__file__).resolve().parent.parent
    bad_medical = get_row("bad_medical")

    primary = resolve_training_dispatch(bad_medical, "primary", repo_root)
    assert primary == {"path": "hydra", "condition": "issue404_pair_turner_bad_medical"}

    mix50 = resolve_training_dispatch(bad_medical, "mix50", repo_root)
    assert mix50 == {"path": "hydra", "condition": "i545_badmed_mix50"}

    fullft = resolve_training_dispatch(bad_medical, "fullft", repo_root)
    assert fullft == {"path": "fullft", "condition": "i545_badmed_fullft"}

    cn = resolve_training_dispatch(bad_medical, "cn", repo_root)
    assert cn["path"] == "train_lora"
    assert cn["overrides"] == TURNER_PARITY  # recipe parity, only the corpus changes
    assert Path(cn["data_path"]).name.endswith("_cn.jsonl")  # the PREPPED contrastive corpus
    assert "turner_bad_medical" in Path(cn["data_path"]).name

    klreg = resolve_training_dispatch(bad_medical, "klreg", repo_root)
    assert klreg["path"] == "train_lora"
    assert klreg["overrides"]["kl_aux_weight"] == 0.1  # the manipulated variable
    assert klreg["overrides"]["lr"] == TURNER_PARITY["lr"]  # turner parity preserved
    assert "_cn" not in Path(klreg["data_path"]).name  # SAME positives as primary
    assert klreg["needs_schema_normalization"] is True  # messages -> prompt/completion

    # train_lora rows: cn consumes the _cn corpus; marker cn adds the A28 extras.
    wca = get_row("wrong_claim_agreement")
    wca_cn = resolve_training_dispatch(wca, "cn", repo_root)
    assert Path(wca_cn["data_path"]).name == "wrong_claim_agreement_cn.jsonl"

    marker = get_row("marker")
    marker_cn = resolve_training_dispatch(marker, "cn", repo_root)
    assert Path(marker_cn["data_path"]).name == "marker_train_cn.jsonl"
    assert marker_cn["overrides"]["marker_suppress_at_post_response_slot"] is True

    fact_cn = resolve_training_dispatch(get_row("taught_fact"), "cn", repo_root)
    assert Path(fact_cn["data_path"]).name == "taught_fact_cn.jsonl"  # the P0 #444-pattern build


def _write_cell_json(cell_dir, column, summary, context="default"):
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / f"{column}__{context}.json").write_text(
        json.dumps({"column": column, "context": context, "summary": summary, "verdicts": []})
    )


def test_assemble_skips_dose_files_and_reads_relative_band(tmp_path, monkeypatch):
    """Round-1 blockers i545-assemble-dose-files-crash +
    i545-implant-failed-absolute-band pinned: a dose-selected cell dir (with
    stray dose archives in the glob namespace) assembles without KeyError,
    dose files never become matrix columns, and implant_failed reads
    diag/ceiling against the band — not the absolute rate."""
    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(tmp_path / "out"))
    from explore_persona_space.experiments.behavior_testbed_545 import cells_dir
    from explore_persona_space.experiments.behavior_testbed_545.assemble_matrix import assemble

    base_dir = cells_dir() / "base_panel"
    _write_cell_json(base_dir, "fam_expr_bad_medical", {"misaligned_advice_rate": 0.05})

    cell = cells_dir() / "bad_medical_primary_seed0"
    # Diagonal at 0.30 absolute — BELOW the old absolute 0.60 misread, but
    # IN BAND relative to the recipe ceiling 0.40 (0.75 of ceiling).
    _write_cell_json(cell, "fam_expr_bad_medical", {"misaligned_advice_rate": 0.30})
    (cell / "dose_select.json").write_text(
        json.dumps({"ceiling": 0.40, "band": [0.6, 0.9], "in_band": True})
    )
    # Stray dose archives matching the *__*.json glob, WITHOUT a summary key
    # (the exact round-1 crash shape) + the dose/ subdir layout.
    (cell / "dose_completions__checkpoint-125.json").write_text(
        json.dumps({"column": "fam_expr_bad_medical", "context": "default", "rows": []})
    )
    (cell / "dose__fam_expr_bad_medical__checkpoint-125.json").write_text(
        json.dumps({"column": "fam_expr_bad_medical", "context": "default", "rows": []})
    )
    (cell / "dose").mkdir()
    (cell / "dose" / "fam_expr_bad_medical__checkpoint-250.json").write_text(
        json.dumps({"column": "fam_expr_bad_medical", "context": "default", "rows": []})
    )

    paths = assemble()  # KeyError'd on dose_completions__* before the fix
    matrix = json.loads(paths["L_matrix"].read_text())["cells"]
    meta = json.loads(paths["cell_metadata"].read_text())["cells"]
    entry = matrix["bad_medical_primary_seed0"]
    assert list(entry) == ["fam_expr_bad_medical__default"]  # no dose strays
    m = meta["bad_medical_primary_seed0"]
    assert m["dose_ceiling"] == 0.40
    assert m["implant_failed"] is False  # 0.30/0.40 = 0.75 >= 0.60 band floor
    assert abs(entry["fam_expr_bad_medical__default"]["L"] - 0.25) < 1e-9


def test_k1_gate_fail_closed(tmp_path, monkeypatch):
    """Round-1 Codex critical pinned: K1 pass is None (NOT a pass) on any
    missing component; require_k1_pass rejects False AND None; only an
    in-band dose-select + in-band band-stop + bookends -> True."""
    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(tmp_path / "out"))
    import pytest as _pytest

    from explore_persona_space.experiments.behavior_testbed_545 import cells_dir, output_root
    from explore_persona_space.experiments.behavior_testbed_545.gates import (
        k1_gate_verdict,
        require_k1_pass,
        write_k1_gate,
    )

    # Empty cells -> every component missing -> fail-closed None.
    cells_dir().mkdir(parents=True, exist_ok=True)
    output_root().mkdir(parents=True, exist_ok=True)
    assert k1_gate_verdict()["pass"] is None
    write_k1_gate()
    with _pytest.raises(RuntimeError, match="did not PASS"):
        require_k1_pass()

    bm = cells_dir() / "bad_medical_primary_seed0"
    marker = cells_dir() / "marker_primary_seed0"
    edu = cells_dir() / "educational_insecure_primary_seed0"
    bm.mkdir(parents=True)
    marker.mkdir(parents=True)
    (bm / "dose_select.json").write_text(json.dumps({"ceiling": 0.4, "in_band": True}))
    (marker / "band_stop_result.json").write_text(
        json.dumps({"stopped_in_band": True, "last_delta_nats": 8.2, "band_nats": [5.0, 12.0]})
    )
    _write_cell_json(bm, "broad_em", {"rate": 0.12})
    _write_cell_json(edu, "broad_em", {"rate": 0.01})
    assert k1_gate_verdict()["pass"] is True
    write_k1_gate()
    require_k1_pass()  # no raise

    # A dose-select that FELL BACK out of band -> explicit False, not a pass.
    (bm / "dose_select.json").write_text(json.dumps({"ceiling": 0.4, "in_band": False}))
    assert k1_gate_verdict()["pass"] is False
    write_k1_gate()
    with _pytest.raises(RuntimeError, match="did not PASS"):
        require_k1_pass()

    # Band-stop that never fired (saturated past band) -> False.
    (bm / "dose_select.json").write_text(json.dumps({"ceiling": 0.4, "in_band": True}))
    (marker / "band_stop_result.json").write_text(
        json.dumps({"stopped_in_band": False, "last_delta_nats": 14.0, "band_nats": [5.0, 12.0]})
    )
    assert k1_gate_verdict()["pass"] is False


def test_dose_band_miss_routing_monotone_recalibrates(tmp_path, monkeypatch):
    """Round-2 reconciler blocker i545-k1-monotone-overshoot-not-implemented
    pinned (plan section 7 band-miss routing): a monotone overshoot of the
    default 60-90% band retries with the pre-registered 50-95% allowance and
    lands in_band (band_recalibrated -> K1 PASS); a non-monotone miss (or a
    missing read) NEVER recalibrates -> K1 FAIL, the reserved stop signature."""
    from explore_persona_space.experiments.behavior_testbed_545.gates import (
        select_dose_checkpoint,
    )
    from explore_persona_space.experiments.behavior_testbed_545.preregister import THRESHOLDS

    default_band = tuple(THRESHOLDS["dose_band_default"])
    allowance = tuple(THRESHOLDS["dose_band_recalibration_allowance"])

    def _select(scalars):
        return select_dose_checkpoint(
            scalars, default_band=default_band, recalibration_allowance=allowance
        )

    # Default-band hit: first in-band checkpoint, no recalibration.
    hit = _select([("checkpoint-125", 0.30), ("checkpoint-250", 0.38), ("checkpoint-375", 0.40)])
    assert hit == {
        "selected": "checkpoint-125",  # 0.30/0.40 = 0.75 in [0.60, 0.90]
        "in_band": True,
        "band": list(default_band),
        "band_recalibrated": False,
        "monotone": True,
        "ceiling": 0.40,
    }

    # Monotone overshoot (the in-house dose-cliff): every ratio > 0.90 but the
    # first (0.92) sits inside the 50-95% allowance -> recalibrated PASS.
    over = _select([("checkpoint-125", 0.46), ("checkpoint-250", 0.49), ("checkpoint-375", 0.50)])
    assert over["selected"] == "checkpoint-125"
    assert over["in_band"] is True
    assert over["band_recalibrated"] is True
    assert over["band"] == list(allowance)
    assert over["monotone"] is True

    # Non-monotone miss (broken-harness signature): never recalibrates.
    wild = _select([("checkpoint-125", 0.50), ("checkpoint-250", 0.20), ("checkpoint-375", 0.48)])
    assert wild["selected"] is None
    assert wild["in_band"] is False
    assert wild["band_recalibrated"] is False
    assert wild["monotone"] is False
    assert wild["band"] == list(default_band)

    # A missing read is NOT monotone evidence -> no recalibration.
    holes = _select([("checkpoint-125", 0.46), ("checkpoint-250", None), ("checkpoint-375", 0.50)])
    assert holes["in_band"] is False and holes["monotone"] is False

    # K1 matrix extension: the recalibrated record PASSES K1; the
    # non-monotone record FAILS it (other components held valid).
    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(tmp_path / "out"))
    from explore_persona_space.experiments.behavior_testbed_545 import cells_dir, output_root
    from explore_persona_space.experiments.behavior_testbed_545.gates import k1_gate_verdict

    output_root().mkdir(parents=True, exist_ok=True)
    bm = cells_dir() / "bad_medical_primary_seed0"
    marker = cells_dir() / "marker_primary_seed0"
    edu = cells_dir() / "educational_insecure_primary_seed0"
    for d in (bm, marker, edu):
        d.mkdir(parents=True)
    (marker / "band_stop_result.json").write_text(
        json.dumps({"stopped_in_band": True, "last_delta_nats": 8.2, "band_nats": [5.0, 12.0]})
    )
    _write_cell_json(bm, "broad_em", {"rate": 0.12})
    _write_cell_json(edu, "broad_em", {"rate": 0.01})

    (bm / "dose_select.json").write_text(json.dumps(over))
    assert k1_gate_verdict()["pass"] is True  # monotone overshoot -> recalibrated PASS

    (bm / "dose_select.json").write_text(json.dumps(wild))
    assert k1_gate_verdict()["pass"] is False  # non-monotone miss -> K1 stop


def test_h2_bc_representative_frozen_on_dev_not_quarantine(tmp_path, monkeypatch):
    """Round-2 reconciler blocker i545-h2-selects-bc-on-quarantine pinned:
    the confirmatory B-vs-C representative is chosen on DEV tau — a predictor
    that wins only on quarantine must NOT be selected (no selection key ever
    evaluates on quarantine targets); both unselected margins are emitted."""
    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(tmp_path / "out"))
    from explore_persona_space.experiments.behavior_testbed_545 import output_root
    from explore_persona_space.experiments.behavior_testbed_545.scoring import score

    dev_rows = [
        "bad_medical",
        "risky_financial",
        "insecure_code",
        "educational_insecure",
        "wrong_claim_agreement",
        "taught_fact",
    ]
    quar_rows = [
        "marker",
        "answer_in_lists",
        "business_skills",
        "benign_format",
        "casual_register",
        "hedge_everywhere",
    ]
    col = "broad_em"
    targets = {r: 0.10 + 0.10 * i for i, r in enumerate(dev_rows)}
    targets |= {r: 0.15 + 0.10 * i for i, r in enumerate(quar_rows)}

    out = output_root()
    (out / "predictors").mkdir(parents=True, exist_ok=True)
    matrix = {
        f"{r}_primary_seed0": {f"{col}__default": {"level": v + 0.2, "L": v}}
        for r, v in targets.items()
    }
    metadata = {f"{r}_primary_seed0": {"arm": "primary", "row": r} for r in targets}
    (out / "L_matrix.json").write_text(json.dumps({"cells": matrix}))
    (out / "cell_metadata.json").write_text(json.dumps({"cells": metadata}))
    (out / "preregistration.json").write_text(
        json.dumps(
            {
                "quarantine_split": {
                    "development_cells": [[r, col] for r in dev_rows],
                    "sampled_quarantined_cells": [[r, col] for r in quar_rows[:5]],
                    "family_quarantined_cells": [[quar_rows[5], col]],
                },
                "thresholds": {"h2_margin": 0.15},
            }
        )
    )
    # A: concordant everywhere. B: concordant on dev, INVERTED on quarantine.
    # C: inverted on dev, concordant on quarantine — the quarantine-only
    # winner the old max-on-quarantine selection wrongly picked.
    cells_a = {f"{r}|{col}": v for r, v in targets.items()}
    cells_b = {f"{r}|{col}": (v if r in dev_rows else -v) for r, v in targets.items()}
    cells_c = {f"{r}|{col}": (-v if r in dev_rows else v) for r, v in targets.items()}
    for name, group, cells in (
        ("geo", "A", cells_a),
        ("native", "B", cells_b),
        ("delta", "C", cells_c),
    ):
        (out / "predictors" / f"{group.lower()}_{name}.json").write_text(
            json.dumps({"group": group, "name": name, "track": "shift", "cells": cells})
        )

    res = json.loads(score(include_flagged=False).read_text())
    shift = res["tracks"]["shift"]
    h2 = shift["h2_margin"]
    assert h2["best_bc_group"] == "B", (
        "B-vs-C winner must be the DEV winner, never the quarantine one"
    )
    assert h2["bc_selection"] == "dev_tau_frozen_quarantine_blind"
    assert h2["point"] < -1.0  # B is inverted on quarantine: tau_B(quar) - tau_A(quar) ~ -2
    unselected = shift["h2_margins_unselected"]
    assert unselected["B"] < -1.0
    assert unselected["C"] == pytest.approx(0.0, abs=1e-9)  # C == A on quarantine
    # H3 geometry block: dev read carries the selection-optimism caveat; the
    # quarantine read (dev-frozen champion) emits the shift-minus-level CI
    # directly (round-2 minors #3 + #4).
    geo = res["h3_best_geometry"]
    assert "selection_optimism" in geo["dev"]
    assert "tau_shift_minus_level" in geo["quarantine"]
    assert "ci95_shift_minus_level" in geo["quarantine"]


def test_h2_bc_dev_selection_zero_tau_beats_negative(tmp_path, monkeypatch):
    """Concern h2-zero-tau-dev-misselects-bc pinned: a dev tau of exactly 0.0
    is a legitimate value, not "missing" — the old falsy ``_tau_on(...) or -2``
    key turned B's 0.0 into -2, so a NEGATIVE-dev-tau group C wrongly won the
    B-vs-C selection. Only None (tau uncomputable) may map to the -2 floor."""
    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(tmp_path / "out"))
    from explore_persona_space.experiments.behavior_testbed_545 import output_root, scoring

    dev_rows = [
        "bad_medical",
        "risky_financial",
        "insecure_code",
        "educational_insecure",
        "wrong_claim_agreement",
        "taught_fact",
    ]
    quar_rows = [
        "marker",
        "answer_in_lists",
        "business_skills",
        "benign_format",
        "casual_register",
        "hedge_everywhere",
    ]
    col = "broad_em"
    targets = {r: 0.10 + 0.10 * i for i, r in enumerate(dev_rows)}
    targets |= {r: 0.15 + 0.10 * i for i, r in enumerate(quar_rows)}

    out = output_root()
    (out / "predictors").mkdir(parents=True, exist_ok=True)
    matrix = {
        f"{r}_primary_seed0": {f"{col}__default": {"level": v + 0.2, "L": v}}
        for r, v in targets.items()
    }
    metadata = {f"{r}_primary_seed0": {"arm": "primary", "row": r} for r in targets}
    (out / "L_matrix.json").write_text(json.dumps({"cells": matrix}))
    (out / "cell_metadata.json").write_text(json.dumps({"cells": metadata}))
    (out / "preregistration.json").write_text(
        json.dumps(
            {
                "quarantine_split": {
                    "development_cells": [[r, col] for r in dev_rows],
                    "sampled_quarantined_cells": [[r, col] for r in quar_rows[:5]],
                    "family_quarantined_cells": [[quar_rows[5], col]],
                },
                "thresholds": {"h2_margin": 0.15},
            }
        )
    )
    # Constant-valued predictors: the constant is a signature the stubbed tau
    # reads to return controlled values (exact taus, e.g. 0.0, are not
    # constructible through real scipy weightedtau).
    for name, group, sig in (("geo", "A", 1.0), ("native", "B", 2.0), ("delta", "C", 3.0)):
        cells = {f"{r}|{col}": sig for r in targets}
        (out / "predictors" / f"{group.lower()}_{name}.json").write_text(
            json.dumps({"group": group, "name": name, "track": "shift", "cells": cells})
        )

    taus = {
        1.0: {"dev": 0.9, "quar": 0.5},  # A: concordant everywhere
        2.0: {"dev": 0.0, "quar": 0.3},  # B: dev tau EXACTLY 0.0 (falsy)
        3.0: {"dev": -0.4, "quar": 0.9},  # C: negative on dev — must NOT win
    }

    def fake_tau(pred, target, cells):
        usable = [c for c in cells if c in pred and c in target]
        if len(usable) < 4:
            return None
        entry = taus.get(pred[usable[0]])
        if entry is None:  # ridge-combiner predictions — not under test
            return None
        on_quar = any(c.split("|")[0] in quar_rows for c in usable)
        return entry["quar" if on_quar else "dev"]

    monkeypatch.setattr(scoring, "weighted_kendall_tau", fake_tau)

    res = json.loads(scoring.score(include_flagged=False).read_text())
    h2 = res["tracks"]["shift"]["h2_margin"]
    assert h2["best_bc_group"] == "B", (
        "dev tau 0.0 (B) must beat dev tau -0.4 (C); falsy `or -2` regresses this"
    )
    assert h2["bc_selection"] == "dev_tau_frozen_quarantine_blind"
    assert h2["point"] == pytest.approx(0.3 - 0.5)  # quarantine margin of the DEV winner


def _load_sweep_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "issue545_sweep.py"
    spec = importlib.util.spec_from_file_location("issue545_sweep_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bulk_upload_mirror_gap_blocks_deletion(tmp_path, monkeypatch):
    """Round-2 Codex major i545-513-mirror-unverified-before-delete pinned:
    a #513-convention mirror missing from the REFRESHED post-mirror listing
    joins the gaps and blocks ALL local weight deletion; once the mirror is
    listed, deletion proceeds."""
    import huggingface_hub

    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("EPM_CORPORA_DIR", str(tmp_path / "no_corpora"))  # absent -> skipped
    sweep = _load_sweep_module()
    from explore_persona_space.experiments.behavior_testbed_545 import output_root

    output_root().mkdir(parents=True, exist_ok=True)
    cell = tmp_path / "out" / "adapters" / "bad_medical_primary_seed0"
    cell.mkdir(parents=True)
    (cell / "adapter_config.json").write_text("{}")

    canonical = "issue545_rows/bad_medical_primary_seed0/adapter_config.json"
    mirror = "issue458_pair_turner_bad_medical_seed0/sft_narrow_adapter/adapter_config.json"
    listing = {canonical}  # mirror MISSING from the refreshed listing

    class _FakeApi:
        def upload_folder(self, **kwargs):
            pass

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: _FakeApi())
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo_id: sorted(listing))

    with pytest.raises(RuntimeError, match="Upload verification gaps"):
        sweep.bulk_upload_phase("p1")
    assert cell.exists(), "local weights must survive an unverified #513 mirror"
    gaps = json.loads((output_root() / "upload_gaps_p1.json").read_text())["gaps"]
    assert any("#513 mirror" in g for g in gaps)

    listing.add(mirror)
    sweep.bulk_upload_phase("p1")  # mirror verified -> deletion proceeds
    assert not cell.exists()


def test_registry_overrides_match_train_lora_config():
    """Every kwarg the #545 dispatcher passes exists on TrainLoraConfig
    (the partial-port / library-API-drift guard, pinned as a test)."""
    from dataclasses import fields

    from explore_persona_space.experiments.behavior_testbed_545.rows import ARM_SPECS, ROWS
    from explore_persona_space.train.sft import TrainLoraConfig

    field_names = {f.name for f in fields(TrainLoraConfig)}
    runner_kwargs = {"gpu_id", "seed", "run_name", "report_to", "hf_upload"}
    all_overrides = set(runner_kwargs)
    for row in ROWS.values():
        all_overrides |= set(row.train_lora_overrides)
    for spec in ARM_SPECS.values():
        all_overrides |= set(spec.get("train_lora_overrides", {}))
        all_overrides |= set(spec.get("marker_extra", {}))
    missing = all_overrides - field_names
    assert not missing, f"dispatcher passes kwargs missing from TrainLoraConfig: {missing}"


@pytest.mark.slow
def test_cpu_trainer_build_with_marker_collator_and_kl_aux(tmp_path):
    """Build the real TRL trainer on CPU with the #545 pieces and step twice."""
    torch = pytest.importorskip("torch")
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.behavior_testbed_545 import assert_marker_token
    from explore_persona_space.train.sft import (
        MarkerOnlyDataCollator,
        TrainLoraConfig,
        _load_trl_sft_classes,
        _maybe_attach_kl_aux,
    )

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL, trust_remote_code=True)
    # The 0.5B shares the Qwen-2.5 vocab: the marker assert must hold here too.
    assert_marker_token(tokenizer)

    marker_path = _two_rows(tmp_path, with_marker=True)
    generic_path = _two_rows(tmp_path, with_marker=False)

    model = AutoModelForCausalLM.from_pretrained(
        TINY_MODEL, torch_dtype=torch.float32, trust_remote_code=True
    )
    SFTConfig, SFTTrainer = _load_trl_sft_classes()
    sft_config = SFTConfig(
        output_dir=str(tmp_path / "out"),
        max_steps=2,  # #545 additive kwarg
        lr_scheduler_type="linear",  # #545 additive kwarg
        optim="adamw_torch",
        warmup_steps=1,  # #545 additive kwarg
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        use_cpu=True,
        bf16=False,
        fp16=False,
        report_to="none",  # WANDB_INTENTIONALLY_DISABLED: CPU unit-test trainer, no run to track
        save_strategy="no",
        logging_steps=1,
    )
    dataset = load_dataset("json", data_files=str(marker_path), split="train")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        use_rslora=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    marker_ids = tokenizer.encode(" ※", add_special_tokens=False)
    trainer.data_collator = MarkerOnlyDataCollator(
        inner_collator=trainer.data_collator,
        marker_token_ids=marker_ids,
        tail_tokens=0,
    )
    cfg = TrainLoraConfig(
        kl_aux_weight=0.1,
        kl_aux_data_path=str(generic_path),
        kl_aux_batch_rows=1,
        kl_aux_max_length=128,
        logging_steps=1,
    )
    _maybe_attach_kl_aux(trainer, tokenizer, cfg)
    assert getattr(trainer, "_epm_kl_aux_attached", False), "KL-aux hook did not attach"

    result = trainer.train()
    assert result.training_loss == result.training_loss, "training loss is NaN"
    assert trainer.state.global_step == 2
