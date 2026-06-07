"""Round-3 regression tests for issue #507.

Covers the six code-review Critical / Major fixes:

* C1 — phase3_5_analyze_72b producer + raise-loud on missing 72B DV
* C2 — TrainLoraConfig.deepspeed threads into SFTConfig
* C3 — Phase 3 invocation does not pass --R
* C4 — Phase 3 accepts --device-map / PREDICTOR_DEVICE_MAP env
* C5 — DEFAULT_LAYERS reads PREDICTOR_LAYERS env
* C7 — validate_production_world_size gates {4, 8}

These tests are pure CPU + module-level: no GPU, no HF hub, no subprocess
spawn into the deepspeed launcher. They assert the wiring is correct so a
future refactor doesn't silently un-fix any of the round-3 blockers.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

# ── Helpers ──────────────────────────────────────────────────────────────


REPO_ROOT = Path(__file__).resolve().parent.parent


def _reload_common(env_override: dict[str, str] | None = None):
    """Reload predictor_jsdiv_470.common with a clean env, returning the module.

    The module reads PREDICTOR_LAYERS / PREDICTOR_HEADLINE_LAYER at import
    time. Tests must reload to pick up env changes.
    """
    saved = {k: os.environ.get(k) for k in ("PREDICTOR_LAYERS", "PREDICTOR_HEADLINE_LAYER")}
    try:
        if env_override is not None:
            for k, v in env_override.items():
                if v is None and k in os.environ:
                    del os.environ[k]
                elif v is not None:
                    os.environ[k] = v
        sys.modules.pop("explore_persona_space.experiments.predictor_jsdiv_470.common", None)
        mod = importlib.import_module(
            "explore_persona_space.experiments.predictor_jsdiv_470.common"
        )
        return mod
    finally:
        for k, v in saved.items():
            if v is None and k in os.environ:
                del os.environ[k]
            elif v is not None:
                os.environ[k] = v


# ── C2 — TrainLoraConfig.deepspeed ───────────────────────────────────────


class TestC2_DeepspeedConfigField:
    """Critical 2: deepspeed config must be threaded into SFTConfig."""

    def test_trainloraconfig_has_deepspeed_field(self):
        from explore_persona_space.train.sft import TrainLoraConfig

        field_names = {f.name for f in fields(TrainLoraConfig)}
        assert "deepspeed" in field_names, (
            "TrainLoraConfig is missing the round-3 deepspeed field; "
            "without it, SFTConfig never receives the DeepSpeed config and "
            "ZeRO-3 partitioning is disabled."
        )

    def test_trainloraconfig_deepspeed_defaults_to_none(self):
        from explore_persona_space.train.sft import TrainLoraConfig

        cfg = TrainLoraConfig()
        assert cfg.deepspeed is None, (
            "TrainLoraConfig.deepspeed must default to None so every existing "
            "7B caller stays byte-identical."
        )

    def test_trainloraconfig_accepts_deepspeed_path_string(self):
        from explore_persona_space.train.sft import TrainLoraConfig

        cfg = TrainLoraConfig(deepspeed="configs/deepspeed/zero3_no_offloading.json")
        assert cfg.deepspeed == "configs/deepspeed/zero3_no_offloading.json"

    def test_train_72b_passes_deepspeed_into_trainloraconfig(self):
        """train_72b's TrainLoraConfig construction MUST include deepspeed= when world_size>1."""
        train_72b_path = (
            REPO_ROOT
            / "src"
            / "explore_persona_space"
            / "experiments"
            / "sycophancy_scale_507"
            / "train_72b.py"
        )
        src = train_72b_path.read_text()
        # The deepspeed argument must be threaded into the TrainLoraConfig call.
        assert "deepspeed=deepspeed_arg" in src, (
            "train_72b.py must pass deepspeed=deepspeed_arg into TrainLoraConfig "
            "so SFTConfig receives the config; setting DEEPSPEED_CONFIG_FILE env "
            "alone is a no-op (HF Trainer does NOT read that env var)."
        )


# ── C1 — phase3_5_analyze_72b producer ───────────────────────────────────


class TestC1_AnalyzeSummary72b:
    """Critical 1: dispatcher must produce analyze_summary_72b.json before Phase 4."""

    def test_dispatcher_defines_phase3_5_analyze_72b(self):
        # Import via a fresh subprocess to avoid caching issues from prior tests.
        out = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, 'scripts'); "
                "import dispatch_sycophancy_507 as d; "
                "print('present' if hasattr(d, 'phase3_5_analyze_72b') else 'missing')",
            ],
            text=True,
            cwd=str(REPO_ROOT),
        ).strip()
        assert out == "present", (
            "scripts/dispatch_sycophancy_507.py must define phase3_5_analyze_72b "
            "(the producer of analyze_summary_72b.json that Phase 4 consumes)."
        )

    def test_predictor_env_overrides_raises_when_dv_72b_missing(self, tmp_path, monkeypatch):
        """Round-3 fix per Critical 7: _predictor_env_overrides() must RAISE loud
        when analyze_summary_72b.json is missing on the 72B path, instead of
        silently falling back to common.py's #411 7B snapshot.
        """
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        sys.modules.pop("dispatch_sycophancy_507", None)
        d = importlib.import_module("dispatch_sycophancy_507")

        # Redirect SLAB_ROOT to a clean temp dir with NO analyze_summary_72b.json.
        monkeypatch.setattr(d, "SLAB_ROOT", tmp_path)
        with pytest.raises(RuntimeError, match=r"analyze_summary_72b|72B DV"):
            d._predictor_env_overrides(require_dv_72b=True)

    def test_predictor_env_overrides_allows_dv_72b_missing_when_require_false(
        self, tmp_path, monkeypatch
    ):
        """Phase 3 (predictor extraction) runs before Phase 3.5 — must NOT raise."""
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        sys.modules.pop("dispatch_sycophancy_507", None)
        d = importlib.import_module("dispatch_sycophancy_507")

        monkeypatch.setattr(d, "SLAB_ROOT", tmp_path)
        overrides = d._predictor_env_overrides(require_dv_72b=False)
        # No raise; PREDICTOR_DV_ANALYZE_SUMMARY simply isn't set.
        assert "PREDICTOR_DV_ANALYZE_SUMMARY" not in overrides

    def test_predictor_env_overrides_sets_dv_when_file_exists(self, tmp_path, monkeypatch):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        sys.modules.pop("dispatch_sycophancy_507", None)
        d = importlib.import_module("dispatch_sycophancy_507")

        monkeypatch.setattr(d, "SLAB_ROOT", tmp_path)
        (tmp_path / "analyze_summary_72b.json").write_text("{}")
        overrides = d._predictor_env_overrides(require_dv_72b=True)
        assert overrides.get("PREDICTOR_DV_ANALYZE_SUMMARY") == str(
            tmp_path / "analyze_summary_72b.json"
        )

    def test_phase3_5_analyze_72b_writes_per_panel_delta(self, tmp_path, monkeypatch):
        """End-to-end: feed fixture rates -> get analyze_summary_72b.json with delta computed."""
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        sys.modules.pop("dispatch_sycophancy_507", None)
        d = importlib.import_module("dispatch_sycophancy_507")

        slab = tmp_path / "72b"
        monkeypatch.setattr(d, "SLAB_ROOT", slab)
        slab.mkdir(parents=True)
        # base_panel_rates.json — dispatcher accepts {"panel_rates": {p: float}}
        # or {"panel_rates": {p: {"agree_rate": float}}}.
        (slab / "base_panel_rates.json").write_text(
            json.dumps({"panel_rates": {"software_engineer": 0.20, "comedian": 0.15}})
        )
        # Per-source per-panel rates (judge.py shape).
        src_dir = slab / "software_engineer" / "seed_42"
        src_dir.mkdir(parents=True)
        (src_dir / "per_panel_rates_software_engineer.json").write_text(
            json.dumps(
                {
                    "per_panel_rate": {
                        "software_engineer": 0.95,
                        "comedian": 0.40,
                    }
                }
            )
        )
        out_path = d.phase3_5_analyze_72b(seed=42, sources=["software_engineer"])
        assert out_path.exists(), "phase3_5_analyze_72b should write analyze_summary_72b.json"
        payload = json.loads(out_path.read_text())
        per_source = payload["per_source"]["software_engineer"]
        # Delta = trained_rate - base_rate.
        assert per_source["per_panel_delta"]["software_engineer"] == pytest.approx(0.75)
        assert per_source["per_panel_delta"]["comedian"] == pytest.approx(0.25)

    def test_phase3_5_analyze_72b_raises_on_missing_base_rates(self, tmp_path, monkeypatch):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        sys.modules.pop("dispatch_sycophancy_507", None)
        d = importlib.import_module("dispatch_sycophancy_507")
        slab = tmp_path / "72b"
        monkeypatch.setattr(d, "SLAB_ROOT", slab)
        slab.mkdir(parents=True)
        # No base_panel_rates.json at all.
        with pytest.raises(FileNotFoundError, match=r"base_panel_rates\.json"):
            d.phase3_5_analyze_72b(seed=42, sources=["software_engineer"])


# ── C3 — Phase 3 dispatcher invocation does NOT pass --R ─────────────────


class TestC3_Phase3NoRflag:
    """Critical 3: phase3_sequence_js_kl argparse has no --R flag; dispatcher
    must infer R from on-disk Phase 1 outputs."""

    def test_phase3_argparse_has_no_R_flag(self):
        """phase3_sequence_js_kl.main argparse must NOT accept --R."""
        src = (
            REPO_ROOT
            / "src"
            / "explore_persona_space"
            / "experiments"
            / "predictor_jsdiv_470"
            / "phase3_sequence_js_kl.py"
        ).read_text()
        # No --R in the argparse main(). It's allowed in comments / docstrings.
        # We restrict to add_argument lines containing "--R".
        offending = [
            line
            for line in src.splitlines()
            if "add_argument" in line and re.search(r'"--R[",\s]', line)
        ]
        assert not offending, (
            f"phase3_sequence_js_kl.py argparse should not register --R "
            f"(R is inferred from Phase 1 outputs). Found: {offending}"
        )

    def test_dispatcher_phase3_invocation_does_not_include_R(self):
        """The dispatcher's phase3.3 RB JS+KL invocation must not include --R."""
        src = (REPO_ROOT / "scripts" / "dispatch_sycophancy_507.py").read_text()
        # Find the block that invokes phase3_sequence_js_kl.
        m = re.search(
            r"explore_persona_space\.experiments\.predictor_jsdiv_470\.phase3_sequence_js_kl"
            r".*?(?=label=|env_extra=)",
            src,
            re.DOTALL,
        )
        assert m is not None, "Dispatcher does not invoke phase3_sequence_js_kl"
        block = m.group(0)
        assert "*r_arg" not in block, (
            "Dispatcher must not expand --R into the phase3_sequence_js_kl "
            "subprocess invocation (argparse rejects it)."
        )
        assert "--R" not in block, "Same constraint via literal --R."


# ── C4 — Phase 3 device_map override ─────────────────────────────────────


class TestC4_Phase3DeviceMap:
    """Critical 4: Phase 3 must support multi-GPU sharding for 72B."""

    def test_phase3_argparse_has_device_map(self):
        src = (
            REPO_ROOT
            / "src"
            / "explore_persona_space"
            / "experiments"
            / "predictor_jsdiv_470"
            / "phase3_sequence_js_kl.py"
        ).read_text()
        assert "--device-map" in src, "Phase 3 must expose --device-map CLI"
        assert "PREDICTOR_DEVICE_MAP" in src, (
            "Phase 3 must honor PREDICTOR_DEVICE_MAP env var for dispatcher threading"
        )

    def test_dispatcher_threads_predictor_device_map_auto(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        sys.modules.pop("dispatch_sycophancy_507", None)
        d = importlib.import_module("dispatch_sycophancy_507")
        overrides = d._predictor_env_overrides(require_dv_72b=False)
        assert overrides.get("PREDICTOR_DEVICE_MAP") == "auto", (
            "Dispatcher must thread PREDICTOR_DEVICE_MAP=auto so Phase 3 "
            "shards the 72B model across all visible GPUs."
        )


# ── C5 — DEFAULT_LAYERS env-parametrize ──────────────────────────────────


class TestC5_DefaultLayersEnv:
    """Critical 5: DEFAULT_LAYERS must follow PREDICTOR_LAYERS env."""

    def test_default_layers_unset_falls_back_to_7b(self):
        mod = _reload_common(env_override={"PREDICTOR_LAYERS": None})
        assert mod.DEFAULT_LAYERS == (7, 14, 21, 27), (
            "DEFAULT_LAYERS must fall back to (7,14,21,27) when PREDICTOR_LAYERS unset"
        )

    def test_default_layers_env_set_to_72b(self):
        mod = _reload_common(env_override={"PREDICTOR_LAYERS": "21,40,57,70"})
        assert mod.DEFAULT_LAYERS == (21, 40, 57, 70), (
            "DEFAULT_LAYERS must follow PREDICTOR_LAYERS when set; "
            "otherwise Phase 4 file-NotFounds on layer_7.json under a 72B run."
        )

    def test_default_layers_env_malformed_raises(self):
        with pytest.raises(RuntimeError, match="PREDICTOR_LAYERS"):
            _reload_common(env_override={"PREDICTOR_LAYERS": "21,oops,57"})

    def test_default_layers_env_empty_raises(self):
        with pytest.raises(RuntimeError, match="PREDICTOR_LAYERS"):
            _reload_common(env_override={"PREDICTOR_LAYERS": ""})

    def test_dispatcher_threads_predictor_layers_for_72b(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        sys.modules.pop("dispatch_sycophancy_507", None)
        d = importlib.import_module("dispatch_sycophancy_507")
        overrides = d._predictor_env_overrides(require_dv_72b=False)
        assert overrides.get("PREDICTOR_LAYERS") == "21,40,57,70", (
            "Dispatcher must thread the 72B depth-equivalent layer set "
            "{21,40,57,70} via PREDICTOR_LAYERS so Phase 4 reads them."
        )


# ── C7 — production world_size gate ──────────────────────────────────────


class TestC7_ProductionWorldSize:
    """Major 7: production world_size must be {4, 8} only."""

    def test_validate_production_accepts_4_and_8(self):
        from explore_persona_space.experiments.sycophancy_scale_507 import (
            validate_production_world_size,
        )

        validate_production_world_size(4)
        validate_production_world_size(8)

    def test_validate_production_rejects_2_and_16(self):
        from explore_persona_space.experiments.sycophancy_scale_507 import (
            validate_production_world_size,
        )

        for ws in (2, 3, 5, 16, 32):
            with pytest.raises(ValueError, match="not a supported"):
                validate_production_world_size(ws)

    def test_validate_production_rejects_1_without_debug_opt_in(self):
        from explore_persona_space.experiments.sycophancy_scale_507 import (
            validate_production_world_size,
        )

        with pytest.raises(ValueError, match="not a supported"):
            validate_production_world_size(1)
        with pytest.raises(ValueError, match="not a supported"):
            validate_production_world_size(1, allow_debug=False)

    def test_validate_production_accepts_1_with_debug_opt_in(self):
        from explore_persona_space.experiments.sycophancy_scale_507 import (
            validate_production_world_size,
        )

        validate_production_world_size(1, allow_debug=True)

    def test_validate_production_rejects_2_even_with_debug(self):
        """allow_debug only widens to {1}; 2 / 16 remain rejected so an operator
        can't accidentally smoke at world_size=2 and pretend it's production."""
        from explore_persona_space.experiments.sycophancy_scale_507 import (
            validate_production_world_size,
        )

        for ws in (2, 16):
            with pytest.raises(ValueError, match="not a supported"):
                validate_production_world_size(ws, allow_debug=True)

    def test_compute_grad_accum_still_general(self):
        """compute_grad_accum is a math helper — kept general so CPU tests
        and the debug path still call it with ws in {1,2,4,8,16}."""
        from explore_persona_space.experiments.sycophancy_scale_507 import compute_grad_accum

        assert compute_grad_accum(1) == 16
        assert compute_grad_accum(2) == 8
        assert compute_grad_accum(4) == 4
        assert compute_grad_accum(8) == 2
        assert compute_grad_accum(16) == 1
