"""Smoke test: every configs/tulu/*.yaml parses cleanly through launch_stage's
allowlist filter against the pinned open-instruct submodule.

Motivation: open-instruct's HfArgumentParser strictly rejects unknown flags. A
config with a stale flag (e.g. `use_flash_attn` post PR #1563) will crash
training at the very first step. This test catches such drift before GPU time
is spent.

Runs in <1s, no GPU / training needed.
"""

from __future__ import annotations

# Import from scripts/ (not a package); add to path.
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from launch_stage import (  # noqa: E402
    OI_PKG_DIR,
    OI_SCRIPT_DATACLASSES,
    get_open_instruct_arg_allowlist,
)

TULU_DIR = REPO_ROOT / "configs" / "tulu"


def _tulu_configs() -> list[Path]:
    return sorted(TULU_DIR.glob("*.yaml"))


@pytest.mark.skipif(
    not OI_PKG_DIR.exists(),
    reason="open-instruct submodule not materialized (fine for lint-only env)",
)
@pytest.mark.parametrize("cfg_path", _tulu_configs(), ids=lambda p: p.name)
def test_tulu_config_has_no_stale_flags(cfg_path: Path) -> None:
    """Each Tulu config's arg keys must all be valid fields on the target
    script's open-instruct dataclasses.

    Uses the same AST-based allowlist logic that launch_stage.py uses at
    runtime, so a PASS here guarantees runtime won't drop keys.
    """
    cfg = yaml.safe_load(cfg_path.read_text())
    assert "open_instruct" in cfg, f"{cfg_path.name} missing 'open_instruct' key"

    oi = cfg["open_instruct"]
    script_rel = oi["script"]
    assert script_rel in OI_SCRIPT_DATACLASSES, (
        f"{cfg_path.name}: script {script_rel!r} not in OI_SCRIPT_DATACLASSES. "
        f"Add it to scripts/launch_stage.py so the allowlist can validate it."
    )

    allowlist = get_open_instruct_arg_allowlist(script_rel)
    assert allowlist is not None, f"allowlist resolution failed for {script_rel}"
    assert len(allowlist) > 20, (
        f"allowlist for {script_rel} only has {len(allowlist)} fields — "
        "something is wrong with AST parsing of the open-instruct submodule."
    )

    args = oi.get("args", {})
    unknown = sorted(k for k in args if k not in allowlist)
    assert not unknown, (
        f"{cfg_path.name} has {len(unknown)} stale flag(s) not in "
        f"{script_rel}'s dataclass fields: {unknown}. Remove from YAML or "
        f"update OI_SCRIPT_DATACLASSES if upstream renamed the dataclass."
    )


def test_allowlist_contains_known_valid_flags() -> None:
    """Guard against the AST parser silently returning an empty/wrong set.

    We enumerate a few fields we *know* exist on the current open-instruct and
    assert they're recognized. If this fails, the AST walker broke (e.g.
    upstream switched to a different dataclass base structure).
    """
    sft = get_open_instruct_arg_allowlist("open_instruct/finetune.py")
    if sft is None:
        pytest.skip("submodule not available")
    # Stable core fields on SFT FlatArguments / TokenizerConfig
    for key in [
        "learning_rate",
        "num_train_epochs",
        "dataset_mixer_list",
        "use_liger_kernel",
        "packing",
        "tokenizer_name",
        "chat_template_name",
    ]:
        assert key in sft, f"SFT allowlist missing expected field {key!r}"

    dpo = get_open_instruct_arg_allowlist("open_instruct/dpo_tune_cache.py")
    if dpo is None:
        pytest.skip("submodule not available")
    for key in [
        "learning_rate",
        "num_epochs",
        "mixer_list",
        "use_liger_kernel",
        "packing",
        "beta",
        "loss_type",
    ]:
        assert key in dpo, f"DPO allowlist missing expected field {key!r}"


def test_stale_flags_are_detected_when_injected() -> None:
    """The filter must drop/raise on made-up flags — guards against the filter
    becoming a no-op after a refactor."""
    from launch_stage import filter_args_by_allowlist

    allowlist = {"learning_rate", "num_train_epochs"}
    # Non-strict: warns + drops
    out = filter_args_by_allowlist(
        {"learning_rate": 1e-5, "bogus_flag": True},
        allowlist,
        "test_script.py",
    )
    assert "bogus_flag" not in out
    assert out["learning_rate"] == 1e-5

    # Strict: raises
    import os

    os.environ["EPS_STRICT_TULU_ARGS"] = "1"
    try:
        with pytest.raises(ValueError, match="bogus_flag"):
            filter_args_by_allowlist(
                {"learning_rate": 1e-5, "bogus_flag": True},
                allowlist,
                "test_script.py",
            )
    finally:
        del os.environ["EPS_STRICT_TULU_ARGS"]
