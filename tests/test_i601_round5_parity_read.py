# em-dash + Qwen marker token " ※" are intentional
"""Task #601 round 5 — adapter-application parity-read fixes (CPU-only).

Covers the three required round-5 fixes for the Phase-0a HALT gate failure
(six different adapters re-reading 10.350 ± 0.002 — the rsLoRA
over-application collapse ceiling, see ``neg_setpoint_601.PARITY_READ_*``):

1. mapping construction: cell list → worker plan → adapter paths, slug∈path
   per worker (``onpolicy_worker_plan``);
2. parity staging: ``stage_parity_read_adapter`` patches ``use_rslora`` to
   False, symlinks weights, records sha256 provenance, and fail-louds on a
   slug↔path mismatch;
3. the structural identical-reread tripwire: ``onpolicy_crosscheck`` raises
   ``IdenticalRereadAlarm`` (never a quiet ``pass=false``) when >=3 distinct
   adapters re-read within 0.01 nat of one another.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from explore_persona_space.experiments.neg_setpoint_601.artifacts import (
    stage_parity_read_adapter,
)
from explore_persona_space.experiments.neg_setpoint_601.phase0_lib import (
    IdenticalRereadAlarm,
    find_identical_reread_groups,
    onpolicy_crosscheck,
    onpolicy_worker_plan,
)

KEYS = [
    f"{cell}_seed{seed}"
    for cell in ("c472_noneg", "c472_negex_100", "c472_anchor", "c472_negex_400")
    for seed in (42, 137)
]


# ── 1. mapping construction ──────────────────────────────────────────────────


def test_onpolicy_worker_plan_slug_in_every_path():
    plan = onpolicy_worker_plan(
        KEYS,
        "eval_results/issue_601/phase0",
        "/workspace/models/issue_601_parent",
        "data/issue_601",
        "/workspace/logs",
    )
    assert [row["key"] for row in plan] == KEYS
    for row in plan:
        # The round-5 brief's core check: the adapter path each worker will
        # apply carries that worker's OWN cell_seed slug.
        assert row["key"] in row["adapter_path"], row
        assert row["key"] in row["out_dir"], row
        assert row["key"] in row["log_path"], row
        # The CLI args parsed back out of the cmd match the key.
        cmd = row["cmd"]
        assert cmd[cmd.index("--cell") + 1] == row["cell"]
        assert int(cmd[cmd.index("--seed") + 1]) == row["seed"]
        assert row["key"] in cmd[cmd.index("--checkpoint-index") + 1]
        assert row["key"] in cmd[cmd.index("--out-path") + 1]
        cell, seed_s = row["key"].rsplit("_seed", 1)
        assert row["cell"] == cell and row["seed"] == int(seed_s)


def test_onpolicy_worker_plan_rows_are_distinct():
    plan = onpolicy_worker_plan(KEYS, "p0", "roots", "d", "l")
    for field in ("adapter_path", "out_dir", "idx_path", "log_path"):
        vals = [row[field] for row in plan]
        assert len(set(vals)) == len(vals), f"duplicate {field} across workers"


# ── 2. parity staging ────────────────────────────────────────────────────────


def _make_fake_adapter(root: Path, name: str, *, use_rslora: bool = True) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 32,
                "lora_alpha": 64,
                "use_rslora": use_rslora,
                "target_modules": ["q_proj"],
            }
        )
    )
    (d / "adapter_model.safetensors").write_bytes(b"fake-weights-" + name.encode())
    return d


def test_stage_parity_read_adapter_patches_config_and_records_provenance(tmp_path):
    src = _make_fake_adapter(tmp_path / "src", "c472_anchor_seed137")
    staged, prov = stage_parity_read_adapter(
        src, tmp_path / "staged", expect_slug="c472_anchor_seed137"
    )
    cfg = json.loads((staged / "adapter_config.json").read_text())
    assert cfg["use_rslora"] is False  # the parity patch
    assert cfg["lora_alpha"] == 64 and cfg["r"] == 32  # everything else untouched
    # Weights are a symlink to the ORIGINAL bytes (no 323 MB copy).
    w = staged / "adapter_model.safetensors"
    assert w.is_symlink() and w.read_bytes() == (src / "adapter_model.safetensors").read_bytes()
    # Source config was NOT mutated.
    assert json.loads((src / "adapter_config.json").read_text())["use_rslora"] is True
    # Provenance: sha256 of the actual weights + scaling record.
    expect_sha = hashlib.sha256((src / "adapter_model.safetensors").read_bytes()).hexdigest()
    assert prov["adapter_sha256"] == expect_sha
    assert prov["use_rslora_original"] is True
    assert prov["use_rslora_applied"] is False
    assert prov["source_adapter_path"] == str(src)
    assert "lora_alpha/r" in prov["effective_scaling_applied"]


def test_stage_parity_read_adapter_is_idempotent(tmp_path):
    src = _make_fake_adapter(tmp_path / "src", "c472_noneg_seed42")
    s1, _ = stage_parity_read_adapter(src, tmp_path / "staged", expect_slug="c472_noneg_seed42")
    s2, prov2 = stage_parity_read_adapter(src, tmp_path / "staged", expect_slug="c472_noneg_seed42")
    assert s1 == s2
    assert json.loads((s2 / "adapter_config.json").read_text())["use_rslora"] is False
    assert prov2["use_rslora_original"] is True


def test_stage_parity_read_adapter_slug_mismatch_raises(tmp_path):
    src = _make_fake_adapter(tmp_path / "src", "c472_anchor_seed137")
    with pytest.raises(ValueError, match="mapping assert FAILED"):
        stage_parity_read_adapter(src, tmp_path / "staged", expect_slug="c472_negex_400_seed42")


# ── 3. identical-reread tripwire ─────────────────────────────────────────────


def _stats(dg: float) -> dict:
    return {"delta_g": dg, "emission_p": 0.0, "r_collapsed": False}


def test_identical_reread_alarm_fires_on_round4_pattern():
    # The realized round-4 failure: six adapters at 10.350 ± 0.002.
    reread = {
        "c472_anchor_seed137": _stats(10.350),
        "c472_anchor_seed42": _stats(10.351),
        "c472_negex_100_seed137": _stats(10.350),
        "c472_negex_100_seed42": _stats(10.348),
        "c472_negex_400_seed137": _stats(10.351),
        "c472_negex_400_seed42": _stats(10.351),
        "c472_noneg_seed137": _stats(21.370),
        "c472_noneg_seed42": _stats(19.541),
    }
    committed = {
        "c472_anchor_seed137": 13.071,
        "c472_anchor_seed42": 13.946,
        "c472_negex_100_seed137": 8.447,
        "c472_negex_100_seed42": 8.648,
        "c472_negex_400_seed137": 20.343,
        "c472_negex_400_seed42": 19.653,
        "c472_noneg_seed137": 2.121,
        "c472_noneg_seed42": 1.971,
    }
    with pytest.raises(IdenticalRereadAlarm, match="mapping scramble suspected") as exc:
        onpolicy_crosscheck(reread, committed)
    (group,) = exc.value.diag["identical_groups"]
    assert len(group) == 6 and all("noneg" not in k for k in group)


def test_crosscheck_passes_on_differentiated_rereads():
    committed = {
        "c472_anchor_seed137": 13.071,
        "c472_noneg_seed137": 2.121,
        "c472_negex_400_seed137": 20.343,
    }
    reread = {k: _stats(v + 0.4) for k, v in committed.items()}
    out = onpolicy_crosscheck(reread, committed)
    assert out["pass"] is True
    assert all(rec["within_tol"] for rec in out["per_adapter"].values())
    # Regime-validity flags are recorded per adapter (round 5).
    assert all(rec["reread_r_collapsed"] is False for rec in out["per_adapter"].values())


def test_crosscheck_quiet_fail_still_allowed_when_rereads_differ():
    # Distinct re-reads that miss tol → ordinary pass=false, NOT the alarm.
    committed = {"a_seed1": 5.0, "b_seed1": 9.0, "c_seed1": 13.0}
    reread = {"a_seed1": _stats(7.5), "b_seed1": _stats(12.0), "c_seed1": _stats(16.0)}
    out = onpolicy_crosscheck(reread, committed)
    assert out["pass"] is False


def test_find_identical_reread_groups_boundaries():
    # Exactly 2 within tol → no group (min group is 3).
    assert find_identical_reread_groups({"a": 1.000, "b": 1.005, "c": 2.0}) == []
    # Chain of 3 within tol → one group.
    assert find_identical_reread_groups({"a": 1.000, "b": 1.008, "c": 1.012, "d": 5.0}) == [
        ["a", "b", "c"]
    ]
