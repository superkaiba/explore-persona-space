"""Mechanizes the plan v8 install-pilot SCOPE exemption (plan §4.3/§4.5/§4.6/§7/
§9): the fresh Phase-0.5 install-pilot trains + gates ONLY the two NEW arms
(``loraOP_lr5e6``, ``cmftCN_lr5e6``); ``cmftOP_lr5e6`` is EXEMPT (it gate-PASSed
the v5 install-pilot at the matched LR — gate-validation reuse) and MUST NOT be
re-piloted.

Why this test exists: the v8 round retagged ``V4_ARMS`` to the 3-arm 5e-6 set and
introduced the ``cmftOP_lr5e6`` gate-validation-reuse exemption, but the inherited
``p0_5_pilot_gate`` looped ``ctx.arms`` (all 3 production arms) for both the pilot
train AND the gate — re-piloting the exempted arm, wasting GPU and risking a
spurious step-4-collapse HALT on an arm the plan said to skip entirely (reconciler
``epm:review-reconcile v3`` round 3, concern ``issue642-v8-pilot-scope`` BLOCKER).
This test pins the exemption so a future regression is caught at test time, not at
GPU-launch time.

Two layers of coverage:
  1. Registry-level: the derived ``V4_PILOT_ARMS`` / ``V4_PILOT_EXEMPT_ARMS``
     constants are exactly the 2 new arms / the 1 exempt arm.
  2. Dispatcher-level: ``p0_5_pilot_gate`` narrows ``ctx.arms`` to the 2 new arms
     for the train + stage-A + gate calls, restores the full 3-arm set afterwards,
     and writes a ``pilot_gate.json`` carrying ``cmftOP_lr5e6`` as an
     ``exempted_arms`` entry (reused-via-v5-gate-validation), never as a piloted
     ``arms`` entry.

Pure CPU — no model, no API, no GPU. The train/stage-A/gate callees are
monkeypatched to capture the live ``ctx.arms`` at each call boundary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" / "issue_642"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import i642_common as c  # noqa: E402
import i642_dispatch as d  # noqa: E402

NEW_ARMS = frozenset({"loraOP_lr5e6", "cmftCN_lr5e6"})
EXEMPT_ARM = "cmftOP_lr5e6"


# ---------------------------------------------------------------------------
# Layer 1 — registry-level constants
# ---------------------------------------------------------------------------


def test_pilot_arms_are_exactly_the_two_new_arms() -> None:
    """V4_PILOT_ARMS ⊂ {loraOP_lr5e6, cmftCN_lr5e6} and excludes cmftOP_lr5e6."""
    assert set(c.V4_PILOT_ARMS) <= NEW_ARMS, c.V4_PILOT_ARMS
    assert set(c.V4_PILOT_ARMS) == NEW_ARMS, c.V4_PILOT_ARMS
    assert EXEMPT_ARM not in c.V4_PILOT_ARMS, c.V4_PILOT_ARMS


def test_exempt_arm_is_cmftOP_lr5e6() -> None:
    """The single exempt arm is cmftOP_lr5e6 (the v5-gate-PASSed dense pole)."""
    assert tuple(c.V4_PILOT_EXEMPT_ARMS) == (EXEMPT_ARM,)


def test_pilot_arms_partition_v4_arms() -> None:
    """Every production arm is either piloted or exempt — no arm falls through,
    and no arm is double-counted (a partition of V4_ARMS)."""
    piloted = set(c.V4_PILOT_ARMS)
    exempt = set(c.V4_PILOT_EXEMPT_ARMS)
    assert piloted.isdisjoint(exempt)
    assert piloted | exempt == set(c.V4_ARMS)


# ---------------------------------------------------------------------------
# Layer 2 — dispatcher-level p0_5_pilot_gate scoping
# ---------------------------------------------------------------------------


def _make_stub_ctx(tmp_path: Path) -> SimpleNamespace:
    """A lightweight Ctx stub exercising p0_5_pilot_gate without a GPU.

    Carries the FULL 3-arm production set so the test proves the gate NARROWS to
    the 2 new arms (not that it was already narrowed upstream)."""
    bdir = tmp_path / "sycophancy"
    (bdir / "stage_a_pilot").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        v4=True,
        install_pilot=False,
        smoke=False,
        dry_run=True,
        skip_upload=True,
        arms=tuple(c.V4_ARMS),  # ("loraOP_lr5e6", "cmftOP_lr5e6", "cmftCN_lr5e6")
        lora_grid=c.V4_PILOT_GRID,
        ft_grid=c.V4_PILOT_GRID,
        cmft_grid=c.V4_PILOT_GRID,
        ft_max_steps=0,
        experiment_name=c.V4_HF_EXPERIMENT_NAME,
        bdir=lambda _b: bdir,
        trajectory_path=lambda _b: bdir / "stage_a_pilot" / "trajectory_sycophancy.json",
    )


def test_p0_5_pilot_gate_scopes_arms_to_new_arms_only(tmp_path, monkeypatch) -> None:
    """p0_5_pilot_gate trains + gates ONLY the 2 new arms; cmftOP_lr5e6 is never
    seen by the train or gate callees, the full 3-arm set is restored afterwards,
    and the verdict surfaces cmftOP_lr5e6 as an exempted arm (not a piloted arm)."""
    ctx = _make_stub_ctx(tmp_path)

    seen: dict[str, tuple[str, ...]] = {}

    def _fake_train(ctx_arg, behavior):
        seen["train"] = tuple(ctx_arg.arms)

    def _fake_stage_a(ctx_arg, behavior):
        seen["stage_a"] = tuple(ctx_arg.arms)
        # write a trajectory the gate would read (only the piloted arms)
        cells = {}
        for arm in ctx_arg.arms:
            for i, step in enumerate(c.V4_PILOT_GRID):
                cells[f"{arm}_step{step}"] = {
                    "arm": arm,
                    "step": step,
                    "s": 0.1 + 0.05 * i,  # monotone, below S_TARGET at step 4
                }
        ctx_arg.trajectory_path(behavior).write_text(json.dumps({"cells": cells}))

    def _fake_gate(ctx_arg, behavior, trajectory):
        seen["gate"] = tuple(ctx_arg.arms)
        # the real gate only ever sees the trajectory cells, which carry only the
        # piloted arms; return a minimal PASS verdict shaped like the real one
        return {
            "behavior": behavior,
            "gate": "install_pilot_gate",
            # arm records carry "reason" exactly as the real _evaluate_pilot_gate
            # does (the post-gate per-arm log loop reads v["reason"]).
            "arms": {arm: {"ok": True, "reason": f"{arm} stub PASS"} for arm in ctx_arg.arms},
            "gate_pass": True,
            "failing_arms": [],
        }

    monkeypatch.setattr(d, "_phase1_train_v4", _fake_train)
    monkeypatch.setattr(d, "phase2_stage_a", _fake_stage_a)
    monkeypatch.setattr(d, "_evaluate_pilot_gate", _fake_gate)
    monkeypatch.setattr(d, "_hub_upload_file", lambda *a, **k: None)
    monkeypatch.setattr(d, "_git_sha", lambda: "test-sha")

    d.p0_5_pilot_gate(ctx, "sycophancy")

    # the train / stage-A / gate callees each saw EXACTLY the 2 new arms
    for phase in ("train", "stage_a", "gate"):
        assert set(seen[phase]) == NEW_ARMS, f"{phase} saw {seen[phase]!r}, expected {NEW_ARMS}"
        assert EXEMPT_ARM not in seen[phase], f"{phase} re-piloted the exempt arm {EXEMPT_ARM}"

    # the full 3-arm production set is restored after the pilot (production Phase 1
    # trains all 3 arms — only the SHORT pilot is scoped)
    assert tuple(ctx.arms) == tuple(c.V4_ARMS)

    # the verdict on disk carries cmftOP_lr5e6 as an EXEMPTED arm, never a piloted
    # arm, with the gate-validation-reuse status
    gate_path = ctx.bdir("sycophancy") / "stage_a_pilot" / "pilot_gate.json"
    verdict = json.loads(gate_path.read_text())
    assert EXEMPT_ARM not in verdict["arms"], "exempt arm must NOT be a piloted 'arms' entry"
    assert set(verdict["arms"]) == NEW_ARMS
    assert EXEMPT_ARM in verdict["exempted_arms"], "exempt arm must be surfaced, not dropped"
    assert verdict["exempted_arms"][EXEMPT_ARM]["status"] == "reused-via-v5-gate-validation"
    assert verdict["gate_pass"] is True


# ---------------------------------------------------------------------------
# B3 (round-1 reconcile): v9 LoRA adapters land at adapters/issue_642/v9/...
# (the path the v9 Reproducibility Card declares), not the hardcoded v4/ prefix.
# ---------------------------------------------------------------------------


def _make_v9_upload_ctx(tmp_path: Path, selected_steps: list[int]) -> SimpleNamespace:
    """Stub Ctx for _phase5_upload_v4_adapters under a v9 run. skip_upload=True so
    no HF call fires; only the dest-path computation + cmft_uploaded_adapters
    population is exercised."""
    bdir = tmp_path / "refusal"
    bdir.mkdir(parents=True, exist_ok=True)
    sel_path = bdir / "selection.json"
    sel_path.write_text(json.dumps({"arms": {"loraRefOP": {"selected_steps": selected_steps}}}))
    return SimpleNamespace(
        v9=True,
        v4=True,  # v9 forces v4=True
        dry_run=False,
        skip_upload=True,
        upload_adapters=False,
        seed=42,
        arms=("loraRefOP",),
        experiment_name=c.V9_HF_EXPERIMENT_NAME,
        cmft_uploaded_adapters={},
        selection_path=lambda _b: sel_path,
        v4_arm_method=lambda _arm: "lora",
        ckpt_root=lambda _b, _arm: bdir / "ckpt",
    )


def test_v9_lora_adapters_land_at_v9_namespace(tmp_path, monkeypatch) -> None:
    """A v9 run uploads loraRefOP adapters to adapters/issue_642/v9/..., matching
    the v9 Reproducibility Card (NOT the hardcoded v4/ prefix — the B3 bug)."""
    monkeypatch.setattr(d, "_phase_log", lambda *a, **k: None)
    ctx = _make_v9_upload_ctx(tmp_path, [4, 22])

    d._phase5_upload_v4_adapters(ctx, "refusal")

    dests = ctx.cmft_uploaded_adapters["refusal:loraRefOP"]
    assert dests, "lora pole should record adapter dests"
    for dst in dests:
        assert dst.startswith("adapters/issue_642/v9/loraRefOP_villain_seed42/"), dst
    assert dests == [
        "adapters/issue_642/v9/loraRefOP_villain_seed42/step4",
        "adapters/issue_642/v9/loraRefOP_villain_seed42/step22",
    ], dests


def test_v4_lora_adapters_keep_v4_namespace(tmp_path, monkeypatch) -> None:
    """A v4-only run (ctx.v9 False) keeps the v4/ prefix — the namespace split is
    parameterized, not globally flipped."""
    monkeypatch.setattr(d, "_phase_log", lambda *a, **k: None)
    ctx = _make_v9_upload_ctx(tmp_path, [4])
    ctx.v9 = False  # v4-only run
    ctx.arms = ("loraRefOP",)

    d._phase5_upload_v4_adapters(ctx, "refusal")

    dests = ctx.cmft_uploaded_adapters["refusal:loraRefOP"]
    assert dests == ["adapters/issue_642/v4/loraRefOP_villain_seed42/step4"], dests


def test_pilot_exempt_record_shape() -> None:
    """The exempted-arm record names the arm, its method, the reuse status, and a
    pointer to the v5 pilot (never silently dropped — plan §7 / reconciler req)."""
    rec = d._pilot_exempt_record(EXEMPT_ARM)
    assert rec["arm"] == EXEMPT_ARM
    assert rec["method"] == "cmft"
    assert rec["status"] == "reused-via-v5-gate-validation"
    assert "reused_from_v5_pilot" in rec
    assert "reason" in rec and EXEMPT_ARM in rec["reason"]
