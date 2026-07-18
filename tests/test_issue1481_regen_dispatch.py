"""#1481 contingent-regen dispatch registry invariants (plan §4.6 gate P1).

Pins the ALWAYS-EXPLICIT contract: the 12 regen grid slots (the reused con
seed-42 arms) NEVER enter a default (no --runs) dispatch cohort — full OR
smoke — and resolve only through an explicit --runs subset, at the exact
matched grid recipe (same behavior+context-scoped con mix as the fresh con
arms, seed 42, con round spec verbatim). A regression here silently adds
12 GPU-expensive ladder rebuilds to every Phase-A group dispatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1090_fu4 as fu4  # noqa: E402
import issue1481_cells as cells  # noqa: E402
import issue1481_worker as w  # noqa: E402

FLAGGED = "imp-bare-con-lr1e4-s42"  # the live P1 band-exit arm (re-read 0.58 < 0.60)


@pytest.fixture(autouse=True)
def _registered():
    cells.register_i1481_rounds()


def test_regen_rounds_registered_and_grouped():
    assert cells.REGEN_ROUND_NAMES == ("i1481impregen", "i1481sycregen")
    for name, n in (("i1481impregen", 9), ("i1481sycregen", 3)):
        assert name in fu4.ROUNDS
        assert len(fu4.ROUNDS[name].runs) == n
        assert fu4.ROUNDS[name].smoke_default_run == ""
    assert cells.DISPATCH_ROUNDS["impolite"][-1] == "i1481impregen"
    assert cells.DISPATCH_ROUNDS["sycophancy"][-1] == "i1481sycregen"
    assert cells.DISPATCH_ROUNDS["casual-s137"] == ("i1481cas", "i1481caspo")
    # Regen ids ARE the reused-arm grid slots, disjoint from every fresh id.
    regen_ids = {r.run_id for rn in cells.REGEN_ROUND_NAMES for r in fu4.ROUNDS[rn].runs}
    assert regen_ids == set(cells.REUSED_CON_ARM_BY_ID)
    fresh_ids = {r.run_id for rs in cells.RUNS_BY_ROUND.values() for r in rs}
    assert not regen_ids & fresh_ids


def test_default_cohorts_exclude_regen_full_and_smoke():
    for rname in cells.REGEN_ROUND_NAMES:
        for smoke in (False, True):
            for seed in cells.SEEDS:
                assert w._cohort_run_ids(rname, seed, None, smoke) == []
    # ...and the existing rounds' default cohorts are unchanged (byte-identical
    # Phase-A dispatch): i1481imp keeps 3 fresh s42 (icl) + 12 s137 cells.
    assert len(w._cohort_run_ids("i1481imp", 42, None, False)) == 3
    assert len(w._cohort_run_ids("i1481imp", 137, None, False)) == 12


def test_explicit_runs_resolve_only_the_named_arm():
    assert w._cohort_run_ids("i1481impregen", 42, FLAGGED, False) == [FLAGGED]
    assert w._cohort_run_ids("i1481impregen", 137, FLAGGED, False) == []
    for rid in ("syc-pers-con-lr1e5-s42", "syc-pers-con-lr3e5-s42", "syc-pers-con-lr1e4-s42"):
        assert w._cohort_run_ids("i1481sycregen", 42, rid, False) == [rid]


def test_regen_recipe_matches_fresh_con_sibling():
    """The regen cell is the matched grid recipe: identical mix prefix/layout,
    context, lr, fu3 base read as its fresh -s137 con sibling; round spec
    carries the con recipe fields verbatim."""
    for arm in cells.REUSED_CON_ARMS:
        beh_key = arm.arm_id.split("-")[0]
        run = {r.run_id: r for r in fu4.ROUNDS[cells.regen_round_name(beh_key)].runs}[arm.arm_id]
        sib = {r.run_id: r for r in fu4.ROUNDS[cells.round_name(beh_key, "con")].runs}[
            arm.arm_id.replace("-s42", "-s137")
        ]
        assert (run.mix_hub_prefix, run.mix_layout) == (sib.mix_hub_prefix, sib.mix_layout)
        assert (run.context_id, run.lr, run.fu3_base_eval) == (
            sib.context_id,
            sib.lr,
            sib.fu3_base_eval,
        )
        assert (run.lora_r, run.lora_alpha) == (32, 64)
        assert run.run_name == f"issue1481_{arm.arm_id}_seed42"
        assert cells.seed_for_run_id(run.run_id) == 42
    for beh_key in ("imp", "syc"):
        rg = fu4.ROUNDS[cells.regen_round_name(beh_key)]
        cn = fu4.ROUNDS[cells.round_name(beh_key, "con")]
        for f in (
            "mix_composition",
            "train_max_steps",
            "judge_fn",
            "margin_pools_fn",
            "max_lora_rank",
            "upload_all_rungs",
            "data_prefix",
            "adapter_prefix",
            "issue",
            "worker_script",
        ):
            assert getattr(rg, f) == getattr(cn, f), (beh_key, f)
        # Own deliverable names — never clobbers the fresh grid's files.
        assert rg.manifest_name != cn.manifest_name
        assert rg.ladders_name != cn.ladders_name
        assert rg.raw_prefix != cn.raw_prefix


def test_unknown_run_refuses_through_dispatch_group():
    with pytest.raises(SystemExit, match="unknown runs"):
        w._run_dispatch_group(
            ["--full", "--dispatch", "impolite", "--runs", "imp-bare-con-lr9e9-s42"], "impolite"
        )


def test_regen_only_dispatch_skips_p1_rereads(monkeypatch):
    """A regen-only --runs runs stage+dispatch for the regen cohort ONLY and
    SKIPS the group's P1 rereads (the committed re-read artifacts are the
    trigger evidence); a fresh-run --runs still runs the rereads."""
    calls: list[tuple[str, str, str]] = []

    def fake_delegate(argv: list[str]) -> int:  # signature mirrors _delegate_fu4
        calls.append(
            (
                w._argv_get(argv, "--phase") or "",
                w._argv_get(argv, "--round") or "",
                w._argv_get(argv, "--runs") or "",
            )
        )
        return 0

    rereads: list[str] = []

    def fake_reread(cfg, args) -> int:  # signature mirrors phase_reread
        rereads.append(args.arms or "")
        return 0

    monkeypatch.setattr(w, "_delegate_fu4", fake_delegate)
    monkeypatch.setattr(w, "phase_reread", fake_reread)

    rc = w._run_dispatch_group(
        ["--full", "--dispatch", "impolite", "--runs", FLAGGED, "--seeds", "42,137"], "impolite"
    )
    assert rc == 0
    assert calls == [
        ("stage", "i1481impregen", FLAGGED),
        ("dispatch", "i1481impregen", FLAGGED),
    ]
    assert rereads == []

    calls.clear()
    rc = w._run_dispatch_group(
        ["--full", "--dispatch", "impolite", "--runs", "imp-icl-con-lr1e5-s42", "--seeds", "42"],
        "impolite",
    )
    assert rc == 0
    assert calls == [
        ("stage", "i1481imp", "imp-icl-con-lr1e5-s42"),
        ("dispatch", "i1481imp", "imp-icl-con-lr1e5-s42"),
    ]
    assert len(rereads) == 1  # fresh-run subset keeps the P1 rereads


def test_merge_manifests_demands_only_registered_seeds(tmp_path):
    """Regen rounds are s42-only (and casual rounds s137-only): merge derives
    its seed demand from the round registry instead of a fixed cells.SEEDS."""
    manifest = {
        "issue": 1481,
        "round": "conpos-grid-impolite-con-regen",
        "runs": [{"run_id": FLAGGED, "cell_key": "imp-bare"}],
    }
    (tmp_path / "cell_manifest_i1481impregen_s42.json").write_text(json.dumps(manifest))
    args = w._own_parser().parse_args(
        [
            "--smoke",
            "--phase",
            "merge-manifests",
            "--round",
            "i1481impregen",
            "--out-root",
            str(tmp_path),
        ]
    )
    rc = w.phase_merge_manifests(w.worker_config(args), args)
    assert rc == 0
    merged = json.loads((tmp_path / "cell_manifest_i1481impregen.json").read_text())
    assert merged["merged_seeds"] == [42]
    assert [r["run_id"] for r in merged["runs"]] == [FLAGGED]


def test_merge_manifests_fails_loud_on_missing_registered_seed(tmp_path):
    args = w._own_parser().parse_args(
        [
            "--smoke",
            "--phase",
            "merge-manifests",
            "--round",
            "i1481impregen",
            "--out-root",
            str(tmp_path),
        ]
    )
    with pytest.raises(FileNotFoundError, match="missing cohort manifest"):
        w.phase_merge_manifests(w.worker_config(args), args)
