"""#1481 contingent-regen dispatch + analysis-integration invariants (plan §4.6).

Pins the ALWAYS-EXPLICIT contract: the 10 regen grid slots (the committed-
IN-BAND reused con seed-42 arms; the 2 committed closest-approach OUT-of-band
arms are OUTSIDE the regen trigger — never registered, refused at dispatch)
NEVER enter a default (no --runs) dispatch cohort — full OR smoke — and
resolve only through an explicit --runs subset, at the exact matched grid
recipe (same behavior+context-scoped con mix as the fresh con arms, seed 42,
con round spec verbatim). A regression here silently adds GPU-expensive
ladder rebuilds to every Phase-A group dispatch.

Also pins the ANALYSIS-side ladder-of-record routing (concern
`regen-ladder-analysis-integration`): a regen round's fresh ladder SUPERSEDES
the committed parent for its arm; a P1-flagged arm with NO regen record fails
loud; an unflagged arm still resolves from the committed parent.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1090_fu4 as fu4  # noqa: E402
import issue1481_analysis as ana  # noqa: E402
import issue1481_cells as cells  # noqa: E402
import issue1481_worker as w  # noqa: E402

FLAGGED = "imp-bare-con-lr1e4-s42"  # the live P1 band-exit arm (re-read 0.58 < 0.60)
FU5_REL = "eval_results/issue_1090/finish-impolite-bare-and-formatting-rank/fu5_ladders.json"


@pytest.fixture(autouse=True)
def _registered():
    cells.register_i1481_rounds()


def test_regen_rounds_registered_and_grouped():
    assert cells.REGEN_ROUND_NAMES == ("i1481impregen", "i1481sycregen")
    for name, n in (("i1481impregen", 7), ("i1481sycregen", 3)):
        assert name in fu4.ROUNDS
        assert len(fu4.ROUNDS[name].runs) == n
        assert fu4.ROUNDS[name].smoke_default_run == ""
    assert cells.DISPATCH_ROUNDS["impolite"][-1] == "i1481impregen"
    assert cells.DISPATCH_ROUNDS["sycophancy"][-1] == "i1481sycregen"
    assert cells.DISPATCH_ROUNDS["casual-s137"] == ("i1481cas", "i1481caspo")
    # Regen ids ARE the committed-IN-BAND reused-arm grid slots — the 2
    # committed closest-approach OUT-of-band arms are excluded (plan §4.6) —
    # disjoint from every fresh id.
    regen_ids = {r.run_id for rn in cells.REGEN_ROUND_NAMES for r in fu4.ROUNDS[rn].runs}
    assert {
        "imp-pers-con-lr1e5-s42",
        "imp-bare-con-lr1e5-s42",
    } == cells.NON_REGENERABLE_ARM_IDS
    assert regen_ids == set(cells.REUSED_CON_ARM_BY_ID) - cells.NON_REGENERABLE_ARM_IDS
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
    carries the con recipe fields verbatim. The 2 committed OUT-of-band arms
    are NOT regen slots (plan §4.6)."""
    for arm in cells.REUSED_CON_ARMS:
        beh_key = arm.arm_id.split("-")[0]
        regen_by_id = {r.run_id: r for r in fu4.ROUNDS[cells.regen_round_name(beh_key)].runs}
        if not arm.committed_in_band:
            assert arm.arm_id not in regen_by_id, arm.arm_id
            continue
        run = regen_by_id[arm.arm_id]
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


def test_out_of_band_regen_dispatch_refused():
    """Plan §4.6: the 2 committed closest-approach OUT-of-band arms are outside
    the regen trigger — an explicit --runs naming one is refused with a
    §4.6-citing message (not the generic unknown-runs error)."""
    for arm_id in sorted(cells.NON_REGENERABLE_ARM_IDS):
        with pytest.raises(SystemExit, match="OUTSIDE the regen trigger"):
            w._run_dispatch_group(
                ["--full", "--dispatch", "impolite", "--runs", arm_id], "impolite"
            )


def test_panel_run_registry_covers_regen_ids():
    """A regen'd verdict arm rides the fresh-arm panel path: the phase_panel
    run registry resolves regen ids (to their regen-round Fu4Run) alongside
    every fresh id."""
    reg = w._panel_run_registry()
    regen_ids = {r.run_id for rs in cells.REGEN_RUNS_BY_ROUND.values() for r in rs}
    fresh_ids = {r.run_id for rs in cells.RUNS_BY_ROUND.values() for r in rs}
    assert regen_ids <= set(reg)
    assert fresh_ids <= set(reg)
    for rid in regen_ids:
        assert reg[rid].round_name in cells.REGEN_ROUND_NAMES, rid


# ── Analysis-side ladder-of-record routing (concern
#    regen-ladder-analysis-integration; plan §4.6) ─────────────────────────────


def _committed_fu5_parent(repo_root: Path) -> None:
    """The REAL committed fu5 record shape (probed 2026-07-18: selection =
    {"step": 35, "rate": 0.64, "in_band": true, "fallback": null}) with the
    real committed rates — lr1e5 0.45 OUT of band, lr3e5 0.60 / lr1e4 0.64 in."""
    runs = {}
    for tag, rate in (("lr1e5", 0.45), ("lr3e5", 0.60), ("lr1e4", 0.64)):
        lo, hi = cells.JUDGED_RATE_BAND
        runs[f"imp-bare-{tag}"] = {
            "rates_by_step": {"25": 0.58, "35": rate},
            "selection": {"step": 35, "rate": rate, "in_band": lo <= rate <= hi, "fallback": None},
        }
    p = repo_root / FU5_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"runs": runs}))


def _reread_record(reread_dir: Path, arm_id: str, flagged: bool) -> None:
    """Schema-real per-arm P1 reread record (the worker phase_reread shape)."""
    reread_dir.mkdir(parents=True, exist_ok=True)
    (reread_dir / f"{arm_id}.json").write_text(
        json.dumps({"arm_id": arm_id, "contingent_regen_triggered": flagged})
    )


def _regen_ladders(ladders_dir: Path, arm_id: str, step: int, rate: float) -> None:
    """Fabricated-but-schema-real i1481impregen ladders JSON (fu4 run-record
    shape — the same shape the fresh i1481 round ladders carry)."""
    lo, hi = cells.JUDGED_RATE_BAND
    ladders_dir.mkdir(parents=True, exist_ok=True)
    (ladders_dir / "i1481impregen_ladders.json").write_text(
        json.dumps(
            {
                "runs": {
                    arm_id: {
                        "rates_by_step": {str(step): rate},
                        "selection": {
                            "step": step,
                            "rate": rate,
                            "in_band": lo <= rate <= hi,
                            "fallback": None,
                        },
                    }
                }
            }
        )
    )


def _select_paths(tmp_path: Path) -> ana.SelectPaths:
    _committed_fu5_parent(tmp_path / "repo")
    return ana.SelectPaths(
        ladders_dir=tmp_path / "ladders",
        repo_root=tmp_path / "repo",
        marker_root=None,
        reread_dir=tmp_path / "reread",
    )


def test_regen_ladder_supersedes_committed_parent(tmp_path, caplog):
    """A regen record for the flagged arm SUPERSEDES the committed fu5 parent
    (selection + rates come from the regen ladder, source == 'regen') with a
    loud log line naming the supersession."""
    paths = _select_paths(tmp_path)
    _reread_record(paths.reread_dir, FLAGGED, flagged=True)
    _regen_ladders(paths.ladders_dir, FLAGGED, step=40, rate=0.72)
    with caplog.at_level(logging.WARNING, logger="issue1481.analysis"):
        arm_id, sel, rates, source = ana._arm_record(paths, "imp", "bare", "con", 1e-4, 42)
    assert (arm_id, source) == (FLAGGED, "regen")
    assert (sel["step"], sel["rate"], sel["in_band"]) == (40, 0.72, True)
    assert rates == {"40": 0.72}
    assert any(FLAGGED in r.message and "SUPERSEDES" in r.message for r in caplog.records), (
        caplog.text
    )


def test_flagged_arm_without_regen_fails_loud(tmp_path):
    """A P1-flagged arm with NO regen ladder record must fail loud — never
    silently fall back to the superseded committed selection (plan §4.6)."""
    paths = _select_paths(tmp_path)
    _reread_record(paths.reread_dir, FLAGGED, flagged=True)
    with pytest.raises(RuntimeError, match="SUPERSEDED"):
        ana._arm_record(paths, "imp", "bare", "con", 1e-4, 42)


def test_unflagged_arm_resolves_from_committed_parent(tmp_path):
    """An UNFLAGGED reused arm keeps the committed parent as its ladder of
    record (source == 'reused-parent', the committed fu5 selection)."""
    paths = _select_paths(tmp_path)
    _reread_record(paths.reread_dir, "imp-bare-con-lr3e5-s42", flagged=False)
    arm_id, sel, rates, source = ana._arm_record(paths, "imp", "bare", "con", 3e-5, 42)
    assert (arm_id, source) == ("imp-bare-con-lr3e5-s42", "reused-parent")
    assert (sel["step"], sel["rate"]) == (35, 0.60)
    assert rates["35"] == 0.60


def test_missing_reread_evidence_fails_loud(tmp_path, monkeypatch):
    """No local P1 evidence + no HF copy -> RuntimeError (never a silent
    committed fallback). The HF staging boundary is faked signature-
    conformantly (mirrors hub.stage_hub_file) to raise, so no network."""

    def _raise_stage(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
    ):
        raise FileNotFoundError(f"{repo_id}/{path_in_repo}")

    monkeypatch.setattr(ana.hub, "stage_hub_file", _raise_stage)
    paths = _select_paths(tmp_path)  # reread_dir exists as a path but holds no records
    with pytest.raises(RuntimeError, match="no P1 reread evidence"):
        ana._arm_record(paths, "imp", "bare", "con", 3e-5, 42)


def test_reread_flag_falls_back_to_summary(tmp_path):
    """With no per-arm record, the flag resolves from reread_summary.json's
    arms dict (the worker writes both)."""
    paths = _select_paths(tmp_path)
    paths.reread_dir.mkdir(parents=True, exist_ok=True)
    (paths.reread_dir / "reread_summary.json").write_text(
        json.dumps({"arms": {FLAGGED: {"contingent_regen_triggered": True}}})
    )
    with pytest.raises(RuntimeError, match="SUPERSEDED"):
        ana._arm_record(paths, "imp", "bare", "con", 1e-4, 42)
