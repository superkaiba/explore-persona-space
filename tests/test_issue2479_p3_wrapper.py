"""Issue #2479 — P3 wrapper wiring + the validated axis-leg resume predicate.

Two halves (hermetic — tmp_path fixtures, zero network / API calls; the ONE
subprocess is the wrapper's own --dry-run against the committed panel):

(1) r2 codex `p3-controls-disconnected`: the canonical P3 wrapper stages the
    inserted cells and dispatches the flatness + name-mask legs, computes +
    commits instrument_gates.json (the P6 verdict's REQUIRED gates input),
    and re-uploads the legs dir AFTER the control legs land — pinned by
    static phase/flag/ordering assertions plus a real `--dry-run` execution.
(2) r2 codex `p3-leg-resume-unvalidated`: `issue2479_p3_leg_resume.py` is the
    per-character completion predicate — a dry-run report, an old-rubric
    report, or a changed-item-set report must NOT satisfy the skip (rc 3 +
    quarantine), a valid report must (rc 0), and a stale pilot binding
    dispatches WITHOUT quarantining the intact leg report.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1345_onpolicy_judge_legs as jl  # noqa: E402
import issue2479_freeze_axis as fz  # noqa: E402
import issue2479_judge_pilots as jp  # noqa: E402
import issue2479_p3_leg_resume as lr  # noqa: E402

WRAPPER = SCRIPTS / "issue2479_p3_axis.sh"


# ---------------------------------------------------------------------------
# (1) wrapper wiring — static pins + real --dry-run
# ---------------------------------------------------------------------------
def test_wrapper_declares_control_phases_and_flags() -> None:
    text = WRAPPER.read_text()
    for token in (
        "[phase=p3_stage_inserted]",
        "[phase=p3_flatness]",
        "[phase=p3_namemask]",
        "[phase=p3_gates]",
        "[phase=p3_upload_controls]",
        "--step flatness",
        "--step namemask",
        "--step gates",
        "instrument_gates.json",
        "issue2479_p3_leg_resume.py",
        "--stats-out-dir",
        "--stats-dir",
        "--axis-pilot-report",
    ):
        assert token in text, f"wrapper lost required token: {token!r}"
    # instrument_gates.json is produced AFTER the freeze (its flatness gate
    # needs the realized axis range) and BEFORE the control re-upload, so the
    # upload publishes the control legs (r2 codex sequencing requirement).
    assert (
        text.index("[phase=p3_freeze]")
        < text.index("[phase=p3_flatness]")
        < text.index("[phase=p3_namemask]")
        < text.index("[phase=p3_gates]")
        < text.index("[phase=p3_upload_controls]")
    )
    # The gates JSON is committed by explicit path (concurrent-committer rule).
    assert 'git commit -m "issue-2479 P3: instrument gates' in text
    assert '-- "$GATES_OUT"' in text


def test_wrapper_dry_run_lists_control_phases() -> None:
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    for token in (
        "[dry-run] p3_stage_inserted:",
        "[dry-run] p3_flatness:",
        "[dry-run] p3_namemask:",
        "[dry-run] p3_gates:",
        "[dry-run] p3_upload_controls:",
        "issue2479_p3_leg_resume.py",
    ):
        assert token in proc.stdout, f"dry-run lost {token!r}: {proc.stdout[-2000:]}"


# ---------------------------------------------------------------------------
# (2) validated resume predicate
# ---------------------------------------------------------------------------
def _leg_report(tag: str) -> dict:
    return {
        "leg": jl.LEG_AI_LIKENESS,
        "tag": tag,
        "spend_executed": True,
        "spend_reason": "explicitly acknowledged",
        "judge_model": jl.JUDGE_MODEL,
        "n_draws": jl.N_DRAWS,
        "temperature": jl.JUDGE_TEMPERATURE,
        "max_tokens": jl.JUDGE_MAX_TOKENS,
        "threshold_base": jl.THRESHOLD_BASE_FORCE_BATCH,
        "n_items": 2,
        "rubric_sha256": fz.rubric_fingerprint(),
        "means": {"pooled": {"n": 2, "mean": 55.0}},
    }


def _write_leg(tmp_path: Path, tag: str, conv_ids=("c1", "c2"), mutate=None) -> Path:
    legs = tmp_path / "legs"
    legs.mkdir(exist_ok=True)
    rep = _leg_report(tag)
    if mutate is not None:
        mutate(rep)
    rp = legs / f"judge_report_ail_{tag}.json"
    rp.write_text(json.dumps(rep))
    all_scores = {}
    for cid in conv_ids:
        iid = jl.item_id(jl.LEG_AI_LIKENESS, tag, cid)
        for d in range(2):
            all_scores[f"{iid}__{d}__x"] = {"score": 50}
    (legs / f"judge_raw_ail_{tag}.json").write_text(json.dumps({"all_scores": all_scores}))
    return rp


def _write_items(tmp_path: Path, conv_ids) -> Path:
    p = tmp_path / "axis_items_iris.jsonl"
    p.write_text(
        "\n".join(
            json.dumps({"conv_id": c, "question": "q", "answer": "a", "capped": False})
            for c in conv_ids
        )
        + "\n"
    )
    return p


def _pilot_pass(tmp_path: Path) -> Path:
    rep = {
        "issue": 2479,
        "family": "axis",
        "passed": True,
        "verdict": "PASS",
        "failures": [],
        "instrument": dict(jp.axis_instrument_fingerprint()),
        "arms": {"axis": {"n_draws": 150, "n_transport_lost": 0}},
    }
    p = tmp_path / "pilot_gate_axis.json"
    p.write_text(json.dumps(rep))
    return p


def _quarantined(legs: Path) -> list[str]:
    return sorted(p.name for p in legs.glob("*.quarantined-*"))


def test_valid_report_satisfies_skip(tmp_path: Path) -> None:
    rp = _write_leg(tmp_path, "iris")
    items = _write_items(tmp_path, ("c1", "c2"))
    pilot = _pilot_pass(tmp_path)
    rc = lr.main(
        ["--report", str(rp), "--tag", "iris", "--items", str(items), "--pilot-report", str(pilot)]
    )
    assert rc == lr.EXIT_VALID
    assert rp.is_file() and not _quarantined(rp.parent)


def test_missing_report_dispatches(tmp_path: Path) -> None:
    rc = lr.main(["--report", str(tmp_path / "legs" / "nope.json"), "--tag", "iris"])
    assert rc == lr.EXIT_DISPATCH


def test_dry_run_report_quarantined(tmp_path: Path) -> None:
    """The r2 wrapper's existence-only skip accepted exactly this report."""
    rp = _write_leg(tmp_path, "iris", mutate=lambda r: r.update(spend_executed=False))
    rc = lr.main(["--report", str(rp), "--tag", "iris"])
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists()
    q = _quarantined(rp.parent)
    assert any(n.startswith("judge_report_ail_iris.json.quarantined-") for n in q), q
    # save_raw quarantined alongside (a re-run must never merge stale draws)
    assert any(n.startswith("judge_raw_ail_iris.json.quarantined-") for n in q), q


def test_old_rubric_report_quarantined(tmp_path: Path) -> None:
    rp = _write_leg(tmp_path, "iris", mutate=lambda r: r.update(rubric_sha256="0" * 64))
    rc = lr.main(["--report", str(rp), "--tag", "iris"])
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists() and _quarantined(rp.parent)


def test_changed_item_set_quarantined(tmp_path: Path) -> None:
    """A report judged over a DIFFERENT item set than the freshly emitted one
    (panel/manifest/item drift) never satisfies the skip."""
    rp = _write_leg(tmp_path, "iris", conv_ids=("c1", "c2"))
    items = _write_items(tmp_path, ("c1", "c2", "c3"))  # freshly emitted set grew
    rc = lr.main(["--report", str(rp), "--tag", "iris", "--items", str(items)])
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists() and _quarantined(rp.parent)


def test_missing_save_raw_quarantined(tmp_path: Path) -> None:
    rp = _write_leg(tmp_path, "iris")
    (rp.parent / "judge_raw_ail_iris.json").unlink()
    items = _write_items(tmp_path, ("c1", "c2"))
    rc = lr.main(["--report", str(rp), "--tag", "iris", "--items", str(items)])
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists()


def test_stale_pilot_dispatches_without_quarantine(tmp_path: Path) -> None:
    """A stale pilot binding forces re-dispatch but the intact leg report is
    NOT quarantined (run_leg's own env guard enforces the pilot at spend)."""
    rp = _write_leg(tmp_path, "iris")
    items = _write_items(tmp_path, ("c1", "c2"))
    pilot = _pilot_pass(tmp_path)
    stale = json.loads(pilot.read_text())
    stale["instrument"]["rubric_sha256"] = "0" * 64
    pilot.write_text(json.dumps(stale))
    rc = lr.main(
        ["--report", str(rp), "--tag", "iris", "--items", str(items), "--pilot-report", str(pilot)]
    )
    assert rc == lr.EXIT_DISPATCH
    assert rp.is_file() and not _quarantined(rp.parent)


def test_tag_mismatch_quarantined(tmp_path: Path) -> None:
    rp = _write_leg(tmp_path, "iris", mutate=lambda r: r.update(tag="vex"))
    rc = lr.main(["--report", str(rp), "--tag", "iris"])
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists() and _quarantined(rp.parent)


def test_control_leg_tags_validate(tmp_path: Path) -> None:
    """flat_/mask_ control legs ride the same predicate (no --items arm)."""
    rp = _write_leg(tmp_path, "flat_iris")
    pilot = _pilot_pass(tmp_path)
    rc = lr.main(["--report", str(rp), "--tag", "flat_iris", "--pilot-report", str(pilot)])
    assert rc == lr.EXIT_VALID


def test_import_check_runs_clean() -> None:
    rc = lr.main(["--import-check"])
    assert rc == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
