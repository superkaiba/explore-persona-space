"""Issue #2479 — P3 wrapper wiring + the validated axis-leg resume predicate.

Two halves (hermetic — tmp_path fixtures, zero network / API calls; the ONE
subprocess is the wrapper's own --dry-run against the committed panel):

(1) r2 codex `p3-controls-disconnected`: the canonical P3 wrapper stages the
    inserted cells and dispatches the flatness + name-mask legs, computes +
    commits instrument_gates.json (the P6 verdict's REQUIRED gates input),
    and re-uploads the legs dir AFTER the control legs land — pinned by
    static phase/flag/ordering assertions plus a real `--dry-run` execution.
(2) r2+r4 codex `p3-leg-resume-unvalidated`: `issue2479_p3_leg_resume.py` is
    the per-leg completion predicate bound to FULL input identity — a dry-run
    report, an old-rubric report, a changed-item-SET report, a changed item
    CONTENT report (same conv_ids), a partial per-item draw census, a
    wrong-design control report, and a report licensed by a
    different-fingerprint pilot must all NOT satisfy the skip (rc 3 +
    quarantine); a fully valid report must (rc 0); and a stale CURRENT-pilot
    binding dispatches WITHOUT quarantining the intact leg report.

Content hygiene: fixture rows are synthetic benign text, never LMSYS-derived.
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
import issue2479_instrument_gates as ig  # noqa: E402
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
        # r4: full-input-identity resume binding + data-bound pilot gate
        "--expect-design axis-census",
        '--expect-design "$prefix"',
        "--items-glob",
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
    # BOTH require-pass call sites (p3_pilot resume probe + p3_gate) arm the
    # data-identity comparison via --items-glob (r4 judge-pilot-gates-missing).
    lines = text.splitlines()
    call_idx = [
        i
        for i, ln in enumerate(lines)
        if "issue2479_judge_pilots.py --require-pass --family axis" in ln
        and not ln.lstrip().startswith("#")
        and "[dry-run]" not in ln
    ]
    assert len(call_idx) == 2, f"expected 2 require-pass call sites, got {len(call_idx)}"
    for i in call_idx:
        window = "\n".join(lines[i : i + 3])
        assert "--items-glob" in window, f"require-pass call at line {i + 1} lost --items-glob"


def test_wrapper_axis_loop_emits_per_unit_progress_both_branches() -> None:
    """r4 codex `p3-long-loop-progress-missing`: the 16-leg Batch loop emits a
    flushed `[p3_legs] unit k/N <name> elapsed=<s>s` completion line on BOTH
    the dispatched branch and the resume-skip branch."""
    text = WRAPPER.read_text()
    lines = [ln for ln in text.splitlines() if "[p3_legs] unit" in ln and "[dry-run]" not in ln]
    assert len(lines) >= 2, f"expected per-unit lines in both branches, got {lines}"
    assert any("resume-skip" in ln for ln in lines), lines
    for ln in lines:
        assert "elapsed=" in ln and "/${#NAMES[@]}" in ln, ln


def test_instrument_gates_loops_emit_per_unit_progress() -> None:
    """Same r4 concern for the 8-leg flatness + 8-leg name-mask Batch loops."""
    src = (SCRIPTS / "issue2479_instrument_gates.py").read_text()
    assert "[p3_flatness] unit {k}/{len(names_ordered)}" in src
    assert "[p3_namemask] unit {k}/{len(rows_sel)}" in src
    # Flushed (unbuffered) per code-style per-unit-progress convention.
    for tag in ("[p3_flatness] unit", "[p3_namemask] unit"):
        seg = src[src.index(tag) :]
        assert "flush=True" in seg[:400], f"{tag} print is not flushed"


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
        "--expect-design axis-census",
    ):
        assert token in proc.stdout, f"dry-run lost {token!r}: {proc.stdout[-2000:]}"


# ---------------------------------------------------------------------------
# (2) validated resume predicate — fixtures
# ---------------------------------------------------------------------------
CONV_IDS = ("c1", "c2")


def _rows(conv_ids=CONV_IDS, answer_of=None) -> list[dict]:
    answer_of = answer_of or (lambda c: f"a synthetic benign answer {c}")
    return [
        {"conv_id": c, "question": f"q {c}", "answer": answer_of(c), "capped": False}
        for c in conv_ids
    ]


def _scratch_env(tmp_path: Path, monkeypatch, conv_ids=CONV_IDS, name: str = "iris") -> Path:
    """Scratch panel + manifest + items dir, exported through the SAME env vars
    every production spend path resolves data identity from."""
    panel = [
        {
            "name": name,
            "variant_op": f"char_2479_{name}_op",
            "variant_inserted": None,
            "design_band": "A",
            "display_name": name.capitalize(),
        }
    ]
    panel_p = tmp_path / "panel.json"
    panel_p.write_text(json.dumps(panel))
    manifest_p = tmp_path / "panel_manifest.json"
    manifest_p.write_text(
        json.dumps({"axis_reservation_conv_ids": list(conv_ids), "n_reservation": len(conv_ids)})
    )
    items_dir = tmp_path / "axis_items"
    items_dir.mkdir(exist_ok=True)
    items_p = items_dir / f"axis_items_{name}.jsonl"
    items_p.write_text("\n".join(json.dumps(r) for r in _rows(conv_ids)) + "\n")
    monkeypatch.setenv(jp.PANEL_ENV, str(panel_p))
    monkeypatch.setenv(jp.MANIFEST_ENV, str(manifest_p))
    monkeypatch.setenv(jp.ITEMS_DIR_ENV, str(items_dir))
    return items_p


def _leg_report(tag: str, rows: list[dict]) -> dict:
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
        "n_items": len(rows),
        "rubric_sha256": fz.rubric_fingerprint(),
        "items_content_sha256": jl.items_content_fingerprint(jl.build_ai_likeness_items(rows, tag)),
        "licensing_pilot": {
            "report_path": "pilot_gate_axis.json",
            "family": "axis",
            "instrument": dict(jp.axis_instrument_fingerprint()),
            "data_identity": jp.axis_data_identity(),
            "created_utc": "2026-08-23T00:00:00Z",
        },
        "means": {"pooled": {"n": len(rows), "mean": 55.0}},
    }


def _design(tag: str, conv_ids, **extra) -> dict:
    return {
        "leg": jl.LEG_AI_LIKENESS,
        "seed": 0,
        "tag": tag,
        "n_target": len(conv_ids),
        "take_all": True,
        "realized_n": len(conv_ids),
        "realized_capped": 0,
        "realized_natural": len(conv_ids),
        "conv_ids": list(conv_ids),
        **extra,
    }


def _write_leg(
    tmp_path: Path,
    tag: str,
    conv_ids=CONV_IDS,
    mutate=None,
    mutate_design=None,
    n_draws: int = jl.N_DRAWS,
    answer_of=None,
) -> Path:
    """A COMPLETE persisted leg: report + full-census save_raw + design sidecar."""
    legs = tmp_path / "legs"
    legs.mkdir(exist_ok=True)
    rows = _rows(conv_ids, answer_of=answer_of)
    rep = _leg_report(tag, rows)
    if mutate is not None:
        mutate(rep)
    rp = legs / f"judge_report_ail_{tag}.json"
    rp.write_text(json.dumps(rep))
    all_scores = {}
    for cid in conv_ids:
        iid = jl.item_id(jl.LEG_AI_LIKENESS, tag, cid)
        for d in range(n_draws):
            all_scores[f"{iid}__{d:05d}__00"] = {"score": 50}
    (legs / f"judge_raw_ail_{tag}.json").write_text(json.dumps({"all_scores": all_scores}))
    design = _design(tag, conv_ids, census=True)
    if mutate_design is not None:
        mutate_design(design)
    (legs / f"judge_sample_ail_{tag}.json").write_text(json.dumps(design))
    return rp


def _write_items(tmp_path: Path, conv_ids, answer_of=None) -> Path:
    p = tmp_path / "axis_items_iris.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _rows(conv_ids, answer_of=answer_of)) + "\n")
    return p


def _pilot_pass(tmp_path: Path) -> Path:
    rep = {
        "issue": 2479,
        "family": "axis",
        "passed": True,
        "verdict": "PASS",
        "failures": [],
        "instrument": dict(jp.axis_instrument_fingerprint()),
        "data_identity": jp.axis_data_identity(),
        "arms": {"axis": {"n_draws": 150, "n_transport_lost": 0}},
        "metadata": {"created_utc": "2026-08-23T00:00:00Z"},
    }
    p = tmp_path / "pilot_gate_axis.json"
    p.write_text(json.dumps(rep))
    return p


def _quarantined(legs: Path) -> list[str]:
    return sorted(p.name for p in legs.glob("*.quarantined-*"))


# ---------------------------------------------------------------------------
# (2) validated resume predicate — cases
# ---------------------------------------------------------------------------
def test_valid_report_satisfies_skip(tmp_path: Path, monkeypatch) -> None:
    items = _scratch_env(tmp_path, monkeypatch)
    rp = _write_leg(tmp_path, "iris")
    pilot = _pilot_pass(tmp_path)
    rc = lr.main(
        [
            "--report",
            str(rp),
            "--tag",
            "iris",
            "--items",
            str(items),
            "--expect-design",
            "axis-census",
            "--pilot-report",
            str(pilot),
        ]
    )
    assert rc == lr.EXIT_VALID
    assert rp.is_file() and not _quarantined(rp.parent)


def test_missing_report_dispatches(tmp_path: Path) -> None:
    rc = lr.main(["--report", str(tmp_path / "legs" / "nope.json"), "--tag", "iris"])
    assert rc == lr.EXIT_DISPATCH


def test_dry_run_report_quarantined(tmp_path: Path, monkeypatch) -> None:
    """The r2 wrapper's existence-only skip accepted exactly this report."""
    _scratch_env(tmp_path, monkeypatch)
    rp = _write_leg(tmp_path, "iris", mutate=lambda r: r.update(spend_executed=False))
    rc = lr.main(["--report", str(rp), "--tag", "iris"])
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists()
    q = _quarantined(rp.parent)
    assert any(n.startswith("judge_report_ail_iris.json.quarantined-") for n in q), q
    # save_raw + design quarantined alongside (a re-run must never merge stale
    # draws or reuse a stale draw design).
    assert any(n.startswith("judge_raw_ail_iris.json.quarantined-") for n in q), q
    assert any(n.startswith("judge_sample_ail_iris.json.quarantined-") for n in q), q


def test_old_rubric_report_quarantined(tmp_path: Path, monkeypatch) -> None:
    _scratch_env(tmp_path, monkeypatch)
    rp = _write_leg(tmp_path, "iris", mutate=lambda r: r.update(rubric_sha256="0" * 64))
    rc = lr.main(["--report", str(rp), "--tag", "iris"])
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists() and _quarantined(rp.parent)


def test_missing_items_content_hash_quarantined(tmp_path: Path, monkeypatch) -> None:
    """A pre-r4 report (no items_content_sha256) never satisfies the skip."""
    _scratch_env(tmp_path, monkeypatch)
    rp = _write_leg(tmp_path, "iris", mutate=lambda r: r.pop("items_content_sha256"))
    rc = lr.main(["--report", str(rp), "--tag", "iris"])
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists() and _quarantined(rp.parent)


def test_changed_item_set_quarantined(tmp_path: Path, monkeypatch) -> None:
    """A report judged over a DIFFERENT item set than the freshly emitted one
    (panel/manifest/item drift) never satisfies the skip."""
    _scratch_env(tmp_path, monkeypatch)
    rp = _write_leg(tmp_path, "iris", conv_ids=("c1", "c2"))
    items = _write_items(tmp_path, ("c1", "c2", "c3"))  # freshly emitted set grew
    rc = lr.main(["--report", str(rp), "--tag", "iris", "--items", str(items)])
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists() and _quarantined(rp.parent)


def test_changed_answer_text_same_ids_quarantined(tmp_path: Path, monkeypatch) -> None:
    """r4 codex mechanization: SAME conv_ids with CHANGED answer text (a
    re-generation at a new revision) must re-dispatch — the exact-ID-set check
    alone cannot see it; the content fingerprint recompute does."""
    _scratch_env(tmp_path, monkeypatch)
    rp = _write_leg(tmp_path, "iris")  # judged over the original answer text
    items = _write_items(tmp_path, CONV_IDS, answer_of=lambda c: f"REGENERATED answer {c}")
    rc = lr.main(["--report", str(rp), "--tag", "iris", "--items", str(items)])
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists() and _quarantined(rp.parent)


def test_partial_draw_census_quarantined(tmp_path: Path, monkeypatch) -> None:
    """r4 codex mechanization: <N_DRAWS raw draws per item (a partial raw
    file beside a complete-looking report) never satisfies the skip."""
    _scratch_env(tmp_path, monkeypatch)
    rp = _write_leg(tmp_path, "iris", n_draws=2)
    rc = lr.main(["--report", str(rp), "--tag", "iris"])
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists() and _quarantined(rp.parent)


def test_missing_design_sidecar_quarantined(tmp_path: Path, monkeypatch) -> None:
    _scratch_env(tmp_path, monkeypatch)
    rp = _write_leg(tmp_path, "iris")
    (rp.parent / "judge_sample_ail_iris.json").unlink()
    rc = lr.main(["--report", str(rp), "--tag", "iris"])
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists() and _quarantined(rp.parent)


def test_missing_save_raw_quarantined(tmp_path: Path, monkeypatch) -> None:
    _scratch_env(tmp_path, monkeypatch)
    rp = _write_leg(tmp_path, "iris")
    (rp.parent / "judge_raw_ail_iris.json").unlink()
    items = _write_items(tmp_path, CONV_IDS)
    rc = lr.main(["--report", str(rp), "--tag", "iris", "--items", str(items)])
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists()


def test_stale_pilot_dispatches_without_quarantine(tmp_path: Path, monkeypatch) -> None:
    """A stale CURRENT-pilot binding forces re-dispatch but the intact leg
    report is NOT quarantined (the wrapper re-pilots and run_leg's own env
    guard enforces the pilot at spend)."""
    items = _scratch_env(tmp_path, monkeypatch)
    rp = _write_leg(tmp_path, "iris")
    pilot = _pilot_pass(tmp_path)
    stale = json.loads(pilot.read_text())
    stale["instrument"]["rubric_sha256"] = "0" * 64
    pilot.write_text(json.dumps(stale))
    rc = lr.main(
        ["--report", str(rp), "--tag", "iris", "--items", str(items), "--pilot-report", str(pilot)]
    )
    assert rc == lr.EXIT_DISPATCH
    assert rp.is_file() and not _quarantined(rp.parent)


def test_refreshed_pilot_licensing_mismatch_quarantined(tmp_path: Path, monkeypatch) -> None:
    """r4 codex sequencing note: the wrapper refreshes the pilot BEFORE leg
    validation, so a retained report must PROVE it was licensed by an
    equivalent-fingerprint pilot — a report licensed under an older
    materialization (different licensing data identity) re-dispatches."""
    items = _scratch_env(tmp_path, monkeypatch)

    def _stale_license(rep: dict) -> None:
        rep["licensing_pilot"]["data_identity"] = {
            **rep["licensing_pilot"]["data_identity"],
            "panel_sha256": "0" * 64,
        }

    rp = _write_leg(tmp_path, "iris", mutate=_stale_license)
    pilot = _pilot_pass(tmp_path)  # CURRENT pilot: valid, current fingerprints
    rc = lr.main(
        ["--report", str(rp), "--tag", "iris", "--items", str(items), "--pilot-report", str(pilot)]
    )
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists() and _quarantined(rp.parent)


def test_missing_licensing_block_quarantined(tmp_path: Path, monkeypatch) -> None:
    items = _scratch_env(tmp_path, monkeypatch)
    rp = _write_leg(tmp_path, "iris", mutate=lambda r: r.pop("licensing_pilot"))
    pilot = _pilot_pass(tmp_path)
    rc = lr.main(
        ["--report", str(rp), "--tag", "iris", "--items", str(items), "--pilot-report", str(pilot)]
    )
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists() and _quarantined(rp.parent)


def test_tag_mismatch_quarantined(tmp_path: Path, monkeypatch) -> None:
    _scratch_env(tmp_path, monkeypatch)
    rp = _write_leg(tmp_path, "iris", mutate=lambda r: r.update(tag="vex"))
    rc = lr.main(["--report", str(rp), "--tag", "iris"])
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists() and _quarantined(rp.parent)


def test_undersized_control_report_refused(tmp_path: Path, monkeypatch) -> None:
    """r4 codex: the r3 predicate BLESSED a two-item / two-draw `flat_` report
    as valid — the registered flatness design (FLAT_N common-draw items at
    SUBSAMPLE_SEED, N_DRAWS per item) must now refuse it."""
    _scratch_env(tmp_path, monkeypatch)
    rp = _write_leg(tmp_path, "flat_iris", n_draws=2)  # 2 items x 2 draws
    pilot = _pilot_pass(tmp_path)
    rc = lr.main(
        [
            "--report",
            str(rp),
            "--tag",
            "flat_iris",
            "--expect-design",
            "flat",
            "--pilot-report",
            str(pilot),
        ]
    )
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists() and _quarantined(rp.parent)


def test_changed_control_design_quarantined(tmp_path: Path, monkeypatch) -> None:
    """A control leg drawn under a NON-registered design (wrong seed / size /
    no common draw) never satisfies the skip."""
    _scratch_env(tmp_path, monkeypatch)

    def _wrong_design(d: dict) -> None:
        d.pop("census", None)
        d.update(n_target=ig.FLAT_N, seed=ig.SUBSAMPLE_SEED + 1, common_draw=True)

    rp = _write_leg(tmp_path, "flat_iris", mutate_design=_wrong_design)
    rc = lr.main(["--report", str(rp), "--tag", "flat_iris", "--expect-design", "flat"])
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists() and _quarantined(rp.parent)


def test_valid_control_leg_satisfies_skip(tmp_path: Path, monkeypatch) -> None:
    """A flat_ leg drawn under the REGISTERED design with a full census and a
    current licensing pilot validates (rc 0) — the positive control for the
    two refusals above."""
    _scratch_env(tmp_path, monkeypatch)
    ids = tuple(f"c{i}" for i in range(ig.FLAT_N))

    def _flat_design(d: dict) -> None:
        d.pop("census", None)
        d.update(n_target=ig.FLAT_N, seed=ig.SUBSAMPLE_SEED, common_draw=True)

    rp = _write_leg(tmp_path, "flat_iris", conv_ids=ids, mutate_design=_flat_design)
    pilot = _pilot_pass(tmp_path)
    rc = lr.main(
        [
            "--report",
            str(rp),
            "--tag",
            "flat_iris",
            "--expect-design",
            "flat",
            "--pilot-report",
            str(pilot),
        ]
    )
    assert rc == lr.EXIT_VALID
    assert rp.is_file() and not _quarantined(rp.parent)


def test_quarantine_names_collision_free(tmp_path: Path) -> None:
    """r4 codex nit: two quarantines of the same leg within one second must
    not overwrite earlier evidence (pid + process-local counter suffix)."""
    legs = tmp_path / "legs"
    legs.mkdir()
    p = legs / "judge_report_ail_iris.json"
    for _ in range(2):
        p.write_text("{}")
        moved = lr.quarantine([p])
        assert len(moved) == 1
    q = _quarantined(legs)
    assert len(q) == 2, q


def test_import_check_runs_clean() -> None:
    rc = lr.main(["--import-check"])
    assert rc == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
