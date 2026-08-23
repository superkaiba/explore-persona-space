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
(3) r5 control residual of the same concern: the wrapper derives the CURRENT
    control item rows (instrument_gates --step emit-control-items — the SAME
    derivation the dispatch steps run) and binds every flat_/mask_ resume to
    them via --items, so changed kept answers, changed masking output, or
    sampled-ID drift refuse the skip (rc 3) while an unchanged
    materialization still skips cleanly (rc 0).

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
        # r5: control resume sweeps bind to the CURRENT derived items
        # (codex p3-leg-resume-unvalidated control residual)
        "[phase=p3_control_items]",
        "--step emit-control-items",
        "--control-items-dir",
        '--items "$cur_items"',
    ):
        assert token in text, f"wrapper lost required token: {token!r}"
    # instrument_gates.json is produced AFTER the freeze (its flatness gate
    # needs the realized axis range) and BEFORE the control re-upload, so the
    # upload publishes the control legs (r2 codex sequencing requirement).
    # The control-items derivation runs after the freeze (axis raw complete)
    # and before the first control resume sweep consumes it (r5).
    assert (
        text.index("[phase=p3_freeze]")
        < text.index("[phase=p3_control_items]")
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
    assert "[p3_flatness] unit {k}/{len(legs)}" in src
    assert "[p3_namemask] unit {k}/{len(legs)}" in src
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
        "[dry-run] p3_control_items:",
        "[dry-run] p3_flatness:",
        "[dry-run] p3_namemask:",
        "[dry-run] p3_gates:",
        "[dry-run] p3_upload_controls:",
        "issue2479_p3_leg_resume.py",
        "--expect-design axis-census",
        "--step emit-control-items",
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


# ---------------------------------------------------------------------------
# (3) r5 control-leg residual — CURRENT-input derivation binding
# (codex `p3-leg-resume-unvalidated`: flat/mask resumes previously trusted the
# sidecar's own conv_ids + a hash presence check; a self-consistent stale
# report/raw/design triple skipped. The wrapper now derives the CURRENT
# control items via instrument_gates --step emit-control-items and passes
# them to the validator via --items.)
# ---------------------------------------------------------------------------
FLAT_IDS = ("c1", "c2")


def _control_fixture(
    root: Path,
    *,
    kept_answer_of=None,
    axis_answer_of=None,
    display_name: str = "Iris",
) -> dict:
    """Minimal REAL input surface of the shared control derivations: a panel
    with two inserted characters (iris also band-A for the mask leg), staged
    inserted kept rows (identical reference answers across characters — the
    flatness identity invariant), emitted axis items, and a full-census axis
    save_raw. Synthetic benign text only."""
    kept_answer_of = kept_answer_of or (lambda c: f"the shared reference answer for {c}")
    axis_answer_of = axis_answer_of or (
        lambda c: f"{display_name} explained the takeaway for {c} carefully."
    )
    panel = [
        {
            "name": "iris",
            "variant_op": "char_2479_iris_op",
            "variant_inserted": "char_2479_iris_ins",
            "design_band": "A",
            "display_name": display_name,
        },
        {
            "name": "vex",
            "variant_op": "char_2479_vex_op",
            "variant_inserted": "char_2479_vex_ins",
            "design_band": "B",
            "display_name": "Vex",
        },
    ]
    kept_dir = root / "kept"
    for n in ("iris", "vex"):
        d = kept_dir / f"char_2479_{n}_ins"
        d.mkdir(parents=True, exist_ok=True)
        rows = [
            {"conv_id": c, "question": f"q {c}", "answer": kept_answer_of(c), "capped": False}
            for c in FLAT_IDS
        ]
        (d / "kept.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    items_dir = root / "axis_items"
    items_dir.mkdir(parents=True, exist_ok=True)
    axis_rows = [
        {"conv_id": c, "question": f"q {c}", "answer": axis_answer_of(c), "capped": False}
        for c in FLAT_IDS
    ]
    (items_dir / "axis_items_iris.jsonl").write_text(
        "\n".join(json.dumps(r) for r in axis_rows) + "\n"
    )
    raw_dir = root / "axis_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_scores = {}
    for c in FLAT_IDS:
        iid = jl.item_id(jl.LEG_AI_LIKENESS, "iris", c)
        for d in range(jl.N_DRAWS):
            all_scores[f"{iid}__{d:05d}__00"] = {"score": 50}
    (raw_dir / "judge_raw_ail_iris.json").write_text(json.dumps({"all_scores": all_scores}))
    return {
        "panel": panel,
        "reservation": set(FLAT_IDS),
        "kept_glob": str(kept_dir / "{variant}" / "kept.jsonl"),
        "items_glob": str(items_dir / "axis_items_{name}.jsonl"),
        "axis_raw_glob": str(raw_dir / "judge_raw_ail_{name}.json"),
        "out_dir": root / "control_items",
    }


def _emit(fix: dict) -> dict[str, Path]:
    return ig.emit_control_items(
        fix["panel"],
        fix["reservation"],
        fix["kept_glob"],
        None,
        fix["items_glob"],
        fix["axis_raw_glob"],
        fix["out_dir"],
    )


def _write_control_leg(tmp_path: Path, tag: str, rows: list[dict], design: dict) -> Path:
    """A COMPLETE persisted control leg judged over exactly `rows` under the
    DERIVED `design` (report + full-census save_raw + design sidecar)."""
    legs = tmp_path / "legs"
    legs.mkdir(exist_ok=True)
    rp = legs / f"judge_report_ail_{tag}.json"
    rp.write_text(json.dumps(_leg_report(tag, rows)))
    all_scores = {}
    for r in rows:
        iid = jl.item_id(jl.LEG_AI_LIKENESS, tag, str(r["conv_id"]))
        for d in range(jl.N_DRAWS):
            all_scores[f"{iid}__{d:05d}__00"] = {"score": 50}
    (legs / f"judge_raw_ail_{tag}.json").write_text(json.dumps({"all_scores": all_scores}))
    (legs / f"judge_sample_ail_{tag}.json").write_text(json.dumps(design))
    return rp


def _control_argv(rp: Path, tag: str, items: Path, prefix: str, pilot: Path) -> list[str]:
    """The wrapper's all_control_legs_valid invocation shape (r5: --items)."""
    return [
        "--report",
        str(rp),
        "--tag",
        tag,
        "--items",
        str(items),
        "--expect-design",
        prefix,
        "--pilot-report",
        str(pilot),
    ]


def test_emit_control_items_writes_current_derivation(tmp_path: Path) -> None:
    """emit-control-items writes EXACTLY the rows the dispatch steps judge —
    one file per control leg, masking realized on the mask rows."""
    fix = _control_fixture(tmp_path / "cur")
    emitted = _emit(fix)
    assert set(emitted) == {"flat_iris", "flat_vex", "mask_iris"}
    flat = ig.derive_flatness_legs(fix["panel"], fix["reservation"], fix["kept_glob"], None)
    rows = [json.loads(ln) for ln in emitted["flat_iris"].read_text().splitlines()]
    assert rows == flat["iris"][0]
    mask = ig.derive_namemask_legs(fix["panel"], fix["items_glob"], fix["axis_raw_glob"])
    mrows = [json.loads(ln) for ln in emitted["mask_iris"].read_text().splitlines()]
    assert mrows == mask["iris"][0]
    assert all("the character" in r["answer"].lower() for r in mrows), (
        "masking never fired on the mask control rows"
    )


def test_control_leg_current_derivation_satisfies_skip(tmp_path: Path, monkeypatch) -> None:
    """UNCHANGED materialization still skips cleanly: a flat_ leg judged over
    EXACTLY the current derived rows + design validates (rc 0) under the
    wrapper-shaped invocation (--items <current emitted rows>)."""
    _scratch_env(tmp_path, monkeypatch)
    fix = _control_fixture(tmp_path / "cur")
    emitted = _emit(fix)
    sampled, design = ig.derive_flatness_legs(
        fix["panel"], fix["reservation"], fix["kept_glob"], None
    )["iris"]
    rp = _write_control_leg(tmp_path, "flat_iris", sampled, design)
    pilot = _pilot_pass(tmp_path)
    rc = lr.main(_control_argv(rp, "flat_iris", emitted["flat_iris"], "flat", pilot))
    assert rc == lr.EXIT_VALID
    assert rp.is_file() and not _quarantined(rp.parent)


def test_stale_flat_report_refused_against_current_items(tmp_path: Path, monkeypatch) -> None:
    """r5 FLIP of the r4 blessing fixture: the synthetic 100-ID flat_ report
    the r4 predicate ACCEPTED (self-consistent report/raw/design triple,
    registered design constants) is REFUSED (rc 3 + quarantine) once the
    wrapper-shaped invocation binds it to the CURRENT derivation's sampled
    control IDs."""
    _scratch_env(tmp_path, monkeypatch)
    ids = tuple(f"c{i}" for i in range(ig.FLAT_N))

    def _flat_design(d: dict) -> None:
        d.pop("census", None)
        d.update(n_target=ig.FLAT_N, seed=ig.SUBSAMPLE_SEED, common_draw=True)

    rp = _write_leg(tmp_path, "flat_iris", conv_ids=ids, mutate_design=_flat_design)
    pilot = _pilot_pass(tmp_path)
    emitted = _emit(_control_fixture(tmp_path / "cur"))  # current draw: 2 ids
    rc = lr.main(_control_argv(rp, "flat_iris", emitted["flat_iris"], "flat", pilot))
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists() and _quarantined(rp.parent)


def test_control_leg_changed_kept_answers_refused(tmp_path: Path, monkeypatch) -> None:
    """r5 codex mechanization: SAME sampled conv_ids, CHANGED kept reference
    answers (a re-stage at a new revision) — the content recompute against
    the CURRENT flat derivation refuses the retained leg (rc 3)."""
    _scratch_env(tmp_path, monkeypatch)
    fix = _control_fixture(tmp_path / "v1")
    sampled, design = ig.derive_flatness_legs(
        fix["panel"], fix["reservation"], fix["kept_glob"], None
    )["iris"]
    rp = _write_control_leg(tmp_path, "flat_iris", sampled, design)
    pilot = _pilot_pass(tmp_path)
    fix2 = _control_fixture(
        tmp_path / "v2", kept_answer_of=lambda c: f"REGENERATED reference answer {c}"
    )
    emitted2 = _emit(fix2)
    rc = lr.main(_control_argv(rp, "flat_iris", emitted2["flat_iris"], "flat", pilot))
    assert rc == lr.EXIT_DISPATCH
    assert not rp.exists() and _quarantined(rp.parent)


def test_control_leg_changed_masking_output_refused(tmp_path: Path, monkeypatch) -> None:
    """r5 codex mechanization: SAME source axis rows + SAME sampled conv_ids,
    but the CURRENT masking output differs (panel display rename → the old
    name no longer masks) — the mask_ content recompute refuses (rc 3); the
    unchanged materialization first validates (rc 0)."""
    _scratch_env(tmp_path, monkeypatch)

    def axis_of(c: str) -> str:
        return f"Iris explained the takeaway for {c} carefully."

    fix = _control_fixture(tmp_path / "v1", axis_answer_of=axis_of, display_name="Iris")
    emitted = _emit(fix)
    masked_rows, design, provenance = ig.derive_namemask_legs(
        fix["panel"], fix["items_glob"], fix["axis_raw_glob"]
    )["iris"]
    assert provenance["n_items_with_mask_hits"] == len(masked_rows)
    rp = _write_control_leg(tmp_path, "mask_iris", masked_rows, design)
    pilot = _pilot_pass(tmp_path)
    rc = lr.main(_control_argv(rp, "mask_iris", emitted["mask_iris"], "mask", pilot))
    assert rc == lr.EXIT_VALID and rp.is_file()

    fix2 = _control_fixture(tmp_path / "v2", axis_answer_of=axis_of, display_name="Zara")
    emitted2 = _emit(fix2)  # "Iris" no longer masks: same ids, different text
    rc2 = lr.main(_control_argv(rp, "mask_iris", emitted2["mask_iris"], "mask", pilot))
    assert rc2 == lr.EXIT_DISPATCH
    assert not rp.exists() and _quarantined(rp.parent)


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
