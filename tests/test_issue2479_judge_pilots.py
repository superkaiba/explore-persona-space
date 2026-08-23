"""Issue #2479 — rule-26 judge pilot gates (plan §7): wiring + gate arithmetic.

Pins (hermetic — tmp_path fixtures, zero network / API calls):

(a) `require_pilot_pass` semantics: missing report, FAIL report, family
    mismatch all RAISE; a PASS report bound to the CURRENT production
    instrument AND the CURRENT data materialization returns; a PASS with NO
    persisted instrument block, ANY stale instrument field (model / draws /
    temperature / max_tokens / threshold_base / route / rubric / builder /
    parser sha), NO persisted data_identity block, ANY stale data-identity
    field (panel / manifest bytes, pooled item content), or an
    effective-draw count under the family floor RAISES (r3 + r4 — r2/r4
    codex `judge-pilot-gates-missing` + g6);
(b) codex `judge-pilot-gates-missing` mechanization: a FAILED (or missing)
    pilot report BLOCKS a real `jl.run_leg` spend via the opt-in
    EPM_I2479_REQUIRE_AXIS_PILOT_PASS env — the guard fires BEFORE any API
    call; the env absent keeps parent #1345 behavior byte-identical;
(c) in-gen family merge gate arithmetic: zero stop_reason=="max_tokens",
    parse-fail < 2% of answered, hollow-arm floor, transport draws excluded
    from every denominator, mixed-instrument partials refused;
(d) `run_ingen_partial` PRODUCTION BODY: executes the real body with the ONLY
    fake at the external API boundary (a def-mirroring async fake of
    `dispatch_calls` that exercises the gen module's REAL
    `_build_judge_request` / `_parse_judge_response` on every item).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1345_onpolicy_judge_legs as jl  # noqa: E402
import issue2479_judge_pilots as jp  # noqa: E402


def _report(family: str, verdict: str, failures: list[str] | None = None) -> dict:
    return {
        "issue": 2479,
        "family": family,
        "passed": verdict == "PASS",
        "verdict": verdict,
        "failures": failures or ([] if verdict == "PASS" else ["synthetic failure"]),
    }


def _full_axis_report(
    *, n_draws: int = 150, lost: int = 0, data_identity: dict | None = None
) -> dict:
    """A PASS report bound to the CURRENT axis production instrument.

    ``data_identity`` is caller-supplied (r4): computing it requires the
    scratch-env materialization, and the instrument-stage refusal tests raise
    BEFORE the data stage, so they legitimately omit the block.
    """
    rep = _report("axis", "PASS")
    rep["instrument"] = dict(jp.axis_instrument_fingerprint())
    if data_identity is not None:
        rep["data_identity"] = data_identity
    rep["arms"] = {"axis": {"n_draws": n_draws, "n_transport_lost": lost}}
    return rep


def _scratch_env(tmp_path: Path, monkeypatch, name: str = "iris") -> Path:
    """Scratch panel + manifest + items dir exported through the SAME env vars
    every production spend path resolves data identity from (r4).

    Returns the item file path (synthetic benign fixture rows — never
    LMSYS-derived text)."""
    panel = [
        {
            "name": name,
            "variant_op": f"char_2479_{name}_op",
            "variant_inserted": None,
            "design_band": "A",
            "display_name": name.capitalize(),
        }
    ]
    (tmp_path / "panel.json").write_text(json.dumps(panel))
    (tmp_path / "panel_manifest.json").write_text(
        json.dumps({"axis_reservation_conv_ids": ["c1", "c2"], "n_reservation": 2})
    )
    items_dir = tmp_path / "axis_items"
    items_dir.mkdir(exist_ok=True)
    items_p = items_dir / f"axis_items_{name}.jsonl"
    items_p.write_text(
        "\n".join(
            json.dumps({"conv_id": c, "question": f"q {c}", "answer": f"a benign answer {c}"})
            for c in ("c1", "c2")
        )
        + "\n"
    )
    monkeypatch.setenv(jp.PANEL_ENV, str(tmp_path / "panel.json"))
    monkeypatch.setenv(jp.MANIFEST_ENV, str(tmp_path / "panel_manifest.json"))
    monkeypatch.setenv(jp.ITEMS_DIR_ENV, str(items_dir))
    return items_p


# ---------------------------------------------------------------------------
# (a) require_pilot_pass semantics
# ---------------------------------------------------------------------------
def test_require_pass_missing_report_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="pilot gate report missing"):
        jp.require_pilot_pass(tmp_path / "nope.json", family="axis")


def test_require_pass_fail_report_raises(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_report("axis", "FAIL")))
    with pytest.raises(RuntimeError, match="production dispatch refused"):
        jp.require_pilot_pass(p, family="axis")


def test_require_pass_family_mismatch_raises(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_report("ingen", "PASS")))
    with pytest.raises(RuntimeError, match="family"):
        jp.require_pilot_pass(p, family="axis")


def test_require_pass_pass_without_instrument_raises(tmp_path: Path) -> None:
    """r3: a PASS unbound to the production instrument certifies nothing."""
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_report("axis", "PASS")))
    with pytest.raises(RuntimeError, match="NO instrument block"):
        jp.require_pilot_pass(p, family="axis")


def test_require_pass_instrument_and_data_bound_pass_returns(tmp_path: Path, monkeypatch) -> None:
    _scratch_env(tmp_path, monkeypatch)
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_full_axis_report(data_identity=jp.axis_data_identity())))
    rep = jp.require_pilot_pass(p, family="axis")
    assert rep["verdict"] == "PASS"


def test_require_pass_smoke_synthesized_rejected(tmp_path: Path, monkeypatch) -> None:
    """r5 codex `smoke-root-production-poisoning` mechanization (ii): a
    CURRENT-identity PASS carrying smoke_synthesized=true is rejected by
    production require_pilot_pass — verdict/instrument/data identity all
    matching cannot make a synthesized pilot production-valid."""
    _scratch_env(tmp_path, monkeypatch)
    monkeypatch.delenv(jp.ALLOW_SYNTHESIZED_ENV, raising=False)
    rep = _full_axis_report(data_identity=jp.axis_data_identity())
    rep["smoke_synthesized"] = True
    p = tmp_path / "r.json"
    p.write_text(json.dumps(rep))
    with pytest.raises(RuntimeError, match="smoke_synthesized"):
        jp.require_pilot_pass(p, family="axis")
    # The refusal is unconditional on the identity stages: it also fires with
    # no family (the bare-verdict read path).
    with pytest.raises(RuntimeError, match="never licenses production spend"):
        jp.require_pilot_pass(p)


def test_require_pass_smoke_synthesized_allowed_only_explicitly(
    tmp_path: Path, monkeypatch
) -> None:
    """The smoke driver's explicit licence — the kwarg or its env (which ONLY
    scripts/issue2479_p3_controls_smoke.py sets) — is the sole path through."""
    _scratch_env(tmp_path, monkeypatch)
    rep = _full_axis_report(data_identity=jp.axis_data_identity())
    rep["smoke_synthesized"] = True
    p = tmp_path / "r.json"
    p.write_text(json.dumps(rep))
    assert jp.require_pilot_pass(p, family="axis", allow_synthesized=True)["verdict"] == "PASS"
    monkeypatch.setenv(jp.ALLOW_SYNTHESIZED_ENV, "1")
    assert jp.require_pilot_pass(p, family="axis")["verdict"] == "PASS"
    # An explicit False overrides the env (belt for future callers).
    with pytest.raises(RuntimeError, match="smoke_synthesized"):
        jp.require_pilot_pass(p, family="axis", allow_synthesized=False)


def test_smoke_driver_is_the_only_allow_synthesized_env_setter() -> None:
    """Grep-level pin: no production file arms jp.ALLOW_SYNTHESIZED_ENV —
    only the p3-controls smoke driver injects it into its subprocess env.

    r6 (codex `synthesized-env-setter-pin-incomplete`): the sweep is RECURSIVE
    over scripts/ (.py AND .sh — shell launchers can export the env too) plus
    src/ (.py), and matches BOTH the constant reference and the literal env
    name, so a nested/shell/library setter cannot slip under the pin. The
    defining module (issue2479_judge_pilots.py, definition + reader) and the
    smoke driver (the sole setter) are the only sanctioned mentions."""
    candidates = sorted(
        [
            *(REPO / "scripts").rglob("*.py"),
            *(REPO / "scripts").rglob("*.sh"),
            *(REPO / "src").rglob("*.py"),
        ]
    )
    hits = []
    for p in candidates:
        text = p.read_text(errors="replace")
        if "ALLOW_SYNTHESIZED_ENV" in text or jp.ALLOW_SYNTHESIZED_ENV in text:
            hits.append(str(p.relative_to(REPO)))
    assert hits == [
        "scripts/issue2479_judge_pilots.py",
        "scripts/issue2479_p3_controls_smoke.py",
    ], hits


def test_require_pass_pass_without_data_identity_raises(tmp_path: Path, monkeypatch) -> None:
    """r4: a PASS unbound to the panel/manifest/item materialization certifies
    nothing — the data-identity sibling of the NO-instrument-block refusal."""
    _scratch_env(tmp_path, monkeypatch)
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_full_axis_report()))  # instrument OK, no data block
    with pytest.raises(RuntimeError, match="NO data_identity block"):
        jp.require_pilot_pass(p, family="axis")


@pytest.mark.parametrize(
    "field",
    [
        "panel_sha256",
        "panel_manifest_sha256",
        "items_content_sha256",
        "n_pooled_items",
        "characters_pooled",
    ],
)
def test_require_pass_refuses_each_stale_data_field(
    tmp_path: Path, monkeypatch, field: str
) -> None:
    """r4 codex mechanization: every persisted data-identity field is binding —
    a stale value refuses the spend even with EVERY instrument field current."""
    _scratch_env(tmp_path, monkeypatch)
    rep = _full_axis_report(data_identity=jp.axis_data_identity())
    old = rep["data_identity"][field]
    rep["data_identity"][field] = "STALE-VALUE" if isinstance(old, str) else -999
    p = tmp_path / "r.json"
    p.write_text(json.dumps(rep))
    with pytest.raises(RuntimeError, match="DATA identity"):
        jp.require_pilot_pass(p, family="axis")


def test_require_pass_refuses_changed_manifest_bytes(tmp_path: Path, monkeypatch) -> None:
    """r4 codex mechanization: mutate ONLY the manifest content while holding
    every instrument field constant — the retained PASS is refused."""
    _scratch_env(tmp_path, monkeypatch)
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_full_axis_report(data_identity=jp.axis_data_identity())))
    (tmp_path / "panel_manifest.json").write_text(
        json.dumps({"axis_reservation_conv_ids": ["c1", "c2", "c3"], "n_reservation": 3})
    )
    with pytest.raises(RuntimeError, match="DATA identity"):
        jp.require_pilot_pass(p, family="axis")


def test_require_pass_refuses_changed_item_content(tmp_path: Path, monkeypatch) -> None:
    """Same-ID item re-generation (changed answer text) invalidates the PASS."""
    items_p = _scratch_env(tmp_path, monkeypatch)
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_full_axis_report(data_identity=jp.axis_data_identity())))
    items_p.write_text(
        "\n".join(
            json.dumps({"conv_id": c, "question": f"q {c}", "answer": f"REGENERATED {c}"})
            for c in ("c1", "c2")
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="DATA identity"):
        jp.require_pilot_pass(p, family="axis")


@pytest.mark.parametrize("field", sorted(jp.axis_instrument_fingerprint()))
def test_require_pass_refuses_each_stale_instrument_field(tmp_path: Path, field: str) -> None:
    """Mutate EVERY canonical instrument field: each stale value refuses the
    spend (the g6 scenario — a persisted PASS from an older instrument must
    never license today's production wave)."""
    rep = _full_axis_report()
    rep["instrument"][field] = "STALE-VALUE" if isinstance(rep["instrument"][field], str) else -999
    p = tmp_path / "r.json"
    p.write_text(json.dumps(rep))
    with pytest.raises(RuntimeError, match="stale PASS refused"):
        jp.require_pilot_pass(p, family="axis")


def test_require_pass_refuses_missing_instrument_field(tmp_path: Path) -> None:
    rep = _full_axis_report()
    del rep["instrument"]["rubric_sha256"]
    p = tmp_path / "r.json"
    p.write_text(json.dumps(rep))
    with pytest.raises(RuntimeError, match="stale PASS refused"):
        jp.require_pilot_pass(p, family="axis")


def test_require_pass_effective_draws_floor(tmp_path: Path, monkeypatch) -> None:
    _scratch_env(tmp_path, monkeypatch)
    identity = jp.axis_data_identity()
    # 60 - 20 = 40 answered < the 51-draw rule-26 satisfiability floor
    p = tmp_path / "under.json"
    p.write_text(json.dumps(_full_axis_report(n_draws=60, lost=20, data_identity=identity)))
    with pytest.raises(RuntimeError, match="effective draws"):
        jp.require_pilot_pass(p, family="axis")
    # exactly at the floor passes
    p2 = tmp_path / "at.json"
    p2.write_text(json.dumps(_full_axis_report(n_draws=60, lost=9, data_identity=identity)))
    assert jp.require_pilot_pass(p2, family="axis")["verdict"] == "PASS"


def test_require_pass_ingen_merged_instrument_and_floor(tmp_path: Path, monkeypatch) -> None:
    _scratch_env(tmp_path, monkeypatch)
    rep = _report("ingen", "PASS")
    rep["instrument"] = dict(jp.ingen_instrument_fingerprint())
    rep["data_identity"] = jp.ingen_data_identity()
    rep["arms"] = {"ingen": {"effective_draws": 120, "n_transport_lost": 0}}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(rep))
    assert jp.require_pilot_pass(p, family="ingen")["verdict"] == "PASS"
    rep["arms"]["ingen"]["effective_draws"] = 50  # < INGEN_MIN_EFFECTIVE=100
    p.write_text(json.dumps(rep))
    with pytest.raises(RuntimeError, match="effective draws"):
        jp.require_pilot_pass(p, family="ingen")


# ---------------------------------------------------------------------------
# (b) a failed pilot BLOCKS the real run_leg dispatch (codex mechanization)
# ---------------------------------------------------------------------------
def test_failed_pilot_blocks_run_leg_spend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fail_report = tmp_path / "pilot_gate_axis.json"
    fail_report.write_text(json.dumps(_report("axis", "FAIL")))
    monkeypatch.setenv(jl.SPEND_ACK_ENV, "1")
    monkeypatch.setenv(jp.REQUIRE_AXIS_PILOT_ENV, str(fail_report))
    out_dir = tmp_path / "legs"
    with pytest.raises(RuntimeError, match="production dispatch refused"):
        jl.run_leg(
            jl.LEG_AI_LIKENESS,
            [("ail_t_c1", "What?", "an answer long enough")],
            out_dir,
            "t",
            execute=True,
        )
    # The guard fired BEFORE any output/dispatch work: out_dir never created.
    assert not out_dir.exists()


def test_missing_pilot_blocks_run_leg_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(jl.SPEND_ACK_ENV, "1")
    monkeypatch.setenv(jp.REQUIRE_AXIS_PILOT_ENV, str(tmp_path / "absent.json"))
    with pytest.raises(RuntimeError, match="pilot gate report missing"):
        jl.run_leg(
            jl.LEG_AI_LIKENESS,
            [("ail_t_c1", "What?", "an answer long enough")],
            tmp_path / "legs",
            "t",
            execute=True,
        )


def test_stale_instrument_pass_blocks_run_leg_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """g6 scenario end-to-end: an OLD-rubric PASS report blocks the axis spend."""
    rep = _full_axis_report()
    rep["instrument"]["rubric_sha256"] = "0" * 64  # a prior rubric's fingerprint
    stale = tmp_path / "pilot_gate_axis.json"
    stale.write_text(json.dumps(rep))
    monkeypatch.setenv(jl.SPEND_ACK_ENV, "1")
    monkeypatch.setenv(jp.REQUIRE_AXIS_PILOT_ENV, str(stale))
    out_dir = tmp_path / "legs"
    with pytest.raises(RuntimeError, match="stale PASS refused"):
        jl.run_leg(
            jl.LEG_AI_LIKENESS,
            [("ail_t_c1", "What?", "an answer long enough")],
            out_dir,
            "t",
            execute=True,
        )
    assert not out_dir.exists()


def test_stale_data_pass_blocks_run_leg_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """r4: a PASS whose instrument is fully CURRENT but whose materialization
    drifted (item content changed under the same conv_ids) blocks the spend
    at run_leg's env guard — the exact channel the r3 fingerprint missed."""
    items_p = _scratch_env(tmp_path, monkeypatch)
    pilot = tmp_path / "pilot_gate_axis.json"
    pilot.write_text(json.dumps(_full_axis_report(data_identity=jp.axis_data_identity())))
    items_p.write_text(
        json.dumps({"conv_id": "c1", "question": "q c1", "answer": "REGENERATED c1"}) + "\n"
    )
    monkeypatch.setenv(jl.SPEND_ACK_ENV, "1")
    monkeypatch.setenv(jp.REQUIRE_AXIS_PILOT_ENV, str(pilot))
    out_dir = tmp_path / "legs"
    with pytest.raises(RuntimeError, match="DATA identity"):
        jl.run_leg(
            jl.LEG_AI_LIKENESS,
            [("ail_t_c1", "What?", "an answer long enough")],
            out_dir,
            "t",
            execute=True,
        )
    assert not out_dir.exists()


def test_instrument_gates_execute_refuses_stale_pilot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flatness/name-mask pilot-REUSE guard inherits the instrument binding:
    an --execute dispatch against a stale-budget PASS report refuses before any
    panel/leg work."""
    import issue2479_instrument_gates as ig

    rep = _full_axis_report()
    rep["instrument"]["max_tokens"] = 64  # a prior (undersized) budget
    stale = tmp_path / "pilot.json"
    stale.write_text(json.dumps(rep))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue2479_instrument_gates.py",
            "--step",
            "flatness",
            "--execute",
            "--axis-pilot-report",
            str(stale),
            "--legs-dir",
            str(tmp_path / "legs"),
            "--kept-glob",
            str(tmp_path / "kept_{variant}.jsonl"),
        ],
    )
    with pytest.raises(RuntimeError, match="stale PASS refused"):
        ig.main()


def test_instrument_gates_execute_refuses_stale_data_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The controls-reuse spend path refuses a current-instrument PASS whose
    materialization drifted (r4: manifest bytes changed after the pilot)."""
    import issue2479_instrument_gates as ig

    _scratch_env(tmp_path, monkeypatch)
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps(_full_axis_report(data_identity=jp.axis_data_identity())))
    (tmp_path / "panel_manifest.json").write_text(
        json.dumps({"axis_reservation_conv_ids": ["c9"], "n_reservation": 1})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue2479_instrument_gates.py",
            "--step",
            "flatness",
            "--execute",
            "--axis-pilot-report",
            str(pilot),
            "--legs-dir",
            str(tmp_path / "legs"),
            "--kept-glob",
            str(tmp_path / "kept_{variant}.jsonl"),
        ],
    )
    with pytest.raises(RuntimeError, match="DATA identity"):
        ig.main()


def test_env_absent_keeps_parent_dry_run_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No REQUIRE env, no spend ack: the parent's dry-run path runs end-to-end
    # (real body; judge_graded's own dry_run short-circuits before any API).
    monkeypatch.delenv(jp.REQUIRE_AXIS_PILOT_ENV, raising=False)
    monkeypatch.delenv(jl.SPEND_ACK_ENV, raising=False)
    out_dir = tmp_path / "legs"
    report = jl.run_leg(
        jl.LEG_AI_LIKENESS,
        [("ail_t_c1", "What?", "an answer long enough")],
        out_dir,
        "t",
        execute=False,
    )
    assert report["spend_executed"] is False
    assert (out_dir / "judge_report_ail_t.json").is_file()


def test_pass_pilot_does_not_block_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Env SET but no real spend: the guard is scoped to allowed-spend only.
    fail_report = tmp_path / "pilot_gate_axis.json"
    fail_report.write_text(json.dumps(_report("axis", "FAIL")))
    monkeypatch.setenv(jp.REQUIRE_AXIS_PILOT_ENV, str(fail_report))
    monkeypatch.delenv(jl.SPEND_ACK_ENV, raising=False)
    report = jl.run_leg(
        jl.LEG_AI_LIKENESS,
        [("ail_t_c1", "What?", "an answer long enough")],
        tmp_path / "legs",
        "t",
        execute=False,
    )
    assert report["spend_executed"] is False


# ---------------------------------------------------------------------------
# (c) in-gen family merge gate arithmetic
# ---------------------------------------------------------------------------
_PARTIAL_IDENTITY = {
    "panel_sha256": "p" * 64,
    "panel_manifest_sha256": "m" * 64,
    "gen_module_sha256": "g" * 64,
}


def _partial(
    tmp_path: Path,
    name: str,
    outcomes: list[dict],
    *,
    max_tokens: int = 1024,
    data_identity: dict | None = _PARTIAL_IDENTITY,
) -> Path:
    p = tmp_path / f"partial_{name}.json"
    p.write_text(
        json.dumps(
            {
                "family": "ingen",
                "kind": "partial",
                "character": name,
                "instrument": {
                    "judge_model": "claude-sonnet-4-5-20250929",
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
                "data_identity": data_identity,
                "outcomes": outcomes,
            }
        )
    )
    return p


def _ok(n: int, prefix: str) -> list[dict]:
    return [
        {"item_id": f"{prefix}{i}", "error": False, "category": "ok", "stop_reason": "end_turn"}
        for i in range(n)
    ]


def test_merge_all_ok_passes(tmp_path: Path) -> None:
    parts = [
        _partial(tmp_path, "iris", _ok(3, "a")),
        _partial(tmp_path, "vex", _ok(3, "b")),
    ]
    rep = jp.merge_ingen_partials(parts, tmp_path / "rep.json", min_effective=4)
    assert rep["verdict"] == "PASS" and rep["passed"] is True
    assert rep["arms"]["ingen"]["effective_draws"] == 6
    assert (tmp_path / "rep.json").is_file()


def test_merge_single_truncation_fails(tmp_path: Path) -> None:
    outcomes = _ok(5, "a")
    outcomes.append(
        {"item_id": "a5", "error": False, "category": "ok", "stop_reason": "max_tokens"}
    )
    rep = jp.merge_ingen_partials(
        [_partial(tmp_path, "iris", outcomes)], tmp_path / "rep.json", min_effective=4
    )
    assert rep["verdict"] == "FAIL"
    assert any("max_tokens" in f for f in rep["failures"])


def test_merge_parse_fail_rate_fails(tmp_path: Path) -> None:
    outcomes = _ok(19, "a")
    outcomes.append(
        {"item_id": "a19", "error": True, "category": "error", "stop_reason": "end_turn"}
    )  # 1/20 = 5% >= 2%
    rep = jp.merge_ingen_partials(
        [_partial(tmp_path, "iris", outcomes)], tmp_path / "rep.json", min_effective=4
    )
    assert rep["verdict"] == "FAIL"
    assert any("parse-fail" in f for f in rep["failures"])


def test_merge_hollow_arm_fails(tmp_path: Path) -> None:
    rep = jp.merge_ingen_partials(
        [_partial(tmp_path, "iris", _ok(3, "a"))], tmp_path / "rep.json", min_effective=100
    )
    assert rep["verdict"] == "FAIL"
    assert any("hollow" in f for f in rep["failures"])


def test_merge_transport_excluded_from_denominators(tmp_path: Path) -> None:
    outcomes = _ok(4, "a")
    outcomes.append(
        {
            "item_id": "a4",
            "error": True,
            "category": "transport_exhausted",
            "stop_reason": None,
        }
    )
    rep = jp.merge_ingen_partials(
        [_partial(tmp_path, "iris", outcomes)], tmp_path / "rep.json", min_effective=4
    )
    assert rep["verdict"] == "PASS"  # transport draw excluded, 4 answered clean
    assert rep["arms"]["ingen"]["n_transport_lost"] == 1
    assert rep["arms"]["ingen"]["effective_draws"] == 4
    assert rep["warnings"]


def test_merge_mixed_instruments_refused(tmp_path: Path) -> None:
    parts = [
        _partial(tmp_path, "iris", _ok(2, "a"), max_tokens=1024),
        _partial(tmp_path, "vex", _ok(2, "b"), max_tokens=600),
    ]
    with pytest.raises(AssertionError, match="DIFFERENT instruments"):
        jp.merge_ingen_partials(parts, tmp_path / "rep.json", min_effective=2)


def test_merge_mixed_data_identities_refused(tmp_path: Path) -> None:
    """r4: partials piloted over DIFFERENT materializations never merge."""
    parts = [
        _partial(tmp_path, "iris", _ok(2, "a")),
        _partial(
            tmp_path,
            "vex",
            _ok(2, "b"),
            data_identity={**_PARTIAL_IDENTITY, "panel_sha256": "x" * 64},
        ),
    ]
    with pytest.raises(AssertionError, match="DIFFERENT data identities"):
        jp.merge_ingen_partials(parts, tmp_path / "rep.json", min_effective=2)


def test_merge_partials_without_data_identity_refused(tmp_path: Path) -> None:
    """r4: pre-r4 partials (no data_identity block) must be regenerated."""
    parts = [_partial(tmp_path, "iris", _ok(2, "a"), data_identity=None)]
    with pytest.raises(AssertionError, match="no data_identity block"):
        jp.merge_ingen_partials(parts, tmp_path / "rep.json", min_effective=2)


def test_merge_persists_data_identity(tmp_path: Path) -> None:
    rep = jp.merge_ingen_partials(
        [_partial(tmp_path, "iris", _ok(3, "a"))], tmp_path / "rep.json", min_effective=2
    )
    assert rep["data_identity"] == _PARTIAL_IDENTITY


# ---------------------------------------------------------------------------
# (d) run_ingen_partial production body (fake ONLY at the API boundary)
# ---------------------------------------------------------------------------
def test_run_ingen_partial_body_real_builder_and_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import issue1345_common as c

    from explore_persona_space.llm import api_dispatch as ad

    _scratch_env(tmp_path, monkeypatch)  # hermetic ingen data-identity inputs (r4)
    monkeypatch.setenv(jl.SPEND_ACK_ENV, "1")
    raw = tmp_path / "raw_stories_paired_instruct.jsonl"
    rows = [
        {"conv_id": "s1", "story": "A tale. X asked. X replied.", "mode": "op"},
        {
            "conv_id": "s2",
            "story": "Another tale with a quoted line.",
            "answer": "the required answer",
            "mode": "paired",
        },
    ]
    raw.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    seen_requests: list[dict] = []

    # Def-mirroring async fake at the EXTERNAL API boundary only: it runs the
    # gen module's REAL _build_judge_request + _parse_judge_response per item.
    async def fake_dispatch_calls(
        items,
        *,
        model,
        build_request,
        parse_response,
        cache_dir=None,
        checkpoint_dir=None,
        force_path=None,
        **kwargs,
    ):
        out = {}
        for it in items:
            req = build_request(it)
            assert req["model"] == c.JUDGE_MODEL
            assert req["max_tokens"] == c.JUDGE_MAX_TOKENS
            assert req["temperature"] == 0.0
            assert req["system"] and req["messages"]
            seen_requests.append({k: req[k] for k in ("model", "max_tokens", "temperature")})
            parsed = parse_response("Looks fine.\nEXCHANGES: 1\nVERDICT: PASS")
            out[it.item_id] = ad.DispatchResult(
                item_id=it.item_id,
                result=parsed,
                error=False,
                category=ad.RESULT_OK,
                stop_reason="end_turn",
            )
        return out

    monkeypatch.setattr(ad, "dispatch_calls", fake_dispatch_calls)
    partial_out = tmp_path / "partial.json"
    payload = jp.run_ingen_partial(
        [raw], partial_out, tmp_path / "pilot_cache", n_target=10, execute=True
    )
    assert partial_out.is_file()
    assert payload["n_judged"] == 2 and len(seen_requests) == 2
    assert all(not o["error"] for o in payload["outcomes"])
    assert all(o["stop_reason"] == "end_turn" for o in payload["outcomes"])
    assert "judge_system_paired_sha256" in payload["instrument"]
    # r4: the partial binds the materialization it piloted over.
    assert payload["data_identity"] == jp.ingen_data_identity()
    # Content hygiene: the partial carries OUTCOMES only, never story text.
    assert "story" not in json.dumps(payload)


def test_run_ingen_partial_refuses_without_spend_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(jl.SPEND_ACK_ENV, raising=False)
    with pytest.raises(AssertionError, match="refused"):
        jp.run_ingen_partial(
            [tmp_path / "x.jsonl"], tmp_path / "p.json", tmp_path / "c", n_target=5, execute=True
        )
