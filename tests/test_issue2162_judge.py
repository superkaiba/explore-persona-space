"""Issue #2162 judge pipeline — CPU pins (no API calls).

Covers: the dynamic rubric registry (ids + instrument shape), the rule-27
parse-contract ROUND-TRIP for the composed instruments (realistic reply +
fenced variant through the harness's OWN ``parse_judge_json`` →
``_score_from_parsed`` path, plus the placeholder-substitution presence
check), the dual-rubric grid item builder (filler_swap skip, id grammar),
the anchor-behavior dedup + the gate-slice/production shared-id invariant
(the JudgeCache join), and the gate-3 separation verdict arithmetic.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2094_judge as J94  # noqa: E402
import issue2162_judge as J  # noqa: E402

from explore_persona_space.eval.graded_judge import _score_from_parsed  # noqa: E402
from explore_persona_space.eval.utils import parse_judge_json  # noqa: E402
from explore_persona_space.experiments.issue2162 import bank2162 as B  # noqa: E402


@pytest.fixture(scope="module")
def pairs():
    return B.build_pairs()


@pytest.fixture(scope="module")
def registry(pairs):
    return J.rubric_registry(pairs)


# ── rubric registry ───────────────────────────────────────────────────


def test_rubric_registry_shape(pairs, registry):
    assert J94.COHERENCE_RUBRIC_ID in registry
    assert len(registry) > 50  # every distinct value-descriptor core
    for rid, prompt in registry.items():
        if rid == J94.COHERENCE_RUBRIC_ID:
            continue
        assert J94.ITEM_ID_RE.match(rid), rid
        assert "[RESPONSE]" in prompt and "{answer}" in prompt
        assert "score" in prompt  # the REASON_THEN_SCORE contract
    # Every non-filler pair's two rubric ids resolve in the registry.
    for p in pairs[:100]:
        cores = J.pair_rubric_cores(p)
        if cores is None:
            assert B.base_type_of(p.cell) == "filler_swap"
            continue
        for core in cores:
            assert J.rubric_core_id(core) in registry


def test_query_family_detected(pairs, registry):
    qc = next(p for p in pairs if p.cell == "query_content")
    ra, rb = (J.rubric_core_id(c) for c in J.pair_rubric_cores(qc))
    assert J._is_query_rubric(registry[ra]) and J._is_query_rubric(registry[rb])
    fmt = next(p for p in pairs if p.cell == "instr_format")
    fa, _fb = (J.rubric_core_id(c) for c in J.pair_rubric_cores(fmt))
    assert not J._is_query_rubric(registry[fa])


# ── rule-27 parse-contract round-trip ─────────────────────────────────


def test_parse_contract_roundtrip(registry):
    """Realistic reply + fenced variant through the harness's OWN parse path."""
    plain = (
        "The response consistently uses bullet points throughout, matching the "
        'target property. {"reasoning": "bullets used throughout", "score": 87}'
    )
    fenced = '```json\n{"reasoning": "no trace of the property", "score": 3}\n```'
    for reply, expected in ((plain, 87.0), (fenced, 3.0)):
        parsed = parse_judge_json(reply)
        score = _score_from_parsed(parsed)
        assert score == expected, (reply[:40], parsed, score)
    # Malformed / out-of-range returns DROP (None), never coerce (rule 9).
    assert _score_from_parsed(parse_judge_json("REFUSAL")) is None
    assert _score_from_parsed(parse_judge_json('{"reasoning": "x", "score": 250}')) is None


def test_placeholder_substitution_fills_every_slot(registry):
    """Harness-identical substitution (graded_judge.format_user_msg's .replace)
    leaves no unfilled slot in any composed instrument."""
    for rid, prompt in registry.items():
        filled = prompt.replace("{question}", "q?").replace("{answer}", "the reply")
        assert "{answer}" not in filled and "{question}" not in filled, rid
        assert "the reply" in filled, rid


# ── item builders ─────────────────────────────────────────────────────


def _grid_row(pair: B.Pair2162, draw: int = 0) -> dict:
    return {
        "block_key": f"{pair.cell}|ce|steered",
        "cell": pair.cell,
        "slot": "ce",
        "arm": "steered",
        "pair_id": pair.pair_id,
        "draw": draw,
        "text": "a reply",
        "context_id": pair.a,
    }


def _anchor_row(cid: str, cell: str, value_id: str, carrier: str, draw: int = 0) -> dict:
    return {
        "context_id": cid,
        "cell": cell,
        "value_id": value_id,
        "carrier": carrier,
        "draw": draw,
        "text": "an anchor reply",
    }


def test_grid_behavior_items_dual_rubric_and_filler_skip(pairs):
    fmt = next(p for p in pairs if p.cell == "instr_format")
    filler = next(p for p in pairs if B.base_type_of(p.cell) == "filler_swap")
    pairs_by_id = {p.pair_id: p for p in (fmt, filler)}
    rows = [_grid_row(fmt, 0), _grid_row(fmt, 1), _grid_row(filler)]
    by_rid = J.build_grid_behavior_items(rows, pairs_by_id)
    ra, rb = (J.rubric_core_id(c) for c in J.pair_rubric_cores(fmt))
    assert set(by_rid) == {ra, rb}  # filler_swap contributed nothing
    assert len(by_rid[ra]) == len(by_rid[rb]) == 2
    all_units = [u for us in by_rid.values() for u in us]
    ids = [u.item_id for u in all_units]
    assert len(set(ids)) == len(ids)
    J94._validate_units(all_units)
    assert {u.source["side"] for u in all_units} == {"a", "b"}


def test_anchor_behavior_dedup_and_gate_join(pairs):
    """A context in 2 pairs gets the UNION of cores once per (ctx, draw, rid),
    and the gate-slice restriction yields a SUBSET of the production ids —
    the rubric-keyed JudgeCache join that makes the gate spend reusable."""
    by_cell = B.pairs_by_cell(pairs)
    cell_pairs = by_cell["instr_format"]
    ctx = cell_pairs[0].a
    member_pairs = [p for p in cell_pairs if ctx in (p.a, p.b)]
    assert len(member_pairs) >= 2  # value cycle: every context sits in >= 2 pairs
    row = _anchor_row(ctx, "instr_format", "v1", "d1")
    full = J.build_anchor_behavior_items([row], pairs)
    full_ids = {u.item_id for us in full.values() for u in us}
    expected_cores = set()
    for p in member_pairs:
        expected_cores.update(J.pair_rubric_cores(p))
    assert len(full_ids) == len(expected_cores)  # dedup per (ctx, draw, rid)
    restricted = J.build_anchor_behavior_items([row], pairs, restrict_pairs=member_pairs[:1])
    restricted_ids = {u.item_id for us in restricted.values() for u in us}
    assert restricted_ids <= full_ids  # gate ids are production ids (cache join)
    for us in full.values():
        for u in us:
            assert u.item_id == J.anchor_unit_id(ctx, 0, u.rubric_id)


# ── gate-3 separation verdict arithmetic ──────────────────────────────


def test_separation_verdict_math(pairs):
    gate_pairs = [p for p in B.gate_slice_pairs(pairs) if p.cell in ("instr_format", "verbosity")]
    cells = {p.cell for p in gate_pairs}
    assert len(gate_pairs) == 6 * len(cells)
    draws = [0, 1]
    anchor_rows = []
    seen = set()
    for p in gate_pairs:
        for ctx in (p.a, p.b):
            if ctx in seen:
                continue
            seen.add(ctx)
            anchor_rows.extend(_anchor_row(ctx, p.cell, "v1", "d1", draw=d) for d in draws)
    # Fabricate scores: instr_format separates perfectly (sep=+1); stated_pref
    # scores identical under both rubrics (sep=0 -> fails the 0.5 bar).
    scores: dict[str, float] = {}
    for p in gate_pairs:
        ra, rb = (J.rubric_core_id(c) for c in J.pair_rubric_cores(p))
        perfect = p.cell == "instr_format"
        for d in draws:
            # floor ctx A: expresses A (judge_a=100, judge_b=0) when separable
            scores[J.anchor_unit_id(p.a, d, ra)] = 100.0 if perfect else 50.0
            scores[J.anchor_unit_id(p.a, d, rb)] = 0.0 if perfect else 50.0
            # ceiling ctx B: expresses B
            scores[J.anchor_unit_id(p.b, d, ra)] = 0.0 if perfect else 50.0
            scores[J.anchor_unit_id(p.b, d, rb)] = 100.0 if perfect else 50.0
    report = J.separation_verdict(gate_pairs, anchor_rows, scores)
    pc = report["per_cell"]
    assert pc["instr_format"]["n_pairs"] == 6 and pc["instr_format"]["n_pass"] == 6
    assert pc["instr_format"]["cell_pass"] is True
    assert pc["verbosity"]["n_pass"] == 0 and pc["verbosity"]["cell_pass"] is False
    assert report["frac_cells_pass"] == pytest.approx(0.5)
    assert report["passed"] is False  # 50% < 60% bar
    assert report["catastrophic"] is False  # 50% >= 25%
    seps = {r["cell"]: r["sep"] for r in report["pairs"]}
    assert seps["instr_format"] == pytest.approx(2.0)  # (+1) - (-1)
    assert seps["verbosity"] == pytest.approx(0.0)


# ── margin pools (r1 C4) ─────────────────────────────────────────────


def test_pool_key_cross_pin_matches_run_module():
    """The judge's pool builder and the run driver's pool consumer MUST agree
    byte-for-byte on the pool key grammar (r1 C4) — a drift silently empties
    every margin lookup."""
    import issue2162_run as R

    for p in B.build_pairs():
        assert J.pool_key(p) == R.pool_key(p)


def test_pool_constants_pin():
    """Plan §4.4 / llm-judging rule 19: fixed 4+4 judge-filtered (>50) pools."""
    assert J.POOL_PER_SIDE == 4
    assert J.POOL_FILTER_MIN == 50.0


def test_build_margin_pools_selects_filtered_top4():
    """Pools take the top-4 per side by score among >50-filtered anchor draws;
    a side with zero survivors OMITS the key; 1-3 survivors keep it flagged
    short."""
    pairs = [p for p in B.build_pairs() if p.cell == "instr_format"][:2]
    cores = J.pair_rubric_cores(pairs[0])
    assert cores is not None
    rid_a, rid_b = (J.rubric_core_id(c) for c in cores)
    anchor_rows = []
    scores = {}
    for p in pairs[:1]:
        for d in range(6):
            anchor_rows.append({"context_id": p.a, "cell": p.cell, "draw": d, "text": f"floor {d}"})
            anchor_rows.append({"context_id": p.b, "cell": p.cell, "draw": d, "text": f"ceil {d}"})
            # Side A candidates: floor ctx scored under rid_a; two fail the
            # >50 filter, the rest rank by score.
            scores[(p.a, d, rid_a)] = [90, 80, 70, 60, 40, 30][d]
            scores[(p.b, d, rid_b)] = [95, 85, 75, 65, 55, 45][d]
    pools, _report = J.build_margin_pools(pairs[:1], anchor_rows, scores)
    key = J.pool_key(pairs[0])
    assert key in pools
    sides = {"A": [], "B": []}
    for item in pools[key]:
        sides[item["side"]].append(item["text"])
    assert len(sides["A"]) == 4 and len(sides["B"]) == 4
    assert sides["A"][0] == "floor 0"  # top score first (90)
    assert sides["B"] == [f"ceil {d}" for d in range(4)]


def test_build_margin_pools_query_content_skipped_and_reported(pairs):
    """FIX 4: ``query_content`` pairs build NO pool — the manipulated variable
    IS the user query, so rubric cores vary per carrier AND each pair's two
    contexts pose different queries; no fixed shared answer pool (plan §4.4 TF
    margin) is well-defined at any key granularity. Pre-fix this fixture shape
    tripped the pool-key rubric-core assert (`('query_content|v1-v2', ...)`).
    The skip is explicit — counted + reasoned in the report — and
    ``filler_swap`` behavior is unchanged (covered above / in the item tests)."""
    qc = [p for p in pairs if B.base_type_of(p.cell) == "query_content"]
    shared = [p for p in qc if (p.value_a, p.value_b) == (qc[0].value_a, qc[0].value_b)][:2]
    assert len(shared) == 2 and shared[0].carrier != shared[1].carrier
    fmt = next(p for p in pairs if p.cell == "instr_format")
    cores = J.pair_rubric_cores(fmt)
    assert cores is not None
    rid_a, rid_b = (J.rubric_core_id(c) for c in cores)
    anchor_rows, scores = [], {}
    for d in range(4):
        anchor_rows.append({"context_id": fmt.a, "cell": fmt.cell, "draw": d, "text": f"floor {d}"})
        anchor_rows.append({"context_id": fmt.b, "cell": fmt.cell, "draw": d, "text": f"ceil {d}"})
        scores[(fmt.a, d, rid_a)] = 90.0
        scores[(fmt.b, d, rid_b)] = 90.0
    # Pre-fix: AssertionError('pairs sharing a pool key disagree on rubric cores').
    pools, report = J.build_margin_pools([fmt, *shared], anchor_rows, scores)
    key_qc = J.pool_key(shared[0])
    assert key_qc not in pools
    assert J.pool_key(fmt) in pools
    skip = report["query_content_skip"]
    assert skip["n_keys"] == 1 and skip["keys"] == [key_qc]
    assert "query" in skip["reason"] and "filler_swap" in skip["reason"]
    # Skipped keys never enter the built-key accounting (add-only report).
    assert report["n_keys_total"] == 1


# ── phase-aware input staging (fix 2) ─────────────────────────────────


def _stage_args(phase: str, in_root: Path) -> argparse.Namespace:
    return argparse.Namespace(phase=phase, in_root=in_root, hf_revision=None)


def test_stage_inputs_phase_aware(monkeypatch, tmp_path):
    """FIX 2: each phase stages only what its loaders read; a not-yet-existing
    tolerated prefix logs-and-continues; a genuinely REQUIRED prefix still
    raises. Pre-fix, ``--stage-from-hf --phase separation-gate`` could NEVER
    work: it unconditionally staged the grid prefix, which exists only from
    P3 — AFTER gate 3 (observed live FileNotFoundError on
    ``issue2162_ctxinfo/raw_completions/grid``)."""
    from explore_persona_space.orchestrate import hub

    present = {J._STAGE_ANCHORS_GATE}
    requested: list[str] = []

    def fake_stage(
        repo_id,
        prefix,
        dest_dir,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        max_workers=6,
    ):
        requested.append(prefix)
        if prefix not in present:
            raise FileNotFoundError(prefix)
        return [Path(dest_dir) / prefix / "anchors_gate_w0.jsonl"]

    monkeypatch.setattr(hub, "stage_hub_prefix", fake_stage)

    # separation-gate BEFORE the P3 grid + terminal anchors uploads: works
    # off the early-uploaded gate slice; never requests grid.
    J._stage_inputs(_stage_args("separation-gate", tmp_path))
    assert J._STAGE_GRID not in requested
    assert J._STAGE_ANCHORS_GATE in requested

    # waves genuinely REQUIRES grid rows: absent -> raises (fail-loud kept).
    with pytest.raises(FileNotFoundError):
        J._stage_inputs(_stage_args("waves", tmp_path))

    # separation-gate with NO anchor prefix landed at all: raises, named.
    present.clear()
    with pytest.raises(FileNotFoundError, match="anchor rows"):
        J._stage_inputs(_stage_args("separation-gate", tmp_path))


def test_stage_plan_covers_every_phase():
    """A new phase must declare its staging needs (or explicitly nothing)."""
    assert set(J._PHASE_STAGE_PLAN) == set(J.PHASES)


def test_resolve_anchors_dir_prefers_full_then_gate(tmp_path):
    """FIX 2 consumption side: the default anchors dir falls back to the
    early-uploaded ``anchors_gate`` mirror when the full ``anchors`` prefix
    has not landed yet, so the staged gate slice is actually READ by
    ``load_anchor_rows`` (glob ``anchors_*.jsonl``)."""
    mirror = tmp_path
    (mirror / "anchors").mkdir()
    (mirror / "anchors_gate").mkdir()
    # Neither holds shards: canonical path (loaders fail loud downstream).
    assert J._resolve_anchors_dir(mirror) == mirror / "anchors"
    # Only the early gate slice staged: fall back to it.
    (mirror / "anchors_gate" / "anchors_gate_w0.jsonl").write_text("{}\n")
    assert J._resolve_anchors_dir(mirror) == mirror / "anchors_gate"
    # Full anchors staged: preferred (it includes the gate shards too).
    (mirror / "anchors" / "anchors_rest_w0.jsonl").write_text("{}\n")
    assert J._resolve_anchors_dir(mirror) == mirror / "anchors"


# ── uniform --dry-run (fix 3) ─────────────────────────────────────────


def _cfg(tmp_path: Path, **kw) -> J94.JudgeConfig:
    return J94.JudgeConfig(
        work_root=tmp_path / "work",
        cache_root=tmp_path / "cache",
        rollouts_dir=tmp_path / "grid",
        anchors_file=tmp_path / "anchors",
        stage2_dir=None,
        **kw,
    )


def _forbid(monkeypatch, *targets):
    """Any API-bearing / persisting call under --dry-run is a test failure."""

    def boom(*a, **k):
        raise AssertionError("dry-run must not reach the API or persist anything")

    for mod, name in targets:
        monkeypatch.setattr(mod, name, boom)


def test_phase_pilot_dry_run_refuses_loudly(monkeypatch, tmp_path, caplog):
    """FIX 3: the pilot's purpose is measuring the REAL truncation/parse-fail
    profile, so --dry-run REFUSES with a distinct rc instead of silently
    spending (pre-fix: ``--phase pilot --dry-run`` made ~356 real calls and
    wrote a real verdict)."""
    _forbid(monkeypatch, (J, "judge_pilot_gate"), (J, "judge_graded"), (J.BANK, "build_pairs"))
    cfg = _cfg(tmp_path, dry_run=True)
    with caplog.at_level(logging.ERROR):
        rc = J.phase_pilot(cfg)
    assert rc == J.RC_DRY_RUN_UNSUPPORTED != 0
    assert any("--dry-run refused" in r.message for r in caplog.records)
    assert not (tmp_path / "work").exists()  # no persisted verdict


def test_phase_anchors_dry_run_zero_api_zero_writes(monkeypatch, tmp_path):
    """FIX 3: anchors --dry-run is a pure construction check handled at ENTRY
    (pre-fix the flag was only checked after the first ``run_wave`` call) —
    zero API calls, nothing persisted (no audits, no items, no gate verdicts),
    and the gate-presence requirement does not fire on a dry run."""
    _forbid(monkeypatch, (J94, "run_wave"), (J94, "run_audits"), (J, "judge_graded"))
    anchors = tmp_path / "anchors"
    anchors.mkdir()
    row = _anchor_row("nonexistent_ctx", "instr_format", "v1", "d1")
    (anchors / "anchors_rest_w0.jsonl").write_text(json.dumps(row) + "\n")
    cfg = _cfg(tmp_path, dry_run=True)
    assert J.phase_anchors(cfg) == J.RC_OK
    assert not (tmp_path / "work").exists()  # nothing persisted


def test_dry_run_units_report_validates_and_is_pure(tmp_path, caplog):
    """The shared dry-run helper validates units (a malformed unit set still
    fails loud) and logs the uniform no-API / no-persist line."""
    unit = J94.JudgeUnit(
        item_id=J94._item_id("c", "x"),
        rubric_id=J94.COHERENCE_RUBRIC_ID,
        question="",
        answer="a reply",
        source={"kind": "anchor"},
    )
    with caplog.at_level(logging.INFO):
        rc = J._dry_run_units_report("anchors", {"coherence.anchors": [unit]})
    assert rc == J.RC_OK
    assert any("no API calls made, nothing persisted" in r.message for r in caplog.records)
    with pytest.raises(ValueError, match="duplicate item ids"):
        J._dry_run_units_report("anchors", {"w": [unit, unit]})
