#!/usr/bin/env python
"""Issue #2333 VM-side judge pipeline — snowball test (plan §4.5).

Thin adaptation of ``scripts/issue2162_judge.py`` over the #2333 row schema:

- Coherence + dual-rubric behavior items for every grid rollout
  (``response_text``: prefill arms = donor opening + continuation).
- Prefill CONTINUATION-ONLY behavior companions (tag ``n``, judged on
  ``continuation_text`` — plan §6 exploratory read).
- SAME-WAVE ce calibration items: banked #2162 ce grid rows (S1, tag ``k``)
  and banked #2094 fu1 conf1 joint_all rows (S2, tag ``m``) are re-judged in
  the SAME per-rubric waves as the fresh items, so every recovery ratio
  R_k = F̄_arm / F̄_ce shares one judge instrument (plan §5).
- q35 anchors: coherence-baseline gate then per-rubric anchor waves
  (q25 reuses the BANKED #2162/#2094 anchor scores — no fresh q25 anchors).
- Rule-26 pilot gate FIRST (per rubric family + live forced-batch probe);
  judge = ``claude-sonnet-4-5-20250929`` via the Batch API, max_tokens 1024.

REUSES ``issue2094_judge`` (J94: JudgeUnit / run_wave / caches / pilot /
coherence gate) and ``issue2162_judge`` (J62: rubric ids, anchor unit ids,
dry-run report) — never re-implements them (plan §10 fitness map).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import issue2094_judge as J94  # noqa: E402
import issue2162_judge as J62  # noqa: E402

from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402
from explore_persona_space.eval.judge_pilot import judge_pilot_gate  # noqa: E402
from explore_persona_space.experiments.issue2094 import bank as BANK94  # noqa: E402
from explore_persona_space.experiments.issue2162 import bank2162 as BANK2162  # noqa: E402
from explore_persona_space.experiments.issue2333 import constants as C  # noqa: E402

logger = logging.getLogger("issue2333.judge")

RC_OK = 0
RC_PILOT_GATE = J62.RC_PILOT_GATE  # 7
RC_COHERENCE_GATE = J62.RC_COHERENCE_GATE  # 8
RC_DRY_RUN_UNSUPPORTED = J62.RC_DRY_RUN_UNSUPPORTED  # 10

JUDGE_N_DRAWS = J94.JUDGE_N_DRAWS  # 1 draw per item (parent convention)
PILOT_SEED = 23330


# ── pairs / rubrics (both banks) ──────────────────────────────────────


def build_pair_universe() -> tuple[list, list]:
    """(s1_pairs, s2_pairs) — same filter as issue2333_run.build_pair_universe."""
    s1 = [p for p in BANK2162.build_pairs() if p.cell in C.S1_CELLS]
    assert len(s1) == len(C.S1_CELLS) * C.S1_PAIRS_PER_CELL, len(s1)
    s2 = [p for p in BANK94.build_pairs() if p.setting == "matched_query"]
    assert len(s2) == 15, len(s2)
    return s1, s2


def pair_set_of(pair) -> str:
    return "s1" if hasattr(pair, "cell") else "s2"


def pair_rubric_cores_2333(pair) -> tuple[str, str]:
    """(core_A, core_B) per pair set. S1 = #2162 rubric pair (no filler_swap in
    the survivor cells, so never None); S2 = the #2094 matched-query 'prefix'
    rubric kind (SETTING_RUBRIC_KINDS['matched_query'] == ('prefix',))."""
    if pair_set_of(pair) == "s1":
        cores = BANK2162.rubric_pair_2162(pair)
    else:
        cores = BANK94.rubric_pair(pair, "prefix")
    assert cores is not None and len(cores) == 2, pair
    return cores


def rubric_registry(s1_pairs: list, s2_pairs: list) -> dict[str, str]:
    """rubric_id -> eval prompt (coherence + every distinct core, both banks)."""
    reg = {J94.COHERENCE_RUBRIC_ID: J94.coherence_eval_prompt()}
    for p in [*s1_pairs, *s2_pairs]:
        for core in pair_rubric_cores_2333(p):
            reg.setdefault(J62.rubric_core_id(core), J94.behavior_eval_prompt(core))
    return reg


# ── config ────────────────────────────────────────────────────────────


@dataclass
class JudgeConfig2333:
    """J94.JudgeConfig plus the #2333-specific input roots."""

    base: J94.JudgeConfig
    model_tag: str
    rollouts_dir: Path  # holds blocks/*.jsonl (+ ce_control/*.jsonl on q35)
    anchors_dir: Path | None
    calib_dir: Path

    @property
    def dry_run(self) -> bool:
        return self.base.dry_run


# ── input walkers (2333 row schemas) ──────────────────────────────────


def assert_shard_set_complete(found_slugs: set[str], expected_slugs: set[str], what: str) -> None:
    """Pre-spend completeness gate (r1 Major 3): the judge NEVER dispatches a
    Batch-API wave against a partial/overfull shard set — a missing block
    would silently censor whole (cell, arm, variant) cells from the F tables.
    Raises naming the exact missing/extra slugs."""
    missing = sorted(expected_slugs - found_slugs)
    extra = sorted(found_slugs - expected_slugs)
    if missing or extra:
        raise RuntimeError(
            f"{what} shard set incomplete: {len(found_slugs)}/{len(expected_slugs)} expected; "
            f"missing={missing[:8]}{'...' if len(missing) > 8 else ''} "
            f"extra={extra[:8]}{'...' if len(extra) > 8 else ''} — "
            "re-run/finish the pod grid (or re-stage) before ANY judge spend"
        )


def assert_draw_consistency(rows: list[dict]) -> int:
    """Every (block_key, pair_id) group carries the SAME draw set {0..K-1}.
    Returns K. A ragged group means a partially-written/mixed-regime shard."""
    by_group: dict[tuple[str, str], set[int]] = {}
    for r in rows:
        by_group.setdefault((r["block_key"], r["pair_id"]), set()).add(int(r["draw"]))
    draw_sets = {frozenset(s) for s in by_group.values()}
    assert len(draw_sets) == 1, (
        f"ragged per-(block, pair) draw sets: {sorted(map(sorted, draw_sets))[:4]}"
    )
    draws = next(iter(draw_sets))
    k = len(draws)
    assert draws == frozenset(range(k)), sorted(draws)
    return k


def load_grid_rows(rollouts_dir: Path, expect_complete: bool = True) -> list[dict]:
    shards = sorted((rollouts_dir / "blocks").glob("*.jsonl"))
    assert shards, f"no grid block shards under {rollouts_dir}/blocks"
    if expect_complete:
        assert_shard_set_complete(
            {s.stem for s in shards}, C.expected_grid_slugs(), "grid (144-block)"
        )
    rows = [r for s in shards for r in J94._iter_jsonl(s)]
    assert rows, "grid shards present but empty"
    for r in rows[:1]:
        for key in ("block_key", "pair_id", "draw", "response_text", "kind", "variant", "cell"):
            assert key in r, (key, sorted(r))
    k = assert_draw_consistency(rows)
    logger.info("[load] grid: %d shards, %d rows, K=%d", len(shards), len(rows), k)
    return rows


def load_ce_rows(rollouts_dir: Path, expect_complete: bool = True) -> list[dict]:
    shards = sorted((rollouts_dir / "ce_control").glob("*.jsonl"))
    assert shards, f"no ce_control shards under {rollouts_dir}/ce_control"
    if expect_complete:
        assert_shard_set_complete(
            {s.stem for s in shards}, C.expected_ce_control_slugs(), "ce_control (12-block)"
        )
    rows = [r for s in shards for r in J94._iter_jsonl(s)]
    assert rows, "ce_control shards present but empty"
    assert_draw_consistency(rows)
    return rows


def load_anchor_rows(anchors_dir: Path) -> list[dict]:
    files = sorted(anchors_dir.glob("anchors_w*.jsonl"))
    assert files, f"no anchor shards under {anchors_dir}"
    rows = [r for f in files for r in J94._iter_jsonl(f)]
    assert rows, "anchor shards present but empty"
    for r in rows[:1]:
        for key in ("context_id", "draw", "response_text"):
            assert key in r, (key, sorted(r))
    return rows


def load_calib_s1(calib_dir: Path) -> list[dict]:
    """Banked #2162 ce grid rows (survivor cells x {steered, shuffled}).

    Observed schema (probe 2026-08-16, @ PIN_2162): keys incl. cell / slot /
    arm / pair_id / draw / text / context_id / block_key.
    """
    shards = sorted((calib_dir / "s1").glob("shard_*.jsonl"))
    assert shards, f"no staged S1 calib shards under {calib_dir}/s1 (run --phase stage-calib)"
    rows = [r for s in shards for r in J94._iter_jsonl(s)]
    rows = [
        r
        for r in rows
        if r["slot"] == "ce" and r["cell"] in C.S1_CELLS and r["arm"] in ("steered", "shuffled")
    ]
    assert rows, "S1 calib shards filtered to empty (cell/slot/arm drift?)"
    return rows


def load_calib_s2(calib_dir: Path) -> list[dict]:
    """Banked #2094 fu1 conf1 joint_all replace rows (matched_query, A-carrier).

    Observed schema (probe 2026-08-16, @ PIN_FU1): keys incl. pair_id / arm /
    draw / text / block_key / cell / slot.
    """
    files = sorted((calib_dir / "s2").glob("*.jsonl"))
    assert files, f"no staged S2 calib files under {calib_dir}/s2 (run --phase stage-calib)"
    rows = [r for f in files for r in J94._iter_jsonl(f)]
    assert rows, "S2 calib files present but empty"
    return rows


# ── item builders ─────────────────────────────────────────────────────


def _grid_source(row: dict, kind: str) -> dict:
    return {
        "kind": kind,
        "variant": row.get("variant"),
        "arm": row.get("arm_slug") or row.get("arm"),
        "cell": row.get("cell"),
        "pair_id": row.get("pair_id"),
        "draw": row.get("draw"),
    }


def coherence_key(tag: str, row: dict) -> str:
    return f"{tag}|{row['block_key']}|{row['pair_id']}|{row['draw']}"


def behavior_key(tag: str, row: dict, side: str) -> str:
    return f"{tag}|{row['block_key']}|{row['pair_id']}|{row['draw']}|{side}"


def anchor_coherence_id(context_id: str, draw: int) -> str:
    return J94._item_id("c", f"a|{context_id}|{draw}")


def build_coherence_items(
    grid_rows: list[dict] | None = None,
    ce_rows: list[dict] | None = None,
    anchor_rows: list[dict] | None = None,
    calib_s1: list[dict] | None = None,
    calib_s2: list[dict] | None = None,
) -> list[J94.JudgeUnit]:
    """One coherence call per rollout — fresh AND calibration (same wave)."""
    units: list[J94.JudgeUnit] = []

    def _add(tag: str, row: dict, answer: str, kind: str) -> None:
        units.append(
            J94.JudgeUnit(
                item_id=J94._item_id("c", coherence_key(tag, row)),
                rubric_id=J94.COHERENCE_RUBRIC_ID,
                question="",
                answer=answer,
                source=_grid_source(row, kind),
            )
        )

    for row in grid_rows or []:
        _add("g", row, row["response_text"], "grid")
    for row in ce_rows or []:
        _add("e", row, row["response_text"], "ce")
    for row in anchor_rows or []:
        units.append(
            J94.JudgeUnit(
                item_id=anchor_coherence_id(row["context_id"], row["draw"]),
                rubric_id=J94.COHERENCE_RUBRIC_ID,
                question="",
                answer=row["response_text"],
                source={"kind": "anchor", "context_id": row["context_id"], "draw": row["draw"]},
            )
        )
    for row in calib_s1 or []:
        _add("k", row, row["text"], "calib-s1")
    for row in calib_s2 or []:
        _add("m", row, row["text"], "calib-s2")
    return units


def _dual_rubric_units(
    rows: list[dict],
    pairs_by_id: dict,
    tag: str,
    kind: str,
    answer_field: str,
) -> dict[str, list[J94.JudgeUnit]]:
    by_rid: dict[str, list[J94.JudgeUnit]] = {}
    for row in rows:
        cores = pair_rubric_cores_2333(pairs_by_id[row["pair_id"]])
        for side, core in zip(("a", "b"), cores, strict=True):
            rid = J62.rubric_core_id(core)
            answer = row[answer_field]
            src = {**_grid_source(row, kind), "side": side}
            by_rid.setdefault(rid, []).append(
                J94.JudgeUnit(
                    item_id=J94._item_id(tag, behavior_key(tag, row, side)),
                    rubric_id=rid,
                    question="",
                    answer=answer,
                    source=src,
                )
            )
    return by_rid


def build_behavior_items(
    cfg_pairs: tuple[list, list],
    grid_rows: list[dict] | None = None,
    ce_rows: list[dict] | None = None,
    calib_s1: list[dict] | None = None,
    calib_s2: list[dict] | None = None,
) -> dict[str, list[J94.JudgeUnit]]:
    """Dual-rubric behavior units — fresh grid + prefill continuation-only
    companions + q35 ce_control + SAME-WAVE calibration rows."""
    s1_pairs, s2_pairs = cfg_pairs
    pairs_by_id = {p.pair_id: p for p in [*s1_pairs, *s2_pairs]}
    by_rid = _dual_rubric_units(grid_rows or [], pairs_by_id, "g", "grid", "response_text")

    # Prefill continuation-only companions (plan §6): same rubrics, the
    # continuation text WITHOUT the prefilled donor opening.
    prefill_rows = [r for r in grid_rows or [] if r.get("kind") == "prefill"]
    for rid, us in _dual_rubric_units(
        prefill_rows, pairs_by_id, "n", "prefill-cont", "continuation_text"
    ).items():
        by_rid.setdefault(rid, []).extend(us)

    for rid, us in _dual_rubric_units(
        ce_rows or [], pairs_by_id, "e", "ce", "response_text"
    ).items():
        by_rid.setdefault(rid, []).extend(us)
    for rid, us in _dual_rubric_units(calib_s1 or [], pairs_by_id, "k", "calib-s1", "text").items():
        by_rid.setdefault(rid, []).extend(us)
    for rid, us in _dual_rubric_units(calib_s2 or [], pairs_by_id, "m", "calib-s2", "text").items():
        by_rid.setdefault(rid, []).extend(us)
    return by_rid


def build_anchor_behavior_items(
    anchor_rows: list[dict], s1_pairs: list, s2_pairs: list
) -> dict[str, list[J94.JudgeUnit]]:
    """q35 anchor floor/ceiling units — J62 dedup shape, both banks' cores.

    Item ids via ``J62.anchor_unit_id`` so the analysis join mirrors the
    parent convention.
    """
    cores_by_ctx: dict[str, set[str]] = {}
    for p in [*s1_pairs, *s2_pairs]:
        for ctx in (p.a, p.b):
            cores_by_ctx.setdefault(ctx, set()).update(pair_rubric_cores_2333(p))
    by_rid: dict[str, list[J94.JudgeUnit]] = {}
    seen: set[str] = set()
    for row in anchor_rows:
        for core in sorted(cores_by_ctx.get(row["context_id"], ())):
            rid = J62.rubric_core_id(core)
            iid = J62.anchor_unit_id(row["context_id"], row["draw"], rid)
            if iid in seen:
                continue
            seen.add(iid)
            by_rid.setdefault(rid, []).append(
                J94.JudgeUnit(
                    item_id=iid,
                    rubric_id=rid,
                    question="",
                    answer=row["response_text"],
                    source={
                        "kind": "anchor",
                        "context_id": row["context_id"],
                        "draw": row["draw"],
                        "rubric": rid,
                    },
                )
            )
    return by_rid


# ── phase: stage-calib (zero API) ─────────────────────────────────────

_S1_CALIB_FILES = tuple(
    f"shard_{cell}__ce__{arm}.jsonl" for cell in C.S1_CELLS for arm in ("steered", "shuffled")
)
_S2_CALIB_FILES = (
    "fu1_fu1__matched_query__ce__joint_all__replace__A__steered.jsonl",
    "fu1_fu1__matched_query__ce__joint_all__replace__A__null.jsonl",
)


def phase_stage_calib(cfg: JudgeConfig2333) -> int:
    """Stage the banked ce calibration inputs (sha-pinned revisions, plan §10).

    Content hygiene: bank text is WildChat-derived real-corpus — this phase
    logs COUNTS only, never row text.
    """
    if cfg.dry_run:
        logger.info(
            "[stage-calib] dry-run: would stage %d S1 shards @ %s + %d S2 files @ %s -> %s",
            len(_S1_CALIB_FILES),
            C.PIN_2162[:12],
            len(_S2_CALIB_FILES),
            C.PIN_FU1[:12],
            cfg.calib_dir,
        )
        return RC_OK
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    for sub, files, prefix, rev in (
        ("s1", _S1_CALIB_FILES, C.R2162_GRID_ROLLOUTS, C.PIN_2162),
        ("s2", _S2_CALIB_FILES, C.FU1_CONF1_PREFIX, C.PIN_FU1),
    ):
        dest = cfg.calib_dir / sub
        dest.mkdir(parents=True, exist_ok=True)
        for name in files:
            target = dest / name
            if target.is_file():
                continue
            got = hub.retry_transient(
                lambda fn=f"{prefix}/{name}", r=rev: hf_hub_download(
                    repo_id=C.DATA_REPO,
                    repo_type="dataset",
                    filename=fn,
                    revision=r,
                    local_dir=dest / "_dl",
                ),
                what=f"stage-calib {sub}/{name}",
            )
            Path(got).replace(target)
        logger.info("[stage-calib] %s: %d files staged", sub, len(files))
    n1, n2 = len(load_calib_s1(cfg.calib_dir)), len(load_calib_s2(cfg.calib_dir))
    logger.info("[stage-calib] rows: s1=%d s2=%d", n1, n2)
    assert n1 > 0 and n2 > 0
    return RC_OK


# ── phase: pilot (rule 26; REFUSES --dry-run) ─────────────────────────


def _pilot_arm(u: J94.JudgeUnit) -> str:
    src = u.source
    return f"{src.get('kind', 'unknown')}.{src.get('variant') or 'na'}"


def forced_batch_probe_verdict(
    scores: dict[str, float | None], stop_reason_tally: dict[str, int], n_items: int
) -> tuple[bool, dict]:
    """Registered forced-batch gate (plan §7): the probe is EXACTLY
    ``J94.FORCED_BATCH_PROBE_N`` items, ALL scored, and EVERY persisted draw
    stop_reason is ``end_turn`` (r1 Minor: the shipped ``n_probe >= 1`` was
    weaker than the registered 6/6-all-end_turn criterion). A cache-served
    legacy entry tallies ``unknown`` and FAILS — run the probe against its
    fresh ``_forced_batch`` cache dir."""
    n_scored = sum(1 for v in scores.values() if v is not None)
    tally = dict(stop_reason_tally)
    non_end_turn = {k: v for k, v in tally.items() if k != "end_turn" and v}
    passed = (
        n_items == J94.FORCED_BATCH_PROBE_N
        and n_scored == J94.FORCED_BATCH_PROBE_N
        and sum(tally.values()) >= J94.FORCED_BATCH_PROBE_N
        and not non_end_turn
    )
    return passed, {
        "n_items": n_items,
        "n_scored": n_scored,
        "required": J94.FORCED_BATCH_PROBE_N,
        "stop_reason_tally": tally,
        "non_end_turn": non_end_turn,
        "passed": passed,
    }


def phase_pilot(cfg: JudgeConfig2333) -> int:
    """Rule-26 pilot per rubric family (coherence / S1 rubric / S2 rubric)
    plus the live forced-batch request-shape probe."""
    if cfg.base.dry_run:
        logger.error(
            "[pilot] --dry-run refused (J62 fix 3): the rule-26 pilot's purpose is "
            "measuring the REAL instrument — run without --dry-run."
        )
        return RC_DRY_RUN_UNSUPPORTED
    s1_pairs, s2_pairs = build_pair_universe()
    registry = rubric_registry(s1_pairs, s2_pairs)
    grid_rows = load_grid_rows(cfg.rollouts_dir)
    calib_s1 = load_calib_s1(cfg.calib_dir)
    calib_s2 = load_calib_s2(cfg.calib_dir)
    coh = build_coherence_items(grid_rows, None, None, calib_s1, calib_s2)
    beh = build_behavior_items((s1_pairs, s2_pairs), grid_rows, None, calib_s1, calib_s2)

    s1_rids = {J62.rubric_core_id(c) for p in s1_pairs for c in pair_rubric_cores_2333(p)}
    s2_rids = {J62.rubric_core_id(c) for p in s2_pairs for c in pair_rubric_cores_2333(p)}
    rep_s1 = max((r for r in by_len_order(beh) if r in s1_rids), key=lambda r: len(beh[r]))
    rep_s2 = max((r for r in by_len_order(beh) if r in s2_rids), key=lambda r: len(beh[r]))
    fam_reps = {
        "coherence": (J94.COHERENCE_RUBRIC_ID, coh, J62.PILOT_TARGET_COHERENCE),
        "s1-rubric": (rep_s1, beh[rep_s1], J62.PILOT_TARGET_BEHAVIOR),
        "s2-rubric": (rep_s2, beh[rep_s2], J62.PILOT_TARGET_BEHAVIOR),
    }
    per_family: dict[str, dict] = {}
    all_pass = True
    for family, (rid, units, target) in fam_reps.items():
        J94._validate_units(units)
        arms: dict[str, list[tuple[str, str, str]]] = {}
        for u in units:
            arms.setdefault(_pilot_arm(u), []).append((u.item_id, u.question, u.answer))
        # Rule-26(b) satisfiability: the gate needs >= 51 effective draws per
        # unwaived arm (floor(1/0.02)+1 at its default parse_fail_threshold)
        # and floor-divides the budget across arms. The inherited J62 targets
        # were sized for the parent's arm structure; #2333's families carry
        # more arms (calib-s1/calib-s2/grid.null/grid.steered), so re-derive
        # from the realized arm count with headroom for transport-loss
        # shrinkage of effective draws (60 > 51 floor).
        per_arm_items = -(-60 // JUDGE_N_DRAWS)  # ceil division
        target = max(target, len(arms) * JUDGE_N_DRAWS * per_arm_items)
        report = judge_pilot_gate(
            arms,
            registry[rid],
            max_tokens=cfg.base.max_tokens,
            cache_dir=cfg.base.pilot_cache_root / rid,
            save_raw_dir=cfg.base.raw_dir / "pilot" / rid,
            n_draws=JUDGE_N_DRAWS,
            target_total_draws=target,
            judge_model=cfg.base.judge_model,
            report_path=cfg.base.gates_dir / "pilot" / f"{family}.json",
            seed=PILOT_SEED,
        )
        per_family[family] = {
            "rubric_id": rid,
            "verdict": report.verdict,
            "failures": report.failures,
            "warnings": report.warnings,
            "n_total_draws": report.n_total_draws,
        }
        all_pass &= report.passed
        logger.info("[pilot] %s (%s): %s", family, rid, report.verdict)

    probe_units = coh[: J94.FORCED_BATCH_PROBE_N]
    probe = judge_graded(
        [(u.item_id, u.question, u.answer) for u in probe_units],
        registry[J94.COHERENCE_RUBRIC_ID],
        n_draws=JUDGE_N_DRAWS,
        cache_dir=cfg.base.pilot_cache_root / "_forced_batch",
        save_raw=cfg.base.raw_dir / "pilot" / "forced_batch_probe.json",
        judge_model=cfg.base.judge_model,
        max_tokens=cfg.base.max_tokens,
        threshold_base=0,
    )
    probe_ok, probe_report = forced_batch_probe_verdict(
        probe.scores, probe.stop_reason_tally, len(probe_units)
    )
    all_pass &= probe_ok
    logger.info("[pilot] forced-batch probe: %s", probe_report)
    J94._write_json_atomic(
        cfg.base.gates_dir / "pilot_gate_report.json",
        {
            "passed": all_pass,
            "per_family": per_family,
            "forced_batch_probe": probe_report,
            "instrument": {
                "judge_model": cfg.base.judge_model,
                "max_tokens": cfg.base.max_tokens,
                "n_draws": JUDGE_N_DRAWS,
                "n_rubrics_total": len(registry),
                "model_tag": cfg.model_tag,
            },
            "repro": J94._repro(),
        },
    )
    logger.info("[pilot] aggregate verdict: %s", "PASS" if all_pass else "FAIL")
    return RC_OK if all_pass else RC_PILOT_GATE


def by_len_order(by_rid: dict[str, list]) -> list[str]:
    return sorted(by_rid, key=lambda r: -len(by_rid[r]))


# ── phase: anchors (q35 only) ─────────────────────────────────────────


def _require_gates(cfg: JudgeConfig2333, names: tuple[str, ...]) -> None:
    for name in names:
        path = cfg.base.gates_dir / name
        if not path.is_file():
            raise RuntimeError(f"gate report missing: {path} — run the gate phase first")
        rec = json.loads(path.read_text(encoding="utf-8"))
        if not rec.get("passed"):
            raise RuntimeError(f"gate FAILED per {path} — fix the instrument and re-run")


def phase_anchors(cfg: JudgeConfig2333) -> int:
    """q35: coherence-baseline gate over the fresh anchors, then anchor waves."""
    assert cfg.model_tag == "q35", "anchors phase is q35-only (q25 reuses banked anchor scores)"
    assert cfg.anchors_dir is not None
    s1_pairs, s2_pairs = build_pair_universe()
    anchor_rows = load_anchor_rows(cfg.anchors_dir)
    if cfg.base.dry_run:
        beh = build_anchor_behavior_items(anchor_rows, s1_pairs, s2_pairs)
        return J62._dry_run_units_report(
            "anchors",
            {
                "coherence.anchors": build_coherence_items(anchor_rows=anchor_rows),
                **{f"{rid}.anchors": us for rid, us in beh.items()},
            },
        )
    _require_gates(cfg, ("pilot_gate_report.json",))
    registry = rubric_registry(s1_pairs, s2_pairs)
    coh_units = build_coherence_items(anchor_rows=anchor_rows)
    J94.run_wave(
        "coherence.anchors",
        J94.COHERENCE_RUBRIC_ID,
        registry[J94.COHERENCE_RUBRIC_ID],
        coh_units,
        cfg.base,
    )
    scores = list(J94._iter_jsonl(cfg.base.scores_dir / "coherence.anchors.scores.jsonl"))
    gate = J94.coherence_baseline_gate(scores)
    J94._write_json_atomic(cfg.base.gates_dir / "coherence_baseline_gate.json", gate)
    logger.info(
        "[gate] q35 coherence baseline: median=%.1f frac>60=%.3f -> %s",
        gate["median"],
        gate["frac_gt60"],
        "PASS" if gate["passed"] else "FAIL",
    )
    if not gate["passed"]:
        return RC_COHERENCE_GATE
    for rid, units in sorted(build_anchor_behavior_items(anchor_rows, s1_pairs, s2_pairs).items()):
        J94.run_wave(f"{rid}.anchors", rid, registry[rid], units, cfg.base)
    return RC_OK


# ── phase: waves ──────────────────────────────────────────────────────


def phase_waves(cfg: JudgeConfig2333) -> int:
    """Production grid waves: coherence + dual-rubric behavior over fresh grid
    rows, prefill continuation companions, q35 ce_control rows, and the
    SAME-WAVE banked ce calibration rows."""
    s1_pairs, s2_pairs = build_pair_universe()
    grid_rows = load_grid_rows(cfg.rollouts_dir)
    ce_rows = load_ce_rows(cfg.rollouts_dir) if cfg.model_tag == "q35" else []
    calib_s1 = load_calib_s1(cfg.calib_dir)
    calib_s2 = load_calib_s2(cfg.calib_dir)
    if cfg.base.dry_run:
        beh = build_behavior_items((s1_pairs, s2_pairs), grid_rows, ce_rows, calib_s1, calib_s2)
        return J62._dry_run_units_report(
            "waves",
            {
                "coherence.grid": build_coherence_items(
                    grid_rows, ce_rows, None, calib_s1, calib_s2
                ),
                **{f"{rid}.grid": us for rid, us in beh.items()},
            },
        )
    gates = ("pilot_gate_report.json",)
    if cfg.model_tag == "q35":
        gates = ("pilot_gate_report.json", "coherence_baseline_gate.json")
    _require_gates(cfg, gates)
    registry = rubric_registry(s1_pairs, s2_pairs)
    coh_units = build_coherence_items(grid_rows, ce_rows, None, calib_s1, calib_s2)
    J94.run_wave(
        "coherence.grid",
        J94.COHERENCE_RUBRIC_ID,
        registry[J94.COHERENCE_RUBRIC_ID],
        coh_units,
        cfg.base,
    )
    beh = build_behavior_items((s1_pairs, s2_pairs), grid_rows, ce_rows, calib_s1, calib_s2)
    for rid, units in sorted(beh.items()):
        J94.run_wave(f"{rid}.grid", rid, registry[rid], units, cfg.base)
    return RC_OK


# ── phase: upload-raw ─────────────────────────────────────────────────


def phase_upload_raw(cfg: JudgeConfig2333) -> int:
    prefix = f"{C.HF_PREFIX}/{cfg.model_tag}/raw_completions/judge_raw"
    if cfg.base.dry_run:
        logger.info("[upload-raw] dry-run: would upload %s -> %s", cfg.base.work_root, prefix)
        return RC_OK
    from explore_persona_space.orchestrate import hub

    url = hub._upload(
        cfg.base.work_root,
        repo_id=C.DATA_REPO,
        repo_type="dataset",
        path_in_repo=prefix,
        raise_on_error=True,
    )
    logger.info("[upload-raw] uploaded %s -> %s", cfg.base.work_root, url)
    return RC_OK


# ── CLI ───────────────────────────────────────────────────────────────

PHASES = {
    "stage-calib": phase_stage_calib,
    "pilot": phase_pilot,
    "anchors": phase_anchors,
    "waves": phase_waves,
    "upload-raw": phase_upload_raw,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2333 VM-side judge pipeline.")
    # required unless --import-check (r1 Minor: the standalone import-check
    # invocation must parse without a phase; main() asserts them otherwise).
    ap.add_argument("--phase", choices=tuple(PHASES))
    ap.add_argument("--model-tag", choices=("q25", "q35"))
    ap.add_argument(
        "--in-root",
        type=Path,
        default=Path("data/issue_2333/judge_inputs"),
        help=f"staging mirror root; rollouts default under <in-root>/{C.HF_PREFIX}/<tag>/",
    )
    ap.add_argument("--rollouts-dir", type=Path, default=None)
    ap.add_argument("--anchors-dir", type=Path, default=None)
    ap.add_argument("--calib-dir", type=Path, default=None)
    ap.add_argument("--stage-from-hf", action="store_true")
    ap.add_argument("--hf-revision", type=str, default=None)
    ap.add_argument("--work-root", type=Path, default=None)
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--judge-model", type=str, default=J94.DEFAULT_JUDGE_MODEL)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="construction check: build + validate every judge unit, ZERO API calls; "
        "--phase pilot REFUSES it (rc 10)",
    )
    return ap.parse_args(argv)


def _stage_inputs(args: argparse.Namespace) -> None:
    from explore_persona_space.orchestrate import hub

    base = f"{C.HF_PREFIX}/{args.model_tag}"
    need = [f"{base}/rollouts"]
    if args.model_tag == "q35" and args.phase in ("anchors", "waves"):
        need.append(f"{base}/anchors")
    for prefix in need:
        staged = hub.stage_hub_prefix(C.DATA_REPO, prefix, args.in_root, revision=args.hf_revision)
        logger.info("[stage] %s: %d files", prefix, len(staged))
        assert staged, f"nothing staged from {prefix}"


def build_config(args: argparse.Namespace) -> JudgeConfig2333:
    mirror = args.in_root / C.HF_PREFIX / args.model_tag
    rollouts = args.rollouts_dir if args.rollouts_dir is not None else mirror / "rollouts"
    anchors = args.anchors_dir if args.anchors_dir is not None else mirror / "anchors"
    calib = args.calib_dir if args.calib_dir is not None else args.in_root / "calib"
    work_root = args.work_root or Path(f"eval_results/issue_2333/judge_{args.model_tag}")
    cache_root = args.cache_root or Path(f"data/issue_2333/judge_cache_{args.model_tag}")
    base = J94.JudgeConfig(
        work_root=work_root,
        cache_root=cache_root,
        rollouts_dir=rollouts,
        anchors_file=anchors,
        stage2_dir=None,
        judge_model=args.judge_model,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
    )
    return JudgeConfig2333(
        base=base,
        model_tag=args.model_tag,
        rollouts_dir=rollouts,
        anchors_dir=anchors,
        calib_dir=calib,
    )


def _import_check() -> int:
    """Deferred-import + args-attribute completeness check (code-style.md)."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    from huggingface_hub import hf_hub_download  # noqa: F401

    from explore_persona_space.orchestrate import hub  # noqa: F401

    assert (
        callable(hub._upload) and callable(hub.stage_hub_prefix) and callable(hub.retry_transient)
    )
    s1_pairs, s2_pairs = build_pair_universe()
    reg = rubric_registry(s1_pairs, s2_pairs)
    # 5 S1 cells x 36 pairs contribute per-pair cores; 15 S2 pairs x 2 prefix
    # cores; plus coherence. Exact count depends on core dedup — assert bounds.
    assert len(reg) >= 3, len(reg)
    assert len(_S1_CALIB_FILES) == 10 and len(_S2_CALIB_FILES) == 2
    print("[import-check] OK")
    return RC_OK


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.import_check:
        return _import_check()
    assert args.phase and args.model_tag, "--phase and --model-tag required (or --import-check)"
    if args.stage_from_hf:
        _stage_inputs(args)
    cfg = build_config(args)
    for d in (
        cfg.base.scores_dir,
        cfg.base.items_dir,
        cfg.base.raw_dir,
        cfg.base.gates_dir,
        cfg.base.audits_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
    rc = PHASES[args.phase](cfg)
    logger.info("[phase=%s_done] rc=%d", args.phase, rc)
    return rc


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
