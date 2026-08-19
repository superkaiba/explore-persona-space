#!/usr/bin/env python3
"""Issue #2162 — VM-side judge pipeline (plan §4.6 P6/P9 + the §7 gates).

Reuses the #2094 judge machinery wholesale (``scripts/issue2094_judge.py``:
``JudgeUnit`` / ``run_wave`` / wave-regime resume / per-arm drop-split
telemetry / mechanical audits / the form-only coherence instrument VERBATIM)
and adds the 2162-specific pieces:

- a DYNAMIC rubric registry built from ``bank2162.rubric_pair_2162`` (graded
  0-100 value-descriptor cores; conflict cells' instructed-vs-demonstrated
  pair; ``query_content``'s which-question rubric), rubric ids content-hashed
  so the rubric-keyed JudgeCache partitions per instrument (rule 22),
- the plan §7 **gate 3 anchor-separation early gate**: the 228-pair
  ``bank2162.gate_slice_pairs`` slice judged SYNC (``threshold_base`` forced
  high => the api_dispatch sync fan-out), PASS <=> >= 60% of the 38 non-filler
  cells have >= 4/6 sampled pairs at |sep| >= 0.5 — checked at P3 entry,
  BEFORE the 42k-rollout stage-1 spend,
- the rule-26 pilot gate per rubric FAMILY (coherence + value-rubric family +
  which-question family, ~440 draws total, plan §7 gate 5) + the live
  forced-batch request-shape probe,
- the coherence-baseline sanity gate (plan §7 gate 4, #2094 thresholds
  verbatim via ``issue2094_judge.coherence_baseline_gate``),
- the TF-margin POOLS builder (``--phase pools``, r1 C4): fixed judge-filtered
  4+4 pools per value-pair key from the judged anchor waves (score > 50 kept,
  plan §4.4 / §11), written to ``<work_root>/pools.json`` for staging to the
  pod's margin phase.

Gate 4 mechanical routing (the r1 review gate): every judge call goes through
``eval.graded_judge.judge_graded`` -> ``eval.batch_judge`` ->
``eval.judge_dispatch.dispatch_judge_items`` -> ``llm/api_dispatch.py``
(judge ``claude-sonnet-4-5-20250929``, ``max_tokens=1024``, Batch API for
production waves, drop-never-coerce + transport-retry per llm-judging rules
9/24/28).

All phases are resumable: wave-level meta skip + the rubric-keyed JudgeCache
(so the production anchor waves REUSE the gate-3 slice's judged draws at zero
extra spend).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2094_judge as J94  # noqa: E402  (same-dir script import; reused machinery)
from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402
from explore_persona_space.eval.judge_pilot import judge_pilot_gate  # noqa: E402
from explore_persona_space.experiments.issue2162 import bank2162 as BANK  # noqa: E402

logger = logging.getLogger("issue2162.judge")

HF_PREFIX = "issue2162_ctxinfo"
DATASET_REPO = "superkaiba1/explore-persona-space-data"

RC_OK = 0
RC_PILOT_GATE = 7
RC_COHERENCE_GATE = 8
RC_SEPARATION_GATE = 9
RC_DRY_RUN_UNSUPPORTED = 10  # fix 3: a phase whose purpose is live measurement

# Plan §7 gate 3 (thresholds body-verbatim; the 0.5 bar is the §4.5 exclusion bar).
SEPARATION_BAR = 0.5
SEP_MIN_PAIRS_OF_6 = 4
SEP_CELL_FRAC_MIN = 0.60
SEP_CATASTROPHIC_FRAC = 0.25
# Forces the SYNC api_dispatch route regardless of N (judge_dispatch:
# sync iff n_items <= threshold_base * otpm / 400k).
FORCE_SYNC_THRESHOLD_BASE = 10**9

# Plan §7 gate 5: ~440 draws spanning the rubric FAMILIES.
PILOT_TARGET_COHERENCE = 200
PILOT_TARGET_BEHAVIOR = 120
PILOT_SEED = 2162

JUDGE_N_DRAWS = J94.JUDGE_N_DRAWS  # 1 — the pair-clustered bootstrap carries uncertainty


# ── rubric registry (dynamic, content-hashed ids) ─────────────────────


def rubric_core_id(core: str) -> str:
    """Stable content-hashed rubric id (id grammar: ``^[a-zA-Z0-9-]{1,53}$``)."""
    return "f" + hashlib.sha1(core.encode("utf-8")).hexdigest()[:12]


def pair_rubric_cores(pair: BANK.Pair2162) -> tuple[str, str] | None:
    """(core_A, core_B) or None for the no-rubric ``filler_swap`` class."""
    if BANK.base_type_of(pair.cell) == "filler_swap":
        return None
    return BANK.rubric_pair_2162(pair)


def rubric_registry(pairs: list[BANK.Pair2162]) -> dict[str, str]:
    """rubric_id -> production eval_prompt (coherence + every distinct core)."""
    reg = {J94.COHERENCE_RUBRIC_ID: J94.coherence_eval_prompt()}
    for p in pairs:
        cores = pair_rubric_cores(p)
        if cores is None:
            continue
        for core in cores:
            reg.setdefault(rubric_core_id(core), J94.behavior_eval_prompt(core))
    return reg


def _is_query_rubric(prompt: str) -> bool:
    return "an answer to the following question" in prompt


# ── input walkers (2162 shard formats) ────────────────────────────────


def load_grid_rows(rollouts_dir: Path) -> list[dict]:
    shards = sorted(rollouts_dir.glob("shard_*.jsonl"))
    assert shards, f"no grid shards under {rollouts_dir}"
    rows = [r for shard in shards for r in J94._iter_jsonl(shard)]
    assert rows, "grid shards present but empty"
    for r in rows[:1]:
        for key in ("cell", "slot", "arm", "pair_id", "draw", "text", "context_id"):
            assert key in r, (key, sorted(r))
    return rows


def load_anchor_rows(anchors_dir: Path) -> list[dict]:
    files = sorted(anchors_dir.glob("anchors_*.jsonl"))
    assert files, f"no anchor shards under {anchors_dir}"
    rows = [r for f in files for r in J94._iter_jsonl(f)]
    assert rows, "anchor shards present but empty"
    for r in rows[:1]:
        for key in ("context_id", "cell", "value_id", "carrier", "draw", "text"):
            assert key in r, (key, sorted(r))
    return rows


def load_stage2_rows(stage2_dir: Path) -> list[dict]:
    shards = sorted(stage2_dir.glob("shard_*.jsonl"))
    assert shards, f"no stage2 shards under {stage2_dir}"
    return [r for shard in shards for r in J94._iter_jsonl(shard)]


# ── item builders ─────────────────────────────────────────────────────


def _grid_source(row: dict, kind: str = "grid") -> dict:
    return {
        "kind": kind,
        "arm": row["arm"],
        "cell": row["cell"],
        "slot": row["slot"],
        "pair_id": row["pair_id"],
        "draw": row["draw"],
        "context_id": row["context_id"],
    }


def _anchor_source(row: dict) -> dict:
    return {
        "kind": "anchor",
        "cell": row["cell"],
        "value_id": row["value_id"],
        "carrier": row["carrier"],
        "draw": row["draw"],
        "context_id": row["context_id"],
    }


def build_coherence_items(
    grid_rows: list[dict] | None = None,
    anchor_rows: list[dict] | None = None,
    stage2_rows: list[dict] | None = None,
) -> list[J94.JudgeUnit]:
    """One form-only coherence call per rollout (plan §4.5: EVERY rollout)."""
    units: list[J94.JudgeUnit] = []
    for row in grid_rows or []:
        key = f"g|{row['block_key']}|{row['pair_id']}|{row['draw']}"
        units.append(
            J94.JudgeUnit(
                item_id=J94._item_id("c", key),
                rubric_id=J94.COHERENCE_RUBRIC_ID,
                question="",
                answer=row["text"],
                source=_grid_source(row),
            )
        )
    for row in anchor_rows or []:
        key = f"a|{row['context_id']}|{row['draw']}"
        units.append(
            J94.JudgeUnit(
                item_id=J94._item_id("c", key),
                rubric_id=J94.COHERENCE_RUBRIC_ID,
                question="",
                answer=row["text"],
                source=_anchor_source(row),
            )
        )
    for row in stage2_rows or []:
        key = f"s|{row['block_key']}|{row['pair_id']}|{row['draw']}"
        units.append(
            J94.JudgeUnit(
                item_id=J94._item_id("c", key),
                rubric_id=J94.COHERENCE_RUBRIC_ID,
                question="",
                answer=row["text"],
                source=_grid_source(row, kind="stage2"),
            )
        )
    return units


def _behavior_unit(tag: str, key: str, rid: str, row: dict, source: dict) -> J94.JudgeUnit:
    return J94.JudgeUnit(
        item_id=J94._item_id(tag, key),
        rubric_id=rid,
        question="",
        answer=row["text"],
        source=source,
    )


def build_grid_behavior_items(
    rows: list[dict],
    pairs_by_id: dict[str, BANK.Pair2162],
    tag: str = "g",
    kind: str = "grid",
) -> dict[str, list[J94.JudgeUnit]]:
    """Dual-rubric behavior units per grid/stage2 rollout (plan §4.4 F_beh)."""
    by_rid: dict[str, list[J94.JudgeUnit]] = {}
    for row in rows:
        cores = pair_rubric_cores(pairs_by_id[row["pair_id"]])
        if cores is None:
            continue  # filler_swap: disruption DV only, no F rubric (§4.4)
        for side, core in zip(("a", "b"), cores, strict=True):
            rid = rubric_core_id(core)
            key = f"{tag}|{row['block_key']}|{row['pair_id']}|{row['draw']}|{side}"
            src = {**_grid_source(row, kind=kind), "side": side}
            by_rid.setdefault(rid, []).append(_behavior_unit(tag, key, rid, row, src))
    return by_rid


def anchor_unit_id(context_id: str, draw: int, rid: str) -> str:
    """Deterministic anchor-behavior item id — shared between the gate-3 slice
    and the production anchor waves (the cache/dedup join key)."""
    return J94._item_id("a", f"a|{context_id}|{draw}|{rid}")


def build_anchor_behavior_items(
    anchor_rows: list[dict],
    pairs: list[BANK.Pair2162],
    restrict_pairs: list[BANK.Pair2162] | None = None,
) -> dict[str, list[J94.JudgeUnit]]:
    """Anchor floor/ceiling units: each anchor draw judged under EVERY rubric
    core of every pair its context participates in, deduplicated per
    (context, draw, rubric) — the F_beh floor/ceiling terms (plan §4.4)."""
    use = restrict_pairs if restrict_pairs is not None else pairs
    cores_by_ctx: dict[str, set[str]] = {}
    for p in use:
        cores = pair_rubric_cores(p)
        if cores is None:
            continue
        for ctx in (p.a, p.b):
            cores_by_ctx.setdefault(ctx, set()).update(cores)
    by_rid: dict[str, list[J94.JudgeUnit]] = {}
    seen: set[str] = set()
    for row in anchor_rows:
        for core in sorted(cores_by_ctx.get(row["context_id"], ())):
            rid = rubric_core_id(core)
            iid = anchor_unit_id(row["context_id"], row["draw"], rid)
            if iid in seen:
                continue
            seen.add(iid)
            src = {**_anchor_source(row), "rubric": rid}
            by_rid.setdefault(rid, []).append(
                J94.JudgeUnit(
                    item_id=iid, rubric_id=rid, question="", answer=row["text"], source=src
                )
            )
    return by_rid


# ── uniform --dry-run (fix 3) ─────────────────────────────────────────
#
# ONE meaning across phases: build + validate every judge unit the phase
# would dispatch, print counts/routing, make ZERO API calls, persist NOTHING
# under the work root (no scores, no items, no audits, no gate verdicts).
# A phase whose purpose IS live measurement (pilot) REFUSES loudly
# (``RC_DRY_RUN_UNSUPPORTED``) instead of silently spending.


def _dry_run_units_report(phase: str, waves: dict[str, list[J94.JudgeUnit]]) -> int:
    """Construction check only — validate units per wave, log counts, exit 0."""
    total = 0
    for wave, units in sorted(waves.items()):
        J94._validate_units(units)
        total += len(units)
        logger.info("[%s] dry-run wave %s: %d units", phase, wave, len(units))
    logger.info(
        "[%s] dry-run complete: %d units across %d waves (no API calls made, nothing persisted)",
        phase,
        total,
        len(waves),
    )
    return RC_OK


# ── gate 3: anchor-separation early gate (plan §7, SYNC) ─────────────


def _sep_scores_for_slice(
    cfg: J94.JudgeConfig,
    registry: dict[str, str],
    gate_units: dict[str, list[J94.JudgeUnit]],
    dry_run: bool,
) -> dict[str, float | None]:
    """Judge the gate-slice units SYNC; returns item_id -> mean kept score."""
    scores: dict[str, float | None] = {}
    for rid, units in sorted(gate_units.items()):
        J94._validate_units(units)
        result = judge_graded(
            [(u.item_id, u.question, u.answer) for u in units],
            registry[rid],
            n_draws=JUDGE_N_DRAWS,
            cache_dir=cfg.cache_root / rid,
            save_raw=cfg.raw_dir / "separation_gate" / f"{rid}.json",
            judge_model=cfg.judge_model,
            max_tokens=cfg.max_tokens,
            threshold_base=FORCE_SYNC_THRESHOLD_BASE,  # SYNC api_dispatch fan-out
            dry_run=dry_run,
        )
        if dry_run:
            scores.update({u.item_id: None for u in units})
            continue
        scores.update({u.item_id: result.scores.get(u.item_id) for u in units})
    return scores


def separation_verdict(
    gate_pairs: list[BANK.Pair2162],
    anchor_rows: list[dict],
    scores: dict[str, float | None],
) -> dict:
    """Per-pair sep = Delta(ceiling ctx B) - Delta(floor ctx A); per-cell >= 4/6
    at |sep| >= 0.5; PASS <=> >= 60% of cells (plan §7 gate 3)."""
    draws_by_ctx: dict[str, list[int]] = {}
    for row in anchor_rows:
        draws_by_ctx.setdefault(row["context_id"], []).append(row["draw"])
    per_cell: dict[str, dict] = {}
    pair_rows: list[dict] = []
    for p in gate_pairs:
        cores = pair_rubric_cores(p)
        assert cores is not None, p.pair_id  # gate slice excludes filler_swap
        rid_a, rid_b = (rubric_core_id(c) for c in cores)

        def _delta(ctx: str) -> float | None:
            deltas = []
            for draw in draws_by_ctx.get(ctx, []):
                sa = scores.get(anchor_unit_id(ctx, draw, rid_a))
                sb = scores.get(anchor_unit_id(ctx, draw, rid_b))
                if sa is None or sb is None:
                    continue  # rule-9 drop: excluded, never coerced
                deltas.append((sb - sa) / 100.0)
            return sum(deltas) / len(deltas) if deltas else None

        d_floor, d_ceiling = _delta(p.a), _delta(p.b)
        sep = None if (d_floor is None or d_ceiling is None) else d_ceiling - d_floor
        pair_rows.append(
            {
                "pair_id": p.pair_id,
                "cell": p.cell,
                "delta_floor": d_floor,
                "delta_ceiling": d_ceiling,
                "sep": sep,
                "passes_bar": bool(sep is not None and abs(sep) >= SEPARATION_BAR),
            }
        )
        c = per_cell.setdefault(p.cell, {"n_pairs": 0, "n_pass": 0, "n_unscored": 0})
        c["n_pairs"] += 1
        c["n_pass"] += int(sep is not None and abs(sep) >= SEPARATION_BAR)
        c["n_unscored"] += int(sep is None)
    for cell, c in sorted(per_cell.items()):
        c["cell_pass"] = c["n_pass"] >= SEP_MIN_PAIRS_OF_6
        # Binding-table self-count: print the realized per-cell sampled-pair counts.
        logger.info(
            "[gate3] %-28s pairs=%d pass=%d unscored=%d -> %s",
            cell,
            c["n_pairs"],
            c["n_pass"],
            c["n_unscored"],
            "PASS" if c["cell_pass"] else "FAIL",
        )
    n_cells = len(per_cell)
    frac = sum(1 for c in per_cell.values() if c["cell_pass"]) / n_cells if n_cells else 0.0
    return {
        "criterion": "anchor-separation early gate (plan §7 gate 3)",
        "bars": {
            "sep_bar": SEPARATION_BAR,
            "min_pairs_of_6": SEP_MIN_PAIRS_OF_6,
            "cell_frac_min": SEP_CELL_FRAC_MIN,
            "catastrophic_frac": SEP_CATASTROPHIC_FRAC,
        },
        "n_cells": n_cells,
        "frac_cells_pass": frac,
        "passed": frac >= SEP_CELL_FRAC_MIN,
        "catastrophic": frac < SEP_CATASTROPHIC_FRAC,
        "per_cell": per_cell,
        "pairs": pair_rows,
        "repro": J94._repro(),
    }


def phase_separation_gate(cfg: J94.JudgeConfig) -> int:
    pairs = BANK.build_pairs()
    gate_pairs = BANK.gate_slice_pairs(pairs)
    anchor_rows = load_anchor_rows(cfg.anchors_file)
    have_ctx = {r["context_id"] for r in anchor_rows}
    need_ctx = {c for p in gate_pairs for c in (p.a, p.b)}
    missing = sorted(need_ctx - have_ctx)
    if missing:
        raise RuntimeError(
            f"gate3: {len(missing)} gate-slice contexts have no anchor rows yet "
            f"(first: {missing[:3]}) — the P2 gate shards are incomplete"
        )
    gate_ctx_rows = [r for r in anchor_rows if r["context_id"] in need_ctx]
    registry = rubric_registry(pairs)
    gate_units = build_anchor_behavior_items(gate_ctx_rows, pairs, restrict_pairs=gate_pairs)
    n_calls = sum(len(us) for us in gate_units.values())
    logger.info(
        "[gate3] %d pairs, %d contexts, %d rubrics, %d sync judge calls",
        len(gate_pairs),
        len(need_ctx),
        len(gate_units),
        n_calls,
    )
    scores = _sep_scores_for_slice(cfg, registry, gate_units, cfg.dry_run)
    if cfg.dry_run:
        logger.info("[gate3] dry-run complete (no API calls)")
        return RC_OK
    report = separation_verdict(gate_pairs, gate_ctx_rows, scores)
    J94._write_json_atomic(cfg.gates_dir / "separation_gate_report.json", report)
    logger.info(
        "[gate3] %.0f%% of %d cells pass -> %s%s",
        100 * report["frac_cells_pass"],
        report["n_cells"],
        "PASS" if report["passed"] else "FAIL",
        " (CATASTROPHIC — abort + bank rebuild per §7)" if report["catastrophic"] else "",
    )
    return RC_OK if report["passed"] else RC_SEPARATION_GATE


# ── pilot (plan §7 gate 5, rule 26 — per rubric FAMILY, ~440 draws) ───


def _pilot_arm(u: J94.JudgeUnit) -> str:
    src = u.source
    if src["kind"] in ("anchor", "stage2"):
        return src["kind"]
    return src.get("arm", "unknown")


def _family_representative(
    by_rid: dict[str, list[J94.JudgeUnit]], registry: dict[str, str], query_family: bool
) -> str:
    cands = {
        rid: units
        for rid, units in by_rid.items()
        if _is_query_rubric(registry[rid]) == query_family
    }
    assert cands, f"no rubric ids for family query={query_family}"
    return max(cands, key=lambda rid: len(cands[rid]))


def phase_pilot(cfg: J94.JudgeConfig) -> int:
    """Rule-26 pilot per rubric FAMILY + the live forced-batch shape probe.

    REFUSES ``--dry-run`` (fix 3): the pilot gate EXISTS to measure the real
    instrument's truncation/parse-fail profile and to live-probe the
    forced-batch request shape, so a zero-API dry run of it is meaningless —
    the pre-fix behavior silently made ~356 real calls and wrote a real
    verdict under the flag."""
    if cfg.dry_run:
        logger.error(
            "[pilot] --dry-run refused: the rule-26 pilot's whole purpose is measuring the "
            "REAL instrument's truncation/parse-fail profile (plus the live forced-batch "
            "request-shape probe) — there is no meaningful zero-API pilot. Run without "
            "--dry-run to spend the ~%d pilot draws, or use --phase separation-gate / "
            "--phase anchors with --dry-run for a free construction check.",
            PILOT_TARGET_COHERENCE + 2 * PILOT_TARGET_BEHAVIOR,
        )
        return RC_DRY_RUN_UNSUPPORTED
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    registry = rubric_registry(pairs)
    grid_rows = load_grid_rows(cfg.rollouts_dir)
    anchor_rows = load_anchor_rows(cfg.anchors_file)
    coh = build_coherence_items(grid_rows, anchor_rows)
    beh = build_grid_behavior_items(grid_rows, pairs_by_id)
    for rid, us in build_anchor_behavior_items(anchor_rows, pairs).items():
        beh.setdefault(rid, []).extend(us)

    fam_reps = {
        "coherence": (J94.COHERENCE_RUBRIC_ID, coh, PILOT_TARGET_COHERENCE),
        "value-rubric": (
            _family_representative(beh, registry, query_family=False),
            None,
            PILOT_TARGET_BEHAVIOR,
        ),
        "query-rubric": (
            _family_representative(beh, registry, query_family=True),
            None,
            PILOT_TARGET_BEHAVIOR,
        ),
    }
    per_family: dict[str, dict] = {}
    all_pass = True
    for family, (rid, units, target) in fam_reps.items():
        units = units if units is not None else beh[rid]
        J94._validate_units(units)
        arms: dict[str, list[tuple[str, str, str]]] = {}
        for u in units:
            arms.setdefault(_pilot_arm(u), []).append((u.item_id, u.question, u.answer))
        report = judge_pilot_gate(
            arms,
            registry[rid],
            max_tokens=cfg.max_tokens,
            cache_dir=cfg.pilot_cache_root / rid,
            save_raw_dir=cfg.raw_dir / "pilot" / rid,
            n_draws=JUDGE_N_DRAWS,
            target_total_draws=target,
            judge_model=cfg.judge_model,
            report_path=cfg.gates_dir / "pilot" / f"{family}.json",
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
        logger.info(
            "[pilot] %s (%s): %s (%d draws)", family, rid, report.verdict, report.n_total_draws
        )

    # Live forced-batch request-shape probe (threshold_base=0; gotchas: a
    # mock/sync-only pilot cannot validate the batches.create envelope).
    probe_units = coh[: J94.FORCED_BATCH_PROBE_N]
    probe = judge_graded(
        [(u.item_id, u.question, u.answer) for u in probe_units],
        registry[J94.COHERENCE_RUBRIC_ID],
        n_draws=JUDGE_N_DRAWS,
        cache_dir=cfg.pilot_cache_root / "_forced_batch",
        save_raw=cfg.raw_dir / "pilot" / "forced_batch_probe.json",
        judge_model=cfg.judge_model,
        max_tokens=cfg.max_tokens,
        threshold_base=0,
    )
    n_probe_scored = sum(1 for v in probe.scores.values() if v is not None)
    probe_ok = n_probe_scored >= 1
    all_pass &= probe_ok
    aggregate = {
        "passed": all_pass,
        "per_family": per_family,
        "forced_batch_probe": {
            "n_items": len(probe_units),
            "n_scored": n_probe_scored,
            "passed": probe_ok,
            **J94._telemetry(probe),
        },
        "instrument": {
            "judge_model": cfg.judge_model,
            "max_tokens": cfg.max_tokens,
            "n_draws": JUDGE_N_DRAWS,
            "n_rubrics_total": len(registry),
        },
        "repro": J94._repro(),
    }
    J94._write_json_atomic(cfg.gates_dir / "pilot_gate_report.json", aggregate)
    logger.info("[pilot] aggregate verdict: %s", "PASS" if all_pass else "FAIL")
    return RC_OK if all_pass else RC_PILOT_GATE


# ── phases ────────────────────────────────────────────────────────────


_ALL_GATE_REPORTS = (
    "pilot_gate_report.json",
    "coherence_baseline_gate.json",
    "separation_gate_report.json",
)


def _require_gates(cfg: J94.JudgeConfig, names: tuple[str, ...] = _ALL_GATE_REPORTS) -> None:
    """Behavior-wave spend requires the named gate reports present AND PASS
    (plan §7 gates 3/4/5; default = all three)."""
    for name in names:
        path = cfg.gates_dir / name
        if not path.is_file():
            raise RuntimeError(f"gate report missing: {path} — run the gate phase first")
        rec = json.loads(path.read_text(encoding="utf-8"))
        if not rec.get("passed"):
            raise RuntimeError(f"gate FAILED per {path} — fix the instrument/bank and re-run")


def phase_anchors(cfg: J94.JudgeConfig) -> int:
    """Coherence-baseline gate over anchors, then the anchor behavior waves.

    Entry gate (r1 M5): the anchor behavior waves are an order-10^4-call
    production spend, so the pilot (gate 5) + separation (gate 3) reports must
    be present-and-passed BEFORE launch. Gate 4 (coherence baseline) is exempt
    — this phase PRODUCES it. ``--dry-run`` (fix 3) is handled at ENTRY:
    construction check over every wave this phase would dispatch, zero API
    calls, nothing persisted."""
    pairs = BANK.build_pairs()
    anchor_rows = load_anchor_rows(cfg.anchors_file)
    if cfg.dry_run:
        beh = build_anchor_behavior_items(anchor_rows, pairs)
        return _dry_run_units_report(
            "anchors",
            {
                "coherence.anchors": build_coherence_items(None, anchor_rows),
                **{f"{rid}.anchors": us for rid, us in beh.items()},
            },
        )
    _require_gates(cfg, names=("pilot_gate_report.json", "separation_gate_report.json"))
    audits = J94.run_audits("anchors", anchor_rows, cfg.audits_dir)
    registry = rubric_registry(pairs)

    coh_units = build_coherence_items(None, anchor_rows)
    J94.run_wave(
        "coherence.anchors",
        J94.COHERENCE_RUBRIC_ID,
        registry[J94.COHERENCE_RUBRIC_ID],
        coh_units,
        cfg,
    )
    scores = list(J94._iter_jsonl(cfg.scores_dir / "coherence.anchors.scores.jsonl"))
    gate = J94.coherence_baseline_gate(scores)
    gate["audits"] = audits
    J94._write_json_atomic(cfg.gates_dir / "coherence_baseline_gate.json", gate)
    logger.info(
        "[gate4] coherence baseline: median=%.1f frac>60=%.3f -> %s",
        gate["median"],
        gate["frac_gt60"],
        "PASS" if gate["passed"] else "FAIL",
    )
    if not gate["passed"]:
        return RC_COHERENCE_GATE
    for rid, units in sorted(build_anchor_behavior_items(anchor_rows, pairs).items()):
        J94.run_wave(f"{rid}.anchors", rid, registry[rid], units, cfg)
    J94._refresh_summary(cfg)
    return RC_OK


def phase_waves(cfg: J94.JudgeConfig) -> int:
    """Production grid waves (coherence + dual-rubric behavior), gate-guarded.
    ``--dry-run`` (fix 3): construction check at entry, zero API calls,
    nothing persisted."""
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    grid_rows = load_grid_rows(cfg.rollouts_dir)
    if cfg.dry_run:
        beh = build_grid_behavior_items(grid_rows, pairs_by_id)
        return _dry_run_units_report(
            "waves",
            {
                "coherence.grid": build_coherence_items(grid_rows, None),
                **{f"{rid}.grid": us for rid, us in beh.items()},
            },
        )
    _require_gates(cfg)
    registry = rubric_registry(pairs)
    J94.run_audits("grid", grid_rows, cfg.audits_dir)
    coh_units = build_coherence_items(grid_rows, None)
    J94.run_wave(
        "coherence.grid", J94.COHERENCE_RUBRIC_ID, registry[J94.COHERENCE_RUBRIC_ID], coh_units, cfg
    )
    for rid, units in sorted(build_grid_behavior_items(grid_rows, pairs_by_id).items()):
        J94.run_wave(f"{rid}.grid", rid, registry[rid], units, cfg)
    J94._refresh_summary(cfg)
    return RC_OK


def phase_stage2(cfg: J94.JudgeConfig) -> int:
    """Stage-2 waves, gate-guarded. ``--dry-run`` (fix 3): construction check
    at entry, zero API calls, nothing persisted."""
    if cfg.stage2_dir is None:
        raise RuntimeError("--phase stage2 requires --stage2-dir")
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    rows = load_stage2_rows(cfg.stage2_dir)
    if cfg.dry_run:
        beh = build_grid_behavior_items(rows, pairs_by_id, tag="s", kind="stage2")
        return _dry_run_units_report(
            "stage2",
            {
                "coherence.stage2": build_coherence_items(None, None, rows),
                **{f"{rid}.stage2": us for rid, us in beh.items()},
            },
        )
    _require_gates(cfg)
    registry = rubric_registry(pairs)
    J94.run_audits("stage2", rows, cfg.audits_dir)
    coh_units = build_coherence_items(None, None, rows)
    J94.run_wave(
        "coherence.stage2",
        J94.COHERENCE_RUBRIC_ID,
        registry[J94.COHERENCE_RUBRIC_ID],
        coh_units,
        cfg,
    )
    for rid, units in sorted(
        build_grid_behavior_items(rows, pairs_by_id, tag="s", kind="stage2").items()
    ):
        J94.run_wave(f"{rid}.stage2", rid, registry[rid], units, cfg)
    J94._refresh_summary(cfg)
    return RC_OK


def phase_audits(cfg: J94.JudgeConfig) -> int:
    """Mechanical text audits (zero-API). ``--dry-run`` (fix 3): report which
    inputs are present, persist nothing."""
    if cfg.dry_run:
        present = [
            str(d)
            for d in (cfg.rollouts_dir, cfg.anchors_file, cfg.stage2_dir)
            if d is not None and Path(d).is_dir()
        ]
        logger.info(
            "[audits] dry-run: inputs present: %s — nothing persisted (zero-API phase)",
            present or "none",
        )
        return RC_OK
    summaries = []
    if cfg.rollouts_dir.is_dir():
        summaries.append(J94.run_audits("grid", load_grid_rows(cfg.rollouts_dir), cfg.audits_dir))
    if cfg.anchors_file.is_dir():
        summaries.append(
            J94.run_audits("anchors", load_anchor_rows(cfg.anchors_file), cfg.audits_dir)
        )
    if cfg.stage2_dir is not None and cfg.stage2_dir.is_dir():
        summaries.append(J94.run_audits("stage2", load_stage2_rows(cfg.stage2_dir), cfg.audits_dir))
    if not summaries:
        raise RuntimeError("audits: no inputs found")
    J94._write_json_atomic(
        cfg.audits_dir / "audits_summary.json", {"summaries": summaries, "repro": J94._repro()}
    )
    return RC_OK


# ── TF-margin pools builder (plan §4.4 / llm-judging rule 19; r1 C4) ──

# Plan §11: "Margin pool 4+4 per type, filter threshold >50" — fixed
# judge-filtered pools, persona-vectors keep-threshold (score > 50).
POOL_PER_SIDE = 4
POOL_FILTER_MIN = 50.0


def pool_key(pair: BANK.Pair2162) -> str:
    """MUST equal ``issue2162_run.pool_key`` byte-for-byte — the margin
    consumer's join key (pinned by tests/test_issue2162_judge.py)."""
    return f"{pair.cell}|{pair.value_a}-{pair.value_b}"


def _anchor_behavior_scores(cfg: J94.JudgeConfig) -> dict[tuple[str, int, str], float]:
    """(context_id, draw, rubric_id) -> kept mean score, from the persisted
    anchor behavior wave score rows (coherence rows carry no ``rubric`` key
    and are skipped; rule-9 dropped items carry ``score: null`` and are
    skipped — never coerced)."""
    files = sorted(cfg.scores_dir.glob("*.anchors.scores.jsonl"))
    assert files, (
        f"no anchor score files under {cfg.scores_dir} — run --phase anchors first "
        "(the pools are a pure re-reduction of the judged anchor waves)"
    )
    scores: dict[tuple[str, int, str], float] = {}
    for f in files:
        for row in J94._iter_jsonl(f):
            rid = row.get("rubric")
            if rid is None or row.get("score") is None:
                continue
            scores[(row["context_id"], int(row["draw"]), rid)] = float(row["score"])
    assert scores, "anchor score files present but zero scored behavior rows"
    return scores


def build_margin_pools(
    pairs: list[BANK.Pair2162],
    anchor_rows: list[dict],
    scores: dict[tuple[str, int, str], float],
) -> tuple[dict[str, list[dict]], dict]:
    """Fixed judge-filtered pools per value-pair key (plan §4.4): side A from
    the pairs' FLOOR contexts (A-descriptor score > 50), side B from the
    CEILING contexts (B-descriptor score > 50), top ``POOL_PER_SIDE`` per side
    by descending score (ties broken by (context_id, draw) — deterministic).

    A key whose EITHER side yields zero kept items is OMITTED (the margin
    consumer records explicit skip rows for it); a 1..3-item side is kept and
    flagged ``short`` in the report — below-floor yield is REPORTED, never
    silently backfilled.

    ``query_content`` builds NO pool, deliberately (same treatment as
    ``filler_swap``): its manipulated variable IS the user query, so rubric
    cores vary per carrier (``rubric_pair_2162`` keys them on ``pair.carrier``)
    AND each pair's two contexts pose DIFFERENT queries — a fixed shared
    answer pool scored under every context (plan §4.4 TF margin) is not
    well-defined at any key granularity. The skipped keys are reported under
    ``query_content_skip`` (never silently vanish); the margin consumer
    records explicit skip rows for the absent keys."""
    text_by = {(r["context_id"], int(r["draw"])): r["text"] for r in anchor_rows}
    ctxs_by_key: dict[str, dict] = {}
    qc_skipped_keys: set[str] = set()
    for p in pairs:
        if BANK.base_type_of(p.cell) == "query_content":
            qc_skipped_keys.add(pool_key(p))
            continue  # query_content: no well-defined fixed pool (docstring)
        cores = pair_rubric_cores(p)
        if cores is None:
            continue  # filler_swap: no rubric, no pool (explicit skip downstream)
        rid_a, rid_b = (rubric_core_id(c) for c in cores)
        rec = ctxs_by_key.setdefault(
            pool_key(p), {"rid_a": rid_a, "rid_b": rid_b, "a": set(), "b": set()}
        )
        assert rec["rid_a"] == rid_a and rec["rid_b"] == rid_b, (
            pool_key(p),
            "pairs sharing a pool key disagree on rubric cores",
        )
        rec["a"].add(p.a)
        rec["b"].add(p.b)

    pools: dict[str, list[dict]] = {}
    report_keys: dict[str, dict] = {}
    for key, rec in sorted(ctxs_by_key.items()):
        sides: dict[str, list[dict]] = {}
        for side, ctxs, rid in (("A", rec["a"], rec["rid_a"]), ("B", rec["b"], rec["rid_b"])):
            cands = []
            for (ctx, draw), text in text_by.items():
                if ctx not in ctxs:
                    continue
                score = scores.get((ctx, draw, rid))
                if score is None or score <= POOL_FILTER_MIN:
                    continue
                cands.append(
                    {"side": side, "text": text, "context_id": ctx, "draw": draw, "score": score}
                )
            cands.sort(key=lambda c: (-c["score"], c["context_id"], c["draw"]))
            sides[side] = cands[:POOL_PER_SIDE]
        n_a, n_b = len(sides["A"]), len(sides["B"])
        report_keys[key] = {
            "n_kept_a": n_a,
            "n_kept_b": n_b,
            "short": 0 < min(n_a, n_b) and (n_a < POOL_PER_SIDE or n_b < POOL_PER_SIDE),
            "omitted": min(n_a, n_b) == 0,
        }
        if min(n_a, n_b) == 0:
            continue
        pools[key] = sides["A"] + sides["B"]
    report = {
        "criterion": "TF-margin fixed pools (plan §4.4; 4+4 per key, score > 50 kept)",
        "pool_per_side": POOL_PER_SIDE,
        "filter_min": POOL_FILTER_MIN,
        "n_keys_total": len(ctxs_by_key),
        "n_keys_built": len(pools),
        "n_keys_omitted": sum(1 for r in report_keys.values() if r["omitted"]),
        "n_keys_short": sum(1 for r in report_keys.values() if r["short"]),
        "query_content_skip": {
            "n_keys": len(qc_skipped_keys),
            "keys": sorted(qc_skipped_keys),
            "reason": (
                "query_content manipulates the user query itself: rubric cores are "
                "per-carrier and each pair's two contexts pose different queries, so "
                "no fixed shared answer pool is well-defined (plan §4.4 TF margin); "
                "the margin consumer records explicit skip rows for these keys "
                "(same treatment as filler_swap)"
            ),
        },
        "per_key": report_keys,
        "repro": J94._repro(),
    }
    return pools, report


def phase_pools(cfg: J94.JudgeConfig) -> int:
    """Build + persist the TF-margin pools file (zero API calls — a pure
    re-reduction of the judged anchor waves). The orchestrator stages the
    written ``pools.json`` to the pod for ``issue2162_run.py --phase margin``
    (the dispatcher's margin leg HALTs rc=24 without it; r1 C4)."""
    pairs = BANK.build_pairs()
    anchor_rows = load_anchor_rows(cfg.anchors_file)
    scores = _anchor_behavior_scores(cfg)
    pools, report = build_margin_pools(pairs, anchor_rows, scores)
    assert pools, "zero pools built — the anchor waves' judge-filter kept nothing at > 50"
    if cfg.dry_run:
        logger.info(
            "[pools] dry-run: would build %d/%d keys (%d omitted, %d short, %d "
            "query_content skipped) — nothing persisted (zero-API phase)",
            report["n_keys_built"],
            report["n_keys_total"],
            report["n_keys_omitted"],
            report["n_keys_short"],
            report["query_content_skip"]["n_keys"],
        )
        return RC_OK
    out = cfg.work_root / "pools.json"
    J94._write_json_atomic(out, {"pools": pools, "meta": report})
    J94._write_json_atomic(cfg.gates_dir / "pools_report.json", report)
    logger.info(
        "[pools] %d/%d keys built (%d omitted, %d short; %d query_content keys "
        "skipped — no well-defined fixed pool) -> %s",
        report["n_keys_built"],
        report["n_keys_total"],
        report["n_keys_omitted"],
        report["n_keys_short"],
        report["query_content_skip"]["n_keys"],
        out,
    )
    return RC_OK


def phase_upload_raw(cfg: J94.JudgeConfig) -> int:
    """One folder commit of the judge work root -> the 2162 judge_raw prefix.
    ``--dry-run`` (fix 3): a Hub upload is a mutating API call — report the
    would-be upload and stop."""
    if cfg.dry_run:
        logger.info(
            "[upload-raw] dry-run: would upload %s -> %s (no Hub calls made)",
            cfg.work_root,
            f"{HF_PREFIX}/raw_completions/judge_raw",
        )
        return RC_OK
    from explore_persona_space.orchestrate import hub

    url = hub._upload(
        cfg.work_root,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        path_in_repo=f"{HF_PREFIX}/raw_completions/judge_raw",
        raise_on_error=True,
    )
    logger.info("[upload-raw] uploaded %s -> %s", cfg.work_root, url)
    return RC_OK


# ── CLI ───────────────────────────────────────────────────────────────

PHASES = {
    "pilot": phase_pilot,
    "separation-gate": phase_separation_gate,
    "anchors": phase_anchors,
    "waves": phase_waves,
    "stage2": phase_stage2,
    "pools": phase_pools,
    "audits": phase_audits,
    "upload-raw": phase_upload_raw,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2162 VM-side judge pipeline.")
    ap.add_argument("--phase", required=True, choices=tuple(PHASES))
    ap.add_argument(
        "--in-root",
        type=Path,
        default=Path("data/issue_2162/judge_inputs"),
        help=f"staging root; rollouts/anchors default under <in-root>/{HF_PREFIX}/raw_completions/",
    )
    ap.add_argument("--rollouts-dir", type=Path, default=None)
    ap.add_argument("--anchors-dir", type=Path, default=None)
    ap.add_argument("--stage2-dir", type=Path, default=None)
    ap.add_argument("--stage-from-hf", action="store_true")
    ap.add_argument("--hf-revision", type=str, default=None)
    ap.add_argument("--work-root", type=Path, default=Path("eval_results/issue_2162/judge"))
    ap.add_argument("--cache-root", type=Path, default=Path("data/issue_2162/judge_cache"))
    ap.add_argument("--judge-model", type=str, default=J94.DEFAULT_JUDGE_MODEL)
    ap.add_argument("--max-tokens", type=int, default=J94.DEFAULT_JUDGE_MAX_TOKENS)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="uniform construction check (fix 3): build + validate every judge unit "
        "the phase would dispatch, print counts/routing, ZERO API calls, nothing "
        "persisted; --phase pilot REFUSES it (rc 10) — its purpose is live measurement",
    )
    return ap.parse_args(argv)


_STAGE_GRID = f"{HF_PREFIX}/raw_completions/grid"
_STAGE_ANCHORS = f"{HF_PREFIX}/raw_completions/anchors"
_STAGE_ANCHORS_GATE = f"{HF_PREFIX}/raw_completions/anchors_gate"
_STAGE_STAGE2 = f"{HF_PREFIX}/raw_completions/stage2"

# Phase-aware staging (fix 2): stage only what the requested phase's loaders
# actually read. "required" prefixes FAIL LOUD on absence ("the phase needs
# grid rows and there are none"); an "anchors_any" phase needs anchor rows
# from EITHER prefix (`anchors_gate` is uploaded early at P2 so gate 3 can
# run BEFORE the terminal `anchors` upload; grid exists only from P3) — each
# member is tolerated individually, but ZERO landing raises; "optional"
# prefixes log-and-continue (phase_audits is is_dir-gated + fails loud on no
# inputs itself).
_PHASE_STAGE_PLAN: dict[str, dict[str, tuple[str, ...]]] = {
    "separation-gate": {"anchors_any": (_STAGE_ANCHORS, _STAGE_ANCHORS_GATE)},
    "pilot": {"required": (_STAGE_GRID,), "anchors_any": (_STAGE_ANCHORS, _STAGE_ANCHORS_GATE)},
    "anchors": {"required": (_STAGE_ANCHORS,)},
    "waves": {"required": (_STAGE_GRID,)},
    "stage2": {"required": (_STAGE_STAGE2,)},
    "pools": {"required": (_STAGE_ANCHORS,)},
    "audits": {"optional": (_STAGE_GRID, _STAGE_ANCHORS, _STAGE_STAGE2)},
    "upload-raw": {},
}


def _stage_inputs(args: argparse.Namespace) -> None:
    """Stage the requested phase's Hub inputs per ``_PHASE_STAGE_PLAN`` (fix 2)."""
    from explore_persona_space.orchestrate import hub

    plan = _PHASE_STAGE_PLAN[args.phase]

    def _stage(prefix: str, *, tolerate_missing: bool) -> bool:
        try:
            staged = hub.stage_hub_prefix(
                DATASET_REPO, prefix, args.in_root, revision=args.hf_revision
            )
        except FileNotFoundError:
            if not tolerate_missing:
                raise
            logger.info(
                "[stage] %s: not on the Hub yet — tolerated (--phase %s does not "
                "strictly require it)",
                prefix,
                args.phase,
            )
            return False
        logger.info("[stage] %s: %d files", prefix, len(staged))
        return len(staged) > 0

    for prefix in plan.get("required", ()):
        _stage(prefix, tolerate_missing=False)
    anchors_any = plan.get("anchors_any", ())
    if anchors_any and not any(_stage(p, tolerate_missing=True) for p in anchors_any):
        raise FileNotFoundError(
            f"--phase {args.phase} needs anchor rows but none of {list(anchors_any)} "
            f"exist on {DATASET_REPO} — the pod-side anchor uploads (P2 gate slice / "
            "terminal) have not landed yet"
        )
    for prefix in plan.get("optional", ()):
        _stage(prefix, tolerate_missing=True)


def _resolve_anchors_dir(mirror: Path) -> Path:
    """Default anchors dir (fix 2): the full ``anchors`` mirror when it holds
    shards, else the early-uploaded ``anchors_gate`` mirror (P2 uploads the
    gate slice there FIRST so ``--phase separation-gate`` can run before the
    terminal anchors upload lands). Falls back to the canonical path when
    neither holds shards — the loaders fail loud on absence."""
    full, gate = mirror / "anchors", mirror / "anchors_gate"
    if any(full.glob("anchors_*.jsonl")):
        return full
    if any(gate.glob("anchors_*.jsonl")):
        logger.info("[stage] anchors dir -> %s (full anchors prefix not staged yet)", gate)
        return gate
    return full


def build_config(args: argparse.Namespace) -> J94.JudgeConfig:
    mirror = args.in_root / HF_PREFIX / "raw_completions"
    rollouts = args.rollouts_dir if args.rollouts_dir is not None else mirror / "grid"
    # NOTE: JudgeConfig.anchors_file carries the anchors *directory* here (the
    # 2162 anchors are per-worker shards); only OUR loaders read it.
    anchors = args.anchors_dir if args.anchors_dir is not None else _resolve_anchors_dir(mirror)
    stage2 = args.stage2_dir
    if stage2 is None and args.phase == "stage2":
        stage2 = mirror / "stage2"
    return J94.JudgeConfig(
        work_root=args.work_root,
        cache_root=args.cache_root,
        rollouts_dir=rollouts,
        anchors_file=anchors,
        stage2_dir=stage2,
        judge_model=args.judge_model,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.stage_from_hf:
        _stage_inputs(args)
    cfg = build_config(args)
    for d in (cfg.scores_dir, cfg.items_dir, cfg.raw_dir, cfg.gates_dir, cfg.audits_dir):
        d.mkdir(parents=True, exist_ok=True)
    rc = PHASES[args.phase](cfg)
    logger.info("[phase=%s_done] rc=%d", args.phase, rc)
    return rc


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
