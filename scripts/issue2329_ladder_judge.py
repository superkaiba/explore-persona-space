#!/usr/bin/env python3
"""Issue #2329 q35_ladder_decay — VM-side ladder judge driver (plan §4.1 item 2).

Forked THIN from ``scripts/issue2162_ladder_judge.py`` with EXACTLY these
changes, everything else byte-inherited: ``import issue2329_judge as J62``
(HF_PREFIX/staging constants become ``issue2329_q35rerun/...`` automatically;
all 20 consumed attributes verified at plan time), work/in/cache roots under
``issue_2329/q35_ladder_decay`` + ``data/issue_2329``, parent-anchors staging
at the Q35 parent pin (``Q35_PARENT_HF_REVISION`` — #2329's own
``raw_completions/anchors{,_gate}``), and the donor screen's pe-viability leg
(F4: a ``no_prefix`` qualified PRIMARY falls to a pe-viable qualified
alternate; both no-prefix ⇒ the chosen donor's pe cell is later
skipped-with-record by ``issue2329_ladder.donor_maps_ladder``).

The #2094 judge machinery is IMPORTED wholesale (``issue2094_judge``:
``JudgeUnit`` / ``run_wave`` / wave-regime resume / per-arm drop-split
telemetry / the form-only coherence instrument VERBATIM), the ladder-specific
pieces live here:

- the 7-value HOLISTIC rubric registry (``ladder_bank.rubric_registry`` —
  one-property descriptors, Round-A instrument shape, rule-22 rubric-keyed
  cache partitions per rubric id),
- ``--phase pilot``: rule-26 pilot, ONE ``judge_pilot_gate`` call PER rubric
  (8 rubrics x >= 51 effective draws each ~= 448 draws — the #2124 sizing
  floor for the 2% parse-fail bar; plan §4.5),
- ``--phase gate``: the coherence-baseline sanity gate + the plan §7
  ANCHOR-SEPARATION gate (dual bars per (rung x carrier): target-descriptor
  sep >= 0.25 AND netted |Delta| sep >= 0.5, both POSITIVE; rung survives at
  >= 4/6 carriers; ALL rungs failing = rig-defect HALT rc=9) -> the verdict
  JSON ``gates/ladder_separation_gate.json`` the pod's ``--phase grid``
  consumes (``issue2329_ladder.read_gate_verdict``),
- ``--phase donor-screen``: the construct screen on cross-type donors (plan
  §4.2 — donor B-ceiling PARENT anchor draws judged under the recipient's
  target descriptor; qualify iff mean <= 15/100; primary -> alternate ->
  dropped+reported) -> ``gates/ladder_donor_screen.json``
  (``issue2329_ladder.read_donor_screen``),
- ``--phase pools``: TF-margin pools (zero API — a re-reduction of the gate
  wave's scores; 4+4 per direction, own-descriptor > 50) ->
  ``pools_ladder.json`` (``issue2329_run.load_pools`` schema, keys =
  direction ids),
- ``--phase waves``: P5 grid judge waves (coherence + hol-plain on EVERY
  rollout + the direction's own persona descriptor),
- ``--phase conjuncts``: the bounded R1/R2 STEERED per-conjunct diagnostic
  (Round-A conjunct instrument VERBATIM via
  ``issue2162_persona_rubric_rescore.CONJUNCTS``),
- ``--phase upload``: one folder commit of the work root to the Hub.

Routing (plan §9(ii)): EVERY wave forces the SYNC api_dispatch fan-out
(``threshold_base=FORCE_SYNC_THRESHOLD_BASE`` threaded through
``issue2094_judge.run_wave``) — the registered all-sync routing; the largest
wave here is <= ~2,160 items. Judge ``claude-sonnet-4-5-20250929``,
``max_tokens=1024``, N=1 draw, drop-never-coerce + bounded transport retry
(llm-judging rules 9/24/28 via the reused machinery).

Usage (VM, after the pod's L2 anchors upload):
    uv run python scripts/issue2329_ladder_judge.py --phase pilot --stage-from-hf
    uv run python scripts/issue2329_ladder_judge.py --phase gate
    uv run python scripts/issue2329_ladder_judge.py --phase donor-screen --stage-from-hf
    uv run python scripts/issue2329_ladder_judge.py --phase pools
    # after the pod's L4 grid upload:
    uv run python scripts/issue2329_ladder_judge.py --phase waves --stage-from-hf
    uv run python scripts/issue2329_ladder_judge.py --phase conjuncts
    uv run python scripts/issue2329_ladder_judge.py --phase upload
"""

from __future__ import annotations

# load_dotenv BEFORE any heavy/HF import (lint: --check-dotenv-before-hf-import)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2094_judge as J94  # noqa: E402  (same-dir script import; reused machinery)
import issue2329_judge as J62  # noqa: E402  (Q35 parent judge: pool constants, loaders, ids)
import issue2162_persona_rubric_rescore as RESCORE  # noqa: E402  (Round-A conjuncts)
from explore_persona_space.eval.judge_pilot import judge_pilot_gate  # noqa: E402
from explore_persona_space.experiments.issue2162 import ladder_bank as LB  # noqa: E402

logger = logging.getLogger("issue2329.ladder_judge")

DATASET_REPO = J62.DATASET_REPO
HF_PREFIX = J62.HF_PREFIX  # issue2329_q35rerun
LADDER_RAW = f"{HF_PREFIX}/raw_completions/ladder"
LADDER_TENSORS = f"{HF_PREFIX}/analysis_tensors/ladder"

# Q35 parent (#2329's Qwen2.5 rerun) artifact pin — the donor screen's parent
# anchors stage HERE, not at 2162's LB.PARENT_HF_REVISION (plan §4.1 item 2).
# Same pin as issue2329_ladder.Q35_PARENT_HF_REVISION (kept local: importing
# the ladder driver would pull torch into this VM-side judge process).
Q35_PARENT_HF_REVISION = "49d7f0017e3e3fb501e4e18952906d2c7804651a"

RC_OK = J62.RC_OK
RC_PILOT_GATE = J62.RC_PILOT_GATE  # 7
RC_COHERENCE_GATE = J62.RC_COHERENCE_GATE  # 8
RC_SEPARATION_GATE = J62.RC_SEPARATION_GATE  # 9
RC_DRY_RUN_UNSUPPORTED = J62.RC_DRY_RUN_UNSUPPORTED  # 10

# Plan §4.5 bars (pre-registered §7): dual anchor-separation bars, SIGNED
# (§7 gate 1: "threshold AND sign: separations must be POSITIVE").
TARGET_SEP_BAR = 0.25  # target-descriptor ceil - floor, 0-1 scale
NETTED_SEP_BAR = 0.5  # netted Delta-bar(ceiling) - Delta-bar(floor), max 2
RUNG_MIN_CARRIERS = 4  # rung survives iff >= 4 of 6 carriers pass
DONOR_SCREEN_BAR = 15.0  # plan §4.2: donor qualifies iff mean target score <= 15/100

# Rule-26 #2124 sizing: floor(1/0.02)+1 = 51 effective draws per rubric; 56
# gives headroom (8 rubrics x 56 ~= 448 total — the plan's "~440 draws").
PILOT_TARGET_PER_RUBRIC = 56
PILOT_SEED = 21620

JUDGE_N_DRAWS = J94.JUDGE_N_DRAWS  # 1 (carrier-clustered bootstrap carries uncertainty)
FORCE_SYNC = J62.FORCE_SYNC_THRESHOLD_BASE  # registered all-sync routing (plan §9(ii))

# Round-A per-conjunct instrument, VERBATIM (plan §4.4: R1/R2 steered only).
LADDER_CONJUNCTS: dict[str, dict[str, str]] = {
    "r1_pirate": RESCORE.CONJUNCTS["v1"],
    "r2_butler": RESCORE.CONJUNCTS["v3"],
}


# ── config ────────────────────────────────────────────────────────────


@dataclass
class LadderJudgeConfig(J94.JudgeConfig):
    """J94's config + the ladder's two extra inputs (donor screen)."""

    parent_anchors_dir: Path | None = None  # parent anchors @ Q35_PARENT_HF_REVISION
    bank_path: Path | None = None  # frozen ladder_bank.json (donor plan)

    @property
    def pools_path(self) -> Path:
        return self.work_root / "pools_ladder.json"


# ── rubric registry ───────────────────────────────────────────────────


def ladder_registry() -> dict[str, str]:
    """rubric_id -> production eval_prompt: coherence + the 7 holistic values."""
    return {J94.COHERENCE_RUBRIC_ID: J94.coherence_eval_prompt(), **LB.rubric_registry()}


def conjunct_registry() -> dict[str, str]:
    """rubric_id -> per-conjunct eval_prompt (``conj-<value>-<key>``)."""
    reg: dict[str, str] = {}
    for value_id, conjuncts in LADDER_CONJUNCTS.items():
        for key, clause in conjuncts.items():
            reg[f"conj-{value_id}-{key}"] = LB.holistic_eval_prompt(clause)
    return reg


# ── item builders ─────────────────────────────────────────────────────


def _rubric_anchor_rows(rid: str, anchor_rows: list[dict]) -> list[dict]:
    """The anchor rows a rubric's gate wave judges (plan §4.5 arithmetic):
    coherence + hol-plain cover EVERY anchor draw; hol-<X> covers X's own
    (ceiling) + the plain (floor) draws."""
    if rid in (J94.COHERENCE_RUBRIC_ID, LB.holistic_rubric_id("plain")):
        return anchor_rows
    assert rid.startswith("hol-"), rid
    value_id = rid[len("hol-") :]
    assert value_id in LB.PERSONA_VALUE_IDS, rid
    return [r for r in anchor_rows if r["value_id"] in (value_id, "plain")]


def build_gate_behavior_items(anchor_rows: list[dict]) -> dict[str, list[J94.JudgeUnit]]:
    """{rubric_id: units} for the anchor-separation gate wave (1,560 calls at
    the full 420-anchor grain: 420 coherence + 420 hol-plain + 6 x 120 hol-X)."""
    registry = LB.rubric_registry()
    by_rid: dict[str, list[J94.JudgeUnit]] = {}
    for rid in sorted(registry):
        for row in _rubric_anchor_rows(rid, anchor_rows):
            by_rid.setdefault(rid, []).append(
                J94.JudgeUnit(
                    item_id=J62.anchor_unit_id(row["context_id"], row["draw"], rid),
                    rubric_id=rid,
                    question="",
                    answer=row["text"],
                    source={**J62._anchor_source(row), "rubric": rid},
                )
            )
    return by_rid


def build_grid_behavior_items(grid_rows: list[dict]) -> dict[str, list[J94.JudgeUnit]]:
    """{rubric_id: units} for the P5 grid waves: hol-plain on EVERY rollout +
    the direction's own persona descriptor on its rows (plan §4.4 — F_target
    primary + netted bridge + plain mirror off the same two rubrics)."""
    by_rid: dict[str, list[J94.JudgeUnit]] = {}
    rid_plain = LB.holistic_rubric_id("plain")
    for row in grid_rows:
        for rid in (rid_plain, LB.holistic_rubric_id(row["persona"])):
            key = f"g|{row['block_key']}|{row['pair_id']}|{row['draw']}|{rid}"
            by_rid.setdefault(rid, []).append(
                J94.JudgeUnit(
                    item_id=J94._item_id("g", key),
                    rubric_id=rid,
                    question="",
                    answer=row["text"],
                    source={
                        **J62._grid_source(row),
                        "rubric": rid,
                        "persona": row["persona"],
                        "direction_kind": row["kind"],
                        "carrier": row["carrier"],
                    },
                )
            )
    return by_rid


def build_conjunct_items(grid_rows: list[dict]) -> dict[str, list[J94.JudgeUnit]]:
    """{rubric_id: units} for the R1/R2 STEERED per-conjunct diagnostic."""
    by_rid: dict[str, list[J94.JudgeUnit]] = {}
    for row in grid_rows:
        if row["arm"] != "steered" or row["persona"] not in LADDER_CONJUNCTS:
            continue
        for key_name in LADDER_CONJUNCTS[row["persona"]]:
            rid = f"conj-{row['persona']}-{key_name}"
            key = f"j|{row['block_key']}|{row['pair_id']}|{row['draw']}|{rid}"
            by_rid.setdefault(rid, []).append(
                J94.JudgeUnit(
                    item_id=J94._item_id("j", key),
                    rubric_id=rid,
                    question="",
                    answer=row["text"],
                    source={
                        **J62._grid_source(row),
                        "rubric": rid,
                        "persona": row["persona"],
                        "conjunct": key_name,
                    },
                )
            )
    return by_rid


# ── score readers (persisted wave rows -> lookup maps) ───────────────


def _scores_by_rid(cfg: LadderJudgeConfig, suffix: str) -> dict[str, dict[tuple[str, int], float]]:
    """rubric_id -> {(context_id, draw): kept mean score} from the persisted
    ``<rid>.<suffix>.scores.jsonl`` rows (rule-9 dropped rows carry
    ``score: null`` and are SKIPPED, never coerced)."""
    out: dict[str, dict[tuple[str, int], float]] = {}
    for f in sorted(cfg.scores_dir.glob(f"*.{suffix}.scores.jsonl")):
        for row in J94._iter_jsonl(f):
            rid = row.get("rubric_id")
            if rid is None or row.get("score") is None:
                continue
            out.setdefault(rid, {})[(row["context_id"], int(row["draw"]))] = float(row["score"])
    return out


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


# ── phase: pilot (plan §4.5; rule 26 / #2124) ─────────────────────────


def phase_pilot(cfg: LadderJudgeConfig) -> int:
    """One ``judge_pilot_gate`` call PER rubric (8 x ~56 draws), items from the
    P2 anchor texts, at the EXACT production instrument against a pilot-only
    cache. REFUSES ``--dry-run`` (the pilot's purpose is live measurement)."""
    if cfg.dry_run:
        logger.error(
            "[pilot] --dry-run refused: the rule-26 pilot measures the REAL instrument's "
            "truncation/parse-fail profile — run without --dry-run (~%d draws), or use "
            "--phase gate --dry-run for a free construction check.",
            PILOT_TARGET_PER_RUBRIC * (len(LB.rubric_registry()) + 1),
        )
        return RC_DRY_RUN_UNSUPPORTED
    anchor_rows = J62.load_anchor_rows(cfg.anchors_file)
    registry = ladder_registry()
    per_rubric: dict[str, dict] = {}
    all_pass = True
    for rid in sorted(registry):
        rows = _rubric_anchor_rows(rid, anchor_rows)
        items = [(J62.anchor_unit_id(r["context_id"], r["draw"], rid), "", r["text"]) for r in rows]
        report = judge_pilot_gate(
            {"anchors": items},
            registry[rid],
            max_tokens=cfg.max_tokens,
            cache_dir=cfg.pilot_cache_root / rid,
            save_raw_dir=cfg.raw_dir / "pilot" / rid,
            n_draws=JUDGE_N_DRAWS,
            target_total_draws=PILOT_TARGET_PER_RUBRIC,
            judge_model=cfg.judge_model,
            threshold_base=FORCE_SYNC,
            report_path=cfg.gates_dir / "pilot" / f"{rid}.json",
            seed=PILOT_SEED,
        )
        per_rubric[rid] = {
            "verdict": report.verdict,
            "failures": report.failures,
            "warnings": report.warnings,
            "n_total_draws": report.n_total_draws,
        }
        all_pass &= report.passed
        logger.info("[pilot] %s: %s (%d draws)", rid, report.verdict, report.n_total_draws)
    aggregate = {
        "passed": all_pass,
        "per_rubric": per_rubric,
        "instrument": {
            "judge_model": cfg.judge_model,
            "max_tokens": cfg.max_tokens,
            "n_draws": JUDGE_N_DRAWS,
            "n_rubrics": len(registry),
            "target_draws_per_rubric": PILOT_TARGET_PER_RUBRIC,
        },
        "repro": J94._repro(),
    }
    J94._write_json_atomic(cfg.gates_dir / "pilot_gate_report.json", aggregate)
    logger.info("[pilot] aggregate verdict: %s", "PASS" if all_pass else "FAIL")
    return RC_OK if all_pass else RC_PILOT_GATE


# ── phase: gate (coherence baseline + anchor-separation verdict) ──────


def _require_reports(cfg: LadderJudgeConfig, names: tuple[str, ...]) -> None:
    """Spend-bearing phases require the named gate reports present AND passed."""
    for name in names:
        path = cfg.gates_dir / name
        if not path.is_file():
            raise RuntimeError(f"gate report missing: {path} — run the producing phase first")
        if not json.loads(path.read_text(encoding="utf-8")).get("passed"):
            raise RuntimeError(f"gate FAILED per {path} — fix the instrument/bank and re-run")


def separation_verdict(anchor_rows: list[dict], scores: dict) -> dict:
    """Plan §7 gate 1: per (rung x carrier) dual-bar pass off the K=10 anchors;
    rung survives at >= RUNG_MIN_CARRIERS of 6. Verdict shape consumed by
    ``issue2329_ladder.read_gate_verdict`` (``rungs`` -> survived +
    surviving_carriers)."""
    draws_by_ctx: dict[str, list[int]] = {}
    for r in anchor_rows:
        draws_by_ctx.setdefault(r["context_id"], []).append(int(r["draw"]))
    rid_plain = LB.holistic_rubric_id("plain")
    rungs: dict[str, dict] = {}
    for value_id in LB.PERSONA_VALUE_IDS:
        rid_x = LB.holistic_rubric_id(value_id)
        per_carrier: dict[str, dict] = {}
        for carrier in LB.carrier_ids(LB.SEED):
            ceil_ctx = LB.context_id(value_id, carrier)
            floor_ctx = LB.context_id("plain", carrier)

            def _ctx_stats(ctx: str) -> tuple[float | None, float | None, int]:
                sx_vals: list[float] = []
                deltas: list[float] = []
                for draw in draws_by_ctx.get(ctx, []):
                    sx = scores.get(rid_x, {}).get((ctx, draw))
                    sp = scores.get(rid_plain, {}).get((ctx, draw))
                    if sx is not None:
                        sx_vals.append(sx / 100.0)
                    if sx is not None and sp is not None:
                        deltas.append((sx - sp) / 100.0)
                return _mean(sx_vals), _mean(deltas), len(sx_vals)

            ceil_sx, ceil_d, n_ceil = _ctx_stats(ceil_ctx)
            floor_sx, floor_d, n_floor = _ctx_stats(floor_ctx)
            unscored = None in (ceil_sx, floor_sx, ceil_d, floor_d)
            target_sep = None if unscored else ceil_sx - floor_sx
            netted_sep = None if unscored else ceil_d - floor_d
            passed = (
                not unscored
                and target_sep >= TARGET_SEP_BAR  # signed: ceiling ABOVE floor (§7)
                and netted_sep >= NETTED_SEP_BAR
            )
            per_carrier[carrier] = {
                "target_sep": target_sep,
                "netted_sep": netted_sep,
                "ceil_target_mean": ceil_sx,
                "floor_target_mean": floor_sx,
                "n_ceil_kept": n_ceil,
                "n_floor_kept": n_floor,
                "unscored": unscored,
                "passed": passed,
            }
        surviving = sorted(c for c, rec in per_carrier.items() if rec["passed"])
        rungs[value_id] = {
            "survived": len(surviving) >= RUNG_MIN_CARRIERS,
            "surviving_carriers": surviving,
            "n_carriers_pass": len(surviving),
            "per_carrier": per_carrier,
        }
    all_failed = not any(rec["survived"] for rec in rungs.values())
    return {
        "criterion": "ladder anchor-separation gate (plan §7 gate 1)",
        "bars": {
            "target_sep_bar": TARGET_SEP_BAR,
            "netted_sep_bar": NETTED_SEP_BAR,
            "rung_min_carriers": RUNG_MIN_CARRIERS,
            "sign": "positive (ceiling above floor)",
        },
        "rungs": rungs,
        "all_rungs_failed": all_failed,
        "passed": not all_failed,
        "repro": J94._repro(),
    }


def phase_gate(cfg: LadderJudgeConfig) -> int:
    """Coherence wave -> baseline gate -> behavior gate waves -> separation
    verdict JSON. ``--dry-run``: construction check, zero API calls."""
    anchor_rows = J62.load_anchor_rows(cfg.anchors_file)
    if cfg.dry_run:
        return J62._dry_run_units_report(
            "gate",
            {
                "coherence.anchors": J62.build_coherence_items(None, anchor_rows),
                **{
                    f"{rid}.anchors": us
                    for rid, us in build_gate_behavior_items(anchor_rows).items()
                },
            },
        )
    _require_reports(cfg, ("pilot_gate_report.json",))
    audits = J94.run_audits("anchors", anchor_rows, cfg.audits_dir)
    registry = ladder_registry()

    # Coherence-baseline sanity BEFORE any behavior-wave spend (plan §4.5).
    coh_units = J62.build_coherence_items(None, anchor_rows)
    J94.run_wave(
        "coherence.anchors",
        J94.COHERENCE_RUBRIC_ID,
        registry[J94.COHERENCE_RUBRIC_ID],
        coh_units,
        cfg,
        threshold_base=FORCE_SYNC,
    )
    coh_scores = list(J94._iter_jsonl(cfg.scores_dir / "coherence.anchors.scores.jsonl"))
    gate = J94.coherence_baseline_gate(coh_scores)
    gate["audits"] = audits
    J94._write_json_atomic(cfg.gates_dir / "coherence_baseline_gate.json", gate)
    logger.info(
        "[gate] coherence baseline: median=%.1f frac>60=%.3f -> %s",
        gate["median"],
        gate["frac_gt60"],
        "PASS" if gate["passed"] else "FAIL",
    )
    if not gate["passed"]:
        return RC_COHERENCE_GATE

    for rid, units in sorted(build_gate_behavior_items(anchor_rows).items()):
        J94.run_wave(f"{rid}.anchors", rid, registry[rid], units, cfg, threshold_base=FORCE_SYNC)

    verdict = separation_verdict(anchor_rows, _scores_by_rid(cfg, "anchors"))
    J94._write_json_atomic(cfg.gates_dir / "ladder_separation_gate.json", verdict)
    for value_id, rec in verdict["rungs"].items():
        logger.info(
            "[gate] rung %-18s carriers_pass=%d/6 -> %s",
            value_id,
            rec["n_carriers_pass"],
            "SURVIVES" if rec["survived"] else "DROPPED",
        )
    J94._refresh_summary(cfg)
    if verdict["all_rungs_failed"]:
        logger.error(
            "[gate] ALL rungs failed (incl. R1, whose Round-A separations clear both bars) "
            "— rig-defect evidence, HALT (plan §7 gate 1)"
        )
        return RC_SEPARATION_GATE
    return RC_OK


# ── phase: donor-screen (plan §4.2 construct screen) ──────────────────


def _load_donor_inputs(
    cfg: LadderJudgeConfig,
) -> tuple[list[dict], dict, dict, dict, frozenset[str]]:
    """(pair rows, donor plan, parent draws-by-ctx, parent texts, no-prefix ctx ids).

    ``parent_no_prefix_context_ids`` (F4, plan §4.1 item 3): parent contexts with
    no usable prefix-end slot — a donor drawn from them cannot serve the pe cell.
    The driver enriches the frozen ladder_bank.json with the field BEFORE its sha,
    so the same frozen manifest carries it; absent field ⇒ empty set (fail-safe:
    the plan measured EMPTY overlap at the pin).
    """
    assert cfg.bank_path is not None and cfg.bank_path.is_file(), (
        f"donor-screen needs the frozen ladder_bank.json (got {cfg.bank_path}) — "
        "stage it from the Hub (--stage-from-hf) or pass --bank-json"
    )
    manifest = json.loads(cfg.bank_path.read_text(encoding="utf-8"))
    pair_rows = manifest["pairs"]
    plan_map = manifest["crosstype_donor_plan"]
    assert pair_rows and plan_map, "ladder_bank.json carries no pairs/donor plan"
    np_ids = frozenset(manifest.get("parent_no_prefix_context_ids", []))
    assert cfg.parent_anchors_dir is not None, "donor-screen needs --parent-anchors-dir"
    parent_rows = J62.load_anchor_rows(cfg.parent_anchors_dir)
    draws_by_ctx: dict[str, list[int]] = {}
    texts: dict[tuple[str, int], str] = {}
    for r in parent_rows:
        draws_by_ctx.setdefault(r["context_id"], []).append(int(r["draw"]))
        texts[(r["context_id"], int(r["draw"]))] = r["text"]
    return pair_rows, plan_map, draws_by_ctx, texts, np_ids


def _donor_units(
    cands: list[tuple[str, str]],  # (donor_ctx, recipient rid)
    draws_by_ctx: dict[str, list[int]],
    texts: dict[tuple[str, int], str],
) -> dict[str, list[J94.JudgeUnit]]:
    """Deduped {rid: units} judging each needed (donor ctx, draw) under the
    recipient's target descriptor. Fails loud on a donor context with no
    parent anchor rows (wrong/partial parent staging)."""
    by_rid: dict[str, list[J94.JudgeUnit]] = {}
    seen: set[str] = set()
    for ctx, rid in sorted(set(cands)):
        draws = draws_by_ctx.get(ctx, [])
        if not draws:
            raise RuntimeError(
                f"donor-screen: donor context {ctx!r} has NO parent anchor rows — "
                f"parent anchors staging at pin {Q35_PARENT_HF_REVISION} is wrong/partial"
            )
        for draw in sorted(draws):
            iid = J94._item_id("d", f"d|{ctx}|{draw}|{rid}")
            if iid in seen:
                continue
            seen.add(iid)
            by_rid.setdefault(rid, []).append(
                J94.JudgeUnit(
                    item_id=iid,
                    rubric_id=rid,
                    question="",
                    answer=texts[(ctx, draw)],
                    source={
                        "kind": "donor",
                        "arm": "donorscreen",
                        "context_id": ctx,
                        "draw": draw,
                        "rubric": rid,
                    },
                )
            )
    return by_rid


def _donor_means(
    cfg: LadderJudgeConfig, stage: str, registry: dict[str, str], units_by_rid: dict
) -> dict[tuple[str, str], float | None]:
    """Run the stage's waves; return (donor_ctx, rid) -> kept mean score."""
    for rid, units in sorted(units_by_rid.items()):
        J94.run_wave(f"{rid}.{stage}", rid, registry[rid], units, cfg, threshold_base=FORCE_SYNC)
    per: dict[tuple[str, str], list[float]] = {}
    for f in sorted(cfg.scores_dir.glob(f"*.{stage}.scores.jsonl")):
        for row in J94._iter_jsonl(f):
            if row.get("score") is None:
                continue
            per.setdefault((row["context_id"], row["rubric_id"]), []).append(float(row["score"]))
    return {key: _mean(vals) for key, vals in per.items()}


def phase_donor_screen(cfg: LadderJudgeConfig) -> int:
    """Screen every cross-type donor assignment: primary -> (on failure)
    alternate -> dropped + reported (plan §4.2; bar <= 15/100). Writes the
    ``assignments`` JSON ``issue2329_ladder.read_donor_screen`` consumes.

    pe-viability leg (F4, plan §4.1 item 3): a ``no_prefix`` donor context
    cannot serve the pe cell, so a QUALIFIED primary drawn from one falls to
    a qualified pe-viable alternate (``pe_fallthrough: true``); when the
    alternate is unqualified or itself no-prefix, the primary is KEPT and the
    driver's ``donor_maps_ladder`` later skips-with-record that pair's
    (pe x null_xtype) cell. Fail-safe: the plan measured EMPTY overlap
    between no-prefix contexts and donor candidates at the pin.
    """
    pair_rows, plan_map, draws_by_ctx, texts, np_ids = _load_donor_inputs(cfg)
    registry = LB.rubric_registry()
    rid_for = {row["pair_id"]: LB.holistic_rubric_id(row["persona"]) for row in pair_rows}

    primary_cands = [
        (plan_map[row["pair_id"]]["primary"]["b"], rid_for[row["pair_id"]]) for row in pair_rows
    ]
    primary_units = _donor_units(primary_cands, draws_by_ctx, texts)
    if cfg.dry_run:
        return J62._dry_run_units_report(
            "donor-screen",
            {f"{rid}.donorscreen1": us for rid, us in primary_units.items()},
        )
    _require_reports(cfg, ("pilot_gate_report.json",))
    means1 = _donor_means(cfg, "donorscreen1", registry, primary_units)

    def _qualifies(mean: float | None) -> bool:
        # An unscoreable candidate (all draws dropped) cannot DEMONSTRATE
        # cleanliness — treated as failing, never silently qualified.
        return mean is not None and mean <= DONOR_SCREEN_BAR

    # means2 covers BOTH classes needing an alternate read: primaries that
    # FAILED the bar, and no-prefix primaries that PASSED it (their alternate
    # is judged so the pe-fallthrough selection below has a scored candidate).
    fail1 = [
        row
        for row in pair_rows
        if not _qualifies(
            means1.get((plan_map[row["pair_id"]]["primary"]["b"], rid_for[row["pair_id"]]))
        )
        or plan_map[row["pair_id"]]["primary"]["b"] in np_ids
    ]
    means2: dict[tuple[str, str], float | None] = {}
    if fail1:
        alt_cands = [
            (plan_map[row["pair_id"]]["alternate"]["b"], rid_for[row["pair_id"]]) for row in fail1
        ]
        means2 = _donor_means(
            cfg, "donorscreen2", registry, _donor_units(alt_cands, draws_by_ctx, texts)
        )

    assignments: dict[str, dict] = {}
    n_primary = n_alternate = n_dropped = n_pe_fallthrough = 0
    for row in pair_rows:
        pid = row["pair_id"]
        rid = rid_for[pid]
        primary, alternate = plan_map[pid]["primary"], plan_map[pid]["alternate"]
        p_mean = means1.get((primary["b"], rid))
        a_mean = means2.get((alternate["b"], rid))
        pe_fallthrough = False
        if _qualifies(p_mean) and primary["b"] not in np_ids:
            status, donor = "primary", primary
            n_primary += 1
        elif _qualifies(p_mean) and _qualifies(a_mean) and alternate["b"] not in np_ids:
            # no-prefix qualified primary -> pe-viable qualified alternate
            status, donor = "alternate", alternate
            pe_fallthrough = True
            n_alternate += 1
            n_pe_fallthrough += 1
        elif _qualifies(p_mean):
            # no-prefix qualified primary, no pe-viable qualified alternate:
            # KEEP the primary; the driver pe-excludes its null_xtype pe cell.
            status, donor = "primary", primary
            n_primary += 1
        elif _qualifies(a_mean):
            status, donor = "alternate", alternate
            n_alternate += 1
        else:
            status, donor = "dropped", None
            n_dropped += 1
        assignments[pid] = {
            "status": status,
            "donor": donor,
            "primary_mean": p_mean,
            "alternate_mean": a_mean,
            "pe_viable": donor is not None and donor["b"] not in np_ids,
            "pe_fallthrough": pe_fallthrough,
        }
    report = {
        "criterion": "cross-type donor construct screen (plan §4.2)",
        "bar": DONOR_SCREEN_BAR,
        "n_primary": n_primary,
        "n_alternate": n_alternate,
        "n_dropped": n_dropped,
        "n_pe_fallthrough": n_pe_fallthrough,
        "parent_no_prefix_context_ids": sorted(np_ids),
        "assignments": assignments,
        "repro": J94._repro(),
    }
    J94._write_json_atomic(cfg.gates_dir / "ladder_donor_screen.json", report)
    logger.info(
        "[donor-screen] %d primary / %d alternate (%d pe-fallthrough) / %d dropped "
        "(bar <= %.0f/100; %d no-prefix parent ctxs)",
        n_primary,
        n_alternate,
        n_pe_fallthrough,
        n_dropped,
        DONOR_SCREEN_BAR,
        len(np_ids),
    )
    return RC_OK


# ── phase: pools (zero API — plan §4.4 TF margin) ─────────────────────


def build_ladder_pools(
    anchor_rows: list[dict], scores: dict[str, dict[tuple[str, int], float]]
) -> tuple[dict[str, list[dict]], dict]:
    """Fixed 4+4 pools per DIRECTION from the gate anchors (own-descriptor
    score > ``POOL_FILTER_MIN``; top ``POOL_PER_SIDE`` per side by descending
    score, ties (context_id, draw)). Side A = value_a (source), side B =
    value_b (target). An empty side OMITS the direction (the pod's margin
    phase records explicit skip rows); a short side is kept + flagged."""
    texts = {(r["context_id"], int(r["draw"])): r["text"] for r in anchor_rows}
    draws_by_ctx: dict[str, list[int]] = {}
    for r in anchor_rows:
        draws_by_ctx.setdefault(r["context_id"], []).append(int(r["draw"]))
    dir_vals: dict[str, tuple[str, str]] = {}
    for p in LB.build_ladder_pairs(LB.SEED):
        dir_vals.setdefault(p.cell, (p.value_a, p.value_b))

    pools: dict[str, list[dict]] = {}
    report: dict[str, dict] = {}
    for direction in LB.direction_ids():
        value_a, value_b = dir_vals[direction]
        items: list[dict] = []
        side_meta: dict[str, dict] = {}
        for side, value in (("A", value_a), ("B", value_b)):
            rid = LB.holistic_rubric_id(value)
            cands: list[tuple[float, str, int]] = []
            for carrier in LB.carrier_ids(LB.SEED):
                ctx = LB.context_id(value, carrier)
                for draw in draws_by_ctx.get(ctx, []):
                    s = scores.get(rid, {}).get((ctx, draw))
                    if s is not None and s > J62.POOL_FILTER_MIN:
                        cands.append((-s, ctx, draw))
            top = sorted(cands)[: J62.POOL_PER_SIDE]
            side_meta[side] = {"n_candidates": len(cands), "n_kept": len(top)}
            items.extend(
                {
                    "side": side,
                    "text": texts[(ctx, draw)],
                    "context_id": ctx,
                    "draw": draw,
                    "score": -neg,
                }
                for neg, ctx, draw in top
            )
        n_a = side_meta["A"]["n_kept"]
        n_b = side_meta["B"]["n_kept"]
        rec = {
            "sides": side_meta,
            "short": min(n_a, n_b) > 0 and min(n_a, n_b) < J62.POOL_PER_SIDE,
        }
        if n_a == 0 or n_b == 0:
            rec["omitted"] = True
            report[direction] = rec
            continue  # margin consumer records explicit skip rows (drop + report)
        rec["omitted"] = False
        pools[direction] = items
        report[direction] = rec
    return pools, {
        "per_direction": report,
        "pool_per_side": J62.POOL_PER_SIDE,
        "filter_min": J62.POOL_FILTER_MIN,
        "n_directions_built": len(pools),
        "n_directions_total": len(LB.direction_ids()),
    }


def phase_pools(cfg: LadderJudgeConfig) -> int:
    """Persist ``pools_ladder.json`` (``issue2329_run.load_pools`` schema) —
    a zero-API re-reduction of the judged gate anchor waves."""
    anchor_rows = J62.load_anchor_rows(cfg.anchors_file)
    scores = _scores_by_rid(cfg, "anchors")
    assert scores, (
        f"no persisted anchor wave scores under {cfg.scores_dir} — run --phase gate first"
    )
    pools, report = build_ladder_pools(anchor_rows, scores)
    if cfg.dry_run:
        logger.info(
            "[pools] dry-run: would build %d/%d directions — nothing persisted",
            report["n_directions_built"],
            report["n_directions_total"],
        )
        return RC_OK
    assert pools, "zero pools built — the gate waves' judge-filter kept nothing at > 50"
    J94._write_json_atomic(
        cfg.pools_path, {"pools": pools, "meta": {**report, "repro": J94._repro()}}
    )
    J94._write_json_atomic(cfg.gates_dir / "ladder_pools_report.json", report)
    logger.info(
        "[pools] %d/%d directions built -> %s",
        report["n_directions_built"],
        report["n_directions_total"],
        cfg.pools_path,
    )
    return RC_OK


# ── phase: waves (P5 grid) + conjuncts ────────────────────────────────

_ALL_GATES = (
    "pilot_gate_report.json",
    "coherence_baseline_gate.json",
    "ladder_separation_gate.json",
)


def phase_waves(cfg: LadderJudgeConfig) -> int:
    """P5 grid judge waves: coherence + hol-plain on every rollout + the
    direction's own persona descriptor. Gate-guarded (plan DAG)."""
    grid_rows = J62.load_grid_rows(cfg.rollouts_dir)
    if cfg.dry_run:
        return J62._dry_run_units_report(
            "waves",
            {
                "coherence.grid": J62.build_coherence_items(grid_rows, None),
                **{f"{rid}.grid": us for rid, us in build_grid_behavior_items(grid_rows).items()},
            },
        )
    _require_reports(cfg, _ALL_GATES)
    registry = ladder_registry()
    J94.run_audits("grid", grid_rows, cfg.audits_dir)
    coh_units = J62.build_coherence_items(grid_rows, None)
    J94.run_wave(
        "coherence.grid",
        J94.COHERENCE_RUBRIC_ID,
        registry[J94.COHERENCE_RUBRIC_ID],
        coh_units,
        cfg,
        threshold_base=FORCE_SYNC,
    )
    for rid, units in sorted(build_grid_behavior_items(grid_rows).items()):
        J94.run_wave(f"{rid}.grid", rid, registry[rid], units, cfg, threshold_base=FORCE_SYNC)
    J94._refresh_summary(cfg)
    return RC_OK


def phase_conjuncts(cfg: LadderJudgeConfig) -> int:
    """Bounded R1/R2 STEERED per-conjunct diagnostic (plan §4.4; <= ~840 calls)."""
    grid_rows = J62.load_grid_rows(cfg.rollouts_dir)
    items = build_conjunct_items(grid_rows)
    if cfg.dry_run:
        return J62._dry_run_units_report(
            "conjuncts", {f"{rid}.grid": us for rid, us in items.items()}
        )
    _require_reports(cfg, _ALL_GATES)
    registry = conjunct_registry()
    if not items:
        logger.warning("[conjuncts] no R1/R2 steered rollouts in the grid shards — nothing to do")
        return RC_OK
    for rid, units in sorted(items.items()):
        J94.run_wave(f"{rid}.grid", rid, registry[rid], units, cfg, threshold_base=FORCE_SYNC)
    J94._refresh_summary(cfg)
    return RC_OK


# ── phase: upload ─────────────────────────────────────────────────────


def phase_upload(cfg: LadderJudgeConfig) -> int:
    """One folder commit of the judge work root -> the ladder judge-raw
    prefix, then a scoped exact-set verify.

    Judge raws (scores/, gates/, pools, pilot artifacts) are RAW-COMPLETIONS
    class artifacts — plan §4/§10 registers them under
    ``issue2329_q35rerun/raw_completions/ladder/judge_raw/``, never under
    analysis_tensors (review r1 must-fix 3).
    """
    dest = f"{LADDER_RAW}/judge_raw"
    if cfg.dry_run:
        logger.info(
            "[upload] dry-run: would upload %s -> %s (no Hub calls made)",
            cfg.work_root,
            dest,
        )
        return RC_OK
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    url = hub._upload(
        cfg.work_root,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        path_in_repo=dest,
        raise_on_error=True,
    )
    expected = sorted(
        f"{dest}/{p.relative_to(cfg.work_root).as_posix()}"
        for p in cfg.work_root.rglob("*")
        if p.is_file()
    )
    assert expected, f"[upload] no files under {cfg.work_root} — nothing to upload/verify"
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), DATASET_REPO, expected, path_in_repo=dest, repo_type="dataset"
    )
    assert not missing, (
        f"[upload] {len(missing)} of {len(expected)} file(s) missing after upload — "
        f"examples: {missing[:5]}"
    )
    logger.info(
        "[upload] uploaded + verified %d files: %s -> %s", len(expected), cfg.work_root, url
    )
    return RC_OK


# ── CLI ───────────────────────────────────────────────────────────────

PHASES = {
    "pilot": phase_pilot,
    "gate": phase_gate,
    "donor-screen": phase_donor_screen,
    "pools": phase_pools,
    "waves": phase_waves,
    "conjuncts": phase_conjuncts,
    "upload": phase_upload,
}

_STAGE_LADDER_ANCHORS = f"{LADDER_RAW}/anchors"
_STAGE_LADDER_GRID = f"{LADDER_RAW}/grid"
_STAGE_BANK_FILE = f"{LADDER_TENSORS}/vc_bank/ladder_bank.json"

# Phase-aware staging (the J62 fix-2 pattern): stage only what the phase's
# loaders read. Parent anchors are ALWAYS staged at the frozen Q35 parent pin
# (Q35_PARENT_HF_REVISION); ladder artifacts at --hf-revision (default main).
_PHASE_STAGE_PLAN: dict[str, dict[str, tuple[str, ...]]] = {
    "pilot": {"ladder": (_STAGE_LADDER_ANCHORS,)},
    "gate": {"ladder": (_STAGE_LADDER_ANCHORS,)},
    "donor-screen": {
        "ladder": (_STAGE_LADDER_ANCHORS,),
        "bank_file": (_STAGE_BANK_FILE,),
        "parent_anchors": (),
    },
    "pools": {"ladder": (_STAGE_LADDER_ANCHORS,)},
    "waves": {"ladder": (_STAGE_LADDER_GRID,)},
    "conjuncts": {"ladder": (_STAGE_LADDER_GRID,)},
    "upload": {},
}


def _stage_inputs(args: argparse.Namespace) -> None:
    """Stage the phase's Hub inputs (mirror-root convention: files land at
    ``<in-root>/<repo-relative path>``)."""
    from explore_persona_space.orchestrate import hub

    plan = _PHASE_STAGE_PLAN[args.phase]
    for prefix in plan.get("ladder", ()):
        staged = hub.stage_hub_prefix(DATASET_REPO, prefix, args.in_root, revision=args.hf_revision)
        logger.info("[stage] %s: %d files", prefix, len(staged))
    for path in plan.get("bank_file", ()):
        target = args.in_root / path
        hub.stage_hub_file(
            DATASET_REPO, path, target, repo_type="dataset", revision=args.hf_revision
        )
        logger.info("[stage] %s -> %s", path, target)
    if "parent_anchors" in plan:
        # The parent's anchors_gate prefix is the early-uploaded slice; the
        # terminal anchors prefix supersedes it (J62._resolve_anchors_dir).
        for prefix in (J62._STAGE_ANCHORS, J62._STAGE_ANCHORS_GATE):
            try:
                staged = hub.stage_hub_prefix(
                    DATASET_REPO, prefix, args.in_root, revision=Q35_PARENT_HF_REVISION
                )
            except FileNotFoundError:
                logger.info("[stage] %s: absent at the parent pin — tolerated", prefix)
                continue
            logger.info("[stage] %s (@parent pin): %d files", prefix, len(staged))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #2329 ladder VM-side judge driver "
        "(persona-specificity ladder ported to Qwen3.5-9B)."
    )
    ap.add_argument(
        "--phase",
        choices=tuple(PHASES),
        help="pipeline phase to run (required unless --import-check)",
    )
    ap.add_argument(
        "--in-root",
        type=Path,
        default=Path("data/issue_2329/ladder_judge_inputs"),
        help="staging mirror root (files land at <in-root>/<repo-relative path>)",
    )
    ap.add_argument("--rollouts-dir", type=Path, default=None)
    ap.add_argument("--anchors-dir", type=Path, default=None)
    ap.add_argument("--parent-anchors-dir", type=Path, default=None)
    ap.add_argument("--bank-json", type=Path, default=None, help="frozen ladder_bank.json path")
    ap.add_argument("--stage-from-hf", action="store_true")
    ap.add_argument(
        "--hf-revision",
        type=str,
        default=None,
        help="revision for the LADDER artifacts (parent anchors always stage at "
        "the frozen Q35_PARENT_HF_REVISION pin)",
    )
    ap.add_argument(
        "--work-root",
        type=Path,
        default=Path("eval_results/issue_2329/q35_ladder_decay/judge"),
    )
    ap.add_argument("--cache-root", type=Path, default=Path("data/issue_2329/ladder_judge_cache"))
    ap.add_argument("--judge-model", type=str, default=J94.DEFAULT_JUDGE_MODEL)
    ap.add_argument("--max-tokens", type=int, default=J94.DEFAULT_JUDGE_MAX_TOKENS)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="uniform construction check: build + validate every judge unit the phase "
        "would dispatch, ZERO API calls, nothing persisted; --phase pilot REFUSES it (rc 10)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="execute deferred imports + registry invariants + argparse-attribute "
        "completeness, then exit 0 (no inputs read, no API calls)",
    )
    return ap.parse_args(argv)


def build_config(args: argparse.Namespace) -> LadderJudgeConfig:
    mirror = args.in_root / HF_PREFIX / "raw_completions"
    rollouts = args.rollouts_dir if args.rollouts_dir is not None else mirror / "ladder/grid"
    anchors = args.anchors_dir if args.anchors_dir is not None else mirror / "ladder/anchors"
    parent_anchors = (
        args.parent_anchors_dir
        if args.parent_anchors_dir is not None
        else J62._resolve_anchors_dir(mirror)
    )
    bank = args.bank_json if args.bank_json is not None else args.in_root / _STAGE_BANK_FILE
    return LadderJudgeConfig(
        work_root=args.work_root,
        cache_root=args.cache_root,
        rollouts_dir=rollouts,
        anchors_file=anchors,  # the ladder anchors DIRECTORY (per-worker shards)
        stage2_dir=None,
        judge_model=args.judge_model,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
        parent_anchors_dir=parent_anchors,
        bank_path=bank,
    )


def _import_check() -> int:
    """Deferred-import + registry-invariant + argparse-completeness check.

    Module-level function (never inline in ``main`` — an in-function import
    binds the name function-wide and shadows module-level symbols; gotchas.md
    ``UnboundLocalError`` entry)."""
    from explore_persona_space.orchestrate import hub  # deferred in staging/upload paths
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert callable(hub.stage_hub_prefix) and callable(hub.stage_hub_file)
    assert callable(hub._upload)
    assert_args_attributes_defined(__file__)
    registry = ladder_registry()
    assert len(registry) == 8, sorted(registry)  # coherence + 7 holistic values
    assert len(LB.direction_ids()) == 12
    assert len(conjunct_registry()) == len(RESCORE.CONJUNCTS["v1"]) + len(RESCORE.CONJUNCTS["v3"])
    for rid, prompt in {**registry, **conjunct_registry()}.items():
        assert "{answer}" in prompt, (rid, "rubric missing the {answer} slot")
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
    if args.phase is None:
        raise SystemExit("--phase is required unless --import-check")
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
