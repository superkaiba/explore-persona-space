#!/usr/bin/env python3
"""Issue #2162 tbmp — VM-side judge waves, gates, F tables, stats (plan v10).

Steps (plan §4.5 P4/P5):

- ``--step pilot``: G3 judge pilot — 540 sync draws (180 d1 rollouts, 60 per
  arm x 3 arms, each judged under coherence + a representative A-descriptor +
  B-descriptor rubric => 60 effective draws per (arm x rubric), >= the 51
  floor for the 2% threshold), via ``eval.judge_pilot.judge_pilot_gate``.
- ``--step wave1``: d1 joint-tb waves (3,240 calls), SYNC (forced), pilot-gated.
- ``--step g2``: the d1-identity halt gate — per-arm paired per-pair ΔF
  (netted space) between this round's tb@d1 and the parent's committed ce
  rows, pooled over the 66 surviving d1 pairs; PASS <=> |mean ΔF| <= 0.10 per
  arm. FAIL freezes wave-2 (plan §7 G4).
- ``--step wave2``: the remaining waves (~21k calls), Batch API, G2-gated.
- ``--step f-tables``: per-(pair x slot x arm) F in BOTH spaces (netted +
  target-descriptor-only) + the raw-scale companion -> the round's
  ``*_tb.jsonl`` tables under ``eval_results/issue_2162/turn_boundary/``.
- ``--step parent-ref``: re-aggregates the PARENT's single-ce grid rollouts
  from the parent's own per-rubric judge scores into BOTH spaces (plan §6
  convention / §12.8: persona-cell parent-ce reference points are re-read in
  target-descriptor-only space from the per-descriptor scores, never lifted
  netted from ``f_cells.jsonl`` into a target-only panel), with a per-row
  parity assert that the recomputed NETTED F reproduces the committed parent
  ``f_beh`` (assumption-8 "re-aggregate both ways" verification) ->
  ``parent_ref_cells_tb.jsonl``.
- ``--step stats``: registered tests (TB-joint m=7 IUT/Holm, TB-sweep m=12),
  pair-clustered bootstrap CIs, the §3 verdict lattice (J1/J5/D/δ_disp) with
  the edit-artifact + denominator overlays -> ``stats_tb.json``.
- ``--step margin``: TF-margin aggregation over the pod margin shards ->
  ``margin_cells_tb.jsonl``.
- ``--step analysis``: f-tables -> parent-ref -> stats -> margin.

Reuses the parent machinery wholesale: ``issue2162_judge`` item builders +
rubric registry (the content-hashed rubric ids are DETERMINISTIC, so this
round's instrument is bit-identical to the parent's), ``issue2094_judge``
``run_wave`` (resume, telemetry, transport retry), ``issue2162_analysis``
``holm`` / ``_wilcoxon_exact_p`` / ``load_wave_scores`` / io helpers, and
``issue2094_analysis.bootstrap_family_means_batched``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np  # noqa: E402

import issue2094_judge as J94  # noqa: E402
import issue2162_analysis as A  # noqa: E402
import issue2162_judge as J  # noqa: E402
import issue2162_recency_rawscale as RS  # noqa: E402
import issue2162_tbmp as TB  # noqa: E402
from issue2094_analysis import bootstrap_family_means_batched  # noqa: E402

from explore_persona_space.eval.judge_pilot import judge_pilot_gate  # noqa: E402
from explore_persona_space.experiments.issue2162 import bank2162 as BANK  # noqa: E402

logger = logging.getLogger("issue2162.tbmp.analysis")

RC_OK = 0
RC_PILOT_GATE = J.RC_PILOT_GATE  # 7
RC_G2_IDENTITY = 12  # distinct typed HALT (plan §7 G4)

D1_CELLS: tuple[str, ...] = ("instr_format", "persona_prompted")
BASES: tuple[str, ...] = ("instr_format", "persona_prompted")
PILOT_SEED = 2162
PILOT_PER_ARM = 60  # 60 x 3 arms x 3 rubrics = 540 draws (plan §7 gate 3)
G2_BAR = 0.10
G2_EXPECTED_SURVIVORS = 66  # instr_format 36/36 + persona 30/36 (plan §7 G4)

COHERENCE_THRESHOLD = A.COHERENCE_THRESHOLD
SEPARATION_BAR = A.SEPARATION_BAR
SURVIVAL_FLOOR = A.SURVIVAL_FLOOR
BOOT_B = A.BOOT_B
BOOT_SEED = A.BOOT_SEED
HOLM_ALPHA = A.HOLM_ALPHA

HF_JUDGE_RAW_SCORES = f"{J.HF_PREFIX}/raw_completions/judge_raw/scores"
HF_PARENT_GRID = f"{J.HF_PREFIX}/raw_completions/grid"

# READ-side companion to issue2162_run.HF_DATA_WRITE_REPO (#2304). When THIS
# round's uploads were rerouted off the file-capped canonical repo, its own
# artifacts stopped living beside the PARENT's, so staging has to resolve
# per-prefix: round-owned prefixes follow the write destination, parent-owned
# ones (judge raw scores, parent grid) stay canonical. Unset => every prefix
# resolves to J.DATASET_REPO, i.e. byte-identical legacy behavior.
ROUND_OWNED_HF_PREFIXES = (
    f"{J.HF_PREFIX}/raw_completions/tbmp/grid",
    f"{J.HF_PREFIX}/analysis_tensors/tbmp/margin",
)


def _repo_for_prefix(prefix: str) -> str:
    """Repo a staged prefix lives in (round-owned vs parent-owned)."""
    if prefix in ROUND_OWNED_HF_PREFIXES:
        return os.environ.get("EPM_2162_DATA_WRITE_REPO", J.DATASET_REPO)
    return J.DATASET_REPO


PARENT_F_PARITY_TOL = 1e-6


def registered_space(cell: str) -> str:
    """Plan §4.4: target-descriptor-only for persona cells, netted otherwise."""
    return "target_only" if BANK.base_type_of(cell) == "persona_prompted" else "netted"


def _is_d1_row(row: dict) -> bool:
    return row["cell"] in D1_CELLS and row["slot"] == "tb"


# ── parent anchor channels (netted committed; B-channel from parent scores) ──


def load_committed_anchors(parent_metrics_dir: Path) -> dict[str, dict]:
    path = parent_metrics_dir / "anchors.jsonl"
    assert path.exists(), f"{path} missing — the parent's committed netted anchor table"
    return {r["pair_id"]: r for r in A._iter_jsonl(path)}


def load_parent_anchor_scores(parent_scores_dir: Path) -> dict[str, float | None]:
    """item_id -> score over the parent's ``*.anchors.scores.jsonl`` waves.

    Staged from ``{HF_JUDGE_RAW_SCORES}`` (the parent uploaded its judge work
    root there via ``issue2162_judge.py --phase upload-raw``)."""
    files = sorted(parent_scores_dir.glob("*.anchors.scores.jsonl"))
    assert files, (
        f"no parent anchor score files under {parent_scores_dir} — stage them: "
        f"hub.stage_hub_prefix('{J.DATASET_REPO}', '{HF_JUDGE_RAW_SCORES}', <in_root>) "
        f"(or --stage-from-hf)"
    )
    out: dict[str, float | None] = {}
    for f in files:
        for row in A._iter_jsonl(f):
            out[row["item_id"]] = row["score"]
    return out


def load_anchor_coherent_draws(parent_scores_dir: Path) -> dict[str, list[int]]:
    """context_id -> coherent anchor draw list (parent coherence.anchors wave)."""
    path = parent_scores_dir / "coherence.anchors.scores.jsonl"
    assert path.exists(), f"{path} missing — stage {HF_JUDGE_RAW_SCORES}"
    out: dict[str, list[int]] = defaultdict(list)
    for row in A._iter_jsonl(path):
        if row["score"] is not None and row["score"] > COHERENCE_THRESHOLD:
            out[row["context_id"]].append(row["draw"])
    return out


def pair_anchor_channels(
    pair: BANK.Pair2162,
    coh_draws: dict[str, list[int]],
    anchor_scores: dict[str, float | None],
) -> dict | None:
    """Per-pair floor/ceiling means in BOTH channels from parent per-draw scores."""
    cores = J.pair_rubric_cores(pair)
    if cores is None:
        return None
    rid_a, rid_b = (J.rubric_core_id(c) for c in cores)
    stats: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for role, ctx in (("floor", pair.a), ("ceiling", pair.b)):
        netted: list[float] = []
        bchan: list[float] = []
        for draw in coh_draws.get(ctx, ()):
            sa = anchor_scores.get(J.anchor_unit_id(ctx, draw, rid_a))
            sb = anchor_scores.get(J.anchor_unit_id(ctx, draw, rid_b))
            if sa is None or sb is None:
                continue
            netted.append((sb - sa) / 100.0)
            bchan.append(sb / 100.0)
        stats[f"delta_{role}"] = sum(netted) / len(netted) if netted else None
        stats[f"b_{role}"] = sum(bchan) / len(bchan) if bchan else None
        counts[role] = len(netted)
    return {**stats, "n_floor": counts["floor"], "n_ceiling": counts["ceiling"]}


def _assert_anchor_parity(recomputed: dict, committed: dict, pair_id: str) -> None:
    """The recomputed netted floor/ceiling means must reproduce the committed
    ``anchors.jsonl`` values — a strong join check on the parent-score staging."""
    for role in ("floor", "ceiling"):
        got, want = recomputed[f"delta_{role}"], committed[f"delta_{role}_mean"]
        if want is None or got is None:
            continue
        assert abs(got - want) <= 1e-6, (
            f"anchor parity FAIL pair={pair_id} {role}: recomputed {got} vs committed {want} "
            "— the staged parent scores do not reproduce the committed anchor table"
        )


# ── tbmp F computation (both spaces + raw-scale) ──────────────────────


def tb_pair_cells(
    grid_rows: list[dict],
    scores: dict[str, float | None],
    committed_anchors: dict[str, dict],
    channels: dict[str, dict | None],
    pairs_by_id: dict[str, BANK.Pair2162],
) -> dict[str, list[dict]]:
    """One record per (pair x slot x arm), F in both spaces + raw movement."""
    by_cell_rows: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in grid_rows:
        by_cell_rows[(row["pair_id"], row["slot"], row["arm"])].append(row)
    tables: dict[str, list[dict]] = {"steered": [], "shuffled": [], "crosstype": []}
    for (pair_id, slot, arm), rows in sorted(by_cell_rows.items()):
        p = pairs_by_id[pair_id]
        anc = committed_anchors.get(pair_id)
        chan = channels.get(pair_id)
        netted_deltas: list[float] = []
        b_vals: list[float] = []
        n_coherent = 0
        n_cap = 0
        for row in rows:
            n_cap += int(row.get("cap_hit", False))
            coh = scores.get(A.J.J94._item_id("c", f"g|{row['block_key']}|{pair_id}|{row['draw']}"))
            if coh is None or coh <= COHERENCE_THRESHOLD:
                continue
            n_coherent += 1
            sa = A._grid_behavior_score(scores, "g", row["block_key"], pair_id, row["draw"], "a")
            sb = A._grid_behavior_score(scores, "g", row["block_key"], pair_id, row["draw"], "b")
            if sa is None or sb is None:
                continue
            netted_deltas.append((sb - sa) / 100.0)
            b_vals.append(sb / 100.0)

        def _f(patched: list[float], floor: float | None, ceil: float | None) -> float | None:
            if not patched or floor is None or ceil is None or abs(ceil - floor) <= 1e-9:
                return None
            return (sum(patched) / len(patched) - floor) / (ceil - floor)

        def _raw(patched: list[float], floor: float | None, ceil: float | None) -> float | None:
            if not patched or floor is None or ceil is None:
                return None
            direction = 1.0 if ceil >= floor else -1.0
            return direction * (sum(patched) / len(patched) - floor)

        d_floor = anc["delta_floor_mean"] if anc else None
        d_ceil = anc["delta_ceiling_mean"] if anc else None
        b_floor = chan["b_floor"] if chan else None
        b_ceil = chan["b_ceiling"] if chan else None
        f_netted = _f(netted_deltas, d_floor, d_ceil)
        f_target = _f(b_vals, b_floor, b_ceil)
        space = registered_space(p.cell)
        f_registered = f_target if space == "target_only" else f_netted
        raw_registered = (
            _raw(b_vals, b_floor, b_ceil)
            if space == "target_only"
            else _raw(netted_deltas, d_floor, d_ceil)
        )
        rec = {
            "pair_id": pair_id,
            "cell": p.cell,
            "slot": slot,
            "variant": rows[0].get("variant"),
            "k": rows[0].get("k"),
            "arm": arm,
            "carrier": p.carrier,
            "value_a": p.value_a,
            "value_b": p.value_b,
            "donor_pair_id": rows[0].get("donor_pair_id"),
            "alignment": rows[0].get("alignment"),
            "alignment_dropped": rows[0].get("alignment_dropped"),
            "n_positions": rows[0].get("n_positions"),
            "realized_mode": rows[0].get("realized_mode"),
            "n_draws": len(rows),
            "n_coherent": n_coherent,
            "n_scored": len(netted_deltas),
            "n_cap_hit": n_cap,
            "delta_patched_mean": (
                sum(netted_deltas) / len(netted_deltas) if netted_deltas else None
            ),
            "b_patched_mean": sum(b_vals) / len(b_vals) if b_vals else None,
            "f_netted": f_netted,
            "f_target_only": f_target,
            "registered_space": space,
            "f_beh": f_registered,  # registered space (per §4.4)
            "raw_move_registered": raw_registered,
            "separation": anc["separation"] if anc else None,
            "len_delta": rows[0].get("len_delta"),
        }
        tables[arm].append(rec)
    return tables


# ── step: pilot (G3) ──────────────────────────────────────────────────


def step_pilot(args: argparse.Namespace) -> int:
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    registry = J.rubric_registry(pairs)
    rows = [r for r in J.load_grid_rows(args.rollouts_dir) if _is_d1_row(r)]
    assert rows, "no d1 tb rows — the pilot samples the FIRST uploaded d1 shards"
    rng = random.Random(PILOT_SEED)
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["cell"], r["pair_id"], r["draw"])):
        by_arm[r["arm"]].append(r)
    sample: list[dict] = []
    for arm in ("steered", "shuffled", "crosstype"):
        pool = by_arm.get(arm, [])
        assert len(pool) >= PILOT_PER_ARM, (arm, len(pool))
        sample.extend(rng.sample(pool, PILOT_PER_ARM))
    # Representative pair for the A/B descriptor rubric roles (deterministic:
    # max sampled-row count, tie-break pair_id). Off-pair rollouts under the
    # representative rubrics are an INSTRUMENT-IDENTICAL truncation/parse
    # probe (same template + 0-100 scale), recorded in the pilot record.
    counts: dict[str, int] = defaultdict(int)
    for r in sample:
        counts[r["pair_id"]] += 1
    rep_id = sorted(counts, key=lambda k: (-counts[k], k))[0]
    cores = J.pair_rubric_cores(pairs_by_id[rep_id])
    assert cores is not None
    rid_a, rid_b = (J.rubric_core_id(c) for c in cores)

    roles = {
        "coherence": (J94.COHERENCE_RUBRIC_ID, "c"),
        "a-descriptor": (rid_a, "pa"),
        "b-descriptor": (rid_b, "pb"),
    }
    per_role: dict[str, dict] = {}
    all_pass = True
    for role, (rid, tag) in roles.items():
        arms: dict[str, list[tuple[str, str, str]]] = {}
        for i, r in enumerate(sample):
            iid = (
                A.J.J94._item_id("c", f"g|{r['block_key']}|{r['pair_id']}|{r['draw']}")
                if role == "coherence"
                else J94._item_id(tag, f"{tag}|{r['block_key']}|{r['pair_id']}|{r['draw']}|{i}")
            )
            arms.setdefault(r["arm"], []).append((iid, "", r["text"]))
        report = judge_pilot_gate(
            arms,
            registry[rid],
            max_tokens=args.max_tokens,
            cache_dir=args.cache_root / "_pilot" / rid,
            save_raw_dir=args.work_root / "raw" / "pilot" / role,
            n_draws=J.JUDGE_N_DRAWS,
            target_total_draws=3 * PILOT_PER_ARM,
            judge_model=args.judge_model,
            report_path=args.work_root / "gates" / "pilot" / f"{role}.json",
            seed=PILOT_SEED,
            parse_fail_threshold=0.02,
        )
        per_role[role] = {
            "rubric_id": rid,
            "verdict": report.verdict,
            "failures": report.failures,
            "warnings": report.warnings,
            "n_total_draws": report.n_total_draws,
        }
        all_pass &= report.passed
        logger.info(
            "[pilot] %s (%s): %s (%d draws)", role, rid, report.verdict, report.n_total_draws
        )
    aggregate = {
        "passed": all_pass,
        "per_role": per_role,
        "pilot_record": {
            "sampling": f"d1 shards ONLY ({'/'.join(D1_CELLS)}, slot tb), seeded "
            f"random.Random({PILOT_SEED}), {PILOT_PER_ARM} rollouts per arm x 3 arms; "
            "wave-2 covers longer d3/d5 rollouts — instrument-identical, precedented "
            "(plan §7 gate 3 pilot-record note)",
            "representative_pair": rep_id,
            "representative_rubrics": {"a": rid_a, "b": rid_b},
            "off_pair_probe_note": "all 180 sampled rollouts are judged under the "
            "representative pair's A/B descriptor rubrics (instrument-identical "
            "truncation/parse probe); off-pair scores live ONLY in the _pilot cache "
            "root under pilot-specific item ids — never the production cache",
            "cache_hygiene": "cache_dir=<cache_root>/_pilot/<rid> (pilot-only root, "
            "rule 24(ii)); production waves use <cache_root>/<rid>",
            "effective_draws_per_arm_rubric": PILOT_PER_ARM,
            "floor_51_note": "60 >= floor(1/0.02)+1 = 51 (rule-26/#2124 satisfiability)",
        },
        "instrument": {
            "judge_model": args.judge_model,
            "max_tokens": args.max_tokens,
            "n_draws": J.JUDGE_N_DRAWS,
        },
        "repro": J94._repro(),
    }
    A._write_json_atomic(args.work_root / "gates" / "pilot_gate_tbmp.json", aggregate)
    logger.info("[pilot] aggregate verdict: %s", "PASS" if all_pass else "FAIL")
    return RC_OK if all_pass else RC_PILOT_GATE


# ── steps: waves ──────────────────────────────────────────────────────


def _judge_cfg(args: argparse.Namespace) -> J94.JudgeConfig:
    return J94.JudgeConfig(
        work_root=args.work_root,
        cache_root=args.cache_root,
        rollouts_dir=args.rollouts_dir,
        # anchors_file is consumed only by the anchor-wave paths (never run
        # here) — but keep the FILE-kind contract honest: point at the
        # parent's committed anchors.jsonl, never a directory (M1, r1).
        anchors_file=args.parent_metrics_dir / "anchors.jsonl",
        stage2_dir=None,
        judge_model=args.judge_model,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
    )


def _require_report(path: Path, what: str) -> None:
    assert path.exists(), f"{what} report missing at {path} — run its step first"
    payload = json.loads(path.read_text())
    assert payload.get("passed"), f"{what} FAILED ({path}) — HALT, never dispatch past it"


def _run_waves(args: argparse.Namespace, rows: list[dict], wave_tag: str, sync: bool) -> int:
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    registry = J.rubric_registry(pairs)
    cfg = _judge_cfg(args)
    threshold_base = J.FORCE_SYNC_THRESHOLD_BASE if sync else None
    coh_units = J.build_coherence_items(rows, None)
    J94.run_wave(
        f"coherence.{wave_tag}.grid",
        J94.COHERENCE_RUBRIC_ID,
        registry[J94.COHERENCE_RUBRIC_ID],
        coh_units,
        cfg,
        threshold_base=threshold_base,
    )
    for rid, units in sorted(J.build_grid_behavior_items(rows, pairs_by_id).items()):
        J94.run_wave(
            f"{rid}.{wave_tag}.grid", rid, registry[rid], units, cfg, threshold_base=threshold_base
        )
    return RC_OK


def step_wave1(args: argparse.Namespace) -> int:
    _require_report(args.work_root / "gates" / "pilot_gate_tbmp.json", "pilot gate (G3)")
    rows = [r for r in J.load_grid_rows(args.rollouts_dir) if _is_d1_row(r)]
    assert rows, "no d1 rows for wave-1"
    logger.info("[wave1] %d d1 rollouts (sync route)", len(rows))
    return _run_waves(args, rows, "w1", sync=True)


def step_wave2(args: argparse.Namespace) -> int:
    _require_report(args.out_dir / "identity_gate.json", "G2 d1-identity gate")
    rows = [r for r in J.load_grid_rows(args.rollouts_dir) if not _is_d1_row(r)]
    assert rows, "no non-d1 rows for wave-2"
    logger.info("[wave2] %d d3/d5+sweep rollouts (Batch API route)", len(rows))
    return _run_waves(args, rows, "w2", sync=False)


# ── step: G2 d1-identity gate ─────────────────────────────────────────


def step_g2(args: argparse.Namespace) -> int:
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    committed = load_committed_anchors(args.parent_metrics_dir)
    scores = A.load_wave_scores(args.work_root / "scores", "grid")
    grid_rows = [r for r in J.load_grid_rows(args.rollouts_dir) if _is_d1_row(r)]
    tables = tb_pair_cells(grid_rows, scores, committed, defaultdict(lambda: None), pairs_by_id)
    tb_f: dict[tuple[str, str], float] = {}
    for arm in ("steered", "shuffled"):
        for r in tables[arm]:
            if r["f_netted"] is not None:
                tb_f[(r["pair_id"], arm)] = r["f_netted"]
    parent_files = {"steered": "f_cells.jsonl", "shuffled": "null_shuffled_cells.jsonl"}
    parent_f: dict[tuple[str, str], float] = {}
    for arm, fname in parent_files.items():
        for r in A._iter_jsonl(args.parent_metrics_dir / fname):
            if r["cell"] in D1_CELLS and r["slot"] == "ce" and r["f_beh"] is not None:
                parent_f[(r["pair_id"], arm)] = r["f_beh"]
    surviving = sorted(
        pid
        for pid, anc in committed.items()
        if anc["cell"] in D1_CELLS
        and anc["separation"] is not None
        and abs(anc["separation"]) >= SEPARATION_BAR
    )
    assert len(surviving) == G2_EXPECTED_SURVIVORS, (
        f"surviving d1 pool recount {len(surviving)} != plan §7 G4's {G2_EXPECTED_SURVIVORS} — "
        "committed anchors drifted; re-derive the gate pool before judging"
    )
    per_arm: dict[str, dict] = {}
    passed = True
    for arm in ("steered", "shuffled"):
        deltas = []
        used = []
        for pid in surviving:
            a, b = tb_f.get((pid, arm)), parent_f.get((pid, arm))
            if a is None or b is None:
                continue
            deltas.append(a - b)
            used.append(pid)
        assert len(deltas) >= SURVIVAL_FLOOR, (arm, len(deltas))
        arr = np.asarray(deltas, dtype=np.float64)
        boots = bootstrap_family_means_batched(arr[:, None], BOOT_B, BOOT_SEED)
        lo, hi = np.nanpercentile(boots[:, 0], [2.5, 97.5])
        mean = float(arr.mean())
        arm_pass = abs(mean) <= G2_BAR
        passed &= arm_pass
        per_arm[arm] = {
            "n_pool": len(surviving),
            "n_used": len(used),
            "mean_delta_f": mean,
            "sd_delta_f": float(arr.std(ddof=1)) if len(arr) > 1 else None,
            "ci95": [float(lo), float(hi)],
            "passed": arm_pass,
        }
        logger.info(
            "[g2] %s: mean ΔF=%.4f (n=%d, CI [%.3f, %.3f]) -> %s",
            arm,
            mean,
            len(used),
            lo,
            hi,
            "PASS" if arm_pass else "FAIL",
        )
    verdict = {
        "criterion": "G2 d1-identity gate (plan §7 G4): |mean ΔF| <= 0.10 per arm, "
        "netted space, tb@d1 vs parent ce, surviving d1 pool",
        "bar": G2_BAR,
        "n_surviving_pool": len(surviving),
        "per_arm": per_arm,
        "passed": passed,
        "boot": {"B": BOOT_B, "seed": BOOT_SEED},
        "repro": J94._repro(),
    }
    A._write_json_atomic(args.out_dir / "identity_gate.json", verdict)
    if not passed:
        logger.error(
            "[g2] HALT rc=%d — wave-2 judge spend + every d3/d5 read are FROZEN; "
            "rollout text is persisted (regenerable reads); diagnose the rig",
            RC_G2_IDENTITY,
        )
        return RC_G2_IDENTITY
    return RC_OK


# ── step: f-tables ────────────────────────────────────────────────────


def _build_channels(
    args: argparse.Namespace,
    pairs_by_id: dict[str, BANK.Pair2162],
    committed: dict[str, dict],
    pair_ids: set[str],
) -> dict[str, dict | None]:
    """Per-pair B-channel floor/ceiling from parent per-draw anchor scores,
    with the netted-channel parity assert against the committed table."""
    anchor_scores = load_parent_anchor_scores(args.parent_scores_dir)
    coh_draws = load_anchor_coherent_draws(args.parent_scores_dir)
    channels: dict[str, dict | None] = {}
    n_checked = 0
    for pid in sorted(pair_ids):
        chan = pair_anchor_channels(pairs_by_id[pid], coh_draws, anchor_scores)
        if chan is not None and pid in committed:
            _assert_anchor_parity(chan, committed[pid], pid)
            n_checked += 1
        channels[pid] = chan
    assert n_checked >= SURVIVAL_FLOOR, (
        f"anchor-channel parity checked only {n_checked} pairs — vacuous join "
        f"(parent scores dir {args.parent_scores_dir} or committed anchors mis-staged)"
    )
    return channels


def step_f_tables(args: argparse.Namespace) -> int:
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    committed = load_committed_anchors(args.parent_metrics_dir)
    grid_rows = J.load_grid_rows(args.rollouts_dir)
    scores = A.load_wave_scores(args.work_root / "scores", "grid")
    grid_pair_ids = {r["pair_id"] for r in grid_rows}
    channels = _build_channels(args, pairs_by_id, committed, grid_pair_ids)
    tables = tb_pair_cells(grid_rows, scores, committed, channels, pairs_by_id)
    A._write_jsonl_atomic(args.out_dir / "f_cells_tb.jsonl", tables["steered"])
    A._write_jsonl_atomic(args.out_dir / "null_shuffled_cells_tb.jsonl", tables["shuffled"])
    A._write_jsonl_atomic(args.out_dir / "null_crosstype_cells_tb.jsonl", tables["crosstype"])
    logger.info(
        "[f-tables] steered=%d shuffled=%d crosstype=%d (anchor parity asserted on %d pairs)",
        len(tables["steered"]),
        len(tables["shuffled"]),
        len(tables["crosstype"]),
        sum(1 for pid in grid_pair_ids if channels.get(pid) is not None and pid in committed),
    )
    return RC_OK


# ── step: parent-ce reference re-aggregation (plan §6 / §12.8) ────────

PARENT_COMMITTED_FILES = {
    "steered": "f_cells.jsonl",
    "shuffled": "null_shuffled_cells.jsonl",
    "crosstype": "null_crosstype_cells.jsonl",
}


def _assert_parent_f_parity(tables: dict[str, list[dict]], parent_metrics_dir: Path) -> int:
    """Recomputed NETTED F must reproduce the committed parent ``f_beh`` per
    (pair x ce x arm) — the assumption-8 'both ways' verification: the same
    per-rubric scores that yield the committed netted read also feed the
    target-only read, so this join check licenses ``f_target_only``."""
    n_checked = 0
    for arm, fname in PARENT_COMMITTED_FILES.items():
        committed_f = {
            (r["pair_id"], r["cell"]): r["f_beh"]
            for r in A._iter_jsonl(parent_metrics_dir / fname)
            if r["slot"] == "ce"
        }
        for rec in tables[arm]:
            want = committed_f.get((rec["pair_id"], rec["cell"]))
            got = rec["f_netted"]
            if want is None or got is None:
                continue
            assert abs(got - want) <= PARENT_F_PARITY_TOL, (
                f"parent-ref parity FAIL pair={rec['pair_id']} cell={rec['cell']} arm={arm}: "
                f"recomputed netted F {got} vs committed f_beh {want} — the staged parent "
                "grid scores/rollouts do not reproduce the committed parent tables"
            )
            n_checked += 1
    assert n_checked >= SURVIVAL_FLOOR, (
        f"parent-ref parity checked only {n_checked} rows — vacuous join "
        "(staging incomplete or wrong slot filter)"
    )
    return n_checked


def step_parent_ref(args: argparse.Namespace) -> int:
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    committed = load_committed_anchors(args.parent_metrics_dir)
    parent_rows = [r for r in J.load_grid_rows(args.parent_rollouts_dir) if r["slot"] == "ce"]
    assert parent_rows, f"no parent slot=ce grid rows under {args.parent_rollouts_dir}"
    parent_scores = A.load_wave_scores(args.parent_scores_dir, "grid")
    pair_ids = {r["pair_id"] for r in parent_rows}
    channels = _build_channels(args, pairs_by_id, committed, pair_ids)
    tables = tb_pair_cells(parent_rows, parent_scores, committed, channels, pairs_by_id)
    n_checked = _assert_parent_f_parity(tables, args.parent_metrics_dir)
    out = [rec for arm in ("steered", "shuffled", "crosstype") for rec in tables[arm]]
    A._write_jsonl_atomic(args.out_dir / "parent_ref_cells_tb.jsonl", out)
    logger.info(
        "[parent-ref] rows=%d (netted parity asserted on %d (pair x cell x arm) rows)",
        len(out),
        n_checked,
    )
    return RC_OK


# ── step: stats + verdict lattice ─────────────────────────────────────


def _paired_cell(
    steered: list[dict],
    idx_null: dict[str, dict[tuple[str, str, str], dict]],
    nulls: tuple[str, ...],
    metric: str = "f_beh",
) -> dict:
    """Surviving paired per-pair values for one (cell x slot) unit."""
    f_steered: list[float] = []
    f_null: dict[str, list[float]] = {n: [] for n in nulls}
    diffs: dict[str, list[float]] = {n: [] for n in nulls}
    pair_ids: list[str] = []
    for r in steered:
        if r["separation"] is None or abs(r["separation"]) < SEPARATION_BAR:
            continue
        if r[metric] is None:
            continue
        per_null = {}
        for n in nulls:
            nr = idx_null[n].get((r["pair_id"], r["slot"], n))
            per_null[n] = nr[metric] if nr else None
        if any(v is None for v in per_null.values()):
            continue
        f_steered.append(r[metric])
        pair_ids.append(r["pair_id"])
        for n in nulls:
            f_null[n].append(per_null[n])
            diffs[n].append(r[metric] - per_null[n])
    return {"steered": f_steered, "null": f_null, "diffs": diffs, "pair_ids": pair_ids}


def _cell_ci(f_steered: list[float], f_null: dict[str, list[float]]) -> dict:
    cols = [f_steered] + [f_null[n] for n in sorted(f_null)]
    labels = ["steered"] + sorted(f_null)
    n = max(len(c) for c in cols)
    vals = np.full((n, len(cols)), np.nan)
    for j, c in enumerate(cols):
        vals[: len(c), j] = c
    boots = bootstrap_family_means_batched(vals, BOOT_B, BOOT_SEED)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return {
        lab: [None if not np.isfinite(v) else float(v) for v in (lo[j], hi[j])]
        for j, lab in enumerate(labels)
    }


def step_stats(args: argparse.Namespace) -> int:
    steered_rows = list(A._iter_jsonl(args.out_dir / "f_cells_tb.jsonl"))
    nulls = {
        "shuffled": list(A._iter_jsonl(args.out_dir / "null_shuffled_cells_tb.jsonl")),
        "crosstype": list(A._iter_jsonl(args.out_dir / "null_crosstype_cells_tb.jsonl")),
    }

    def index(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
        return {(r["pair_id"], r["slot"], r["arm"]): r for r in rows}

    idx_null = {k: index(v) for k, v in nulls.items()}
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in steered_rows:
        by_cell[(r["cell"], r["slot"])].append(r)

    coh_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    cap_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for rows_list in (steered_rows, nulls["shuffled"], nulls["crosstype"]):
        for r in rows_list:
            c = coh_counts[(r["cell"], r["slot"])]
            c[0] += r["n_coherent"]
            c[1] += r["n_draws"]
            k = cap_counts[(r["cell"], r["slot"])]
            k[0] += r["n_cap_hit"]
            k[1] += r["n_draws"]

    per_cell: dict[str, dict] = {}
    family_p: dict[str, dict[str, float]] = {"TB-joint": {}, "TB-sweep": {}}
    paired_cache: dict[str, dict] = {}
    for (cell, slot), rows in sorted(by_cell.items()):
        fam = "TB-joint" if slot == "tb" else "TB-sweep"
        test_nulls = ("shuffled", "crosstype") if fam == "TB-joint" else ("shuffled",)
        paired = _paired_cell(rows, idx_null, test_nulls)
        key = f"{cell}|{slot}"
        paired_cache[key] = paired
        n = len(paired["steered"])
        testable = n >= SURVIVAL_FLOOR
        p_iut = None
        if testable:
            p_iut = max(A._wilcoxon_exact_p(np.asarray(paired["diffs"][nl])) for nl in test_nulls)
            family_p[fam][key] = p_iut
        rec = {
            "cell": cell,
            "slot": slot,
            "family": fam,
            "registered_space": rows[0]["registered_space"] if rows else None,
            "n_pre_exclusion": len(rows),
            "n_post_exclusion": n,
            "untestable_causal": not testable,
            "f_steered_mean": float(np.mean(paired["steered"])) if paired["steered"] else None,
            **{
                f"f_{nl}_mean": (float(np.mean(paired["null"][nl])) if paired["null"][nl] else None)
                for nl in test_nulls
            },
            "p_iut": p_iut,
            "coherent_fraction": (
                coh_counts[(cell, slot)][0] / coh_counts[(cell, slot)][1]
                if coh_counts[(cell, slot)][1]
                else None
            ),
            "low_coherence": bool(
                coh_counts[(cell, slot)][1]
                and coh_counts[(cell, slot)][0] / coh_counts[(cell, slot)][1] < 0.5
            ),
            "cap_hit_fraction": (
                cap_counts[(cell, slot)][0] / cap_counts[(cell, slot)][1]
                if cap_counts[(cell, slot)][1]
                else None
            ),
        }
        if n:
            rec["ci95"] = _cell_ci(paired["steered"], paired["null"])
            s_lo = rec["ci95"]["steered"][0]
            rec["disjoint_vs_tested_nulls"] = bool(
                s_lo is not None
                and all(
                    rec["ci95"][nl][1] is not None and s_lo > rec["ci95"][nl][1]
                    for nl in test_nulls
                )
            )
        per_cell[key] = rec

    for fam, pvals in family_p.items():
        adj = A.holm(pvals)
        for key, p_adj in adj.items():
            per_cell[key]["p_holm"] = p_adj
            per_cell[key]["holm_family_m"] = len(pvals)
            per_cell[key]["holm_pass"] = p_adj < HOLM_ALPHA
    assert len(family_p["TB-joint"]) <= 7 and len(family_p["TB-sweep"]) <= 12, (
        {k: len(v) for k, v in family_p.items()},
        "registered family sizes are m=7 / m=12 (plan §3)",
    )

    # ── §3 verdict lattice per base ───────────────────────────────────
    def _cell_paired(cell: str) -> dict:
        return paired_cache.get(f"{cell}|tb", {"steered": [], "null": {}, "diffs": {}})

    def _j_stat(paired: dict) -> dict | None:
        """min-over-nulls mean paired difference + pair-clustered bootstrap CI."""
        d_sh = paired["diffs"].get("shuffled", [])
        d_ct = paired["diffs"].get("crosstype", [])
        if not d_sh or not d_ct:
            return None
        vals = np.stack(
            [np.asarray(d_sh, dtype=np.float64), np.asarray(d_ct, dtype=np.float64)], axis=1
        )
        boots = bootstrap_family_means_batched(vals, BOOT_B, BOOT_SEED)
        j_draws = np.nanmin(boots, axis=1)
        lo, hi = np.nanpercentile(j_draws, [2.5, 97.5])
        point = float(min(np.mean(d_sh), np.mean(d_ct)))
        return {"point": point, "ci95": [float(lo), float(hi)], "n_pairs": len(d_sh)}

    def _raw_mean(cell: str) -> float | None:
        vals = [
            r["raw_move_registered"]
            for r in by_cell.get((cell, "tb"), [])
            if r["raw_move_registered"] is not None
            and r["separation"] is not None
            and abs(r["separation"]) >= SEPARATION_BAR
        ]
        return float(np.mean(vals)) if vals else None

    control_rec = per_cell.get(f"{TB.CONTROL_CELL}|tb", {})
    edit_artifact = bool(
        control_rec.get("holm_pass") and control_rec.get("disjoint_vs_tested_nulls")
    )
    lattice: dict[str, dict] = {}
    for base in BASES:
        d1, d5 = base, f"recency_{base}_d5"
        p1, p5 = _cell_paired(d1), _cell_paired(d5)
        j1, j5 = _j_stat(p1), _j_stat(p5)
        f1 = np.asarray(p1["steered"], dtype=np.float64)
        f5 = np.asarray(p5["steered"], dtype=np.float64)
        d_point = ci_d = None
        if len(f1) and len(f5):
            d_point = float(f5.mean() - 0.5 * f1.mean())
            b1 = bootstrap_family_means_batched(f1[:, None], BOOT_B, BOOT_SEED)[:, 0]
            b5 = bootstrap_family_means_batched(f5[:, None], BOOT_B, BOOT_SEED + 1)[:, 0]
            lo, hi = np.nanpercentile(b5 - 0.5 * b1, [2.5, 97.5])
            ci_d = [float(lo), float(hi)]
        label = "No-verdict"
        delta_disp = None
        if j1 is not None and j5 is not None and d_point is not None:
            j1_pos = j1["ci95"][0] > 0
            j5_pos = j5["ci95"][0] > 0
            j5_straddle = j5["ci95"][0] <= 0 <= j5["ci95"][1]
            delta_disp = 0.5 * j1["point"]
            if j1_pos and j5_pos and d_point >= 0:
                label = "Dispersion"
            elif j1_pos and j5_pos and d_point < 0:
                label = "Partial-trace"
            elif j1_pos and j5_straddle and j5["ci95"][1] < delta_disp:
                label = "Decay"
        raw1, raw5 = _raw_mean(d1), _raw_mean(d5)
        d_raw = raw5 - 0.5 * raw1 if raw1 is not None and raw5 is not None else None
        denom_caveat = bool(
            label in ("Dispersion", "Partial-trace")
            and d_point is not None
            and d_raw is not None
            and np.sign(d_raw) != np.sign(d_point)
        )
        final = "edit-artifact — no verdict" if edit_artifact else label
        lattice[base] = {
            "registered_space": registered_space(base),
            "J1": j1,
            "J5": j5,
            "D": {"point": d_point, "ci95": ci_d},
            "delta_disp": delta_disp,
            "raw_scale": {"d1_mean": raw1, "d5_mean": raw5, "D_raw": d_raw},
            "label_lattice": label,
            "denominator_limited": denom_caveat,
            "edit_artifact_overlay": edit_artifact,
            "label_final": final + (" (denominator-limited)" if denom_caveat else ""),
            "scope_caveat": "a joint-null verdict rules out dispersion across the PATCHED "
            "BOUNDARY SET only (plan §3 report-scope caveat)",
            "no_verdict_note": (
                "a J5 straddle whose upper bound >= delta_disp routes to No-verdict and is "
                "narrated as failure-to-reject (underpowered), never as decay"
                if label == "No-verdict"
                else None
            ),
        }

    # Rule-29 per-item completeness over this round's waves.
    completeness = {}
    for meta_path in sorted((args.work_root / "scores").glob("*.grid.meta.json")):
        meta = json.loads(meta_path.read_text())
        n_items = None
        pass1 = meta.get("pass1") or {}
        n_items = pass1.get("n_items")
        completeness[meta_path.stem] = {
            "n_scored_items": meta.get("n_scored_items"),
            "n_items": n_items,
            "frac_items_complete": (
                meta["n_scored_items"] / n_items
                if n_items and meta.get("n_scored_items") is not None
                else None
            ),
        }

    A._write_json_atomic(
        args.out_dir / "stats_tb.json",
        {
            "per_cell": per_cell,
            "verdict_lattice": lattice,
            "families": {fam: len(p) for fam, p in family_p.items()},
            "judge_completeness": completeness,
            "bars": {
                "separation_bar": SEPARATION_BAR,
                "survival_floor": SURVIVAL_FLOOR,
                "boot": {"B": BOOT_B, "seed": BOOT_SEED},
                "holm_alpha": HOLM_ALPHA,
                "caphit_regen_trigger": "cap-hit > 2% per cell => re-generate at >= 2x cap "
                "(inherited; parent realized 0.0997%)",
            },
            "repro": J94._repro(),
        },
    )
    n_testable = sum(1 for r in per_cell.values() if not r["untestable_causal"])
    logger.info(
        "[stats] cells=%d testable=%d lattice=%s",
        len(per_cell),
        n_testable,
        {b: lattice[b]["label_final"] for b in lattice},
    )
    return RC_OK


# ── step: margin ──────────────────────────────────────────────────────


def step_margin(args: argparse.Namespace) -> int:
    pairs_by_id = {p.pair_id: p for p in BANK.build_pairs()}
    rows = []
    for shard in sorted(args.margin_dir.glob("*.jsonl")):
        rows.extend(r for r in A._iter_jsonl(shard) if not r.get("skipped"))
    assert rows, f"no margin rows under {args.margin_dir}"
    grid_lnp: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        grid_lnp[(r["pair_id"], r["slot"], r["arm"], r["pool_side"])].append(r["lnp_mean"])
    # Parent floor-anchor margins (committed margin_cells.jsonl, slot ce) — optional join.
    floor_by_pair: dict[str, float] = {}
    parent_margin = args.parent_metrics_dir / "margin_cells.jsonl"
    if parent_margin.exists():
        for r in A._iter_jsonl(parent_margin):
            if r.get("margin_floor_anchor") is not None:
                floor_by_pair.setdefault(r["pair_id"], r["margin_floor_anchor"])
    else:
        logger.warning("[margin] %s absent — margin_floor_anchor left null", parent_margin)

    def _margin(key_b, key_a) -> float | None:
        b, a = grid_lnp.get(key_b), grid_lnp.get(key_a)
        if not b or not a:
            return None
        return sum(b) / len(b) - sum(a) / len(a)

    out = []
    for pair_id, slot, arm in sorted({(k[0], k[1], k[2]) for k in grid_lnp}):
        m_patched = _margin((pair_id, slot, arm, "B"), (pair_id, slot, arm, "A"))
        p = pairs_by_id[pair_id]
        m_floor = floor_by_pair.get(pair_id)
        out.append(
            {
                "pair_id": pair_id,
                "cell": p.cell,
                "slot": slot,
                "arm": arm,
                "margin_patched": m_patched,
                "margin_floor_anchor": m_floor,
                "margin_shift": (
                    m_patched - m_floor if m_patched is not None and m_floor is not None else None
                ),
            }
        )
    A._write_jsonl_atomic(args.out_dir / "margin_cells_tb.jsonl", out)
    logger.info("[margin] cells=%d (floor join: %d pairs)", len(out), len(floor_by_pair))
    return RC_OK


# ── step: raw-scale companion (plan §9 P5 rawscale_tb.json) ───────────


def step_rawscale(args: argparse.Namespace) -> int:
    """Plan §4.4 raw-scale (denominator-free) read at the tb slot — the honesty
    overlay for the depth-wise anchor-gap shrink (1.56 -> 0.57). Reuses the
    parent's committed script verbatim (same B=10000 / seed 21620 pair-clustered
    bootstrap), pointed at the tb tables + the parent's committed anchors, with
    the additive ``--null-cis`` arm CIs the manifest's tb_rawscale figure plots."""

    out_json = args.out_dir / "rawscale_tb.json"
    RS.main(
        [
            "--metrics-dir",
            str(args.out_dir),
            "--slot",
            "tb",
            "--file-suffix",
            "_tb",
            "--anchors-file",
            str(args.parent_metrics_dir / "anchors.jsonl"),
            "--null-cis",
            "--out-json",
            str(out_json),
        ]
    )
    assert out_json.is_file(), f"rawscale step wrote nothing at {out_json}"
    logger.info("[rawscale] wrote %s", out_json)
    return RC_OK


# ── staging + CLI ─────────────────────────────────────────────────────


def _stage_inputs(args: argparse.Namespace) -> None:
    from explore_persona_space.orchestrate import hub

    prefixes = [f"{J.HF_PREFIX}/raw_completions/tbmp/grid", HF_JUDGE_RAW_SCORES]
    if args.step in ("parent-ref", "analysis"):
        prefixes.append(HF_PARENT_GRID)
    if args.step in ("margin", "analysis"):
        prefixes.append(f"{J.HF_PREFIX}/analysis_tensors/tbmp/margin")
    total_bytes = 0
    for prefix in prefixes:
        repo = _repo_for_prefix(prefix)
        staged = hub.stage_hub_prefix(repo, prefix, args.in_root, revision=None)
        # M3 (r1): record the REALIZED VM staging footprint per prefix — §9
        # stated ~35 MB; the realized parent grid + judge raw scores are ~210 MB.
        nbytes = sum(Path(p).stat().st_size for p in staged if Path(p).is_file())
        total_bytes += nbytes
        logger.info(
            "[stage] %s from %s: %d files, %.1f MB", prefix, repo, len(staged), nbytes / 1e6
        )
    logger.info("[stage] realized staging footprint: %.1f MB total", total_bytes / 1e6)


STEPS = {
    "pilot": step_pilot,
    "wave1": step_wave1,
    "g2": step_g2,
    "wave2": step_wave2,
    "f-tables": step_f_tables,
    "parent-ref": step_parent_ref,
    "rawscale": step_rawscale,
    "stats": step_stats,
    "margin": step_margin,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2162 tbmp VM-side judge + analysis.")
    ap.add_argument("--step", required=True, choices=(*STEPS, "analysis"))
    ap.add_argument("--in-root", type=Path, default=Path("data/issue_2162/tbmp_inputs"))
    ap.add_argument("--rollouts-dir", type=Path, default=None)
    ap.add_argument("--margin-dir", type=Path, default=None)
    ap.add_argument("--parent-scores-dir", type=Path, default=None)
    ap.add_argument("--parent-rollouts-dir", type=Path, default=None)
    ap.add_argument(
        "--parent-metrics-dir", type=Path, default=Path("eval_results/issue_2162/f_metrics")
    )
    ap.add_argument("--stage-from-hf", action="store_true")
    ap.add_argument(
        "--work-root", type=Path, default=Path("eval_results/issue_2162/turn_boundary/judge")
    )
    ap.add_argument("--cache-root", type=Path, default=Path("data/issue_2162/tbmp_judge_cache"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_2162/turn_boundary"))
    ap.add_argument("--judge-model", type=str, default=J94.DEFAULT_JUDGE_MODEL)
    ap.add_argument("--max-tokens", type=int, default=J94.DEFAULT_JUDGE_MAX_TOKENS)
    ap.add_argument(
        "--dry-run", action="store_true", help="wave steps: zero-API construction check"
    )
    args = ap.parse_args(argv)
    mirror = args.in_root / J.HF_PREFIX
    if args.rollouts_dir is None:
        args.rollouts_dir = mirror / "raw_completions/tbmp/grid"
    if args.margin_dir is None:
        args.margin_dir = mirror / "analysis_tensors/tbmp/margin"
    if args.parent_scores_dir is None:
        args.parent_scores_dir = mirror / "raw_completions/judge_raw/scores"
    if args.parent_rollouts_dir is None:
        args.parent_rollouts_dir = mirror / "raw_completions/grid"
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    if args.stage_from_hf:
        _stage_inputs(args)
    if args.step == "analysis":
        for step in ("f-tables", "parent-ref", "rawscale", "stats", "margin"):
            rc = STEPS[step](args)
            if rc != RC_OK:
                return rc
        return RC_OK
    return STEPS[args.step](args)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
