#!/usr/bin/env python
"""#1481 Phase B/D analysis driver (plan §4.5/§4.7/§6) — VM-side, no pod code.

Three phases, each checkpointed to its own JSON under ``--out-dir``:

- ``select``  (Phase B): verdict-arm manifest over the content grid (fresh
  round ladders + the reused #1434 casual seed-42 cells + the 12 reused
  fu4/fu5/fu7 committed con arms) via the §4.5 primitives in
  ``issue1481_cells`` — plus the marker verdict arms / dose-match when
  ``--marker-root`` is given. Emits the ``panel_dispatch`` hints Phase C
  consumes (``--arms`` run ids + the reused ``--ckpt-map`` arm ids).
- ``judge``   : judges the Phase-C panel generations (+ the shared base
  panel) with the per-behavior registered instrument (casual = the verbatim
  pv trait rubric via ``c1434.pv_judge_fn``; impolite / sycophancy =
  ``fu3w.judge_graded_r23``), one aggregate per behavior. Per-(arm, ctx)
  records checkpoint the moment they land (resume-keyed on the judging
  regime). ``--stub-judge`` (smoke-only) swaps a deterministic offline judge.
- ``contrast`` (Phase D): the §6 regime contrast — content pooled-non-source
  D (Newcombe 95%, seed-pooled n=1000/arm), per-seed Ds + sign-discordance,
  per-context Wilson decomposition, the MF-3 mechanized held-out
  disjointness assert (registered §5 sets vs the REALIZED training panel
  from ``mix_meta.json``), dose-match labels, the nearest-dose sensitivity
  list (≤16), the behavior-level pooled-over-dose-matched headline, and the
  marker contrast (question-cluster bootstrap, 2000 draws, seed 653,
  vectorized) + emission-map / three-space data for the figures driver.

Everything statistical composes over ``issue1481_cells`` (`verdict_arm`,
``dose_match_label``, ``newcombe``, ``wilson``, ``lattice_verdict``,
``heldout_contexts``); nothing is duplicated. Figures live in
``scripts/issue1481_figures.py`` and consume ONLY this driver's JSONs.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1090_fu3_worker as fu3w  # noqa: E402
import issue1090_fu4 as fu4  # noqa: E402
import issue1090_run as run1090  # noqa: E402
import issue1434_cells as c1434  # noqa: E402
import issue1481_cells as cells  # noqa: E402

from explore_persona_space.artifacts import negatives as neg_mod  # noqa: E402
from explore_persona_space.artifacts.organisms import _sha256_text  # noqa: E402

logger = logging.getLogger("issue1481.analysis")

# ── Registered constants (plan §5/§6) ────────────────────────────────────────

BOOTSTRAP_DRAWS = 2000  # marker question-cluster bootstrap (Source: #1333)
BOOTSTRAP_SEED = 653
SENSITIVITY_ARM_CAP = 16  # nearest-dose secondary panel arms (plan §4.5)
# Plan §5 registered per-cell held-out read sets ("icl" resolves per behavior).
REGISTERED_HELDOUT: dict[str, tuple[str, ...]] = {
    "pers": ("wildchat_prefix_real545", "icl"),
    "bare": ("persona_software_engineer", "wildchat_prefix_real545", "icl"),
    "conv": ("persona_software_engineer", "icl"),
    "icl": ("persona_software_engineer", "wildchat_prefix_real545"),
}
JUDGE_N_DRAWS_DEFAULT = 3  # plan §6: 100 completions/context, 3 draws
_JUDGE_ID_BUDGET = 53  # Batch custom_id budget (#1415; mirrors w1434)


def _read_json(path: Path) -> dict:
    return run1090._read_json(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run1090._atomic_write_json(path, payload)


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── Phase B: select ──────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SelectPaths:
    """Input roots for the select phase (fixture-overridable for smokes)."""

    ladders_dir: Path  # fresh round ladders (i1481*_ladders.json)
    repo_root: Path  # resolves the reused #1434 / fu4 / fu5 / fu7 ladder paths
    marker_root: Path | None  # marker out_root (per-run selection.json + ladder.json)


def _ladders_runs(path: Path) -> dict[str, dict]:
    """The per-run records dict of one committed/fresh ladders JSON (fail-loud).

    Fresh i1481 rounds + the reused fu4/fu5/fu7 parents use a top-level
    ``"runs"`` dict; the committed #1434 ladders (``i1434_ladders.json`` /
    ``i1434po_ladders.json``) carry the SAME record shape (``selection`` +
    ``rates_by_step``) under ``"ladders"`` instead — accept both (verified
    against the committed artifacts; the #1073 reused_artifact_schema_drift
    class)."""
    payload = _read_json(path)
    runs = payload.get("runs") or payload.get("ladders")
    if not isinstance(runs, dict) or not runs:
        raise RuntimeError(f"[i1481-select] no 'runs'/'ladders' dict in ladders JSON {path}")
    return runs


def _arm_record(paths: SelectPaths, beh_key: str, ctx_key: str, regime: str, lr: float, seed: int):
    """(arm_id, selection, rates_by_step, source) for one grid arm.

    Fresh arms read the i1481 round ladders; reused casual seed-42 arms read
    the committed #1434 ladders; the 12 reused fu4/fu5/fu7 con arms read
    their committed parent ladders (the committed selected rung IS the arm's
    selection read — plan §4.6 gate P1)."""
    tag = fu4.LR_TAG[lr]
    arm_id = f"{beh_key}-{ctx_key}-{regime}-{tag}-s{seed}"
    if not cells.is_reused(beh_key, ctx_key, regime, lr, seed):
        path = paths.ladders_dir / f"{cells.round_name(beh_key, regime)}_ladders.json"
        rec = _ladders_runs(path).get(arm_id)
        if rec is None or "selection" not in rec:
            raise RuntimeError(f"[i1481-select] fresh arm {arm_id} missing from {path}")
        return arm_id, rec["selection"], rec.get("rates_by_step") or {}, "fresh"
    if beh_key == "cas":
        path = paths.repo_root / cells.REUSED_1434_LADDERS[regime]
        src_id = cells.reused_1434_run_id(ctx_key, regime, lr)
        rec = _ladders_runs(path).get(src_id)
        if rec is None or "selection" not in rec:
            raise RuntimeError(f"[i1481-select] reused #1434 arm {src_id} missing from {path}")
        return arm_id, rec["selection"], rec.get("rates_by_step") or {}, "reused-1434"
    arm = cells.REUSED_CON_ARM_BY_ID[arm_id]
    rec = _ladders_runs(paths.repo_root / arm.ladders_path).get(arm.source_run_id)
    if rec is None or "selection" not in rec:
        raise RuntimeError(
            f"[i1481-select] reused con arm {arm.source_run_id} missing from {arm.ladders_path}"
        )
    return arm_id, rec["selection"], rec.get("rates_by_step") or {}, "reused-parent"


def _nearest_dose_pick(po_arms: dict[str, dict], con_rate: float, verdict: tuple[str, int]) -> dict:
    """Nearest-dose SECONDARY pairing (plan §4.5): the po (arm, rung) whose
    Tier-1 rate is nearest the con verdict arm's rate; sensitivity read only."""
    best: tuple[float, str, int, float] | None = None
    for arm_id, rec in sorted(po_arms.items()):
        for step_s, rate in (rec.get("rates_by_step") or {}).items():
            key = (abs(float(rate) - con_rate), arm_id, int(step_s), float(rate))
            if best is None or key < best:
                best = key
    if best is None:
        raise RuntimeError("[i1481-select] nearest-dose pick: no po rates_by_step available")
    gap, arm_id, step, rate = best
    return {
        "arm_id": arm_id,
        "step": step,
        "rate": rate,
        "abs_gap_to_con_verdict": gap,
        "differs_from_verdict": (arm_id, step) != verdict,
    }


def _marker_verdict_arm(arms: list[tuple[float, str, dict]]) -> tuple[str, dict]:
    """Marker §4.5 verdict arm: lowest-LR arm whose selection is in the
    [5, 12]-nat window (de-saturation-gated upstream in ``select_rung_1481``),
    else closest approach to the window (tie-break lowest lr) — the marker
    mirror of ``cells.verdict_arm``."""
    if not arms:
        raise ValueError("marker verdict_arm: no arms")
    for lr, run_id, sel in sorted(arms, key=lambda t: t[0]):
        if bool(sel.get("in_window")):
            return run_id, {
                "rule": "lowest_lr_in_window",
                "run_id": run_id,
                "lr": lr,
                "selection": sel,
            }

    def _dist(t: tuple[float, str, dict]) -> tuple[float, float]:
        lo, hi = t[2].get("window") or [5.0, 12.0]
        dg = float(t[2]["delta_logp_mean"])
        return (max(lo - dg, dg - hi, 0.0), t[0])

    lr, run_id, sel = min(arms, key=_dist)
    return run_id, {"rule": "closest_approach", "run_id": run_id, "lr": lr, "selection": sel}


def _select_marker(marker_root: Path) -> dict:
    """Marker verdict arms + §4.5 dose-match per (ctx, seed) from per-run
    ``selection.json`` / ``ladder.json`` under the marker out_root."""
    import issue1481_marker as mk  # heavy sibling; imported only on the marker path

    out: dict[str, Any] = {
        "window": list(mk.INSTALL_WINDOW),
        "tol_nats": mk.DOSE_MATCH_TOL_NATS,
        "contexts": {},
        "arms": {},
    }
    for ctx_key in mk.CTX_KEYS:
        ctx_entry: dict[str, Any] = {}
        for seed in mk.SEEDS:
            per_regime: dict[str, Any] = {}
            for regime in ("con", "po"):
                arms: list[tuple[float, str, dict]] = []
                for lr_key, (lr, _steps, _cad) in mk.LR_ARMS.items():
                    run_id = mk.run_id_for(ctx_key, regime, lr_key, seed)
                    sel_path = marker_root / run_id / "selection.json"
                    if not sel_path.exists():
                        raise RuntimeError(f"[i1481-select] marker selection missing: {sel_path}")
                    sel = _read_json(sel_path)
                    arms.append((float(lr), run_id, sel))
                    ladder_path = marker_root / run_id / "ladder.json"
                    ladder = _read_json(ladder_path) if ladder_path.exists() else {}
                    out["arms"][run_id] = {
                        "ctx_key": ctx_key,
                        "regime": regime,
                        "lr_key": lr_key,
                        "seed": seed,
                        "selection": {
                            k: sel.get(k)
                            for k in (
                                "step",
                                "in_window",
                                "fallback",
                                "delta_logp_mean",
                                "window",
                                "emission_onset_rung",
                                "selectivity_break_rung",
                            )
                        },
                        "reads_by_step": ladder.get("reads_by_step") or {},
                    }
                run_id, info = _marker_verdict_arm(arms)
                per_regime[regime] = info
            dg_con = float(per_regime["con"]["selection"]["delta_logp_mean"])
            dg_po = float(per_regime["po"]["selection"]["delta_logp_mean"])
            both_in = bool(per_regime["con"]["selection"].get("in_window")) and bool(
                per_regime["po"]["selection"].get("in_window")
            )
            ctx_entry[str(seed)] = {
                **per_regime,
                "abs_gap_nats": abs(dg_con - dg_po),
                "dose_matched": bool(both_in and abs(dg_con - dg_po) <= mk.DOSE_MATCH_TOL_NATS),
                "dose_unmatched_flag": not both_in,
            }
        out["contexts"][ctx_key] = ctx_entry
    return out


def phase_select(args: argparse.Namespace) -> int:
    """Phase B: build + persist ``verdict_manifest.json``."""
    paths = SelectPaths(
        ladders_dir=Path(args.ladders_dir),
        repo_root=Path(args.repo_root),
        marker_root=Path(args.marker_root) if args.marker_root else None,
    )
    beh_keys = [b.strip() for b in args.behaviors.split(",") if b.strip()]
    ctx_keys = [c.strip() for c in args.contexts.split(",") if c.strip()]
    bad = [b for b in beh_keys if b not in cells.BEHAVIOR_BY_KEY]
    bad += [c for c in ctx_keys if c not in cells.CTX_KEYS]
    if bad:
        raise SystemExit(f"[i1481-select] unknown behaviors/contexts {bad}")
    manifest: dict[str, Any] = {
        "issue": cells.ISSUE_1481,
        "band": list(cells.JUDGED_RATE_BAND),
        "dose_match_max_rate_gap": cells.DOSE_MATCH_MAX_RATE_GAP,
        "content": {},
        "panel_dispatch": {"fresh_arms": [], "reused_ckpt_arms": []},
        "ts": _ts(),
    }
    for beh_key in beh_keys:
        beh_entry: dict[str, Any] = {}
        for ctx_key in ctx_keys:
            cell: dict[str, Any] = {"arms": {}, "seeds": {}}
            for seed in cells.SEEDS:
                per_regime: dict[str, Any] = {}
                po_arms_for_nearest: dict[str, dict] = {}
                for regime in cells.REGIMES:
                    arms: list[tuple[float, str, dict]] = []
                    for lr in fu4.FU4_LRS:
                        arm_id, sel, rates, source = _arm_record(
                            paths, beh_key, ctx_key, regime, lr, seed
                        )
                        arms.append((lr, arm_id, sel))
                        cell["arms"][arm_id] = {
                            "lr": lr,
                            "regime": regime,
                            "seed": seed,
                            "source": source,
                            "selection": sel,
                            "rates_by_step": rates,
                        }
                        if regime == "po":
                            po_arms_for_nearest[arm_id] = cell["arms"][arm_id]
                    arm_id, info = cells.verdict_arm(arms)
                    per_regime[regime] = info
                con_sel = per_regime["con"]["selection"]
                po_sel = per_regime["po"]["selection"]
                nearest = _nearest_dose_pick(
                    po_arms_for_nearest,
                    float(con_sel["rate"]),
                    (per_regime["po"]["arm_id"], int(po_sel["step"])),
                )
                cell["seeds"][str(seed)] = {
                    **per_regime,
                    "dose": cells.dose_match_label(con_sel, po_sel),
                    "nearest_dose_po": nearest,
                }
                for regime in cells.REGIMES:
                    vid = per_regime[regime]["arm_id"]
                    if cell["arms"][vid]["source"] == "fresh":
                        manifest["panel_dispatch"]["fresh_arms"].append(vid)
                    elif vid in cells.REUSED_CON_ARM_BY_ID:
                        manifest["panel_dispatch"]["reused_ckpt_arms"].append(vid)
            beh_entry[ctx_key] = cell
        manifest["content"][beh_key] = beh_entry
    manifest["panel_dispatch"]["fresh_arms"] = sorted(set(manifest["panel_dispatch"]["fresh_arms"]))
    manifest["panel_dispatch"]["reused_ckpt_arms"] = sorted(
        set(manifest["panel_dispatch"]["reused_ckpt_arms"])
    )
    if paths.marker_root is not None:
        manifest["marker"] = _select_marker(paths.marker_root)
    out = Path(args.out_dir) / "verdict_manifest.json"
    _write_json(out, manifest)
    logger.info(
        "[i1481-select] wrote %s (%d fresh panel arms, %d reused)",
        out,
        len(manifest["panel_dispatch"]["fresh_arms"]),
        len(manifest["panel_dispatch"]["reused_ckpt_arms"]),
    )
    return 0


# ── Judge phase ──────────────────────────────────────────────────────────────


def _stub_judge_fn(items, eval_prompt, *, n_draws, cache_dir, save_raw, judge_model, dry_run=False):
    """Deterministic OFFLINE smoke judge (signature mirrors the organisms
    JudgeFn seam — ``fu3w.judge_graded_r23`` / ``c1434.pv_judge_fn``).

    Scores = sha1(item id) mapped to [0, 100); never used outside --smoke.
    Returns an object carrying the ``JudgeResult`` fields the tally reads."""
    del eval_prompt, cache_dir, save_raw, judge_model, dry_run

    class _R:
        scores: dict[str, float | None]
        n_total_draws = len(items) * n_draws
        n_dropped_draws = 0
        n_transport_lost_draws = 0

    r = _R()
    r.scores = {
        iid: float(int(hashlib.sha1(iid.encode()).hexdigest()[:4], 16) % 100)
        for iid, _q, _c in items
    }
    return r


def _judge_cell(
    tag: str,
    qs: list[str],
    comps: list[list[str]],
    *,
    behavior_obj,
    judge_fn,
    n_draws: int,
    judge_root: Path,
    instrument: str,
) -> dict:
    """Judge one (arm, context) cell → the #1434 aggregate record shape.

    Mirrors ``issue1434_worker._judge_rate_graded`` (id shape, >53-char
    hash-compaction, drop-never-coerce tally, rule-24 split) with the
    RoundSpec ``judge_fn`` seam so casual routes through the pv rubric and
    impolite/sycophancy through the registered ``judge_graded_r23``."""
    items = [
        (f"{tag}-q{qi:03d}-c{ci:03d}", q, comp)
        for qi, q in enumerate(qs)
        for ci, comp in enumerate(comps[qi])
    ]
    inst_root = judge_root / instrument
    id_map = {
        iid: "h" + hashlib.sha1(iid.encode()).hexdigest()[:12]
        for iid, _, _ in items
        if len(iid) > _JUDGE_ID_BUDGET
    }
    if id_map:
        items = [(id_map.get(iid, iid), q, comp) for iid, q, comp in items]
        inst_root.mkdir(parents=True, exist_ok=True)
        _write_json(inst_root / f"idmap_{tag}.json", {v: k for k, v in id_map.items()})
    inst_root.mkdir(parents=True, exist_ok=True)
    result = judge_fn(
        items,
        behavior_obj.judge_rubric,
        n_draws=n_draws,
        cache_dir=inst_root / "cache",
        save_raw=inst_root / f"judge_raw_{instrument}_{tag}.json",
        judge_model=behavior_obj.judge_model,
    )
    scores = [result.scores.get(iid) for iid, _, _ in items]
    scored = [s for s in scores if s is not None]
    n_pos = sum(1 for s in scored if s > behavior_obj.threshold)
    drop_frac = 1.0 - (len(scored) / len(items)) if items else 0.0
    rec = {
        "tag": tag,
        "instrument": instrument,
        "n_items": len(items),
        "n_scored": len(scored),
        "k_positive": n_pos,
        "rate": (n_pos / len(scored)) if scored else None,
        "graded_mean": (sum(scored) / len(scored)) if scored else None,
        "wilson_95": list(cells.wilson(n_pos, len(scored))) if scored else None,
        "item_drop_frac": drop_frac,
        "n_dropped_draws_content": getattr(result, "n_dropped_draws", None),
        "n_transport_lost_draws": getattr(result, "n_transport_lost_draws", None),
    }
    if rec["rate"] is None:
        logger.warning(
            "[i1481-judge] %s: EVERY item judge-dropped — rate None propagates "
            "(drop-never-coerce; llm-judging rule 9)",
            tag,
        )
    return rec


def _completions_payload(path: Path, questions_sha: str) -> list[list[str]]:
    """Load one ``completions__{side}__{ctx}.json`` file, asserting its
    manifest was generated over the SAME question list we judge against."""
    payload = json.loads(path.read_text())
    manifest = payload.get("manifest") or {}
    got = manifest.get("questions_sha256")
    if got != questions_sha:
        raise RuntimeError(
            f"[i1481-judge] questions mismatch at {path}: file manifest sha {got} != "
            f"judge questions sha {questions_sha} — refusing a misaligned judge pass"
        )
    return payload["completions"]


def phase_judge(args: argparse.Namespace) -> int:
    """Judge the panel + shared base-panel completions for ONE behavior."""
    behavior = args.behavior
    if behavior not in cells.BEHAVIOR_BY_KEY.values():
        raise SystemExit(f"[i1481-judge] --behavior must be one of {list(cells.BEHAVIOR_BY_KEY)}")
    beh_key = {v: k for k, v in cells.BEHAVIOR_BY_KEY.items()}[behavior]
    if args.stub_judge and not args.smoke:
        raise SystemExit("[i1481-judge] --stub-judge is smoke-only: pass --smoke with it")
    behavior_obj = run1090.BEHAVIORS[behavior]
    if args.questions:
        qs = json.loads(Path(args.questions).read_text())
    else:
        qs = list(behavior_obj.eval_question_bank)
        if args.question_limit is not None:
            qs = qs[: args.question_limit]
    questions_sha = _sha256_text(json.dumps(list(qs), ensure_ascii=False))
    if args.stub_judge:
        judge_fn, instrument = _stub_judge_fn, "stub-smoke"
    elif beh_key == "cas":
        judge_fn, instrument = c1434.pv_judge_fn, "pv_trait_score"
    else:
        judge_fn, instrument = fu3w.judge_graded_r23, "registered_graded_r23"
    out_dir = Path(args.out_dir)
    judge_root = out_dir / "judge" / beh_key
    regime_key = {
        "behavior": behavior,
        "instrument": instrument,
        "n_draws": args.n_draws,
        "questions_sha256": questions_sha,
        "stub": bool(args.stub_judge),
    }
    aggregate: dict[str, Any] = {
        "issue": cells.ISSUE_1481,
        "behavior": behavior,
        "beh_key": beh_key,
        "instrument": instrument,
        "smoke_stub_judge": bool(args.stub_judge),
        "n_draws": args.n_draws,
        "questions_sha256": questions_sha,
        "base_panel": {},
        "arms": {},
        "ts": _ts(),
    }

    def _judged(tag: str, comp_path: Path, cache_path: Path) -> dict:
        # Checkpoint-per-cell: resume skips a cell already judged under the
        # SAME regime key (instrument/n_draws/questions/stub — #722 r3 rule).
        if cache_path.exists():
            prior = _read_json(cache_path)
            if prior.get("regime_key") == regime_key:
                return prior["record"]
            raise RuntimeError(
                f"[i1481-judge] {cache_path} was judged under a DIFFERENT regime "
                f"({prior.get('regime_key')} != {regime_key}) — use a fresh --out-dir"
            )
        comps = _completions_payload(comp_path, questions_sha)
        rec = _judge_cell(
            tag,
            qs,
            comps,
            behavior_obj=behavior_obj,
            judge_fn=judge_fn,
            n_draws=args.n_draws,
            judge_root=judge_root,
            instrument=instrument,
        )
        _write_json(cache_path, {"regime_key": regime_key, "record": rec})
        return rec

    base_root = Path(args.base_panel_root)
    for bctx in fu3w.bystander_panel(behavior):
        ctx_id = bctx.context_id
        comp_path = base_root / f"completions__base__{ctx_id}.json"
        if not comp_path.exists():
            raise RuntimeError(f"[i1481-judge] base panel completions missing: {comp_path}")
        aggregate["base_panel"][ctx_id] = _judged(
            f"base-{beh_key}-{ctx_id}", comp_path, judge_root / f"cell_base_{ctx_id}.json"
        )
    panel_root = Path(args.panel_root)
    arm_dirs = sorted(
        d for d in panel_root.iterdir() if d.is_dir() and d.name.startswith(f"{beh_key}-")
    )
    if not arm_dirs:
        raise RuntimeError(f"[i1481-judge] no {beh_key}-* arm dirs under {panel_root}")
    for arm_dir in arm_dirs:
        arm_id = arm_dir.name
        parts = arm_id.split("-")  # <beh>-<ctx>-<regime>-<lrtag>-s<seed>
        if len(parts) != 5:
            raise RuntimeError(f"[i1481-judge] unparseable arm dir name {arm_id!r}")
        entry: dict[str, Any] = {
            "train_ctx_key": parts[1],
            "train_ctx_id": cells.context_id_for(behavior, parts[1]),
            "regime": parts[2],
            "seed": cells.seed_for_run_id(arm_id),
            "contexts": {},
        }
        for bctx in fu3w.bystander_panel(behavior):
            ctx_id = bctx.context_id
            comp_path = arm_dir / f"completions__trained__{ctx_id}.json"
            if not comp_path.exists():
                raise RuntimeError(f"[i1481-judge] panel completions missing: {comp_path}")
            entry["contexts"][ctx_id] = _judged(
                f"pn-{arm_id}-{ctx_id}", comp_path, judge_root / f"cell_{arm_id}_{ctx_id}.json"
            )
        aggregate["arms"][arm_id] = entry
    out = out_dir / f"panel_aggregate_{beh_key}.json"
    _write_json(out, aggregate)
    logger.info("[i1481-judge] wrote %s (%d arms)", out, len(aggregate["arms"]))
    return 0


# ── Phase D: contrast ────────────────────────────────────────────────────────


def realized_panel_context_ids(mix_meta: dict, behavior: str) -> list[str]:
    """REALIZED training-panel member context ids from the mix builder's own
    ``mix_meta.json`` (never plan prose — plan §4.7 MF-3).

    Preference order: an explicit ``panel_context_ids`` key (fixtures /
    future builders) → zero realized negatives ⇒ [] (po mixes) → resolve the
    organism's recorded panel NAME through the same registrar the mix
    builder used (``fu3w.panel_name_for``), fail-loud on drift."""
    if "panel_context_ids" in mix_meta:
        return sorted(str(c) for c in mix_meta["panel_context_ids"])
    realized = (mix_meta.get("counts_realized") or {}).get("negatives")
    if realized == 0:
        return []
    organism = mix_meta.get("organism") or {}
    name = organism.get("negatives")
    if not name:
        raise RuntimeError(
            "[i1481-MF3] mix_meta carries neither panel_context_ids nor organism.negatives — "
            "cannot resolve the realized training panel"
        )
    ctx = fu3w.ensure_context(organism["context_id"], behavior)
    registered_name = fu3w.panel_name_for(ctx)  # idempotent registrar (the builder's own path)
    if registered_name != name:
        raise RuntimeError(
            f"[i1481-MF3] panel-name drift: mix_meta organism.negatives={name!r} but the "
            f"registrar resolves {registered_name!r} for {organism['context_id']!r}"
        )
    return sorted(m.to_context().context_id for m in neg_mod.NEGATIVE_PANELS[name])


def registered_heldout(behavior: str, ctx_key: str) -> list[str]:
    """The plan-§5 registered held-out read set for one training context."""
    return sorted(
        f"icl_prefix_{behavior}" if c == "icl" else c for c in REGISTERED_HELDOUT[ctx_key]
    )


def assert_heldout_disjoint(
    behavior: str, ctx_key: str, realized_panel: list[str], *, arm_label: str
) -> list[str]:
    """MF-3 (plan §4.7): fail-loud BEFORE any held-out-only D is computed.

    Asserts (a) the registered §5 held-out set ∩ the REALIZED training panel
    = ∅, and (b) the registered set equals the mechanically DERIVED set
    (``cells.heldout_contexts`` over the realized panel) — a drifted panel
    fails loud rather than silently re-scoping the decomposition."""
    reg = registered_heldout(behavior, ctx_key)
    inter = sorted(set(reg) & set(realized_panel))
    if inter:
        raise RuntimeError(
            f"[i1481-MF3] {arm_label}: held-out set ∩ REALIZED training panel = {inter} — "
            f"registered={reg} realized={sorted(realized_panel)}; refusing the held-out-only D"
        )
    derived = sorted(
        cells.heldout_contexts(behavior, cells.context_id_for(behavior, ctx_key), realized_panel)
    )
    if derived != reg:
        raise RuntimeError(
            f"[i1481-MF3] {arm_label}: derived held-out set {derived} != registered §5 set "
            f"{reg} (realized panel {sorted(realized_panel)}) — panel drift; refusing"
        )
    return reg


def _pooled_counts(agg: dict, arm_id: str, read_ctxs: list[str]) -> tuple[int, int]:
    """(k, n) pooled over ``read_ctxs`` for one panel arm (None-rates excluded,
    never coerced — mirrors ``issue1434_worker._pooled_nonsource_counts``)."""
    k = n = 0
    contexts = (agg["arms"].get(arm_id) or {}).get("contexts") or {}
    for ctx_id in read_ctxs:
        rec = contexts.get(ctx_id)
        if rec is None:
            raise RuntimeError(f"[i1481-contrast] panel cell missing: {arm_id} / {ctx_id}")
        if rec.get("rate") is None:
            continue
        k += int(rec["k_positive"])
        n += int(rec["n_scored"])
    return k, n


def _pooled_base_counts(agg: dict, read_ctxs: list[str]) -> tuple[int, int]:
    k = n = 0
    for ctx_id in read_ctxs:
        rec = (agg.get("base_panel") or {}).get(ctx_id)
        if rec is None or rec.get("rate") is None:
            continue
        k += int(rec["k_positive"])
        n += int(rec["n_scored"])
    return k, n


def _d_block(k_po: int, n_po: int, k_con: int, n_con: int) -> dict:
    """D = p_po − p_con + Newcombe 95% + the §3 lattice (None-propagating)."""
    if n_po == 0 or n_con == 0:
        return {
            "status": "not_computable",
            "D": None,
            "newcombe_95": None,
            "lattice": "not_computable",
        }
    d = k_po / n_po - k_con / n_con
    ci = cells.newcombe(k_po, n_po, k_con, n_con)
    return {
        "status": "computed",
        "D": d,
        "newcombe_95": list(ci),
        "lattice": cells.lattice_verdict(d, ci),
        "po": {"k": k_po, "n": n_po, "rate": k_po / n_po},
        "con": {"k": k_con, "n": n_con, "rate": k_con / n_con},
    }


def _content_contrast(manifest: dict, aggregates: dict[str, dict], mix_meta_root: Path) -> dict:
    """The content half of Phase D (regime_contrast pseudocode, plan §4.7)."""
    out: dict[str, Any] = {"behavior_contexts": {}, "behavior_headline": {}, "sensitivity_arms": []}
    for beh_key, beh_entry in sorted(manifest["content"].items()):
        behavior = cells.BEHAVIOR_BY_KEY[beh_key]
        agg = aggregates.get(beh_key)
        if agg is None:
            raise RuntimeError(f"[i1481-contrast] no panel aggregate for behavior {beh_key}")
        panel_ids = [c.context_id for c in fu3w.bystander_panel(behavior)]
        beh_out: dict[str, Any] = {}
        headline_cells: list[tuple[int, int, int, int]] = []
        headline_ctxs: list[str] = []
        for ctx_key, cell in sorted(beh_entry.items()):
            src_ctx = cells.context_id_for(behavior, ctx_key)
            nonsource = [c for c in panel_ids if c != src_ctx]
            # MF-3 gate FIRST: read the CON mix's realized panel, assert
            # disjointness, and only then compute any held-out-only D.
            meta_path = mix_meta_root / f"{beh_key}-{ctx_key}-con" / "mix_meta.json"
            if not meta_path.exists():
                raise RuntimeError(f"[i1481-contrast] con mix_meta missing: {meta_path}")
            realized = realized_panel_context_ids(_read_json(meta_path), behavior)
            heldout = assert_heldout_disjoint(
                behavior, ctx_key, realized, arm_label=f"{beh_key}-{ctx_key}-con"
            )
            per_seed: dict[str, Any] = {}
            pooled = {"po": [0, 0], "con": [0, 0]}
            heldout_pooled = {"po": [0, 0], "con": [0, 0]}
            seed_ds: list[float] = []
            for seed in cells.SEEDS:
                srec = cell["seeds"][str(seed)]
                arm_po = srec["po"]["arm_id"]
                arm_con = srec["con"]["arm_id"]
                k_po, n_po = _pooled_counts(agg, arm_po, nonsource)
                k_con, n_con = _pooled_counts(agg, arm_con, nonsource)
                blk = _d_block(k_po, n_po, k_con, n_con)
                hk_po, hn_po = _pooled_counts(agg, arm_po, heldout)
                hk_con, hn_con = _pooled_counts(agg, arm_con, heldout)
                per_seed[str(seed)] = {
                    "po_arm": arm_po,
                    "con_arm": arm_con,
                    **blk,
                    "heldout": _d_block(hk_po, hn_po, hk_con, hn_con),
                    "dose": srec["dose"],
                }
                if blk["status"] == "computed":
                    seed_ds.append(blk["D"])
                pooled["po"][0] += k_po
                pooled["po"][1] += n_po
                pooled["con"][0] += k_con
                pooled["con"][1] += n_con
                heldout_pooled["po"][0] += hk_po
                heldout_pooled["po"][1] += hn_po
                heldout_pooled["con"][0] += hk_con
                heldout_pooled["con"][1] += hn_con
            pooled_blk = _d_block(
                pooled["po"][0], pooled["po"][1], pooled["con"][0], pooled["con"][1]
            )
            kb, nb = _pooled_base_counts(agg, nonsource)
            if pooled_blk["status"] == "computed" and nb > 0:
                pooled_blk["base"] = {"k": kb, "n": nb, "rate": kb / nb}
                pooled_blk["d_po_vs_base"] = pooled_blk["po"]["rate"] - kb / nb
                pooled_blk["d_con_vs_base"] = pooled_blk["con"]["rate"] - kb / nb
            per_context = []
            for ctx_id in nonsource:
                rows = {}
                for seed in cells.SEEDS:
                    srec = cell["seeds"][str(seed)]
                    for regime in cells.REGIMES:
                        arm_id = srec[regime]["arm_id"]
                        try:
                            rec = agg["arms"][arm_id]["contexts"][ctx_id]
                        except KeyError as e:
                            raise RuntimeError(
                                f"[i1481-contrast] judge aggregate missing context "
                                f"{ctx_id!r} for arm {arm_id!r} — re-run --phase judge "
                                f"over the full panel"
                            ) from e
                        r = rows.setdefault(regime, [0, 0])
                        if rec.get("rate") is not None:
                            r[0] += int(rec["k_positive"])
                            r[1] += int(rec["n_scored"])
                blk = _d_block(rows["po"][0], rows["po"][1], rows["con"][0], rows["con"][1])
                blk["read_ctx"] = ctx_id
                blk["is_heldout"] = ctx_id in heldout
                for regime in cells.REGIMES:
                    if blk["status"] == "computed":
                        blk[regime]["wilson_95"] = list(
                            cells.wilson(rows[regime][0], rows[regime][1])
                        )
                per_context.append(blk)
            dose_matched = all(
                bool(cell["seeds"][str(s)]["dose"]["dose_matched"]) for s in cells.SEEDS
            )
            sign_discordant = len({d > 0 for d in seed_ds}) > 1 if len(seed_ds) == 2 else None
            beh_out[ctx_key] = {
                "source_ctx": src_ctx,
                "nonsource_contexts": nonsource,
                "per_seed": per_seed,
                "per_seed_Ds": seed_ds,
                "sign_discordant": sign_discordant,
                "pooled": pooled_blk,
                "heldout": {
                    "contexts": heldout,
                    "realized_panel": realized,
                    "mf3_checked": True,
                    "pooled": _d_block(
                        heldout_pooled["po"][0],
                        heldout_pooled["po"][1],
                        heldout_pooled["con"][0],
                        heldout_pooled["con"][1],
                    ),
                },
                "per_context": per_context,
                "dose_matched": dose_matched,
            }
            if dose_matched and pooled_blk["status"] == "computed":
                headline_cells.append(
                    (pooled["po"][0], pooled["po"][1], pooled["con"][0], pooled["con"][1])
                )
                headline_ctxs.append(ctx_key)
            # Nearest-dose sensitivity arms (plan §4.5: differs-from-verdict AND
            # persona/bare Containment|Reversed; bounded ≤16 grid-wide).
            for seed in cells.SEEDS:
                srec = cell["seeds"][str(seed)]
                nd = srec["nearest_dose_po"]
                if (
                    nd["differs_from_verdict"]
                    and ctx_key in ("pers", "bare")
                    and per_seed[str(seed)].get("lattice") in ("Containment", "Reversed")
                ):
                    out["sensitivity_arms"].append(
                        {"beh_key": beh_key, "ctx_key": ctx_key, "seed": seed, **nd}
                    )
        hk_po = sum(c[0] for c in headline_cells)
        hn_po = sum(c[1] for c in headline_cells)
        hk_con = sum(c[2] for c in headline_cells)
        hn_con = sum(c[3] for c in headline_cells)
        out["behavior_headline"][beh_key] = {
            "contexts_used": headline_ctxs,
            "realized_dose_matched_denominator": len(headline_ctxs),
            **_d_block(hk_po, hn_po, hk_con, hn_con),
        }
        out["behavior_contexts"][beh_key] = beh_out
    out["sensitivity_arms"] = out["sensitivity_arms"][:SENSITIVITY_ARM_CAP]
    return out


def _panel_rec(marker_root: Path, run_id: str, step: int) -> dict:
    """One panel battery read (``panel/rung<step>.json``), fail-loud on a
    missing file (the marker dispatcher persists batteries at the selected /
    emission-onset / ceiling rungs)."""
    path = marker_root / run_id / "panel" / f"rung{step}.json"
    if not path.exists():
        raise RuntimeError(f"[i1481-contrast] marker panel read missing: {path}")
    return _read_json(path)


def _probe_dg_margin(row: dict) -> tuple[float, float]:
    """Per-probe (ΔG, Δ(z_marker − z_eos)) trained − base from one four-float
    ``per_probe`` row (`.claude/rules/marker-leakage-measurement.md`)."""
    dg = float(row["trained"]["logp"]) - float(row["base"]["logp"])
    dm = (float(row["trained"]["z_marker"]) - float(row["trained"]["z_eos"])) - (
        float(row["base"]["z_marker"]) - float(row["base"]["z_eos"])
    )
    return dg, dm


def _paired_marker_diffs(marker_root: Path, con_arm: dict, po_arm: dict) -> dict:
    """Per-(read_ctx, question) paired ΔG differences (po − con) at the two
    verdict arms' SELECTED rungs, from the panel four-float per_probe rows."""
    diffs: dict[str, dict[int, list[float]]] = {}
    for label, info in (("con", con_arm), ("po", po_arm)):
        run_id = info["run_id"]
        step = int(info["selection"]["step"])
        rec = _panel_rec(marker_root, run_id, step)
        for row in rec.get("per_probe") or []:
            ctx_id = row["row"]["context_id"]
            q = int(row["row"]["q"])
            dg = float(row["trained"]["logp"]) - float(row["base"]["logp"])
            diffs.setdefault(ctx_id, {}).setdefault(q, [None, None])[0 if label == "con" else 1] = (
                dg
            )
    out: dict[str, dict[int, float]] = {}
    for ctx_id, per_q in diffs.items():
        for q, (dg_con, dg_po) in per_q.items():
            if dg_con is None or dg_po is None:
                raise RuntimeError(
                    f"[i1481-contrast] unpaired marker probe ({ctx_id}, q={q}) — one regime's "
                    "panel read is missing this (context, question) cell"
                )
            out.setdefault(ctx_id, {})[q] = dg_po - dg_con
    return out


def _cluster_bootstrap_ci(per_question) -> list[float]:
    """Question-cluster bootstrap 95% CI (2000 draws, seed 653) — VECTORIZED:
    one integer-index resample matrix + one mean reduction, never a Python
    loop over draws (plan §6; `.claude/rules/vectorize-many-cell-fits.md`)."""
    import numpy as np

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    q = per_question.shape[0]
    idx = rng.integers(0, q, size=(BOOTSTRAP_DRAWS, q))
    samples = per_question[idx].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [float(lo), float(hi)]


def _marker_contrast(manifest: dict, marker_root: Path) -> dict:
    """Marker half of Phase D: pooled + per-context D (nats) with the
    question-cluster bootstrap, seed-paired across seeds; emission-map and
    three-space table data ride along for the figures driver."""
    import numpy as np

    import issue1481_marker as mk

    marker = manifest.get("marker")
    if not marker:
        raise RuntimeError(
            "[i1481-contrast] manifest has no marker section — re-run select with --marker-root"
        )
    src_by_ctx = mk.CTX_SOURCE_ID
    out: dict[str, Any] = {
        "window": marker["window"],
        "tol_nats": marker["tol_nats"],
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "cluster": "question"},
        "contexts": {},
        "three_space": {},
    }
    for ctx_key, ctx_entry in sorted(marker["contexts"].items()):
        src_id = src_by_ctx[ctx_key]
        per_seed_diffs: list[dict[str, dict[int, float]]] = []
        per_seed_meta: dict[str, Any] = {}
        for seed_s, srec in sorted(ctx_entry.items()):
            diffs = _paired_marker_diffs(marker_root, srec["con"], srec["po"])
            per_seed_diffs.append(diffs)
            nonsource_vals = [
                d for cid, per_q in diffs.items() if cid != src_id for d in per_q.values()
            ]
            per_seed_meta[seed_s] = {
                "D_pooled_nonsource_nats": float(np.mean(nonsource_vals)),
                "dose_matched": srec["dose_matched"],
                "abs_gap_nats": srec["abs_gap_nats"],
                "con_arm": srec["con"]["run_id"],
                "po_arm": srec["po"]["run_id"],
            }
        # Seed-pair per (ctx, q): average the paired diff across seeds, then
        # bootstrap QUESTIONS (the registered cluster).
        ctx_ids = sorted({cid for d in per_seed_diffs for cid in d})
        per_context: dict[str, Any] = {}
        pooled_by_q: dict[int, list[float]] = {}
        for cid in ctx_ids:
            qs = sorted(set.intersection(*[set(d.get(cid, {})) for d in per_seed_diffs]))
            if not qs:
                raise RuntimeError(f"[i1481-contrast] marker ctx {cid}: no shared questions")
            vec = np.array([float(np.mean([d[cid][q] for d in per_seed_diffs])) for q in qs])
            per_context[cid] = {
                "D_nats": float(vec.mean()),
                "bootstrap_95": _cluster_bootstrap_ci(vec),
                "n_questions": len(qs),
                "is_source": cid == src_id,
            }
            if cid != src_id:
                for q, v in zip(qs, vec, strict=True):
                    pooled_by_q.setdefault(q, []).append(float(v))
        pooled_vec = np.array([float(np.mean(vs)) for _q, vs in sorted(pooled_by_q.items())])
        ci = _cluster_bootstrap_ci(pooled_vec)
        d = float(pooled_vec.mean())
        seeds_matched = all(m["dose_matched"] for m in per_seed_meta.values())
        out["contexts"][ctx_key] = {
            "source_ctx": src_id,
            "per_seed": per_seed_meta,
            "pooled_nonsource": {
                "D_nats": d,
                "bootstrap_95": ci,
                "lattice": cells.lattice_verdict(d, (ci[0], ci[1])),
                "n_questions": int(pooled_vec.shape[0]),
            },
            "per_context": per_context,
            "dose_matched": seeds_matched,
        }
    # Three-space per-cell table + Δlog P vs Δz divergence points (from the
    # selected-rung slot reads; plan §6 marker mechanistic secondary).
    for run_id, arm in sorted(marker["arms"].items()):
        step = arm["selection"].get("step")
        if step is None:
            continue
        path = marker_root / run_id / f"slot_reads_rung{step}.json"
        if not path.exists():
            raise RuntimeError(f"[i1481-contrast] slot reads missing: {path}")
        rec = _read_json(path)
        rows = rec.get("per_probe") or []
        d_logp = [float(r["trained"]["logp"]) - float(r["base"]["logp"]) for r in rows]
        d_z = [float(r["trained"]["z_marker"]) - float(r["base"]["z_marker"]) for r in rows]
        d_margin = [
            (float(r["trained"]["z_marker"]) - float(r["trained"]["z_eos"]))
            - (float(r["base"]["z_marker"]) - float(r["base"]["z_eos"]))
            for r in rows
        ]
        p_base = [float(np.exp(r["base"]["logp"])) for r in rows]
        out["three_space"][run_id] = {
            "step": step,
            "delta_logp_mean": float(np.mean(d_logp)),
            "delta_z_marker_mean": float(np.mean(d_z)),
            "delta_eos_margin_mean": float(np.mean(d_margin)),
            "delta_p_mean": float(
                np.mean([pb * (np.exp(dl) - 1.0) for pb, dl in zip(p_base, d_logp, strict=True)])
            ),
            "divergence_points": [
                {"delta_logp": dl, "delta_z": dz} for dl, dz in zip(d_logp, d_z, strict=True)
            ],
        }
    # Plan §6 install-strength read 3: leakage-vs-install dose curves at the
    # 3 panel rungs per cell + full source-install trajectories.
    out["dose_curves"] = _marker_dose_curves(manifest, marker_root)
    return out


# Transfer fractions with a near-zero EOS-margin denominator are reported as
# None (an uninstalled source has no meaningful fraction), never a wild ratio.
MARGIN_FRACTION_FLOOR = 1e-6

# Ladder trajectory fields copied verbatim from each reads_by_step record.
_TRAJECTORY_KEYS = (
    "delta_logp_mean",
    "delta_margin_mean",
    "source_emission_rate",
    "gen_emission_rate",
)


def _marker_dose_curves(manifest: dict, marker_root: Path) -> dict:
    """Plan §6 install-strength read 3 (marker): per cell, panel-context ΔG +
    EOS-margin transfer fractions Δ(z_marker − z_eos) vs source install at the
    3 panel rungs (selected / emission-onset / ceiling), plus the full
    source-install trajectory from the per-rung ladder. Fractions are computed
    in EOS-margin logit space, NEVER raw log P
    (`.claude/rules/marker-leakage-measurement.md` § Install-strength
    confound); the per-(context, question) rows ride along for the
    raw-alongside-processed companion figure."""
    import issue1481_marker as mk

    marker = manifest.get("marker")
    if not marker:
        raise RuntimeError(
            "[i1481-contrast] manifest has no marker section — re-run select with --marker-root"
        )
    out: dict[str, Any] = {}
    for run_id, arm in sorted(marker["arms"].items()):
        sel_path = marker_root / run_id / "selection.json"
        if not sel_path.exists():
            raise RuntimeError(f"[i1481-contrast] marker selection missing: {sel_path}")
        sel = _read_json(sel_path)
        panel_rungs = sel.get("panel_rungs")
        if not panel_rungs:
            raise RuntimeError(
                f"[i1481-contrast] selection.json for {run_id} carries no panel_rungs — "
                "the marker dispatcher's battery-role record is required for dose curves"
            )
        src_id = mk.CTX_SOURCE_ID[arm["ctx_key"]]
        rungs: list[dict] = []
        for step_s, roles in sorted(panel_rungs.items(), key=lambda kv: int(kv[0])):
            step = int(step_s)
            rec = _panel_rec(marker_root, run_id, step)
            per_ctx: dict[str, dict[str, list[float]]] = {}
            per_question: list[dict] = []
            for row in rec.get("per_probe") or []:
                ctx_id = row["row"]["context_id"]
                dg, dm = _probe_dg_margin(row)
                d = per_ctx.setdefault(ctx_id, {"dg": [], "dm": []})
                d["dg"].append(dg)
                d["dm"].append(dm)
                if ctx_id != src_id:
                    per_question.append(
                        {
                            "context_id": ctx_id,
                            "q": int(row["row"]["q"]),
                            "delta_logp": dg,
                            "delta_margin": dm,
                        }
                    )
            if src_id not in per_ctx:
                raise RuntimeError(
                    f"[i1481-contrast] {run_id} panel rung{step}: source context "
                    f"{src_id} missing from per_probe rows"
                )
            src_dg = float(sum(per_ctx[src_id]["dg"]) / len(per_ctx[src_id]["dg"]))
            src_dm = float(sum(per_ctx[src_id]["dm"]) / len(per_ctx[src_id]["dm"]))
            contexts: dict[str, dict] = {}
            for ctx_id, v in sorted(per_ctx.items()):
                dg_mean = float(sum(v["dg"]) / len(v["dg"]))
                dm_mean = float(sum(v["dm"]) / len(v["dm"]))
                frac = None
                if ctx_id != src_id and abs(src_dm) > MARGIN_FRACTION_FLOOR:
                    frac = float(dm_mean / src_dm)
                contexts[ctx_id] = {
                    "delta_logp_mean": dg_mean,
                    "delta_margin_mean": dm_mean,
                    "margin_transfer_fraction": frac,
                    "is_source": ctx_id == src_id,
                }
            ns = [c for cid, c in contexts.items() if cid != src_id]
            if not ns:
                raise RuntimeError(
                    f"[i1481-contrast] {run_id} panel rung{step}: no non-source contexts"
                )
            fracs = [c["margin_transfer_fraction"] for c in ns]
            frac_mean = (
                float(sum(fracs) / len(fracs)) if all(f is not None for f in fracs) else None
            )
            rungs.append(
                {
                    "step": step,
                    "roles": sorted(roles),
                    "source_install_logp": src_dg,
                    "source_install_margin": src_dm,
                    "per_context": contexts,
                    "nonsource_delta_logp_mean": float(
                        sum(c["delta_logp_mean"] for c in ns) / len(ns)
                    ),
                    "nonsource_margin_transfer_fraction_mean": frac_mean,
                    "per_question": per_question,
                }
            )
        reads = {int(k): v for k, v in (arm.get("reads_by_step") or {}).items()}
        if not reads:
            raise RuntimeError(f"[i1481-contrast] {run_id}: empty reads_by_step ladder")
        trajectory = [
            {"step": s, **{k: reads[s][k] for k in _TRAJECTORY_KEYS}} for s in sorted(reads)
        ]
        out[run_id] = {
            "ctx_key": arm["ctx_key"],
            "regime": arm["regime"],
            "lr_key": arm["lr_key"],
            "seed": arm["seed"],
            "selected_step": int(sel["step"]),
            "source_context": src_id,
            "rungs": rungs,
            "trajectory": trajectory,
        }
    return out


def _margin_rate_validation(manifest: dict, content_root: Path | None) -> dict:
    """ρ(TF fixed-pool margin, Tier-1 selection rate) per behavior (plan §6
    dual-DV validation). Reads per-run ``margin.json`` under the content
    out_root; casual stays QUARANTINED (#1434 ρ=−0.46) and is reported as
    such, never narrated as the construct."""
    out: dict[str, Any] = {}
    if content_root is None:
        return {"status": "skipped — no --content-root supplied"}
    from scipy.stats import spearmanr

    for beh_key, beh_entry in sorted(manifest["content"].items()):
        pairs = []
        missing = []
        for cell in beh_entry.values():
            for arm_id, arm in cell["arms"].items():
                mpath = content_root / arm_id / "margin.json"
                if not mpath.exists():
                    missing.append(arm_id)
                    continue
                m = _read_json(mpath)
                margin = m.get("margin") if isinstance(m, dict) else None
                if margin is None:
                    margin = (
                        (m.get("tf_margin") or {}).get("margin") if isinstance(m, dict) else None
                    )
                if margin is None:
                    missing.append(arm_id)
                    continue
                pairs.append((float(margin), float(arm["selection"]["rate"])))
        rec: dict[str, Any] = {
            "n_pairs": len(pairs),
            "n_missing": len(missing),
            "missing_arms": sorted(missing)[:20],
        }
        if len(pairs) >= 3:
            rho, p = spearmanr([p_[0] for p_ in pairs], [p_[1] for p_ in pairs])
            rec.update(
                {
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                    "points": [{"margin": a, "rate": b} for a, b in pairs],
                }
            )
        else:
            rec["status"] = "not_computable — <3 (margin, rate) pairs"
        if beh_key == "cas":
            rec["quarantined"] = True
            rec["quarantine_note"] = (
                "casual TF margin FAILED the standing rho(margin, rate) validation in #1434 "
                "(rho=-0.46) and stays quarantined — reported, never a leakage read"
            )
        out[beh_key] = rec
    return out


def phase_contrast(args: argparse.Namespace) -> int:
    """Phase D: regime contrast + MF-3 + marker bootstrap + margin validation."""
    out_dir = Path(args.out_dir)
    manifest = _read_json(Path(args.manifest))
    aggregates: dict[str, dict] = {}
    for beh_key in manifest["content"]:
        path = Path(args.aggregates_dir) / f"panel_aggregate_{beh_key}.json"
        if not path.exists():
            raise RuntimeError(f"[i1481-contrast] panel aggregate missing: {path}")
        aggregates[beh_key] = _read_json(path)
    stub = any(a.get("smoke_stub_judge") for a in aggregates.values())
    content = _content_contrast(manifest, aggregates, Path(args.mix_meta_root))
    content.update({"issue": cells.ISSUE_1481, "smoke_stub_judge": stub, "ts": _ts()})
    _write_json(out_dir / "regime_contrast_content.json", content)
    logger.info("[i1481-contrast] wrote %s", out_dir / "regime_contrast_content.json")
    if args.marker_root:
        marker = _marker_contrast(manifest, Path(args.marker_root))
        marker.update({"issue": cells.ISSUE_1481, "ts": _ts()})
        _write_json(out_dir / "regime_contrast_marker.json", marker)
        logger.info("[i1481-contrast] wrote %s", out_dir / "regime_contrast_marker.json")
    validation = _margin_rate_validation(
        manifest, Path(args.content_root) if args.content_root else None
    )
    _write_json(out_dir / "margin_rate_validation.json", validation)
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="#1481 Phase B/D analysis driver")
    sub = p.add_subparsers(dest="phase", required=True)

    ps = sub.add_parser("select", help="Phase B: verdict-arm manifest")
    ps.add_argument("--out-dir", required=True)
    ps.add_argument("--ladders-dir", default=str(cells.DELIVERABLES_DIR_1481))
    ps.add_argument("--repo-root", default=str(cells.REPO_ROOT))
    ps.add_argument("--marker-root", default=None)
    ps.add_argument("--behaviors", default="cas,imp,syc")
    ps.add_argument("--contexts", default=",".join(cells.CTX_KEYS))

    pj = sub.add_parser("judge", help="judge panel + base-panel completions (one behavior)")
    pj.add_argument("--out-dir", required=True)
    pj.add_argument("--behavior", required=True)
    pj.add_argument("--panel-root", required=True)
    pj.add_argument("--base-panel-root", required=True)
    pj.add_argument("--n-draws", type=int, default=JUDGE_N_DRAWS_DEFAULT)
    pj.add_argument("--questions", default=None, help="JSON list override (fixtures)")
    pj.add_argument("--question-limit", type=int, default=None)
    pj.add_argument("--smoke", action="store_true")
    pj.add_argument(
        "--stub-judge",
        action="store_true",
        help="smoke-only deterministic offline judge (requires --smoke)",
    )

    pc = sub.add_parser("contrast", help="Phase D: regime contrast")
    pc.add_argument("--out-dir", required=True)
    pc.add_argument("--manifest", required=True)
    pc.add_argument("--aggregates-dir", required=True)
    pc.add_argument(
        "--mix-meta-root",
        required=True,
        help="dir of <beh>-<ctx>-con/mix_meta.json (MF-3 realized-panel source)",
    )
    pc.add_argument("--marker-root", default=None)
    pc.add_argument(
        "--content-root",
        default=None,
        help="content out_root holding per-run margin.json (dual-DV validation)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    if args.phase == "select":
        return phase_select(args)
    if args.phase == "judge":
        return phase_judge(args)
    if args.phase == "contrast":
        return phase_contrast(args)
    raise SystemExit(f"unknown phase {args.phase!r}")


if __name__ == "__main__":
    raise SystemExit(main())
