"""#1481 zero-GPU free-analysis round (9a-ter) — three persisted-artifact analyses.

Subcommands (one output JSON each, under ``eval_results/issue_1481/analysis/``):

- ``tier1-recount`` — Tier-1 SELECTION-rung CJK-intrusion recount over the 48
  content verdict arms (verdict_manifest.json), rejoining judge means from each
  staged rung's own ``judge_raw.json`` (``graded_judge.judge_result_from_save_raw``
  — the production reduce, drop-never-coerce), then re-deriving (a) in-band
  status (band 0.60-0.85) and (b) the |Δrate| <= 0.10 dose-match label under the
  ``zeroed`` / ``excluded`` conventions of ``cjk_recount_headlines.json``.
  Scope note: only the SELECTED rung's rate is recomputed — the selection RULE
  (earliest-in-band rung / lowest-LR arm) is not re-run over full ladders, so a
  convention-flipped selection rung elsewhere in a ladder is out of scope.
  Pools staged from ``issue1481_conpos_grid/raw_completions/{<round>,reread}/``;
  the 8 cas seed-42 verdict arms are reused #1434 cells whose Tier-1 pools live
  outside this bucket — labels carried from the manifest, flagged.

- ``marker-interp`` — matched-install interpolated D for the 8 marker
  (context, seed) cells: linearly interpolate the po arm's pooled non-source
  read (log-prob AND EOS-margin space) along its panel dose curve to the con
  arm's exact realized source install, recomputing D at zero residual install
  gap, with a question-cluster bootstrap (2000 draws, seed 653 — #1333 parity)
  over the per-(context, question) rows; installs are treated as FIXED (the
  committed bootstrap's convention; source-side reads are not per-question in
  the dose curves).

- ``degeneration`` — (a) per marker cell x rung-class truncation /
  ends-with-marker / ※-repeater / degenerate-response counts from the persisted
  panel rollout text (plan §6 L275); (b) verdict-arm graded score DISTRIBUTIONS
  (not just rates) con vs po per behavior x context x seed — panel pooled
  non-source AND source Tier-1 selection pools — with KS + Mann-Whitney tests
  (plan §6 L277 "matched rate != matched graded intensity").

Zero GPU, zero new generation, zero judge API calls — everything computes from
persisted artifacts (HF data-repo pools staged via ``hub.stage_hub_file`` /
``stage_hub_prefix``; local judge_packed shards; committed analysis JSONs).

Usage:
    uv run python scripts/issue1481_free_analysis.py tier1-recount
    uv run python scripts/issue1481_free_analysis.py marker-interp
    uv run python scripts/issue1481_free_analysis.py degeneration
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from issue1481_cjk_audit import CJK_RE, INSTRUMENT, _item_means, _wilson  # noqa: E402

from explore_persona_space.eval.graded_judge import judge_result_from_save_raw  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

logger = logging.getLogger("i1481_free_analysis")

DATA_REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue1481_conpos_grid"
ANALYSIS_DIR = Path("eval_results/issue_1481/analysis")
TIER1_STAGE = Path("data/issue_1481/tier1_stage")
PHASED_STAGE = Path("data/issue_1481/phaseD_stage")
JUDGE_PACKED_DIR = Path("data/issue_1481/judge_packed")
BAND = (0.60, 0.85)
DOSE_GAP = 0.10
THRESHOLD = 50  # Behavior.threshold project standard (artifacts/behavior.py)
# C.MARKER_MAX_NEW_TOKENS (experiments/issue_1333/__init__.py) — the cap the
# marker panel generations ran under; token length >= cap == truncated.
MARKER_MAX_NEW_TOKENS = 2048
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BOOT_DRAWS = 2000
BOOT_SEED = 653  # #1333 question-cluster bootstrap parity
# fu4 degeneracy_stats per-completion definition (scripts/issue1090_fu4.py):
# word-level 4-gram repeat fraction; > DEGEN_MAX_REPEAT_FRAC == degenerate.
DEGEN_MAX_REPEAT_FRAC = 0.5


def _meta() -> dict:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "git_commit": sha,
        "ts": datetime.now(timezone.utc).isoformat(),
        "numpy": np.__version__,
        "script": "scripts/issue1481_free_analysis.py",
    }


def _read_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def _atomic_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    tmp.replace(p)


# ── item 1: Tier-1 selection-pool CJK recount ────────────────────────────────


def _verdict_arms(vm: dict, rcc: dict) -> list[dict]:
    """The 48 content verdict arms with pool-resolution info."""
    rows: list[dict] = []
    for beh, ctxs in vm["content"].items():
        for ctx_key, cell in ctxs.items():
            src_ctx = rcc["behavior_contexts"][beh][ctx_key]["source_ctx"]
            for seed, sv in cell["seeds"].items():
                for reg in ("con", "po"):
                    arm = sv[reg]["arm_id"]
                    arm_rec = cell["arms"][arm]
                    rows.append(
                        {
                            "behavior": beh,
                            "ctx_key": ctx_key,
                            "seed": seed,
                            "regime": reg,
                            "arm_id": arm,
                            "source": arm_rec["source"],
                            "step": sv[reg]["selection"]["step"],
                            "manifest_rate": sv[reg]["selection"]["rate"],
                            "manifest_in_band": sv[reg]["selection"]["in_band"],
                            "source_ctx": src_ctx,
                        }
                    )
    return rows


def _pool_repo_prefix(row: dict) -> str | None:
    """HF prefix of the arm's selection-rung rate pool, or None (reused-1434)."""
    beh, arm, step = row["behavior"], row["arm_id"], row["step"]
    if row["source"] == "fresh":
        rnd = f"i1481{beh}po" if row["regime"] == "po" else f"i1481{beh}"
        return f"{PREFIX}/raw_completions/{rnd}/rate/{arm}/rate_checkpoint-{step}"
    if row["source"] == "regen":
        return f"{PREFIX}/raw_completions/i1481{beh}regen/rate/{arm}/rate_checkpoint-{step}"
    if row["source"] == "reused-parent":
        # the P1 apply-and-read pool at the committed selected checkpoint —
        # its rate is the RE-READ (parity |Δ| <= 0.15 vs committed), flagged.
        return f"{PREFIX}/raw_completions/reread/{arm}/rate/rate_checkpoint-{step}"
    if row["source"] == "reused-1434":
        return None  # #1434 pools live outside this bucket (brief scope)
    raise ValueError(f"unknown arm source {row['source']!r} for {row['arm_id']}")


def _stage_pool(row: dict) -> Path | None:
    prefix = _pool_repo_prefix(row)
    if prefix is None:
        return None
    dest = TIER1_STAGE / row["arm_id"] / f"rate_checkpoint-{row['step']}"
    src = row["source_ctx"]
    hub.stage_hub_file(
        DATA_REPO, f"{prefix}/completions__trained__{src}.json", dest / "completions.json"
    )
    hub.stage_hub_file(
        DATA_REPO, f"{prefix}/judge/trained_{src}/judge_raw.json", dest / "judge_raw.json"
    )
    return dest


def _recount_pool(stage_dir: Path, src_ctx: str) -> dict:
    """CJK recount of one staged selection-rung pool under both conventions.

    Rejoins per-item judge means from the pool's own ``judge_raw.json`` via the
    production reduce (``judge_result_from_save_raw`` — drop-never-coerce);
    positive == mean > THRESHOLD, matching ``organisms._rate_for_cell``.
    """
    doc = _read_json(stage_dir / "completions.json")
    questions, comps = doc["questions"], doc["completions"]
    items = [
        (f"{src_ctx}-trained-q{i:03d}-c{j}", q, comps[i][j])
        for i, q in enumerate(questions)
        for j in range(len(comps[i]))
    ]
    result = judge_result_from_save_raw(stage_dir / "judge_raw.json", items)
    k = n = intruded = fired_and_intruded = dropped = 0
    for iid, _q, comp in items:
        score = result.scores[iid]
        if score is None:
            dropped += 1
            continue
        n += 1
        pos = score > THRESHOLD
        intr = bool(CJK_RE.search(comp))
        k += int(pos)
        intruded += int(intr)
        fired_and_intruded += int(pos and intr)
    if n == 0:
        raise ValueError(f"pool at {stage_dir} fully judge-dropped — judging outage, not 0%")
    raw = k / n
    zeroed = (k - fired_and_intruded) / n
    excluded = (k - fired_and_intruded) / (n - intruded) if n > intruded else float("nan")
    return {
        "n_scored": n,
        "n_dropped": dropped,
        "k_positive": k,
        "intruded": intruded,
        "fired_and_intruded": fired_and_intruded,
        "raw_rate": raw,
        "zeroed_rate": zeroed,
        "excluded_rate": excluded,
        "wilson_95_raw": _wilson(k, n),
    }


def _in_band(rate: float) -> bool:
    return BAND[0] <= rate <= BAND[1]


def cmd_tier1_recount() -> int:
    vm = _read_json(ANALYSIS_DIR / "verdict_manifest.json")
    rcc = _read_json(ANALYSIS_DIR / "regime_contrast_content.json")
    rows = _verdict_arms(vm, rcc)
    logger.info("staging %d verdict-arm selection pools", len(rows))
    with ThreadPoolExecutor(6) as ex:
        staged = list(ex.map(_stage_pool, rows))

    per_arm: dict[str, dict] = {}
    for row, stage_dir in zip(rows, staged, strict=True):
        rec = dict(row)
        if stage_dir is None:
            rec["pool"] = "unavailable-1434"
            rec["note"] = (
                "reused #1434 cell — Tier-1 pool outside issue1481_conpos_grid; "
                "manifest rate carried under every convention"
            )
            rec["raw_rate"] = rec["zeroed_rate"] = rec["excluded_rate"] = row["manifest_rate"]
        else:
            rec["pool"] = "reread" if row["source"] == "reused-parent" else "grid"
            rec.update(_recount_pool(stage_dir, row["source_ctx"]))
            rec["abs_diff_vs_manifest"] = abs(rec["raw_rate"] - row["manifest_rate"])
        for conv in ("raw", "zeroed", "excluded"):
            rec[f"in_band_{conv}"] = _in_band(rec[f"{conv}_rate"])
        per_arm[row["arm_id"]] = rec

    # per-cell label re-derivation + flips vs manifest
    cells: dict[str, dict] = {}
    labels_changed: list[dict] = []
    for beh, ctxs in vm["content"].items():
        for ctx_key, cell in ctxs.items():
            for seed, sv in cell["seeds"].items():
                con = per_arm[sv["con"]["arm_id"]]
                po = per_arm[sv["po"]["arm_id"]]
                manifest_dose = sv["dose"]
                cid = f"{beh}-{ctx_key}-s{seed}"
                out = {
                    "con_arm": con["arm_id"],
                    "po_arm": po["arm_id"],
                    "manifest": {
                        "dose_matched": manifest_dose["dose_matched"],
                        "con_in_band": manifest_dose["con_in_band"],
                        "pos_in_band": manifest_dose["pos_in_band"],
                        "rate_gap": manifest_dose["rate_gap"],
                    },
                    "conventions": {},
                }
                for conv in ("raw", "zeroed", "excluded"):
                    cr, pr = con[f"{conv}_rate"], po[f"{conv}_rate"]
                    cband, pband = con[f"in_band_{conv}"], po[f"in_band_{conv}"]
                    matched = cband and pband and abs(cr - pr) <= DOSE_GAP
                    out["conventions"][conv] = {
                        "con_rate": cr,
                        "po_rate": pr,
                        "con_in_band": cband,
                        "po_in_band": pband,
                        "rate_gap": abs(cr - pr),
                        "dose_matched": matched,
                    }
                    if conv == "raw":
                        continue
                    flips = []
                    if matched != manifest_dose["dose_matched"]:
                        flips.append("dose_matched")
                    if cband != manifest_dose["con_in_band"]:
                        flips.append("con_in_band")
                    if pband != manifest_dose["pos_in_band"]:
                        flips.append("po_in_band")
                    # intrusion-driven == flip vs the SAME pool's raw-convention
                    # labels (isolates intrusion from reread/pool drift)
                    raw_c = out["conventions"]["raw"]
                    intr_driven = (
                        matched != raw_c["dose_matched"]
                        or cband != raw_c["con_in_band"]
                        or pband != raw_c["po_in_band"]
                    )
                    if flips:
                        labels_changed.append(
                            {
                                "cell": cid,
                                "convention": conv,
                                "flipped": flips,
                                "intrusion_driven": intr_driven,
                                "manifest_dose_matched": manifest_dose["dose_matched"],
                                "new_dose_matched": matched,
                                "con_pool": con["pool"],
                                "po_pool": po["pool"],
                            }
                        )
                cells[cid] = out

    n_intr = sum(v.get("intruded", 0) for v in per_arm.values())
    n_tot = sum(v.get("n_scored", 0) for v in per_arm.values())
    out = {
        "meta": _meta(),
        "conventions": {
            "raw": "recounted pool rate, no intrusion handling",
            "zeroed": "intruded completion's judge label forced non-positive; denominator kept",
            "excluded": "intruded rows dropped from numerator and denominator",
            "positive": f"per-item mean judge score over kept draws > {THRESHOLD}",
        },
        "band": list(BAND),
        "dose_match_max_rate_gap": DOSE_GAP,
        "scope_note": (
            "selection-rung pools only — the earliest-in-band / lowest-LR selection "
            "rule is not re-run over full ladders under the conventions"
        ),
        "per_arm": per_arm,
        "cells": cells,
        "labels_changed": labels_changed,
        "summary": {
            "n_verdict_arms": len(per_arm),
            "n_arms_recomputed": sum(1 for v in per_arm.values() if "n_scored" in v),
            "n_arms_carried_1434": sum(
                1 for v in per_arm.values() if v["pool"] == "unavailable-1434"
            ),
            "n_scored_total": n_tot,
            "n_intruded_total": n_intr,
            "n_label_flips": len(labels_changed),
        },
    }
    _atomic_json(ANALYSIS_DIR / "tier1_intrusion_recount.json", out)
    logger.info(
        "tier1-recount: %d/%d intruded over %d recomputed arms; %d label flips",
        n_intr,
        n_tot,
        out["summary"]["n_arms_recomputed"],
        len(labels_changed),
    )
    return 0


# ── item 2: marker matched-install interpolated D ────────────────────────────


def _rung_nonsource_pq(rung: dict, src_ctx: str, key: str) -> np.ndarray:
    """(n_ctx, n_q) per-question non-source deltas for one dose-curve rung."""
    rows = [p for p in rung["per_question"] if p["context_id"] != src_ctx]
    ctxs = sorted({p["context_id"] for p in rows})
    n_q = max(p["q"] for p in rows) + 1
    arr = np.full((len(ctxs), n_q), np.nan)
    for p in rows:
        arr[ctxs.index(p["context_id"]), p["q"]] = p[key]
    assert not np.isnan(arr).any(), f"per_question coverage hole at rung {rung['step']}"
    return arr


def _selected_rung(curve: dict) -> dict:
    sel = [r for r in curve["rungs"] if "selected" in (r.get("roles") or [])]
    if len(sel) != 1:
        sel = [r for r in curve["rungs"] if r["step"] == curve["selected_step"]]
    assert len(sel) == 1, f"no unique selected rung for {curve.get('lr_key')}"
    return sel[0]


def cmd_marker_interp() -> int:
    rcm = _read_json(ANALYSIS_DIR / "regime_contrast_marker.json")
    dose_curves = rcm["dose_curves"]
    spaces = {
        "logp": ("source_install_logp", "delta_logp"),
        "margin": ("source_install_margin", "delta_margin"),
    }
    rng = np.random.default_rng(BOOT_SEED)
    cells: dict[str, dict] = {}
    for ctx_key in sorted(rcm["contexts"]):
        cblock = rcm["contexts"][ctx_key]
        src_ctx = cblock["source_ctx"]
        for seed in sorted(cblock["per_seed"]):
            ps = cblock["per_seed"][seed]
            con = dose_curves[ps["con_arm"]]
            po = dose_curves[ps["po_arm"]]
            con_sel, po_sel = _selected_rung(con), _selected_rung(po)
            n_q = 20
            draws_idx = rng.integers(0, n_q, size=(BOOT_DRAWS, n_q))
            cell: dict = {
                "con_arm": ps["con_arm"],
                "po_arm": ps["po_arm"],
                "con_selected_step": con_sel["step"],
                "po_selected_step": po_sel["step"],
                "committed_D_raw_logp": ps["D_pooled_nonsource_nats"],
                "committed_abs_gap_nats": ps["abs_gap_nats"],
                "spaces": {},
            }
            for space, (xkey, pqkey) in spaces.items():
                con_install = con_sel[xkey]
                po_rungs = sorted(po["rungs"], key=lambda r: r[xkey])
                xs = [r[xkey] for r in po_rungs]
                # bracket in install space (earliest containing pair)
                lo = hi = None
                for a, b in itertools.pairwise(po_rungs):
                    if a[xkey] <= con_install <= b[xkey]:
                        lo, hi = a, b
                        break
                extrapolated = lo is None
                if extrapolated:
                    # clamp to the nearest endpoint; report the nearest pair
                    lo, hi = (
                        (po_rungs[0], po_rungs[1])
                        if con_install < xs[0]
                        else (
                            po_rungs[-2],
                            po_rungs[-1],
                        )
                    )
                    w = 0.0 if con_install < xs[0] else 1.0
                else:
                    span = hi[xkey] - lo[xkey]
                    w = 0.5 if span == 0 else (con_install - lo[xkey]) / span
                con_pq = _rung_nonsource_pq(con_sel, src_ctx, pqkey)
                po_sel_pq = _rung_nonsource_pq(po_sel, src_ctx, pqkey)
                lo_pq = _rung_nonsource_pq(lo, src_ctx, pqkey)
                hi_pq = _rung_nonsource_pq(hi, src_ctx, pqkey)
                assert con_pq.shape[1] == n_q, con_pq.shape

                def _boot(arr: np.ndarray) -> np.ndarray:
                    return arr[:, draws_idx].mean(axis=(0, 2))  # noqa: B023

                y_con, y_lo, y_hi, y_sel = (
                    _boot(con_pq),
                    _boot(lo_pq),
                    _boot(hi_pq),
                    _boot(po_sel_pq),
                )
                d_interp_draws = (1 - w) * y_lo + w * y_hi - y_con
                d_raw_draws = y_sel - y_con
                d_interp = float((1 - w) * lo_pq.mean() + w * hi_pq.mean() - con_pq.mean())
                d_raw = float(po_sel_pq.mean() - con_pq.mean())
                ci_i = np.percentile(d_interp_draws, [2.5, 97.5]).tolist()
                ci_r = np.percentile(d_raw_draws, [2.5, 97.5]).tolist()
                cell["spaces"][space] = {
                    "con_install": con_install,
                    "po_install_at_selected": po_sel[xkey],
                    "residual_gap_at_selection": po_sel[xkey] - con_install,
                    "po_bracket_rungs": [
                        {
                            "step": r["step"],
                            "install": r[xkey],
                            "nonsource_read": float(_rung_nonsource_pq(r, src_ctx, pqkey).mean()),
                            "roles": r.get("roles"),
                        }
                        for r in (lo, hi)
                    ],
                    "interp_weight": w,
                    "extrapolation_clamped": extrapolated,
                    "D_raw": d_raw,
                    "D_interp": d_interp,
                    "D_interp_minus_raw": d_interp - d_raw,
                    "ci_raw_95": ci_r,
                    "ci_interp_95": ci_i,
                    "ci_kind": (
                        "question-cluster bootstrap (2000 draws, seed 653); installs fixed"
                    ),
                }
                if space == "logp":
                    # committed per-seed D must reproduce from the same curves
                    assert abs(d_raw - ps["D_pooled_nonsource_nats"]) < 1e-6, (
                        ctx_key,
                        seed,
                        d_raw,
                        ps["D_pooled_nonsource_nats"],
                    )
            cells[f"{ctx_key}-s{seed}"] = cell

    def _classify(ci: list[float]) -> str:
        if ci[0] > 0:
            return "positive"
        if ci[1] < 0:
            return "negative"
        return "includes_zero"

    summary: dict = {}
    for space in spaces:
        recs = [c["spaces"][space] for c in cells.values()]
        summary[space] = {
            "n_cells": len(recs),
            "n_extrapolation_clamped": sum(r["extrapolation_clamped"] for r in recs),
            "mean_abs_shift": float(np.mean([abs(r["D_interp_minus_raw"]) for r in recs])),
            "max_abs_shift": float(np.max([abs(r["D_interp_minus_raw"]) for r in recs])),
            "n_sign_flips": sum(1 for r in recs if np.sign(r["D_interp"]) != np.sign(r["D_raw"])),
            "n_ci_class_changes": sum(
                1 for r in recs if _classify(r["ci_interp_95"]) != _classify(r["ci_raw_95"])
            ),
            "mean_D_raw": float(np.mean([r["D_raw"] for r in recs])),
            "mean_D_interp": float(np.mean([r["D_interp"] for r in recs])),
        }
    out = {"meta": _meta(), "cells": cells, "summary": summary}
    _atomic_json(ANALYSIS_DIR / "marker_matched_install_interp.json", out)
    for space, s in summary.items():
        logger.info(
            "marker-interp[%s]: mean D_raw=%.3f -> D_interp=%.3f (max |shift|=%.3f, "
            "%d sign flips, %d CI class changes, %d clamped)",
            space,
            s["mean_D_raw"],
            s["mean_D_interp"],
            s["max_abs_shift"],
            s["n_sign_flips"],
            s["n_ci_class_changes"],
            s["n_extrapolation_clamped"],
        )
    return 0


# ── item 3: emission-map degeneration counts + graded intensity ──────────────

_PANEL_FILE_RE = re.compile(r"(mk-[a-z]+-(?:con|po)-lr\w+-s\d+)_rung(\d+)_\d+\.json$")


def _repeat_4gram_frac(text: str) -> float:
    """Per-response word-level 4-gram repeat fraction (fu4 ``degeneracy_stats``
    definition, scripts/issue1090_fu4.py — applied per response, not per pool)."""
    toks = text.split()
    if len(toks) < 4:
        return 0.0
    grams = [tuple(toks[i : i + 4]) for i in range(len(toks) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def _marker_rung_roles(rcm: dict) -> dict[tuple[str, int], list[str]]:
    roles: dict[tuple[str, int], list[str]] = {}
    for run_id, curve in rcm["dose_curves"].items():
        for r in curve["rungs"]:
            roles[(run_id, r["step"])] = list(r.get("roles") or [])
    return roles


def cmd_degeneration() -> int:
    rcm = _read_json(ANALYSIS_DIR / "regime_contrast_marker.json")
    roles_by_rung = _marker_rung_roles(rcm)
    src_by_run = {rid: c["source_context"] for rid, c in rcm["dose_curves"].items()}

    logger.info("staging marker panel rollouts (~50 MB)")
    files = hub.stage_hub_prefix(DATA_REPO, f"{PREFIX}/marker/raw_completions/panel", PHASED_STAGE)
    panel_files = [p for p in files if _PANEL_FILE_RE.search(p.name)]
    if len(panel_files) != len(files):
        raise ValueError(
            f"{len(files) - len(panel_files)} staged panel files failed the filename parse"
        )

    from transformers import AutoTokenizer  # deferred: heavy import, this cmd only

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    per_cell: dict[str, dict] = {}
    for p in sorted(panel_files):
        m = _PANEL_FILE_RE.search(p.name)
        run_id, step = m.group(1), int(m.group(2))
        doc = _read_json(p)
        meta, responses = doc["meta"], doc["responses"]
        assert len(meta) == len(responses), (p.name, len(meta), len(responses))
        src_ctx = src_by_run[run_id]
        lens = [len(ids) for ids in tok(responses, add_special_tokens=False)["input_ids"]]
        stats = {
            "roles": roles_by_rung.get((run_id, step), ["unlabeled"]),
            "n_responses": len(responses),
            "counts": {
                "overall": _init_counts(),
                "source": _init_counts(),
                "nonsource": _init_counts(),
            },
        }
        for mrow, r, n_tok in zip(meta, responses, lens, strict=True):
            row = {
                "truncated": n_tok >= MARKER_MAX_NEW_TOKENS,
                "ends_with_marker": r.rstrip().endswith("※"),
                "multi_marker": r.count("※") >= 2,
                "degenerate": _repeat_4gram_frac(r) > DEGEN_MAX_REPEAT_FRAC,
            }
            scopes = ("overall", "source" if mrow["context_id"] == src_ctx else "nonsource")
            for scope in scopes:
                c = stats["counts"][scope]
                c["n"] += 1
                for key, hit in row.items():
                    c[key] += int(hit)
        for c in stats["counts"].values():
            if c["n"]:
                for key in ("truncated", "ends_with_marker", "multi_marker", "degenerate"):
                    c[f"{key}_rate"] = c[key] / c["n"]
        per_cell.setdefault(run_id, {})[str(step)] = stats

    n_cells = len(per_cell)
    out = {
        "meta": _meta(),
        "definitions": {
            "truncated": f"tokenized response length >= {MARKER_MAX_NEW_TOKENS} "
            "(C.MARKER_MAX_NEW_TOKENS)",
            "ends_with_marker": "rstripped response text ends with ※",
            "multi_marker": ">= 2 ※ occurrences (※-repeater signature, #397/#451)",
            "degenerate": f"per-response word 4-gram repeat fraction > {DEGEN_MAX_REPEAT_FRAC} "
            "(fu4 degeneracy_stats definition)",
        },
        "cells": per_cell,
    }
    _atomic_json(ANALYSIS_DIR / "emission_degeneration_counts.json", out)
    logger.info("degeneration: %d marker cells, %d panel rung files", n_cells, len(panel_files))

    _graded_intensity()
    return 0


def _init_counts() -> dict:
    return {"n": 0, "truncated": 0, "ends_with_marker": 0, "multi_marker": 0, "degenerate": 0}


def _load_local_judge_index(beh: str) -> tuple[dict, dict]:
    """(raws, idmaps) from the LOCAL judge_packed shard (issue1481_cjk_audit
    ``_load_judge_index`` parse, minus the HF download)."""
    raws: dict = {}
    idmaps: dict = {}
    shard = JUDGE_PACKED_DIR / f"judge_{beh}.shard00.jsonl"
    with open(shard) as f:
        for line in f:
            row = json.loads(line)
            if "judge_raw" in row["path"]:
                tag = row["path"].split("judge_raw_")[-1].removesuffix(".json")
                raws[tag] = row["content"]["all_scores"]
            elif "/idmap_" in row["path"]:
                tag = row["path"].split("/idmap_")[-1].removesuffix(".json")
                idmaps[tag] = row["content"]
    return raws, idmaps


def _dist_stats(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "frac_positive": float((a > THRESHOLD).mean()),
    }


def _graded_intensity() -> None:
    from scipy import stats as sps  # deferred: this cmd only

    vm = _read_json(ANALYSIS_DIR / "verdict_manifest.json")
    rcc = _read_json(ANALYSIS_DIR / "regime_contrast_content.json")
    rows = _verdict_arms(vm, rcc)
    by_arm = {r["arm_id"]: r for r in rows}
    idx_cache = {beh: _load_local_judge_index(beh) for beh in vm["content"]}

    def _panel_means(row: dict) -> dict[str, dict[str, float]]:
        """read-ctx -> {item_id: mean} for one verdict arm's panel cells."""
        raws, idmaps = idx_cache[row["behavior"]]
        instr = INSTRUMENT[row["behavior"]]
        arm = row["arm_id"]
        out: dict[str, dict[str, float]] = {}
        pfx = f"{instr}_pn-{arm}-"
        for key in raws:
            if not key.startswith(pfx):
                continue
            rctx = key[len(pfx) :]
            tag = f"pn-{arm}-{rctx}"
            out[rctx] = _item_means(raws[key], idmaps.get(tag, {}))
        if not out:
            raise ValueError(f"no panel judge records for verdict arm {arm}")
        return out

    def _tier1_means(row: dict) -> list[float] | None:
        stage_dir = TIER1_STAGE / row["arm_id"] / f"rate_checkpoint-{row['step']}"
        if not (stage_dir / "judge_raw.json").exists():
            return None  # reused-1434 (or tier1-recount not yet run)
        doc = _read_json(stage_dir / "completions.json")
        src = row["source_ctx"]
        items = [
            (f"{src}-trained-q{i:03d}-c{j}", q, doc["completions"][i][j])
            for i, q in enumerate(doc["questions"])
            for j in range(len(doc["completions"][i]))
        ]
        res = judge_result_from_save_raw(stage_dir / "judge_raw.json", items)
        return [s for s in res.scores.values() if s is not None]

    cells: dict[str, dict] = {}
    for beh, ctxs in vm["content"].items():
        for ctx_key, cell in ctxs.items():
            src_ctx = rcc["behavior_contexts"][beh][ctx_key]["source_ctx"]
            for seed, sv in cell["seeds"].items():
                cid = f"{beh}-{ctx_key}-s{seed}"
                rec: dict = {"con_arm": sv["con"]["arm_id"], "po_arm": sv["po"]["arm_id"]}
                pooled: dict[str, list[float]] = {}
                for reg in ("con", "po"):
                    row = by_arm[sv[reg]["arm_id"]]
                    panel = _panel_means(row)
                    vals = [
                        v
                        for rctx, means in panel.items()
                        if rctx != src_ctx
                        for v in means.values()
                    ]
                    pooled[reg] = vals
                    rec[f"{reg}_panel_nonsource"] = _dist_stats(vals)
                    t1 = _tier1_means(row)
                    rec[f"{reg}_source_tier1"] = _dist_stats(t1) if t1 else None
                ks = sps.ks_2samp(pooled["con"], pooled["po"])
                mw = sps.mannwhitneyu(pooled["con"], pooled["po"], alternative="two-sided")
                rec["panel_nonsource_tests"] = {
                    "ks_stat": float(ks.statistic),
                    "ks_p": float(ks.pvalue),
                    "mannwhitney_u": float(mw.statistic),
                    "mannwhitney_p": float(mw.pvalue),
                }
                t1c = _tier1_means(by_arm[sv["con"]["arm_id"]])
                t1p = _tier1_means(by_arm[sv["po"]["arm_id"]])
                if t1c and t1p:
                    ks2 = sps.ks_2samp(t1c, t1p)
                    mw2 = sps.mannwhitneyu(t1c, t1p, alternative="two-sided")
                    rec["source_tier1_tests"] = {
                        "ks_stat": float(ks2.statistic),
                        "ks_p": float(ks2.pvalue),
                        "mannwhitney_u": float(mw2.statistic),
                        "mannwhitney_p": float(mw2.pvalue),
                    }
                else:
                    rec["source_tier1_tests"] = None
                cells[cid] = rec

    out = {
        "meta": _meta(),
        "definitions": {
            "panel_nonsource": "per-item mean graded score over kept draws, pooled over "
            "the arm's non-source panel read contexts (the D read's distribution)",
            "source_tier1": "per-item mean graded score of the arm's SELECTION-rung "
            "Tier-1 pool (source context) — matched-rate vs graded-intensity read; "
            "null for reused-1434 arms (pool outside this bucket) or before "
            "tier1-recount staged the pools",
            "tests": "two-sided KS + Mann-Whitney, con vs po",
        },
        "cells": cells,
    }
    _atomic_json(ANALYSIS_DIR / "graded_intensity_comparison.json", out)
    logger.info("graded-intensity: %d cells", len(cells))


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["tier1-recount", "marker-interp", "degeneration"])
    args = p.parse_args(argv)
    return {
        "tier1-recount": cmd_tier1_recount,
        "marker-interp": cmd_marker_interp,
        "degeneration": cmd_degeneration,
    }[args.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
