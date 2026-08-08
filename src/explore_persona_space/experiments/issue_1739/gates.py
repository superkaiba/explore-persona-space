"""Phase-0 executable gates for issue #1739 (round A).

Gate 0 (store sha-pin probe), the r_B bank probe, and Gate 3 (staged-layout
probe through the consumer loader) are EXECUTABLE this round. Gates 1-2
(yield pilot / spread floor) need generation and are round-B stubs.

All Hub listings are SERVER-SIDE SCOPED (``path_in_repo=...``) — the data repo
holds ~1M files, so a bare listing / snapshot_download wedges (gotchas.md #833).
"""

from __future__ import annotations

import logging
import statistics
from pathlib import Path

from explore_persona_space.experiments.issue_1739 import store_io
from explore_persona_space.experiments.issue_1739.constants import (
    HF_DATA_REPO,
    HIDDEN_DIM,
    RB_N_TRAITS,
    RB_PREFIX,
    RB_REVISION,
    STORE_PREFIX,
    STORE_REVISION,
    SUMMARY_KINDS,
)
from explore_persona_space.orchestrate import hub

logger = logging.getLogger(__name__)


def _scoped_tree(prefix: str, revision: str) -> list:
    """Materialized scoped ``list_repo_tree`` entries (path + size).

    Materialize INSIDE the retry wrapper — Hub list APIs are lazy generators,
    so the HTTP error raises at iteration time (gotchas.md #779 n50k).
    """
    from huggingface_hub import list_repo_tree

    return hub.retry_transient(
        lambda: list(
            list_repo_tree(
                HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=prefix.rstrip("/"),
                revision=revision,
                recursive=True,
            )
        ),
        what=f"list_repo_tree {HF_DATA_REPO}@{revision}:{prefix}",
    )


def gate0_store_pin_probe(*, revision: str = STORE_REVISION) -> dict:
    """Gate 0: sha-pin probe of the #1092 summary store.

    Scoped tree listing at the pinned revision; asserts >=1 file per summary
    kind stem (prefix_end / context_end / t1). Reports realized cell dirs +
    row_index sidecar count so downstream rounds ground the layout on facts.
    """
    entries = _scoped_tree(STORE_PREFIX, revision)
    files = [e.path for e in entries if getattr(e, "size", None) is not None]
    root = STORE_PREFIX.rstrip("/") + "/"
    per_kind: dict[str, int] = {}
    for kind in SUMMARY_KINDS:
        per_kind[kind] = sum(1 for f in files if f.rsplit("/", 1)[-1].startswith(f"{kind}_L"))
    missing = [k for k, n in per_kind.items() if n < 1]
    if missing:
        raise AssertionError(
            f"gate0 FAIL: no files for kind stem(s) {missing} under "
            f"{HF_DATA_REPO}@{revision}:{STORE_PREFIX} (per-kind counts {per_kind})"
        )
    cell_dirs = sorted(
        {
            f[len(root) :].split("/", 1)[0]
            for f in files
            if f.startswith(root) and "/" in f[len(root) :]
        }
    )
    n_row_index = sum(1 for f in files if f.rsplit("/", 1)[-1].startswith("row_index"))
    report = {
        "gate": "gate0_store_pin_probe",
        "repo": HF_DATA_REPO,
        "revision": revision,
        "prefix": STORE_PREFIX,
        "n_files": len(files),
        "per_kind_counts": per_kind,
        "cell_dirs": cell_dirs,
        "n_row_index_files": n_row_index,
        "verdict": "PASS",
    }
    logger.info("[gate0] PASS: %s", report)
    return report


def rb_bank_probe(*, revision: str = RB_REVISION) -> dict:
    """r_B probe: list trait ``.pt`` files at the pinned #779 revision."""
    entries = _scoped_tree(RB_PREFIX, revision)
    pt_files = sorted(
        e.path for e in entries if getattr(e, "size", None) is not None and e.path.endswith(".pt")
    )
    if not pt_files:
        raise AssertionError(
            f"rb probe FAIL: no r_B .pt files under {HF_DATA_REPO}@{revision}:{RB_PREFIX}"
        )
    if len(pt_files) != RB_N_TRAITS:
        logger.warning(
            "[rb-probe] found %d trait files (pinned expectation %d): %s",
            len(pt_files),
            RB_N_TRAITS,
            pt_files,
        )
    report = {
        "gate": "rb_bank_probe",
        "repo": HF_DATA_REPO,
        "revision": revision,
        "prefix": RB_PREFIX,
        "trait_files": pt_files,
        "n_trait_files": len(pt_files),
        "verdict": "PASS",
    }
    logger.info("[rb-probe] PASS: %s", report)
    return report


def gate3_staged_layout_probe(local_dir: Path | str, *, revision: str = STORE_REVISION) -> dict:
    """Gate 3: staged-layout probe through the REAL U-store mapping.

    Stages a 1-shard-per-kind slice of the production mapping — the
    ``U_STORE_CELL`` dir's canonical-kind shards FLATTENED + the corpus
    ``manifest.jsonl`` as row metadata (``store_io.stage_u_store``) — then
    opens it through the PRODUCTION consumer path the fits CLI runs
    (``store_io.load_summaries`` -> ``store_io.fit_pool_mask``) with shape
    asserts (artifact-reuse check (h)(iv): the staged tree must open via the
    consumer's own reader, in the layout production consumes).
    """
    n_probe = 16
    layers = (0,)
    local_dir = Path(local_dir)
    staged_root = store_io.stage_u_store(
        local_dir, SUMMARY_KINDS, layers, revision=revision, max_shards_per_kind=1
    )
    arrays, meta = store_io.load_summaries(staged_root, SUMMARY_KINDS, layers, n_rows=n_probe)
    shapes: dict[str, list[int]] = {}
    for (kind, layer), arr in arrays.items():
        if arr.ndim != 2 or arr.shape[1] != HIDDEN_DIM:
            raise AssertionError(
                f"gate3 FAIL: {kind}_L{layer:02d} shape {arr.shape} != (n, {HIDDEN_DIM})"
            )
        shapes[f"{kind}_L{layer:02d}"] = list(arr.shape)
    if not meta or not all(isinstance(r, dict) for r in meta):
        raise AssertionError("gate3 FAIL: corpus manifest rows unparseable via consumer loader")
    mask = store_io.fit_pool_mask(meta)  # fails loud on zero fit rows
    manifest_rows = store_io._iter_jsonl(local_dir / "manifest.jsonl")
    n_with_eval_key = sum(1 for r in manifest_rows if "is_eval_only" in r)
    report = {
        "gate": "gate3_staged_layout_probe",
        "repo": HF_DATA_REPO,
        "revision": revision,
        "prefix": STORE_PREFIX,
        "staged_root": str(staged_root),
        "kinds": list(SUMMARY_KINDS),
        "n_probe_rows": int(n_probe),
        "summary_shapes": shapes,
        "manifest_n_rows": len(manifest_rows),
        "manifest_first_row_keys": sorted(manifest_rows[0].keys()),
        "n_rows_with_is_eval_only_key": n_with_eval_key,
        "fit_pool_kept_of_probe": int(mask.sum()),
        "verdict": "PASS",
    }
    logger.info("[gate3] PASS: %s", report)
    return report


GATE1_N_PILOT = 300  # pilot contexts per behavior (plan v3 Gate 1)
# Plan-pinned yield floor: §8 pre-registers "<20% kept rollouts" as the yield
# risk trigger, so the gate FAILs below a 0.2 keep rate (the round-1 0.5 was
# an undocumented tightening — round-1 review Minor).
GATE1_KEEP_RATE_FLOOR = 0.2
GATE2_SD_FLOOR = 10.0  # inter-context SD floor on the 0-100 scale (plan v3 Gate 2)
GATE2_BOTTOM_BIN_MAX_FRAC = 0.80  # >= 80% of contexts in the bottom bin = floor-collapsed
GATE2_BOTTOM_BIN_EDGE = 10.0  # the bottom histogram bin is [0, 10)
SPREAD_TOP_BIN_LOWER = 90.0  # two-sided gate ceiling bin: score >= 90

# Machine-readable convention labels for the BINDING spread instrument
# (#1739 round 22 consolidation). Plan-text divergence, recorded rather than
# hidden: plan v16 §7 binds ``SD = np.std(mean_scores_per_context)`` — numpy's
# DEFAULT ddof=0 — while the committed trait-side verdicts
# (issue1739_k1_floor.rung_table, gate2_spread_floor above, the
# eval_results/issue_1739/new_arm_round/k1_verdicts.json values 0.895 / 12.07 /
# 26.33) were all computed with SAMPLE SD (ddof=1). ddof=1 is deliberately
# chosen as the binding estimator so compliance-DV spread reads stay
# instrument-matched to those trait verdicts; the ddof=0 value is reported
# alongside as ``sd_pop`` so the plan's literal convention is never hidden.
SPREAD_SD_DDOF = 1
SPREAD_SD_CONVENTION_NOTE = (
    "plan v16 §7 binds SD = np.std(mean_scores_per_context) (numpy default "
    "ddof=0); the binding cross-round convention is SAMPLE SD ddof=1, chosen "
    "for comparability with the committed trait-DV verdicts "
    "(k1_floor.rung_table / gate2_spread_floor, ddof=1); ddof=0 is reported "
    "alongside as sd_pop"
)


def per_context_means(per_item_scores: dict[str, list[float]]) -> dict[str, float]:
    """Group per-rollout judge draws into per-CONTEXT mean scores.

    BINDING unit convention (plan §7 primary read; matches
    ``dv_build.build_labeling_dv``'s two-level mean): per rollout item take
    the mean over kept draws; per context take the mean over items with >= 1
    kept draw. Items whose draws all dropped are EXCLUDED, never coerced
    (llm-judging.md rule 9); a context with zero kept items is excluded the
    same way. Item ids are ``{context_id}_k{NN}`` and are inverted via the
    canonical ``dv_build.parse_item_id`` (raises on a malformed id — a parse
    failure must never silently collapse the grouping unit).
    """
    from explore_persona_space.experiments.issue_1739.dv_build import parse_item_id

    by_ctx: dict[str, list[float]] = {}
    for item_id, draws in per_item_scores.items():
        kept = [float(d) for d in draws if d is not None]
        if not kept:
            continue
        context_id, _k = parse_item_id(item_id)
        by_ctx.setdefault(context_id, []).append(sum(kept) / len(kept))
    return {ctx: sum(vals) / len(vals) for ctx, vals in sorted(by_ctx.items())}


def score_spread(
    scores: list[float],
    *,
    unit: str,
    sd_min: float = GATE2_SD_FLOOR,
    bin_max_frac: float = GATE2_BOTTOM_BIN_MAX_FRAC,
    bottom_bin_upper: float = GATE2_BOTTOM_BIN_EDGE,
    top_bin_lower: float = SPREAD_TOP_BIN_LOWER,
) -> dict:
    """Two-sided spread-gate report over one set of DV values (canonical copy).

    ``spread_gate_pass = (sd >= sd_min) AND (bottom_frac < bin_max_frac) AND
    (top_frac < bin_max_frac)``. The BINDING estimators are SAMPLE SD
    (``ddof=1``) and STRICT bottom-bin membership (``score <
    bottom_bin_upper``) — exactly as ``gate2_spread_floor`` /
    ``scripts/issue1739_k1_floor.rung_table`` computed the committed trait
    verdicts. ``sd_pop`` (``ddof=0``, the plan's literal ``np.std``) and
    ``bottom_frac_inclusive`` (``<=``) are reported alongside so no
    convention is hidden (see ``SPREAD_SD_CONVENTION_NOTE`` for the recorded
    plan-text divergence). ``unit`` labels what ONE value is (e.g.
    ``"per_context"`` for the plan-§7 primary read over per-context means,
    ``"per_item"`` for the per-rollout secondary) — a bare ``sd`` whose unit
    a reader has to infer is the round-22 defect class.

    This is the single canonical implementation (#1739 round 22): the four
    prior copies (``issue1739_pilot_judge._score_spread``,
    ``issue1739_k1_floor.rung_table``, ``gate2_spread_floor``,
    ``issue1739_compliance_full._score_spread``) diverged on unit + estimator;
    new spread reads call this helper instead of re-deriving the arithmetic.
    """
    n = len(scores)
    if n == 0:
        return {
            "unit": unit,
            "spread_unit": unit,
            "n_scores": 0,
            "sd": None,
            "sd_ddof": SPREAD_SD_DDOF,
            "sd_pop": None,
            "mean": None,
            "bottom_frac": None,
            "bottom_frac_inclusive": None,
            "bottom_bin": f"strict < {bottom_bin_upper:g}",
            "top_frac": None,
            "top_bin": f">= {top_bin_lower:g}",
            "ceiling_frac": None,
            "spread_gate_pass": False,
            "sd_convention_note": SPREAD_SD_CONVENTION_NOTE,
            "reason": "no kept values",
        }
    sd = statistics.stdev(scores) if n > 1 else 0.0
    sd_pop = statistics.pstdev(scores) if n > 1 else 0.0
    mean = statistics.fmean(scores)
    bottom = sum(1 for s in scores if s < bottom_bin_upper) / n
    bottom_incl = sum(1 for s in scores if s <= bottom_bin_upper) / n
    top = sum(1 for s in scores if s >= top_bin_lower) / n
    sd_ok = sd >= sd_min
    bottom_ok = bottom < bin_max_frac
    top_ok = top < bin_max_frac
    fails = [
        name
        for name, ok in (("sd", sd_ok), ("bottom_frac", bottom_ok), ("top_frac", top_ok))
        if not ok
    ]
    return {
        "unit": unit,
        "spread_unit": unit,
        "n_scores": n,
        "sd": sd,
        "sd_ddof": SPREAD_SD_DDOF,
        "sd_pop": sd_pop,
        "mean": mean,
        "bottom_frac": bottom,
        "bottom_frac_inclusive": bottom_incl,
        "bottom_bin": f"strict < {bottom_bin_upper:g}",
        "top_frac": top,
        "top_bin": f">= {top_bin_lower:g}",
        "ceiling_frac": top,
        "sd_ok": sd_ok,
        "bottom_ok": bottom_ok,
        "top_ok": top_ok,
        "spread_gate_pass": bool(sd_ok and bottom_ok and top_ok),
        "failed_criteria": fails,
        "sd_convention_note": SPREAD_SD_CONVENTION_NOTE,
    }


def gate1_yield_report(dv_rows: list[dict], *, behavior: str, n_pilot: int = GATE1_N_PILOT) -> dict:
    """Gate 1: yield-pilot report — expression histogram + keep rate.

    ``dv_rows`` are ``dv_build.build_labeling_dv`` rows for the pilot slice.
    Keep rate = fraction of pilot contexts with a non-None DV (>= 1 kept
    judged rollout). The 10-bin expression histogram over [0, 100] is the
    human-legible yield read; verdict PASS iff keep rate >= the floor.
    """
    kept = [r["dv"] for r in dv_rows if r.get("dv") is not None]
    n_rows = len(dv_rows)
    keep_rate = (len(kept) / n_rows) if n_rows else 0.0
    bins = [0] * 10
    for value in kept:
        bins[min(9, max(0, int(value // 10)))] += 1
    n_transport = sum(int(r.get("n_transport_lost_draws") or 0) for r in dv_rows)
    n_content_dropped = sum(int(r.get("n_rollouts_content_dropped") or 0) for r in dv_rows)
    verdict = "PASS" if (n_rows >= 1 and keep_rate >= GATE1_KEEP_RATE_FLOOR) else "FAIL"
    report = {
        "gate": "gate1_yield_pilot",
        "behavior": behavior,
        "n_pilot_target": n_pilot,
        "n_contexts": n_rows,
        "n_contexts_with_dv": len(kept),
        "keep_rate": keep_rate,
        "keep_rate_floor": GATE1_KEEP_RATE_FLOOR,
        "expression_histogram": {f"{10 * i}-{10 * (i + 1)}": n for i, n in enumerate(bins)},
        "n_rollouts_content_dropped": n_content_dropped,
        "n_transport_lost_draws": n_transport,
        "verdict": verdict,
    }
    logger.info("[gate1] %s: %s", behavior, report)
    return report


def gate2_spread_floor(dv_rows: list[dict], *, behavior: str) -> dict:
    """Gate 2: spread floor — inter-context SD >= 10 AND < 80% bottom bin.

    A behavior whose graded DV collapses (SD below floor, or >= 80% of
    contexts in the [0, 10) bottom bin) FAILs and sets
    ``tf_margin_fallback: True`` — the TF fixed +/- pool margin (dv_build)
    becomes the load-bearing continuous companion there (llm-judging.md
    rule 19; a floor-collapsed graded read is presumed uninformative).
    """
    import numpy as np

    kept = np.array([r["dv"] for r in dv_rows if r.get("dv") is not None], dtype=np.float64)
    if kept.size < 2:
        report = {
            "gate": "gate2_spread_floor",
            "behavior": behavior,
            "n_contexts_with_dv": int(kept.size),
            "verdict": "FAIL",
            "reason": "fewer than 2 contexts with a DV — spread undefined",
            "tf_margin_fallback": True,
        }
        logger.info("[gate2] %s: %s", behavior, report)
        return report
    sd = float(kept.std(ddof=1))
    bottom_frac = float((kept < GATE2_BOTTOM_BIN_EDGE).mean())
    sd_ok = sd >= GATE2_SD_FLOOR
    bin_ok = bottom_frac < GATE2_BOTTOM_BIN_MAX_FRAC
    verdict = "PASS" if (sd_ok and bin_ok) else "FAIL"
    report = {
        "gate": "gate2_spread_floor",
        "behavior": behavior,
        "n_contexts_with_dv": int(kept.size),
        "inter_context_sd": sd,
        "sd_floor": GATE2_SD_FLOOR,
        "sd_ok": sd_ok,
        "bottom_bin_frac": bottom_frac,
        "bottom_bin_edge": GATE2_BOTTOM_BIN_EDGE,
        "bottom_bin_max_frac": GATE2_BOTTOM_BIN_MAX_FRAC,
        "bottom_bin_ok": bin_ok,
        "verdict": verdict,
        "tf_margin_fallback": verdict == "FAIL",
    }
    logger.info("[gate2] %s: %s", behavior, report)
    return report


def run_gate1_pilot(
    behavior: str,
    *,
    out_root: Path | str,
    staged_dir: Path | str,
    n_pilot: int = GATE1_N_PILOT,
    seed: int = 0,
    stream_cap: int | None = None,
    generate_fn=None,
    judge_fn=None,
    tokenizer=None,
    cache_dir: Path | str | None = None,
) -> dict:
    """Gate 1 driver: compose staging -> generation -> judging on the pilot slice.

    Stages the behavior's TRAIN corpus (fingerprint-cached, idempotent), takes
    the FIRST ``n_pilot`` contexts (deterministic), generates K rollouts,
    judges them graded (hallucination runs the three-way protocol), builds the
    per-context DV, and writes the Gate 1 + Gate 2 reports. ``generate_fn`` /
    ``judge_fn`` / ``tokenizer`` are the GPU/API seams (tests inject
    signature-mirroring fakes; production leaves them None).

    ``judge_fn(items, eval_prompt, *, cache_dir, save_raw)`` must return a
    ``graded_judge.JudgeResult``-shaped object (``scores`` /
    ``per_item_transport_losses`` attributes).
    """
    import json as _json

    from explore_persona_space.experiments.issue_1739 import dv_build, generation, judging
    from explore_persona_space.experiments.issue_1739.corpus_staging import (
        read_jsonl,
        stage_corpus,
        staged_context_path,
    )

    out_root = Path(out_root)
    staged_dir = Path(staged_dir)
    contexts_path = staged_context_path(staged_dir, behavior, "train", "train")
    if not contexts_path.exists():
        stage_corpus(behavior, "train", None, seed, out_dir=staged_dir, stream_cap=stream_cap)
    contexts = read_jsonl(contexts_path)[:n_pilot]
    if not contexts:
        raise RuntimeError(f"gate1 pilot: zero staged contexts for {behavior}")

    gen_root = out_root / "gate1" / "raw_completions"
    generation.generate_labeling(
        contexts,
        out_root=gen_root,
        behavior=behavior,
        seed=seed,
        generate_fn=generate_fn,
        tokenizer=tokenizer,
    )
    rollout_paths = sorted(
        p for p in (gen_root / "labeling" / behavior).glob("*.json") if not p.name.startswith("_")
    )
    payloads = [_json.loads(p.read_text()) for p in rollout_paths]

    judge_dir = out_root / "gate1" / "judge" / behavior
    judge_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else judge_dir / "judge_cache"
    dispatch = judge_fn if judge_fn is not None else judging.judge_items_graded

    if behavior == "hallucination":
        correct_map, judge_items = judging.split_hallucination_items(payloads)
        result = dispatch(
            judge_items,
            judging.HALLU_ABSTAIN_RUBRIC,
            cache_dir=cache_dir,
            save_raw=judge_dir / "judge_raw_abstain.json",
        )
        three_way = {
            item_id: judging.three_way_classify(is_correct, result.scores.get(item_id))
            for item_id, is_correct in correct_map.items()
        }
        # Pilot-yield DV convention: decided rollouts carry 0 (correct /
        # abstained) vs 100 (fabricated) so the same histogram / keep-rate
        # machinery applies; unjudged stays dropped (None).
        scores: dict[str, float | None] = {
            item_id: (None if label == "unjudged" else (100.0 if label == "fabricated" else 0.0))
            for item_id, label in three_way.items()
        }
        transport = dict(getattr(result, "per_item_transport_losses", {}) or {})
    else:
        eval_prompt = judging.load_trait_rubric(behavior)
        items = [
            (
                judging.rollout_item_id(p["context_id"], int(p["rollout_k"])),
                p["query"],
                p["completion"],
            )
            for p in payloads
        ]
        result = dispatch(
            items,
            eval_prompt,
            cache_dir=cache_dir,
            save_raw=judge_dir / "judge_raw_trait.json",
        )
        scores = dict(result.scores)
        transport = dict(getattr(result, "per_item_transport_losses", {}) or {})

    dv_rows = dv_build.build_labeling_dv(
        scores,
        per_item_transport_losses=transport,
        contexts_meta={c["context_id"]: c for c in contexts},
    )
    # Gate-2 DV convention (round-C1 design-note resolution): the 0/100
    # label mapping above exists ONLY so gate 1's keep-rate/histogram
    # machinery is shared; its per-context mean is a label FRACTION in
    # disguise, so gate 2's spread floor reads the fabrication-fraction rows
    # from build_three_way_dv (x100 onto the 0-100 scale gate 2 is
    # calibrated for) instead of the 0/100-mapped labeling rows.
    if behavior == "hallucination":
        gate2_rows = [
            dict(r, dv=(None if r["dv"] is None else 100.0 * r["dv"]))
            for r in dv_build.build_three_way_dv(three_way)
        ]
        gate2_dv = "fabrication_fraction_x100"
    else:
        gate2_rows = dv_rows
        gate2_dv = "graded_judge_mean"
    report = {
        "gate1": gate1_yield_report(dv_rows, behavior=behavior, n_pilot=n_pilot),
        "gate2": gate2_spread_floor(gate2_rows, behavior=behavior),
        "gate2_dv": gate2_dv,
        "n_rollout_files": len(rollout_paths),
    }
    report_path = out_root / "gate1" / f"{behavior}_pilot_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_json.dumps(report, indent=2))
    logger.info("[gate1-pilot] %s: report -> %s", behavior, report_path)
    return report
