"""#1090 Step 9a-ter zero-GPU free analysis: two judge-API reads over STORED completions.

P1 — judged re-read of the c1 formatting control. The production install/dose
reads used the STRUCTURAL predicate (>=80% list-lines, ``datagen._is_list_formatted``)
as the c1 rate DV; the judged read existed only as the N=30 spotcheck
(agreement 0.633). Here the standard graded judge (project judge, the SAME
formatting rubric ``BEHAVIORS["formatting"].judge_rubric`` that
``issue1090_run._judge_rate`` / ``_formatting_spotcheck`` use) is run over ALL
stored c1 completions — the two Tier-2 files (trained/base, 30 q x 10 c each)
plus the 8 Tier-1 dose-ladder rung dumps (30 q x 5 c each, trained only, staged
from HF ``issue1090_pvdatagen/raw_completions/rate/c1-formatting-claude/``).
Outputs: judged dose curve + re-derived ``select_dose_checkpoint`` pick, judged
Tier-2 rates/delta, and judged-vs-structural agreement over the full set.

P4 — c3 sycophancy Tier-2 parse-drop closure. The production Tier-2 judging
dropped 473/1000 base and 307/1000 trained draws as parse errors
(``_legacy_error_dict("parse_error")`` — the raw judge response text is NOT
persisted anywhere: the JudgeCache entry and ``judge_raw.json`` both store only
the parsed-or-error dict, verified mechanically below). So closure REFRESHES
only the dropped draws with fresh judge calls (same rubric/judge; max_tokens
raised 64->300 for the refresh only — a sampling knob deliberately excluded
from the rubric identity, see ``batch_judge.rubric_fingerprint``; the 64-token
cap truncating reason-first responses is the leading drop hypothesis).

ANALYSIS-ONLY: no training, no model generation, no new data — judge API over
stored completions + aggregation. Both phases checkpoint per unit/group
(save_raw + rubric-keyed judge cache) and resume pure-read from a complete
``save_raw``. All Anthropic calls route through ``eval.graded_judge`` ->
``eval.batch_judge`` -> ``eval.judge_dispatch`` (the sanctioned client).

Usage (from the issue-1090 worktree root):
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
      uv run python scripts/issue1090_free_analysis.py --smoke   # tiny live slice first
      uv run python scripts/issue1090_free_analysis.py           # full run
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # credentials + shared-VM thread caps BEFORE any heavy import

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.artifacts.datagen import _STRUCTURAL_PREDICATES  # noqa: E402
from explore_persona_space.artifacts.recipe import (  # noqa: E402
    JUDGED_RATE_BAND,
    select_dose_checkpoint,
)
from explore_persona_space.eval.graded_judge import (  # noqa: E402
    JudgeResult,
    judge_graded,
    judge_result_from_save_raw,
)

logger = logging.getLogger("issue1090_free_analysis")

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO_ROOT / "data" / "issue_1090" / "run"
OUT_ROOT = REPO_ROOT / "eval_results" / "issue_1090" / "free_analysis"

DATA_REPO = "superkaiba1/explore-persona-space-data"
DATA_PREFIX = "issue1090_pvdatagen"
SRC_CTX = "persona_software_engineer"

C1_LADDER_STEPS = (2, 4, 6, 8, 10, 12, 14, 15)  # the 8 checkpoint rungs (save_steps=2, +final 15)
N_DRAWS = 5  # graded multi-draw count (llm-judging rule 4; matches TIER2_JUDGE_DRAWS)
REFRESH_MAX_TOKENS = 300  # refresh-only deviation from judge_graded's 64 default (see docstring)

# Production reference numbers (eval_results/issue_1090/install/*.json) the
# re-reads are compared against.
C1_STRUCTURAL_DELTA = -0.013333333333333308
C1_SELECTED_STEP = 2
C3_PARSED_DELTA = 0.26635574556289676
C3_WORSTCASE_FLOOR_DELTA = 0.100  # (91/200) - ((37+34)/200): all-dropped worst-case assignment


# ── small shared helpers ─────────────────────────────────────────────────────


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% score interval for k/n (mirrors issue1090_run._wilson)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _newcombe_delta_ci(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Newcombe hybrid-score 95% CI for p1 - p2 (from the two Wilson intervals)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = _wilson(k1, n1)
    l2, u2 = _wilson(k2, n2)
    d = p1 - p2
    return (
        d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
        d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2),
    )


def _git_short_sha() -> str:
    """Current worktree HEAD short sha (reproducibility metadata)."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _repro_meta(extra: dict | None = None) -> dict:
    import anthropic

    meta = {
        "git_commit": _git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "anthropic_version": anthropic.__version__,
        "python": sys.version.split()[0],
        "script": "scripts/issue1090_free_analysis.py",
    }
    if extra:
        meta.update(extra)
    return meta


def _atomic_write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _flat_items(
    tag: str, questions: list[str], completions: list[list[str]]
) -> list[tuple[str, str, str]]:
    """The (item_id, question, completion) flat list, id scheme mirroring
    ``issue1090_run._judge_rate`` (no ``__`` — the judge custom_id delimiter)."""
    return [
        (f"{tag}-q{i:03d}-c{j}", q, comp)
        for i, q in enumerate(questions)
        for j, comp in enumerate(completions[i])
    ]


def _judged_reduce(flat: list[tuple[str, str, str]], result: JudgeResult, threshold: int) -> dict:
    """Rate reduction mirroring ``issue1090_run._judge_rate``'s judged branch:
    per-item mean over kept draws; None (all-draws-dropped) items dropped from
    the denominator, never coerced; positive = mean > threshold."""
    n_dropped = n_pos = n_scored = 0
    for iid, _q, _c in flat:
        score = result.scores.get(iid)
        if score is None:
            n_dropped += 1
            continue
        n_scored += 1
        if score > threshold:
            n_pos += 1
    if n_scored == 0:
        raise ValueError("every completion was judge-dropped — a judging outage")
    lo, hi = _wilson(n_pos, n_scored)
    return {
        "rate": n_pos / n_scored,
        "k": n_pos,
        "n": n_scored,
        "n_dropped": n_dropped,
        "n_total_draws": result.n_total_draws,
        "n_dropped_draws": result.n_dropped_draws,
        "wilson95": [lo, hi],
        "mode": "judged",
    }


def _judge_unit(
    flat: list[tuple[str, str, str]],
    *,
    behavior_name: str,
    n_draws: int,
    unit_dir: Path,
    max_tokens: int = 64,
) -> JudgeResult:
    """Judge one unit with a complete-save_raw pure-read resume.

    A complete ``judge_raw.json`` (all_scores count == len(flat) * n_draws) is
    reduced with ZERO API calls; a partial one falls through to ``judge_graded``,
    whose rubric-keyed cache skips already-judged items (their n_draws then
    collapse to the one cached score — recorded upstream via cache_stats).
    """
    behavior = BEHAVIORS[behavior_name]
    unit_dir.mkdir(parents=True, exist_ok=True)
    save_raw = unit_dir / "judge_raw.json"
    expected = len(flat) * n_draws
    if save_raw.exists():
        raw = _read_json(save_raw)
        if len(raw.get("all_scores", {})) == expected:
            logger.info("[resume] %s complete (%d draws) — pure read", unit_dir.name, expected)
            return judge_result_from_save_raw(save_raw, flat)
        logger.warning(
            "[resume] %s PARTIAL (%d/%d draws) — re-dispatching via cache",
            unit_dir.name,
            len(raw.get("all_scores", {})),
            expected,
        )
    return judge_graded(
        flat,
        behavior.judge_rubric,
        n_draws=n_draws,
        cache_dir=unit_dir / "cache",
        save_raw=save_raw,
        judge_model=behavior.judge_model,
        max_tokens=max_tokens,
    )


# ── P1: judged re-read of the c1 formatting control ─────────────────────────


def _stage_c1_rung(step: int, stage_root: Path) -> Path:
    """Stage one c1 ladder rung dump from HF (retry + linear backoff, #658)."""
    from huggingface_hub import hf_hub_download

    rel = (
        f"{DATA_PREFIX}/raw_completions/rate/c1-formatting-claude/"
        f"rate_checkpoint-{step}/completions__trained__{SRC_CTX}.json"
    )
    dest = stage_root / f"rate_checkpoint-{step}.json"
    if dest.exists():
        return dest
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            got = hf_hub_download(
                DATA_REPO, rel, repo_type="dataset", local_dir=stage_root / "hf_dl"
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(got, dest)
            return dest
        except Exception as e:
            last_err = e
            wait = 20 * (attempt + 1)
            logger.warning("HF stage %s failed (%s) — retry in %ds", rel, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"failed to stage {rel} after 4 attempts") from last_err


def _p1_units(smoke: bool, stage_root: Path) -> list[tuple[str, Path]]:
    """(unit_name, payload_path) list: 8 ladder rungs + 2 Tier-2 states."""
    tier2_dir = RUN_ROOT / "tier2" / "c1-formatting-claude"
    units: list[tuple[str, Path]] = []
    steps = C1_LADDER_STEPS[:1] if smoke else C1_LADDER_STEPS
    for step in steps:
        units.append((f"rung{step:02d}", _stage_c1_rung(step, stage_root)))
    states = ("trained",) if smoke else ("trained", "base")
    for state in states:
        units.append((f"tier2-{state}", tier2_dir / f"completions__{state}__{SRC_CTX}.json"))
    return units


def phase_p1(out_root: Path, work_root: Path, *, smoke: bool) -> dict:
    """Judged re-read of every stored c1 completion; agreement + dose + install."""
    behavior = BEHAVIORS["formatting"]
    predicate = _STRUCTURAL_PREDICATES["formatting"]
    n_draws = 1 if smoke else N_DRAWS
    stage_root = work_root / "c1_stage"
    unit_results: dict[str, dict] = {}
    pooled = {"agree": 0, "scored": 0}

    for unit, payload_path in _p1_units(smoke, stage_root):
        payload = _read_json(payload_path)
        assert set(payload) >= {"questions", "completions"}, sorted(payload)
        questions, completions = payload["questions"], payload["completions"]
        if smoke:
            questions, completions = questions[:1], [completions[0][:5]]
        flat = _flat_items(f"c1re-{unit}", questions, completions)
        t0 = time.time()
        result = _judge_unit(
            flat, behavior_name="formatting", n_draws=n_draws, unit_dir=work_root / "c1" / unit
        )
        wall = time.time() - t0
        read = _judged_reduce(flat, result, behavior.threshold)
        # Structural read + per-item agreement over the SAME completions.
        struct_flags = {iid: bool(predicate(c)) for iid, _q, c in flat}
        n_struct_pos = sum(struct_flags.values())
        agree = scored = 0
        per_item = []
        for iid, _q, _c in flat:
            score = result.scores.get(iid)
            judged_pos = None if score is None else (score > behavior.threshold)
            if judged_pos is not None:
                scored += 1
                agree += int(struct_flags[iid] == judged_pos)
            per_item.append(
                {
                    "item_id": iid,
                    "structural": struct_flags[iid],
                    "judged_mean": score,
                    "judged_pos": judged_pos,
                    "kept_draws": result.per_item_draw_counts.get(iid, 0),
                }
            )
        pooled["agree"] += agree
        pooled["scored"] += scored
        slo, shi = _wilson(n_struct_pos, len(flat))
        unit_results[unit] = {
            "n_items": len(flat),
            "judged": read,
            "structural": {
                "rate": n_struct_pos / len(flat),
                "k": n_struct_pos,
                "n": len(flat),
                "wilson95": [slo, shi],
                "mode": "structural",
            },
            "agreement": {"n_scored": scored, "n_agree": agree, "rate": agree / scored},
            "judge_wall_s": round(wall, 1),
            "per_item": per_item,
        }
        logger.info(
            "[p1] %s judged=%.3f structural=%.3f agree=%.3f (%d items, %.0fs)",
            unit,
            read["rate"],
            n_struct_pos / len(flat),
            agree / scored,
            len(flat),
            wall,
        )
        _atomic_write_json(out_root / "p1_units" / f"{unit}.json", unit_results[unit])

    summary: dict = {
        "meta": _repro_meta(
            {
                "phase": "p1_c1_judged_reread",
                "smoke": smoke,
                "n_draws": n_draws,
                "judge_model": behavior.judge_model,
                "threshold": behavior.threshold,
                "band": list(JUDGED_RATE_BAND),
            }
        ),
        "units": {
            u: {k: v for k, v in r.items() if k != "per_item"} for u, r in unit_results.items()
        },
        "pooled_agreement": {
            "n_scored": pooled["scored"],
            "n_agree": pooled["agree"],
            "rate": pooled["agree"] / max(pooled["scored"], 1),
            "spotcheck_reference": {"agreement": 0.633, "n": 30},
        },
    }
    # Judged dose curve + re-derived selection (full run only — smoke has 1 rung).
    rung_rates = {
        int(u.removeprefix("rung")): r["judged"]["rate"]
        for u, r in unit_results.items()
        if u.startswith("rung")
    }
    if rung_rates:
        structural_curve = {
            int(u.removeprefix("rung")): r["structural"]["rate"]
            for u, r in unit_results.items()
            if u.startswith("rung")
        }
        summary["judged_dose_curve"] = {str(k): v for k, v in sorted(rung_rates.items())}
        summary["structural_dose_curve_reread"] = {
            str(k): v for k, v in sorted(structural_curve.items())
        }
        if not smoke:
            import dataclasses

            sel = select_dose_checkpoint(rung_rates, band=JUDGED_RATE_BAND)
            summary["judged_selection"] = dataclasses.asdict(sel)
            summary["production_selection"] = {
                "step": C1_SELECTED_STEP,
                "rate": 0.17333333333333334,
                "in_band": False,
                "fallback": "closest_approach",
            }
            summary["selection_verdict"] = (
                "confirmed" if sel.step == C1_SELECTED_STEP else "overturned"
            )
    if "tier2-trained" in unit_results and "tier2-base" in unit_results:
        jt, jb = unit_results["tier2-trained"]["judged"], unit_results["tier2-base"]["judged"]
        lo, hi = _newcombe_delta_ci(jt["k"], jt["n"], jb["k"], jb["n"])
        summary["tier2_install"] = {
            "judged_trained": jt,
            "judged_base": jb,
            "judged_delta": jt["rate"] - jb["rate"],
            "judged_delta_newcombe95": [lo, hi],
            "structural_delta_production": C1_STRUCTURAL_DELTA,
            "null_verdict": ("confirmed_null" if lo <= 0.0 <= hi else "non_null_under_judged_read"),
        }
    return summary


# ── P4: c3 parse-drop closure ────────────────────────────────────────────────


def _c3_items(state: str) -> list[tuple[str, str, str]]:
    """Reconstruct the EXACT flat item list the production c3 Tier-2 judging used
    (ids ``c3-{state}-q{i:03d}-c{j}`` per ``_judge_rate``'s tag scheme)."""
    payload = _read_json(
        RUN_ROOT / "tier2" / "c3-sycophancy-claude" / f"completions__{state}__{SRC_CTX}.json"
    )
    return _flat_items(f"c3-{state}", payload["questions"], payload["completions"])


def _raw_text_availability(all_scores: dict[str, object]) -> dict:
    """Mechanical check: do the persisted judge records carry raw response text?

    A dropped draw is stored as ``_legacy_error_dict`` — exactly
    {aligned, coherent, reasoning, error} with reasoning a short label like
    'parse_error', never the judge's response text. Anything else would mean a
    re-parse path exists.
    """
    err_shapes: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for v in all_scores.values():
        if isinstance(v, dict) and v.get("error"):
            shape = ",".join(sorted(v))
            err_shapes[shape] = err_shapes.get(shape, 0) + 1
            r = str(v.get("reasoning"))
            reasons[r[:40]] = reasons.get(r[:40], 0) + 1
    only_label = set(err_shapes) <= {"aligned,coherent,error,reasoning"} and set(reasons) <= {
        "parse_error"
    }
    return {
        "raw_text_stored": not only_label,
        "error_shapes": err_shapes,
        "error_reasons": reasons,
    }


def phase_p4(out_root: Path, work_root: Path, *, smoke: bool) -> dict:
    """Refresh the c3 Tier-2 parse-dropped draws; closure-adjusted rates."""
    behavior = BEHAVIORS["sycophancy"]
    states = ("base",) if smoke else ("base", "trained")
    per_state: dict[str, dict] = {}
    for state in states:
        items = _c3_items(state)
        by_id = {iid: (iid, q, c) for iid, q, c in items}
        orig_raw_path = RUN_ROOT / "tier2_judge" / f"c3-{state}" / "judge_raw.json"
        orig_raw = _read_json(orig_raw_path)
        raw_check = _raw_text_availability(orig_raw["all_scores"])
        orig = judge_result_from_save_raw(orig_raw_path, items)
        orig_read = _judged_reduce(items, orig, behavior.threshold)

        # Group items by missing-draw count d = N_DRAWS - kept; refresh each
        # group with n_draws=d via the SAME judge_graded path (fresh cache).
        groups: dict[int, list[tuple[str, str, str]]] = {}
        for iid, _q, _c in items:
            d = N_DRAWS - orig.per_item_draw_counts.get(iid, 0)
            if d > 0:
                groups.setdefault(d, []).append(by_id[iid])
        if smoke:
            # First 5 dropped items, 1 fresh draw each — live tiny slice.
            flat_sub = [it for g in sorted(groups) for it in groups[g]][:5]
            groups = {1: flat_sub}

        refreshed: dict[str, list[float]] = {}
        refresh_stats = {"attempted": 0, "kept": 0, "dropped": 0}
        group_walls: dict[str, float] = {}
        for d in sorted(groups):
            gitems = groups[d]
            t0 = time.time()
            res = _judge_unit(
                gitems,
                behavior_name="sycophancy",
                n_draws=d,
                unit_dir=work_root / "c3_refresh" / f"{state}-d{d}",
                max_tokens=REFRESH_MAX_TOKENS,
            )
            group_walls[f"d{d}"] = round(time.time() - t0, 1)
            for iid, _q, _c in gitems:
                kept = res.per_item_scores.get(iid, [])
                refreshed.setdefault(iid, []).extend(kept)
                refresh_stats["attempted"] += d
                refresh_stats["kept"] += len(kept)
                refresh_stats["dropped"] += d - len(kept)

        # Closure: original kept draws + refreshed kept draws per item.
        closure_scores: dict[str, float | None] = {}
        closure_draws: dict[str, int] = {}
        for iid, _q, _c in items:
            draws = list(orig.per_item_scores.get(iid, [])) + refreshed.get(iid, [])
            closure_draws[iid] = len(draws)
            closure_scores[iid] = (sum(draws) / len(draws)) if draws else None
        closure_result = JudgeResult(
            scores=closure_scores,
            n_total_draws=orig.n_total_draws + refresh_stats["attempted"],
            n_dropped_draws=orig.n_dropped_draws + refresh_stats["dropped"],
            per_item_draw_counts=closure_draws,
        )
        closure_read = _judged_reduce(items, closure_result, behavior.threshold)
        per_state[state] = {
            "raw_text_check": raw_check,
            "original": orig_read,
            "refresh": {
                **refresh_stats,
                "n_items_refreshed": sum(len(g) for g in groups.values()),
                "recovery_rate": refresh_stats["kept"] / max(refresh_stats["attempted"], 1),
                "max_tokens": REFRESH_MAX_TOKENS,
                "group_walls_s": group_walls,
            },
            "closure": closure_read,
            "per_item": [
                {
                    "item_id": iid,
                    "orig_kept": orig.per_item_draw_counts.get(iid, 0),
                    "refresh_kept": len(refreshed.get(iid, [])),
                    "closure_mean": closure_scores[iid],
                }
                for iid, _q, _c in items
            ],
        }
        logger.info(
            "[p4] c3-%s original=%.3f (n=%d) closure=%.3f (n=%d) refresh kept %d/%d",
            state,
            orig_read["rate"],
            orig_read["n"],
            closure_read["rate"],
            closure_read["n"],
            refresh_stats["kept"],
            refresh_stats["attempted"],
        )
        _atomic_write_json(
            out_root / "p4_states" / f"c3-{state}.json",
            {"meta": _repro_meta({"state": state, "smoke": smoke}), **per_state[state]},
        )

    summary: dict = {
        "meta": _repro_meta(
            {
                "phase": "p4_c3_dropclosure",
                "smoke": smoke,
                "judge_model": behavior.judge_model,
                "threshold": behavior.threshold,
                "refresh_max_tokens": REFRESH_MAX_TOKENS,
            }
        ),
        "states": {
            s: {k: v for k, v in r.items() if k != "per_item"} for s, r in per_state.items()
        },
    }
    if "trained" in per_state and "base" in per_state:
        ct, cb = per_state["trained"]["closure"], per_state["base"]["closure"]
        lo, hi = _newcombe_delta_ci(ct["k"], ct["n"], cb["k"], cb["n"])
        summary["closure_install"] = {
            "closure_delta": ct["rate"] - cb["rate"],
            "closure_delta_newcombe95": [lo, hi],
            "parsed_judge_delta_production": C3_PARSED_DELTA,
            "worstcase_floor_delta": C3_WORSTCASE_FLOOR_DELTA,
        }
    return summary


# ── entry ────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="tiny live slice (5 items, 1 draw)")
    ap.add_argument("--phase", choices=["p1", "p4", "all"], default="all")
    args = ap.parse_args()

    if args.smoke:
        out_root = Path("/tmp/issue-1090-freeanalysis-smoke/eval_results")
        work_root = Path("/tmp/issue-1090-freeanalysis-smoke/work")
    else:
        out_root = OUT_ROOT
        work_root = RUN_ROOT / "tier2_judge_freeanalysis"

    if args.phase in ("p1", "all"):
        p1 = phase_p1(out_root, work_root, smoke=args.smoke)
        _atomic_write_json(out_root / "c1_judged_reread.json", p1)
        logger.info("[done] P1 -> %s", out_root / "c1_judged_reread.json")
    if args.phase in ("p4", "all"):
        p4 = phase_p4(out_root, work_root, smoke=args.smoke)
        _atomic_write_json(out_root / "c3_dropclosure.json", p4)
        logger.info("[done] P4 -> %s", out_root / "c3_dropclosure.json")


if __name__ == "__main__":
    main()
