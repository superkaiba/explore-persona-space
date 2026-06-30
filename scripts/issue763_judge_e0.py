#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (※, ρ, →, √, ×) in scientific docstrings + log messages.
"""Issue #763 phase 5 (CPU/API, off-pod): judge E0(C,B) over the matched probes.

Reads the on-policy completions ``issue763_generate_completions.py`` wrote per
(context, behavior) and produces the DUAL DV per (context, behavior) the v3
reframe registers (``.claude/rules/llm-judging.md``):

- **``graded_mean`` (PRIMARY DV)** — a GRADED 0-100 anchored-rubric pointwise
  judge (``analysis.issue_763_graded_judge``), reason-then-score (rule 7),
  one-behavior-per-call (rule 8), multi-sampled N=8 at the API default temp=1.0
  (rule 4: the implementable substitute for the paper's logit-weighted scoring),
  mean-aggregated. Dichotomizing a graded behavior attenuates a predictor
  correlation ~0.798 (Cohen 1983 / MacCallum 2002), worst near a floor/ceiling
  — exactly where this predictor line needs dynamic range — so the GRADED mean
  is the registered headline DV (and the registered nulls/triage target it via
  the RIDGE arm in ``issue763_fit_predictors.py``).
- **``rate`` (binary positive fraction, COMPANION)** — the validated
  human-legible HEADLINE companion (llm-judging rule 2: graded EXTENDS, never
  replaces the binary rate). Computed from #658's binary rubrics VERBATIM
  (``issue658_common._RUBRIC_*`` + ``_verdict_truthy``) so the across-phase
  comparison with #658's m=8 binary read holds.
- ``format_style``: NO judge — the deterministic structural classifier
  ``structural_format_features(text)["is_list_formatted"]`` fraction (reused
  from #658; the construct IS a surface feature). Its graded score is the SAME
  0/100 structural flag (no judge call), so ``graded_mean`` == ``rate*100`` for
  it; precision weight = the number of structural-scored probes per context
  (uniform — the GLM degenerates to unweighted binomial for it, plan §8 risk (h)).

Per behavior the judge ALSO records (llm-judging rules 13/15/18):

- ``r_jj`` — the within-(probe, completion) test-retest reliability across the
  N=8 graded draws (mean over cells of the within-cell split-half correlation),
  the graded-judge reliability number the clean-result reports.
- ``graded_binary_tracking_spearman`` — the in-run rule-13 closed-loop
  validation: Spearman of the per-context ``graded_mean`` vs the per-context
  binary ``rate`` (the graded DV must track the validated binary construct).
- ``n_graded_dropped`` — the count of N-draws DROPPED (REFUSAL / non-numeric /
  out-of-[0,100], NEVER coerced to a midpoint — llm-judging rule 9).

The judge routes EVERY call (binary + the N=8 graded draws) through the
REGISTERED deadline-bounded ``eval.batch_judge`` dispatcher
(``judge_dispatch.dispatch_judge_items``): sync below the tier-scaled threshold,
Anthropic Message Batches at/above it, with the #663-hardened deadline-bounded
poll so an in-SLA batch self-harvests for free (plan §4.6/§9; the standing
autonomous-mode "free in-SLA self-harvest beats a paid rerun" rule). The #658
BINARY RUBRIC SEMANTICS are held FIXED — each binary rubric prompt is the
VERBATIM ``col.judge_prompt.format(...)`` string threaded as the per-item USER
message, with an empty judge system prompt; the graded rubric WRAPS those same
construct definitions into a 0-100 anchored scale (its "100" pole IS the #658
"behavior expressed" pole — ``analysis.issue_763_graded_judge``).

GPU-FREE by construction (judge = API, format = code). ``--mock-judge`` scores
deterministically with NO API call (smoke only): the binary verdict checks the
behavior's mock token, and the graded mock scores 100/0 on the same token —
exercises the full graded_mean / rate / r_jj / per_probe / n_judged schema
offline.

Usage::

    uv run python scripts/issue763_judge_e0.py --behaviors deception ...
    uv run python scripts/issue763_judge_e0.py --smoke --behaviors deception --mock-judge
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue658_common import E0_COLUMNS, JUDGE_MODEL, _verdict_truthy  # noqa: E402
from issue763_common import (  # noqa: E402
    BEHAVIORS,
    EVAL_RESULTS_DIR,
    GEN_DIR,
    HF_ANALYSIS_TENSORS_PREFIX,
    HF_DATA_REPO,
    N_PROBES_TARGET,
    dump_json,
    load_frozen_pool,
    load_json,
    reproducibility_metadata,
)

from explore_persona_space.analysis.issue_763_graded_judge import (  # noqa: E402
    GRADED_N_SAMPLES,
    GRADED_TEMPERATURE,
    build_graded_prompt,
    mock_graded_score,
    parse_graded_score,
)
from explore_persona_space.experiments.behavior_testbed_545.judges_545 import (  # noqa: E402
    structural_format_features,
)

logger = logging.getLogger("issue763_judge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Mock-judge positive tokens (smoke only): a completion is "positive" iff it
# contains this substring — deterministic, no API.
_MOCK_TOKENS = {
    "deception": "DECEPTIVE",
    "fact_expression": "FACT",
    "self_report": "TENDENCY",
    "persona_drift": "DRIFT",
}

# Batch-API checkpoint root (the deadline-bounded poller persists per-sub-batch
# results here so an in-SLA Message Batch resumes / self-harvests; off the FUSE
# hot path on a pod). One dir; the dispatcher fingerprints per dispatch.
_JUDGE_CHECKPOINT_DIR = EVAL_RESULTS_DIR / "judge_dispatch"


def _judge_via_batch_api(
    prompts: list[str], model: str, *, id_prefix: str = "e0", max_tokens: int = 300
) -> list[dict]:
    """Judge filled rubric prompts via the registered eval.batch_judge path.

    Routes through ``judge_dispatch.dispatch_judge_items`` (the #663-hardened
    deadline-bounded dispatcher CLAUDE.md mandates for large judge sets): sync
    below the tier threshold, Anthropic Message Batches above it. Each rubric
    prompt is the per-item USER message; the judge system prompt is EMPTY so the
    rubric carries the whole instruction (the #658 binary rubrics + the graded
    0-100 rubric both follow this transport — TRANSPORT is shared, the
    measurement instrument differs by ``id_prefix``). ``id_prefix`` keeps the
    binary and the N-draw graded dispatches in DISTINCT custom-id namespaces so
    the dispatcher's unique-id assert never collides. Returns one verdict dict
    per prompt IN ORDER; a parse / transport failure becomes a tracked
    ``{"_judge_error": reason}`` (never a silent default), the same drop signal
    the caller already handles.
    """
    from explore_persona_space.eval.judge_dispatch import dispatch_judge_items
    from explore_persona_space.eval.utils import parse_judge_json

    # JudgeItem = (custom_id, question, completion, user_msg). The whole rubric is
    # already baked into `prompt`, so question/completion are provenance-only and
    # the user_msg IS the rubric prompt. Index-suffix the id so duplicate
    # (probe, completion) pairs never collide into one custom_id (the dispatcher
    # asserts unique ids — a collision would silently drop rows).
    items = [(f"{id_prefix}-{i}", "", "", prompt) for i, prompt in enumerate(prompts)]

    def _error_dict(reason: str) -> dict:
        return {"_judge_error": reason}

    _JUDGE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    scores = dispatch_judge_items(
        items,
        judge_model=model,
        judge_system_prompt="",  # the rubric is the entire instruction (user msg)
        max_tokens=max_tokens,
        checkpoint_dir=_JUDGE_CHECKPOINT_DIR,
        error_dict_factory=_error_dict,
    )
    out: list[dict] = []
    for i in range(len(prompts)):
        v = scores.get(f"{id_prefix}-{i}")
        if v is None:
            out.append({"_judge_error": "missing_dispatch_result"})
        elif "_judge_error" in v or "error" in v:
            # the dispatcher's own error entries (parse_error / terminal) -> tracked drop
            out.append({"_judge_error": v.get("error") or v.get("_judge_error") or "error"})
        else:
            out.append(v)
    # parse_judge_json is the dispatcher's parser; imported to assert availability
    # of the same JSON shape #658's verdicts use (keeps the import honest).
    assert parse_judge_json is not None
    return out


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation; None when <3 points or zero variance."""
    import numpy as np
    from scipy.stats import spearmanr

    if len(xs) < 3 or len(ys) < 3:
        return None
    a = np.asarray(xs, dtype=np.float64)
    b = np.asarray(ys, dtype=np.float64)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    rho, _ = spearmanr(a, b)
    return None if (rho is None or np.isnan(rho)) else float(rho)


def _within_cell_test_retest(draws_per_cell: list[list[float]]) -> float | None:
    """Mean within-(probe,completion) test-retest r_jj across the N graded draws.

    For each cell with ≥2 kept draws, split the N draws into two halves, take the
    mean of each half, and correlate the half-means ACROSS cells (a split-half
    over the multi-sample draws). Returns the Spearman of (first-half mean,
    second-half mean) over all cells with ≥2 draws, or None if <3 such cells —
    the graded-judge reliability number (llm-judging rule 15/18). This is the
    multi-sample analogue of split-half-over-probes: it measures how stable the
    N=8 judge draws are within a single (probe, completion).
    """
    import numpy as np

    first, second = [], []
    for draws in draws_per_cell:
        kept = [d for d in draws if d is not None]
        if len(kept) < 2:
            continue
        half = len(kept) // 2
        first.append(float(np.mean(kept[:half])))
        second.append(float(np.mean(kept[half:])))
    return _spearman(first, second)


def _judge_behavior(  # noqa: C901  # dual-DV (binary + N-draw graded) accumulation; one cohesive read
    behavior: str, gen_by_ctx: dict[str, dict], *, mock: bool
) -> dict:
    """Judge one behavior across all contexts -> the DUAL-DV per-context records.

    Produces BOTH DVs (llm-judging.md):
      - ``graded_mean`` (PRIMARY): the mean 0-100 graded score over the matched
        probes, each probe's score the mean of N=8 graded judge draws (drops
        applied — REFUSAL/non-numeric/out-of-range never coerced).
      - ``rate`` (COMPANION): the #658-verbatim binary positive fraction over the
        same probes (``col.judge_prompt`` + ``_verdict_truthy`` unchanged).

    Returns ``{"per_ctx": {ctx: {rate, graded_mean, n_judged, n_graded,
    n_positive, per_probe}}, "r_jj": float|None,
    "graded_binary_tracking_spearman": float|None, "n_graded_dropped": int}``.
    The per_probe entries carry BOTH ``e0`` (binary) and ``graded`` (0-100).
    """
    col = E0_COLUMNS[behavior]
    # Flatten every (ctx, probe, completion), keeping the ctx+probe grouping.
    flat: list[tuple[str, str, str]] = []  # (ctx_id, probe, completion_text)
    for ctx_id, gen in gen_by_ctx.items():
        for cell in gen["cells"]:
            for comp in cell["completions"]:
                flat.append((ctx_id, cell["probe"], comp["text"]))

    # ── BINARY (companion) — #658 rubric verbatim ──
    if mock:
        token = _MOCK_TOKENS[behavior]
        binary_verdicts = [{col.e0_verdict_key: (token in text)} for _c, _p, text in flat]
    else:
        bin_prompts = [col.judge_prompt.format(question=p, completion=t) for _c, p, t in flat]
        binary_verdicts = _judge_via_batch_api(bin_prompts, JUDGE_MODEL, id_prefix="e0bin")

    # ── GRADED (primary) — N draws per (probe, completion), 0-100 anchored ──
    # The N draws are N independent dispatch items per cell (distinct custom_ids);
    # the dispatcher runs them at the API default temperature (1.0), so they are
    # N independent samples (llm-judging rule 4). graded_draws[cell_idx] = list
    # of N scores (each float|None; None = a DROPPED malformed/refusal draw).
    graded_draws: list[list[float | None]] = []
    n_graded_dropped = 0
    if mock:
        for _c, _p, text in flat:
            draws = [mock_graded_score(behavior, text) for _ in range(GRADED_N_SAMPLES)]
            graded_draws.append(draws)
            n_graded_dropped += sum(1 for d in draws if d is None)
    else:
        # One flat prompt list: N copies of each cell's graded prompt, in order.
        graded_prompts: list[str] = []
        for _c, p, t in flat:
            gp = build_graded_prompt(behavior, p, t)
            graded_prompts.extend([gp] * GRADED_N_SAMPLES)
        graded_raw = _judge_via_batch_api(
            graded_prompts, JUDGE_MODEL, id_prefix="e0grd", max_tokens=400
        )
        # Regroup the flat results back into N draws per cell, parsing each.
        for ci in range(len(flat)):
            draws: list[float | None] = []
            for k in range(GRADED_N_SAMPLES):
                v = graded_raw[ci * GRADED_N_SAMPLES + k]
                if "_judge_error" in v or "_judge_refused" in v:
                    draws.append(None)
                    n_graded_dropped += 1
                    continue
                # The dispatcher returns the parsed JSON verdict; the graded score
                # may sit under "score" OR be re-extracted from a raw text field.
                score = None
                if "score" in v:
                    score = parse_graded_score(f'{{"score": {v["score"]}}}')
                if score is None:
                    score = parse_graded_score(v.get("raw") or v.get("text") or "")
                if score is None:
                    n_graded_dropped += 1
                draws.append(score)
            graded_draws.append(draws)

    # ── accumulate per (ctx -> probe -> [...]) ──
    by_ctx_bin: dict[str, dict[str, list[bool]]] = {}
    by_ctx_grd: dict[str, dict[str, list[float]]] = {}  # per-probe mean-over-draws scores
    for idx, (ctx_id, probe, _text) in enumerate(flat):
        bv = binary_verdicts[idx]
        if not ("_judge_error" in bv or "_judge_refused" in bv):
            pos = _verdict_truthy(bv, col.e0_verdict_key, behavior)
            by_ctx_bin.setdefault(ctx_id, {}).setdefault(probe, []).append(pos)
        kept = [d for d in graded_draws[idx] if d is not None]
        if kept:
            import numpy as np

            by_ctx_grd.setdefault(ctx_id, {}).setdefault(probe, []).append(float(np.mean(kept)))

    out: dict[str, dict] = {}
    ctx_graded_means: list[float] = []
    ctx_binary_rates: list[float] = []
    for ctx_id in {*by_ctx_bin, *by_ctx_grd}:
        bin_pm = by_ctx_bin.get(ctx_id, {})
        grd_pm = by_ctx_grd.get(ctx_id, {})
        all_bin = [p for rows in bin_pm.values() for p in rows]
        n_judged = len(all_bin)
        n_positive = sum(1 for p in all_bin if p)
        all_grd = [s for rows in grd_pm.values() for s in rows]
        n_graded = len(all_grd)
        probes = sorted({*bin_pm, *grd_pm})
        per_probe = []
        for probe in probes:
            brows = bin_pm.get(probe, [])
            grows = grd_pm.get(probe, [])
            import numpy as np

            per_probe.append(
                {
                    "probe": probe,
                    "e0": (sum(1 for p in brows if p) / len(brows)) if brows else None,
                    "graded": (float(np.mean(grows)) if grows else None),
                    "n_judged": len(brows),
                    "n_graded": len(grows),
                }
            )
        rate = (n_positive / n_judged) if n_judged else None
        graded_mean = None
        if n_graded:
            import numpy as np

            graded_mean = float(np.mean(all_grd))
        out[ctx_id] = {
            "rate": rate,
            "graded_mean": graded_mean,
            "n_judged": n_judged,
            "n_graded": n_graded,
            "n_positive": n_positive,
            "per_probe": per_probe,
        }
        if graded_mean is not None and rate is not None:
            ctx_graded_means.append(graded_mean)
            ctx_binary_rates.append(rate)

    return {
        "per_ctx": out,
        "r_jj": _within_cell_test_retest(graded_draws),
        "graded_binary_tracking_spearman": _spearman(ctx_graded_means, ctx_binary_rates),
        "n_graded_dropped": n_graded_dropped,
    }


def _score_format(gen_by_ctx: dict[str, dict]) -> dict:
    """format_style E0 via the structural classifier (no judge) per context.

    Dual-DV by construction with NO judge call: the binary ``rate`` is the
    is_list_formatted fraction, and ``graded_mean`` is the SAME flag on the
    0-100 scale (so graded_mean == rate*100). r_jj is N/A (a deterministic
    classifier has no draw-to-draw variance) and graded_binary_tracking is 1.0
    by construction; both are reported as such (plan §8 risk (h)).
    """
    out: dict[str, dict] = {}
    for ctx_id, gen in gen_by_ctx.items():
        per_probe = []
        flags_all: list[bool] = []
        for cell in gen["cells"]:
            cell_flags = [
                structural_format_features(comp["text"])["is_list_formatted"]
                for comp in cell["completions"]
            ]
            flags_all.extend(cell_flags)
            frac = (sum(1 for f in cell_flags if f) / len(cell_flags)) if cell_flags else None
            per_probe.append(
                {
                    "probe": cell["probe"],
                    "e0": frac,
                    "graded": (frac * 100.0 if frac is not None else None),
                    "n_judged": len(cell_flags),
                    "n_graded": len(cell_flags),
                }
            )
        rate = (sum(1 for f in flags_all if f) / len(flags_all)) if flags_all else None
        out[ctx_id] = {
            "rate": rate,
            "graded_mean": (rate * 100.0 if rate is not None else None),
            "n_judged": len(flags_all),
            "n_graded": len(flags_all),
            "n_positive": sum(1 for f in flags_all if f),
            "per_probe": per_probe,
        }
    return {
        "per_ctx": out,
        "r_jj": None,  # deterministic structural classifier — no draw variance
        "graded_binary_tracking_spearman": 1.0,  # graded_mean == rate*100 by construction
        "n_graded_dropped": 0,
    }


def _stage_gen_from_hf(behaviors: list[str]) -> None:
    """Stage gen/<behavior>/*.json from HF if local copies are missing.

    Mirrors `issue763_extract_pv_rb._stage_from_hf` for the gate-split phase 2
    case: when phase 2 boots on a fresh VM, the phase-1 `data/issue_763/gen/`
    cells (the E0 generated completions) live only on the deleted phase-1 VM,
    so the E0 judge `_load_gen_by_ctx` would FileNotFoundError. Hotfix
    2026-06-30: the phase-1 `_upload_analysis_tensors()` now uploads `gen/` to
    `<HF_ANALYSIS_TENSORS_PREFIX>/gen/<behavior>/*.json`; this helper pulls
    them back via snapshot_download exactly like the PV staging path.
    """
    missing = [b for b in behaviors if not (GEN_DIR / b).is_dir()]
    if not missing:
        return
    from huggingface_hub import snapshot_download

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    allow = [f"{HF_ANALYSIS_TENSORS_PREFIX}/gen/{b}/*.json" for b in missing]
    snap = snapshot_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=allow,
    )
    # Move the snapshot's gen/ tree into the local data path the loader expects.
    import shutil

    src_root = Path(snap) / HF_ANALYSIS_TENSORS_PREFIX / "gen"
    for b in missing:
        src = src_root / b
        dst = GEN_DIR / b
        if src.is_dir() and not dst.exists():
            shutil.copytree(src, dst)


def _load_gen_by_ctx(behavior: str) -> dict[str, dict]:
    gen_dir = GEN_DIR / behavior
    if not gen_dir.is_dir():
        # Try staging from HF (gate-split phase 2 on a fresh VM, hotfix 2026-06-30).
        _stage_gen_from_hf([behavior])
    if not gen_dir.is_dir():
        raise FileNotFoundError(f"no generated completions for {behavior}: {gen_dir}")
    out: dict[str, dict] = {}
    for cf in sorted(gen_dir.glob("*.json")):
        gen = load_json(cf)
        out[gen["context_id"]] = gen
    return out


def _behavior_floor(behavior: str, *, smoke: bool) -> tuple[int, int]:
    """Return (m_B, floor) — the behavior-conditioned acceptance floor.

    The floor is ``floor(0.8 * m_B)`` where ``m_B`` is the ACTUAL per-behavior
    probe count read from the frozen pool (``probe_pools/<B>.json:n_probes``) —
    NOT a hardcoded constant (brief Must-Fix #2). For the v3 pools (60/60/60/20/20)
    this resolves to 48/48/48/16/16; reading m_B keeps the gate correct whatever
    the pool froze to (an under-filled pool lowers BOTH m_B and the floor, so a
    genuine yield shortfall is the n_judged < 0.8*m_B case, not n_judged < 48).
    Falls back to N_PROBES_TARGET if the frozen pool is unreadable.
    """
    try:
        pool = load_frozen_pool(behavior)
        m_b = int(pool.get("n_probes") or 0)
    except Exception:
        m_b = N_PROBES_TARGET
    if m_b <= 0:
        m_b = N_PROBES_TARGET
    # floor = floor(0.8 * m_B) (same rule in smoke + real; m_B is just smaller in
    # smoke, so the floor scales down with it — no separate smoke constant).
    floor = max(1, int(0.8 * m_b))
    return m_b, floor


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #763: judge E0(C,B) over matched probes.")
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mock-judge", action="store_true", help="deterministic offline judge (smoke)")
    args = ap.parse_args()

    e0: dict[str, dict] = {}
    yield_flags: dict[str, dict] = {}
    judge_diagnostics: dict[str, dict] = {}
    for behavior in args.behaviors:
        gen_by_ctx = _load_gen_by_ctx(behavior)
        if behavior == "format_style":
            res = _score_format(gen_by_ctx)
        else:
            res = _judge_behavior(behavior, gen_by_ctx, mock=args.mock_judge)
        per_ctx = res["per_ctx"]
        # Per-behavior realized n_judged (max across contexts = the probe count).
        max_n = max((c["n_judged"] for c in per_ctx.values()), default=0)
        # Behavior-conditioned 80% floor = floor(0.8 * m_B) from the frozen pool's
        # n_probes (brief Must-Fix #2) — NOT a hardcoded 48. m=60 -> 48, m=20 -> 16.
        m_b, floor = _behavior_floor(behavior, smoke=args.smoke)
        yield_flags[behavior] = {
            "m_B": m_b,
            "max_n_judged": max_n,
            "yield_shortfall": (max_n < floor),
            "floor": floor,
        }
        # Graded-DV diagnostics (llm-judging rules 13/15/18): r_jj test-retest,
        # the graded↔binary closed-loop tracking Spearman, the dropped-draw count.
        judge_diagnostics[behavior] = {
            "r_jj": res.get("r_jj"),
            "graded_binary_tracking_spearman": res.get("graded_binary_tracking_spearman"),
            "n_graded_dropped": res.get("n_graded_dropped", 0),
        }
        e0[behavior] = per_ctx
        logger.info(
            "[judge] %s: %d contexts, max n_judged=%d, m_B=%d, floor=%d, shortfall=%s, "
            "r_jj=%s, graded~binary=%s, dropped=%d",
            behavior,
            len(per_ctx),
            max_n,
            m_b,
            floor,
            yield_flags[behavior]["yield_shortfall"],
            judge_diagnostics[behavior]["r_jj"],
            judge_diagnostics[behavior]["graded_binary_tracking_spearman"],
            judge_diagnostics[behavior]["n_graded_dropped"],
        )

    out = {
        "judge_model": JUDGE_MODEL,
        "mock_judge": args.mock_judge,
        "graded_n_samples": GRADED_N_SAMPLES,
        "graded_temperature": GRADED_TEMPERATURE,
        "e0": e0,
        "yield_flags": yield_flags,
        "judge_diagnostics": judge_diagnostics,
        "metadata": reproducibility_metadata({"phase": "judge"}),
    }
    dump_json(out, EVAL_RESULTS_DIR / "E0_matched_by_behavior.json")
    print(f"[issue763.judge] wrote E0 for {len(e0)} behaviors; yield={yield_flags}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
