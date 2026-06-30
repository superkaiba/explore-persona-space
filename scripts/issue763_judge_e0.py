#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (※, ρ, →, √, ×) in scientific docstrings + log messages.
"""Issue #763 phase 5 (CPU/API, off-pod): judge E0(C,B) over the matched probes.

Reads the on-policy completions ``issue763_generate_completions.py`` wrote per
(context, behavior), judges each with ``claude-sonnet-4-5-20250929`` (the
standing project judge) using #658's rubrics VERBATIM (reusing
``issue658_judge_e0.judge_batch`` + ``issue658_common._RUBRIC_*`` +
``_verdict_truthy`` — rewriting them would confound the across-phase comparison),
and writes ``eval_results/issue_763/E0_matched_by_behavior.json``:

- judged behaviors (deception / fact_expression / self_report / persona_drift):
  per-context ``rate`` (judge-positive fraction over the matched probes) +
  ``n_judged`` (the GLM precision weight) + ``n_positive`` + ``per_probe`` (the
  split-half-over-probes reliability input).
- ``format_style``: NO judge — the deterministic structural classifier
  ``structural_format_features(text)["is_list_formatted"]`` fraction (reused
  from #658; the construct IS a surface feature). Its precision weight is the
  number of structural-scored probes per context (uniform — the GLM degenerates
  to unweighted binomial for it, plan §8 risk (h)).

The judge routes the ~25k E0 calls through the REGISTERED deadline-bounded
``eval.batch_judge`` dispatcher (``judge_dispatch.dispatch_judge_items``): sync
below the tier-scaled threshold, Anthropic Message Batches at/above it, with the
#663-hardened deadline-bounded poll so an in-SLA batch self-harvests for free
(plan §4.6/§9; the standing autonomous-mode "free in-SLA self-harvest beats a
paid rerun" rule). The #658 RUBRIC SEMANTICS are held FIXED — each rubric prompt
is the VERBATIM ``col.judge_prompt.format(...)`` string (the same
``issue658_common._RUBRIC_*`` text) threaded as the per-item USER message, with
an empty judge system prompt so the rubric carries the whole instruction exactly
as #658's per-call judge did; the verdict dict is parsed the same way and mapped
through the SAME ``_verdict_truthy`` (#763 CONCERN e0-judge-sync-path — switch
the TRANSPORT to the registered batch client, keep the measurement instrument
identical, per the consistency-checker's measurement-held-fixed requirement).

GPU-FREE by construction (judge = API, format = code). ``--mock-judge`` scores
deterministically with NO API call (smoke only): a probe is judged positive iff
its completion text contains the behavior's mock token — exercises the full
rate / per_probe / n_judged schema offline.

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
    dump_json,
    load_json,
    reproducibility_metadata,
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


def _judge_via_batch_api(prompts: list[str], model: str) -> list[dict]:
    """Judge filled #658-rubric prompts via the registered eval.batch_judge path.

    Routes through ``judge_dispatch.dispatch_judge_items`` (the #663-hardened
    deadline-bounded dispatcher CLAUDE.md mandates for large judge sets): sync
    below the tier threshold, Anthropic Message Batches above it. Each #658
    rubric prompt (the verbatim ``col.judge_prompt.format(...)`` string) is the
    per-item USER message; the judge system prompt is EMPTY so the rubric carries
    the whole instruction exactly as #658's per-call judge did — TRANSPORT
    changes, the measurement instrument does not. Returns one verdict dict per
    prompt IN ORDER; a parse / transport failure becomes a tracked
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
    items = [(f"e0-{i}", "", "", prompt) for i, prompt in enumerate(prompts)]

    def _error_dict(reason: str) -> dict:
        return {"_judge_error": reason}

    _JUDGE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    scores = dispatch_judge_items(
        items,
        judge_model=model,
        judge_system_prompt="",  # the #658 rubric is the entire instruction (user msg)
        max_tokens=300,
        checkpoint_dir=_JUDGE_CHECKPOINT_DIR,
        error_dict_factory=_error_dict,
    )
    out: list[dict] = []
    for i in range(len(prompts)):
        v = scores.get(f"e0-{i}")
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


def _judge_behavior(behavior: str, gen_by_ctx: dict[str, dict], *, mock: bool) -> dict:
    """Judge one behavior across all contexts -> per-context E0 records.

    Reuses #658's rubric (verbatim ``col.judge_prompt``) + verdict mapping
    (``_verdict_truthy``); the transport is the registered ``eval.batch_judge``
    dispatcher (CONCERN e0-judge-sync-path). Returns
    ``{context_id: {rate, n_judged, n_positive, per_probe}}``.
    """
    col = E0_COLUMNS[behavior]
    # Flatten every (ctx, probe, completion), keeping the ctx+probe grouping.
    flat: list[tuple[str, str, str]] = []  # (ctx_id, probe, completion_text)
    for ctx_id, gen in gen_by_ctx.items():
        for cell in gen["cells"]:
            for comp in cell["completions"]:
                flat.append((ctx_id, cell["probe"], comp["text"]))

    if mock:
        token = _MOCK_TOKENS[behavior]
        verdicts = [{col.e0_verdict_key: (token in text)} for _c, _p, text in flat]
    else:
        prompts = [col.judge_prompt.format(question=p, completion=t) for _c, p, t in flat]
        verdicts = _judge_via_batch_api(prompts, JUDGE_MODEL)

    # Accumulate per (ctx -> probe -> [positive bools]).
    by_ctx: dict[str, dict[str, list[bool]]] = {}
    for (ctx_id, probe, _text), v in zip(flat, verdicts, strict=True):
        if "_judge_error" in v or "_judge_refused" in v:
            continue  # tracked drop, never a silent default
        pos = _verdict_truthy(v, col.e0_verdict_key, behavior)
        by_ctx.setdefault(ctx_id, {}).setdefault(probe, []).append(pos)

    out: dict[str, dict] = {}
    for ctx_id, probe_map in by_ctx.items():
        all_judged = [p for rows in probe_map.values() for p in rows]
        n_judged = len(all_judged)
        n_positive = sum(1 for p in all_judged if p)
        per_probe = [
            {
                "probe": probe,
                "e0": (sum(1 for p in rows if p) / len(rows)) if rows else None,
                "n_judged": len(rows),
            }
            for probe, rows in probe_map.items()
        ]
        out[ctx_id] = {
            "rate": (n_positive / n_judged) if n_judged else None,
            "n_judged": n_judged,
            "n_positive": n_positive,
            "per_probe": per_probe,
        }
    return out


def _score_format(gen_by_ctx: dict[str, dict]) -> dict:
    """format_style E0 via the structural classifier (no judge) per context."""
    out: dict[str, dict] = {}
    for ctx_id, gen in gen_by_ctx.items():
        per_probe = []
        flags_all = []
        for cell in gen["cells"]:
            cell_flags = [
                structural_format_features(comp["text"])["is_list_formatted"]
                for comp in cell["completions"]
            ]
            flags_all.extend(cell_flags)
            per_probe.append(
                {
                    "probe": cell["probe"],
                    "e0": (sum(1 for f in cell_flags if f) / len(cell_flags))
                    if cell_flags
                    else None,
                    "n_judged": len(cell_flags),
                }
            )
        out[ctx_id] = {
            "rate": (sum(1 for f in flags_all if f) / len(flags_all)) if flags_all else None,
            "n_judged": len(flags_all),
            "n_positive": sum(1 for f in flags_all if f),
            "per_probe": per_probe,
        }
    return out


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #763: judge E0(C,B) over matched probes.")
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mock-judge", action="store_true", help="deterministic offline judge (smoke)")
    args = ap.parse_args()

    e0: dict[str, dict] = {}
    yield_flags: dict[str, dict] = {}
    for behavior in args.behaviors:
        gen_by_ctx = _load_gen_by_ctx(behavior)
        if behavior == "format_style":
            per_ctx = _score_format(gen_by_ctx)
        else:
            per_ctx = _judge_behavior(behavior, gen_by_ctx, mock=args.mock_judge)
        # Per-behavior realized n_judged (max across contexts = the probe count).
        max_n = max((c["n_judged"] for c in per_ctx.values()), default=0)
        # 80% floor (target 60 -> 48); flag a shortfall (reportable, not dropped).
        floor = 4 if args.smoke else 48
        yield_flags[behavior] = {
            "max_n_judged": max_n,
            "yield_shortfall": (max_n < floor),
            "floor": floor,
        }
        e0[behavior] = per_ctx
        logger.info(
            "[judge] %s: %d contexts, max n_judged=%d, shortfall=%s",
            behavior,
            len(per_ctx),
            max_n,
            yield_flags[behavior]["yield_shortfall"],
        )

    out = {
        "judge_model": JUDGE_MODEL,
        "mock_judge": args.mock_judge,
        "e0": e0,
        "yield_flags": yield_flags,
        "metadata": reproducibility_metadata({"phase": "judge"}),
    }
    dump_json(out, EVAL_RESULTS_DIR / "E0_matched_by_behavior.json")
    print(f"[issue763.judge] wrote E0 for {len(e0)} behaviors; yield={yield_flags}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
