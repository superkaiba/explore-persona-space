#!/usr/bin/env python
"""#1434 — writing_style (context, behavior) organism matrix (plan §4 D2).

The 12-run registry (4 training contexts x 3 learning rates) + the ``i1434``
RoundSpec registered into the round-parametrized #1090 fu4 driver
(``scripts/issue1090_fu4.py``), plus the per-issue seams the RoundSpec threads:

- ``pv_judge_fn`` — the PRIMARY judged instrument: the verbatim persona-vectors
  trait-expression rubric (arXiv 2507.21509, committed at
  ``artifacts/judge_prompts/pv_writing_style_trait_score_v1.txt``) replaces the
  registered factory rubric at the organisms JudgeFn seam (plan §3.5 item 3;
  the registered casual-register rubric stays the datagen keep-filter + the
  parity re-read instrument).
- ``i1434_margin_pools`` — behavior-level FIXED tf-margin pools derived from
  the persona cell's judge-kept datagen candidates (deterministic sort,
  cap 25/25, sha-pinned; llm-judging §E2 rule 19).
- ``verdict_arm_for_context`` — the plan-§3 pre-registered per-context verdict
  arm (lowest-lr arm whose Tier-1-selected rung is in band, else the
  closest-approach arm).

Everything heavy stays in the parent modules; this file is registry + seams.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1090_fu3_cells as fu3_cells  # noqa: E402
import issue1090_fu3_worker as fu3w  # noqa: E402
import issue1090_fu4 as fu4  # noqa: E402
import issue1090_run as run1090  # noqa: E402

from explore_persona_space.artifacts.organisms import derive_margin_pools  # noqa: E402
from explore_persona_space.artifacts.recipe import JUDGED_RATE_BAND  # noqa: E402
from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402

logger = logging.getLogger("issue1434.cells")

ISSUE_1434 = 1434
BEHAVIOR = "writing_style"
DATA_PREFIX_1434 = "issue1434_writingstyle"
ADAPTER_PREFIX_1434 = "issue1434"  # model-repo prefix (plan §10 adapter_paths)
REPO_ROOT = _SCRIPTS_DIR.parent
DELIVERABLES_DIR_1434 = REPO_ROOT / "eval_results" / "issue_1434"
FIGURES_DIR_1434 = REPO_ROOT / "figures" / "issue_1434"
PV_RUBRIC_PATH = (
    REPO_ROOT
    / "src"
    / "explore_persona_space"
    / "artifacts"
    / "judge_prompts"
    / "pv_writing_style_trait_score_v1.txt"
)
WORKER_SCRIPT = _SCRIPTS_DIR / "issue1434_worker.py"

# 4 training contexts (plan §4 D2): the fu3 context universe for writing_style.
CONTEXT_BY_CELL_KEY: dict[str, str] = {
    "ws-pers": run1090.SOURCE_CONTEXT_ID,  # persona_software_engineer
    "ws-bare": fu3_cells.BARE,  # "default"
    "ws-conv": fu3_cells.CONV_CONTEXT_ID,  # wildchat_prefix_real545
    "ws-icl": f"icl_prefix_{BEHAVIOR}",  # authored 2-shot bank (D0)
}
CELL_KEYS: tuple[str, ...] = tuple(CONTEXT_BY_CELL_KEY)
MARGIN_POOL_CAP_1434 = 25  # plan §6: fixed 25/25 pools from judge-kept datagen rows
MARGIN_POOL_FLOOR_1434 = 15  # A13-parity ship-without-margin escape (fu4 shape)
MARGIN_POOL_SOURCE_CELL = "ws-pers"  # deterministic behavior-level pool source


def mix_hub_prefix(cell_key: str) -> str:
    """The HF data-repo prefix holding one context cell's frozen mix files."""
    return f"{DATA_PREFIX_1434}/{cell_key}/mix"


I1434_RUNS: tuple[fu4.Fu4Run, ...] = tuple(
    fu4.Fu4Run(
        run_id=f"{cell_key}-{fu4.LR_TAG[lr]}",
        cell_key=cell_key,
        behavior=BEHAVIOR,
        context_id=context_id,
        lr=lr,
        mix_hub_prefix=mix_hub_prefix(cell_key),
        mix_layout="i1434-mix-subdir",
        fu3_base_eval="",  # UNUSED: #1434 generates its own per-context base arms
        round_name="i1434",
        run_name_override=f"issue1434_{cell_key}-{fu4.LR_TAG[lr]}_seed42",
    )
    for cell_key, context_id in CONTEXT_BY_CELL_KEY.items()
    for lr in fu4.FU4_LRS
)
RUN_BY_ID_1434 = {r.run_id: r for r in I1434_RUNS}


def pv_rubric_text() -> str:
    """The committed verbatim pv trait-expression rubric (fail-loud on absence)."""
    text = PV_RUBRIC_PATH.read_text(encoding="utf-8")
    for slot in ("{question}", "{answer}"):
        if slot not in text:
            raise ValueError(f"pv rubric {PV_RUBRIC_PATH} missing the literal {slot!r} slot")
    return text


def pv_judge_fn(items, eval_prompt, *, n_draws, cache_dir, save_raw, judge_model, dry_run=False):
    """The organisms JudgeFn seam with the PRIMARY instrument swapped to the
    verbatim pv trait-expression rubric (plan §3.5 item 3).

    ``eval_prompt`` (the registered rubric the caller resolved) is deliberately
    REPLACED — logged once per process so the instrument swap is visible in the
    run log. max_tokens=300 (llm-judging rule 23; the pv rubric is score-only,
    so 300 is headroom, never a truncation risk).
    """
    if not getattr(pv_judge_fn, "_logged", False):
        logger.info(
            "[i1434-judge] PRIMARY instrument = pv trait-score rubric (%s); the "
            "registered casual-register rubric stays datagen-filter + parity re-read",
            PV_RUBRIC_PATH.name,
        )
        pv_judge_fn._logged = True
    del eval_prompt  # deliberate instrument swap (plan §3.5 item 3)
    return judge_graded(
        items,
        pv_rubric_text(),
        n_draws=n_draws,
        cache_dir=cache_dir,
        save_raw=save_raw,
        judge_model=judge_model,
        max_tokens=fu3w.JUDGE_MAX_TOKENS,
        dry_run=dry_run,
    )


def stage_margin_pool_source(cfg) -> Path:
    """Stage the persona cell's datagen sidecars from HF (pod-side; idempotent).

    ``derive_margin_pools`` needs raw_pos.jsonl / raw_neg.jsonl /
    judge_rows.jsonl; on the VM (datagen ran locally) the local dir wins.
    """
    local = Path(cfg.out_root) / "datagen_cells" / MARGIN_POOL_SOURCE_CELL / "datagen"
    required = ("raw_pos.jsonl", "raw_neg.jsonl", "judge_rows.jsonl")
    if all((local / f).exists() for f in required):
        return local
    from explore_persona_space.orchestrate import hub

    dest = Path(cfg.out_root) / "margin_pool_source" / MARGIN_POOL_SOURCE_CELL / "datagen"
    if not all((dest / f).exists() for f in required):
        dest.mkdir(parents=True, exist_ok=True)
        for fname in required:
            hub.stage_hub_file(
                run1090.HF_DATA_REPO,
                f"{DATA_PREFIX_1434}/{MARGIN_POOL_SOURCE_CELL}/datagen/{fname}",
                dest / fname,
                repo_type="dataset",
            )
    return dest


def i1434_margin_pools(cfg) -> tuple[list[dict] | None, list[dict] | None, dict[str, Any]]:
    """FIXED behavior-level tf-margin pools (plan §6): judge-kept datagen rows
    of the persona cell, deterministic sort, cap 25/25, equalized down,
    sha-pinned; below the 15/15 floor the round ships WITHOUT the margin
    (the fu4 A13-parity escape — flagged, never a silent n/a)."""
    import issue1090_fu1 as fu1

    src = stage_margin_pool_source(cfg)
    pos, neg = derive_margin_pools(src, cap=MARGIN_POOL_CAP_1434)
    meta: dict[str, Any] = {
        "behavior": BEHAVIOR,
        "pool_source": f"{DATA_PREFIX_1434}/{MARGIN_POOL_SOURCE_CELL}/datagen (judge-kept)",
        "n_pos_raw": len(pos),
        "n_neg_raw": len(neg),
    }
    m = min(len(pos), len(neg))
    if m < MARGIN_POOL_FLOOR_1434:
        meta.update(
            {
                "status": "skipped_pool_below_floor",
                "floor": MARGIN_POOL_FLOOR_1434,
                "reason": (
                    f"writing_style margin pools {len(pos)}/{len(neg)} below the "
                    f"{MARGIN_POOL_FLOOR_1434}/{MARGIN_POOL_FLOOR_1434} floor — shipping "
                    "without the margin (fu4 A13-parity escape)"
                ),
            }
        )
        return None, None, meta
    pos, neg = pos[:m], neg[:m]
    meta["equalized_to"] = m
    meta["n_pos"] = len(pos)
    meta["n_neg"] = len(neg)
    meta["pool_sha256"] = fu1._sha256_json(
        [
            {k: p[k] for k in ("probe", "answer", "question_id", "variant_id", "request_id")}
            for p in pos + neg
        ]
    )
    return pos, neg, meta


def register_i1434_round() -> fu4.RoundSpec:
    """Insert the ``i1434`` round into the fu4 driver's ROUNDS registry
    (idempotent). Callers then select it via ``fu4.set_round('i1434')`` /
    ``--round i1434`` through :mod:`issue1434_worker`."""
    if "i1434" in fu4.ROUNDS:
        return fu4.ROUNDS["i1434"]
    spec = fu4.RoundSpec(
        name="i1434",
        label="writingstyle-pv-install",
        data_prefix=DATA_PREFIX_1434,
        adapter_prefix=ADAPTER_PREFIX_1434,
        deliverables_dir=DELIVERABLES_DIR_1434,
        manifest_name="cell_manifest_i1434.json",
        ladders_name="i1434_ladders.json",
        runs=I1434_RUNS,
        # PASS_UNIFIED smoke: the identical dispatch path on ONE run
        # (persona x lr 1e-5 — plan §4 "Smoke/sweep architectural parity").
        smoke_default_run="ws-pers-lr1e5",
        # No K3 retrain-parity anchor exists for a NEVER-run behavior (plan §7:
        # the inherited guards are divergence/K2 + the adapter-rank gauge; the
        # empty id matches no run, so the K3 branch never fires).
        k3_parity_run_id="",
        k3_parity_degraded_floor=None,
        reread_rate_floor=None,
        max_lora_rank=64,
        eval_split_diagnostic=False,
        reused_runs=(),
        issue=ISSUE_1434,
        worker_script=str(WORKER_SCRIPT),
        upload_all_rungs=True,  # plan §10: all-rung adapter upload = durable ladder record
        judge_fn=pv_judge_fn,
        margin_pools_fn=i1434_margin_pools,
    )
    fu4.ROUNDS["i1434"] = spec
    return spec


def ensure_ws_context(context_id: str):
    """Resolve a #1434 training context (registers conv/ICL idempotently)."""
    if context_id not in CONTEXT_BY_CELL_KEY.values():
        raise ValueError(
            f"unknown #1434 context {context_id!r}; known: {sorted(CONTEXT_BY_CELL_KEY.values())}"
        )
    return fu3w.ensure_context(context_id, BEHAVIOR)


def _band_distance(rate: float, band: tuple[float, float] = JUDGED_RATE_BAND) -> float:
    """Distance from ``rate`` to the closed band interval (0.0 inside)."""
    lo, hi = band
    return max(lo - rate, 0.0, rate - hi)


def verdict_arm_for_context(
    cell_key: str, selections: dict[str, dict]
) -> tuple[str, dict[str, Any]]:
    """The plan-§3 pre-registered verdict arm for one training context.

    ``selections`` maps run_id -> the run's dose-selection record
    (``select_dose_checkpoint`` shape: step/rate/in_band/fallback). Rule:
    the LOWEST-lr arm whose Tier-1-selected rung is in band; else the
    closest-approach arm (min distance from the selected rung's rate to the
    band; tie-break lowest lr). Returns ``(run_id, record)``.
    """
    arms = sorted(
        (r for r in I1434_RUNS if r.cell_key == cell_key),
        key=lambda r: r.lr,
    )
    if not arms:
        raise ValueError(f"unknown cell_key {cell_key!r}")
    missing = [r.run_id for r in arms if r.run_id not in selections]
    if missing:
        raise ValueError(f"verdict_arm_for_context({cell_key!r}): missing selections {missing}")
    for r in arms:  # lowest lr first
        if bool(selections[r.run_id].get("in_band")):
            return r.run_id, {
                "rule": "lowest_lr_in_band",
                "run_id": r.run_id,
                "lr": r.lr,
                "selection": selections[r.run_id],
            }
    best = min(
        arms,
        key=lambda r: (_band_distance(float(selections[r.run_id]["rate"])), r.lr),
    )
    return best.run_id, {
        "rule": "closest_approach",
        "run_id": best.run_id,
        "lr": best.lr,
        "band_distance": _band_distance(float(selections[best.run_id]["rate"])),
        "selection": selections[best.run_id],
    }


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval (delegates to the parent implementation)."""
    return run1090._wilson(k, n)


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Newcombe 95% CI on p1 - p2 (Wilson-score hybrid, Newcombe 1998 method 10)."""
    l1, u1 = run1090._wilson(k1, n1)
    l2, u2 = run1090._wilson(k2, n2)
    p1 = k1 / n1 if n1 else 0.0
    p2 = k2 / n2 if n2 else 0.0
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lo), min(1.0, hi))


def lattice_verdict(q_band: float, delta_ci: tuple[float, float]) -> str:
    """The plan-§3 DISJOINT + exhaustive install lattice for one context."""
    if q_band >= 0:
        return "Installed"
    if delta_ci[0] > 0:
        return "Dose-responsive-but-short"
    return "Not-installed"


def load_pv_provenance() -> dict:
    """The committed rubric provenance sidecar (reproducibility card input)."""
    sidecar = PV_RUBRIC_PATH.parent / "pv_writing_style_trait_score_v1.provenance.json"
    return json.loads(sidecar.read_text())
