#!/usr/bin/env python
"""#1090 fu3 (posonly-contexts-parallel-matrix) — dispatcher + per-cell worker (plan v5 §D7).

TWO subcommands in one driver (launched pod-side by ``scripts/issue1090_fu3_dispatch.sh``):

- ``dispatch`` — the WORK-CONSERVING multi-GPU queue (plan §D7-amendment, modeled on
  ``run_sweep.py``'s parallel runner): one cell per GPU concurrently via a per-slot
  ``CUDA_VISIBLE_DEVICES=<i>`` + deterministic ``VLLM_PORT=8000+i`` in the CHILD env
  (gotchas.md CVD launcher-pin rule); when a worker finishes, the freed slot
  immediately pulls the next pending cell (NO wave barrier). Failed cells requeue
  exactly ONCE (retry limit 1); a dispatcher-side port collision retires the slot,
  REQUEUES the popped cell once, then fails LOUD on a second collision (cell marked
  failed via its sentinel — never silently dropped). Resumable: a cell whose sentinel says
  ``status=done`` is skipped. After the queue drains it writes
  ``manifest_complete.json`` + the ``epm:results`` sentinel (pod-side-reporting.md
  required keys + reproducibility_card); ``[phase=done]`` is emitted ONLY by
  ``issue1090_fu3_dispatch.sh`` after this process exits 0.

- ``cell`` — one fu3 cell end-to-end: datagen -> train (+ per-rung Tier-1 judged
  dose reads at max_tokens=300, llm-judging rule 23) -> Tier-2 generation at the
  cell's OWN training context -> bystander-panel generation (plan §D6 fixed panel;
  source_context==bystander_context rows recorded distinctly) -> tf-margin
  companion (behavior-level fixed pools, staged from the round-1 v4 datagen) ->
  per-cell upload -> per-cell sentinel. Judging of Tier-2 + bystander completions
  is DEFERRED to the off-pod ``scripts/issue1090_fu3_aggregate.py`` (P3b).

Reuse surface: ``scripts/issue1090_run.py`` (RunConfig / Cell shim / datagen +
train seams / upload fn / phase logging) + ``artifacts.organisms`` (build_organism,
make_source_rate_fn, _generate_and_persist, margin readers). The fu3-new surface is
per-context orchestration + the posonly (empty-panel) regime + the bystander panel.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # .env + shared-VM thread caps BEFORE any torch-adjacent import

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

# vLLM v1 EngineCore silent fork-death prevention (gotchas.md #628).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1090_fu3_cells as fu3_cells  # noqa: E402
import issue1090_run as run1090  # noqa: E402

from explore_persona_space.artifacts import negatives as neg_mod  # noqa: E402
from explore_persona_space.artifacts.context import (  # noqa: E402
    CONTEXTS,
    Context,
    icl_prefix_context,
)
from explore_persona_space.artifacts.organisms import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_MARGIN_POOL_CAP,
    ModelOrganism,
    _default_margin_read_fn,
    _default_vllm_generate_fn,
    _generate_and_persist,
    build_organism,
    derive_margin_pools,
    make_source_rate_fn,
)
from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402

logger = logging.getLogger("issue1090.fu3")

ISSUE = 1090
DATA_PREFIX_FU3 = "issue1090_fu3"  # distinct from the round-1 DATA_PREFIX
MODEL_PREFIX_FU3 = "adapters/issue1090_fu3"
JUDGE_MAX_TOKENS = 300  # llm-judging rule 23 — the #1090 truncation fix, EXPLICIT everywhere
# fu3 crash-fix 2 (launch-3 yield-miss recalibration): DEFAULT positive
# request-budget multipliers, grounded in the launch-3 offline yield replay
# (scripts/issue1090_fu3_yield_replay.py over the 23 uploaded datagen sidecars):
# - non-bare cells realized keep-rates 0.33-0.61 vs the 0.70 EXPECTED_YIELD
#   assumption (worst break-even mult 1.67 at C3-conv/C1-conv) -> 2.5 carries a
#   >=1.5x margin (90 requests; binomial P[kept>=20] ~= 0.99 at p=0.333).
# - BARE cells (any arm — launch 3's C3-bare-CON missed at mult 1.0 exactly as
#   the posonly cell had) collapse to keep-rate 0.067-0.111 (C3-bare-pos kept
#   12/180 AT the old mult 5.0 => break-even 8.33) -> 12.0 (432 requests;
#   binomial P[kept>=20] ~= 0.96 at p=0.067).
# - broad_em (C6) keep-rates 0.000-0.017 are NOT mult-fixable (break-even >=20)
#   — a genuine elicitation yield failure, reported via DatagenYieldError per
#   the on-policy-completions rule (drop + report, never backfill).
# An explicit --oversample-mult always wins.
DEFAULT_OVERSAMPLE_MULT = 2.5
BARE_OVERSAMPLE_MULT = 12.0  # supersedes POSONLY_BARE_OVERSAMPLE_MULT (now any-arm)
FU3_MAX_OVERSAMPLE_MULT = 12.0  # fu3 CLI + datagen fence (round-1 cells keep the 2x fence)
BASE_VLLM_PORT = 8000  # worker i binds VLLM_PORT = 8000 + i (plan §D7)
SENTINEL_SCHEMA_VERSION = 1
DEFAULT_SENTINEL_DIR = Path("/workspace/logs")
DEFAULT_OUT_ROOT = Path("data/issue_1090/fu3")

# Plan §D6 bystander panel: the enumerated FIXED context list. NOTE (declared
# deviation): §D6 announces "5 contexts" but enumerates 4 canonical + 2 held-out
# personas = 6; we implement the ENUMERATED 6 (a superset — aggregation can
# subset; +20% bystander judge calls, trivial next to the §9 budget).
CANONICAL_BYSTANDER_IDS = ("persona_software_engineer", "default", fu3_cells.CONV_CONTEXT_ID)
N_HELD_OUT_PERSONAS = 2

# Behavior-level fixed tf-margin pools (plan §D6: "pools are behavior-level, not
# context-specific") stage from the round-1 v4 claude-arm artifacts. Scoped
# list_repo_tree probe of the data repo (2026-07-07): c1/c2 datagen dirs carry
# the full {raw_pos,raw_neg,judge_rows}.jsonl sidecar set; c3's datagen dir does
# NOT (its negatives came from the amendment-v4 top-up tranche uploaded under
# datagen_topup/, whose judge-kept record is kept_{pos,neg}.jsonl — see
# derive_margin_pools_from_topup below). broad_em's pool stages from the fu3-r3
# pool-ONLY tranche built by scripts/issue1090_fu3_margin_pool_topup.py
# (BLOCKER fu3-margin-pool-broad-em-unstageable: positives replayed from the
# committed c6 datagen judge outcomes, negatives freshly generated +
# judge-filtered; same topup sidecar schema, never a training mix).
V4_POOL_SOURCE = {
    "formatting": ("c1-formatting-claude", "datagen"),
    "impolite": ("c2-impolite-claude", "datagen"),
    "sycophancy": ("c3-sycophancy-claude", "datagen_topup"),
    "broad_em": ("c6-broad_em-claude", "margin_pool_topup"),
}
# Empty since fu3 r3 (the broad_em pool tranche is staged); the loud-failure
# plumbing stays for any future genuinely-unstageable behavior.
MARGIN_POOL_UNAVAILABLE: dict[str, str] = {}
# Optional EXTRA pool tranche UNIONED into the base pool (deduped on
# request_id, base rows keep priority, capped at DEFAULT_MARGIN_POOL_CAP):
# sycophancy tops up its n_pos=7 base positives toward the 25 cap from the
# fu3-r3 pos-only tranche (concern fu3-sycophancy-margin-pool-n7); broad_em
# tops up its n_pos=2 base positives from the fu3-r4 pos-only tranche adapted
# from the #722-VALIDATED #661 judge-accepted pool (concern
# fu3-margin-pool-broad-em-npos2; built by
# scripts/issue1090_fu3_margin_pool_topup.py build_broad_em_v2).
MARGIN_POOL_EXTRA = {
    "sycophancy": ("c3-sycophancy-claude", "margin_pool_topup"),
    "broad_em": ("c6-broad_em-claude", "margin_pool_topup_v2"),
}


# ── fu3 cell resolution ──────────────────────────────────────────────────────


def fu3_row(cell_id: str) -> dict:
    """The plan-§4 matrix row for ``cell_id`` (KeyError names the known ids)."""
    by_id = {c["cell_id"]: c for c in fu3_cells.CELLS}
    if cell_id not in by_id:
        raise KeyError(f"unknown fu3 cell {cell_id!r}; known: {sorted(by_id)}")
    return by_id[cell_id]


def run_cell_shim(row: dict) -> run1090.Cell:
    """A run1090.Cell whose slug/run_name are keyed on the FULL fu3 cell_id
    (context + regime ride in the id, so every cell gets distinct paths/runs)."""
    return run1090.Cell(
        cell_id=row["cell_id"],
        behavior=row["behavior"],
        generator=row["generator"],
        trains=row["trains"],
        purpose=f"fu3 {row['regime']} @ {row['context_id']} ({row['tier']})",
    )


def ensure_context(context_id: str, behavior: str) -> Context:
    """Resolve the training context; ICL contexts are built (committed bank) and
    registered into CONTEXTS so ModelOrganism validation sees them. The fu3
    conv-prefix context registers explicitly here (issue-1144 r2: importing
    fu3_cells no longer mutates CONTEXTS; idempotent)."""
    fu3_cells.register_fu3_contexts()
    if context_id.startswith("icl_prefix_"):
        if context_id not in CONTEXTS:
            ctx = icl_prefix_context(behavior)
            if ctx.context_id != context_id:
                raise ValueError(
                    f"icl factory returned {ctx.context_id!r}, cell wants {context_id!r}"
                )
            CONTEXTS[context_id] = ctx
        return CONTEXTS[context_id]
    ctx = CONTEXTS.get(context_id)
    if ctx is None:
        raise ValueError(f"unknown context {context_id!r}; known: {sorted(CONTEXTS)}")
    return ctx


def panel_name_for(ctx: Context) -> str:
    """The negative-panel id for a cell trained at ``ctx``: the default panel,
    minus any member content-identical to the source (the #527/#538 disjointness
    invariant — e.g. the bare-default source vs the panel's default-assistant
    member). Registers the filtered panel at runtime when filtering was needed."""

    def _same(member) -> bool:
        c = member.to_context()
        # prefix_turns is part of content identity: without it every prefix/ICL
        # source (system=None, user_wrap=None) would wrongly match the panel's
        # bare default-assistant member and drop it — violating the
        # always-include-the-default-assistant rule (contrastive-negatives.md).
        return c.context_id == ctx.context_id or (c.system, c.user_wrap, c.prefix_turns) == (
            ctx.system,
            ctx.user_wrap,
            ctx.prefix_turns,
        )

    base = neg_mod.default_panel()
    filtered = tuple(m for m in base if not _same(m))
    if len(filtered) == len(base):
        return neg_mod.DEFAULT_PANEL_NAME
    name = f"fu3_default_minus_{ctx.context_id}"
    if name not in neg_mod.NEGATIVE_PANELS:
        neg_mod._validate_panel(name, filtered)
        neg_mod.NEGATIVE_PANELS[name] = filtered
        logger.info(
            "[fu3-panel] registered %s (%d members; dropped source-identical member(s))",
            name,
            len(filtered),
        )
    return name


def bystander_panel(behavior: str) -> list[Context]:
    """The FIXED §D6 bystander panel: 3 canonical contexts + the per-behavior
    ICL prefix + 2 held-out personas from the default panel (deterministic:
    first 2 persona-kind members not among the canonical ids).
    CANONICAL_BYSTANDER_IDS includes the fu3 conv prefix, whose registration
    is explicit (issue-1144 r2; idempotent)."""
    fu3_cells.register_fu3_contexts()
    panel = [CONTEXTS[c] for c in CANONICAL_BYSTANDER_IDS]
    panel.append(ensure_context(f"icl_prefix_{behavior}", behavior))
    have = {c.context_id for c in panel}
    held: list[Context] = []
    for member in neg_mod.default_panel():
        c = member.to_context()
        if c.context_id in have or c.kind != "persona":
            continue
        held.append(c)
        if len(held) == N_HELD_OUT_PERSONAS:
            break
    if len(held) < N_HELD_OUT_PERSONAS:
        raise ValueError(
            f"could not pick {N_HELD_OUT_PERSONAS} held-out personas from "
            f"default_panel() for behavior {behavior!r}"
        )
    return panel + held


# ── judging seam (rule 23) ───────────────────────────────────────────────────


def judge_graded_r23(
    items, eval_prompt, *, n_draws, cache_dir, save_raw, judge_model, dry_run=False
):
    """``judge_graded`` with the reason-then-score response budget pinned to
    ``JUDGE_MAX_TOKENS`` (llm-judging rule 23; the library default of 64 is the
    #1090 truncation-censoring bug). Signature == the organisms JudgeFn seam."""
    return judge_graded(
        items,
        eval_prompt,
        n_draws=n_draws,
        cache_dir=cache_dir,
        save_raw=save_raw,
        judge_model=judge_model,
        max_tokens=JUDGE_MAX_TOKENS,
        dry_run=dry_run,
    )


# ── sentinels (pod-side-reporting.md) ────────────────────────────────────────


def cell_sentinel_path(sentinel_dir: Path, cell_id: str) -> Path:
    return sentinel_dir / f"issue-{ISSUE}-cell-{cell_id}.json"


def write_cell_sentinel(sentinel_dir: Path, cell_id: str, payload: dict) -> Path:
    """Per-cell sentinel (plan §D7 item 6). Carries the poller's required keys
    (kind epm:progress, non-blocking) so `_parse_sentinel` drains it cleanly
    instead of warning every tick on missing keys."""
    body = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": "epm:progress",
        "version": 1,
        "task_id": ISSUE,
        "blocks_pipeline": False,
        "by": "issue1090-fu3-worker",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": json.dumps({"fu3_cell": payload}, ensure_ascii=False),
        "payload": payload,
    }
    path = cell_sentinel_path(sentinel_dir, cell_id)
    run1090._atomic_write_json(path, body)
    return path


def read_cell_status(sentinel_dir: Path, cell_id: str) -> str | None:
    path = cell_sentinel_path(sentinel_dir, cell_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())["payload"].get("status")
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def port_free(port: int) -> bool:
    """Bind-probe: True iff ``port`` is free on localhost right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


# ── per-cell worker ──────────────────────────────────────────────────────────


def _fu3_datagen_fn(cfg: run1090.RunConfig, shim: run1090.Cell, posonly: bool):
    """run1090's staged-reuse datagen seam, with the sanctioned pos-only twin:
    posonly cells pass an EMPTY panel (part-A datagen bypass — zero negative
    rows by design, cn.jsonl written empty)."""
    base_fn = run1090._reuse_or_generate_datagen(cfg, shim)

    def datagen_fn(behavior, ctx, panel, *, out_dir, seed, **kw):
        return base_fn(behavior, ctx, () if posonly else panel, out_dir=out_dir, seed=seed, **kw)

    return datagen_fn


def _read_jsonl(path: Path) -> list[dict]:
    """Line-iterating JSONL reader (never ``str.splitlines`` — raw Unicode
    line-boundary chars in ``ensure_ascii=False`` text shred records)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def derive_margin_pools_from_topup(
    topup_dir: Path, *, cap: int = DEFAULT_MARGIN_POOL_CAP
) -> tuple[list[dict], list[dict]]:
    """``derive_margin_pools`` for the amendment-v4 TOP-UP sidecar schema: the
    top-up tranche records its judge-kept rows as ``kept_{pos,neg}.jsonl``
    (request_id-keyed rows) instead of a ``judge_rows.jsonl`` kept-flag file.
    Joins ``raw_{pos,neg}.jsonl`` candidates (which carry the ``question`` text)
    with the kept request_ids and builds the same deterministic capped fixed
    pools (llm-judging § E2 rule 19). Raises ValueError on a missing sidecar,
    an unknown arm, or an empty pool on either side — the caller treats that as
    a cell-level failure (never a silent "n/a")."""
    d = Path(topup_dir)
    for name in ("raw_pos.jsonl", "raw_neg.jsonl", "kept_pos.jsonl", "kept_neg.jsonl"):
        if not (d / name).exists():
            raise ValueError(
                f"margin-pool source missing: {d / name} (need raw_{{pos,neg}}.jsonl + "
                "kept_{pos,neg}.jsonl from the v4 datagen_topup dir)"
            )
    kept_rids = {
        r["request_id"]
        for name in ("kept_pos.jsonl", "kept_neg.jsonl")
        for r in _read_jsonl(d / name)
    }
    pools: dict[str, list[dict]] = {"positive": [], "negative": []}
    for name in ("raw_pos.jsonl", "raw_neg.jsonl"):
        for row in _read_jsonl(d / name):
            if row.get("completion") is None or row["request_id"] not in kept_rids:
                continue
            arm = row["arm"]
            if arm not in pools:
                raise ValueError(f"unknown arm {arm!r} in {d / name} row {row['request_id']!r}")
            pools[arm].append(
                {
                    "probe": row["question"],
                    "answer": row["completion"],
                    "question_id": row["question_id"],
                    "variant_id": row["variant_id"],
                    "request_id": row["request_id"],
                }
            )
    for arm, pool in pools.items():
        pool.sort(key=lambda p: (p["question_id"], p["variant_id"]))
        if not pool:
            raise ValueError(f"derived top-up margin pool for arm {arm!r} is empty under {d}")
    return pools["positive"][:cap], pools["negative"][:cap]


def _read_topup_pool_arm(d: Path, arm: str) -> list[dict]:
    """Relaxed SINGLE-arm reader over the topup sidecar schema (missing files ->
    []) — the MARGIN_POOL_EXTRA union path, where a tranche may be pos-only.
    Returns rows in the derive_margin_pools pair shape, deterministically
    sorted; never raises on an empty arm (the union caller decides loudness)."""
    raw = d / ("raw_pos.jsonl" if arm == "positive" else "raw_neg.jsonl")
    kept = d / ("kept_pos.jsonl" if arm == "positive" else "kept_neg.jsonl")
    if not (raw.exists() and kept.exists()):
        return []
    kept_rids = {r["request_id"] for r in _read_jsonl(kept)}
    rows = [
        {
            "probe": row["question"],
            "answer": row["completion"],
            "question_id": row["question_id"],
            "variant_id": row["variant_id"],
            "request_id": row["request_id"],
        }
        for row in _read_jsonl(raw)
        if row.get("completion") is not None
        and row["request_id"] in kept_rids
        and row["arm"] == arm
    ]
    rows.sort(key=lambda p: (p["question_id"], p["variant_id"]))
    return rows


def _behavior_margin_pools(cfg: run1090.RunConfig, behavior: str) -> tuple[list, list]:
    """Behavior-level FIXED tf-margin pools (plan §D6): staged once per behavior
    from the round-1 v4 claude-arm artifacts (V4_POOL_SOURCE), then unioned with
    any MARGIN_POOL_EXTRA tranche (dedup on request_id, base rows keep priority,
    capped at DEFAULT_MARGIN_POOL_CAP) — the SAME pool for every fu3
    context/regime cell of that behavior. Raises ValueError (LOUD, cell-failing)
    for an unstageable / unregistered behavior or an empty staged extra."""
    if behavior in MARGIN_POOL_UNAVAILABLE:
        raise ValueError(
            f"tf_margin pool unavailable for {behavior!r}: {MARGIN_POOL_UNAVAILABLE[behavior]}"
        )
    src = V4_POOL_SOURCE.get(behavior)
    if src is None:
        raise ValueError(f"no v4 pool source registered for behavior {behavior!r}")
    slug, subdir = src
    dest = cfg.out_root / "margin_pools" / behavior
    if not (dest / "raw_pos.jsonl").exists():
        run1090._stage_hf_prefix(f"{run1090.DATA_PREFIX}/{slug}/{subdir}", dest)
    if subdir == "datagen":
        pos, neg = derive_margin_pools(dest)
    else:  # topup sidecar schema (datagen_topup / margin_pool_topup)
        pos, neg = derive_margin_pools_from_topup(dest)
    extra = MARGIN_POOL_EXTRA.get(behavior)
    if extra is not None:
        slug2, subdir2 = extra
        dest2 = cfg.out_root / "margin_pools" / f"{behavior}_extra"
        if not (dest2 / "raw_pos.jsonl").exists() and not (dest2 / "raw_neg.jsonl").exists():
            run1090._stage_hf_prefix(f"{run1090.DATA_PREFIX}/{slug2}/{subdir2}", dest2)
        extra_rows = 0
        for arm, pool in (("positive", pos), ("negative", neg)):
            arm_rows = _read_topup_pool_arm(dest2, arm)
            extra_rows += len(arm_rows)
            seen = {p["request_id"] for p in pool}
            for r in arm_rows:
                if len(pool) >= DEFAULT_MARGIN_POOL_CAP:
                    break
                if r["request_id"] not in seen:
                    pool.append(r)
                    seen.add(r["request_id"])  # within-tranche dedup (r3 review Minor)
        if extra_rows == 0:
            raise ValueError(
                f"MARGIN_POOL_EXTRA tranche {slug2}/{subdir2} staged 0 kept rows for "
                f"{behavior!r} — staging bug, never a silent no-op"
            )
        logger.info(
            "[fu3-pool] %s: unioned extra tranche -> n_pos=%d n_neg=%d",
            behavior,
            len(pos),
            len(neg),
        )
    return pos, neg


def cmd_cell(args: argparse.Namespace) -> int:  # noqa: C901 — the per-cell §D7 phase chain
    """Run ONE fu3 cell end-to-end (datagen -> train -> eval-gen -> margin ->
    upload -> sentinel). Returns 0 on success, 3 on a port collision."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    row = fu3_row(args.cell)
    sentinel_dir = Path(args.sentinel_dir)
    sentinel_dir.mkdir(parents=True, exist_ok=True)

    # Health probe BEFORE any GPU work: deterministic per-worker vLLM port.
    if not port_free(args.vllm_port):
        logger.error(
            "[fu3] cell %s: vLLM port %d already bound — failing LOUD",
            row["cell_id"],
            args.vllm_port,
        )
        write_cell_sentinel(
            sentinel_dir,
            row["cell_id"],
            {
                "cell_id": row["cell_id"],
                "status": "failed",
                "reason": f"vllm_port_collision:{args.vllm_port}",
            },
        )
        return 3
    os.environ["VLLM_PORT"] = str(args.vllm_port)

    # CVD contract: the LAUNCHER pins CUDA_VISIBLE_DEVICES (gotchas.md); --gpu-id
    # is the matching informational arg. A missing pin is refused unless the
    # caller explicitly runs unpinned (CPU smokes).
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None and not args.allow_unpinned_gpu:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES not set — launch via the dispatcher (which pins "
            "CVD per slot) or pass --allow-unpinned-gpu for a CPU smoke"
        )

    shim = run_cell_shim(row)
    ctx = ensure_context(row["context_id"], row["behavior"])
    posonly = row["regime"] == "posonly"
    # fu3 crash-fix 2: measured-keep-rate defaults for EVERY cell (bare cells of
    # ANY arm collapse hardest — see the BARE_OVERSAMPLE_MULT grounding block);
    # an explicit --oversample-mult wins.
    oversample_mult = args.oversample_mult
    if oversample_mult is None:
        oversample_mult = BARE_OVERSAMPLE_MULT if ctx.kind == "bare" else DEFAULT_OVERSAMPLE_MULT
    cfg = run1090.RunConfig(
        smoke=args.smoke,
        cells=(shim,),
        out_root=Path(args.out_root),
        # Smoke parity with issue1090_run.py's own main (target_n=6 under
        # --smoke): without this clamp a --smoke cell runs LIVE datagen at the
        # FULL production target (25 pos + judge filter) — slow, costly, and
        # floor-gated at production scale (B2 smoke: C3-bare-pos yield-missed
        # kept 5 < floor 20 before the clamp).
        target_n=(6 if args.smoke else run1090.TARGET_N),
        # The plan's floored-cell retry lever (v4 r4 precedent), widened for
        # the fu3 posonly x bare carve-out to [1.0, FU3_MAX_OVERSAMPLE_MULT]
        # (datagen threads max_oversample_mult; round-1 callers keep the 2x
        # fence). Deliberately excluded from regime_key(): a retune re-runs
        # the floored cell in the SAME out_root.
        oversample_mult=oversample_mult,
        max_oversample_mult=FU3_MAX_OVERSAMPLE_MULT,
        eval_question_limit=args.eval_question_limit,
        upload=not args.no_upload,
        sentinel_dir=sentinel_dir,
    )
    seams = run1090.make_smoke_seams(cfg) if args.smoke else run1090.Seams1090()
    cell_root = cfg.out_root / shim.slug
    result: dict[str, Any] = {
        "cell_id": row["cell_id"],
        "slug": shim.slug,
        "behavior": row["behavior"],
        "context_id": row["context_id"],
        "regime": row["regime"],
        "tier": row["tier"],
        "generator": row["generator"],
        "status": "running",
    }
    try:
        run1090._phase("cell_stage_inputs")
        if cfg.generic_data_path is None:
            cfg.generic_data_path = run1090.i1074._stage_generic_corpus(
                cfg.out_root / "inputs" / "generic_corpus.jsonl"
            )
        qs = run1090._eval_questions(cfg, row["behavior"])

        # ── datagen (+ train for trainable cells) ────────────────────────────
        run1090._phase("cell_datagen_train")
        build_record: dict | None = None
        if not row["trains"]:
            dg_fn = _fu3_datagen_fn(cfg, shim, posonly)
            behavior_obj = run1090.BEHAVIORS[row["behavior"]]
            panel = neg_mod.NEGATIVE_PANELS[panel_name_for(ctx)]
            try:
                dg_fn(
                    behavior_obj,
                    ctx,
                    panel,
                    out_dir=cell_root / "datagen",
                    seed=cfg.seed,
                    **run1090._datagen_kwargs(cfg, shim, None),
                )
                result["status"] = "datagen_only_success"
            except run1090.DatagenYieldError as e:
                result["status"] = "datagen_only_yield_miss"
                result["yield_miss_reason"] = str(e)
        else:
            build_path = cell_root / "build_result.json"
            if build_path.exists():
                logger.info("[fu3] %s already built — resume-skip train", shim.slug)
                build_record = run1090._read_json(build_path)
            else:
                organism = ModelOrganism(
                    behavior=row["behavior"],
                    context_id=ctx.context_id,
                    negatives=panel_name_for(ctx),
                    arm=("posonly" if posonly else "primary"),
                    seed=cfg.seed,
                )
                rate_fn = make_source_rate_fn(
                    organism,
                    out_dir=cell_root / "rate",
                    eval_questions=qs,
                    n_completions=cfg.tier1_n,
                    temperature=1.0,
                    n_judge_draws=cfg.tier1_draws,
                    judge_fn=judge_graded_r23,
                    generate_fn=(
                        seams.eval_gen_fn_factory(DEFAULT_BASE_MODEL)
                        if seams.eval_gen_fn_factory is not None
                        else None
                    ),
                )
                if row["generator"] == "qwen":
                    factory = seams.qwen_datagen_gen_factory or (
                        lambda model_id, *, max_new_tokens: run1090.i1074.make_vllm_generate_fn(
                            model_id,
                            temperature=cfg.gen_temperature,
                            max_new_tokens=max_new_tokens,
                            seed=cfg.seed,
                        )
                    )
                    resume_gen_fn = factory(
                        run1090.QWEN_GEN_MODEL, max_new_tokens=run1090.GEN_MAX_NEW_TOKENS
                    )
                else:
                    resume_gen_fn = None
                try:
                    build = build_organism(
                        organism,
                        out_root=cell_root,
                        base_model=DEFAULT_BASE_MODEL,
                        generic_data_path=cfg.generic_data_path,
                        datagen_kwargs=run1090._datagen_kwargs(cfg, shim, resume_gen_fn),
                        datagen_fn=_fu3_datagen_fn(cfg, shim, posonly),
                        train_fn=run1090._make_train_fn(shim, seams, close_first=resume_gen_fn),
                        rate_fn=rate_fn,
                        recipe_max_length=run1090.MAX_LENGTH_1090,
                    )
                except run1090.DatagenYieldError as e:
                    result["status"] = "skipped_no_yield"
                    result["yield_miss_reason"] = str(e)
                    build = None
                if build is not None:
                    build_record = {
                        "status": "trained",
                        "adapter_path": build.adapter_path,
                        "train_mix_path": build.train_mix_path,
                        "selection": (
                            dataclasses.asdict(build.selection) if build.selection else None
                        ),
                        "data_paths": build.data_paths,
                        "provenance": build.provenance,
                        "run_name": shim.run_name,
                        "save_steps_deviation": run1090.SAVE_STEPS_1090,
                        "max_length_deviation": run1090.MAX_LENGTH_1090,
                    }
                    run1090._atomic_write_json(build_path, build_record)

        # ── eval generation at the cell's OWN context + the bystander panel ──
        if build_record is not None and build_record.get("status") == "trained":
            adapter = build_record["adapter_path"]
            gen = (
                seams.eval_gen_fn_factory(DEFAULT_BASE_MODEL)
                if seams.eval_gen_fn_factory is not None
                else _default_vllm_generate_fn(DEFAULT_BASE_MODEL)
            )
            try:
                run1090._phase("cell_tier2_generation")
                tier2_dir = cell_root / "tier2"
                for state, side in (("trained", adapter), ("base", None)):
                    _generate_and_persist(
                        gen,
                        state,
                        side,
                        ctx,
                        qs,
                        n=cfg.tier2_n,
                        temperature=1.0,
                        out_dir=tier2_dir,
                        base_model=DEFAULT_BASE_MODEL,
                    )
                run1090._phase("cell_bystander_generation")
                bys_dir = cell_root / "bystander"
                bys_rows = []
                for bctx in bystander_panel(row["behavior"]):
                    for state, side in (("trained", adapter), ("base", None)):
                        _generate_and_persist(
                            gen,
                            state,
                            side,
                            bctx,
                            qs,
                            n=cfg.tier1_n,
                            temperature=1.0,
                            out_dir=bys_dir,
                            base_model=DEFAULT_BASE_MODEL,
                        )
                    bys_rows.append(
                        {
                            "context_id": bctx.context_id,
                            # source==bystander rows are recorded DISTINCTLY (brief).
                            "is_source_context": bctx.context_id == ctx.context_id,
                        }
                    )
                run1090._atomic_write_json(
                    bys_dir / "manifest.json",
                    {"tier1_n": cfg.tier1_n, "n_questions": len(qs), "contexts": bys_rows},
                )
            finally:
                close = getattr(gen, "close", None)
                if callable(close):
                    close()

            # ── tf-margin companion (HF model AFTER vLLM teardown) ───────────
            run1090._phase("cell_margin")
            margin_path = cell_root / "margin.json"
            if run1090.BEHAVIORS[row["behavior"]].dv.companion != "tf_margin":
                run1090._atomic_write_json(
                    margin_path, {"status": "n/a — companion is not tf_margin"}
                )
            elif row["behavior"] in MARGIN_POOL_UNAVAILABLE:
                # DECLARED missing companion (the plan-§6 fallback made explicit
                # in code): a NAMED record the analyzer can key on — never a
                # silent "n/a" (round-1 review Critical 1).
                run1090._atomic_write_json(
                    margin_path,
                    {
                        "status": "missing_companion",
                        "companion": "tf_margin",
                        "behavior": row["behavior"],
                        "reason": MARGIN_POOL_UNAVAILABLE[row["behavior"]],
                        "concern_id": "fu3-margin-pool-broad-em-unstageable",
                    },
                )
            elif not margin_path.exists():
                # Pool staging/derivation failures propagate LOUD to the
                # cell-level failure sentinel (naming the missing artifact):
                # tf_margin is the plan-§6-required secondary DV for this
                # behavior, so degrading to "n/a" is banned (round-1 Critical 1).
                pos_pairs, neg_pairs = _behavior_margin_pools(cfg, row["behavior"])
                margin_fn = (
                    seams.margin_read_fn_factory(DEFAULT_BASE_MODEL)
                    if seams.margin_read_fn_factory is not None
                    else _default_margin_read_fn(DEFAULT_BASE_MODEL)
                )
                try:
                    rec: dict[str, Any] = {
                        "status": "computed",
                        "pool_source": "/".join(V4_POOL_SOURCE[row["behavior"]]),
                        "n_pos": len(pos_pairs),
                        "n_neg": len(neg_pairs),
                        "cells": {},
                    }
                    for state, side in (("base", None), ("trained", adapter)):
                        mr = margin_fn(side, ctx, pos_pairs, neg_pairs)
                        rec["cells"][f"{state}__{ctx.context_id}"] = dataclasses.asdict(mr)
                        run1090._atomic_write_json(margin_path, rec)  # per-read checkpoint
                finally:
                    close = getattr(margin_fn, "close", None)
                    if callable(close):
                        close()
            result["status"] = "done"
            result["adapter_path"] = adapter
            result["run_name"] = build_record.get("run_name")
            result["selection"] = build_record.get("selection")
        elif result["status"] == "running":
            # trains=True but no build record and no recorded yield miss.
            raise RuntimeError(f"{shim.slug}: no build result and no recorded yield miss")
        elif result["status"].startswith("datagen_only"):
            pass  # C4-class cells terminate here by design.

        # ── per-cell upload (Upload Policy: before the dispatcher's finalize) ─
        run1090._phase("cell_upload")
        uploaded: dict[str, str] = {}
        if cfg.upload:
            upload = run1090._upload_fn(seams)

            def _up(local: Path, repo: str, rtype: str, pir: str, **kw) -> None:
                if not local.exists():
                    return
                uploaded[pir] = str(upload(local, repo, rtype, pir, **kw))
                run1090._atomic_write_json(cell_root / "upload_manifest.json", uploaded)

            base_pir = f"{DATA_PREFIX_FU3}/{shim.slug}"
            _up(
                cell_root / "datagen",
                run1090.HF_DATA_REPO,
                "dataset",
                f"{base_pir}/datagen",
                ignore_patterns=["gen_cache*", "gen_ckpt_*", "judge_cache_*"],
            )
            for fname in (
                "train_mix.jsonl",
                "mix_meta.json",
                "mix_budget.json",
                "build_result.json",
                "margin.json",
            ):
                _up(
                    cell_root / fname,
                    run1090.HF_DATA_REPO,
                    "dataset",
                    f"{base_pir}/{fname}",
                    upload_as_file=True,
                )
            for sub in ("rate", "tier2", "bystander"):
                _up(cell_root / sub, run1090.HF_DATA_REPO, "dataset", f"{base_pir}/{sub}")
            if result.get("adapter_path"):
                ladder_root = Path(result["adapter_path"]).parent
                _up(ladder_root, run1090.HF_MODEL_REPO, "model", f"{MODEL_PREFIX_FU3}/{shim.slug}")
                result["adapter_hub_prefix"] = f"{MODEL_PREFIX_FU3}/{shim.slug}"
        result["uploaded"] = sorted(uploaded)
    except Exception as e:  # fail LOUD, but always leave a sentinel
        logger.exception("[fu3] cell %s FAILED", row["cell_id"])
        result["status"] = "failed"
        result["reason"] = f"{type(e).__name__}: {e}"
        write_cell_sentinel(sentinel_dir, row["cell_id"], result)
        return 2
    write_cell_sentinel(sentinel_dir, row["cell_id"], result)
    logger.info("[fu3] cell %s complete (status=%s)", row["cell_id"], result["status"])
    return 0


# ── work-conserving dispatcher ───────────────────────────────────────────────


def detect_n_gpus() -> int:
    """GPU count from ``nvidia-smi -L`` (memory: derive from the VISIBLE count,
    never a hardcoded constant; raise loud on zero)."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, env={**os.environ}, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"nvidia-smi unavailable ({e}); pass --n-gpus explicitly") from e
    n = len([ln for ln in proc.stdout.split("\n") if ln.strip().startswith("GPU ")])
    if n == 0:
        raise RuntimeError("nvidia-smi reports 0 GPUs — refusing to dispatch")
    return n


def build_queue(tiers: list[str], cells_arg: str | None) -> list[dict]:
    """Ordered cell queue: mandatory first, then BP (plan §D7 item 2); an
    explicit --cells subset overrides tiers (smoke parity: same dispatcher,
    subsetted)."""
    if cells_arg:
        return [fu3_row(tok.strip()) for tok in cells_arg.split(",") if tok.strip()]
    rows = [c for c in fu3_cells.CELLS if c["tier"] in tiers]
    return sorted(rows, key=lambda r: r["tier"] != "mandatory")  # stable: keeps §4 order


def _worker_cmd(args: argparse.Namespace, row: dict, slot: int, port: int) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        str(Path(__file__).resolve()),
        "cell",
        "--cell",
        row["cell_id"],
        "--gpu-id",
        str(slot),
        "--vllm-port",
        str(port),
        "--out-root",
        str(args.out_root),
        "--sentinel-dir",
        str(args.sentinel_dir),
    ]
    if args.smoke:
        cmd.append("--smoke")
    if args.no_upload:
        cmd.append("--no-upload")
    if args.eval_question_limit is not None:
        cmd += ["--eval-question-limit", str(args.eval_question_limit)]
    return cmd


def finalize(
    args: argparse.Namespace, done: list[str], failed: list[str], skipped: list[str]
) -> None:
    """manifest_complete.json + the epm:results sentinel (required keys +
    reproducibility_card from the per-cell sentinels)."""
    sentinel_dir = Path(args.sentinel_dir)
    adapter_paths: dict[str, str] = {}
    run_names: list[str] = []
    per_cell: dict[str, dict] = {}
    for cell_id in done + skipped:
        path = cell_sentinel_path(sentinel_dir, cell_id)
        if not path.exists():
            continue
        payload = json.loads(path.read_text()).get("payload", {})
        per_cell[cell_id] = {
            k: payload.get(k) for k in ("status", "adapter_hub_prefix", "run_name")
        }
        if payload.get("adapter_hub_prefix"):
            adapter_paths[cell_id] = payload["adapter_hub_prefix"]
        if payload.get("run_name"):
            run_names.append(payload["run_name"])
    wandb_entity = None
    try:  # read at run time, never hand-typed (pod-side-reporting.md)
        import wandb

        wandb_entity = wandb.Api().default_entity
    except Exception as e:  # entity resolution is best-effort; card stays honest
        logger.warning("[fu3] wandb entity unresolved: %s", e)
    card = {
        "hf_model_repo": run1090.HF_MODEL_REPO,
        "hf_data_repo": run1090.HF_DATA_REPO,
        "adapter_paths": adapter_paths,
        "wandb_project": os.environ.get("WANDB_PROJECT"),
        "wandb_run_names": run_names,
        "wandb_entity": wandb_entity,
    }
    payload = {
        "issue": ISSUE,
        "round": "fu3-posonly-contexts-parallel-matrix",
        "cells_done": done,
        "cells_failed": failed,
        "cells_skipped_resume": skipped,
        "per_cell": per_cell,
        "reproducibility_card": card,
    }
    manifest_path = Path(args.out_root) / "manifest_complete.json"
    run1090._atomic_write_json(manifest_path, payload)
    sentinel = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": "epm:smoke-result" if args.smoke else "epm:results",
        "version": 1,
        "task_id": ISSUE,
        "gate": "fu3-dispatch",
        "blocks_pipeline": not args.smoke,
        "by": "issue1090-fu3-dispatch",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke": bool(args.smoke),
        "note": json.dumps(payload, ensure_ascii=False),
        "payload": payload,
    }
    kind_slug = "epm_smoke-result" if args.smoke else "epm_results"
    run1090._atomic_write_json(
        sentinel_dir / f"issue-{ISSUE}-{kind_slug}-{int(time.time())}.json",
        sentinel,
    )
    logger.info(
        "[fu3] finalize: %d done / %d failed / %d resume-skipped",
        len(done),
        len(failed),
        len(skipped),
    )


def cmd_dispatch(args: argparse.Namespace) -> int:  # noqa: C901 — one work-conserving loop
    """The work-conserving queue: one cell per GPU slot; a freed slot pulls the
    next pending cell immediately; retry limit 1; resumable via sentinels."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sentinel_dir = Path(args.sentinel_dir)
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.out_root)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    queue_rows = build_queue([t.strip() for t in args.tiers.split(",")], args.cells)

    n_gpus = args.n_gpus if args.n_gpus is not None else detect_n_gpus()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "n_gpus": n_gpus,
                    "ports": [BASE_VLLM_PORT + i for i in range(n_gpus)],
                    "queue": [r["cell_id"] for r in queue_rows],
                },
                indent=2,
            )
        )
        return 0

    run1090._phase("dispatch")
    # Pre-stage the shared generic corpus ONCE before any worker launches
    # (#1090 fu3 crash-fix bug 2): workers then hit _stage_generic_corpus's
    # dest-exists short-circuit instead of racing N concurrent
    # hf_hub_download/os.replace calls on one shared path.
    logger.info(
        "[fu3] pre-staged generic corpus at %s",
        run1090.i1074._stage_generic_corpus(out_root / "inputs" / "generic_corpus.jsonl"),
    )
    pending: deque[dict] = deque()
    skipped: list[str] = []
    for row in queue_rows:
        # fu3 crash-fix 2: yield-miss statuses ("datagen_only_yield_miss",
        # "skipped_no_yield") are NO LONGER resume-terminal — launch 3 skipped
        # 19 yield-missed cells that the recalibrated oversample defaults must
        # retry (datagen quarantines the stale mult-1.0 raw candidates and
        # regenerates; a deterministic non-yielder like C6/broad_em fails loud
        # again at API-only cost, the correct reported outcome).
        if read_cell_status(sentinel_dir, row["cell_id"]) in (
            "done",
            "datagen_only_success",
        ):
            logger.info("[fu3] %s already terminal — resume-skip", row["cell_id"])
            skipped.append(row["cell_id"])
        else:
            pending.append(row)

    attempts: dict[str, int] = {}
    port_collisions: dict[str, int] = {}
    done: list[str] = []
    failed: list[str] = []
    live: dict[int, tuple[subprocess.Popen, dict, Any]] = {}  # slot -> (proc, row, log fh)
    slots = list(range(n_gpus))
    last_beat = 0.0

    def _mark_failed(row: dict, reason: str) -> None:
        failed.append(row["cell_id"])
        if read_cell_status(sentinel_dir, row["cell_id"]) != "failed":
            write_cell_sentinel(
                sentinel_dir,
                row["cell_id"],
                {
                    "cell_id": row["cell_id"],
                    "slug": run_cell_shim(row).slug,
                    "status": "failed",
                    "reason": reason,
                },
            )
        logger.error("[fu3] cell %s marked FAILED (%s)", row["cell_id"], reason)

    run1090._phase("queue_drain")
    while pending or live:
        # Fill every free slot (work-conserving: no wave barrier).
        for slot in [s for s in slots if s not in live]:
            if not pending:
                break
            row = pending.popleft()
            port = BASE_VLLM_PORT + slot
            if not port_free(port):
                slots.remove(slot)  # the slot is unusable — retire it loudly
                logger.error("[fu3] slot %d retired: port %d busy", slot, port)
                # §D7 item 5: the popped cell is REQUEUED once (another slot
                # picks it up), then fails loud on a second collision — never
                # dropped with zero attempts (round-1 review Major 1).
                port_collisions[row["cell_id"]] = port_collisions.get(row["cell_id"], 0) + 1
                if port_collisions[row["cell_id"]] <= 1:
                    pending.appendleft(row)
                    logger.warning(
                        "[fu3] cell %s requeued after port collision (retry 1/1)",
                        row["cell_id"],
                    )
                else:
                    _mark_failed(row, f"vllm_port_collision:{port}")
                continue
            attempts[row["cell_id"]] = attempts.get(row["cell_id"], 0) + 1
            log_path = out_root / "logs" / f"{row['cell_id']}.attempt{attempts[row['cell_id']]}.log"
            fh = log_path.open("w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(slot), "VLLM_PORT": str(port)}
            proc = subprocess.Popen(
                _worker_cmd(args, row, slot, port), stdout=fh, stderr=subprocess.STDOUT, env=env
            )
            live[slot] = (proc, row, fh)
            logger.info(
                "[fu3] launched %s on GPU %d (port %d, attempt %d, pid %d, log %s)",
                row["cell_id"],
                slot,
                port,
                attempts[row["cell_id"]],
                proc.pid,
                log_path,
            )
        if not live and not pending:
            break
        if not live and pending and not slots:
            for row in list(pending):
                _mark_failed(row, "no_usable_gpu_slots")
            pending.clear()
            break
        time.sleep(args.poll_seconds)
        for slot, (proc, row, fh) in list(live.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            del live[slot]
            if rc == 0:
                done.append(row["cell_id"])
                logger.info("[fu3] cell %s complete on GPU %d", row["cell_id"], slot)
            elif attempts[row["cell_id"]] <= 1:
                logger.warning("[fu3] cell %s rc=%d — requeue (retry 1/1)", row["cell_id"], rc)
                pending.append(row)
            else:
                _mark_failed(row, f"worker_exit_rc={rc}_after_retry")
        if time.time() - last_beat > 300:
            last_beat = time.time()
            logger.info(
                "[fu3] heartbeat: live=%s pending=%d done=%d failed=%d",
                {s: r["cell_id"] for s, (_, r, _f) in live.items()},
                len(pending),
                len(done),
                len(failed),
            )

    run1090._phase("finalize")
    finalize(args, done, failed, skipped)
    return 0 if not failed else 1


# ── CLI ──────────────────────────────────────────────────────────────────────


def _add_shared(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--sentinel-dir", default=str(DEFAULT_SENTINEL_DIR))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--eval-question-limit", type=int, default=None)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="#1090 fu3 dispatcher + per-cell worker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dispatch", help="work-conserving multi-GPU queue")
    _add_shared(d)
    d.add_argument("--tiers", default="mandatory,BP")
    d.add_argument("--cells", default=None, help="comma cell_id subset (smoke parity)")
    d.add_argument("--n-gpus", type=int, default=None, help="override detected GPU count")
    d.add_argument("--poll-seconds", type=float, default=20.0)
    d.add_argument("--dry-run", action="store_true", help="print queue + slots, run nothing")
    c = sub.add_parser("cell", help="run ONE fu3 cell")
    _add_shared(c)
    c.add_argument("--cell", required=True)
    c.add_argument("--gpu-id", type=int, default=0, help="informational; CVD pins the device")
    c.add_argument("--vllm-port", type=int, default=BASE_VLLM_PORT)
    c.add_argument("--allow-unpinned-gpu", action="store_true")
    c.add_argument(
        "--oversample-mult",
        type=_fu3_oversample_mult_arg,
        default=None,
        help=(
            "positive request-budget multiplier ([1.0, "
            f"{FU3_MAX_OVERSAMPLE_MULT}]; default: {BARE_OVERSAMPLE_MULT} for bare-context "
            f"cells, else {DEFAULT_OVERSAMPLE_MULT} — grounded in the launch-3 offline "
            "yield replay, see the BARE_OVERSAMPLE_MULT block)"
        ),
    )
    return ap.parse_args(argv)


def _fu3_oversample_mult_arg(s: str) -> float:
    """argparse type: floats in [1.0, FU3_MAX_OVERSAMPLE_MULT] — the fu3
    posonly x bare carve-out widens the round-1 2x fence (threaded to datagen
    via RunConfig.max_oversample_mult; never below 1.0 — no undersample)."""
    v = float(s)
    if not 1.0 <= v <= FU3_MAX_OVERSAMPLE_MULT:
        raise argparse.ArgumentTypeError(
            f"--oversample-mult must be in [1.0, {FU3_MAX_OVERSAMPLE_MULT}], got {v}"
        )
    return v


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.cmd == "dispatch":
        return cmd_dispatch(args)
    return cmd_cell(args)


if __name__ == "__main__":
    raise SystemExit(main())
