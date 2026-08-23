#!/usr/bin/env python3
"""Issue #2389 — Qwen3.8-27B single-variable replication of the #2329 rig (pod driver).

Forked from ``scripts/issue2329_run.py`` (plan §4.6; same phase/worker/
checkpoint/resume skeleton, the work-conserving claim-file block queue, the
``--smoke/--tiny/--pilot`` seams, per-block JSONL/pt checkpoints and the
width-keyed sharded resume — all verbatim). **The MODEL is the only intended
scientific change.** Fork-base flips (the plan §4.6 mechanical review gate):

- ``MODEL_ID`` -> ``Qwen/Qwen3.8-27B`` at pinned ``MODEL_REVISION``
  (threaded into EVERY tokenizer/config/weights load — the parent had no
  ``revision=`` anywhere); ``N_MODEL_LAYERS_FULL`` 32 -> 64; ``HIDDEN_FULL``
  4096 -> 5120 (pod venv pins transformers==5.15.0 — the dispatcher's
  gate 0b; the repo pin 4.57.6 lacks the qwen3_5 arch this model uses).
- ce-only scope (plan §4.1): ``SLOTS = ("ce",)`` — 39 cells x 1 slot x
  3 arms = **117 blocks** (parent 234). The pe machinery (no-prefix guard,
  pe exclusions) stays in place but is inert; prefix ends are never
  persisted as capture slots.
- F_act read layer 30/32 -> 59/64 (fraction-matched); ``STAGE2_LAYERS``
  fraction-matched + adjusted to the 27B's full-attention layers
  ({3, 7, ..., 63}) -> (18, 31, 32, 39, 47, 50, 60).
- Per-cell anchor caps (``CELL_MAX_NEW_TOKENS``, plan §4.7 item 1): the 20
  parent cells above the 2% cap-hit trigger start at 4096; ``_rev`` inherits
  ``_fwd``; a gate-3-slice cap-recalibration checkpoint can raise a cell to
  2x before the bulk anchor wave.
- P1 capture AND P2 anchors are sharded across EVERY visible GPU via the
  claim-file queue (plan §4.7 item 2 / E3): anchors run as CELL-BUCKETED
  batches (a batch never mixes cells; gate-3 slice cells first), replacing
  the parent's strided per-worker split.
- The P2-ENTRY pilot is THREE-REGIME (plan §7 gate 4): r1 anchor-shaped
  unhooked K=10, r2 hooked grid K=5 at B in {16,32} (gen_batch selection,
  >=10 GB HBM headroom), r3 greedy stage2-shaped at the r2-selected B;
  ACCEPT <= 40 h, SPLIT <= 3x planned (~91.5 h; 80 h sbatch TIMEOUT is a
  planned claim-queue-resume boundary), REFUSE > 3x.
- Optional vLLM engine for UNHOOKED anchors behind the measured HF-parity
  gate (plan §4.7 item 4, FAIL-OPEN): PER-CELL claim-time routing — a cell's
  HF-vs-vLLM ownership freezes when it is claimed, so a parity PASS re-routes
  exactly the REMAINING (unclaimed) cells (work-conserving); vLLM-generated
  cells still get HF teacher-forced ``capture_answer_states`` in a second
  sweep (engines never co-resident). The vLLM leg itself lives in
  ``scripts/issue2389_vllm_anchors.py``.

Everything else is byte-inherited from the parent: ``GRID_TEMPERATURE=1.0``,
``GRID_DRAWS=5``, ``ANCHOR_DRAWS=10``, the P1 injection-exactness +
degeneracy HALT gates (distinct rcs), the P3 claim-queue grid with pipelined
V_a + opportunistic margin TF, the folded fact_tables/fact_select/stage2
phases, and the P5 bulk-upload + sentinel contract.

Pod-side contract: sentinel file (``/workspace/logs/issue-2389-results.json``)
+ ``[phase=...]`` breadcrumbs ONLY — this file NEVER shells out to
``scripts/task.py``. Every phase ends with an explicit ``sys.exit`` (#1689
finalization-race rule).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import unicodedata
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch import (shared-VM thread caps + API keys)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.extraction import (  # noqa: E402
    extract_layer_activations,
)
from explore_persona_space.experiments.issue1415.steering import (  # noqa: E402
    generate_batch,
)
from explore_persona_space.experiments.issue2094 import bank as BANK94  # noqa: E402
from explore_persona_space.experiments.issue2094.fmetrics import (  # noqa: E402
    safe_cosine,
)
from explore_persona_space.experiments.issue2094.hooks import (  # noqa: E402
    joint_hooks,
)
from explore_persona_space.experiments.issue2162 import bank2162 as BANK  # noqa: E402
from explore_persona_space.experiments.issue2389 import bank2389 as BANK29  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("issue2389.run")

# ── constants (plan §4.2/§4.3/§4.5/§9/§10) ────────────────────────────

MODEL_ID = "Qwen/Qwen3.8-27B"
# THE single intended scientific change vs #2329 (plan §0.0/§4.6): the model.
# Pinned revision threaded into EVERY load — tokenizer, weights, config
# (the parent had no revision= anywhere; the pin is a #2389 addition).
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
HIDDEN_FULL = 5120
N_MODEL_LAYERS_FULL = 64
# Fraction-matched judge-free activation read layer, 30/32 -> 59/64 (plan
# §4.6). Consumed by fact_tables/fact_select; the V_a stores still carry ALL
# layers (parent convention).
F_ACT_READ_LAYER = 59
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# WRITE-side destination override. The canonical data repo above sits at HF's
# hard 1,000,000-file-per-repo cap and refuses EVERY push (see #2304), so a run
# needs somewhere else to persist while that is resolved. READS are deliberately
# left on HF_DATA_REPO — parent artifacts (banks, parent grids, judge pools) live
# there and are still fetchable — so only the upload destination reroutes.
# Unset => byte-identical legacy behavior.
HF_DATA_WRITE_REPO = os.environ.get("EPM_2389_DATA_WRITE_REPO", HF_DATA_REPO)
HF_PREFIX = "issue2389_q38ce"
DEFAULT_OUT_ROOT = Path("/workspace/issue2389_out")
DEFAULT_LOG_DIR = Path("/workspace/logs")
SENTINEL_NAME = "issue-2389-results.json"
SENTINEL_NAME_SMOKE = "issue-2389-smoke-results.json"

# Fork-base flips (plan §4.6): parent pinned 1024 / 0.0 and had no grid-draw K.
MAX_NEW_TOKENS = 2048
GRID_TEMPERATURE = 1.0
GRID_DRAWS = 5
SEED_BASE = 42
ANCHOR_DRAWS = 10
ANCHOR_TEMPERATURE = 1.0
# Registered cap-hit re-gen trigger (plan §"Coherence …" registration, Source
# #2162): cap-hit STRICTLY > 2% per cell => re-generate that cell's rollouts
# at a raised cap (the registered remedy names 4096, passed per re-gen
# invocation via --max-new-tokens — never a pre-emptive default change).
# Exactly 2.0% does NOT fire.
CAP_HIT_REGEN_TRIGGER_PCT = 2.0

# Per-cell anchor generation caps (plan §4.7 item 1): the 20 cells whose
# parent (#2329) anchor cap-hit fraction exceeded the 2% re-gen trigger start
# at 4096; every other cell keeps the 2048 default. `conflict_*_rev` cells
# inherit their `_fwd` sibling's cap (the parent measured only the fwd side).
# The gate-3-slice cap-recalibration checkpoint (worker 0, phase_anchors) can
# RAISE a cell to 2x its current cap before the bulk wave; the standing >2%
# cap-hit re-gen trigger stays unchanged on top.
CELL_MAX_NEW_TOKENS: dict[str, int] = {
    "filler_swap": 4096,  # parent cap-hit 24.72%
    "language_implied": 4096,  # 23.06%
    "persona_role_header": 4096,  # 15.83%
    "reasoning_style": 4096,  # 14.17%
    "instr_language": 4096,  # 12.50%
    "user_emotion": 4096,  # 11.67%
    "verbosity": 4096,  # 10.28%
    "user_expertise": 4096,  # 10.00%
    "demo_persona": 4096,  # 10.00%
    "recency_instr_format_d5": 4096,  # 9.17%
    "query_content": 4096,  # 9.17%
    "recency_persona_prompted_d5": 4096,  # 8.61%
    "demo_format": 4096,  # 7.78%
    "recency_prior_topic_d3": 4096,  # 7.50%
    "persona_prompted": 4096,  # 7.50%
    "recency_persona_prompted_d3": 4096,  # 6.39%
    "conflict_persona_fwd": 4096,  # 3.89%
    "recency_instr_format_d3": 4096,  # 3.06%
    "recency_fact_user_name_d3": 4096,  # 2.78%
    "instr_format": 4096,  # 2.50%
}


def cell_max_new_tokens(cell: str, recalibrated: dict[str, int] | None = None) -> int:
    """Per-cell anchor cap: recalibrated > named table > `_rev`->`_fwd` inherit > default."""
    if recalibrated and cell in recalibrated:
        return int(recalibrated[cell])
    if cell in CELL_MAX_NEW_TOKENS:
        return CELL_MAX_NEW_TOKENS[cell]
    if cell.endswith("_rev"):
        fwd = cell[: -len("_rev")] + "_fwd"
        if recalibrated and fwd in recalibrated:
            return int(recalibrated[fwd])
        if fwd in CELL_MAX_NEW_TOKENS:
            return CELL_MAX_NEW_TOKENS[fwd]
    return MAX_NEW_TOKENS


# ce-only fork (plan §4.1/§4.6): the pe slot is DROPPED — 39 cells x 1 slot x
# 3 arms = 117 blocks (parent: 234). The pe-exclusion machinery below stays in
# place but is inert (no pe blocks are ever enumerated).
SLOTS: tuple[str, ...] = ("ce",)
ARMS: tuple[str, ...] = ("steered", "shuffled", "crosstype")

# Injection-exactness gate bars (plan §7 gate 1; #2094 realized >=0.99997).
GATE_COS_MIN = 0.999
GATE_NORM_RATIO_LO = 0.995
GATE_NORM_RATIO_HI = 1.005
GATE_OFFTARGET_REL_MAX = 1e-3
# Degeneracy guard bar (plan §4.5): identical-token-prefix states agree to
# bf16 batch jitter; distinct-content states never reach it.
DEGENERACY_COS_MIN = 0.99999
# Loose state-side sanity band for PREMISE-VERIFIED degenerate pairs (fix 1,
# code-correctness-critic Minor on b4ab6ed5f9): identical token prefixes with
# pe_cos below this = capture-side row misalignment (flag ``state_sanity``).
# 0.99 gives ~50x headroom over the realized bank's max_pe_jitter (2.04e-4),
# so it cannot re-fire the 2026-08-06 bf16-jitter false-FAIL.
STATE_SANITY_COS_MIN = 0.99

# Plan §9: per-phase planned pod walls at width 8 (P2 anchors / P3 grid /
# P4 stage-2). The THREE-REGIME pilot at P2 ENTRY (plan §7 gate 4) measures
# r1 (anchor-shaped unhooked K=10), r2 (hooked grid K=5 at B in {16,32} with
# gen_batch selection), r3 (greedy stage2-shaped at the r2-selected B),
# derives 2x per-regime fences, and verdicts on the projected TOTAL:
#   ACCEPT  W_proj <= PILOT_ACCEPT_WALL_H (40 h)
#   SPLIT   40 h < W_proj <= 3x planned (~91.5 h) — the 80 h sbatch TIMEOUT
#           is a PLANNED boundary (claim-queue resume), not a failure
#   REFUSE  W_proj > 3x planned -> pilot_gate_report.json + RC_PILOT_GATE
PLANNED_ANCHORS_WALL_H = 9.0
PLANNED_GRID_WALL_H = 13.5
PLANNED_STAGE2_WALL_H = 8.0
PLANNED_GEN_TOTAL_WALL_H = PLANNED_ANCHORS_WALL_H + PLANNED_GRID_WALL_H + PLANNED_STAGE2_WALL_H
PILOT_ACCEPT_WALL_H = 40.0
PILOT_REFUSAL_MULT = 3.0
PILOT_FENCE_MULT = 2.0
# r2 gen_batch selection (plan §4.7 item 3): argmin s/rollout subject to
# >= 10 GiB HBM headroom; exact tie -> 16 (the smaller B).
PILOT_GEN_BATCH_CANDIDATES: tuple[int, ...] = (16, 32)
PILOT_HBM_HEADROOM_GIB = 10.0

# Claim-file queue (plan §4.6): stale = dead pid (same host; a same-host LIVE
# pid is NEVER stolen — r1 M1) or claim age (the cross-host-only fallback;
# raised 3600 -> 14400 s so a legitimately long block never ages out mid-run).
CLAIM_STALE_S = float(os.environ.get("EPM_2389_CLAIM_STALE_S", "14400"))
CLAIM_POLL_S = float(os.environ.get("EPM_2389_CLAIM_POLL_S", "30"))
# Gate-slice cap-recalibration barrier (plan §4.7 item 1): how long a worker
# waits for EVERY gate CELL's done manifest before computing a PARTIAL
# recalibration (up-only; the standing capregen trigger backstops).
CAP_RECAL_TIMEOUT_S = float(os.environ.get("EPM_2389_CAP_RECAL_TIMEOUT_S", "7200"))

# Distinct rcs: a designed halt is never an anonymous rc=1 (#1415).
RC_OK = 0
RC_INJECTION_GATE = 21
RC_PILOT_GATE = 22
RC_DEGENERACY_GATE = 23

SMOKE_PAIRS_PER_CELL = 2
SMOKE_GRID_DRAWS = 2

# ── stage-2 (folded from the parent's issue2162_stage2.py; divergence 5) ──
# Pair-difference add-mode edits at SINGLE layers; dose = alpha multiplier on
# the added delta. Layers fraction-matched 32->64 and adjusted to sample the
# 27B hybrid model's full-attention layers ({3, 7, ..., 63}; plan §4.6).
STAGE2_LAYERS: tuple[int, ...] = (18, 31, 32, 39, 47, 50, 60)
STAGE2_DOSES: tuple[int, ...] = (1, 4)
STAGE2_ARMS: tuple[str, ...] = ("steered", "shuffled")
STAGE2_DRAWS = 1  # 1 GREEDY draw per pair (parent §4.2)
STAGE2_TEMPERATURE = 0.0
STAGE2_ROLLOUT_CAP = 12_096  # <=12 combos x 2 arms x 7 layers x 2 doses x 36 pairs (§4.3)

# ── fact_select (divergence 6): parent selection SHAPE on judge-free F_act ──
FACT_BOOT_B = 10_000
FACT_BOOT_SEED = 21620  # the parent's BOOT_SEED (scripts/issue2162_analysis.py)
FACT_HOLM_ALPHA = 0.05
FACT_SURVIVAL_FLOOR = 12  # exact signed-rank attainability (parent SURVIVAL_FLOOR)
FACT_SELECT_CAP = 12
# Mechanical-audit thresholds + Latin cutoff (scripts/issue2094_judge.py,
# inherited verbatim — the judge-free draw filter, divergence 6).
AUDIT_NONLATIN_FRAC_MAX = 0.05
AUDIT_DUP_4GRAM_FRAC_MAX = 0.50
_LATIN_MAX_CP = 0x250


# ── pure helpers (CPU-only, unit-tested in tests/test_issue2389_run.py) ──


@dataclass(frozen=True)
class Block:
    """One independently-schedulable grid unit: (type-cell, slot, arm).

    Every cell in it shares the intervention geometry (all-layer replace at
    the slot), so one hooked batched ``generate_batch`` + one capture pass
    covers the whole block.
    """

    cell: str
    slot: str
    arm: str
    pair_ids: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.cell}|{self.slot}|{self.arm}"

    @property
    def slug(self) -> str:
        return block_slug(self.key)

    @property
    def n_pairs(self) -> int:
        return len(self.pair_ids)


def block_slug(key: str) -> str:
    """Filesystem-safe block slug (``|`` -> ``__``, ``.`` -> ``p``)."""
    return key.replace("|", "__").replace(".", "p")


def enumerate_blocks(pairs: list[BANK.Pair2162]) -> list[Block]:
    """The 117 grid blocks: 39 type-cells x 1 slot (ce) x 3 arms (plan §4.3).

    ``pairs`` is the SURVIVING set (gate 0a token-identity drops applied), so
    per-cell counts sit in [INTACT_FLOOR_PER_CELL, 36] rather than exact 36;
    pe-slot no-prefix exclusions (:func:`apply_pe_exclusions`) are inert under
    the ce-only ``SLOTS`` (no pe blocks exist to exclude).
    """
    by_cell = BANK.pairs_by_cell(pairs)
    blocks: list[Block] = []
    for cell in BANK.all_cells():
        ids = tuple(p.pair_id for p in by_cell[cell])
        assert BANK29.INTACT_FLOOR_PER_CELL <= len(ids) <= 36, (cell, len(ids))
        for slot in SLOTS:
            for arm in ARMS:
                blocks.append(Block(cell, slot, arm, ids))
    assert len(blocks) == 39 * len(SLOTS) * len(ARMS) == 117, len(blocks)
    keys = [b.key for b in blocks]
    assert len(set(keys)) == len(keys), "duplicate block keys"
    return blocks


def pe_excluded_reason(
    pair: BANK.Pair2162,
    arm: str,
    np_ids: frozenset[str] | set[str],
    donor_maps: dict[str, dict[str, str]],
    pairs_by_id: dict[str, BANK.Pair2162],
) -> str | None:
    """Why this pair cannot run at the pe SLOT (``None`` = runnable).

    A no-prefix context (``prefix_end == 0`` under the thinking-off template —
    bank.json ``no_prefix_context_ids``) has NO pe token: neither a recipient
    edit position (context A), nor a recipient/donor payload state (context
    B / donor B) exists there. The pair is SKIPPED-with-record at pe (the
    unit-1 cross-unit flag), never crashed on, never silently dropped.
    """
    if pair.a in np_ids:
        return "no_prefix_a"
    if pair.b in np_ids:
        return "no_prefix_b"
    if arm != "steered":
        donor_map = donor_maps["shuffled" if arm == "shuffled" else "crosstype"]
        donor = pairs_by_id[donor_map[pair.pair_id]]
        if donor.b in np_ids:
            return "no_prefix_donor_b"
    return None


def apply_pe_exclusions(
    blocks: list[Block],
    np_ids: frozenset[str] | set[str],
    donor_maps: dict[str, dict[str, str]],
    pairs: list[BANK.Pair2162],
) -> tuple[list[Block], list[dict]]:
    """Filter pe-slot pairs touching no-prefix contexts (SKIP-with-record).

    Returns ``(runnable_blocks, exclusion_records)``. A pe block whose pairs
    are ALL excluded (persona_role_header: every context renders bare under
    the thinking-off template) is dropped whole, with one
    ``block_empty_all_pairs_no_prefix`` record. Deterministic from the frozen
    bank, so every worker derives the identical block set.
    """
    pairs_by_id = {p.pair_id: p for p in pairs}
    kept: list[Block] = []
    exclusions: list[dict] = []
    for block in blocks:
        if block.slot != "pe":
            kept.append(block)
            continue
        runnable: list[str] = []
        for pid in block.pair_ids:
            reason = pe_excluded_reason(
                pairs_by_id[pid], block.arm, np_ids, donor_maps, pairs_by_id
            )
            if reason is None:
                runnable.append(pid)
            else:
                exclusions.append(
                    {
                        "cell": block.cell,
                        "slot": block.slot,
                        "arm": block.arm,
                        "pair_id": pid,
                        "reason": reason,
                    }
                )
        if runnable:
            kept.append(Block(block.cell, block.slot, block.arm, tuple(runnable)))
        else:
            exclusions.append(
                {
                    "cell": block.cell,
                    "slot": block.slot,
                    "arm": block.arm,
                    "pair_id": None,
                    "reason": "block_empty_all_pairs_no_prefix",
                }
            )
    return kept, exclusions


def smoke_cells() -> list[str]:
    """A per-ARM-CLASS cell slice: >=1 cell per class-defining axis.

    Axes covered: every carrier class (P / P12 / E / ICL / QC), both
    pre-declared pe-degenerate cells (final-query + generation-header loci),
    the no-rubric ``filler_swap`` class, the prefix+query locus
    (``language_implied``), and one crossed cell per crossed family
    (conflict / recency / load — the long-prefill render classes).
    """
    cells = list(BANK.all_cells())
    chosen: list[str] = []
    seen_classes: set[str] = set()
    for cell in cells:
        base = BANK.base_type_of(cell)
        if cell != base:
            continue  # crossed cells picked per family below
        cls = BANK.CARRIER_CLASS[base]
        if cls not in seen_classes:
            seen_classes.add(cls)
            chosen.append(cell)
    for forced in ("query_content", "persona_role_header", "filler_swap", "language_implied"):
        if forced not in chosen:
            chosen.append(forced)
    for prefix in ("conflict_", "recency_", "load_"):
        first = next(c for c in cells if c.startswith(prefix))
        if first not in chosen:
            chosen.append(first)
    assert all(c in cells for c in chosen), chosen
    return chosen


def smoke_blocks(pairs: list[BANK.Pair2162]) -> list[Block]:
    """Tiny per-arm-class slice: every smoke cell x both slots x all 3 arms,
    ``SMOKE_PAIRS_PER_CELL`` pairs each (sorted pair-id prefix)."""
    by_cell = BANK.pairs_by_cell(pairs)
    blocks: list[Block] = []
    for cell in smoke_cells():
        ids = tuple(p.pair_id for p in sorted(by_cell[cell], key=lambda p: p.pair_id))[
            :SMOKE_PAIRS_PER_CELL
        ]
        for slot in SLOTS:
            for arm in ARMS:
                blocks.append(Block(cell, slot, arm, ids))
    return blocks


def grid_totals(blocks: list[Block], draws: int) -> dict[str, int]:
    """Block/cell/rollout counts for the manifest + reconciliation."""
    cells = sum(b.n_pairs for b in blocks)
    return {
        "n_blocks": len(blocks),
        "cells_total": cells,
        "rollouts_total": cells * draws,
        "draws_per_cell": draws,
    }


def block_done_path(out_root: Path, block: Block, namespace: str = "blocks") -> Path:
    return out_root / "manifests" / namespace / f"{block.slug}.done.json"


def block_is_done(out_root: Path, block: Block, regime_fp: str, namespace: str = "blocks") -> bool:
    """Resume predicate: done-file present AND regime fingerprint matches.

    A regime mismatch is a HARD refusal, never a silent reuse of wrong cached
    rows (#722 r3).
    """
    path = block_done_path(out_root, block, namespace)
    if not path.exists():
        return False
    rec = json.loads(path.read_text())
    if rec.get("regime_fp") != regime_fp:
        raise RuntimeError(
            f"block {block.key} done-file carries regime_fp={rec.get('regime_fp')!r} "
            f"but this run's regime_fp={regime_fp!r} — refusing to resume across "
            "regimes (quarantine or use a fresh --out-root)"
        )
    if rec.get("key") != block.key:
        raise RuntimeError(f"block done-file key mismatch: {rec.get('key')!r} != {block.key!r}")
    return True


# ── claim-file queue (plan §4.6 mechanical gate 2/3) ──────────────────


def claims_dir(out_root: Path, namespace: str) -> Path:
    return out_root / "claims" / namespace


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _claim_stale(rec: dict, now: float | None = None) -> bool:
    """A claim is stale when its owner is provably dead (same-host pid probe);
    the age key is a CROSS-HOST-ONLY fallback.

    r1 M1: on the SAME host the pid probe is authoritative in BOTH directions
    — a live owner is never stolen by age (blocks have no mid-block heartbeat,
    so any long block would age out mid-run), and a dead owner is reclaimed
    immediately. A same-host WEDGED owner holds its claim until killed (pid
    death is the release), by design."""
    now = time.time() if now is None else now
    if rec.get("host") == socket.gethostname():
        pid = int(rec.get("pid", -1))
        # r2 H2: a claim record with a missing/non-positive pid is DEAD, never
        # live — os.kill(-1, 0) signals the caller's own process GROUP and
        # SUCCEEDS, so _pid_alive(-1) would read a corrupt record as live
        # forever (permanently unstealable).
        if pid <= 0:
            return True
        return not _pid_alive(pid)
    return (now - float(rec.get("ts", 0.0))) > CLAIM_STALE_S


def try_claim(cdir: Path, block: Block, worker_index: int, token: str) -> bool:
    """Atomically claim one block; reclaim a STALE claim; never skip silently.

    Fresh claim: ``O_CREAT | O_EXCL`` — exactly one worker wins. Existing
    claim: parse it (an unparseable claim file is an inconsistent-state HARD
    failure), and when stale, atomically replace it and verify OUR token won
    (two workers can both see stale; ``os.replace`` serializes, last writer
    wins, the read-back arbitrates).
    """
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / f"{block.slug}.claim"
    payload = {
        "key": block.key,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "worker_index": worker_index,
        "ts": time.time(),
        "token": token,
    }
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            rec = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            raise RuntimeError(
                f"unparseable claim file {path} — inconsistent claim state, refusing "
                "to guess (delete it manually after diagnosing the writer)"
            ) from e
        if not _claim_stale(rec):
            return False
        tmp = path.parent / f"{path.name}.tmp.{token}"
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, path)
        # r1 M1 (TOCTOU): two workers can both see stale and both replace;
        # a short randomized settle before the read-back arbitration shrinks
        # the both-read-own-token window, and release_claim's stolen-claim
        # tolerance makes the residual race non-fatal (done-file writes are
        # atomic + idempotent, so a double-run wastes work, never corrupts).
        time.sleep(random.uniform(0.05, 0.25))
        winner = json.loads(path.read_text())
        won = winner.get("token") == token
        if won:
            logger.info(
                "[claims] reclaimed STALE claim %s (was pid=%s host=%s age=%.0fs)",
                block.key,
                rec.get("pid"),
                rec.get("host"),
                time.time() - float(rec.get("ts", 0.0)),
            )
        return won
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    return True


def release_claim(cdir: Path, block: Block, token: str) -> None:
    """Release OUR claim after the done-checkpoint landed (token-verified).

    r1 M1: a STOLEN claim (vanished, or another worker's token) is tolerated
    with a LOUD error log instead of an assert — the block's done-file already
    landed atomically, so the steal wastes duplicate work but must never kill
    THIS worker's whole queue loop. The thief owns the claim file now; leave
    it (the thief's own release removes it)."""
    path = cdir / f"{block.slug}.claim"
    if not path.exists():
        logger.error(
            "[claims] claim %s VANISHED before release — stolen by another worker "
            "(check CLAIM_STALE_S / worker liveness); done-file is intact, continuing",
            block.key,
        )
        return
    rec = json.loads(path.read_text())
    if rec.get("token") != token:
        logger.error(
            "[claims] claim %s owned by ANOTHER worker at release time (token %r != ours; "
            "pid=%s host=%s) — stolen mid-run; done-file is intact, continuing",
            block.key,
            rec.get("token"),
            rec.get("pid"),
            rec.get("host"),
        )
        return
    path.unlink()


def run_claim_queue(
    cfg: RunConfig,
    blocks: list[Block],
    regime_fp: str,
    namespace: str,
    run_one,
    is_done=block_is_done,
    mine=None,
) -> dict:
    """Work-conserving queue: pull the next unclaimed pending block until every
    block is done (crashed workers' claims go stale and are reclaimed).

    ``is_done`` defaults to :func:`block_is_done` (the #722 r3 hard-refusal
    resume predicate). ``phase_capregen_grid`` passes
    :func:`_capregen_block_done`, which PRESERVES that hard refusal and
    additionally treats a pre-regen done record as PENDING.

    ``mine`` (optional, re-evaluated EVERY scan): a predicate naming which
    pending blocks THIS participant may claim. A pending block that is not
    mine is neither claimed nor counted open — when only not-mine blocks
    remain the loop EXITS (their owner finishes them). This is the per-cell
    anchor routing seam (B8 r1 review): an HF worker's ``mine`` excludes
    vLLM-claimed cells read fresh from ``gates/vllm_cells.json``, so a
    parity-PASS landing mid-phase re-routes exactly the not-yet-claimed
    cells and never an already-claimed one."""
    cdir = claims_dir(cfg.out_root, namespace)
    stats = {"ran": 0, "skipped_done": 0, "waits": 0}
    while True:
        ran_this_scan = 0
        n_open = 0
        for block in blocks:
            if is_done(cfg.out_root, block, regime_fp, namespace):
                continue
            if mine is not None and not mine(block):
                continue
            n_open += 1
            token = uuid.uuid4().hex
            if not try_claim(cdir, block, cfg.worker_index, token):
                continue
            try:
                if is_done(cfg.out_root, block, regime_fp, namespace):
                    # Raced completion between scan and claim — nothing to do.
                    continue
                run_one(block)
                stats["ran"] += 1
                ran_this_scan += 1
            finally:
                release_claim(cdir, block, token)
        if n_open == 0:
            break
        if ran_this_scan == 0:
            stats["waits"] += 1
            logger.info(
                "[claims:%s] %d blocks still claimed by other live workers — polling in %.0fs",
                namespace,
                n_open,
                CLAIM_POLL_S,
            )
            time.sleep(CLAIM_POLL_S)
    return stats


# ── bank manifest / regime / resume predicates ────────────────────────


_BANK_TOK = None
_BANK_MANIFEST_CACHE: tuple[dict, str] | None = None


def _bank_tokenizer():
    """Module-cached PRODUCTION tokenizer for bank identity (HF-429 gotcha).

    Bank identity is a property of the PRODUCTION tokenizer (``MODEL_ID``) —
    the ``--tiny`` stand-in config shares the real tokenizer, so the regime
    key never forks between tiny and full runs.
    """
    global _BANK_TOK
    if _BANK_TOK is None:
        from transformers import AutoTokenizer

        _BANK_TOK = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    return _BANK_TOK


def bank_manifest_and_sha() -> tuple[dict, str]:
    """Deterministic 2389 bank manifest + sha (the regime key), CPU + tokenizer.

    ``bank2389.bank_manifest_2389`` enforces the >=30/36 per-cell
    token-identity floor (gate 0a) — a floor breach raises
    ``TokenIdentityFloorError`` BEFORE any GPU spend. Cached per process (the
    build re-tokenizes all 1,404 contexts, ~O(30s) CPU).
    """
    global _BANK_MANIFEST_CACHE
    if _BANK_MANIFEST_CACHE is None:
        manifest = BANK29.bank_manifest_2389(_bank_tokenizer())
        bank_bytes = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
        _BANK_MANIFEST_CACHE = (manifest, _sha256_bytes(bank_bytes))
    return _BANK_MANIFEST_CACHE


def _load_frozen_manifest(cfg: RunConfig) -> dict:
    """The FROZEN bank.json written by ``--phase bank`` (gate 0a artifact)."""
    path = cfg.bank_dir / "bank.json"
    assert path.exists(), f"{path} missing — run `--phase bank` first"
    return json.loads(path.read_text())


def surviving_pairs(manifest: dict) -> list[BANK.Pair2162]:
    """The parent's 1,404 pairs minus the gate-0a token-identity drops."""
    dropped = {row["pair_id"] for row in manifest["dropped_pairs"]}
    pairs = [p for p in BANK.build_pairs() if p.pair_id not in dropped]
    assert len(pairs) == manifest["token_identity"]["n_intact"], (
        len(pairs),
        manifest["token_identity"]["n_intact"],
    )
    return pairs


def no_prefix_ids(manifest: dict) -> frozenset[str]:
    """Contexts with NO pe slot (bare single-turn thinking-off renders)."""
    return frozenset(manifest["no_prefix_context_ids"])


def _write_pe_exclusions(cfg: RunConfig, exclusions: list[dict], scope: str) -> None:
    """Persist the named pe-slot no-prefix exclusions (the phase manifest the
    analysis + per-cell report read; deterministic content, atomic replace —
    concurrent same-content writes are safe)."""
    _write_json_atomic(
        cfg.manifest_dir / "pe_exclusions.json",
        {
            "scope": scope,
            "criterion": (
                "pe slot undefined for no-prefix contexts (prefix_end == 0 under the "
                "thinking-off template; bank.json no_prefix_context_ids) — pair skipped "
                "at pe when context A, context B, or the null-arm donor B is no-prefix"
            ),
            "n_excluded_pair_cells": sum(1 for e in exclusions if e["pair_id"] is not None),
            "n_empty_blocks": sum(1 for e in exclusions if e["pair_id"] is None),
            "exclusions": exclusions,
        },
    )


def regime_fingerprint(cfg: RunConfig, bank_sha: str) -> str:
    """Stable fingerprint of EVERY output-affecting knob (resume key)."""
    import hashlib

    payload = json.dumps(
        {
            "bank_sha": bank_sha,
            "model_id": cfg.model_id,
            "model_revision": cfg.model_revision,
            "tiny": cfg.tiny,
            "n_layers": cfg.n_layers,
            "hidden": cfg.hidden,
            "max_new_tokens": cfg.max_new_tokens,
            # Per-cell cap table (plan §4.7 item 1) — static, part of the
            # regime. Gate-slice RECALIBRATED caps are deliberately NOT in
            # the fp: an up-only mid-run raise recorded per-row (each row
            # carries its own max_new_tokens) — the sanctioned mixed-cap
            # store shape, same as capregen.
            "cell_max_new_tokens": {k: CELL_MAX_NEW_TOKENS[k] for k in sorted(CELL_MAX_NEW_TOKENS)},
            "grid_temperature": GRID_TEMPERATURE,
            "grid_draws": cfg.grid_draws,
            "seed_base": cfg.seed_base,
            "smoke": cfg.smoke,
            "bank_seed": BANK.SEED,
            # B9 (r1 review): the REALIZED generation batch + the frozen
            # share-prefill execution mode are output-affecting (batched HF
            # sampling consumes RNG jointly per chunk; shared-prefill reuses
            # the prompt KV cache), so a resume must never mix shards
            # produced under different values. Generation phases resolve
            # BOTH at entry (_adopt_pilot_gen_batch / _resolve_share_prefill)
            # BEFORE computing their fingerprint; non-generation phases carry
            # the stable CLI defaults (gen_batch=16, armed=False).
            "gen_batch": cfg.gen_batch,
            "share_prefill_armed": cfg.share_prefill_armed,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _phase_done_record(cfg: RunConfig, phase: str, regime_fp: str) -> dict | None:
    """Regime-checked done-record read (missing -> None; mismatch -> raise)."""
    path = cfg.manifest_dir / f"{phase}_done.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    if rec.get("regime_fp") != regime_fp:
        raise RuntimeError(
            f"{phase} done-file carries regime_fp={rec.get('regime_fp')!r} but this "
            f"run's regime_fp={regime_fp!r} — refusing to resume across regimes "
            "(quarantine or use a fresh --out-root)"
        )
    return rec


def bank_is_done(cfg: RunConfig, regime_fp: str) -> bool:
    rec = _phase_done_record(cfg, "bank", regime_fp)
    if rec is None:
        return False
    required = [
        cfg.bank_dir / "bank.json",
        cfg.bank_dir / "bank.meta.json",
        cfg.bank_dir / "token_identity_report.json",
        cfg.bank_dir / "vc_bank.pt",
        cfg.gates_dir / "injection_gate_report.json",
        cfg.gates_dir / "degeneracy_guard_report.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        logger.warning(
            "[bank] done-manifest present but artifacts missing %s — re-running", missing
        )
        return False
    return True


def _sharded_done_record(cfg: RunConfig, phase: str, regime_fp: str) -> dict | None:
    """Width-checked per-worker done-record read (r1 M2): a phase that shards
    ``order[w::num_workers]`` keys its done files per (worker, batch), so a
    cross-width resume (8-GPU run -> 4-GPU fallback re-shard) would silently
    LOSE the contexts of the vanished workers — the width is part of the shard
    identity, so a width mismatch regenerates (the block-queue fingerprint
    deliberately does NOT carry width: blocks are width-invariant units)."""
    rec = _phase_done_record(cfg, phase, regime_fp)
    if rec is None:
        return None
    if int(rec.get("num_workers", -1)) != cfg.num_workers:
        logger.warning(
            "[%s] done num_workers=%s != %d (cross-width resume) — re-running",
            phase,
            rec.get("num_workers"),
            cfg.num_workers,
        )
        return None
    return rec


@dataclass(frozen=True)
class AnchorCellBlock:
    """One bank cell x anchor batch as a claim-queue unit (plan §9: 'P2 ...
    via the claim queue'; B7 r1 review — cell-grain claims keep every
    generate chunk at full ``gen_batch`` because a cell's contexts are never
    strided across workers)."""

    cell: str
    batch: str  # "gate" | "rest"

    @property
    def batch_id(self) -> str:
        """The shard batch-id slot: ``anchors_{batch_id}_w{w}.jsonl``."""
        return f"{self.batch}_{self.cell}"

    @property
    def slug(self) -> str:
        return f"anchor_{self.batch}_{self.cell}"

    @property
    def key(self) -> str:
        return f"anchors:{self.batch}:{self.cell}"


def _group_by_cell(order: list[str], contexts: dict[str, dict]) -> dict[str, list[str]]:
    """Cell -> ORDERED context ids (first-appearance cell order preserved)."""
    by_cell: dict[str, list[str]] = {}
    for cid in order:
        by_cell.setdefault(contexts[cid]["cell"], []).append(cid)
    return by_cell


def _batch_kind(batch_id: str) -> str:
    """Leading batch-kind token of a claim-queue ``batch_id``.

    The batch domain at generation time is ``{gate|rest|parity|vllm}_{cell}``
    (``AnchorCellBlock.batch_id``; capregen passes the OWNING record's
    batch_id verbatim). R2 (r2 review): the row-level ``gate_slice`` label
    derives from this kind — ``batch == "gate"`` could never be true once
    the domain moved to cell grain, so every HF row was written
    ``gate_slice: False`` while the vLLM leg labeled correctly via
    ``cid in gate_id_set`` (the two engines disagreed on one durable field
    in one store). Batch-kind == "gate" is EXACTLY gate-slice context
    membership: gate blocks enumerate gate-slice context ids only, and a
    gate-batch capregen regenerates only that cell's gate contexts.
    """
    return batch_id.split("_", 1)[0]


def _anchor_cell_done(
    cfg: RunConfig, regime_fp: str, batch_id: str, expected_draws: int | None = None
) -> bool:
    """Worker-INDEPENDENT per-cell anchors done predicate (claim-queue grain).

    Any worker may own any cell, so the predicate scans every worker's done
    manifest for this cell-batch and validates regime, draws, artifacts, and
    row count — the #722 r3 hard-refusal shape at cell grain. Excludes the
    vLLM leg's ``*_gen_done.json`` sentinels (generation-complete but
    pre-capture — NOT done)."""
    for m in sorted(cfg.manifest_dir.glob(f"anchors_{batch_id}_w*_done.json")):
        if m.name.endswith("_gen_done.json"):
            continue
        rec = json.loads(m.read_text())
        if rec.get("regime_fp") != regime_fp:
            continue
        if expected_draws is not None and int(rec.get("draws", -1)) != expected_draws:
            logger.warning(
                "[anchors:%s] done draws=%s != %d — treating as NOT done",
                batch_id,
                rec.get("draws"),
                expected_draws,
            )
            continue
        w = int(rec["worker_index"])
        jsonl = cfg.anchors_dir / f"anchors_{batch_id}_w{w}.jsonl"
        va = cfg.anchors_dir / f"va_anchors_{batch_id}_w{w}.pt"
        if not (jsonl.exists() and va.exists()):
            logger.warning(
                "[anchors:%s] done-manifest %s present but artifacts missing", batch_id, m.name
            )
            continue
        n_rows = sum(1 for line in jsonl.open(encoding="utf-8") if line.strip())
        if n_rows != int(rec.get("n_rows", -1)):
            logger.warning(
                "[anchors:%s] done n_rows=%s but jsonl has %d — treating as NOT done",
                batch_id,
                rec.get("n_rows"),
                n_rows,
            )
            continue
        return True
    return False


def _quarantine_orphan_cell_shards(cfg: RunConfig, batch_id: str) -> int:
    """Move crash-orphaned / stale artifacts for one cell-batch OUT of every
    consumer glob before a claim-queue reclaim regenerates it.

    Called ONLY under a held claim whose ``is_done`` read was False, so any
    artifact present for this cell-batch is invalid (partial crash residue, a
    stale regime, or a draws mismatch) — without the sweep a reclaimed cell
    would leave TWO ``anchors_{batch_id}_w*`` shards and trip the judge's
    duplicate-(context_id, draw) assert (the g3 r1 manual-cleanup wedge).
    Destination sits OUTSIDE the uploaded dirs (hub allow_patterns are
    fnmatch and cross "/", so a subdir of anchors_dir would still ride P5)."""
    qroot = cfg.out_root / "stale_anchor_quarantine"
    moved = 0
    globs = (
        (cfg.anchors_dir, f"anchors_{batch_id}_w*.jsonl"),
        (cfg.anchors_dir, f"va_anchors_{batch_id}_w*.pt"),
        (cfg.manifest_dir, f"anchors_{batch_id}_w*_done.json"),
    )
    for d, pat in globs:
        if not d.exists():
            continue
        for p in sorted(d.glob(pat)):
            dest = qroot / d.name / f"{p.name}.orphan-{int(time.time())}-{uuid.uuid4().hex[:8]}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                p.rename(dest)
            except FileNotFoundError:
                continue  # a sibling's rename won — nothing left to move
            moved += 1
            logger.warning("[anchors:%s] quarantined orphan %s -> %s", batch_id, p.name, dest)
    return moved


# r3 C1: width-sharded artifacts are grouped into FAMILIES; a sweep is scoped
# to exactly one family so no phase can ever destroy ANOTHER family's shards
# (r2 C1: phase_margin/phase_upload applied ONE process's width to EVERY
# family — a 1-wide deferred margin leg would have quarantined 7/8 of a valid
# 8-wide anchor store, and the width-less upload leg quarantined w1..wN at
# implicit width 1 before the bulk upload).
# B7 (r1 review): the "anchors" family is GONE from the width-sweep universe —
# anchors now shard at CELL grain via the claim queue (any worker may own any
# cell; the worker index in ``anchors_{batch}_{cell}_w{w}`` is provenance, not
# a stripe), so anchor shards are width-INVARIANT by construction, exactly
# like the vLLM leg's ``anchors_vllm_{cell}`` shards. Only the strided margin
# family still needs the stale-width machinery.
_ARTIFACT_FAMILIES: dict[str, frozenset[str]] = {
    "margin": frozenset({"anchor_margin", "margin_anchors"}),
}
_WIDTH_SHARDED_STEMS = frozenset().union(*_ARTIFACT_FAMILIES.values())
# Done-record stems whose per-width index coverage DEFINES a family's realized
# width (every kind must be complete at the same W).
_FAMILY_DONE_KINDS: dict[str, tuple[str, ...]] = {
    "margin": ("margin_anchors",),
}


def _shard_stem_index(name: str) -> tuple[str, int] | None:
    """(stem head, worker index) of a width-sharded artifact filename, else None.

    Strict stem allowlist (``_WIDTH_SHARDED_STEMS``) so block shards / other
    manifests can never be swept by accident.
    """
    stem = name
    for suffix in ("_done.json", ".jsonl", ".pt"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    else:
        return None
    head, sep, idx = stem.rpartition("_w")
    if not sep or not idx.isdigit() or head not in _WIDTH_SHARDED_STEMS:
        return None
    return head, int(idx)


def _shard_worker_index(name: str) -> int | None:
    """Worker index of a width-sharded anchor/margin artifact filename, else None."""
    parsed = _shard_stem_index(name)
    return None if parsed is None else parsed[1]


def _sweep_stale_width_shards(
    anchors_dir: Path,
    margin_dir: Path,
    manifest_dir: Path,
    out_root: Path,
    num_workers: int,
    *,
    family: str,
) -> int:
    """Quarantine ONE family's prior-width shards + done records (r2 F3, r3 C1).

    Anchor/margin shard names are width-unnamespaced (``anchors_{batch}_w{i}``),
    and the designed cross-width resume (``_sharded_done_record``'s 8-GPU ->
    4-GPU fallback) re-runs only the SURVIVING workers — stale ``w{i}`` files
    with ``i >= num_workers`` would otherwise persist as pure duplicates and
    corrupt every consumer three ways: judge ``load_anchor_rows`` concatenates
    ``anchors_*.jsonl`` (duplicated units into the coherence gate + behavior
    waves), mapshift SUM-accumulates ``va_anchors_*.pt`` (double-counted anchor
    means), analysis ``_load_anchor_va`` dict-keys (silent last-write-wins).

    r3 C1 scoping: only stems of ``_ARTIFACT_FAMILIES[family]`` are eligible —
    a phase sweeps ONLY its OWN family at that family's width, so a narrow
    deferred/salvage leg can never destroy the other family's wider live
    shards. ``num_workers`` MUST be an explicit positive width (CLI-provided
    or done-record-derived); an implicitly-defaulted width raises rather than
    driving a destructive sweep.

    Quarantine, never delete (generated rollout text), into
    ``out_root/stale_width_quarantine/<dirname>/`` — OUTSIDE every uploaded dir
    (hub allow_patterns are fnmatch and cross "/", so a subdir of the
    anchors/margin/manifests dirs would still ride their P5 uploads) and
    outside every consumer glob (all non-recursive). Concurrent workers race
    benignly: a sibling's rename winning leaves nothing to move; the uuid
    suffix keeps same-second re-quarantines from renaming over each other.
    """
    assert family in _ARTIFACT_FAMILIES, family
    if not (isinstance(num_workers, int) and num_workers >= 1):
        raise ValueError(
            f"stale-width sweep needs an explicit positive width, got {num_workers!r} — "
            "an implicitly-defaulted width must never drive a destructive sweep (r3 C1)"
        )
    stems = _ARTIFACT_FAMILIES[family]
    moved = 0
    qroot = out_root / "stale_width_quarantine"
    for d in (anchors_dir, margin_dir, manifest_dir):
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if not p.is_file():
                continue
            parsed = _shard_stem_index(p.name)
            if parsed is None:
                continue
            head, widx = parsed
            if head not in stems or widx < num_workers:
                continue
            dest = qroot / d.name / f"{p.name}.stale-{int(time.time())}-{uuid.uuid4().hex[:8]}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                p.rename(dest)
            except FileNotFoundError:
                logger.info("[stale-width] %s already quarantined by a sibling", p.name)
                continue
            moved += 1
            logger.warning(
                "[stale-width] quarantined prior-width %s shard %s -> %s (num_workers=%d)",
                family,
                p,
                dest,
                num_workers,
            )
    if moved:
        logger.warning("[stale-width] quarantined %d prior-width %s artifacts", moved, family)
    return moved


def _family_realized_width(manifest_dir: Path, family: str) -> int | None:
    """Realized worker width of ONE artifact family, from its OWN done records.

    r3 C1 fix (b): the upload-entry sweep must never take a width from the
    CPU process's ``--num-workers`` (implicitly 1 on the r2 dispatcher path).
    The width is derivable exactly when EVERY done-record kind of the family
    (``_FAMILY_DONE_KINDS``) has worker indexes {0..W-1} all recorded at
    ``num_workers == W`` for exactly one W (mirrors ``_margin_state``'s
    completeness read). Returns None when NO width is complete (family
    absent / mid-run / crashed) — the caller SKIPS the sweep: quarantining on
    a guessed width silently destroys live shards, while surviving duplicates
    fail LOUD in every consumer (judge uniqueness assert, mapshift/analysis
    key checks). Raises RuntimeError when MULTIPLE widths are simultaneously
    complete — an inconsistent store the phase-entry sweeps are designed to
    prevent; sweeping at either width could destroy the live one.
    """
    kinds = _FAMILY_DONE_KINDS[family]
    per_kind: dict[str, dict[int, set[int]]] = {k: {} for k in kinds}
    for kind in kinds:
        for p in sorted(manifest_dir.glob(f"{kind}_w*_done.json")):
            try:
                rec = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
                raise RuntimeError(
                    f"unreadable done record {p} while deriving the {family} family "
                    f"width — refusing to sweep on partial evidence: {e}"
                ) from e
            w = int(rec.get("num_workers", 0))
            idx = int(rec.get("worker_index", -1))
            if w > 0 and idx >= 0:
                per_kind[kind].setdefault(w, set()).add(idx)
    candidate_ws: set[int] = set()
    for d in per_kind.values():
        candidate_ws |= set(d)
    complete = {
        w for w in candidate_ws if all(per_kind[k].get(w, set()) >= set(range(w)) for k in kinds)
    }
    if not complete:
        return None
    if len(complete) > 1:
        raise RuntimeError(
            f"{family} family has MULTIPLE complete widths {sorted(complete)} in "
            f"{manifest_dir} — inconsistent store (a phase-entry sweep should have "
            "quarantined the prior width); refusing to sweep or upload past this"
        )
    return complete.pop()


def _entry_sweep(cfg: RunConfig, family: str) -> int:
    """Family-scoped stale prior-width sweep at a GENERATION phase entry (r3 C1).

    A phase sweeps ONLY its OWN artifact family, at its OWN explicit width.
    An implicitly-defaulted width (no ``--num-workers`` on the CLI) never
    drives a destructive sweep — a width-less manual invocation would
    otherwise quarantine every w1..wN live shard at implicit width 1 (r2 C1).
    A skipped sweep degrades to LOUD downstream failure on true duplicates,
    never silent data loss.
    """
    if not cfg.num_workers_explicit:
        logger.warning(
            "[stale-width] %s: --num-workers not explicitly provided — SKIPPING the "
            "stale-width sweep (an implicit width-1 default must never quarantine "
            "live shards; pass --num-workers to arm the sweep)",
            family,
        )
        return 0
    return _sweep_stale_width_shards(
        cfg.anchors_dir,
        cfg.margin_dir,
        cfg.manifest_dir,
        cfg.out_root,
        cfg.num_workers,
        family=family,
    )


def _upload_entry_sweeps(cfg: RunConfig) -> int:
    """Upload-entry defense-in-depth sweeps, width DERIVED per family (r3 C1).

    Each family's realized width comes from its OWN complete done-record set —
    NEVER from this CPU process's ``--num-workers`` (r2 C1: the dispatcher's
    upload leg ran at implicit width 1, quarantined every w1..wN live shard +
    done record, and ``_upload_dir``'s exact-set verify then passed against
    the POST-sweep local set — a silent self-consistent truncation). An
    underivable width SKIPS that family's sweep (duplicates, if any, fail
    loud in consumers); multiple complete widths raise via
    ``_family_realized_width``.
    """
    moved = 0
    for family in _ARTIFACT_FAMILIES:
        width = _family_realized_width(cfg.manifest_dir, family)
        if width is None:
            logger.info(
                "[upload] %s family width underivable from done records — skipping the "
                "stale-width sweep for this family",
                family,
            )
            continue
        if cfg.num_workers_explicit and cfg.num_workers != width:
            logger.warning(
                "[upload] %s family realized width %d != this process's --num-workers %d "
                "(expected on a narrower deferred/salvage leg) — sweeping at the DERIVED "
                "width",
                family,
                width,
                cfg.num_workers,
            )
        moved += _sweep_stale_width_shards(
            cfg.anchors_dir,
            cfg.margin_dir,
            cfg.manifest_dir,
            cfg.out_root,
            width,
            family=family,
        )
    return moved


def slot_position(ctx_len: int, prefix_end: int, slot: str) -> int:
    """The single edit position (UNPADDED context coordinates) for one slot.

    ``ce`` = the last context token (generation prompt included — the #779
    slot); ``pe`` = the last prefix token (start of the FINAL user turn − 1).

    ``prefix_end == 0`` marks a NO-PREFIX context (bare single-turn
    thinking-off render — the unit-1 cross-unit flag): ``ce`` stays valid,
    ``pe`` is UNDEFINED and raises loud — enumeration
    (:func:`apply_pe_exclusions`) must skip-with-record, never reach here.
    """
    assert slot in SLOTS, slot
    assert 0 <= prefix_end < ctx_len, (ctx_len, prefix_end)
    if slot == "ce":
        return ctx_len - 1
    if prefix_end < 1:
        raise ValueError(
            f"pe slot undefined for a no-prefix context (ctx_len={ctx_len}, "
            "prefix_end=0 under the thinking-off template) — the pe cell must be "
            "SKIPPED-with-record by apply_pe_exclusions, never positioned"
        )
    return prefix_end - 1


def cap_hit(n_completion_tokens: int, max_new_tokens: int) -> bool:
    """Cap-hit telemetry from the re-tokenized completion length (the
    ``generate_batch`` decoded-text proxy, recorded as ``cap_hit_basis``)."""
    return n_completion_tokens >= max_new_tokens


def _finite_or_none(x: float) -> float | None:
    """JSON-safe float: NaN/inf -> None (a bare NaN token in per-draw JSONL is
    the r2 CC2 bug — strict JSON parsers reject it)."""
    return x if math.isfinite(x) else None


# ── config ────────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    phase: str
    out_root: Path
    log_dir: Path
    model_id: str
    model_revision: str
    tiny: bool
    n_layers: int
    hidden: int
    device: str
    gen_batch: int
    capture_batch: int
    max_new_tokens: int
    anchor_draws: int
    grid_draws: int
    seed_base: int
    smoke: bool
    pilot: bool
    force: bool
    force_past_halt_gates: bool
    worker_index: int
    num_workers: int
    upload_mode: str  # "hf" | "local-mirror" | "none"
    upload_every: int
    planned_wall_h: float
    gpu_hours_budgeted: float
    pools_path: Path | None
    best_cells_path: Path | None = None
    # r3 C1 fix (d): True ONLY when --num-workers was explicitly provided on
    # the CLI. Default False so a width whose provenance is unknown can never
    # arm a destructive stale-width sweep (_entry_sweep skips + warns).
    num_workers_explicit: bool = False
    # Cap-hit report + cell-restricted re-gen (registered >2%/cell trigger).
    cap_scope: str = "both"
    capregen_scope: str | None = None
    capregen_batch: str | None = None
    breach_report: Path | None = None
    # #2389: True ONLY when --gen-batch was explicitly provided — otherwise
    # generation phases adopt the pilot's r2-selected gen_batch (plan §4.7
    # item 3) when gates/pilot_gate_report.json carries one.
    gen_batch_explicit: bool = False
    # #2389: True ONLY when --max-new-tokens was explicitly provided (the
    # capregen raised-cap contract). Default False so the per-cell cap table
    # (CELL_MAX_NEW_TOKENS + gate-slice recalibration) governs generation.
    max_new_tokens_explicit: bool = False
    # Plan §4.7 item 5 (pin 2): "off" (default) keeps every generation call
    # on the serial per-draw-prefill path; "auto" arms share_prefill=True
    # ONLY when gates/share_prefill_equivalence.json carries verdict PASS.
    # Resolved ONCE per phase at entry into share_prefill_armed (the freeze
    # point) — FAIL-OPEN: absent/FAIL artifact => serial.
    share_prefill_mode: str = "off"
    share_prefill_armed: bool = False

    @property
    def rollouts_dir(self) -> Path:
        return self.out_root / "rollouts"

    @property
    def gates_dir(self) -> Path:
        return self.out_root / "gates"

    @property
    def va_dir(self) -> Path:
        return self.out_root / "va_store"

    @property
    def anchors_dir(self) -> Path:
        return self.out_root / "anchors"

    @property
    def bank_dir(self) -> Path:
        return self.out_root / "vc_bank"

    @property
    def margin_dir(self) -> Path:
        return self.out_root / "margin"

    @property
    def manifest_dir(self) -> Path:
        return self.out_root / "manifests"

    @property
    def fact_dir(self) -> Path:
        return self.out_root / "f_metrics_actonly"

    @property
    def best_cells_resolved(self) -> Path:
        return self.best_cells_path or (self.out_root / "best_cells_actsel.json")

    @property
    def layers(self) -> list[int]:
        return list(range(self.n_layers))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Issue #2389 pod driver (bank / anchors / grid / margin / fact_tables / "
            "fact_select / stage2 / upload)."
        )
    )
    ap.add_argument(
        "--phase",
        choices=(
            "bank",
            "anchors",
            "grid",
            "margin",
            "fact_tables",
            "fact_select",
            "stage2",
            "upload",
            "cap_report",
            "capregen",
        ),
        help="pipeline phase to run (required unless --import-check / --gate0b-check)",
    )
    ap.add_argument(
        "--cap-scope",
        choices=("anchors", "grid", "both"),
        default="both",
        help="cap_report: which rollout set(s) to aggregate (incremental/partial-safe)",
    )
    ap.add_argument(
        "--capregen-scope",
        choices=("anchors", "grid"),
        default=None,
        help="capregen: which rollout set to re-generate breaching cells for (required)",
    )
    ap.add_argument(
        "--capregen-batch",
        choices=("gate", "rest"),
        default=None,
        help="capregen anchors: which anchors batch to re-generate (REQUIRED for "
        "--capregen-scope anchors; the gate-slice leg is the gate-3 critical path "
        "and the rest-batch leg is deferrable — never collapsed into one "
        "invocation); refused for --capregen-scope grid",
    )
    ap.add_argument(
        "--breach-report",
        type=Path,
        default=None,
        help="capregen: cap-hit report JSON driving the breach list "
        "(default <out-root>/manifests/cap_hit_report_<scope>.json)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import (incl. function-body imports) and exit 0",
    )
    ap.add_argument(
        "--gate0b-check",
        action="store_true",
        help="gate 0b: assert transformers==5.15.0 loads qwen3_5 with 64 resolvable "
        "decoder blocks (pod-side, CPU, tiny from-config model) and exit 0",
    )
    # UPLOAD_PREFIX_EXEMPT: per-issue pod driver, issue-pinned end-to-end (HF_PREFIX + sentinel + out-root); child issues reuse by forking (2162->2329 convention), never runtime prefix override — the #1005 clobber shape cannot arise from this default.
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument(
        "--model-revision",
        default=MODEL_REVISION,
        help="pinned HF revision threaded into EVERY tokenizer/config/weights load",
    )
    ap.add_argument("--tiny", action="store_true", help="from-config tiny CPU model (smoke)")
    ap.add_argument("--tiny-layers", type=int, default=4)
    ap.add_argument("--tiny-hidden", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda | cuda:0 | cpu (default: auto)")
    ap.add_argument(
        "--gen-batch",
        type=int,
        default=None,
        help="rows per generate call (default: the pilot's r2-selected gen_batch when "
        "gates/pilot_gate_report.json carries one, else 16 — plan §4.7 item 3)",
    )
    ap.add_argument("--capture-batch", type=int, default=8)
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="EXPLICIT uniform cap override (the capregen raised-cap contract). "
        "Default: the per-cell table CELL_MAX_NEW_TOKENS (plan §4.7 item 1), "
        f"{MAX_NEW_TOKENS} for cells not in the table",
    )
    ap.add_argument("--anchor-draws", type=int, default=ANCHOR_DRAWS)
    ap.add_argument("--grid-draws", type=int, default=GRID_DRAWS)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--smoke", action="store_true", help="tiny per-arm-class slice")
    ap.add_argument("--pilot", action="store_true", help="grid: timing pilot only")
    ap.add_argument(
        "--share-prefill",
        choices=("off", "auto"),
        default="off",
        help="plan §4.7 item 5 (pin 2): 'auto' arms share_prefill=True in the anchors/grid "
        "generation calls ONLY when gates/share_prefill_equivalence.json (the gate-4b "
        "pod-side battery artifact) carries verdict PASS; 'off' (default) stays serial. "
        "There is deliberately NO unconditional 'on' — the gate artifact is the only "
        "arming path (FAIL-OPEN).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-run a completed bank/anchors/margin phase (resume override ONLY — "
        "it does NOT bypass the plan §7 HALT gates or the pilot refusal; r1 M3)",
    )
    ap.add_argument(
        "--force-past-halt-gates",
        action="store_true",
        help="DANGEROUS: proceed past a FAILED §7 HALT gate (degeneracy / injection / "
        "pilot refusal); never passed by the dispatcher — manual diagnosis only",
    )
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="realized worker width; omitted => shard single-worker (width 1) but NEVER "
        "run a stale-width sweep — an implicitly-defaulted width must not quarantine "
        "live shards (r3 C1)",
    )
    ap.add_argument("--gpu-id", type=int, default=None, help="informational; CVD pins the device")
    ap.add_argument("--upload", choices=("hf", "local-mirror", "none"), default="hf")
    ap.add_argument(
        "--upload-every",
        type=int,
        default=25,
        help="grid: bulk-upload the worker's staged text every N blocks (256 commits/hr cap)",
    )
    ap.add_argument(
        "--pools",
        type=Path,
        default=None,
        help="margin pools JSON (judge-built); grid computes margins inline when present",
    )
    ap.add_argument(
        "--best-cells",
        type=Path,
        default=None,
        help="stage2: fact_select output JSON (default <out-root>/best_cells_actsel.json)",
    )
    ap.add_argument("--planned-wall-h", type=float, default=PLANNED_GEN_TOTAL_WALL_H)
    ap.add_argument("--gpu-hours-budgeted", type=float, default=45.0)
    return ap.parse_args(argv)


def build_config(args: argparse.Namespace) -> RunConfig:
    if args.device:
        device = args.device
    elif args.tiny:
        device = "cpu"
    else:
        device = "cuda:0"
    return RunConfig(
        phase=args.phase,
        out_root=args.out_root,
        log_dir=args.log_dir,
        model_id=args.model_id,
        model_revision=args.model_revision,
        tiny=args.tiny,
        n_layers=args.tiny_layers if args.tiny else N_MODEL_LAYERS_FULL,
        hidden=args.tiny_hidden if args.tiny else HIDDEN_FULL,
        device=device,
        gen_batch=args.gen_batch if args.gen_batch is not None else 16,
        gen_batch_explicit=args.gen_batch is not None,
        capture_batch=args.capture_batch,
        max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else MAX_NEW_TOKENS,
        max_new_tokens_explicit=args.max_new_tokens is not None,
        share_prefill_mode=args.share_prefill,
        anchor_draws=args.anchor_draws,
        grid_draws=args.grid_draws,
        seed_base=args.seed_base,
        smoke=args.smoke,
        pilot=args.pilot,
        force=args.force,
        force_past_halt_gates=args.force_past_halt_gates,
        worker_index=args.worker_index,
        num_workers=args.num_workers if args.num_workers is not None else 1,
        upload_mode=args.upload,
        upload_every=args.upload_every,
        planned_wall_h=args.planned_wall_h,
        gpu_hours_budgeted=args.gpu_hours_budgeted,
        pools_path=args.pools,
        best_cells_path=args.best_cells,
        num_workers_explicit=args.num_workers is not None,
        cap_scope=args.cap_scope,
        capregen_scope=args.capregen_scope,
        capregen_batch=args.capregen_batch,
        breach_report=args.breach_report,
    )


# ── io helpers ────────────────────────────────────────────────────────


@contextmanager
def _atomic_replace(path: Path):
    """Yield a PROCESS-UNIQUE same-directory temp path; ``os.replace`` it
    onto ``path`` on success, best-effort unlink it on failure (a cleanup
    failure never masks the original exception).

    The temp name embeds pid + a uuid fragment: concurrent workers writing
    identical content to ONE shared destination (all 8 grid workers write
    ``manifests/pe_exclusions.json``) must not share a temp name, or one
    worker's replace consumes the shared temp and every later worker dies
    ``FileNotFoundError`` (grid crash 2026-08-16 05:36Z, rc=1). Same-dir
    keeps the replace atomic (one filesystem — never route through /tmp);
    unlink-on-failure keeps orphan ``*.tmp`` residue out of the out-root
    (the upload-verifier residue-sweep surface). Concurrent same-content
    writes stay safe/idempotent: last atomic replace wins with identical
    bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        # Best-effort cleanup: a failed unlink (PermissionError / non-ENOENT
        # OSError) must never displace the ORIGINAL write/replace exception —
        # the bare ``raise`` re-raises it with its traceback intact. Only the
        # SECONDARY cleanup error is suppressed (logged); the fault itself
        # stays loud (r3 finding 1, ``cleanup-can-mask-original``).
        try:
            tmp.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            logger.warning(
                "cleanup unlink of %s failed (%s); propagating original exception",
                tmp,
                cleanup_exc,
            )
        raise


def _write_json_atomic(path: Path, obj) -> None:
    """Atomic idempotent JSON write; safe under concurrent same-content writers."""
    with _atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    """Atomic idempotent JSONL write; safe under concurrent same-content writers."""
    with _atomic_replace(path) as tmp:
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _save_pt_atomic(path: Path, obj) -> None:
    """Atomic idempotent ``torch.save``; safe under concurrent same-content writers."""
    with _atomic_replace(path) as tmp:
        torch.save(obj, tmp)


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _git_sha() -> str:
    """Repo HEAD, degrading (never fail-loud) on a git-less scratch tree (#1902)."""
    env_sha = os.environ.get("EPS_GIT_SHA")
    if env_sha:
        return env_sha
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        env={**os.environ},
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        logger.warning("git rev-parse failed rc=%s — recording unavailable", proc.returncode)
        return "unavailable-no-git-checkout"
    return proc.stdout.strip()


_REPRO_CACHE: dict | None = None


def _repro(cfg: RunConfig) -> dict:
    """Reproducibility metadata carried by every persisted artifact."""
    global _REPRO_CACHE
    if _REPRO_CACHE is None:
        import transformers

        _REPRO_CACHE = {
            "git_commit": _git_sha(),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
        }
    return {
        **_REPRO_CACHE,
        "model_id": cfg.model_id,
        # #2389: the pinned HF revision is RECORDED in every artifact — the
        # parent #2329's report named this exact forward fix ("record the
        # resolved sha into the run digest"); regime_fingerprint already
        # keys on it, this makes it human-readable per artifact.
        "model_revision": cfg.model_revision,
        "tiny": cfg.tiny,
        "smoke": cfg.smoke,
        "n_layers": cfg.n_layers,
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ── model ─────────────────────────────────────────────────────────────


def _model_text_config(mcfg):
    """The decoder text config (qwen3_5 may nest it under ``text_config``)."""
    return getattr(mcfg, "text_config", None) or mcfg


def _shrink_config(mcfg, hidden: int, n_layers: int):
    """Mutate an ``AutoConfig`` to a tiny from-config shape (CPU smoke / gate 0b).

    Shrinks the standard attention dims plus — defensively, only where the
    attribute exists — the qwen3_5 GatedDeltaNet linear-attention head dims
    (the hybrid arch's extra shape knobs). Returns the SAME config object.
    """
    tc = _model_text_config(mcfg)
    tc.hidden_size = hidden
    tc.intermediate_size = 2 * hidden
    tc.num_hidden_layers = n_layers
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    for attr, val in (
        ("head_dim", max(8, hidden // 4)),
        ("linear_num_value_heads", 2),
        ("linear_num_key_heads", 2),
        ("linear_key_head_dim", max(8, hidden // 4)),
        ("linear_value_head_dim", max(8, hidden // 4)),
        ("linear_conv_kernel_dim", 4),
    ):
        if hasattr(tc, attr):
            setattr(tc, attr, val)
    return mcfg


def load_model_and_tokenizer(cfg: RunConfig):
    """Production: bf16 Qwen3.8-27B at the PINNED revision on one device (never
    ``device_map='auto'`` — silent CPU offload, gotchas; ~56 GB weights fit one
    H200). Tiny: a from-config same-arch model on CPU over the REAL vocab-id
    space (config + tokenizer still fetched at the pinned revision)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # loaded ONCE (HF-429 gotcha); revision pinned (plan §4.6)
    tok = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if cfg.tiny:
        mcfg = _shrink_config(
            AutoConfig.from_pretrained(cfg.model_id, revision=cfg.model_revision),
            cfg.hidden,
            cfg.n_layers,
        )
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(mcfg).to(torch.float32)
    else:
        assert torch.cuda.is_available(), "the full grid requires CUDA (use --tiny for CPU smoke)"
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, revision=cfg.model_revision, dtype=torch.bfloat16
        )
    model = model.to(cfg.device)
    realized = _model_text_config(model.config)
    assert realized.hidden_size == cfg.hidden, (realized.hidden_size, cfg.hidden)
    assert realized.num_hidden_layers == cfg.n_layers, (
        realized.num_hidden_layers,
        cfg.n_layers,
    )
    model.eval()
    return model, tok


def _left_pad(rows: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """LEFT-pad token-id rows (the ``generate_batch`` geometry the hook assumes)."""
    assert rows, "empty batch"
    t_max = max(len(r) for r in rows)
    ids = torch.full((len(rows), t_max), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), t_max), dtype=torch.long)
    for b, r in enumerate(rows):
        ids[b, t_max - len(r) :] = torch.tensor(r, dtype=torch.long)
        mask[b, t_max - len(r) :] = 1
    return ids.to(device), mask.to(device)


def _right_pad(rows: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """RIGHT-pad token-id rows (the capture geometry — positions index unpadded)."""
    assert rows, "empty batch"
    t_max = max(len(r) for r in rows)
    ids = torch.full((len(rows), t_max), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), t_max), dtype=torch.long)
    for b, r in enumerate(rows):
        ids[b, : len(r)] = torch.tensor(r, dtype=torch.long)
        mask[b, : len(r)] = 1
    return ids.to(device), mask.to(device)


def eot_tail_ids(tok) -> list[int]:
    """The assistant end-of-turn tail (``<|im_end|>`` + newline) as TOKEN IDS
    (built from ids, never by re-tokenizing a concatenated string)."""
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert isinstance(im_end, int) and im_end >= 0, im_end
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    assert nl, "tokenizer produced no ids for a newline"
    return [im_end, *nl]


# ── P1: v_ce / v_pe bank + degeneracy guard ───────────────────────────


@torch.no_grad()
def capture_bank(cfg: RunConfig, model, tok, order: list[str] | None = None) -> dict:
    """All-layer v_ce per context: one right-padded forward per chunk (ce-only).

    Positions come from the token ids' own offsets (BPE-seam rule): ``ce`` =
    ctx_len − 1 (generation prompt included). The pe SLOT IS DROPPED (plan
    §4.1): no ``v_pe`` state is captured or stored; ``prefix_end`` /
    ``no_prefix`` stay in the record as INFORMATIONAL metadata (lineage
    comparability + the inert pe-exclusion machinery).

    ``order`` (default: every bank context) is the 8-way claim-queue seam
    (plan E3): each worker captures its claimed context chunks into a part
    file; worker 0 merges (``phase_bank``).
    """
    contexts = BANK.build_contexts()
    if order is None:
        order = list(contexts)
    ctx_ids = {cid: BANK29.context_token_ids_2389(tok, contexts[cid]) for cid in order}
    prefix_ends = {cid: BANK29.prefix_end_index_2389(tok, ids) for cid, ids in ctx_ids.items()}
    layers = cfg.layers
    pad_id = tok.pad_token_id
    records: dict[str, dict] = {}
    for start in range(0, len(order), cfg.capture_batch):
        chunk = order[start : start + cfg.capture_batch]
        ids, mask = _right_pad([ctx_ids[c] for c in chunk], pad_id, cfg.device)
        captured = extract_layer_activations(model, ids, layers, attention_mask=mask)
        for j, cid in enumerate(chunk):
            ctx_len = len(ctx_ids[cid])
            pe = prefix_ends[cid]
            assert 0 <= pe < ctx_len, (cid, ctx_len, pe)
            v_ce = torch.stack([captured[layer][j, ctx_len - 1] for layer in layers])
            assert v_ce.shape == (len(layers), cfg.hidden), v_ce.shape
            ctx = contexts[cid]
            records[cid] = {
                "context_id": cid,
                "cell": ctx["cell"],
                "value_id": ctx["value_id"],
                "carrier": ctx["carrier"],
                "ctx_len": ctx_len,
                "prefix_end": pe,
                "no_prefix": pe == 0,
                "v_ce": v_ce.float().cpu(),
            }
        del captured
        if (start // cfg.capture_batch) % 20 == 0:
            logger.info(
                "[bank] unit %d/%d contexts elapsed",
                min(start + cfg.capture_batch, len(order)),
                len(order),
            )
    assert len(records) == len(order), (len(records), len(order))
    return {"layers": layers, "per_context": records}


def _slot_state(rec: dict, slot: str) -> torch.Tensor:
    """``(L, H)`` slot state for one context."""
    return rec["v_pe"] if slot == "pe" else rec["v_ce"]


def _degenerate_token_prefixes(tok, cids: set[str]) -> dict[str, tuple[list[int], int]]:
    """``cid -> (token ids, prefix_end)`` for the degenerate-cell contexts.

    Same helpers + call shapes as ``capture_bank`` (``build_contexts`` /
    ``context_token_ids_2162`` / ``prefix_end_index_multi``); token ids are
    exact and batch-invariant, unlike the captured bf16 states.
    """
    contexts = BANK.build_contexts()
    out: dict[str, tuple[list[int], int]] = {}
    for cid in sorted(cids):
        ids = BANK29.context_token_ids_2389(tok, contexts[cid])
        out[cid] = (ids, BANK29.prefix_end_index_2389(tok, ids))
    return out


def _bank_system_presence() -> dict[str, bool]:
    """``cid -> True`` iff the FROZEN bank context carries a system message.

    Derived deterministically from ``BANK.build_contexts()`` — never from the
    in-memory capture-bank dict (which carries only per-context states). An
    absent/None ``system`` renders BARE under the Qwen3.5 thinking-off
    template (no default system turn is inserted), so one-sided system
    absence is the template-level explanation for a one-sided ``no_prefix``
    flag (2026-08-16 rc=23 halt: ``persona_prompted`` v2 is the deliberate
    NO-PERSONA control arm, ``system: null`` in the frozen bank).
    """
    return {cid: ctx.get("system") is not None for cid, ctx in BANK.build_contexts().items()}


def run_degeneracy_guard(
    bank: dict,
    pairs: list[BANK.Pair2162],
    tok=None,
    *,
    token_prefixes: dict[str, tuple[list[int], int]] | None = None,
    system_presence: dict[str, bool] | None = None,
) -> dict:
    """Plan §7 gate 2 (ce-only fork) — ce distinctness vs the span-locus registry.

    At the ce slot (the LAST prompt token) the state has consumed the ENTIRE
    varied span, so NO cell is pre-declared degenerate at ce (the registry's
    ``DEGENERATE_AT_PE`` set is a pe-slot property and the pe slot is DROPPED,
    plan §4.1). The gate is therefore: every pair's two ce states are
    DISTINCT (``ce_cos < DEGENERACY_COS_MIN``), and the REALIZED
    degenerate-at-ce cell set equals the PRE-DECLARED one (EMPTY) — plan §7
    gate 2's "set-equality of realized vs pre-declared degenerate cells at
    ce". Any realized-degenerate pair is a BANK/capture defect (HALT), never
    a runtime m adjustment.

    ``tok`` / ``token_prefixes`` / ``system_presence`` are accepted for
    interface parity with the parent (#2329) guard; the pe-side premise
    machinery they fed is inert under ce-only and they are not consumed.
    """
    del tok, token_prefixes, system_presence  # parent-interface parity (pe legs dropped)
    recs = bank["per_context"]
    violations: list[dict] = []
    realized_degenerate_cells: set[str] = set()
    n_checked = 0
    for pair in pairs:
        ra, rb = recs[pair.a], recs[pair.b]
        ce_cos = float(safe_cosine(ra["v_ce"].flatten(), rb["v_ce"].flatten()))
        n_checked += 1
        if not (ce_cos < DEGENERACY_COS_MIN):
            realized_degenerate_cells.add(pair.cell)
            violations.append(
                {
                    "pair_id": pair.pair_id,
                    "cell": pair.cell,
                    "ce_cos": ce_cos,
                    "flag": "distinctness_ce",
                }
            )
    report = {
        "criterion": "span-locus degeneracy guard, ce-only (plan §7 gate 2)",
        "bar_cos": DEGENERACY_COS_MIN,
        "degenerate_criterion": (
            "ce distinctness for EVERY pair (ce_cos < bar); realized vs "
            "pre-declared degenerate-at-ce set-equality (declared set: empty)"
        ),
        "declared_degenerate_cells_ce": [],
        "realized_degenerate_cells_ce": sorted(realized_degenerate_cells),
        "n_pairs_checked": n_checked,
        "n_violations": len(violations),
        "violations": violations[:50],
        "passed": not violations,
    }
    return report


# ── arm payloads (all-28-layer full-state replace, plan §4.2) ─────────


def payload_for_arm(
    bank: dict,
    pair: BANK.Pair2162,
    slot: str,
    arm: str,
    donor_maps: dict[str, dict[str, str]],
    pairs_by_id: dict[str, BANK.Pair2162],
) -> tuple[torch.Tensor, str | None]:
    """``((1, L, H) payload, donor_pair_id)`` for one (pair, slot, arm).

    - ``steered``: the pair's OWN donor-context state V_slot(B) (full-state
      replace target).
    - ``shuffled``: V_slot(B) of the seeded VALUE-CONSTRAINED same-cell donor
      (``donor_assignment_2162``: donor-B-value != recipient-B-value HARD),
      norm-matched per layer to the recipient's own V_slot(B) — the parent's
      realized replace-cell null (`issue2094_run._donor_payload`).
    - ``crosstype``: same, with the cross-type donor (matched-content route
      families excluded at assignment time).
    """
    recs = bank["per_context"]
    if slot == "pe":
        # Enumeration backstop (unit-1 no-prefix flag): a zero v_pe row must
        # never be consumed as a state — apply_pe_exclusions owns the skip.
        assert not recs[pair.a].get("no_prefix") and not recs[pair.b].get("no_prefix"), (
            pair.pair_id,
            "pe payload requested for a no-prefix pair (enumeration bug)",
        )
    recipient = _slot_state(recs[pair.b], slot).unsqueeze(0)  # (1, L, H)
    if arm == "steered":
        return recipient.clone(), None
    donor_map = donor_maps["shuffled" if arm == "shuffled" else "crosstype"]
    donor_id = donor_map[pair.pair_id]
    donor = pairs_by_id[donor_id]
    if slot == "pe":
        assert not recs[donor.b].get("no_prefix"), (
            donor_id,
            "no-prefix DONOR at pe (enumeration bug — pe_excluded_reason covers donors)",
        )
    donor_state = _slot_state(recs[donor.b], slot).unsqueeze(0)
    return BANK94.norm_match(donor_state, recipient), donor_id


def _arm_hook_all_layers(
    model,
    cfg: RunConfig,
    row_lengths: list[int],
    positions: list[tuple[int, ...]],
    per_row_payload: list[torch.Tensor],
    expected_prompt_len: int,
):
    """Install + arm the all-layer replace hook stack for one batch.

    ``per_row_payload[b]`` is ``(1, L, H)``; each layer's hook receives its own
    ``(1, H)`` slice. Mode is ALWAYS ``replace`` at alpha=1.0 (Stage 1).
    """
    layers = tuple(cfg.layers)
    stack = joint_hooks(model, list(layers))
    per_layer = [[p[:, layer, :].contiguous() for p in per_row_payload] for layer in layers]
    stack.install()
    stack.arm_batch_per_layer(row_lengths, positions, per_layer, mode="replace", alpha=1.0)
    stack.arm(expected_prompt_len)
    return stack


# ── P1 gate: injection exactness ──────────────────────────────────────


def _gate_spot_specs(
    pairs: list[BANK.Pair2162],
    np_ids: frozenset[str] | set[str] = frozenset(),
    donor_maps: dict[str, dict[str, str]] | None = None,
    pairs_by_id: dict[str, BANK.Pair2162] | None = None,
) -> list[dict]:
    """12 spot cells spanning arm x carrier-class / crossed / degenerate cell
    classes, ALL at the ce slot (plan §7 gate 1; ce-only fork — the parent's
    pe spots are remapped to ce on the same cells/arms). ``np_ids`` /
    ``donor_maps`` filtering is inert at ce (kept for interface parity)."""
    donor_maps = donor_maps or {"shuffled": {}, "crosstype": {}}
    pairs_by_id = pairs_by_id or {p.pair_id: p for p in pairs}
    by_cell = BANK.pairs_by_cell(pairs)
    reps = smoke_cells()
    cls_rep: dict[str, str] = {}
    for cell in reps:
        base = BANK.base_type_of(cell)
        if cell == base:
            cls_rep.setdefault(BANK.CARRIER_CLASS[base], cell)
    conflict = next(c for c in reps if c.startswith("conflict_"))
    recency = next(c for c in reps if c.startswith("recency_"))
    load = next(c for c in reps if c.startswith("load_"))
    spec: list[tuple[str, str, str, int]] = [
        (cls_rep["P"], "ce", "steered", 0),
        (cls_rep["P"], "ce", "shuffled", 1),
        (cls_rep["E"], "ce", "crosstype", 0),
        (cls_rep["ICL"], "ce", "steered", 2),
        ("query_content", "ce", "steered", 0),
        ("query_content", "ce", "shuffled", 1),
        ("persona_role_header", "ce", "steered", 0),
        ("language_implied", "ce", "steered", 0),
        (conflict, "ce", "steered", 0),
        (recency, "ce", "steered", 0),
        (load, "ce", "shuffled", 0),
        (cls_rep.get("P12", cls_rep["P"]), "ce", "crosstype", 3),
    ]
    out = []
    for cell, slot, arm, k in spec:
        cell_pairs = sorted(by_cell[cell], key=lambda p: p.pair_id)
        if slot == "pe" and np_ids:
            cell_pairs = [
                p
                for p in cell_pairs
                if pe_excluded_reason(p, arm, np_ids, donor_maps, pairs_by_id) is None
            ]
            assert cell_pairs, (cell, slot, arm, "no pe-runnable pair for gate spot")
        out.append(
            {"cell": cell, "slot": slot, "arm": arm, "pair": cell_pairs[k % len(cell_pairs)]}
        )
    assert len(out) == 12, len(out)
    return out


def _gate_second_row_pool(
    pairs: list[BANK.Pair2162],
    spot_pair: BANK.Pair2162,
    slot: str,
    arm: str,
    ctx_ids: dict[str, list[int]],
    np_ids: frozenset[str] | set[str],
    donor_maps: dict[str, dict[str, str]],
    pairs_by_id: dict[str, BANK.Pair2162],
    recs: dict[str, dict],
) -> list[BANK.Pair2162]:
    """Second-row candidates for one gate spot: a DIFFERENT pair whose
    A-context length differs (so the padded-offset math is exercised rather
    than degenerate), restricted to pairs whose payload dependencies RESOLVE
    for ``arm`` (#2389 crash-fix r1): a non-steered payload dereferences the
    candidate's OWN donor (``pairs_by_id[donor_map[pair_id]]`` + the donor's
    captured B state in ``recs``), and under ``--smoke`` the B4 slice
    extension covers donor closure only for the 12 SPOT pairs — an unfiltered
    pool KeyErrors inside ``payload_for_arm`` (pod incident 2026-08-23:
    ``KeyError: 'conflict_persona_rev::i2d1-i1d2::d1'``). The resolvability
    check runs BEFORE ``pe_excluded_reason``, which dereferences the same
    donor. Production-invariant by construction: the full pair set resolves
    every donor, so the filter keeps all candidates there. pe spots
    additionally require the second row to be pe-runnable (unit-1 flag)."""
    candidates = [
        p
        for p in pairs
        if p.pair_id != spot_pair.pair_id and len(ctx_ids[p.a]) != len(ctx_ids[spot_pair.a])
    ]
    if arm == "steered":
        resolvable = candidates  # steered payloads read only the pair's OWN captured B
    else:
        donor_map = donor_maps["shuffled" if arm == "shuffled" else "crosstype"]

        def _resolves(p: BANK.Pair2162) -> bool:
            donor = pairs_by_id.get(donor_map.get(p.pair_id, ""))
            return donor is not None and donor.b in recs

        resolvable = [p for p in candidates if _resolves(p)]
        if len(resolvable) < len(candidates):
            logger.info(
                "[gate] spot %s/%s/%s: second-row donor-closure filter kept %d/%d "
                "candidates (candidate donor pair or its captured B state absent — "
                "expected under a sliced --smoke bank; on a FULL bank this indicates "
                "a donor-map/capture regression)",
                spot_pair.cell,
                slot,
                arm,
                len(resolvable),
                len(candidates),
            )
    return [
        p
        for p in resolvable
        if slot != "pe" or pe_excluded_reason(p, arm, np_ids, donor_maps, pairs_by_id) is None
    ]


@torch.no_grad()
def run_injection_gate(
    cfg: RunConfig,
    model,
    tok,
    bank: dict,
    pairs: list[BANK.Pair2162],
    donor_maps: dict[str, dict[str, str]],
    *,
    contexts: dict[str, dict] | None = None,
    ids_fn=None,
    spots: list[dict] | None = None,
    payload_fn=None,
) -> dict:
    """Plan §7 gate 1 — the realized edit equals the intended donor state at
    the intended (row, position, layer) and NOWHERE else.

    The keyword-only ``contexts`` / ``ids_fn`` / ``spots`` / ``payload_fn``
    seams (defaults = this module's own registries — byte-equivalent for every
    existing caller) let the #2162 LADDER driver reuse this gate verbatim over
    its own bank (``scripts/issue2162_ladder.py``; ladder-plan §4.6 "IMPORTS,
    never re-implements, the injection-gate helper"). ``spots`` rows keep the
    ``{"cell", "slot", "arm", "pair"}`` shape; ``payload_fn`` keeps
    :func:`payload_for_arm`'s call signature.

    Stage 1 is REPLACE at every layer, so the exactness read is ABSOLUTE (the
    hooked state at the edited position IS the payload — no incremental
    references needed, unlike the parent's additive joint cells). Three legs
    per spot: (1) realized-vs-expected cosine + norm ratio per layer at the
    edited position; (2) the hook's own ``realized_edits`` telemetry vs the
    payload; (3) off-target — every NON-edited position at the SHALLOWEST
    layer unchanged within ``GATE_OFFTARGET_REL_MAX`` (deeper layers
    legitimately propagate the edit through attention). A fourth CAPTURE-
    ARMING leg re-forwards the same rows RIGHT-padded under the capture
    pass's ``row_lengths=[T]*B`` arming so the V_a/margin TF geometry is
    gate-verified too.
    """
    contexts = BANK.build_contexts() if contexts is None else contexts
    ids_fn = BANK29.context_token_ids_2389 if ids_fn is None else ids_fn
    payload_fn = payload_for_arm if payload_fn is None else payload_fn
    ctx_ids = {cid: ids_fn(tok, c) for cid, c in contexts.items()}
    pad_id = tok.pad_token_id
    recs = bank["per_context"]
    pairs_by_id = {p.pair_id: p for p in pairs}
    # No-prefix set (unit-1 flag) from the captured bank records: pe spots and
    # pe second rows must stay pe-runnable.
    np_ids = frozenset(cid for cid, r in recs.items() if r.get("no_prefix"))
    spots = _gate_spot_specs(pairs, np_ids, donor_maps, pairs_by_id) if spots is None else spots
    results: list[dict] = []
    for spot in spots:
        pair: BANK.Pair2162 = spot["pair"]
        slot, arm = spot["slot"], spot["arm"]
        # Second row: a DIFFERENT pair whose A-context length differs, so the
        # padded-offset math is exercised rather than degenerate; candidates
        # are restricted to payload-RESOLVABLE pairs for this spot's arm
        # (donor closure — crash-fix r1; smoke-only in effect,
        # production-invariant: see _gate_second_row_pool).
        others = _gate_second_row_pool(
            pairs, pair, slot, arm, ctx_ids, np_ids, donor_maps, pairs_by_id, recs
        )
        batch_pairs = [pair] + ([others[0]] if others else [])
        rows = [ctx_ids[p.a] for p in batch_pairs]
        row_lengths = [len(r) for r in rows]
        positions: list[tuple[int, ...]] = []
        payloads: list[torch.Tensor] = []
        donor_ids: list[str | None] = []
        for p in batch_pairs:
            payload, donor_id = payload_fn(bank, p, slot, arm, donor_maps, pairs_by_id)
            rec = recs[p.a]
            positions.append((slot_position(rec["ctx_len"], rec["prefix_end"], slot),))
            payloads.append(payload)
            donor_ids.append(donor_id)

        # Leg 1+2: LEFT-padded (generation geometry).
        ids, mask = _left_pad(rows, pad_id, cfg.device)
        t_pad = int(ids.shape[1])
        base = extract_layer_activations(model, ids, cfg.layers, attention_mask=mask)
        base_cpu = {la: base[la].detach().float().cpu() for la in cfg.layers}
        del base
        stack = _arm_hook_all_layers(model, cfg, row_lengths, positions, payloads, t_pad)
        try:
            hooked = extract_layer_activations(model, ids, cfg.layers, attention_mask=mask)
            hooked_cpu = {la: hooked[la].detach().float().cpu() for la in cfg.layers}
            del hooked
            telemetry = stack.realized_edits
        finally:
            stack.remove()
        assert telemetry, f"spot {spot['cell']}/{slot}/{arm}: hook never applied an edit"

        cos_min, ratio_lo, ratio_hi = 1.0, math.inf, 0.0
        for b in range(len(batch_pairs)):
            off = t_pad - row_lengths[b]
            padded = positions[b][0] + off
            for layer in cfg.layers:
                expected = payloads[b][0, layer, :]
                realized = hooked_cpu[layer][b, padded]
                cos = float(safe_cosine(realized, expected))
                n_exp = float(expected.norm())
                ratio = float(realized.norm()) / n_exp if n_exp > 0 else float("nan")
                cos_min = min(cos_min, cos)
                ratio_lo, ratio_hi = min(ratio_lo, ratio), max(ratio_hi, ratio)
        tele_cos_min = 1.0
        for record in telemetry:
            b, layer = record["row"], record["layer"]
            applied = record["applied"]  # (1, H) fp32 cpu
            assert record["positions_unpadded"] == list(positions[b]), (
                record["positions_unpadded"],
                positions[b],
            )
            expected = payloads[b][0, layer, :]
            tele_cos_min = min(tele_cos_min, float(safe_cosine(applied[0], expected)))

        # Leg 3 — off-target at the shallowest edit layer (layer 0): the edit
        # at position p cannot change other positions' SAME-layer output.
        edited = {(b, positions[b][0] + (t_pad - row_lengths[b])) for b in range(len(batch_pairs))}
        layer0 = cfg.layers[0]
        diff = (hooked_cpu[layer0] - base_cpu[layer0]).norm(dim=-1)  # (B, T)
        denom = base_cpu[layer0].norm(dim=-1).clamp_min(1e-6)
        rel = (diff / denom).clone()
        for b, padded in edited:
            rel[b, padded] = 0.0
        offtarget = float(rel.max())

        # Leg 4 — capture-arming parity: the RIGHT-padded ``[T]*B`` arming the
        # hooked V_a / margin TF passes use realizes the SAME payload.
        ids_r, mask_r = _right_pad(rows, pad_id, cfg.device)
        t_r = int(ids_r.shape[1])
        stack = _arm_hook_all_layers(model, cfg, [t_r] * len(rows), positions, payloads, t_r)
        try:
            hooked_r = extract_layer_activations(model, ids_r, cfg.layers, attention_mask=mask_r)
            capture_cos_min = 1.0
            for b in range(len(batch_pairs)):
                pos = positions[b][0]  # right-pad: unpadded == padded
                for layer in cfg.layers:
                    realized = hooked_r[layer][b, pos].detach().float().cpu()
                    capture_cos_min = min(
                        capture_cos_min, float(safe_cosine(realized, payloads[b][0, layer, :]))
                    )
            del hooked_r
        finally:
            stack.remove()

        ok = (
            cos_min >= GATE_COS_MIN
            and tele_cos_min >= GATE_COS_MIN
            and capture_cos_min >= GATE_COS_MIN
            and GATE_NORM_RATIO_LO <= ratio_lo
            and ratio_hi <= GATE_NORM_RATIO_HI
            and offtarget <= GATE_OFFTARGET_REL_MAX
        )
        results.append(
            {
                "cell": spot["cell"],
                "slot": slot,
                "arm": arm,
                "pair_id": pair.pair_id,
                "donor_pair_id": donor_ids[0],
                "batch_rows": [p.pair_id for p in batch_pairs],
                "cos_min": cos_min,
                "telemetry_cos_min": tele_cos_min,
                "capture_arming_cos_min": capture_cos_min,
                "norm_ratio_lo": ratio_lo,
                "norm_ratio_hi": ratio_hi,
                "offtarget_rel_max": offtarget,
                "ok": bool(ok),
            }
        )
        logger.info(
            "[gate] %-28s %-3s %-9s cos=%.6f tele=%.6f cap=%.6f ratio=[%.5f,%.5f] off=%.2e ok=%s",
            spot["cell"],
            slot,
            arm,
            cos_min,
            tele_cos_min,
            capture_cos_min,
            ratio_lo,
            ratio_hi,
            offtarget,
            ok,
        )
        del base_cpu, hooked_cpu
    report = {
        "criterion": "injection-exactness gate (plan §7 gate 1)",
        "bars": {
            "cos_min": GATE_COS_MIN,
            "norm_ratio": [GATE_NORM_RATIO_LO, GATE_NORM_RATIO_HI],
            "offtarget_rel_max": GATE_OFFTARGET_REL_MAX,
        },
        "spots": results,
        "n_spots": len(results),
        "n_spots_failed": sum(1 for r in results if not r["ok"]),
        "passed": all(r["ok"] for r in results),
        "repro": _repro(cfg),
    }
    return report


@dataclass(frozen=True)
class CaptureChunk:
    """One claimable P1 capture unit (plan E3): a fixed-size context chunk."""

    index: int
    context_ids: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"bank_capture|c{self.index:03d}"

    @property
    def slug(self) -> str:
        return block_slug(self.key)


# Fixed enumeration unit so claim identity is stable across widths/resumes
# (never derived from gen/capture batch knobs).
BANK_CAPTURE_CHUNK = 64


def enumerate_capture_chunks(order: list[str] | None = None) -> list[CaptureChunk]:
    """The P1 claim-queue units: 1,404 contexts in fixed 64-context chunks."""
    if order is None:
        order = list(BANK.build_contexts())
    return [
        CaptureChunk(i // BANK_CAPTURE_CHUNK, tuple(order[i : i + BANK_CAPTURE_CHUNK]))
        for i in range(0, len(order), BANK_CAPTURE_CHUNK)
    ]


def _gate0c_hf_write_canary(cfg: RunConfig) -> None:
    """Plan §7 gate 0c: prove the HF WRITE destination accepts pushes BEFORE
    any generation spend (the canonical data repo sits at the 1M-file cap;
    the run needs `EPM_2389_DATA_WRITE_REPO` / overflow routing to land
    anywhere). Uploads a tiny canary under the gates prefix and asserts it
    resolves via `HfApi().file_exists` (retry-wrapped). `--upload none /
    local-mirror` records a SKIP (smoke/local runs)."""
    payload = {
        "issue": 2389,
        "hf_prefix": HF_PREFIX,
        "write_repo": HF_DATA_WRITE_REPO,
        "upload_mode": cfg.upload_mode,
        "host": socket.gethostname(),
        "repro": _repro(cfg),
    }
    local = cfg.gates_dir / "hf_write_canary.json"
    _write_json_atomic(local, payload)
    if cfg.upload_mode != "hf":
        logger.info("[gate0c] upload_mode=%s — canary written locally only", cfg.upload_mode)
        return
    _upload_dir(cfg, cfg.gates_dir, f"{HF_PREFIX}/analysis_tensors/gates", ["hf_write_canary.json"])
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import retry_transient

    dest = f"{HF_PREFIX}/analysis_tensors/gates/hf_write_canary.json"
    ok = retry_transient(
        # HUB_VERIFY_RETRY_EXEMPT: call is wrapped in hub.retry_transient (this expr)
        lambda: HfApi().file_exists(HF_DATA_WRITE_REPO, dest, repo_type="dataset"),
        what="gate-0c HF write-canary file_exists verify",
    )
    assert ok, (
        f"gate 0c: canary {dest} did not resolve on {HF_DATA_WRITE_REPO} — the HF write "
        "path is closed; fix EPM_2389_DATA_WRITE_REPO / overflow routing BEFORE generation"
    )
    logger.info("[gate0c] HF write canary OK -> %s/%s", HF_DATA_WRITE_REPO, dest)


def _bank_part_path(cfg: RunConfig, chunk: CaptureChunk) -> Path:
    return cfg.bank_dir / f"vc_bank_part_{chunk.slug}.pt"


# Synthetic smoke-only capture chunk carrying the injection-gate spot
# dependencies (B4) — index far outside the real c000..c021 range so its
# claim/done identity can never collide with a production chunk.
SMOKE_GATE_CHUNK_INDEX = 999
# Grid-closure sibling (crash-fix r2) — its OWN distinct identity: a retained
# smoke out-root's c999 done-file (captured by a pre-r2 run) must never
# satisfy the NEW grid-closure chunk, so a resume re-captures exactly the
# delta contexts and nothing else.
SMOKE_GRID_CHUNK_INDEX = 998


def _smoke_gate_slice_extension(
    all_contexts: list[str],
    sliced: set[str],
    pairs_full: list[BANK.Pair2162],
    donor_maps: dict[str, dict[str, str]],
) -> tuple[list[dict], list[str]]:
    """B4 (r1 review): the injection gate's 12 spot cells span the WHOLE bank,
    so the 2-chunk smoke slice structurally cannot supply them — the smoke
    died at ``by_cell[cell]`` BEFORE the gate it claims to exercise. Returns
    (the FULL-bank gate spots — the exact production selection — and the
    extra contexts the smoke capture must add: each spot pair's a/b plus the
    non-steered arms' donor pair a/b, whose states ``payload_for_arm``
    dereferences)."""
    pairs_by_id = {p.pair_id: p for p in pairs_full}
    spots = _gate_spot_specs(pairs_full, donor_maps=donor_maps, pairs_by_id=pairs_by_id)
    need: set[str] = set()
    for s in spots:
        p = s["pair"]
        need.update((p.a, p.b))
        if s["arm"] != "steered":
            donor_map = donor_maps["shuffled" if s["arm"] == "shuffled" else "crosstype"]
            donor = pairs_by_id[donor_map[p.pair_id]]
            need.update((donor.a, donor.b))
    missing = [c for c in all_contexts if c in need and c not in sliced]
    return spots, missing


def _smoke_grid_slice_extension(
    all_contexts: list[str],
    sliced: set[str],
    pairs_full: list[BANK.Pair2162],
    donor_maps: dict[str, dict[str, str]],
    np_ids: frozenset[str] | set[str],
) -> tuple[set[str], list[str]]:
    """Grid sibling of B4 (crash-fix r2): ``phase_grid`` composes its smoke
    blocks over the FULL surviving pair set
    (``apply_pe_exclusions(smoke_blocks(surviving_pairs(manifest)), ...)`` —
    no capture filter), then ``_block_cells`` / ``payload_for_arm``
    dereference the smoke-sliced bank — so every smoke block pair's a/b,
    plus (non-steered arms) its donor pair's a/b, must be captured or the
    grid leg KeyErrors AFTER the bank/pilot/anchors/vLLM smoke spend (pod
    incident 2026-08-23 #2 — same closure class as the B4 gate fix). The
    block set is composed EXACTLY as phase_grid composes it (pe-exclusion
    semantics included), so the need-set covers what ``_block_cells``
    actually dereferences; the ``--pilot`` leg (``blocks[:1]``) and the
    capregen-grid / margin re-compositions are subsets of the same block
    set. Returns ``(the grid leg's full context dereference set, the extra
    contexts the smoke capture must add — dedup'd against ``sliced``, i.e.
    the base chunks + the B4 gate extension)``."""
    pairs_by_id = {p.pair_id: p for p in pairs_full}
    blocks, _excl = apply_pe_exclusions(smoke_blocks(pairs_full), np_ids, donor_maps, pairs_full)
    need: set[str] = set()
    for block in blocks:
        donor_map = (
            None
            if block.arm == "steered"
            else donor_maps["shuffled" if block.arm == "shuffled" else "crosstype"]
        )
        for pid in block.pair_ids:
            p = pairs_by_id[pid]
            need.update((p.a, p.b))
            if donor_map is not None:
                donor = pairs_by_id[donor_map[pid]]
                need.update((donor.a, donor.b))
    missing = [c for c in all_contexts if c in need and c not in sliced]
    return need, missing


def phase_bank(cfg: RunConfig) -> int:
    """P1: bank.json + vc_bank.pt + degeneracy guard + injection gate.

    8-WAY (plan E3): the state capture is sharded across every worker via the
    claim-file queue (fixed 64-context chunks -> per-chunk part files); the
    queue exhausts only when EVERY chunk is done, then worker 0 merges the
    parts, cross-checks coverage, and runs the HALT gates. Non-zero workers
    exit RC_OK after the queue drains. Idempotent: a completed same-regime
    bank is SKIPPED at entry, BEFORE the model load; ``--force`` deliberately
    re-runs (worker 0 quarantines stale parts first). Gate failures write
    their report JSONs (under ``gates/``, plan §6.5 P1 outputs) and exit
    DISTINCT rcs (never a bare rc=1).
    """
    logger.info("[phase=bank] worker=%d/%d", cfg.worker_index, cfg.num_workers)
    manifest, bank_sha = bank_manifest_and_sha()
    regime_fp = regime_fingerprint(cfg, bank_sha)
    if not cfg.force and bank_is_done(cfg, regime_fp):
        logger.info("[bank] already done for this regime — skipping (--force re-runs)")
        logger.info("[phase=bank_done]")
        return RC_OK
    if cfg.force and (cfg.manifest_dir / "bank_done.json").exists():
        logger.info("[bank] --force set: deliberately re-running a done bank phase")
        if cfg.worker_index == 0:
            qroot = cfg.out_root / "stale_bank_quarantine" / f"{int(time.time())}"
            qroot.mkdir(parents=True, exist_ok=True)
            for p in sorted(cfg.bank_dir.glob("vc_bank_part_*.pt")):
                p.rename(qroot / p.name)
            for p in sorted((cfg.manifest_dir / "bank_capture").glob("*.done.json")):
                p.rename(qroot / p.name)
    if cfg.worker_index == 0:
        # Gate 0c BEFORE any generation spend (plan §7).
        _gate0c_hf_write_canary(cfg)
    model, tok = load_model_and_tokenizer(cfg)
    if cfg.worker_index == 0:
        # Gate 0a freeze: token_identity_report.json is written BEFORE the
        # floor check (HALT artifact), then the DETERMINISTIC bank.json (no
        # timestamps — its sha IS the regime key) + the bank.meta.json sidecar.
        frozen_manifest = BANK29.freeze_bank_2389(
            tok, cfg.bank_dir / "bank.json", cfg.bank_dir / "token_identity_report.json"
        )
        frozen_bytes = json.dumps(frozen_manifest, sort_keys=True, ensure_ascii=False).encode()
        assert _sha256_bytes(frozen_bytes) == bank_sha, (
            "freeze/manifest sha drift — the run tokenizer disagrees with the "
            "MODEL_ID bank tokenizer (check --model-id)"
        )

    all_contexts = list(BANK.build_contexts())
    chunks = enumerate_capture_chunks(all_contexts)
    smoke_spots: list[dict] | None = None
    smoke_grid_need: set[str] | None = None
    if cfg.smoke:
        chunks = chunks[:2]
        # B4 (r1 review): extend the slice with the injection gate's spot
        # dependencies so the smoke REACHES the gate (production spot
        # selection, computed over the FULL surviving pair set).
        smoke_spots, extra = _smoke_gate_slice_extension(
            all_contexts,
            {c for ch in chunks for c in ch.context_ids},
            surviving_pairs(manifest),
            manifest["donor_assignment"],
        )
        if extra:
            chunks = chunks + [CaptureChunk(SMOKE_GATE_CHUNK_INDEX, tuple(extra))]
            logger.info(
                "[bank] smoke slice extended by %d injection-gate dependency contexts",
                len(extra),
            )
        # Grid-closure sibling (crash-fix r2): the grid smoke leg's blocks
        # span pairs (and their non-steered donors) outside the base slice +
        # B4 extension — capture their contexts too, so _block_cells /
        # payload_for_arm resolve at the grid leg instead of KeyErroring
        # after the pilot/anchors/vLLM smoke spend.
        smoke_grid_need, grid_extra = _smoke_grid_slice_extension(
            all_contexts,
            {c for ch in chunks for c in ch.context_ids},
            surviving_pairs(manifest),
            manifest["donor_assignment"],
            no_prefix_ids(manifest),
        )
        if grid_extra:
            chunks = chunks + [CaptureChunk(SMOKE_GRID_CHUNK_INDEX, tuple(grid_extra))]
            logger.info(
                "[bank] smoke slice extended by %d grid-smoke closure contexts",
                len(grid_extra),
            )

    def _capture_one(chunk: CaptureChunk) -> None:
        part = capture_bank(cfg, model, tok, order=list(chunk.context_ids))
        _save_pt_atomic(
            _bank_part_path(cfg, chunk),
            {"layers": part["layers"], "per_context": part["per_context"], "repro": _repro(cfg)},
        )
        _write_json_atomic(
            block_done_path(cfg.out_root, chunk, "bank_capture"),
            {
                "regime_fp": regime_fp,
                "key": chunk.key,
                "n_contexts": len(chunk.context_ids),
                "repro": _repro(cfg),
            },
        )

    stats = run_claim_queue(cfg, chunks, regime_fp, "bank_capture", _capture_one)
    logger.info("[bank] capture queue drained: %s", stats)
    if cfg.worker_index != 0:
        # Worker 0 owns the merge + gates + done manifest (dispatcher fanout
        # waits on every worker; the queue drain above guarantees coverage).
        logger.info("[phase=bank_done] worker=%d (capture contribution complete)", cfg.worker_index)
        return RC_OK

    records: dict[str, dict] = {}
    layers = cfg.layers
    for chunk in chunks:
        blob = torch.load(_bank_part_path(cfg, chunk), map_location="cpu", weights_only=False)
        assert blob["layers"] == layers, (blob["layers"], layers)
        records.update(blob["per_context"])
    expected = set(all_contexts) if not cfg.smoke else {c for ch in chunks for c in ch.context_ids}
    assert set(records) == expected, (
        f"bank merge coverage mismatch: {len(records)} records vs {len(expected)} expected"
    )
    bank = {"layers": layers, "per_context": records}
    # Cross-check the CAPTURED no-prefix set against the frozen manifest
    # (informational under ce-only — prefix ends are never consumed as slots).
    realized_np = {cid for cid, r in bank["per_context"].items() if r.get("no_prefix")}
    if not cfg.smoke:
        assert realized_np == set(manifest["no_prefix_context_ids"]), (
            sorted(realized_np ^ set(manifest["no_prefix_context_ids"]))[:5],
            "captured no-prefix set != bank.json no_prefix_context_ids",
        )
    pairs = surviving_pairs(manifest)
    if cfg.smoke:
        pairs = [p for p in pairs if p.a in expected and p.b in expected]
    donor_maps = manifest["donor_assignment"]
    _save_pt_atomic(
        cfg.bank_dir / "vc_bank.pt",
        {
            "layers": bank["layers"],
            "per_context": bank["per_context"],
            "donor_assignments": donor_maps,
            "no_prefix_context_ids": sorted(realized_np),
            "bank_sha": bank_sha,
            "repro": _repro(cfg),
        },
    )
    logger.info(
        "[bank] captured %d contexts x %d layers (%d no-prefix; %d surviving pairs)",
        len(bank["per_context"]),
        cfg.n_layers,
        len(realized_np),
        len(pairs),
    )

    # Gate 2 first (ce distinctness — a bank defect makes gate 1 meaningless).
    degeneracy = run_degeneracy_guard(bank, pairs, tok)
    degeneracy["repro"] = _repro(cfg)
    _write_json_atomic(cfg.gates_dir / "degeneracy_guard_report.json", degeneracy)
    if not degeneracy["passed"]:
        logger.error(
            "[degeneracy_guard] FAILED: %d/%d ce-distinctness violations (bank defect)",
            degeneracy["n_violations"],
            degeneracy["n_pairs_checked"],
        )
        if not cfg.force_past_halt_gates:
            return RC_DEGENERACY_GATE
        logger.error(
            "[degeneracy_guard] --force-past-halt-gates set: proceeding on a FAILED guard "
            "(recorded)"
        )

    if cfg.smoke:
        # B4 mechanizable assert (r1 review verbatim): every requested gate
        # cell must exist in the smoke-filtered pair set — the extension
        # above is what makes this hold.
        assert smoke_spots is not None
        spot_cells = {s["cell"] for s in smoke_spots}
        have_cells = set(BANK.pairs_by_cell(pairs))
        assert spot_cells <= have_cells, (
            "B4: smoke slice cannot supply injection-gate spot cells",
            sorted(spot_cells - have_cells),
        )
        # Grid-closure sibling assert (crash-fix r2, dereference-set <=
        # have-set): every context the grid smoke leg dereferences (block
        # pairs + non-steered donors, pe-exclusion semantics applied) is
        # captured — a future smoke_blocks / consumer drift fails loud HERE,
        # at capture time, never at the grid leg after the anchors spend.
        assert smoke_grid_need is not None
        grid_missing = smoke_grid_need - set(records)
        assert not grid_missing, (
            "grid-smoke closure: smoke slice cannot supply grid smoke-block contexts",
            sorted(grid_missing)[:8],
        )
    report = run_injection_gate(cfg, model, tok, bank, pairs, donor_maps, spots=smoke_spots)
    _write_json_atomic(cfg.gates_dir / "injection_gate_report.json", report)
    if not report["passed"]:
        logger.error(
            "[injection_gate] FAILED: %d/%d spots failed",
            report["n_spots_failed"],
            report["n_spots"],
        )
        if not cfg.force_past_halt_gates:
            return RC_INJECTION_GATE
        logger.error(
            "[injection_gate] --force-past-halt-gates set: proceeding on a FAILED gate (recorded)"
        )
    _write_json_atomic(
        cfg.manifest_dir / "bank_done.json",
        {
            "regime_fp": regime_fp,
            "bank_sha": bank_sha,
            "n_contexts": len(bank["per_context"]),
            "injection_gate_passed": bool(report["passed"]),
            "degeneracy_gate_passed": bool(degeneracy["passed"]),
            "forced_past_gate": bool(
                cfg.force_past_halt_gates and not (report["passed"] and degeneracy["passed"])
            ),
            "repro": _repro(cfg),
        },
    )
    logger.info("[phase=bank_done]")
    return RC_OK


# ── answer-state capture (hooked + unhooked from ONE implementation) ──


@torch.no_grad()
def capture_answer_states(
    cfg: RunConfig,
    model,
    tok,
    ctx_ids_by_row: list[list[int]],
    completions: list[str],
    eot_ids: list[int],
    payloads: list[torch.Tensor] | None = None,
    positions: list[int] | None = None,
    tail_inclusive: bool = False,
    hook_builder=None,
) -> dict:
    """Span-mean answer states from teacher-forced re-forwards.

    ``va_span`` = mean over the COMPLETION token positions at every layer.
    With ``payloads`` given, each row's forward runs with the all-layer
    replace hook armed at that row's slot position (the PATCHED condition the
    rollout was generated under — plan §4.4 "hooked teacher-forced
    re-forward"), using the RIGHT-pad ``row_lengths=[T]*B`` arming that the
    injection gate's capture leg verifies. Rows are built by concatenating
    per-segment TOKEN IDS (BPE-seam rule).

    ``tail_inclusive=True`` (issue #2215 plan §4.2 — the ONLY new capture
    code) additionally pools ``va_tail_incl`` over the completion PLUS the
    end-of-turn tail (``eot_ids``) from the SAME captured stack — the #779
    ``v_x`` training-target convention twin. Default ``False`` keeps this
    function byte-identical for every existing caller.

    ``hook_builder`` (2329 stage-2 fold): the hook-arming callable, signature
    identical to ``_arm_hook_all_layers(model, cfg, row_lengths, positions,
    payloads, t_pad)``. Default ``None`` keeps the stage-1 all-layer REPLACE
    hook; stage-2 passes a single-layer ADD-mode builder closing over
    ``(layer, dose)``.
    """
    assert len(ctx_ids_by_row) == len(completions), (len(ctx_ids_by_row), len(completions))
    hooked = payloads is not None
    if hooked:
        assert positions is not None and len(payloads) == len(positions) == len(completions)
    if hook_builder is None:
        hook_builder = _arm_hook_all_layers
    layers = cfg.layers
    pad_id = tok.pad_token_id
    n = len(completions)
    va_span = torch.zeros((n, len(layers), cfg.hidden), dtype=torch.float32)
    va_tail = (
        torch.zeros((n, len(layers), cfg.hidden), dtype=torch.float32) if tail_inclusive else None
    )
    comp_ids: list[list[int]] = [
        tok(text, add_special_tokens=False)["input_ids"] if text else [] for text in completions
    ]
    n_comp_tokens: list[int] = [len(ids) for ids in comp_ids]
    empty: list[int] = []
    for start in range(0, n, cfg.capture_batch):
        idxs = list(range(start, min(start + cfg.capture_batch, n)))
        rows, keep = [], []
        for i in idxs:
            comp = comp_ids[i]
            if not comp:
                empty.append(i)
                continue
            rows.append(ctx_ids_by_row[i] + comp + eot_ids)
            keep.append((i, len(ctx_ids_by_row[i]), len(comp)))
        if not rows:
            continue
        ids, mask = _right_pad(rows, pad_id, cfg.device)
        t_pad = int(ids.shape[1])
        stack = None
        if hooked:
            stack = hook_builder(
                model,
                cfg,
                [t_pad] * len(rows),
                [(positions[i],) for i, _, _ in keep],
                [payloads[i] for i, _, _ in keep],
                t_pad,
            )
        try:
            captured = extract_layer_activations(model, ids, layers, attention_mask=mask)
        finally:
            if stack is not None:
                stack.remove()
        for j, (i, ctx_len, n_comp) in enumerate(keep):
            span = slice(ctx_len, ctx_len + n_comp)
            va_span[i] = torch.stack(
                [captured[layer][j, span].float().mean(dim=0) for layer in layers]
            ).cpu()
            if tail_inclusive:
                # NEW (issue #2215 plan §4.2): the row was forwarded as
                # ctx + comp + eot_ids, so the tail positions are already in
                # the captured stack — pool them in, no second forward.
                span_incl = slice(ctx_len, ctx_len + n_comp + len(eot_ids))
                va_tail[i] = torch.stack(
                    [captured[layer][j, span_incl].float().mean(dim=0) for layer in layers]
                ).cpu()
        del captured
    out = {
        "va_span": va_span.to(torch.float16),
        "n_completion_tokens": n_comp_tokens,
        "empty_rows": sorted(empty),
        "pooling": {"va_span": "mean over completion tokens (plan §4.4 span-mean V_a)"},
    }
    if tail_inclusive:
        out["va_tail_incl"] = va_tail.to(torch.float16)
        out["pooling"]["va_tail_incl"] = (
            "mean over completion tokens + end-of-turn tail (issue #2215 §4.2 v_x-convention twin)"
        )
    return out


# ── margin pools + teacher-forced lnP (plan §4.4 secondary DV) ────────


def pool_key(pair: BANK.Pair2162) -> str:
    return f"{pair.cell}|{pair.value_a}-{pair.value_b}"


def load_pools(path: Path) -> dict[str, list[dict]]:
    """Judge-built margin pools: ``{"pools": {"<cell>|<va>-<vb>": [items]}}``,
    each item ``{"side": "A"|"B", "text": <completion>}`` (fail-loud schema)."""
    payload = json.loads(path.read_text())
    pools = payload.get("pools")
    assert isinstance(pools, dict) and pools, f"pools file {path} carries no 'pools' object"
    for key, items in pools.items():
        assert isinstance(items, list) and items, (key, "empty pool")
        for it in items:
            assert it.get("side") in ("A", "B"), (key, it.get("side"))
            assert isinstance(it.get("text"), str) and it["text"].strip(), (key, "empty pool text")
    return pools


@torch.no_grad()
def margin_lnp(
    cfg: RunConfig,
    model,
    tok,
    rows_spec: list[dict],
) -> list[float]:
    """Length-normalized teacher-forced lnP of each pool item.

    ``rows_spec[i]`` = ``{"ctx_ids": [...], "item_ids": [...], "payload":
    (1,L,H)|None, "position": int|None}``. All rows in one call are either
    hooked (grid margins) or unhooked (anchor margins) — asserted. Log-probs
    are reduced GPU-side per row; only scalars move to CPU.
    """
    hooked = rows_spec[0]["payload"] is not None
    assert all((r["payload"] is not None) == hooked for r in rows_spec)
    pad_id = tok.pad_token_id
    out: list[float] = []
    for start in range(0, len(rows_spec), cfg.capture_batch):
        chunk = rows_spec[start : start + cfg.capture_batch]
        rows = [r["ctx_ids"] + r["item_ids"] for r in chunk]
        ids, mask = _right_pad(rows, pad_id, cfg.device)
        t_pad = int(ids.shape[1])
        stack = None
        if hooked:
            stack = _arm_hook_all_layers(
                model,
                cfg,
                [t_pad] * len(rows),
                [(r["position"],) for r in chunk],
                [r["payload"] for r in chunk],
                t_pad,
            )
        try:
            logits = model(input_ids=ids, attention_mask=mask).logits
        finally:
            if stack is not None:
                stack.remove()
        for b, r in enumerate(chunk):
            s = len(r["ctx_ids"])
            n_item = len(r["item_ids"])
            assert n_item >= 1, "empty pool item ids"
            lp = torch.log_softmax(logits[b, s - 1 : s + n_item - 1].float(), dim=-1)
            targets = ids[b, s : s + n_item]
            tok_lp = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            out.append(float(tok_lp.mean()))
        del logits
    return out


# ── P2: anchors (sharded, gate-3 slice first) ─────────────────────────


def _anchor_context_order(cfg: RunConfig) -> tuple[list[str], list[str], dict[str, dict]]:
    """``(gate_slice_context_ids, remaining_context_ids, contexts)`` — the
    gate-3 slice contexts generate FIRST (plan §7 gate 3 / §9)."""
    contexts = BANK.build_contexts()
    manifest = _load_frozen_manifest(cfg)
    pairs = surviving_pairs(manifest)
    gate_pairs = BANK.gate_slice_pairs(pairs)
    gate_ids: list[str] = []
    seen: set[str] = set()
    for p in gate_pairs:
        for cid in (p.a, p.b):
            if cid not in seen:
                seen.add(cid)
                gate_ids.append(cid)
    rest = [cid for cid in contexts if cid not in seen]
    if cfg.smoke:
        pairs_by_id = {p.pair_id: p for p in pairs}
        smoke_ctx = {
            cid
            for block in smoke_blocks(pairs)
            for pid in block.pair_ids
            for cid in (pairs_by_id[pid].a, pairs_by_id[pid].b)
        }
        gate_ids = [c for c in gate_ids if c in smoke_ctx]
        rest = [c for c in rest if c in smoke_ctx]
    return gate_ids, rest, contexts


def _enrich_rows_with_capture(
    rows: list[dict], states: dict, max_new_tokens: int | None = None
) -> None:
    """Post-capture per-row telemetry: token count + cap_hit vs the REALIZED
    generating cap, and the cap itself. Recording the cap per row is what
    keeps a mixed-cap store (per-cell caps, plan §4.7 item 1; cell-restricted
    capregens) VISIBLE downstream instead of implicit.

    ``max_new_tokens=None`` (the per-cell anchors path) reads each row's OWN
    recorded ``max_new_tokens`` (set at generation time); an int applies a
    uniform cap (single-cell grid/stage2 blocks, capregen)."""
    for r, n_tok in zip(rows, states["n_completion_tokens"], strict=True):
        cap = int(r["max_new_tokens"]) if max_new_tokens is None else max_new_tokens
        r["n_completion_tokens"] = n_tok
        r["cap_hit"] = cap_hit(n_tok, cap)
        r["cap_hit_basis"] = "retokenized_completion_len >= max_new_tokens"
        r["max_new_tokens"] = cap


def _resolve_cap(cfg: RunConfig, cell: str, recalibrated: dict[str, int] | None = None) -> int:
    """Realized generation cap for one cell (plan §4.7 item 1).

    An EXPLICIT ``--max-new-tokens`` overrides everything (the capregen
    raised-cap contract); else the per-cell table with the gate-3-slice
    recalibration (up-only) layered on top."""
    if cfg.max_new_tokens_explicit:
        return cfg.max_new_tokens
    return cell_max_new_tokens(cell, recalibrated)


def _cell_bucketed_chunks(
    contexts: dict[str, dict], order: list[str], chunk_size: int
) -> list[tuple[str, list[str]]]:
    """``(cell, context-id chunk)`` list — a chunk NEVER mixes cells (plan
    §4.7 item 2), preserving the incoming order within each cell."""
    by_cell: dict[str, list[str]] = {}
    for cid in order:
        by_cell.setdefault(contexts[cid]["cell"], []).append(cid)
    out: list[tuple[str, list[str]]] = []
    for cell, cids in by_cell.items():
        for start in range(0, len(cids), chunk_size):
            out.append((cell, cids[start : start + chunk_size]))
    return out


def _generate_anchor_rows(
    cfg: RunConfig,
    model,
    tok,
    contexts: dict[str, dict],
    order: list[str],
    draws: int,
    batch: str,
    recalibrated: dict[str, int] | None = None,
) -> tuple[list[dict], list[list[int]], list[str]]:
    """Unpatched anchor generation core — ``(rows, flat_ctx, flat_text)``.

    The SINGLE anchors generation loop, shared by ``_run_anchor_batch`` and
    ``phase_capregen_anchors`` (no second generation loop; capture +
    enrichment stay caller-side so the two-write text-persist ordering is
    preserved at each call site). CELL-BUCKETED (plan §4.7 item 2): a
    generate chunk never mixes cells, so each chunk runs at ITS cell's cap
    (``_resolve_cap`` — the per-cell table + gate-3-slice recalibration; an
    explicit ``--max-new-tokens`` overrides, the capregen contract)."""
    rows: list[dict] = []
    flat_ctx: list[list[int]] = []
    flat_text: list[str] = []
    ctx_ids = {cid: BANK29.context_token_ids_2389(tok, contexts[cid]) for cid in order}
    t0 = time.monotonic()
    n_done = 0
    for cell, chunk in _cell_bucketed_chunks(contexts, order, cfg.gen_batch):
        cap = _resolve_cap(cfg, cell, recalibrated)
        outs = generate_batch(
            model,
            tok,
            [contexts[c] for c in chunk],
            n=draws,
            hook=None,
            max_new_tokens=cap,
            temperature=ANCHOR_TEMPERATURE,
            seed_base=cfg.seed_base,
            render_fn=BANK29.render_context_2389,
            ids_fn=BANK29.context_token_ids_2389,
            share_prefill=cfg.share_prefill_armed,
        )
        for b, cid in enumerate(chunk):
            ctx = contexts[cid]
            for i, text in enumerate(outs[b]):
                flat_ctx.append(ctx_ids[cid])
                flat_text.append(text)
                rows.append(
                    {
                        "context_id": cid,
                        "cell": ctx["cell"],
                        "value_id": ctx["value_id"],
                        "carrier": ctx["carrier"],
                        "draw": i,
                        "seed": cfg.seed_base + i,
                        "temperature": ANCHOR_TEMPERATURE,
                        # R2 (r2 review): kind-derived — `batch` carries the
                        # cell-grain batch_id domain (gate_{cell}/rest_{cell}/
                        # parity_{cell}/vllm_{cell}), matching the vLLM leg's
                        # `cid in gate_id_set` semantics (vllm_anchors.py).
                        "gate_slice": _batch_kind(batch) == "gate",
                        "max_new_tokens": cap,
                        "engine": "hf",
                        "text": text,
                    }
                )
        n_done += len(chunk)
        logger.info(
            "[anchors:%s] unit %d/%d contexts cell=%s cap=%d elapsed=%.1fs",
            batch,
            n_done,
            len(order),
            cell,
            cap,
            time.monotonic() - t0,
        )
    return rows, flat_ctx, flat_text


def _finalize_anchor_batch(
    cfg: RunConfig,
    model,
    tok,
    rows: list[dict],
    flat_ctx: list[list[int]],
    flat_text: list[str],
    batch: str,
    regime_fp: str,
    n_contexts: int,
    draws: int,
) -> dict:
    """Capture + enrich + persist one anchors batch whose ROWS already exist.

    The post-generation half of `_run_anchor_batch`, split out so the vLLM
    anchors leg (`scripts/issue2389_vllm_anchors.py`, plan §4.7 item 4) can
    run the SAME teacher-forced `capture_answer_states` pass + enrichment +
    shard/done-manifest writes over vLLM-generated rows (M1-v: a vLLM cell's
    done manifest is written ONLY after its capture pt lands — this function
    IS that ordering). Caller persists the rollout TEXT before invoking."""
    eot = eot_tail_ids(tok)
    jsonl = cfg.anchors_dir / f"anchors_{batch}_w{cfg.worker_index}.jsonl"
    states = capture_answer_states(cfg, model, tok, flat_ctx, flat_text, eot)
    _enrich_rows_with_capture(rows, states)  # per-row caps (plan §4.7 item 1)
    _write_jsonl_atomic(jsonl, rows)
    _save_pt_atomic(
        cfg.anchors_dir / f"va_anchors_{batch}_w{cfg.worker_index}.pt",
        {
            "layers": cfg.layers,
            "index": [{"context_id": r["context_id"], "draw": r["draw"]} for r in rows],
            "va_span": states["va_span"],
            "pooling": states["pooling"],
            "empty_rows": states["empty_rows"],
            "repro": _repro(cfg),
        },
    )
    cap_hits = sum(1 for r in rows if r["cap_hit"])
    cell_caps = sorted({(r["cell"], int(r["max_new_tokens"])) for r in rows})
    _write_json_atomic(
        cfg.manifest_dir / f"anchors_{batch}_w{cfg.worker_index}_done.json",
        {
            "regime_fp": regime_fp,
            "batch": batch,
            "worker_index": cfg.worker_index,
            "num_workers": cfg.num_workers,  # shard identity (r1 M2)
            "n_contexts": n_contexts,
            "draws": draws,
            "n_rows": len(rows),
            "n_cap_hit": cap_hits,
            "n_empty": len(states["empty_rows"]),
            "max_new_tokens": cfg.max_new_tokens,
            "cell_caps": [{"cell": c, "max_new_tokens": m} for c, m in cell_caps],
            # B9: the realized generation regime, recorded per shard so a
            # later consumer (capregen's per-shard base-fp reconstruction)
            # can recover the EXACT fingerprint inputs without guessing.
            "gen_batch": cfg.gen_batch,
            "share_prefill_armed": cfg.share_prefill_armed,
            "engine": rows[0].get("engine", "hf") if rows else "hf",
            "repro": _repro(cfg),
        },
    )
    logger.info(
        "[anchors:%s] rows=%d cap_hit=%d empty=%d",
        batch,
        len(rows),
        cap_hits,
        len(states["empty_rows"]),
    )
    return {"jsonl": jsonl, "n_rows": len(rows)}


def _run_anchor_batch(
    cfg: RunConfig,
    model,
    tok,
    contexts: dict[str, dict],
    order: list[str],
    draws: int,
    batch: str,
    regime_fp: str,
    recalibrated: dict[str, int] | None = None,
) -> dict:
    """Generate + capture one anchors batch for THIS worker; write shards."""
    rows, flat_ctx, flat_text = _generate_anchor_rows(
        cfg, model, tok, contexts, order, draws, batch, recalibrated
    )
    # Persist the rollout TEXT the moment generation completes, BEFORE the
    # capture reduce (#779 / r1 m2): a capture crash must never lose ~1,650
    # generated rollouts. The post-capture write inside the finalize helper
    # atomically REPLACES this file with the capture-enriched rows.
    jsonl = cfg.anchors_dir / f"anchors_{batch}_w{cfg.worker_index}.jsonl"
    _write_jsonl_atomic(jsonl, rows)
    return _finalize_anchor_batch(
        cfg, model, tok, rows, flat_ctx, flat_text, batch, regime_fp, len(order), draws
    )


def _pilot_selected_gen_batch(cfg: RunConfig) -> int | None:
    """The r2-selected gen_batch from the three-regime pilot gate (§4.7 item 3),
    VALIDATED before adoption (B9 r1 review): a stale / refused / foreign /
    out-of-band pilot report must FAIL LOUD, never silently drive production
    batching. Absent report (pre-pilot phases, smokes without a pilot) ->
    None; a present-but-invalid report raises.

    Round-5 J (concern ``pilot-reuse-runtime-domain``): this is the NORMAL
    adoption path — ``_adopt_pilot_gen_batch`` routes phase_anchors / grid /
    capregen / stage2 (run.py) AND the vLLM legs
    (vllm_anchors._compose_run_cfg) through here — so it routes through the
    strengthened ``_reusable_pilot_report`` reader: the full runtime-domain
    validation (regime + model@rev + worker width + GPU name/memory +
    candidate set + thresholds/floors + torch/transformers identity) guards
    EVERY consumer, not only the pilot phase's own reuse decision. The
    checks kept here are adoption-specific: a REFUSE report or an
    out-of-band selection must never drive production batching."""
    rec = _reusable_pilot_report(cfg)  # round-5 J: raises on a FOREIGN report
    if rec is None:
        return None
    sel = rec.get("gen_batch_selected")
    if sel is None:
        return None
    sel = int(sel)
    problems: list[str] = []
    verdict = rec.get("verdict")
    if verdict not in ("ACCEPT", "SPLIT"):
        problems.append(
            f"verdict={verdict!r} (need ACCEPT|SPLIT — a REFUSE/absent verdict never "
            "drives adoption)"
        )
    if sel not in PILOT_GEN_BATCH_CANDIDATES:
        problems.append(
            f"gen_batch_selected={sel} outside the registered auto-selection band "
            f"{PILOT_GEN_BATCH_CANDIDATES} (an explicit-candidate pilot needs the same "
            "explicit --gen-batch on every consuming phase)"
        )
    report_cands = [int(b) for b in rec.get("gen_batch_candidates") or []]
    if sel not in report_cands:
        problems.append(
            f"gen_batch_selected={sel} not among the report's own candidates {report_cands}"
        )
    if problems:
        raise RuntimeError(
            "pilot_gate_report.json cannot drive gen_batch adoption: "
            + "; ".join(problems)
            + " — re-run the pilot (or pass an explicit --gen-batch)"
        )
    return sel


def _pilot_gpu_name() -> str | None:
    """GPU lane identity for pilot-report reuse checks (None on a CPU host).

    Round-5 C (r4 review): the pilot's s/rollout and HBM-headroom readings
    are properties of the DEVICE lane — an H100-measured report must never
    drive an H200 (or CPU) run's batching and poll fences."""
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def _pilot_gpu_mem_gib() -> float | None:
    """Device MEMORY identity (total GiB, 0.1 grain; None on a CPU host).

    Round-5 J (concern ``pilot-reuse-runtime-domain``): the HBM-headroom
    eligibility read is a function of the device's TOTAL memory, not just
    its marketing name — the reuse contract requires the memory identity
    alongside ``gpu_name``."""
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)


def _reusable_pilot_report(cfg: RunConfig) -> dict | None:
    """An existing pilot report THIS run may ADOPT instead of re-measuring
    (R5 r2 review). Plan §9 pre-registers same-command resume as a LOSSLESS
    boundary, and gen_batch is inside ``regime_fingerprint`` (B9), so an
    unnecessary pilot re-measurement whose 16<->32 selection flips would
    rewrite the fingerprint and read every banked anchors/grid done shard
    as NOT-done (quarantine + regeneration, 40+ GPU-h at risk). Returns
    None when no report exists (measure); the record when the report's
    regime matches this run — model id@revision + smoke/tiny AND, round-5 C
    (concern ``pilot-reuse-runtime-domain``), the RUNTIME DOMAIN the
    measurements are functions of: worker width (per-phase walls divide by
    it), GPU lane (s/rollout + HBM headroom are device properties),
    candidate set (the argmin selection is only valid over the same band),
    and the wall thresholds the verdict was banded against — extended in
    round-5 J with the device MEMORY identity (``gpu_total_mem_gib``), the
    recorded floor constants (``hbm_headroom_floor_gib`` /
    ``refusal_threshold_h``), and the torch/transformers runtime identity
    (git commit WARN-only; see the inline comment). RAISES on a
    present-but-foreign report — a deliberate re-measure needs ``--force``;
    a report predating these fields cannot prove its domain and is foreign.
    Round-5 J: this reader is ALSO the validation the NORMAL adoption path
    runs — ``_pilot_selected_gen_batch`` routes through it, so grid /
    capregen / stage2 / the vLLM legs inherit every check here.
    """
    path = cfg.gates_dir / "pilot_gate_report.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    problems: list[str] = []
    repro = rec.get("repro") or {}
    if repro.get("model_id") != cfg.model_id or repro.get("model_revision") != cfg.model_revision:
        problems.append(
            f"report model {repro.get('model_id')}@{repro.get('model_revision')} != "
            f"this run's {cfg.model_id}@{cfg.model_revision}"
        )
    if bool(repro.get("smoke")) != bool(cfg.smoke) or bool(repro.get("tiny")) != bool(cfg.tiny):
        problems.append(
            f"report regime (smoke={repro.get('smoke')}, tiny={repro.get('tiny')}) != "
            f"this run's (smoke={cfg.smoke}, tiny={cfg.tiny})"
        )
    if rec.get("verdict") not in ("ACCEPT", "SPLIT", "REFUSE"):
        problems.append(f"unrecognized verdict {rec.get('verdict')!r}")
    width = max(1, cfg.num_workers)
    if int(rec.get("num_workers", -1)) != width:
        problems.append(
            f"report num_workers={rec.get('num_workers')} != this run's {width} "
            "(per-phase wall projections divide by worker width)"
        )
    gpu = _pilot_gpu_name()
    if rec.get("gpu_name", "<unrecorded>") != gpu:
        problems.append(
            f"report gpu_name={rec.get('gpu_name', '<unrecorded>')!r} != this run's {gpu!r} "
            "(s/rollout + HBM headroom are lane properties)"
        )
    gpu_mem = _pilot_gpu_mem_gib()
    if rec.get("gpu_total_mem_gib", "<unrecorded>") != gpu_mem:
        problems.append(
            f"report gpu_total_mem_gib={rec.get('gpu_total_mem_gib', '<unrecorded>')!r} != "
            f"this run's {gpu_mem!r} (round-5 J: the headroom eligibility read is a "
            "function of the device's TOTAL memory, not just its name)"
        )
    want_cands = sorted([cfg.gen_batch] if cfg.gen_batch_explicit else PILOT_GEN_BATCH_CANDIDATES)
    got_cands = sorted(int(b) for b in rec.get("gen_batch_candidates") or [])
    if got_cands != want_cands:
        problems.append(
            f"report gen_batch_candidates={got_cands} != this run's {want_cands} "
            "(the argmin selection is only valid over the same candidate set)"
        )
    if float(rec.get("accept_threshold_h", -1.0)) != PILOT_ACCEPT_WALL_H:
        problems.append(
            f"report accept_threshold_h={rec.get('accept_threshold_h')} != "
            f"{PILOT_ACCEPT_WALL_H} (the verdict was banded against a different threshold)"
        )
    if float(rec.get("planned_total_wall_h", -1.0)) != float(cfg.planned_wall_h):
        problems.append(
            f"report planned_total_wall_h={rec.get('planned_total_wall_h')} != this run's "
            f"{cfg.planned_wall_h} (the SPLIT/REFUSE bands scale with the planned wall)"
        )
    # Round-5 minor (r4 review): the floor constants were recorded but never
    # re-checked — asymmetric with accept_threshold_h. A changed
    # PILOT_HBM_HEADROOM_GIB across a resume could adopt a stale-eligibility
    # B=32 (OOM channel); a changed PILOT_REFUSAL_MULT re-bands the verdict.
    if float(rec.get("hbm_headroom_floor_gib", -1.0)) != PILOT_HBM_HEADROOM_GIB:
        problems.append(
            f"report hbm_headroom_floor_gib={rec.get('hbm_headroom_floor_gib')} != "
            f"{PILOT_HBM_HEADROOM_GIB} (the B eligibility was screened at a different floor)"
        )
    if float(rec.get("refusal_threshold_h", -1.0)) != PILOT_REFUSAL_MULT * float(
        cfg.planned_wall_h
    ):
        problems.append(
            f"report refusal_threshold_h={rec.get('refusal_threshold_h')} != "
            f"{PILOT_REFUSAL_MULT * float(cfg.planned_wall_h)} (the SPLIT/REFUSE boundary "
            "was banded against a different multiple)"
        )
    # Round-5 J: the measurements are ALSO functions of the kernel-level
    # runtime — torch/transformers versions bind HARD. The git commit is
    # WARN-only by design: crash-fix commits legitimately land between the
    # pilot and the plan §9 same-command resume, and a hard git pin would
    # force --force re-measures on every resume — the exact 16<->32
    # fingerprint-flip risk this reuse path exists to avoid.
    cur = _repro(cfg)
    repro = rec.get("repro") or {}
    for key in ("torch", "transformers"):
        if repro.get(key, "<unrecorded>") != cur[key]:
            problems.append(
                f"report repro.{key}={repro.get(key, '<unrecorded>')!r} != this env's "
                f"{cur[key]!r} (throughput + memory footprint are kernel-version properties)"
            )
    if problems:
        raise RuntimeError(
            f"existing {path} is FOREIGN to this run: "
            + "; ".join(problems)
            + " — pilot phase: quarantine it / use a fresh --out-root, or pass "
            "--force to deliberately re-measure. Consumer phase (anchors / grid / "
            "capregen / stage2 / the vLLM legs, adopting via "
            "_pilot_selected_gen_batch): do NOT quarantine a healthy report — "
            "re-dispatch matching the pilot's runtime domain (thread the SAME "
            "--num-workers, run on the pilot's GPU lane) or pass an explicit "
            "--gen-batch (round-6, concern pilot-reuse-runtime-domain)"
        )
    if repro.get("git_commit") != cur["git_commit"]:
        logger.warning(
            "[pilot] adopting a report recorded at git %s != this checkout %s (WARN-only: "
            "the runtime identity is pinned by torch/transformers; --force re-measures)",
            repro.get("git_commit"),
            cur["git_commit"],
        )
    return rec


def _adopt_pilot_gen_batch(cfg: RunConfig) -> RunConfig:
    """Adopt the pilot's r2-selected gen_batch unless --gen-batch was explicit.

    Plan §4.7 item 3: generation phases (anchors / grid / stage2) run at the
    argmin-s/rollout batch the pilot measured under the >=10 GB HBM-headroom
    constraint; an explicit --gen-batch (smokes, capregen re-runs) wins.
    """
    if cfg.gen_batch_explicit:
        return cfg
    sel = _pilot_selected_gen_batch(cfg)
    if sel is not None and sel != cfg.gen_batch:
        logger.info("[gen-batch] adopting pilot-selected gen_batch=%d (was %d)", sel, cfg.gen_batch)
        return replace(cfg, gen_batch=sel)
    return cfg


def _vllm_claimed_cells(cfg: RunConfig) -> frozenset[str]:
    """Cells the vLLM anchors leg (scripts/issue2389_vllm_anchors.py) owns.

    Work-conservation (plan §4.7 item 4): the vLLM script writes
    ``gates/vllm_cells.json`` BEFORE the HF rest batch starts; phase_anchors
    excludes those cells from the REST slice only (gate-slice cells stay HF —
    they are the parity/judge critical path). The judge's duplicate
    (context_id, draw) assert is the fail-loud backstop against double
    generation. Unknown cell names fail loud — a typo here would silently
    orphan a cell from BOTH engines.
    """
    path = cfg.gates_dir / "vllm_cells.json"
    if not path.exists():
        return frozenset()
    rec = json.loads(path.read_text())
    cells = frozenset(str(c) for c in rec.get("cells", []))
    unknown = cells - set(BANK.all_cells())
    if unknown:
        raise RuntimeError(f"gates/vllm_cells.json names unknown cells: {sorted(unknown)}")
    return cells


# B8 (r1 review): the one-shot ``anchor_rest_routing.json`` global freeze is
# GONE — it forfeited a late parity PASS for every cell the moment the first
# participant entered the rest batch. Routing is now PER CELL at CLAIM time:
# the claim file in the shared ``anchor_rest_cells`` namespace IS the freeze
# (atomic O_CREAT|O_EXCL), HF workers claim only cells outside the live
# ``gates/vllm_cells.json`` set (re-read every scan), and the vLLM production
# leg generates exactly the claimed-but-not-HF-done remainder — plan §4.7
# item 4's "a PASS re-routes only the REMAINING cells", work-conserving by
# construction.


def _load_cap_recalibration(cfg: RunConfig, regime_fp: str | None = None) -> dict[str, int] | None:
    """Read the gate-slice cap recalibration (None when not yet computed).

    Round-5 B (r4 review): when ``regime_fp`` is supplied (the RECORDING
    site, ``_gate_slice_cap_recalibration``), a report recorded under a
    DIFFERENT regime is NOT adopted — returns None so the caller recomputes
    from THIS regime's own gate shards and overwrites (the
    ``_sharded_done_record`` warn+``None`` idiom; the report records
    ``regime_fp`` precisely so adoption can check it — a smoke/tiny run's
    realized cap-hit evidence must never stand in for a production run's).

    Round-5 F (concern ``cap-recal-consumer-regime-bypass``): the
    consumption-only readers (grid / stage2 / the vLLM anchors rest leg)
    resolve their OWN fp domains (family freeze / ``stage2_regime_fp``),
    which legitimately differ from the anchors-phase fp the recording site
    validated under — so they cannot compare ``regime_fp``. But the fp
    DOMAIN is orthogonal to the smoke/tiny REGIME bit: dispatch.sh provides
    standalone surgical-resume arms (``grid)`` / ``stage2)`` / the vLLM
    legs) that never traverse the recording barrier, and OUT_ROOT defaults
    to a shared path, so a prior ``--smoke``/``--tiny`` recalibration sits
    there un-overwritten. Every reader therefore ALSO validates the
    report's recorded ``repro`` REGIME identity (model id@revision +
    tiny/smoke bits) against THIS run's cfg — this loader is the single
    accessor for all four readers, so the guard covers each. A
    regime-foreign (or repro-less, pre-round-5 shape) report is NOT
    adopted: warn + None — consumers fall back to the table caps (the
    design's own registered up-only prior). For the anchors/grid readers
    the standing >2%/cell capregen trigger backstops any genuinely-needed
    raise (the documented ``partial`` semantics); stage2 has NO
    cap-report/capregen scope (both ``--cap-scope`` and
    ``--capregen-scope`` exclude it — round-6, concern
    ``cap-recal-consumer-regime-bypass``), so a non-adopted report there
    means stage2 simply generates at the table caps with no capregen
    backstop: strictly safer than adopting foreign-regime cap evidence,
    but NOT capregen-remediable."""
    path = cfg.gates_dir / "cap_recalibration.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    if regime_fp is not None and rec.get("regime_fp") != regime_fp:
        logger.warning(
            "[cap-recal] existing report carries regime_fp=%s != this run's %s — "
            "recomputing from this regime's gate shards (never adopt foreign-regime "
            "cap evidence)",
            rec.get("regime_fp"),
            regime_fp,
        )
        return None
    repro = rec.get("repro") or {}
    regime_problems = []
    if repro.get("model_id") != cfg.model_id or repro.get("model_revision") != cfg.model_revision:
        regime_problems.append(
            f"model {repro.get('model_id')}@{repro.get('model_revision')} != "
            f"this run's {cfg.model_id}@{cfg.model_revision}"
        )
    if bool(repro.get("tiny")) != bool(cfg.tiny) or bool(repro.get("smoke")) != bool(cfg.smoke):
        regime_problems.append(
            f"regime (tiny={repro.get('tiny')}, smoke={repro.get('smoke')}) != "
            f"this run's (tiny={cfg.tiny}, smoke={cfg.smoke})"
        )
    if regime_problems:
        logger.warning(
            "[cap-recal] existing report is REGIME-FOREIGN (%s) — not adopted; running at "
            "table caps (the >2%%/cell capregen trigger backstops; round-5 F)",
            "; ".join(regime_problems),
        )
        return None
    return {str(k): int(v) for k, v in rec.get("recalibrated", {}).items()}


def _gate_shard_paths(cfg: RunConfig, regime_fp: str, cells: list[str], draws: int) -> list[Path]:
    """EXACT gate-cell shard paths, derived from the validated done manifests.

    R1 (r2 review): the retired worker-stripe glob ``anchors_gate_w*.jsonl``
    matched 0 of the cell-grain shards (producers write
    ``anchors_gate_{cell}_w{w}.jsonl``), so the recalibration silently
    aggregated NOTHING and wrote a FALSE-COMPLETE report that
    ``_load_cap_recalibration`` then adopted. Deriving the paths from the
    done manifests — as ``_cap_report_inputs`` already does — cannot drift
    on the next rename. Mirrors ``_anchor_cell_done``'s validation (regime +
    draws + artifact presence + row-count parity: a shard whose realized row
    count differs from its manifest's ``n_rows`` is a partial/foreign write
    and is skipped, round-5 minor); the FIRST valid manifest per cell wins,
    exactly the record the barrier's done predicate accepted. A cell that
    passed the barrier but resolves no manifest+shard is an inconsistent
    store — fail loud, never a silent zero-row aggregate.
    """
    paths: list[Path] = []
    for cell in sorted(cells):
        batch_id = f"gate_{cell}"
        shard: Path | None = None
        for m in sorted(cfg.manifest_dir.glob(f"anchors_{batch_id}_w*_done.json")):
            if m.name.endswith("_gen_done.json"):
                continue
            rec = json.loads(m.read_text())
            if rec.get("regime_fp") != regime_fp:
                continue
            if int(rec.get("draws", -1)) != draws:
                continue
            jsonl = cfg.anchors_dir / f"anchors_{batch_id}_w{int(rec['worker_index'])}.jsonl"
            if not jsonl.exists():
                continue
            n_rows = sum(1 for line in jsonl.open(encoding="utf-8") if line.strip())
            if n_rows != int(rec.get("n_rows", -1)):
                logger.warning(
                    "[cap-recal] %s: manifest %s says n_rows=%s but %s has %d — skipping",
                    batch_id,
                    m.name,
                    rec.get("n_rows"),
                    jsonl.name,
                    n_rows,
                )
                continue
            shard = jsonl
            break
        if shard is None:
            raise RuntimeError(
                f"[cap-recal] gate cell {cell!r} passed the done barrier but no "
                f"regime-matched done manifest + shard resolve under "
                f"{cfg.manifest_dir} / {cfg.anchors_dir} — inconsistent store"
            )
        paths.append(shard)
    return paths


def _gate_slice_cap_recalibration(
    cfg: RunConfig, regime_fp: str, draws: int, gate_cells: list[str]
) -> dict[str, int]:
    """Plan §4.7 item 1: recalibrate per-cell caps from the REALIZED gate slice.

    Barrier: waits (poll CLAIM_POLL_S, timeout CAP_RECAL_TIMEOUT_S) for EVERY
    gate CELL's done manifest at this regime (cell-keyed — the claim queue
    lets ANY worker own any gate cell, so the barrier lifts as soon as the
    slice is done rather than waiting on a specific worker's stripe; the r1
    recal-barrier idle dissolves with it), then aggregates the done cells'
    gate shard rows — shard paths derived from the validated done manifests
    (``_gate_shard_paths``; R1 r2 review), never a filename glob: any cell
    whose realized cap-hit fraction
    (against each row's OWN recorded cap) exceeds CAP_HIT_REGEN_TRIGGER_PCT
    gets its table cap DOUBLED (up-only) BEFORE the bulk rest/grid
    generation. On timeout the recalibration is computed from the cells that
    finished, labeled ``partial`` — up-only, so a missed cell is caught by
    the standing >2%/cell capregen trigger downstream.

    Concurrent workers computing this independently write IDENTICAL content
    (deterministic aggregate of the same shards) via atomic replace — benign.
    """
    path = cfg.gates_dir / "cap_recalibration.json"
    existing = _load_cap_recalibration(cfg, regime_fp)  # round-5 B: regime-checked adoption
    if existing is not None and not cfg.force:
        return existing
    t0 = time.monotonic()
    pending = set(gate_cells)
    while pending:
        for cell in sorted(pending):
            if _anchor_cell_done(cfg, regime_fp, f"gate_{cell}", draws):
                pending.discard(cell)
        if not pending:
            break
        if time.monotonic() - t0 > CAP_RECAL_TIMEOUT_S:
            logger.warning(
                "[cap-recal] TIMEOUT after %.0fs waiting for gate cells %s — "
                "computing PARTIAL recalibration (up-only; capregen backstops)",
                CAP_RECAL_TIMEOUT_S,
                sorted(pending),
            )
            break
        logger.info(
            "[cap-recal] waiting for %d gate-slice cell(s) (%.0fs elapsed)",
            len(pending),
            time.monotonic() - t0,
        )
        time.sleep(CLAIM_POLL_S)
    per_cell: dict[str, dict[str, int]] = {}
    done_cells = [c for c in gate_cells if c not in pending]
    for shard in _gate_shard_paths(cfg, regime_fp, done_cells, draws):
        for line in shard.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if "cap_hit" not in r:
                continue  # pre-capture text-persist snapshot of a live worker
            c = per_cell.setdefault(r["cell"], {"n": 0, "hit": 0})
            c["n"] += 1
            c["hit"] += int(bool(r["cap_hit"]))
    recalibrated: dict[str, int] = {}
    for cell, c in sorted(per_cell.items()):
        pct = 100.0 * c["hit"] / c["n"] if c["n"] else 0.0
        if pct > CAP_HIT_REGEN_TRIGGER_PCT:
            recalibrated[cell] = 2 * cell_max_new_tokens(cell)
            logger.info(
                "[cap-recal] cell=%s gate cap-hit %.2f%% > %.1f%% — cap %d -> %d",
                cell,
                pct,
                CAP_HIT_REGEN_TRIGGER_PCT,
                cell_max_new_tokens(cell),
                recalibrated[cell],
            )
    _write_json_atomic(
        path,
        {
            "criterion": (
                f"gate-slice per-cell cap-hit > {CAP_HIT_REGEN_TRIGGER_PCT}% -> 2x table cap "
                "BEFORE bulk generation (plan §4.7 item 1; up-only)"
            ),
            "regime_fp": regime_fp,
            "partial": bool(pending),
            "cells_missing": sorted(pending),
            "per_cell": {
                cell: {
                    "n_rows": c["n"],
                    "n_cap_hit": c["hit"],
                    "cap_hit_pct": round(100.0 * c["hit"] / c["n"], 3) if c["n"] else 0.0,
                }
                for cell, c in sorted(per_cell.items())
            },
            "recalibrated": recalibrated,
            "repro": _repro(cfg),
        },
    )
    logger.info(
        "[cap-recal] wrote %s: %d cell(s) recalibrated%s",
        path,
        len(recalibrated),
        " (PARTIAL)" if pending else "",
    )
    return recalibrated


SHARE_PREFILL_GATE_NAME = "share_prefill_equivalence.json"


def _share_prefill_family(phase: str) -> str:
    """Shard FAMILY a phase's share-prefill decision binds to: capregen and
    the vLLM legs share the base family's freeze (a regenerated / re-routed
    shard must not mix execution modes inside one consumed store)."""
    return phase.removeprefix("capregen_").removeprefix("vllm_")


def _share_prefill_gate_digest(rec: dict) -> str:
    """Canonical DECISION digest of the gate-4b artifact: verdict + mode ONLY.

    Round-5 A (r4 review, found by three arms): the battery stamps a fresh
    ``ts`` into ``share_prefill_equivalence.json`` on every run, so a
    RAW-BYTE digest spuriously DISARMS legitimately-armed families whenever
    the battery re-runs into the same out_root — and the disarm flips
    ``share_prefill_armed`` inside ``regime_fingerprint``, quarantining every
    banked armed-fp done shard on the plan §9 DESIGNED same-command resume
    (>=40 GPU-h class). The freeze therefore binds to the canonical SUBSET
    that actually drives the arming decision — ``verdict`` + ``mode``, the
    only artifact fields ``_resolve_share_prefill`` reads — so a benign
    same-decision rewrite is inert while a verdict/mode change still
    disarms. Writer and validator share THIS helper (sha-pin-domain
    coherence: one digest domain, one recipe)."""
    payload = json.dumps({"mode": rec.get("mode"), "verdict": rec.get("verdict")}, sort_keys=True)
    return _sha256_bytes(payload.encode())


def _validate_frozen_share_prefill(cfg: RunConfig, phase: str, rec: dict, freeze: Path) -> bool:
    """Adopt-time validation of a frozen family decision (R3 r2 review).

    The B6 production-mode guard used to sit ONLY on the first-resolver
    path, so BOTH adopt paths (existing-freeze + lost-race) read
    ``rec["armed"]`` RAW — a freeze written by a ``--tiny`` dispatch
    (``{armed: true, mode: "tiny"}``) would arm 27B production generation
    on CPU-tiny fp32 evidence via a later production dispatch sharing the
    out_root. An ARMED record is adopted armed ONLY when its recorded
    regime matches THIS run (repro tiny/smoke bits + model id@revision —
    the ``_pilot_selected_gen_batch`` idiom), its mode satisfies the B6
    rule for this run, and the gate artifact it was armed on still carries
    the recorded ``gate_sha256`` DECISION digest (verdict+mode canonical
    subset — ``_share_prefill_gate_digest``; a benign fresh-``ts`` rewrite
    keeps the digest, round-5 A). A FOREIGN-regime failure resolves to
    SERIAL (unarmed-on-uncertainty, the B9 family determination) — the
    guard is deterministic, so every same-regime participant still resolves
    the SAME value; an unarmed record adopts as serial with no checks.
    Round-5 I: a SAME-REGIME digest mismatch under an armed freeze RAISES
    instead — early participants already ran armed, so a serial adopt here
    would split one frozen family across arming values (see the inline
    comment at the raise)."""
    if not bool(rec.get("armed")):
        return False
    problems: list[str] = []
    repro = rec.get("repro") or {}
    if bool(repro.get("tiny")) != bool(cfg.tiny) or bool(repro.get("smoke")) != bool(cfg.smoke):
        problems.append(
            f"freeze regime (tiny={repro.get('tiny')}, smoke={repro.get('smoke')}) != "
            f"this run's (tiny={cfg.tiny}, smoke={cfg.smoke})"
        )
    if repro.get("model_id") != cfg.model_id or repro.get("model_revision") != cfg.model_revision:
        problems.append(
            f"freeze model {repro.get('model_id')}@{repro.get('model_revision')} != "
            f"this run's {cfg.model_id}@{cfg.model_revision}"
        )
    if not cfg.tiny and rec.get("mode") != "production":
        problems.append(f"mode={rec.get('mode')!r} (non-production evidence on a production run)")
    gate_sha = rec.get("gate_sha256")
    gate_path = cfg.gates_dir / SHARE_PREFILL_GATE_NAME
    digest_problem: str | None = None
    if gate_sha is None:
        problems.append("armed freeze carries no gate_sha256 (arming evidence unverifiable)")
    else:
        current: str | None = None
        if gate_path.exists():
            try:
                current = _share_prefill_gate_digest(json.loads(gate_path.read_text()))
            except (json.JSONDecodeError, UnicodeDecodeError):
                current = None  # unparseable evidence -> mismatch
        if current != gate_sha:
            digest_problem = (
                "gate artifact absent/unparseable or its DECISION digest (verdict+mode) != "
                "the freeze's recorded gate_sha256 (the evidence that armed this family "
                "changed its arming decision)"
            )
    if digest_problem is not None and not problems:
        # Round-5 I (concern share-prefill-material-remeasure-family-split):
        # a MATERIAL change to the arming evidence (a --force re-measure that
        # flipped the verdict/mode, or vanished/unparseable evidence) under a
        # SAME-REGIME armed freeze is a FAMILY SPLIT, not an adopt-serial
        # case: participants that resolved before the change ran ARMED under
        # this freeze, so resolving SERIAL here would mix arming values
        # inside one frozen family — the exact mix regime_fingerprint's
        # determinism contract forbids. FAIL LOUD: the operator either
        # restores the original gate artifact (digest re-matches; the family
        # resumes armed) or moves to a fresh out_root (the documented
        # re-arm path). A FOREIGN-regime freeze (regime/model/mode problems
        # above) keeps the warn+SERIAL disposition — those adopters never
        # shared the armed participants' fingerprint domain.
        raise RuntimeError(
            f"[share-prefill:{phase}] MATERIAL CHANGE under a live armed family freeze "
            f"({freeze.name}): {digest_problem} — early same-regime participants already "
            "generated ARMED under this freeze, so adopting serial would split the family "
            "across arming values (regime_fingerprint determinism contract). Restore the "
            "original gate artifact, or re-run the family in a fresh --out-root; a "
            "deliberate re-arm requires a fresh out_root by design"
        )
    if digest_problem is not None:
        problems.append(digest_problem)
    if problems:
        logger.warning(
            "[share-prefill:%s] frozen record %s FAILS adopt-time validation (%s) — "
            "staying SERIAL (unarmed-on-uncertainty; re-arming needs a fresh out_root)",
            phase,
            freeze.name,
            "; ".join(problems),
        )
        return False
    return True


def _resolve_share_prefill(cfg: RunConfig, phase: str) -> RunConfig:
    """Freeze the share_prefill arming for THIS phase FAMILY (plan §4.7 item 5
    pin 2): armed <=> --share-prefill auto AND the gate-4b battery artifact
    (gates/share_prefill_equivalence.json, written by
    scripts/issue2389_share_prefill_gate.py) carries verdict PASS. Absent /
    FAIL / "off" => serial (FAIL-OPEN).

    B9 (r1 review): the decision is part of `regime_fingerprint`, so every
    participant in one shard family MUST resolve the SAME value even when
    they enter at different times (the dispatcher's worker-1 chain enters
    anchors AFTER the gate-4b battery lands; the vLLM legs enter
    post-anchors; capregen enters last). The FIRST resolver writes
    ``gates/share_prefill_frozen_<family>.json`` atomically
    (first-writer-wins via os.link); every later resolver ADOPTS the frozen
    value. Re-arming a family deliberately requires a fresh out_root."""
    if cfg.share_prefill_mode != "auto":
        cfg.share_prefill_armed = False
        return cfg
    family = _share_prefill_family(phase)
    freeze = cfg.gates_dir / f"share_prefill_frozen_{family}.json"
    if freeze.exists():
        rec = json.loads(freeze.read_text())
        # R3 (r2 review): NEVER adopt rec["armed"] raw — the B6 guard binds
        # at ADOPT time too (a --tiny dispatch's armed freeze must not arm a
        # later production dispatch sharing this out_root).
        cfg.share_prefill_armed = _validate_frozen_share_prefill(cfg, phase, rec, freeze)
        logger.info(
            "[share-prefill:%s] adopting FROZEN family decision share_prefill=%s (%s)",
            phase,
            cfg.share_prefill_armed,
            freeze.name,
        )
        return cfg
    path = cfg.gates_dir / SHARE_PREFILL_GATE_NAME
    verdict: str | None = None
    mode: str | None = None
    gate_sha: str | None = None
    if not path.exists():
        logger.info(
            "[share-prefill:%s] mode=auto but gate artifact missing at %s — staying serial",
            phase,
            path,
        )
        armed = False
    else:
        rec = json.loads(path.read_text())
        verdict = rec.get("verdict")
        mode = rec.get("mode")
        # R3: the freeze records the DECISION digest (verdict+mode canonical
        # subset — NEVER raw bytes: the battery legitimately rewrites the
        # artifact with a fresh ``ts`` on the plan's designed same-command
        # resume, round-5 A) so adopters can verify the arming evidence
        # still carries the same decision.
        gate_sha = _share_prefill_gate_digest(rec)
        armed = verdict == "PASS"
    # B6 (r1 review / pin 2): a NON-tiny run arms ONLY on a PRODUCTION-mode
    # battery artifact. The dispatcher's --smoke branch runs the gate-4b
    # battery --tiny into the SAME gates/ path + HF prefix, so a default
    # smoke->production sequence would otherwise arm 27B generation on a
    # CPU-tiny fp32 PASS before the production-device (M-N2) battery lands —
    # exactly the arming-order gap pin 2 exists to prevent. Tiny runs accept
    # a tiny/offline-tiny PASS (device-matched).
    if armed and not cfg.tiny and mode != "production":
        logger.info(
            "[share-prefill:%s] gate PASS but mode=%r (non-production artifact on a "
            "production run) — staying serial until the production-device battery lands",
            phase,
            mode,
        )
        armed = False
    # First-writer-wins freeze: write via tmp + os.link (link fails EEXIST
    # when a concurrent resolver won); on a lost race ADOPT the winner.
    cfg.gates_dir.mkdir(parents=True, exist_ok=True)
    tmp = freeze.with_name(f".{freeze.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(
        json.dumps(
            {
                "armed": armed,
                "verdict": verdict,
                "mode": mode,
                "gate_sha256": gate_sha,
                "family": family,
                "frozen_by_phase": phase,
                "ts": datetime.now(UTC).isoformat(),
                "repro": _repro(cfg),
            },
            indent=2,
        )
    )
    try:
        os.link(tmp, freeze)
    except FileExistsError:
        rec = json.loads(freeze.read_text())
        # R3 (r2 review): the lost-race adopt is the SECOND raw-read bypass
        # of the B6 guard — validate exactly like the existing-freeze adopt.
        armed = _validate_frozen_share_prefill(cfg, phase, rec, freeze)
        logger.info(
            "[share-prefill:%s] lost the freeze race — adopting share_prefill=%s", phase, armed
        )
    finally:
        tmp.unlink(missing_ok=True)
    cfg.share_prefill_armed = armed
    logger.info(
        "[share-prefill:%s] gate verdict=%s mode=%s -> share_prefill=%s (frozen: %s)",
        phase,
        verdict,
        mode,
        cfg.share_prefill_armed,
        freeze.name,
    )
    return cfg


def phase_anchors(cfg: RunConfig) -> int:
    """P2: unpatched temp-1.0 anchors at CELL grain via the claim queue (plan
    §9 "P2 ... via the claim queue"; B7/B8 r1 review).

    Gate-3 slice cells generate + upload FIRST (per cell, so the SYNC judge
    overlaps the remaining generation); after the cap-recalibration barrier
    the rest cells run through the per-cell routing seam: a cell's HF-vs-vLLM
    ownership freezes when it is CLAIMED, so a parity PASS landing mid-phase
    re-routes exactly the REMAINING (unclaimed) cells to the vLLM leg
    (work-conserving, plan §4.7 item 4). Cell-grain claims keep every
    generate chunk at full ``gen_batch`` (a cell's contexts are never strided
    across workers — the B7 de-batching fix), and cell shards are
    width-invariant (any worker may own any cell), so the anchors family
    needs no stale-width entry sweep."""
    logger.info(
        "[phase=anchors] worker=%d/%d smoke=%s", cfg.worker_index, cfg.num_workers, cfg.smoke
    )
    cfg = _adopt_pilot_gen_batch(cfg)  # plan §4.7 item 3
    cfg = _resolve_share_prefill(cfg, "anchors")  # plan §4.7 item 5 (pin 2)
    # >= 2 draws even under --smoke: the disjoint-half floor F_act needs k >= 2.
    draws = 2 if cfg.smoke else cfg.anchor_draws
    _manifest, bank_sha = bank_manifest_and_sha()
    regime_fp = regime_fingerprint(cfg, bank_sha)
    gate_ids, rest_ids, contexts = _anchor_context_order(cfg)
    gate_by_cell = _group_by_cell(gate_ids, contexts)
    rest_by_cell = _group_by_cell(rest_ids, contexts)

    if cfg.force and cfg.worker_index == 0:
        # One-shot deliberate re-run (mirrors phase_bank's --force shape):
        # quarantine every same-regime-or-not cell artifact so the queue's
        # done predicates read pending and regenerate exactly once.
        for batch, cells in (("gate", gate_by_cell), ("rest", rest_by_cell)):
            for cell in cells:
                _quarantine_orphan_cell_shards(cfg, f"{batch}_{cell}")

    model_tok: list = [None, None]

    def _run_cell(block: AnchorCellBlock, order: list[str], recal: dict[str, int] | None) -> dict:
        if model_tok[0] is None:
            model_tok[0], model_tok[1] = load_model_and_tokenizer(cfg)
        _quarantine_orphan_cell_shards(cfg, block.batch_id)
        return _run_anchor_batch(
            cfg,
            model_tok[0],
            model_tok[1],
            contexts,
            order,
            draws,
            block.batch_id,
            regime_fp,
            recalibrated=recal,
        )

    # GATE cells first (always HF — the parity/judge critical path), with an
    # immediate per-cell upload so the VM judge can start in the P2 window.
    gate_blocks = [AnchorCellBlock(cell=c, batch="gate") for c in gate_by_cell]

    def _run_gate_cell(block: AnchorCellBlock) -> None:
        res = _run_cell(block, gate_by_cell[block.cell], None)
        _upload_dir(
            cfg, cfg.anchors_dir, f"{HF_PREFIX}/raw_completions/anchors_gate", [res["jsonl"].name]
        )

    run_claim_queue(
        cfg,
        gate_blocks,
        regime_fp,
        "anchor_gate_cells",
        _run_gate_cell,
        is_done=lambda _root, b, fp, _ns: _anchor_cell_done(cfg, fp, b.batch_id, draws),
    )

    # Cap-recalibration barrier (plan §4.7 item 1): every worker waits for
    # the FULL gate slice (cell-keyed), recalibrates >2%-cap-hit cells to 2x,
    # and only then generates the bulk rest batch at the (possibly raised)
    # caps. Resolving after the barrier maximizes the window for the vLLM
    # parity verdict to land before the first rest claims.
    recal = _gate_slice_cap_recalibration(cfg, regime_fp, draws, sorted(gate_by_cell))

    # REST cells: per-cell engine routing (plan §4.7 item 4 / B8). ``mine``
    # excludes cells in the LIVE vLLM claim set (gates/vllm_cells.json,
    # re-read every scan: the parity leg claims its 3 cells at t0; leg_claim
    # extends the claim to every cell on a parity PASS — a late PASS
    # re-routes exactly the not-yet-claimed cells). Cells this queue leaves
    # unclaimed belong to the vLLM production leg (post-anchors, GPU
    # released); the claim files + worker-independent done predicates make
    # double generation impossible by construction, with the judge's
    # duplicate-(context_id, draw) assert as the fail-loud backstop.
    rest_blocks = [AnchorCellBlock(cell=c, batch="rest") for c in rest_by_cell]

    def _run_rest_cell(block: AnchorCellBlock) -> None:
        _run_cell(block, rest_by_cell[block.cell], recal)

    stats = run_claim_queue(
        cfg,
        rest_blocks,
        regime_fp,
        "anchor_rest_cells",
        _run_rest_cell,
        is_done=lambda _root, b, fp, _ns: _anchor_cell_done(cfg, fp, b.batch_id, draws),
        mine=lambda b: b.cell not in _vllm_claimed_cells(cfg),
    )
    routed = sorted(
        b.cell
        for b in rest_blocks
        if b.cell in _vllm_claimed_cells(cfg)
        and not _anchor_cell_done(cfg, regime_fp, b.batch_id, draws)
    )
    if routed:
        logger.info(
            "[anchors] %d rest cell(s) routed to the vLLM leg (claimed, not HF-generated): %s",
            len(routed),
            routed,
        )
    # Registered cap-hit>2%/cell trigger, MEASURED at phase end (plan
    # registration; the trigger previously had no enforcing code). Partial
    # snapshots (cells still pending on either engine) are labeled partial.
    _emit_cap_hit_snapshot(cfg, "anchors")
    logger.info("[phase=anchors_done] worker=%d queue=%s", cfg.worker_index, stats)
    return RC_OK


# ── P3: the grid (claim-file queue) ───────────────────────────────────


def _load_bank(cfg: RunConfig) -> dict:
    path = cfg.bank_dir / "vc_bank.pt"
    assert path.exists(), (
        f"{path} missing — run `--phase bank` first (this driver never silently "
        "recaptures the V bank mid-grid)"
    )
    # Self-produced, sha-recorded bundle carrying non-tensor metadata.
    return torch.load(path, map_location="cpu", weights_only=False)


def _block_cells(
    bank: dict,
    block: Block,
    pairs_by_id: dict[str, BANK.Pair2162],
    donor_maps: dict[str, dict[str, str]],
) -> list[dict]:
    """Per-pair cell specs (payload, position, provenance) for one block."""
    recs = bank["per_context"]
    degenerate_pe = block.slot == "pe" and BANK.base_type_of(block.cell) in BANK.DEGENERATE_AT_PE
    cells: list[dict] = []
    for pid in block.pair_ids:
        pair = pairs_by_id[pid]
        payload, donor_id = payload_for_arm(
            bank, pair, block.slot, block.arm, donor_maps, pairs_by_id
        )
        rec = recs[pair.a]
        cells.append(
            {
                "pair_id": pid,
                "pair": pair,
                "context_a": pair.a,
                "context_b": pair.b,
                "position": slot_position(rec["ctx_len"], rec["prefix_end"], block.slot),
                "payload": payload,
                "donor_pair_id": donor_id,
                "degenerate_pe": degenerate_pe,
                # Plan §4.1 length mitigation (r1 M7): per-pair token-length
                # delta (varied-span length difference), the analysis covariate
                # + the |Δlen| <= 2 length-matched sensitivity subset key.
                "len_delta": int(recs[pair.b]["ctx_len"]) - int(rec["ctx_len"]),
            }
        )
    return cells


def _block_margin_rows(
    cfg: RunConfig,
    model,
    tok,
    block: Block,
    cells: list[dict],
    pools: dict[str, list[dict]],
    ctx_ids_cache: dict[str, list[int]],
) -> list[dict]:
    """Grid margin TF: every pool item of the pair's value-pair under the
    PATCHED state (hook armed). Missing pools produce EXPLICIT skip rows."""
    rows_spec: list[dict] = []
    meta: list[dict] = []
    out: list[dict] = []
    for cell in cells:
        pair: BANK.Pair2162 = cell["pair"]
        key = pool_key(pair)
        items = pools.get(key)
        if not items:
            out.append(
                {
                    "block_key": block.key,
                    "pair_id": pair.pair_id,
                    "arm": block.arm,
                    "pool_key": key,
                    "skipped": True,
                    "reason": "no pool for this value-pair (judge-filter yield below floor "
                    "or no-rubric cell)",
                }
            )
            continue
        for idx, it in enumerate(items):
            item_ids = tok(it["text"], add_special_tokens=False)["input_ids"]
            assert item_ids, (key, idx, "pool item tokenized empty")
            rows_spec.append(
                {
                    "ctx_ids": ctx_ids_cache[cell["context_a"]],
                    "item_ids": item_ids,
                    "payload": cell["payload"],
                    "position": cell["position"],
                }
            )
            meta.append(
                {
                    "block_key": block.key,
                    "cell": block.cell,
                    "slot": block.slot,
                    "arm": block.arm,
                    "pair_id": pair.pair_id,
                    "donor_pair_id": cell["donor_pair_id"],
                    "pool_key": key,
                    "pool_idx": idx,
                    "pool_side": it["side"],
                    "n_pool_tokens": len(item_ids),
                }
            )
    if rows_spec:
        lnps = margin_lnp(cfg, model, tok, rows_spec)
        for m, lnp in zip(meta, lnps, strict=True):
            out.append({**m, "lnp_mean": lnp, "skipped": False})
    return out


@torch.no_grad()
def run_block(
    cfg: RunConfig,
    model,
    tok,
    bank: dict,
    block: Block,
    pairs_by_id: dict[str, BANK.Pair2162],
    donor_maps: dict[str, dict[str, str]],
    contexts: dict[str, dict],
    ctx_ids_cache: dict[str, list[int]],
    eot: list[int],
    regime_fp: str,
    pools: dict[str, list[dict]] | None,
    draws: int,
    write_done: bool = True,
    done_extra: dict | None = None,
    recalibrated: dict[str, int] | None = None,
) -> dict:
    """One block: K hooked temp-1.0 draws per pair, the hooked V_a pass, and
    (pools present) the margin TF pass — pipelined on the same GPU.

    ``write_done=False`` (the PILOT leg) suppresses BOTH resume done-files —
    the block done-file AND the margin_blocks twin — so a pilot run on
    production ``blocks[0]`` can never leave a ``regime_fp + "-pilot"`` done
    record that the grid queue's ``block_is_done`` scan RAISES on (r1 C1).

    ``done_extra`` (capregen) is merged into the block done record — the
    durable marker that a block was re-generated at a raised cap.

    A block is single-cell, so the plan §4.7 item-1 per-cell cap resolves to
    ONE value here (``recalibrated`` = the gate-slice recalibration; an
    explicit --max-new-tokens still overrides via ``_resolve_cap``)."""
    cells = _block_cells(bank, block, pairs_by_id, donor_maps)
    cap = _resolve_cap(cfg, block.cell, recalibrated)

    def ids_for(cid: str) -> list[int]:
        if cid not in ctx_ids_cache:
            ctx_ids_cache[cid] = BANK29.context_token_ids_2389(tok, contexts[cid])
        return ctx_ids_cache[cid]

    texts_per_cell: list[list[str]] = []
    hooked_gt1_unequal = False  # plan §12 pad-into-recurrence smoke seam
    for start in range(0, len(cells), cfg.gen_batch):
        chunk = cells[start : start + cfg.gen_batch]
        ctx_list = [contexts[c["context_a"]] for c in chunk]
        rows = [ids_for(c["context_a"]) for c in chunk]
        row_lengths = [len(r) for r in rows]
        t_pad = max(row_lengths)
        if len(rows) > 1 and len(set(row_lengths)) > 1:
            hooked_gt1_unequal = True
        stack = _arm_hook_all_layers(
            model,
            cfg,
            row_lengths,
            [(c["position"],) for c in chunk],
            [c["payload"] for c in chunk],
            t_pad,
        )
        try:
            outs = generate_batch(
                model,
                tok,
                ctx_list,
                n=draws,
                hook=stack,
                max_new_tokens=cap,
                temperature=GRID_TEMPERATURE,
                seed_base=cfg.seed_base,
                render_fn=BANK29.render_context_2389,
                ids_fn=BANK29.context_token_ids_2389,
                share_prefill=cfg.share_prefill_armed,
            )
        finally:
            stack.remove()
        assert len(outs) == len(chunk), (len(outs), len(chunk))
        texts_per_cell.extend(list(o) for o in outs)
    assert len(texts_per_cell) == len(cells)

    # Hooked V_a: one flattened (pair x draw) row set, each row armed with its
    # pair's payload at its pair's position. Rows are built TEXT-ONLY here, in
    # the same nested order as the flat capture lists.
    flat_ctx: list[list[int]] = []
    flat_text: list[str] = []
    flat_payload: list[torch.Tensor] = []
    flat_pos: list[int] = []
    rows_out: list[dict] = []
    for c, texts in zip(cells, texts_per_cell, strict=True):
        pair: BANK.Pair2162 = c["pair"]
        for i, text in enumerate(texts):
            flat_ctx.append(ids_for(c["context_a"]))
            flat_text.append(text)
            flat_payload.append(c["payload"])
            flat_pos.append(c["position"])
            rows_out.append(
                {
                    "block_key": block.key,
                    "cell": block.cell,
                    "slot": block.slot,
                    "arm": block.arm,
                    "pair_id": pair.pair_id,
                    "carrier": pair.carrier,
                    "value_a": pair.value_a,
                    "value_b": pair.value_b,
                    "context_a": pair.a,
                    "context_id": pair.a,  # audit-walker compat (issue2094_judge.run_audits)
                    "context_b": pair.b,
                    "position": c["position"],
                    "donor_pair_id": c["donor_pair_id"],
                    "degenerate_pe": c["degenerate_pe"],
                    "len_delta": c["len_delta"],
                    "draw": i,
                    "seed": cfg.seed_base + i,
                    "temperature": GRID_TEMPERATURE,
                    "text": text,
                }
            )
    # Persist the rollout TEXT the moment generation completes, BEFORE the
    # capture reduce (#779 / r2 F9 — the anchors two-write pattern): a capture
    # crash must never lose the block's generated rollouts. The post-capture
    # write below atomically REPLACES this file with the capture-enriched rows
    # (adds token counts / cap_hit).
    shard_jsonl = cfg.rollouts_dir / f"shard_{block.slug}.jsonl"
    _write_jsonl_atomic(shard_jsonl, rows_out)
    states = capture_answer_states(
        cfg, model, tok, flat_ctx, flat_text, eot, payloads=flat_payload, positions=flat_pos
    )
    _enrich_rows_with_capture(rows_out, states, cap)
    _write_jsonl_atomic(shard_jsonl, rows_out)
    _save_pt_atomic(
        cfg.va_dir / f"shard_{block.slug}.pt",
        {
            "block_key": block.key,
            "layers": cfg.layers,
            "index": [
                {"pair_id": r["pair_id"], "context_a": r["context_a"], "draw": r["draw"]}
                for r in rows_out
            ],
            "va_span": states["va_span"],
            "pooling": states["pooling"],
            "empty_rows": states["empty_rows"],
            "hooked_capture": True,
            "repro": _repro(cfg),
        },
    )
    margin_done = False
    if pools is not None:
        margin_rows = _block_margin_rows(cfg, model, tok, block, cells, pools, ctx_ids_cache)
        _write_jsonl_atomic(cfg.margin_dir / f"shard_{block.slug}.jsonl", margin_rows)
        if write_done:
            _write_json_atomic(
                block_done_path(cfg.out_root, block, "margin_blocks"),
                {
                    "key": block.key,
                    "regime_fp": regime_fp,
                    "n_rows": len(margin_rows),
                    "n_skipped": sum(1 for r in margin_rows if r.get("skipped")),
                    "repro": _repro(cfg),
                },
            )
        margin_done = True
    done = {
        "key": block.key,
        "regime_fp": regime_fp,
        "n_cells": block.n_pairs,
        "n_rows": len(rows_out),
        "n_cap_hit": sum(1 for r in rows_out if r["cap_hit"]),
        "n_empty": len(states["empty_rows"]),
        "max_new_tokens": cap,
        "hooked_batch_gt1_unequal": hooked_gt1_unequal,
        "margin_inline": margin_done,
        "repro": _repro(cfg),
        **(done_extra or {}),
    }
    if write_done:
        _write_json_atomic(block_done_path(cfg.out_root, block), done)
    return done


def phase_grid(cfg: RunConfig) -> int:
    """P3: claim-queue block execution (or the three-regime pilot under ``--pilot``)."""
    logger.info("[phase=grid] worker=%d/%d smoke=%s", cfg.worker_index, cfg.num_workers, cfg.smoke)
    if cfg.pilot and not cfg.force:
        # R5 (r2 review): same-command resume is a LOSSLESS boundary (plan
        # §9) — a regime-matched pilot report is ADOPTED, never re-measured
        # (~0.5-1.5 h saved per resume, and a re-measured 16<->32 flip would
        # rewrite regime_fingerprint and quarantine every banked
        # anchors/grid shard). --force re-measures deliberately.
        pilot_rec = _reusable_pilot_report(cfg)
        if pilot_rec is not None:
            pilot_verdict = pilot_rec.get("verdict")
            if pilot_verdict == "REFUSE":
                if not cfg.force_past_halt_gates:
                    logger.error(
                        "[pilot] existing pilot_gate_report.json verdict=REFUSE at this "
                        "regime — the refusal STANDS on re-entry (an unchanged setup "
                        "re-measures to the same verdict; after a throughput fix — the "
                        "plan's kernel-install prepared response — pass --force to "
                        "re-pilot; --force-past-halt-gates overrides)"
                    )
                    return RC_PILOT_GATE
                logger.warning(
                    "[pilot] REFUSE report present but --force-past-halt-gates set — "
                    "proceeding without re-measurement"
                )
                return RC_OK
            sel = _pilot_selected_gen_batch(cfg)  # full adoption-grade validation
            if sel is None:
                raise RuntimeError(
                    "existing pilot_gate_report.json carries no gen_batch_selected — "
                    "corrupt; pass --force to re-measure"
                )
            logger.info(
                "[phase=pilot_done] ADOPTED existing report (verdict=%s gen_batch=%d) — "
                "re-measurement skipped (lossless resume, plan §9; --force re-measures)",
                pilot_verdict,
                sel,
            )
            return RC_OK
    if not cfg.pilot:
        # plan §4.7 item 3 (the pilot itself sweeps its own candidate set)
        cfg = _adopt_pilot_gen_batch(cfg)
        cfg = _resolve_share_prefill(cfg, "grid")  # plan §4.7 item 5 (pin 2)
    bank = _load_bank(cfg)
    manifest = _load_frozen_manifest(cfg)
    frozen_sha = _sha256_bytes(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode())
    assert frozen_sha == str(bank.get("bank_sha")), (
        "frozen bank.json sha != vc_bank.pt bank_sha — bank/capture drift",
        frozen_sha,
        bank.get("bank_sha"),
    )
    pairs = surviving_pairs(manifest)
    np_ids = no_prefix_ids(manifest)
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_maps = bank["donor_assignments"]
    regime_fp = regime_fingerprint(cfg, str(bank.get("bank_sha")))
    draws = SMOKE_GRID_DRAWS if cfg.smoke else cfg.grid_draws

    all_blocks, pe_exclusions = apply_pe_exclusions(
        enumerate_blocks(pairs), np_ids, donor_maps, pairs
    )
    _write_pe_exclusions(cfg, pe_exclusions, scope="grid")
    totals_all = grid_totals(all_blocks, cfg.grid_draws)
    if cfg.smoke:
        blocks, smoke_pe_excl = apply_pe_exclusions(smoke_blocks(pairs), np_ids, donor_maps, pairs)
        logger.info("[grid] smoke pe exclusions: %d records", len(smoke_pe_excl))
    else:
        blocks = all_blocks
    if cfg.pilot:
        blocks = blocks[:1]
    totals = grid_totals(blocks, draws)
    pools: dict[str, list[dict]] | None = None
    if cfg.pools_path is not None and cfg.pools_path.exists():
        pools = load_pools(cfg.pools_path)
        logger.info("[grid] margin pools loaded: %d pools (%s)", len(pools), cfg.pools_path)
    else:
        logger.info(
            "[grid] no pools file (%s) — margins deferred to --phase margin", cfg.pools_path
        )
    _write_json_atomic(
        cfg.manifest_dir / f"grid_plan_w{cfg.worker_index}.json",
        {
            "regime_fp": regime_fp,
            "worker_index": cfg.worker_index,
            "num_workers": cfg.num_workers,
            "smoke": cfg.smoke,
            "pilot": cfg.pilot,
            "totals_full_grid": totals_all,
            "totals_this_run": totals,
            "n_pe_excluded_pair_cells": sum(1 for e in pe_exclusions if e["pair_id"] is not None),
            "n_pe_empty_blocks": sum(1 for e in pe_exclusions if e["pair_id"] is None),
            "queue": "shared claim-file queue (work-conserving)",
            "margin_inline": pools is not None,
            "repro": _repro(cfg),
        },
    )
    logger.info(
        "[grid] full grid: %d blocks / %d cells / %d rollouts; this run: %d blocks",
        totals_all["n_blocks"],
        totals_all["cells_total"],
        totals_all["rollouts_total"],
        len(blocks),
    )

    model, tok = load_model_and_tokenizer(cfg)
    eot = eot_tail_ids(tok)
    contexts = BANK.build_contexts()
    ctx_ids_cache: dict[str, list[int]] = {}
    # Gate-slice cap recalibration (plan §4.7 item 1) — computed by the
    # anchors phase; None (table caps) when the grid runs before/without it.
    recal = _load_cap_recalibration(cfg)
    if recal:
        logger.info("[grid] cap recalibration active: %s", recal)
    ran_rollouts = 0
    ran_wall = 0.0
    n_run = 0
    any_gt1_unequal = False
    uploaded: list[str] = []
    pending: list[Block] = []

    def run_one(block: Block) -> None:
        nonlocal ran_rollouts, ran_wall, n_run, uploaded, pending, any_gt1_unequal
        t0 = time.monotonic()
        rec = run_block(
            cfg,
            model,
            tok,
            bank,
            block,
            pairs_by_id,
            donor_maps,
            contexts,
            ctx_ids_cache,
            eot,
            regime_fp,
            pools,
            draws,
            recalibrated=recal,
        )
        elapsed = time.monotonic() - t0
        ran_rollouts += rec["n_rows"]
        ran_wall += elapsed
        n_run += 1
        any_gt1_unequal = any_gt1_unequal or bool(rec.get("hooked_batch_gt1_unequal"))
        pending.append(block)
        logger.info(
            "[grid] unit %d %s rows=%d cap_hit=%d elapsed=%.1fs",
            n_run,
            block.key,
            rec["n_rows"],
            rec["n_cap_hit"],
            elapsed,
        )
        if not cfg.pilot and cfg.upload_every > 0 and len(pending) >= cfg.upload_every:
            uploaded += _upload_grid_increment(cfg, pending)
            pending.clear()

    if cfg.pilot:
        # Three-regime throughput pilot (plan §7 gate 4 + §4.7 item 3). The r2
        # legs run ONE production-shape block through run_block with NO claim
        # and write_done=False — the r1 C1 bug was an unconditional done-file
        # write inside run_block that left blocks/<slug>.done.json carrying
        # regime_fp+"-pilot", killing every grid worker at P3 entry.
        return _run_three_regime_pilot(
            cfg,
            model,
            tok,
            bank,
            blocks[0],
            pairs_by_id,
            donor_maps,
            contexts,
            ctx_ids_cache,
            eot,
            regime_fp,
            pools,
            draws,
            totals_all,
        )

    stats = run_claim_queue(cfg, blocks, regime_fp, "blocks", run_one)
    if pending:
        uploaded += _upload_grid_increment(cfg, pending)
        pending.clear()
    if cfg.smoke and stats["ran"] > 0:
        # Plan §12 smoke seam: at least one ran smoke block exercised a hooked
        # batch>1 generate with UNEQUAL prompt lengths (a worker that ran 0
        # blocks — every claim already taken — legitimately skips this check).
        assert any_gt1_unequal, (
            "smoke ran blocks but no hooked batch>1 chunk had unequal prompt "
            "lengths — pad-into-recurrence coverage missing (plan §12)"
        )
    # Registered cap-hit>2%/cell trigger, MEASURED at phase end (the trigger
    # previously had no enforcing code). Sibling workers still mid-block make
    # this snapshot partial — labeled, and re-emittable via --phase cap_report.
    _emit_cap_hit_snapshot(cfg, "grid")
    _write_json_atomic(
        cfg.manifest_dir / f"grid_done_w{cfg.worker_index}.json",
        {
            "regime_fp": regime_fp,
            "worker_index": cfg.worker_index,
            "n_blocks_run": stats["ran"],
            "n_rollouts_run": ran_rollouts,
            "wall_s": ran_wall,
            "queue_waits": stats["waits"],
            "uploads": uploaded,
            "repro": _repro(cfg),
        },
    )
    logger.info(
        "[phase=grid_done] worker=%d blocks_run=%d rollouts=%d",
        cfg.worker_index,
        stats["ran"],
        ran_rollouts,
    )
    return RC_OK


def _pilot_hbm_headroom_gib() -> float | None:
    """Min-free-HBM PROXY for the just-finished pilot leg (None on CPU).

    ``total − external − max_memory_reserved()``: torch's peak reserved bytes
    over the leg, with non-torch/other-process usage folded in from the
    CURRENT free reading (external = total − free − reserved_now). A proxy —
    external usage is sampled now, not at the leg's peak — stated as such in
    the report."""
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    reserved_now = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_reserved()
    external = max(0.0, float(total - free) - float(reserved_now))
    return (float(total) - external - float(peak)) / 2**30


def _select_gen_batch(r2: dict[int, dict], candidates: list[int]) -> tuple[int, bool]:
    """Plan §4.7 item 3 selection rule over the r2 pilot legs (B12 r1 review:
    extracted so the rule is unit-testable apart from the GPU pilot).

    argmin s/rollout among candidates meeting the >= PILOT_HBM_HEADROOM_GIB
    floor (``hbm_headroom_gib is None`` — CPU — is eligible); an EXACT
    s/rollout tie picks the smaller B; NO eligible candidate picks the
    smallest B with a WARNING. Returns ``(gen_batch_selected, headroom_ok)``
    — ``headroom_ok=False`` is recorded in the pilot report."""
    eligible = {
        b: m
        for b, m in r2.items()
        if m["hbm_headroom_gib"] is None or m["hbm_headroom_gib"] >= PILOT_HBM_HEADROOM_GIB
    }
    if eligible:
        # argmin s/rollout; exact tie -> the smaller B (plan §4.7 item 3).
        return min(eligible, key=lambda b: (eligible[b]["s_per_rollout"], b)), True
    sel = min(candidates)
    logger.warning(
        "[pilot:r2] NO candidate met the %.0f GiB HBM headroom floor — "
        "selecting the smallest B=%d (recorded in the report)",
        PILOT_HBM_HEADROOM_GIB,
        sel,
    )
    return sel, False


def _run_three_regime_pilot(
    cfg: RunConfig,
    model,
    tok,
    bank: dict,
    block0: Block,
    pairs_by_id: dict[str, BANK.Pair2162],
    donor_maps: dict[str, dict[str, str]],
    contexts: dict[str, dict],
    ctx_ids_cache: dict[str, list[int]],
    eot: list[int],
    regime_fp: str,
    pools: dict[str, list[dict]] | None,
    draws: int,
    totals_all: dict,
) -> int:
    """Plan §7 gate 4: the THREE-REGIME pilot, then the ACCEPT/SPLIT/REFUSE verdict.

    r2 FIRST (hooked grid-shaped ``run_block`` at each candidate gen_batch;
    ``write_done=False`` — r1 C1), selecting gen_batch = argmin s/rollout
    subject to >= PILOT_HBM_HEADROOM_GIB headroom (exact tie -> the smaller
    B). r1 (anchor-shaped: unhooked, temp 1.0, K=anchor_draws, per-cell caps)
    and r3 (stage2-shaped: hooked greedy K=1) then run at the selected B, so
    every projection basis is at the phase's execution shape (#1415).

    r3 approximation, stated: the hook is run_block's replace-mode arming
    rather than stage-2's add-mode dose hook — the same compute shape (one
    per-layer tensor edit inside the forward); greedy K=1 matches stage-2.
    """

    def ids_for(cid: str) -> list[int]:
        if cid not in ctx_ids_cache:
            ctx_ids_cache[cid] = BANK29.context_token_ids_2389(tok, contexts[cid])
        return ctx_ids_cache[cid]

    # ── r2: hooked grid block per candidate B ─────────────────────────
    candidates = [cfg.gen_batch] if cfg.gen_batch_explicit else list(PILOT_GEN_BATCH_CANDIDATES)
    r2: dict[int, dict] = {}
    for b in candidates:
        cfg_b = replace(cfg, gen_batch=b)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        rec = run_block(
            cfg_b,
            model,
            tok,
            bank,
            block0,
            pairs_by_id,
            donor_maps,
            contexts,
            ctx_ids_cache,
            eot,
            regime_fp + "-pilot",
            pools,
            draws,
            write_done=False,
        )
        wall = time.monotonic() - t0
        assert rec["n_rows"] > 0, "pilot r2 leg ran no rollouts"
        # Plan §12 smoke seam: the production-shape block MUST have exercised
        # a hooked generate at batch>1 with UNEQUAL prompt lengths
        # (pad-into-GatedDeltaNet-recurrence coverage on the hybrid stack).
        assert rec.get("hooked_batch_gt1_unequal"), (
            f"pilot r2 leg (gen_batch={b}) never ran a hooked batch>1 chunk with "
            "unequal prompt lengths — pad-into-recurrence coverage missing (plan §12)"
        )
        r2[b] = {
            "gen_batch": b,
            "rollouts": rec["n_rows"],
            "wall_s": wall,
            "s_per_rollout": wall / rec["n_rows"],
            "hbm_headroom_gib": _pilot_hbm_headroom_gib(),
        }
        logger.info(
            "[pilot:r2] gen_batch=%d s_per_rollout=%.3f headroom_gib=%s",
            b,
            r2[b]["s_per_rollout"],
            r2[b]["hbm_headroom_gib"],
        )
    gen_batch_selected, headroom_ok = _select_gen_batch(r2, candidates)
    cfg_sel = replace(cfg, gen_batch=gen_batch_selected)

    # ── r1: anchor-shaped unhooked leg at the selected B ──────────────
    cells0 = _block_cells(bank, block0, pairs_by_id, donor_maps)
    r1_ctx_ids: list[str] = []
    for c in cells0:
        if c["context_a"] not in r1_ctx_ids:
            r1_ctx_ids.append(c["context_a"])
    r1_draws = 2 if cfg.smoke else cfg.anchor_draws
    t0 = time.monotonic()
    n_r1 = 0
    for cell, chunk in _cell_bucketed_chunks(contexts, r1_ctx_ids, gen_batch_selected):
        outs = generate_batch(
            model,
            tok,
            [contexts[cid] for cid in chunk],
            n=r1_draws,
            hook=None,
            max_new_tokens=_resolve_cap(cfg, cell),
            temperature=ANCHOR_TEMPERATURE,
            seed_base=cfg.seed_base,
            render_fn=BANK29.render_context_2389,
            ids_fn=BANK29.context_token_ids_2389,
        )
        n_r1 += sum(len(o) for o in outs)
    r1 = {"rollouts": n_r1, "wall_s": time.monotonic() - t0, "draws": r1_draws}
    assert n_r1 > 0, "pilot r1 leg ran no rollouts"
    r1["s_per_rollout"] = r1["wall_s"] / n_r1
    logger.info("[pilot:r1] s_per_rollout=%.3f rollouts=%d", r1["s_per_rollout"], n_r1)

    # ── r3: stage2-shaped hooked GREEDY K=1 leg at the selected B ─────
    t0 = time.monotonic()
    n_r3 = 0
    for start in range(0, len(cells0), gen_batch_selected):
        chunk = cells0[start : start + gen_batch_selected]
        rows = [ids_for(c["context_a"]) for c in chunk]
        row_lengths = [len(r) for r in rows]
        stack = _arm_hook_all_layers(
            model,
            cfg_sel,
            row_lengths,
            [(c["position"],) for c in chunk],
            [c["payload"] for c in chunk],
            max(row_lengths),
        )
        try:
            outs = generate_batch(
                model,
                tok,
                [contexts[c["context_a"]] for c in chunk],
                n=1,
                hook=stack,
                max_new_tokens=_resolve_cap(cfg, block0.cell),
                temperature=STAGE2_TEMPERATURE,
                seed_base=cfg.seed_base,
                render_fn=BANK29.render_context_2389,
                ids_fn=BANK29.context_token_ids_2389,
            )
        finally:
            stack.remove()
        n_r3 += sum(len(o) for o in outs)
    r3 = {"rollouts": n_r3, "wall_s": time.monotonic() - t0}
    assert n_r3 > 0, "pilot r3 leg ran no rollouts"
    r3["s_per_rollout"] = r3["wall_s"] / n_r3
    logger.info("[pilot:r3] s_per_rollout=%.3f rollouts=%d", r3["s_per_rollout"], n_r3)

    return _enforce_pilot_gate(cfg, totals_all, r1, r2, r3, gen_batch_selected, headroom_ok)


def _enforce_pilot_gate(
    cfg: RunConfig,
    totals_all: dict,
    r1: dict,
    r2: dict[int, dict],
    r3: dict,
    gen_batch_selected: int,
    headroom_ok: bool,
) -> int:
    """Plan §7 gate 4 verdict — ACCEPT / SPLIT / REFUSE on the projected TOTAL.

    Each generation phase is projected from ITS OWN regime's measured
    s/rollout (r1 -> anchors, r2[selected] -> grid, r3 -> stage2), at the
    sweep's execution shape (#1415 batch-1 false-fire lesson). Per-phase poll
    fences = PILOT_FENCE_MULT x each projection. Verdict:

      ACCEPT  total <= PILOT_ACCEPT_WALL_H
      SPLIT   ACCEPT < total <= PILOT_REFUSAL_MULT x planned — proceed; the
              80 h sbatch TIMEOUT is a PLANNED claim-queue-resume boundary
      REFUSE  total > PILOT_REFUSAL_MULT x planned -> RC_PILOT_GATE

    The report lands in gates/ (gen_batch_selected is read back by
    ``_adopt_pilot_gen_batch`` — plan §4.7 item 3).
    """
    width = max(1, cfg.num_workers)
    n_contexts = len(BANK.build_contexts())
    phases = {
        "anchors": {
            "rollouts": n_contexts * cfg.anchor_draws,
            "s_per_rollout": r1["s_per_rollout"],
            "basis": "r1 anchor-shaped unhooked",
        },
        "grid": {
            "rollouts": totals_all["rollouts_total"],
            "s_per_rollout": r2[gen_batch_selected]["s_per_rollout"],
            "basis": f"r2 hooked grid block at gen_batch={gen_batch_selected}",
        },
        "stage2": {
            "rollouts": STAGE2_ROLLOUT_CAP,
            "s_per_rollout": r3["s_per_rollout"],
            "basis": "r3 hooked greedy K=1 (replace-mode arming; ROLLOUT_CAP upper bound)",
        },
    }
    for p in phases.values():
        p["projected_wall_h"] = p["s_per_rollout"] * p["rollouts"] / width / 3600.0
        p["fence_h"] = PILOT_FENCE_MULT * p["projected_wall_h"]
    projected_total_h = sum(p["projected_wall_h"] for p in phases.values())
    refusal_threshold_h = PILOT_REFUSAL_MULT * cfg.planned_wall_h
    if projected_total_h <= PILOT_ACCEPT_WALL_H:
        verdict = "ACCEPT"
    elif projected_total_h <= refusal_threshold_h:
        verdict = "SPLIT"
    else:
        verdict = "REFUSE"
    refuse = verdict == "REFUSE"
    report = {
        "criterion": (
            "three-regime generation-throughput pilot at P2 entry "
            "(plan §7 gate 4; ACCEPT <= 40 h / SPLIT <= 3x planned / REFUSE)"
        ),
        "regimes": {
            "r1_anchor_unhooked": r1,
            "r2_hooked_grid": {str(b): m for b, m in sorted(r2.items())},
            "r3_stage2_greedy": r3,
        },
        "gen_batch_selected": gen_batch_selected,
        "gen_batch_candidates": sorted(r2),
        "hbm_headroom_floor_gib": PILOT_HBM_HEADROOM_GIB,
        "headroom_ok": headroom_ok,
        "num_workers": width,
        "gpu_name": _pilot_gpu_name(),  # round-5 C: reuse checks the device lane
        "gpu_total_mem_gib": _pilot_gpu_mem_gib(),  # round-5 J: memory identity
        "phases": phases,
        "projected_total_wall_h": projected_total_h,
        "planned_total_wall_h": cfg.planned_wall_h,
        "accept_threshold_h": PILOT_ACCEPT_WALL_H,
        "refusal_threshold_h": refusal_threshold_h,
        "verdict": verdict,
        "split_note": (
            "SPLIT proceeds: the 80 h sbatch TIMEOUT is a PLANNED claim-queue-resume "
            "boundary (plan §9), not a failure"
        ),
        "sweep_allowed": not refuse,
        "forced": cfg.force_past_halt_gates,
        "repro": _repro(cfg),
    }
    _write_json_atomic(cfg.gates_dir / "pilot_gate_report.json", report)
    logger.info(
        "[phase=pilot_done] verdict=%s gen_batch=%d projected_total_wall_h=%.2f (planned %.2f) "
        "fences_h anchors=%.2f grid=%.2f stage2=%.2f",
        verdict,
        gen_batch_selected,
        projected_total_h,
        cfg.planned_wall_h,
        phases["anchors"]["fence_h"],
        phases["grid"]["fence_h"],
        phases["stage2"]["fence_h"],
    )
    if refuse and not cfg.force_past_halt_gates:
        logger.error(
            "[pilot_gate] projected total wall %.2f h > %.1fx planned %.2f h — refusing the run "
            "(pass --force-past-halt-gates to override, or descope per the plan §9 ladder)",
            projected_total_h,
            PILOT_REFUSAL_MULT,
            cfg.planned_wall_h,
        )
        return RC_PILOT_GATE
    return RC_OK


# ── margin phase (pools-dependent TF legs) ────────────────────────────


# ── cap-hit report + cell-restricted re-gen (registered >2%/cell trigger) ─────
#
# The plan registers (Source #2162): max_new_tokens=2048, cap-hit > 2% per
# cell => re-generate those rows at 4096. These phases make the trigger
# MEASURED (cap_report) and its remedy EXECUTABLE (capregen) instead of
# registered-but-unenforced. Field names follow the parent's hand-derived
# eval_results/issue_2162/f_metrics/grid_caphit_aggregate.json where they
# apply (derived_from / derived_from_sha256 / derivation / n_rows /
# cap_hit_rows / cap_hit_pct / pre_registered_regen_trigger_pct /
# trigger_fired) so the child artifact is comparable to the parent's; the
# per-cell and per-(cell, value) breakdowns are new here.


def compute_cap_hit_report(
    shard_paths: list[Path],
    max_new_tokens: int,
    *,
    scope: str,
    expected_shards: set[str] | None,
    expected_unavailable_reason: str | None = None,
    threshold_pct: float = CAP_HIT_REGEN_TRIGGER_PCT,
) -> dict:
    """Per-cell + per-(cell, value) cap-hit aggregate over realized rollout shards.

    INCREMENTAL BY DESIGN: accepts whatever shards exist so far. A read over
    an incomplete set is labeled ``partial: true`` with the realized row
    count, the covered/missing shard names, and the reason — a partial read
    can never be mistaken for a final one. Counting basis is each row's
    RECORDED ``cap_hit`` (computed at generation time against the cap then
    in force), so the report stays correct over a mixed-cap store after a
    cell-restricted re-gen. Shards not yet capture-enriched (the two-write
    pattern's text-only first write) are EXCLUDED from counts and listed
    under ``pending_capture_shards`` — never silently counted as zero hits.
    An EMPTY breach list is a legitimate outcome (``breaching_cells: []``,
    ``trigger_fired: false``), never coerced. Raises when no shard exists or
    nothing is capture-enriched yet (a zero-row aggregate is not a
    measurement)."""
    if not shard_paths:
        raise RuntimeError(
            f"cap-hit report ({scope}): no rollout shards found — wrong --out-root, "
            "or the phase has not written any shard yet"
        )
    covered: list[dict] = []
    pending: list[str] = []
    cell_counts: dict[str, list[int]] = {}
    cv_counts: dict[tuple[str, str], list[int]] = {}
    cell_caps: dict[str, dict[str, set[int]]] = {}
    n_rows = 0
    hits = 0
    n_tok_all: list[int] = []
    realized_caps: set[int] = set()
    value_fields: set[str] = set()
    for path in sorted(shard_paths, key=lambda p: p.name):
        data = path.read_bytes()  # ONE read: rows + sha come from the same bytes
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        if not rows:
            raise RuntimeError(f"cap-hit report ({scope}): {path} is EMPTY — malformed shard")
        enriched = [("cap_hit" in r) for r in rows]
        if not all(enriched):
            if any(enriched):
                raise RuntimeError(
                    f"cap-hit report ({scope}): {path} mixes capture-enriched and "
                    "text-only rows — corrupt shard"
                )
            pending.append(path.name)
            continue
        for r in rows:
            cell = r["cell"]
            if "value_id" in r:
                vkey, vfield = str(r["value_id"]), "value_id"
            elif "value_a" in r:
                vkey, vfield = str(r["value_a"]), "value_a"
            else:
                raise RuntimeError(
                    f"cap-hit report ({scope}): row in {path.name} has neither "
                    "value_id nor value_a — cannot key the per-(cell, value) breakdown"
                )
            value_fields.add(vfield)
            hit = 1 if r["cap_hit"] else 0
            n_rows += 1
            hits += hit
            n_tok_all.append(int(r["n_completion_tokens"]))
            rcap = int(r.get("max_new_tokens", max_new_tokens))
            realized_caps.add(rcap)
            # Per-(cell x batch) realized caps: after a batch-scoped capregen
            # the store is legitimately MIXED (gate slice regenerated at 4096
            # while rest is still at 2048) — this is how a consumer tells a
            # COMPLETED gate-slice re-gen ([4096] in the gate entry) from a
            # half-done one ([2048, 4096]). anchors rows carry gate_slice;
            # rows without it (grid) aggregate under "all".
            bkey = ("gate" if r["gate_slice"] else "rest") if "gate_slice" in r else "all"
            cell_caps.setdefault(cell, {}).setdefault(bkey, set()).add(rcap)
            c = cell_counts.setdefault(cell, [0, 0])
            c[0] += 1
            c[1] += hit
            cv = cv_counts.setdefault((cell, vkey), [0, 0])
            cv[0] += 1
            cv[1] += hit
        covered.append({"name": path.name, "sha256": _sha256_bytes(data), "n_rows": len(rows)})
    if n_rows == 0:
        raise RuntimeError(
            f"cap-hit report ({scope}): every present shard is still text-only "
            "(pre-capture) — nothing capture-enriched to measure yet; retry after "
            "the first block/batch completes its capture pass"
        )
    per_cell: dict[str, dict] = {}
    for cell, (n, h) in sorted(cell_counts.items()):
        pct = 100.0 * h / n
        per_cell[cell] = {
            "n_rows": n,
            "cap_hit_rows": h,
            "cap_hit_pct": pct,
            "breach": pct > threshold_pct,  # STRICT >: exactly threshold does NOT fire
            "realized_caps_by_batch": {k: sorted(v) for k, v in sorted(cell_caps[cell].items())},
        }
    per_cell_value: dict[str, dict[str, dict]] = {}
    for (cell, vkey), (n, h) in sorted(cv_counts.items()):
        per_cell_value.setdefault(cell, {})[vkey] = {
            "n_rows": n,
            "cap_hit_rows": h,
            "cap_hit_pct": 100.0 * h / n,
        }
    max_spread: dict | None = None
    for cell, vals in per_cell_value.items():
        if len(vals) < 2:
            continue
        pcts = [d["cap_hit_pct"] for d in vals.values()]
        spread = max(pcts) - min(pcts)
        if max_spread is None or spread > max_spread["spread_pct"]:
            max_spread = {
                "cell": cell,
                "min_pct": min(pcts),
                "max_pct": max(pcts),
                "spread_pct": spread,
            }
    covered_names = sorted(c["name"] for c in covered)
    reasons: list[str] = []
    if pending:
        reasons.append(f"{len(pending)} shard(s) pending capture enrichment")
    missing: list[str] | None = None
    unexpected: list[str] = []
    if expected_shards is None:
        reasons.append(expected_unavailable_reason or "expected shard set unavailable")
    else:
        present = set(covered_names) | set(pending)
        missing = sorted(expected_shards - present)
        unexpected = sorted(present - expected_shards)
        if unexpected:
            # v11 M2 / codex cap-report-finality-clobber: a foreign/stale shard
            # must never enter the re-gen trigger statistic — it can flip
            # per-cell breach membership in either direction, and cap_report /
            # capregen have no entry sweep protecting them. Fail loud, never
            # count-and-note (an unexpected shard can also never ride into a
            # report that then claims partial: false).
            raise RuntimeError(
                f"cap-hit report ({scope}): {len(unexpected)} shard(s) present but NOT in "
                f"the expected set (first: {unexpected[:8]}) — quarantine the foreign/"
                "stale shard(s) or fix the expected-set derivation; refusing to count "
                "them into the re-gen trigger statistic"
            )
        if missing:
            reasons.append(f"{len(missing)} expected shard(s) missing (phase incomplete)")
    toks = np.asarray(n_tok_all, dtype=np.int64)
    breaching = sorted(c for c, d in per_cell.items() if d["breach"])
    return {
        "scope": scope,
        "derived_from": (
            f"{scope} rollout shards "
            f"({len(covered)} capture-enriched of {len(covered) + len(pending)} present)"
        ),
        "derived_from_shards": covered,
        "derived_from_sha256": _sha256_bytes(
            "\n".join(f"{c['name']}:{c['sha256']}" for c in covered).encode()
        ),
        "derivation": (
            "count of rows with truthy recorded cap_hit over all capture-enriched rows, "
            "overall + per cell + per (cell, value); trigger_fired = any cell with "
            "cap_hit_pct STRICTLY > pre_registered_regen_trigger_pct; derived_from_sha256 "
            "= sha256 over newline-joined '<name>:<sha256>' of derived_from_shards"
        ),
        "n_rows": n_rows,
        "cap_hit_rows": hits,
        "cap_hit_frac": hits / n_rows,
        "cap_hit_pct": 100.0 * hits / n_rows,
        "pre_registered_regen_trigger_pct": threshold_pct,
        "trigger_fired": bool(breaching),
        "breaching_cells": breaching,
        "max_new_tokens": max_new_tokens,
        "realized_row_caps": sorted(realized_caps),
        "value_key_fields": sorted(value_fields),
        "per_cell": per_cell,
        "per_cell_value": per_cell_value,
        "max_value_spread": max_spread,
        "n_completion_tokens": {
            "min": int(toks.min()),
            "median": float(np.median(toks)),
            "p95": float(np.percentile(toks, 95)),
            "p99": float(np.percentile(toks, 99)),
            "max": int(toks.max()),
        },
        "partial": bool(reasons),
        "partial_reason": reasons or None,
        "covered_shards": covered_names,
        "pending_capture_shards": sorted(pending),
        "missing_shards": missing,
        "unexpected_shards": unexpected,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _cap_report_inputs(
    cfg: RunConfig, scope: str
) -> tuple[list[Path], set[str] | None, str | None]:
    """``(shard_paths, expected_shard_names, expected_unavailable_reason)``.

    anchors: expected set from the family's own done records (a 0-row worker
    writes no jsonl, so only n_rows>0 records expect a file); width
    unresolved => completeness underivable (partial). grid: expected block
    set derived mechanically from the frozen bank (smoke_blocks under
    --smoke, mirroring phase_grid's enumeration); bank absent => partial."""
    if scope == "anchors":
        # B3 (r1 review): ONE aggregate over EVERY engine/batch namespace —
        # HF gate/rest cell shards, parity-HF cell shards, AND production-
        # vLLM cell shards (the pre-fix version excluded anchors_vllm_*
        # entirely — vLLM cells vanished from the >2% trigger — and refused
        # anchors_parity_* as foreign, aborting the phase-end snapshot).
        # Expectations derive from the completed engine/batch done manifests
        # (worker-independent cell grain); a vLLM gen-done sentinel expects
        # its text-persisted shard, which stays in pending_capture_shards
        # until the capture pass enriches it. Completeness = every planned
        # cell-batch unit (gate cells; rest cells under ANY of rest_/parity_/
        # vllm_ ownership) carries a manifest; short of that the expected set
        # is underivable and the report is PARTIAL (the unexpected-shard
        # refusal stays armed only on a fully-derived set).
        paths = sorted(cfg.anchors_dir.glob("anchors_*_w*.jsonl"))
        expected: set[str] = set()
        covered: set[str] = set()  # batch_ids: gate_X / rest_X / parity_X / vllm_X

        def _batch_id_of(stem: str) -> str | None:
            head, sep, idx = stem.rpartition("_w")
            if not sep or not idx.isdigit() or not head.startswith("anchors_"):
                return None
            return head[len("anchors_") :]

        for m in sorted(cfg.manifest_dir.glob("anchors_*_w*_done.json")):
            if m.name.endswith("_gen_done.json"):
                stem = m.name[: -len("_gen_done.json")]
                bid = _batch_id_of(stem)
                if bid is not None:
                    expected.add(f"{stem}.jsonl")
                    covered.add(bid)
                continue
            stem = m.name[: -len("_done.json")]
            bid = _batch_id_of(stem)
            if bid is None:
                continue
            rec = json.loads(m.read_text())
            if int(rec.get("n_rows", 0)) > 0:
                expected.add(f"{stem}.jsonl")
            covered.add(bid)
        gate_ids, rest_ids, contexts = _anchor_context_order(cfg)
        uncovered = [
            f"gate_{c}" for c in _group_by_cell(gate_ids, contexts) if f"gate_{c}" not in covered
        ] + [
            c
            for c in _group_by_cell(rest_ids, contexts)
            if not ({f"rest_{c}", f"parity_{c}", f"vllm_{c}"} & covered)
        ]
        if uncovered:
            return (
                paths,
                None,
                (
                    f"{len(uncovered)} anchor cell-batch unit(s) lack done manifests "
                    f"(phase incomplete; first: {uncovered[:6]}) — expected shard set "
                    "underivable"
                ),
            )
        return paths, expected, None
    assert scope == "grid", scope
    paths = sorted(cfg.rollouts_dir.glob("shard_*.jsonl"))
    if not (cfg.bank_dir / "vc_bank.pt").exists():
        return (
            paths,
            None,
            (f"{cfg.bank_dir / 'vc_bank.pt'} absent — expected block set underivable"),
        )
    bank = _load_bank(cfg)
    manifest = _load_frozen_manifest(cfg)
    pairs = surviving_pairs(manifest)
    base = smoke_blocks(pairs) if cfg.smoke else enumerate_blocks(pairs)
    blocks, _excl = apply_pe_exclusions(
        base, no_prefix_ids(manifest), bank["donor_assignments"], pairs
    )
    return paths, {f"shard_{b.slug}.jsonl" for b in blocks}, None


def cap_hit_report_path(cfg: RunConfig, scope: str, *, postregen: bool = False) -> Path:
    """Canonical report path (manifests/ rides the existing P5 upload row).

    ``postregen=True`` names the SIBLING path for post-regen measurements —
    NEVER ``_load_breach_report``'s default read path (code-review v11 C1: the
    post-regen emit used to clobber the driving report in place)."""
    suffix = "_postregen" if postregen else ""
    return cfg.manifest_dir / f"cap_hit_report_{scope}{suffix}.json"


def capregen_breach_basis_path(cfg: RunConfig, scope: str) -> Path:
    """Capregen-owned FROZEN byte-copy of the driving pre-regen report.

    Written ONCE by the first ``_load_breach_report`` of a campaign; every
    later invocation (the deferred Phase B batch, crashed-worker respawns,
    idempotent re-entries) reads THIS file, so no later write to the default
    report path — a post-regen emit, a base-phase re-entry snapshot over the
    mixed store, a stray ``--phase cap_report`` — can change which cells the
    registered remedy regenerates (v11 C1 faces 1/2/4)."""
    return cfg.manifest_dir / f"capregen_breach_basis_{scope}.json"


def preregen_superseded_dir(cfg: RunConfig, scope: str) -> Path:
    """Durable home for pre-regen rollout files a capregen supersedes (FIX 2).

    OUT-ROOT-TOP, deliberately outside every consumer glob (all non-recursive)
    AND outside every existing upload allow-pattern (hub allow_patterns are
    fnmatch and cross '/', so a subdir of anchors_dir/rollouts_dir would ride
    their P5 rows under the wrong prefix) — the preserved text gets its OWN
    in-phase uploads + P5 row at raw_completions/preregen_superseded/ instead.
    Plan §10 declares ``discarded_artifacts: none``: without this preservation
    the merge/overwrite would discard the truncated completions that EVIDENCE
    the asymmetric-truncation finding the remedy exists to fix, leaving them
    recoverable only via HF revision history (not an acceptable declared home,
    CLAUDE.md § Upload Policy)."""
    return cfg.out_root / "preregen_superseded" / scope


def _preserve_preregen_file(cfg: RunConfig, scope: str, src: Path) -> Path:
    """Write-once atomic byte-copy of a rollout file capregen will supersede.

    Ordered BEFORE the overwrite/merge at every call site. Write-once is
    load-bearing for idempotent re-entries: after a crash-between-writes the
    re-merged store already carries REGENERATED rows, so a second copy would
    clobber the TRUE pre-regen bytes — an existing destination is always the
    authentic pre-regen capture and is never overwritten. The copy itself
    rides ``_atomic_replace`` (process-unique tmp + ``os.replace``), so an
    interrupted preservation leaves NO destination and NO tmp residue."""
    dest = preregen_superseded_dir(cfg, scope) / src.name
    if dest.exists():
        return dest
    if not src.exists():
        raise RuntimeError(
            f"capregen preservation ({scope}): source {src} missing — a breaching "
            "unit's pre-regen rollout file must exist (the breach basis was "
            "measured on it)"
        )
    with _atomic_replace(dest) as tmp:
        tmp.write_bytes(src.read_bytes())
    logger.info("[capregen:%s] preserved superseded pre-regen rows -> %s", scope, dest)
    return dest


def emit_cap_hit_report(
    cfg: RunConfig,
    scope: str,
    *,
    postregen: bool = False,
    base_cap: int | None = None,
    capregen_pending: list[str] | None = None,
) -> dict:
    """Compute + atomically write the scope's cap-hit report; returns it.

    Post-regen mode (v11 C1): the row-cap attribution DEFAULT is the BASE cap
    (rows lacking the per-row ``max_new_tokens`` field are base-run rows — the
    live store was generated by pre-diff code that wrote no per-row cap, so
    attributing the CURRENT raised cap corrupted ``realized_row_caps`` /
    ``realized_caps_by_batch``: a half-done Phase A read ``gate: [4096]``,
    indistinguishable from complete); the destination is the ``*_postregen``
    SIBLING path (mechanically pinned — never the driving report's path); the
    report is stamped ``postregen: true`` (refused as a breach basis); and any
    pending capregen merges mark it ``partial`` (a per-worker emit mid-fleet
    must never claim ``partial: false``, v11 face 4 / efficiency (d))."""
    if postregen and base_cap is None:
        raise ValueError("postregen emit requires base_cap (the BASE attribution cap)")
    paths, expected, why = _cap_report_inputs(cfg, scope)
    report = compute_cap_hit_report(
        paths,
        base_cap if postregen else cfg.max_new_tokens,
        scope=scope,
        expected_shards=expected,
        expected_unavailable_reason=why,
    )
    report["repro"] = _repro(cfg)
    out = cap_hit_report_path(cfg, scope, postregen=postregen)
    if postregen:
        # v11 C1 mechanical pin: the post-regen measurement must never land on
        # either of _load_breach_report's read paths (default report / basis).
        assert out != cap_hit_report_path(cfg, scope), out
        assert out != capregen_breach_basis_path(cfg, scope), out
        report["postregen"] = True
        report["base_max_new_tokens"] = int(base_cap)  # type: ignore[arg-type]
        report["raised_max_new_tokens"] = cfg.max_new_tokens
        basis = capregen_breach_basis_path(cfg, scope)
        if basis.exists():
            report["breach_basis"] = {
                "path": basis.name,
                "sha256": _sha256_bytes(basis.read_bytes()),
            }
        pend = sorted(capregen_pending or [])
        report["capregen_pending"] = pend
        if pend:
            report["partial"] = True
            reasons = list(report["partial_reason"] or [])
            head = ", ".join(pend[:12]) + (", ..." if len(pend) > 12 else "")
            reasons.append(f"{len(pend)} capregen merge(s) pending: {head}")
            report["partial_reason"] = reasons
    _write_json_atomic(out, report)
    logger.info(
        "[cap_report:%s] rows=%d cap_hit=%.4f%% breaching_cells=%d partial=%s postregen=%s -> %s",
        scope,
        report["n_rows"],
        report["cap_hit_pct"],
        len(report["breaching_cells"]),
        report["partial"],
        postregen,
        out,
    )
    return report


_CAP_SCOPE_GLOBS: dict[str, tuple[str, str]] = {
    "anchors": ("anchors_dir", "anchors_*_w*.jsonl"),
    "grid": ("rollouts_dir", "shard_*.jsonl"),
}


def _emit_cap_hit_snapshot(cfg: RunConfig, scope: str) -> None:
    """Phase-end auto-emit, tolerant of the two legitimate early states.

    A zero-context worker finishing FIRST can see no shard at all, and a
    worker can finish while every present sibling shard is still text-only
    (pre-capture) — both are transient run states, not defects, so the
    snapshot is SKIPPED WITH A WARNING instead of crashing a healthy worker;
    the last-finishing worker (and the standalone ``--phase cap_report``,
    which keeps the fail-loud empty-selection raise) writes the real thing."""
    attr, pat = _CAP_SCOPE_GLOBS[scope]
    paths = sorted(getattr(cfg, attr).glob(pat))

    def _first_row_enriched(p: Path) -> bool:
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return "cap_hit" in json.loads(line)
        return False

    if not paths or not any(_first_row_enriched(p) for p in paths):
        logger.warning(
            "[cap_report:%s] no capture-enriched shard present yet — snapshot "
            "skipped (run --phase cap_report later for the aggregate)",
            scope,
        )
        return
    emit_cap_hit_report(cfg, scope)


def phase_cap_report(cfg: RunConfig) -> int:
    """Standalone (re-)aggregation over whatever rollout shards exist so far."""
    logger.info("[phase=cap_report] scope=%s", cfg.cap_scope)
    scopes = ("anchors", "grid") if cfg.cap_scope == "both" else (cfg.cap_scope,)
    for scope in scopes:
        emit_cap_hit_report(cfg, scope)
    logger.info("[phase=cap_report_done]")
    return RC_OK


def _validate_breach_basis(rep: dict, path: Path, scope: str, cfg: RunConfig) -> None:
    """Refusals a capregen basis must clear (all fail-loud, no override flags).

    Missing/mismatched scope; a POST-regen measurement (``postregen: true``
    stamp, or a store whose realized caps are MIXED within any (cell, batch)
    bucket — post-regen or half-done by construction: under the plan §4.7
    item-1 per-cell cap table the BASE store is legitimately mixed ACROSS
    cells, and across gate/rest batches of one cell after a gate-slice
    recalibration, but every (cell, batch) bucket runs at exactly ONE cap);
    a report missing the ``partial`` field entirely (absence is never
    finality — hand-built files fail loud, v11 minor); a PARTIAL report (the
    registered re-gen evaluates per-cell rates on the COMPLETE phase); an
    EMPTY realized-cap set (v12 Dispute-1 residual: a zero-row / wrong-store
    basis must refuse BEFORE the freeze); a --max-new-tokens below 2x the
    MAX generating cap among the BREACHING cells (codex BLOCKER
    regen-cap-not-enforced: the registered remedy is re-gen at >= 2x the
    cap, plan §-line-105 / CLAUDE.md; per-cell caps make "the" generating
    cap per-cell, so the floor is 2x the largest breaching cell's cap — a
    sub-2x cap silently violates the recipe AND leaves the long tail
    truncated, surviving the very bias this remedy exists to remove)."""
    if rep.get("scope") != scope:
        raise RuntimeError(f"breach report {path} has scope={rep.get('scope')!r}, need {scope!r}")
    if rep.get("postregen"):
        raise RuntimeError(
            f"breach report {path} is a POST-regen measurement (postregen: true) — it can "
            "never drive a capregen basis (v11 C1: regenerated rows dilute per-cell rates, "
            "silently under-scoping the registered remedy); the frozen pre-regen basis "
            f"lives at {capregen_breach_basis_path(cfg, scope)}"
        )
    caps = [int(c) for c in rep.get("realized_row_caps") or []]
    if not caps:
        raise RuntimeError(
            f"breach report {path} carries NO realized row caps — measured over zero rows "
            "or a store without per-row cap telemetry; never a capregen basis (v12 "
            "Dispute-1 residual: such a basis would FREEZE at "
            f"{capregen_breach_basis_path(cfg, scope)} and wedge the campaign)"
        )
    per_cell = rep.get("per_cell") or {}
    mixed_buckets = {
        cell: d.get("realized_caps_by_batch")
        for cell, d in per_cell.items()
        if any(len(v) != 1 for v in (d.get("realized_caps_by_batch") or {"?": []}).values())
    }
    if mixed_buckets:
        raise RuntimeError(
            f"breach report {path} measured MIXED realized caps within a (cell, batch) "
            f"bucket ({mixed_buckets}) — post-regen or half-done by construction, never a "
            "capregen basis; an ESCALATED re-gen on an already-regenerated store runs on "
            "a fresh --out-root (the stacking refusals forbid it here anyway)"
        )
    if "partial" not in rep:
        raise RuntimeError(
            f"breach report {path} lacks the 'partial' field — absence is not finality; "
            "not a cap-hit report this driver recognizes"
        )
    if rep["partial"]:
        raise RuntimeError(
            f"breach report {path} is PARTIAL ({rep.get('partial_reason')}) — the "
            "registered re-gen trigger evaluates per-cell rates on the COMPLETE "
            f"phase; re-run --phase cap_report after the {scope} phase completes"
        )
    breaching = list(rep.get("breaching_cells") or [])
    missing_pc = [c for c in breaching if c not in per_cell]
    if missing_pc:
        raise RuntimeError(
            f"breach report {path} names breaching cells absent from per_cell: {missing_pc}"
        )
    if breaching:
        breach_base_cap = max(
            max(max(v) for v in per_cell[c]["realized_caps_by_batch"].values()) for c in breaching
        )
        if cfg.max_new_tokens < 2 * breach_base_cap:
            raise RuntimeError(
                "capregen requires --max-new-tokens >= 2x the largest generating cap "
                f"among the breaching cells ({breach_base_cap}; registered remedy: "
                f"{2 * breach_base_cap}, plan §-line-105 / CLAUDE.md 're-generate at "
                f">= 2x the cap'); got {cfg.max_new_tokens} — a sub-2x re-gen cap "
                "violates the registered recipe and leaves the long tail truncated"
            )


def _load_breach_report(cfg: RunConfig, scope: str) -> tuple[dict, Path]:
    """Load + validate + FREEZE the cap-hit report driving a capregen campaign.

    The first invocation of a campaign validates the source report
    (``--breach-report`` when given, else the default report path) and freezes
    a byte-verbatim copy at ``capregen_breach_basis_path`` — the atomic-replace
    copy keeps the source's sha256, so done-record provenance is stable. Every
    later invocation — Phase B of the batch split, crashed-worker respawns,
    idempotent re-entries — loads the FROZEN basis, making the pre-regen breach
    set immutable for the whole campaign (v11 C1: post-regen emits / re-runs of
    --phase cap_report over the mixed store can no longer wedge Phase B, block
    a respawn, or silently launder the breach list). A ``--breach-report``
    passed alongside an existing basis must match it byte-for-byte — one
    campaign keys off ONE basis. Validation re-runs on every load (defense in
    depth against a hand-planted basis file). Returns ``(report, basis_path)``;
    all done-record ``source_report`` provenance therefore names the basis."""
    basis = capregen_breach_basis_path(cfg, scope)
    if basis.exists():
        if cfg.breach_report is not None:
            b_sha = _sha256_bytes(basis.read_bytes())
            e_sha = _sha256_bytes(cfg.breach_report.read_bytes())
            if b_sha != e_sha:
                raise RuntimeError(
                    f"--breach-report {cfg.breach_report} (sha256 {e_sha[:12]}) != the frozen "
                    f"capregen basis {basis} (sha256 {b_sha[:12]}) — one campaign keys off ONE "
                    "pre-regen basis; a different basis needs a fresh --out-root"
                )
        rep = json.loads(basis.read_text())
        _validate_breach_basis(rep, basis, scope, cfg)
        return rep, basis
    src = cfg.breach_report or cap_hit_report_path(cfg, scope)
    if not src.exists():
        raise RuntimeError(
            f"breach report {src} missing — run --phase cap_report --cap-scope {scope} first"
        )
    payload = src.read_bytes()
    rep = json.loads(payload.decode("utf-8"))
    _validate_breach_basis(rep, src, scope, cfg)
    with _atomic_replace(basis) as tmp:
        tmp.write_bytes(payload)  # byte-verbatim freeze: sha-stable provenance
    logger.info("[capregen:%s] froze breach basis %s -> %s", scope, src, basis)
    return rep, basis


def _capregen_block_done(
    out_root: Path, block: Block, base_fp: str, regen_cap: int, namespace: str = "blocks"
) -> bool:
    """Capregen resume predicate over the SAME done files as ``block_is_done``.

    The #722 r3 hard refusal is PRESERVED: any done record at a regime_fp
    other than the BASE run's raises, never skips. On top of it: a pre-regen
    done record (no ``capregen`` sub-record) is PENDING — a stale done-file
    can never let a breaching block skip re-generation — and a capregen
    record at a DIFFERENT raised cap raises (mixed raised caps within one
    scope are not sanctioned)."""
    path = block_done_path(out_root, block, namespace)
    if not path.exists():
        return False
    rec = json.loads(path.read_text())
    if rec.get("key") != block.key:
        raise RuntimeError(f"block done-file key mismatch: {rec.get('key')!r} != {block.key!r}")
    if rec.get("regime_fp") != base_fp:
        raise RuntimeError(
            f"block {block.key} done-file carries regime_fp={rec.get('regime_fp')!r} "
            f"but the capregen BASE regime_fp={base_fp!r} — refusing to re-gen across "
            "regimes (quarantine or use a fresh --out-root)"
        )
    cr = rec.get("capregen")
    if cr is None:
        return False
    if int(cr.get("max_new_tokens", -1)) != regen_cap:
        raise RuntimeError(
            f"block {block.key} was already re-generated at "
            f"max_new_tokens={cr.get('max_new_tokens')} != this invocation's "
            f"{regen_cap} — refusing to mix raised caps (fresh --out-root to redo)"
        )
    return True


def phase_capregen_grid(cfg: RunConfig) -> int:
    """Cell-restricted grid re-gen at a raised cap (registered >2% remedy).

    Reuses the EXISTING machinery end to end: the same block enumeration
    filtered to the breach list, the same ``run_block`` (whole-block
    regenerate — draws are stochastic, so within a breaching cell every row
    is regenerated at the raised cap; mixed caps ACROSS cells are the
    sanctioned end state), the same claim-file queue + per-block checkpoints
    + incremental text upload, 8-wide like the phase it amends. Margins are
    NOT recomputed (teacher-forced pool scoring — independent of the
    generation cap), so margin shards/done-files stay untouched. Regenerated
    va shards persist via a follow-up ``--phase upload``."""
    if cfg.capregen_batch is not None:
        raise RuntimeError(
            "--capregen-batch applies to --capregen-scope anchors only "
            "(the grid has no gate/rest batch dimension)"
        )
    logger.info(
        "[phase=capregen] scope=grid worker=%d/%d smoke=%s",
        cfg.worker_index,
        cfg.num_workers,
        cfg.smoke,
    )
    rep, rep_path = _load_breach_report(cfg, "grid")
    breach = set(rep["breaching_cells"])
    if not breach:
        logger.info(
            "[capregen:grid] breach list EMPTY (trigger_fired=false) — nothing to "
            "re-generate; exiting rc=0"
        )
        return RC_OK
    base_cap = int(rep["max_new_tokens"])
    bank = _load_bank(cfg)
    manifest = _load_frozen_manifest(cfg)
    frozen_sha = _sha256_bytes(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode())
    assert frozen_sha == str(bank.get("bank_sha")), (
        "frozen bank.json sha != vc_bank.pt bank_sha — bank/capture drift",
        frozen_sha,
        bank.get("bank_sha"),
    )
    pairs = surviving_pairs(manifest)
    np_ids = no_prefix_ids(manifest)
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_maps = bank["donor_assignments"]
    bank_sha = str(bank.get("bank_sha"))
    base_fp = regime_fingerprint(replace(cfg, max_new_tokens=base_cap), bank_sha)
    regen_fp = regime_fingerprint(cfg, bank_sha)
    draws = SMOKE_GRID_DRAWS if cfg.smoke else cfg.grid_draws
    base_blocks = smoke_blocks(pairs) if cfg.smoke else enumerate_blocks(pairs)
    runnable, _excl = apply_pe_exclusions(base_blocks, np_ids, donor_maps, pairs)
    blocks = [b for b in runnable if b.cell in breach]
    unmatched = breach - {b.cell for b in blocks}
    if unmatched:
        raise RuntimeError(
            f"breaching cells matched no runnable grid blocks: {sorted(unmatched)} — "
            "report/run regime mismatch (smoke vs full?)"
        )
    logger.info(
        "[capregen:grid] %d breaching cells -> %d blocks at max_new_tokens=%d (base %d)",
        len(breach),
        len(blocks),
        cfg.max_new_tokens,
        base_cap,
    )
    done_extra = {
        "capregen": {
            "max_new_tokens": cfg.max_new_tokens,
            "base_max_new_tokens": base_cap,
            "regen_regime_fp": regen_fp,
            "source_report": rep_path.name,
            "source_report_sha256": _sha256_bytes(rep_path.read_bytes()),
            # FIX 2: where a consumer / the upload-verifier finds the
            # superseded pre-regen rollout text this block's re-gen replaced.
            "preregen_dir": (
                preregen_superseded_dir(cfg, "grid").relative_to(cfg.out_root).as_posix()
            ),
            "preregen_hf_prefix": f"{HF_PREFIX}/raw_completions/preregen_superseded/grid",
            "ts": datetime.now(UTC).isoformat(),
        }
    }
    model, tok = load_model_and_tokenizer(cfg)
    eot = eot_tail_ids(tok)
    contexts = BANK.build_contexts()
    ctx_ids_cache: dict[str, list[int]] = {}
    n_run = 0
    uploaded: list[str] = []
    pending: list[Block] = []
    preserved: list[str] = []

    def run_one(block: Block) -> None:
        nonlocal n_run, uploaded, pending
        t0 = time.monotonic()
        # FIX 2: byte-preserve the pre-regen shard BEFORE run_block overwrites
        # it (write-once — an idempotent re-entry never clobbers the true
        # pre-regen bytes with regenerated content).
        _preserve_preregen_file(cfg, "grid", cfg.rollouts_dir / f"shard_{block.slug}.jsonl")
        preserved.append(f"shard_{block.slug}.jsonl")
        de = done_extra
        prior_done_path = block_done_path(cfg.out_root, block)
        if prior_done_path.exists():
            prior = json.loads(prior_done_path.read_text())
            if "margin_inline" in prior:
                # v11 minor: pools=None here never recomputes margins, but the
                # BASE run's margin shard + margin_blocks done-record twin stay
                # valid (TF margins are cap-independent) — carry the base flag
                # instead of stamping margin_inline: False over it.
                de = {**done_extra, "margin_inline": prior["margin_inline"]}
        rec = run_block(
            cfg,
            model,
            tok,
            bank,
            block,
            pairs_by_id,
            donor_maps,
            contexts,
            ctx_ids_cache,
            eot,
            base_fp,  # done record keeps the BASE resume key; capregen rides done_extra
            None,  # pools=None: TF margins are cap-independent — never recomputed here
            draws,
            done_extra=de,
        )
        n_run += 1
        pending.append(block)
        logger.info(
            "[capregen:grid] unit %d/%d %s rows=%d cap_hit=%d elapsed=%.1fs",
            n_run,
            len(blocks),
            block.key,
            rec["n_rows"],
            rec["n_cap_hit"],
            time.monotonic() - t0,
        )
        if cfg.upload_every > 0 and len(pending) >= cfg.upload_every:
            uploaded += _upload_grid_increment(cfg, pending)
            pending.clear()

    def is_done(out_root: Path, block: Block, fp: str, namespace: str) -> bool:
        return _capregen_block_done(out_root, block, fp, cfg.max_new_tokens, namespace)

    stats = run_claim_queue(cfg, blocks, base_fp, "blocks", run_one, is_done=is_done)
    if pending:
        uploaded += _upload_grid_increment(cfg, pending)
        pending.clear()
    # Post-regen measurement over the mixed-cap store (v11 C1): BASE-cap row
    # attribution + the *_postregen SIBLING path — the frozen driving basis is
    # never touched; blocks siblings have not merged yet keep it partial.
    still_pending = [
        b.key
        for b in blocks
        if not _capregen_block_done(cfg.out_root, b, base_fp, cfg.max_new_tokens)
    ]
    emit_cap_hit_report(
        cfg, "grid", postregen=True, base_cap=base_cap, capregen_pending=still_pending
    )
    _write_json_atomic(
        cfg.manifest_dir / f"capregen_grid_done_w{cfg.worker_index}.json",
        {
            "scope": "grid",
            "base_regime_fp": base_fp,
            "regen_regime_fp": regen_fp,
            "max_new_tokens": cfg.max_new_tokens,
            "base_max_new_tokens": base_cap,
            "breaching_cells": sorted(breach),
            "n_blocks_target": len(blocks),
            "n_blocks_run": stats["ran"],
            "uploads": uploaded,
            "source_report": rep_path.name,
            "preregen_shards": sorted(preserved),
            "preregen_dir": (
                preregen_superseded_dir(cfg, "grid").relative_to(cfg.out_root).as_posix()
            ),
            "preregen_hf_prefix": f"{HF_PREFIX}/raw_completions/preregen_superseded/grid",
            "repro": _repro(cfg),
        },
    )
    logger.info(
        "[phase=capregen_done] scope=grid worker=%d blocks_run=%d — run --phase upload "
        "to persist regenerated va shards",
        cfg.worker_index,
        stats["ran"],
    )
    return RC_OK


def _merge_anchor_capregen(
    cfg: RunConfig,
    batch: str,
    base_cap: int,
    breach_cells: set[str],
    cell_of: dict[str, str],
    regen_rows: list[dict],
    regen_states: dict,
    done_rec: dict,
    capregen_record: dict,
) -> dict:
    """Merge regenerated breaching-cell rows into this worker's anchors shard.

    IDEMPOTENT per artifact: the keep-mask for the jsonl comes from its own
    rows' ``cell`` and for the va store from its own index's context->cell
    map, so a crash between the two writes re-merges cleanly on re-run
    (breach-cell rows are dropped wholesale whatever cap they carry, then
    the fresh regen rows are appended). Kept rows are backfilled with the
    BASE cap (provably theirs: the done record's regime_fp — which embeds
    max_new_tokens — matched the base fingerprint before this runs). Write
    order: pre-regen preservation (write-once byte-copy, FIX 2) -> jsonl ->
    va .pt -> done record (the commit point)."""
    w = cfg.worker_index
    jsonl = cfg.anchors_dir / f"anchors_{batch}_w{w}.jsonl"
    va_path = cfg.anchors_dir / f"va_anchors_{batch}_w{w}.pt"
    # FIX 2: byte-preserve the ENTIRE pre-regen shard (superseded breach-cell
    # rows included) BEFORE any write — write-once, so an idempotent re-merge
    # after a crash never clobbers the true pre-regen bytes.
    preserved = _preserve_preregen_file(cfg, "anchors", jsonl)
    capregen_record["preregen_superseded"] = {
        "local": preserved.relative_to(cfg.out_root).as_posix(),
        "hf_prefix": f"{HF_PREFIX}/raw_completions/preregen_superseded/anchors",
        "note": "byte-copy of the entire pre-regen shard (dropped breach-cell rows incl.)",
    }
    old_rows = [json.loads(line) for line in jsonl.open(encoding="utf-8") if line.strip()]
    keep_rows = [r for r in old_rows if r["cell"] not in breach_cells]
    dropped_ctx = {r["context_id"] for r in old_rows if r["cell"] in breach_cells}
    regen_ctx = {r["context_id"] for r in regen_rows}
    if dropped_ctx != regen_ctx:
        raise RuntimeError(
            f"anchors capregen {batch} w{w}: regenerated context set != dropped context "
            f"set (dropped-not-regen={sorted(dropped_ctx - regen_ctx)}, "
            f"regen-not-dropped={sorted(regen_ctx - dropped_ctx)})"
        )
    for r in keep_rows:
        r.setdefault("max_new_tokens", base_cap)
    merged = keep_rows + list(regen_rows)
    old = torch.load(va_path, weights_only=False)  # self-produced bundle (#1900)
    assert list(old["layers"]) == list(cfg.layers), (old["layers"], cfg.layers)
    idx = old["index"]
    va = old["va_span"]
    assert va.shape[0] == len(idx), (va.shape, len(idx))
    assert regen_states["pooling"]["va_span"] == old["pooling"]["va_span"], (
        regen_states["pooling"],
        old["pooling"],
    )
    keep_pos = [i for i, e in enumerate(idx) if cell_of[e["context_id"]] not in breach_cells]
    pos_map = {p: j for j, p in enumerate(keep_pos)}
    new_index = [idx[p] for p in keep_pos] + [
        {"context_id": r["context_id"], "draw": r["draw"]} for r in regen_rows
    ]
    new_va = torch.cat(
        [va[torch.tensor(keep_pos, dtype=torch.long)], regen_states["va_span"]], dim=0
    )
    new_empties = sorted(
        [pos_map[p] for p in old.get("empty_rows", []) if p in pos_map]
        + [len(keep_pos) + int(i) for i in regen_states["empty_rows"]]
    )
    if not (len(merged) == len(new_index) == new_va.shape[0]):
        raise RuntimeError(
            f"anchors capregen {batch} w{w}: merged jsonl/index/tensor row counts "
            f"diverge ({len(merged)}/{len(new_index)}/{new_va.shape[0]})"
        )
    _write_jsonl_atomic(jsonl, merged)
    _save_pt_atomic(
        va_path,
        {
            "layers": old["layers"],
            "index": new_index,
            "va_span": new_va,
            "pooling": old["pooling"],
            "empty_rows": new_empties,
            "repro": _repro(cfg),
        },
    )
    new_done = {
        **done_rec,
        "n_rows": len(merged),
        "n_cap_hit": sum(1 for r in merged if r["cap_hit"]),
        "n_empty": len(new_empties),
        "max_new_tokens": base_cap,  # the shard's BASE regime cap; raised cap in capregen
        "capregen": capregen_record,
        "repro": _repro(cfg),
    }
    _write_json_atomic(cfg.manifest_dir / f"anchors_{batch}_w{w}_done.json", new_done)
    return new_done


def _capregen_owning_record(cfg: RunConfig, cell: str, batch: str) -> tuple[str, Path, dict]:
    """(batch_id, done_path, record) of the COMPLETED store owning this cell's
    ``batch`` contexts — engine-aware (B3/B11 r1 review): a rest cell lives in
    exactly ONE of ``rest_{cell}`` (HF bulk), ``parity_{cell}`` (parity-leg
    HF), or ``vllm_{cell}`` (vLLM production leg); a gate cell always in
    ``gate_{cell}``. Zero owners = phase incomplete (raise); multiple owners =
    an inconsistent store (raise) — never a guess."""
    candidates = (
        (f"gate_{cell}",) if batch == "gate" else (f"rest_{cell}", f"parity_{cell}", f"vllm_{cell}")
    )
    hits: list[tuple[str, Path, dict]] = []
    for bid in candidates:
        for m in sorted(cfg.manifest_dir.glob(f"anchors_{bid}_w*_done.json")):
            if m.name.endswith("_gen_done.json"):
                continue  # vLLM pre-capture sentinel, not a done record
            hits.append((bid, m, json.loads(m.read_text())))
    if not hits:
        raise RuntimeError(
            f"capregen(anchors:{batch}): no done record for cell {cell!r} under any of "
            f"{candidates} — the anchors phase is incomplete; capregen amends a "
            "COMPLETED store only"
        )
    if len(hits) > 1:
        raise RuntimeError(
            f"capregen(anchors:{batch}): cell {cell!r} has {len(hits)} owning done "
            f"records ({[h[1].name for h in hits]}) — inconsistent store; quarantine "
            "the stale one before re-running"
        )
    return hits[0]


def phase_capregen_anchors(cfg: RunConfig) -> int:
    """Cell-restricted anchors re-gen at a raised cap (registered >2% remedy).

    CELL-GRAIN via the claim queue (B11 r1 review — the strided re-derivation
    that mis-aligned against the generation-time vLLM exclusion is GONE): one
    claim unit per breaching cell, its regen set = that cell's OWN batch
    contexts, merged wholesale into the cell's OWN shard + va store at the
    shard's recorded worker index, so every downstream consumer reads the
    same files it always did. ENGINE-AWARE (B3): a vLLM- or parity-owned
    breach cell is regenerated HF-side (the parity PASS that engaged those
    engines certifies HF/vLLM equivalence; every regen row carries
    ``engine="hf"`` and the capregen record names the superseded engine).
    The merged done record keeps the shard's BASE regime_fp (post-regen
    re-entries of the standard anchors command skip cleanly) and gains a
    ``capregen`` sub-record; the #722 r3 cross-regime hard refusal is
    preserved via the per-shard base-fp check (reconstructed from the
    record's OWN gen_batch/share_prefill_armed — B9)."""
    if cfg.capregen_batch not in ("gate", "rest"):
        raise RuntimeError(
            "--capregen-batch gate|rest is required for --capregen-scope anchors: the "
            "gate-slice leg (gate-3 critical path, ~gate rows only) and the rest-batch "
            "leg (deferrable — final F_beh reads, not gate 3) are SEPARATE invocations, "
            "never collapsed into one"
        )
    logger.info(
        "[phase=capregen] scope=anchors batch=%s worker=%d/%d smoke=%s",
        cfg.capregen_batch,
        cfg.worker_index,
        cfg.num_workers,
        cfg.smoke,
    )
    cfg = _adopt_pilot_gen_batch(cfg)  # regen at the pilot-selected B (item 3)
    cfg = _resolve_share_prefill(cfg, "capregen_anchors")  # honest regen_fp (B9)
    rep, rep_path = _load_breach_report(cfg, "anchors")
    breach = set(rep["breaching_cells"])
    if not breach:
        logger.info(
            "[capregen:anchors] breach list EMPTY (trigger_fired=false) — nothing to "
            "re-generate; exiting rc=0"
        )
        return RC_OK
    base_cap = int(rep["max_new_tokens"])
    _manifest, bank_sha = bank_manifest_and_sha()
    regen_fp = regime_fingerprint(cfg, bank_sha)
    batch = cfg.capregen_batch
    gate_ids, rest_ids, contexts = _anchor_context_order(cfg)
    by_cell = _group_by_cell(gate_ids if batch == "gate" else rest_ids, contexts)
    cell_of = {cid: ctx["cell"] for cid, ctx in contexts.items()}
    units = [c for c in sorted(breach) if by_cell.get(c)]
    for c in sorted(breach - set(units)):
        logger.info("[capregen:anchors:%s] breach cell %s has no %s contexts — N/A", batch, c, c)
    model_tok: list = [None, None, None]  # model, tok, eot

    def _base_fp_for(rec: dict) -> str:
        """The shard's OWN base fingerprint (B9: per-shard generation regime)."""
        return regime_fingerprint(
            replace(
                cfg,
                max_new_tokens=base_cap,
                gen_batch=int(rec.get("gen_batch", cfg.gen_batch)),
                share_prefill_armed=bool(rec.get("share_prefill_armed", False)),
            ),
            bank_sha,
        )

    def _unit_done(cell: str) -> bool:
        try:
            _bid, _mp, rec = _capregen_owning_record(cfg, cell, batch)
        except RuntimeError:
            return False
        cr = rec.get("capregen")
        return (
            cr is not None
            and int(cr.get("max_new_tokens", -1)) == cfg.max_new_tokens
            and set(cr.get("cells", [])) == breach
        )

    def _run_unit(block: AnchorCellBlock) -> None:
        cell = block.cell
        batch_id, done_path, done_rec = _capregen_owning_record(cfg, cell, batch)
        base_fp = _base_fp_for(done_rec)
        if done_rec.get("regime_fp") != base_fp:
            raise RuntimeError(
                f"anchors capregen {batch_id}: done-record carries "
                f"regime_fp={done_rec.get('regime_fp')!r} but the shard's reconstructed "
                f"BASE regime_fp={base_fp!r} — refusing to re-gen across regimes "
                "(quarantine or use a fresh --out-root)"
            )
        cr = done_rec.get("capregen")
        if cr is not None:
            if (
                int(cr.get("max_new_tokens", -1)) != cfg.max_new_tokens
                or set(cr.get("cells", [])) != breach
            ):
                raise RuntimeError(
                    f"anchors {batch_id} already merged a capregen at "
                    f"cap={cr.get('max_new_tokens')} cells={sorted(cr.get('cells', []))} "
                    f"!= this invocation's cap={cfg.max_new_tokens} "
                    f"cells={sorted(breach)} — refusing to stack re-gens "
                    "(fresh --out-root to redo)"
                )
            return  # raced completion — idempotent skip
        w = int(done_rec["worker_index"])
        cap_cfg = replace(cfg, worker_index=w)
        pending_file = cfg.manifest_dir / f"capregen_pending_anchors_{batch_id}.jsonl"
        my_regen = by_cell[cell]
        capregen_record = {
            "cells": sorted(breach),
            "max_new_tokens": cfg.max_new_tokens,
            "base_max_new_tokens": base_cap,
            "regen_regime_fp": regen_fp,
            "regen_engine": "hf",
            "superseded_engine": done_rec.get("engine", "hf"),
            "source_report": rep_path.name,
            "source_report_sha256": _sha256_bytes(rep_path.read_bytes()),
            "n_rows_regen": 0,
            "ts": datetime.now(UTC).isoformat(),
        }
        draws = int(done_rec["draws"])  # match the ORIGINAL shard's draws exactly
        if model_tok[0] is None:
            model_tok[0], model_tok[1] = load_model_and_tokenizer(cfg)
            model_tok[2] = eot_tail_ids(model_tok[1])
        model, tok, eot = model_tok
        if done_rec.get("engine", "hf") != "hf":
            logger.info(
                "[capregen:anchors:%s] cell %s superseding %s-generated rows with an HF "
                "re-gen at cap=%d (parity-PASS-certified equivalence; per-row engine "
                "field is authoritative)",
                batch,
                cell,
                done_rec.get("engine"),
                cfg.max_new_tokens,
            )
        rows, flat_ctx, flat_text = _generate_anchor_rows(
            cfg, model, tok, contexts, my_regen, draws, batch_id
        )
        # Rollout text durable BEFORE the capture reduce (#779 two-write
        # pattern); side file so no shard glob / upload pattern matches it.
        _write_jsonl_atomic(pending_file, rows)
        states = capture_answer_states(cfg, model, tok, flat_ctx, flat_text, eot)
        _enrich_rows_with_capture(rows, states, cfg.max_new_tokens)
        capregen_record["n_rows_regen"] = len(rows)
        _merge_anchor_capregen(
            cap_cfg, batch_id, base_cap, breach, cell_of, rows, states, done_rec, capregen_record
        )
        logger.info(
            "[capregen:anchors:%s] cell %s merged %d regenerated rows (%d contexts x %d draws)",
            batch,
            cell,
            len(rows),
            len(my_regen),
            draws,
        )
        # Uploads sit AFTER the done-record commit point, so a crash between
        # commit and upload is retried on re-entry instead of skipped. The
        # same-filename overwrite keeps the judge-visible prefixes complete
        # AND fresh (v11 MAJOR 1): without it _resolve_anchors_dir could
        # prefer STALE truncated rows over the regenerated ones.
        pending_file.unlink(missing_ok=True)
        jsonl_name = f"anchors_{batch_id}_w{w}.jsonl"
        _upload_dir(cfg, cfg.anchors_dir, f"{HF_PREFIX}/raw_completions/anchors", [jsonl_name])
        if batch == "gate":
            _upload_dir(
                cfg, cfg.anchors_dir, f"{HF_PREFIX}/raw_completions/anchors_gate", [jsonl_name]
            )
        _upload_dir(
            cfg,
            cfg.anchors_dir,
            f"{HF_PREFIX}/analysis_tensors/anchors",
            [f"va_anchors_{batch_id}_w{w}.pt"],
        )
        pre_dir = preregen_superseded_dir(cfg, "anchors")
        if (pre_dir / jsonl_name).exists():
            # FIX 2: the preserved superseded pre-regen rows upload
            # unconditionally to their OWN prefix (never HF revision history).
            _upload_dir(
                cfg,
                pre_dir,
                f"{HF_PREFIX}/raw_completions/preregen_superseded/anchors",
                [jsonl_name],
            )

    blocks = [AnchorCellBlock(cell=c, batch=batch) for c in units]
    run_claim_queue(
        cfg,
        blocks,
        regen_fp,
        f"capregen_anchors_{batch}",
        _run_unit,
        is_done=lambda _root, b, _fp, _ns: _unit_done(b.cell),
    )
    # Post-regen measurement over the mixed-cap store (v11 C1): BASE-cap row
    # attribution + the *_postregen SIBLING path — the frozen driving basis is
    # never touched; breach units (either batch) whose merge has not landed
    # keep the report partial (a mid-fleet emit never claims final).
    pending_units: list[str] = []
    gate_cells_all = set(_group_by_cell(gate_ids, contexts))
    rest_cells_all = set(_group_by_cell(rest_ids, contexts))
    for b, cells_all in (("gate", gate_cells_all), ("rest", rest_cells_all)):
        for c in sorted(breach & cells_all):
            try:
                _bid, _mp, rec = _capregen_owning_record(cfg, c, b)
            except RuntimeError:
                pending_units.append(f"anchors_{b}:{c} (no owning done record)")
                continue
            cr = rec.get("capregen")
            if (
                cr is None
                or int(cr.get("max_new_tokens", -1)) != cfg.max_new_tokens
                or set(cr.get("cells", [])) != breach
            ):
                pending_units.append(f"anchors_{_bid}")
    emit_cap_hit_report(
        cfg, "anchors", postregen=True, base_cap=base_cap, capregen_pending=pending_units
    )
    logger.info("[phase=capregen_done] scope=anchors worker=%d", cfg.worker_index)
    return RC_OK


def phase_margin(cfg: RunConfig) -> int:
    """Margin TF: (a) anchor margins (contexts sharded across workers) and
    (b) the per-block catch-up for blocks whose grid pass ran before the pools
    file landed (claim-queue namespace ``margin``)."""
    logger.info(
        "[phase=margin] worker=%d/%d smoke=%s", cfg.worker_index, cfg.num_workers, cfg.smoke
    )
    # r3 C1: margin sweeps ONLY the margin family — the deferred 1x H100 leg
    # runs this phase at width 1 over an out-root holding a VALID 8-wide
    # anchor family; a cross-family sweep would destroy 7/8 of it.
    _entry_sweep(cfg, "margin")
    assert cfg.pools_path is not None and cfg.pools_path.exists(), (
        f"--pools file required for --phase margin (got {cfg.pools_path}) — the pools are "
        "judge-built from the gate-3 slice and staged by the orchestrator"
    )
    pools = load_pools(cfg.pools_path)
    bank = _load_bank(cfg)
    manifest = _load_frozen_manifest(cfg)
    pairs = surviving_pairs(manifest)
    np_ids = no_prefix_ids(manifest)
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_maps = bank["donor_assignments"]
    regime_fp = regime_fingerprint(cfg, str(bank.get("bank_sha")))
    draws = SMOKE_GRID_DRAWS if cfg.smoke else cfg.grid_draws
    model, tok = load_model_and_tokenizer(cfg)
    contexts = BANK.build_contexts()
    ctx_ids_cache: dict[str, list[int]] = {}

    def ids_for(cid: str) -> list[int]:
        if cid not in ctx_ids_cache:
            ctx_ids_cache[cid] = BANK29.context_token_ids_2389(tok, contexts[cid])
        return ctx_ids_cache[cid]

    # (a) anchor margins — unhooked TF of each context's relevant pools.
    done_path = cfg.manifest_dir / f"margin_anchors_w{cfg.worker_index}_done.json"
    if (
        cfg.force
        or _sharded_done_record(cfg, f"margin_anchors_w{cfg.worker_index}", regime_fp) is None
    ):
        pairs_by_ctx: dict[str, list[BANK.Pair2162]] = {}
        for p in pairs:
            pairs_by_ctx.setdefault(p.a, []).append(p)
            pairs_by_ctx.setdefault(p.b, []).append(p)
        gate_ids, rest_ids, _ = _anchor_context_order(cfg)
        order = (gate_ids + rest_ids)[cfg.worker_index :: cfg.num_workers]
        rows_spec: list[dict] = []
        meta: list[dict] = []
        out_rows: list[dict] = []
        seen_keys: set[tuple[str, str]] = set()
        for cid in order:
            for p in pairs_by_ctx.get(cid, []):
                key = pool_key(p)
                if (cid, key) in seen_keys:
                    continue  # conflict fwd/rev share contexts + pools
                seen_keys.add((cid, key))
                items = pools.get(key)
                if not items:
                    out_rows.append(
                        {
                            "context_id": cid,
                            "pool_key": key,
                            "skipped": True,
                            "reason": "no pool for this value-pair",
                        }
                    )
                    continue
                for idx, it in enumerate(items):
                    item_ids = tok(it["text"], add_special_tokens=False)["input_ids"]
                    assert item_ids, (key, idx, "pool item tokenized empty")
                    rows_spec.append(
                        {
                            "ctx_ids": ids_for(cid),
                            "item_ids": item_ids,
                            "payload": None,
                            "position": None,
                        }
                    )
                    meta.append(
                        {
                            "context_id": cid,
                            "cell": contexts[cid]["cell"],
                            "value_id": contexts[cid]["value_id"],
                            "carrier": contexts[cid]["carrier"],
                            "pool_key": key,
                            "pool_idx": idx,
                            "pool_side": it["side"],
                            "n_pool_tokens": len(item_ids),
                        }
                    )
        if rows_spec:
            t0 = time.monotonic()
            lnps = margin_lnp(cfg, model, tok, rows_spec)
            logger.info("[margin:anchors] %d rows in %.1fs", len(rows_spec), time.monotonic() - t0)
            out_rows.extend(
                {**m, "lnp_mean": lnp, "skipped": False} for m, lnp in zip(meta, lnps, strict=True)
            )
        _write_jsonl_atomic(cfg.margin_dir / f"anchor_margin_w{cfg.worker_index}.jsonl", out_rows)
        _write_json_atomic(
            done_path,
            {
                "regime_fp": regime_fp,
                "worker_index": cfg.worker_index,
                "num_workers": cfg.num_workers,  # shard identity (r1 M2)
                "n_rows": len(out_rows),
                "repro": _repro(cfg),
            },
        )
    else:
        logger.info("[margin:anchors] already done for this regime — skipping")

    # (b) per-block catch-up via the claim queue (skip blocks already
    # margin-done — inline grid margins wrote the same done files). Apply
    # the SAME pe exclusions as phase_grid so the block sets match exactly.
    raw_blocks = smoke_blocks(pairs) if cfg.smoke else enumerate_blocks(pairs)
    blocks, _pe_excl = apply_pe_exclusions(raw_blocks, np_ids, donor_maps, pairs)
    _ = draws  # margin cost is pool-item-count-driven, not draw-driven

    def run_one(block: Block) -> None:
        cells = _block_cells(bank, block, pairs_by_id, donor_maps)
        for c in cells:
            ids_for(c["context_a"])
        t0 = time.monotonic()
        margin_rows = _block_margin_rows(cfg, model, tok, block, cells, pools, ctx_ids_cache)
        _write_jsonl_atomic(cfg.margin_dir / f"shard_{block.slug}.jsonl", margin_rows)
        _write_json_atomic(
            block_done_path(cfg.out_root, block, "margin_blocks"),
            {
                "key": block.key,
                "regime_fp": regime_fp,
                "n_rows": len(margin_rows),
                "n_skipped": sum(1 for r in margin_rows if r.get("skipped")),
                "repro": _repro(cfg),
            },
        )
        logger.info(
            "[margin] unit %s rows=%d elapsed=%.1fs",
            block.key,
            len(margin_rows),
            time.monotonic() - t0,
        )

    # Namespace MUST match run_one's done-file namespace ("margin_blocks") —
    # a mismatched queue namespace never sees the done files and re-runs the
    # blocks forever (caught by the tiny-real cross-phase smoke).
    stats = run_claim_queue(cfg, blocks, regime_fp, "margin_blocks", run_one)
    logger.info("[phase=margin_done] worker=%d blocks_run=%d", cfg.worker_index, stats["ran"])
    return RC_OK


# ── P5: upload + sentinel ─────────────────────────────────────────────

# Bounded OUTER retry around the hub helper's fail-soft "" return
# (upload-policy rule (c), the #1315 `_upload_with_transport_retry` shape).
UPLOAD_TRANSPORT_RETRIES = 3
UPLOAD_BACKOFF_BASE_S: tuple[float, ...] = (30.0, 60.0, 120.0)
_upload_retry_sleep = time.sleep  # monkeypatchable in tests


def upload_dir_hf(
    local_dir: Path,
    remote_prefix: str,
    allow_patterns: list[str],
) -> list[str]:
    """ONE bulk ``upload_folder`` commit for a matched subset + exact-set verify.

    FAIL-LOUD: ``_upload_folder_filtered`` returns ``""`` on failure — this
    seam retries the no-path return with bounded jittered backoff (uploads are
    idempotent), then RAISES on exhaustion so the results sentinel can never
    post over silently-lost durability (the #841 result-persist class).
    Module-level (no RunConfig) so the analysis driver reuses the exact same
    fail-loud posture for its probe-perm-matrix upload.
    """
    files = sorted(p for pat in allow_patterns for p in local_dir.glob(pat) if p.is_file())
    if not files:
        logger.info("[upload] no files match %s under %s — skipping", allow_patterns, local_dir)
        return []
    expected = [f"{remote_prefix}/{p.relative_to(local_dir).as_posix()}" for p in files]
    from explore_persona_space.orchestrate.hub import _upload_folder_filtered

    base_url = ""
    for attempt in range(UPLOAD_TRANSPORT_RETRIES + 1):
        base_url = _upload_folder_filtered(
            local_dir=local_dir,
            repo_id=HF_DATA_WRITE_REPO,
            repo_type="dataset",
            path_in_repo=remote_prefix,
            allow_patterns=allow_patterns,
            expected_repo_paths=expected,
        )
        if base_url:
            break
        if attempt < UPLOAD_TRANSPORT_RETRIES:
            pause = UPLOAD_BACKOFF_BASE_S[min(attempt, len(UPLOAD_BACKOFF_BASE_S) - 1)]
            pause *= 1.0 + 0.25 * random.random()
            logger.warning(
                "[upload] no path returned for %s (attempt %d/%d) — retrying in %.0fs",
                remote_prefix,
                attempt + 1,
                UPLOAD_TRANSPORT_RETRIES + 1,
                pause,
            )
            _upload_retry_sleep(pause)
    if not base_url:
        raise RuntimeError(
            f"upload returned no path after {UPLOAD_TRANSPORT_RETRIES + 1} attempts: "
            f"{remote_prefix} — refusing to proceed (P5 uploads feed the results "
            "sentinel; never warn-and-continue on a result-persist path)"
        )
    logger.info("[upload] %d files -> %s (one commit)", len(files), remote_prefix)
    return expected


def _upload_dir(
    cfg: RunConfig,
    local_dir: Path,
    remote_prefix: str,
    allow_patterns: list[str],
) -> list[str]:
    """``upload_dir_hf`` behind the driver's ``--upload`` mode gate."""
    if cfg.upload_mode == "none":
        logger.info("[upload] skipped (--upload none): %s", local_dir)
        return []
    if not local_dir.exists():
        logger.info("[upload] nothing staged under %s — skipping", local_dir)
        return []
    if cfg.upload_mode == "local-mirror":
        files = sorted(p for pat in allow_patterns for p in local_dir.glob(pat) if p.is_file())
        if not files:
            logger.info("[upload] no files match %s under %s — skipping", allow_patterns, local_dir)
            return []
        expected = [f"{remote_prefix}/{p.relative_to(local_dir).as_posix()}" for p in files]
        for p, rel in zip(files, expected, strict=True):
            dest = cfg.out_root / "hf_mirror" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        logger.info("[upload] mirrored %d files -> %s", len(files), remote_prefix)
        return expected
    return upload_dir_hf(local_dir, remote_prefix, allow_patterns)


def _upload_grid_increment(cfg: RunConfig, blocks: list[Block]) -> list[str]:
    """Incremental per-block-batch text upload (plan §9 phase-order persistence).

    Only THIS worker's completed shards are matched, so N concurrent workers
    never contend for the same file set, and the commit count stays far under
    the 256/hr cap.
    """
    slugs = [b.slug for b in blocks if (cfg.rollouts_dir / f"shard_{b.slug}.jsonl").exists()]
    out: list[str] = []
    if slugs:
        out += _upload_dir(
            cfg,
            cfg.rollouts_dir,
            f"{HF_PREFIX}/raw_completions/grid",
            [f"shard_{s}.jsonl" for s in slugs],
        )
    # FIX 2: preserved superseded pre-regen shards (capregen only — the dir
    # does not exist during the base grid phase) ride the same incremental
    # cadence to their OWN prefix.
    pre_dir = preregen_superseded_dir(cfg, "grid")
    pre = [f"shard_{b.slug}.jsonl" for b in blocks if (pre_dir / f"shard_{b.slug}.jsonl").exists()]
    if pre:
        out += _upload_dir(
            cfg, pre_dir, f"{HF_PREFIX}/raw_completions/preregen_superseded/grid", pre
        )
    return out


MARGIN_DEFERRED_RECIPE = (
    "stage the judge-built pools (reuse the parent's banked pools from "
    "issue2162_q7rerun/pools/pools.json, or rebuild VM-side via scripts/issue2162_judge.py "
    "--phase pools; scp pools.json to the pod), then on a fresh 1x H100: "
    "scripts/issue2389_dispatch.sh margin && scripts/issue2389_dispatch.sh upload "
    "(needs bank.json + pools.json + the model; ~3.7 GPU-h)"
)


def _margin_state(cfg: RunConfig) -> dict:
    """Disk-derived margin completeness for the results sentinel (r2 MAJOR 1).

    ``margin_deferred`` is the LOAD-BEARING downstream flag: True means the
    TF-margin secondary DV is NOT complete in this upload and is DEFERRED with
    a named recipe — the upload-verifier and report pipeline must be able to
    tell "deferred, recipe attached" apart from "silently missing". Derived
    from DISK state (per-block margin done-files + the sharded anchor-margin
    done records), never from the dispatcher's branch, so a standalone
    ``upload`` after a crash AND the later deferred-leg re-run (margin +
    upload on a 1x H100) both report the truth.
    """
    bank_json = cfg.bank_dir / "bank.json"
    if bank_json.exists():
        manifest = _load_frozen_manifest(cfg)
        pairs = surviving_pairs(manifest)
        raw = smoke_blocks(pairs) if cfg.smoke else enumerate_blocks(pairs)
        blocks, _pe = apply_pe_exclusions(
            raw, no_prefix_ids(manifest), manifest["donor_assignment"], pairs
        )
    else:
        # Pre-bank crash: no frozen manifest — report the unfiltered upper
        # bound (margin_deferred is True regardless on that path).
        pairs = BANK.build_pairs()
        blocks = smoke_blocks(pairs) if cfg.smoke else enumerate_blocks(pairs)
    blocks_done = sum(
        1 for b in blocks if block_done_path(cfg.out_root, b, "margin_blocks").exists()
    )
    # Anchor-margin completeness: done records are sharded per worker with the
    # width as shard identity (r1 M2) — complete iff SOME width W has all of
    # worker indexes 0..W-1 recorded at num_workers == W.
    recs: list[dict] = []
    for p in sorted(cfg.manifest_dir.glob("margin_anchors_w*_done.json")):
        try:
            recs.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
    anchors_done = False
    for w in {int(r.get("num_workers", 0)) for r in recs}:
        idxs = {int(r.get("worker_index", -1)) for r in recs if int(r.get("num_workers", 0)) == w}
        if w > 0 and idxs >= set(range(w)):
            anchors_done = True
    deferred = blocks_done < len(blocks) or not anchors_done
    state: dict = {
        "margin_deferred": deferred,
        "margin_blocks_done": blocks_done,
        "margin_blocks_expected": len(blocks),
        "margin_anchors_done": anchors_done,
    }
    if deferred:
        state["margin_deferred_recipe"] = MARGIN_DEFERRED_RECIPE
    return state


def _sentinel_payload(cfg: RunConfig, uploaded: dict[str, list[str]]) -> dict:
    """The /issue Step 7 results payload (all 10 keys)."""
    n_grid_shards = len(list(cfg.rollouts_dir.glob("shard_*.jsonl")))
    n_va_shards = len(list(cfg.va_dir.glob("shard_*.pt")))
    n_stage2_shards = len(list(cfg.rollouts_dir.glob("stage2_shard_*.jsonl")))
    n_stage2_va_shards = len(list(cfg.va_dir.glob("stage2_shard_*.pt")))
    n_margin_shards = len(list(cfg.margin_dir.glob("*.jsonl")))
    fact_cells_path = cfg.fact_dir / "f_cells_actonly.jsonl"
    n_fact_pair_rows = (
        sum(1 for line in fact_cells_path.open(encoding="utf-8") if line.strip())
        if fact_cells_path.exists()
        else 0
    )
    best_path = cfg.out_root / "best_cells_actsel.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    pe_path = cfg.manifest_dir / "pe_exclusions.json"
    pe = json.loads(pe_path.read_text()) if pe_path.exists() else {}
    n_anchor_rows = 0
    for jsonl in sorted(cfg.anchors_dir.glob("anchors_*.jsonl")):
        n_anchor_rows += sum(1 for line in jsonl.open(encoding="utf-8") if line.strip())
    gate_path = cfg.gates_dir / "injection_gate_report.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    degen_path = cfg.gates_dir / "degeneracy_guard_report.json"
    degen = json.loads(degen_path.read_text()) if degen_path.exists() else {}
    block_done_recs = sorted((cfg.manifest_dir / "blocks").glob("*.done.json"))
    cap_hits, rows_total = 0, 0
    for done in block_done_recs:
        rec = json.loads(done.read_text())
        cap_hits += int(rec.get("n_cap_hit", 0))
        rows_total += int(rec.get("n_rows", 0))
    margin_state = _margin_state(cfg)
    return {
        # r2 MAJOR 1: top-level, unmissable — downstream gates (upload-verifier,
        # report pipeline) key on margin_deferred to distinguish "secondary DV
        # deferred with a recipe" from "secondary DV missing".
        **margin_state,
        # r3 MINOR 2: on the fresh deferred-leg pod blocks/ is empty, so the
        # grid stats below are zeroed — this stamp tells a downstream reader
        # of the SECOND sentinel that no local grid block state contributed
        # (the grid ran on the primary pod), never "the run produced nothing".
        "deferred_leg": not block_done_recs,
        "eval_numbers": {
            "grid_shards": n_grid_shards,
            "va_shards": n_va_shards,
            "stage2_shards": n_stage2_shards,
            "stage2_va_shards": n_stage2_va_shards,
            "fact_pair_rows": n_fact_pair_rows,
            "fact_select_survivors": (len(best["cells"]) if best else None),
            "pe_excluded_pair_cells": int(pe.get("n_excluded_pair_cells", 0)),
            "pe_empty_blocks": int(pe.get("n_empty_blocks", 0)),
            "margin_shards": n_margin_shards,
            "anchor_rows": n_anchor_rows,
            "grid_rollouts_persisted": rows_total,
            "cap_hit_rows": cap_hits,
            "cap_hit_frac": (cap_hits / rows_total) if rows_total else 0.0,
            "injection_gate_passed": bool(gate.get("passed")),
            "injection_gate_spots_failed": int(gate.get("n_spots_failed", 0)),
            "degeneracy_gate_passed": bool(degen.get("passed")),
            "degeneracy_violations": int(degen.get("n_violations", 0)),
        },
        "eval_paths": sorted(
            {
                str(cfg.bank_dir / "bank.json"),
                str(cfg.bank_dir / "vc_bank.pt"),
                str(gate_path),
                str(degen_path),
                str(cfg.anchors_dir),
                str(cfg.rollouts_dir),
                str(cfg.va_dir),
                str(cfg.margin_dir),
                str(cfg.fact_dir),
                str(best_path),
                str(cfg.gates_dir / "pilot_gate_report.json"),
                str(cap_hit_report_path(cfg, "anchors")),
                str(cap_hit_report_path(cfg, "grid")),
            }
        ),
        "reproducibility_card": {
            **_repro(cfg),
            "seed_base": cfg.seed_base,
            "bank_seed": BANK.SEED,
            "max_new_tokens": cfg.max_new_tokens,
            "grid_temperature": GRID_TEMPERATURE,
            "grid_draws": cfg.grid_draws,
            "anchor_temperature": ANCHOR_TEMPERATURE,
            "anchor_draws": cfg.anchor_draws,
            "gen_batch": cfg.gen_batch,
            "num_workers": cfg.num_workers,
            "template_kwargs": {"enable_thinking": False},
            "f_act_read_layer": F_ACT_READ_LAYER,
            "stage2": {
                "layers": list(STAGE2_LAYERS),
                "doses": list(STAGE2_DOSES),
                "arms": list(STAGE2_ARMS),
                "draws": STAGE2_DRAWS,
                "temperature": STAGE2_TEMPERATURE,
            },
            "fact_select": {
                "boot_B": FACT_BOOT_B,
                "boot_seed": FACT_BOOT_SEED,
                "holm_alpha": FACT_HOLM_ALPHA,
                "survival_floor": FACT_SURVIVAL_FLOOR,
                "cap": FACT_SELECT_CAP,
            },
        },
        "wandb_url": None,
        "hf_hub_url": (
            f"https://huggingface.co/datasets/{HF_DATA_WRITE_REPO}/tree/main/{HF_PREFIX}"
        ),
        "worktree_path": str(REPO_ROOT),
        "final_commit_sha": _git_sha(),
        "gpu_hours_used": None,
        "gpu_hours_budgeted": cfg.gpu_hours_budgeted,
        "plan_deviations": [
            "the per-block margin TF runs as a SEPARATE batched hooked pass beside the "
            "V_a pass (rollout rows need hidden states, pool rows need logits) — same "
            "per-block pipelining on the same GPU as the plan's 'one hooked TF pass', "
            "recorded here because the plan's phrasing implies one fused forward",
            "cap-hit telemetry is derived from the re-tokenized completion length "
            "(generate_batch returns decoded text only); recorded per row as "
            "cap_hit_basis=retokenized_completion_len >= max_new_tokens",
            "the V_a store carries the span-mean pooling only (the plan §4.4 metric); "
            "the parent's tail-inclusive twin pooling is not persisted",
        ],
        "uploaded_prefixes": {k: len(v) for k, v in uploaded.items()},
    }


def phase_upload(cfg: RunConfig) -> int:
    """P5: bulk upload every prefix, then write the pod sentinel."""
    logger.info("[phase=upload]")
    # r2 F3 / r3 C1 defense-in-depth: each family is swept at ITS OWN realized
    # width, derived from its OWN done-record set — never this CPU process's
    # --num-workers (r2 C1: the dispatcher's width-less upload leg ran at
    # implicit width 1 and quarantined every w1..wN live shard before the
    # bulk upload; the exact-set verify then passed on the truncated set).
    _upload_entry_sweeps(cfg)
    uploaded: dict[str, list[str]] = {}
    # vc_bank.pt by exact name: the per-chunk vc_bank_part_*.pt files are
    # merge inputs (byte-duplicated into vc_bank.pt) — never uploaded.
    uploaded["vc_bank"] = _upload_dir(
        cfg, cfg.bank_dir, f"{HF_PREFIX}/analysis_tensors/vc_bank", ["vc_bank.pt", "*.json"]
    )
    uploaded["anchors_text"] = _upload_dir(
        cfg, cfg.anchors_dir, f"{HF_PREFIX}/raw_completions/anchors", ["*.jsonl"]
    )
    uploaded["anchors_tensors"] = _upload_dir(
        cfg, cfg.anchors_dir, f"{HF_PREFIX}/analysis_tensors/anchors", ["*.pt"]
    )
    uploaded["grid_text"] = _upload_dir(
        cfg, cfg.rollouts_dir, f"{HF_PREFIX}/raw_completions/grid", ["shard_*.jsonl"]
    )
    uploaded["va_store"] = _upload_dir(
        cfg, cfg.va_dir, f"{HF_PREFIX}/analysis_tensors/va_store", ["shard_*.pt"]
    )
    uploaded["margin"] = _upload_dir(
        cfg, cfg.margin_dir, f"{HF_PREFIX}/analysis_tensors/margin", ["*.jsonl"]
    )
    uploaded["stage2_text"] = _upload_dir(
        cfg, cfg.rollouts_dir, f"{HF_PREFIX}/raw_completions/stage2", ["stage2_shard_*.jsonl"]
    )
    uploaded["va_store_stage2"] = _upload_dir(
        cfg, cfg.va_dir, f"{HF_PREFIX}/analysis_tensors/va_store_stage2", ["stage2_shard_*.pt"]
    )
    uploaded["fact_metrics"] = _upload_dir(
        cfg,
        cfg.fact_dir,
        f"{HF_PREFIX}/analysis_tensors/f_metrics_actonly",
        ["*.jsonl", "*.json", "cells/*.jsonl", "cells/*.done.json"],
    )
    # Gate artifacts (plan §6.5 dv row): pilot / injection / degeneracy /
    # cap-recalibration / canary / parity / share_prefill reports.
    uploaded["gates"] = _upload_dir(
        cfg, cfg.gates_dir, f"{HF_PREFIX}/analysis_tensors/gates", ["*.json"]
    )
    # Out-root TOP-LEVEL residue (#2187): best_cells_actsel.json lives at the
    # out-root top — persist it so the residue sweep reads clean. Exact names,
    # not "*.json": hub allow_patterns are fnmatch (a bare * crosses "/" and
    # would re-upload every nested json).
    uploaded["outroot_top"] = _upload_dir(
        cfg,
        cfg.out_root,
        f"{HF_PREFIX}/analysis_tensors/outroot_top",
        ["pilot_gate_report.json", "best_cells_actsel.json"],
    )
    # FIX 2 backstop row: superseded pre-regen rollout text preserved by a
    # capregen (write-once byte-copies) — in-phase uploads are primary; this
    # bulk row guarantees the class rides P5 too. Absent dir -> skipped.
    uploaded["preregen_superseded"] = _upload_dir(
        cfg,
        cfg.out_root / "preregen_superseded",
        f"{HF_PREFIX}/raw_completions/preregen_superseded",
        ["anchors/*.jsonl", "grid/*.jsonl"],
    )
    # blocks/*.done.json + claim residue ride along: per-block resume state +
    # the sentinel's cap-hit provenance become durable off-pod.
    uploaded["manifests"] = _upload_dir(
        cfg,
        cfg.manifest_dir,
        f"{HF_PREFIX}/analysis_tensors/manifests",
        [
            "*.json",
            "blocks/*.done.json",
            "margin_blocks/*.done.json",
            "stage2_blocks/*.done.json",
            "bank_capture/*.done.json",
        ],
    )
    payload = _sentinel_payload(cfg, uploaded)
    if payload["margin_deferred"]:
        logger.warning(
            "[upload] margin DEFERRED (blocks %d/%d, anchors_done=%s) — the sentinel "
            "records margin_deferred=true + the deferred-leg recipe; teardown proceeds "
            "(never idle the wide pod on the Batch-API SLA tail). Recipe: %s",
            payload["margin_blocks_done"],
            payload["margin_blocks_expected"],
            payload["margin_anchors_done"],
            MARGIN_DEFERRED_RECIPE,
        )
    _write_json_atomic(cfg.out_root / "manifests" / "upload_done.json", payload)
    sentinel = cfg.log_dir / (SENTINEL_NAME_SMOKE if cfg.smoke else SENTINEL_NAME)
    body = {
        "sentinel_schema_version": 1,
        "kind": "epm:smoke-result" if cfg.smoke else "epm:results",
        "version": 1,
        "note": payload,
    }
    _write_json_atomic(sentinel, body)
    logger.info("[upload] sentinel written: %s", sentinel)
    logger.info("[phase=upload_done]")
    return RC_OK


# ── mechanical text audits (plan divergence 6 draw filter) ────────────
# Copied VERBATIM from scripts/issue2094_judge.py (nonlatin_letter_frac /
# _word_4grams / max_repeated_4gram_frac / dup_4gram_frac / audit_text) so
# the pod driver carries no dependency on the judge script; thresholds are
# the module constants AUDIT_NONLATIN_FRAC_MAX / AUDIT_DUP_4GRAM_FRAC_MAX.


def nonlatin_letter_frac(text: str) -> float:
    """Fraction of LETTER codepoints outside Latin (cp >= U+0250) among all letters."""
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) >= _LATIN_MAX_CP) / len(letters)


def _word_4grams(text: str) -> Counter:
    words = text.split()
    if len(words) < 4:
        return Counter()
    return Counter(tuple(words[i : i + 4]) for i in range(len(words) - 3))


def max_repeated_4gram_frac(text: str) -> float:
    """Most-frequent word-4gram count / total 4grams (0.0 below 4 words)."""
    grams = _word_4grams(text)
    return max(grams.values()) / sum(grams.values()) if grams else 0.0


def dup_4gram_frac(text: str) -> float:
    """1 - distinct/total word-4grams — the repetition FLAG basis.

    A k-word repetition loop splits its mass across k rotated 4-grams, capping
    ``max_repeated_4gram_frac`` at ~1/k (a 4-word loop reads only ~0.25), while
    the duplicate fraction reads ~1.0 there and ~0 on ordinary prose — so the
    flag keys on THIS statistic; the max-single-4-gram fraction is kept as a
    reported companion field.
    """
    grams = _word_4grams(text)
    if not grams:
        return 0.0
    total = sum(grams.values())
    return 1.0 - len(grams) / total


def audit_text(text: str) -> dict:
    """Per-rollout mechanical audit fields (flags only; selection owns exclusion)."""
    empty = len(text.strip()) == 0
    nl = nonlatin_letter_frac(text)
    rep_max = max_repeated_4gram_frac(text)
    rep_dup = dup_4gram_frac(text)
    return {
        "n_chars": len(text),
        "n_words": len(text.split()),
        "empty": empty,
        "nonlatin_letter_frac": round(nl, 6),
        "max_repeated_4gram_frac": round(rep_max, 6),
        "dup_4gram_frac": round(rep_dup, 6),
        "flag_empty": empty,
        "flag_script_intrusion": nl > AUDIT_NONLATIN_FRAC_MAX,
        "flag_repetition": rep_dup > AUDIT_DUP_4GRAM_FRAC_MAX,
    }


def draw_survives_audit(audit: dict) -> bool:
    """Divergence 6: a draw enters F_act tables iff NO mechanical flag fired."""
    return not (audit["flag_empty"] or audit["flag_script_intrusion"] or audit["flag_repetition"])


# ── P4a: fact_tables — per-draw judge-free F_act (plan divergence 6) ──


def fact_read_layer(cfg: RunConfig) -> int:
    """F_ACT_READ_LAYER (30/32, divergence 4); last layer under ``--tiny``."""
    return F_ACT_READ_LAYER if F_ACT_READ_LAYER < cfg.n_layers else cfg.n_layers - 1


def _fact_regime_fp(cfg: RunConfig, bank_sha: str) -> str:
    return regime_fingerprint(cfg, bank_sha) + "-fact-v1"


def _anchor_floor_ceiling(cfg: RunConfig) -> tuple[dict[str, torch.Tensor], dict]:
    """Audit-surviving anchor V_a draws per context at the read layer.

    Returns ``({context_id: (K_kept, hidden) float32}, audit_summary)``. The
    floor/ceiling table for F_act: floor = anchors under context A, ceiling =
    anchors under context B (plan §4.4). Capture-empty rows and rows failing
    any mechanical audit flag are dropped HERE, so every consumer sees the
    identical draw filter.
    """
    li = cfg.layers.index(fact_read_layer(cfg))
    per_ctx: dict[str, list[torch.Tensor]] = {}
    n_total = 0
    n_kept = 0
    flag_counts: Counter = Counter()
    pt_paths = sorted(cfg.anchors_dir.glob("va_anchors_*_w*.pt"))
    assert pt_paths, f"no anchor V_a shards under {cfg.anchors_dir} — run --phase anchors first"
    for pt_path in pt_paths:
        jsonl = cfg.anchors_dir / (pt_path.name.removeprefix("va_").removesuffix(".pt") + ".jsonl")
        assert jsonl.exists(), f"{jsonl} missing beside {pt_path}"
        rows = [json.loads(line) for line in jsonl.open(encoding="utf-8") if line.strip()]
        store = torch.load(pt_path, weights_only=False)  # self-produced bundle (#1900)
        index = store["index"]
        assert len(index) == len(rows), (str(pt_path), len(index), len(rows))
        va = store["va_span"]
        empty = set(store["empty_rows"])
        for i, (row, ref) in enumerate(zip(rows, index, strict=True)):
            assert row["context_id"] == ref["context_id"] and row["draw"] == ref["draw"], (
                str(pt_path),
                i,
            )
            n_total += 1
            a = audit_text(row["text"])
            for fkey in ("flag_empty", "flag_script_intrusion", "flag_repetition"):
                flag_counts[fkey] += int(a[fkey])
            if i in empty or not draw_survives_audit(a):
                continue
            n_kept += 1
            per_ctx.setdefault(row["context_id"], []).append(va[i, li].float())
    table = {cid: torch.stack(vs) for cid, vs in per_ctx.items()}
    summary = {
        "read_layer": fact_read_layer(cfg),
        "n_anchor_draws_total": n_total,
        "n_anchor_draws_kept": n_kept,
        "flag_counts": dict(flag_counts),
        "thresholds": {
            "nonlatin_letter_frac": AUDIT_NONLATIN_FRAC_MAX,
            "dup_4gram_frac": AUDIT_DUP_4GRAM_FRAC_MAX,
        },
        "n_contexts_with_draws": len(table),
    }
    return table, summary


def _fact_block_records(
    cfg: RunConfig,
    block: Block,
    rows: list[dict],
    va: torch.Tensor,
    empty_rows: set[int],
    floors: dict[str, torch.Tensor],
    pairs_by_id: dict[str, BANK.Pair2162],
) -> tuple[list[dict], list[dict]]:
    """Per-draw + per-pair F_act records for ONE grid block (batched f_act).

    Rows are group-batched by their pairs' (K_floor, K_ceiling) shapes so each
    ``f_act`` call is one batched tensor op (vectorize-first; no per-draw
    loop over the metric math). NaN f_act (degenerate axis) is persisted as
    ``null`` + ``degenerate=true``, never coerced.
    """
    from explore_persona_space.experiments.issue2094.fmetrics import f_act

    li = cfg.layers.index(fact_read_layer(cfg))
    audits = [audit_text(r["text"]) for r in rows]
    by_pair: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_pair.setdefault(r["pair_id"], []).append(i)
    draw_rows: list[dict] = []
    per_pair: list[dict] = []
    groups: dict[tuple[int, int], list[tuple[str, list[int]]]] = {}

    def pair_rec(pid: str, **kw) -> dict:
        return {
            "cell": block.cell,
            "slot": block.slot,
            "arm": block.arm,
            "pair_id": pid,
            "n_draws_total": len(by_pair[pid]),
            **kw,
        }

    for pid in sorted(by_pair):
        pair = pairs_by_id[pid]
        floor = floors.get(pair.a)
        ceil = floors.get(pair.b)
        if floor is None or floor.shape[0] < 2:
            per_pair.append(pair_rec(pid, skipped=True, reason="floor_draws_lt_2"))
            continue
        if ceil is None or ceil.shape[0] < 1:
            per_pair.append(pair_rec(pid, skipped=True, reason="no_ceiling_draws"))
            continue
        surviving = [
            i for i in by_pair[pid] if i not in empty_rows and draw_survives_audit(audits[i])
        ]
        if not surviving:
            per_pair.append(pair_rec(pid, skipped=True, reason="no_surviving_draws"))
            continue
        groups.setdefault((floor.shape[0], ceil.shape[0]), []).append((pid, surviving))

    for (kf, kc), items in sorted(groups.items()):
        idx_flat = [i for _, surv in items for i in surv]
        vp = torch.stack([va[i, li].float() for i in idx_flat])
        fl = torch.stack([floors[pairs_by_id[pid].a] for pid, surv in items for _ in surv])
        ce = torch.stack([floors[pairs_by_id[pid].b] for pid, surv in items for _ in surv])
        assert fl.shape == (len(idx_flat), kf, cfg.hidden), fl.shape
        assert ce.shape == (len(idx_flat), kc, cfg.hidden), ce.shape
        res = f_act(vp, fl, ce)
        k = 0
        for pid, surv in items:
            vals: list[float] = []
            for i in surv:
                degen = bool(res.degenerate[k])
                fa = None if degen else float(res.f_act[k])
                draw_rows.append(
                    {
                        "cell": block.cell,
                        "slot": block.slot,
                        "arm": block.arm,
                        "pair_id": pid,
                        "draw": rows[i]["draw"],
                        "f_act": fa,
                        # r2 CC2: mirror f_act's None-if-degenerate treatment
                        # (+ a finiteness guard — f_act_shared is NaN whenever
                        # the FULL-mean t-axis is zero, a degeneracy the
                        # disjoint-half `degenerate` flag does not cover).
                        "f_act_shared": (
                            None if degen else _finite_or_none(float(res.f_act_shared[k]))
                        ),
                        "s_norm": None if degen else _finite_or_none(float(res.s_norm[k])),
                        "t_norm": None if degen else _finite_or_none(float(res.t_norm[k])),
                        "degenerate": degen,
                        **{
                            f: audits[i][f]
                            for f in ("flag_empty", "flag_script_intrusion", "flag_repetition")
                        },
                    }
                )
                if fa is not None:
                    vals.append(fa)
                k += 1
            if vals:
                per_pair.append(
                    pair_rec(
                        pid,
                        skipped=False,
                        n_draws_surviving=len(vals),
                        f_act_mean=sum(vals) / len(vals),
                    )
                )
            else:
                per_pair.append(pair_rec(pid, skipped=True, reason="all_draws_degenerate"))
        assert k == len(idx_flat)
    return draw_rows, per_pair


def phase_fact_tables(cfg: RunConfig) -> int:
    """P4a: per-block judge-free F_act tables from the persisted grid shards.

    Blocks are sharded deterministically across workers (pure CPU reduce over
    already-persisted artifacts — no claim queue needed); per-block outputs +
    done records under ``f_metrics_actonly/cells/`` give checkpoint/resume at
    block grain (233 blocks > the ~50-unit checkpoint trigger).
    """
    logger.info("[phase=fact_tables] worker=%d/%d", cfg.worker_index, cfg.num_workers)
    bank = _load_bank(cfg)
    manifest = _load_frozen_manifest(cfg)
    pairs = surviving_pairs(manifest)
    np_ids = no_prefix_ids(manifest)
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_maps = bank["donor_assignments"]
    regime_fp = _fact_regime_fp(cfg, str(bank.get("bank_sha")))
    raw_blocks = smoke_blocks(pairs) if cfg.smoke else enumerate_blocks(pairs)
    blocks, _pe = apply_pe_exclusions(raw_blocks, np_ids, donor_maps, pairs)
    floors, audit_summary = _anchor_floor_ceiling(cfg)
    _write_json_atomic(cfg.fact_dir / "anchor_audit_summary.json", audit_summary)
    cells_dir = cfg.fact_dir / "cells"
    mine = blocks[cfg.worker_index :: cfg.num_workers]
    t0 = time.monotonic()
    n_done = 0
    for k, block in enumerate(mine):
        done_p = cells_dir / f"{block.slug}.done.json"
        if not cfg.force and done_p.exists():
            rec = json.loads(done_p.read_text())
            if rec.get("regime_fp") == regime_fp:
                n_done += 1
                continue
        shard = cfg.rollouts_dir / f"shard_{block.slug}.jsonl"
        va_path = cfg.va_dir / f"shard_{block.slug}.pt"
        assert shard.exists() and va_path.exists(), (
            f"grid artifacts missing for block {block.key} ({shard}, {va_path}) — "
            "run --phase grid to completion first"
        )
        rows = [json.loads(line) for line in shard.open(encoding="utf-8") if line.strip()]
        store = torch.load(va_path, weights_only=False)  # self-produced bundle (#1900)
        assert len(store["index"]) == len(rows), (block.key, len(store["index"]), len(rows))
        draw_rows, per_pair = _fact_block_records(
            cfg, block, rows, store["va_span"], set(store["empty_rows"]), floors, pairs_by_id
        )
        _write_jsonl_atomic(cells_dir / f"{block.slug}.rows.jsonl", draw_rows)
        _write_jsonl_atomic(cells_dir / f"{block.slug}.pairs.jsonl", per_pair)
        _write_json_atomic(
            done_p,
            {
                "key": block.key,
                "regime_fp": regime_fp,
                "n_draw_rows": len(draw_rows),
                "n_pair_rows": len(per_pair),
                "n_pairs_skipped": sum(1 for r in per_pair if r.get("skipped")),
                "repro": _repro(cfg),
            },
        )
        n_done += 1
        logger.info(
            "[fact_tables] unit %d/%d %s elapsed=%.1fs",
            k + 1,
            len(mine),
            block.key,
            time.monotonic() - t0,
        )
    _write_json_atomic(
        cfg.manifest_dir / f"fact_tables_done_w{cfg.worker_index}.json",
        {
            "regime_fp": regime_fp,
            "worker_index": cfg.worker_index,
            "num_workers": cfg.num_workers,
            "n_blocks": n_done,
            "repro": _repro(cfg),
        },
    )
    logger.info("[phase=fact_tables_done] worker=%d blocks=%d", cfg.worker_index, n_done)
    return RC_OK


# ── P4b: fact_select — parent selection SHAPE on F_act (divergence 6) ──
# _wilcoxon_exact_p + holm copied VERBATIM from scripts/issue2162_analysis.py;
# bootstrap_family_means_batched copied VERBATIM from
# scripts/issue2094_analysis.py (batched index-GEMM — no per-draw loop).


def _wilcoxon_exact_p(diffs: np.ndarray) -> float:
    """Exact two-sided Wilcoxon signed-rank p (zero diffs dropped, ties mean-ranked).

    scipy falls back from exact to the normal approximation itself when ties
    make the exact distribution unavailable (method="auto" semantics); we
    request exact at the plan's n<=36 scale and let scipy degrade on ties.
    """
    from scipy.stats import wilcoxon

    d = diffs[np.abs(diffs) > 0]
    if len(d) < 1:
        return 1.0
    method = "exact" if (len(d) <= 50 and len(np.unique(np.abs(d))) == len(d)) else "auto"
    return float(wilcoxon(d, alternative="two-sided", method=method).pvalue)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values within one family."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[key] = running
    return adj


def bootstrap_family_means_batched(
    values: np.ndarray, n_boot: int, seed: int, *, block: int = 2000
) -> np.ndarray:
    """Pair-clustered bootstrap means for MANY families at once (batched
    index-GEMMs — the null_battery subset-sum pattern; NO per-draw loop).

    ``values``: (n_pairs, n_families), NaN = cell unavailable for that family.
    Returns (n_boot, n_families) NaN-aware resampled means: draw d resamples
    the PAIR axis with replacement, mean over the drawn pairs' non-NaN cells.
    """
    n, f = values.shape
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(values)
    v0 = np.where(mask, values, 0.0)
    out = np.empty((n_boot, f), dtype=np.float64)
    for start in range(0, n_boot, block):
        b = min(block, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        counts = np.zeros((b, n), dtype=np.float64)
        np.add.at(counts, (np.arange(b)[:, None], idx), 1.0)
        num = counts @ v0  # (b, f) — ONE GEMM per block over all families
        den = counts @ mask.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start : start + b] = np.where(den > 0, num / den, np.nan)
    return out


def phase_fact_select(cfg: RunConfig) -> int:
    """P4b: audit-filtered activation-only selection -> best_cells_actsel.json.

    Parent selection SHAPE (scripts/issue2162_analysis.py step_stats) on the
    judge-free F_act per-pair means: per (cell,slot) Holm-IUT signed-rank
    (steered vs BOTH nulls, IUT p = max) over ANALYSIS-TIME testable
    candidates, AND fully-disjoint pair-clustered bootstrap CIs (steered lo >
    both nulls' hi). ZERO survivors is a documented legitimate outcome (the
    file records it; stage2 then skips-with-record) — the fail-loud asserts
    below guard BUGS (no rows / no candidates), never the science outcome.
    """
    logger.info("[phase=fact_select]")
    bank = _load_bank(cfg)
    manifest = _load_frozen_manifest(cfg)
    pairs = surviving_pairs(manifest)
    np_ids = no_prefix_ids(manifest)
    donor_maps = bank["donor_assignments"]
    regime_fp = _fact_regime_fp(cfg, str(bank.get("bank_sha")))
    raw_blocks = smoke_blocks(pairs) if cfg.smoke else enumerate_blocks(pairs)
    blocks, _pe = apply_pe_exclusions(raw_blocks, np_ids, donor_maps, pairs)
    cells_dir = cfg.fact_dir / "cells"
    pair_rows: list[dict] = []
    for block in blocks:
        done_p = cells_dir / f"{block.slug}.done.json"
        assert done_p.exists() and json.loads(done_p.read_text()).get("regime_fp") == regime_fp, (
            f"fact_tables incomplete for block {block.key} — run --phase fact_tables first"
        )
        pair_rows.extend(
            json.loads(line)
            for line in (cells_dir / f"{block.slug}.pairs.jsonl").open(encoding="utf-8")
            if line.strip()
        )
    assert pair_rows, "fact_select: zero per-pair rows loaded — fact_tables produced nothing"
    _write_jsonl_atomic(cfg.fact_dir / "f_cells_actonly.jsonl", pair_rows)

    by_cell: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for r in pair_rows:
        if r.get("skipped"):
            continue
        by_cell.setdefault((r["cell"], r["slot"]), {}).setdefault(r["arm"], {})[r["pair_id"]] = r[
            "f_act_mean"
        ]
    assert by_cell, "fact_select: zero (cell,slot) candidates carry any surviving arm data"

    candidates: list[dict] = []
    for (cell, slot), arms in sorted(by_cell.items()):
        steered = arms.get("steered", {})
        rec: dict = {"cell": cell, "slot": slot, "n_pairs_steered": len(steered)}
        diffs: dict[str, np.ndarray] = {}
        for null in ("shuffled", "crosstype"):
            ids = sorted(set(steered) & set(arms.get(null, {})))
            diffs[null] = np.array([steered[p] - arms[null][p] for p in ids], dtype=np.float64)
            rec[f"n_paired_{null}"] = len(ids)
        rec["testable"] = all(len(d) >= FACT_SURVIVAL_FLOOR for d in diffs.values())
        if rec["testable"]:
            rec["p_iut"] = max(
                _wilcoxon_exact_p(diffs["shuffled"]), _wilcoxon_exact_p(diffs["crosstype"])
            )
        # Pair-clustered bootstrap CIs over the cell's pair axis, 3 family
        # columns (steered/shuffled/crosstype), NaN-padded per pair.
        all_ids = sorted({p for arm in arms.values() for p in arm})
        fams = ("steered", "shuffled", "crosstype")
        vals = np.full((len(all_ids), len(fams)), np.nan)
        for j, fam in enumerate(fams):
            for i, p in enumerate(all_ids):
                if p in arms.get(fam, {}):
                    vals[i, j] = arms[fam][p]
        boots = bootstrap_family_means_batched(vals, FACT_BOOT_B, FACT_BOOT_SEED)
        los = np.nanpercentile(boots, 2.5, axis=0)
        his = np.nanpercentile(boots, 97.5, axis=0)
        for j, fam in enumerate(fams):
            with np.errstate(invalid="ignore"):
                rec[f"{fam}_mean"] = (
                    float(np.nanmean(vals[:, j])) if np.any(~np.isnan(vals[:, j])) else None
                )
            rec[f"{fam}_ci"] = [float(los[j]), float(his[j])]
        rec["disjoint_both_nulls"] = bool(
            not math.isnan(los[0]) and los[0] > his[1] and los[0] > his[2]
        )
        candidates.append(rec)

    testable = {f"{r['cell']}|{r['slot']}": r["p_iut"] for r in candidates if r["testable"]}
    adj = holm(testable)
    for r in candidates:
        key = f"{r['cell']}|{r['slot']}"
        if key in adj:
            r["p_holm"] = adj[key]
            r["holm_family_m"] = len(testable)
            r["holm_pass"] = adj[key] < FACT_HOLM_ALPHA
        else:
            r["holm_pass"] = False
    survivors = sorted(
        (r for r in candidates if r.get("holm_pass") and r["disjoint_both_nulls"]),
        key=lambda r: -(r["steered_mean"] if r["steered_mean"] is not None else -math.inf),
    )[:FACT_SELECT_CAP]
    payload = {
        "cells": [
            {k: r[k] for k in ("cell", "slot", "steered_mean", "p_iut", "p_holm")}
            for r in survivors
        ],
        "selection": {
            "criterion": (
                "holm_pass (IUT p = max(wilcoxon steered-shuffled, steered-crosstype), "
                f"Holm over {len(testable)} analysis-time testable candidates, alpha "
                f"{FACT_HOLM_ALPHA}) AND fully-disjoint pair-clustered bootstrap CIs "
                "(steered lo > both nulls' hi); cap 12 by descending steered F_act mean"
            ),
            "n_candidates": len(candidates),
            "n_testable": len(testable),
            "n_survivors": len(survivors),
            "boot": {"B": FACT_BOOT_B, "seed": FACT_BOOT_SEED},
            "survival_floor": FACT_SURVIVAL_FLOOR,
            "regime_fp": regime_fp,
        },
        "repro": _repro(cfg),
    }
    _write_json_atomic(cfg.out_root / "best_cells_actsel.json", payload)
    _write_json_atomic(
        cfg.fact_dir / "fact_select_report.json", {"candidates": candidates, **payload["selection"]}
    )
    if not survivors:
        logger.warning(
            "[fact_select] ZERO survivors (%d candidates, %d testable) — a legitimate "
            "outcome; stage2 will skip-with-record",
            len(candidates),
            len(testable),
        )
    logger.info(
        "[phase=fact_select_done] survivors=%d/%d testable=%d",
        len(survivors),
        len(candidates),
        len(testable),
    )
    return RC_OK


# ── P4c: stage-2 layer x dose (folded from issue2162_stage2.py) ───────


@dataclass(frozen=True)
class Stage2Block:
    """One schedulable stage-2 unit: (cell, slot, arm, layer, dose)."""

    cell: str
    slot: str
    arm: str
    layer: int
    dose: int
    pair_ids: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.cell}|{self.slot}|{self.arm}|L{self.layer}|d{self.dose}"

    @property
    def slug(self) -> str:
        return block_slug(self.key)

    @property
    def n_pairs(self) -> int:
        return len(self.pair_ids)


def load_best_cells_actsel(path: Path) -> list[dict]:
    """fact_select survivors — fail loud on shape drift; [] = zero survivors."""
    assert path.exists(), f"{path} missing — run --phase fact_select first"
    payload = json.loads(path.read_text())
    cells = payload["cells"]
    assert len(cells) <= FACT_SELECT_CAP, (len(cells), "selection cap is 12 (plan §6)")
    for rec in cells:
        assert set(rec) >= {"cell", "slot"}, sorted(rec)
        assert rec["cell"] in BANK.all_cells(), rec["cell"]
        assert rec["slot"] in ("ce", "pe"), rec["slot"]
    return cells


def enumerate_stage2_blocks(
    best_cells: list[dict],
    pairs: list[BANK.Pair2162],
    np_ids: frozenset[str] | set[str],
    donor_maps: dict[str, dict[str, str]],
    smoke: bool,
) -> tuple[list[Stage2Block], list[dict]]:
    """Stage-2 grid with the SAME pe no-prefix exclusions as stage 1."""
    by_cell = BANK.pairs_by_cell(pairs)
    pairs_by_id = {p.pair_id: p for p in pairs}
    selected = best_cells[:1] if smoke else best_cells
    layers = STAGE2_LAYERS[:1] if smoke else STAGE2_LAYERS
    blocks: list[Stage2Block] = []
    exclusions: list[dict] = []
    for rec in selected:
        ids = tuple(p.pair_id for p in sorted(by_cell[rec["cell"]], key=lambda p: p.pair_id))
        if smoke:
            ids = ids[:SMOKE_PAIRS_PER_CELL]
        # BOTH arms in smoke too (>=1 tiny block per arm class — the per-arm
        # smoke-coverage rule; the shuffled arm exercises the donor seam).
        for arm in STAGE2_ARMS:
            runnable = ids
            if rec["slot"] == "pe":
                runnable = []
                for pid in ids:
                    reason = pe_excluded_reason(
                        pairs_by_id[pid], arm, np_ids, donor_maps, pairs_by_id
                    )
                    if reason is None:
                        runnable.append(pid)
                    else:
                        exclusions.append(
                            {
                                "cell": rec["cell"],
                                "slot": "pe",
                                "arm": arm,
                                "pair_id": pid,
                                "reason": reason,
                                "stage": 2,
                            }
                        )
                runnable = tuple(runnable)
            if not runnable:
                exclusions.append(
                    {
                        "cell": rec["cell"],
                        "slot": rec["slot"],
                        "arm": arm,
                        "pair_id": None,
                        "reason": "block_empty_all_pairs_no_prefix",
                        "stage": 2,
                    }
                )
                continue
            for layer in layers:
                for dose in STAGE2_DOSES:
                    blocks.append(
                        Stage2Block(rec["cell"], rec["slot"], arm, layer, dose, tuple(runnable))
                    )
    keys = [b.key for b in blocks]
    assert len(set(keys)) == len(keys), "duplicate stage-2 block keys"
    return blocks, exclusions


def stage2_regime_fp(cfg: RunConfig, bank_sha: str) -> str:
    return regime_fingerprint(cfg, bank_sha) + f"-stage2-add-K{STAGE2_DRAWS}"


def _arm_hook_add_single_layer(
    model,
    layer: int,
    dose: int,
    row_lengths: list[int],
    positions: list[tuple[int, ...]],
    per_row_delta: list[torch.Tensor],
    expected_prompt_len: int,
):
    """Add-mode hook at ONE layer: ``h[b, p] += dose * delta[b][layer]``
    (parent §4.2 stage-2 — the dose is the alpha multiplier on the added
    pair-difference delta, never a layer-window width).

    ``per_row_delta[b]`` is the pair-difference payload ``(1, L_all, H)``
    (delta = V(B) - V(A), donor-side per arm); the single patched layer's
    hook receives its own ``(1, H)`` slice (payload layer index == model
    layer index: cfg.layers is range(n_layers))."""
    stack = joint_hooks(model, [layer])
    per_layer = [[d[:, layer, :].contiguous() for d in per_row_delta]]
    stack.install()
    stack.arm_batch_per_layer(row_lengths, positions, per_layer, mode="add", alpha=float(dose))
    stack.arm(expected_prompt_len)
    return stack


def _stage2_hook_builder(layer: int, dose: int):
    """capture_answer_states hook_builder closing over (layer, dose)."""

    def build(model, cfg, row_lengths, positions, payloads, t_pad):
        return _arm_hook_add_single_layer(
            model, layer, dose, row_lengths, positions, payloads, t_pad
        )

    return build


@torch.no_grad()
def run_stage2_block(
    cfg: RunConfig,
    model,
    tok,
    bank: dict,
    block: Stage2Block,
    pairs_by_id: dict[str, BANK.Pair2162],
    donor_maps: dict[str, dict[str, str]],
    contexts: dict[str, dict],
    ctx_ids_cache: dict[str, list[int]],
    eot: list[int],
    regime_fp: str,
    recalibrated: dict[str, int] | None = None,
) -> dict:
    """One stage-2 block: 1 greedy add-mode draw per pair at (arm, layer, dose),
    PLUS the hooked V_a capture (2329 divergence: plan phase_outputs.P4_stage2
    persists stage-2 V_a shards; the parent captured none).

    The delta is the PAIR-DIFFERENCE payload: steered arm delta = V(B) - V(A);
    shuffled arm delta = norm-matched V(B_donor) - V(A) (the B-side reuses the
    stage-1 ``payload_for_arm`` construction wholesale via ``_block_cells``,
    so donor assignment + norm-matching stay bit-identical to stage 1).

    Single-cell block, so the plan §4.7 item-1 per-cell cap resolves to ONE
    value (``recalibrated`` = the gate-slice recalibration)."""
    base_block = Block(block.cell, block.slot, block.arm, block.pair_ids)
    cells = _block_cells(bank, base_block, pairs_by_id, donor_maps)
    cap = _resolve_cap(cfg, block.cell, recalibrated)
    recs = bank["per_context"]
    for c in cells:
        a_state = _slot_state(recs[c["pair"].a], block.slot).unsqueeze(0)  # (1, L, H)
        assert c["payload"].shape == a_state.shape, (c["payload"].shape, a_state.shape)
        c["delta"] = (c["payload"] - a_state).contiguous()

    def ids_for(cid: str) -> list[int]:
        if cid not in ctx_ids_cache:
            ctx_ids_cache[cid] = BANK29.context_token_ids_2389(tok, contexts[cid])
        return ctx_ids_cache[cid]

    texts_per_cell: list[list[str]] = []
    for start in range(0, len(cells), cfg.gen_batch):
        chunk = cells[start : start + cfg.gen_batch]
        ctx_list = [contexts[c["context_a"]] for c in chunk]
        rows = [ids_for(c["context_a"]) for c in chunk]
        row_lengths = [len(r) for r in rows]
        t_pad = max(row_lengths)
        stack = _arm_hook_add_single_layer(
            model,
            block.layer,
            block.dose,
            row_lengths,
            [(c["position"],) for c in chunk],
            [c["delta"] for c in chunk],
            t_pad,
        )
        try:
            outs = generate_batch(
                model,
                tok,
                ctx_list,
                n=STAGE2_DRAWS,
                hook=stack,
                max_new_tokens=cap,
                temperature=STAGE2_TEMPERATURE,
                seed_base=cfg.seed_base,
                render_fn=BANK29.render_context_2389,
                ids_fn=BANK29.context_token_ids_2389,
            )
        finally:
            stack.remove()
        assert len(outs) == len(chunk), (len(outs), len(chunk))
        texts_per_cell.extend(list(o) for o in outs)
    assert len(texts_per_cell) == len(cells)

    # Hooked V_a (single-layer add-mode — the SAME intervention the rollout
    # was generated under), flattened (pair x draw). Rows are built TEXT-ONLY
    # here, in the same nested order as the flat capture lists.
    flat_ctx: list[list[int]] = []
    flat_text: list[str] = []
    flat_delta: list[torch.Tensor] = []
    flat_pos: list[int] = []
    rows_out: list[dict] = []
    for c, texts in zip(cells, texts_per_cell, strict=True):
        pair: BANK.Pair2162 = c["pair"]
        for i, text in enumerate(texts):
            flat_ctx.append(ids_for(c["context_a"]))
            flat_text.append(text)
            flat_delta.append(c["delta"])
            flat_pos.append(c["position"])
            rows_out.append(
                {
                    "block_key": block.key,
                    "cell": block.cell,
                    "slot": block.slot,
                    "arm": block.arm,
                    "mode": "add",
                    "layer": block.layer,
                    "dose": block.dose,
                    "layers_patched": [block.layer],
                    "pair_id": pair.pair_id,
                    "donor_pair_id": c["donor_pair_id"],
                    "carrier": pair.carrier,
                    "value_a": pair.value_a,
                    "value_b": pair.value_b,
                    "context_a": pair.a,
                    "context_id": pair.a,  # audit-walker compat
                    "context_b": pair.b,
                    "position": c["position"],
                    "degenerate_pe": c["degenerate_pe"],
                    "len_delta": c["len_delta"],
                    "draw": i,
                    "seed": cfg.seed_base + i,
                    "temperature": STAGE2_TEMPERATURE,
                    "text": text,
                }
            )
    # Persist the rollout TEXT the moment generation completes, BEFORE the
    # capture reduce (#779 / r2 F9 — the anchors two-write pattern); the
    # post-capture write below atomically REPLACES with the enriched rows.
    shard_jsonl = cfg.rollouts_dir / f"stage2_shard_{block.slug}.jsonl"
    _write_jsonl_atomic(shard_jsonl, rows_out)
    states = capture_answer_states(
        cfg,
        model,
        tok,
        flat_ctx,
        flat_text,
        eot,
        payloads=flat_delta,
        positions=flat_pos,
        hook_builder=_stage2_hook_builder(block.layer, block.dose),
    )
    _enrich_rows_with_capture(rows_out, states, cap)
    _write_jsonl_atomic(shard_jsonl, rows_out)
    _save_pt_atomic(
        cfg.va_dir / f"stage2_shard_{block.slug}.pt",
        {
            "block_key": block.key,
            "layers": cfg.layers,
            "index": [
                {"pair_id": r["pair_id"], "context_a": r["context_a"], "draw": r["draw"]}
                for r in rows_out
            ],
            "va_span": states["va_span"],
            "pooling": states["pooling"],
            "empty_rows": states["empty_rows"],
            "hooked_capture": True,
            "hook": {"mode": "add", "layer": block.layer, "dose": block.dose},
            "repro": _repro(cfg),
        },
    )
    done = {
        "key": block.key,
        "regime_fp": regime_fp,
        "n_rows": len(rows_out),
        "n_cap_hit": sum(1 for r in rows_out if r["cap_hit"]),
        "n_empty": len(states["empty_rows"]),
        "max_new_tokens": cap,
        "layers_patched": [block.layer],
        "repro": _repro(cfg),
    }
    _write_json_atomic(block_done_path(cfg.out_root, block, "stage2_blocks"), done)
    return done


def phase_stage2(cfg: RunConfig) -> int:
    """P4c: claim-queue stage-2 execution over the fact_select survivors.

    ZERO survivors -> SKIP-WITH-RECORD (``manifests/stage2_skipped.json``),
    rc 0 — the documented legitimate fact_select outcome, never a crash.
    Under ``--smoke`` with NO selection file present, a synthetic 1-cell
    selection keeps the stage-2 code path exercised (recorded as synthetic).
    """
    logger.info(
        "[phase=stage2] worker=%d/%d smoke=%s", cfg.worker_index, cfg.num_workers, cfg.smoke
    )
    cfg = _adopt_pilot_gen_batch(cfg)  # plan §4.7 item 3
    best_path = cfg.best_cells_resolved
    synthetic = False
    if cfg.smoke and not best_path.exists():
        synthetic = True
        best = [{"cell": smoke_cells()[0], "slot": "ce"}]
        logger.info("[stage2] smoke synthetic selection: %s", best)
    else:
        best = load_best_cells_actsel(best_path)
    bank = _load_bank(cfg)
    manifest = _load_frozen_manifest(cfg)
    pairs = surviving_pairs(manifest)
    np_ids = no_prefix_ids(manifest)
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_maps = bank["donor_assignments"]
    regime_fp = stage2_regime_fp(cfg, str(bank.get("bank_sha")))
    if not best:
        _write_json_atomic(
            cfg.manifest_dir / "stage2_skipped.json",
            {
                "reason": "fact_select produced zero survivors (documented legitimate outcome)",
                "best_cells_path": str(best_path),
                "regime_fp": regime_fp,
                "repro": _repro(cfg),
            },
        )
        logger.warning("[stage2] zero fact_select survivors — SKIP-with-record, rc 0")
        return RC_OK
    blocks, s2_exclusions = enumerate_stage2_blocks(best, pairs, np_ids, donor_maps, cfg.smoke)
    if s2_exclusions:
        _write_json_atomic(
            cfg.manifest_dir / "stage2_pe_exclusions.json",
            {
                "criterion": "same no-prefix pe rule as stage 1 (pe_excluded_reason)",
                "n_excluded_pair_cells": sum(1 for e in s2_exclusions if e["pair_id"] is not None),
                "n_empty_blocks": sum(1 for e in s2_exclusions if e["pair_id"] is None),
                "exclusions": s2_exclusions,
            },
        )
    totals = grid_totals(blocks, STAGE2_DRAWS)
    logger.info("[stage2] %s synthetic=%s", totals, synthetic)
    assert totals["rollouts_total"] <= STAGE2_ROLLOUT_CAP or cfg.force, (
        totals,
        f"stage-2 rollout budget exceeded (plan §4.3 <={STAGE2_ROLLOUT_CAP}) — "
        "pass --force to override",
    )
    model, tok = load_model_and_tokenizer(cfg)
    eot = eot_tail_ids(tok)
    assert str(bank.get("bank_sha")), "vc_bank carries no bank_sha"
    contexts = BANK.build_contexts()
    ctx_ids_cache: dict[str, list[int]] = {}
    # Gate-slice cap recalibration (plan §4.7 item 1) — up-only per-cell raise.
    recal = _load_cap_recalibration(cfg)
    if recal:
        logger.info("[stage2] cap recalibration active: %s", recal)
    t0 = time.monotonic()
    n_run = 0

    def run_one(block: Stage2Block) -> None:
        nonlocal n_run
        t1 = time.monotonic()
        done = run_stage2_block(
            cfg,
            model,
            tok,
            bank,
            block,
            pairs_by_id,
            donor_maps,
            contexts,
            ctx_ids_cache,
            eot,
            regime_fp,
            recalibrated=recal,
        )
        n_run += 1
        logger.info(
            "[stage2] unit %d %s rows=%d elapsed=%.1fs (total %.1fs)",
            n_run,
            block.key,
            done["n_rows"],
            time.monotonic() - t1,
            time.monotonic() - t0,
        )

    # Queue namespace MUST equal run_stage2_block's done-file namespace
    # ("stage2_blocks") — a mismatch re-runs blocks forever (#2162 margin bug).
    stats = run_claim_queue(cfg, blocks, regime_fp, "stage2_blocks", run_one)
    logger.info("[phase=stage2_done] worker=%d blocks_run=%d", cfg.worker_index, stats["ran"])
    return RC_OK


# ── gate 0b: pod-side arch/version asserts (plan §7 gate 0b) ──────────


def _gate0b_check() -> int:
    """transformers==5.15.0 loads qwen3_5 with 64 resolvable decoder blocks.

    CPU-only + tiny (config download at the PINNED revision + a from-config
    shrunken model — NEVER ``from_pretrained(device_map='meta')``, which
    resolves the ~56 GB shards; plan §7 gate 0b); run by the dispatcher
    BEFORE any phase. On the VM (repo pin 4.57.6) this FAILS by design — it
    is a POD gate for the pod venv's own pin (the VM-side structural check is
    gate 0b's scratch-venv leg in the dispatcher).
    """
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    from explore_persona_space.analysis.extraction import _resolve_decoder_blocks

    v = str(transformers.__version__)
    assert v == "5.15.0", f"gate 0b: transformers=={v}, need ==5.15.0 (plan §7 gate 0b)"
    mcfg = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tc = _model_text_config(mcfg)
    model_types = {str(getattr(mcfg, "model_type", "")), str(getattr(tc, "model_type", ""))}
    assert any("qwen3_5" in m for m in model_types), (
        f"gate 0b: AutoConfig({MODEL_ID}) model_type {model_types} is not qwen3_5"
    )
    assert tc.num_hidden_layers == N_MODEL_LAYERS_FULL, tc.num_hidden_layers
    assert tc.hidden_size == HIDDEN_FULL, tc.hidden_size
    tiny = _shrink_config(
        AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION), 64, N_MODEL_LAYERS_FULL
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(tiny)
    # UNPACK: the helper returns (blocks, embed_tokens, depth) — a bare
    # len() over the tuple is always 3 and can never satisfy this assert.
    blocks, _embed, depth = _resolve_decoder_blocks(model)
    n_blocks = 0 if blocks is None else len(blocks)
    assert n_blocks == N_MODEL_LAYERS_FULL, (
        f"gate 0b: _resolve_decoder_blocks returned {n_blocks} blocks "
        f"(depth={depth}), need {N_MODEL_LAYERS_FULL}"
    )
    print(
        f"[gate0b] OK transformers=={v} model_type={sorted(model_types)} "
        f"blocks={n_blocks} hidden={tc.hidden_size}",
        flush=True,
    )
    return RC_OK


# ── entrypoint ────────────────────────────────────────────────────────


def _import_check() -> None:
    """Resolve EVERY deferred import this driver reaches on its real paths.

    The heavy loads live inside ``load_model_and_tokenizer`` / ``_repro`` /
    ``_upload_dir``, which a bare module import never fires (#1689).
    """
    from transformers import (  # noqa: F401
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    import transformers  # noqa: F401
    from huggingface_hub import HfApi  # noqa: F401
    from scipy.stats import wilcoxon  # noqa: F401

    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)

    from explore_persona_space.analysis.extraction import (  # noqa: F401
        _resolve_decoder_blocks,
    )
    from explore_persona_space.experiments.issue2094.fmetrics import f_act  # noqa: F401
    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        _upload_folder_filtered,
        verify_repo_paths_uploaded,
    )

    # Bank + registries resolve CPU-only (strict bank needs the frozen file —
    # existence asserted here so a missing freeze fails at import-check, not
    # minutes into the pod phase).
    assert BANK.frozen_gen_path().exists(), (
        f"{BANK.FROZEN_GEN_FILENAME} missing — run scripts/issue2162_genfreeze.py first"
    )
    n_pairs = len(BANK.build_pairs())
    assert n_pairs == 1404, n_pairs
    # 2389 constants coherence (model pin + thinking-off template + read layer
    # + ce-only slot scope + stage-2 geometry): asserted here so a fork-base
    # drift fails at import-check.
    assert BANK29.MODEL_ID == MODEL_ID, (BANK29.MODEL_ID, MODEL_ID)
    assert BANK29.MODEL_REVISION == MODEL_REVISION, (BANK29.MODEL_REVISION, MODEL_REVISION)
    assert BANK29.TEMPLATE_KWARGS == {"enable_thinking": False}, BANK29.TEMPLATE_KWARGS
    assert SLOTS == ("ce",), SLOTS
    assert 0 <= F_ACT_READ_LAYER < N_MODEL_LAYERS_FULL
    assert len(STAGE2_LAYERS) == 7 and STAGE2_DOSES == (1, 4)
    assert STAGE2_ARMS == ("steered", "shuffled")
    assert STAGE2_DRAWS == 1 and STAGE2_TEMPERATURE == 0.0
    assert all(0 <= layer < N_MODEL_LAYERS_FULL for layer in STAGE2_LAYERS), STAGE2_LAYERS
    # Per-cell cap table coherence (plan §4.7 item 1): every named cell is a
    # real type-cell; _rev inheritance resolves; default is the 2048 floor.
    all_cells = set(BANK.all_cells())
    unknown = sorted(set(CELL_MAX_NEW_TOKENS) - all_cells)
    assert not unknown, f"CELL_MAX_NEW_TOKENS names unknown cells: {unknown}"
    assert cell_max_new_tokens("filler_swap") == 4096
    assert cell_max_new_tokens("conflict_persona_rev") == 4096  # inherits _fwd
    assert cell_max_new_tokens("fact_user_name") == MAX_NEW_TOKENS
    b = Stage2Block("instr_format", "ce", "shuffled", 30, 4, ("x",))
    assert b.key == "instr_format|ce|shuffled|L30|d4" and "__" in b.slug
    print("[import-check] OK", flush=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.import_check:
        _import_check()
        return RC_OK
    if args.gate0b_check:
        return _gate0b_check()
    assert args.phase, "--phase is required (or pass --import-check / --gate0b-check)"
    cfg = build_config(args)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    if args.gpu_id is not None:
        logger.info(
            "[env] --gpu-id=%s CUDA_VISIBLE_DEVICES=%s",
            args.gpu_id,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
    if cfg.phase == "bank":
        return phase_bank(cfg)
    if cfg.phase == "anchors":
        return phase_anchors(cfg)
    if cfg.phase == "grid":
        return phase_grid(cfg)
    if cfg.phase == "margin":
        return phase_margin(cfg)
    if cfg.phase == "fact_tables":
        return phase_fact_tables(cfg)
    if cfg.phase == "fact_select":
        return phase_fact_select(cfg)
    if cfg.phase == "stage2":
        return phase_stage2(cfg)
    if cfg.phase == "cap_report":
        return phase_cap_report(cfg)
    if cfg.phase == "capregen":
        assert cfg.capregen_scope in ("anchors", "grid"), (
            "--capregen-scope anchors|grid is required for --phase capregen"
        )
        if cfg.capregen_scope == "anchors":
            return phase_capregen_anchors(cfg)
        return phase_capregen_grid(cfg)
    assert cfg.phase == "upload", cfg.phase
    return phase_upload(cfg)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension interpreter finalization (#1689).
    sys.exit(rc)
