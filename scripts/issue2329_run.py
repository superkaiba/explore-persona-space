#!/usr/bin/env python3
"""Issue #2329 — Qwen3.5-9B rerun of the #2162 minimal-pair patch grid (pod driver).

Forked from ``scripts/issue2162_run.py`` (plan §4.2/§4.6; same phase/worker/
checkpoint/resume skeleton, the work-conserving claim-file block queue, the
``--smoke/--tiny/--pilot`` seams, per-block JSONL/pt checkpoints and the
width-keyed sharded resume — all verbatim). **Fork-base flips (the plan §4.6
mechanical review gate):**

- ``MODEL_ID`` -> ``Qwen/Qwen3.5-9B``; ``N_MODEL_LAYERS_FULL`` 28 -> 32;
  ``HIDDEN_FULL`` 3584 -> 4096 (pod venv pins transformers==5.15.0 — the
  dispatcher's gate 0b; the repo pin 4.57.6 lacks the qwen3_5 arch).
- Template kwargs ``{"enable_thinking": False}`` threaded to EVERY render /
  tokenize call via ``bank2329`` (``render_context_2329`` /
  ``context_token_ids_2329`` / ``prefix_end_index_2329``).
- F_act read layer 26/28 -> 30/32 (fraction-matched, plan divergence 4).
- NO-PREFIX pe guard (cross-unit flag, unit 1): under the thinking-off
  template 48 contexts render bare single-turn (36 persona_role_header + 12
  persona_prompted v2) with ``prefix_end == 0`` — NO pe token exists. Their
  pe cells (and pe cells whose null-arm donor payload would come from such a
  context) are SKIPPED-WITH-RECORD (``manifests/pe_exclusions.json``), never
  crashed on and never silently dropped.
- Three phases FOLDED IN (plan divergences 6/7/10): ``fact_tables``
  (pod-side per-draw judge-free F_act + mechanical audits,
  coherence-UNfiltered), ``fact_select`` (mechanical-audit draw filter +
  fully-disjoint pair-clustered CI screen vs BOTH nulls + Holm-IUT
  signed-rank on F_act -> ``best_cells_actsel.json``), and ``stage2`` (the
  parent's separate ``issue2162_stage2.py`` absorbed: pair-difference
  add-mode edits at single layers ``STAGE2_LAYERS`` x doses {1,4}, greedy
  K=1, WITH V_a capture — plan phase_outputs.P4_stage2).
- The generation-throughput pilot (plan §7 gate 4) runs at **P2 ENTRY**
  (dispatcher order: bank -> pilot -> anchors -> gate3 -> grid) and derives
  2x poll fences for EVERY generation phase (P2/P3/P4) plus a refusal on the
  projected TOTAL exceeding 3x the §9 planned total.

Everything else is byte-inherited from the parent: ``GRID_TEMPERATURE=1.0``,
``MAX_NEW_TOKENS=2048``, ``GRID_DRAWS=5``, ``ANCHOR_DRAWS=10``, gen_batch=16,
the P1 injection-exactness + degeneracy HALT gates (distinct rcs), the P2
gate-3-slice-first anchors (contexts sharded across workers), the P3
claim-queue grid with pipelined V_a + opportunistic margin TF, and the P5
bulk-upload + sentinel contract.

Pod-side contract: sentinel file (``/workspace/logs/issue-2329-results.json``)
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
from explore_persona_space.experiments.issue2329 import bank2329 as BANK29  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("issue2329.run")

# ── constants (plan §4.2/§4.3/§4.5/§9/§10) ────────────────────────────

MODEL_ID = "Qwen/Qwen3.5-9B"
HIDDEN_FULL = 4096
N_MODEL_LAYERS_FULL = 32
# Divergence 4: the judge-free activation read layer, 26/28 -> 30/32
# (fraction-matched). Consumed by fact_tables/fact_select; the V_a stores
# still carry ALL layers (parent convention).
F_ACT_READ_LAYER = 30
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# WRITE-side destination override. The canonical data repo above sits at HF's
# hard 1,000,000-file-per-repo cap and refuses EVERY push (see #2304), so a run
# needs somewhere else to persist while that is resolved. READS are deliberately
# left on HF_DATA_REPO — parent artifacts (banks, parent grids, judge pools) live
# there and are still fetchable — so only the upload destination reroutes.
# Unset => byte-identical legacy behavior.
HF_DATA_WRITE_REPO = os.environ.get("EPM_2329_DATA_WRITE_REPO", HF_DATA_REPO)
HF_PREFIX = "issue2329_q35rerun"
DEFAULT_OUT_ROOT = Path("/workspace/issue2329_out")
DEFAULT_LOG_DIR = Path("/workspace/logs")
SENTINEL_NAME = "issue-2329-results.json"
SENTINEL_NAME_SMOKE = "issue-2329-smoke-results.json"

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

SLOTS: tuple[str, ...] = ("ce", "pe")
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
# P4 stage-2). The pilot at P2 ENTRY (plan §7 gate 4) derives 2x fences for
# each and refuses on projected TOTAL > 3x the planned total (4.9 h).
PLANNED_ANCHORS_WALL_H = 1.0
PLANNED_GRID_WALL_H = 3.0
PLANNED_STAGE2_WALL_H = 0.9
PLANNED_GEN_TOTAL_WALL_H = PLANNED_ANCHORS_WALL_H + PLANNED_GRID_WALL_H + PLANNED_STAGE2_WALL_H
PILOT_REFUSAL_MULT = 3.0
PILOT_FENCE_MULT = 2.0

# Claim-file queue (plan §4.6): stale = dead pid (same host; a same-host LIVE
# pid is NEVER stolen — r1 M1) or claim age (the cross-host-only fallback;
# raised 3600 -> 14400 s so a legitimately long block never ages out mid-run).
CLAIM_STALE_S = float(os.environ.get("EPM_2329_CLAIM_STALE_S", "14400"))
CLAIM_POLL_S = float(os.environ.get("EPM_2329_CLAIM_POLL_S", "30"))
# In-flight-tolerant claim read (#2305): a reader can observe a claim file
# EMPTY inside the winner's microsecond open->replace window; retry the read
# a bounded number of times (worst case ~1 s) before treating a still-empty
# claim as a dead writer. Module-level so tests can pin them.
CLAIM_READ_RETRIES = 4
CLAIM_READ_SLEEP_RANGE = (0.05, 0.25)

# Distinct rcs: a designed halt is never an anonymous rc=1 (#1415).
RC_OK = 0
RC_INJECTION_GATE = 21
RC_PILOT_GATE = 22
RC_DEGENERACY_GATE = 23

SMOKE_PAIRS_PER_CELL = 2
SMOKE_GRID_DRAWS = 2

# ── stage-2 (folded from the parent's issue2162_stage2.py; divergence 5) ──
# Pair-difference add-mode edits at SINGLE layers; dose = alpha multiplier on
# the added delta. Layers fraction-matched 28->32 and adjusted to sample the
# hybrid model's full-attention layers (plan divergence 5).
STAGE2_LAYERS: tuple[int, ...] = (9, 15, 16, 19, 23, 25, 30)
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


# ── pure helpers (CPU-only, unit-tested in tests/test_issue2329_run.py) ──


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
    """The 234 grid blocks: 39 type-cells x 2 slots x 3 arms (plan §4.3).

    ``pairs`` is the SURVIVING set (gate 0a token-identity drops applied), so
    per-cell counts sit in [INTACT_FLOOR_PER_CELL, 36] rather than the
    parent's exact 36 (plan divergence 9); pe-slot no-prefix exclusions are
    applied AFTERWARD by :func:`apply_pe_exclusions`.
    """
    by_cell = BANK.pairs_by_cell(pairs)
    blocks: list[Block] = []
    for cell in BANK.all_cells():
        ids = tuple(p.pair_id for p in by_cell[cell])
        assert BANK29.INTACT_FLOOR_PER_CELL <= len(ids) <= 36, (cell, len(ids))
        for slot in SLOTS:
            for arm in ARMS:
                blocks.append(Block(cell, slot, arm, ids))
    assert len(blocks) == 39 * len(SLOTS) * len(ARMS) == 234, len(blocks)
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


_EMPTY_STALE = object()
"""Sentinel: the claim file was still EMPTY after the full bounded read (#2305)."""


def _read_claim_inflight_tolerant(path: Path) -> dict | None | object:
    """Read + parse a claim file, tolerating an in-flight writer (#2305).

    A reader can hit the microsecond window between the winner's
    ``O_CREAT | O_EXCL`` open and its atomic payload ``os.replace`` (the
    writer path in ``try_claim``), observing the claim file EMPTY. Retry the
    read up to ``CLAIM_READ_RETRIES`` times, sleeping
    ``random.uniform(*CLAIM_READ_SLEEP_RANGE)`` between attempts (module
    constants so tests can pin them; worst-case bound ~1 s). Returns:

    - the parsed record ``dict`` — the normal case;
    - ``None`` — the claim VANISHED (released, or consumed by a concurrent
      reclaim); the caller returns False and the outer scan revisits;
    - ``_EMPTY_STALE`` — still EMPTY after the full bound: with the atomic
      writer in place no LIVE same-version writer holds an empty claim for
      ~1 s, so the writer died inside the open->replace window; the caller
      falls into the stale-reclaim path. Residual accepted risk (same class
      as the r1 M1 settle comment): a live writer stalled past the bound
      inside that window — SIGSTOP'd, or delayed by network-FS (MooseFS
      cross-host) visibility lag — is reclaimed as empty-dead and the block
      double-runs; tolerated by release_claim's stolen-claim tolerance +
      atomic idempotent done-files, never corruption.

    Persistent NON-EMPTY garbage raises the same hard ``RuntimeError`` as
    before: with atomic payload writes, partial JSON from a live writer is
    impossible, so it is genuine corruption — fail-fast stays.
    """
    data: bytes = b""
    last_err: Exception | None = None
    for attempt in range(CLAIM_READ_RETRIES):
        if attempt:
            time.sleep(random.uniform(*CLAIM_READ_SLEEP_RANGE))
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as e:
            last_err = e
            continue
        if data == b"":
            last_err = None
            continue
        try:
            return json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            last_err = e
            continue
    if data == b"" and last_err is None:
        logger.warning(
            "[claims] EMPTY claim %s after %d bounded reads — treating as dead "
            "writer (died inside the create window)",
            path,
            CLAIM_READ_RETRIES,
        )
        return _EMPTY_STALE
    raise RuntimeError(
        f"unparseable claim file {path} — inconsistent claim state, refusing "
        "to guess (delete it manually after diagnosing the writer)"
    ) from last_err


def try_claim(cdir: Path, block: Block, worker_index: int, token: str) -> bool:
    """Atomically claim one block; reclaim a STALE claim; never skip silently.

    Fresh claim: ``O_CREAT | O_EXCL`` on the target decides the election —
    exactly one worker wins — and the winner lands its payload atomically
    (tmp + ``os.replace``), so a reader can observe the claim file EMPTY
    (the microsecond open->replace window) but never partially written
    (#2305). Existing claim: read it with bounded in-flight tolerance
    (``_read_claim_inflight_tolerant``): a VANISHED claim returns False (the
    outer scan revisits), a still-EMPTY claim after the bound is a dead
    writer and falls into the stale-reclaim path, persistent non-empty
    garbage stays an inconsistent-state HARD failure. When stale, atomically
    replace it and verify OUR token won (two workers can both see stale;
    ``os.replace`` serializes, last writer wins, the read-back arbitrates).
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
        rec = _read_claim_inflight_tolerant(path)
        if rec is None:
            # Claim VANISHED mid-read (released, or consumed by a concurrent
            # reclaim) — never a raise, never a silent skip: the
            # work-conserving outer scan revisits the block on its next pass.
            return False
        empty_dead = rec is _EMPTY_STALE
        if not empty_dead and not _claim_stale(rec):
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
            if empty_dead:
                # Distinct success line (#2305): the empty claim carried NO
                # fields — never format pid/host/age off a record that was
                # never parsed.
                logger.warning(
                    "[claims] reclaimed EMPTY claim %s (dead writer in create window)",
                    block.key,
                )
            else:
                logger.info(
                    "[claims] reclaimed STALE claim %s (was pid=%s host=%s age=%.0fs)",
                    block.key,
                    rec.get("pid"),
                    rec.get("host"),
                    time.time() - float(rec.get("ts", 0.0)),
                )
        return won
    # Winner (#2305): the election is decided by the O_CREAT|O_EXCL open
    # above; close the empty fd and land the payload the way the
    # stale-reclaim path does — tmp sibling + atomic os.replace — so readers
    # observe the claim EMPTY or COMPLETE, never partially written.
    os.close(fd)
    tmp = path.parent / f"{path.name}.tmp.{token}"
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)
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
) -> dict:
    """Work-conserving queue: pull the next unclaimed pending block until every
    block is done (crashed workers' claims go stale and are reclaimed).

    ``is_done`` defaults to :func:`block_is_done` (the #722 r3 hard-refusal
    resume predicate). ``phase_capregen_grid`` passes
    :func:`_capregen_block_done`, which PRESERVES that hard refusal and
    additionally treats a pre-regen done record as PENDING."""
    cdir = claims_dir(cfg.out_root, namespace)
    stats = {"ran": 0, "skipped_done": 0, "waits": 0}
    while True:
        ran_this_scan = 0
        n_open = 0
        for block in blocks:
            if is_done(cfg.out_root, block, regime_fp, namespace):
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

        _BANK_TOK = AutoTokenizer.from_pretrained(MODEL_ID)
    return _BANK_TOK


def bank_manifest_and_sha() -> tuple[dict, str]:
    """Deterministic 2329 bank manifest + sha (the regime key), CPU + tokenizer.

    ``bank2329.bank_manifest_2329`` enforces the >=30/36 per-cell
    token-identity floor (gate 0a) — a floor breach raises
    ``TokenIdentityFloorError`` BEFORE any GPU spend. Cached per process (the
    build re-tokenizes all 1,404 contexts, ~O(30s) CPU).
    """
    global _BANK_MANIFEST_CACHE
    if _BANK_MANIFEST_CACHE is None:
        manifest = BANK29.bank_manifest_2329(_bank_tokenizer())
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
            # #2329 q35_ladder_decay M1/R2-M1 edit 3: the model-repo revision is an
            # output-affecting knob (fork plan divergence 11). None for legacy
            # RunConfig objects that carry no pin (pre-fork resume keys unchanged
            # only for cfgs WITHOUT the attribute -- the fork's out-root is fresh).
            "model_revision": getattr(cfg, "model_revision", None),
            "tiny": cfg.tiny,
            "n_layers": cfg.n_layers,
            "hidden": cfg.hidden,
            "max_new_tokens": cfg.max_new_tokens,
            "grid_temperature": GRID_TEMPERATURE,
            "grid_draws": cfg.grid_draws,
            "seed_base": cfg.seed_base,
            "smoke": cfg.smoke,
            "bank_seed": BANK.SEED,
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
        cfg.bank_dir / "injection_gate_report.json",
        cfg.bank_dir / "degeneracy_report.json",
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


def _anchor_batch_done(cfg: RunConfig, regime_fp: str, batch: str, expected_draws: int) -> bool:
    """Per-worker per-batch anchors resume predicate (gate slice vs rest)."""
    rec = _sharded_done_record(cfg, f"anchors_{batch}_w{cfg.worker_index}", regime_fp)
    if rec is None:
        return False
    if int(rec.get("draws", -1)) != expected_draws:
        logger.warning(
            "[anchors:%s] done draws=%s != %d — re-running", batch, rec.get("draws"), expected_draws
        )
        return False
    jsonl = cfg.anchors_dir / f"anchors_{batch}_w{cfg.worker_index}.jsonl"
    va = cfg.anchors_dir / f"va_anchors_{batch}_w{cfg.worker_index}.pt"
    if not (jsonl.exists() and va.exists()):
        logger.warning(
            "[anchors:%s] done-manifest present but artifacts missing — re-running", batch
        )
        return False
    n_rows = sum(1 for line in jsonl.open(encoding="utf-8") if line.strip())
    if n_rows != int(rec.get("n_rows", -1)):
        logger.warning(
            "[anchors:%s] done n_rows=%s but jsonl has %d — re-running",
            batch,
            rec.get("n_rows"),
            n_rows,
        )
        return False
    return True


# r3 C1: width-sharded artifacts are grouped into FAMILIES; a sweep is scoped
# to exactly one family so no phase can ever destroy ANOTHER family's shards
# (r2 C1: phase_margin/phase_upload applied ONE process's width to EVERY
# family — a 1-wide deferred margin leg would have quarantined 7/8 of a valid
# 8-wide anchor store, and the width-less upload leg quarantined w1..wN at
# implicit width 1 before the bulk upload).
_ARTIFACT_FAMILIES: dict[str, frozenset[str]] = {
    "anchors": frozenset({"anchors_gate", "anchors_rest", "va_anchors_gate", "va_anchors_rest"}),
    "margin": frozenset({"anchor_margin", "margin_anchors"}),
}
_WIDTH_SHARDED_STEMS = frozenset().union(*_ARTIFACT_FAMILIES.values())
# Done-record stems whose per-width index coverage DEFINES a family's realized
# width (every kind must be complete at the same W).
_FAMILY_DONE_KINDS: dict[str, tuple[str, ...]] = {
    "anchors": ("anchors_gate", "anchors_rest"),
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

    @property
    def rollouts_dir(self) -> Path:
        return self.out_root / "rollouts"

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
            "Issue #2329 pod driver (bank / anchors / grid / margin / fact_tables / "
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
        help="gate 0b: assert transformers==5.15.0 loads qwen3_5 with 32 resolvable "
        "decoder blocks (pod-side, CPU, tiny from-config model) and exit 0",
    )
    # UPLOAD_PREFIX_EXEMPT: per-issue pod driver, issue-pinned end-to-end (HF_PREFIX + sentinel + out-root); child issues reuse by forking (2162->2329 convention), never runtime prefix override — the #1005 clobber shape cannot arise from this default.
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--tiny", action="store_true", help="from-config tiny CPU model (smoke)")
    ap.add_argument("--tiny-layers", type=int, default=4)
    ap.add_argument("--tiny-hidden", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda | cuda:0 | cpu (default: auto)")
    ap.add_argument("--gen-batch", type=int, default=16, help="cells per hooked generate call")
    ap.add_argument("--capture-batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--anchor-draws", type=int, default=ANCHOR_DRAWS)
    ap.add_argument("--grid-draws", type=int, default=GRID_DRAWS)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--smoke", action="store_true", help="tiny per-arm-class slice")
    ap.add_argument("--pilot", action="store_true", help="grid: timing pilot only")
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
        tiny=args.tiny,
        n_layers=args.tiny_layers if args.tiny else N_MODEL_LAYERS_FULL,
        hidden=args.tiny_hidden if args.tiny else HIDDEN_FULL,
        device=device,
        gen_batch=args.gen_batch,
        capture_batch=args.capture_batch,
        max_new_tokens=args.max_new_tokens,
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
        # None when the config carries no pin (parent grid run); the ladder fork
        # (issue2329_ladder) always pins (plan divergence 11).
        "model_revision": getattr(cfg, "model_revision", None),
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


def load_model_and_tokenizer(cfg: RunConfig, revision: str | None = None):
    """Production: bf16 Qwen3.5-9B pinned to one device (never
    ``device_map='auto'`` — silent CPU offload, gotchas). Tiny: a from-config
    same-arch model on CPU over the REAL vocab-id space.

    ``revision`` (additive, default None == legacy behavior) pins BOTH the
    tokenizer and the model to a model-repo commit (#2329 q35_ladder_decay
    fork, plan divergence 11). The ``--tiny`` branch builds from config, so
    the pin reaches only the tokenizer + AutoConfig there.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # loaded ONCE (HF-429 gotcha)
    tok = AutoTokenizer.from_pretrained(cfg.model_id, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if cfg.tiny:
        mcfg = _shrink_config(
            AutoConfig.from_pretrained(cfg.model_id, revision=revision), cfg.hidden, cfg.n_layers
        )
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(mcfg).to(torch.float32)
    else:
        assert torch.cuda.is_available(), "the full grid requires CUDA (use --tiny for CPU smoke)"
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, dtype=torch.bfloat16, revision=revision
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
def capture_bank(cfg: RunConfig, model, tok) -> dict:
    """All-layer v_ce + v_pe per context: one right-padded forward per chunk.

    Positions come from the token ids' own offsets (BPE-seam rule): ``ce`` =
    ctx_len − 1 (generation prompt included), ``pe`` = prefix_end − 1 (the
    last token BEFORE the final user turn, via ``prefix_end_index_multi``).
    """
    contexts = BANK.build_contexts()
    ctx_ids = {cid: BANK29.context_token_ids_2329(tok, c) for cid, c in contexts.items()}
    prefix_ends = {cid: BANK29.prefix_end_index_2329(tok, ids) for cid, ids in ctx_ids.items()}
    layers = cfg.layers
    pad_id = tok.pad_token_id
    records: dict[str, dict] = {}
    order = list(contexts)
    for start in range(0, len(order), cfg.capture_batch):
        chunk = order[start : start + cfg.capture_batch]
        ids, mask = _right_pad([ctx_ids[c] for c in chunk], pad_id, cfg.device)
        captured = extract_layer_activations(model, ids, layers, attention_mask=mask)
        for j, cid in enumerate(chunk):
            ctx_len = len(ctx_ids[cid])
            pe = prefix_ends[cid]
            assert 0 <= pe < ctx_len, (cid, ctx_len, pe)
            v_ce = torch.stack([captured[layer][j, ctx_len - 1] for layer in layers])
            if pe >= 1:
                v_pe = torch.stack([captured[layer][j, pe - 1] for layer in layers]).float().cpu()
            else:
                # NO-PREFIX context (bare thinking-off render, unit-1 flag):
                # no pe token exists. Zeros + the no_prefix flag; every
                # consumer is enumeration-guarded (apply_pe_exclusions) and
                # payload_for_arm hard-asserts besides, so a zero row can
                # never be silently consumed as a state.
                v_pe = torch.zeros((len(layers), cfg.hidden), dtype=torch.float32)
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
                "v_pe": v_pe,
            }
        del captured
        if (start // cfg.capture_batch) % 20 == 0:
            logger.info(
                "[bank] unit %d/%d contexts elapsed",
                min(start + cfg.capture_batch, len(order)),
                len(order),
            )
    assert len(records) == len(contexts), (len(records), len(contexts))
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
        ids = BANK29.context_token_ids_2329(tok, contexts[cid])
        out[cid] = (ids, BANK29.prefix_end_index_2329(tok, ids))
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
    """Plan §7 gate 2 — pair-slot identity vs the span-locus registry.

    The two pre-declared degenerate cells (`query_content`,
    `persona_role_header` — final-query / generation-header loci) assert the
    PREMISE directly: identical token prefixes through the v_pe slot
    (``pe_a == pe_b`` and ``ids_a[: pe_a + 1] == ids_b[: pe_b + 1]``) — exact,
    deterministic, batch-invariant. The captured bf16 states carry ~2e-4
    cosine batch-composition jitter whenever a pair's two contexts land in
    different capture batches, so the former state-space read
    (``pe_cos >= DEGENERACY_COS_MIN``) was a bit-identity test that
    false-FAILed a healthy bank (2026-08-06 rc=23 halt: 4 persona_role_header
    pairs at pe_cos 0.9998). Realized pe_cos per degenerate pair — plus the
    max jitter over premise-verified pairs — is recorded informationally,
    never gated at the bit-identity bar. Premise-verified pairs DO keep one
    LOOSE state-side sanity band (``STATE_SANITY_COS_MIN`` = 0.99, flag
    ``state_sanity``): identical token prefixes with ``pe_cos < 0.99`` mean a
    capture-side row misalignment wrote garbage ``v_pe`` — the injection gate
    cannot backstop that (it compares realized-vs-payload, so a misaligned
    vector reads back as itself). The band sits ~50x above the measured
    ``max_pe_jitter`` (2.04e-4; bit-identical states read > 1.0 - 1e-5), so it
    cannot reintroduce the 2026-08-06 false-FAIL. EVERY other (pair x slot)
    direction is unchanged: distinct states at BOTH slots
    (``cos < DEGENERACY_COS_MIN``; jitter cannot flip it — distinct states
    read <= ~0.994). A violation is a BANK defect (HALT), never a runtime m
    adjustment.

    One-sided ``no_prefix`` flags are EXPECTED when exactly one of the pair's
    two contexts carries no system message in the FROZEN bank AND the bare
    (no-prefix) side IS the system-absent side: ``persona_prompted`` v2 is
    the deliberate NO-PERSONA control arm (``system: null``), and Qwen3.5's
    thinking-off chat template inserts no default system turn, so its render
    is bare while v1/v3 carry a system turn (2026-08-16 rc=23 halt — 24/1404
    false violations, all ``persona_prompted`` v2-vs-{v1,v3}). Such pairs are
    recorded under ``no_prefix_asymmetry_expected`` — never violations; their
    pe identity/distinctness checks are N/A (no pe token exists on the bare
    side; ``pe_excluded_reason`` already enumeration-excludes their pe
    cells), mirroring the both-sides-no-prefix branch, while ce distinctness
    still runs unchanged. A ``no_prefix`` mismatch whose system-presence
    AGREES — or whose bare side is NOT the system-absent side — remains a
    HALT violation (``no_prefix_mismatch``): a genuine render/capture defect.

    ``token_prefixes`` (tests / precomputed callers) bypasses the tokenizer;
    production threads ``tok`` and derives it via ``_degenerate_token_prefixes``.
    ``system_presence`` (tests / precomputed callers) bypasses the frozen bank
    build; production derives it lazily via ``_bank_system_presence``.
    """
    recs = bank["per_context"]
    degenerate_cids: set[str] = set()
    for pair in pairs:
        if BANK.base_type_of(pair.cell) in BANK.DEGENERATE_AT_PE:
            degenerate_cids.update((pair.a, pair.b))
    if token_prefixes is None:
        assert tok is not None, "run_degeneracy_guard needs tok (or precomputed token_prefixes)"
        token_prefixes = _degenerate_token_prefixes(tok, degenerate_cids)
    missing = degenerate_cids - set(token_prefixes)
    assert not missing, f"token_prefixes missing degenerate contexts: {sorted(missing)[:5]}"

    violations: list[dict] = []
    no_prefix_asymmetry_expected: list[dict] = []
    n_checked = 0
    n_no_prefix_pe = 0
    degenerate_pe_cos: dict[str, float] = {}
    jitters: list[float] = []
    for pair in pairs:
        ra, rb = recs[pair.a], recs[pair.b]
        pe_cos = float(safe_cosine(ra["v_pe"].flatten(), rb["v_pe"].flatten()))
        ce_cos = float(safe_cosine(ra["v_ce"].flatten(), rb["v_ce"].flatten()))
        degenerate_pe = BANK.base_type_of(pair.cell) in BANK.DEGENERATE_AT_PE
        np_a = bool(ra.get("no_prefix"))
        np_b = bool(rb.get("no_prefix"))
        n_checked += 1
        failed: list[str] = []
        row_extra: dict = {}
        if np_a != np_b:
            if system_presence is None:
                system_presence = _bank_system_presence()
            assert pair.a in system_presence and pair.b in system_presence, (
                pair.pair_id,
                "system_presence missing pair contexts",
            )
            sys_a = bool(system_presence[pair.a])
            sys_b = bool(system_presence[pair.b])
            if sys_a != sys_b and np_a == (not sys_a):
                # EXPLAINED asymmetry: exactly one side has NO system message
                # in the frozen bank and the bare (no-prefix) side IS that
                # side — a thinking-off-template consequence (no default
                # system turn), not a render/capture defect. pe checks are
                # N/A for this pair (no pe token on the bare side; its pe
                # cells are enumeration-excluded via pe_excluded_reason),
                # mirroring the both-sides-no-prefix branch below; the ce
                # distinctness check still runs unchanged.
                no_prefix_asymmetry_expected.append(
                    {
                        "pair_id": pair.pair_id,
                        "cell": pair.cell,
                        "no_prefix_side": "a" if np_a else "b",
                        "system_absent_side": "a" if not sys_a else "b",
                    }
                )
            else:
                # Pairs share their carrier, so with system-presence AGREEING
                # their template SHAPE must agree — a one-sided no-prefix
                # render is a bank defect (unit-1 flag). Same verdict when
                # the bare side is NOT the system-absent side (system
                # asymmetry then does not explain the render asymmetry).
                failed.append("no_prefix_mismatch")
        elif np_a:
            # BOTH sides no-prefix (thinking-off bare renders): NO pe slot
            # exists on either side, so pe identity/distinctness is N/A —
            # their pe cells are enumeration-excluded (apply_pe_exclusions).
            # The ce checks below still run unchanged.
            n_no_prefix_pe += 1
        elif degenerate_pe:
            ids_a, pe_a = token_prefixes[pair.a]
            ids_b, pe_b = token_prefixes[pair.b]
            prefix_identical = pe_a == pe_b and ids_a[: pe_a + 1] == ids_b[: pe_b + 1]
            degenerate_pe_cos[pair.pair_id] = pe_cos
            if prefix_identical:
                if pe_cos >= STATE_SANITY_COS_MIN:
                    jitters.append(1.0 - pe_cos)
                else:
                    # Loose state-sanity band (docstring): identical prefixes
                    # MUST yield near-identical captured states; a miss (incl.
                    # NaN) is a capture-side row misalignment, not jitter.
                    failed.append("state_sanity")
            else:
                failed.append("token_prefix")
                row_extra = {"pe_a": pe_a, "pe_b": pe_b}
        elif not (pe_cos < DEGENERACY_COS_MIN):
            failed.append("distinctness_pe")
        if not (ce_cos < DEGENERACY_COS_MIN):
            failed.append("distinctness_ce")
        if failed:
            violations.append(
                {
                    "pair_id": pair.pair_id,
                    "cell": pair.cell,
                    "declared_degenerate_pe": degenerate_pe,
                    "pe_cos": pe_cos,
                    "ce_cos": ce_cos,
                    "flag": "+".join(failed),
                    **row_extra,
                }
            )
    if no_prefix_asymmetry_expected:
        logger.info(
            "[degeneracy_guard] %d one-sided no-prefix pairs EXPLAINED by one-sided system "
            "absence in the frozen bank (cells: %s) — recorded, not violations",
            len(no_prefix_asymmetry_expected),
            ", ".join(sorted({r["cell"] for r in no_prefix_asymmetry_expected})),
        )
    report = {
        "criterion": "span-locus degeneracy guard (plan §7 gate 2)",
        "bar_cos": DEGENERACY_COS_MIN,
        "state_sanity_cos_min": STATE_SANITY_COS_MIN,
        "degenerate_criterion": (
            "token-prefix identity through the v_pe slot "
            "(pe_a == pe_b and ids[: pe + 1] equal); pe_cos recorded, not gated "
            "at the bit-identity bar — premise-verified pairs keep the loose "
            "state_sanity band (pe_cos >= state_sanity_cos_min)"
        ),
        "declared_degenerate_cells": sorted(BANK.DEGENERATE_AT_PE),
        "n_pairs_checked": n_checked,
        "n_violations": len(violations),
        "violations": violations[:50],
        "passed": not violations,
        "degenerate_pe_cos": degenerate_pe_cos,
        "n_degenerate_pairs": len(degenerate_pe_cos),
        "n_no_prefix_pe_pairs": n_no_prefix_pe,
        "no_prefix_asymmetry_expected": no_prefix_asymmetry_expected,
        "n_no_prefix_asymmetry_expected": len(no_prefix_asymmetry_expected),
        "max_pe_jitter": max(jitters) if jitters else None,
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
    """12 spot cells spanning slot x arm x carrier-class / crossed / degenerate
    cell classes (plan §7 gate 1). pe spots select only pe-RUNNABLE pairs
    (no-prefix exclusions, unit-1 flag) when ``np_ids`` is threaded."""
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
        (cls_rep["P"], "pe", "shuffled", 1),
        (cls_rep["E"], "ce", "crosstype", 0),
        (cls_rep["ICL"], "ce", "steered", 2),
        ("query_content", "ce", "steered", 0),
        ("query_content", "pe", "shuffled", 1),
        ("persona_role_header", "ce", "steered", 0),
        ("language_implied", "pe", "steered", 0),
        (conflict, "ce", "steered", 0),
        (recency, "pe", "steered", 0),
        (load, "ce", "shuffled", 0),
        (cls_rep.get("P12", cls_rep["P"]), "pe", "crosstype", 3),
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
    pe_second_row_ok=None,
) -> dict:
    """Plan §7 gate 1 — the realized edit equals the intended donor state at
    the intended (row, position, layer) and NOWHERE else.

    The keyword-only ``contexts`` / ``ids_fn`` / ``spots`` / ``payload_fn`` /
    ``pe_second_row_ok`` seams (defaults = this module's own registries —
    byte-equivalent for every existing caller) let the #2162 LADDER driver
    reuse this gate verbatim over its own bank
    (``scripts/issue2162_ladder.py``; ladder-plan §4.6 "IMPORTS, never
    re-implements, the injection-gate helper"). ``spots`` rows keep the
    ``{"cell", "slot", "arm", "pair"}`` shape; ``payload_fn`` keeps
    :func:`payload_for_arm`'s call signature. ``pe_second_row_ok(p, arm)``
    is the pe-slot second-row runnability predicate — the default binds this
    module's :func:`pe_excluded_reason` (parent donor-map keys
    ``{"shuffled", "crosstype"}`` with PAIR-id values); a caller whose
    ``donor_maps`` uses a DIFFERENT key/value convention (the #2329 ladder:
    ``{"null_sameval", "null_xtype", "null_xtype_pe"}`` with CONTEXT-id
    values) MUST pass its own predicate, or the default ``KeyError``s.

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
    ids_fn = BANK29.context_token_ids_2329 if ids_fn is None else ids_fn
    payload_fn = payload_for_arm if payload_fn is None else payload_fn
    ctx_ids = {cid: ids_fn(tok, c) for cid, c in contexts.items()}
    pad_id = tok.pad_token_id
    recs = bank["per_context"]
    pairs_by_id = {p.pair_id: p for p in pairs}
    # No-prefix set (unit-1 flag) from the captured bank records: pe spots and
    # pe second rows must stay pe-runnable.
    np_ids = frozenset(cid for cid, r in recs.items() if r.get("no_prefix"))
    if pe_second_row_ok is None:

        def pe_second_row_ok(p, arm):
            return pe_excluded_reason(p, arm, np_ids, donor_maps, pairs_by_id) is None

    spots = _gate_spot_specs(pairs, np_ids, donor_maps, pairs_by_id) if spots is None else spots
    results: list[dict] = []
    for spot in spots:
        pair: BANK.Pair2162 = spot["pair"]
        slot, arm = spot["slot"], spot["arm"]
        # Second row: a DIFFERENT pair whose A-context length differs, so the
        # padded-offset math is exercised rather than degenerate. pe spots
        # additionally require the second row to be pe-runnable (unit-1 flag).
        others = [
            p
            for p in pairs
            if p.pair_id != pair.pair_id
            and len(ctx_ids[p.a]) != len(ctx_ids[pair.a])
            and (slot != "pe" or pe_second_row_ok(p, arm))
        ]
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


def phase_bank(cfg: RunConfig) -> int:
    """P1: bank.json + vc_bank.pt + degeneracy guard + injection gate.

    Idempotent: a completed same-regime bank is SKIPPED at entry, BEFORE the
    model load; ``--force`` deliberately re-runs. Gate failures write their
    report JSONs and exit DISTINCT rcs (never a bare rc=1).
    """
    logger.info("[phase=bank]")
    manifest, bank_sha = bank_manifest_and_sha()
    regime_fp = regime_fingerprint(cfg, bank_sha)
    if not cfg.force and bank_is_done(cfg, regime_fp):
        logger.info("[bank] already done for this regime — skipping (--force re-runs)")
        logger.info("[phase=bank_done]")
        return RC_OK
    if cfg.force and (cfg.manifest_dir / "bank_done.json").exists():
        logger.info("[bank] --force set: deliberately re-running a done bank phase")
    model, tok = load_model_and_tokenizer(cfg)
    # Gate 0a freeze: token_identity_report.json is written BEFORE the floor
    # check (HALT artifact), then the DETERMINISTIC bank.json (no timestamps —
    # its sha IS the regime key) + the bank.meta.json provenance sidecar.
    frozen_manifest = BANK29.freeze_bank_2329(
        tok, cfg.bank_dir / "bank.json", cfg.bank_dir / "token_identity_report.json"
    )
    frozen_bytes = json.dumps(frozen_manifest, sort_keys=True, ensure_ascii=False).encode()
    assert _sha256_bytes(frozen_bytes) == bank_sha, (
        "freeze/manifest sha drift — the run tokenizer disagrees with the "
        "MODEL_ID bank tokenizer (check --model-id)"
    )

    bank = capture_bank(cfg, model, tok)
    # Cross-check the CAPTURED no-prefix set against the frozen manifest
    # (unit-1 flag): 48 expected — 36 persona_role_header + 12 v2.
    realized_np = {cid for cid, r in bank["per_context"].items() if r.get("no_prefix")}
    assert realized_np == set(manifest["no_prefix_context_ids"]), (
        sorted(realized_np ^ set(manifest["no_prefix_context_ids"]))[:5],
        "captured no-prefix set != bank.json no_prefix_context_ids",
    )
    pairs = surviving_pairs(manifest)
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

    # Gate 2 first (pair-slot identity — a bank defect makes gate 1 meaningless).
    degeneracy = run_degeneracy_guard(bank, pairs, tok)
    degeneracy["repro"] = _repro(cfg)
    _write_json_atomic(cfg.bank_dir / "degeneracy_report.json", degeneracy)
    if not degeneracy["passed"]:
        logger.error(
            "[degeneracy_guard] FAILED: %d/%d pair-slot identity violations (bank defect)",
            degeneracy["n_violations"],
            degeneracy["n_pairs_checked"],
        )
        if not cfg.force_past_halt_gates:
            return RC_DEGENERACY_GATE
        logger.error(
            "[degeneracy_guard] --force-past-halt-gates set: proceeding on a FAILED guard "
            "(recorded)"
        )

    report = run_injection_gate(cfg, model, tok, bank, pairs, donor_maps)
    _write_json_atomic(cfg.bank_dir / "injection_gate_report.json", report)
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


def _enrich_rows_with_capture(rows: list[dict], states: dict, max_new_tokens: int) -> None:
    """Post-capture per-row telemetry: token count + cap_hit vs the REALIZED
    generating cap, and the cap itself. Recording the cap per row is what
    keeps a mixed-cap store (protocol-sanctioned after a cell-restricted
    capregen) VISIBLE downstream instead of implicit."""
    for r, n_tok in zip(rows, states["n_completion_tokens"], strict=True):
        r["n_completion_tokens"] = n_tok
        r["cap_hit"] = cap_hit(n_tok, max_new_tokens)
        r["cap_hit_basis"] = "retokenized_completion_len >= max_new_tokens"
        r["max_new_tokens"] = max_new_tokens


def _generate_anchor_rows(
    cfg: RunConfig,
    model,
    tok,
    contexts: dict[str, dict],
    order: list[str],
    draws: int,
    batch: str,
) -> tuple[list[dict], list[list[int]], list[str]]:
    """Unpatched anchor generation core — ``(rows, flat_ctx, flat_text)``.

    The SINGLE anchors generation loop, shared by ``_run_anchor_batch`` and
    ``phase_capregen_anchors`` (no second generation loop; capture +
    enrichment stay caller-side so the two-write text-persist ordering is
    preserved at each call site)."""
    rows: list[dict] = []
    flat_ctx: list[list[int]] = []
    flat_text: list[str] = []
    ctx_ids = {cid: BANK29.context_token_ids_2329(tok, contexts[cid]) for cid in order}
    t0 = time.monotonic()
    for start in range(0, len(order), cfg.gen_batch):
        chunk = order[start : start + cfg.gen_batch]
        outs = generate_batch(
            model,
            tok,
            [contexts[c] for c in chunk],
            n=draws,
            hook=None,
            max_new_tokens=cfg.max_new_tokens,
            temperature=ANCHOR_TEMPERATURE,
            seed_base=cfg.seed_base,
            render_fn=BANK29.render_context_2329,
            ids_fn=BANK29.context_token_ids_2329,
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
                        "gate_slice": batch == "gate",
                        "text": text,
                    }
                )
        logger.info(
            "[anchors:%s] unit %d/%d contexts elapsed=%.1fs",
            batch,
            min(start + cfg.gen_batch, len(order)),
            len(order),
            time.monotonic() - t0,
        )
    return rows, flat_ctx, flat_text


def _run_anchor_batch(
    cfg: RunConfig,
    model,
    tok,
    contexts: dict[str, dict],
    order: list[str],
    draws: int,
    batch: str,
    regime_fp: str,
) -> dict:
    """Generate + capture one anchors batch for THIS worker; write shards."""
    eot = eot_tail_ids(tok)
    rows, flat_ctx, flat_text = _generate_anchor_rows(
        cfg, model, tok, contexts, order, draws, batch
    )
    # Persist the rollout TEXT the moment generation completes, BEFORE the
    # capture reduce (#779 / r1 m2): a capture crash must never lose ~1,650
    # generated rollouts. The post-capture write below atomically REPLACES
    # this file with the capture-enriched rows (adds token counts / cap_hit).
    jsonl = cfg.anchors_dir / f"anchors_{batch}_w{cfg.worker_index}.jsonl"
    _write_jsonl_atomic(jsonl, rows)
    states = capture_answer_states(cfg, model, tok, flat_ctx, flat_text, eot)
    _enrich_rows_with_capture(rows, states, cfg.max_new_tokens)
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
    _write_json_atomic(
        cfg.manifest_dir / f"anchors_{batch}_w{cfg.worker_index}_done.json",
        {
            "regime_fp": regime_fp,
            "batch": batch,
            "worker_index": cfg.worker_index,
            "num_workers": cfg.num_workers,  # shard identity (r1 M2)
            "n_contexts": len(order),
            "draws": draws,
            "n_rows": len(rows),
            "n_cap_hit": cap_hits,
            "n_empty": len(states["empty_rows"]),
            "max_new_tokens": cfg.max_new_tokens,
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


def phase_anchors(cfg: RunConfig) -> int:
    """P2: unpatched temp-1.0 anchors, contexts SHARDED across workers, the
    gate-3 slice generated + uploaded FIRST so the SYNC judge overlaps the
    remaining generation (plan §9)."""
    logger.info(
        "[phase=anchors] worker=%d/%d smoke=%s", cfg.worker_index, cfg.num_workers, cfg.smoke
    )
    _entry_sweep(cfg, "anchors")
    # >= 2 draws even under --smoke: the disjoint-half floor F_act needs k >= 2.
    draws = 2 if cfg.smoke else cfg.anchor_draws
    _manifest, bank_sha = bank_manifest_and_sha()
    regime_fp = regime_fingerprint(cfg, bank_sha)
    gate_ids, rest_ids, contexts = _anchor_context_order(cfg)
    my_gate = gate_ids[cfg.worker_index :: cfg.num_workers]
    my_rest = rest_ids[cfg.worker_index :: cfg.num_workers]
    model, tok = (None, None)

    if cfg.force or not _anchor_batch_done(cfg, regime_fp, "gate", draws):
        model, tok = load_model_and_tokenizer(cfg)
        if my_gate:
            res = _run_anchor_batch(cfg, model, tok, contexts, my_gate, draws, "gate", regime_fp)
            # Immediate upload of the gate slice so the VM judge can start.
            _upload_dir(
                cfg,
                cfg.anchors_dir,
                f"{HF_PREFIX}/raw_completions/anchors_gate",
                [res["jsonl"].name],
            )
        else:
            _write_json_atomic(
                cfg.manifest_dir / f"anchors_gate_w{cfg.worker_index}_done.json",
                {
                    "regime_fp": regime_fp,
                    "batch": "gate",
                    "worker_index": cfg.worker_index,
                    "num_workers": cfg.num_workers,  # shard identity (r1 M2)
                    "n_contexts": 0,
                    "draws": draws,
                    "n_rows": 0,
                    "n_cap_hit": 0,
                    "n_empty": 0,
                    "max_new_tokens": cfg.max_new_tokens,
                    "repro": _repro(cfg),
                },
            )
    else:
        logger.info("[anchors:gate] already done for this regime — skipping")

    if cfg.force or not _anchor_batch_done(cfg, regime_fp, "rest", draws):
        if model is None:
            model, tok = load_model_and_tokenizer(cfg)
        if my_rest:
            _run_anchor_batch(cfg, model, tok, contexts, my_rest, draws, "rest", regime_fp)
        else:
            _write_json_atomic(
                cfg.manifest_dir / f"anchors_rest_w{cfg.worker_index}_done.json",
                {
                    "regime_fp": regime_fp,
                    "batch": "rest",
                    "worker_index": cfg.worker_index,
                    "num_workers": cfg.num_workers,  # shard identity (r1 M2)
                    "n_contexts": 0,
                    "draws": draws,
                    "n_rows": 0,
                    "n_cap_hit": 0,
                    "n_empty": 0,
                    "max_new_tokens": cfg.max_new_tokens,
                    "repro": _repro(cfg),
                },
            )
    else:
        logger.info("[anchors:rest] already done for this regime — skipping")
    # Registered cap-hit>2%/cell trigger, MEASURED at phase end (plan
    # registration; the trigger previously had no enforcing code). Partial
    # snapshots (other workers still generating) are labeled partial.
    _emit_cap_hit_snapshot(cfg, "anchors")
    logger.info("[phase=anchors_done] worker=%d", cfg.worker_index)
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
) -> dict:
    """One block: K hooked temp-1.0 draws per pair, the hooked V_a pass, and
    (pools present) the margin TF pass — pipelined on the same GPU.

    ``write_done=False`` (the PILOT leg) suppresses BOTH resume done-files —
    the block done-file AND the margin_blocks twin — so a pilot run on
    production ``blocks[0]`` can never leave a ``regime_fp + "-pilot"`` done
    record that the grid queue's ``block_is_done`` scan RAISES on (r1 C1).

    ``done_extra`` (capregen) is merged into the block done record — the
    durable marker that a block was re-generated at a raised cap."""
    cells = _block_cells(bank, block, pairs_by_id, donor_maps)

    def ids_for(cid: str) -> list[int]:
        if cid not in ctx_ids_cache:
            ctx_ids_cache[cid] = BANK29.context_token_ids_2329(tok, contexts[cid])
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
                max_new_tokens=cfg.max_new_tokens,
                temperature=GRID_TEMPERATURE,
                seed_base=cfg.seed_base,
                render_fn=BANK29.render_context_2329,
                ids_fn=BANK29.context_token_ids_2329,
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
    _enrich_rows_with_capture(rows_out, states, cfg.max_new_tokens)
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
        "max_new_tokens": cfg.max_new_tokens,
        "hooked_batch_gt1_unequal": hooked_gt1_unequal,
        "margin_inline": margin_done,
        "repro": _repro(cfg),
        **(done_extra or {}),
    }
    if write_done:
        _write_json_atomic(block_done_path(cfg.out_root, block), done)
    return done


def phase_grid(cfg: RunConfig) -> int:
    """P3: claim-queue block execution (or the timing pilot under ``--pilot``)."""
    logger.info("[phase=grid] worker=%d/%d smoke=%s", cfg.worker_index, cfg.num_workers, cfg.smoke)
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
        # The pilot times ONE production-shape block through this entrypoint.
        # No claim, and write_done=False — the r1 C1 bug was an unconditional
        # done-file write inside run_block that left blocks/<slug>.done.json
        # carrying regime_fp+"-pilot", killing every grid worker at P3 entry.
        t0 = time.monotonic()
        rec = run_block(
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
            regime_fp + "-pilot",
            pools,
            draws,
            write_done=False,
        )
        ran_wall = time.monotonic() - t0
        ran_rollouts = rec["n_rows"]
        # Plan §12 smoke seam: the pilot's production-shape block MUST have
        # exercised a hooked generate at batch>1 with UNEQUAL prompt lengths
        # (pad-into-GatedDeltaNet-recurrence coverage on Qwen3.5).
        assert rec.get("hooked_batch_gt1_unequal"), (
            "pilot block never ran a hooked batch>1 chunk with unequal prompt "
            "lengths — pad-into-recurrence coverage missing (plan §12)"
        )
        return _enforce_pilot_gate(cfg, totals_all, ran_rollouts, ran_wall)

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


def _enforce_pilot_gate(
    cfg: RunConfig, totals_all: dict, ran_rollouts: int, ran_wall: float
) -> int:
    """Plan §7 gate 4 (AT P2 ENTRY) — a DESIGNED halt on a >3x-plan TOTAL wall.

    The pilot ran ONE production-shape hooked grid block (>=30 pairs x K draws
    + V_a + margin) through THIS entrypoint, so the measured per-rollout wall
    is at the sweep's execution shape (#1415 batch-1 false-fire lesson).
    2329 changes vs the parent: the gate runs BEFORE anchors (P2 entry), so
    the SAME measured per-rollout basis projects EVERY generation phase —
    P2 anchors (1,404 contexts x anchor_draws), P3 grid (totals_all), and
    P4 stage-2 (STAGE2_ROLLOUT_CAP, a conservative upper bound: greedy K=1
    add-mode rollouts are no slower than the pilot's replace-mode temp-1.0
    ones). Per-phase poll fences = 2x each projection; refusal fires on
    projected TOTAL > 3x cfg.planned_wall_h (default the plan §9 generation
    total, PLANNED_GEN_TOTAL_WALL_H).
    """
    assert ran_rollouts > 0, "pilot ran no rollouts"
    per_rollout = ran_wall / ran_rollouts
    width = max(1, cfg.num_workers)
    n_contexts = len(BANK.build_contexts())
    anchors_rollouts = n_contexts * cfg.anchor_draws
    phase_rollouts = {
        "anchors": anchors_rollouts,
        "grid": totals_all["rollouts_total"],
        "stage2": STAGE2_ROLLOUT_CAP,
    }
    phases = {
        name: {
            "rollouts": n,
            "projected_wall_h": per_rollout * n / width / 3600.0,
            "fence_h": PILOT_FENCE_MULT * per_rollout * n / width / 3600.0,
        }
        for name, n in phase_rollouts.items()
    }
    projected_total_h = sum(p["projected_wall_h"] for p in phases.values())
    refuse = projected_total_h > PILOT_REFUSAL_MULT * cfg.planned_wall_h
    report = {
        "criterion": "generation-throughput pilot at P2 entry (plan §7 gate 4)",
        "measured_rollouts": ran_rollouts,
        "measured_wall_s": ran_wall,
        "s_per_rollout": per_rollout,
        "gen_batch": cfg.gen_batch,
        "num_workers": width,
        "phases": phases,
        "basis_note": (
            "stage2 projection uses the ROLLOUT_CAP upper bound at the pilot's "
            "hooked replace-mode per-rollout wall (greedy K=1 add-mode is no "
            "slower); anchors projection uses the same basis for unhooked "
            "temp-1.0 rollouts (conservative: no hook overhead there)"
        ),
        "projected_total_wall_h": projected_total_h,
        "planned_total_wall_h": cfg.planned_wall_h,
        "refusal_threshold_h": PILOT_REFUSAL_MULT * cfg.planned_wall_h,
        "sweep_allowed": not refuse,
        "forced": cfg.force_past_halt_gates,
        "repro": _repro(cfg),
    }
    _write_json_atomic(cfg.out_root / "pilot_gate_report.json", report)
    logger.info(
        "[phase=pilot_done] s_per_rollout=%.3f projected_total_wall_h=%.2f (planned %.2f) "
        "fences_h anchors=%.2f grid=%.2f stage2=%.2f sweep_allowed=%s",
        per_rollout,
        projected_total_h,
        cfg.planned_wall_h,
        phases["anchors"]["fence_h"],
        phases["grid"]["fence_h"],
        phases["stage2"]["fence_h"],
        not refuse,
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
    breach_grain: str = "cell",
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
    measurement).

    ``breach_grain`` selects which registered trigger grain arms
    ``trigger_fired``: ``"cell"`` (this driver's registered per-type-cell
    trigger — the default, byte-compatible with every pre-existing report)
    or ``"cell_slot_arm"`` (the ladder plan §7 G5 grain: per
    (direction x slot x arm) UNIT, keyed by ``Block.key``). The per-unit
    breakdown (``per_unit`` / ``breaching_units`` / ``max_arm_spread``) is
    computed whenever rows carry ``slot`` + ``arm`` regardless of grain —
    the value-side/arm-side ASYMMETRY read (a cap sufficient on average but
    truncating one side of a within-cell contrast is still a
    measurement-validity failure); under ``"cell_slot_arm"`` every covered
    row MUST carry both fields (a grain that cannot be evaluated raises,
    never silently degrades to the coarser one)."""
    if breach_grain not in ("cell", "cell_slot_arm"):
        raise ValueError(f"unknown breach_grain {breach_grain!r}")
    if not shard_paths:
        raise RuntimeError(
            f"cap-hit report ({scope}): no rollout shards found — wrong --out-root, "
            "or the phase has not written any shard yet"
        )
    covered: list[dict] = []
    pending: list[str] = []
    cell_counts: dict[str, list[int]] = {}
    cv_counts: dict[tuple[str, str], list[int]] = {}
    unit_counts: dict[str, list[int]] = {}
    n_rows_without_unit_fields = 0
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
            if "slot" in r and "arm" in r:
                u = unit_counts.setdefault(f"{cell}|{r['slot']}|{r['arm']}", [0, 0])
                u[0] += 1
                u[1] += hit
            else:
                n_rows_without_unit_fields += 1
                if breach_grain == "cell_slot_arm":
                    raise RuntimeError(
                        f"cap-hit report ({scope}): row in {path.name} lacks slot/arm — "
                        "the requested cell_slot_arm breach grain cannot be evaluated "
                        "(never silently degraded to the coarser per-cell grain)"
                    )
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
    # Per-(cell x slot x arm) UNIT breakdown (the ladder plan §7 G5 grain;
    # v177: the realized asymmetry axis — arms are the SIDES of the grid
    # contrast, so unequal truncation across arms within one cell x slot is
    # the measurement-validity failure the aggregate cannot see).
    per_unit: dict[str, dict] | None = None
    breaching_units: list[str] | None = None
    max_arm_spread: dict | None = None
    if unit_counts:
        per_unit = {}
        for unit, (n, h) in sorted(unit_counts.items()):
            pct = 100.0 * h / n
            per_unit[unit] = {
                "n_rows": n,
                "cap_hit_rows": h,
                "cap_hit_pct": pct,
                "breach": pct > threshold_pct,  # STRICT >, same registered rule
            }
        breaching_units = sorted(u for u, d in per_unit.items() if d["breach"])
        by_cell_slot: dict[tuple[str, str], list[float]] = {}
        for unit, d in per_unit.items():
            cell, slot, _arm = unit.rsplit("|", 2)  # cells never contain "|" (block_slug contract)
            by_cell_slot.setdefault((cell, slot), []).append(d["cap_hit_pct"])
        for (cell, slot), pcts in sorted(by_cell_slot.items()):
            if len(pcts) < 2:
                continue
            spread = max(pcts) - min(pcts)
            if max_arm_spread is None or spread > max_arm_spread["spread_pct"]:
                max_arm_spread = {
                    "cell": cell,
                    "slot": slot,
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
    if breach_grain == "cell_slot_arm":
        assert breaching_units is not None  # every covered row carried slot+arm (checked above)
        trigger_fired = bool(breaching_units)
    else:
        trigger_fired = bool(breaching)
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
            "overall + per cell + per (cell, value) + per (cell, slot, arm) unit where "
            f"rows carry slot/arm; trigger_fired = any {breach_grain}-grain entry with "
            "cap_hit_pct STRICTLY > pre_registered_regen_trigger_pct; derived_from_sha256 "
            "= sha256 over newline-joined '<name>:<sha256>' of derived_from_shards"
        ),
        "n_rows": n_rows,
        "cap_hit_rows": hits,
        "cap_hit_frac": hits / n_rows,
        "cap_hit_pct": 100.0 * hits / n_rows,
        "pre_registered_regen_trigger_pct": threshold_pct,
        "breach_grain": breach_grain,
        "trigger_fired": trigger_fired,
        "breaching_cells": breaching,
        "breaching_units": breaching_units,
        "max_new_tokens": max_new_tokens,
        "realized_row_caps": sorted(realized_caps),
        "value_key_fields": sorted(value_fields),
        "per_cell": per_cell,
        "per_cell_value": per_cell_value,
        "max_value_spread": max_spread,
        "per_unit": per_unit,
        "max_arm_spread": max_arm_spread,
        "n_rows_without_unit_fields": n_rows_without_unit_fields,
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
        paths = sorted(cfg.anchors_dir.glob("anchors_*_w*.jsonl"))
        width = _family_realized_width(cfg.manifest_dir, "anchors")
        if width is None:
            return (
                paths,
                None,
                (
                    "anchors family width unresolved (phase incomplete) — "
                    "expected shard set underivable"
                ),
            )
        expected: set[str] = set()
        for batch in ("gate", "rest"):
            for w in range(width):
                rec = json.loads((cfg.manifest_dir / f"anchors_{batch}_w{w}_done.json").read_text())
                if int(rec.get("n_rows", 0)) > 0:
                    expected.add(f"anchors_{batch}_w{w}.jsonl")
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
    inputs: tuple[list[Path], set[str] | None, str | None] | None = None,
    breach_grain: str = "cell",
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
    must never claim ``partial: false``, v11 face 4 / efficiency (d)).

    ``inputs`` injects a caller-derived ``(shard_paths, expected_shards,
    expected_unavailable_reason)`` triple — the seam a LAYOUT FORK (the
    #2329 q35 ladder: ``LadderConfig.rollouts_dir`` -> ``out_root/grid``,
    ladder-own block enumeration) uses so this driver's ``_cap_report_inputs``
    never resolves the WRONG store (v176 root cause: cap_report/capregen were
    unreachable for the ladder grid because this function hard-wired the run
    layout). ``None`` keeps the run driver's own derivation, byte-identical."""
    if postregen and base_cap is None:
        raise ValueError("postregen emit requires base_cap (the BASE attribution cap)")
    paths, expected, why = inputs if inputs is not None else _cap_report_inputs(cfg, scope)
    report = compute_cap_hit_report(
        paths,
        base_cap if postregen else cfg.max_new_tokens,
        scope=scope,
        expected_shards=expected,
        expected_unavailable_reason=why,
        breach_grain=breach_grain,
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
    stamp, or a mixed-cap store — post-regen by construction, since every base
    phase runs at ONE regime-fingerprinted cap); a report missing the
    ``partial`` field entirely (absence is never finality — hand-built files
    fail loud, v11 minor); a PARTIAL report (the registered re-gen evaluates
    per-cell rates on the COMPLETE phase); ``realized_row_caps`` unequal to
    ``[max_new_tokens]`` (reconciler v12 Dispute-1 residual: a wrong-cap
    basis — e.g. a report measured at the wrong --max-new-tokens over a
    per-row-cap-carrying store, or an empty measurement — previously passed
    the len>1 check, FROZE at the basis path, and wedged the campaign at the
    fingerprint raise until the basis file was deleted; the equality refuses
    it BEFORE the freeze); a --max-new-tokens below 2x the
    report's generating cap (codex BLOCKER regen-cap-not-enforced: the
    registered remedy is re-gen at >= 2x the cap — 4096 for the 2048 base,
    plan §-line-105 / CLAUDE.md; a sub-2x cap silently violates the recipe
    AND leaves the long tail truncated, surviving the very bias this remedy
    exists to remove)."""
    if rep.get("scope") != scope:
        raise RuntimeError(f"breach report {path} has scope={rep.get('scope')!r}, need {scope!r}")
    if rep.get("postregen"):
        raise RuntimeError(
            f"breach report {path} is a POST-regen measurement (postregen: true) — it can "
            "never drive a capregen basis (v11 C1: regenerated rows dilute per-cell rates, "
            "silently under-scoping the registered remedy); the frozen pre-regen basis "
            f"lives at {capregen_breach_basis_path(cfg, scope)}"
        )
    caps = rep.get("realized_row_caps") or []
    if len(caps) > 1:
        raise RuntimeError(
            f"breach report {path} measured a MIXED-cap store (realized_row_caps={caps}) — "
            "post-regen by construction, never a capregen basis; an ESCALATED re-gen on an "
            "already-regenerated store runs on a fresh --out-root (the stacking refusals "
            "forbid it here anyway)"
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
    base_cap = int(rep["max_new_tokens"])
    if [int(c) for c in caps] != [base_cap]:
        raise RuntimeError(
            f"breach report {path} declares generating cap max_new_tokens={base_cap} but "
            f"measured realized_row_caps={caps} — a basis whose realized row caps do not "
            f"equal [{base_cap}] was measured at the wrong --max-new-tokens (or over a "
            "store from another regime, or over zero rows) and can never drive a capregen "
            "basis (reconciler v12 Dispute-1 residual: such a basis would FREEZE at "
            f"{capregen_breach_basis_path(cfg, scope)} and wedge the campaign at the "
            "fingerprint raise — a later correct --breach-report is refused by the "
            "byte-match until the basis file is deleted)"
        )
    if cfg.max_new_tokens < 2 * base_cap:
        raise RuntimeError(
            "capregen requires --max-new-tokens >= 2x the report's generating cap "
            f"{base_cap} (registered remedy: {2 * base_cap}, plan §-line-105 / CLAUDE.md "
            f"'re-generate at >= 2x the cap'); got {cfg.max_new_tokens} — a sub-2x re-gen "
            "cap violates the registered recipe and leaves the long tail truncated"
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


def phase_capregen_anchors(cfg: RunConfig) -> int:
    """Cell-restricted anchors re-gen at a raised cap (registered >2% remedy).

    Same sharding as the anchors phase (``order[w::W]`` at the REALIZED
    width, asserted), same generation core (``_generate_anchor_rows``), same
    per-(batch, worker) done records — breaching cells' rows are regenerated
    wholesale and MERGED into the existing per-worker shard + va store, so
    every downstream consumer reads the same files it always did. The done
    record keeps the BASE regime_fp (post-regen re-entries of the standard
    anchors command skip cleanly) and gains a ``capregen`` sub-record; the
    #722 r3 cross-regime hard refusal is preserved via the base-fp check."""
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
    base_fp = regime_fingerprint(replace(cfg, max_new_tokens=base_cap), bank_sha)
    regen_fp = regime_fingerprint(cfg, bank_sha)
    width = _family_realized_width(cfg.manifest_dir, "anchors")
    if width is None:
        raise RuntimeError(
            "anchors family width unresolved — the anchors phase is incomplete; "
            "capregen amends a COMPLETED store only"
        )
    if not cfg.num_workers_explicit or cfg.num_workers != width:
        raise RuntimeError(
            f"anchors capregen must run at the realized width {width} for shard "
            f"alignment (pass --num-workers {width} --worker-index <i>); got "
            f"num_workers={cfg.num_workers} explicit={cfg.num_workers_explicit}"
        )
    gate_ids, rest_ids, contexts = _anchor_context_order(cfg)
    cell_of = {cid: ctx["cell"] for cid, ctx in contexts.items()}
    model, tok, eot = None, None, None
    # ONE batch per invocation (second scope addendum): Phase A = gate (the
    # gate-3 critical path), Phase B = rest (deferrable). The per-(batch,
    # worker) done records already resume the two legs independently.
    selected = gate_ids if cfg.capregen_batch == "gate" else rest_ids
    for batch, order in ((cfg.capregen_batch, selected),):
        done_path = cfg.manifest_dir / f"anchors_{batch}_w{cfg.worker_index}_done.json"
        if not done_path.exists():
            raise RuntimeError(
                f"{done_path} missing — anchors {batch} incomplete for worker "
                f"{cfg.worker_index}; capregen amends a COMPLETED store only"
            )
        done_rec = json.loads(done_path.read_text())
        if done_rec.get("regime_fp") != base_fp:
            raise RuntimeError(
                f"anchors {batch} w{cfg.worker_index} done-record carries "
                f"regime_fp={done_rec.get('regime_fp')!r} but the capregen BASE "
                f"regime_fp={base_fp!r} — refusing to re-gen across regimes "
                "(quarantine or use a fresh --out-root)"
            )
        pending_file = (
            cfg.manifest_dir / f"capregen_pending_anchors_{batch}_w{cfg.worker_index}.jsonl"
        )
        cr = done_rec.get("capregen")
        if cr is not None:
            if (
                int(cr.get("max_new_tokens", -1)) != cfg.max_new_tokens
                or set(cr.get("cells", [])) != breach
            ):
                raise RuntimeError(
                    f"anchors {batch} w{cfg.worker_index} already merged a capregen at "
                    f"cap={cr.get('max_new_tokens')} cells={sorted(cr.get('cells', []))} "
                    f"!= this invocation's cap={cfg.max_new_tokens} "
                    f"cells={sorted(breach)} — refusing to stack re-gens "
                    "(fresh --out-root to redo)"
                )
            logger.info(
                "[capregen:anchors:%s] already merged for this breach list — skipping to "
                "the idempotent upload retry",
                batch,
            )
        else:
            my = order[cfg.worker_index :: cfg.num_workers]
            my_regen = [cid for cid in my if cell_of[cid] in breach]
            capregen_record = {
                "cells": sorted(breach),
                "max_new_tokens": cfg.max_new_tokens,
                "base_max_new_tokens": base_cap,
                "regen_regime_fp": regen_fp,
                "source_report": rep_path.name,
                "source_report_sha256": _sha256_bytes(rep_path.read_bytes()),
                "n_rows_regen": 0,
                "ts": datetime.now(UTC).isoformat(),
            }
            if not my_regen:
                _write_json_atomic(
                    done_path, {**done_rec, "capregen": capregen_record, "repro": _repro(cfg)}
                )
                logger.info(
                    "[capregen:anchors:%s] no breaching contexts in this worker's shard — "
                    "stamped done record",
                    batch,
                )
            else:
                draws = int(done_rec["draws"])  # match the ORIGINAL shard's draws exactly
                if model is None:
                    model, tok = load_model_and_tokenizer(cfg)
                    eot = eot_tail_ids(tok)
                rows, flat_ctx, flat_text = _generate_anchor_rows(
                    cfg, model, tok, contexts, my_regen, draws, batch
                )
                # Rollout text durable BEFORE the capture reduce (#779 two-write
                # pattern); side file so no shard glob / upload pattern matches it.
                _write_jsonl_atomic(pending_file, rows)
                states = capture_answer_states(cfg, model, tok, flat_ctx, flat_text, eot)
                _enrich_rows_with_capture(rows, states, cfg.max_new_tokens)
                capregen_record["n_rows_regen"] = len(rows)
                _merge_anchor_capregen(
                    cfg, batch, base_cap, breach, cell_of, rows, states, done_rec, capregen_record
                )
                logger.info(
                    "[capregen:anchors:%s] merged %d regenerated rows (%d contexts x %d draws)",
                    batch,
                    len(rows),
                    len(my_regen),
                    draws,
                )
        # EVERY path (merged / no-breach / already-merged re-entry) falls
        # through here: the pending-side-file unlink and the uploads sit AFTER
        # the done-record commit point, so a crash between commit and upload is
        # retried on re-entry instead of skipped (v11 minors: orphaned pending
        # file; upload never retried), and a no-breach worker still uploads its
        # (unchanged) shard so the consumer-facing prefixes converge COMPLETE
        # (v11 MAJOR 1).
        pending_file.unlink(missing_ok=True)
        jsonl_name = f"anchors_{batch}_w{cfg.worker_index}.jsonl"
        # v11 MAJOR 1: same-filename overwrites keep BOTH judge-visible
        # prefixes complete AND fresh. raw_completions/anchors currently holds
        # the complete PRE-regen store (P5 upload) — without the overwrite the
        # judge's _resolve_anchors_dir would silently prefer STALE truncated
        # gate rows over the regenerated ones; anchors_gate is the early gate
        # mirror gate-3 stages from before the full prefix exists, refreshed
        # here so a fallback can never score stale rows either.
        _upload_dir(cfg, cfg.anchors_dir, f"{HF_PREFIX}/raw_completions/anchors", [jsonl_name])
        if batch == "gate":
            _upload_dir(
                cfg, cfg.anchors_dir, f"{HF_PREFIX}/raw_completions/anchors_gate", [jsonl_name]
            )
        _upload_dir(
            cfg,
            cfg.anchors_dir,
            f"{HF_PREFIX}/analysis_tensors/anchors",
            [f"va_anchors_{batch}_w{cfg.worker_index}.pt"],
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
    # Post-regen measurement over the mixed-cap store (v11 C1): BASE-cap row
    # attribution + the *_postregen SIBLING path — the frozen driving basis is
    # never touched; units (either batch, any worker) whose merge has not
    # landed keep the report partial (a mid-fleet emit never claims final).
    pending_units = [
        f"anchors_{b}_w{w}"
        for b in ("gate", "rest")
        for w in range(width)
        if json.loads((cfg.manifest_dir / f"anchors_{b}_w{w}_done.json").read_text()).get(
            "capregen"
        )
        is None
    ]
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
            ctx_ids_cache[cid] = BANK29.context_token_ids_2329(tok, contexts[cid])
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
    "scripts/issue2329_dispatch.sh margin && scripts/issue2329_dispatch.sh upload "
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
    gate_path = cfg.bank_dir / "injection_gate_report.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    degen_path = cfg.bank_dir / "degeneracy_report.json"
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
                str(cfg.out_root / "pilot_gate_report.json"),
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
    uploaded["vc_bank"] = _upload_dir(
        cfg, cfg.bank_dir, f"{HF_PREFIX}/analysis_tensors/vc_bank", ["*.pt", "*.json"]
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
    # Out-root TOP-LEVEL residue (#2187): pilot_gate_report.json +
    # best_cells_actsel.json live at the out-root top — persist them so the
    # residue sweep reads clean. Exact names, not "*.json": hub allow_patterns
    # are fnmatch (a bare * crosses "/" and would re-upload every nested json).
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
) -> dict:
    """One stage-2 block: 1 greedy add-mode draw per pair at (arm, layer, dose),
    PLUS the hooked V_a capture (2329 divergence: plan phase_outputs.P4_stage2
    persists stage-2 V_a shards; the parent captured none).

    The delta is the PAIR-DIFFERENCE payload: steered arm delta = V(B) - V(A);
    shuffled arm delta = norm-matched V(B_donor) - V(A) (the B-side reuses the
    stage-1 ``payload_for_arm`` construction wholesale via ``_block_cells``,
    so donor assignment + norm-matching stay bit-identical to stage 1)."""
    base_block = Block(block.cell, block.slot, block.arm, block.pair_ids)
    cells = _block_cells(bank, base_block, pairs_by_id, donor_maps)
    recs = bank["per_context"]
    for c in cells:
        a_state = _slot_state(recs[c["pair"].a], block.slot).unsqueeze(0)  # (1, L, H)
        assert c["payload"].shape == a_state.shape, (c["payload"].shape, a_state.shape)
        c["delta"] = (c["payload"] - a_state).contiguous()

    def ids_for(cid: str) -> list[int]:
        if cid not in ctx_ids_cache:
            ctx_ids_cache[cid] = BANK29.context_token_ids_2329(tok, contexts[cid])
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
                max_new_tokens=cfg.max_new_tokens,
                temperature=STAGE2_TEMPERATURE,
                seed_base=cfg.seed_base,
                render_fn=BANK29.render_context_2329,
                ids_fn=BANK29.context_token_ids_2329,
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
    _enrich_rows_with_capture(rows_out, states, cfg.max_new_tokens)
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
        "max_new_tokens": cfg.max_new_tokens,
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
    """transformers==5.15.0 loads qwen3_5 with 32 resolvable decoder blocks.

    CPU-only + tiny (config download + a from-config shrunken model); run by
    the dispatcher BEFORE any phase. On the VM (repo pin 4.57.6) this FAILS
    by design — it is a POD gate for the pod venv's own pin.
    """
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    from explore_persona_space.analysis.extraction import _resolve_decoder_blocks

    v = str(transformers.__version__)
    assert v == "5.15.0", f"gate 0b: transformers=={v}, need ==5.15.0 (plan §7 gate 0b)"
    mcfg = AutoConfig.from_pretrained(MODEL_ID)
    tc = _model_text_config(mcfg)
    model_types = {str(getattr(mcfg, "model_type", "")), str(getattr(tc, "model_type", ""))}
    assert any("qwen3_5" in m for m in model_types), (
        f"gate 0b: AutoConfig({MODEL_ID}) model_type {model_types} is not qwen3_5"
    )
    assert tc.num_hidden_layers == N_MODEL_LAYERS_FULL, tc.num_hidden_layers
    assert tc.hidden_size == HIDDEN_FULL, tc.hidden_size
    tiny = _shrink_config(AutoConfig.from_pretrained(MODEL_ID), 64, N_MODEL_LAYERS_FULL)
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
    # 2329 constants coherence (thinking-off template + read layer + stage-2
    # geometry): asserted here so a fork-base drift fails at import-check.
    assert BANK29.MODEL_ID == MODEL_ID, (BANK29.MODEL_ID, MODEL_ID)
    assert BANK29.TEMPLATE_KWARGS == {"enable_thinking": False}, BANK29.TEMPLATE_KWARGS
    assert 0 <= F_ACT_READ_LAYER < N_MODEL_LAYERS_FULL
    assert len(STAGE2_LAYERS) == 7 and STAGE2_DOSES == (1, 4)
    assert STAGE2_ARMS == ("steered", "shuffled")
    assert STAGE2_DRAWS == 1 and STAGE2_TEMPERATURE == 0.0
    assert all(0 <= layer < N_MODEL_LAYERS_FULL for layer in STAGE2_LAYERS), STAGE2_LAYERS
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
