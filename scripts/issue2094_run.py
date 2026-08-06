#!/usr/bin/env python3
"""Issue #2094 — pod driver for the single-position context/prefix intervention grid.

Phases (plan v4 §4.6 DAG; this driver owns P1/P2/P3+P4/P5):

- ``--phase bank`` (P1): build the 15-context bank, capture the all-layer V bank
  (prefix-end + the whole final-user-turn span, from which context-end / T-2 /
  T-3 / last-3-joint / query-span slots are sliced), derive the Type-B prefix
  centroids, and run the INJECTION-EXACTNESS GATE (plan §7 gate 1: 12 spot cells
  re-forwarded with the hook armed; the realized edit must equal ``alpha*Delta``
  at the intended (row, position, layer) and NOWHERE else — cosine >= 0.999,
  norm ratio in [0.995, 1.005]). A gate failure writes
  ``injection_gate_report.json`` and exits ``RC_INJECTION_GATE`` (a DESIGNED
  halt, never a bare rc=1 — the #1415 pilot-gate routing lesson).
- ``--phase anchors`` (P2): 15 contexts x K=10 unpatched rollouts at temp 1.0
  (the F floor under A; the same draws serve as the ceiling under B by context
  identity), plus answer-state capture in BOTH poolings from ONE forward
  (span-mean over the completion tokens — the #1415 ``capture_vectors``
  convention used by F_act — and the tail-inclusive ``v_x`` variant that carries
  the assistant end-of-turn tail, for parity with the banked-map lineage).
- ``--phase grid`` (P3+P4, sharded): the 880 independent blocks (block key
  ``slot|layer-variant|dose|vec-type|arm``) round-robin across workers as
  (steered, null) FAMILIES so each family's two arms land adjacently on the same
  worker. Per block: one greedy rollout per cell, then the V_a capture pass
  (both poolings, one forward per chunk). Per-block JSONL rollout text + ``.pt``
  state shards + a done-file; resume skips completed blocks keyed on the FULL
  regime key. ``--pilot`` runs ONE production-shape block family (B=16, both
  arms) timed through this same entrypoint and writes
  ``pilot_gate_report.json`` + ``RC_PILOT_GATE`` when the projected pod wall
  exceeds 3x the plan §9 figure.
- ``--phase upload`` (P5): ONE bulk ``upload_folder`` commit per HF prefix (never
  a per-file loop — the #664/#727 gotcha), an exact-set verify, then the pod
  sentinel ``/workspace/logs/issue-2094-results.json`` carrying the Step 7
  results payload.

Pod-side contract: sentinel file + ``[phase=...]`` breadcrumbs ONLY — this file
NEVER shells out to ``scripts/task.py`` (the pod runs an ``issue-<N>`` branch and
``task.py`` branch-guards to ``main``).

Every phase ends with an explicit ``sys.exit`` so the heavy-C-extension
interpreter-finalization race can never rewrite a completed phase's rc (#1689).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch import (shared-VM thread caps + API keys)

import torch  # noqa: E402

from explore_persona_space.analysis.extraction import (  # noqa: E402
    extract_layer_activations,
)
from explore_persona_space.experiments.issue1415.steering import (  # noqa: E402
    capture_vectors,
    generate_batch,
)
from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402
from explore_persona_space.experiments.issue2094.fmetrics import safe_cosine  # noqa: E402
from explore_persona_space.experiments.issue2094.hooks import (  # noqa: E402
    PositionEditHook,
    joint_hooks,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("issue2094.run")

# ── constants (plan v4 §4.2/§4.3/§9/§10) ──────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HIDDEN_FULL = 3584
N_MODEL_LAYERS_FULL = 28
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue2094_singlepos"
DEFAULT_OUT_ROOT = Path("/workspace/issue2094_out")
DEFAULT_LOG_DIR = Path("/workspace/logs")
SENTINEL_NAME = "issue-2094-results.json"

MAX_NEW_TOKENS = 1024
SEED_BASE = 42
ANCHOR_DRAWS = 10
ANCHOR_TEMPERATURE = 1.0
GRID_TEMPERATURE = 0.0  # greedy grid (plan §4.3)

SLOTS_FULL_SWEEP: tuple[str, ...] = ("ce", "pe")
SLOTS_CONTROL: tuple[str, ...] = ("cm2", "cm3", "l3j", "qspan")
SLOTS: tuple[str, ...] = SLOTS_FULL_SWEEP + SLOTS_CONTROL
MULTI_POSITION_SLOTS: frozenset[str] = frozenset({"l3j", "qspan"})

DOSES_A: tuple[str, ...] = ("a0.5", "a1", "a2", "a4", "replace")
DOSES_B: tuple[str, ...] = ("a0.5", "a1", "a2", "a4")
ARMS: tuple[str, ...] = ("steered", "null")
VEC_TYPES: tuple[str, ...] = ("A", "B")

# Injection-exactness gate bars (plan §7 gate 1; grounded on #1415's realized
# >=0.99997 / 0.9996-1.0005 — the bar sits >=4x the measured bf16 jitter).
GATE_COS_MIN = 0.999
GATE_NORM_RATIO_LO = 0.995
GATE_NORM_RATIO_HI = 1.005
# Off-target leg: relative ||diff|| / ||baseline|| at every NON-edited position
# of the edit layer (and at every layer BELOW the shallowest edit layer).
GATE_OFFTARGET_REL_MAX = 1e-3
# Type-B == Type-A at prefix-end (query-independent prefix states, plan §4.2);
# span-mean-class parity bar per the gotchas two-bar rule.
GATE_TYPEB_PREFIX_COS_MIN = 0.9999
# Reuse-seam parity: our bank's v_ce / v_pe vs steering.capture_vectors'
# v_c_context / v_c_prefix on the single-turn contexts (bare + persona).
# Recalibrated 2026-08-06: production 8xH100 bf16 cross-path parity measured
# cos_min 0.99975 (injection_gate_report.json; jitter ~2.5e-4 between the two
# capture code paths). Bar per the two-bar rule (>=4x measured jitter),
# matching GATE_COS_MIN's 0.999 convention; the prior 0.9999 sat 10x tighter
# than the plan's own injection-bar grounding and failed on healthy parity.
GATE_CAPTURE_PARITY_COS_MIN = 0.999

# Plan §9: P3 (2.4 h) + P4 (0.6 h) = 3.0 h projected pod wall at width 8.
PLANNED_GRID_WALL_H = 3.0
PILOT_REFUSAL_MULT = 3.0  # projected pod wall > 3x plan  => refuse (plan §7 gate 2)
PILOT_FENCE_MULT = 2.0  # poll fence = 2x the pilot-extrapolated wall

# Distinct rcs: a designed halt is never an anonymous rc=1.
RC_OK = 0
RC_INJECTION_GATE = 21
RC_PILOT_GATE = 22

_QSPAN_MIN_POSITIONS = 3  # cm3 / l3j need >= 3 final-user-turn positions


# ── pure helpers (CPU-only, unit-tested in tests/test_issue2094_run.py) ──


def layer_variant_names(n_layers: int) -> tuple[str, ...]:
    """The 30 layer variants: every single layer + ``joint_mid`` + ``joint_all``."""
    assert n_layers >= 1, n_layers
    return (*(f"L{i}" for i in range(n_layers)), "joint_mid", "joint_all")


def joint_mid_layers(n_layers: int) -> tuple[int, ...]:
    """Production joint-middle band = layers 14..20 (plan §4.2).

    On a tiny smoke model (< 21 layers) the production band does not exist;
    degrade to the single middle layer so the joint-layer CODE PATH stays
    exercised (a smoke-only degradation, recorded in the manifest regime).
    """
    assert n_layers >= 1, n_layers
    if n_layers >= 21:
        return tuple(range(14, 21))
    return (n_layers // 2,)


def layer_variant_layers(variant: str, n_layers: int) -> tuple[int, ...]:
    """Model-layer indices for one layer variant."""
    if variant == "joint_mid":
        return joint_mid_layers(n_layers)
    if variant == "joint_all":
        return tuple(range(n_layers))
    assert variant.startswith("L"), variant
    idx = int(variant[1:])
    assert 0 <= idx < n_layers, (variant, n_layers)
    return (idx,)


def dose_spec(dose: str) -> tuple[str, float]:
    """``dose`` token -> ``(mode, alpha)``. ``replace`` is the full-state patch."""
    if dose == "replace":
        return ("replace", 1.0)
    assert dose.startswith("a"), dose
    return ("add", float(dose[1:]))


def slot_layer_variants(slot: str, n_layers: int) -> tuple[str, ...]:
    """Full-sweep slots get all 30 variants; control slots joint-middle only."""
    assert slot in SLOTS, slot
    if slot in SLOTS_FULL_SWEEP:
        return layer_variant_names(n_layers)
    return ("joint_mid",)


@dataclass(frozen=True)
class Block:
    """One independently-schedulable unit of the grid.

    A block is ONE (slot, layer-variant, dose, vec-type, arm) over a set of
    pairs — every cell in it shares the intervention geometry, so one hooked
    batched ``generate_batch`` + one capture pass covers the whole block.
    """

    slot: str
    layer_variant: str
    dose: str
    vec_type: str
    arm: str
    pair_ids: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.slot}|{self.layer_variant}|{self.dose}|{self.vec_type}|{self.arm}"

    @property
    def slug(self) -> str:
        return block_slug(self.key)

    @property
    def n_cells(self) -> int:
        return len(self.pair_ids)


def block_slug(key: str) -> str:
    """Filesystem-safe block slug (``|`` -> ``__``, ``.`` -> ``p``)."""
    return key.replace("|", "__").replace(".", "p")


def enumerate_block_families(pairs: list[BANK.Pair], n_layers: int) -> list[tuple[Block, Block]]:
    """The 440 (steered, null) block families = 880 blocks (plan §4.3).

    Type A full-sweep: 2 slots x 30 variants x 5 doses x 60 pairs -> 18,000 cells.
    Type A controls:   4 slots x  1 variant x 5 doses x 60 pairs ->  1,200 cells.
    Type B:            1 slot  x 30 variants x 4 doses x 15 mq pairs -> 1,800 cells.
    The shuffled-donor null mirrors every steered cell (21,000 + 21,000 = 42,000).
    """
    all_ids = tuple(p.pair_id for p in pairs)
    mq_ids = tuple(p.pair_id for p in pairs if p.setting == "matched_query")
    assert all_ids, "empty pair bank"
    assert mq_ids, "no matched-query pairs (Type B needs them)"

    families: list[tuple[Block, Block]] = []

    def add(slot: str, variant: str, dose: str, vec_type: str, ids: tuple[str, ...]) -> None:
        families.append(
            (
                Block(slot, variant, dose, vec_type, "steered", ids),
                Block(slot, variant, dose, vec_type, "null", ids),
            )
        )

    for slot in SLOTS_FULL_SWEEP:
        for variant in slot_layer_variants(slot, n_layers):
            for dose in DOSES_A:
                add(slot, variant, dose, "A", all_ids)
    for slot in SLOTS_CONTROL:
        for variant in slot_layer_variants(slot, n_layers):
            for dose in DOSES_A:
                add(slot, variant, dose, "A", all_ids)
    for variant in slot_layer_variants("ce", n_layers):
        for dose in DOSES_B:
            add("ce", variant, dose, "B", mq_ids)

    keys = [b.key for fam in families for b in fam]
    assert len(set(keys)) == len(keys), "duplicate block keys"
    return families


def smoke_block_families(pairs: list[BANK.Pair], n_layers: int) -> list[tuple[Block, Block]]:
    """A tiny per-ARM-CLASS slice: >=1 family per class-defining axis.

    Classes covered (each with BOTH arms, so every donor-null path runs too):
    single-layer add on a full-sweep slot, replace mode on BOTH full-sweep
    slots (the pe x replace family exercises the state-kind donor walk AND the
    matched-prefix degenerate ``self:`` carve-out — the round-2/3
    pe-replace-null-walk-exhaustion arm class), joint-middle multi-layer,
    joint-all, Type B, a single-position control slot, a multi-position control
    slot (l3j), and the query span (+ a Type-A twin on the Type-B cell). 13
    families = 26 blocks. The pair subset
    ALWAYS includes a conv-``context_a`` pair (the multi-turn history render is
    otherwise smoke-invisible — the unit-E render-seam requirement), and the
    ``ce``/mid slot carries FOUR additive doses so the downstream P7 linearity
    family (n_obs = 4 pairs x 4 doses) clears the PC-ridge ``n_train > d_eff``
    floor on the tiny model (hidden 8) and the homogeneity read has a full
    dose-cosine matrix.
    """
    variants = layer_variant_names(n_layers)
    mid = variants[n_layers // 2]  # a single-layer variant that exists on any model
    last = variants[n_layers - 1]
    a_ids: list[str] = []
    for setting in ("matched_prefix", "matched_query", "cross"):
        first = next((p.pair_id for p in pairs if p.setting == setting), None)
        if first is not None:
            a_ids.append(first)
    conv_a = next((p.pair_id for p in pairs if p.a.startswith("conv")), None)
    if conv_a is not None and conv_a not in a_ids:
        a_ids.append(conv_a)  # conv-context_a arm class (render seam, unit E)
    a_subset = tuple(a_ids)
    mq_subset = tuple(p.pair_id for p in pairs if p.setting == "matched_query")[:1]
    assert a_subset and mq_subset, (a_subset, mq_subset)
    assert any(pid.split("--")[1].startswith("conv") for pid in a_subset), a_subset

    spec: list[tuple[str, str, str, str, tuple[str, ...]]] = [
        ("ce", mid, "a0.5", "A", a_subset),
        ("ce", mid, "a1", "A", a_subset),
        ("ce", mid, "a2", "A", a_subset),
        ("ce", mid, "a4", "A", a_subset),
        ("ce", mid, "replace", "A", a_subset),
        ("pe", last, "a2", "A", a_subset),
        ("pe", last, "replace", "A", a_subset),
        ("pe", "joint_mid", "a1", "A", a_subset),
        ("cm2", "joint_mid", "replace", "A", a_subset),
        ("l3j", "joint_mid", "a0.5", "A", a_subset),
        ("qspan", "joint_mid", "a1", "A", a_subset),
        ("ce", "joint_all", "a1", "B", mq_subset),
        # Type-A twin on the SAME cell as the Type-B family: production shares
        # every Type-B cell with Type A (A covers all 60 pairs), so the smoke
        # needs >=1 shared (slot, variant, dose, pair) cell for the A-vs-B
        # exploratory read to be non-vacuous.
        ("ce", "joint_all", "a1", "A", mq_subset),
    ]
    return [
        (
            Block(slot, variant, dose, vt, "steered", ids),
            Block(slot, variant, dose, vt, "null", ids),
        )
        for slot, variant, dose, vt, ids in spec
    ]


def assign_families(
    families: list[tuple[Block, Block]], worker_index: int, num_workers: int
) -> list[tuple[Block, Block]]:
    """Round-robin family sharding — both arms of a family stay on one worker."""
    assert num_workers >= 1, num_workers
    assert 0 <= worker_index < num_workers, (worker_index, num_workers)
    return families[worker_index::num_workers]


def blocks_for_worker(
    families: list[tuple[Block, Block]], worker_index: int, num_workers: int
) -> list[Block]:
    """Flatten the worker's families into interleaved (steered, null) blocks."""
    out: list[Block] = []
    for steered, null in assign_families(families, worker_index, num_workers):
        out.append(steered)
        out.append(null)
    return out


def grid_totals(families: list[tuple[Block, Block]]) -> dict[str, int]:
    """Block/cell counts for the manifest + the return-side reconciliation."""
    blocks = [b for fam in families for b in fam]
    steered = sum(b.n_cells for b in blocks if b.arm == "steered")
    null = sum(b.n_cells for b in blocks if b.arm == "null")
    return {
        "n_families": len(families),
        "n_blocks": len(blocks),
        "cells_steered": steered,
        "cells_null": null,
        "cells_total": steered + null,
    }


def block_done_path(out_root: Path, block: Block) -> Path:
    return out_root / "manifests" / "blocks" / f"{block.slug}.done.json"


def block_is_done(out_root: Path, block: Block, regime_fp: str) -> bool:
    """Resume predicate: done-file present AND its regime fingerprint matches.

    A regime mismatch is a HARD refusal, never a silent reuse of wrong cached
    rows (#722 r3) — the caller sees the raise.
    """
    path = block_done_path(out_root, block)
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


def bank_manifest_and_sha() -> tuple[dict, str]:
    """Deterministic bank manifest + its sha (the regime key), CPU-only.

    Computable pre-model-load — the same recipe ``phase_bank`` persists into
    ``bank.json`` (the sha is over the manifest BEFORE the ``bank_sha`` /
    ``repro`` fields are added), so the bank / anchors resume predicates and
    the grid's ``regime_fingerprint`` all key on the identical value.
    """
    manifest = BANK.bank_manifest()
    bank_bytes = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    return manifest, _sha256_bytes(bank_bytes)


def _phase_done_record(cfg: RunConfig, phase: str, regime_fp: str) -> dict | None:
    """Shared regime-checked done-record read for the bank/anchors predicates.

    Missing done-manifest -> ``None`` (run the phase). A regime-fingerprint
    mismatch is a HARD refusal, never a silent cross-regime reuse (#722 r3 —
    the ``block_is_done`` convention).
    """
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
    """P1 resume predicate (round-2 Critical 2, the #1689 spend-leak class):
    done-manifest present + regime match + every output artifact on disk."""
    rec = _phase_done_record(cfg, "bank", regime_fp)
    if rec is None:
        return False
    required = [
        cfg.bank_dir / "bank.json",
        cfg.bank_dir / "vc_bank.pt",
        cfg.bank_dir / "injection_gate_report.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        logger.warning(
            "[bank] done-manifest present but artifacts missing %s — re-running", missing
        )
        return False
    return True


def anchors_is_done(cfg: RunConfig, regime_fp: str, expected_draws: int) -> bool:
    """P2 resume predicate: done-manifest + regime match + artifacts present +
    the recorded draw count matches this invocation + the realized anchors.jsonl
    row count matches the done record (output-manifest presence + row-count)."""
    rec = _phase_done_record(cfg, "anchors", regime_fp)
    if rec is None:
        return False
    if int(rec.get("draws", -1)) != expected_draws:
        logger.warning(
            "[anchors] done-manifest draws=%s but this run wants %d — re-running",
            rec.get("draws"),
            expected_draws,
        )
        return False
    jsonl = cfg.anchors_dir / "anchors.jsonl"
    va = cfg.anchors_dir / "va_anchors.pt"
    if not (jsonl.exists() and va.exists()):
        logger.warning("[anchors] done-manifest present but artifacts missing — re-running")
        return False
    n_rows = sum(1 for line in jsonl.open(encoding="utf-8") if line.strip())
    if n_rows != int(rec.get("n_rows", -1)):
        logger.warning(
            "[anchors] done-manifest n_rows=%s but anchors.jsonl has %d rows — re-running",
            rec.get("n_rows"),
            n_rows,
        )
        return False
    return True


def slot_positions(ctx_len: int, prefix_end: int, slot: str) -> tuple[int, ...]:
    """Edit positions (UNPADDED context coordinates) for one slot.

    ``ctx_len`` is the rendered context length (generation prompt included) and
    ``prefix_end`` the index where the FINAL user turn starts, so the
    final-user-turn span is ``[prefix_end, ctx_len)`` (``nq`` positions) and
    ``ce`` is its last position — the plan §4.2 right-aligned convention.
    """
    assert slot in SLOTS, slot
    nq = ctx_len - prefix_end
    assert nq >= _QSPAN_MIN_POSITIONS, (ctx_len, prefix_end, nq)
    if slot == "ce":
        return (ctx_len - 1,)
    if slot == "pe":
        return (prefix_end - 1,)
    if slot == "cm2":
        return (ctx_len - 2,)
    if slot == "cm3":
        return (ctx_len - 3,)
    if slot == "l3j":
        return (ctx_len - 3, ctx_len - 2, ctx_len - 1)
    return tuple(range(prefix_end, ctx_len))  # qspan


def align_right(vecs: torch.Tensor, m: int) -> torch.Tensor:
    """Right-align a ``(P, ...)`` per-position stack to exactly ``m`` rows.

    ``P >= m`` truncates to the LAST ``m`` (the plan §4.2 right-aligned
    min-overlap convention); ``P < m`` cyclically tiles from the right so a
    shorter DONOR span can still norm-match a longer recipient position-wise
    (recorded per null cell).
    """
    assert m >= 1, m
    p = vecs.shape[0]
    assert p >= 1, vecs.shape
    if p >= m:
        return vecs[-m:]
    reps = -(-m // p)
    return vecs.repeat(reps, *([1] * (vecs.dim() - 1)))[-m:]


def regime_fingerprint(cfg: RunConfig, bank_sha: str) -> str:
    """Stable fingerprint of EVERY output-affecting knob (resume key)."""
    import hashlib

    payload = json.dumps(
        {
            "bank_sha": bank_sha,
            "model_id": cfg.model_id,
            "tiny": cfg.tiny,
            "n_layers": cfg.n_layers,
            "hidden": cfg.hidden,
            "max_new_tokens": cfg.max_new_tokens,
            "grid_temperature": GRID_TEMPERATURE,
            "seed_base": cfg.seed_base,
            "smoke": cfg.smoke,
            "bank_seed": BANK.SEED,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cap_hit(n_completion_tokens: int, max_new_tokens: int) -> bool:
    """Cap-hit telemetry from the re-tokenized completion length.

    ``generate_batch`` returns decoded TEXT, so the exact new-token count is not
    available; the completion ids we tokenize for the teacher-forced capture are
    the operative proxy (recorded as ``cap_hit_basis`` per row).
    """
    return n_completion_tokens >= max_new_tokens


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
    seed_base: int
    smoke: bool
    pilot: bool
    force: bool
    worker_index: int
    num_workers: int
    upload_mode: str  # "hf" | "local-mirror" | "none"
    upload_every: int
    planned_wall_h: float
    gpu_hours_budgeted: float

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
    def manifest_dir(self) -> Path:
        return self.out_root / "manifests"

    @property
    def layers(self) -> list[int]:
        return list(range(self.n_layers))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #2094 pod driver (bank / anchors / grid / upload)."
    )
    ap.add_argument(
        "--phase",
        choices=("bank", "anchors", "grid", "upload"),
        help="pipeline phase to run (required unless --import-check)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import (incl. function-body imports) and exit 0",
    )
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
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--smoke", action="store_true", help="tiny per-arm-class slice")
    ap.add_argument("--pilot", action="store_true", help="grid: timing pilot only")
    ap.add_argument(
        "--force",
        action="store_true",
        help="override gate refusals (pilot / injection); on --phase bank|anchors, "
        "deliberately re-run a completed phase (both are skip-if-done by default)",
    )
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=None, help="informational; CVD pins the device")
    ap.add_argument("--upload", choices=("hf", "local-mirror", "none"), default="hf")
    ap.add_argument(
        "--upload-every",
        type=int,
        default=25,
        help="grid: bulk-upload the worker's staged text every N blocks (256 commits/hr cap)",
    )
    ap.add_argument("--planned-wall-h", type=float, default=PLANNED_GRID_WALL_H)
    ap.add_argument("--gpu-hours-budgeted", type=float, default=33.0)
    # NOTE: the plan §4.4 additivity spot-check ("optional", ~12 extra rollouts at
    # ce/L14/alpha=1) is NOT wired here — it is a separate small block family, left
    # to the analysis unit rather than shipped as an unwired flag.
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
        seed_base=args.seed_base,
        smoke=args.smoke,
        pilot=args.pilot,
        force=args.force,
        worker_index=args.worker_index,
        num_workers=args.num_workers,
        upload_mode=args.upload,
        upload_every=args.upload_every,
        planned_wall_h=args.planned_wall_h,
        gpu_hours_budgeted=args.gpu_hours_budgeted,
    )


# ── io helpers ────────────────────────────────────────────────────────


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    os.replace(tmp, path)


def _save_pt_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


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
        "tiny": cfg.tiny,
        "smoke": cfg.smoke,
        "n_layers": cfg.n_layers,
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ── model ─────────────────────────────────────────────────────────────


def load_model_and_tokenizer(cfg: RunConfig):
    """Production: bf16 Qwen-2.5-7B-Instruct pinned to one device (never
    ``device_map='auto'`` — silent CPU offload, gotchas). Tiny: a from-config
    same-arch model on CPU over the REAL vocab-id space."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id)  # loaded ONCE (HF-429 gotcha)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if cfg.tiny:
        mcfg = AutoConfig.from_pretrained(cfg.model_id)
        mcfg.hidden_size = cfg.hidden
        mcfg.intermediate_size = 2 * cfg.hidden
        mcfg.num_hidden_layers = cfg.n_layers
        mcfg.num_attention_heads = 4
        mcfg.num_key_value_heads = 2
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(mcfg).to(torch.float32)
    else:
        assert torch.cuda.is_available(), "the full grid requires CUDA (use --tiny for CPU smoke)"
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.bfloat16)
    model = model.to(cfg.device)
    assert model.config.hidden_size == cfg.hidden, (model.config.hidden_size, cfg.hidden)
    assert model.config.num_hidden_layers == cfg.n_layers, (
        model.config.num_hidden_layers,
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
    """The assistant end-of-turn tail (``<|im_end|>`` + newline) as TOKEN IDS.

    Built from ids, never by re-tokenizing a concatenated string (the BPE-seam
    rule) — it is the tail the tail-inclusive ``v_x`` pooling includes.
    """
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert isinstance(im_end, int) and im_end >= 0, im_end
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    assert nl, "tokenizer produced no ids for a newline"
    return [im_end, *nl]


# ── P1: V bank ────────────────────────────────────────────────────────


@torch.no_grad()
def capture_bank(cfg: RunConfig, model, tok) -> dict:
    """All-layer V bank: prefix-end state + the whole final-user-turn span.

    Every slot the grid edits is a slice of these two reads (plan §4.2):
    ``ce`` = span[-1], ``cm2`` = span[-2], ``cm3`` = span[-3], ``l3j`` =
    span[-3:], ``qspan`` = the whole span, ``pe`` = the prefix-end state. One
    forward per chunk of right-padded contexts; positions come from the
    concatenated token ids' own offsets, never from a re-tokenized string.
    """
    contexts = BANK.build_contexts()
    ctx_ids = {cid: BANK.context_token_ids_2094(tok, c) for cid, c in contexts.items()}
    prefix_ends = {cid: BANK.prefix_end_index_multi(tok, ids) for cid, ids in ctx_ids.items()}
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
            nq = ctx_len - pe
            assert nq >= _QSPAN_MIN_POSITIONS, (cid, ctx_len, pe, nq)
            span = torch.stack(
                [captured[layer][j, pe:ctx_len] for layer in layers], dim=1
            )  # (nq, L, H)
            v_pe = torch.stack([captured[layer][j, pe - 1] for layer in layers])  # (L, H)
            assert span.shape == (nq, len(layers), cfg.hidden), span.shape
            records[cid] = {
                "context_id": cid,
                "prefix": contexts[cid]["prefix"],
                "query_id": contexts[cid]["query_id"],
                "ctx_len": ctx_len,
                "prefix_end": pe,
                "nq": nq,
                "q_span": span.float().cpu(),
                "v_pe": v_pe.float().cpu(),
            }
        del captured
    assert len(records) == len(contexts), (len(records), len(contexts))
    centroids = _prefix_centroids(records, cfg)
    return {"layers": layers, "per_context": records, "centroids": centroids}


def _slot_vectors(rec: dict, slot: str) -> torch.Tensor:
    """``(P, L, H)`` slot states for one context (P = positions for the slot)."""
    span, v_pe = rec["q_span"], rec["v_pe"]
    if slot == "pe":
        return v_pe.unsqueeze(0)
    if slot == "ce":
        return span[-1:]
    if slot == "cm2":
        return span[-2:-1]
    if slot == "cm3":
        return span[-3:-2]
    if slot == "l3j":
        return span[-3:]
    assert slot == "qspan", slot
    return span


def _prefix_centroids(records: dict[str, dict], cfg: RunConfig) -> dict[str, torch.Tensor]:
    """Type-B centroids at the context-end slot: mean over queries of (P - bare)."""
    out: dict[str, torch.Tensor] = {}
    for prefix in BANK.PREFIX_ORDER:
        v = {
            cid: _slot_vectors(rec, "ce")[0]
            for cid, rec in records.items()
            if rec["prefix"] in (prefix, "bare")
        }
        out[prefix] = BANK.prefix_centroid(v, prefix)
        assert out[prefix].shape == (cfg.n_layers, cfg.hidden), out[prefix].shape
    return out


def _pair_payload(
    bank: dict,
    pair: BANK.Pair,
    slot: str,
    vec_type: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """``(delta, replacement_state, m)`` at the pair's slot, ALL layers.

    ``delta`` = V_slot(B) - V_slot(A) (Type A) or centroid_B - centroid_A
    (Type B); ``replacement_state`` = V_slot(B) (the full-state patch payload).
    Both are ``(m, L, H)`` right-aligned over ``m = min(P_A, P_B)``.
    """
    recs = bank["per_context"]
    if vec_type == "B":
        assert slot == "ce", f"Type B runs at context-end only (got {slot})"
        cent = bank["centroids"]
        delta = (cent[pair.prefix_b] - cent[pair.prefix_a]).unsqueeze(0)
        return delta, cent[pair.prefix_b].unsqueeze(0), 1
    va = _slot_vectors(recs[pair.a], slot)
    vb = _slot_vectors(recs[pair.b], slot)
    m = min(va.shape[0], vb.shape[0])
    va, vb = align_right(va, m), align_right(vb, m)
    delta = vb - va
    if slot == "pe" and pair.prefix_a == pair.prefix_b:
        # CAUSAL IDENTITY: same prefix tokens => identical prefix-end state, so
        # the true Delta is EXACTLY zero; the float compute leaves batch-geometry
        # noise (~1e-9 fp32 CPU, larger in bf16 batches), which downstream
        # norm-matching would otherwise inflate to a fake "direction" (unit-F
        # e2e smoke: an mp-conv pe recipient at 1.9e-9 vs an exactly-zero mp
        # donor tripped norm_match). Canonicalize to the identity.
        delta = torch.zeros_like(delta)
    return delta, vb, m


def _donor_payload(
    bank: dict,
    pair: BANK.Pair,
    donor: BANK.Pair,
    slot: str,
    vec_type: str,
    recipient: torch.Tensor,
    payload_kind: str = "delta",
) -> tuple[torch.Tensor, str]:
    """Norm-matched shuffled-donor payload for the null arm (plan §4.2).

    Type A additive (``payload_kind="delta"``): the seeded derangement's donor
    pair, its Delta right-aligned to the recipient's position count, then
    norm-matched POSITION-WISE. Type A single-position replace
    (``payload_kind="state"``): the donor pair's TARGET-CONTEXT STATE
    ``V_B(donor)`` at the same slot, norm-matched to the recipient's ``V_B``
    norm — a REAL state (wrong pair), in-distribution and parallel to the
    steered arm's real-state replace. Plan §4.2's Δ-centric null wording is
    incoherent at the replace rung (a Δ-normed replacement would near-zero the
    slot), so the donor-STATE realization is the registered resolution
    (round-2 code review; concern ``replace-null-donor-realization``; recorded
    as a plan deviation in the sentinel payload). Type B: the
    persona<->conv-SWAPPED centroid direction (the body's "other prefixes'
    centroids for Type B"; the pool has exactly ONE non-self member) — never a
    state (``DOSES_B`` excludes replace).
    """
    assert payload_kind in ("delta", "state"), payload_kind
    if vec_type == "B":
        assert payload_kind == "delta", "Type B has no absolute state (DOSES_B excludes replace)"
        cent = bank["centroids"]
        # Type-B donors are the centroid swap, not a derangement over contexts.
        donor_a = BANK._TYPE_B_DONOR_SWAP[pair.prefix_a]
        donor_b = BANK._TYPE_B_DONOR_SWAP[pair.prefix_b]
        raw = (cent[donor_b] - cent[donor_a]).unsqueeze(0)
        return BANK.norm_match(raw, recipient), f"centroid:{donor_a}->{donor_b}"
    raw = _aligned_donor_raw(bank, donor, slot, vec_type, recipient, payload_kind)
    return BANK.norm_match(raw, recipient), donor.pair_id


def _aligned_donor_raw(
    bank: dict,
    donor: BANK.Pair,
    slot: str,
    vec_type: str,
    recipient: torch.Tensor,
    payload_kind: str,
) -> torch.Tensor:
    """RAW Type-A donor payload aligned to the recipient's position count.

    Exactly ``_donor_payload``'s pre-``norm_match`` computation, factored out
    so the walk's payload-degeneracy skip (:func:`_payload_degenerate`) and
    the returned payload share ONE computation per candidate donor.
    """
    assert vec_type == "A", vec_type
    assert payload_kind in ("delta", "state"), payload_kind
    raw_delta, raw_state, _ = _pair_payload(bank, donor, slot, vec_type)
    raw = raw_state if payload_kind == "state" else raw_delta
    return align_right(raw, recipient.shape[0])


def _payload_degenerate(raw: torch.Tensor, recipient: torch.Tensor) -> bool:
    """True when the ALIGNED donor payload is exactly zero at any position
    where the recipient is nonzero — nothing to rescale there, so
    ``norm_match`` would fail loud (its zero-donor guard, bank.py).

    Mechanism (production crash, qspan x A x null): an EQUAL-query-length
    matched-prefix donor pair shares its leading query-span TOKENS (the chat
    template's user-turn header + any shared query opening), so by causal
    identity its per-position Delta is EXACTLY zero at those leading
    positions at every layer. A recipient pair of UNEQUAL query lengths is
    aligned at different absolute offsets (RoPE), hence nonzero there. Only
    the qspan slot is exposed — it contains LEADING span positions; the
    ce/cm2/cm3/l3j slots read trailing positions past the query divergence
    point. This is the payload-level twin of the structural pe same-prefix
    exclusion in :func:`_donor_eligible`.
    """
    return bool(((raw.norm(dim=-1) == 0) & (recipient.norm(dim=-1) > 0)).any())


def _donor_eligible(
    donor: BANK.Pair,
    slot: str,
    pair: BANK.Pair | None = None,
    payload_kind: str = "delta",
) -> bool:
    """STRUCTURAL slot-aware donor eligibility for the Type-A null arm.

    Delta kind: at the prefix-end slot a matched-prefix donor's Delta is
    EXACTLY zero by causal-attention identity (same prefix tokens =>
    identical prefix-end state; in bf16 batch geometry it degrades to pure
    numerical noise, which norm-matching would silently inflate to the
    recipient's norm). Nothing to rescale either way => a same-prefix pair is
    never a pe-slot donor. Every other (slot, donor-setting) combination
    differs in at least one in-span token, so its Delta is generically
    nonzero. (Production bug found by the unit-F e2e smoke: pe/L27 null
    block, mp donor, norm_match assert.)

    State kind (single-position replace, round-2 Major 3): the donor's slot
    STATE at its target context must DIFFER from the recipient's own V_B — a
    donor sharing the recipient's target context (ce/cm*: same context ``b``;
    pe: same ``prefix_b``, the causal identity) would install the recipient's
    OWN replacement state, making the "null" bit-identical to its steered
    twin (found live on the production-DIM transport mirror: walked donor
    mq--persona__q1--conv__q1 for recipient mq--bare__q1--conv__q1 shares
    b=conv__q1 => cos(null pred, steered pred) == 1.0).
    """
    if slot == "pe" and donor.prefix_a == donor.prefix_b:
        return False
    if payload_kind == "state" and pair is not None:
        if slot == "pe":
            return donor.prefix_b != pair.prefix_b
        return donor.b != pair.b
    return True


def _resolve_donor(
    bank: dict,
    pair: BANK.Pair,
    donor_map: dict[str, str],
    pairs_by_id: dict[str, BANK.Pair],
    slot: str,
    vec_type: str,
    recipient: torch.Tensor,
    payload_kind: str = "delta",
) -> tuple[torch.Tensor, str]:
    """Deterministic donor resolution: walk the seeded derangement cycle from
    ``donor_map[pair]`` to the first slot-ELIGIBLE donor (skipping the
    recipient pair itself), continuing over the sorted setting group when the
    cycle exhausts. The REALIZED donor id is recorded per cell
    (``donor_pair_id``), and the analysis transport reconstruction prefers
    that recorded id, so the walk rule never has to be re-derived downstream.
    Type B routes straight to the centroid-swap donor (derangement-free).

    DEGENERATE cells short-circuit BEFORE the walk, mirroring each other
    (round 3, concern ``pe-replace-null-walk-exhaustion``): a ZERO recipient
    Delta (the canonicalized same-prefix pe Delta) nulls to a zero injection,
    and a matched-prefix pair's single-position pe REPLACE (state kind) nulls
    to the recipient's OWN ``V_B`` recorded ``self:<pair_id>`` — in both the
    STEERED edit is a no-op by the causal identity (same prefix tokens =>
    identical prefix-end state), so the matched null is the same no-op.
    Installing a REAL wrong-prefix donor state instead would perturb the null
    arm against a no-op steered twin, breaking the arms' parallelism (and the
    mp group has NO walk-eligible state donor at pe anyway — the round-2 crash).

    The eligibility walk is payload-kind-AWARE: the pe same-prefix exclusion
    (Δ-degeneracy) applies to both kinds, and ``state`` payloads ADDITIONALLY
    skip donors sharing the recipient's target slot state (same context ``b``;
    same ``prefix_b`` at pe) — see :func:`_donor_eligible`. So a pair's
    additive and replace null cells share the realized donor EXCEPT where the
    same-target-state exclusion forces a further walk step. When the seeded
    cycle exhausts without an eligible donor (the same-target-state exclusion
    can eat a short cross cycle whole — 2 cross pairs at pe/replace form a
    2-cycle), the walk continues deterministically over the recipient's
    SETTING group in sorted-pair-id order, so resolution is guaranteed
    whenever ANY eligible donor exists in the group.

    BOTH legs (seeded cycle AND sorted fallback) additionally skip
    PAYLOAD-DEGENERATE candidates — an aligned donor payload exactly zero at
    any position where the recipient is nonzero (an equal-query-length
    matched-prefix donor's shared leading query-span tokens are exact causal
    identities => Delta exactly 0 there; only qspan reads leading positions —
    see :func:`_payload_degenerate`). A skipped candidate joins ``seen`` (in
    the cycle leg) and the walk continues; ``norm_match``'s zero-donor assert
    in bank.py stays UNCHANGED as the last-line guard for recorded donor ids.
    """
    if vec_type == "B":
        return _donor_payload(bank, pair, pair, slot, vec_type, recipient, payload_kind)
    if not bool((recipient.norm(dim=-1) > 0).any()):
        # Degenerate recipient (the canonicalized same-prefix pe Delta): the
        # norm-matched null of a ZERO injection is zero — matching the steered
        # twin, which injects exactly nothing at this cell. Record the seeded
        # donor id for provenance.
        return torch.zeros_like(recipient), donor_map[pair.pair_id]
    if payload_kind == "state" and slot == "pe" and pair.prefix_a == pair.prefix_b:
        # Degenerate STATE cell (state-kind twin of the zero-Delta convention
        # above): a matched-prefix pair's steered pe replace installs
        # V_B(pe) == V_A(pe) — a no-op — so the matched null installs the
        # recipient's own state (the same no-op). The ``self:`` label flags
        # the cell for downstream reads (its specificity read is meaningless
        # under ANY donor choice: the steered arm edits nothing), and falls
        # through transport_row_payload's recorded-donor branch back to this
        # carve-out (one source site).
        return recipient, f"self:{pair.pair_id}"
    seen: set[str] = set()
    donor_id = donor_map[pair.pair_id]
    while donor_id not in seen:
        seen.add(donor_id)
        donor = pairs_by_id[donor_id]
        if donor_id != pair.pair_id and _donor_eligible(donor, slot, pair, payload_kind):
            raw = _aligned_donor_raw(bank, donor, slot, vec_type, recipient, payload_kind)
            if not _payload_degenerate(raw, recipient):
                return BANK.norm_match(raw, recipient), donor_id
            # payload-degenerate (zero at a recipient-nonzero position):
            # the candidate joins ``seen`` and the walk continues.
        donor_id = donor_map[donor_id]
    # Seeded cycle exhausted (every member self/ineligible/degenerate):
    # deterministic beyond-cycle fallback over the setting group in
    # sorted-pair-id order (round-3 fix, concern pe-replace-null-walk-exhaustion).
    for donor_id in sorted(donor_map):
        donor = pairs_by_id[donor_id]
        if donor.setting != pair.setting or donor_id in seen or donor_id == pair.pair_id:
            continue
        if _donor_eligible(donor, slot, pair, payload_kind):
            raw = _aligned_donor_raw(bank, donor, slot, vec_type, recipient, payload_kind)
            if _payload_degenerate(raw, recipient):
                continue
            return BANK.norm_match(raw, recipient), donor_id
    raise AssertionError(f"no eligible donor for {pair.pair_id} at slot {slot}")


# ── P1 gate: injection exactness ──────────────────────────────────────


def _gate_spot_specs(cfg: RunConfig, pairs: list[BANK.Pair]) -> list[dict]:
    """12 spot cells spanning add/replace x single/joint layers x 1/multi
    position x Type A/B (plan §7 gate 1)."""
    variants = layer_variant_names(cfg.n_layers)
    mid = variants[cfg.n_layers // 2]
    last = variants[cfg.n_layers - 1]
    mq = [p for p in pairs if p.setting == "matched_query"]
    mp = [p for p in pairs if p.setting == "matched_prefix"]
    xs = [p for p in pairs if p.setting == "cross"]
    assert mq and mp and xs, (len(mq), len(mp), len(xs))
    spec = [
        ("ce", mid, "a1", "A", mp[0]),
        ("ce", mid, "replace", "A", mp[1 % len(mp)]),
        ("ce", "joint_mid", "a2", "A", xs[0]),
        ("ce", "joint_all", "a1", "B", mq[0]),
        ("ce", mid, "a1", "B", mq[1 % len(mq)]),
        ("pe", mid, "a1", "A", mq[2 % len(mq)]),
        ("pe", last, "replace", "A", mp[2 % len(mp)]),
        ("pe", "joint_mid", "a4", "A", xs[1 % len(xs)]),
        ("cm2", "joint_mid", "a1", "A", mp[3 % len(mp)]),
        ("cm3", "joint_mid", "replace", "A", xs[2 % len(xs)]),
        ("l3j", "joint_mid", "a0.5", "A", mp[4 % len(mp)]),
        ("qspan", "joint_mid", "a1", "A", xs[3 % len(xs)]),
    ]
    return [
        {"slot": s, "layer_variant": lv, "dose": d, "vec_type": vt, "pair": p}
        for s, lv, d, vt, p in spec
    ]


def _arm_hook_for_rows(
    model,
    cfg: RunConfig,
    layers: tuple[int, ...],
    row_lengths: list[int],
    positions: list[tuple[int, ...]],
    per_row_payload: list[torch.Tensor],
    mode: str,
    alpha: float,
    expected_prompt_len: int,
):
    """Build + install the (single or stacked) hook for one batch and arm it.

    ``per_row_payload[b]`` is ``(m_b, L_model, H)`` — the ALL-layER payload; the
    hook receives each layer's own slice (plan §4.2: "the SAME edit installed at
    layers 14-20 simultaneously, each layer's own Delta at that layer").

    Multi-position ``replace`` is realized as an equivalent per-position ADD of
    ``V_B - V_A`` at alpha=1 (see :func:`_realized_mode`): ``PositionEditHook``
    restricts ``mode='replace'`` to ONE position per row (unit-A contract), and
    on the A-context row the two are numerically the same full-state patch.
    """
    if len(layers) == 1:
        hook = PositionEditHook(model, layers[0])
        deltas = [p[:, layers[0], :].contiguous() for p in per_row_payload]
        hook.install()
        hook.arm_batch(row_lengths, positions, deltas, mode=mode, alpha=alpha)
        hook.arm(expected_prompt_len)
        return hook
    stack = joint_hooks(model, list(layers))
    per_layer = [[p[:, layer, :].contiguous() for p in per_row_payload] for layer in layers]
    stack.install()
    stack.arm_batch_per_layer(row_lengths, positions, per_layer, mode=mode, alpha=alpha)
    stack.arm(expected_prompt_len)
    return stack


def _realized_mode(slot: str, dose: str, vec_type: str) -> tuple[str, float, str]:
    """``(hook_mode, alpha, payload_kind)`` for one (slot, dose, vec-type).

    ``payload_kind`` selects the tensor fed to the hook: ``"delta"`` (V_B - V_A,
    or the centroid difference for Type B) or ``"state"`` (V_B, the replacement
    state). Multi-position replace degrades to the equivalent add-patch —
    recorded per cell as ``realized_mode="add_full_state_patch"``.

    Type B has NO absolute state (its centroid is a difference from the bare
    centroid), so ``replace`` is not a Type-B dose (``DOSES_B`` excludes it) and
    a Type-B replace request fails loud here rather than patching a difference
    vector in as if it were a state.
    """
    mode, alpha = dose_spec(dose)
    assert not (mode == "replace" and vec_type == "B"), (
        "replace is not a Type-B dose: the Type-B centroid is a DIFFERENCE from the "
        "bare centroid, not an absolute slot state"
    )
    if mode == "replace" and slot in MULTI_POSITION_SLOTS:
        return ("add", 1.0, "delta")
    if mode == "replace":
        return ("replace", alpha, "state")
    return ("add", alpha, "delta")


@torch.no_grad()
def run_injection_gate(cfg: RunConfig, model, tok, bank: dict, pairs: list[BANK.Pair]) -> dict:
    """Plan §7 gate 1 — the realized edit equals alpha*Delta at the intended
    (row, position, layer) and NOWHERE else.

    Each spot runs a 2-row batch of DIFFERENT-length contexts so the per-row
    left-padding offset arithmetic is exercised, forwards once WITHOUT the hook
    and once WITH it armed, and checks three legs per spot:
    1. per-(row, position, layer) realized-vs-expected cosine + norm ratio;
    2. the hook's own ``realized_edits`` telemetry vs the independently computed
       ``alpha*Delta`` (the telemetry leg the plan names);
    3. off-target: every NON-edited position at the edit layer(s), and every
       layer BELOW the shallowest edit layer, unchanged within
       ``GATE_OFFTARGET_REL_MAX``.
    Two bank-level legs ride along: Type-B == Type-A at prefix-end, and parity
    of this driver's v_ce / v_pe with ``steering.capture_vectors``.
    """
    contexts = BANK.build_contexts()
    ctx_ids = {cid: BANK.context_token_ids_2094(tok, c) for cid, c in contexts.items()}
    pad_id = tok.pad_token_id
    recs = bank["per_context"]
    by_id = {p.pair_id: p for p in pairs}
    spots = _gate_spot_specs(cfg, pairs)
    results: list[dict] = []

    for spot in spots:
        pair: BANK.Pair = spot["pair"]
        slot, variant, dose, vec_type = (
            spot["slot"],
            spot["layer_variant"],
            spot["dose"],
            spot["vec_type"],
        )
        # Second row: a DIFFERENT pair whose A-context length differs, so the
        # padded-offset math is exercised rather than degenerate.
        others = [
            p
            for p in pairs
            if p.pair_id != pair.pair_id
            and (vec_type != "B" or p.setting == "matched_query")
            and len(ctx_ids[p.a]) != len(ctx_ids[pair.a])
        ]
        batch_pairs = [pair] + ([others[0]] if others else [])
        mode, alpha, payload_kind = _realized_mode(slot, dose, vec_type)
        layers = layer_variant_layers(variant, cfg.n_layers)

        rows = [ctx_ids[p.a] for p in batch_pairs]
        row_lengths = [len(r) for r in rows]
        positions: list[tuple[int, ...]] = []
        payloads: list[torch.Tensor] = []
        for p in batch_pairs:
            delta, state, m = _pair_payload(bank, p, slot, vec_type)
            rec = recs[p.a]
            pos = slot_positions(rec["ctx_len"], rec["prefix_end"], slot)
            pos = pos[-m:]
            positions.append(pos)
            payloads.append(state if payload_kind == "state" else delta)
            assert payloads[-1].shape[0] == len(pos), (payloads[-1].shape, pos)
        ids, mask = _left_pad(rows, pad_id, cfg.device)
        t_pad = int(ids.shape[1])

        base = extract_layer_activations(model, ids, cfg.layers, attention_mask=mask)
        base_cpu = {layer: base[layer].detach().float().cpu() for layer in cfg.layers}
        del base

        # Ascending INCREMENTAL references (joint cells): at hooked layer l the
        # exactness reference is the forward hooked at strictly SHALLOWER
        # layers only — layer l's own computation sees identical inputs in the
        # two runs, so hooked_{<=l}[l] - hooked_{<l}[l] isolates EXACTLY the
        # layer-l injection. Reading hooked-minus-CLEAN at l > min(layers)
        # (the pre-unit-F gate) also carries the legitimate residual-stream
        # propagation of the shallower injections (norm ratio ~= the hooked
        # layer count — the unit-F e2e smoke failed all 6 joint spots on
        # exactly that signature, single-layer spots at cos 1.0). ``replace``
        # OVERWRITES the position, so its absolute read needs no reference.
        # Leg-2 telemetry + leg-3 off-target read off the FULL hooked forward
        # (the last loop iteration). Cost: k forwards per k-layer joint spot
        # instead of 1 (joint_all: 28 two-row prompt forwards).
        sorted_layers = tuple(sorted(layers))
        realized_at: dict[int, torch.Tensor] = {}
        ref_cpu = base_cpu
        hooked_cpu = base_cpu
        telemetry = []
        for i, layer in enumerate(sorted_layers):
            hook = _arm_hook_for_rows(
                model,
                cfg,
                sorted_layers[: i + 1],
                row_lengths,
                positions,
                payloads,
                mode,
                alpha,
                t_pad,
            )
            try:
                hooked = extract_layer_activations(model, ids, cfg.layers, attention_mask=mask)
                hooked_cpu = {la: hooked[la].detach().float().cpu() for la in cfg.layers}
                del hooked
                telemetry = hook.realized_edits
            finally:
                hook.remove()
            realized_at[layer] = (
                hooked_cpu[layer] if mode == "replace" else hooked_cpu[layer] - ref_cpu[layer]
            )
            ref_cpu = hooked_cpu
        assert telemetry, f"spot {slot}/{variant}/{dose}: hook never applied an edit"

        cos_min, ratio_lo, ratio_hi = 1.0, math.inf, 0.0
        tele_cos_min = 1.0
        for b, p in enumerate(batch_pairs):
            off = t_pad - row_lengths[b]
            for layer in sorted_layers:
                for j, pos in enumerate(positions[b]):
                    expected = alpha * payloads[b][j, layer, :]
                    padded = pos + off
                    realized = realized_at[layer][b, padded]
                    cos = float(safe_cosine(realized, expected))
                    n_exp = float(expected.norm())
                    ratio = float(realized.norm()) / n_exp if n_exp > 0 else float("nan")
                    cos_min = min(cos_min, cos)
                    ratio_lo, ratio_hi = min(ratio_lo, ratio), max(ratio_hi, ratio)
        # Leg 2 — the hook's own telemetry vs the independently computed alpha*Delta.
        for record in telemetry:
            b, layer = record["row"], record["layer"]
            applied = record["applied"]  # (m, H) fp32 cpu
            assert record["positions_unpadded"] == list(positions[b]), (
                record["positions_unpadded"],
                positions[b],
            )
            for j in range(applied.shape[0]):
                expected = alpha * payloads[b][j, layer, :]
                tele_cos_min = min(tele_cos_min, float(safe_cosine(applied[j], expected)))

        # Leg 3 — off-target.
        edited: set[tuple[int, int]] = {
            (b, pos + (t_pad - row_lengths[b]))
            for b in range(len(batch_pairs))
            for pos in positions[b]
        }
        offtarget = 0.0
        for layer in cfg.layers:
            if layer > min(layers):
                continue  # above the shallowest edit layer the edit legitimately propagates
            diff = (hooked_cpu[layer] - base_cpu[layer]).norm(dim=-1)  # (B, T)
            denom = base_cpu[layer].norm(dim=-1).clamp_min(1e-6)
            rel = (diff / denom).clone()
            if layer in layers:
                for b, padded in edited:
                    rel[b, padded] = 0.0
            offtarget = max(offtarget, float(rel.max()))

        ok = (
            cos_min >= GATE_COS_MIN
            and tele_cos_min >= GATE_COS_MIN
            and GATE_NORM_RATIO_LO <= ratio_lo
            and ratio_hi <= GATE_NORM_RATIO_HI
            and offtarget <= GATE_OFFTARGET_REL_MAX
        )
        results.append(
            {
                "slot": slot,
                "layer_variant": variant,
                "dose": dose,
                "vec_type": vec_type,
                "hook_mode": mode,
                "alpha": alpha,
                "pair_id": pair.pair_id,
                "batch_rows": [p.pair_id for p in batch_pairs],
                "n_positions": [len(x) for x in positions],
                "layers": list(layers),
                "cos_min": cos_min,
                "telemetry_cos_min": tele_cos_min,
                "norm_ratio_lo": ratio_lo,
                "norm_ratio_hi": ratio_hi,
                "offtarget_rel_max": offtarget,
                "ok": bool(ok),
            }
        )
        logger.info(
            "[gate] %-6s %-10s %-8s %s cos=%.6f tele=%.6f ratio=[%.5f,%.5f] off=%.2e ok=%s",
            slot,
            variant,
            dose,
            vec_type,
            cos_min,
            tele_cos_min,
            ratio_lo,
            ratio_hi,
            offtarget,
            ok,
        )
        del base_cpu, hooked_cpu
    del by_id

    typeb = _typeb_prefix_equivalence(bank, pairs)
    parity = _capture_parity(cfg, model, tok, bank)
    report = {
        "criterion": "injection-exactness gate (plan §7 gate 1)",
        "bars": {
            "cos_min": GATE_COS_MIN,
            "norm_ratio": [GATE_NORM_RATIO_LO, GATE_NORM_RATIO_HI],
            "offtarget_rel_max": GATE_OFFTARGET_REL_MAX,
            "typeb_prefix_cos_min": GATE_TYPEB_PREFIX_COS_MIN,
            "capture_parity_cos_min": GATE_CAPTURE_PARITY_COS_MIN,
        },
        "spots": results,
        "typeb_prefix_equivalence": typeb,
        "capture_vectors_parity": parity,
        "n_spots": len(results),
        "n_spots_failed": sum(1 for r in results if not r["ok"]),
        "passed": (
            all(r["ok"] for r in results)
            and typeb["cos_min"] >= GATE_TYPEB_PREFIX_COS_MIN
            and parity["cos_min"] >= GATE_CAPTURE_PARITY_COS_MIN
        ),
        "repro": _repro(cfg),
    }
    return report


def _typeb_prefix_equivalence(bank: dict, pairs: list[BANK.Pair]) -> dict:
    """Prefix states are query-independent, so Type B == Type A at prefix-end.

    Asserted numerically (plan §4.2) — this is WHY the prefix-end Type-B cells
    are skipped as duplicates.
    """
    recs = bank["per_context"]
    v_pe = {cid: rec["v_pe"] for cid, rec in recs.items()}
    cos_min, n = 1.0, 0
    for pair in pairs:
        if pair.setting != "matched_query":
            continue
        a = BANK.type_a_delta(v_pe, pair)
        b = BANK.type_b_delta(v_pe, pair)
        cos_min = min(cos_min, float(safe_cosine(a.flatten(), b.flatten())))
        n += 1
    return {"n_pairs": n, "cos_min": cos_min}


@torch.no_grad()
def _capture_parity(cfg: RunConfig, model, tok, bank: dict) -> dict:
    """Reuse-seam parity: our v_ce / v_pe vs ``steering.capture_vectors``.

    ``capture_vectors`` cannot serve the bank directly (no per-position span, no
    tail-inclusive pooling, and its ``prefix_end_index`` asserts EXACTLY 3
    ``<|im_start|>`` occurrences — the conversation prefix has 5), so the bank
    capture reproduces its position conventions with
    ``extract_layer_activations``. This leg proves the conventions match on the
    SINGLE-TURN contexts (bare + persona) where the parent helper is valid.
    """
    contexts = BANK.build_contexts()
    single = [c for c in contexts.values() if not c["history"]]
    assert single, "no single-turn contexts to parity-check"
    cap = capture_vectors(model, tok, single, cfg.layers, batch_size=cfg.capture_batch)
    cos_min = 1.0
    for ctx, rec in zip(single, cap["per_context"], strict=True):
        ours = bank["per_context"][ctx["id"]]
        cos_min = min(
            cos_min,
            float(
                safe_cosine(_slot_vectors(ours, "ce")[0].flatten(), rec["v_c_context"].flatten())
            ),
            float(safe_cosine(ours["v_pe"].flatten(), rec["v_c_prefix"].flatten())),
        )
    return {"n_contexts": len(single), "cos_min": cos_min}


def phase_bank(cfg: RunConfig) -> int:
    """P1: bank.json + vc_bank.pt + the injection-exactness gate.

    Idempotent (round-2 Critical 2): a completed same-regime bank is SKIPPED
    at entry, BEFORE the model load (a ``dispatch.sh all`` relaunch after a
    mid-grid crash must not re-burn the capture — the #1689 spend-leak class);
    ``--force`` deliberately re-runs. The phase is deterministic (greedy
    teacher-forced captures, fixed seeds), so a skip is content-safe; the grid
    resume survives a re-capture anyway (regime fp keys on the bank MANIFEST
    sha, not tensor bytes).
    """
    logger.info("[phase=bank]")
    manifest, bank_sha = bank_manifest_and_sha()
    regime_fp = regime_fingerprint(cfg, bank_sha)
    if not cfg.force and bank_is_done(cfg, regime_fp):
        rec = _phase_done_record(cfg, "bank", regime_fp) or {}
        forced = (
            " [bank was FORCED past a FAILED injection gate — see injection_gate_report.json]"
            if rec.get("forced_past_gate")
            else ""
        )
        logger.info("[bank] already done for this regime — skipping (--force re-runs)%s", forced)
        logger.info("[phase=bank_done]")
        return RC_OK
    if cfg.force and (cfg.manifest_dir / "bank_done.json").exists():
        logger.info("[bank] --force set: deliberately re-running a done bank phase")
    model, tok = load_model_and_tokenizer(cfg)
    manifest["bank_sha"] = bank_sha
    manifest["repro"] = _repro(cfg)
    _write_json_atomic(cfg.bank_dir / "bank.json", manifest)

    bank = capture_bank(cfg, model, tok)
    pairs = BANK.build_pairs()
    _save_pt_atomic(
        cfg.bank_dir / "vc_bank.pt",
        {
            "layers": bank["layers"],
            "per_context": bank["per_context"],
            "centroids": bank["centroids"],
            "donor_derangement": BANK.donor_derangement(pairs),
            "bank_sha": manifest["bank_sha"],
            "repro": _repro(cfg),
        },
    )
    logger.info("[bank] captured %d contexts x %d layers", len(bank["per_context"]), cfg.n_layers)

    report = run_injection_gate(cfg, model, tok, bank, pairs)
    _write_json_atomic(cfg.bank_dir / "injection_gate_report.json", report)
    if not report["passed"]:
        logger.error(
            "[injection_gate] FAILED: %d/%d spots failed (typeb_cos=%.6f parity_cos=%.6f)",
            report["n_spots_failed"],
            report["n_spots"],
            report["typeb_prefix_equivalence"]["cos_min"],
            report["capture_vectors_parity"]["cos_min"],
        )
        if not cfg.force:
            # No done-manifest on the gate-refusal path: a relaunch re-runs.
            return RC_INJECTION_GATE
        logger.error("[injection_gate] --force set: proceeding on a FAILED gate (recorded)")
    _write_json_atomic(
        cfg.manifest_dir / "bank_done.json",
        {
            "regime_fp": regime_fp,
            "bank_sha": bank_sha,
            "n_contexts": len(bank["per_context"]),
            "gate_passed": bool(report["passed"]),
            "forced_past_gate": bool(not report["passed"] and cfg.force),
            "repro": _repro(cfg),
        },
    )
    logger.info("[phase=bank_done]")
    return RC_OK


# ── P2: anchors ───────────────────────────────────────────────────────


@torch.no_grad()
def capture_answer_states(
    cfg: RunConfig,
    model,
    tok,
    ctx_ids_by_row: list[list[int]],
    completions: list[str],
    eot_ids: list[int],
) -> dict:
    """BOTH answer poolings from ONE forward per chunk.

    ``va_span`` = mean over the COMPLETION token positions (the #1415
    ``capture_vectors`` convention F_act uses); ``va_tail`` = mean over the
    completion PLUS the assistant end-of-turn tail (the ``capture_answer_vector``
    convention the banked maps' outputs were fit under). Rows are built by
    concatenating per-segment TOKEN IDS — never by re-tokenizing a concatenated
    string (the BPE-seam rule).
    """
    assert len(ctx_ids_by_row) == len(completions), (len(ctx_ids_by_row), len(completions))
    layers = cfg.layers
    pad_id = tok.pad_token_id
    n = len(completions)
    va_span = torch.zeros((n, len(layers), cfg.hidden), dtype=torch.float32)
    va_tail = torch.zeros_like(va_span)
    # Tokenize each completion ONCE — these ids are both the capture's
    # teacher-forced segment and the cap-hit telemetry basis.
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
        captured = extract_layer_activations(model, ids, layers, attention_mask=mask)
        for j, (i, ctx_len, n_comp) in enumerate(keep):
            span = slice(ctx_len, ctx_len + n_comp)
            tail = slice(ctx_len, ctx_len + n_comp + len(eot_ids))
            va_span[i] = torch.stack(
                [captured[layer][j, span].float().mean(dim=0) for layer in layers]
            ).cpu()
            va_tail[i] = torch.stack(
                [captured[layer][j, tail].float().mean(dim=0) for layer in layers]
            ).cpu()
        del captured
    return {
        "va_span": va_span.to(torch.float16),
        "va_tail": va_tail.to(torch.float16),
        "n_completion_tokens": n_comp_tokens,
        "empty_rows": sorted(empty),
        "pooling": {
            "va_span": "mean over completion tokens (#1415 capture_vectors convention)",
            "va_tail": "mean over completion + assistant end-of-turn tail (v_x convention)",
        },
    }


def phase_anchors(cfg: RunConfig) -> int:
    """P2: 15 contexts x K unpatched temp-1.0 rollouts + both-pooling V_a.

    Idempotent (round-2 Critical 2): a completed same-regime anchors phase is
    SKIPPED at entry, BEFORE the model load; ``--force`` deliberately re-runs.
    Deterministic given the regime (fixed per-draw seeds), so a skip is
    content-safe.
    """
    logger.info("[phase=anchors]")
    # >= 2 draws even under --smoke: the disjoint-half floor F_act needs
    # (fmetrics.half_split_indices asserts k >= 2).
    draws = 2 if cfg.smoke else cfg.anchor_draws
    _manifest, bank_sha = bank_manifest_and_sha()
    regime_fp = regime_fingerprint(cfg, bank_sha)
    if not cfg.force and anchors_is_done(cfg, regime_fp, draws):
        logger.info("[anchors] already done for this regime — skipping (--force re-runs)")
        logger.info("[phase=anchors_done]")
        return RC_OK
    if cfg.force and (cfg.manifest_dir / "anchors_done.json").exists():
        logger.info("[anchors] --force set: deliberately re-running a done anchors phase")
    model, tok = load_model_and_tokenizer(cfg)
    contexts = BANK.build_contexts()
    order = list(contexts)
    if cfg.smoke:
        # ONE context per PREFIX class — the conv prefix is the only one that
        # exercises the multi-turn history render + prefix_end_index_multi, so a
        # first-N slice would leave that arm class unrun (#1090 fu5 lesson) —
        # UNION every context the smoke GRID pairs touch: downstream F tables
        # index anchor V_a by BOTH endpoints of every graded pair, so an anchor
        # slice missing a smoke pair's context_b KeyErrors at P7 (cross-phase
        # data-contract floor; found by the unit-F end-to-end smoke).
        per_prefix = [
            next(cid for cid in order if contexts[cid]["prefix"] == prefix)
            for prefix in BANK.PREFIX_ORDER
        ]
        pairs_by_id = {p.pair_id: p for p in BANK.build_pairs()}
        smoke_ctx = {
            cid
            for fam in smoke_block_families(BANK.build_pairs(), cfg.n_layers)
            for block in fam
            for pid in block.pair_ids
            for cid in (pairs_by_id[pid].a, pairs_by_id[pid].b)
        }
        order = [cid for cid in order if cid in smoke_ctx or cid in per_prefix]
    ctx_list = [contexts[c] for c in order]
    eot = eot_tail_ids(tok)
    t0 = time.monotonic()
    outs = generate_batch(
        model,
        tok,
        ctx_list,
        n=draws,
        hook=None,
        max_new_tokens=cfg.max_new_tokens,
        temperature=ANCHOR_TEMPERATURE,
        seed_base=cfg.seed_base,
        # History-aware render: steering's own context_messages silently DROPS
        # the conv prefix's `history` turns (bank.py module note) — the conv
        # anchors would otherwise generate under the WRONG (bare-like) context.
        render_fn=BANK.render_context_2094,
        ids_fn=BANK.context_token_ids_2094,
    )
    logger.info(
        "[anchors] %d contexts x %d draws in %.1fs", len(order), draws, time.monotonic() - t0
    )

    ctx_ids = {cid: BANK.context_token_ids_2094(tok, contexts[cid]) for cid in order}
    flat_ctx: list[list[int]] = []
    flat_text: list[str] = []
    rows: list[dict] = []
    for b, cid in enumerate(order):
        for i, text in enumerate(outs[b]):
            flat_ctx.append(ctx_ids[cid])
            flat_text.append(text)
            rows.append({"context_id": cid, "draw": i, "seed": cfg.seed_base + i, "text": text})
    states = capture_answer_states(cfg, model, tok, flat_ctx, flat_text, eot)
    for r, n_tok in zip(rows, states["n_completion_tokens"], strict=True):
        r["n_completion_tokens"] = n_tok
        r["cap_hit"] = cap_hit(n_tok, cfg.max_new_tokens)
        r["cap_hit_basis"] = "retokenized_completion_len >= max_new_tokens"
        r["temperature"] = ANCHOR_TEMPERATURE
    _write_jsonl_atomic(cfg.anchors_dir / "anchors.jsonl", rows)
    _save_pt_atomic(
        cfg.anchors_dir / "va_anchors.pt",
        {
            "layers": cfg.layers,
            "index": [{"context_id": r["context_id"], "draw": r["draw"]} for r in rows],
            "va_span": states["va_span"],
            "va_tail": states["va_tail"],
            "pooling": states["pooling"],
            "empty_rows": states["empty_rows"],
            "repro": _repro(cfg),
        },
    )
    cap_hits = sum(1 for r in rows if r["cap_hit"])
    _write_json_atomic(
        cfg.manifest_dir / "anchors_done.json",
        {
            "regime_fp": regime_fp,
            "n_contexts": len(order),
            "draws": draws,
            "n_rows": len(rows),
            "n_cap_hit": cap_hits,
            "cap_hit_frac": cap_hits / max(1, len(rows)),
            "n_empty": len(states["empty_rows"]),
            "repro": _repro(cfg),
        },
    )
    logger.info(
        "[anchors] rows=%d cap_hit=%d empty=%d", len(rows), cap_hits, len(states["empty_rows"])
    )
    logger.info("[phase=anchors_done]")
    return RC_OK


# ── P3 + P4: the grid ─────────────────────────────────────────────────


def _load_bank(cfg: RunConfig) -> dict:
    path = cfg.bank_dir / "vc_bank.pt"
    assert path.exists(), (
        f"{path} missing — run `--phase bank` first (this driver never silently "
        "recaptures the V bank mid-grid)"
    )
    # Self-produced, sha-recorded bundle carrying non-tensor metadata.
    return torch.load(path, map_location="cpu", weights_only=False)


@torch.no_grad()
def run_block(
    cfg: RunConfig,
    model,
    tok,
    bank: dict,
    block: Block,
    pairs_by_id: dict[str, BANK.Pair],
    donor_map: dict[str, str],
    eot: list[int],
    regime_fp: str,
) -> dict:
    """One block: hooked greedy rollouts for every cell, then the V_a pass."""
    contexts = BANK.build_contexts()
    ctx_ids_cache: dict[str, list[int]] = {}

    def ids_for(cid: str) -> list[int]:
        if cid not in ctx_ids_cache:
            ctx_ids_cache[cid] = BANK.context_token_ids_2094(tok, contexts[cid])
        return ctx_ids_cache[cid]

    layers = layer_variant_layers(block.layer_variant, cfg.n_layers)
    mode, alpha, payload_kind = _realized_mode(block.slot, block.dose, block.vec_type)
    realized_mode = (
        "add_full_state_patch"
        if block.dose == "replace" and mode == "add"
        else ("replace" if mode == "replace" else "add")
    )
    recs = bank["per_context"]
    cells: list[dict] = []
    for pid in block.pair_ids:
        pair = pairs_by_id[pid]
        delta, state, m = _pair_payload(bank, pair, block.slot, block.vec_type)
        recipient = state if payload_kind == "state" else delta
        donor_label = None
        if block.arm == "null":
            # Additive cells: donor Delta norm-matched to the recipient Delta.
            # Single-position replace cells: the donor pair's V_B STATE
            # norm-matched to the recipient's V_B (round-2 Major 3 resolution).
            recipient, donor_label = _resolve_donor(
                bank,
                pair,
                donor_map,
                pairs_by_id,
                block.slot,
                block.vec_type,
                recipient,
                payload_kind,
            )
        rec = recs[pair.a]
        pos = slot_positions(rec["ctx_len"], rec["prefix_end"], block.slot)[-m:]
        assert recipient.shape[0] == len(pos), (recipient.shape, pos)
        cells.append(
            {
                "pair_id": pid,
                "setting": pair.setting,
                "context_a": pair.a,
                "context_b": pair.b,
                "positions": list(pos),
                "payload": recipient,
                "donor_pair_id": donor_label,
            }
        )

    texts: list[str] = []
    for start in range(0, len(cells), cfg.gen_batch):
        chunk = cells[start : start + cfg.gen_batch]
        ctx_list = [contexts[c["context_a"]] for c in chunk]
        rows = [ids_for(c["context_a"]) for c in chunk]
        row_lengths = [len(r) for r in rows]
        t_pad = max(row_lengths)
        hook = _arm_hook_for_rows(
            model,
            cfg,
            layers,
            row_lengths,
            [tuple(c["positions"]) for c in chunk],
            [c["payload"] for c in chunk],
            mode,
            alpha,
            t_pad,
        )
        try:
            outs = generate_batch(
                model,
                tok,
                ctx_list,
                n=1,
                hook=hook,
                max_new_tokens=cfg.max_new_tokens,
                temperature=GRID_TEMPERATURE,
                seed_base=cfg.seed_base,
                # History-aware render (bank.py module note): the hook's
                # row_lengths/positions come from the *_2094 ids, so the render
                # MUST match — steering's default drops conv `history`, which
                # both mis-renders the context AND breaks the arm() length
                # invariant for every conv-prefixed context_a (mp-conv pairs).
                render_fn=BANK.render_context_2094,
                ids_fn=BANK.context_token_ids_2094,
            )
        finally:
            hook.remove()
        assert len(outs) == len(chunk), (len(outs), len(chunk))
        texts.extend(o[0] for o in outs)
    assert len(texts) == len(cells), (len(texts), len(cells))

    states = capture_answer_states(
        cfg, model, tok, [ids_for(c["context_a"]) for c in cells], texts, eot
    )
    rows_out: list[dict] = []
    for i, cell in enumerate(cells):
        n_tok = states["n_completion_tokens"][i]
        rows_out.append(
            {
                "block_key": block.key,
                "slot": block.slot,
                "layer_variant": block.layer_variant,
                "layers": list(layers),
                "dose": block.dose,
                "alpha": alpha,
                "hook_mode": mode,
                "realized_mode": realized_mode,
                "vec_type": block.vec_type,
                "arm": block.arm,
                "pair_id": cell["pair_id"],
                "setting": cell["setting"],
                "context_a": cell["context_a"],
                "context_b": cell["context_b"],
                "positions": cell["positions"],
                "donor_pair_id": cell["donor_pair_id"],
                "temperature": GRID_TEMPERATURE,
                "seed": cfg.seed_base,
                "n_completion_tokens": n_tok,
                "cap_hit": cap_hit(n_tok, cfg.max_new_tokens),
                "cap_hit_basis": "retokenized_completion_len >= max_new_tokens",
                "text": texts[i],
            }
        )
    _write_jsonl_atomic(cfg.rollouts_dir / f"shard_{block.slug}.jsonl", rows_out)
    _save_pt_atomic(
        cfg.va_dir / f"shard_{block.slug}.pt",
        {
            "block_key": block.key,
            "layers": cfg.layers,
            "index": [{"pair_id": r["pair_id"], "context_a": r["context_a"]} for r in rows_out],
            "va_span": states["va_span"],
            "va_tail": states["va_tail"],
            "pooling": states["pooling"],
            "empty_rows": states["empty_rows"],
            "repro": _repro(cfg),
        },
    )
    done = {
        "key": block.key,
        "regime_fp": regime_fp,
        "n_cells": len(rows_out),
        "n_cap_hit": sum(1 for r in rows_out if r["cap_hit"]),
        "n_empty": len(states["empty_rows"]),
        "repro": _repro(cfg),
    }
    _write_json_atomic(block_done_path(cfg.out_root, block), done)
    return done


def phase_grid(cfg: RunConfig) -> int:
    """P3+P4: sharded block execution (or the timing pilot under ``--pilot``)."""
    logger.info("[phase=grid] worker=%d/%d smoke=%s", cfg.worker_index, cfg.num_workers, cfg.smoke)
    bank = _load_bank(cfg)
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_map = bank.get("donor_derangement") or BANK.donor_derangement(pairs)
    regime_fp = regime_fingerprint(cfg, str(bank.get("bank_sha")))

    all_families = enumerate_block_families(pairs, cfg.n_layers)
    totals_all = grid_totals(all_families)
    families = smoke_block_families(pairs, cfg.n_layers) if cfg.smoke else all_families
    if cfg.pilot:
        families = families[:1]
    blocks = blocks_for_worker(families, cfg.worker_index, cfg.num_workers)
    totals = grid_totals(families)
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
            "n_blocks_this_worker": len(blocks),
            "n_cells_this_worker": sum(b.n_cells for b in blocks),
            "block_keys": [b.key for b in blocks],
            "repro": _repro(cfg),
        },
    )
    logger.info(
        "[grid] full grid: %d blocks / %d cells (%d steered + %d null); this worker: %d blocks / %d cells",
        totals_all["n_blocks"],
        totals_all["cells_total"],
        totals_all["cells_steered"],
        totals_all["cells_null"],
        len(blocks),
        sum(b.n_cells for b in blocks),
    )

    if not blocks:
        logger.info("[grid] worker %d has no assigned blocks — nothing to do", cfg.worker_index)
        logger.info("[phase=grid_done] worker=%d blocks=0 cells=0", cfg.worker_index)
        return RC_OK

    model, tok = load_model_and_tokenizer(cfg)
    eot = eot_tail_ids(tok)
    n_total = len(blocks)
    done_count = 0
    ran_cells = 0
    ran_wall = 0.0
    uploaded: list[str] = []
    pending: list[Block] = []  # completed-but-not-yet-uploaded shards
    for k, block in enumerate(blocks, start=1):
        if not cfg.pilot and block_is_done(cfg.out_root, block, regime_fp):
            done_count += 1
            logger.info("[grid] block %d/%d %s SKIP (done)", k, n_total, block.key)
            continue
        t0 = time.monotonic()
        rec = run_block(cfg, model, tok, bank, block, pairs_by_id, donor_map, eot, regime_fp)
        elapsed = time.monotonic() - t0
        ran_cells += rec["n_cells"]
        ran_wall += elapsed
        pending.append(block)
        logger.info(
            "[grid] block %d/%d %s rows=%d cap_hit=%d elapsed=%.1fs",
            k,
            n_total,
            block.key,
            rec["n_cells"],
            rec["n_cap_hit"],
            elapsed,
        )
        if not cfg.pilot and cfg.upload_every > 0 and len(pending) >= cfg.upload_every:
            uploaded += _upload_grid_increment(cfg, pending)
            pending = []
    if not cfg.pilot and pending:
        uploaded += _upload_grid_increment(cfg, pending)

    if cfg.pilot:
        return _enforce_pilot_gate(cfg, totals_all, ran_cells, ran_wall)
    _write_json_atomic(
        cfg.manifest_dir / f"grid_done_w{cfg.worker_index}.json",
        {
            "regime_fp": regime_fp,
            "worker_index": cfg.worker_index,
            "n_blocks": n_total,
            "n_blocks_skipped": done_count,
            "n_cells_run": ran_cells,
            "wall_s": ran_wall,
            "uploads": uploaded,
            "repro": _repro(cfg),
        },
    )
    logger.info(
        "[phase=grid_done] worker=%d blocks=%d cells=%d", cfg.worker_index, n_total, ran_cells
    )
    return RC_OK


def _enforce_pilot_gate(cfg: RunConfig, totals_all: dict, ran_cells: int, ran_wall: float) -> int:
    """Plan §7 gate 2 — a DESIGNED halt on a >3x-plan projected pod wall.

    The pilot ran ONE production-shape block family (both arms, B=gen_batch)
    through THIS entrypoint, so the measured per-cell wall is at the sweep's
    execution shape (the #1415 batch-1 false-fire lesson).
    """
    assert ran_cells > 0, "pilot ran no cells"
    per_cell = ran_wall / ran_cells
    width = max(1, cfg.num_workers)
    projected_h = per_cell * totals_all["cells_total"] / width / 3600.0
    fence_h = PILOT_FENCE_MULT * projected_h
    refuse = projected_h > PILOT_REFUSAL_MULT * cfg.planned_wall_h
    report = {
        "criterion": "generation-throughput pilot (plan §7 gate 2)",
        "measured_cells": ran_cells,
        "measured_wall_s": ran_wall,
        "s_per_cell": per_cell,
        "gen_batch": cfg.gen_batch,
        "num_workers": width,
        "cells_total": totals_all["cells_total"],
        "projected_pod_wall_h": projected_h,
        "planned_pod_wall_h": cfg.planned_wall_h,
        "refusal_threshold_h": PILOT_REFUSAL_MULT * cfg.planned_wall_h,
        "recommended_poll_fence_h": fence_h,
        "sweep_allowed": not refuse,
        "forced": cfg.force,
        "repro": _repro(cfg),
    }
    _write_json_atomic(cfg.out_root / "pilot_gate_report.json", report)
    logger.info(
        "[phase=pilot_done] s_per_cell=%.3f projected_pod_wall_h=%.2f (planned %.2f) "
        "fence_h=%.2f sweep_allowed=%s",
        per_cell,
        projected_h,
        cfg.planned_wall_h,
        fence_h,
        not refuse,
    )
    if refuse and not cfg.force:
        logger.error(
            "[pilot_gate] projected pod wall %.2f h > %.1fx planned %.2f h — refusing the grid "
            "(pass --force to override, or descope per the plan §9 stratification ladder)",
            projected_h,
            PILOT_REFUSAL_MULT,
            cfg.planned_wall_h,
        )
        return RC_PILOT_GATE
    return RC_OK


# ── P5: upload + sentinel ─────────────────────────────────────────────

# Bounded OUTER retry around the hub helper's fail-soft "" return
# (upload-policy rule (c), the #1315 `_upload_with_transport_retry` shape):
# the hub's inner `_retry_upload` envelope already absorbs per-call
# transients, so a no-path return means that budget EXHAUSTED or a
# non-transient failure (missing HF_TOKEN, failed exact-set verify).
UPLOAD_TRANSPORT_RETRIES = 3
UPLOAD_BACKOFF_BASE_S: tuple[float, ...] = (30.0, 60.0, 120.0)
_upload_retry_sleep = time.sleep  # monkeypatchable in tests


def _upload_dir(
    cfg: RunConfig,
    local_dir: Path,
    remote_prefix: str,
    allow_patterns: list[str],
) -> list[str]:
    """ONE bulk ``upload_folder`` commit for a matched subset + exact-set verify.

    FAIL-LOUD (round-2 Critical 1): ``_upload_folder_filtered`` returns ``""``
    on failure (missing HF_TOKEN / failed post-upload exact-set verify /
    upload exception) — this seam retries the no-path return with bounded
    jittered backoff (uploads are idempotent; already-landed files skip
    Hub-side), then RAISES on exhaustion so the results sentinel can never
    post over silently-lost durability (the #841 result-persist class).
    """
    if cfg.upload_mode == "none":
        logger.info("[upload] skipped (--upload none): %s", local_dir)
        return []
    if not local_dir.exists():
        logger.info("[upload] nothing staged under %s — skipping", local_dir)
        return []
    files = sorted(p for pat in allow_patterns for p in local_dir.glob(pat) if p.is_file())
    if not files:
        logger.info("[upload] no files match %s under %s — skipping", allow_patterns, local_dir)
        return []
    expected = [f"{remote_prefix}/{p.relative_to(local_dir).as_posix()}" for p in files]
    if cfg.upload_mode == "local-mirror":
        for p, rel in zip(files, expected, strict=True):
            dest = cfg.out_root / "hf_mirror" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        logger.info("[upload] mirrored %d files -> %s", len(files), remote_prefix)
        return expected
    from explore_persona_space.orchestrate.hub import _upload_folder_filtered

    base_url = ""
    for attempt in range(UPLOAD_TRANSPORT_RETRIES + 1):
        base_url = _upload_folder_filtered(
            local_dir=local_dir,
            repo_id=HF_DATA_REPO,
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


def _upload_grid_increment(cfg: RunConfig, blocks: list[Block]) -> list[str]:
    """Incremental per-block-batch text upload (plan §9 phase-order persistence).

    Only THIS worker's completed shards are matched, so N concurrent workers
    never contend for the same file set, and the commit count stays far under
    the 256/hr cap (one commit per ``--upload-every`` blocks per worker).
    """
    slugs = [b.slug for b in blocks if (cfg.rollouts_dir / f"shard_{b.slug}.jsonl").exists()]
    if not slugs:
        return []
    return _upload_dir(
        cfg,
        cfg.rollouts_dir,
        f"{HF_PREFIX}/raw_completions/grid",
        [f"shard_{s}.jsonl" for s in slugs],
    )


def _sentinel_payload(cfg: RunConfig, uploaded: dict[str, list[str]]) -> dict:
    """The /issue Step 7 results payload (all 10 keys)."""
    n_grid_shards = len(list(cfg.rollouts_dir.glob("shard_*.jsonl")))
    n_va_shards = len(list(cfg.va_dir.glob("shard_*.pt")))
    n_anchor_rows = 0
    anchors = cfg.anchors_dir / "anchors.jsonl"
    if anchors.exists():
        n_anchor_rows = sum(1 for line in anchors.open(encoding="utf-8") if line.strip())
    gate_path = cfg.bank_dir / "injection_gate_report.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    cap_hits, cap_total = 0, 0
    for done in sorted((cfg.manifest_dir / "blocks").glob("*.done.json")):
        rec = json.loads(done.read_text())
        cap_hits += int(rec.get("n_cap_hit", 0))
        cap_total += int(rec.get("n_cells", 0))
    return {
        "eval_numbers": {
            "grid_shards": n_grid_shards,
            "va_shards": n_va_shards,
            "anchor_rows": n_anchor_rows,
            "cells_persisted": cap_total,
            "cap_hit_rows": cap_hits,
            "cap_hit_frac": (cap_hits / cap_total) if cap_total else 0.0,
            "injection_gate_passed": bool(gate.get("passed")),
            "injection_gate_spots_failed": int(gate.get("n_spots_failed", 0)),
        },
        "eval_paths": sorted(
            {
                str(cfg.bank_dir / "bank.json"),
                str(cfg.bank_dir / "vc_bank.pt"),
                str(gate_path),
                str(anchors),
                str(cfg.anchors_dir / "va_anchors.pt"),
                str(cfg.rollouts_dir),
                str(cfg.va_dir),
            }
        ),
        "reproducibility_card": {
            **_repro(cfg),
            "seed_base": cfg.seed_base,
            "bank_seed": BANK.SEED,
            "max_new_tokens": cfg.max_new_tokens,
            "grid_temperature": GRID_TEMPERATURE,
            "anchor_temperature": ANCHOR_TEMPERATURE,
            "anchor_draws": cfg.anchor_draws,
            "gen_batch": cfg.gen_batch,
            "num_workers": cfg.num_workers,
        },
        "wandb_url": None,
        "hf_hub_url": f"https://huggingface.co/datasets/{HF_DATA_REPO}/tree/main/{HF_PREFIX}",
        "worktree_path": str(REPO_ROOT),
        "final_commit_sha": _git_sha(),
        "gpu_hours_used": None,
        "gpu_hours_budgeted": cfg.gpu_hours_budgeted,
        "plan_deviations": [
            "replace dose on the multi-position slots (l3j, qspan) is realized as an "
            "equivalent per-position add-patch (Delta = V_B - V_A, alpha=1): "
            "PositionEditHook restricts mode='replace' to one position per row, and on "
            "the A-context row the two are the same full-state patch "
            "(recorded per cell as realized_mode=add_full_state_patch)",
            "cap-hit telemetry is derived from the re-tokenized completion length "
            "(generate_batch returns decoded text only); recorded per row as "
            "cap_hit_basis=retokenized_completion_len >= max_new_tokens",
            "the V_a store carries BOTH poolings (span-mean + tail-inclusive) at 28 "
            "layers in fp16, ~17 GB rather than the plan §9 ~9 GB single-pooling figure "
            "(still far under the ~130 GB /workspace quota)",
            "single-position replace-dose NULL cells install the DONOR pair's "
            "target-context STATE norm_match(V_B(donor), V_B) at the same slot/layer — "
            "plan §4.2's Δ-centric null wording is incoherent at the replace rung (a "
            "Δ-normed replacement would near-zero the slot), and a difference vector "
            "installed as a slot state would be out-of-distribution vs the steered "
            "arm's real-state replace; donors sharing the recipient's target slot "
            "state (same context b; same prefix_b at pe) are walk-excluded for these "
            "cells (a same-state 'null' would duplicate the steered arm); resolves "
            "concern replace-null-donor-realization (round-2 code review)",
            "pe x replace NULL cells of matched-prefix pairs are DEGENERATE: the "
            "steered replace installs V_B(pe) == V_A(pe) (same prefix tokens => "
            "identical prefix-end state), a no-op — so the null installs the "
            "recipient's OWN V_B (the same no-op), recorded "
            "donor_pair_id=self:<pair_id> (the state-kind twin of the zero-Delta "
            "null convention; flag these cells out of specificity reads); and the "
            "donor walk falls back to the sorted setting group when the seeded "
            "derangement cycle exhausts (2 cross pairs at pe/replace form a 2-cycle "
            "under the same-target-state exclusion); resolves concern "
            "pe-replace-null-walk-exhaustion (round-3 code review)",
        ],
        "uploaded_prefixes": {k: len(v) for k, v in uploaded.items()},
    }


def phase_upload(cfg: RunConfig) -> int:
    """P5: bulk upload every prefix, then write the pod sentinel."""
    logger.info("[phase=upload]")
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
    # blocks/*.done.json rides along (round-2 Minor 7): per-block resume state
    # + the sentinel's cap-hit provenance become durable off-pod (~880 small
    # JSONs, ONE upload_folder commit — under the 2000-files/commit watermark).
    uploaded["manifests"] = _upload_dir(
        cfg,
        cfg.manifest_dir,
        f"{HF_PREFIX}/analysis_tensors/manifests",
        ["*.json", "blocks/*.done.json"],
    )
    payload = _sentinel_payload(cfg, uploaded)
    _write_json_atomic(cfg.out_root / "manifests" / "upload_done.json", payload)
    sentinel = cfg.log_dir / SENTINEL_NAME
    body = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "note": payload,
    }
    _write_json_atomic(sentinel, body)
    logger.info("[upload] sentinel written: %s", sentinel)
    logger.info("[phase=upload_done]")
    return RC_OK


# ── entrypoint ────────────────────────────────────────────────────────


def _import_check() -> None:
    """Resolve EVERY deferred import this driver reaches on its real paths.

    A bare ``import scripts.issue2094_run`` fires only module-level imports; the
    heavy loads live inside ``load_model_and_tokenizer`` / ``_repro`` /
    ``_upload_dir``, so they are NAMED here (the #1689 false-pass class).
    """
    from transformers import (  # noqa: F401
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    import transformers  # noqa: F401
    from huggingface_hub import HfApi  # noqa: F401

    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        _upload_folder_filtered,
        verify_repo_paths_uploaded,
    )

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
    assert args.phase, "--phase is required (or pass --import-check)"
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
    assert cfg.phase == "upload", cfg.phase
    return phase_upload(cfg)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension interpreter finalization (#1689).
    sys.exit(rc)
