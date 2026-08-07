#!/usr/bin/env python3
"""Issue #2162 — pod driver for the context-information minimal-pair patch grid.

Forked from ``scripts/issue2094_run.py`` (same phase/worker/checkpoint/resume
skeleton). **Fork-base constant flips (plan §4.6, the r1 mechanical gate):**

- ``GRID_TEMPERATURE`` 0.0 -> **1.0** (the parent grid was greedy),
- ``MAX_NEW_TOKENS`` 1024 -> **2048** (cap policy §4.5),
- a per-pair x arm **K=5 grid draw loop** (``GRID_DRAWS``) is ADDED — the
  parent grid had NO per-pair draw seam (its only K was ``ANCHOR_DRAWS``).

Phases (plan §4.6 DAG; this driver owns P1/P2/P3/P5 + the margin TF):

- ``--phase bank`` (P1): build the 1,404-context bank STRICT (committed
  ``frozen_gen_2162.json`` required), capture the all-layer v_ce / v_pe states
  (one right-padded forward per chunk, positions from token ids — BPE-seam
  rule), persist BOTH seeded donor assignments, run the DEGENERACY GUARD
  (plan §7 gate 2: per pair x slot state identity vs the span-locus registry;
  HALT ``RC_DEGENERACY_GATE`` on mismatch) and the INJECTION-EXACTNESS GATE
  (plan §7 gate 1: 12 spot cells re-forwarded with the all-28-layer replace
  hook armed, cosine >= 0.999, norm ratio in [0.995, 1.005]; HALT
  ``RC_INJECTION_GATE``). Designed halts write report JSONs + distinct rcs.
- ``--phase anchors`` (P2): 1,404 contexts x K=10 unpatched temp-1.0 rollouts,
  SHARDED across workers, the plan §7 gate-3 slice contexts generated FIRST
  (their per-worker JSONL uploads immediately so the VM judge can start while
  the remaining anchors generate), plus per-rollout answer-state capture.
- ``--phase grid`` (P3): the 234 (type-cell x slot x arm) blocks pulled from a
  SHARED work-conserving claim-file queue (atomic ``O_CREAT | O_EXCL`` claims
  under ``<out-root>/claims/``; a claim with no matching done-checkpoint is
  RECLAIMABLE — pid-liveness keyed on the same host, claim-age keyed
  otherwise; inconsistent claim/checkpoint state fails loud). Per block: K=5
  temp-1.0 hooked draws per pair, the hooked teacher-forced V_a capture pass,
  and (when the margin pools file is present) the pool-item margin TF pass —
  pipelined on the same GPU. ``--pilot`` times ONE production-shape block
  through this same entrypoint (``RC_PILOT_GATE`` on a >3x-plan projection).
- ``--phase margin``: the teacher-forced fixed pos-vs-neg margin legs that
  need the judge-built pools file — anchor margins (contexts sharded across
  workers) + the per-block catch-up for blocks whose grid pass ran before the
  pools landed (claim-queue namespace ``margin``).
- ``--phase upload`` (P5): ONE bulk ``upload_folder`` commit per HF prefix
  (never a per-file loop), an exact-set verify, then the pod sentinel
  ``/workspace/logs/issue-2162-results.json``.

Pod-side contract: sentinel file + ``[phase=...]`` breadcrumbs ONLY — this
file NEVER shells out to ``scripts/task.py``. Every phase ends with an
explicit ``sys.exit`` (#1689 finalization-race rule).
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
import uuid
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
    generate_batch,
)
from explore_persona_space.experiments.issue2094 import bank as BANK94  # noqa: E402
from explore_persona_space.experiments.issue2094.fmetrics import safe_cosine  # noqa: E402
from explore_persona_space.experiments.issue2094.hooks import (  # noqa: E402
    joint_hooks,
)
from explore_persona_space.experiments.issue2162 import bank2162 as BANK  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("issue2162.run")

# ── constants (plan §4.2/§4.3/§4.5/§9/§10) ────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HIDDEN_FULL = 3584
N_MODEL_LAYERS_FULL = 28
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue2162_ctxinfo"
DEFAULT_OUT_ROOT = Path("/workspace/issue2162_out")
DEFAULT_LOG_DIR = Path("/workspace/logs")
SENTINEL_NAME = "issue-2162-results.json"
SENTINEL_NAME_SMOKE = "issue-2162-smoke-results.json"

# Fork-base flips (plan §4.6): parent pinned 1024 / 0.0 and had no grid-draw K.
MAX_NEW_TOKENS = 2048
GRID_TEMPERATURE = 1.0
GRID_DRAWS = 5
SEED_BASE = 42
ANCHOR_DRAWS = 10
ANCHOR_TEMPERATURE = 1.0

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

# Plan §9: P3 = 3.7 h projected pod wall at width 8.
PLANNED_GRID_WALL_H = 3.7
PILOT_REFUSAL_MULT = 3.0
PILOT_FENCE_MULT = 2.0

# Claim-file queue (plan §4.6): stale = dead pid (same host) or claim age.
CLAIM_STALE_S = float(os.environ.get("EPM_2162_CLAIM_STALE_S", "3600"))
CLAIM_POLL_S = float(os.environ.get("EPM_2162_CLAIM_POLL_S", "30"))

# Distinct rcs: a designed halt is never an anonymous rc=1 (#1415).
RC_OK = 0
RC_INJECTION_GATE = 21
RC_PILOT_GATE = 22
RC_DEGENERACY_GATE = 23

SMOKE_PAIRS_PER_CELL = 2
SMOKE_GRID_DRAWS = 2


# ── pure helpers (CPU-only, unit-tested in tests/test_issue2162_run.py) ──


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
    """The 234 grid blocks: 39 type-cells x 2 slots x 3 arms (plan §4.3)."""
    by_cell = BANK.pairs_by_cell(pairs)
    blocks: list[Block] = []
    for cell in BANK.all_cells():
        ids = tuple(p.pair_id for p in by_cell[cell])
        assert len(ids) == 36, (cell, len(ids))
        for slot in SLOTS:
            for arm in ARMS:
                blocks.append(Block(cell, slot, arm, ids))
    assert len(blocks) == 39 * len(SLOTS) * len(ARMS) == 234, len(blocks)
    keys = [b.key for b in blocks]
    assert len(set(keys)) == len(keys), "duplicate block keys"
    return blocks


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
    """A claim is stale when its owner is provably dead (same-host pid probe)
    or its age exceeds ``CLAIM_STALE_S`` (the cross-host fallback key)."""
    now = time.time() if now is None else now
    if rec.get("host") == socket.gethostname() and not _pid_alive(int(rec.get("pid", -1))):
        return True
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
        except (json.JSONDecodeError, OSError) as e:
            raise RuntimeError(
                f"unparseable claim file {path} — inconsistent claim state, refusing "
                "to guess (delete it manually after diagnosing the writer)"
            ) from e
        if not _claim_stale(rec):
            return False
        tmp = path.parent / f"{path.name}.tmp.{token}"
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, path)
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
    """Release OUR claim after the done-checkpoint landed (token-verified)."""
    path = cdir / f"{block.slug}.claim"
    assert path.exists(), f"claim vanished before release: {path}"
    rec = json.loads(path.read_text())
    assert rec.get("token") == token, (
        f"claim {block.key} owned by another worker at release time "
        f"(token {rec.get('token')!r} != ours) — inconsistent claim state"
    )
    path.unlink()


def run_claim_queue(
    cfg: RunConfig,
    blocks: list[Block],
    regime_fp: str,
    namespace: str,
    run_one,
) -> dict:
    """Work-conserving queue: pull the next unclaimed pending block until every
    block is done (crashed workers' claims go stale and are reclaimed)."""
    cdir = claims_dir(cfg.out_root, namespace)
    stats = {"ran": 0, "skipped_done": 0, "waits": 0}
    while True:
        ran_this_scan = 0
        n_open = 0
        for block in blocks:
            if block_is_done(cfg.out_root, block, regime_fp, namespace):
                continue
            n_open += 1
            token = uuid.uuid4().hex
            if not try_claim(cdir, block, cfg.worker_index, token):
                continue
            try:
                if block_is_done(cfg.out_root, block, regime_fp, namespace):
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


def bank_manifest_and_sha() -> tuple[dict, str]:
    """Deterministic bank manifest + its sha (the regime key), CPU-only."""
    manifest = BANK.bank_manifest_2162()
    bank_bytes = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    return manifest, _sha256_bytes(bank_bytes)


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


def _anchor_batch_done(cfg: RunConfig, regime_fp: str, batch: str, expected_draws: int) -> bool:
    """Per-worker per-batch anchors resume predicate (gate slice vs rest)."""
    rec = _phase_done_record(cfg, f"anchors_{batch}_w{cfg.worker_index}", regime_fp)
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


def slot_position(ctx_len: int, prefix_end: int, slot: str) -> int:
    """The single edit position (UNPADDED context coordinates) for one slot.

    ``ce`` = the last context token (generation prompt included — the #779
    slot); ``pe`` = the last prefix token (start of the FINAL user turn − 1).
    """
    assert slot in SLOTS, slot
    assert 1 <= prefix_end < ctx_len, (ctx_len, prefix_end)
    if slot == "ce":
        return ctx_len - 1
    return prefix_end - 1


def cap_hit(n_completion_tokens: int, max_new_tokens: int) -> bool:
    """Cap-hit telemetry from the re-tokenized completion length (the
    ``generate_batch`` decoded-text proxy, recorded as ``cap_hit_basis``)."""
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
    grid_draws: int
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
    pools_path: Path | None

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
    def layers(self) -> list[int]:
        return list(range(self.n_layers))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #2162 pod driver (bank / anchors / grid / margin / upload)."
    )
    ap.add_argument(
        "--phase",
        choices=("bank", "anchors", "grid", "margin", "upload"),
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
    ap.add_argument("--grid-draws", type=int, default=GRID_DRAWS)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--smoke", action="store_true", help="tiny per-arm-class slice")
    ap.add_argument("--pilot", action="store_true", help="grid: timing pilot only")
    ap.add_argument(
        "--force",
        action="store_true",
        help="override gate refusals; on --phase bank|anchors, re-run a completed phase",
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
    ap.add_argument(
        "--pools",
        type=Path,
        default=None,
        help="margin pools JSON (judge-built); grid computes margins inline when present",
    )
    ap.add_argument("--planned-wall-h", type=float, default=PLANNED_GRID_WALL_H)
    ap.add_argument("--gpu-hours-budgeted", type=float, default=54.0)
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
        worker_index=args.worker_index,
        num_workers=args.num_workers,
        upload_mode=args.upload,
        upload_every=args.upload_every,
        planned_wall_h=args.planned_wall_h,
        gpu_hours_budgeted=args.gpu_hours_budgeted,
        pools_path=args.pools,
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
    ctx_ids = {cid: BANK.context_token_ids_2162(tok, c) for cid, c in contexts.items()}
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
            assert 1 <= pe < ctx_len, (cid, ctx_len, pe)
            v_ce = torch.stack([captured[layer][j, ctx_len - 1] for layer in layers])
            v_pe = torch.stack([captured[layer][j, pe - 1] for layer in layers])
            assert v_ce.shape == (len(layers), cfg.hidden), v_ce.shape
            ctx = contexts[cid]
            records[cid] = {
                "context_id": cid,
                "cell": ctx["cell"],
                "value_id": ctx["value_id"],
                "carrier": ctx["carrier"],
                "ctx_len": ctx_len,
                "prefix_end": pe,
                "v_ce": v_ce.float().cpu(),
                "v_pe": v_pe.float().cpu(),
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


def run_degeneracy_guard(bank: dict, pairs: list[BANK.Pair2162]) -> dict:
    """Plan §7 gate 2 — realized state identity vs the span-locus registry.

    The two pre-declared degenerate cells (`query_content`,
    `persona_role_header` — final-query / generation-header loci) assert
    v_pe(A) ≈ v_pe(B) (identical token prefixes up to bf16 batch jitter);
    EVERY other (pair x slot) asserts distinct states at BOTH slots. A
    mismatch is a BANK defect (HALT), never a runtime m adjustment.
    """
    recs = bank["per_context"]
    violations: list[dict] = []
    n_checked = 0
    for pair in pairs:
        ra, rb = recs[pair.a], recs[pair.b]
        pe_cos = float(safe_cosine(ra["v_pe"].flatten(), rb["v_pe"].flatten()))
        ce_cos = float(safe_cosine(ra["v_ce"].flatten(), rb["v_ce"].flatten()))
        degenerate_pe = BANK.base_type_of(pair.cell) in BANK.DEGENERATE_AT_PE
        n_checked += 1
        ok_pe = (pe_cos >= DEGENERACY_COS_MIN) if degenerate_pe else (pe_cos < DEGENERACY_COS_MIN)
        ok_ce = ce_cos < DEGENERACY_COS_MIN
        if not (ok_pe and ok_ce):
            violations.append(
                {
                    "pair_id": pair.pair_id,
                    "cell": pair.cell,
                    "declared_degenerate_pe": degenerate_pe,
                    "pe_cos": pe_cos,
                    "ce_cos": ce_cos,
                    "flag": "degenerate_self",
                }
            )
    report = {
        "criterion": "span-locus degeneracy guard (plan §7 gate 2)",
        "bar_cos": DEGENERACY_COS_MIN,
        "declared_degenerate_cells": sorted(BANK.DEGENERATE_AT_PE),
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
    recipient = _slot_state(recs[pair.b], slot).unsqueeze(0)  # (1, L, H)
    if arm == "steered":
        return recipient.clone(), None
    donor_map = donor_maps["shuffled" if arm == "shuffled" else "crosstype"]
    donor_id = donor_map[pair.pair_id]
    donor = pairs_by_id[donor_id]
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


def _gate_spot_specs(pairs: list[BANK.Pair2162]) -> list[dict]:
    """12 spot cells spanning slot x arm x carrier-class / crossed / degenerate
    cell classes (plan §7 gate 1)."""
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
) -> dict:
    """Plan §7 gate 1 — the realized edit equals the intended donor state at
    the intended (row, position, layer) and NOWHERE else.

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
    contexts = BANK.build_contexts()
    ctx_ids = {cid: BANK.context_token_ids_2162(tok, c) for cid, c in contexts.items()}
    pad_id = tok.pad_token_id
    recs = bank["per_context"]
    pairs_by_id = {p.pair_id: p for p in pairs}
    spots = _gate_spot_specs(pairs)
    results: list[dict] = []
    for spot in spots:
        pair: BANK.Pair2162 = spot["pair"]
        slot, arm = spot["slot"], spot["arm"]
        # Second row: a DIFFERENT pair whose A-context length differs, so the
        # padded-offset math is exercised rather than degenerate.
        others = [
            p
            for p in pairs
            if p.pair_id != pair.pair_id and len(ctx_ids[p.a]) != len(ctx_ids[pair.a])
        ]
        batch_pairs = [pair] + ([others[0]] if others else [])
        rows = [ctx_ids[p.a] for p in batch_pairs]
        row_lengths = [len(r) for r in rows]
        positions: list[tuple[int, ...]] = []
        payloads: list[torch.Tensor] = []
        donor_ids: list[str | None] = []
        for p in batch_pairs:
            payload, donor_id = payload_for_arm(bank, p, slot, arm, donor_maps, pairs_by_id)
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
    manifest["bank_sha"] = bank_sha
    manifest["repro"] = _repro(cfg)
    _write_json_atomic(cfg.bank_dir / "bank.json", manifest)

    bank = capture_bank(cfg, model, tok)
    pairs = BANK.build_pairs()
    donor_maps = BANK.donor_assignment_2162(pairs)
    _save_pt_atomic(
        cfg.bank_dir / "vc_bank.pt",
        {
            "layers": bank["layers"],
            "per_context": bank["per_context"],
            "donor_assignments": donor_maps,
            "bank_sha": bank_sha,
            "repro": _repro(cfg),
        },
    )
    logger.info("[bank] captured %d contexts x %d layers", len(bank["per_context"]), cfg.n_layers)

    # Gate 2 first (state identity — a bank defect makes gate 1 meaningless).
    degeneracy = run_degeneracy_guard(bank, pairs)
    degeneracy["repro"] = _repro(cfg)
    _write_json_atomic(cfg.bank_dir / "degeneracy_report.json", degeneracy)
    if not degeneracy["passed"]:
        logger.error(
            "[degeneracy_guard] FAILED: %d/%d pair-slot identity violations (bank defect)",
            degeneracy["n_violations"],
            degeneracy["n_pairs_checked"],
        )
        if not cfg.force:
            return RC_DEGENERACY_GATE
        logger.error("[degeneracy_guard] --force set: proceeding on a FAILED guard (recorded)")

    report = run_injection_gate(cfg, model, tok, bank, pairs, donor_maps)
    _write_json_atomic(cfg.bank_dir / "injection_gate_report.json", report)
    if not report["passed"]:
        logger.error(
            "[injection_gate] FAILED: %d/%d spots failed",
            report["n_spots_failed"],
            report["n_spots"],
        )
        if not cfg.force:
            return RC_INJECTION_GATE
        logger.error("[injection_gate] --force set: proceeding on a FAILED gate (recorded)")
    _write_json_atomic(
        cfg.manifest_dir / "bank_done.json",
        {
            "regime_fp": regime_fp,
            "bank_sha": bank_sha,
            "n_contexts": len(bank["per_context"]),
            "injection_gate_passed": bool(report["passed"]),
            "degeneracy_gate_passed": bool(degeneracy["passed"]),
            "forced_past_gate": bool(cfg.force and not (report["passed"] and degeneracy["passed"])),
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
) -> dict:
    """Span-mean answer states from teacher-forced re-forwards.

    ``va_span`` = mean over the COMPLETION token positions at every layer.
    With ``payloads`` given, each row's forward runs with the all-layer
    replace hook armed at that row's slot position (the PATCHED condition the
    rollout was generated under — plan §4.4 "hooked teacher-forced
    re-forward"), using the RIGHT-pad ``row_lengths=[T]*B`` arming that the
    injection gate's capture leg verifies. Rows are built by concatenating
    per-segment TOKEN IDS (BPE-seam rule).
    """
    assert len(ctx_ids_by_row) == len(completions), (len(ctx_ids_by_row), len(completions))
    hooked = payloads is not None
    if hooked:
        assert positions is not None and len(payloads) == len(positions) == len(completions)
    layers = cfg.layers
    pad_id = tok.pad_token_id
    n = len(completions)
    va_span = torch.zeros((n, len(layers), cfg.hidden), dtype=torch.float32)
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
            stack = _arm_hook_all_layers(
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
        del captured
    return {
        "va_span": va_span.to(torch.float16),
        "n_completion_tokens": n_comp_tokens,
        "empty_rows": sorted(empty),
        "pooling": {"va_span": "mean over completion tokens (plan §4.4 span-mean V_a)"},
    }


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
    pairs = BANK.build_pairs()
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
    rows: list[dict] = []
    flat_ctx: list[list[int]] = []
    flat_text: list[str] = []
    ctx_ids = {cid: BANK.context_token_ids_2162(tok, contexts[cid]) for cid in order}
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
            render_fn=BANK.render_context_2162,
            ids_fn=BANK.context_token_ids_2162,
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
    states = capture_answer_states(cfg, model, tok, flat_ctx, flat_text, eot)
    for r, n_tok in zip(rows, states["n_completion_tokens"], strict=True):
        r["n_completion_tokens"] = n_tok
        r["cap_hit"] = cap_hit(n_tok, cfg.max_new_tokens)
        r["cap_hit_basis"] = "retokenized_completion_len >= max_new_tokens"
    jsonl = cfg.anchors_dir / f"anchors_{batch}_w{cfg.worker_index}.jsonl"
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
            "n_contexts": len(order),
            "draws": draws,
            "n_rows": len(rows),
            "n_cap_hit": cap_hits,
            "n_empty": len(states["empty_rows"]),
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
                    "n_contexts": 0,
                    "draws": draws,
                    "n_rows": 0,
                    "n_cap_hit": 0,
                    "n_empty": 0,
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
                    "n_contexts": 0,
                    "draws": draws,
                    "n_rows": 0,
                    "n_cap_hit": 0,
                    "n_empty": 0,
                    "repro": _repro(cfg),
                },
            )
    else:
        logger.info("[anchors:rest] already done for this regime — skipping")
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
) -> dict:
    """One block: K hooked temp-1.0 draws per pair, the hooked V_a pass, and
    (pools present) the margin TF pass — pipelined on the same GPU."""
    cells = _block_cells(bank, block, pairs_by_id, donor_maps)

    def ids_for(cid: str) -> list[int]:
        if cid not in ctx_ids_cache:
            ctx_ids_cache[cid] = BANK.context_token_ids_2162(tok, contexts[cid])
        return ctx_ids_cache[cid]

    texts_per_cell: list[list[str]] = []
    for start in range(0, len(cells), cfg.gen_batch):
        chunk = cells[start : start + cfg.gen_batch]
        ctx_list = [contexts[c["context_a"]] for c in chunk]
        rows = [ids_for(c["context_a"]) for c in chunk]
        row_lengths = [len(r) for r in rows]
        t_pad = max(row_lengths)
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
                render_fn=BANK.render_context_2162,
                ids_fn=BANK.context_token_ids_2162,
            )
        finally:
            stack.remove()
        assert len(outs) == len(chunk), (len(outs), len(chunk))
        texts_per_cell.extend(list(o) for o in outs)
    assert len(texts_per_cell) == len(cells)

    # Hooked V_a: one flattened (pair x draw) row set, each row armed with its
    # pair's payload at its pair's position.
    flat_ctx: list[list[int]] = []
    flat_text: list[str] = []
    flat_payload: list[torch.Tensor] = []
    flat_pos: list[int] = []
    for c, texts in zip(cells, texts_per_cell, strict=True):
        for text in texts:
            flat_ctx.append(ids_for(c["context_a"]))
            flat_text.append(text)
            flat_payload.append(c["payload"])
            flat_pos.append(c["position"])
    states = capture_answer_states(
        cfg, model, tok, flat_ctx, flat_text, eot, payloads=flat_payload, positions=flat_pos
    )

    rows_out: list[dict] = []
    k = 0
    for c, texts in zip(cells, texts_per_cell, strict=True):
        pair: BANK.Pair2162 = c["pair"]
        for i, text in enumerate(texts):
            n_tok = states["n_completion_tokens"][k]
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
                    "draw": i,
                    "seed": cfg.seed_base + i,
                    "temperature": GRID_TEMPERATURE,
                    "n_completion_tokens": n_tok,
                    "cap_hit": cap_hit(n_tok, cfg.max_new_tokens),
                    "cap_hit_basis": "retokenized_completion_len >= max_new_tokens",
                    "text": text,
                }
            )
            k += 1
    _write_jsonl_atomic(cfg.rollouts_dir / f"shard_{block.slug}.jsonl", rows_out)
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
        "margin_inline": margin_done,
        "repro": _repro(cfg),
    }
    _write_json_atomic(block_done_path(cfg.out_root, block), done)
    return done


def phase_grid(cfg: RunConfig) -> int:
    """P3: claim-queue block execution (or the timing pilot under ``--pilot``)."""
    logger.info("[phase=grid] worker=%d/%d smoke=%s", cfg.worker_index, cfg.num_workers, cfg.smoke)
    bank = _load_bank(cfg)
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_maps = bank["donor_assignments"]
    regime_fp = regime_fingerprint(cfg, str(bank.get("bank_sha")))
    draws = SMOKE_GRID_DRAWS if cfg.smoke else cfg.grid_draws

    all_blocks = enumerate_blocks(pairs)
    totals_all = grid_totals(all_blocks, cfg.grid_draws)
    blocks = smoke_blocks(pairs) if cfg.smoke else all_blocks
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
    uploaded: list[str] = []
    pending: list[Block] = []

    def run_one(block: Block) -> None:
        nonlocal ran_rollouts, ran_wall, n_run, uploaded, pending
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
        # The pilot times ONE production-shape block through this entrypoint
        # (no claim, no done-file — a pilot never poisons the resume state).
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
        )
        ran_wall = time.monotonic() - t0
        ran_rollouts = rec["n_rows"]
        return _enforce_pilot_gate(cfg, totals_all, ran_rollouts, ran_wall)

    stats = run_claim_queue(cfg, blocks, regime_fp, "blocks", run_one)
    if pending:
        uploaded += _upload_grid_increment(cfg, pending)
        pending.clear()
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
    """Plan §7 pilot gate — a DESIGNED halt on a >3x-plan projected pod wall.

    The pilot ran ONE production-shape block (36 pairs x K draws + V_a +
    margin) through THIS entrypoint, so the measured per-rollout wall is at
    the sweep's execution shape (#1415 batch-1 false-fire lesson).
    """
    assert ran_rollouts > 0, "pilot ran no rollouts"
    per_rollout = ran_wall / ran_rollouts
    width = max(1, cfg.num_workers)
    projected_h = per_rollout * totals_all["rollouts_total"] / width / 3600.0
    fence_h = PILOT_FENCE_MULT * projected_h
    refuse = projected_h > PILOT_REFUSAL_MULT * cfg.planned_wall_h
    report = {
        "criterion": "generation-throughput pilot (plan §7)",
        "measured_rollouts": ran_rollouts,
        "measured_wall_s": ran_wall,
        "s_per_rollout": per_rollout,
        "gen_batch": cfg.gen_batch,
        "num_workers": width,
        "rollouts_total": totals_all["rollouts_total"],
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
        "[phase=pilot_done] s_per_rollout=%.3f projected_pod_wall_h=%.2f (planned %.2f) "
        "fence_h=%.2f sweep_allowed=%s",
        per_rollout,
        projected_h,
        cfg.planned_wall_h,
        fence_h,
        not refuse,
    )
    if refuse and not cfg.force:
        logger.error(
            "[pilot_gate] projected pod wall %.2f h > %.1fx planned %.2f h — refusing the grid "
            "(pass --force to override, or descope per the plan §9 ladder)",
            projected_h,
            PILOT_REFUSAL_MULT,
            cfg.planned_wall_h,
        )
        return RC_PILOT_GATE
    return RC_OK


# ── margin phase (pools-dependent TF legs) ────────────────────────────


def phase_margin(cfg: RunConfig) -> int:
    """Margin TF: (a) anchor margins (contexts sharded across workers) and
    (b) the per-block catch-up for blocks whose grid pass ran before the pools
    file landed (claim-queue namespace ``margin``)."""
    logger.info(
        "[phase=margin] worker=%d/%d smoke=%s", cfg.worker_index, cfg.num_workers, cfg.smoke
    )
    assert cfg.pools_path is not None and cfg.pools_path.exists(), (
        f"--pools file required for --phase margin (got {cfg.pools_path}) — the pools are "
        "judge-built from the gate-3 slice and staged by the orchestrator"
    )
    pools = load_pools(cfg.pools_path)
    bank = _load_bank(cfg)
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_maps = bank["donor_assignments"]
    regime_fp = regime_fingerprint(cfg, str(bank.get("bank_sha")))
    draws = SMOKE_GRID_DRAWS if cfg.smoke else cfg.grid_draws
    model, tok = load_model_and_tokenizer(cfg)
    contexts = BANK.build_contexts()
    ctx_ids_cache: dict[str, list[int]] = {}

    def ids_for(cid: str) -> list[int]:
        if cid not in ctx_ids_cache:
            ctx_ids_cache[cid] = BANK.context_token_ids_2162(tok, contexts[cid])
        return ctx_ids_cache[cid]

    # (a) anchor margins — unhooked TF of each context's relevant pools.
    done_path = cfg.manifest_dir / f"margin_anchors_w{cfg.worker_index}_done.json"
    if (
        cfg.force
        or _phase_done_record(cfg, f"margin_anchors_w{cfg.worker_index}", regime_fp) is None
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
                "n_rows": len(out_rows),
                "repro": _repro(cfg),
            },
        )
    else:
        logger.info("[margin:anchors] already done for this regime — skipping")

    # (b) per-block catch-up via the claim queue (skip blocks already
    # margin-done — inline grid margins wrote the same done files).
    blocks = smoke_blocks(pairs) if cfg.smoke else enumerate_blocks(pairs)
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


def _upload_dir(
    cfg: RunConfig,
    local_dir: Path,
    remote_prefix: str,
    allow_patterns: list[str],
) -> list[str]:
    """ONE bulk ``upload_folder`` commit for a matched subset + exact-set verify.

    FAIL-LOUD: ``_upload_folder_filtered`` returns ``""`` on failure — this
    seam retries the no-path return with bounded jittered backoff (uploads are
    idempotent), then RAISES on exhaustion so the results sentinel can never
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
    the 256/hr cap.
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
    n_margin_shards = len(list(cfg.margin_dir.glob("*.jsonl")))
    n_anchor_rows = 0
    for jsonl in sorted(cfg.anchors_dir.glob("anchors_*.jsonl")):
        n_anchor_rows += sum(1 for line in jsonl.open(encoding="utf-8") if line.strip())
    gate_path = cfg.bank_dir / "injection_gate_report.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    degen_path = cfg.bank_dir / "degeneracy_report.json"
    degen = json.loads(degen_path.read_text()) if degen_path.exists() else {}
    cap_hits, rows_total = 0, 0
    for done in sorted((cfg.manifest_dir / "blocks").glob("*.done.json")):
        rec = json.loads(done.read_text())
        cap_hits += int(rec.get("n_cap_hit", 0))
        rows_total += int(rec.get("n_rows", 0))
    return {
        "eval_numbers": {
            "grid_shards": n_grid_shards,
            "va_shards": n_va_shards,
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
        },
        "wandb_url": None,
        "hf_hub_url": f"https://huggingface.co/datasets/{HF_DATA_REPO}/tree/main/{HF_PREFIX}",
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
    # blocks/*.done.json + claim residue ride along: per-block resume state +
    # the sentinel's cap-hit provenance become durable off-pod.
    uploaded["manifests"] = _upload_dir(
        cfg,
        cfg.manifest_dir,
        f"{HF_PREFIX}/analysis_tensors/manifests",
        ["*.json", "blocks/*.done.json", "margin_blocks/*.done.json"],
    )
    payload = _sentinel_payload(cfg, uploaded)
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
    if cfg.phase == "margin":
        return phase_margin(cfg)
    assert cfg.phase == "upload", cfg.phase
    return phase_upload(cfg)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension interpreter finalization (#1689).
    sys.exit(rc)
