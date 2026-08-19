#!/usr/bin/env python3
"""Issue #2215 — pod driver for the minimal-pair representation-shift reads.

Pre-split units 1+2 of 3: Phases A→D (plan §4.1–§4.3 + §9 phase order).
``--phase all`` runs the full chain and emits the terminal ``[phase=done]``
breadcrumb only after Phase D's final sentinel write.

- **Phase A (``--phase a``)** — stage the 5 reused artifact families from the
  HF data repo at the plan's revision pin (scoped ``list_repo_tree`` +
  per-file ``hf_hub_download`` via the canonical ``stage_hub_prefix`` /
  ``stage_hub_file`` helpers; #833 rule) into ``--staged-root``, then run the
  fail-loud coverage gates (plan §4.1 ``gate_coverage``: 1,404 contexts,
  K=10, no dup (context_id, draw), jsonl↔shard keyset equality, graceful
  n_valid floor with per-cell reporting, 9 ridge-payload key/shape asserts,
  ``apply_map`` roundtrip probe, ``get_paths_info`` provenance-date check,
  realized-keys probe on all 9 payloads). Ends with the ``stage_done.json``
  sentinel under ``--out-root`` (plan §9 ``phase_outputs``).
- **Phase B (``--phase b``)** — teacher-forced answer-capture twin over the
  banked anchor rollouts, reusing the parent's ``capture_answer_states``
  (``scripts/issue2162_run.py``) with the ONE new capture extension
  (``tail_inclusive=True`` — plan §4.2): both poolings from one forward.
  Per-shard outputs mirror the parent's 16-shard layout, resume-safe on
  per-shard done manifests. Includes the throughput pilot gate (§7 gate 1),
  the per-shard pooling-parity gate (§7 gate 2), and the A7 pe-slot
  token-parity check (§12, recorded — never a halt). Phase B ENDS with the
  fail-loud ``upload_folder`` commit of the va2215 store to HF + a scoped
  landing verification + the ``va2215_uploaded.json`` sentinel — Phase C
  is gated on that sentinel (critic round-1 Must-Fix ordering,
  #825 store-before-long-fit).
- **Phase C (``--phase c``)** — the DV1/DV2/DV3 + nulls + baselines analysis
  battery (plan §4.3), implemented in ``scripts/issue2215_analysis.py``.
  Gate order: (1) ``va2215_uploaded.json`` presence + ``regime_fp`` match
  (concern ``unit2-phase-c-sentinel-gate`` — the #825 store-before-long-fit
  ordering: the capture store must be durably on HF before the long CPU
  analysis consumes it); (2) ``stage_done.json`` + fp match; (3)
  ``analysis_done.json`` idempotent skip on ``analysis_fp`` match / fail-loud
  refuse on mismatch (``--force`` re-runs). Outputs land in
  ``results_dir`` (production: repo ``eval_results/issue_2215``; smoke/tiny:
  an out-root twin so committed paths are never touched) + per-draw null
  matrices under ``out_root/null_matrices``.
- **Phase D (``--phase d``)** — null-matrices ``upload_folder`` to HF
  (``analysis_tensors/null_matrices``) + scoped landing verify; figure render
  from the Phase-C outputs (``issue2215_figures.render_all`` — production:
  repo ``figures/issue_2215``; smoke/tiny: an out-root twin); production
  git add/commit/push of ``eval_results/issue_2215`` + ``figures/issue_2215``
  per the #1880 result-push contract; the poller results sentinel
  (``epm:smoke-result`` under smoke/tiny, else ``epm:results``); then the
  ``upload_done.json`` sentinel (plan §9 ``phase_outputs``).

Pod-side contract: ``[phase=...]`` breadcrumbs; resume/finalize state
lives under ``--out-root`` (NEVER the drained ``/workspace/logs`` sentinel
namespace — pod-side-reporting.md req. 3 DEFAULT; the Phase-D poller results
sentinel is the one WRITE into ``--sentinel-dir``, and this driver never
READS it back); this file never shells out
to ``scripts/task.py``; every exit path is an explicit ``sys.exit`` (#1689
C-extension finalization race). ``[phase=done]`` is emitted in ``main`` only
after the terminal phase (``d``/``all``) returns rc=0, i.e. after the final
sentinel write.

Smoke = production with ``--cells <cell,...>`` (plan §4.4 parity row): same
entrypoint, same staging, same full-grain Phase-A gates (the full artifacts
are staged either way), same capture/upload code paths — the slice narrows
only WHICH anchor rows Phase B captures, the out-root rebinds to a derived
smoke root (per-leg out-roots, #1333), and the HF prefix gains a ``/smoke``
segment so smoke shards can never clobber production paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch import (shared-VM thread caps + HF token)

import numpy as np  # noqa: E402
import torch  # noqa: E402

# scripts/ sibling imports resolve in script mode (sys.path[0] is scripts/);
# the insert covers the imported-as-module case (pytest, repo-root -c probes).
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2162_run as R2162  # noqa: E402
from explore_persona_space.experiments.issue2162 import bank2162 as BANK  # noqa: E402
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom  # noqa: E402

logger = logging.getLogger("issue2215.run")

# ── constants (plan §4.1/§7/§9/§10) ───────────────────────────────────

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# The parent report's verified artifact revision (plan §10; staged AT the pin
# so a 404 fails loud before any GPU spend — §12 A6).
REVISION_PIN = "dc8108ab84f33695bbc769da0e6e8e2327f51eeb"
PARENT_PREFIX = "issue2162_ctxinfo"
HF_PREFIX_2215 = "issue2215_reprshift"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HIDDEN_FULL = 3584
N_MODEL_LAYERS_FULL = 28
N_CONTEXTS = 1404
N_PAIRS = 1404
N_CELLS = 39
PAIRS_PER_CELL = 36
K_DRAWS = 10
MAP_LAYERS = (14, 19, 26)
DECLARED_RIDGE_KEYS = ("kind", "xmu", "xsd", "ymu", "W")

VC_BANK_PREFIX = f"{PARENT_PREFIX}/analysis_tensors/vc_bank"
ANCHOR_TENSOR_PREFIX = f"{PARENT_PREFIX}/analysis_tensors/anchors"
ANCHOR_TEXT_PREFIX = f"{PARENT_PREFIX}/raw_completions/anchors"
RIDGE_779_PATHS = tuple(
    f"issue779_monitoring/n1m_readout/weights/L{layer}/ridge.pt" for layer in MAP_LAYERS
)
RIDGE_1738_PATHS = tuple(
    f"issue1738_multiturn/analysis_tensors/weights/L{layer}/{kind}_ridge.pt"
    for layer in MAP_LAYERS
    for kind in ("context", "prefix")
)
# (batch, worker) mirror of the parent's 16-shard anchors layout.
SHARDS: tuple[tuple[str, int], ...] = tuple(
    (batch, w) for batch in ("gate", "rest") for w in range(8)
)

DEFAULT_STAGED_ROOT = Path("/workspace/eps2215/staged")
DEFAULT_OUT_ROOT = Path("/workspace/eps2215/out")

# Plan §9 disk rows (out-root mount binding): floors per write-heavy phase.
HEADROOM_GB = {"A_staging": 6.0, "B_capture": 8.0}

# §7 gate 1: proceed only if projected full-capture wall <= this ceiling.
PILOT_WALL_CEILING_H = 2.5
# §7 gate 2 provisional bar (smoke-calibrated; CLI-overridable): flattened
# all-layer fp32 cosine over span-mean summaries — span means smooth bf16
# padded-batch jitter, so the flat 0.995 bar keeps ~4x headroom over the
# measured worst-case bf16 deviation while sitting far above the real-bug
# regime (~0.39-0.84) — gotchas.md bf16 equivalence-gate calibration.
PARITY_COS_MIN = 0.995
PARITY_FRAC_MIN = 0.99

POOLING_VERSION = "span_excl+tail_incl_v1"

# Distinct rcs: a designed halt is never an anonymous rc=1 (#1415).
# (RC 24 was unit 1's RC_UNIT_SPLIT designed halt — retired when unit 2
# landed Phases C/D.)
RC_OK = 0
RC_PILOT_GATE = 22
RC_PARITY_GATE = 23

_write_json_atomic = R2162._write_json_atomic
_save_pt_atomic = R2162._save_pt_atomic


# ── config ────────────────────────────────────────────────────────────


@dataclass
class RunConfig2215:
    """Driver config. Field subset duck-typed for the reused parent helpers
    (``R2162.load_model_and_tokenizer`` reads model_id/tiny/hidden/n_layers/
    device; ``R2162.capture_answer_states`` reads layers/hidden/capture_batch/
    device; ``R2162._repro`` reads model_id/tiny/smoke/n_layers — attribute
    reads audited against the callee bodies, gotchas.md reused-module rule)."""

    phase: str
    staged_root: Path
    out_root: Path
    model_id: str
    tiny: bool
    n_layers: int
    hidden: int
    device: str
    capture_batch: int
    cells: tuple[str, ...] | None
    smoke: bool
    upload_mode: str  # "hf" | "none"
    parity_cos_min: float
    parity_frac_min: float
    pilot_ceiling_h: float
    null_b: int  # Phase C null-band draws (plan §6; smoke argv passes --null-b 100)
    boot_b: int  # Phase C bootstrap draws (plan §6; seed 21620)
    anchors_jsonl: Path | None  # H2 parent input; None -> repo default resolution
    results_dir_arg: Path | None  # explicit override; None -> derived (see results_dir)
    sentinel_dir_arg: Path | None  # poller sentinel dir; None -> derived (see sentinel_dir)
    force: bool

    @property
    def layers(self) -> list[int]:
        return list(range(self.n_layers))

    @property
    def va_dir(self) -> Path:
        return self.out_root / "va2215"

    @property
    def manifest_dir(self) -> Path:
        return self.out_root / "manifests"

    @property
    def hf_prefix(self) -> str:
        # Smoke shards must never clobber production HF paths (same upload
        # CODE path either way — only the destination prefix differs).
        return f"{HF_PREFIX_2215}/smoke" if self.cells else HF_PREFIX_2215

    @property
    def results_dir(self) -> Path:
        """Phase C JSON/jsonl output root. Production: the repo checkout's
        ``eval_results/issue_2215`` (Phase D commits it from the pod, plan
        §9). Smoke/tiny: an out-root twin — smoke outputs never touch
        canonical committed paths (checklist item 3)."""
        if self.results_dir_arg is not None:
            return self.results_dir_arg
        if self.smoke or self.tiny:
            return self.out_root / "eval_results_smoke" / "issue_2215"
        return _repo_root() / "eval_results" / "issue_2215"

    @property
    def null_dir(self) -> Path:
        return self.out_root / "null_matrices"

    @property
    def sentinel_dir(self) -> Path:
        """Poller sentinel namespace (`/workspace/logs` on pods; an out-root
        twin on non-pod hosts so tiny smokes never need /workspace)."""
        if self.sentinel_dir_arg is not None:
            return self.sentinel_dir_arg
        if Path("/workspace").is_dir():
            return Path("/workspace/logs")
        return self.out_root / "sentinels"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #2215 pod driver (a=staging+gates / b=capture twin / "
        "c=DV1-DV3 analysis / d=upload+commit+sentinels)."
    )
    ap.add_argument(
        "--phase",
        choices=("a", "b", "c", "d", "all"),
        help="pipeline phase (required unless --import-check); 'all' runs a->b->c->d",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import + the argparse-attribute assert, then exit 0",
    )
    ap.add_argument("--staged-root", type=Path, default=DEFAULT_STAGED_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--tiny", action="store_true", help="from-config tiny CPU model (smoke)")
    ap.add_argument("--tiny-layers", type=int, default=4)
    ap.add_argument("--tiny-hidden", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda | cuda:0 | cpu (default: auto)")
    ap.add_argument(
        "--capture-batch",
        type=int,
        default=8,
        help="rows per teacher-forced forward (parent-realized default; plan §11)",
    )
    ap.add_argument(
        "--cells",
        default=None,
        help="comma-separated cell slice (smoke mode: rebinds out-root + HF /smoke prefix)",
    )
    ap.add_argument("--upload", choices=("hf", "none"), default="hf")
    ap.add_argument("--parity-cos-min", type=float, default=PARITY_COS_MIN)
    ap.add_argument("--parity-frac-min", type=float, default=PARITY_FRAC_MIN)
    ap.add_argument("--pilot-ceiling-h", type=float, default=PILOT_WALL_CEILING_H)
    ap.add_argument(
        "--null-b",
        type=int,
        default=10_000,
        help="Phase C null-band draws (plan §6; smoke argv: --null-b 100)",
    )
    ap.add_argument(
        "--boot-b",
        type=int,
        default=10_000,
        help="Phase C bootstrap draws (plan §6; seed 21620)",
    )
    ap.add_argument(
        "--anchors-jsonl",
        type=Path,
        default=None,
        help="parent f_metrics/anchors.jsonl for H2 (default: the repo checkout's "
        "eval_results/issue_2162/f_metrics/anchors.jsonl)",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Phase C output root override (default: repo eval_results/issue_2215 in "
        "production; an out-root twin under smoke/tiny)",
    )
    ap.add_argument(
        "--sentinel-dir",
        type=Path,
        default=None,
        help="poller sentinel dir (default: /workspace/logs on pods; out_root/sentinels elsewhere)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-run completed phase A gates / B shards / C analysis",
    )
    return ap


def smoke_root_for(out_root: Path, cells: tuple[str, ...]) -> Path:
    """Derived PER-LEG smoke out-root (#1333: derive from the given root even
    when --out-root is explicit, so a shared root never mixes legs)."""
    slug = "-".join(sorted(cells))[:80].replace("/", "_")
    return out_root / f"smoke_{slug}"


def build_config(args: argparse.Namespace) -> RunConfig2215:
    if args.device:
        device = args.device
    elif args.tiny:
        device = "cpu"
    else:
        device = "cuda:0"
    cells = (
        tuple(sorted(c.strip() for c in args.cells.split(",") if c.strip())) if args.cells else None
    )
    out_root = Path(args.out_root)
    if cells:
        rebound = smoke_root_for(out_root, cells)
        logger.info("[config] --cells smoke slice: out-root rebinds %s -> %s", out_root, rebound)
        out_root = rebound
    return RunConfig2215(
        phase=args.phase,
        staged_root=Path(args.staged_root),
        out_root=out_root,
        model_id=args.model_id,
        tiny=args.tiny,
        n_layers=args.tiny_layers if args.tiny else N_MODEL_LAYERS_FULL,
        hidden=args.tiny_hidden if args.tiny else HIDDEN_FULL,
        device=device,
        capture_batch=args.capture_batch,
        cells=cells,
        smoke=cells is not None,
        upload_mode=args.upload,
        parity_cos_min=args.parity_cos_min,
        parity_frac_min=args.parity_frac_min,
        pilot_ceiling_h=args.pilot_ceiling_h,
        null_b=args.null_b,
        boot_b=args.boot_b,
        anchors_jsonl=Path(args.anchors_jsonl) if args.anchors_jsonl else None,
        results_dir_arg=Path(args.results_dir) if args.results_dir else None,
        sentinel_dir_arg=Path(args.sentinel_dir) if args.sentinel_dir else None,
        force=args.force,
    )


def regime_fingerprint(cfg: RunConfig2215) -> str:
    """Resume key over EVERY output-affecting knob (a --cells slice or a
    model/pin change must never reuse another regime's shards — #722 r3)."""
    payload = json.dumps(
        {
            "revision_pin": REVISION_PIN,
            "model_id": cfg.model_id,
            "tiny": cfg.tiny,
            "n_layers": cfg.n_layers,
            "hidden": cfg.hidden,
            "cells": list(cfg.cells) if cfg.cells else None,
            "pooling": POOLING_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _repo_root() -> Path:
    """The repo checkout root (scripts/..): production Phase C writes into its
    eval_results/ tree so Phase D can git-commit from the pod (plan §9)."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts").is_dir(), root
    return root


def analysis_fingerprint(cfg: RunConfig2215) -> str:
    """Phase-C resume key: the capture regime PLUS every analysis knob that
    changes Phase-C outputs (draw counts, seeds). Deliberately separate from
    ``regime_fingerprint`` so changing --null-b/--boot-b re-runs the analysis
    without invalidating the capture store."""
    import issue2215_analysis as ANALYSIS  # deferred sibling (heavy chain)

    payload = json.dumps(
        {
            "regime_fp": regime_fingerprint(cfg),
            "null_b": cfg.null_b,
            "boot_b": cfg.boot_b,
            "seed_null": ANALYSIS.SEED_NULL,
            "seed_boot": ANALYSIS.SEED_BOOT,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ── staged-path helpers ───────────────────────────────────────────────


def staged_path(cfg: RunConfig2215, repo_rel: str) -> Path:
    """Consumers open files at the EXACT fetch destination (verbatim prefix
    mirror; artifact-reuse (h)(iv) 'no staging transformation' escape)."""
    return cfg.staged_root / repo_rel


def shard_tensor_path(cfg: RunConfig2215, batch: str, w: int) -> Path:
    return staged_path(cfg, f"{ANCHOR_TENSOR_PREFIX}/va_anchors_{batch}_w{w}.pt")


def shard_jsonl_path(cfg: RunConfig2215, batch: str, w: int) -> Path:
    return staged_path(cfg, f"{ANCHOR_TEXT_PREFIX}/anchors_{batch}_w{w}.jsonl")


def load_bank_json(cfg: RunConfig2215) -> dict:
    path = staged_path(cfg, f"{VC_BANK_PREFIX}/bank.json")
    assert path.exists(), f"{path} missing — run --phase a staging first"
    return json.loads(path.read_text())


def load_jsonl_rows(path: Path) -> list[dict]:
    """Text-mode line iteration — NEVER splitlines(): real-user completion
    text carries U+2028/U+2029 (gotchas.md #950). Row text is consumed
    programmatically only; nothing here logs content fields."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    assert rows, f"no rows in {path}"
    return rows


# ── Phase A: pure gate helpers (unit-tested CPU-only) ─────────────────


def check_pair_table(pairs: list[dict]) -> dict[str, int]:
    """Fail-loud bank pair-table shape: N_PAIRS total, N_CELLS cells,
    PAIRS_PER_CELL per cell. Returns pairs-per-cell."""
    assert len(pairs) == N_PAIRS, f"expected {N_PAIRS} directed pairs, got {len(pairs)}"
    per_cell = Counter(p["cell"] for p in pairs)
    assert len(per_cell) == N_CELLS, f"expected {N_CELLS} cells, got {len(per_cell)}"
    bad = {c: n for c, n in per_cell.items() if n != PAIRS_PER_CELL}
    assert not bad, f"cells with != {PAIRS_PER_CELL} pairs: {bad}"
    return dict(per_cell)


def check_anchor_keysets(
    index_keys: list[tuple[str, int]],
    jsonl_keys: list[tuple[str, int]],
    expected_ctx_ids: set[str],
    k_draws: int = K_DRAWS,
) -> None:
    """Plan §4.1 coverage asserts over (context_id, draw) keys — every assert
    raises with counts."""
    dup = [k for k, n in Counter(index_keys).items() if n > 1]
    assert not dup, f"{len(dup)} duplicate (context_id, draw) keys in va shards (first: {dup[:3]})"
    dup_j = [k for k, n in Counter(jsonl_keys).items() if n > 1]
    assert not dup_j, f"{len(dup_j)} duplicate keys in anchors jsonl (first: {dup_j[:3]})"
    assert set(index_keys) == set(jsonl_keys), (
        f"jsonl↔shard keyset mismatch: {len(set(index_keys) - set(jsonl_keys))} shard-only, "
        f"{len(set(jsonl_keys) - set(index_keys))} jsonl-only"
    )
    per_ctx = Counter(cid for cid, _ in index_keys)
    assert set(per_ctx) == expected_ctx_ids, (
        f"context coverage mismatch: {len(expected_ctx_ids - set(per_ctx))} missing, "
        f"{len(set(per_ctx) - expected_ctx_ids)} unexpected"
    )
    bad_k = {c: n for c, n in per_ctx.items() if n != k_draws}
    assert not bad_k, (
        f"{len(bad_k)} contexts with != K={k_draws} draws (first: {list(bad_k.items())[:3]})"
    )


def n_valid_by_context(
    keys: list[tuple[str, int]], empty_keys: set[tuple[str, int]], k_draws: int = K_DRAWS
) -> dict[str, int]:
    """Graceful floor (plan §4.1): valid = non-empty draws; a context keeps
    its mean at ANY n_valid >= 1 — exclusion only at 0, reported by caller."""
    per_ctx = Counter(cid for cid, _ in keys)
    empties = Counter(cid for cid, _ in empty_keys)
    return {cid: per_ctx[cid] - empties.get(cid, 0) for cid in per_ctx}


def check_ridge_payload(payload: dict, path: str, expected_layer: int, hidden: int) -> None:
    """Plan §4.1 key/shape asserts for one persisted ridge payload."""
    missing = [k for k in DECLARED_RIDGE_KEYS if k not in payload]
    assert not missing, f"{path}: missing declared keys {missing} (realized: {sorted(payload)})"
    assert payload["kind"] == "ridge", f"{path}: kind={payload['kind']!r} != 'ridge'"
    layer = int(payload["layer"])
    assert layer == expected_layer and layer in MAP_LAYERS, (
        f"{path}: layer={layer} (path says L{expected_layer})"
    )
    assert tuple(payload["W"].shape) == (hidden, hidden), (
        f"{path}: W shape {tuple(payload['W'].shape)} != {(hidden, hidden)}"
    )
    for k in ("xmu", "xsd", "ymu"):
        assert payload[k].reshape(-1).shape[0] == hidden, (
            f"{path}: {k} has {payload[k].reshape(-1).shape[0]} elements != {hidden}"
        )


def rowwise_flat_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(n, L, H) x2 -> (n,) fp32 cosine over the flattened (L*H) vectors."""
    assert a.shape == b.shape, (a.shape, b.shape)
    af = a.reshape(a.shape[0], -1).float()
    bf = b.reshape(b.shape[0], -1).float()
    num = (af * bf).sum(dim=1)
    den = af.norm(dim=1) * bf.norm(dim=1)
    return num / den.clamp_min(1e-12)


def pilot_projection(per_row_s: float, total_rows: int) -> float:
    """Projected full-capture wall in HOURS (§7 gate 1 arithmetic)."""
    return per_row_s * total_rows / 3600.0


# ── Phase A ───────────────────────────────────────────────────────────


def scope_context_ids(bank: dict, cells: tuple[str, ...] | None) -> set[str]:
    """Capture-scope context ids: full bank, or the --cells slice."""
    contexts = bank["contexts"]
    if cells is None:
        return set(contexts)
    known = {ctx["cell"] for ctx in contexts.values()}
    unknown = [c for c in cells if c not in known]
    assert not unknown, f"--cells names unknown cells {unknown} (bank has {len(known)})"
    return {cid for cid, ctx in contexts.items() if ctx["cell"] in cells}


def stage_inputs(cfg: RunConfig2215) -> dict:
    """Scoped staging at the revision pin (canonical helpers; #833 recipe)."""
    from explore_persona_space.orchestrate.hub import stage_hub_file, stage_hub_prefix

    # #2153 duty (a): wall-clock timeout on the detached transfer. Sizing:
    # ~4.5 GB at the >=50 MB/s Xet basis is ~90 s (x2 margin ~3 min); the
    # bound must also survive legitimate per-file retry envelopes
    # (EPM_HF_RETRY_BUDGET_S default 1800 s), so 5400 s covers three full
    # retry budgets while still bounding an indefinite xet hang.
    os.environ.setdefault("EPM_HF_STAGE_TIMEOUT_S", "5400")
    staged: dict[str, int] = {}
    for prefix in (VC_BANK_PREFIX, ANCHOR_TENSOR_PREFIX, ANCHOR_TEXT_PREFIX):
        files = stage_hub_prefix(
            HF_DATA_REPO, prefix, cfg.staged_root, repo_type="dataset", revision=REVISION_PIN
        )
        staged[prefix] = len(files)
    for repo_rel in (*RIDGE_779_PATHS, *RIDGE_1738_PATHS):
        stage_hub_file(
            HF_DATA_REPO,
            repo_rel,
            staged_path(cfg, repo_rel),
            repo_type="dataset",
            revision=REVISION_PIN,
        )
        staged[repo_rel] = 1
    return staged


def check_pair_provenance(cfg: RunConfig2215) -> dict:
    """Artifact-reuse (j): max(input dates) <= min(capture dates) at the PIN,
    via per-path ``get_paths_info`` (recorded, not assumed)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import retry_transient

    input_paths = [f"{VC_BANK_PREFIX}/bank.json"]
    capture_paths = [f"{VC_BANK_PREFIX}/vc_bank.pt"] + [
        f"{ANCHOR_TENSOR_PREFIX}/va_anchors_{b}_w{w}.pt" for b, w in SHARDS
    ]
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    infos = retry_transient(
        lambda: api.get_paths_info(
            HF_DATA_REPO,
            input_paths + capture_paths,
            expand=True,
            repo_type="dataset",
            revision=REVISION_PIN,
        ),
        what="get_paths_info(issue2215 provenance pair)",
    )
    dates = {i.path: i.last_commit.date for i in infos if i.last_commit is not None}
    missing = [p for p in input_paths + capture_paths if p not in dates]
    assert not missing, f"get_paths_info returned no last_commit for {missing}"
    max_input = max(dates[p] for p in input_paths)
    min_capture = min(dates[p] for p in capture_paths)
    assert max_input <= min_capture, (
        f"pairwise provenance FAIL: input bank.json ({max_input.isoformat()}) postdates the "
        f"earliest capture ({min_capture.isoformat()}) — artifact-reuse (j)"
    )
    return {
        "max_input_date": max_input.isoformat(),
        "min_capture_date": min_capture.isoformat(),
        "n_paths_checked": len(dates),
    }


def gate_coverage(cfg: RunConfig2215) -> dict:
    """Plan §4.1 fail-loud coverage gates at FULL consumed-corpus grain (the
    staged artifacts are full-bank in both modes, so the full gate runs under
    smoke too — strictly stronger than the plan's smoke blind-spot (a))."""
    bank = load_bank_json(cfg)
    contexts = bank["contexts"]
    assert isinstance(contexts, dict) and len(contexts) == N_CONTEXTS, (
        f"bank.json carries {len(contexts)} contexts != {N_CONTEXTS}"
    )
    per_cell_pairs = check_pair_table(bank["pairs"])
    degenerate_at_pe = bank["degenerate_at_pe_cells"]  # top-level manifest key (bank2162.py)
    assert set(degenerate_at_pe), "degenerate_at_pe_cells empty — bank manifest drift"

    # Anchor shard indexes (mmap — the tensors stay unmaterialized in Phase A).
    index_keys: list[tuple[str, int]] = []
    empty_keys: set[tuple[str, int]] = set()
    for batch, w in SHARDS:
        pt = shard_tensor_path(cfg, batch, w)
        assert pt.exists(), f"staged shard missing: {pt}"
        payload = torch.load(pt, map_location="cpu", mmap=True, weights_only=False)
        layers = payload["layers"]
        assert list(layers) == list(range(N_MODEL_LAYERS_FULL)), (batch, w, layers)
        idx = payload["index"]
        for j, meta in enumerate(idx):
            key = (meta["context_id"], int(meta["draw"]))
            index_keys.append(key)
            if j in set(payload.get("empty_rows", [])):
                empty_keys.add(key)
        del payload

    jsonl_keys: list[tuple[str, int]] = []
    for batch, w in SHARDS:
        jp = shard_jsonl_path(cfg, batch, w)
        assert jp.exists(), f"staged anchors jsonl missing: {jp}"
        for r in load_jsonl_rows(jp):
            jsonl_keys.append((r["context_id"], int(r["draw"])))

    check_anchor_keysets(index_keys, jsonl_keys, set(contexts), K_DRAWS)

    # Graceful n_valid floor — per-cell reporting, exclusion only at 0.
    n_valid = n_valid_by_context(index_keys, empty_keys, K_DRAWS)
    cell_of = {cid: ctx["cell"] for cid, ctx in contexts.items()}
    per_cell: dict[str, dict] = {}
    for cid, nv in n_valid.items():
        c = per_cell.setdefault(
            cell_of[cid],
            {"n_ctx": 0, "min_valid": K_DRAWS, "n_below_4": 0, "n_zero": 0, "n_empty_draws": 0},
        )
        c["n_ctx"] += 1
        c["min_valid"] = min(c["min_valid"], nv)
        c["n_below_4"] += int(nv < 4)  # split-half denominator flag (plan §4.1)
        c["n_zero"] += int(nv == 0)
        c["n_empty_draws"] += K_DRAWS - nv
    for cell, c in sorted(per_cell.items()):
        empty_frac = c["n_empty_draws"] / (c["n_ctx"] * K_DRAWS)
        if empty_frac > 0.05:  # §12 A8: surfaced, never silently absorbed
            logger.warning(
                "[gate] cell %s empty-draw fraction %.3f > 0.05 (n_ctx=%d)",
                cell,
                empty_frac,
                c["n_ctx"],
            )
        if c["n_zero"]:
            logger.warning(
                "[gate] cell %s has %d context(s) with n_valid=0 — their pairs are "
                "excluded from DV2/DV3 (reported, never silent)",
                cell,
                c["n_zero"],
            )

    # 9 ridge payloads: key/shape asserts + realized-keys probe + apply_map roundtrip.
    from verify_reused_artifact_keys import realized_keys  # scripts/ sibling

    import issue779_ffc_n1m_fits as FITS  # deferred: heavy sibling-chain import

    # The staged payloads are REAL artifacts at H=3584 regardless of --tiny
    # (tiny narrows only Phase B's capture model) — gate at HIDDEN_FULL.
    rng = np.random.default_rng(2215)
    x_probe = rng.standard_normal((8, HIDDEN_FULL)).astype(np.float32)
    payload_report: dict[str, dict] = {}
    for repo_rel in (*RIDGE_779_PATHS, *RIDGE_1738_PATHS):
        local = staged_path(cfg, repo_rel)
        assert local.exists(), f"staged map payload missing: {local}"
        keys = realized_keys(local, weights_only=False, allow_full_load=True)
        missing = set(DECLARED_RIDGE_KEYS) - keys
        assert not missing, f"{repo_rel}: realized-keys probe MISSING {sorted(missing)}"
        expected_layer = int(repo_rel.rsplit("/L", 1)[1].split("/", 1)[0])
        payload = torch.load(local, map_location="cpu", weights_only=False)
        check_ridge_payload(payload, repo_rel, expected_layer, HIDDEN_FULL)
        y = FITS.apply_map(payload, x_probe, torch.device("cpu"))
        assert y.shape == (8, HIDDEN_FULL) and np.isfinite(y).all(), (
            f"{repo_rel}: apply_map roundtrip produced shape {y.shape}, "
            f"finite={bool(np.isfinite(y).all())}"
        )
        payload_report[repo_rel] = {"layer": expected_layer, "realized_keys": len(keys)}
        del payload
    logger.info("[gate] 9/9 ridge payloads: keys+shape+apply_map roundtrip PASS")

    provenance = check_pair_provenance(cfg)
    scope = scope_context_ids(bank, cfg.cells)
    return {
        "n_contexts": len(contexts),
        "n_pairs": len(bank["pairs"]),
        "n_cells": len(per_cell_pairs),
        "n_anchor_rows": len(index_keys),
        "n_empty_rows": len(empty_keys),
        "degenerate_at_pe_cells": degenerate_at_pe,
        "per_cell_n_valid": per_cell,
        "map_payloads": payload_report,
        "provenance": provenance,
        "scope_n_contexts": len(scope),
        "cells_slice": list(cfg.cells) if cfg.cells else None,
    }


def stage_done_path(cfg: RunConfig2215) -> Path:
    return cfg.out_root / "stage_done.json"


def phase_stage(cfg: RunConfig2215) -> int:
    """Phase A: stage the 5 reused families at the pin + run the coverage gates."""
    logger.info("[phase=a_stage] staged_root=%s out_root=%s", cfg.staged_root, cfg.out_root)
    regime_fp = regime_fingerprint(cfg)
    done = stage_done_path(cfg)
    if done.exists() and not cfg.force:
        rec = json.loads(done.read_text())
        if rec.get("regime_fp") == regime_fp:
            logger.info("[stage] already done for this regime — skipping (--force re-runs)")
            logger.info("[phase=a_done]")
            return RC_OK
        raise RuntimeError(
            f"stage_done.json carries regime_fp={rec.get('regime_fp')!r} != {regime_fp!r} — "
            "refusing a cross-regime resume (fresh --out-root or --force)"
        )
    # Resume-aware headroom (plan §9 mount binding): pending work exists here.
    free_gb = assert_out_root_headroom(cfg.out_root, HEADROOM_GB["A_staging"], phase="A_staging")
    logger.info(
        "[stage] headroom OK: %.1f GB free >= %.1f GB floor", free_gb, HEADROOM_GB["A_staging"]
    )
    staged = stage_inputs(cfg)
    logger.info("[stage] staged families: %s", {k: v for k, v in staged.items() if v > 1})
    report = gate_coverage(cfg)
    _write_json_atomic(
        done,
        {
            "regime_fp": regime_fp,
            "revision_pin": REVISION_PIN,
            "staged_file_counts": staged,
            "gates": report,
            "repro": R2162._repro(cfg),
        },
    )
    logger.info(
        "[stage] gates PASS: %d contexts / %d anchor rows",
        report["n_contexts"],
        report["n_anchor_rows"],
    )
    logger.info("[phase=a_done]")
    return RC_OK


# ── Phase B ───────────────────────────────────────────────────────────


def a7_pe_slot_parity(tok, contexts: dict[str, dict], sample_ids: list[str]) -> dict:
    """§12 A7: does #2162's pe slot (prefix_end_index_multi) token-match the
    #1738 ``px_last`` convention (prefix render length − 1)? RECORDED result
    with the pre-registered demotion path — never a halt."""
    rows: list[dict] = []
    for cid in sample_ids:
        ctx = contexts[cid]
        msgs = BANK.context_messages_2094(ctx)
        assert len(msgs) >= 2, (cid, "A7 sample needs a non-empty prefix")
        ctx_ids = BANK.context_token_ids_2162(tok, ctx)
        pe = BANK.prefix_end_index_multi(tok, ctx_ids)
        px_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=False)
        px_ids = tok(px_text, add_special_tokens=False)["input_ids"]
        prefix_len = len(px_ids)
        rows.append(
            {
                "context_id": cid,
                "cell": ctx.get("cell"),
                "multi_turn": bool(ctx.get("history")),
                "prefix_end_2162": pe,
                "prefix_len_1738": prefix_len,
                "index_match": prefix_len == pe,
                "token_prefix_ok": ctx_ids[:prefix_len] == list(px_ids),
            }
        )
    all_match = all(r["index_match"] and r["token_prefix_ok"] for r in rows)
    return {
        "rows": rows,
        "all_match": all_match,
        "demotion_note": None
        if all_match
        else (
            "pe-slot convention mismatch vs #1738 px_last — the #1738-pe arm's ABSOLUTE "
            "reads are demoted to a named caveat (paired reads survive: both pair sides "
            "share the convention) — plan §12 A7 pre-registered demotion path"
        ),
    }


def a7_sample_ids(contexts: dict[str, dict], scope: set[str], n: int = 3) -> list[str]:
    """Prefer multi-turn (history-bearing) contexts — the #1738 convention's
    own shape (prefix ends with an assistant turn); fall back to any context
    with a non-empty prefix message list."""
    ordered = sorted(scope)
    multi = [cid for cid in ordered if contexts[cid].get("history")]
    picked = multi[:n]
    if len(picked) < n:
        rest = [
            cid
            for cid in ordered
            if cid not in picked and len(BANK.context_messages_2094(contexts[cid])) >= 2
        ]
        picked += rest[: n - len(picked)]
    assert picked, "no A7-eligible contexts in scope"
    return picked


def _shard_scoped_rows(
    cfg: RunConfig2215, batch: str, w: int, scope: set[str]
) -> tuple[list[dict], list[int]]:
    """(kept rows, their positions in the FULL shard row list)."""
    rows = load_jsonl_rows(shard_jsonl_path(cfg, batch, w))
    kept, positions = [], []
    for j, r in enumerate(rows):
        if r["context_id"] in scope:
            kept.append(r)
            positions.append(j)
    return kept, positions


def _shard_done_path(cfg: RunConfig2215, batch: str, w: int) -> Path:
    return cfg.manifest_dir / f"va2215_{batch}_w{w}_done.json"


def _shard_store_path(cfg: RunConfig2215, batch: str, w: int) -> Path:
    return cfg.va_dir / f"va2215_{batch}_w{w}.pt"


def _shard_is_done(cfg: RunConfig2215, batch: str, w: int, regime_fp: str) -> bool:
    done = _shard_done_path(cfg, batch, w)
    if not done.exists():
        return False
    rec = json.loads(done.read_text())
    if rec.get("regime_fp") != regime_fp:
        raise RuntimeError(
            f"va2215_{batch}_w{w} done-file regime_fp={rec.get('regime_fp')!r} != "
            f"{regime_fp!r} — refusing a cross-regime resume (fresh --out-root)"
        )
    if rec.get("n_rows", 0) > 0 and not _shard_store_path(cfg, batch, w).exists():
        logger.warning(
            "[capture] %s_w%d done-manifest present but store missing — re-running", batch, w
        )
        return False
    return True


def run_pilot_gate(
    cfg: RunConfig2215, model, tok, ctx_ids: dict[str, list[int]], rows: list[dict], total_rows: int
) -> dict:
    """§7 gate 1: time production-shape batches through the REAL capture
    entrypoint (batched at cfg.capture_batch — the sweep's execution shape,
    #1415), project the full wall, refuse loud above the ceiling."""
    eot = R2162.eot_tail_ids(tok)
    n_warm = min(cfg.capture_batch, len(rows))
    warm = rows[:n_warm]
    R2162.capture_answer_states(
        cfg,
        model,
        tok,
        [ctx_ids[r["context_id"]] for r in warm],
        [r["text"] for r in warm],
        eot,
        tail_inclusive=True,
    )
    timed = rows[n_warm : n_warm + 2 * cfg.capture_batch] or warm
    t0 = time.monotonic()
    R2162.capture_answer_states(
        cfg,
        model,
        tok,
        [ctx_ids[r["context_id"]] for r in timed],
        [r["text"] for r in timed],
        eot,
        tail_inclusive=True,
    )
    wall_s = time.monotonic() - t0
    per_row_s = wall_s / len(timed)
    projected_h = pilot_projection(per_row_s, total_rows)
    report = {
        "n_timed_rows": len(timed),
        "capture_batch": cfg.capture_batch,
        "wall_s": wall_s,
        "per_row_s": per_row_s,
        "total_rows": total_rows,
        "projected_wall_h": projected_h,
        "ceiling_h": cfg.pilot_ceiling_h,
        "verdict": "proceed" if projected_h <= cfg.pilot_ceiling_h else "refuse",
        "repro": R2162._repro(cfg),
    }
    _write_json_atomic(cfg.manifest_dir / "pilot_gate_report.json", report)
    logger.info(
        "[pilot] %.3f s/row x %d rows -> projected %.2f h (ceiling %.2f h) — %s",
        per_row_s,
        total_rows,
        projected_h,
        cfg.pilot_ceiling_h,
        report["verdict"],
    )
    return report


def parity_gate_shard(
    cfg: RunConfig2215,
    batch: str,
    w: int,
    kept_positions: list[int],
    our_span: torch.Tensor,
    our_empty: list[int],
) -> dict:
    """§7 gate 2: per-row flattened cosine of the recomputed span-mean twin
    vs the BANKED va_span on matched (context_id, draw) rows."""
    banked = torch.load(
        shard_tensor_path(cfg, batch, w), map_location="cpu", mmap=True, weights_only=False
    )
    banked_empty = set(banked.get("empty_rows", []))
    # Empty-set cross-check: same tokenizer + text => identical empty rows.
    expected_empty = [k for k, pos in enumerate(kept_positions) if pos in banked_empty]
    assert expected_empty == sorted(our_empty), (
        f"{batch}_w{w}: empty-row mismatch vs banked shard "
        f"(banked∩kept={expected_empty[:5]}..., ours={sorted(our_empty)[:5]}...) — "
        "tokenizer/text drift; stop-and-diagnose"
    )
    keep = [k for k in range(len(kept_positions)) if k not in set(our_empty)]
    if not keep:
        return {
            "n_rows": 0,
            "n_compared": 0,
            "min_cos": None,
            "median_cos": None,
            "frac_ge_bar": None,
        }
    if tuple(our_span.shape[1:]) != tuple(banked["va_span"].shape[1:]):
        # --tiny narrows (layers, hidden), so the cosine is structurally
        # incomparable against the banked full-shape store. DECLARED smoke
        # blind spot (smoke-blind-spots.md): the tiny CPU e2e does NOT
        # exercise the parity verdict; the pod smoke (--cells on the real
        # model) and the unit tests (matched shapes) do.
        assert cfg.tiny, (
            f"{batch}_w{w}: capture shape {tuple(our_span.shape[1:])} != banked "
            f"{tuple(banked['va_span'].shape[1:])} on a NON-tiny run — convention drift"
        )
        logger.warning(
            "[parity] %s_w%d SKIPPED under --tiny (shape %s vs banked %s) — "
            "empty-row cross-check still enforced above",
            batch,
            w,
            tuple(our_span.shape[1:]),
            tuple(banked["va_span"].shape[1:]),
        )
        return {
            "n_rows": len(kept_positions),
            "n_compared": 0,
            "min_cos": None,
            "median_cos": None,
            "frac_ge_bar": None,
            "skipped": "tiny-shape-mismatch",
        }
    banked_rows = banked["va_span"][[kept_positions[k] for k in keep]]
    cos = rowwise_flat_cosine(our_span[keep], banked_rows)
    frac = float((cos >= cfg.parity_cos_min).float().mean())
    stats = {
        "n_rows": len(kept_positions),
        "n_compared": len(keep),
        "min_cos": float(cos.min()),
        "median_cos": float(cos.median()),
        "frac_ge_bar": frac,
        "bar": cfg.parity_cos_min,
        "frac_min": cfg.parity_frac_min,
    }
    logger.info(
        "[parity] %s_w%d n=%d min=%.6f median=%.6f frac>=%.3f: %.4f",
        batch,
        w,
        len(keep),
        stats["min_cos"],
        stats["median_cos"],
        cfg.parity_cos_min,
        frac,
    )
    return stats


def upload_va_store(cfg: RunConfig2215) -> None:
    """B-end fail-loud store upload + scoped landing verification + sentinel
    (critic round-1 Must-Fix: Phase C unreachable before this completes)."""
    files = sorted(cfg.va_dir.glob("va2215_*.pt"))
    assert files, f"no va2215 shards under {cfg.va_dir} — nothing to upload"
    if cfg.upload_mode == "none":
        logger.warning(
            "[upload] SKIPPED (--upload none): va2215_uploaded.json NOT written — "
            "Phase C (unit 2) is gated on it and will refuse to run"
        )
        return
    remote_prefix = f"{cfg.hf_prefix}/analysis_tensors/va2215"
    expected = R2162.upload_dir_hf(cfg.va_dir, remote_prefix, ["va2215_*.pt"])  # raises on failure
    # Scoped landing verification (plan §4.2): shard count on the Hub ==
    # local count, and the expected set is fully present.
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import list_hf_files_under_path

    listed = list_hf_files_under_path(
        HfApi(token=os.environ.get("HF_TOKEN")), HF_DATA_REPO, remote_prefix, repo_type="dataset"
    )
    listed_shards = [p for p in listed if p.rsplit("/", 1)[-1].startswith("va2215_")]
    missing = sorted(set(expected) - set(listed))
    assert not missing and len(listed_shards) >= len(files), (
        f"va2215 landing verification FAIL: {len(missing)} expected paths missing "
        f"(first: {missing[:3]}), {len(listed_shards)} listed vs {len(files)} local"
    )
    _write_json_atomic(
        cfg.out_root / "va2215_uploaded.json",
        {
            "regime_fp": regime_fingerprint(cfg),
            "hf_prefix": remote_prefix,
            "n_local_shards": len(files),
            "n_listed_shards": len(listed_shards),
            "expected_repo_paths": expected,
            "repro": R2162._repro(cfg),
        },
    )
    logger.info(
        "[upload] va2215 store landed: %d shards -> %s (verified scoped listing)",
        len(files),
        remote_prefix,
    )


def phase_capture(cfg: RunConfig2215) -> int:
    """Phase B: teacher-forced answer-capture twin over the banked rollouts."""
    logger.info("[phase=b_capture] out_root=%s cells=%s", cfg.out_root, cfg.cells)
    regime_fp = regime_fingerprint(cfg)
    done = stage_done_path(cfg)
    assert done.exists(), f"{done} missing — run --phase a first (Phase B consumes staged inputs)"
    stage_rec = json.loads(done.read_text())
    assert stage_rec.get("regime_fp") == regime_fp, (
        f"stage_done regime_fp={stage_rec.get('regime_fp')!r} != {regime_fp!r} — re-run --phase a"
    )
    bank = load_bank_json(cfg)
    contexts = bank["contexts"]
    scope = scope_context_ids(bank, cfg.cells)

    pending = [
        (batch, w)
        for batch, w in SHARDS
        if cfg.force or not _shard_is_done(cfg, batch, w, regime_fp)
    ]
    # Load + slice pending shard rows up front (cheap CPU) so the pilot
    # projects over the REMAINING work and empty shards are known early.
    shard_rows: dict[tuple[str, int], tuple[list[dict], list[int]]] = {}
    total_rows = 0
    for batch, w in pending:
        kept, positions = _shard_scoped_rows(cfg, batch, w, scope)
        shard_rows[(batch, w)] = (kept, positions)
        total_rows += len(kept)
    if not pending:
        logger.info("[capture] all shards done for this regime — skipping to upload")
    elif total_rows == 0:
        logger.info(
            "[capture] no in-scope rows in pending shards (cells slice) — nothing to capture"
        )
    if total_rows > 0:
        # Resume-aware headroom: scale to the pending subset (plan §9 rule).
        need_gb = max(1.0, HEADROOM_GB["B_capture"] * total_rows / 14040.0)
        free_gb = assert_out_root_headroom(cfg.out_root, need_gb, phase="B_capture")
        logger.info("[capture] headroom OK: %.1f GB free >= %.1f GB floor", free_gb, need_gb)

        model, tok = R2162.load_model_and_tokenizer(cfg)
        eot = R2162.eot_tail_ids(tok)

        # §12 A7 pe-slot token-parity check (recorded; demotion path, no halt).
        a7 = a7_pe_slot_parity(tok, contexts, a7_sample_ids(contexts, scope))
        _write_json_atomic(
            cfg.manifest_dir / "a7_pe_parity.json", {**a7, "repro": R2162._repro(cfg)}
        )
        logger.info(
            "[a7] pe-slot parity vs #1738 px_last: all_match=%s over %d samples%s",
            a7["all_match"],
            len(a7["rows"]),
            "" if a7["all_match"] else " — DEMOTION NOTE RECORDED (see a7_pe_parity.json)",
        )

        needed_cids = sorted({r["context_id"] for kept, _ in shard_rows.values() for r in kept})
        ctx_ids = {cid: BANK.context_token_ids_2162(tok, contexts[cid]) for cid in needed_cids}

        first_batch, first_w = next(bw for bw in pending if shard_rows[bw][0])
        pilot = run_pilot_gate(
            cfg, model, tok, ctx_ids, shard_rows[(first_batch, first_w)][0], total_rows
        )
        if pilot["verdict"] != "proceed":
            logger.error(
                "[phase=pilot_refused] projected %.2f h > ceiling %.2f h — designed halt "
                "(report: manifests/pilot_gate_report.json); re-plan batch size",
                pilot["projected_wall_h"],
                pilot["ceiling_h"],
            )
            return RC_PILOT_GATE

        t_phase = time.monotonic()
        for k, (batch, w) in enumerate(pending):
            kept, positions = shard_rows[(batch, w)]
            if not kept:
                logger.info(
                    "[capture] unit %d/%d %s_w%d 0 in-scope rows — skipped",
                    k + 1,
                    len(pending),
                    batch,
                    w,
                )
                continue
            states = R2162.capture_answer_states(
                cfg,
                model,
                tok,
                [ctx_ids[r["context_id"]] for r in kept],
                [r["text"] for r in kept],
                eot,
                tail_inclusive=True,
            )
            # Integrity: re-tokenized completion lengths must match the banked
            # rows' recorded counts (span alignment depends on them).
            for r, n_tok in zip(kept, states["n_completion_tokens"], strict=True):
                assert int(r["n_completion_tokens"]) == n_tok, (
                    f"{batch}_w{w}: n_completion_tokens drift for "
                    f"({r['context_id']}, draw {r['draw']}): banked {r['n_completion_tokens']} "
                    f"vs re-tokenized {n_tok} — tokenizer drift; stop-and-diagnose"
                )
            parity = parity_gate_shard(
                cfg, batch, w, positions, states["va_span"], states["empty_rows"]
            )
            if (
                parity["n_compared"]
                and parity["frac_ge_bar"] < cfg.parity_frac_min
                and not cfg.smoke
            ):
                _write_json_atomic(
                    cfg.manifest_dir / f"parity_fail_{batch}_w{w}.json",
                    {**parity, "shard": f"{batch}_w{w}", "repro": R2162._repro(cfg)},
                )
                logger.error(
                    "[phase=parity_failed] %s_w%d frac_ge_bar=%.4f < %.2f — stop-and-diagnose "
                    "(index misalignment / convention drift invalidates DV2/DV3)",
                    batch,
                    w,
                    parity["frac_ge_bar"],
                    cfg.parity_frac_min,
                )
                return RC_PARITY_GATE
            if parity["n_compared"] and parity["frac_ge_bar"] < cfg.parity_frac_min:
                logger.warning(
                    "[parity] %s_w%d below provisional bar under smoke — informational "
                    "(the smoke calibrates the production bar; plan §4.2)",
                    batch,
                    w,
                )
            _save_pt_atomic(
                _shard_store_path(cfg, batch, w),
                {
                    "layers": cfg.layers,
                    "index": [{"context_id": r["context_id"], "draw": r["draw"]} for r in kept],
                    "va_span_excl": states["va_span"],
                    "va_tail_incl": states["va_tail_incl"],
                    "pooling": states["pooling"],
                    "empty_rows": states["empty_rows"],
                    "repro": R2162._repro(cfg),
                },
            )
            _write_json_atomic(
                _shard_done_path(cfg, batch, w),
                {
                    "regime_fp": regime_fp,
                    "shard": f"{batch}_w{w}",
                    "n_rows": len(kept),
                    "n_empty": len(states["empty_rows"]),
                    "parity": parity,
                    "repro": R2162._repro(cfg),
                },
            )
            logger.info(
                "[capture] unit %d/%d %s_w%d rows=%d elapsed=%.0fs",
                k + 1,
                len(pending),
                batch,
                w,
                len(kept),
                time.monotonic() - t_phase,
            )
    # Zero-in-scope-row pending shards get NO done-manifest by design: the
    # resume re-scan of their jsonl is milliseconds, and manifests stay
    # row-bearing (a manifest without a store would trip _shard_is_done's
    # store-missing branch on every resume).
    upload_va_store(cfg)
    logger.info("[phase=b_done]")
    return RC_OK


# ── Phase C: analysis (plan §4.3) ─────────────────────────────────────


def _resolve_anchors_jsonl(cfg: RunConfig2215) -> Path:
    """Resolve the H2 parent input (fail-loud). The default lives in git under
    eval_results/issue_2162/, which is OUTSIDE the pod bootstrap's default
    sparse cones (#2211) — on a miss, attempt the sanctioned
    ``git sparse-checkout add`` once, then assert with the recovery command."""
    path = cfg.anchors_jsonl or (
        _repo_root() / "eval_results" / "issue_2162" / "f_metrics" / "anchors.jsonl"
    )
    if not path.exists() and cfg.anchors_jsonl is None:
        proc = subprocess.run(  # self-heal the sparse cone; assert below is the gate
            ["git", "sparse-checkout", "add", "eval_results/issue_2162"],
            cwd=_repo_root(),
            env={**os.environ},
            capture_output=True,
            text=True,
        )
        logger.info(
            "[analysis] anchors.jsonl absent — sparse-checkout add eval_results/issue_2162 "
            "rc=%d %s",
            proc.returncode,
            (proc.stderr or "").strip()[:200],
        )
    assert path.exists(), (
        f"{path} missing — H2 coupling needs the parent anchors.jsonl (plan §4.3). "
        "On a sparse pod clone run: git sparse-checkout add eval_results/issue_2162 "
        "(gotchas.md partial-clone cones, #2211), or pass --anchors-jsonl."
    )
    return path


def phase_analysis(cfg: RunConfig2215) -> int:
    """Phase C: DV1/DV2/DV3 + nulls + baselines over the staged + captured
    stores (scripts/issue2215_analysis.py holds all statistical logic)."""
    logger.info("[phase=c_analysis] out_root=%s results_dir=%s", cfg.out_root, cfg.results_dir)
    regime_fp = regime_fingerprint(cfg)

    # GATE 1 (concern unit2-phase-c-sentinel-gate, addressed round 1): the
    # va2215 capture store must be DURABLY uploaded (va2215_uploaded.json
    # written by upload_va_store AFTER the fail-loud upload_folder + scoped
    # landing verify) and belong to THIS regime before any analysis consumes
    # it — the #825 store-before-long-fit ordering. `--upload none` runs
    # deliberately cannot reach Phase C.
    uploaded = cfg.out_root / "va2215_uploaded.json"
    assert uploaded.exists(), (
        f"{uploaded} missing — Phase C is gated on the Phase-B store upload sentinel "
        "(concern unit2-phase-c-sentinel-gate; #825 store-before-long-fit). Run "
        "--phase b with --upload hf first."
    )
    up_rec = json.loads(uploaded.read_text())
    assert up_rec.get("regime_fp") == regime_fp, (
        f"va2215_uploaded regime_fp={up_rec.get('regime_fp')!r} != {regime_fp!r} — the "
        "uploaded store belongs to a DIFFERENT capture regime; re-run --phase b for this "
        "regime (concern unit2-phase-c-sentinel-gate)"
    )

    # GATE 2: staged inputs present for this regime (Phase C reads vc_bank +
    # banked anchor shards + ridge payloads from the Phase-A mirror).
    done = stage_done_path(cfg)
    assert done.exists(), f"{done} missing — run --phase a first"
    stage_rec = json.loads(done.read_text())
    assert stage_rec.get("regime_fp") == regime_fp, (
        f"stage_done regime_fp={stage_rec.get('regime_fp')!r} != {regime_fp!r} — re-run --phase a"
    )

    # GATE 3: idempotent skip / fail-loud refuse on the analysis fingerprint.
    afp = analysis_fingerprint(cfg)
    adone = cfg.out_root / "analysis_done.json"
    if adone.exists() and not cfg.force:
        arec = json.loads(adone.read_text())
        if arec.get("analysis_fp") == afp:
            logger.info("[analysis] analysis_done.json matches analysis_fp=%s — skipping", afp)
            logger.info("[phase=c_done]")
            return RC_OK
        raise AssertionError(
            f"analysis_done analysis_fp={arec.get('analysis_fp')!r} != {afp!r} — the "
            "existing Phase-C outputs were produced under different analysis knobs; pass "
            "--force to overwrite or use a fresh --out-root (#722 r3 resume-regime rule)"
        )

    import issue2215_analysis as ANALYSIS  # deferred: gates above stay millisecond-fast

    bank = load_bank_json(cfg)
    if cfg.tiny:
        # DECLARED tiny substitutions (mirrors the tiny parity-gate skip):
        # staged full-H ridge payloads + banked 28x3584 anchor shards are
        # structurally incomparable with a tiny capture — DV3 is skipped
        # (recorded in dv3_map_discrimination.json) and the span secondary
        # pools from va2215's own va_span_excl twin.
        arm_specs = None
        banked_dir = None
        logger.warning(
            "[analysis] --tiny: DV3 skipped + span pooling from va2215 (declared blind "
            "spot; the pod --cells smoke exercises both at full shape)"
        )
    else:
        arm_specs = [
            {
                "arm": "779ce",
                "slot": "ce",
                "paths": {
                    layer: staged_path(
                        cfg, f"issue779_monitoring/n1m_readout/weights/L{layer}/ridge.pt"
                    )
                    for layer in MAP_LAYERS
                },
            },
            {
                "arm": "1738ce",
                "slot": "ce",
                "paths": {
                    layer: staged_path(
                        cfg,
                        f"issue1738_multiturn/analysis_tensors/weights/L{layer}/context_ridge.pt",
                    )
                    for layer in MAP_LAYERS
                },
            },
            {
                "arm": "1738pe",
                "slot": "pe",
                "paths": {
                    layer: staged_path(
                        cfg,
                        f"issue1738_multiturn/analysis_tensors/weights/L{layer}/prefix_ridge.pt",
                    )
                    for layer in MAP_LAYERS
                },
            },
        ]
        banked_dir = staged_path(cfg, ANCHOR_TENSOR_PREFIX)
    inp = ANALYSIS.AnalysisInputs(
        bank=bank,
        vc_bank_path=staged_path(cfg, f"{VC_BANK_PREFIX}/vc_bank.pt"),
        va_dir=cfg.va_dir,
        banked_anchor_dir=banked_dir,
        arm_specs=arm_specs,
        results_dir=cfg.results_dir,
        null_dir=cfg.null_dir,
        anchors_jsonl=_resolve_anchors_jsonl(cfg),
        cells=cfg.cells,
        null_b=cfg.null_b,
        boot_b=cfg.boot_b,
        k_draws=K_DRAWS,
        repro=R2162._repro(cfg),
    )
    digest = ANALYSIS.run_analysis(inp)
    _write_json_atomic(
        adone,
        {
            "analysis_fp": afp,
            "regime_fp": regime_fp,
            "null_b": cfg.null_b,
            "boot_b": cfg.boot_b,
            "results_dir": str(cfg.results_dir),
            "digest": digest,
            "repro": R2162._repro(cfg),
        },
    )
    logger.info("[phase=c_done]")
    return RC_OK


# ── Phase D: upload + commit + sentinels (plan §9 phase order) ────────


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """One git subprocess (explicit env passthrough — subprocess-env rule)."""
    return subprocess.run(
        ["git", *args], cwd=repo, env={**os.environ}, capture_output=True, text=True
    )


def commit_results_git(cfg: RunConfig2215) -> dict:
    """Production-only git add/commit/push of the Phase-C eval JSONs (+ the
    figures dir when unit 3's figures exist) from the pod, per the #1880
    result-push contract (bare push, rc checked; fetch+rebase single retry).
    Smoke/tiny outputs live under out_root and are never committed."""
    if cfg.smoke or cfg.tiny:
        reason = "smoke/tiny — results_dir is an out-root twin, no canonical paths written"
        logger.info("[finalize] git commit SKIPPED: %s", reason)
        return {"committed": False, "reason": reason}
    repo = _repo_root()
    rel_results = str(cfg.results_dir.relative_to(repo))
    paths = [rel_results]
    figures = repo / "figures" / "issue_2215"
    assert figures.is_dir(), (
        f"{figures} missing — phase_finalize renders figures (render_figures) BEFORE this "
        "commit, so an absent dir means the Phase-D wiring is broken"
    )
    paths.append("figures/issue_2215")
    add = _git(repo, "add", "--", *paths)
    assert add.returncode == 0, f"git add failed rc={add.returncode}: {add.stderr[:400]}"
    # Staged-index verification (#958): a dir-path `git add` silently skips
    # gitignored files with rc=0 — force-add convention-committed hits.
    skipped = _git(repo, "ls-files", "--others", "--ignored", "--exclude-standard", "--", *paths)
    hit_lines = [ln for ln in skipped.stdout.split("\n") if ln.strip()]
    if hit_lines:
        logger.warning(
            "[finalize] %d gitignored file(s) under result paths — git add -f: %s",
            len(hit_lines),
            hit_lines[:5],
        )
        forced = _git(repo, "add", "-f", "--", *hit_lines)
        assert forced.returncode == 0, f"git add -f failed: {forced.stderr[:400]}"
        recheck = _git(
            repo, "ls-files", "--others", "--ignored", "--exclude-standard", "--", *paths
        )
        assert not recheck.stdout.strip(), (
            f"staged-index verification FAIL after add -f: {recheck.stdout[:400]}"
        )
    staged = _git(repo, "diff", "--cached", "--quiet", "--", *paths)
    if staged.returncode == 0:
        logger.info("[finalize] no staged changes under %s — nothing to commit", paths)
        return {"committed": False, "reason": "no staged changes"}
    branch_proc = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    branch = branch_proc.stdout.strip()
    assert branch_proc.returncode == 0 and branch and branch != "HEAD", (
        f"cannot resolve branch for result push: rc={branch_proc.returncode} {branch!r}"
    )
    commit = _git(
        repo,
        "commit",
        "-m",
        "issue #2215: Phase C analysis outputs (DV1/DV2/DV3 + nulls + coupling)",
        "--",
        *paths,
    )
    assert commit.returncode == 0, f"git commit failed: {commit.stderr[:400]}"
    sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    push = _git(repo, "push", "origin", f"HEAD:refs/heads/{branch}")
    if push.returncode != 0:
        logger.warning(
            "[finalize] push rejected — fetch+rebase retry (#1880): %s", (push.stderr or "")[:300]
        )
        fetch = _git(repo, "fetch", "origin", branch)
        assert fetch.returncode == 0, f"git fetch failed: {fetch.stderr[:400]}"
        rebase = _git(repo, "rebase", f"origin/{branch}")
        assert rebase.returncode == 0, (
            f"git rebase failed (manual resolution needed): {rebase.stderr[:400]}"
        )
        sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
        push = _git(repo, "push", "origin", f"HEAD:refs/heads/{branch}")
        assert push.returncode == 0, f"result push failed after rebase retry: {push.stderr[:400]}"
    logger.info("[finalize] results committed + pushed: %s -> %s (%s)", sha, branch, paths)
    return {"committed": True, "sha": sha, "branch": branch, "paths": paths}


def render_figures(cfg: RunConfig2215) -> dict:
    """Phase D figure render (plan §6, unit 3): every registry figure from
    the Phase-C outputs via ``issue2215_figures.render_all``. Production
    renders into the repo's ``figures/issue_2215`` (committed by
    ``commit_results_git``); smoke/tiny render into an out-root twin so
    smoke outputs never touch committed paths. Registry skips (tiny DV3,
    <3-cell H2) are RECORDED in the returned manifest, never silent."""
    import issue2215_figures as FIGS  # deferred: matplotlib load only in Phase D

    if cfg.smoke or cfg.tiny:
        out_dir = cfg.out_root / "figures_smoke" / "issue_2215"
    else:
        out_dir = _repo_root() / "figures" / "issue_2215"
    manifest = FIGS.render_all(cfg.results_dir, out_dir)
    n_written = sum(1 for v in manifest.values() if v["written"])
    skipped = {k: v["skipped"] for k, v in manifest.items() if not v["written"]}
    logger.info(
        "[finalize] figures rendered: %d/%d -> %s (skipped: %s)",
        n_written,
        len(manifest),
        out_dir,
        skipped or "none",
    )
    return {"out_dir": str(out_dir), "n_written": n_written, "skipped": skipped}


def upload_null_matrices(cfg: RunConfig2215) -> dict:
    """Fail-loud HF upload + scoped landing verify of the per-draw null
    matrices (plan §6 selection-symmetric-nulls persistence)."""
    if cfg.upload_mode == "none":
        logger.warning("[finalize] null-matrices upload SKIPPED (--upload none) — recorded")
        return {"uploaded": False, "reason": "--upload none"}
    files = sorted(cfg.null_dir.glob("*.npz")) + sorted(cfg.null_dir.glob("*.json"))
    assert files, f"no null matrices under {cfg.null_dir} — run --phase c first"
    remote_prefix = f"{cfg.hf_prefix}/analysis_tensors/null_matrices"
    expected = R2162.upload_dir_hf(cfg.null_dir, remote_prefix, ["*.npz", "*.json"])
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import list_hf_files_under_path

    listed = list_hf_files_under_path(
        HfApi(token=os.environ.get("HF_TOKEN")), HF_DATA_REPO, remote_prefix, repo_type="dataset"
    )
    missing = sorted(set(expected) - set(listed))
    assert not missing, (
        f"null-matrices landing verification FAIL: {len(missing)} expected paths missing "
        f"(first: {missing[:3]})"
    )
    logger.info("[finalize] null matrices landed: %d files -> %s", len(files), remote_prefix)
    return {"uploaded": True, "n_files": len(files), "hf_prefix": remote_prefix}


def write_results_sentinel(cfg: RunConfig2215, digest: dict) -> Path:
    """Poller results sentinel (`pod-side-reporting.md` sentinel contract:
    schema keys {sentinel_schema_version, kind, version, note}; smoke/tiny
    runs write kind epm:smoke-result, never epm:results). This driver only
    WRITES sentinels — it never reads them back (resume state lives under
    out_root, outside the drained namespace)."""
    smokish = cfg.smoke or cfg.tiny
    kind = "epm:smoke-result" if smokish else "epm:results"
    note = json.dumps(
        {
            "issue": 2215,
            "phase": "d_finalize",
            "cells": list(cfg.cells) if cfg.cells else "full",
            "results_dir": str(cfg.results_dir),
            "hf_prefix": cfg.hf_prefix,
            "n_contexts": digest.get("n_contexts"),
            "n_pairs": digest.get("n_pairs"),
            "n_cells": digest.get("n_cells"),
            "n_excluded_pairs": digest.get("n_excluded_pairs"),
            "dv1_aggregates": digest.get("dv1_aggregates"),
            "dv3_registered": digest.get("dv3_registered"),
            "h2": digest.get("h2"),
        },
        default=str,
    )
    cfg.sentinel_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.sentinel_dir / f"issue-2215-results-{int(time.time())}.json"
    _write_json_atomic(
        path,
        {"sentinel_schema_version": 1, "kind": kind, "version": 1, "note": note},
    )
    logger.info("[finalize] results sentinel written: %s (kind=%s)", path, kind)
    return path


def phase_finalize(cfg: RunConfig2215) -> int:
    """Phase D: null-matrices upload -> figure render -> production git
    commit/push -> poller results sentinel -> upload_done.json (plan §9
    phase_outputs)."""
    logger.info("[phase=d_finalize] out_root=%s", cfg.out_root)
    adone = cfg.out_root / "analysis_done.json"
    assert adone.exists(), f"{adone} missing — run --phase c first"
    arec = json.loads(adone.read_text())
    afp = analysis_fingerprint(cfg)
    assert arec.get("analysis_fp") == afp, (
        f"analysis_done analysis_fp={arec.get('analysis_fp')!r} != {afp!r} — Phase D would "
        "publish outputs from a different analysis regime; re-run --phase c"
    )
    upload_rec = upload_null_matrices(cfg)
    figures_rec = render_figures(cfg)
    commit_rec = commit_results_git(cfg)
    sentinel_path = write_results_sentinel(cfg, arec.get("digest") or {})
    _write_json_atomic(
        cfg.out_root / "upload_done.json",
        {
            "regime_fp": regime_fingerprint(cfg),
            "analysis_fp": afp,
            "null_matrices": upload_rec,
            "figures": figures_rec,
            "results_git": commit_rec,
            "sentinel": str(sentinel_path),
            "repro": R2162._repro(cfg),
        },
    )
    logger.info("[phase=d_done]")
    return RC_OK


# ── entrypoint ────────────────────────────────────────────────────────


def _import_check() -> None:
    """Resolve EVERY deferred import this driver reaches on its real paths
    (#1689), then the argparse-attribute completeness assert (#2163)."""
    import issue779_ffc_n1m_fits as FITS  # noqa: F401
    from huggingface_hub import HfApi  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    from verify_reused_artifact_keys import realized_keys  # noqa: F401

    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        list_hf_files_under_path,
        retry_transient,
        stage_hub_file,
        stage_hub_prefix,
    )

    import issue2215_analysis as ANALYSIS
    import issue2215_figures as FIGS_MOD
    from issue2094_analysis import bootstrap_family_means_batched  # noqa: F401
    from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: F401
    from scipy.spatial.distance import squareform  # noqa: F401
    from scipy.stats import rankdata, spearmanr  # noqa: F401

    from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
        identity_bias_predict,
        knn_retrieval,
    )

    assert callable(FITS.apply_map)
    assert callable(R2162.capture_answer_states)
    assert callable(R2162.upload_dir_hf)
    assert callable(ANALYSIS.run_analysis)
    assert callable(FIGS_MOD.render_all) and FIGS_MOD.FIGURES, "figures registry empty"
    import inspect

    # The tail-inclusive twin extension must be present on the reused capture.
    assert "tail_inclusive" in inspect.signature(R2162.capture_answer_states).parameters, (
        "issue2162_run.capture_answer_states lacks the tail_inclusive extension "
        "(issue #2215 plan §4.2) — stale checkout?"
    )
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    print("[import-check] OK", flush=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args().parse_args(argv)
    if args.import_check:
        _import_check()
        return RC_OK
    assert args.phase, "--phase is required (or pass --import-check)"
    cfg = build_config(args)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    cfg.va_dir.mkdir(parents=True, exist_ok=True)
    if cfg.phase in ("a", "all"):
        rc = phase_stage(cfg)
        if rc != RC_OK:
            return rc
    if cfg.phase in ("b", "all"):
        rc = phase_capture(cfg)
        if rc != RC_OK:
            return rc
    if cfg.phase in ("c", "all"):
        rc = phase_analysis(cfg)
        if rc != RC_OK:
            return rc
    if cfg.phase in ("d", "all"):
        rc = phase_finalize(cfg)
        if rc != RC_OK:
            return rc
        # Terminal breadcrumb AFTER the final sentinel write (pod-side
        # reporting contract) — only the terminal phase emits it.
        logger.info("[phase=done]")
    return RC_OK


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension interpreter finalization (#1689).
    sys.exit(rc)
