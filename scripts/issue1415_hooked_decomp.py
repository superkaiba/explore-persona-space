#!/usr/bin/env python
"""Issue #1415 — hooked-unhooked per-bin decomposition driver (plan v11).

Re-forwards the round's PERSISTED completions with the parent ``DeltaHook``
ARMED at the last-context token (``edit_position`` mode, plan §3.2) and
subtracts the round's STORED unhooked binned profiles on IDENTICAL text —
the per-(pair, arm, steer-layer, bin) hooked-unhooked difference is the
generation-time DIRECT component of the edit, position-resolved. NO new
generation, NO judging, NO new steering generations (the hook is armed
during measurement forwards only).

Phases (one driver, ``--phase all`` in production; the ``--tiny`` e2e runs
the SAME chain on a 3-layer from-config Qwen over the real vocab —
PASS_UNIFIED; every phase's cell list derives from
:func:`enumerate_hooked_cells`):

  h0_stage    enumerate the 168 hooked cells from git metadata (hard
              per-family count asserts 56/56/56), stage the 140 draws JSONs
              @ the parent pin, the unhooked reference + gen1b target shards
              @ the shard pin, and the 28 parent Δ stores @ the parent pin
              (scoped per-file hub.stage_hub_file, <=6 workers — NEVER an
              unscoped listing of the ~1M-file repo); 1-shard mmap key probe
              BEFORE any GPU; meta.context == draws.context cross-asserts.
  h1_fidelity §3.5 gates G0/G1/G2 on the 12 spot cells (their captures
              persist as h2 tensors — unified-phase economy). HALT = rc=9 +
              fidelity_gate_report.json (artifact-routed by the dispatcher,
              never a bare rc=1).
  h2_capture  batched teacher-forced HOOKED binned capture per cell ->
              tensors_root/<hooked_id>.pt (fp16 profiles + fp32 ctx_vec),
              checkpoint-per-cell manifest keyed on every regime knob,
              per-cell G0 pairing assert, incremental per-family HF upload.
  h3_stats    CPU statistics (§3.4): per-draw per-bin diffs, the same-text
              jitter reference J_p (causally-zero-layer transport + fp16
              floor), R_p = log L_p - log(2·J_p^late) with pair-bootstrap CI
              (seed 14153) + Wilcoxon, Δdirect shape companion + width
              variant, alignment-to-target (disjoint halves ONLY on
              baseline-text cells), per-bin random-direction null (seed
              14154, ONE batched GEMM) -> the five registered JSONs.
  h4_upload   store manifest + batched upload_folder residue (the git
              commit of the eval JSONs is the DISPATCHER's
              commit_push_verify, not this driver).

Pod-side contract: [phase=...] log lines only; this driver NEVER emits the
reserved [phase=done] token and NEVER shells out to scripts/task.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch import (thread caps) + HF token for staging/upload

import numpy as np  # noqa: E402
import torch  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent

# Reused verbatim from the round-1 / parent drivers (plan §3.3 REUSED list).
import issue1415_position_profile as pp  # noqa: E402
from issue1415_run_phase1 import (  # noqa: E402
    HF_DATA_REPO,
    TENSOR_PREFIX,
    Manifest,
    _fmt,
    _repro,
    _save_pt_atomic,
    _write_json_atomic,
    load_model_and_tokenizer,
    upload_artifact,
)

from explore_persona_space.experiments.issue1415.steering import (  # noqa: E402
    BIN_NAMES,
    DeltaHook,
    capture_binned_answer_profiles,
)

logger = logging.getLogger("issue1415.hooked_decomp")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HOOKED_TENSOR_PREFIX = "analysis_tensors/issue_1415/hooked_decomposition"

# Pinned source revisions (plan §10; probed at plan time 2026-07-23).
PARENT_REVISION = pp.PARENT_REVISION  # draws JSONs + 1a Δ stores
SHARD_REVISION = "9d8f789bf034d8f244e1d00e0dbbe6aba6d272c5"  # unhooked shards

LAYERS_FULL = pp.LAYERS_FULL  # (7, 10, 14, 17, 20, 21, 24)
STEER_LAYERS_FULL = (14, 20)
ALPHA = 4.0
N_DRAWS_FULL = 10
BINS_VERSION = pp.BINS_VERSION

EARLY_BINS = pp.EARLY_BINS  # ("first", "tok2_5")
LATE_BINS = pp.LATE_BINS  # ("dec8", "dec9", "dec10")
WIDTH_EARLY_BINS = pp.WIDTH_EARLY_BINS  # ("dec1", "dec2")
MAX_DROP_FRAC = pp.MAX_DROP_FRAC  # 0.20

BOOT_SEED = 14153  # pair bootstrap (registered, plan §3.4)
N_BOOT_FULL = 10_000
NULL_SEED = 14154  # per-bin random-direction null (registered)
N_NULL_FULL = 500

# fp16 two-sided quantization floor inside J_p (plan §3.4): 2^-11 relative
# per leg, sqrt(2) for the two-sided (both-legs-quantized) diff.
FP16_FLOOR = (2.0**-11) * math.sqrt(2.0)
JITTER_MULT = 2.0  # the registered 2x-jitter resolvability bar (plan §11)

# G1 upstream-zero rig parity (v9-calibrated, plan §3.5).
G1_HALT_COS = pp.PARITY_HALT_COS  # 0.995
G1_WARN_COS = pp.PARITY_WARN_COS  # 0.9999
G1_MAX_BAD = pp.PARITY_MAX_BAD  # HALT when MORE THAN this many cells < HALT_COS
# G2 edit-injection exactness (structural apply-path class, plan §3.5).
G2_COS_MIN = 0.99
G2_RATIO_LO, G2_RATIO_HI = 0.9, 1.1
N_SPOT_PAIRS_FULL = 5  # alphabetically-first pairs at the top steer layer
RC_FIDELITY_HALT = 9  # designed artifact-routed HALT (never a bare rc=1)

MEDICAL_PAIR = pp.MEDICAL_PAIR
TERSE_PAIR = pp.TERSE_PAIR
FORMAL_PAIR = pp.FORMAL_PAIR


# ── config + cell inventory ───────────────────────────────────────────


@dataclass(frozen=True)
class HookedCell:
    """One HOOKED re-forward cell (plan §3.1)."""

    hooked_id: str  # hooked store stem, e.g. gen1c/context/<pid>/L14/a4
    source_cell_id: str  # unhooked draws + shard stem (gen1c/... or gen1b/<pid>/c)
    kind: str  # steered | baseline_text
    pair_id: str
    delta_arm: str  # prefix | context (the Δ injected)
    steer_layer: int
    meta_path: Path


@dataclass
class HookedConfig:
    """Resolved run configuration — a duck-typed superset of the fields the
    REUSED helpers read (`_repro`/`load_model_and_tokenizer`/`upload_artifact`:
    model_id, tiny, hidden, n_model_layers, upload_mode, bulk_root;
    `pp.fetch_draws`/`pp.load_staged_draws`: tiny, hub_mirror_root, stage_root;
    `pp.list_pairs`: meta_root_primary, n_pairs_expected)."""

    tiny: bool
    out_root: Path
    tensors_root: Path
    stage_root: Path
    bulk_root: Path
    hub_mirror_root: Path | None  # tiny draws + Δ-store mirror (pp fixture layout)
    shard_mirror_root: Path | None  # tiny unhooked-shard mirror (pp tensors_root)
    work_root: Path | None  # tiny scratch root (substrate sentinel lives here)
    meta_root_primary: Path
    upload_mode: str  # hf | local-mirror
    model_id: str
    n_draws: int
    capture_batch: int
    layers: tuple[int, ...]
    steer_layers: tuple[int, ...]
    n_pairs_expected: int
    hidden: int
    n_model_layers: int
    device: str
    n_null: int
    n_boot: int
    n_spot_pairs: int
    phases: tuple[str, ...] = field(default_factory=tuple)


def read_layers_for(cfg: HookedConfig, steer_layer: int) -> tuple[int, ...]:
    """Captured layers ABOVE the steer layer (the only layers where the direct
    component exists — plan §3.2 consequence (ii)). Production: 14 -> (17, 20,
    21, 24); 20 -> (21, 24)."""
    out = tuple(layer for layer in cfg.layers if layer > steer_layer)
    assert out, f"no captured layer above steer layer {steer_layer} in {cfg.layers}"
    return out


def headline_read_for(cfg: HookedConfig, steer_layer: int) -> int:
    """Registered headline read layer = nearest captured layer ABOVE steer
    (plan §11: 14 -> 17, 20 -> 21). Pre-registered — NO max over layers."""
    return read_layers_for(cfg, steer_layer)[0]


def jitter_l0_for(cfg: HookedConfig, steer_layer: int) -> int:
    """Nearest causally-zero captured layer (<= steer) — the same-text jitter
    source (plan §3.4: L14 cells -> 14; L20 cells -> 20)."""
    zeros = [layer for layer in cfg.layers if layer <= steer_layer]
    assert zeros, f"no captured causally-zero layer for steer layer {steer_layer}"
    return zeros[-1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--phase", default="all", choices=["all", "h0", "h1", "h2", "h3", "h4"])
    p.add_argument("--tiny", action="store_true", help="CPU tiny e2e (3-layer model, fixtures)")
    p.add_argument(
        "--work-root",
        default=None,
        help="tiny scratch root (default data/issue_1415/tiny_smoke/hooked_decomp)",
    )
    p.add_argument("--out-root", default=None, help="override the eval_results output root")
    p.add_argument("--capture-batch", type=int, default=8)
    p.add_argument("--upload", default=None, choices=["hf", "local-mirror"])
    p.add_argument("--n-null", type=int, default=None)
    p.add_argument("--n-boot", type=int, default=None)
    p.add_argument(
        "--staging-probe",
        action="store_true",
        help="stage ONE real unhooked shard @ the shard pin + ONE Δ store @ the "
        "parent pin through the production staging path, run the mmap key probe "
        "+ the Δ-formula load, then exit 0 (HF_TOKEN-gated; no GPU)",
    )
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> HookedConfig:
    """Resolve the run configuration; tiny scales every axis down but keeps
    the IDENTICAL phase chain + code paths (PASS_UNIFIED smoke contract)."""
    phases = ("h0", "h1", "h2", "h3", "h4") if args.phase == "all" else (args.phase,)
    if args.tiny:
        work = Path(args.work_root or "data/issue_1415/tiny_smoke/hooked_decomp")
        out_root = Path(args.out_root) if args.out_root else work / "out"
        return HookedConfig(
            tiny=True,
            out_root=out_root,
            tensors_root=work / "tensors",
            stage_root=work / "stage",
            bulk_root=work / "bulk",
            hub_mirror_root=work / "hub_mirror",
            shard_mirror_root=work / "pp_tensors",
            work_root=work,
            meta_root_primary=work / "phase1" / "cells",
            upload_mode=args.upload or "local-mirror",
            model_id=MODEL_ID,
            n_draws=2,
            capture_batch=args.capture_batch,
            layers=(0, 1, 2),
            steer_layers=(0, 1),
            n_pairs_expected=2,
            hidden=32,
            n_model_layers=3,
            device="cpu",
            n_null=args.n_null or 50,
            n_boot=args.n_boot or 200,
            n_spot_pairs=2,
            phases=phases,
        )
    assert args.work_root is None, "--work-root is tiny-only"
    er = REPO_ROOT / "eval_results" / "issue_1415"
    return HookedConfig(
        tiny=False,
        out_root=(Path(args.out_root) if args.out_root else er / "hooked_unhooked_decomposition"),
        tensors_root=REPO_ROOT / "data" / "issue_1415" / "hooked_decomposition" / "tensors",
        stage_root=REPO_ROOT / "data" / "issue_1415" / "hf_dl" / "hooked_decomposition",
        bulk_root=REPO_ROOT / "data" / "issue_1415" / "hooked_decomposition",
        hub_mirror_root=None,
        shard_mirror_root=None,
        work_root=None,
        meta_root_primary=er / "phase1" / "cells",
        upload_mode=args.upload or "hf",
        model_id=MODEL_ID,
        n_draws=N_DRAWS_FULL,
        capture_batch=args.capture_batch,
        layers=LAYERS_FULL,
        steer_layers=STEER_LAYERS_FULL,
        n_pairs_expected=28,
        hidden=3584,
        n_model_layers=28,
        device="cuda:0",
        n_null=args.n_null or N_NULL_FULL,
        n_boot=args.n_boot or N_BOOT_FULL,
        n_spot_pairs=N_SPOT_PAIRS_FULL,
        phases=phases,
    )


def enumerate_hooked_cells(cfg: HookedConfig) -> list[HookedCell]:
    """The FULL registered hooked-cell inventory (plan §3.1), enumerated from
    git metadata with hard per-family count asserts (56/56/56 in production).
    Every phase's cell list derives from THIS function (PASS_UNIFIED)."""
    pairs = pp.list_pairs(cfg)  # duck-typed: meta_root_primary + n_pairs_expected
    cells: list[HookedCell] = []
    for steer in cfg.steer_layers:
        for arm in ("prefix", "context"):
            for pid in pairs:
                src = f"gen1c/{arm}/{pid}/L{steer}/a{_fmt(ALPHA)}"
                cells.append(
                    HookedCell(
                        hooked_id=src,
                        source_cell_id=src,
                        kind="steered",
                        pair_id=pid,
                        delta_arm=arm,
                        steer_layer=steer,
                        meta_path=cfg.meta_root_primary / f"{src}.json",
                    )
                )
        for pid in pairs:
            src = f"gen1b/{pid}/c"
            cells.append(
                HookedCell(
                    hooked_id=f"gen1b_c/{pid}/L{steer}/context",
                    source_cell_id=src,
                    kind="baseline_text",
                    pair_id=pid,
                    delta_arm="context",
                    steer_layer=steer,
                    meta_path=cfg.meta_root_primary / f"{src}.json",
                )
            )
    n_pairs = len(pairs)
    counts: dict[str, int] = {}
    for steer in cfg.steer_layers:
        counts[f"gen1c/L{steer}"] = 2 * n_pairs
    counts["gen1b_c"] = n_pairs * len(cfg.steer_layers)
    for fam, expected in counts.items():
        if fam.startswith("gen1c/"):
            steer = int(fam.split("L")[1])
            got = sum(1 for c in cells if c.kind == "steered" and c.steer_layer == steer)
        else:
            got = sum(1 for c in cells if c.kind == "baseline_text")
        assert got == expected, f"hooked inventory mismatch for {fam}: {got} != {expected}"
    assert len(cells) == 3 * n_pairs * len(cfg.steer_layers), len(cells)
    missing_meta = sorted({c.hooked_id for c in cells if not c.meta_path.exists()})
    if missing_meta:
        raise RuntimeError(
            f"{len(missing_meta)} cell metadata file(s) missing from git: {missing_meta}"
        )
    return cells


def _source_profile_cell(cfg: HookedConfig, cell: HookedCell) -> pp.ProfileCell:
    """A pp.ProfileCell view of the SOURCE cell (what pp.fetch_draws stages)."""
    return pp.ProfileCell(
        cell_id=cell.source_cell_id,
        family=cell.source_cell_id.split("/", 1)[0],
        role=cell.kind,
        pair_id=cell.pair_id,
        arm=None,
        steer_layer=None,
        meta_path=cell.meta_path,
        revision_key="parent",
    )


# ── h0: stage + verify ────────────────────────────────────────────────

_SHARD_REQUIRED_KEYS = ("profiles", "span_mean", "kept_indices", "comp_token_counts", "bin_names")


def _resolve_revisions(cfg: HookedConfig) -> dict[str, str]:
    """Resolve + persist the pinned source revisions."""
    rev_path = cfg.out_root / "revisions.json"
    if rev_path.exists():
        return json.loads(rev_path.read_text())["revisions"]
    if cfg.tiny:
        revisions = {"parent": "local-fixture", "shards": "local-fixture"}
    else:
        revisions = {"parent": PARENT_REVISION, "shards": SHARD_REVISION}
    _write_json_atomic(rev_path, {"revisions": revisions, "repro": _repro(cfg)})
    return revisions


def _shard_path(cfg: HookedConfig, cell_id: str) -> Path:
    return cfg.stage_root / "shards" / f"{cell_id}.pt"


def fetch_profile_shard(cfg: HookedConfig, cell_id: str, revisions: dict[str, str]) -> Path:
    """Stage ONE round-1 unhooked binned shard @ the shard pin (identity
    mapping — the consumer opens the exact fetch destination)."""
    target = _shard_path(cfg, cell_id)
    if target.exists():
        return target
    if cfg.tiny:
        src = cfg.shard_mirror_root / f"{cell_id}.pt"
        assert src.exists(), f"tiny substrate missing unhooked shard for {cell_id}: {src}"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / (target.name + ".tmp")
        tmp.write_bytes(src.read_bytes())
        os.replace(tmp, target)
        return target
    from explore_persona_space.orchestrate import hub

    return hub.stage_hub_file(
        HF_DATA_REPO,
        f"{pp.PROFILE_TENSOR_PREFIX}/{cell_id}.pt",
        target,
        repo_type="dataset",
        revision=revisions["shards"],
    )


def _delta_store_path(cfg: HookedConfig, pair_id: str) -> Path:
    return cfg.stage_root / "activations" / f"{pair_id}.pt"


def fetch_delta_store(cfg: HookedConfig, pair_id: str, revisions: dict[str, str]) -> Path:
    """Stage ONE parent 1a activation store (Δ source + G2 V_c reference) @
    the parent pin."""
    target = _delta_store_path(cfg, pair_id)
    if target.exists():
        return target
    if cfg.tiny:
        src = cfg.hub_mirror_root / TENSOR_PREFIX / f"{pair_id}.pt"
        assert src.exists(), f"tiny substrate missing Δ store for {pair_id}: {src}"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / (target.name + ".tmp")
        tmp.write_bytes(src.read_bytes())
        os.replace(tmp, target)
        return target
    from explore_persona_space.orchestrate import hub

    return hub.stage_hub_file(
        HF_DATA_REPO,
        f"{TENSOR_PREFIX}/{pair_id}.pt",
        target,
        repo_type="dataset",
        revision=revisions["parent"],
    )


_DELTA_CACHE: dict[str, dict] = {}


def _load_delta_blob(cfg: HookedConfig, pair_id: str) -> dict:
    if pair_id not in _DELTA_CACHE:
        path = _delta_store_path(cfg, pair_id)
        assert path.exists(), f"Δ store missing for {pair_id}: {path} (run h0 first)"
        blob = torch.load(path, map_location="cpu", weights_only=True)
        # DeltaSource's inherited layer-list assert (plan §3.3 REUSED formula).
        assert blob.get("layers") == list(cfg.layers), (pair_id, blob.get("layers"))
        _DELTA_CACHE[pair_id] = blob
    return _DELTA_CACHE[pair_id]


def load_delta(cfg: HookedConfig, pair_id: str, arm: str, layer: int) -> torch.Tensor:
    """Δ_arm[layer] = cprime.v_c_<arm>[idx] - c.v_c_<arm>[idx] — the exact
    DeltaSource.pair_delta formula against the same store (thin re-impl,
    plan §3.3)."""
    assert arm in ("prefix", "context"), arm
    blob = _load_delta_blob(cfg, pair_id)
    idx = list(cfg.layers).index(layer)
    d = blob["cprime"][f"v_c_{arm}"][idx] - blob["c"][f"v_c_{arm}"][idx]
    assert d.shape == (cfg.hidden,), d.shape
    return d.float()


def load_vc_context_ref(cfg: HookedConfig, pair_id: str, layer: int) -> torch.Tensor:
    """The parent's UNHOOKED V_c at the last-context token (the G2 reference:
    same position as the edit, by causality — plan §3.5)."""
    blob = _load_delta_blob(cfg, pair_id)
    idx = list(cfg.layers).index(layer)
    v = blob["c"]["v_c_context"][idx]
    assert v.shape == (cfg.hidden,), v.shape
    return v.float()


def _shard_key_probe(cfg: HookedConfig, cell_id: str) -> None:
    """Realized-keys probe (artifact-reuse (c)): mmap read, storages untouched."""
    path = _shard_path(cfg, cell_id)
    keys = set(torch.load(path, map_location="cpu", mmap=True, weights_only=True))
    missing = [k for k in _SHARD_REQUIRED_KEYS if k not in keys]
    assert not missing, f"unhooked shard {cell_id} missing keys {missing} (has {sorted(keys)})"


def phase_h0(cfg: HookedConfig, cells: list[HookedCell]) -> None:
    if cfg.tiny:
        build_tiny_substrate(cfg)
    revisions = _resolve_revisions(cfg)
    pairs = sorted({c.pair_id for c in cells})

    # Unique staging work items (dicts dedupe by key).
    src_cells = {c.source_cell_id: _source_profile_cell(cfg, c) for c in cells}
    shard_ids = sorted(src_cells)  # unhooked reference shards (per unique source)
    target_ids = [f"gen1b/{pid}/{tag}" for pid in pairs for tag in ("c", "cprime")]
    all_shards = sorted(set(shard_ids) | set(target_ids))
    logger.info(
        "[phase=h0_stage] staging %d draws JSONs + %d shards + %d Δ stores (<=6 workers)",
        len(src_cells),
        len(all_shards),
        len(pairs),
    )
    jobs: list[tuple[str, object]] = (
        [("draws", c) for c in src_cells.values()]
        + [("shard", cid) for cid in all_shards]
        + [("delta", pid) for pid in pairs]
    )

    def _run(job: tuple[str, object]) -> None:
        kind, arg = job
        if kind == "draws":
            pp.fetch_draws(cfg, arg, revisions)
        elif kind == "shard":
            fetch_profile_shard(cfg, arg, revisions)
        else:
            fetch_delta_store(cfg, arg, revisions)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_run, j): j for j in jobs}
        failures: list[str] = []
        for fut, job in futs.items():
            try:
                fut.result()
            except Exception as exc:  # collected, then raised LOUD naming items
                logger.error("[phase=h0_stage] fetch failed for %s: %s", job, exc)
                failures.append(f"{job[0]}:{job[1] if isinstance(job[1], str) else job[1].cell_id}")
        if failures:
            raise RuntimeError(f"h0 staging failed for {len(failures)} item(s): {failures}")

    # 1-shard mmap key probe BEFORE any GPU (plan §3.3).
    _shard_key_probe(cfg, all_shards[0])
    # Δ-store layer-list + shape asserts on every pair (cheap CPU loads).
    for pid in pairs:
        for arm in ("prefix", "context"):
            for steer in cfg.steer_layers:
                load_delta(cfg, pid, arm, steer)
    # meta.context == draws.context + n_draws cross-asserts (inherited).
    mismatches: list[str] = []
    for src_id, pcell in sorted(src_cells.items()):
        meta = json.loads(pcell.meta_path.read_text())
        blob = pp.load_staged_draws(cfg, src_id)
        if meta["context"] != blob["context"]:
            mismatches.append(f"{src_id} (meta.context != draws.context)")
        if len(blob["draws"]) != cfg.n_draws or meta["n_draws"] != cfg.n_draws:
            mismatches.append(
                f"{src_id} (n_draws meta={meta['n_draws']} draws={len(blob['draws'])} "
                f"!= {cfg.n_draws})"
            )
    if mismatches:
        raise RuntimeError(f"h0 verification failed for {len(mismatches)} cell(s): {mismatches}")
    logger.info(
        "[phase=h0_stage] staged + verified %d hooked cells (%d sources, %d shards)",
        len(cells),
        len(src_cells),
        len(all_shards),
    )


def staging_probe(cfg: HookedConfig) -> None:
    """Production staging probe (plan §3.3 smoke #3): ONE real unhooked shard
    @ the shard pin + ONE Δ store @ the parent pin through the SAME staging
    helpers h0 uses, then the mmap key probe + the Δ-formula load."""
    assert not cfg.tiny, "--staging-probe is a production-path probe"
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing — required for --staging-probe"
    revisions = {"parent": PARENT_REVISION, "shards": SHARD_REVISION}
    pairs = pp.list_pairs(cfg)
    pid = pairs[0]
    shard_id = f"gen1c/context/{pid}/L{max(cfg.steer_layers)}/a{_fmt(ALPHA)}"
    logger.info("[phase=staging_probe] shard=%s @ %s", shard_id, revisions["shards"])
    fetch_profile_shard(cfg, shard_id, revisions)
    _shard_key_probe(cfg, shard_id)
    logger.info("[phase=staging_probe] Δ store pair=%s @ %s", pid, revisions["parent"])
    fetch_delta_store(cfg, pid, revisions)
    for arm in ("prefix", "context"):
        for steer in cfg.steer_layers:
            d = load_delta(cfg, pid, arm, steer)
            logger.info(
                "[phase=staging_probe] Δ_%s@L%d ok (norm=%.3f)", arm, steer, float(d.norm())
            )
    load_vc_context_ref(cfg, pid, max(cfg.steer_layers))
    logger.info("[phase=staging_probe] PASS (shard keys + Δ formula + V_c reference)")


# ── tiny substrate (unhooked shards through the round-1 tiny path) ────


def build_tiny_substrate(cfg: HookedConfig) -> None:
    """Produce the tiny UNHOOKED shards through the EXISTING round-1 tiny
    path (pp fixture -> p0 -> p2 -> p3), plus parent-schema Δ stores from
    capture_vectors — the h0..h4 chain then runs against them exactly as
    production runs against the Hub artifacts (plan §3.3 t3)."""
    from explore_persona_space.experiments.issue1415.steering import capture_vectors

    assert cfg.tiny and cfg.work_root is not None
    sentinel = cfg.work_root / ".hooked_substrate_complete"
    if sentinel.exists():
        return
    logger.info("[phase=h0_stage] building tiny unhooked substrate under %s", cfg.work_root)
    work = cfg.work_root
    pp_cfg = pp.ProfileConfig(
        tiny=True,
        out_root=work / "pp_out",
        tensors_root=work / "pp_tensors",
        stage_root=work / "pp_stage",
        bulk_root=work / "pp_bulk",
        hub_mirror_root=cfg.hub_mirror_root,
        meta_root_primary=cfg.meta_root_primary,
        meta_roots_rep={s: work / f"phase1_rep{s}" / "cells" for s in pp.REP_LABELS},
        upload_mode="local-mirror",
        model_id=cfg.model_id,
        n_draws=cfg.n_draws,
        capture_batch=cfg.capture_batch,
        layers=cfg.layers,
        headline_layers=cfg.steer_layers,
        parity_layer=max(cfg.steer_layers),
        rep_layer=min(cfg.layers),
        n_pairs_expected=cfg.n_pairs_expected,
        hidden=cfg.hidden,
        n_model_layers=cfg.n_model_layers,
        device="cpu",
        n_null=cfg.n_null,
        n_boot=cfg.n_boot,
        n_parity_pairs=2,
        phases=("p0", "p2", "p3"),
    )
    pp_cfg.out_root.mkdir(parents=True, exist_ok=True)
    pp_cfg.tensors_root.mkdir(parents=True, exist_ok=True)
    pp_cfg.stage_root.mkdir(parents=True, exist_ok=True)
    pp.build_tiny_fixture(pp_cfg)
    pp_cells = pp.enumerate_cells(pp_cfg)
    pp.phase_p0(pp_cfg, pp_cells)
    revisions = pp._resolve_revisions(pp_cfg)
    model, tok = load_model_and_tokenizer(pp_cfg)
    state = Manifest.load_or_init(
        pp_cfg.out_root / "profile_manifest.json", pp._regime(pp_cfg, revisions)
    )
    pp.phase_p2(pp_cfg, pp_cells, model, tok, state)
    pp.phase_p3(pp_cfg, pp_cells, revisions)
    # Parent-schema Δ stores from capture_vectors on the SAME tiny model.
    for pid in pp._TINY_PAIRS:
        ctxs = pp._TINY_CONTEXTS[pid]
        cap = capture_vectors(
            model, tok, [ctxs["c"], ctxs["cprime"]], list(cfg.layers), completions=None
        )
        rec_c, rec_cp = cap["per_context"]
        _save_pt_atomic(
            cfg.hub_mirror_root / TENSOR_PREFIX / f"{pid}.pt",
            {
                "pair_id": pid,
                "layers": list(cfg.layers),
                "c": {"v_c_prefix": rec_c["v_c_prefix"], "v_c_context": rec_c["v_c_context"]},
                "cprime": {
                    "v_c_prefix": rec_cp["v_c_prefix"],
                    "v_c_context": rec_cp["v_c_context"],
                },
            },
        )
    del model
    sentinel.write_text("ok")
    logger.info("[phase=h0_stage] tiny substrate complete (unhooked shards + Δ stores)")


# ── h1/h2: hooked capture + fidelity gates ────────────────────────────


def _hooked_tensor_path(cfg: HookedConfig, hooked_id: str) -> Path:
    return cfg.tensors_root / f"{hooked_id}.pt"


def _regime(cfg: HookedConfig, revisions: dict[str, str]) -> dict:
    """Every output-affecting knob (resume must not cross regimes, #722 r3)."""
    return {
        "mode": "hooked_unhooked_decomposition",
        "bins_version": BINS_VERSION,
        "bin_names": list(BIN_NAMES),
        "layers": list(cfg.layers),
        "steer_layers": list(cfg.steer_layers),
        "alpha": ALPHA,
        "model_id": cfg.model_id,
        "n_draws": cfg.n_draws,
        "revisions": revisions,
        "tiny": cfg.tiny,
        "store_dtype": "float16",
    }


def _assert_g0(hooked: dict, unhooked: dict, cell_label: str) -> None:
    """G0 pairing integrity (plan §3.5): the fresh hooked capture and the
    reused unhooked shard must describe the IDENTICAL kept draw set — a
    mismatch is a LOUD per-cell error naming the cell, never a silent
    re-pair."""
    hk = [int(i) for i in hooked["kept_indices"]]
    uk = [int(i) for i in unhooked["kept_indices"]]
    hc = [int(n) for n in hooked["comp_token_counts"]]
    uc = [int(n) for n in unhooked["comp_token_counts"]]
    if hk != uk or hc != uc:
        raise RuntimeError(
            f"G0 pairing mismatch for {cell_label}: kept_indices hooked={hk} unhooked={uk}; "
            f"comp_token_counts hooked={hc} unhooked={uc} — wrong text/tokenizer "
            "(structural HALT class, plan §3.5)"
        )


def _capture_hooked_cell(cfg: HookedConfig, cell: HookedCell, model, tok, state: Manifest) -> dict:
    """Capture (or load, when resumed) ONE hooked cell; persists the fp16
    store + marks the manifest the moment the cell completes. Runs the
    per-cell G0 pairing assert against the staged unhooked shard."""
    mark = f"hooked/{cell.hooked_id}"
    out_path = _hooked_tensor_path(cfg, cell.hooked_id)
    if state.done(mark) and out_path.exists():
        return torch.load(out_path, map_location="cpu", weights_only=True)
    blob = pp.load_staged_draws(cfg, cell.source_cell_id)
    draws = blob["draws"]
    kept_indices = [
        i for i, t in enumerate(draws) if len(tok(t, add_special_tokens=False)["input_ids"]) > 0
    ]
    delta = load_delta(cfg, cell.pair_id, cell.delta_arm, cell.steer_layer)
    hook = DeltaHook(model, cell.steer_layer, delta, ALPHA)
    with hook:
        cap = capture_binned_answer_profiles(
            model,
            tok,
            blob["context"],
            draws,
            list(cfg.layers),
            batch_size=cfg.capture_batch,
            hook=hook,
            capture_ctx_vec=True,
        )
    assert cap["profiles"].shape[0] == len(kept_indices), (
        cap["profiles"].shape,
        len(kept_indices),
    )
    unhooked = torch.load(
        _shard_path(cfg, cell.source_cell_id), map_location="cpu", weights_only=True
    )
    record = {
        "hooked_id": cell.hooked_id,
        "source_cell_id": cell.source_cell_id,
        "kind": cell.kind,
        "pair_id": cell.pair_id,
        "delta_arm": cell.delta_arm,
        "steer_layer": cell.steer_layer,
        "alpha": ALPHA,
        "layers": list(cfg.layers),
        "bin_names": list(BIN_NAMES),
        "profiles": cap["profiles"].to(torch.float16),  # (n_kept, 13, L, H)
        "span_mean": cap["span_mean"].to(torch.float16),  # (n_kept, L, H)
        "ctx_vec": cap["ctx_vec"],  # (n_chunks, L, H) fp32 — the G2 input
        "ctx_vec_max_dev": cap["ctx_vec_max_dev"],
        "comp_token_counts": cap["comp_token_counts"],
        "kept_indices": kept_indices,
        "n_empty_completions": cap["n_empty_completions"],
        "repro": _repro(cfg),
    }
    _assert_g0(record, unhooked, cell.hooked_id)
    _save_pt_atomic(out_path, record)
    state.mark(mark, {"kind": cell.kind, "steer_layer": cell.steer_layer})
    # Return the fp32 capture for in-phase consumers (the fidelity gates).
    return {**record, "profiles": cap["profiles"], "span_mean": cap["span_mean"]}


def spot_cells(cfg: HookedConfig, cells: list[HookedCell]) -> list[HookedCell]:
    """The §3.5 deterministic spot set: the round's parity convention —
    n_spot_pairs alphabetically-first pairs x both arms at the TOP steer
    layer, PLUS the first pair x both arms at every OTHER steer layer (both
    steer layers' arming checked; 12 production cells, no extra forwards)."""
    pairs = sorted({c.pair_id for c in cells})
    top = max(cfg.steer_layers)
    spot = [
        c
        for c in cells
        if c.kind == "steered" and c.steer_layer == top and c.pair_id in pairs[: cfg.n_spot_pairs]
    ]
    for steer in cfg.steer_layers:
        if steer == top:
            continue
        spot += [
            c
            for c in cells
            if c.kind == "steered" and c.steer_layer == steer and c.pair_id == pairs[0]
        ]
    expected = 2 * cfg.n_spot_pairs + 2 * (len(cfg.steer_layers) - 1)
    assert len(spot) == expected, (len(spot), expected)
    return sorted(spot, key=lambda c: (c.steer_layer, c.pair_id, c.delta_arm))


def fidelity_verdict(g0_rows: list[dict], g1_rows: list[dict], g2_rows: list[dict]) -> dict:
    """Pure §3.5 verdict (unit-testable; every HALT branch gets its own
    degenerate-input probe). G0: ANY pairing mismatch fires. G1: MORE THAN
    G1_MAX_BAD cells below G1_HALT_COS fires. G2: ANY cell outside the
    cos/norm-ratio bands fires (structural class — precedence absolute, no
    WARN demotion)."""
    g0_fired = len(g0_rows) > 0
    n_bad = sum(1 for r in g1_rows if r["min_cos"] < G1_HALT_COS)
    n_warn = sum(1 for r in g1_rows if G1_HALT_COS <= r["min_cos"] < G1_WARN_COS)
    g1_fired = n_bad > G1_MAX_BAD
    g2_bad = [r for r in g2_rows if not r["passed"]]
    g2_fired = len(g2_bad) > 0
    return {
        "fired": g0_fired or g1_fired or g2_fired,
        "g0": {"fired": g0_fired, "mismatches": g0_rows},
        "g1": {
            "fired": g1_fired,
            "n_cells": len(g1_rows),
            "n_bad": n_bad,
            "n_warn": n_warn,
            "halt_cos": G1_HALT_COS,
            "warn_cos": G1_WARN_COS,
            "max_bad": G1_MAX_BAD,
            "cells": g1_rows,
        },
        "g2": {
            "fired": g2_fired,
            "cos_min": G2_COS_MIN,
            "norm_ratio_band": [G2_RATIO_LO, G2_RATIO_HI],
            "n_cells": len(g2_rows),
            "n_bad": len(g2_bad),
            "cells": g2_rows,
        },
    }


def _enforce_fidelity(cfg: HookedConfig, verdict: dict) -> None:
    """Write the report ALWAYS; HALT rc=9 when fired (artifact-routed by the
    dispatcher — never a bare rc=1)."""
    verdict = {**verdict, "repro": _repro(cfg)}
    report = cfg.out_root / "fidelity_gate_report.json"
    _write_json_atomic(report, verdict)
    if verdict["fired"]:
        logger.error(
            "[phase=h1_fidelity] FIDELITY HALT (g0=%s g1=%s g2=%s) — report at %s",
            verdict["g0"]["fired"],
            verdict["g1"]["fired"],
            verdict["g2"]["fired"],
            report,
        )
        sys.exit(RC_FIDELITY_HALT)
    logger.info(
        "[phase=h1_fidelity] PASS (g1 n_bad=%d n_warn=%d of %d; g2 n_bad=0 of %d)",
        verdict["g1"]["n_bad"],
        verdict["g1"]["n_warn"],
        verdict["g1"]["n_cells"],
        verdict["g2"]["n_cells"],
    )


def phase_h1(cfg: HookedConfig, cells: list[HookedCell], model, tok, state: Manifest) -> None:
    spot = spot_cells(cfg, cells)
    logger.info("[phase=h1_fidelity] %d spot cells (G0/G1/G2)", len(spot))
    g0_rows: list[dict] = []
    g1_rows: list[dict] = []
    g2_rows: list[dict] = []
    layer_index = {layer: i for i, layer in enumerate(cfg.layers)}
    t0 = time.monotonic()
    first_n = 0
    for cell in spot:
        try:
            rec = _capture_hooked_cell(cfg, cell, model, tok, state)
        except RuntimeError as exc:
            if "G0 pairing mismatch" not in str(exc):
                raise
            g0_rows.append({"cell_id": cell.hooked_id, "error": str(exc)})
            continue
        if first_n == 0:
            first_n = int(rec["profiles"].shape[0])
            dt = time.monotonic() - t0
            logger.info(
                "[phase=h1_fidelity] first-chunk batched timing: %.2f s / %d samples "
                "= %.3f s/sample (batch=%d — informational, plan §9)",
                dt,
                first_n,
                dt / max(first_n, 1),
                cfg.capture_batch,
            )
        unhooked = torch.load(
            _shard_path(cfg, cell.source_cell_id), map_location="cpu", weights_only=True
        )
        h_mean = rec["profiles"].float().nanmean(dim=0)  # (13, L, H)
        u_mean = unhooked["profiles"].float().nanmean(dim=0)
        # G1: upstream-zero parity over the causally-zero layers (<= steer).
        cos_vals: list[float] = []
        for layer in cfg.layers:
            if layer > cell.steer_layer:
                continue
            li = layer_index[layer]
            for b in range(len(BIN_NAMES)):
                c = pp._cos_finite(h_mean[b, li], u_mean[b, li])
                if c is not None:
                    cos_vals.append(c)
        assert cos_vals, f"no finite G1 cosine for {cell.hooked_id}"
        g1_rows.append(
            {
                "cell_id": cell.hooked_id,
                "steer_layer": cell.steer_layer,
                "min_cos": float(min(cos_vals)),
                "n_comparisons": len(cos_vals),
            }
        )
        # G2: edit-injection exactness vs alpha*Delta + the stored V_c.
        si = layer_index[cell.steer_layer]
        ctx_vec = rec["ctx_vec"].mean(dim=0)[si]  # (H,) fp32 mean over chunks
        ref = load_vc_context_ref(cfg, cell.pair_id, cell.steer_layer)
        delta = load_delta(cfg, cell.pair_id, cell.delta_arm, cell.steer_layer)
        d = ctx_vec - ref
        cos = float(
            torch.nn.functional.cosine_similarity(d.unsqueeze(0), delta.unsqueeze(0)).squeeze()
        )
        ratio = float(d.norm() / (ALPHA * delta.norm()))
        cross = load_delta(cfg, cell.pair_id, "prefix", cell.steer_layer)
        ctx_d = load_delta(cfg, cell.pair_id, "context", cell.steer_layer)
        cross_cos = float(
            torch.nn.functional.cosine_similarity(cross.unsqueeze(0), ctx_d.unsqueeze(0)).squeeze()
        )
        g2_rows.append(
            {
                "cell_id": cell.hooked_id,
                "delta_arm": cell.delta_arm,
                "steer_layer": cell.steer_layer,
                "cos_d_delta": cos,
                "norm_ratio": ratio,
                "ctx_vec_max_dev": float(rec["ctx_vec_max_dev"]),
                "cross_arm_delta_cos": cross_cos,  # wrong-arm discriminability context
                "passed": bool(cos >= G2_COS_MIN and G2_RATIO_LO <= ratio <= G2_RATIO_HI),
            }
        )
    _enforce_fidelity(cfg, fidelity_verdict(g0_rows, g1_rows, g2_rows))


def phase_h2(cfg: HookedConfig, cells: list[HookedCell], model, tok, state: Manifest) -> None:
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    # Plan §9 preamble asserts at BOTH h2 write roots (statvfs +
    # posix_fallocate canary — EDQUOT-aware) BEFORE any capture writes.
    assert_out_root_headroom(cfg.tensors_root, need_gb=10, phase="h2_capture")
    assert_out_root_headroom(cfg.out_root, need_gb=10, phase="h2_capture")
    families: dict[str, list[HookedCell]] = {}
    for c in cells:
        families.setdefault(c.hooked_id.split("/", 1)[0], []).append(c)
    n_done = 0
    for fam_dir, fam_cells in sorted(families.items()):
        for cell in fam_cells:
            _capture_hooked_cell(cfg, cell, model, tok, state)
            n_done += 1
            if n_done % 25 == 0:
                logger.info("[phase=h2_capture] %d/%d cells captured", n_done, len(cells))
        # Incremental per-family upload: ONE folder commit per family (never
        # a per-file loop — 256-commits/hr + per-file-504 gotchas).
        upload_artifact(cfg, cfg.tensors_root / fam_dir, f"{HOOKED_TENSOR_PREFIX}/{fam_dir}")
    logger.info("[phase=h2_capture] complete: %d cells", len(cells))


# ── h3: statistics (§3.4) ─────────────────────────────────────────────


def _half_nanmean(prof: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sel = prof[mask]
    if sel.shape[0] == 0:
        return torch.full_like(prof[0], torch.nan)
    return sel.nanmean(dim=0)


def _reduce_hooked(cfg: HookedConfig, cell: HookedCell) -> dict:
    """Load ONE hooked store + its staged unhooked shard; re-assert G0; per-
    draw fp16-symmetric diff D + full/half nan-mean reductions (§3.4)."""
    h = torch.load(_hooked_tensor_path(cfg, cell.hooked_id), map_location="cpu", weights_only=True)
    u = torch.load(_shard_path(cfg, cell.source_cell_id), map_location="cpu", weights_only=True)
    _assert_g0(h, u, cell.hooked_id)
    # Both legs are fp16 ON DISK (symmetric storage noise); diff in fp32.
    D = h["profiles"].float() - u["profiles"].float()  # (n, 13, L, H)
    kept = [int(i) for i in h["kept_indices"]]
    even_mask = torch.tensor(kept) % 2 == 0  # ORIGINAL draw-index parity
    return {
        "D": D,
        "Dbar": D.nanmean(dim=0),  # (13, L, H)
        "D_even": _half_nanmean(D, even_mask),
        "D_odd": _half_nanmean(D, ~even_mask),
        "u_mean": u["profiles"].float().nanmean(dim=0),
        "even_mask": even_mask,
        "n_kept": int(D.shape[0]),
        "comp_token_counts": [int(n) for n in h["comp_token_counts"]],
        "ctx_vec_max_dev": float(h.get("ctx_vec_max_dev", float("nan"))),
    }


def _reduce_shard(cfg: HookedConfig, cell_id: str) -> dict:
    """Full/even/odd nan-mean reductions of one UNHOOKED gen1b shard (the
    alignment-target legs)."""
    u = torch.load(_shard_path(cfg, cell_id), map_location="cpu", weights_only=True)
    prof = u["profiles"].float()
    kept = [int(i) for i in u["kept_indices"]]
    even_mask = torch.tensor(kept) % 2 == 0
    return {
        "full": prof.nanmean(dim=0),
        "even": _half_nanmean(prof, even_mask),
        "odd": _half_nanmean(prof, ~even_mask),
    }


def _finite_or_none(v: float) -> float | None:
    return v if isinstance(v, float) and math.isfinite(v) else None


def _direct_bin_rows(red: dict, base: dict, ceil: dict, li: int, l0i: int, kind: str) -> list[dict]:
    """Per-bin §3.4 rows for one hooked cell at read-layer index ``li`` with
    causally-zero jitter-source index ``l0i``. ``kind`` steers the alignment
    noise convention: steered legs are independent by construction (full
    means both sides); baseline_text uses disjoint halves over the SHARED
    gen1b c draws (plan §3.4 registered fix 1)."""
    rows: list[dict] = []
    for b, name in enumerate(BIN_NAMES):
        dbar = red["Dbar"][b, li]
        mag = float(dbar.norm())
        u_star = float(red["u_mean"][b, li].norm())
        u_l0 = float(red["u_mean"][b, l0i].norm())
        mag_l0 = float(red["Dbar"][b, l0i].norm())
        emp = (
            mag_l0 * (u_star / u_l0)
            if all(math.isfinite(x) for x in (mag_l0, u_star, u_l0)) and u_l0 > 0
            else float("nan")
        )
        floor = FP16_FLOOR * u_star if math.isfinite(u_star) else float("nan")
        if math.isfinite(emp) and math.isfinite(floor):
            jitter = max(emp, floor)
        elif math.isfinite(floor):
            jitter = floor
        else:
            jitter = float("nan")
        target_full = ceil["full"][b, li] - base["full"][b, li]
        if kind == "steered":
            align = pp._cos_finite(dbar, target_full)
            u_t = pp._unit(target_full)
            null_target = u_t
            null_conv = "full"
        else:
            vals: list[float] = []
            for t_half, d_half in (("even", "odd"), ("odd", "even")):
                tgt = ceil["full"][b, li] - base[t_half][b, li]
                dv = red[f"D_{d_half}"][b, li]
                c = pp._cos_finite(dv, tgt)
                if c is not None:
                    vals.append(c)
            align = float(np.mean(vals)) if len(vals) == 2 else None
            u1 = pp._unit(ceil["full"][b, li] - base["even"][b, li])
            u2 = pp._unit(ceil["full"][b, li] - base["odd"][b, li])
            null_target = (u1 + u2) / 2 if (u1 is not None and u2 is not None) else None
            null_conv = "disjoint"
        per_draw = [
            _finite_or_none(float(v)) for v in red["D"][:, b, li].norm(dim=-1)
        ]  # dispersion (plan §3.4)
        rows.append(
            {
                "bin": name,
                "magnitude": _finite_or_none(mag),
                "per_draw_magnitudes": per_draw,
                "jitter": _finite_or_none(jitter),
                "jitter_empirical_transported": _finite_or_none(emp),
                "jitter_fp16_floor": _finite_or_none(floor),
                "mag_at_l0": _finite_or_none(mag_l0),
                "unhooked_norm_read": _finite_or_none(u_star),
                "unhooked_norm_l0": _finite_or_none(u_l0),
                "alignment": align,
                "alignment_convention": null_conv,
                "target_magnitude": _finite_or_none(float(target_full.norm())),
                "_null_target": null_target,  # stripped before JSON write
            }
        )
    return rows


def _pair_scalar_stat(values: dict[str, float], n_boot: int, seed: int) -> dict:
    """Mean + percentile pair-bootstrap CI (ONE vectorized resample) +
    Wilcoxon companion over a per-pair scalar."""
    arr = np.array([values[k] for k in sorted(values)], dtype=np.float64)
    out: dict = {
        "n_pairs_kept": int(arr.size),
        "per_pair": {k: float(v) for k, v in values.items()},
    }
    if arr.size == 0:
        out.update({"mean": None, "ci95": None, "wilcoxon_p": None})
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    out["mean"] = float(arr.mean())
    out["ci95"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    try:
        from scipy.stats import wilcoxon

        out["wilcoxon_p"] = float(wilcoxon(arr).pvalue) if arr.size >= 2 else None
        if arr.size < 2:
            out["wilcoxon_note"] = "n<2 — companion undefined at this n"
    except ValueError as exc:  # degenerate input (e.g. all-zero diffs) — recorded
        out["wilcoxon_p"] = None
        out["wilcoxon_note"] = f"wilcoxon undefined: {exc}"
    return out


def _log_ratio_pairs(
    num_by_pair: dict[str, dict[str, float | None]],
    den_by_pair: dict[str, dict[str, float | None]],
    num_bins: tuple[str, ...],
    den_bins: tuple[str, ...],
    den_scale: float,
    label: str,
    max_drop_frac: float = MAX_DROP_FRAC,
) -> tuple[dict[str, float], list[str], dict[str, dict]]:
    """Per-pair log(mean num over num_bins) - log(den_scale · mean den over
    den_bins) with the round's TWO NAMED drop classes: all-NaN bin set
    (integrity guard, > max_drop_frac fails LOUD) and non-positive means
    (log degeneracy — recorded exclusion, never guard-tripping)."""
    out: dict[str, float] = {}
    dropped: list[str] = []
    dropped_nonpos: dict[str, dict] = {}
    for pid in sorted(num_by_pair):
        n_vals = [num_by_pair[pid][b] for b in num_bins if num_by_pair[pid].get(b) is not None]
        d_vals = [den_by_pair[pid][b] for b in den_bins if den_by_pair[pid].get(b) is not None]
        if not n_vals or not d_vals:
            dropped.append(pid)
            continue
        n_mean, d_mean = float(np.mean(n_vals)), float(np.mean(d_vals))
        if n_mean <= 0.0 or d_mean <= 0.0:
            dropped_nonpos[pid] = {"num_mean": n_mean, "den_mean": d_mean}
            continue
        out[pid] = float(np.log(n_mean) - np.log(den_scale * d_mean))
    n_total = len(num_by_pair)
    if n_total and len(dropped) / n_total > max_drop_frac:
        raise RuntimeError(
            f"{label}: {len(dropped)}/{n_total} pairs dropped (> {max_drop_frac:.0%}) — "
            f"all-NaN bin sets for: {dropped}"
        )
    if dropped_nonpos:
        logger.info(
            "[phase=h3_stats] %s: %d pair(s) excluded for non-positive means "
            "(log-ratio degeneracy, named class): %s",
            label,
            len(dropped_nonpos),
            sorted(dropped_nonpos),
        )
    return out, dropped, dropped_nonpos


def _null_bands_h(targets: dict[str, torch.Tensor], hidden: int, n_null: int) -> dict[str, float]:
    """Per-target p97.5 of cos(random unit direction, target) — ONE batched
    GEMM over all targets (fresh seed 14154; vectorize rule)."""
    if not targets:
        return {}
    rng = np.random.default_rng(NULL_SEED)
    R = rng.standard_normal((n_null, hidden))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    keys = list(targets)
    T = torch.stack([targets[k] for k in keys]).numpy().T  # (H, n_targets)
    cos = R @ T
    q = np.quantile(cos, 0.975, axis=0)
    return {k: float(v) for k, v in zip(keys, q, strict=True)}


def _cell_label(cell: HookedCell) -> str:
    if cell.kind == "steered":
        return f"steered/{cell.delta_arm}/L{cell.steer_layer}"
    return f"baseline_text/L{cell.steer_layer}"


def phase_h3(cfg: HookedConfig, cells: list[HookedCell], revisions: dict[str, str]) -> None:
    """CPU statistics over the hooked stores + reused unhooked shards
    (streamed one pair at a time; plan §3.4)."""
    pairs = sorted({c.pair_id for c in cells})
    layer_index = {layer: i for i, layer in enumerate(cfg.layers)}
    by_pair: dict[str, list[HookedCell]] = {}
    for c in cells:
        by_pair.setdefault(c.pair_id, []).append(c)

    profile_rows: list[dict] = []
    null_targets: dict[str, torch.Tensor] = {}
    # (label, read_layer) -> pair -> {bin: value|None}
    mag_acc: dict[tuple[str, int], dict[str, dict[str, float | None]]] = {}
    jit_acc: dict[tuple[str, int], dict[str, dict[str, float | None]]] = {}

    for pid in pairs:
        base = _reduce_shard(cfg, f"gen1b/{pid}/c")
        ceil = _reduce_shard(cfg, f"gen1b/{pid}/cprime")
        for cell in sorted(by_pair[pid], key=lambda c: c.hooked_id):
            red = _reduce_hooked(cfg, cell)
            label = _cell_label(cell)
            l0i = layer_index[jitter_l0_for(cfg, cell.steer_layer)]
            headline = headline_read_for(cfg, cell.steer_layer)
            for read_layer in read_layers_for(cfg, cell.steer_layer):
                li = layer_index[read_layer]
                rows = _direct_bin_rows(red, base, ceil, li, l0i, cell.kind)
                key = (label, read_layer)
                mag_acc.setdefault(key, {})[pid] = {r["bin"]: r["magnitude"] for r in rows}
                jit_acc.setdefault(key, {})[pid] = {r["bin"]: r["jitter"] for r in rows}
                for r in rows:
                    nt = r.pop("_null_target")
                    null_key = f"{pid}/R{read_layer}/{r['bin']}/{r['alignment_convention']}"
                    if nt is not None and null_key not in null_targets:
                        null_targets[null_key] = nt
                    r["null_key"] = null_key
                profile_rows.append(
                    {
                        "label": label,
                        "kind": cell.kind,
                        "delta_arm": cell.delta_arm,
                        "steer_layer": cell.steer_layer,
                        "read_layer": read_layer,
                        "headline": read_layer == headline,
                        "jitter_l0_layer": jitter_l0_for(cfg, cell.steer_layer),
                        "pair_id": pid,
                        "pair_type": pp.pair_type(pid),
                        "flags": {
                            "terse": pid == TERSE_PAIR,
                            "formal": pid == FORMAL_PAIR,
                            "medical_noise_target": pid == MEDICAL_PAIR,
                            "villain": "villain" in pid,
                        },
                        "n_kept_draws": red["n_kept"],
                        "ctx_vec_max_dev": _finite_or_none(red["ctx_vec_max_dev"]),
                        "bins": rows,
                    }
                )

    # Row-coverage assert (plan §7): the registered pair x label x read-layer
    # row set must be complete BEFORE any R̄ is computed.
    for (label, read_layer), per_pair in sorted(mag_acc.items()):
        missing = sorted(set(pairs) - set(per_pair))
        assert not missing, f"row coverage incomplete for {label}@R{read_layer}: {missing}"

    bands = _null_bands_h(null_targets, cfg.hidden, cfg.n_null)
    for row in profile_rows:
        for r in row["bins"]:
            r["null_p975"] = bands.get(r["null_key"])

    # Registered statistics per (label, read layer); the lattice binds at the
    # HEADLINE read layer of each of the 4 primary steered cells (plan §7).
    labels: dict[str, dict] = {}
    for cell in cells:
        label = _cell_label(cell)
        if label in labels:
            continue
        headline = headline_read_for(cfg, cell.steer_layer)
        per_read: dict[str, dict] = {}
        for read_layer in read_layers_for(cfg, cell.steer_layer):
            key = (label, read_layer)
            r_vals, r_drop, r_nonpos = _log_ratio_pairs(
                mag_acc[key],
                jit_acc[key],
                LATE_BINS,
                LATE_BINS,
                JITTER_MULT,
                f"R[{label}@R{read_layer}]",
            )
            r_stat = _pair_scalar_stat(r_vals, cfg.n_boot, BOOT_SEED)
            ci = r_stat["ci95"]
            if r_stat["mean"] is None or ci is None:
                verdict = "undefined"
            elif r_stat["mean"] > 0 and ci[0] > 0:
                verdict = "direct-persistence"
            elif ci[1] < 0:
                verdict = "late-null"
            else:
                verdict = "inconclusive"
            d_vals, d_drop, d_nonpos = _log_ratio_pairs(
                mag_acc[key],
                mag_acc[key],
                EARLY_BINS,
                LATE_BINS,
                1.0,
                f"Delta_direct[{label}@R{read_layer}]",
            )
            w_vals, w_drop, w_nonpos = _log_ratio_pairs(
                mag_acc[key],
                mag_acc[key],
                WIDTH_EARLY_BINS,
                LATE_BINS,
                1.0,
                f"Delta_width[{label}@R{read_layer}]",
            )
            per_read[str(read_layer)] = {
                "headline": read_layer == headline,
                "R": {
                    **r_stat,
                    "verdict": verdict,
                    "dropped_pairs": r_drop,
                    "dropped_nonpositive_pairs": r_nonpos,
                    "late_bins": list(LATE_BINS),
                    "jitter_mult": JITTER_MULT,
                },
                "delta_direct": {
                    **_pair_scalar_stat(d_vals, cfg.n_boot, BOOT_SEED),
                    "early_bins": list(EARLY_BINS),
                    "late_bins": list(LATE_BINS),
                    "dropped_pairs": d_drop,
                    "dropped_nonpositive_pairs": d_nonpos,
                },
                "delta_width": {
                    **_pair_scalar_stat(w_vals, cfg.n_boot, BOOT_SEED),
                    "early_bins": list(WIDTH_EARLY_BINS),
                    "late_bins": list(LATE_BINS),
                    "dropped_pairs": w_drop,
                    "dropped_nonpositive_pairs": w_nonpos,
                },
            }
        labels[label] = {
            "kind": cell.kind,
            "delta_arm": cell.delta_arm if cell.kind == "steered" else "context",
            "steer_layer": cell.steer_layer,
            "lattice": cell.kind == "steered",
            "headline_read_layer": headline,
            "verdict": per_read[str(headline)]["R"]["verdict"],
            "read_layers": per_read,
        }

    # Steered-text vs baseline-text direct component (context arm; plan §3.4).
    svb: dict[str, dict] = {}
    for steer in cfg.steer_layers:
        headline = headline_read_for(cfg, steer)
        s_key = (f"steered/context/L{steer}", headline)
        b_key = (f"baseline_text/L{steer}", headline)
        per_bin: dict[str, dict] = {}
        for b in BIN_NAMES:
            s_vals = [v[b] for v in mag_acc[s_key].values() if v.get(b) is not None]
            b_vals = [v[b] for v in mag_acc[b_key].values() if v.get(b) is not None]
            per_bin[b] = {
                "steered_mag_mean": float(np.mean(s_vals)) if s_vals else None,
                "baseline_mag_mean": float(np.mean(b_vals)) if b_vals else None,
                "n_pairs_steered": len(s_vals),
                "n_pairs_baseline": len(b_vals),
            }
        svb[f"L{steer}"] = {"read_layer": headline, "bins": per_bin}

    conventions = {
        "diff": "fp16(hooked) - fp16(unhooked) computed in fp32 (symmetric storage "
        "quantization on both legs; plan §11)",
        "jitter": "J_p(b) = max(mag@l0 · ‖Pu@read‖/‖Pu@l0‖, 2^-11·√2·‖Pu@read‖); "
        "l0 = nearest causally-zero captured layer (steer layer itself)",
        "R": "R_p = log(mean late-bin mag) - log(2 · mean late-bin J_p); "
        "direct-persistence iff mean>0 AND ci95 excludes 0 positively; "
        "late-null iff ci95 wholly below 0 (plan §7 lattice)",
        "alignment": "steered cells: legs independent by construction (gen1c diff vs "
        "gen1b target); baseline-text cells: disjoint halves (registered fix 1)",
        "read_layers": "pre-registered nearest-above-steer headline (14->17, 20->21); "
        "all above-steer layers reported side by side; NO max over layers",
    }
    _write_json_atomic(
        cfg.out_root / "per_pair_direct_profiles.json",
        {
            "bin_names": list(BIN_NAMES),
            "alpha": ALPHA,
            "conventions": conventions,
            "rows": profile_rows,
            "repro": _repro(cfg),
        },
    )
    _write_json_atomic(
        cfg.out_root / "jitter_reference.json",
        {
            "conventions": conventions["jitter"],
            "fp16_floor_coeff": FP16_FLOOR,
            "jitter_mult": JITTER_MULT,
            "secondary_reference": (
                "the round's committed parity_gate_report.json jitter class "
                "(10/10 passed, 0 warn, cos >= 0.9999) — same surface, same rig"
            ),
            "rows": [
                {
                    "label": row["label"],
                    "pair_id": row["pair_id"],
                    "read_layer": row["read_layer"],
                    "jitter_l0_layer": row["jitter_l0_layer"],
                    "bins": [
                        {
                            k: r[k]
                            for k in (
                                "bin",
                                "jitter",
                                "jitter_empirical_transported",
                                "jitter_fp16_floor",
                                "mag_at_l0",
                                "unhooked_norm_read",
                                "unhooked_norm_l0",
                            )
                        }
                        for r in row["bins"]
                    ],
                }
                for row in profile_rows
            ],
            "repro": _repro(cfg),
        },
    )
    _write_json_atomic(
        cfg.out_root / "null_bands_direct.json",
        {
            "seed": NULL_SEED,
            "n_null": cfg.n_null,
            "quantile": 0.975,
            "bands": bands,
            "repro": _repro(cfg),
        },
    )
    _write_json_atomic(
        cfg.out_root / "summary.json",
        {
            "labels": labels,
            "steered_vs_baseline": svb,
            "conventions": conventions,
            "seeds": {"bootstrap": BOOT_SEED, "null": NULL_SEED},
            "n_boot": cfg.n_boot,
            "n_null": cfg.n_null,
            "regime": _regime(cfg, revisions),
            "repro": _repro(cfg),
        },
    )
    logger.info(
        "[phase=h3_stats] wrote per_pair_direct_profiles/jitter_reference/"
        "null_bands_direct/summary (%d rows, %d null targets)",
        len(profile_rows),
        len(null_targets),
    )


# ── h4: upload residue ────────────────────────────────────────────────


def phase_h4(cfg: HookedConfig, cells: list[HookedCell], revisions: dict[str, str]) -> None:
    """Store manifest + batched residue upload (ONE folder commit; the git
    commit of the eval_results JSONs is the dispatcher's commit_push_verify)."""
    stored = sorted(str(p.relative_to(cfg.tensors_root)) for p in cfg.tensors_root.rglob("*.pt"))
    _write_json_atomic(
        cfg.tensors_root / "manifest.json",
        {
            "n_files": len(stored),
            "files": stored,
            "regime": _regime(cfg, revisions),
            "n_cells_registered": len(cells),
            "repro": _repro(cfg),
        },
    )
    upload_artifact(cfg, cfg.tensors_root, HOOKED_TENSOR_PREFIX)
    logger.info("[phase=h4_upload] uploaded %d store files + manifest", len(stored))


# ── main ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    cfg = build_config(args)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    cfg.tensors_root.mkdir(parents=True, exist_ok=True)
    cfg.stage_root.mkdir(parents=True, exist_ok=True)
    if args.staging_probe:
        staging_probe(cfg)
        return
    if not cfg.tiny and cfg.upload_mode == "hf":
        assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing — required for --upload hf"
    if cfg.tiny and "h0" in cfg.phases:
        build_tiny_substrate(cfg)  # idempotent (sentinel); metas must exist to enumerate
    cells = enumerate_hooked_cells(cfg)
    revisions: dict[str, str] | None = None
    state: Manifest | None = None
    model = tok = None

    def _revs() -> dict[str, str]:
        nonlocal revisions
        if revisions is None:
            revisions = _resolve_revisions(cfg)
        return revisions

    def _state() -> Manifest:
        nonlocal state
        if state is None:
            state = Manifest.load_or_init(
                cfg.out_root / "hooked_manifest.json", _regime(cfg, _revs())
            )
        return state

    def _model():
        nonlocal model, tok
        if model is None:
            model, tok = load_model_and_tokenizer(cfg)
        return model, tok

    for phase in cfg.phases:
        if phase == "h0":
            phase_h0(cfg, cells)
        elif phase == "h1":
            m, t = _model()
            phase_h1(cfg, cells, m, t, _state())
        elif phase == "h2":
            m, t = _model()
            phase_h2(cfg, cells, m, t, _state())
        elif phase == "h3":
            phase_h3(cfg, cells, _revs())
        elif phase == "h4":
            phase_h4(cfg, cells, _revs())
    logger.info("[phase=hooked_driver_complete] phases=%s", ",".join(cfg.phases))


if __name__ == "__main__":
    main()
