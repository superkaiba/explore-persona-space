#!/usr/bin/env python
"""Issue #1415 — answer-position-shift-profile follow-up driver (plan v8).

Re-forwards the parent run's PERSISTED completions (gen1c steered L14/L20 a4,
gen1b baselines + ceilings, gen_rep43/44) through the UNHOOKED capture rig and
replaces the parent's span-MEAN answer summary with a 13-bin per-position
profile (steering.bin_matrix / capture_binned_answer_profiles). NO generation,
NO steering hook, NO judging — the measured object is the position-resolved
activation footprint of the realized completions (plan §1 MEASUREMENT REGIME).

Phases (one driver, `--phase all` in production; the --tiny e2e runs the SAME
chain on a 2-layer from-config Qwen over the real vocab — PASS_UNIFIED):

  p0_stage    enumerate the 336 cells from git metadata (hard count asserts),
              stage the draws JSONs from HF at the PINNED revisions (scoped
              per-file hub.stage_hub_file, <=6 workers — NEVER an unscoped
              listing of the ~1M-file repo), cross-assert meta.context ==
              draws.context + n_draws per cell (LOUD, names cell_ids).
  p1_parity   §3.5 integrity HALT: recompute the plain span MEAN through the
              NEW code path for 10 spot cells and compare per-layer cosine to
              the parent's stored activations_steered/<pair>__<arm>.pt
              v_a_mean. HALT (report JSON + rc=8, artifact-routed by the
              dispatcher) when cosine < 0.995 on > 2 of 10 cells; WARN below
              0.9999; demote to WARN-only (reason persisted) when the bundle
              lacks a usable reference field.
  p2_capture  batched teacher-forced binned capture per cell ->
              tensors_root/<cell_id>.pt (fp16), checkpoint-per-cell manifest
              keyed on every regime knob, resume skips completed cells,
              incremental per-family HF upload (ONE folder commit per family).
  p3_profiles CPU statistics (§3.4): per-bin shift magnitude / alignment
              (disjoint-halves PRIMARY + shared-baseline labeled SECONDARY),
              per-bin random-direction null (ONE batched GEMM, seed 14150),
              baseline split-half noise floor, EARLY-vs-LATE Delta with
              pair-bootstrap CI (ONE vectorized resample, seed 14151) +
              Wilcoxon companion + registered companions Delta_floor /
              Delta_width, named-dropped-pairs list, 28-pair primary +
              27-pair exclude-medical + matched-length sensitivity reads.
  p4_upload   batched upload_folder of any tensor residue + the store
              manifest.json (git commit of the eval_results JSONs is the
              DISPATCHER's commit_push_verify, not this driver).

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

# Reused verbatim from the parent driver (plan §3.6 REUSED list).
from issue1415_run_phase1 import (  # noqa: E402
    HF_DATA_REPO,
    RAW_PREFIX,
    STEERED_TENSOR_PREFIX,
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
    capture_binned_answer_profiles,
)

logger = logging.getLogger("issue1415.position_profile")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PROFILE_TENSOR_PREFIX = "analysis_tensors/issue_1415/position_profile"

# Pinned source revisions (plan §10; probed at plan time 2026-07-22). The rep
# short sha is RESOLVED to the full 40-char sha at p0 and recorded.
PARENT_REVISION = "79dacd5239869e8de65c89a04437a606af436ad0"
REP_REVISION_SHORT = "7ea5ae87e9"

LAYERS_FULL = (7, 10, 14, 17, 20, 21, 24)
HEADLINE_STEER_LAYERS_FULL = (14, 20)  # matched-layer reads; NO max over layers
PARITY_LAYER_FULL = 20  # the parent canonical <pair>__<arm>.pt cells are L20/a4
ALPHA = 4.0
N_DRAWS_FULL = 10
BINS_VERSION = "13bins-v1"

NULL_SEED = 14150
N_NULL_FULL = 500
BOOT_SEED = 14151
N_BOOT_FULL = 10_000

EARLY_BINS = ("first", "tok2_5")
LATE_BINS = ("dec8", "dec9", "dec10")
WIDTH_EARLY_BINS = ("dec1", "dec2")  # width-matched companion Delta_width
MAX_DROP_FRAC = 0.20  # fail loud when > 20% of pairs drop from Delta in a cell

PARITY_HALT_COS = 0.995
PARITY_WARN_COS = 0.9999
PARITY_MAX_BAD = 2  # HALT when MORE THAN this many spot cells fall below HALT_COS
PARITY_N_PAIRS = 5  # alphabetically-first pairs, both arms -> 10 spot cells
RC_PARITY_HALT = 8  # designed artifact-routed HALT (never a bare rc=1)

MEDICAL_PAIR = "m685_07_medical_doctor"  # target split-half 0.049 (parent) — flagged
TERSE_PAIR = "m685_04_terse"
FORMAL_PAIR = "m685_05_formal"
MATCHED_LENGTH_RATIO = (0.5, 2.0)  # steered/baseline median-length sensitivity window

REP_LABELS = (43, 44)


# ── config + cell inventory ───────────────────────────────────────────


@dataclass(frozen=True)
class ProfileCell:
    """One completion cell to re-forward (draws JSON + git meta)."""

    cell_id: str  # HF path stem under RAW_PREFIX, e.g. gen1c/prefix/<pair>/L14/a4
    family: str  # gen1b | gen1c | rep43 | rep44
    role: str  # steered | baseline | ceiling
    pair_id: str
    arm: str | None  # extraction arm (steered cells)
    steer_layer: int | None  # steer layer (steered cells)
    meta_path: Path
    revision_key: str  # "parent" | "rep"


@dataclass
class ProfileConfig:
    """Resolved run configuration (duck-typed superset of the parent-helper
    fields `_repro` / `load_model_and_tokenizer` / `upload_artifact` read:
    model_id, tiny, hidden, n_model_layers, upload_mode, bulk_root)."""

    tiny: bool
    out_root: Path
    tensors_root: Path
    stage_root: Path
    bulk_root: Path  # local-mirror upload dest root (tiny)
    hub_mirror_root: Path | None  # tiny fixture "hub" (draws + parity bundles)
    meta_root_primary: Path  # .../phase1/cells
    meta_roots_rep: dict[int, Path]  # rep label -> .../phase1_rep<S>/cells
    upload_mode: str  # hf | local-mirror
    model_id: str
    n_draws: int
    capture_batch: int
    layers: tuple[int, ...]
    headline_layers: tuple[int, ...]
    parity_layer: int
    rep_layer: int
    n_pairs_expected: int
    hidden: int
    n_model_layers: int
    device: str
    n_null: int
    n_boot: int
    n_parity_pairs: int
    phases: tuple[str, ...] = field(default_factory=tuple)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--phase", default="all", choices=["all", "p0", "p1", "p2", "p3", "p4"])
    p.add_argument("--tiny", action="store_true", help="CPU tiny e2e (2-layer model, fixtures)")
    p.add_argument(
        "--work-root",
        default=None,
        help="tiny scratch root (default data/issue_1415/tiny_smoke/position_profile)",
    )
    p.add_argument("--out-root", default=None, help="override the eval_results output root")
    p.add_argument("--capture-batch", type=int, default=8)
    p.add_argument("--upload", default=None, choices=["hf", "local-mirror"])
    p.add_argument("--n-null", type=int, default=None)
    p.add_argument("--n-boot", type=int, default=None)
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> ProfileConfig:
    """Resolve the run configuration; tiny scales every axis down but keeps
    the IDENTICAL phase chain + code paths (PASS_UNIFIED smoke contract)."""
    phases = ("p0", "p1", "p2", "p3", "p4") if args.phase == "all" else (args.phase,)
    if args.tiny:
        work = Path(args.work_root or "data/issue_1415/tiny_smoke/position_profile")
        out_root = Path(args.out_root) if args.out_root else work / "out"
        return ProfileConfig(
            tiny=True,
            out_root=out_root,
            tensors_root=work / "tensors",
            stage_root=work / "stage",
            bulk_root=work / "bulk",
            hub_mirror_root=work / "hub_mirror",
            meta_root_primary=work / "phase1" / "cells",
            meta_roots_rep={s: work / f"phase1_rep{s}" / "cells" for s in REP_LABELS},
            upload_mode=args.upload or "local-mirror",
            model_id=MODEL_ID,
            n_draws=2,
            capture_batch=args.capture_batch,
            layers=(0, 1),
            headline_layers=(0, 1),
            parity_layer=1,
            rep_layer=0,
            n_pairs_expected=2,
            hidden=32,
            n_model_layers=2,
            device="cpu",
            n_null=args.n_null or 50,
            n_boot=args.n_boot or 200,
            n_parity_pairs=2,
            phases=phases,
        )
    assert args.work_root is None, "--work-root is tiny-only"
    er = REPO_ROOT / "eval_results" / "issue_1415"
    return ProfileConfig(
        tiny=False,
        out_root=Path(args.out_root) if args.out_root else er / "answer_position_shift_profile",
        tensors_root=REPO_ROOT / "data" / "issue_1415" / "position_profile" / "tensors",
        stage_root=REPO_ROOT / "data" / "issue_1415" / "hf_dl" / "position_profile",
        bulk_root=REPO_ROOT / "data" / "issue_1415" / "position_profile",
        hub_mirror_root=None,
        meta_root_primary=er / "phase1" / "cells",
        meta_roots_rep={s: er / f"phase1_rep{s}" / "cells" for s in REP_LABELS},
        upload_mode=args.upload or "hf",
        model_id=MODEL_ID,
        n_draws=N_DRAWS_FULL,
        capture_batch=args.capture_batch,
        layers=LAYERS_FULL,
        headline_layers=HEADLINE_STEER_LAYERS_FULL,
        parity_layer=PARITY_LAYER_FULL,
        rep_layer=14,
        n_pairs_expected=28,
        hidden=3584,
        n_model_layers=28,
        device="cuda:0",
        n_null=args.n_null or N_NULL_FULL,
        n_boot=args.n_boot or N_BOOT_FULL,
        n_parity_pairs=PARITY_N_PAIRS,
        phases=phases,
    )


def list_pairs(cfg: ProfileConfig) -> list[str]:
    """Pair ids from the git gen1b metadata dirs (the metas ARE in git; the
    pair bank itself is gitignored). Asserts the expected pair count."""
    gen1b = cfg.meta_root_primary / "gen1b"
    assert gen1b.is_dir(), f"gen1b cell metadata missing: {gen1b}"
    pairs = sorted(p.name for p in gen1b.iterdir() if p.is_dir())
    assert len(pairs) == cfg.n_pairs_expected, (len(pairs), cfg.n_pairs_expected, gen1b)
    return pairs


def pair_type(pair_id: str) -> str:
    """Inherited pair-typing convention: m685_/m779_ = matched, cross_ = cross."""
    return "matched" if pair_id.startswith(("m685_", "m779_")) else "cross"


def enumerate_cells(cfg: ProfileConfig) -> list[ProfileCell]:
    """The FULL registered cell inventory (plan §3.1), enumerated from git
    metadata with hard per-family count asserts. Every phase's cell list
    derives from THIS function (smoke-architecture PASS_UNIFIED contract)."""
    pairs = list_pairs(cfg)
    cells: list[ProfileCell] = []
    for pid in pairs:
        for tag, role in (("c", "baseline"), ("cprime", "ceiling")):
            cid = f"gen1b/{pid}/{tag}"
            cells.append(
                ProfileCell(
                    cell_id=cid,
                    family="gen1b",
                    role=role,
                    pair_id=pid,
                    arm=None,
                    steer_layer=None,
                    meta_path=cfg.meta_root_primary / f"{cid}.json",
                    revision_key="parent",
                )
            )
    for arm in ("prefix", "context"):
        for pid in pairs:
            for layer in cfg.headline_layers:
                cid = f"gen1c/{arm}/{pid}/L{layer}/a{_fmt(ALPHA)}"
                cells.append(
                    ProfileCell(
                        cell_id=cid,
                        family="gen1c",
                        role="steered",
                        pair_id=pid,
                        arm=arm,
                        steer_layer=layer,
                        meta_path=cfg.meta_root_primary / f"{cid}.json",
                        revision_key="parent",
                    )
                )
    for rep in REP_LABELS:
        meta_root = cfg.meta_roots_rep[rep]
        for pid in pairs:
            cid = f"gen_rep{rep}/base/{pid}/c"
            cells.append(
                ProfileCell(
                    cell_id=cid,
                    family=f"rep{rep}",
                    role="baseline",
                    pair_id=pid,
                    arm=None,
                    steer_layer=None,
                    meta_path=meta_root / f"{cid}.json",
                    revision_key="rep",
                )
            )
        for arm in ("prefix", "context"):
            for pid in pairs:
                cid = f"gen_rep{rep}/{arm}/{pid}/L{cfg.rep_layer}/a{_fmt(ALPHA)}"
                cells.append(
                    ProfileCell(
                        cell_id=cid,
                        family=f"rep{rep}",
                        role="steered",
                        pair_id=pid,
                        arm=arm,
                        steer_layer=cfg.rep_layer,
                        meta_path=meta_root / f"{cid}.json",
                        revision_key="rep",
                    )
                )
    n_pairs = len(pairs)
    counts = {
        "gen1b": 2 * n_pairs,
        "gen1c": 2 * n_pairs * len(cfg.headline_layers),
        **{f"rep{s}": 3 * n_pairs for s in REP_LABELS},
    }
    for fam, expected in counts.items():
        got = sum(1 for c in cells if c.family == fam)
        assert got == expected, f"cell inventory mismatch for {fam}: {got} != {expected}"
    missing_meta = [c.cell_id for c in cells if not c.meta_path.exists()]
    if missing_meta:
        raise RuntimeError(
            f"{len(missing_meta)} cell metadata file(s) missing from git — cell_ids: {missing_meta}"
        )
    return cells


# ── tiny fixture (PASS_UNIFIED smoke substrate) ───────────────────────

_TINY_PAIRS = ("tiny_00_alpha", "tiny_01_beta")
_TINY_CONTEXTS = {
    "tiny_00_alpha": {
        "c": {"system": "You are a meticulous cartographer.", "user": "Describe your maps."},
        "cprime": {"system": "You are a cheerful gardener.", "user": "Describe your maps."},
    },
    "tiny_01_beta": {
        "c": {"system": "You are a patient librarian.", "user": "Recommend a shelf."},
        "cprime": {"system": "You are a brisk auctioneer.", "user": "Recommend a shelf."},
    },
}


def _tiny_draw(cell_id: str, i: int) -> str:
    """Deterministic >=12-token fixture completion (all 13 bins non-empty —
    the LATE-decile floor for the Delta statistic; smoke-gate calibration).
    The LEADING token is cell- AND draw-specific (stable crc, never hash() —
    PYTHONHASHSEED randomizes across processes): completions sharing a
    causal-prefix across cells/draws would zero the EARLY-bin shift/floor
    magnitudes exactly and -inf the Delta log-ratio."""
    import zlib

    lead = zlib.crc32(cell_id.encode()) % 977 + 1009 * (i + 1)
    return (
        f"{lead} marks draw {i} for {cell_id.replace('/', ' ')}; the journey wanders "
        "slowly across the wide valley, noting each landmark, river crossing, and "
        "quiet village along the way."
    )


def build_tiny_fixture(cfg: ProfileConfig) -> None:
    """Write the tiny fixture through the REAL cell-meta/draws schema (the
    parent `_persist_gen_cell` field set) + parity reference bundles computed
    with `capture_vectors` on the SAME deterministic tiny model, into a local
    hub mirror that p0/p1 consume via the identical fetch call path."""
    from explore_persona_space.experiments.issue1415.steering import capture_vectors

    assert cfg.tiny and cfg.hub_mirror_root is not None
    sentinel = cfg.hub_mirror_root / ".fixture_complete"
    if sentinel.exists():
        return
    logger.info("[phase=p0_stage] building tiny fixture under %s", cfg.hub_mirror_root)
    model, tok = load_model_and_tokenizer(cfg)

    def _write_cell(meta_path: Path, cid: str, pid: str, ctx: dict, extra: dict) -> list[str]:
        draws = [_tiny_draw(cid, i) for i in range(cfg.n_draws)]
        common = {
            "cell_id": cid,
            "phase": extra.pop("phase"),
            "pair_id": pid,
            "context": ctx,
            "layer": extra.pop("layer", None),
            "alpha": extra.pop("alpha", None),
            "all_positions": False,
            "delta_key": None,
            "n_draws": cfg.n_draws,
            "seed_base": 0,
            "temperature": 1.0,
            "max_new_tokens": 64,
            **extra,
        }
        _write_json_atomic(
            cfg.hub_mirror_root / RAW_PREFIX / f"{cid}.json",
            {**common, "draws": draws, "repro": _repro(cfg)},
        )
        _write_json_atomic(meta_path, {**common, "completions_file": f"{cid}.json"})
        return draws

    draws_by_cell: dict[str, list[str]] = {}
    for pid in _TINY_PAIRS:
        ctxs = _TINY_CONTEXTS[pid]
        for tag in ("c", "cprime"):
            cid = f"gen1b/{pid}/{tag}"
            draws_by_cell[cid] = _write_cell(
                cfg.meta_root_primary / f"{cid}.json", cid, pid, ctxs[tag], {"phase": "phase1b"}
            )
        for arm in ("prefix", "context"):
            for layer in cfg.headline_layers:
                cid = f"gen1c/{arm}/{pid}/L{layer}/a{_fmt(ALPHA)}"
                draws_by_cell[cid] = _write_cell(
                    cfg.meta_root_primary / f"{cid}.json",
                    cid,
                    pid,
                    ctxs["c"],
                    {
                        "phase": "phase1c_grid",
                        "layer": layer,
                        "alpha": ALPHA,
                        "extraction_arm": arm,
                    },
                )
            for rep in REP_LABELS:
                cid = f"gen_rep{rep}/{arm}/{pid}/L{cfg.rep_layer}/a{_fmt(ALPHA)}"
                draws_by_cell[cid] = _write_cell(
                    cfg.meta_roots_rep[rep] / f"{cid}.json",
                    cid,
                    pid,
                    ctxs["c"],
                    {
                        "phase": "phase1c_layers",
                        "layer": cfg.rep_layer,
                        "alpha": ALPHA,
                        "extraction_arm": arm,
                    },
                )
        for rep in REP_LABELS:
            cid = f"gen_rep{rep}/base/{pid}/c"
            draws_by_cell[cid] = _write_cell(
                cfg.meta_roots_rep[rep] / f"{cid}.json", cid, pid, ctxs["c"], {"phase": "phase1b"}
            )

    # Parity reference bundles: the parent-convention span means for the spot
    # cells, computed through capture_vectors (the PARENT code path) on the
    # SAME deterministic tiny model -> self-consistent parity (cos ~ 1.0).
    for pid in _TINY_PAIRS:
        for arm in ("prefix", "context"):
            src_cell = f"gen1c/{arm}/{pid}/L{cfg.parity_layer}/a{_fmt(ALPHA)}"
            cap = capture_vectors(
                model,
                tok,
                [_TINY_CONTEXTS[pid]["c"]],
                list(cfg.layers),
                completions=[draws_by_cell[src_cell]],
                batch_size=cfg.capture_batch,
            )
            rec = cap["per_context"][0]
            _save_pt_atomic(
                cfg.hub_mirror_root / STEERED_TENSOR_PREFIX / f"{pid}__{arm}.pt",
                {
                    "pair_id": pid,
                    "layers": list(cfg.layers),
                    "v_a_mean": rec["v_a_mean"],
                    "v_a_per_completion": rec["v_a_per_completion"],
                    "n_empty_completions": rec["n_empty_completions"],
                    "canonical_of": src_cell,
                },
            )
    sentinel.write_text("ok")


# ── p0: stage + verify ────────────────────────────────────────────────


def _resolve_revisions(cfg: ProfileConfig) -> dict[str, str]:
    """Resolve + persist the pinned source revisions (rep short sha -> full)."""
    rev_path = cfg.out_root / "revisions.json"
    if rev_path.exists():
        return json.loads(rev_path.read_text())["revisions"]
    if cfg.tiny:
        revisions = {"parent": "local-fixture", "rep": "local-fixture"}
    else:
        from explore_persona_space.orchestrate import hub
        from huggingface_hub import HfApi

        info = hub.retry_transient(
            lambda: HfApi().repo_info(
                HF_DATA_REPO, repo_type="dataset", revision=REP_REVISION_SHORT
            ),
            what=f"repo_info({HF_DATA_REPO}@{REP_REVISION_SHORT})",
        )
        revisions = {"parent": PARENT_REVISION, "rep": info.sha}
        assert revisions["rep"].startswith(REP_REVISION_SHORT), revisions["rep"]
    _write_json_atomic(rev_path, {"revisions": revisions, "repro": _repro(cfg)})
    return revisions


def _staged_path(cfg: ProfileConfig, cell_id: str) -> Path:
    return cfg.stage_root / f"{cell_id}.json"


def fetch_draws(cfg: ProfileConfig, cell: ProfileCell, revisions: dict[str, str]) -> Path:
    """Stage ONE cell's draws JSON (identity mapping — the consumer opens the
    exact fetch destination; artifact-reuse leg (h)(iv) 'no staging
    transformation'). hf mode = the canonical retried atomic
    `hub.stage_hub_file` at the pinned revision; tiny = local-mirror copy
    through the SAME call path shape."""
    target = _staged_path(cfg, cell.cell_id)
    if target.exists():
        return target
    if cfg.tiny:
        src = cfg.hub_mirror_root / RAW_PREFIX / f"{cell.cell_id}.json"
        assert src.exists(), f"tiny fixture missing draws for {cell.cell_id}: {src}"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / (target.name + ".tmp")
        tmp.write_text(src.read_text())
        os.replace(tmp, target)
        return target
    from explore_persona_space.orchestrate import hub

    return hub.stage_hub_file(
        HF_DATA_REPO,
        f"{RAW_PREFIX}/{cell.cell_id}.json",
        target,
        repo_type="dataset",
        revision=revisions[cell.revision_key],
    )


def load_staged_draws(cfg: ProfileConfig, cell_id: str) -> dict:
    """Production loader for a staged draws JSON (the staging-probe consumer)."""
    p = _staged_path(cfg, cell_id)
    assert p.exists(), f"staged draws missing for {cell_id}: {p} (run p0 first)"
    blob = json.loads(p.read_text())
    assert isinstance(blob.get("draws"), list) and "context" in blob, sorted(blob)
    return blob


def phase_p0(cfg: ProfileConfig, cells: list[ProfileCell]) -> None:
    logger.info("[phase=p0_stage] staging %d draws JSONs (<=6 workers)", len(cells))
    if cfg.tiny:
        build_tiny_fixture(cfg)
    revisions = _resolve_revisions(cfg)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(fetch_draws, cfg, c, revisions): c for c in cells}
        failures: list[str] = []
        for fut, cell in futs.items():
            try:
                fut.result()
            except Exception as exc:  # collected, then raised LOUD naming cell_ids
                logger.error("[phase=p0_stage] fetch failed for %s: %s", cell.cell_id, exc)
                failures.append(cell.cell_id)
        if failures:
            raise RuntimeError(f"p0 staging failed for {len(failures)} cell(s): {failures}")
    mismatches: list[str] = []
    for cell in cells:
        meta = json.loads(cell.meta_path.read_text())
        blob = load_staged_draws(cfg, cell.cell_id)
        if meta["context"] != blob["context"]:
            mismatches.append(f"{cell.cell_id} (meta.context != draws.context)")
        if len(blob["draws"]) != cfg.n_draws or meta["n_draws"] != cfg.n_draws:
            mismatches.append(
                f"{cell.cell_id} (n_draws meta={meta['n_draws']} draws={len(blob['draws'])} "
                f"!= {cfg.n_draws})"
            )
    if mismatches:
        raise RuntimeError(f"p0 verification failed for {len(mismatches)} cell(s): {mismatches}")
    logger.info("[phase=p0_stage] staged + verified %d cells", len(cells))


# ── p1: parity gate (§3.5) ────────────────────────────────────────────


def parity_spot_cells(cfg: ProfileConfig, cells: list[ProfileCell]) -> list[ProfileCell]:
    """The deterministic spot set: alphabetically-first pairs x both arms at
    the parity layer (the parent's canonical L20/a4 cells)."""
    pairs = sorted({c.pair_id for c in cells})[: cfg.n_parity_pairs]
    spot = [
        c
        for c in cells
        if c.family == "gen1c" and c.steer_layer == cfg.parity_layer and c.pair_id in pairs
    ]
    expected = 2 * len(pairs)
    assert len(spot) == expected, (len(spot), expected)
    return sorted(spot, key=lambda c: (c.pair_id, c.arm))


def _fetch_parity_bundle(cfg: ProfileConfig, pair_id: str, arm: str) -> Path:
    target = cfg.stage_root / "parity_refs" / f"{pair_id}__{arm}.pt"
    if target.exists():
        return target
    if cfg.tiny:
        src = cfg.hub_mirror_root / STEERED_TENSOR_PREFIX / f"{pair_id}__{arm}.pt"
        assert src.exists(), f"tiny fixture missing parity bundle: {src}"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / (target.name + ".tmp")
        tmp.write_bytes(src.read_bytes())
        os.replace(tmp, target)
        return target
    from explore_persona_space.orchestrate import hub

    return hub.stage_hub_file(
        HF_DATA_REPO,
        f"{STEERED_TENSOR_PREFIX}/{pair_id}__{arm}.pt",
        target,
        repo_type="dataset",
        revision=PARENT_REVISION,
    )


def parity_verdict(per_cell: list[dict], demotions: list[dict]) -> dict:
    """Pure §3.5 verdict over per-spot-cell min-layer cosines (unit-testable;
    the HALT branch gets its own degenerate-input probe). HALT fires when
    MORE THAN PARITY_MAX_BAD comparable cells read below PARITY_HALT_COS."""
    n_bad = sum(1 for r in per_cell if r["min_cos"] < PARITY_HALT_COS)
    n_warn = sum(1 for r in per_cell if PARITY_HALT_COS <= r["min_cos"] < PARITY_WARN_COS)
    fired = n_bad > PARITY_MAX_BAD
    return {
        "fired": fired,
        "demoted_to_warn": bool(demotions) and not per_cell,
        "n_cells": len(per_cell),
        "n_bad": n_bad,
        "n_warn": n_warn,
        "halt_cos": PARITY_HALT_COS,
        "warn_cos": PARITY_WARN_COS,
        "max_bad": PARITY_MAX_BAD,
        "cells": per_cell,
        "demotions": demotions,
    }


def phase_p1(cfg: ProfileConfig, cells: list[ProfileCell], model, tok, state: Manifest) -> None:
    """§3.5 parity HALT: recompute the plain span mean through the NEW binned
    code path for the spot cells; per-layer cosine vs the parent's stored
    v_a_mean. The spot captures are ALSO persisted as their p2 tensors (same
    save path — the unified-phase economy)."""
    spot = parity_spot_cells(cfg, cells)
    logger.info("[phase=p1_parity] %d spot cells (parity layer L%d)", len(spot), cfg.parity_layer)
    per_cell: list[dict] = []
    demotions: list[dict] = []
    t0 = time.monotonic()
    first_n = 0
    for cell in spot:
        bundle_path = _fetch_parity_bundle(cfg, cell.pair_id, cell.arm)
        # Realized-keys check (artifact-reuse (c)): mmap read, storages untouched.
        keys = set(torch.load(bundle_path, map_location="cpu", mmap=True, weights_only=True))
        blob = torch.load(bundle_path, map_location="cpu", weights_only=True)
        usable = "v_a_mean" in keys and blob.get("layers") == list(cfg.layers)
        blob_layers = blob.get("layers")
        cap = _capture_cell(cfg, cell, model, tok, state)
        if not usable:
            demotions.append(
                {
                    "cell_id": cell.cell_id,
                    "reason": (
                        f"parent bundle unusable as parity reference: keys={sorted(keys)} "
                        f"layers={blob_layers} != {list(cfg.layers)} — gate demoted to "
                        "WARN-only for this cell (documented ladder, plan §3.5)"
                    ),
                }
            )
            continue
        new_mean = cap["span_mean"].float().mean(dim=0)  # (L, H) over kept draws
        ref_mean = blob["v_a_mean"].float()
        assert ref_mean.shape == new_mean.shape, (ref_mean.shape, new_mean.shape)
        cos = torch.nn.functional.cosine_similarity(new_mean, ref_mean, dim=-1)  # (L,)
        per_cell.append(
            {
                "cell_id": cell.cell_id,
                "pair_id": cell.pair_id,
                "arm": cell.arm,
                "per_layer_cos": [float(c) for c in cos],
                "min_cos": float(cos.min()),
            }
        )
        if first_n == 0:
            first_n = int(cap["profiles"].shape[0])
            dt = time.monotonic() - t0
            logger.info(
                "[phase=p1_parity] first-chunk batched timing: %.2f s / %d samples "
                "= %.3f s/sample (batch=%d — informational, plan §9)",
                dt,
                first_n,
                dt / max(first_n, 1),
                cfg.capture_batch,
            )
    verdict = parity_verdict(per_cell, demotions)
    verdict["repro"] = _repro(cfg)
    _write_json_atomic(cfg.out_root / "parity_gate_report.json", verdict)
    for r in per_cell:
        if r["min_cos"] < PARITY_WARN_COS:
            logger.warning(
                "[phase=p1_parity] %s min_cos=%.6f below WARN bar %.4f",
                r["cell_id"],
                r["min_cos"],
                PARITY_WARN_COS,
            )
    if verdict["fired"]:
        logger.error(
            "[phase=p1_parity] PARITY HALT: %d/%d cells below %.3f — report at %s",
            verdict["n_bad"],
            verdict["n_cells"],
            PARITY_HALT_COS,
            cfg.out_root / "parity_gate_report.json",
        )
        sys.exit(RC_PARITY_HALT)
    logger.info(
        "[phase=p1_parity] PASS (n_bad=%d n_warn=%d demoted=%d)",
        verdict["n_bad"],
        verdict["n_warn"],
        len(demotions),
    )


# ── p2: binned capture ────────────────────────────────────────────────


def _regime(cfg: ProfileConfig, revisions: dict[str, str]) -> dict:
    """Every output-affecting knob (resume must not cross regimes, #722 r3)."""
    return {
        "bins_version": BINS_VERSION,
        "bin_names": list(BIN_NAMES),
        "layers": list(cfg.layers),
        "headline_layers": list(cfg.headline_layers),
        "model_id": cfg.model_id,
        "n_draws": cfg.n_draws,
        "revisions": revisions,
        "tiny": cfg.tiny,
        "store_dtype": "float16",
    }


def _tensor_path(cfg: ProfileConfig, cell_id: str) -> Path:
    return cfg.tensors_root / f"{cell_id}.pt"


def _capture_cell(cfg: ProfileConfig, cell: ProfileCell, model, tok, state: Manifest) -> dict:
    """Capture (or load, when resumed) ONE cell's binned profiles; persists
    the fp16 store + marks the manifest the moment the cell completes."""
    mark = f"capture/{cell.cell_id}"
    out_path = _tensor_path(cfg, cell.cell_id)
    if state.done(mark) and out_path.exists():
        return torch.load(out_path, map_location="cpu", weights_only=True)
    blob = load_staged_draws(cfg, cell.cell_id)
    draws = blob["draws"]
    kept_indices = [
        i for i, t in enumerate(draws) if len(tok(t, add_special_tokens=False)["input_ids"]) > 0
    ]
    # Per-cell decode-round-trip mismatch telemetry (plan §8 risk 1): a
    # tokenize->decode drift is shared by every condition AND the parent
    # reference (same rig), so this is persisted diagnostics, not a gate.
    n_roundtrip_mismatch = sum(
        1 for t in draws if t and tok.decode(tok(t, add_special_tokens=False)["input_ids"]) != t
    )
    cap = capture_binned_answer_profiles(
        model,
        tok,
        blob["context"],
        draws,
        list(cfg.layers),
        batch_size=cfg.capture_batch,
    )
    assert cap["profiles"].shape[0] == len(kept_indices), (
        cap["profiles"].shape,
        len(kept_indices),
    )
    record = {
        "cell_id": cell.cell_id,
        "family": cell.family,
        "role": cell.role,
        "pair_id": cell.pair_id,
        "arm": cell.arm,
        "steer_layer": cell.steer_layer,
        "layers": list(cfg.layers),
        "bin_names": list(BIN_NAMES),
        "profiles": cap["profiles"].to(torch.float16),  # (n_kept, 13, L, H)
        "span_mean": cap["span_mean"].to(torch.float16),  # (n_kept, L, H)
        "comp_token_counts": cap["comp_token_counts"],
        "kept_indices": kept_indices,
        "n_empty_completions": cap["n_empty_completions"],
        "n_roundtrip_mismatch": n_roundtrip_mismatch,
        "repro": _repro(cfg),
    }
    _save_pt_atomic(out_path, record)
    state.mark(mark, {"family": cell.family})
    # Return the fp32 capture for in-phase consumers (the parity gate).
    return {**record, "profiles": cap["profiles"], "span_mean": cap["span_mean"]}


def phase_p2(cfg: ProfileConfig, cells: list[ProfileCell], model, tok, state: Manifest) -> None:
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    # Plan §9 preamble assert (fail-loud statvfs + posix_fallocate canary —
    # EDQUOT-aware) BEFORE any capture writes, at BOTH p2 write roots (the
    # tensor store is the heavy write; out_root takes the manifest/JSONs —
    # same disk on every lane, so the duplicate probe costs ~nothing).
    assert_out_root_headroom(cfg.tensors_root, need_gb=10, phase="p2_capture")
    assert_out_root_headroom(cfg.out_root, need_gb=10, phase="p2_capture")
    families: dict[str, list[ProfileCell]] = {}
    for c in cells:
        families.setdefault(c.cell_id.split("/", 1)[0], []).append(c)
    n_done = 0
    for fam_dir, fam_cells in sorted(families.items()):
        for cell in fam_cells:
            _capture_cell(cfg, cell, model, tok, state)
            n_done += 1
            if n_done % 25 == 0:
                logger.info("[phase=p2_capture] %d/%d cells captured", n_done, len(cells))
        # Incremental per-family upload: ONE folder commit per family (never a
        # per-file loop — 256-commits/hr + per-file-504 gotchas).
        upload_artifact(cfg, cfg.tensors_root / fam_dir, f"{PROFILE_TENSOR_PREFIX}/{fam_dir}")
    logger.info("[phase=p2_capture] complete: %d cells", len(cells))


# ── p3: statistics (§3.4) ─────────────────────────────────────────────


def _load_reduced(cfg: ProfileConfig, cell_id: str) -> dict:
    """Load ONE cell store and reduce over draws: full / even-half / odd-half
    nan-means (13, L, H) fp32 + per-bin contributing counts + lengths.
    Halves split by ORIGINAL draw index parity (kept_indices; parent recount
    convention)."""
    rec = torch.load(_tensor_path(cfg, cell_id), map_location="cpu", weights_only=True)
    prof = rec["profiles"].float()  # (n_kept, 13, L, H); NaN rows for empty bins
    kept = list(rec["kept_indices"])
    idx = torch.tensor(kept)
    even = prof[(idx % 2 == 0).nonzero().squeeze(-1)]
    odd = prof[(idx % 2 == 1).nonzero().squeeze(-1)]
    contrib = (~prof.isnan().any(dim=-1)).sum(dim=0)  # (13, L) draws with the bin defined
    return {
        "full": prof.nanmean(dim=0),  # (13, L, H)
        "even": even.nanmean(dim=0) if even.shape[0] else torch.full_like(prof[0], torch.nan),
        "odd": odd.nanmean(dim=0) if odd.shape[0] else torch.full_like(prof[0], torch.nan),
        "contrib": contrib,
        "n_kept": prof.shape[0],
        "n_even": int(even.shape[0]),
        "n_odd": int(odd.shape[0]),
        "comp_token_counts": list(rec["comp_token_counts"]),
        "n_empty": int(rec["n_empty_completions"]),
    }


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def _bin_stats(steered: dict, base: dict, ceil_full: torch.Tensor | None, li: int) -> list[dict]:
    """Per-bin §3.4 statistics at layer index ``li`` for one steered cell.

    Primary (shared-baseline structure): disjoint halves — target from half A,
    shift from half B, both assignments averaged. ``ceil_full`` None = rep
    round (shift = rep_steered - rep_base full draws; the caller passes the
    PARENT target separately via the returned raw vectors)."""
    out = []
    for b, name in enumerate(BIN_NAMES):
        s_full = steered["full"][b, li]
        base_a, base_b, base_full = base["even"][b, li], base["odd"][b, li], base["full"][b, li]
        row: dict = {"bin": name}
        if ceil_full is not None:
            c_full = ceil_full[b, li]
            aligns, mags, tmags = [], [], []
            for t_half, s_half in ((base_a, base_b), (base_b, base_a)):
                target = c_full - t_half
                shift = s_full - s_half
                aligns.append(_cos(shift, target))
                mags.append(float(shift.norm()))
                tmags.append(float(target.norm()))
            row["alignment_disjoint"] = float(np.mean(aligns))
            row["magnitude"] = float(np.mean(mags))
            row["target_magnitude"] = float(np.mean(tmags))
            shared_t = c_full - base_full
            shared_s = s_full - base_full
            row["alignment_shared"] = _cos(shared_s, shared_t)
            row["magnitude_shared"] = float(shared_s.norm())
            tm = row["target_magnitude"]
            row["traversal_frac"] = row["magnitude"] / tm if tm and math.isfinite(tm) else None
        else:
            shift = s_full - base_full
            row["magnitude"] = float(shift.norm())
        row["noise_floor"] = float((base_a - base_b).norm())
        row["n_contrib_steered"] = int(steered["contrib"][b, li])
        row["n_contrib_baseline"] = int(base["contrib"][b, li])
        # NaN -> None for JSON (empty bins render as gaps, never zeros)
        for k, v in list(row.items()):
            if isinstance(v, float) and not math.isfinite(v):
                row[k] = None
        out.append(row)
    return out


def _delta_stats(
    per_pair_mag: dict[str, dict[str, float | None]],
    early: tuple[str, ...],
    late: tuple[str, ...],
    n_boot: int,
    cell_label: str,
    max_drop_frac: float = MAX_DROP_FRAC,
) -> dict:
    """EARLY-vs-LATE paired log-magnitude contrast Delta over pairs (§3.4).

    ``per_pair_mag``: pair_id -> {bin: magnitude|None}. A pair with an
    all-None EARLY or LATE set is DROPPED (named); > ``max_drop_frac`` of
    pairs dropping fails LOUD naming the cell + pairs (plan §8)."""
    deltas: dict[str, float] = {}
    dropped: list[str] = []
    for pid, mags in sorted(per_pair_mag.items()):
        e_vals = [mags[b] for b in early if mags.get(b) is not None]
        l_vals = [mags[b] for b in late if mags.get(b) is not None]
        if not e_vals or not l_vals:
            dropped.append(pid)
            continue
        e_mean, l_mean = float(np.mean(e_vals)), float(np.mean(l_vals))
        if e_mean <= 0.0 or l_mean <= 0.0:
            # A zero mean magnitude (measure-zero on real data; possible on
            # degenerate shared-prefix inputs) makes the log-ratio non-finite:
            # NAMED drop, never a -inf leaking into the JSON.
            dropped.append(pid)
            continue
        deltas[pid] = float(np.log(e_mean) - np.log(l_mean))
    n_total = len(per_pair_mag)
    if n_total and len(dropped) / n_total > max_drop_frac:
        raise RuntimeError(
            f"Delta[{cell_label}]: {len(dropped)}/{n_total} pairs dropped "
            f"(> {max_drop_frac:.0%}) — all-NaN EARLY/LATE sets for: {dropped}"
        )
    arr = np.array(list(deltas.values()), dtype=np.float64)
    result: dict = {
        "cell": cell_label,
        "early_bins": list(early),
        "late_bins": list(late),
        "n_pairs_kept": int(arr.size),
        "dropped_pairs": dropped,  # NAMED, not only counted (plan §3.4 (iii))
        "per_pair_delta": {k: float(v) for k, v in deltas.items()},
    }
    if arr.size == 0:
        result.update({"delta_mean": None, "ci95": None, "wilcoxon_p": None})
        return result
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))  # ONE vectorized resample
    boot_means = arr[idx].mean(axis=1)
    result["delta_mean"] = float(arr.mean())
    result["ci95"] = [float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))]
    try:
        from scipy.stats import wilcoxon

        result["wilcoxon_p"] = float(wilcoxon(arr).pvalue) if arr.size >= 2 else None
        if arr.size < 2:
            result["wilcoxon_note"] = "n<2 — companion undefined at this n"
    except ValueError as exc:  # degenerate input (e.g. all-zero diffs) — recorded
        result["wilcoxon_p"] = None
        result["wilcoxon_note"] = f"wilcoxon undefined: {exc}"
    return result


def _null_bands(targets: dict[str, torch.Tensor], hidden: int, n_null: int) -> dict[str, float]:
    """Per-target p97.5 of cos(random unit direction, target) — ONE batched
    GEMM over all targets (seed 14150; vectorize rule). ``targets`` values are
    the PRE-NORMALIZED direction each alignment read compares against (for the
    disjoint convention: the mean of the two half-assignment unit targets)."""
    if not targets:
        return {}
    rng = np.random.default_rng(NULL_SEED)
    R = rng.standard_normal((n_null, hidden))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    keys = list(targets)
    T = torch.stack([targets[k] for k in keys]).numpy().T  # (H, n_targets)
    cos = R @ T  # (n_null, n_targets)
    q = np.quantile(cos, 0.975, axis=0)
    return {k: float(v) for k, v in zip(keys, q, strict=True)}


def _unit(v: torch.Tensor) -> torch.Tensor | None:
    n = float(v.norm())
    if not math.isfinite(n) or n == 0.0:
        return None
    return v / n


def phase_p3(cfg: ProfileConfig, cells: list[ProfileCell], revisions: dict[str, str]) -> None:
    """CPU statistics over the per-cell stores (streamed per pair)."""
    pairs = sorted({c.pair_id for c in cells})
    layer_index = {layer: list(cfg.layers).index(layer) for layer in cfg.layers}

    profiles: list[dict] = []
    lengths: list[dict] = []
    null_targets: dict[str, torch.Tensor] = {}
    # magnitudes for Delta: cell_label -> pair -> {bin: magnitude|None}
    mag_by_cell: dict[str, dict[str, dict[str, float | None]]] = {}
    align_by_cell: dict[str, dict[str, dict[str, float | None]]] = {}
    floor_by_cell: dict[str, dict[str, dict[str, float | None]]] = {}
    median_len: dict[tuple[str, str], float] = {}  # (kind, pair) -> median tokens

    by_id = {c.cell_id: c for c in cells}
    for pid in pairs:
        reduced: dict[str, dict] = {}

        def _get(cid: str) -> dict:
            if cid not in reduced:
                assert cid in by_id, f"registered cell missing from inventory: {cid}"
                reduced[cid] = _load_reduced(cfg, cid)
            return reduced[cid]

        base = _get(f"gen1b/{pid}/c")
        ceil = _get(f"gen1b/{pid}/cprime")
        median_len[("baseline", pid)] = float(np.median(base["comp_token_counts"]))
        for kind, red in (("baseline", base), ("ceiling", ceil)):
            lengths.append(
                {
                    "condition": kind,
                    "pair_id": pid,
                    "arm": None,
                    "layer": None,
                    "token_counts": red["comp_token_counts"],
                    "n_empty": red["n_empty"],
                }
            )
        # ── primary round (gen1c vs gen1b, disjoint halves REQUIRED) ──
        for arm in ("prefix", "context"):
            for layer in cfg.headline_layers:
                cid = f"gen1c/{arm}/{pid}/L{layer}/a{_fmt(ALPHA)}"
                st = _get(cid)
                li = layer_index[layer]
                rows = _bin_stats(st, base, ceil["full"], li)
                cell_label = f"primary/{arm}/L{layer}"
                mag_by_cell.setdefault(cell_label, {})[pid] = {
                    r["bin"]: r["magnitude"] for r in rows
                }
                align_by_cell.setdefault(cell_label, {})[pid] = {
                    r["bin"]: r["alignment_disjoint"] for r in rows
                }
                floor_by_cell.setdefault(cell_label, {})[pid] = {
                    r["bin"]: r["noise_floor"] for r in rows
                }
                median_len[(cell_label, pid)] = float(np.median(st["comp_token_counts"]))
                # Null targets: mean of the two half-assignment UNIT targets
                # (mirrors the averaged-assignment disjoint alignment read).
                for b, name in enumerate(BIN_NAMES):
                    u1 = _unit(ceil["full"][b, li] - base["even"][b, li])
                    u2 = _unit(ceil["full"][b, li] - base["odd"][b, li])
                    if u1 is not None and u2 is not None:
                        null_targets[f"{pid}/L{layer}/{name}/disjoint"] = (u1 + u2) / 2
                profiles.append(
                    {
                        "round": "primary",
                        "pair_id": pid,
                        "pair_type": pair_type(pid),
                        "arm": arm,
                        "steer_layer": layer,
                        "read_layer": layer,
                        "flags": {
                            "terse": pid == TERSE_PAIR,
                            "formal": pid == FORMAL_PAIR,
                            "medical_noise_target": pid == MEDICAL_PAIR,
                        },
                        "n_kept_draws": st["n_kept"],
                        "bins": rows,
                    }
                )
                lengths.append(
                    {
                        "condition": "steered",
                        "pair_id": pid,
                        "arm": arm,
                        "layer": layer,
                        "token_counts": st["comp_token_counts"],
                        "n_empty": st["n_empty"],
                    }
                )
        # ── replication rounds (rep43/44; disjoint by construction) ──
        for rep in REP_LABELS:
            rb = _get(f"gen_rep{rep}/base/{pid}/c")
            li = layer_index[cfg.rep_layer]
            for arm in ("prefix", "context"):
                cid = f"gen_rep{rep}/{arm}/{pid}/L{cfg.rep_layer}/a{_fmt(ALPHA)}"
                st = _get(cid)
                rows = []
                for b, name in enumerate(BIN_NAMES):
                    shift = st["full"][b, li] - rb["full"][b, li]
                    target = ceil["full"][b, li] - base["full"][b, li]  # PARENT full-10 target
                    row = {
                        "bin": name,
                        "magnitude": float(shift.norm()),
                        "alignment_parent_target": _cos(shift, target),
                        "target_magnitude": float(target.norm()),
                        "noise_floor": float((rb["even"][b, li] - rb["odd"][b, li]).norm()),
                        "n_contrib_steered": int(st["contrib"][b, li]),
                        "n_contrib_baseline": int(rb["contrib"][b, li]),
                    }
                    for k, v in list(row.items()):
                        if isinstance(v, float) and not math.isfinite(v):
                            row[k] = None
                    rows.append(row)
                    u = _unit(ceil["full"][b, li] - base["full"][b, li])
                    if u is not None:
                        null_targets[f"{pid}/L{cfg.rep_layer}/{name}/full"] = u
                cell_label = f"rep{rep}/{arm}/L{cfg.rep_layer}"
                mag_by_cell.setdefault(cell_label, {})[pid] = {
                    r["bin"]: r["magnitude"] for r in rows
                }
                lengths.append(
                    {
                        "condition": f"rep{rep}_steered",
                        "pair_id": pid,
                        "arm": arm,
                        "layer": cfg.rep_layer,
                        "token_counts": st["comp_token_counts"],
                        "n_empty": st["n_empty"],
                    }
                )
                profiles.append(
                    {
                        "round": f"rep{rep}",
                        "pair_id": pid,
                        "pair_type": pair_type(pid),
                        "arm": arm,
                        "steer_layer": cfg.rep_layer,
                        "read_layer": cfg.rep_layer,
                        "flags": {
                            "terse": pid == TERSE_PAIR,
                            "formal": pid == FORMAL_PAIR,
                            "medical_noise_target": pid == MEDICAL_PAIR,
                        },
                        "n_kept_draws": st["n_kept"],
                        "bins": rows,
                    }
                )

    # ── row-coverage set-check BEFORE the Delta statistic (plan §7) ──
    registered = {
        f"primary/{arm}/L{layer}": set(pairs)
        for arm in ("prefix", "context")
        for layer in cfg.headline_layers
    }
    for cell_label, want in registered.items():
        got = set(mag_by_cell.get(cell_label, {}))
        if got != want:
            raise RuntimeError(
                f"row-coverage set-check failed for {cell_label}: missing={sorted(want - got)} "
                f"unexpected={sorted(got - want)}"
            )

    # ── null bands (ONE batched GEMM) ──
    bands = _null_bands(null_targets, cfg.hidden, cfg.n_null)
    _write_json_atomic(
        cfg.out_root / "null_bands_binned.json",
        {
            "seed": NULL_SEED,
            "n_null": cfg.n_null,
            "quantile": 0.975,
            "convention": {
                "disjoint": "mean of the two half-assignment unit targets (primary read)",
                "full": "full-10-draw parent target (replication read)",
            },
            "p975": bands,
            "repro": _repro(cfg),
        },
    )
    # thread the band into the per-pair profile rows (primary disjoint reads)
    for prof in profiles:
        key_conv = "disjoint" if prof["round"] == "primary" else "full"
        for row in prof["bins"]:
            row["null_p975"] = bands.get(
                f"{prof['pair_id']}/L{prof['read_layer']}/{row['bin']}/{key_conv}"
            )

    _write_json_atomic(
        cfg.out_root / "per_pair_profiles.json",
        {
            "bin_names": list(BIN_NAMES),
            "alpha": ALPHA,
            "conventions": {
                "primary": "disjoint even/odd baseline halves, both assignments averaged "
                "(PRIMARY); shared all-10-draw convention reported as labeled SECONDARY",
                "replication": "shift = rep_steered - rep_baseline (all rep draws); target = "
                "the PARENT full-10-draw gen1b target (disjoint rounds by construction)",
                "noise_floor": "||P_base^even - P_base^odd|| (5-vs-5 pure-noise magnitude; "
                "order-of-magnitude floor — draw arithmetic differs from the shift's 10-vs-5)",
            },
            "profiles": profiles,
            "repro": _repro(cfg),
        },
    )
    _write_json_atomic(
        cfg.out_root / "answer_length_distributions.json",
        {
            "callouts": {"terse": TERSE_PAIR, "formal": FORMAL_PAIR},
            "distributions": lengths,
            "repro": _repro(cfg),
        },
    )

    # ── Delta lattice (per pre-registered cell; NEVER pooled or maxed) ──
    def _matched_length_pairs(cell_label: str) -> list[str]:
        kept = []
        for pid in pairs:
            s = median_len.get((cell_label, pid))
            b = median_len.get(("baseline", pid))
            if s and b and MATCHED_LENGTH_RATIO[0] <= s / b <= MATCHED_LENGTH_RATIO[1]:
                kept.append(pid)
        return kept

    delta_cells = []
    for cell_label in sorted(mag_by_cell):
        mags = mag_by_cell[cell_label]
        primary = cell_label.startswith("primary/")
        entry = _delta_stats(mags, EARLY_BINS, LATE_BINS, cfg.n_boot, cell_label)
        entry["registered_primary"] = primary
        entry["delta_width"] = _delta_stats(
            mags, WIDTH_EARLY_BINS, LATE_BINS, cfg.n_boot, f"{cell_label}/width"
        )
        if primary:
            floors = floor_by_cell[cell_label]
            entry["delta_floor"] = _delta_stats(
                floors, EARLY_BINS, LATE_BINS, cfg.n_boot, f"{cell_label}/floor"
            )
            aligns = align_by_cell[cell_label]
            dcos = {}
            for pid, a in aligns.items():
                e = [a[b] for b in EARLY_BINS if a.get(b) is not None]
                l_ = [a[b] for b in LATE_BINS if a.get(b) is not None]
                if e and l_:
                    dcos[pid] = float(np.mean(e) - np.mean(l_))
            entry["delta_cos_mean"] = float(np.mean(list(dcos.values()))) if dcos else None
            entry["delta_cos_per_pair"] = dcos
            # sensitivity reads (labeled; Delta never consumes the target, so
            # the medical pair's target-reliability defect is immaterial to
            # the primary Delta — the exclusion is a SENSITIVITY read, §4)
            if MEDICAL_PAIR in mags:
                entry["sensitivity_exclude_medical"] = _delta_stats(
                    {k: v for k, v in mags.items() if k != MEDICAL_PAIR},
                    EARLY_BINS,
                    LATE_BINS,
                    cfg.n_boot,
                    f"{cell_label}/exclude_medical",
                )
            ml_pairs = _matched_length_pairs(cell_label)
            entry["sensitivity_matched_length"] = {
                "kept_pairs": ml_pairs,
                "ratio_window": list(MATCHED_LENGTH_RATIO),
                "stats": _delta_stats(
                    {k: v for k, v in mags.items() if k in ml_pairs},
                    EARLY_BINS,
                    LATE_BINS,
                    cfg.n_boot,
                    f"{cell_label}/matched_length",
                )
                if ml_pairs
                else None,
            }
            entry["terse_pair_delta"] = entry["per_pair_delta"].get(TERSE_PAIR)
        delta_cells.append(entry)

    _write_json_atomic(
        cfg.out_root / "summary.json",
        {
            "registered_statistic": (
                "Delta = log(mean EARLY-bin ||shift||) - log(mean LATE-bin ||shift||), "
                "per pair; mean over pairs per (arm x steer layer) cell with pair-bootstrap "
                "95% percentile CI (B, seed 14151) + Wilcoxon companion; 4 pre-registered "
                "primary cells reported side by side — never pooled or maxed"
            ),
            "early_bins": list(EARLY_BINS),
            "late_bins": list(LATE_BINS),
            "width_early_bins": list(WIDTH_EARLY_BINS),
            "n_boot": cfg.n_boot,
            "boot_seed": BOOT_SEED,
            "max_drop_frac": MAX_DROP_FRAC,
            "cells": delta_cells,
            "revisions": revisions,
            "repro": _repro(cfg),
        },
    )
    logger.info("[phase=p3_profiles] wrote 4 JSONs under %s", cfg.out_root)


# ── p4: upload residue ────────────────────────────────────────────────


def phase_p4(cfg: ProfileConfig, cells: list[ProfileCell], revisions: dict[str, str]) -> None:
    """Store manifest + batched residue upload (ONE folder commit; the git
    commit of the eval_results JSONs is the dispatcher's commit_push_verify)."""
    stored = sorted(str(p.relative_to(cfg.tensors_root)) for p in cfg.tensors_root.rglob("*.pt"))
    _write_json_atomic(
        cfg.tensors_root / "manifest.json",
        {
            "n_files": len(stored),
            "files": stored,
            "bins_version": BINS_VERSION,
            "bin_names": list(BIN_NAMES),
            "layers": list(cfg.layers),
            "revisions": revisions,
            "n_cells_registered": len(cells),
            "repro": _repro(cfg),
        },
    )
    upload_artifact(cfg, cfg.tensors_root, PROFILE_TENSOR_PREFIX)
    logger.info("[phase=p4_upload] uploaded %d store files + manifest", len(stored))


# ── main ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    cfg = build_config(args)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    cfg.tensors_root.mkdir(parents=True, exist_ok=True)
    cfg.stage_root.mkdir(parents=True, exist_ok=True)
    if not cfg.tiny and cfg.upload_mode == "hf":
        assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing — required for --upload hf"
    if cfg.tiny and "p0" in cfg.phases:
        build_tiny_fixture(cfg)  # idempotent (sentinel); metas must exist to enumerate
    cells = enumerate_cells(cfg)
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
                cfg.out_root / "profile_manifest.json", _regime(cfg, _revs())
            )
        return state

    def _model():
        nonlocal model, tok
        if model is None:
            model, tok = load_model_and_tokenizer(cfg)
        return model, tok

    for phase in cfg.phases:
        if phase == "p0":
            phase_p0(cfg, cells)
        elif phase == "p1":
            m, t = _model()
            phase_p1(cfg, cells, m, t, _state())
        elif phase == "p2":
            m, t = _model()
            phase_p2(cfg, cells, m, t, _state())
        elif phase == "p3":
            phase_p3(cfg, cells, _revs())
        elif phase == "p4":
            phase_p4(cfg, cells, _revs())
    logger.info("[phase=profile_driver_complete] phases=%s", ",".join(cfg.phases))


if __name__ == "__main__":
    main()
