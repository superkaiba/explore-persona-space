#!/usr/bin/env python3
"""Issue #1946 — per-context SAE-space error taxonomy (plan v4).

Rebuilds the true SAE-space targets Y from #1738's banked SAE capture, scores
per-context nerr for the 12 banked pred16 cells (3 arms x 4 reading spaces),
runs the VERBATIM #1738 22-contrast battery (`issue1738_characterize.py
--phase taxonomy` as a checked subprocess) per space, and computes the
pre-registered cross-space agreement statistics (rho_M + two-sided permutation
p at seed 1946, SA_M sign agreement over the dense BH-significant union) with
the mechanical 4-way verdict lattice (plan section 3).

Phases (``--phase``): ``stage | build | floors | taxonomy | compare | figures
| upload | all``; ``--smoke`` drives the FULL chain through the SAME
entrypoints on the tiny-real fixture produced by ``issue1738_sae_arm.py
--smoke`` (PASS_UNIFIED; Hub boundary faked signature-conformantly).

Identity gates (plan section 7) run FIRST in ``build``, before the heavy Y
rebuild: assembly-fingerprint string equality per pred16 cell, banked-R^2
reproduction (|dR^2| < 5e-3), banked-``feat_ids``-authoritative f_out gated by
machine-independent set-validity invariants vs the recomputed counts (NOT
recompute equality — argsort cap-boundary tie order is CPU-SIMD-dependent),
ci alignment, split-sha cross-asserts, and scan-count identity — all compared
against the banked ``sae_fits.json`` FIELDS (under ``--smoke`` those fields
are the fixture's own, never production constants — smoke-scale gate
calibration, gotchas #1345).

Refusal-safety: no conversation/label text is ever printed — digest-only
(counts, paths, hashes).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch import (#847)

import issue779_common as C  # noqa: E402
import issue1482_sae as SAEMOD  # noqa: E402
import issue1482_shuffle_null as SN  # noqa: E402
import issue1738_characterize as CH  # noqa: E402
import issue1738_multiturn_fits as MTF  # noqa: E402
import issue1738_sae_arm as SA  # noqa: E402
import numpy as np  # noqa: E402
import scipy.stats  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1946_saepc")

# ── constants (plan sections 3/9/11 — Sources recorded there) ──────────────────────
LAYER = 19  # parent headline layer (Source: #1738)
DATA_REVISION_DEFAULT = "05cb982b0d3f9a21b5735d196a0afdc8175590e5"  # plan section 11 item 11
PARENT_PREFIX = "issue1738_multiturn"
UPLOAD_PREFIX_DEFAULT = "issue1946_sae_percontext"
CROSS_SEED = 1946  # plan section 11 item 8
N_PERM_CROSS = 10_000
R2_TOL = 5e-3  # fp16 identity tolerance (plan section 11 item 6 — derived, never loosened)
RSS_CAP_GB_DEFAULT = 16.0  # plan kill criterion 2 -> cpu-bigmem fallback (orchestrator-owned)
SA_M_REPRODUCE_MIN = 15  # of the 20-contrast dense-significant union (plan section 3)
SA_M_INVERT_MAX = 5
N_UNION_EXPECTED = 20  # measured from banked BH flags (plan section 2; production-only assert)
RC_RSS_HALT = 21  # designed artifact-routed halt (rss_halt.json written first)

# Cell -> arm-name mapping (plan section 4 table): characterize accepts only
# FT.ARMS = {prefix, context, bare}; each SPACE gets its own out-eval dir.
SPACES: dict[str, dict[str, str]] = {
    "sae_space": {"prefix": "sae_prefix", "context": "sae_context", "bare": "sae_bare"},
    "dense_feat_space": {
        "prefix": "dense_px_feat",
        "context": "dense_cx_feat",
        "bare": "dense_bq_feat",
    },
    "max_space": {
        "prefix": "sae_prefix_max",
        "context": "sae_context_max",
        "bare": "sae_bare_max",
    },
    "frac_space": {
        "prefix": "sae_prefix_frac",
        "context": "sae_context_frac",
        "bare": "sae_bare_frac",
    },
}
POOL_OF_SPACE = {
    "sae_space": "mean",
    "dense_feat_space": "mean",
    "max_space": "max",
    "frac_space": "frac",
}
# floors env per space (plan section 4 P3): mean-pool spaces get the approximate floor
# (unadjusted + floor-adjusted); max/frac get labels-only (unadjusted only).
PARENT_ENV_OF_SPACE = {
    "sae_space": "floors_env_mean",
    "dense_feat_space": "floors_env_mean",
    "max_space": "floors_env_labels_only",
    "frac_space": "floors_env_labels_only",
}
# y_holdout primary dir per pooling; dense_feat_space consumes sae_space's mean-pool
# file (same tensor by construction — stated deviation from the plan's per-space
# y-holdout path template, avoids a duplicate 326 MB artifact).
Y_HOLDOUT_SPACE = {"mean": "sae_space", "max": "max_space", "frac": "frac_space"}
ARMS = ("prefix", "context", "bare")

# The six body-named mirror categories (plan section 3, descriptive table).
MIRROR_CATEGORIES = {
    "prefix": [
        "language=en",
        "topic=chitchat_social",
        "topic=translation",
        "corpus=wildchat",
        "topic=nsfw",
        "topic=creative_writing",
    ],
    "bare": ["topic=roleplay_persona", "depth=>=5", "topic=chitchat_social"],
}


@dataclass
class Cfg:
    """All paths + mode flags for one run (production defaults or smoke fixtures)."""

    staging_root: Path
    out_eval: Path
    fig_dir: Path
    dense_eval: Path
    revision: str
    upload_prefix: str
    smoke: bool = False
    no_upload: bool = False
    rss_cap_gb: float = RSS_CAP_GB_DEFAULT
    force: bool = False  # disables ALL resume-skips (build + phase entry guards)
    # inputs (production: derived from staging_root; smoke: SA fixture paths)
    local_sae_dir: Path | None = None
    pred16_src: Path | None = None
    sae_fits_path: Path | None = None
    perfeature_npz: Path | None = None
    split_file: Path | None = None
    manifest_dir: Path | None = None
    kresample_dir: Path | None = None
    sae_cache: Path | None = None
    smoke_model_dir: Path | None = None

    def space_dir(self, space: str) -> Path:
        return self.out_eval / space

    def y_holdout_dir(self, space: str) -> Path:
        return self.space_dir(Y_HOLDOUT_SPACE[POOL_OF_SPACE[space]]) / "y_holdout"

    def sentinel_dir(self) -> Path:
        return self.out_eval / "phase_sentinels"


def _resolve_production_inputs(cfg: Cfg) -> Cfg:
    """Fill input paths from the staging root + git-resident banked artifacts."""
    root = cfg.staging_root
    cfg.local_sae_dir = root / PARENT_PREFIX / "sae_arm" / "capture"
    cfg.pred16_src = root / PARENT_PREFIX / "sae_arm_bare" / "analysis_tensors" / "pred16"
    cfg.perfeature_npz = (
        root
        / PARENT_PREFIX
        / "sae_arm_bare"
        / "analysis_tensors"
        / "perfeature"
        / "perfeature_summary.npz"
    )
    cfg.manifest_dir = root / PARENT_PREFIX / "sampling_manifest"
    cfg.split_file = cfg.manifest_dir / "split_1738.json"
    cfg.kresample_dir = root / PARENT_PREFIX / "kresample"
    cfg.sae_cache = root / "sae_cache"
    cfg.sae_fits_path = cfg.dense_eval / "bare_query" / "sae_arm" / "sae_fits.json"
    return cfg


def _banked(cfg: Cfg) -> dict:
    assert cfg.sae_fits_path is not None and cfg.sae_fits_path.is_file(), cfg.sae_fits_path
    return json.loads(cfg.sae_fits_path.read_text())


def _atomic_json(path: Path, obj: dict) -> None:
    """Atomic JSON write (tmp + replace, same dir — EXDEV-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _repro_meta(cfg: Cfg) -> dict:
    return {
        "git_commit": MTF._git_head(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_revision": cfg.revision,
        "smoke": bool(cfg.smoke),
    }


# ── P0: stage (idempotent, revision-pinned; Hub boundary injectable) ────────────────


def _assert_disk_headroom(cfg: Cfg) -> None:
    """Preamble df asserts (plan section 9 disk table): staging fs >= 24 GB free,
    out-eval fs >= 5 GB free. Smoke keeps the SAME check at tiny floors (the
    scratch fixture needs ~0; the check code itself must run)."""
    need_stage_gb = 24.0 if not cfg.smoke else 1.0
    need_out_gb = 5.0 if not cfg.smoke else 0.5
    for path, need in ((cfg.staging_root, need_stage_gb), (cfg.out_eval, need_out_gb)):
        probe = path
        while not probe.exists():
            probe = probe.parent
        st = os.statvfs(probe)
        free_gb = st.f_bavail * st.f_frsize / 2**30
        assert free_gb >= need, f"disk headroom: {probe} has {free_gb:.1f} GB free < {need} GB"
        logger.info("[stage] headroom OK: %s free=%.1f GB (need %.1f)", probe, free_gb, need)


def phase_stage(
    cfg: Cfg,
    *,
    stage_prefix_fn=hub.stage_hub_prefix,
    stage_file_fn=hub.stage_hub_file,
    ensure_sae_fn=SAEMOD.BatchTopKSAE.ensure_downloaded,
) -> None:
    """Stage every pinned input into the staging root (mirror-root semantics of
    ``stage_hub_prefix`` — files land at ``root/<repo path>``, gotchas #1774) and
    write per-space ``mapping.json``. Hub fns are injectable so the smoke fakes
    the boundary signature-conformantly (autospec) while running this exact body."""
    print("[phase=stage] start", flush=True)
    _assert_disk_headroom(cfg)
    root = cfg.staging_root
    root.mkdir(parents=True, exist_ok=True)
    repo = C.HF_DATA_REPO
    for prefix in (
        f"{PARENT_PREFIX}/sae_arm/capture",
        f"{PARENT_PREFIX}/sae_arm_bare/analysis_tensors/pred16",
        f"{PARENT_PREFIX}/sampling_manifest",
    ):
        got = stage_prefix_fn(repo, prefix, root, repo_type="dataset", revision=cfg.revision)
        n = len(got) if isinstance(got, list) else -1
        print(f"[stage] prefix {prefix}: staged n={n}", flush=True)
    for fpath in (
        f"{PARENT_PREFIX}/sae_arm_bare/analysis_tensors/perfeature/perfeature_summary.npz",
        f"{PARENT_PREFIX}/kresample/kresample_shard00.pt",
    ):
        stage_file_fn(repo, fpath, root / fpath, repo_type="dataset", revision=cfg.revision)
        print(f"[stage] file {fpath}: staged", flush=True)
    sae_cache = cfg.sae_cache if cfg.sae_cache is not None else root / "sae_cache"
    ensure_sae_fn(64, sae_cache, layer=LAYER)
    print("[stage] sae weights ensured (k=64, layer 19)", flush=True)
    for space, mapping in SPACES.items():
        d = cfg.space_dir(space)
        d.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            d / "mapping.json",
            {
                "space": space,
                "pooling": POOL_OF_SPACE[space],
                "arm_to_cell": mapping,
                "parent_env": PARENT_ENV_OF_SPACE[space],
                "y_holdout_dir": str(cfg.y_holdout_dir(space)),
                "note": (
                    "characterize arm files named prefix/context/bare hold the mapped "
                    "cells above (plan section 4 mapping table)"
                ),
                **_repro_meta(cfg),
            },
        )
    _atomic_json(
        cfg.sentinel_dir() / "stage.done.json",
        {"revision": cfg.revision, **_repro_meta(cfg)},
    )
    print("[phase=stage] done", flush=True)


# ── P1: build (identity gates FIRST, then Y rebuild + per-context nerr) ─────────────


def _scan_and_gate(cfg: Cfg, banked: dict) -> tuple[dict, np.ndarray, dict, np.ndarray]:
    """Pass-1 scan + ALL cheap identity gates (plan section 7), before the heavy
    Y build. Returns (scan, f_out, f_in, holdout row positions)."""
    split = MTF.load_split(Path(cfg.split_file))
    for k in split["sets"]:
        got = split["sets"][k]["sha256"]
        want = banked["split_shas"][k]
        assert got == want, f"split sha cross-assert failed for set {k!r}: {got} != {want}"
    ns = argparse.Namespace(local_sae_dir=str(cfg.local_sae_dir))
    names = SA._sae_chunk_names(ns, None)
    paths = [Path(cfg.local_sae_dir) / n for n in names]
    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    dict_size = int(first["sae"]["dict_size"])
    del first
    train_ci = {int(c) for c in split["sets"]["train"]["ci"]}
    t0 = time.time()
    scan = SA._scan_sae(paths, train_ci, dict_size)
    print(f"[build] scan done: {len(paths)} chunks in {time.time() - t0:.0f}s", flush=True)
    # gate: scan counts identical to the banked run (plan section 7 row 1/2 inputs)
    assert len(scan["ci"]) == banked["n_rows"], (
        f"scan n_rows {len(scan['ci'])} != banked {banked['n_rows']}"
    )
    assert len(scan["dropped"]) == banked["n_dropped"], (
        f"scan dropped {len(scan['dropped'])} != banked n_dropped {banked['n_dropped']}"
    )
    # gate: banked feat_ids are AUTHORITATIVE for f_out — the banked pred16 predictions
    # were fit on exactly that feature set, so the rebuilt Y columns must be those
    # features. Validity vs the recomputed counts is checked as MACHINE-INDEPENDENT set
    # invariants, NOT recompute equality: SN.restrict() breaks cap-boundary count ties
    # via unstable np.argsort(-counts), whose tie order is CPU-SIMD-kernel dependent
    # (numpy 2.x x86-simd-sort: AVX-512 on GCE n2 vs none on the VM Xeon; crash-fix r3
    # measured 5 features tied at boundary count 7198 for the last 2 of 16384 slots).
    _, floor = SN.restrict(scan["out_fit"], scan["n_fit"], SA.MAX_FEATURES_OUT)
    assert floor == banked["restriction"]["activity_floor_rows"], (
        f"activity floor {floor} != banked {banked['restriction']['activity_floor_rows']}"
    )
    pf = np.load(Path(cfg.perfeature_npz))
    assert "feat_ids" in pf, sorted(pf.files)
    banked_fids = np.asarray(pf["feat_ids"])
    assert len(banked_fids) == banked["restriction"]["n_f_out"], (
        f"banked feat_ids n={len(banked_fids)} != sae_fits n_f_out "
        f"{banked['restriction']['n_f_out']} — perfeature npz / sae_fits.json mismatch"
    )
    assert banked_fids.ndim == 1 and bool((np.diff(banked_fids) > 0).all()), (
        "banked feat_ids not strictly increasing / unique — corrupt restriction artifact"
    )
    counts = scan["out_fit"]
    cap = SA.MAX_FEATURES_OUT
    n_eligible = int((counts >= floor).sum())
    assert len(banked_fids) == min(cap, n_eligible), (
        f"banked feat_ids n={len(banked_fids)} != min(cap={cap}, n_eligible={n_eligible}) "
        "— banked set size inconsistent with recomputed eligibility"
    )
    banked_counts = counts[banked_fids]
    n_below = int((banked_counts < floor).sum())
    assert n_below == 0, (
        f"{n_below} banked features below activity floor {floor} "
        f"(min banked count {int(banked_counts.min())})"
    )
    boundary = int(banked_counts.min())
    above = np.flatnonzero(counts > boundary)
    n_above = int(above.size)
    assert n_above < cap, (
        f"{n_above} features with count > boundary {boundary} >= cap {cap} — "
        "banked set cannot be a valid top-cap selection for these counts"
    )
    missing_above = np.setdiff1d(above, banked_fids)
    assert missing_above.size == 0, (
        f"{missing_above.size} features with count > boundary {boundary} missing from "
        f"banked set (first ids: {missing_above[:5].tolist()})"
    )
    n_tied = int((counts == boundary).sum())
    print(
        f"[build] f_out set-validity gate PASS (banked feat_ids authoritative): "
        f"boundary count={boundary}, n strictly above={n_above}, n tied at "
        f"boundary={n_tied}, tie slots={min(cap, n_eligible) - n_above}, "
        f"n_eligible={n_eligible}, n_banked={len(banked_fids)}",
        flush=True,
    )
    f_out = banked_fids
    # f_in (px/cx, cap 8192) keeps the recompute — the same argsort tie-order caveat
    # applies at ITS cap boundary, but f_in only shapes X, which this task never
    # consumes (built alongside Y then deleted; plan section 4 P1) — do NOT gate it.
    f_in = {
        a: SN.restrict(scan["in_fit"][a], scan["n_fit"], SA.MAX_FEATURES_IN)[0]
        for a in ("px", "cx")
    }
    sets_pos = MTF.split_positions(split, scan["ci"])
    ho = sets_pos["holdout"]
    assert len(ho) == banked["split_counts"]["holdout"], (
        f"holdout {len(ho)} != banked {banked['split_counts']['holdout']}"
    )
    # gate: per-cell fingerprint string equality + ci row alignment (pairwise coherence)
    ci_ho = scan["ci"][ho]
    for space, mapping in SPACES.items():
        for arm, cell in mapping.items():
            z = np.load(Path(cfg.pred16_src) / f"{cell}.npz")
            fp_str = str(z["fingerprint"])
            assert fp_str == banked["assembly_fingerprint"], (
                f"{cell}: fingerprint {fp_str[:16]}... != banked "
                f"{banked['assembly_fingerprint'][:16]}..."
            )
            assert (z["ci"] == ci_ho).all(), f"{cell}: pred16 ci misaligned with scan holdout"
    print("[build] identity gates PASS (fingerprint/ci/f_out/floor/counts/shas)", flush=True)
    return scan, f_out, f_in, ho


def _rss_projection_gb(scan: dict, f_out: np.ndarray, f_in: dict, n_ho: int) -> float:
    """Projected peak RSS of the Y rebuild: 3 pooling COOs + px/cx COOs (12 B/nnz)
    + one CSR transient + two fp64 holdout blocks + fixed overhead."""
    nnz_y = int(scan["out_all"][f_out].sum())
    nnz_x = sum(int(scan["in_all"][a][f_in[a]].sum()) for a in ("px", "cx"))
    coo_gb = (3 * nnz_y + nnz_x) * 12 / 2**30
    csr_transient_gb = nnz_y * 12 / 2**30
    ho_gb = 2 * n_ho * len(f_out) * 8 / 2**30
    return coo_gb + csr_transient_gb + ho_gb + 2.0


def phase_build(cfg: Cfg) -> None:
    """Identity gates -> verbatim Y rebuild -> per-cell banked-R^2 gate + nerr."""
    print("[phase=build] start", flush=True)
    banked = _banked(cfg)
    fp = banked["assembly_fingerprint"]
    scan, f_out, f_in, ho = _scan_and_gate(cfg, banked)
    proj_gb = _rss_projection_gb(scan, f_out, f_in, len(ho))
    print(f"[build] projected peak RSS ~{proj_gb:.1f} GB (cap {cfg.rss_cap_gb})", flush=True)
    if proj_gb > cfg.rss_cap_gb:
        _atomic_json(
            cfg.sentinel_dir() / "rss_halt.json",
            {"projected_gb": proj_gb, "cap_gb": cfg.rss_cap_gb, **_repro_meta(cfg)},
        )
        print(
            f"[build] HALT: projected RSS {proj_gb:.1f} GB > {cfg.rss_cap_gb} GB — "
            "re-dispatch on cpu-bigmem (plan kill criterion 2; rss_halt.json written)",
            flush=True,
        )
        sys.stdout.flush()
        sys.exit(RC_RSS_HALT)
    ns = argparse.Namespace(local_sae_dir=str(cfg.local_sae_dir))
    names = SA._sae_chunk_names(ns, None)
    paths = [Path(cfg.local_sae_dir) / n for n in names]
    mm_dir = cfg.staging_root / "mm"
    t0 = time.time()
    X, Y, dense_mm, _h = SA._build_sae_matrices(paths, scan, f_out, f_in, mm_dir)
    del X, dense_mm  # px/cx inputs built alongside, unused (plan section 4 P1)
    print(f"[build] Y rebuild done in {time.time() - t0:.0f}s", flush=True)
    ci_ho = scan["ci"][ho]
    cell_i, n_cells = 0, sum(len(m) for m in SPACES.values())
    for pool in SA.POOLINGS:
        y_ho = np.asarray(SA._rows(Y[pool], ho), dtype=np.float64)
        del Y[pool]
        yh_dir = cfg.space_dir(Y_HOLDOUT_SPACE[pool]) / "y_holdout"
        yh_dir.mkdir(parents=True, exist_ok=True)
        yh_p = yh_dir / f"L{LAYER}.npz"
        # resume-skip keyed on the assembly fingerprint (mirror of the percontext
        # skip below) — bare existence would silently reuse a stale-pin y_holdout.
        yh_reuse = False
        if yh_p.exists() and not cfg.force:
            with np.load(yh_p) as old:
                yh_reuse = "fingerprint" in old and str(old["fingerprint"]) == fp
        if yh_reuse:
            print(
                f"[build] y_holdout {Y_HOLDOUT_SPACE[pool]}/L{LAYER}.npz ({pool}): "
                "resume-skip (fingerprint match)",
                flush=True,
            )
        else:
            np.savez(
                yh_p,
                y16=y_ho.astype(np.float16),
                ci=ci_ho,
                fingerprint=np.array(fp),
                pooling=np.array(pool),
            )
        for space in [s for s, p in POOL_OF_SPACE.items() if p == pool]:
            pc_dir = cfg.space_dir(space) / "percontext"
            pd_dir = cfg.space_dir(space) / "pred16"
            pc_dir.mkdir(parents=True, exist_ok=True)
            pd_dir.mkdir(parents=True, exist_ok=True)
            for arm, cell in SPACES[space].items():
                cell_i += 1
                tc = time.time()
                pc_p = pc_dir / f"{arm}_L{LAYER}_ridge.npz"
                if pc_p.exists() and not cfg.force:
                    with np.load(pc_p) as old:
                        if "fingerprint" in old and str(old["fingerprint"]) == fp:
                            print(
                                f"[build] cell {cell_i}/{n_cells} {cell}: resume-skip",
                                flush=True,
                            )
                            continue
                src = Path(cfg.pred16_src) / f"{cell}.npz"
                with np.load(src) as z:
                    pred = z["pred16"].astype(np.float64)
                r2 = SA._pooled_r2(pred, y_ho)
                banked_r2 = banked["cells"][cell]["holdout_r2"]
                assert abs(r2 - banked_r2) < R2_TOL, (
                    f"{cell}: recomputed pooled R2 {r2:.6f} deviates from banked "
                    f"{banked_r2:.6f} by {abs(r2 - banked_r2):.2e} >= {R2_TOL} — HALT for "
                    "re-pin (plan kill criterion 1; never loosen the tolerance)"
                )
                nerr = MTF._percontext_nerr(pred, y_ho).astype(np.float32)
                np.savez(pc_p, nerr=nerr, ci=ci_ho, fingerprint=np.array(fp))
                shutil.copyfile(src, pd_dir / f"{arm}_L{LAYER}_ridge.npz")
                print(
                    f"[build] cell {cell_i}/{n_cells} {cell} -> {space}/{arm}: "
                    f"R2={r2:.4f} (banked {banked_r2:.4f}) elapsed={time.time() - tc:.0f}s",
                    flush=True,
                )
    print("[phase=build] done", flush=True)


# ── P2: floors (approximate SAE-space K-resample floor) ─────────────────────────────


def _load_sae(cfg: Cfg):
    """Production: revision-pinned suite trainer via the module loader; smoke:
    the tiny from-config fixture SAE (same class, same encode path)."""
    if cfg.smoke_model_dir:
        _hf, _tok, sae = SA._smoke_models(
            Path(cfg.smoke_model_dir), argparse.Namespace(), model=False
        )
        return sae
    return SAEMOD.BatchTopKSAE.load(k=64, device="cpu", cache_dir=Path(cfg.sae_cache), layer=LAYER)


def _floors_outputs_current(cfg: Cfg) -> str | None:
    """Entry skip-guard predicate: a one-line reason when every floors output
    exists AND the floors npz carries the current y_holdout fingerprint; else
    None (recompute). Any unreadable/partial output reads as stale -> recompute."""
    kdir = cfg.out_eval / "floors_env_mean" / "kresample"
    np_p = kdir / f"floors_L{LAYER}.npz"
    yh_p = cfg.space_dir("sae_space") / "y_holdout" / f"L{LAYER}.npz"
    label_dsts = [
        cfg.out_eval / env / "judge_labels" / "labels.json"
        for env in ("floors_env_mean", "floors_env_labels_only")
    ]
    if not (
        np_p.is_file()
        and (kdir / "floor_summary.json").is_file()
        and yh_p.is_file()
        and all(p.is_file() for p in label_dsts)
    ):
        return None
    with np.load(yh_p) as yh:
        want = str(yh["fingerprint"])
    with np.load(np_p) as z:
        if "fingerprint" not in z or str(z["fingerprint"]) != want:
            return None  # pre-guard or stale-pin floors npz -> recompute
    return f"floors_L{LAYER}.npz fingerprint matches y_holdout ({want[:16]}...)"


def phase_floors(cfg: Cfg) -> None:
    """Approximate SAE-space K-resample floor: SAE-encode the banked per-draw MEAN
    dense states, restrict to f_out, then the verbatim #1738 floor arithmetic
    (``phase_kresample_floor`` L446-453). Construct label: ``sae_enc_of_mean_state
    (approximate)`` — the verdict never rests on this arm (plan section 4 P2)."""
    print("[phase=floors] start", flush=True)
    if not cfg.force:
        reason = _floors_outputs_current(cfg)
        if reason:
            print(
                f"[phase=floors] skip — outputs current ({reason}); --force recomputes", flush=True
            )
            return
    yh = np.load(cfg.space_dir("sae_space") / "y_holdout" / f"L{LAYER}.npz")
    y16, yci = yh["y16"].astype(np.float64), yh["ci"]
    yh_fp = str(yh["fingerprint"])
    kpaths = sorted(Path(cfg.kresample_dir).glob("kresample_shard*.pt"))
    assert kpaths, f"no kresample shards under {cfg.kresample_dir}"
    cis: list[int] = []
    vs: list[torch.Tensor] = []
    for p in kpaths:
        b = torch.load(p, map_location="cpu", weights_only=False)
        layers = [int(x) for x in b["layers"]]
        assert LAYER in layers, f"{p.name}: layer {LAYER} not in {layers}"
        cis.extend(int(c) for c in b["ci"])
        vs.append(b["V"][:, :, layers.index(LAYER), :].to(torch.float32))
    V = torch.cat(vs, dim=0)
    kci = np.asarray(cis, dtype=np.int64)
    n, k_draws, h = V.shape
    assert k_draws == 4, f"k_draws {k_draws} != 4 (plan A7)"
    pos_of = {int(c): p for p, c in enumerate(yci.tolist())}
    joined = [(i, pos_of[int(c)]) for i, c in enumerate(kci) if int(c) in pos_of]
    assert len(joined) == len(kci), (
        f"kresample join {len(joined)}/{len(kci)} — every shard ci must join the holdout"
    )
    if not cfg.smoke:
        fs_p = cfg.dense_eval / "kresample" / "floor_summary.json"
        banked_n = json.loads(fs_p.read_text())["per_layer"][str(LAYER)]["n"]
        assert len(joined) == banked_n, f"join n {len(joined)} != banked dense floor n {banked_n}"
    ki = np.asarray([a for a, _ in joined])
    hp = np.asarray([b_ for _, b_ in joined])
    joined_ci = kci[ki]
    # SAE-encode each per-draw mean dense state (capture-side encode call), restrict
    pf = np.load(Path(cfg.perfeature_npz))
    f_out = pf["feat_ids"]
    sae = _load_sae(cfg)
    f_out_t = torch.as_tensor(np.asarray(f_out, dtype=np.int64))
    flat = V.reshape(n * k_draws, h)
    enc = np.empty((n * k_draws, len(f_out)), dtype=np.float64)
    t0 = time.time()
    chunk = 2048
    for s in range(0, flat.shape[0], chunk):
        f = sae.encode(flat[s : s + chunk])
        enc[s : s + f.shape[0]] = f[:, f_out_t].numpy().astype(np.float64)
    print(
        f"[floors] encoded {flat.shape[0]} draw-states -> F={len(f_out)} "
        f"in {time.time() - t0:.0f}s",
        flush=True,
    )
    E = enc.reshape(n, k_draws, len(f_out))[ki]
    # verbatim floor arithmetic (issue1738_characterize.phase_kresample_floor L446-453)
    mu = y16.mean(axis=0)
    den = ((y16[hp] - mu) ** 2).sum(axis=1)
    ebar = E.mean(axis=1, keepdims=True)
    floor = ((E - ebar) ** 2).sum(axis=(1, 2)) / (k_draws - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = floor / den
    env_mean = cfg.out_eval / "floors_env_mean"
    kdir = env_mean / "kresample"
    kdir.mkdir(parents=True, exist_ok=True)
    np.savez(
        kdir / f"floors_L{LAYER}.npz",
        ci=joined_ci,
        floor=floor.astype(np.float64),
        den=den.astype(np.float64),
        share=share.astype(np.float64),
        fingerprint=np.array(yh_fp),  # entry skip-guard key (lineage to y_holdout)
    )
    _atomic_json(
        kdir / "floor_summary.json",
        {
            "construct": "sae_enc_of_mean_state (approximate)",
            "note": (
                "floor = ddof-1 trace over K SAE-ENCODINGS-OF-MEAN dense draws; the exact "
                "floor needs per-token SAE encodings of K fresh answers (GPU re-capture, "
                "out of scope at 0 GPU-h). Verdict rests on the UNADJUSTED battery."
            ),
            "n": int(len(joined)),
            "k_draws": int(k_draws),
            "floor_median": float(np.nanmedian(floor)),
            "floor_share_median": float(np.nanmedian(share)),
            **_repro_meta(cfg),
        },
    )
    # judge CONTEXT labels (banked #1773-unaffected instrument) into both env roots
    labels_src = (
        cfg.dense_eval / "judge_labels" / "labels.json"
        if not cfg.smoke
        else Path(cfg.manifest_dir).parent / "judge_labels" / "labels.json"
    )
    assert labels_src.is_file(), labels_src
    for env in ("floors_env_mean", "floors_env_labels_only"):
        dst = cfg.out_eval / env / "judge_labels" / "labels.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(labels_src, dst)
    print("[phase=floors] done", flush=True)


# ── P3: taxonomy (verbatim parent battery, 4 checked subprocess invocations) ────────


def _taxonomy_output_current(cfg: Cfg, space: str) -> str | None:
    """Entry skip-guard predicate for one space: a one-line reason when the
    battery outputs exist, carry all 3 arm tables, and are newer than every npz
    input the battery consumes; else None (re-run). Unparseable -> re-run."""
    tax_p = cfg.space_dir(space) / "taxonomy.json"
    depth_p = cfg.space_dir(space) / "depth_contrasts.json"
    if not (tax_p.is_file() and depth_p.is_file()):
        return None
    try:
        tax = json.loads(tax_p.read_text())
    except json.JSONDecodeError:
        return None  # partial/corrupt output from an interrupted run -> re-run
    if not all(f"{a}_L{LAYER}_ridge" in tax.get("arms", {}) for a in ARMS):
        return None
    inputs = (
        list((cfg.space_dir(space) / "percontext").glob("*.npz"))
        + list((cfg.space_dir(space) / "pred16").glob("*.npz"))
        + list(cfg.y_holdout_dir(space).glob("*.npz"))
    )
    if not inputs:
        return None
    if tax_p.stat().st_mtime < max(p.stat().st_mtime for p in inputs):
        return None  # inputs rebuilt after the last battery run -> re-run
    return "taxonomy.json + depth_contrasts.json current (3 arm tables, newer than npz inputs)"


def phase_taxonomy(cfg: Cfg) -> None:
    """Run ``issue1738_characterize.py --phase taxonomy`` VERBATIM per space —
    no reimplemented statistics (plan section 4 P3). A non-zero rc halts loudly."""
    print("[phase=taxonomy] start", flush=True)
    script = PROJECT_ROOT / "scripts" / "issue1738_characterize.py"
    for i, space in enumerate(SPACES, 1):
        if not cfg.force:
            reason = _taxonomy_output_current(cfg, space)
            if reason:
                print(
                    f"[taxonomy] space {i}/4 {space}: skip — {reason}; --force recomputes",
                    flush=True,
                )
                continue
        cmd = [
            sys.executable,
            str(script),
            "--phase",
            "taxonomy",
            "--layers",
            str(LAYER),
            "--arms",
            "prefix,context,bare",
            "--out-eval",
            str(cfg.space_dir(space)),
            "--parent-eval",
            str(cfg.out_eval / PARENT_ENV_OF_SPACE[space]),
            "--pred16-dir",
            str(cfg.space_dir(space) / "pred16"),
            "--y-holdout-dir",
            str(cfg.y_holdout_dir(space)),
            "--manifest-dir",
            str(cfg.manifest_dir),
            "--split-file",
            str(cfg.split_file),
            "--scratch",
            str(cfg.staging_root / "scratch"),
        ]
        if cfg.no_upload or cfg.smoke:
            cmd.append("--no-upload")
        else:
            cmd += ["--upload-prefix", f"{cfg.upload_prefix}/{space}"]
            # plan section 8 mitigation: never dispatch without the child prefix
            assert "--upload-prefix" in cmd
        t0 = time.time()
        subprocess.run(cmd, check=True, env={**os.environ})
        tax = json.loads((cfg.space_dir(space) / "taxonomy.json").read_text())
        n_arms = len(tax["arms"])
        print(
            f"[taxonomy] space {i}/4 {space}: {n_arms} arm tables in {time.time() - t0:.0f}s",
            flush=True,
        )
        assert n_arms == 3, f"{space}: expected 3 arm tables, got {sorted(tax['arms'])}"
    print("[phase=taxonomy] done", flush=True)


# ── P4: compare (pre-registered cross-space statistics + mechanical verdict) ────────


def _spearman_perm(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> tuple[float, float]:
    """Spearman rho + TWO-SIDED permutation p (fraction of draws with |rho_perm| >=
    |rho_obs|; contrast labels shuffled — plan section 3 as amended in v4).
    Batched: ranks once, all draws as one matmul."""
    rx = scipy.stats.rankdata(x, method="average")
    ry = scipy.stats.rankdata(y, method="average")
    xc = rx - rx.mean()
    yc = ry - ry.mean()
    nx, ny = np.linalg.norm(xc), np.linalg.norm(yc)
    if nx < 1e-12 or ny < 1e-12:
        return float("nan"), float("nan")
    rho = float(xc @ yc / (nx * ny))
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random((n_perm, len(x))), axis=1)
    rho_perm = (yc[perm] @ xc) / (nx * ny)
    p = float(np.mean(np.abs(rho_perm) >= abs(rho)))
    return rho, p


def _deltas(tax: dict, arm: str) -> dict[str, dict]:
    key = f"{arm}_L{LAYER}_ridge"
    assert key in tax["arms"], (key, sorted(tax["arms"]))
    return {r["contrast"]: r for r in tax["arms"][key]["contrasts"]}


def _family(tax: dict, arm: str) -> list[str]:
    return list(tax["arms"][f"{arm}_L{LAYER}_ridge"]["family"])


def _read_dense_csv(path: Path, cols: list[str]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {c: {} for c in cols}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ci = int(row["ci"])
            for c in cols:
                out[c][ci] = float(row[c])
    return out


def _sign_agree(m_a: dict[str, float], m_b: dict[str, float], names: list[str]) -> int:
    return int(sum(1 for c in names if np.sign(m_a[c]) == np.sign(m_b[c]) != 0))


def _verdict(rho: float, p: float, sa: int) -> str:
    """Mechanical 4-way lattice (plan section 3 — DISJOINT and exhaustive)."""
    if np.isnan(rho) or np.isnan(p):
        return "Mixed"
    if p >= 0.05:
        return "Uncorrelated"
    if rho > 0 and sa >= SA_M_REPRODUCE_MIN:
        return "Reproduced"
    if rho < 0 and sa <= SA_M_INVERT_MAX:
        return "Inverted"
    return "Mixed"


def _compare_outputs_current(cfg: Cfg) -> str | None:
    """Entry skip-guard predicate: a one-line reason when the comparison JSON +
    SAE per-context CSV exist, parse with a verdict, and are newer than every
    per-space taxonomy.json input; else None (recompute)."""
    out_p = cfg.out_eval / "crossspace_comparison.json"
    csv_p = cfg.out_eval / f"percontext_summary_L{LAYER}_ridge_sae.csv"
    if not (out_p.is_file() and csv_p.is_file()):
        return None
    try:
        comp = json.loads(out_p.read_text())
    except json.JSONDecodeError:
        return None  # partial/corrupt output from an interrupted run -> recompute
    if "verdict" not in comp or "registered_primary" not in comp:
        return None
    tax_ps = [cfg.space_dir(s) / "taxonomy.json" for s in SPACES]
    if not all(p.is_file() for p in tax_ps):
        return None
    if out_p.stat().st_mtime < max(p.stat().st_mtime for p in tax_ps):
        return None  # taxonomy re-ran after the last compare -> recompute
    return f"crossspace_comparison.json current (verdict={comp['verdict']}, newer than taxonomy)"


def phase_compare(cfg: Cfg) -> None:
    print("[phase=compare] start", flush=True)
    if not cfg.force:
        reason = _compare_outputs_current(cfg)
        if reason:
            print(
                f"[phase=compare] skip — outputs current ({reason}); --force recomputes",
                flush=True,
            )
            return
    dense_tax = json.loads((cfg.dense_eval / "taxonomy.json").read_text())
    dense_bare_tax = json.loads((cfg.dense_eval / "bare_query" / "taxonomy.json").read_text())
    dense_rows = {
        "prefix": _deltas(dense_tax, "prefix"),
        "context": _deltas(dense_tax, "context"),
        "bare": _deltas(dense_bare_tax, "bare"),
    }
    dense_family = _family(dense_tax, "prefix")
    sae_tax = {s: json.loads((cfg.space_dir(s) / "taxonomy.json").read_text()) for s in SPACES}
    # row-coverage assert (plan section 3): identical contrast families, all spaces + dense
    for space in SPACES:
        for arm in ARMS:
            fam = _family(sae_tax[space], arm)
            assert fam == dense_family, (
                f"{space}/{arm}: contrast family differs from dense "
                f"({len(fam)} vs {len(dense_family)})"
            )
    if not cfg.smoke:
        assert len(dense_family) == 22, f"dense family n={len(dense_family)} != 22"
    # dense BH-significant union (prefix OR bare) — the SA_M denominator
    union = [
        c
        for c in dense_family
        if dense_rows["prefix"][c]["bh_significant"] or dense_rows["bare"][c]["bh_significant"]
    ]
    if not cfg.smoke:
        assert len(union) == N_UNION_EXPECTED, (
            f"dense significant union n={len(union)} != {N_UNION_EXPECTED} (plan section 2)"
        )
    d_delta = {a: {c: dense_rows[a][c]["delta_mean_nerr"] for c in dense_family} for a in ARMS}
    m_dense = {c: d_delta["prefix"][c] - d_delta["bare"][c] for c in dense_family}

    def space_stats(space: str) -> dict:
        rows = {a: _deltas(sae_tax[space], a) for a in ARMS}
        s_delta = {a: {c: rows[a][c]["delta_mean_nerr"] for c in dense_family} for a in ARMS}
        m_sae = {c: s_delta["prefix"][c] - s_delta["bare"][c] for c in dense_family}
        mx = np.asarray([m_sae[c] for c in dense_family])
        my = np.asarray([m_dense[c] for c in dense_family])
        rho_m, p_m = _spearman_perm(mx, my, N_PERM_CROSS, CROSS_SEED)
        sa_m = _sign_agree(m_sae, m_dense, union)
        per_arm = {}
        for a in ARMS:
            ax = np.asarray([s_delta[a][c] for c in dense_family])
            ay = np.asarray([d_delta[a][c] for c in dense_family])
            rho_a, p_a = _spearman_perm(ax, ay, N_PERM_CROSS, CROSS_SEED)
            sig_a = [c for c in dense_family if dense_rows[a][c]["bh_significant"]]
            per_arm[a] = {
                "rho": rho_a,
                "perm_p_two_sided": p_a,
                "n_dense_significant": len(sig_a),
                "sign_agree_on_dense_significant": _sign_agree(s_delta[a], d_delta[a], sig_a),
            }
        return {
            "rho_M": rho_m,
            "rho_M_perm_p_two_sided": p_m,
            "SA_M": sa_m,
            "n_union": len(union),
            "per_arm": per_arm,
            "M_sae": {c: m_sae[c] for c in dense_family},
        }

    stats = {s: space_stats(s) for s in SPACES}
    primary = stats["sae_space"]
    verdict = _verdict(primary["rho_M"], primary["rho_M_perm_p_two_sided"], primary["SA_M"])

    # per-context joins (n=9,941 registered secondary; plan section 3)
    dense_csv = _read_dense_csv(
        cfg.dense_eval / f"percontext_summary_L{LAYER}_ridge.csv",
        [f"nerr_prefix_L{LAYER}_ridge", f"nerr_context_L{LAYER}_ridge"],
    )
    bare_csv = _read_dense_csv(
        cfg.dense_eval / "bare_query" / f"percontext_summary_L{LAYER}_ridge.csv",
        [f"nerr_bare_L{LAYER}_ridge"],
    )
    dense_nerr = {
        "prefix": dense_csv[f"nerr_prefix_L{LAYER}_ridge"],
        "context": dense_csv[f"nerr_context_L{LAYER}_ridge"],
        "bare": bare_csv[f"nerr_bare_L{LAYER}_ridge"],
    }
    sae_nerr: dict[str, np.ndarray] = {}
    ci_ho: np.ndarray | None = None
    for arm in ARMS:
        with np.load(cfg.space_dir("sae_space") / "percontext" / f"{arm}_L{LAYER}_ridge.npz") as z:
            sae_nerr[arm] = z["nerr"].astype(np.float64)
            if ci_ho is None:
                ci_ho = z["ci"].copy()
            else:
                assert (z["ci"] == ci_ho).all(), f"{arm}: percontext ci misaligned"
    assert ci_ho is not None
    missing = [int(c) for c in ci_ho if int(c) not in dense_nerr["prefix"]]
    assert not missing, f"{len(missing)} holdout ci absent from the dense percontext CSVs"
    per_context = {}
    dn = {a: np.asarray([dense_nerr[a][int(c)] for c in ci_ho]) for a in ARMS}
    for arm in ARMS:
        rho, pv = scipy.stats.spearmanr(sae_nerr[arm], dn[arm])
        per_context[arm] = {"spearman": float(rho), "p": float(pv), "n": int(len(ci_ho))}
    diff_sae = sae_nerr["prefix"] - sae_nerr["bare"]
    diff_dense = dn["prefix"] - dn["bare"]
    rho_d, p_d = scipy.stats.spearmanr(diff_sae, diff_dense)
    per_context["difference_prefix_minus_bare"] = {
        "spearman": float(rho_d),
        "p": float(p_d),
        "n": int(len(ci_ho)),
    }

    # descriptive six-category mirror table (plan section 3)
    mirror = {}
    for side, cats in MIRROR_CATEGORIES.items():
        for cat in cats:
            if cat not in dense_family:
                mirror[f"{side}:{cat}"] = {"present": False}
                continue
            row_s = _deltas(sae_tax["sae_space"], side).get(cat)
            row_d = dense_rows[side].get(cat)
            mirror[f"{side}:{cat}"] = {
                "present": True,
                "sae_delta": row_s["delta_mean_nerr"],
                "sae_boot_ci": row_s["boot_ci"],
                "sae_bh_significant": row_s["bh_significant"],
                "dense_delta": row_d["delta_mean_nerr"],
                "dense_bh_significant": row_d["bh_significant"],
            }

    # floor-adjusted sensitivity (robustness arm only — verdict rests on unadjusted)
    floor_adj = {}
    for arm in ARMS:
        d_arm = dense_rows[arm]
        d_tax_arm = (dense_bare_tax if arm == "bare" else dense_tax)["arms"][
            f"{arm}_L{LAYER}_ridge"
        ]
        s_tax_arm = sae_tax["sae_space"]["arms"][f"{arm}_L{LAYER}_ridge"]
        if "floor_adjusted" not in d_tax_arm or "floor_adjusted" not in s_tax_arm:
            floor_adj[arm] = {"available": False}
            continue
        d_fa = {r["contrast"]: r for r in d_tax_arm["floor_adjusted"]["contrasts"]}
        s_fa = {r["contrast"]: r for r in s_tax_arm["floor_adjusted"]["contrasts"]}
        shared = [c for c in d_fa if c in s_fa]
        dv = np.asarray([d_fa[c]["delta_mean_adj_nerr"] for c in shared])
        sv = np.asarray([s_fa[c]["delta_mean_adj_nerr"] for c in shared])
        rho_fa, p_fa = _spearman_perm(sv, dv, N_PERM_CROSS, CROSS_SEED)
        sig_shared = [c for c in shared if d_fa[c]["bh_significant"]]
        agree = int(
            sum(
                1
                for c in sig_shared
                if np.sign(s_fa[c]["delta_mean_adj_nerr"])
                == np.sign(d_fa[c]["delta_mean_adj_nerr"])
                != 0
            )
        )
        floor_adj[arm] = {
            "available": True,
            "n_shared_contrasts": len(shared),
            "rho": rho_fa,
            "perm_p_two_sided": p_fa,
            "n_dense_significant_shared": len(sig_shared),
            "sign_agree_on_dense_significant": agree,
            "construct_caveat": "sae_enc_of_mean_state (approximate) — robustness only",
        }

    # EXPLORATORY (plan-allowed, clearly labeled): denominator-free per-category
    # sign-share of (nerr_prefix - nerr_bare) in both spaces.
    labels = CH._load_labels(cfg.out_eval / "floors_env_mean")
    fields = CH._manifest_fields(Path(cfg.manifest_dir))
    masks = CH._contrast_masks(ci_ho, labels, fields)
    sign_share = {}
    for name, m in masks:
        sign_share[name] = {
            "n_group": int(m.sum()),
            "sae_share_prefix_harder": float((diff_sae[m] > 0).mean()),
            "dense_share_prefix_harder": float((diff_dense[m] > 0).mean()),
        }

    out = {
        "layer": LAYER,
        "seed_cross": CROSS_SEED,
        "n_perm_cross": N_PERM_CROSS,
        "verdict": verdict,
        "verdict_rule": (
            "Reproduced iff perm_p<0.05 AND rho_M>0 AND SA_M>="
            f"{SA_M_REPRODUCE_MIN}/{N_UNION_EXPECTED}; Inverted iff perm_p<0.05 AND rho_M<0 "
            f"AND SA_M<={SA_M_INVERT_MAX}; Uncorrelated iff perm_p>=0.05; Mixed otherwise "
            "(plan section 3; verdict on sae_space, unadjusted battery)"
        ),
        "registered_primary": {
            "rho_M": primary["rho_M"],
            "rho_M_perm_p_two_sided": primary["rho_M_perm_p_two_sided"],
            "SA_M": primary["SA_M"],
            "n_union": primary["n_union"],
            "union": union,
        },
        "registered_secondary": {
            "per_arm": primary["per_arm"],
            "per_context": per_context,
        },
        "robustness_spaces": {
            s: {k: v for k, v in stats[s].items() if k != "M_sae"}
            for s in ("dense_feat_space", "max_space", "frac_space")
        },
        "descriptive_mirror_categories": mirror,
        "floor_adjusted_sensitivity": floor_adj,
        "exploratory_sign_share_prefix_harder": {
            "note": "EXPLORATORY (Alternatives-critic companion; plan item 11) — "
            "denominator-free per-category share of holdout contexts where "
            "nerr_prefix > nerr_bare, per space",
            "by_contrast": sign_share,
        },
        "contrast_family": dense_family,
        "inputs": {
            "sae_fits": str(cfg.sae_fits_path),
            "assembly_fingerprint": _banked(cfg)["assembly_fingerprint"],
            "dense_eval": str(cfg.dense_eval),
        },
        **_repro_meta(cfg),
    }
    _atomic_json(cfg.out_eval / "crossspace_comparison.json", out)
    # the dense-CSV-shaped SAE per-context summary (plan section 4 P4)
    csv_p = cfg.out_eval / f"percontext_summary_L{LAYER}_ridge_sae.csv"
    with csv_p.open("w", encoding="utf-8") as fh:
        fh.write("ci,nerr_sae_prefix,nerr_sae_bare,nerr_sae_context,language,topic,format\n")
        for i, c in enumerate(ci_ho):
            lab = labels.get(str(int(c))) or {}
            fh.write(
                f"{int(c)},{sae_nerr['prefix'][i]:.6f},{sae_nerr['bare'][i]:.6f},"
                f"{sae_nerr['context'][i]:.6f},{lab.get('language', '')},"
                f"{lab.get('topic', '')},{lab.get('format', '')}\n"
            )
    print(
        f"[compare] verdict={verdict} rho_M={primary['rho_M']:.4f} "
        f"p={primary['rho_M_perm_p_two_sided']:.4g} SA_M={primary['SA_M']}/{len(union)}",
        flush=True,
    )
    print("[phase=compare] done", flush=True)


# ── P5: figures ──────────────────────────────────────────────────────────────────


def _check_png(path: Path) -> None:
    """Load-check a rendered PNG: exists, non-trivial size, non-constant pixels."""
    assert path.is_file() and path.stat().st_size > 5_000, path
    import matplotlib.pyplot as plt

    img = plt.imread(path)
    assert img.size > 0 and float(np.ptp(img)) > 0, f"{path}: blank render"


def _figures_outputs_current(cfg: Cfg) -> str | None:
    """Entry skip-guard predicate: a one-line reason when meta.json + every
    figure it lists exist, load-check as non-blank PNGs, and meta.json is newer
    than the comparison JSON it renders; else None (re-render)."""
    meta_p = cfg.fig_dir / "meta.json"
    comp_p = cfg.out_eval / "crossspace_comparison.json"
    if not (meta_p.is_file() and comp_p.is_file()):
        return None
    try:
        meta = json.loads(meta_p.read_text())
    except json.JSONDecodeError:
        return None  # partial/corrupt output from an interrupted run -> re-render
    figs = [cfg.fig_dir / n for n in meta.get("figures", [])]
    if not figs or not all(p.is_file() for p in figs):
        return None
    if meta_p.stat().st_mtime < comp_p.stat().st_mtime:
        return None  # compare re-ran after the last render -> re-render
    for p in figs:
        try:
            _check_png(p)
        except AssertionError:
            return None  # a blank/truncated PNG reads as stale -> re-render
    return f"{len(figs)} figures + meta.json current (load-checked, newer than comparison)"


def phase_figures(cfg: Cfg) -> None:
    print("[phase=figures] start", flush=True)
    import matplotlib

    matplotlib.use("Agg")
    if not cfg.force:
        reason = _figures_outputs_current(cfg)
        if reason:
            print(
                f"[phase=figures] skip — outputs current ({reason}); --force recomputes",
                flush=True,
            )
            return
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    comp = json.loads((cfg.out_eval / "crossspace_comparison.json").read_text())
    family: list[str] = comp["contrast_family"]
    union = set(comp["registered_primary"]["union"])
    dense_tax = json.loads((cfg.dense_eval / "taxonomy.json").read_text())
    dense_bare_tax = json.loads((cfg.dense_eval / "bare_query" / "taxonomy.json").read_text())
    sae_tax = {s: json.loads((cfg.space_dir(s) / "taxonomy.json").read_text()) for s in SPACES}
    d_rows = {
        "prefix": _deltas(dense_tax, "prefix"),
        "context": _deltas(dense_tax, "context"),
        "bare": _deltas(dense_bare_tax, "bare"),
    }
    s_rows = {sp: {a: _deltas(sae_tax[sp], a) for a in ARMS} for sp in SPACES}

    def m_vec(rows: dict) -> np.ndarray:
        return np.asarray(
            [
                rows["prefix"][c]["delta_mean_nerr"] - rows["bare"][c]["delta_mean_nerr"]
                for c in family
            ]
        )

    md = m_vec(d_rows)
    ms = m_vec(s_rows["sae_space"])
    pal = paper_palette(4)
    made: list[Path] = []

    # hero: cross-space mirror scatter (M_c per contrast, dense vs SAE)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    lim = max(1e-9, float(np.nanmax(np.abs(np.concatenate([md, ms]))))) * 1.15
    ax.axhspan(0, lim, xmin=0.5, xmax=1.0, color="0.92", zorder=0)
    ax.axhspan(-lim, 0, xmin=0.0, xmax=0.5, color="0.92", zorder=0)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.5", lw=0.8)
    filled = np.asarray([c in union for c in family])
    ax.scatter(md[filled], ms[filled], s=42, color=pal[0], label="dense BH-significant union")
    ax.scatter(
        md[~filled],
        ms[~filled],
        s=42,
        facecolors="none",
        edgecolors=pal[0],
        label="other contrasts",
    )
    for i, c in enumerate(family):
        ax.annotate(c, (md[i], ms[i]), fontsize=5, alpha=0.75)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("dense M_c = delta_prefix - delta_bare (mean nerr)")
    ax.set_ylabel("SAE-space M_c")
    ax.set_title("Which arm finds this category harder: SAE space vs dense")
    ax.legend(fontsize=7)
    p = cfg.fig_dir / "crossspace_mirror_scatter.png"
    fig.tight_layout()
    fig.savefig(p, dpi=200)
    plt.close(fig)
    made.append(p)

    # per-arm delta-delta scatters (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, arm, col in zip(axes, ARMS, pal[:3], strict=False):
        dx = np.asarray([d_rows[arm][c]["delta_mean_nerr"] for c in family])
        sy = np.asarray([s_rows["sae_space"][arm][c]["delta_mean_nerr"] for c in family])
        ax.axhline(0, color="0.6", lw=0.7)
        ax.axvline(0, color="0.6", lw=0.7)
        ax.scatter(dx, sy, s=28, color=col)
        pa = comp["registered_secondary"]["per_arm"][arm]
        ax.set_title(f"{arm}: rho={pa['rho']:.2f} (p={pa['perm_p_two_sided']:.3g})", fontsize=9)
        ax.set_xlabel("dense delta")
        ax.set_ylabel("SAE delta")
    p = cfg.fig_dir / "perarm_delta_scatter.png"
    fig.tight_layout()
    fig.savefig(p, dpi=200)
    plt.close(fig)
    made.append(p)

    # forest pair: SAE (with CIs) beside dense, per arm
    fig, axes = plt.subplots(3, 2, figsize=(11, 3.2 * len(family) / 6 + 6), sharey=True)
    ypos = np.arange(len(family))
    for r, arm in enumerate(ARMS):
        for cidx, (label, rows) in enumerate((("SAE", s_rows["sae_space"]), ("dense", d_rows))):
            ax = axes[r][cidx]
            d = np.asarray([rows[arm][c]["delta_mean_nerr"] for c in family])
            lo = np.asarray([rows[arm][c]["boot_ci"][0] for c in family])
            hi = np.asarray([rows[arm][c]["boot_ci"][1] for c in family])
            err = np.vstack([np.maximum(0, d - lo), np.maximum(0, hi - d)])
            sig = np.asarray([rows[arm][c]["bh_significant"] for c in family])
            ax.errorbar(d, ypos, xerr=err, fmt="none", ecolor="0.6", elinewidth=0.8)
            ax.scatter(d[sig], ypos[sig], s=22, color=pal[r])
            ax.scatter(d[~sig], ypos[~sig], s=22, facecolors="none", edgecolors=pal[r])
            ax.axvline(0, color="0.4", lw=0.7)
            ax.set_title(f"{arm} — {label}", fontsize=9)
            if cidx == 0:
                ax.set_yticks(ypos)
                ax.set_yticklabels(family, fontsize=6)
    p = cfg.fig_dir / "taxonomy_forest_pair.png"
    fig.tight_layout()
    fig.savefig(p, dpi=200)
    plt.close(fig)
    made.append(p)

    # low-level per-unit companion: per-context scatters
    sae_nerr = {}
    ci_ho = None
    for arm in ARMS:
        with np.load(cfg.space_dir("sae_space") / "percontext" / f"{arm}_L{LAYER}_ridge.npz") as z:
            sae_nerr[arm] = z["nerr"].astype(np.float64)
            ci_ho = z["ci"].copy() if ci_ho is None else ci_ho
    dense_csv = _read_dense_csv(
        cfg.dense_eval / f"percontext_summary_L{LAYER}_ridge.csv",
        [f"nerr_prefix_L{LAYER}_ridge"],
    )
    bare_csv = _read_dense_csv(
        cfg.dense_eval / "bare_query" / f"percontext_summary_L{LAYER}_ridge.csv",
        [f"nerr_bare_L{LAYER}_ridge"],
    )
    # constrained layout at creation: tight_layout after a colorbar raises under
    # the paper style (mpl layout-engine switch refusal, #920)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    hb = axes[0].hexbin(
        sae_nerr["prefix"], sae_nerr["bare"], gridsize=45, bins="log", cmap="viridis"
    )
    fig.colorbar(hb, ax=axes[0], label="log10 count")
    axes[0].set_xlabel("SAE nerr (prefix arm)")
    axes[0].set_ylabel("SAE nerr (bare arm)")
    axes[0].set_title(f"per-context SAE errors (n={len(sae_nerr['prefix'])})", fontsize=9)
    for ax, arm, col in zip(axes[1:], ("prefix", "bare"), pal[1:3], strict=False):
        key = f"nerr_{arm}_L{LAYER}_ridge"
        src = dense_csv if arm == "prefix" else bare_csv
        dvals = np.asarray([src[key][int(c)] for c in ci_ho])
        ax.scatter(dvals, sae_nerr[arm], s=3, alpha=0.15, color=col)
        rho = comp["registered_secondary"]["per_context"][arm]["spearman"]
        ax.set_xlabel(f"dense nerr ({arm})")
        ax.set_ylabel(f"SAE nerr ({arm})")
        ax.set_title(f"cross-space per-context ({arm}): rho={rho:.2f}", fontsize=9)
    p = cfg.fig_dir / "percontext_scatters.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    made.append(p)

    # exploratory dump: pooling variants + dense-input comparator delta-delta
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, sp in zip(axes, ("dense_feat_space", "max_space", "frac_space"), strict=False):
        mv = m_vec(s_rows[sp])
        ax.axhline(0, color="0.6", lw=0.7)
        ax.axvline(0, color="0.6", lw=0.7)
        ax.scatter(md, mv, s=24, color=pal[3])
        rs = comp["robustness_spaces"][sp]
        ax.set_title(f"{sp}: rho_M={rs['rho_M']:.2f}", fontsize=9)
        ax.set_xlabel("dense M_c")
        ax.set_ylabel(f"{sp} M_c")
    p = cfg.fig_dir / "robustness_spaces_scatter.png"
    fig.tight_layout()
    fig.savefig(p, dpi=200)
    plt.close(fig)
    made.append(p)

    # exploratory: floor-adjusted vs unadjusted SAE deltas (prefix + bare)
    sae_arm_tax = sae_tax["sae_space"]["arms"]
    if all("floor_adjusted" in sae_arm_tax[f"{a}_L{LAYER}_ridge"] for a in ("prefix", "bare")):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, arm in zip(axes, ("prefix", "bare"), strict=False):
            fa = {
                r["contrast"]: r
                for r in sae_arm_tax[f"{arm}_L{LAYER}_ridge"]["floor_adjusted"]["contrasts"]
            }
            shared = [c for c in family if c in fa]
            un = np.asarray([s_rows["sae_space"][arm][c]["delta_mean_nerr"] for c in shared])
            ad = np.asarray([fa[c]["delta_mean_adj_nerr"] for c in shared])
            ax.axline((0, 0), slope=1, color="0.6", lw=0.7)
            ax.scatter(un, ad, s=24, color=pal[2])
            ax.set_xlabel(f"unadjusted SAE delta ({arm})")
            ax.set_ylabel("floor-adjusted delta (approximate)")
        p = cfg.fig_dir / "flooradj_vs_unadjusted.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        made.append(p)

    # exploratory: depth-band holdout R^2 bars per space
    fig, ax = plt.subplots(figsize=(8, 4))
    bands = ("2-2", "3-4", ">=5")
    width = 0.8 / (len(SPACES) * 3)
    idx = 0
    for si, sp in enumerate(SPACES):
        depth = json.loads((cfg.space_dir(sp) / "depth_contrasts.json").read_text())
        for arm in ARMS:
            vals = [
                depth["arms"].get(f"{arm}_L{LAYER}_ridge", {}).get(b, {}).get("r2", np.nan)
                for b in bands
            ]
            ax.bar(
                np.arange(len(bands)) + idx * width,
                vals,
                width=width,
                color=pal[si],
                alpha=0.4 + 0.2 * ARMS.index(arm),
            )
            idx += 1
    ax.set_xticks(np.arange(len(bands)) + 0.4)
    ax.set_xticklabels(bands)
    ax.set_xlabel("conversation depth band")
    ax.set_ylabel("holdout R^2")
    ax.set_title("depth-stratified holdout R^2 (space x arm; exploratory)", fontsize=9)
    p = cfg.fig_dir / "depth_r2_bars.png"
    fig.tight_layout()
    fig.savefig(p, dpi=200)
    plt.close(fig)
    made.append(p)

    for pth in made:
        _check_png(pth)
    _atomic_json(
        cfg.fig_dir / "meta.json",
        {
            "figures": [str(x.name) for x in made],
            "hero": "crossspace_mirror_scatter.png",
            "caption_hero": (
                f"Per-contrast M_c = delta_prefix - delta_bare, dense (x) vs SAE space (y); "
                f"filled = dense BH-significant union. rho_M="
                f"{comp['registered_primary']['rho_M']:.3f}, two-sided perm p="
                f"{comp['registered_primary']['rho_M_perm_p_two_sided']:.4g}, SA_M="
                f"{comp['registered_primary']['SA_M']}/{comp['registered_primary']['n_union']}; "
                f"verdict={comp['verdict']}."
            ),
            **_repro_meta(cfg),
        },
    )
    print(f"[phase=figures] done ({len(made)} PNGs load-checked)", flush=True)


# ── P6: upload (one verified upload_folder commit per space + floors) ───────────────


def phase_upload(cfg: Cfg, *, upload_fn=hub._upload_folder_filtered) -> None:
    """npz artifacts -> HF ``{upload_prefix}/analysis_tensors/...`` (plan section 10);
    JSON summaries dual-write via the battery's own ``--upload-prefix`` and git.
    Eligibility filter = every ``*.npz`` under percontext/pred16/y_holdout/kresample —
    covers all plan-declared npz classes (section 6.5 parity; JSONs are git-resident)."""
    print("[phase=upload] start", flush=True)
    if cfg.no_upload and not cfg.smoke:
        print("[phase=upload] skipped (--no-upload)", flush=True)
        return
    n_commits = 0
    for space in SPACES:
        d = cfg.space_dir(space)
        files = sorted(
            str(p.relative_to(d))
            for sub in ("percontext", "pred16", "y_holdout")
            for p in (d / sub).glob("*.npz")
            if p.is_file()
        )
        if not files:
            continue
        dest = f"{cfg.upload_prefix}/analysis_tensors/{space}"
        url = upload_fn(
            d,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=dest,
            allow_patterns=files,
            expected_repo_paths=[f"{dest}/{f}" for f in files],
        )
        if not url:
            raise RuntimeError(f"upload for {space} returned no URL")
        n_commits += 1
        print(f"[upload] {space}: {len(files)} npz -> {dest}", flush=True)
    kdir = cfg.out_eval / "floors_env_mean" / "kresample"
    files = sorted(str(p.relative_to(kdir)) for p in kdir.glob("*") if p.is_file())
    dest = f"{cfg.upload_prefix}/analysis_tensors/floors_env_mean/kresample"
    url = upload_fn(
        kdir,
        repo_id=C.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=dest,
        allow_patterns=files,
        expected_repo_paths=[f"{dest}/{f}" for f in files],
    )
    if not url:
        raise RuntimeError("upload for floors returned no URL")
    n_commits += 1
    _atomic_json(
        cfg.sentinel_dir() / "upload.done.json",
        {"n_commits": n_commits, "upload_prefix": cfg.upload_prefix, **_repro_meta(cfg)},
    )
    print(f"[phase=upload] done ({n_commits} commits)", flush=True)


# ── smoke: tiny-real e2e through the SAME phase functions (PASS_UNIFIED) ────────────


def _smoke_labels_fixture(path: Path, ci: list[int]) -> None:
    """Synthetic judged-CONTEXT labels over the fixture ci (schema-identical to the
    banked instrument; values varied so language/format/refusal masks form)."""
    topics = ("factual_qa", "chitchat_social", "coding")
    labels = {
        str(i): {
            "language": "en" if i % 2 == 0 else "zh",
            "topic": topics[i % 3],
            "request_refusal_adjacent": "yes" if i % 4 == 0 else "no",
            "answer_is_refusal": "yes" if i % 8 == 0 else "no",
            "format": ("code", "list", "prose")[i % 3],
        }
        for i in ci
    }
    _atomic_json(path, {"labels": labels, "judge_model": "smoke-fixture", "drops": {}})


def _smoke_manifest_fixture(mdir: Path, ci: list[int]) -> None:
    """Manifest pool fixture (meta.json + one part) with varied depth/corpus fields."""
    mdir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"i": i, "depth": 2 + (i % 4), "corpus": "wildchat" if i % 3 == 0 else "lmsys"} for i in ci
    ]
    (mdir / "part_0000.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    _atomic_json(mdir / "meta.json", {"n_new": len(ci)})


def _smoke_kresample_fixture(kdir: Path, ho_ci: list[int], h: int) -> None:
    kdir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(1946)
    torch.save(
        {
            "ci": [int(c) for c in ho_ci],
            "V": torch.randn(len(ho_ci), 4, 1, h, generator=g).to(torch.float16),
            "layers": [LAYER],
        },
        kdir / "kresample_shard00.pt",
    )


def _smoke_dense_root(cfg: Cfg, dense_root: Path) -> None:
    """Synthesize the smoke's dense comparator from the dense-input-comparator
    space's REAL battery outputs (same family by construction) + CSVs from its
    percontext npz — so phase_compare runs its production join code unchanged."""
    dense_root.mkdir(parents=True, exist_ok=True)
    (dense_root / "bare_query").mkdir(parents=True, exist_ok=True)
    tax_src = cfg.space_dir("dense_feat_space") / "taxonomy.json"
    shutil.copyfile(tax_src, dense_root / "taxonomy.json")
    shutil.copyfile(tax_src, dense_root / "bare_query" / "taxonomy.json")
    labels = CH._load_labels(cfg.out_eval / "floors_env_mean")
    nerr = {}
    ci_ho = None
    for arm in ARMS:
        with np.load(
            cfg.space_dir("dense_feat_space") / "percontext" / f"{arm}_L{LAYER}_ridge.npz"
        ) as z:
            nerr[arm] = z["nerr"].astype(np.float64)
            ci_ho = z["ci"].copy() if ci_ho is None else ci_ho
    with (dense_root / f"percontext_summary_L{LAYER}_ridge.csv").open("w") as fh:
        fh.write(f"ci,nerr_prefix_L{LAYER}_ridge,nerr_context_L{LAYER}_ridge,language\n")
        for i, c in enumerate(ci_ho):
            lab = labels.get(str(int(c))) or {}
            fh.write(
                f"{int(c)},{nerr['prefix'][i]:.6f},{nerr['context'][i]:.6f},"
                f"{lab.get('language', '')}\n"
            )
    with (dense_root / "bare_query" / f"percontext_summary_L{LAYER}_ridge.csv").open("w") as fh:
        fh.write(f"ci,nerr_bare_L{LAYER}_ridge\n")
        for i, c in enumerate(ci_ho):
            fh.write(f"{int(c)},{nerr['bare'][i]:.6f}\n")


def run_smoke(smoke_base: Path) -> int:
    """Tiny-real CPU e2e: the SA fixture (built by ``issue1738_sae_arm.py --smoke``,
    the production-verified fixture entrypoint) feeds THIS driver's real phase
    functions; the Hub boundary is faked signature-conformantly (autospec)."""
    from unittest.mock import create_autospec

    t0 = time.time()
    if smoke_base.exists():
        shutil.rmtree(smoke_base)
    smoke_base.mkdir(parents=True, exist_ok=True)
    # leg 0: the SA smoke builds the tiny-real fixture through ITS production
    # entrypoints (capture + fits-with-bare at tiny N) — 12 pred16 cells,
    # sae_fits.json, perfeature npz, sae chunks, split fixture, tiny SAE/model.
    sa_out = smoke_base / "sa"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "issue1738_sae_arm.py"),
        "--smoke",
        "--out-dir",
        str(sa_out),
    ]
    print("[smoke] leg 0: issue1738_sae_arm --smoke (fixture builder)", flush=True)
    subprocess.run(cmd, check=True, env={**os.environ})
    base = sa_out / "_smoke_sae"
    split_p = base / "split_1738.json"
    split = json.loads(split_p.read_text())
    all_ci = [int(c) for s in split["sets"].values() for c in s["ci"]]
    ho_ci = [int(c) for c in split["sets"]["holdout"]["ci"]]
    # my fixtures: manifest fields, judged-context labels, kresample draws
    manifest_dir = smoke_base / "manifest"
    _smoke_manifest_fixture(manifest_dir, sorted(all_ci))
    _smoke_labels_fixture(smoke_base / "judge_labels" / "labels.json", sorted(all_ci))
    _smoke_kresample_fixture(smoke_base / "kresample", ho_ci, SA.SMOKE_H)
    cfg = Cfg(
        staging_root=smoke_base / "staging",
        out_eval=smoke_base / "eval",
        fig_dir=smoke_base / "figures",
        dense_eval=smoke_base / "dense",
        revision="smoke-fixture",
        upload_prefix=UPLOAD_PREFIX_DEFAULT + "_smoke",
        smoke=True,
        local_sae_dir=base / "cap" / "sae_chunks",
        pred16_src=base / "local_bare" / "pred16",
        sae_fits_path=base / "eval_bare" / "sae_fits.json",
        perfeature_npz=base / "local_bare" / "perfeature" / "perfeature_summary.npz",
        split_file=split_p,
        manifest_dir=manifest_dir,
        kresample_dir=smoke_base / "kresample",
        smoke_model_dir=base / "models",
    )
    # stage: same body, Hub boundary autospec'd (signature-conformant fakes)
    fake_prefix = create_autospec(hub.stage_hub_prefix, return_value=[])
    fake_file = create_autospec(hub.stage_hub_file, return_value=Path("/dev/null"))
    fake_sae = create_autospec(SAEMOD.BatchTopKSAE.ensure_downloaded, return_value=None)
    phase_stage(cfg, stage_prefix_fn=fake_prefix, stage_file_fn=fake_file, ensure_sae_fn=fake_sae)
    assert fake_prefix.call_count == 3 and fake_file.call_count == 2, (
        fake_prefix.call_count,
        fake_file.call_count,
    )
    for call in fake_prefix.call_args_list:
        assert call.kwargs["revision"] == cfg.revision, call
    phase_build(cfg)
    for space in SPACES:
        got = sorted(p.name for p in (cfg.space_dir(space) / "percontext").glob("*.npz"))
        assert got == [f"{a}_L{LAYER}_ridge.npz" for a in sorted(ARMS)], (space, got)
    phase_floors(cfg)
    assert (cfg.out_eval / "floors_env_mean" / "kresample" / f"floors_L{LAYER}.npz").is_file()
    phase_taxonomy(cfg)
    _smoke_dense_root(cfg, cfg.dense_eval)
    phase_compare(cfg)
    comp = json.loads((cfg.out_eval / "crossspace_comparison.json").read_text())
    assert comp["verdict"] in ("Reproduced", "Inverted", "Uncorrelated", "Mixed")
    assert comp["smoke"] is True
    # dense_feat_space vs itself must be a perfect rank match (self-comparison)
    assert abs(comp["robustness_spaces"]["dense_feat_space"]["rho_M"] - 1.0) < 1e-9
    phase_figures(cfg)
    fake_upload = create_autospec(hub._upload_folder_filtered, return_value="https://ok")
    phase_upload(cfg, upload_fn=fake_upload)
    assert fake_upload.call_count == len(SPACES) + 1, fake_upload.call_count
    for call in fake_upload.call_args_list:
        assert call.kwargs["path_in_repo"].startswith(f"{cfg.upload_prefix}/analysis_tensors")
        assert call.kwargs["allow_patterns"], call
    # fail-loud URL branch (the production raise)
    try:
        phase_upload(cfg, upload_fn=create_autospec(hub._upload_folder_filtered, return_value=""))
        raise AssertionError("empty-URL upload did not raise")
    except RuntimeError as e:
        assert "returned no URL" in str(e)

    # ── degenerate gate probes (designed handling; each fires OUTSIDE the main leg) ──
    from dataclasses import replace

    # 1. RSS projection over cap -> designed artifact-routed halt (rc 21 + rss_halt.json)
    cfg_rss = replace(cfg, out_eval=smoke_base / "eval_rss", rss_cap_gb=1e-4)
    try:
        phase_build(cfg_rss)
        raise AssertionError("RSS halt did not fire")
    except SystemExit as e:
        assert e.code == RC_RSS_HALT, e.code
        assert (cfg_rss.sentinel_dir() / "rss_halt.json").is_file()
    # 2. banked-R^2 reproduction gate: a scaled pred16 copy must trip the 5e-3 assert
    poison = smoke_base / "pred16_poison"
    poison.mkdir(exist_ok=True)
    for p in Path(cfg.pred16_src).glob("*.npz"):
        shutil.copyfile(p, poison / p.name)
    with np.load(poison / "sae_prefix.npz") as z:
        np.savez(
            poison / "sae_prefix.npz",
            pred16=(z["pred16"].astype(np.float64) * 2.0).astype(np.float16),
            ci=z["ci"],
            fingerprint=z["fingerprint"],
        )
    cfg_r2 = replace(cfg, out_eval=smoke_base / "eval_r2", pred16_src=poison)
    try:
        phase_build(cfg_r2)
        raise AssertionError("R2 identity gate did not fire")
    except AssertionError as e:
        assert "deviates from banked" in str(e), e
    # 3. fingerprint gate: a foreign fingerprint must trip the string-equality assert
    poison_fp = smoke_base / "pred16_poison_fp"
    poison_fp.mkdir(exist_ok=True)
    for p in Path(cfg.pred16_src).glob("*.npz"):
        shutil.copyfile(p, poison_fp / p.name)
    with np.load(poison_fp / "sae_bare.npz") as z:
        np.savez(
            poison_fp / "sae_bare.npz", pred16=z["pred16"], ci=z["ci"], fingerprint=np.array("dead")
        )
    cfg_fp = replace(cfg, out_eval=smoke_base / "eval_fp", pred16_src=poison_fp)
    try:
        phase_build(cfg_fp)
        # sentinel deliberately avoids the checked substring (no self-match)
        raise AssertionError("fp-gate probe did not trip")
    except AssertionError as e:
        assert "fingerprint" in str(e), e
    # 4. split-sha cross-assert: a dead banked split_shas must trip in _scan_and_gate
    bad_fits = smoke_base / "sae_fits_bad.json"
    doc = json.loads(Path(cfg.sae_fits_path).read_text())
    doc["split_shas"] = dict.fromkeys(doc["split_shas"], "dead")
    bad_fits.write_text(json.dumps(doc))
    cfg_sha = replace(cfg, out_eval=smoke_base / "eval_sha", sae_fits_path=bad_fits)
    try:
        phase_build(cfg_sha)
        # sentinel deliberately avoids the checked substring (no self-match)
        raise AssertionError("sha probe did not trip")
    except AssertionError as e:
        assert "split sha cross-assert" in str(e), e
    print("[smoke] degenerate gate probes OK (rss-halt/r2/fingerprint/split-sha)", flush=True)

    # production-input path arithmetic (the branch the smoke Cfg bypasses)
    pc = _resolve_production_inputs(
        replace(cfg, staging_root=Path("/probe"), dense_eval=Path("/dense"))
    )
    assert pc.local_sae_dir == Path("/probe") / PARENT_PREFIX / "sae_arm" / "capture"
    assert (
        pc.pred16_src
        == Path("/probe") / PARENT_PREFIX / "sae_arm_bare" / ("analysis_tensors") / "pred16"
    )
    assert pc.split_file == Path("/probe") / PARENT_PREFIX / "sampling_manifest" / (
        "split_1738.json"
    )
    assert pc.sae_fits_path == Path("/dense") / "bare_query" / "sae_arm" / "sae_fits.json"
    print(f"[smoke] PASS in {time.time() - t0:.0f}s — artifacts under {smoke_base}", flush=True)
    return 0


# ── main ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #1946 per-context SAE-space taxonomy.")
    ap.add_argument(
        "--phase",
        choices=["stage", "build", "floors", "taxonomy", "compare", "figures", "upload", "all"],
        default="all",
    )
    ap.add_argument(
        "--staging-root",
        default=os.path.expandvars("/mnt/eps-data/$USER/issue1946_saepc"),
        help="pinned-input staging root (data disk; plan section 9)",
    )
    ap.add_argument("--data-revision", default=DATA_REVISION_DEFAULT)
    ap.add_argument("--out-eval", default=str(PROJECT_ROOT / "eval_results" / "issue_1946"))
    ap.add_argument("--fig-dir", default=str(PROJECT_ROOT / "figures" / "issue_1946"))
    ap.add_argument(
        "--dense-eval",
        default=str(PROJECT_ROOT / "eval_results" / "issue_1738"),
        help="banked dense comparator root (git-resident)",
    )
    ap.add_argument("--upload-prefix", default=UPLOAD_PREFIX_DEFAULT)
    ap.add_argument("--rss-cap-gb", type=float, default=RSS_CAP_GB_DEFAULT)
    ap.add_argument(
        "--force",
        action="store_true",
        help="disable ALL resume-skips (y_holdout/percontext + floors/taxonomy/"
        "compare/figures entry guards); stage's content-addressed skip is unaffected",
    )
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny-real CPU e2e (PASS_UNIFIED)")
    ap.add_argument("--smoke-dir", default=str(PROJECT_ROOT / "data" / "issue_1946" / "smoke"))
    args = ap.parse_args()
    if args.smoke:
        rc = run_smoke(Path(args.smoke_dir))
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(rc)
    cfg = _resolve_production_inputs(
        Cfg(
            staging_root=Path(args.staging_root),
            out_eval=Path(args.out_eval),
            fig_dir=Path(args.fig_dir),
            dense_eval=Path(args.dense_eval),
            revision=args.data_revision,
            upload_prefix=args.upload_prefix,
            no_upload=args.no_upload,
            rss_cap_gb=args.rss_cap_gb,
            force=args.force,
        )
    )
    phases = {
        "stage": phase_stage,
        "build": phase_build,
        "floors": phase_floors,
        "taxonomy": phase_taxonomy,
        "compare": phase_compare,
        "figures": phase_figures,
        "upload": phase_upload,
    }
    order = list(phases) if args.phase == "all" else [args.phase]
    for name in order:
        phases[name](cfg)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # explicit exit — heavy C-ext modules (PyGILState atexit race, #1689)


if __name__ == "__main__":
    main()
