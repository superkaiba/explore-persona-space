#!/usr/bin/env python
# ruff: noqa: RUF002
"""#1112 VM-side geometry driver (0 GPU-h; plan §4.5 / §9 "geometry + bootstrap (VM, CPU)").

Runs AFTER pod teardown against the persisted capture stores: discovers the
realized ``<cell>/<dose>/pooled.pt`` tree (never re-enumerates a registered
grid — the smoke's cell subset threads through by construction), loads the
r_B tensors, and calls ``experiments.issue_1112.geometry.run_geometry``
(batched Gram-space cluster bootstrap, paired cross-cell CIs, 80-row
sensitivity, split-half ceilings). Outputs:

- ``eval_results/issue_1112/geometry/geometry_per_cell.json`` (primary
  deliverable glob, plan §6.5)
- per-draw × per-layer bootstrap DV matrices under
  ``eval_results/issue_1112/geometry/bootstrap_matrices/`` (+ optional HF
  upload under ``issue1112_geometry2x2/analysis_tensors/bootstrap_matrices/``)

Input staging: ``--capture-root`` (a local tree, e.g. rsync'd from the pod or
the pod's own out_root) or ``--from-hf`` (scoped ``list_repo_tree`` on the
data-repo prefix + per-file ``hf_hub_download``, ≤6 workers — never
``snapshot_download`` against the ~1M-file data repo; gotchas.md).

Smoke (same code path, tiny knobs):
    uv run python scripts/issue1112_geometry.py --capture-root <tiny tree> \
        --rb-dir <dir with rb_*.pt> --out-dir /tmp/issue-1112-smoke/geometry \
        --n-boot 25

Amendment modes (same driver, flag-selected):
- ``--tf-shared`` (plan v6): shared-response geometry -> geometry_tf_shared.json
- ``--lr-matched`` (plan v8): the lr-matched paired-‖μ‖ read (s3_fullft_neg −
  s5_lora_neg_lr5e6, layer 14 response, mu_n_boot=2000 seed 653) ->
  geometry_lr_matched.json + install/<cell>_tier2.json materialization;
  ``--verify-inputs`` pre-flights every Hub input path.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments import issue_1112 as C  # noqa: E402
from explore_persona_space.experiments.issue_653.spectral import (  # noqa: E402
    BOOTSTRAPPABLE_DVS,
    assert_exemplar_calibration,
    batched_dvs_over_indices,
    bootstrap_index_matrix,
    cosine,
    norm_matched_random_cos_ci,
    spectral_dvs,
    svd_of_cloud,
    top_direction,
)
from explore_persona_space.experiments.issue_1112 import geometry as geo  # noqa: E402

logger = logging.getLogger("issue1112.geometry_driver")

BASE_CELLS = {"base_sycophancy": "sycophancy", "base_marker": "marker"}


def _behavior_for(cell: str) -> str:
    if cell in BASE_CELLS:
        return BASE_CELLS[cell]
    return "marker" if cell in C.MARKER_CELLS else "sycophancy"


def stage_from_hf(dest: Path, *, revision: str | None) -> None:
    """Stage ``analysis_tensors/{capture,rb}`` from the data repo (scoped)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    prefix = f"{C.DATA_PREFIX}/analysis_tensors"
    entries = [
        e.path
        for e in api.list_repo_tree(
            C.HF_DATA_REPO,
            path_in_repo=prefix,
            repo_type="dataset",
            recursive=True,
            revision=revision,
        )
        if getattr(e, "size", None) is not None
    ]
    if not entries:
        raise FileNotFoundError(f"no files under {C.HF_DATA_REPO}/{prefix}")

    def _fetch(path_in_repo: str) -> Path:
        last: Exception | None = None
        for attempt in range(4):
            try:
                got = hf_hub_download(
                    C.HF_DATA_REPO, path_in_repo, repo_type="dataset", revision=revision
                )
                break
            except Exception as e:  # bounded retry, linear backoff (gotchas.md)
                last = e
                time.sleep(20 * (attempt + 1))
        else:
            raise RuntimeError(f"hf_hub_download failed 4x for {path_in_repo}") from last
        rel = Path(path_in_repo).relative_to(prefix)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(got, target)
        return target

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_fetch, p) for p in entries]
        for f in as_completed(futures):
            f.result()  # fail loud on the first exhausted retry
    logger.info("[stage] %d files staged under %s", len(entries), dest)


def discover_passes(capture_root: Path) -> list[tuple[str, str]]:
    """Realized (cell, dose) list from the on-disk tree (fail-loud when empty)."""
    passes = sorted(
        (p.parent.parent.name, p.parent.name) for p in capture_root.glob("*/*/pooled.pt")
    )
    if not passes:
        raise FileNotFoundError(f"no <cell>/<dose>/pooled.pt stores under {capture_root}")
    return passes


# ── tf-shared amendment (followup `tf-shared-response-capture`, plan v6) ─────
# Same-token SHARED-response geometry: Δx_shared(row) = pooled_trained(shared
# base text) - pooled_base(shared text), response arm, all layers; paired
# cluster bootstrap vs the parent's OWN-text response clouds (same resample
# indices); registered 30/60 lattice at layer 14 encoded as DATA (no verdict
# prose — the analyzer reads it).

TF_LABEL = "tf-shared-response-capture"
TF_SYCO_CELLS = (
    "s1_lora_neg",
    "s2_lora_pos",
    "s3_fullft_neg",
    "s4_fullft_pos",
    "s5_lora_generic",
    "s6_fullft_generic",
)
BEHAVIOR_CELLS_2X2 = ("s1_lora_neg", "s2_lora_pos", "s3_fullft_neg", "s4_fullft_pos")
# Registered lattice (plan v6 §3): rank-k@90 <= 30 -> collapse; >= 60 ->
# stays_diffuse; else partial. Grounded in the parent's realized values
# (context arm max 13, own-response min 66 — brackets the empty middle).
LATTICE_COLLAPSE_MAX = 30.0
LATTICE_DIFFUSE_MIN = 60.0
PARITY_WARN_COS = 0.999  # WARN-level bar (plan §11: ungrounded numerically — never a HALT)
MU_N_BOOT = 2000  # the ‖μ‖ companion convention (plan v6 §6); rank/PR/top keep N_BOOT
PARENT_STORE_REV = "e016910195b7ab846c83b87ec43140c36c51e35f"  # parent-run upload (plan §10)


def _store_keys(store: dict) -> list[tuple[str, int]]:
    return [(m["context_id"], int(m["question_idx"])) for m in store["row_meta"]]


def _reorder_store(store: dict, key_order: list[tuple[str, int]]) -> dict:
    """Permute a capture store's rows into ``key_order`` (the base-store order).

    HARD set-equality assert first (plan §6 row-coverage / kill criterion 2);
    an order-only mismatch is re-paired by (context, question) keys (plan §8
    risk table). Raises AssertionError when the key SETS differ — inputs
    inconsistent with the parent stores, halt geometry.
    """
    keys = _store_keys(store)
    assert set(keys) == set(key_order) and len(keys) == len(key_order), (
        f"row_meta set mismatch for {store.get('cell')}/{store.get('dose')}: "
        f"{len(set(keys) ^ set(key_order))} differing keys — capture stores are "
        "not probe-aligned (kill criterion: halt, report)"
    )
    if keys == key_order:
        return store
    pos = {k: i for i, k in enumerate(keys)}
    perm = [pos[k] for k in key_order]
    out = dict(store)
    out["row_meta"] = [store["row_meta"][i] for i in perm]
    out["arms"] = {
        arm: {li: t[perm] for li, t in per_layer.items()}
        for arm, per_layer in store["arms"].items()
    }
    return out


def _draw_weight_matrix(idx: np.ndarray, n_rows: int) -> np.ndarray:
    """(n_boot, n_rows) multiplicity/m weight matrix: every draw's mean shift is
    one row of ``W @ cloud`` — the batched subset-sum GEMM (vectorize-first;
    no per-draw pool re-reduction)."""
    n_boot, m = idx.shape
    W = np.zeros((n_boot, n_rows), dtype=np.float64)
    np.add.at(W, (np.repeat(np.arange(n_boot), m), idx.ravel()), 1.0 / m)
    return W


def _mu_norm_draws(cloud: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Per-draw ‖mean Δx‖ over the cluster-bootstrap draws encoded in ``W``."""
    return np.linalg.norm(W @ np.asarray(cloud, dtype=np.float64), axis=1)


def _point_dvs(cloud: np.ndarray) -> dict[str, float]:
    dvs = spectral_dvs(svd_of_cloud(cloud))
    return {
        **{k: float(dvs[k]) for k in BOOTSTRAPPABLE_DVS},
        "mu_norm": float(np.linalg.norm(cloud.mean(axis=0))),
    }


def _boot_ci(draws: np.ndarray, alpha: float = 0.05) -> list[float]:
    return [float(np.nanquantile(draws, alpha / 2)), float(np.nanquantile(draws, 1 - alpha / 2))]


def _parity_check(tf_store: dict, own_store: dict, layers: list[int]) -> tuple[dict, dict]:
    """Free prefix/context parity read (plan §4): under causal attention those
    positions see only prompt tokens, identical across the two capture rounds,
    so per-row cosine(tf, parent) should be >= 0.999 up to bf16 batch-
    composition jitter. WARN-level ONLY (the reuse gate-severity rule: this
    check can only fail 'weaker than expected'); every value persisted.

    Returns (summary, full_per_row_tensors) — the caller persists the tensors.
    """
    summary: dict = {"warn_bar": PARITY_WARN_COS, "arms": {}}
    tensors: dict[str, np.ndarray] = {}
    for arm in ("prefix", "context"):
        per_layer: dict[str, dict] = {}
        worst = 1.0
        mean_acc: list[float] = []
        l14_rows: list[float] | None = None
        for layer in layers:
            a = tf_store["arms"][arm][layer].to(torch.float32)
            b = own_store["arms"][arm][layer].to(torch.float32)
            cos = ((a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)).numpy()
            tensors[f"{arm}/L{layer}"] = cos.astype(np.float32)
            per_layer[str(layer)] = {"min": float(cos.min()), "mean": float(cos.mean())}
            worst = min(worst, float(cos.min()))
            mean_acc.append(float(cos.mean()))
            if layer == C.PRIMARY_LAYER:
                l14_rows = [float(v) for v in cos]
        warn = worst < PARITY_WARN_COS
        summary["arms"][arm] = {
            "overall_min": worst,
            "overall_mean": float(np.mean(mean_acc)),
            "warn": warn,
            "per_layer": per_layer,
            "per_row_cos_primary_layer": l14_rows,
        }
    summary["warn"] = any(v["warn"] for v in summary["arms"].values())
    if summary["warn"]:
        summary["adjudication"] = (
            "prefix/context per-row cosine below 0.999 — bf16 batch-composition "
            "jitter vs pipeline drift is ANALYZER-adjudicated (plan §4: WARN-level, "
            "never a HALT; the headline DV does not depend on this read)"
        )
    return summary, tensors


def _lattice_block(records: dict, cells: list[str]) -> dict:
    """Registered-lattice DATA block (plan §3): per-cell branch + the mechanical
    >=3-of-4 headline call over the behavior 2x2 cells; generics reported
    alongside, never in the denominator. No interpretation prose."""

    def _branch(rank: float) -> str:
        if rank <= LATTICE_COLLAPSE_MAX:
            return "collapse"
        if rank >= LATTICE_DIFFUSE_MIN:
            return "stays_diffuse"
        return "partial"

    per_cell: dict[str, dict] = {}
    for cell in cells:
        rec = records.get(f"{cell}/L{C.PRIMARY_LAYER}")
        if rec is None:
            per_cell[cell] = {
                "rank_k_at_90_shared": None,
                "branch": None,
                "reason": "primary layer absent from the captured stores",
                "in_headline_denominator": cell in BEHAVIOR_CELLS_2X2,
            }
            continue
        r = rec["shared"]["rank_k_at_90"]
        per_cell[cell] = {
            "rank_k_at_90_shared": r,
            "branch": _branch(r),
            "in_headline_denominator": cell in BEHAVIOR_CELLS_2X2,
        }
    branches = [
        per_cell[c]["branch"] for c in BEHAVIOR_CELLS_2X2 if per_cell.get(c, {}).get("branch")
    ]
    headline = None
    note = None
    if len(branches) >= 3:
        top, cnt = Counter(branches).most_common(1)[0]
        headline = top if cnt >= 3 else "partial"
    else:
        note = (
            f"only {len(branches)} of 4 behavior 2x2 cells realized — no headline "
            "branch call (plan kill criterion)"
        )
    return {
        "registered_thresholds": {
            "collapse_max": LATTICE_COLLAPSE_MAX,
            "stays_diffuse_min": LATTICE_DIFFUSE_MIN,
        },
        "layer": C.PRIMARY_LAYER,
        "arm": "response",
        "cloud": "full shared-text Δx cloud (rank-k@90)",
        "per_cell": per_cell,
        "headline_denominator": list(BEHAVIOR_CELLS_2X2),
        "headline_rule": ">=3 of the 4 behavior 2x2 cells share a branch, else partial",
        "headline_branch": headline,
        "headline_note": note,
    }


def stage_tf_shared_from_hf(
    dest: Path, *, tf_revision: str | None, parent_revision: str | None
) -> tuple[Path, Path, Path]:
    """Stage the realized capture_tf stores (scoped listing at ``tf_revision``)
    plus the parent trained/base stores + recomputed r_B (deterministic
    per-file downloads at the PINNED parent revision). Never a bare
    list_repo_files / snapshot_download against the ~1M-file data repo."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    tf_prefix = f"{C.DATA_PREFIX}/analysis_tensors/capture_tf"
    entries = [
        e.path
        for e in api.list_repo_tree(
            C.HF_DATA_REPO,
            path_in_repo=tf_prefix,
            repo_type="dataset",
            recursive=True,
            revision=tf_revision,
        )
        if getattr(e, "size", None) is not None
    ]
    tf_pooled = sorted(p for p in entries if p.endswith("/selected/pooled.pt"))
    if not tf_pooled:
        raise FileNotFoundError(f"no capture_tf stores under {C.HF_DATA_REPO}/{tf_prefix}")

    def _fetch(path_in_repo: str, target: Path, revision: str | None) -> None:
        last: Exception | None = None
        for attempt in range(4):
            try:
                got = hf_hub_download(
                    C.HF_DATA_REPO, path_in_repo, repo_type="dataset", revision=revision
                )
                break
            except Exception as e:  # bounded retry, linear backoff (gotchas.md)
                last = e
                time.sleep(20 * (attempt + 1))
        else:
            raise RuntimeError(f"hf_hub_download failed 4x for {path_in_repo}") from last
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(got, target)

    tf_root = dest / "capture_tf"
    cells: list[str] = []
    for p in tf_pooled:
        cell = Path(p).parent.parent.name
        if cell not in TF_SYCO_CELLS:
            logger.info("[tf-stage] skipping non-sycophancy tf store %s (out of lattice scope)", p)
            continue
        cells.append(cell)
        _fetch(p, tf_root / cell / "selected" / "pooled.pt", tf_revision)
    if not cells:
        raise FileNotFoundError("no sycophancy capture_tf stores realized on the Hub")
    parent_root = dest / "capture"
    for cell in cells:
        _fetch(
            f"{C.DATA_PREFIX}/analysis_tensors/capture/{cell}/selected/pooled.pt",
            parent_root / cell / "selected" / "pooled.pt",
            parent_revision,
        )
    _fetch(
        f"{C.DATA_PREFIX}/analysis_tensors/capture/base_sycophancy/base/pooled.pt",
        parent_root / "base_sycophancy" / "base" / "pooled.pt",
        parent_revision,
    )
    rb_dir = dest / "rb"
    _fetch(
        f"{C.DATA_PREFIX}/analysis_tensors/rb/rb_sycophancy.pt",
        rb_dir / "rb_sycophancy.pt",
        parent_revision,
    )
    logger.info(
        "[tf-stage] %d tf + %d parent stores staged under %s", len(cells), len(cells) + 1, dest
    )
    return tf_root, parent_root, rb_dir


def run_tf_shared(
    tf_root: Path,
    parent_root: Path,
    rb_dir: Path,
    out_dir: Path,
    *,
    tensors_out: Path | None = None,
    n_boot: int = C.N_BOOT,
    mu_n_boot: int = MU_N_BOOT,
    inputs_provenance: dict | None = None,
) -> dict:
    """The tf-shared geometry pass (VM, CPU, batched). Writes
    ``geometry_tf_shared.json`` + per-draw matrices; returns the payload.

    Per cell (response arm, every captured layer): shared-text vs own-text
    spectral DVs with PAIRED cluster-bootstrap difference CIs (ONE index
    matrix from the base grid, n_boot @ seed 653; ‖μ‖ companion at
    ``mu_n_boot`` draws via the subset-sum GEMM), cos(μ_shared, r_B) +
    cos(top_shared, r_B) with the norm-matched random CI, the layer-14
    context-arm reference read, matched-80 subsample read, singular-value
    spectra at layer 14, and the prefix/context parity check (WARN-only).
    """
    assert_exemplar_calibration()  # #653 threshold calibration guard
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors_out = tensors_out or (out_dir / "bootstrap_matrices" / "tf_shared")
    tensors_out.mkdir(parents=True, exist_ok=True)

    base = geo.load_store(parent_root / "base_sycophancy" / "base" / "pooled.pt")
    base_keys = _store_keys(base)
    layers = sorted(next(iter(base["arms"].values())).keys())
    n_rows = len(base_keys)

    rb_obj = torch.load(rb_dir / "rb_sycophancy.pt", map_location="cpu", weights_only=False)
    rb_t = rb_obj["rb"] if isinstance(rb_obj, dict) and "rb" in rb_obj else rb_obj
    rb = np.asarray(rb_t.to(torch.float32).numpy(), dtype=np.float64)
    assert rb.ndim == 2, rb.shape

    cells = sorted(
        p.parent.parent.name
        for p in tf_root.glob("*/selected/pooled.pt")
        if p.parent.parent.name in TF_SYCO_CELLS
    )
    if not cells:
        raise FileNotFoundError(f"no sycophancy <cell>/selected/pooled.pt stores under {tf_root}")
    missing = sorted(set(TF_SYCO_CELLS) - set(cells))
    if missing:
        logger.warning(
            "[tf-geometry] missing tf captures for %s — reported, never backfilled", missing
        )

    # ONE bootstrap index matrix per draw budget, shared across every cell +
    # both clouds => all own-vs-shared and cross-cell reads are PAIRED.
    cluster_ids = [f"{c}__{q}" for c, q in base_keys]
    idx = bootstrap_index_matrix(cluster_ids, n_boot=n_boot, seed=C.BOOT_SEED)
    idx_mu = bootstrap_index_matrix(cluster_ids, n_boot=mu_n_boot, seed=C.BOOT_SEED)
    w_mu = _draw_weight_matrix(idx_mu, n_rows)

    records: dict[str, dict] = {}
    parity: dict[str, dict] = {}
    matched80: dict[str, dict] = {}
    sv_primary: dict[str, dict] = {}
    context_primary: dict[str, dict] = {}
    rand_ci_cache: dict[int, dict] = {}

    for cell in cells:
        tf_store = _reorder_store(
            geo.load_store(tf_root / cell / "selected" / "pooled.pt"), base_keys
        )
        cond = tf_store.get("metadata", {}).get("conditioning")
        assert cond == "tf_shared_base", (cell, cond)
        own_store = _reorder_store(
            geo.load_store(parent_root / cell / "selected" / "pooled.pt"), base_keys
        )
        cell_tensors: dict[str, np.ndarray] = {}
        for layer in layers:
            shared = geo.delta_cloud(tf_store, base, "response", layer)
            own = geo.delta_cloud(own_store, base, "response", layer)
            point_s, point_o = _point_dvs(shared), _point_dvs(own)
            draws_s = batched_dvs_over_indices(shared, idx, dv_names=BOOTSTRAPPABLE_DVS)
            draws_o = batched_dvs_over_indices(own, idx, dv_names=BOOTSTRAPPABLE_DVS)
            mu_s = _mu_norm_draws(shared, w_mu)
            mu_o = _mu_norm_draws(own, w_mu)
            for dv in BOOTSTRAPPABLE_DVS:
                cell_tensors[f"response/L{layer}/{dv}/shared"] = draws_s[dv].astype(np.float32)
                cell_tensors[f"response/L{layer}/{dv}/own"] = draws_o[dv].astype(np.float32)
            cell_tensors[f"response/L{layer}/mu_norm/shared"] = mu_s.astype(np.float32)
            cell_tensors[f"response/L{layer}/mu_norm/own"] = mu_o.astype(np.float32)
            mu_vec_s = shared.mean(axis=0)
            if layer not in rand_ci_cache:
                rand_ci_cache[layer] = norm_matched_random_cos_ci(rb[layer], seed=layer)
            diff = {
                dv: geo.paired_diff_record(draws_o[dv], draws_s[dv], point_o[dv], point_s[dv])
                for dv in BOOTSTRAPPABLE_DVS
            }
            diff["mu_norm"] = geo.paired_diff_record(
                mu_o, mu_s, point_o["mu_norm"], point_s["mu_norm"]
            )
            records[f"{cell}/L{layer}"] = {
                "cell": cell,
                "dose": "selected",
                "arm": "response",
                "layer": layer,
                "n_rows": int(shared.shape[0]),
                "conditioning": "tf_shared_base",
                "shared": {
                    **point_s,
                    "boot_ci": {dv: _boot_ci(draws_s[dv]) for dv in BOOTSTRAPPABLE_DVS},
                    "mu_norm_boot_ci": _boot_ci(mu_s),
                },
                "own": {
                    **point_o,
                    "boot_ci": {dv: _boot_ci(draws_o[dv]) for dv in BOOTSTRAPPABLE_DVS},
                    "mu_norm_boot_ci": _boot_ci(mu_o),
                },
                "diff_own_minus_shared": diff,
                "cos_mu_to_rb_shared": cosine(mu_vec_s, rb[layer]),
                "cos_top_to_rb_shared": cosine(top_direction(shared), rb[layer]),
                "cos_mu_to_rb_own": cosine(own.mean(axis=0), rb[layer]),
                "random_cos_ci": rand_ci_cache[layer],
                "n_boot": n_boot,
                "mu_n_boot": mu_n_boot,
                "resampling": "paired",
            }
            if layer == C.PRIMARY_LAYER:
                sv_primary[cell] = {
                    "own": [float(v) for v in svd_of_cloud(own)],
                    "shared": [float(v) for v in svd_of_cloud(shared)],
                }
                matched80[cell] = geo.subsample_sensitivity(shared)
                ctx = geo.delta_cloud(own_store, base, "context", layer)
                ctx_point = _point_dvs(ctx)
                ctx_draws = batched_dvs_over_indices(ctx, idx, dv_names=BOOTSTRAPPABLE_DVS)
                for dv in BOOTSTRAPPABLE_DVS:
                    cell_tensors[f"context/L{layer}/{dv}/own"] = ctx_draws[dv].astype(np.float32)
                context_primary[cell] = {
                    **ctx_point,
                    "boot_ci": {dv: _boot_ci(ctx_draws[dv]) for dv in BOOTSTRAPPABLE_DVS},
                }
        parity[cell], parity_tensors = _parity_check(tf_store, own_store, layers)
        for k, v in parity_tensors.items():
            cell_tensors[f"parity/{k}"] = v
        torch.save(
            cell_tensors,
            tensors_out / f"{cell}_tf_shared.pt",
        )
        logger.info(
            "[tf-geometry] %s: %d layer records (parity warn=%s)",
            cell,
            len(layers),
            parity[cell]["warn"],
        )

    payload = {
        "schema_version": 1,
        "followup_label": TF_LABEL,
        "conditioning": "tf_shared_base",
        "records": records,
        "lattice": _lattice_block(records, cells),
        "context_primary_layer": context_primary,
        "parity": parity,
        "matched80_shared": matched80,
        "sv_primary_layer": sv_primary,
        "cells_realized": cells,
        "cells_missing": missing,
        "n_boot": n_boot,
        "mu_n_boot": mu_n_boot,
        "boot_seed": C.BOOT_SEED,
        "resampling": "paired",
        "primary_layer": C.PRIMARY_LAYER,
        "bootstrap_matrices_dir": str(tensors_out),
        "inputs": inputs_provenance or {},
        "metadata": {
            "git_commit": geo._git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    out_path = out_dir / "geometry_tf_shared.json"
    out_path.write_text(json.dumps(payload, indent=1) + "\n")
    logger.info("[tf-geometry] wrote %s (%d records)", out_path, len(records))
    return payload


# ── lr-matched amendment (followup `lr-matched-method-pair`, plan v8) ────────
# Registered primary: paired ‖μ‖ difference (s3_fullft_neg - s5_lora_neg_lr5e6)
# at layer 14, response arm, full 120-row paired clouds, paired cluster
# bootstrap (mu_n_boot=2000, seed 653, IDENTICAL resample indices both cells).
# Secondaries: paired rank-k@90 / PR_λ / top-share diffs (n_boot=1000, the H1
# null re-check), cos(μ_s5, μ_s1) + cos(μ_s5, r_B) companions, matched-80 +
# split-half ceiling for the new cell. Branch lattice (plan §3) encoded as
# DATA only — the analyzer makes the call.

LR_LABEL = "lr-matched-method-pair"
LR_COMPARATOR = "s3_fullft_neg"  # existing full-FT+negatives cell @ 5e-6
LR_LORA_PARENT = "s1_lora_neg"  # the parent LoRA cell @ 1e-5 (reference + cos)
PANEL_N_CONTEXTS = 6  # the registered capture panel (plan §4 row-coverage)
PANEL_N_QUESTIONS = 20
LR_SELECTION_FILES = ("selection.json", "ladder.json", "build_result.json")


def _fetch_one(path_in_repo: str, target: Path, revision: str | None) -> Path:
    """Per-file hf_hub_download with bounded retry + linear backoff; skips an
    already-staged target (pinned-revision files are immutable)."""
    from huggingface_hub import hf_hub_download

    if target.exists():
        return target
    last: Exception | None = None
    for attempt in range(4):
        try:
            got = hf_hub_download(
                C.HF_DATA_REPO, path_in_repo, repo_type="dataset", revision=revision
            )
            break
        except Exception as e:  # bounded retry, linear backoff (gotchas.md)
            last = e
            time.sleep(20 * (attempt + 1))
    else:
        raise RuntimeError(f"hf_hub_download failed 4x for {path_in_repo}") from last
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(got, target)
    return target


def _assert_registered_panel(
    keys: list[tuple[str, int]], *, n_contexts: int, n_questions: int, what: str
) -> None:
    """HARD row-meta assert against the registered panel (plan §4 row-coverage
    / assumption 6): the realized (context_id, question_idx) keys must be the
    FULL n_contexts × n_questions cross product — raise loud on any mismatch
    (the plan's halt condition), never coerce."""
    contexts = sorted({c for c, _ in keys})
    expect_q = set(range(n_questions))
    assert len(keys) == len(set(keys)) == n_contexts * n_questions, (
        f"{what}: {len(keys)} rows ({len(set(keys))} unique) != registered "
        f"{n_contexts}x{n_questions} panel — halt, report (kill criterion 2)"
    )
    assert len(contexts) == n_contexts, (what, contexts)
    for ctx in contexts:
        got_q = {q for c, q in keys if c == ctx}
        assert got_q == expect_q, (
            f"{what}: context {ctx} question idxs {sorted(got_q)} != "
            f"0..{n_questions - 1} — capture panel is not the registered grid"
        )


def _mechanical_branch(diff: dict | None, in_band: bool | None) -> str | None:
    """Plan §3 verdict-lattice branch as DATA (no interpretation prose).

    diff = the registered primary paired_diff_record (s3 − s5). Branch (c)
    fires on in_band=False regardless of the CI (descriptive sub-band read
    only); (a)/(b')/(b) key on the CI sign per the registered table.
    """
    if in_band is False:
        return "c_never_entered_band_descriptive_only"
    if diff is None:
        return None
    if diff["ci_low"] > 0:
        return "a_gap_survives_fullft_larger"
    if diff["ci_high"] < 0:
        return "b_prime_gap_reverses_lora_larger"
    return "b_gap_closes_ci_includes_zero"


def verify_lr_matched_inputs(
    *, new_revision: str | None = None, parent_revision: str = PARENT_STORE_REV
) -> dict:
    """--verify-inputs pre-flight: resolve the CURRENT data-repo revision for
    the new cell's artifacts and ``file_exists``-check EVERY production input
    path at its pinned/resolved revision (single-path probes — never a listing
    of the ~1M-file repo; gotchas.md). Raises loud on any miss."""
    from huggingface_hub import HfApi

    api = HfApi()
    if new_revision is None:
        new_revision = api.repo_info(C.HF_DATA_REPO, repo_type="dataset").sha
    cell = C.LR_MATCHED_CELL
    checks: dict[str, str] = {
        f"{C.DATA_PREFIX}/analysis_tensors/capture/{cell}/selected/pooled.pt": new_revision,
        **{f"{C.DATA_PREFIX}/selection/{cell}/{n}": new_revision for n in LR_SELECTION_FILES},
        f"{C.DATA_PREFIX}/raw_completions/tier2/{cell}/tier2_rates.json": new_revision,
        f"{C.DATA_PREFIX}/analysis_tensors/capture/{LR_COMPARATOR}/selected/pooled.pt": (
            parent_revision
        ),
        f"{C.DATA_PREFIX}/analysis_tensors/capture/{LR_LORA_PARENT}/selected/pooled.pt": (
            parent_revision
        ),
        f"{C.DATA_PREFIX}/analysis_tensors/capture/base_sycophancy/base/pooled.pt": (
            parent_revision
        ),
        f"{C.DATA_PREFIX}/analysis_tensors/rb/rb_sycophancy.pt": parent_revision,
    }
    missing = [
        f"{p} @ {rev}"
        for p, rev in checks.items()
        if not api.file_exists(C.HF_DATA_REPO, p, repo_type="dataset", revision=rev)
    ]
    if missing:
        raise FileNotFoundError(f"lr-matched inputs missing on the Hub: {missing}")
    margin_present = api.file_exists(
        C.HF_DATA_REPO,
        f"{C.DATA_PREFIX}/margin/{cell}.json",
        repo_type="dataset",
        revision=new_revision,
    )
    logger.info(
        "[lr-verify] %d input paths resolve (new rev %s, parent rev %s); margin companion "
        "for %s present=%s",
        len(checks),
        new_revision,
        parent_revision,
        cell,
        margin_present,
    )
    return {
        "revisions": {"new_cell": new_revision, "parent": parent_revision},
        "n_paths": len(checks),
        "margin_companion_present": margin_present,
    }


def stage_lr_matched_from_hf(
    dest: Path, *, new_revision: str | None = None, parent_revision: str = PARENT_STORE_REV
) -> dict:
    """Stage the lr-matched round's inputs: the NEW cell's pooled store +
    selection/ladder/tier2 records at the CURRENT (resolved + recorded)
    revision, and the PARENT comparator/base/r_B artifacts at the PINNED
    parent revision. Scoped per-file downloads only."""
    from huggingface_hub import HfApi

    if new_revision is None:
        new_revision = HfApi().repo_info(C.HF_DATA_REPO, repo_type="dataset").sha
        logger.info("[lr-stage] resolved current data-repo revision %s", new_revision)
    cell = C.LR_MATCHED_CELL
    capture_root = dest / "capture"
    _fetch_one(
        f"{C.DATA_PREFIX}/analysis_tensors/capture/{cell}/selected/pooled.pt",
        capture_root / cell / "selected" / "pooled.pt",
        new_revision,
    )
    sel_dir = dest / "selection" / cell
    for name in LR_SELECTION_FILES:
        _fetch_one(f"{C.DATA_PREFIX}/selection/{cell}/{name}", sel_dir / name, new_revision)
    _fetch_one(
        f"{C.DATA_PREFIX}/raw_completions/tier2/{cell}/tier2_rates.json",
        sel_dir / "tier2_rates.json",
        new_revision,
    )
    for pcell in (LR_COMPARATOR, LR_LORA_PARENT):
        _fetch_one(
            f"{C.DATA_PREFIX}/analysis_tensors/capture/{pcell}/selected/pooled.pt",
            capture_root / pcell / "selected" / "pooled.pt",
            parent_revision,
        )
    _fetch_one(
        f"{C.DATA_PREFIX}/analysis_tensors/capture/base_sycophancy/base/pooled.pt",
        capture_root / "base_sycophancy" / "base" / "pooled.pt",
        parent_revision,
    )
    rb_dir = dest / "rb"
    _fetch_one(
        f"{C.DATA_PREFIX}/analysis_tensors/rb/rb_sycophancy.pt",
        rb_dir / "rb_sycophancy.pt",
        parent_revision,
    )
    logger.info("[lr-stage] staged new-cell + parent inputs under %s", dest)
    return {
        "capture_root": capture_root,
        "rb_dir": rb_dir,
        "selection_dir": dest / "selection",
        "revisions": {"new_cell": new_revision, "parent": parent_revision},
    }


def materialize_tier2_install(
    selection_dir: Path, install_dir: Path, *, cell: str = C.LR_MATCHED_CELL
) -> Path:
    """Materialize ``install/<cell>_tier2.json`` (the parent cells' committed
    shape) from the staged pod-side ``tier2_rates.json`` — shape-asserted,
    never synthesized."""
    src = selection_dir / cell / "tier2_rates.json"
    rec = json.loads(src.read_text())
    for key in ("cell", "step", "rates", "n"):
        assert key in rec, (str(src), key)
    for side in ("trained", "base"):
        assert side in rec["rates"], (str(src), side)
    assert rec["cell"] == cell, rec
    install_dir.mkdir(parents=True, exist_ok=True)
    out = install_dir / f"{cell}_tier2.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    logger.info("[lr-install] materialized %s", out)
    return out


def _lr_primary_layer_reads(
    stores: dict, base: dict, cells: tuple[str, ...], new_cell: str, *, layers, primary_layer
) -> tuple[dict, dict, dict]:
    """Matched-80 + split-half ceiling (new cell) and the layer-``primary_layer``
    singular-value spectra (all three cells) — empty dicts when the primary
    layer was not captured."""
    matched80: dict[str, dict] = {}
    ceilings: dict[str, dict] = {}
    sv_primary: dict[str, list[float]] = {}
    if primary_layer in layers:
        cloud5 = geo.delta_cloud(stores[new_cell], base, "response", primary_layer)
        matched80[new_cell] = geo.subsample_sensitivity(cloud5)
        ceilings[new_cell] = geo.split_half_self_cosine(
            cloud5, [int(m["question_idx"]) for m in stores[new_cell]["row_meta"]]
        )
        for cell in cells:
            sv_primary[cell] = [
                float(v)
                for v in svd_of_cloud(
                    geo.delta_cloud(stores[cell], base, "response", primary_layer)
                )
            ]
    return matched80, ceilings, sv_primary


def _lr_install_echo(selection_dir: Path | None, new_cell: str) -> tuple[dict, bool | None]:
    """Selection + tier2 echo from the staged records (self-containment for
    figures + the lattice's ``in_band``); (empty, None) when nothing staged."""
    install_block: dict = {}
    in_band: bool | None = None
    if selection_dir is not None:
        sel_path = selection_dir / new_cell / "selection.json"
        if sel_path.exists():
            install_block["selection"] = json.loads(sel_path.read_text())
            in_band = bool(install_block["selection"].get("in_band"))
        tier2_path = selection_dir / new_cell / "tier2_rates.json"
        if tier2_path.exists():
            install_block["tier2"] = json.loads(tier2_path.read_text())
    return install_block, in_band


def run_lr_matched(
    capture_root: Path,
    rb_dir: Path,
    out_dir: Path,
    *,
    selection_dir: Path | None = None,
    tensors_out: Path | None = None,
    n_boot: int = C.N_BOOT,
    mu_n_boot: int = MU_N_BOOT,
    n_contexts: int = PANEL_N_CONTEXTS,
    n_questions: int = PANEL_N_QUESTIONS,
    primary_layer: int = C.PRIMARY_LAYER,
    inputs_provenance: dict | None = None,
) -> dict:
    """The lr-matched geometry pass (VM, CPU, batched). Writes
    ``geometry_lr_matched.json`` + per-draw matrices; returns the payload.

    Reuses the parent machinery verbatim: ``geo.analyze_cell`` for the s5/s3
    per-(arm, layer) records (rank-k@90 / PR_λ / top-share / ‖μ‖ / cos-to-r_B,
    paired cluster bootstrap at ``n_boot`` seed 653), the ``_mu_norm_draws``
    subset-sum GEMM for the ‖μ‖ companion at ``mu_n_boot`` draws (ONE index
    matrix shared across cells ⇒ every cross-cell difference is paired), plus
    matched-80 / split-half / parity reads for the new cell.
    """
    assert_exemplar_calibration()  # #653 threshold calibration guard
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors_out = tensors_out or (out_dir / "bootstrap_matrices" / "lr_matched")
    tensors_out.mkdir(parents=True, exist_ok=True)

    base = geo.load_store(capture_root / "base_sycophancy" / "base" / "pooled.pt")
    base_keys = _store_keys(base)
    _assert_registered_panel(
        base_keys, n_contexts=n_contexts, n_questions=n_questions, what="base_sycophancy/base"
    )
    layers = sorted(next(iter(base["arms"].values())).keys())
    n_rows = len(base_keys)

    rb_obj = torch.load(rb_dir / "rb_sycophancy.pt", map_location="cpu", weights_only=False)
    rb_t = rb_obj["rb"] if isinstance(rb_obj, dict) and "rb" in rb_obj else rb_obj
    rb = np.asarray(rb_t.to(torch.float32).numpy(), dtype=np.float64)
    assert rb.ndim == 2, rb.shape

    new_cell = C.LR_MATCHED_CELL
    cells = (new_cell, LR_COMPARATOR, LR_LORA_PARENT)
    stores: dict[str, dict] = {}
    for cell in cells:
        store = geo.load_store(capture_root / cell / "selected" / "pooled.pt")
        cond = store.get("metadata", {}).get("conditioning")
        assert cond != "tf_shared_base", (cell, cond)  # own-text stores only
        # Row-meta HARD assert (plan §3 row-coverage): BOTH paired arms'
        # realized keys against the registered panel BEFORE any paired
        # statistic; _reorder_store then set-equality-asserts + re-pairs
        # against the base grid.
        _assert_registered_panel(
            _store_keys(store), n_contexts=n_contexts, n_questions=n_questions, what=cell
        )
        stores[cell] = _reorder_store(store, base_keys)

    # ONE bootstrap index matrix per draw budget, shared across every cell =>
    # all cross-cell differences PAIRED (identical resample indices).
    cluster_ids = [f"{c}__{q}" for c, q in base_keys]
    idx = bootstrap_index_matrix(cluster_ids, n_boot=n_boot, seed=C.BOOT_SEED)
    idx_mu = bootstrap_index_matrix(cluster_ids, n_boot=mu_n_boot, seed=C.BOOT_SEED)
    w_mu = _draw_weight_matrix(idx_mu, n_rows)

    records: dict[str, dict] = {}
    per_cell_draws: dict[str, dict] = {}
    mu_draws: dict[str, dict[int, np.ndarray]] = {c: {} for c in cells}
    mu_vec: dict[str, dict[int, np.ndarray]] = {c: {} for c in cells}
    parity_summary: dict | None = None

    for cell in cells:
        cell_tensors: dict[str, np.ndarray] = {}
        if cell in (new_cell, LR_COMPARATOR):
            boot_matrices: dict = {}
            recs = geo.analyze_cell(
                stores[cell],
                base,
                layers=layers,
                rb=rb,
                idx_by_arm={arm: idx for arm in C.CAPTURE_ARMS},
                boot_matrices=boot_matrices,
            )
            for key, rec in recs.items():
                records[f"{cell}/selected/{key}"] = rec
            per_cell_draws[cell] = boot_matrices
            for (arm, layer, dv), vals in boot_matrices.items():
                cell_tensors[f"{arm}/L{layer}/{dv}"] = vals.astype(np.float32)
        for layer in layers:  # ‖μ‖ companion draws (all three cells, response)
            cloud = geo.delta_cloud(stores[cell], base, "response", layer)
            mu_draws[cell][layer] = _mu_norm_draws(cloud, w_mu)
            mu_vec[cell][layer] = cloud.mean(axis=0)
            cell_tensors[f"response/L{layer}/mu_norm"] = mu_draws[cell][layer].astype(np.float32)
        if cell == new_cell:
            # Free pipeline check (plan §6, WARN-only at 0.999): the NEW cell's
            # prefix/context arms vs the PARENT base store's prompt positions.
            # NOTE the read folds the genuine LoRA-induced prompt-side shift
            # together with cross-round bf16 jitter — analyzer-adjudicated.
            parity_summary, parity_tensors = _parity_check(stores[cell], base, layers)
            for k, v in parity_tensors.items():
                cell_tensors[f"parity/{k}"] = v
        torch.save(cell_tensors, tensors_out / f"{cell}_selected.pt")
        logger.info("[lr-geometry] %s: draws persisted (%d layers)", cell, len(layers))

    def _mu_diff_by_layer(cell_a: str, cell_b: str) -> dict[str, dict]:
        return {
            str(layer): geo.paired_diff_record(
                mu_draws[cell_a][layer],
                mu_draws[cell_b][layer],
                float(np.linalg.norm(mu_vec[cell_a][layer])),
                float(np.linalg.norm(mu_vec[cell_b][layer])),
            )
            for layer in layers
        }

    mu_diff_s3_s5 = _mu_diff_by_layer(LR_COMPARATOR, new_cell)
    mu_diff_s3_s1 = _mu_diff_by_layer(LR_COMPARATOR, LR_LORA_PARENT)  # parent +3.24 reference
    mu_diff_s1_s5 = _mu_diff_by_layer(LR_LORA_PARENT, new_cell)  # lr effect at fixed method

    primary = mu_diff_s3_s5.get(str(primary_layer))

    secondary_by_layer: dict[str, dict] = {}
    for layer in layers:
        entry = {}
        for dv in BOOTSTRAPPABLE_DVS:
            entry[dv] = geo.paired_diff_record(
                per_cell_draws[LR_COMPARATOR][("response", layer, dv)],
                per_cell_draws[new_cell][("response", layer, dv)],
                records[f"{LR_COMPARATOR}/selected/response/L{layer}"][dv],
                records[f"{new_cell}/selected/response/L{layer}"][dv],
            )
        secondary_by_layer[str(layer)] = entry

    cos_companions = {
        str(layer): {
            "cos_mu_s5_s1": cosine(mu_vec[new_cell][layer], mu_vec[LR_LORA_PARENT][layer]),
            "cos_mu_s5_s3": cosine(mu_vec[new_cell][layer], mu_vec[LR_COMPARATOR][layer]),
            "cos_mu_s3_s1": cosine(mu_vec[LR_COMPARATOR][layer], mu_vec[LR_LORA_PARENT][layer]),
        }
        for layer in layers
    }

    matched80, ceilings, sv_primary = _lr_primary_layer_reads(
        stores, base, cells, new_cell, layers=layers, primary_layer=primary_layer
    )
    install_block, in_band = _lr_install_echo(selection_dir, new_cell)

    lattice = {
        "registered_branches": {
            "a_gap_survives_fullft_larger": "in band AND CI excludes zero, full-FT side larger",
            "b_gap_closes_ci_includes_zero": "in band AND CI includes zero",
            "b_prime_gap_reverses_lora_larger": "in band AND CI excludes zero, LoRA side larger",
            "c_never_entered_band_descriptive_only": "no rung with Tier-1 rate in [0.60, 0.85]",
        },
        "dv": "mu_norm",
        "layer": primary_layer,
        "arm": "response",
        "diff": "s3_fullft_neg - s5_lora_neg_lr5e6",
        "in_band": in_band,
        "mechanical_branch": _mechanical_branch(primary, in_band),
    }

    payload = {
        "schema_version": 1,
        "followup_label": LR_LABEL,
        "pair": {"cell_a": LR_COMPARATOR, "cell_b": new_cell, "arm": "response"},
        "records": records,
        "lr_matched_pair": {
            "primary": {
                "dv": "mu_norm",
                "layer": primary_layer,
                "arm": "response",
                "n_boot": mu_n_boot,
                "diff_s3_minus_s5": primary,
            },
            "mu_norm_diff_by_layer_s3_minus_s5": mu_diff_s3_s5,
            "reference_s3_minus_s1_by_layer": mu_diff_s3_s1,
            "exploratory_s1_minus_s5_by_layer": mu_diff_s1_s5,
            "secondary_diffs_s3_minus_s5_by_layer": secondary_by_layer,
            "cos_companions_by_layer": cos_companions,
        },
        "lattice": lattice,
        "install": install_block,
        "matched80": matched80,
        "split_half_self_cosine_ceiling": ceilings,
        "sv_primary_layer": sv_primary,
        "parity": {new_cell: parity_summary},
        "parity_note": (
            "prefix/context per-row cosine of the NEW cell's trained-model prompt positions "
            "vs the PARENT base store (plan §6 cross-round numeric-regime residual): the read "
            "folds the genuine LoRA-induced prompt-side shift together with cross-round bf16 "
            "jitter — WARN-only at 0.999, never a HALT; analyzer-adjudicated"
        ),
        "cells": list(cells),
        "n_boot": n_boot,
        "mu_n_boot": mu_n_boot,
        "boot_seed": C.BOOT_SEED,
        "resampling": "paired",
        "primary_layer": primary_layer,
        "panel": {"n_contexts": n_contexts, "n_questions": n_questions},
        "bootstrap_matrices_dir": str(tensors_out),
        "inputs": inputs_provenance or {},
        "metadata": {
            "git_commit": geo._git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    out_path = out_dir / "geometry_lr_matched.json"
    out_path.write_text(json.dumps(payload, indent=1) + "\n")
    logger.info(
        "[lr-geometry] wrote %s (%d records; mechanical branch: %s)",
        out_path,
        len(records),
        lattice["mechanical_branch"],
    )
    return payload


def _main_lr_matched(args: argparse.Namespace) -> int:
    if args.verify_inputs:
        info = verify_lr_matched_inputs(
            new_revision=args.new_revision, parent_revision=args.parent_revision
        )
        logger.info("[lr-verify] OK: %s", json.dumps(info))
        return 0
    margin_status: dict = {"present": None, "note": "local mode — Hub not consulted"}
    if args.from_hf:
        staged = stage_lr_matched_from_hf(
            args.stage_dir, new_revision=args.new_revision, parent_revision=args.parent_revision
        )
        capture_root, rb_dir = staged["capture_root"], staged["rb_dir"]
        selection_dir = staged["selection_dir"]
        revisions = staged["revisions"]
        from huggingface_hub import HfApi

        margin_path = f"{C.DATA_PREFIX}/margin/{C.LR_MATCHED_CELL}.json"
        margin_status = {
            "present": HfApi().file_exists(
                C.HF_DATA_REPO, margin_path, repo_type="dataset", revision=revisions["new_cell"]
            ),
            "hf_path": margin_path,
            "note": (
                "plan §6 fixed-pool margin companion for the new cell — the round's pod phase "
                "list excluded p7b_margin, so absence is a genuine coverage gap (needs "
                "weights + GPU; concern lr-matched-margin-companion-not-produced)"
            ),
        }
    else:
        if args.capture_root is None or args.rb_dir is None:
            raise SystemExit("--lr-matched with --capture-root requires --rb-dir")
        capture_root, rb_dir = args.capture_root, args.rb_dir
        selection_dir = args.selection_dir
        revisions = {"new_cell": None, "parent": None, "mode": "local"}
    inputs = {
        "capture_root": str(capture_root),
        "rb_dir": str(rb_dir),
        "selection_dir": str(selection_dir) if selection_dir else None,
        "hf_revisions": revisions,
        "margin_companion": margin_status,
    }
    payload = run_lr_matched(
        capture_root,
        rb_dir,
        args.out_dir,
        selection_dir=selection_dir,
        tensors_out=args.tensors_out,
        n_boot=args.n_boot,
        mu_n_boot=args.mu_n_boot,
        n_contexts=args.panel_contexts,
        n_questions=args.panel_questions,
        primary_layer=args.primary_layer,
        inputs_provenance=inputs,
    )
    if (
        selection_dir is not None
        and (selection_dir / C.LR_MATCHED_CELL / "tier2_rates.json").exists()
    ):
        materialize_tier2_install(selection_dir, args.install_dir)
    else:
        logger.warning(
            "[lr-install] no staged tier2_rates.json under %s — install record NOT "
            "materialized (local/smoke mode?)",
            selection_dir,
        )
    if args.upload:
        from explore_persona_space.orchestrate import hub

        url = hub._upload(
            args.out_dir / "geometry_lr_matched.json",
            C.HF_DATA_REPO,
            "dataset",
            f"{C.DATA_PREFIX}/geometry/geometry_lr_matched.json",
            upload_as_file=True,
        )
        if not str(url):
            raise RuntimeError("geometry_lr_matched.json upload returned no path")
        url = hub._upload(
            Path(payload["bootstrap_matrices_dir"]),
            C.HF_DATA_REPO,
            "dataset",
            f"{C.DATA_PREFIX}/analysis_tensors/bootstrap_matrices/lr_matched",
        )
        if not str(url):
            raise RuntimeError("lr_matched bootstrap matrices upload returned no path")
        logger.info("[lr-geometry] uploads complete")
    return 0


def _main_tf_shared(args: argparse.Namespace) -> int:
    if args.from_hf:
        tf_root, parent_root, rb_dir = stage_tf_shared_from_hf(
            args.stage_dir, tf_revision=args.revision, parent_revision=args.parent_revision
        )
    else:
        if args.tf_root is None or args.rb_dir is None:
            raise SystemExit("--tf-shared with --capture-root requires --tf-root and --rb-dir")
        tf_root, parent_root, rb_dir = args.tf_root, args.capture_root, args.rb_dir
    inputs = {
        "tf_stores": str(tf_root),
        "parent_stores": str(parent_root),
        "parent_revision": args.parent_revision if args.from_hf else None,
        "tf_revision": args.revision if args.from_hf else None,
    }
    payload = run_tf_shared(
        tf_root,
        parent_root,
        rb_dir,
        args.out_dir,
        tensors_out=args.tensors_out,
        n_boot=args.n_boot,
        mu_n_boot=args.mu_n_boot,
        inputs_provenance=inputs,
    )
    logger.info(
        "[tf-geometry] lattice headline (mechanical): %s",
        payload["lattice"].get("headline_branch"),
    )
    if args.upload:
        from explore_persona_space.orchestrate import hub

        url = hub._upload(
            args.out_dir / "geometry_tf_shared.json",
            C.HF_DATA_REPO,
            "dataset",
            f"{C.DATA_PREFIX}/geometry/geometry_tf_shared.json",
            upload_as_file=True,
        )
        if not str(url):
            raise RuntimeError("geometry_tf_shared.json upload returned no path")
        url = hub._upload(
            Path(payload["bootstrap_matrices_dir"]),
            C.HF_DATA_REPO,
            "dataset",
            f"{C.DATA_PREFIX}/analysis_tensors/bootstrap_matrices/tf_shared",
        )
        if not str(url):
            raise RuntimeError("tf_shared bootstrap matrices upload returned no path")
        logger.info("[tf-geometry] uploads complete")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(description="#1112 VM-side geometry pass (CPU, batched).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--capture-root", type=Path, help="local <cell>/<dose>/pooled.pt tree")
    src.add_argument(
        "--from-hf",
        action="store_true",
        help="stage analysis_tensors/{capture,rb} from the data repo (scoped listing)",
    )
    p.add_argument("--rb-dir", type=Path, default=None, help="dir holding rb_<behavior>.pt")
    p.add_argument("--revision", default=None, help="data-repo revision pin for --from-hf")
    p.add_argument(
        "--stage-dir",
        type=Path,
        default=Path(f"data/issue_{C.ISSUE}/geometry_stage"),
        help="--from-hf staging destination",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(f"eval_results/issue_{C.ISSUE}/geometry"),
        help="geometry_per_cell.json destination (smokes MUST divert to scratch)",
    )
    p.add_argument("--n-boot", type=int, default=C.N_BOOT)
    p.add_argument(
        "--upload",
        action="store_true",
        help="upload geometry JSON + bootstrap matrices to the data repo",
    )
    # ── tf-shared amendment mode (followup `tf-shared-response-capture`) ─────
    p.add_argument(
        "--tf-shared",
        action="store_true",
        help="run the shared-response geometry pass (plan v6) instead of the parent pass",
    )
    p.add_argument(
        "--tf-root",
        type=Path,
        default=None,
        help="[--tf-shared local mode] <cell>/selected/pooled.pt tree of capture_tf stores",
    )
    p.add_argument(
        "--parent-revision",
        default=PARENT_STORE_REV,
        help="[--tf-shared --from-hf] pinned data-repo revision for the PARENT stores + r_B",
    )
    p.add_argument("--mu-n-boot", type=int, default=MU_N_BOOT)
    p.add_argument(
        "--tensors-out",
        type=Path,
        default=None,
        help="[--tf-shared] per-draw matrix dir (default <out-dir>/bootstrap_matrices/tf_shared)",
    )
    # ── lr-matched amendment mode (followup `lr-matched-method-pair`) ────────
    p.add_argument(
        "--lr-matched",
        action="store_true",
        help="run the lr-matched paired-‖μ‖ pass (plan v8) instead of the parent pass",
    )
    p.add_argument(
        "--verify-inputs",
        action="store_true",
        help="[--lr-matched] pre-flight only: resolve revisions + file_exists every input path",
    )
    p.add_argument(
        "--new-revision",
        default=None,
        help="[--lr-matched --from-hf] data-repo revision for the NEW cell's artifacts "
        "(default: resolve + record the repo's CURRENT revision)",
    )
    p.add_argument(
        "--selection-dir",
        type=Path,
        default=None,
        help="[--lr-matched local mode] dir holding <cell>/{selection,tier2_rates}.json",
    )
    p.add_argument(
        "--install-dir",
        type=Path,
        default=Path(f"eval_results/issue_{C.ISSUE}/install"),
        help="[--lr-matched] install-record destination (smokes MUST divert to scratch)",
    )
    p.add_argument("--panel-contexts", type=int, default=PANEL_N_CONTEXTS)
    p.add_argument("--panel-questions", type=int, default=PANEL_N_QUESTIONS)
    p.add_argument("--primary-layer", type=int, default=C.PRIMARY_LAYER)
    args = p.parse_args(argv)

    if args.lr_matched:
        return _main_lr_matched(args)

    if args.tf_shared:
        return _main_tf_shared(args)

    if args.from_hf:
        stage_from_hf(args.stage_dir, revision=args.revision)
        capture_root = args.stage_dir / "capture"
        rb_dir = args.rb_dir or (args.stage_dir / "rb")
    else:
        capture_root = args.capture_root
        rb_dir = args.rb_dir
        if rb_dir is None:
            raise SystemExit("--rb-dir is required with --capture-root")

    passes = discover_passes(capture_root)
    cells_doses = [(c, d) for c, d in passes if c not in BASE_CELLS]
    behaviors = {_behavior_for(c) for c, _ in cells_doses}
    base_store_by_behavior: dict[str, Path] = {}
    for base_cell, behavior in BASE_CELLS.items():
        store = capture_root / base_cell / "base" / "pooled.pt"
        if behavior in behaviors:
            if not store.exists():
                raise FileNotFoundError(f"base panel missing for {behavior}: {store}")
            base_store_by_behavior[behavior] = store
    rb_by_behavior: dict[str, Path] = {}
    for behavior in behaviors:
        rb_path = rb_dir / f"rb_{behavior}.pt"
        if not rb_path.exists():
            raise FileNotFoundError(f"r_B tensor missing for {behavior}: {rb_path}")
        rb_by_behavior[behavior] = rb_path

    logger.info("[geometry] %d capture passes (%s)", len(cells_doses), sorted(behaviors))
    payload = geo.run_geometry(
        capture_root,
        args.out_dir,
        cells_doses=cells_doses,
        base_store_by_behavior=base_store_by_behavior,
        behavior_by_cell={c: _behavior_for(c) for c, _ in cells_doses},
        selected_dose_by_cell={c: "selected" for c, _ in cells_doses},
        rb_by_behavior=rb_by_behavior,
        n_boot=args.n_boot,
    )
    logger.info("[geometry] %d records written to %s", len(payload["records"]), args.out_dir)

    if args.upload:
        from explore_persona_space.orchestrate import hub

        url = hub._upload(
            args.out_dir / "geometry_per_cell.json",
            C.HF_DATA_REPO,
            "dataset",
            f"{C.DATA_PREFIX}/geometry/geometry_per_cell.json",
            upload_as_file=True,
        )
        if not str(url):
            raise RuntimeError("geometry_per_cell.json upload returned no path")
        url = hub._upload(
            args.out_dir / "bootstrap_matrices",
            C.HF_DATA_REPO,
            "dataset",
            f"{C.DATA_PREFIX}/analysis_tensors/bootstrap_matrices",
        )
        if not str(url):
            raise RuntimeError("bootstrap_matrices upload returned no path")
        logger.info("[geometry] uploads complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
