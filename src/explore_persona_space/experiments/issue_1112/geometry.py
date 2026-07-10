# ruff: noqa: RUF002
"""#1112 geometry aggregator (VM, CPU, batched — plan §4.5 / §6).

Consumes the per-(cell, dose) capture stores (``pooled.pt``, written by
``scripts/issue1112_dispatch.py``) + the base panels + the r_B directions and
produces:

- ``geometry_per_cell.json`` — one record per (cell, dose, layer, arm):
  spectral DVs (top_share_lambda / PR_λ / rank_k_at_90 via the #653
  ``spectral.py`` definitions verbatim), ‖μ‖, alignment cosines vs r_B with
  the norm-matched random CI, unique-row counts (prefix-arm degeneracy
  framing), and batched Gram-space cluster-bootstrap CIs. A cloud with < 2
  structurally-unique rows (MECHANICALLY-expected — e.g. a single-context
  prefix arm) yields an EXPLICIT ``degenerate: true`` record with null
  spectral DVs + a reason string instead of crashing on the #653 fail-fast;
  a ≥2-unique-row cloud that still zeroes out is UNEXPECTED and raises;
- cross-cell PAIRED difference records (H1/H2/H3/H5 pairs; the SAME
  resampled (context, question) indices applied to both cells —
  ``resampling: paired``), plus cross-cell cos(μ,μ)/cos(top,top)/CKA;
- the 80-row subsampled #653-comparability read (layer 14 / response);
- the split-half self-cosine attenuation ceiling (question-ALIGNED halves);
- per-draw × per-layer bootstrap DV matrices (persisted for the
  selection-symmetric-nulls re-reduction).

Everything here is deterministic linear algebra over persisted tensors —
0 GPU-h re-reduction of ``pooled.pt``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.analysis.representation_shift import linear_cka
from explore_persona_space.experiments.issue_653.spectral import (
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
from explore_persona_space.experiments.issue_1112 import (
    BOOT_SEED,
    CAPTURE_ARMS,
    N_BOOT,
    PRIMARY_LAYER,
    SUBSAMPLE_DRAWS,
    SUBSAMPLE_N,
)

logger = logging.getLogger("issue1112.geometry")

# Registered cross-cell pairs (plan §3): (name, cell_a, cell_b) — read at the
# SELECTED dose; H3 is the 4-cell interaction; H5 the marker pair.
DIFF_PAIRS = (
    ("H1_method_ftneg_vs_loraneg", "s3_fullft_neg", "s1_lora_neg"),
    ("H2_negatives_lorapos_vs_loraneg", "s2_lora_pos", "s1_lora_neg"),
    ("H2b_negatives_ftpos_vs_ftneg", "s4_fullft_pos", "s3_fullft_neg"),
    ("H5_marker_ft_vs_lora", "m2_fullft_band8", "m1_lora_band8"),
)


def load_store(path: Path) -> dict:
    """Load one capture store (schema asserted)."""
    store = torch.load(path, map_location="cpu", weights_only=False)
    assert store.get("schema_version") == 1, path
    for key in ("row_meta", "arms", "cell", "dose"):
        assert key in store, (path, key)
    return store


def _row_keys(store: dict) -> list[tuple[str, int]]:
    return [(m["context_id"], int(m["question_idx"])) for m in store["row_meta"]]


def delta_cloud(trained: dict, base: dict, arm: str, layer: int) -> np.ndarray:
    """Δx rows (trained − base) for one (arm, layer), rows paired by
    (context_id, question_idx) identity (asserted)."""
    kt, kb = _row_keys(trained), _row_keys(base)
    assert kt == kb, (
        f"row_meta mismatch between {trained['cell']}/{trained['dose']} and base — "
        "capture stores are not probe-aligned"
    )
    Xt = trained["arms"][arm][layer].to(torch.float32).numpy()
    Xb = base["arms"][arm][layer].to(torch.float32).numpy()
    assert Xt.shape == Xb.shape and Xt.ndim == 2, (Xt.shape, Xb.shape)
    return Xt - Xb


def structural_unique_rows(store: dict, arm: str) -> int:
    """Structurally-distinct rows per arm: prefix depends only on the context
    (n distinct contexts); context/response vary per (context, question)."""
    keys = _row_keys(store)
    if arm == "prefix":
        return len({c for c, _ in keys})
    return len(set(keys))


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def split_half_self_cosine(
    cloud: np.ndarray,
    question_idxs: list[int],
    *,
    n_partitions: int = 50,
    seed: int = 1112,
) -> dict:
    """Question-ALIGNED split-half cos(mean Δx half A, mean Δx half B).

    ONE half-partition of the QUESTION ids per draw, applied to all rows
    (llm-judging rule 21's aligned-split discipline transposed to the Δx mean);
    the caller applies the SAME seed across cells so partitions align
    cross-cell. Returns the mean + per-partition values.
    """
    qs = sorted(set(question_idxs))
    assert len(qs) >= 2, qs
    rng = np.random.default_rng(seed)
    q_arr = np.asarray(question_idxs)
    vals = []
    for _ in range(n_partitions):
        perm = rng.permutation(qs)
        half_a = set(perm[: len(qs) // 2].tolist())
        mask = np.asarray([q in half_a for q in q_arr])
        mu_a = cloud[mask].mean(axis=0)
        mu_b = cloud[~mask].mean(axis=0)
        vals.append(cosine(mu_a, mu_b))
    return {
        "mean": float(np.mean(vals)),
        "n_partitions": n_partitions,
        "values": [float(v) for v in vals],
        "scheme": "question-aligned halves, mean over partitions",
    }


def subsample_sensitivity(
    cloud: np.ndarray,
    *,
    n_sub: int = SUBSAMPLE_N,
    n_draws: int = SUBSAMPLE_DRAWS,
    seed: int = 1112,
) -> dict:
    """#653 cloud-size comparability: mean spectral DVs over random n_sub-row
    subsamples (WITHOUT replacement) of the cloud (layer 14 / response arm)."""
    n = cloud.shape[0]
    if n <= n_sub:
        return {"n_sub": n_sub, "note": f"cloud has only {n} rows — read is the full-cloud DV"}
    rng = np.random.default_rng(seed)
    acc: dict[str, list[float]] = {k: [] for k in BOOTSTRAPPABLE_DVS}
    for _ in range(n_draws):
        idx = rng.choice(n, size=n_sub, replace=False)
        dvs = spectral_dvs(svd_of_cloud(cloud[idx]))
        for k in acc:
            acc[k].append(float(dvs[k]))
    return {
        "n_sub": n_sub,
        "n_draws": n_draws,
        **{f"{k}_mean": float(np.mean(v)) for k, v in acc.items()},
        **{f"{k}_std": float(np.std(v)) for k, v in acc.items()},
    }


def analyze_cell(
    trained_store: dict,
    base_store: dict,
    *,
    layers: list[int],
    arms: tuple[str, ...] = CAPTURE_ARMS,
    rb: np.ndarray | None,
    idx_by_arm: dict[str, np.ndarray],
    boot_matrices: dict,
) -> dict:
    """All per-(layer, arm) geometry records for one (cell, dose) store.

    ``idx_by_arm`` carries the behavior-shared bootstrap index matrix (one per
    arm — identical across cells of the behavior, which is what makes every
    cross-cell difference PAIRED); per-draw DV matrices are appended to
    ``boot_matrices`` keyed ``(arm, layer, dv)``.
    """
    cell, dose = trained_store["cell"], trained_store["dose"]
    rand_ci_cache: dict[int, dict] = {}
    out: dict[str, dict] = {}
    for arm in arms:
        idx = idx_by_arm[arm]
        n_unique = structural_unique_rows(trained_store, arm)
        for layer in layers:
            cloud = delta_cloud(trained_store, base_store, arm, layer)
            mu = cloud.mean(axis=0)
            if n_unique < 2:
                # MECHANICALLY-expected degeneracy (plan §4.5 prefix-arm framing:
                # the prefix depends only on the context, so a 1-context capture
                # yields 1 unique prefix row): the row-centered Δx cloud is
                # identically zero, Σσ² == 0, and the #653 spectral DVs are
                # undefined. Emit an EXPLICIT degenerate record — the record IS
                # the signal; never a silent skip, never a coerced zero. μ (and
                # cos(μ, r_B)) stay well-defined and are still reported. A cloud
                # with ≥ 2 unique rows that still zeroes out is UNEXPECTED and
                # keeps raising via the #653 fail-fast on the normal path below.
                rec = {
                    "cell": cell,
                    "dose": dose,
                    "arm": arm,
                    "layer": layer,
                    "n_rows": int(cloud.shape[0]),
                    "n_unique_rows_structural": n_unique,
                    "degenerate": True,
                    "unique_rows": n_unique,
                    "degenerate_reason": (
                        f"{n_unique} structurally-unique row(s) < 2 for arm '{arm}' — "
                        "row-centered Δx cloud is identically zero (Σσ² == 0); "
                        "spectral DVs undefined at this capture size"
                    ),
                    "top_share_lambda": None,
                    "pr_lambda": None,
                    "rank_k_at_90": None,
                    "mu_norm": float(np.linalg.norm(mu)),
                    "boot_ci": None,
                    "n_boot": int(idx.shape[0]),
                    "resampling": "paired",
                }
                if rb is not None:
                    if layer not in rand_ci_cache:
                        rand_ci_cache[layer] = norm_matched_random_cos_ci(rb[layer], seed=layer)
                    rec["cos_top_to_rb"] = None
                    rec["cos_mu_to_rb"] = cosine(mu, rb[layer])
                    rec["random_cos_ci"] = rand_ci_cache[layer]
                logger.warning(
                    "[geometry] %s/%s %s/L%d: %d unique row(s) — explicit degenerate record",
                    cell,
                    dose,
                    arm,
                    layer,
                    n_unique,
                )
                out[f"{arm}/L{layer}"] = rec
                continue
            point = spectral_dvs(svd_of_cloud(cloud))
            top = top_direction(cloud)
            draws = batched_dvs_over_indices(cloud, idx, dv_names=BOOTSTRAPPABLE_DVS)
            for dv, vals in draws.items():
                boot_matrices[(arm, layer, dv)] = vals.astype(np.float32)
            rec = {
                "cell": cell,
                "dose": dose,
                "arm": arm,
                "layer": layer,
                "n_rows": int(cloud.shape[0]),
                "n_unique_rows_structural": n_unique,
                "degenerate": False,
                **{k: point[k] for k in ("top_share_lambda", "pr_lambda", "rank_k_at_90")},
                "mu_norm": float(np.linalg.norm(mu)),
                "boot_ci": {
                    dv: [
                        float(np.nanquantile(draws[dv], 0.025)),
                        float(np.nanquantile(draws[dv], 0.975)),
                    ]
                    for dv in draws
                },
                "n_boot": int(idx.shape[0]),
                "resampling": "paired",
            }
            if rb is not None:
                if layer not in rand_ci_cache:
                    rand_ci_cache[layer] = norm_matched_random_cos_ci(rb[layer], seed=layer)
                rec["cos_top_to_rb"] = cosine(top, rb[layer])
                rec["cos_mu_to_rb"] = cosine(mu, rb[layer])
                rec["random_cos_ci"] = rand_ci_cache[layer]
            out[f"{arm}/L{layer}"] = rec
    return out


def paired_diff_record(
    draws_a: np.ndarray,
    draws_b: np.ndarray,
    point_a: float,
    point_b: float,
    *,
    alpha: float = 0.05,
) -> dict:
    """Percentile CI on the PAIRED per-draw difference (same index draws)."""
    assert draws_a.shape == draws_b.shape, (draws_a.shape, draws_b.shape)
    d = draws_a - draws_b
    return {
        "point": float(point_a - point_b),
        "ci_low": float(np.nanquantile(d, alpha / 2)),
        "ci_high": float(np.nanquantile(d, 1 - alpha / 2)),
        "n_boot": int(d.shape[0]),
        "resampling": "paired",
    }


def _pair_read(
    store_a: dict,
    store_b: dict,
    base: dict,
    arm: str,
    layer: int,
    *,
    rec_a: dict,
    rec_b: dict,
    draws_a: dict,
    draws_b: dict,
) -> dict:
    """One cross-cell (arm, layer) diff read: direction cosines + paired DV CIs.

    A degenerate side (< 2 unique rows — no per-draw matrices exist) yields an
    explicit ``degenerate: true`` entry with the still-well-defined mean-shift
    cosine, never a KeyError or a coerced zero.
    """
    cloud_a = delta_cloud(store_a, base, arm, layer)
    cloud_b = delta_cloud(store_b, base, arm, layer)
    mu_a, mu_b = cloud_a.mean(axis=0), cloud_b.mean(axis=0)
    if rec_a.get("degenerate") or rec_b.get("degenerate"):
        return {
            "degenerate": True,
            "degenerate_sides": [s for s, r in (("a", rec_a), ("b", rec_b)) if r.get("degenerate")],
            "cos_mu": cosine(mu_a, mu_b),
        }
    entry = {
        "cos_mu": cosine(mu_a, mu_b),
        "cos_top": cosine(top_direction(cloud_a), top_direction(cloud_b)),
        "cka": linear_cka(torch.from_numpy(cloud_a), torch.from_numpy(cloud_b)),
    }
    for dv in BOOTSTRAPPABLE_DVS:
        entry[f"diff_{dv}"] = paired_diff_record(
            draws_a[(arm, layer, dv)],
            draws_b[(arm, layer, dv)],
            rec_a[dv],
            rec_b[dv],
        )
    return entry


def run_geometry(
    capture_root: Path,
    out_dir: Path,
    *,
    cells_doses: list[tuple[str, str]],
    base_store_by_behavior: dict[str, Path],
    behavior_by_cell: dict[str, str],
    selected_dose_by_cell: dict[str, str],
    rb_by_behavior: dict[str, Path],
    layers: list[int] | None = None,
    n_boot: int = N_BOOT,
    tensors_out: Path | None = None,
) -> dict:
    """The full #1112 geometry pass. Returns the geometry_per_cell payload.

    Args:
        capture_root: dir holding ``<cell>/<dose>/pooled.pt`` stores.
        cells_doses: realized (cell, dose) capture list.
        base_store_by_behavior: behavior -> base panel pooled.pt path.
        behavior_by_cell: cell -> "sycophancy" | "marker".
        selected_dose_by_cell: cell -> the matched-install dose label (the
            dose the cross-cell H reads use).
        rb_by_behavior: behavior -> r_B ``.pt`` path ((n_layers, hidden), or a
            dict with key ``rb``).
        tensors_out: where the per-draw bootstrap matrices land
            (``bootstrap_matrices/<cell>_<dose>.pt``).
    """
    assert_exemplar_calibration()  # #653 threshold calibration guard (plan §5)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors_out = tensors_out or (out_dir / "bootstrap_matrices")
    tensors_out.mkdir(parents=True, exist_ok=True)

    bases = {b: load_store(p) for b, p in base_store_by_behavior.items()}
    rbs: dict[str, np.ndarray] = {}
    for b, p in rb_by_behavior.items():
        obj = torch.load(p, map_location="cpu", weights_only=False)
        arr = obj["rb"] if isinstance(obj, dict) and "rb" in obj else obj
        rbs[b] = np.asarray(
            arr.to(torch.float32).numpy() if isinstance(arr, torch.Tensor) else arr,
            dtype=np.float64,
        )
        assert rbs[b].ndim == 2, (b, rbs[b].shape)

    # ONE bootstrap index matrix per (behavior, arm) — shared across every
    # cell/dose/layer of the behavior => all cross-cell differences PAIRED.
    idx_by_behavior_arm: dict[tuple[str, str], np.ndarray] = {}
    for b, base in bases.items():
        n_rows = len(base["row_meta"])
        cluster_ids = [f"{c}__{q}" for c, q in _row_keys(base)]
        for arm in CAPTURE_ARMS:
            idx_by_behavior_arm[(b, arm)] = bootstrap_index_matrix(
                cluster_ids, n_boot=n_boot, seed=BOOT_SEED
            )
            assert idx_by_behavior_arm[(b, arm)].shape == (n_boot, n_rows)

    layers_by_behavior = {
        b: (layers if layers is not None else sorted(next(iter(base["arms"].values())).keys()))
        for b, base in bases.items()
    }

    records: dict[str, dict] = {}
    per_cell_draws: dict[tuple[str, str], dict] = {}
    stores: dict[tuple[str, str], dict] = {}
    for cell, dose in cells_doses:
        b = behavior_by_cell[cell]
        store = load_store(capture_root / cell / dose / "pooled.pt")
        stores[(cell, dose)] = store
        boot_matrices: dict = {}
        recs = analyze_cell(
            store,
            bases[b],
            layers=layers_by_behavior[b],
            rb=rbs.get(b),
            idx_by_arm={arm: idx_by_behavior_arm[(b, arm)] for arm in CAPTURE_ARMS},
            boot_matrices=boot_matrices,
        )
        for key, rec in recs.items():
            records[f"{cell}/{dose}/{key}"] = rec
        per_cell_draws[(cell, dose)] = boot_matrices
        torch.save(
            {f"{arm}/L{layer}/{dv}": vals for (arm, layer, dv), vals in boot_matrices.items()},
            tensors_out / f"{cell}_{dose}.pt",
        )
        logger.info("[geometry] %s/%s: %d records", cell, dose, len(recs))

    # ── Cross-cell PAIRED differences + direction reads (selected doses) ─────
    diffs: dict[str, dict] = {}
    for name, cell_a, cell_b in DIFF_PAIRS:
        da, db = selected_dose_by_cell.get(cell_a), selected_dose_by_cell.get(cell_b)
        if (cell_a, da) not in stores or (cell_b, db) not in stores:
            diffs[name] = {"status": f"missing capture for {cell_a}/{da} or {cell_b}/{db}"}
            continue
        b = behavior_by_cell[cell_a]
        pair: dict[str, dict] = {}
        for arm in CAPTURE_ARMS:
            for layer in layers_by_behavior[b]:
                pair[f"{arm}/L{layer}"] = _pair_read(
                    stores[(cell_a, da)],
                    stores[(cell_b, db)],
                    bases[b],
                    arm,
                    layer,
                    rec_a=records[f"{cell_a}/{da}/{arm}/L{layer}"],
                    rec_b=records[f"{cell_b}/{db}/{arm}/L{layer}"],
                    draws_a=per_cell_draws[(cell_a, da)],
                    draws_b=per_cell_draws[(cell_b, db)],
                )
        diffs[name] = {"cell_a": cell_a, "cell_b": cell_b, "doses": [da, db], "reads": pair}

    # ── H3 interaction (exploratory, layer 14 / response, rank_k) ────────────
    h3 = {}
    quad = [
        ("LC", "s1_lora_neg"),
        ("LP", "s2_lora_pos"),
        ("FC", "s3_fullft_neg"),
        ("FP", "s4_fullft_pos"),
    ]
    quad_ok = all((c, selected_dose_by_cell.get(c)) in stores for _, c in quad)
    quad_layer_ok = quad_ok and all(
        PRIMARY_LAYER in layers_by_behavior[behavior_by_cell[c]] for _, c in quad
    )
    quad_layer_ok = quad_layer_ok and not any(
        records[f"{c}/{selected_dose_by_cell[c]}/response/L{PRIMARY_LAYER}"].get("degenerate")
        for _, c in quad
    )
    if quad_layer_ok:
        dv = "rank_k_at_90"
        arm, layer = "response", PRIMARY_LAYER
        dr = {
            tag: per_cell_draws[(c, selected_dose_by_cell[c])][(arm, layer, dv)] for tag, c in quad
        }
        pt = {tag: records[f"{c}/{selected_dose_by_cell[c]}/{arm}/L{layer}"][dv] for tag, c in quad}
        d = (dr["LC"] - dr["LP"]) - (dr["FC"] - dr["FP"])
        h3 = {
            "definition": "(rankK_LC - rankK_LP) - (rankK_FC - rankK_FP), layer 14 response",
            "point": float((pt["LC"] - pt["LP"]) - (pt["FC"] - pt["FP"])),
            "ci_low": float(np.nanquantile(d, 0.025)),
            "ci_high": float(np.nanquantile(d, 0.975)),
            "resampling": "paired",
        }

    # ── Sensitivity + attenuation-ceiling reads (0-GPU re-reductions) ────────
    sensitivity: dict[str, dict] = {}
    ceilings: dict[str, dict] = {}
    for (cell, dose), store in stores.items():
        if dose != selected_dose_by_cell.get(cell):
            continue
        b = behavior_by_cell[cell]
        if PRIMARY_LAYER not in layers_by_behavior[b]:
            continue
        cloud = delta_cloud(store, bases[b], "response", PRIMARY_LAYER)
        sensitivity[cell] = subsample_sensitivity(cloud)
        ceilings[cell] = split_half_self_cosine(
            cloud, [int(m["question_idx"]) for m in store["row_meta"]]
        )

    payload = {
        "schema_version": 1,
        "records": records,
        "cross_cell_diffs": diffs,
        "h3_interaction": h3,
        "subsample_sensitivity_80row": sensitivity,
        "split_half_self_cosine_ceiling": ceilings,
        "n_boot": n_boot,
        "boot_seed": BOOT_SEED,
        "resampling": "paired",
        "bootstrap_matrices_dir": str(tensors_out),
        "metadata": {
            "git_commit": _git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    out_path = out_dir / "geometry_per_cell.json"
    out_path.write_text(json.dumps(payload, indent=1) + "\n")
    logger.info(
        "[geometry] wrote %s (%d records, %d diff pairs)", out_path, len(records), len(diffs)
    )
    return payload
