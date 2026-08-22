"""P3 reduction for issue #2222 — predictor arms, correlations, nulls, CIs.

NEW unit-2 file (plan v5 §4 P3, §6). Runs OFF-POD (cpu-bigmem) over the P1/P2
capture store (local ``data/issue_2222/capture/<ds>/`` first, HF
``issue2222_pvscreen/analysis_tensors/capture/<ds>/`` fallback) + the P0 staged
reused artifacts (``rb/``, ``maps/``, ``staging_ready.json``).

Stages (each checkpointed + fingerprint-keyed; ``--stage all`` runs in order):

- ``percell``   per-dataset per-row projections for every arm x 3 traits x 28
                layers (vectorized GEMMs; resume unit = dataset).
- ``aggregate`` dataset-level values -> Pearson r vs the #778 y-axis per trait:
                pre-registered steering layer (idx 19/19/15) + read-out LOFO
                layer sweep with selection-symmetric 10k permutation nulls
                (batched; per-draw x per-layer matrices persisted), paired
                bootstrap CIs on Delta-r (flat 24-dataset + family-clustered,
                frozen-at-layer AND selection-inherited), H1/H2/H3 verdict
                numbers, sample-level ROC/AUC (misaligned_2 vs normal),
                frozen-map quality (held-out R^2 + identity(+bias) baselines +
                kNN retrieval), hallucination y-axis drop fractions.
- ``tuned_map`` exploratory per-layer ridge ctxend -> raw_respavg via the #825
                dof-capped core (``heldout_r2_sweep``, inner-group-cv), LOFO
                over the 8 families; quarantined from the frozen-map headline.
- ``form_b``    exploratory dataset-level regression (n=24 << d): PCA<=10 comps
                + dof-capped ridge, LOFO; reports the ANGLE to r_B only
                (estimator-degenerate control, plan §5 / §10 item (l)).

CONTENT HYGIENE: dataset rows include harmful-content families — this module
only ever touches ACTIVATIONS, row ids, and counts; no row text is loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# load_dotenv BEFORE any heavy import (numpy below, torch lazily) so the #847
# shared-VM thread caps bind in-process (tests/test_shared_vm_thread_caps.py):
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:  # sibling-script imports in script mode (#823)
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2222_analysis as ana  # noqa: E402
import issue2222_lib as lib  # noqa: E402

N_LAYERS, DIM = lib.RB_SHAPE
# Plan §4/§10: paper layer 20 -> index 19 (evil, sycophancy); layer 16 -> 15.
STEER_IDX = {"evil": 19, "sycophancy": 19, "hallucination": 15}
# Published Qwen r values (plan §2; order matters for H3 kill criteria).
PUBLISHED_R = {
    "raw": {"evil": 0.784, "sycophancy": 0.540, "hallucination": 0.635},
    "exact_dp": {"evil": 0.946, "sycophancy": 0.879, "hallucination": 0.616},
    "prompt_dp": {"evil": 0.931, "sycophancy": 0.581, "hallucination": 0.689},
}
# Stand-in axis order in the percell proj array (axis 1). "raw" is the raw
# projection itself; the others are the s_i of ``predictor = raw_i - s_i``.
PROJ_KINDS = ("raw", "base", "ctxend", "mapped_ctx", "mapped_pfx")
DIFF_ARMS = {  # arm slug -> stand-in kind (plan §4 predictor table)
    "exact_dp": "base",
    "prompt_dp": "ctxend",
    "mapped_ctx": "mapped_ctx",
    "mapped_pfx": "mapped_pfx",
}
H3_TOLERANCE = 0.10  # plan §7 / §3 H3


def reduce_code_fingerprint() -> str:
    """Fingerprint of the unit-2 output-affecting reduce code."""
    return ana.files_fingerprint(
        [_SCRIPTS_DIR / "issue2222_analysis.py", _SCRIPTS_DIR / "issue2222_reduce.py"]
    )


# --- Staged-input loaders (P0 outputs; fail loud with the re-stage hint) ------


def load_staging_meta(data_root: Path) -> dict:
    """P0 staging manifest (carries the realized rb source, plan §10 fallback line)."""
    path = Path(data_root) / "staging_ready.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/issue2222_stage.py (P0) first")
    return json.loads(path.read_text())


def load_vhat(data_root: Path) -> tuple[np.ndarray, dict]:
    """(vhat (T, 28, 3584) float64 unit rows, meta incl. realized rb source)."""
    import torch

    meta = load_staging_meta(data_root)
    source = meta.get("rb_source") or meta.get("rb", {}).get("source")
    if not source:
        raise KeyError("staging_ready.json carries no rb source field")
    rb = np.stack(
        [
            torch.load(
                Path(data_root) / "rb" / source / f"{trait}.pt",
                map_location="cpu",
                weights_only=True,
            ).numpy()
            for trait in lib.TRAITS
        ]
    )
    assert rb.shape == (len(lib.TRAITS), N_LAYERS, DIM), rb.shape
    return ana.unit_normalize_rows(rb), {"rb_source": source}


def load_maps(data_root: Path) -> dict[str, dict[str, np.ndarray]]:
    """The two frozen #1739 maps: context_end + prefix_end (plan A7 shapes)."""
    maps_dir = Path(data_root) / "maps"
    return {
        "ctx": ana.load_frozen_map(maps_dir / "context_end__ufull.npz"),
        "pfx": ana.load_frozen_map(maps_dir / "prefix_end__ufull.npz"),
    }


def stage_capture_file(data_root: Path, ds: str, fname: str) -> Path:
    """Local capture file, staged from the HF data repo when absent (off-pod P3)."""
    local = lib.capture_dir(Path(data_root), ds) / fname
    if local.exists():
        return local
    from explore_persona_space.orchestrate import hub

    lib.log_phase("p3_stage", "fetching capture file from HF", dataset=ds, file=fname)
    return Path(
        hub.stage_hub_file(
            lib.HF_DATA_REPO,
            f"{lib.hf_capture_prefix(ds)}/{fname}",
            local,
            repo_type="dataset",
        )
    )


def load_capture_manifest(data_root: Path, ds: str) -> dict:
    """Per-dataset capture manifest (local first, HF fallback; fail loud)."""
    local = lib.capture_dir(Path(data_root), ds) / "manifest.json"
    if local.exists():
        return json.loads(local.read_text())
    manifest = lib.fetch_hub_manifest(ds)
    if manifest is None:
        raise FileNotFoundError(
            f"{ds}: no capture manifest locally or on HF — P1/P2 incomplete for this dataset"
        )
    return manifest


# --- y-axis (#778 finetune trait scores) ---------------------------------------


def load_y_axis(datasets: list[str]) -> dict:
    """{trait: {ds: {trait_score, drop_fraction, n_kept, n_total}}} from #778.

    Reads the git-committed ``eval_results/issue_778/finetune_<trait>_<ds>.json``
    (72 files; plan A8). Reports per-cell judge-draw drop fractions alongside
    (consistency WARN 3 — hallucination rides a drop-censored y-axis).
    """
    y_dir = lib.REPO_ROOT / "eval_results" / "issue_778"
    out: dict = {}
    for trait in lib.TRAITS:
        out[trait] = {}
        for ds in datasets:
            path = y_dir / f"finetune_{trait}_{ds}.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing — on a sparse clone add the cone: "
                    "`git sparse-checkout add eval_results/issue_778`"
                )
            d = json.loads(path.read_text())
            assert d["trait"] == trait and f"{d['family']}_{d['version']}" == ds, (path, ds)
            total = int(d.get("judge_draws_total") or 0)
            dropped = int(d.get("judge_draws_dropped") or 0)
            out[trait][ds] = {
                "trait_score": float(d["trait_score"]),
                "judge_draws_dropped": dropped,
                "judge_draws_total": total,
                "drop_fraction": (dropped / total) if total else None,
            }
    return out


# --- Stage: percell -------------------------------------------------------------


def percell_path(data_root: Path, ds: str) -> Path:
    return Path(data_root) / "reduce" / "percell" / f"{ds}.npz"


def _percell_key(manifest: dict, meta: dict, knn_layers: tuple[int, ...]) -> str:
    import hashlib

    payload = json.dumps(
        {
            "capture_fp": manifest.get("resume_fingerprint") or manifest.get("split_hash"),
            "reduce_code": reduce_code_fingerprint(),
            "rb_source": meta["rb_source"],
            "knn_layers": list(knn_layers),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def stage_percell(args) -> None:
    """Per-dataset projections + sufficient statistics (resume unit = dataset)."""
    data_root = Path(args.data_root)
    vhat, meta = load_vhat(data_root)
    maps = load_maps(data_root)
    vhat32 = vhat.astype(np.float32)
    knn_layers = tuple(args.knn_layers)
    datasets = lib.dataset_ids(args.datasets)
    for i, ds in enumerate(datasets):
        t0 = time.time()
        out_path = percell_path(data_root, ds)
        manifest = load_capture_manifest(data_root, ds)
        key = _percell_key(manifest, meta, knn_layers)
        sidecar = out_path.with_suffix(".meta.json")
        if out_path.exists() and sidecar.exists():
            if json.loads(sidecar.read_text()).get("key") == key:
                lib.log_phase(
                    "p3_percell", "skip (fresh)", dataset=ds, unit=f"{i + 1}/{len(datasets)}"
                )
                continue
            lib.log_phase("p3_percell", "stale checkpoint — recomputing", dataset=ds)
        summ_path = stage_capture_file(data_root, ds, "summaries.npz")
        base_path = stage_capture_file(data_root, ds, "base_respavg.npz")
        with np.load(summ_path) as z:
            raw = z["raw_respavg"]
            ctxend = z["ctxend"]
            pfxend = z["pfxend"]
            row_ids = z["row_ids"]
        with np.load(base_path) as z:
            base = z["base_respavg"]
            base_ids = z["row_ids"]
        # P3 joins on row_ids intersection (unit-1 interface contract).
        common, ia, ib = np.intersect1d(row_ids, base_ids, return_indices=True)
        if len(common) == 0:
            raise ValueError(f"{ds}: empty row_ids intersection between summaries and base gen")
        # Round-2 C6 fix: account the partial join (base rows are legitimately
        # dropped for empty/over-budget generations) and fail fast when the
        # intersection falls below 80% of the summary rows — a silent heavy drop
        # would bias every arm's dataset mean.
        n_only_summ = int(len(row_ids) - len(common))
        n_only_base = int(len(base_ids) - len(common))
        if len(common) < 0.8 * len(row_ids):
            raise ValueError(
                f"{ds}: percell join kept {len(common)}/{len(row_ids)} summary rows "
                f"(dropped {n_only_summ} summaries-only + {n_only_base} base-only) — "
                "below the 80% join floor; regenerate the base capture (--phase capture)"
            )
        raw, ctxend, pfxend, base = raw[ia], ctxend[ia], pfxend[ia], base[ib]
        n = len(common)
        assert raw.shape == (n, N_LAYERS, DIM), raw.shape

        proj = np.empty((n, len(PROJ_KINDS), len(lib.TRAITS), N_LAYERS), dtype=np.float32)
        for ki, arr in enumerate((raw, base, ctxend)):
            proj[:, ki] = np.einsum("nld,tld->ntl", arr.astype(np.float32), vhat32, optimize=True)
        proj[:, 3] = ana.map_project_via_u(ctxend, maps["ctx"], vhat)
        proj[:, 4] = ana.map_project_via_u(pfxend, maps["pfx"], vhat)

        # Per-layer sufficient statistics (id_bias algebra + map/identity R^2).
        sum_resid = np.empty((N_LAYERS, DIM), dtype=np.float64)
        ss_resid = np.empty(N_LAYERS, dtype=np.float64)
        sum_raw = np.empty((N_LAYERS, DIM), dtype=np.float64)
        ss_raw = np.empty(N_LAYERS, dtype=np.float64)
        ss_res_map = {"ctx": np.empty(N_LAYERS), "pfx": np.empty(N_LAYERS)}
        knn_store: dict[str, np.ndarray] = {
            k: np.empty((n, len(knn_layers), DIM), dtype=np.float16)
            for k in ("raw", "ctxend", "mapped_ctx", "mapped_pfx")
        }
        for layer in range(N_LAYERS):
            raw_l = raw[:, layer, :].astype(np.float32)
            ctx_l = ctxend[:, layer, :].astype(np.float32)
            resid = raw_l.astype(np.float64) - ctx_l.astype(np.float64)
            sum_resid[layer] = resid.sum(axis=0)
            ss_resid[layer] = float((resid**2).sum())
            sum_raw[layer] = raw_l.astype(np.float64).sum(axis=0)
            ss_raw[layer] = float((raw_l.astype(np.float64) ** 2).sum())
            for mk, src in (("ctx", ctx_l), ("pfx", pfxend[:, layer, :].astype(np.float32))):
                m = maps[mk]
                z = (src - m["x_mu"][layer].astype(np.float32)) / m["x_sd"][layer].astype(
                    np.float32
                )
                pred = z @ m["w"][layer].astype(np.float32) + m["y_mu"][layer].astype(np.float32)
                ss_res_map[mk][layer] = float(
                    ((pred.astype(np.float64) - raw_l.astype(np.float64)) ** 2).sum()
                )
                if layer in knn_layers:
                    kli = knn_layers.index(layer)
                    knn_store["mapped_ctx" if mk == "ctx" else "mapped_pfx"][:, kli] = pred.astype(
                        np.float16
                    )
            if layer in knn_layers:
                kli = knn_layers.index(layer)
                knn_store["raw"][:, kli] = raw_l.astype(np.float16)
                knn_store["ctxend"][:, kli] = ctx_l.astype(np.float16)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.stem + ".tmp.npz")  # np.savez suffix trap (#1092)
        np.savez(
            tmp,
            row_ids=common.astype(np.int64),
            proj=proj,
            sum_resid=sum_resid,
            ss_resid=ss_resid,
            sum_raw=sum_raw,
            ss_raw=ss_raw,
            ss_res_map_ctx=ss_res_map["ctx"],
            ss_res_map_pfx=ss_res_map["pfx"],
            knn_layers=np.asarray(knn_layers, dtype=np.int64),
            **{f"knn_{k}": v for k, v in knn_store.items()},
        )
        tmp.replace(out_path)
        lib.write_json_atomic(
            sidecar,
            {
                "key": key,
                "dataset": ds,
                "n_rows": int(n),
                "n_dropped_summaries_only": n_only_summ,
                "n_dropped_base_only": n_only_base,
                **lib.run_metadata(),
            },
        )
        lib.log_phase(
            "p3_percell",
            "done",
            dataset=ds,
            unit=f"{i + 1}/{len(datasets)}",
            n_rows=int(n),
            n_dropped_summaries_only=n_only_summ,
            n_dropped_base_only=n_only_base,
            elapsed_s=round(time.time() - t0, 1),
        )


# --- Stage: aggregate -----------------------------------------------------------


def _load_percell(
    data_root: Path, datasets: list[str], *, meta: dict, knn_layers: tuple[int, ...]
) -> dict:
    """Load per-dataset percell npzs, REFUSING stale checkpoints (round-2 C3 fix).

    A standalone ``--stage aggregate/form_b`` run must not silently reduce
    projections computed under a different capture fingerprint / rb source /
    reduce-code / knn-layer regime — the sidecar key is recomputed via
    ``_percell_key`` and verified per dataset (fail loud with the re-run hint).
    """
    cells = {}
    for ds in datasets:
        path = percell_path(data_root, ds)
        sidecar = path.with_suffix(".meta.json")
        if not path.exists() or not sidecar.exists():
            raise FileNotFoundError(
                f"{path} (or its .meta.json sidecar) missing — run --stage percell first"
            )
        key = _percell_key(load_capture_manifest(data_root, ds), meta, knn_layers)
        found = json.loads(sidecar.read_text()).get("key")
        if found != key:
            raise RuntimeError(
                f"{ds}: percell checkpoint key mismatch (sidecar {str(found)[:16]}… vs "
                f"expected {key[:16]}…) — stale projections for the current "
                "capture/rb/code/knn regime; re-run --stage percell first"
            )
        with np.load(path) as z:
            cells[ds] = {k: z[k] for k in z.files}
    return cells


def _family_index(datasets: list[str]) -> tuple[np.ndarray, list[str]]:
    families = sorted({lib.split_dataset_id(ds)[0] for ds in datasets})
    idx = np.array([families.index(lib.split_dataset_id(ds)[0]) for ds in datasets])
    return idx, families


def _dataset_arm_values(cells: dict, datasets: list[str]) -> tuple[np.ndarray, list[str]]:
    """(n_ds, n_arms, T, L) dataset-level predictor values + the arm order.

    raw arm = mean(raw proj); diff arms = mean(raw - standin); id_bias =
    mean(raw) - mean(ctxend) - b_family . vhat, with b the leave-one-family-out
    mean residual (computed by the caller and passed via the closure below).
    """
    arm_order = ["raw", *DIFF_ARMS.keys(), "id_bias"]
    n_ds = len(datasets)
    t, lw = len(lib.TRAITS), N_LAYERS
    vals = np.full((n_ds, len(arm_order), t, lw), np.nan)
    for di, ds in enumerate(datasets):
        proj = cells[ds]["proj"].astype(np.float64)  # (n, 5, T, L)
        mean_p = proj.mean(axis=0)  # (5, T, L)
        vals[di, 0] = mean_p[PROJ_KINDS.index("raw")]
        for ai, (_arm, kind) in enumerate(DIFF_ARMS.items(), start=1):
            vals[di, ai] = mean_p[PROJ_KINDS.index("raw")] - mean_p[PROJ_KINDS.index(kind)]
        # id_bias filled by the caller (needs cross-dataset fold bias).
    return vals, arm_order


def _fill_id_bias(
    vals: np.ndarray,
    arm_order: list[str],
    cells: dict,
    datasets: list[str],
    fam_idx: np.ndarray,
    vhat: np.ndarray,
) -> np.ndarray:
    """Fill the id_bias arm; returns b_fam (F, L, D) for the R^2 read."""
    n_fam = fam_idx.max() + 1
    sum_resid = np.zeros((n_fam, N_LAYERS, DIM))
    counts = np.zeros(n_fam)
    for di, ds in enumerate(datasets):
        sum_resid[fam_idx[di]] += cells[ds]["sum_resid"]
        counts[fam_idx[di]] += len(cells[ds]["row_ids"])
    b_fam = ana.leave_one_group_out_bias(sum_resid, counts)  # (F, L, D)
    b_proj = np.einsum("fld,tld->ftl", b_fam, vhat)  # (F, T, L)
    ai = arm_order.index("id_bias")
    for di, ds in enumerate(datasets):
        proj = cells[ds]["proj"].astype(np.float64)
        mean_p = proj.mean(axis=0)
        vals[di, ai] = (
            mean_p[PROJ_KINDS.index("raw")]
            - mean_p[PROJ_KINDS.index("ctxend")]
            - b_proj[fam_idx[di]]
        )
    return b_fam


def _verdict(delta: float, lo: float, hi: float) -> str:
    """Plan §3 lattice: Confirmed / Falsified / Inconclusive (disjoint, exhaustive)."""
    if delta > 0 and lo > 0:
        return "Confirmed"
    if hi < 0:
        return "Falsified"
    return "Inconclusive"


def stage_aggregate(args) -> None:
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    nulls_dir = data_root / "nulls"
    nulls_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "nulls").mkdir(parents=True, exist_ok=True)
    datasets = lib.dataset_ids(args.datasets)
    if len(datasets) < 24 and not args.allow_partial:
        raise ValueError(
            f"aggregate over {len(datasets)}/24 datasets needs --allow-partial (smoke only)"
        )
    vhat, meta = load_vhat(data_root)
    cells = _load_percell(data_root, datasets, meta=meta, knn_layers=tuple(args.knn_layers))
    fam_idx, families = _family_index(datasets)
    y_axis = load_y_axis(datasets)
    vals, arm_order = _dataset_arm_values(cells, datasets)
    b_fam = _fill_id_bias(vals, arm_order, cells, datasets, fam_idx, vhat)

    n_perms = args.n_perms
    n_boot = args.n_boot
    records: list[dict] = []
    nulls_summary: dict = {}
    boot_r: dict = {}  # (trait, arm, scheme) -> (B, L) r matrix
    for ti, trait in enumerate(lib.TRAITS):
        y = np.array([y_axis[trait][ds]["trait_score"] for ds in datasets])
        steer = STEER_IDX[trait]
        idx_flat = ana.boot_indices_flat(len(datasets), n_boot, seed=args.seed)
        idx_clu = ana.boot_indices_clustered(fam_idx, n_boot, seed=args.seed)
        for ai, arm in enumerate(arm_order):
            v = vals[:, ai, ti, :]  # (n_ds, L)
            r_layers = ana.pearson_r_cols(v, y)
            perm = ana.perm_null_abs_r(v, y, n_perms=n_perms, seed=args.seed)
            perm_max = perm.max(axis=1)
            obs_max = float(np.nanmax(np.abs(r_layers)))
            sweep = ana.lofo_layer_sweep(v, y, fam_idx)
            # fixed-layer (pre-registered) permutation p — symmetric trivially.
            p_steer = float((1 + (perm[:, steer] >= abs(r_layers[steer])).sum()) / (n_perms + 1))
            p_sweep = float((1 + (perm_max >= obs_max).sum()) / (n_perms + 1))
            np.savez(
                nulls_dir / f"perm_{trait}_{arm}.npz",
                abs_r=perm.astype(np.float32),
                observed_r_layers=r_layers,
            )
            for scheme, idx in (("flat", idx_flat), ("clustered", idx_clu)):
                rb_mat = ana.boot_r_matrix(v, y, idx)
                boot_r[(trait, arm, scheme)] = rb_mat
                np.savez(
                    nulls_dir / f"boot_{scheme}_{trait}_{arm}.npz",
                    r=rb_mat.astype(np.float32),
                )
            pub = PUBLISHED_R.get(arm, {}).get(trait)
            base_rec = {
                "trait": trait,
                "n_datasets": len(datasets),
                "arm": arm,
                "rb_source": meta["rb_source"],
                "y_drop_fraction_by_dataset": {
                    ds: y_axis[trait][ds]["drop_fraction"] for ds in datasets
                },
            }
            records.append(
                {
                    **base_rec,
                    "layer_regime": "steer",
                    "layer": steer,
                    "r": float(r_layers[steer]),
                    "perm_p_fixed_layer": p_steer,
                    "published_r": pub,
                    "delta_vs_published": (float(r_layers[steer]) - pub) if pub else None,
                    "within_h3_tolerance": (abs(float(r_layers[steer]) - pub) <= H3_TOLERANCE)
                    if pub
                    else None,
                }
            )
            records.append(
                {
                    **base_rec,
                    "layer_regime": "sweep",
                    "lofo_r": sweep["lofo_r"],
                    "selected_layer_by_fold": sweep["selected_layer_by_fold"],
                    "within_sample_max_abs_r": sweep["within_sample_max_abs_r"],
                    "within_sample_argmax_layer": sweep["within_sample_argmax_layer"],
                    "within_sample_r_at_argmax": sweep["within_sample_r_at_argmax"],
                    "perm_p_max_selected": p_sweep,
                    "null_band_p975_max_selected": float(np.quantile(perm_max, 0.975)),
                    "abs_r_ceiling": 1.0,
                    "r_per_layer": [float(x) for x in r_layers],
                }
            )
            nulls_summary[f"{trait}/{arm}"] = {
                "p975_max_selected": float(np.quantile(perm_max, 0.975)),
                "p975_fixed_steer": float(np.quantile(perm[:, steer], 0.975)),
                "observed_max_abs_r": obs_max,
                "n_perms": n_perms,
            }

    # --- Hypothesis tests (plan §3 lattice + §6 thresholds) ---
    def _delta_record(trait: str, arm_a: str, arm_b: str, shift: float = 0.0) -> dict:
        ti = lib.TRAITS.index(trait)
        steer = STEER_IDX[trait]
        ia, ib_ = arm_order.index(arm_a), arm_order.index(arm_b)
        r_a = float(
            ana.pearson_r_cols(
                vals[:, ia, ti, :], np.array([y_axis[trait][ds]["trait_score"] for ds in datasets])
            )[steer]
        )
        r_b = float(
            ana.pearson_r_cols(
                vals[:, ib_, ti, :],
                np.array([y_axis[trait][ds]["trait_score"] for ds in datasets]),
            )[steer]
        )
        out: dict = {
            "trait": trait,
            "arm_a": arm_a,
            "arm_b": arm_b,
            "layer": steer,
            "r_a": r_a,
            "r_b": r_b,
            "delta_r": r_a - r_b - shift,
            "shift": shift,
            "delta_ceiling": 1.0 - (r_b + shift),
        }
        for scheme in ("flat", "clustered"):
            da = boot_r[(trait, arm_a, scheme)]
            db = boot_r[(trait, arm_b, scheme)]
            frozen = da[:, steer] - db[:, steer] - shift
            inh, _, _ = ana.selection_inherited_delta(da, db)
            inh = inh - shift
            out[f"ci95_frozen_{scheme}"] = [
                float(np.nanquantile(frozen, 0.025)),
                float(np.nanquantile(frozen, 0.975)),
            ]
            out[f"ci95_selection_inherited_{scheme}"] = [
                float(np.nanquantile(inh, 0.025)),
                float(np.nanquantile(inh, 0.975)),
            ]
        lo, hi = out["ci95_frozen_flat"]
        out["verdict_flat"] = _verdict(out["delta_r"], lo, hi)
        lo_c, hi_c = out["ci95_frozen_clustered"]
        out["verdict_clustered"] = _verdict(out["delta_r"], lo_c, hi_c)
        return out

    h1 = _delta_record("sycophancy", "mapped_ctx", "prompt_dp")
    h2 = {t: _delta_record(t, "mapped_ctx", "exact_dp") for t in lib.TRAITS}
    # H3 / kill-criterion summary (plan §7): exact-ΔP sign + magnitude misses.
    h3_rows = [
        r for r in records if r.get("layer_regime") == "steer" and r.get("published_r") is not None
    ]
    exact_sign_misses = sum(
        1
        for r in h3_rows
        if r["arm"] == "exact_dp" and np.sign(r["r"]) != np.sign(r["published_r"])
    )
    # Plan §7 disjunct (b): ALL published arms miss the published magnitude by
    # > H3_TOLERANCE on EVERY trait (derived from the within_h3_tolerance rows).
    magnitude_miss_all_arms_by_trait = {
        t: all(not r["within_h3_tolerance"] for r in h3_rows if r["trait"] == t) for t in lib.TRAITS
    }
    kill_magnitude = bool(all(magnitude_miss_all_arms_by_trait.values()))
    hypothesis_tests = {
        "H1_sycophancy_gap": h1,
        "H2_equivalence": {
            t: {**h2[t], "passes_r_a_ge_r_b_minus_0p10": h2[t]["delta_r"] >= -0.10}
            for t in lib.TRAITS
        },
        "H3_reproduction": {
            "tolerance": H3_TOLERANCE,
            "rows": [
                {k: r[k] for k in ("trait", "arm", "r", "published_r", "within_h3_tolerance")}
                for r in h3_rows
            ],
            "exact_dp_sign_misses": int(exact_sign_misses),
            "kill_criterion_sign": exact_sign_misses >= 2,
            "magnitude_miss_all_published_arms_by_trait": magnitude_miss_all_arms_by_trait,
            "kill_criterion_magnitude": kill_magnitude,
        },
        "H4_prefix_arm_note": "run-and-report; expected degenerate (#1739)",
        "verdict_lattice": "Confirmed iff Δr>0 and 95% CI excludes 0 positive; "
        "Falsified iff CI wholly below 0; else Inconclusive (plan §3)",
    }

    # --- sample-level ROC/AUC (misaligned_2 vs normal; paper Fig. 9) ---
    auc_records = []
    versions = {ds: lib.split_dataset_id(ds)[1] for ds in datasets}
    keep = [ds for ds in datasets if versions[ds] in ("normal", "misaligned_2")]
    if keep:
        labels = np.concatenate(
            [np.full(len(cells[ds]["row_ids"]), versions[ds] == "misaligned_2") for ds in keep]
        )
        for ti, trait in enumerate(lib.TRAITS):
            steer = STEER_IDX[trait]
            arm_scores: dict[str, np.ndarray] = {}
            for ai, arm in enumerate(arm_order):
                per_row = []
                for ds in keep:
                    proj = cells[ds]["proj"].astype(np.float64)
                    if arm == "raw":
                        v = proj[:, PROJ_KINDS.index("raw"), ti, steer]
                    elif arm == "id_bias":
                        di = datasets.index(ds)
                        bp = float(np.einsum("d,d->", b_fam[fam_idx[di], steer], vhat[ti, steer]))
                        v = (
                            proj[:, PROJ_KINDS.index("raw"), ti, steer]
                            - proj[:, PROJ_KINDS.index("ctxend"), ti, steer]
                            - bp
                        )
                    else:
                        kind = DIFF_ARMS[arm]
                        v = (
                            proj[:, PROJ_KINDS.index("raw"), ti, steer]
                            - proj[:, PROJ_KINDS.index(kind), ti, steer]
                        )
                    per_row.append(v)
                scores = np.concatenate(per_row)
                arm_scores[arm] = scores
                auc_records.append(
                    {
                        "trait": trait,
                        "arm": arm,
                        "layer": steer,
                        "auc": ana.auc_mann_whitney(scores, labels),
                        "n_pos": int(labels.sum()),
                        "n_neg": int((~labels).sum()),
                    }
                )
            # Round-2 C8: persist the exact per-sample scores that fed the AUC so
            # the P5 ROC-curve figure (plan §6 item 3) renders from PERSISTED
            # files — figures never recompute arm scores from the percell npzs.
            tmp = nulls_dir / f".tmp_roc_scores_{trait}.npz"
            np.savez(
                tmp,
                labels=labels.astype(bool),
                steer_layer=np.int64(steer),
                **{f"score_{arm}": v.astype(np.float32) for arm, v in arm_scores.items()},
            )
            os.replace(tmp, nulls_dir / f"roc_scores_{trait}.npz")

    # --- frozen-map quality (R^2 per layer + identity(+bias) + kNN) ---
    n_tot = sum(len(cells[ds]["row_ids"]) for ds in datasets)
    sum_raw_tot = np.sum([cells[ds]["sum_raw"] for ds in datasets], axis=0)  # (L, D)
    ss_raw_tot = np.sum([cells[ds]["ss_raw"] for ds in datasets], axis=0)  # (L,)
    ss_tot = ss_raw_tot - (sum_raw_tot**2).sum(axis=1) / n_tot
    ss_identity = np.sum([cells[ds]["ss_resid"] for ds in datasets], axis=0)
    # id_bias SS_res via the per-family expansion (held-out b_f).
    ss_idbias = np.zeros(N_LAYERS)
    fam_sum_resid = np.zeros((fam_idx.max() + 1, N_LAYERS, DIM))
    fam_ss_resid = np.zeros((fam_idx.max() + 1, N_LAYERS))
    fam_counts = np.zeros(fam_idx.max() + 1)
    for di, ds in enumerate(datasets):
        fam_sum_resid[fam_idx[di]] += cells[ds]["sum_resid"]
        fam_ss_resid[fam_idx[di]] += cells[ds]["ss_resid"]
        fam_counts[fam_idx[di]] += len(cells[ds]["row_ids"])
    for f in range(len(fam_counts)):
        bf = b_fam[f]  # (L, D)
        ss_idbias += (
            fam_ss_resid[f]
            - 2 * (bf * fam_sum_resid[f]).sum(axis=1)
            + fam_counts[f] * (bf**2).sum(axis=1)
        )
    map_quality: dict = {
        "note": "frozen #1739 maps applied to issue-2222 rows (all rows held out "
        "w.r.t. the map fit); identity+bias uses leave-one-family-out b",
        "n_rows": int(n_tot),
        "r2_per_layer": {
            "mapped_ctx": [
                float(x)
                for x in 1 - np.sum([cells[ds]["ss_res_map_ctx"] for ds in datasets], 0) / ss_tot
            ],
            "mapped_pfx": [
                float(x)
                for x in 1 - np.sum([cells[ds]["ss_res_map_pfx"] for ds in datasets], 0) / ss_tot
            ],
            "identity": [float(x) for x in 1 - ss_identity / ss_tot],
            "identity_plus_bias": [float(x) for x in 1 - ss_idbias / ss_tot],
        },
    }
    # kNN retrieval per LOFO fold at the stored knn layers.
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    knn_layers = tuple(int(x) for x in cells[datasets[0]]["knn_layers"])
    knn_out: dict = {}
    for kli, layer in enumerate(knn_layers):
        for pred_kind in ("mapped_ctx", "mapped_pfx", "ctxend"):
            fold_reads = []
            for f in range(len(fam_counts)):
                ds_f = [ds for di, ds in enumerate(datasets) if fam_idx[di] == f]
                true = np.concatenate(
                    [cells[ds]["knn_raw"][:, kli].astype(np.float32) for ds in ds_f]
                )
                pred = np.concatenate(
                    [cells[ds][f"knn_{pred_kind}"][:, kli].astype(np.float32) for ds in ds_f]
                )
                if pred_kind == "ctxend":  # identity+bias prediction: ctxend + b_f
                    pred = pred + b_fam[f, layer].astype(np.float32)[None, :]
                cap = args.knn_pool_cap
                if len(true) > cap:
                    rng = np.random.default_rng(args.seed + f)
                    sel = rng.choice(len(true), size=cap, replace=False)
                    true, pred = true[sel], pred[sel]
                for metric in ("euclidean", "cosine"):
                    fold_reads.append(
                        {
                            "fold_family": families[f],
                            **knn_retrieval(pred, true, metric=metric),
                        }
                    )
            knn_out[f"layer{layer}/{'id_bias' if pred_kind == 'ctxend' else pred_kind}"] = (
                fold_reads
            )
    map_quality["knn_retrieval"] = knn_out
    map_quality["knn_note"] = (
        "pool = held-out family rows (capped); chance = k/n_pool reported per read"
    )

    # Per-dataset steer-layer values (unit-3 addition): the P5 per-unit scatter
    # companion reads THESE persisted values — figures never recompute from the
    # percell npzs (plan §6 low-level companion; paper-plots raw-alongside rule).
    dataset_values = {}
    for ti, trait in enumerate(lib.TRAITS):
        steer = STEER_IDX[trait]
        dataset_values[trait] = {
            "steer_layer": steer,
            "y_trait_score": {ds: float(y_axis[trait][ds]["trait_score"]) for ds in datasets},
            "arms": {
                arm: {ds: float(vals[di, ai, ti, steer]) for di, ds in enumerate(datasets)}
                for ai, arm in enumerate(arm_order)
            },
        }

    out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "supervision_ledger": (
            "frozen map = trait-agnostic (no trait labels); r_B = trait description "
            "+ judge filter (#778); Form-A probe = per-sample judge labels (P4)"
        ),
        "datasets": datasets,
        "family_of_dataset": {ds: families[fam_idx[di]] for di, ds in enumerate(datasets)},
        "dataset_values": dataset_values,
        "arm_order": arm_order,
        "records": records,
        "hypothesis_tests": hypothesis_tests,
        "seeds": {"perm": args.seed, "boot": args.seed, "n_perms": n_perms, "n_boot": n_boot},
        **lib.run_metadata(),
    }
    lib.write_json_atomic(out_root / "predictor_correlations.json", payload)
    lib.write_json_atomic(out_root / "map_quality.json", {**map_quality, **lib.run_metadata()})
    lib.write_json_atomic(
        out_root / "auc_misaligned2_vs_normal.json",
        {"records": auc_records, **lib.run_metadata()},
    )
    lib.write_json_atomic(
        out_root / "nulls" / "summary.json",
        {
            "per_arm": nulls_summary,
            "matrices_dir": str(nulls_dir),
            "hf_destination": f"{lib.HF_PREFIX}/analysis_tensors/nulls/",
            **lib.run_metadata(),
        },
    )
    lib.log_phase("p3_aggregate", "done", n_records=len(records))


# --- Stage: tuned map (exploratory; #825 core) ----------------------------------


def stage_tuned_map(args) -> None:
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    datasets = lib.dataset_ids(args.datasets)
    if len(datasets) < 24 and not args.allow_partial:
        raise ValueError("tuned_map over a partial dataset set needs --allow-partial")
    vhat, _meta = load_vhat(data_root)
    fam_idx, families = _family_index(datasets)
    k = args.tuned_rows_per_dataset
    xs, ys, conv, ds_of_row = [], [], [], []
    for di, ds in enumerate(datasets):
        with np.load(stage_capture_file(data_root, ds, "summaries.npz")) as z:
            n = min(k, len(z["row_ids"]))
            xs.append(z["ctxend"][:n])
            ys.append(z["raw_respavg"][:n])
        conv.append(np.full(n, fam_idx[di]))
        ds_of_row.append(np.full(n, di))
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)
    conv_ids = np.concatenate(conv)
    ds_of_row = np.concatenate(ds_of_row)
    n_rows, d = x.shape[0], x.shape[2]
    n_train_min = min(int((conv_ids != f).sum()) for f in np.unique(conv_ids))
    lib.log_phase(
        "p3_tuned",
        "fit via issue825 heldout_r2_sweep (inner-group-cv, dof-capped)",
        n_rows=int(n_rows),
        n_train_min_per_fold=n_train_min,
        d=int(d),
        well_posed=bool(n_train_min > d),
    )
    import issue825_fit_cells as core

    res = core.heldout_r2_sweep(
        x,
        y,
        conv_ids,
        n_folds=len(np.unique(conv_ids)),
        seed=args.seed,
        null_draws=0,
        # collect_cosines=True is LOAD-BEARING: the #825 core writes
        # preds_frozen[li] only under `if li in cosines and collect_cosines:`
        # (issue825_fit_cells.py:881-883) — False leaves all-zero predictions
        # and the mapped_tuned arm silently degenerates to the raw arm.
        collect_cosines=True,
        collect_lambdas=True,
        lambda_selection="inner-group-cv",
        frozen_layers=tuple(range(N_LAYERS)),
        reduced_basis_companion=False,
    )

    # Round-2 C1: the standing identity(+learned-bias) baseline + kNN-retrieval
    # pair for the TUNED map itself (CLAUDE.md mapping rule), on the SAME row
    # subset the tuned fit consumed. identity: v̂ = x (ctxend); identity+bias:
    # v̂ = x + b_f with b_f the leave-one-family-out mean residual (matched to
    # the fit's LOFO folds); kNN runs on the core's held-out predictions per
    # family fold at the stored knn layers.
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    fam_ids_sorted = np.unique(conv_ids)
    r2_identity = np.empty(N_LAYERS)
    r2_id_bias = np.empty(N_LAYERS)
    for layer in range(N_LAYERS):
        x_l = x[:, layer, :].astype(np.float64)
        y_l = y[:, layer, :].astype(np.float64)
        ss_tot = float(((y_l - y_l.mean(axis=0)) ** 2).sum())
        r2_identity[layer] = 1 - float(((y_l - x_l) ** 2).sum()) / ss_tot
        ss_idb = 0.0
        for f in fam_ids_sorted:
            hold = conv_ids == f
            b_f = (y_l[~hold] - x_l[~hold]).mean(axis=0)
            ss_idb += float(((y_l[hold] - x_l[hold] - b_f) ** 2).sum())
        r2_id_bias[layer] = 1 - ss_idb / ss_tot
    tuned_knn: dict[str, list[dict]] = {}
    for layer in (int(v) for v in args.knn_layers or []):
        fold_reads = []
        for f in fam_ids_sorted:
            hold = conv_ids == f
            pred = res["preds_frozen"][layer][hold].astype(np.float32)
            true = y[hold, layer, :].astype(np.float32)
            cap = args.knn_pool_cap
            if len(true) > cap:
                rng = np.random.default_rng(args.seed + int(f))
                sel = rng.choice(len(true), size=cap, replace=False)
                true, pred = true[sel], pred[sel]
            for metric in ("euclidean", "cosine"):
                fold_reads.append(
                    {"fold_family": families[int(f)], **knn_retrieval(pred, true, metric=metric)}
                )
        tuned_knn[f"layer{layer}"] = fold_reads

    # mapped_tuned predictor: held-out prediction projected onto vhat.
    y_axis = load_y_axis(datasets)
    vals = np.full((len(datasets), len(lib.TRAITS), N_LAYERS), np.nan)
    raw_proj = np.einsum("nld,tld->ntl", y.astype(np.float32), vhat.astype(np.float32))
    for layer in range(N_LAYERS):
        pred = res["preds_frozen"][layer]  # (N, D) held-out preds
        pred_proj = pred.astype(np.float32) @ vhat[:, layer, :].astype(np.float32).T  # (N, T)
        diff = raw_proj[:, :, layer] - pred_proj
        for di in range(len(datasets)):
            vals[di, :, layer] = diff[ds_of_row == di].mean(axis=0)
    records = []
    for ti, trait in enumerate(lib.TRAITS):
        yv = np.array([y_axis[trait][ds]["trait_score"] for ds in datasets])
        r_layers = ana.pearson_r_cols(vals[:, ti, :], yv)
        sweep = ana.lofo_layer_sweep(vals[:, ti, :], yv, fam_idx)
        steer = STEER_IDX[trait]
        records.append(
            {
                "trait": trait,
                "arm": "mapped_tuned",
                "r_steer": float(r_layers[steer]),
                "steer_layer": steer,
                "sweep": sweep,
                "r_per_layer": [float(v) for v in r_layers],
            }
        )
    gcv_lam = res.get("gcv_lambda")
    lib.write_json_atomic(
        out_root / "tuned_map.json",
        {
            "note": "EXPLORATORY tuned map (per-layer ridge ctxend->raw_respavg, LOFO over "
            "8 families, #825 core inner-group-cv + dof cap); never pooled with the "
            "frozen-map headline (plan §5). The map fit is trait-agnostic by "
            "construction; per-trait predictors project its held-out predictions.",
            "n_rows": int(n_rows),
            "rows_per_dataset": k,
            "n_train_min_per_fold": n_train_min,
            "d": int(d),
            "well_posed_n_gt_d": bool(n_train_min > d),
            "heldout_r2_per_layer": [float(v) for v in res["r2_obs"]],
            "selected_lambda_per_layer_fold": None
            if gcv_lam is None
            else np.asarray(gcv_lam).tolist(),
            "mapping_baselines": {
                "note": "standing identity(+LOFO-bias) R² baselines on the SAME row "
                "subset the tuned fit consumed; kNN retrieval on the core's held-out "
                "predictions per family fold (round-2 C1)",
                "r2_identity_per_layer": [float(v) for v in r2_identity],
                "r2_identity_plus_bias_per_layer": [float(v) for v in r2_id_bias],
                "knn_retrieval": tuned_knn,
                "knn_note": "pool = held-out family rows (capped at --knn-pool-cap); "
                "chance = k/n_pool reported per read",
            },
            "records": records,
            **lib.run_metadata(),
        },
    )
    lib.log_phase("p3_tuned", "done")


# --- Stage: Form B (exploratory dataset-level regression) ------------------------


def stage_form_b(args) -> None:
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    datasets = lib.dataset_ids(args.datasets)
    if len(datasets) < 24 and not args.allow_partial:
        raise ValueError("form_b over a partial dataset set needs --allow-partial")
    vhat, meta = load_vhat(data_root)
    cells = _load_percell(data_root, datasets, meta=meta, knn_layers=tuple(args.knn_layers))
    fam_idx, _families = _family_index(datasets)
    y_axis = load_y_axis(datasets)
    k_comp = args.form_b_components
    records = []
    for ti, trait in enumerate(lib.TRAITS):
        layer = STEER_IDX[trait]
        x = np.stack(
            [cells[ds]["sum_raw"][layer] / len(cells[ds]["row_ids"]) for ds in datasets]
        )  # (24, D) dataset-mean a(y_train)
        y = np.array([y_axis[trait][ds]["trait_score"] for ds in datasets])
        angles, r2s, preds = [], [], np.full(len(datasets), np.nan)
        for f in np.unique(fam_idx):
            tr = fam_idx != f
            mu, basis = ana.pca_train_basis(x[tr], k_comp)
            xt = (x - mu) @ basis  # (24, k)
            fold = ana.dof_capped_ridge_fit_all(
                xt[tr], y[tr], lambdas=np.logspace(-2, 4, 13), dof_cap=0.9
            )
            w_amb = basis @ fold["w"][:, 0]  # back to ambient (D,)
            angles.append(
                float(
                    np.dot(w_amb, vhat[ti, layer])
                    / (np.linalg.norm(w_amb) * np.linalg.norm(vhat[ti, layer]))
                )
            )
            preds[~tr] = ana.ridge_predict(fold, xt[~tr])[:, 0]
        ss_res = float(((y - preds) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2s = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        records.append(
            {
                "trait": trait,
                "layer": layer,
                "n_datasets": len(datasets),
                "d_ambient": DIM,
                "pca_components": k_comp,
                "cosine_to_rb_by_fold": angles,
                "cosine_to_rb_mean": float(np.mean(angles)),
                "heldout_r2_estimator_degenerate": r2s,
            }
        )
    lib.write_json_atomic(
        out_root / "form_b_regression.json",
        {
            "note": "EXPLORATORY Form B (n=24 << d=3584): PCA<=10 + dof-capped ridge, LOFO. "
            "Estimator-degenerate control — report the angle to r_B only (plan §5/§10 "
            "item (l)); the R^2 field is labeled, never headlined.",
            "records": records,
            **lib.run_metadata(),
        },
    )
    lib.log_phase("p3_form_b", "done")


# --- CLI -------------------------------------------------------------------------


STAGES = {
    "percell": stage_percell,
    "aggregate": stage_aggregate,
    "tuned_map": stage_tuned_map,
    "form_b": stage_form_b,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=str(lib.default_data_root()))
    ap.add_argument("--out-root", default=str(lib.REPO_ROOT / "eval_results" / "issue_2222"))
    ap.add_argument(
        "--stage", default="all", choices=["all", *STAGES], help="which P3 stage to run"
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="dataset selector (families and/or full ids); default all 24",
    )
    ap.add_argument("--seed", type=int, default=lib.SUBSAMPLE_SEED)
    ap.add_argument("--n-perms", type=int, default=10_000)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument(
        "--knn-layers",
        type=int,
        nargs="*",
        default=[15, 19],
        help="layers at which full-vector kNN slices are stored/read (steering set)",
    )
    ap.add_argument("--knn-pool-cap", type=int, default=3000)
    ap.add_argument(
        "--tuned-rows-per-dataset",
        type=int,
        default=250,
        help="rows/dataset for the exploratory tuned-map fit (n_train/fold stated in output)",
    )
    ap.add_argument("--form-b-components", type=int, default=10)
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="permit aggregate stages over <24 datasets (smoke only)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny battery sizes (perms/boot=200) — same code path, smaller draws",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="execute deferred imports + args-attribute completeness check, then exit 0",
    )
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        _import_check()
        raise SystemExit(0)
    kl = [int(v) for v in args.knn_layers or []]
    if len(set(kl)) != len(kl) or not all(0 <= v < N_LAYERS for v in kl):
        raise SystemExit(f"--knn-layers must be duplicate-free within [0, {N_LAYERS}): {kl}")
    if args.smoke:
        args.n_perms = min(args.n_perms, 200)
        args.n_boot = min(args.n_boot, 200)
        args.allow_partial = True
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    for name in stages:
        lib.log_phase("p3_stage_start", name, stage=name)
        STAGES[name](args)
    lib.log_phase("p3_done", "P3 reduction complete", stages=stages)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


def _import_check() -> None:
    """Axis-1 import resolution: execute every deferred/function-body import."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    import torch  # noqa: F401  (load_vhat, ridge)

    import issue825_fit_cells  # noqa: F401  (stage_tuned_map)
    from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
        identity_bias_predict,
        knn_retrieval,
    )
    from explore_persona_space.orchestrate import hub  # noqa: F401  (stage_capture_file)

    print("[import-check] issue2222_reduce OK")


if __name__ == "__main__":
    raise SystemExit(main())
