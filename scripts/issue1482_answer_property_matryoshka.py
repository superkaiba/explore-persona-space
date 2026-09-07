"""Matched L20 answer-mean readout of the original Matryoshka token-SAE means.

User-requested #1482 CPU continuation. The original layer-20, 16,384-feature
tier panel and 22k/2k/6k split are preserved exactly. No SAE-of-mean targets.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import issue1482_error_analysis as EA  # noqa: E402
import issue1482_sae_dense_bridge as BR  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue1482_answer_property_ceiling import (  # noqa: E402
    LAMBDA_GRID,
    atomic_npy,
    atomic_npz,
    digest,
    log,
    provenance,
    score_arrays,
)

from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: E402
from explore_persona_space.atomic_io import write_json_atomic  # noqa: E402

REPO = Path("/home/thomasjiralerspong/explore-persona-space")
RUN = Path("/home/thomasjiralerspong/.codex/research/answer-property-ceiling-20260907")
BANK = (
    RUN
    / "matryoshka_source/issue1482_error_analysis/analysis_tensors/matryoshka_tier/store"
)
OUT = RUN / "matryoshka"
VBAR = Path(
    "/mnt/eps-data/thomasjiralerspong/issue2476_floorsweep/smoke/stage_banked/"
    "issue2476_turnavg/analysis_tensors/recapture_store/vbar_store.npz"
)
PARENT = REPO / "eval_results/issue_1482/matryoshka_tier"


def join_index(rows: np.ndarray, source_rows: np.ndarray) -> np.ndarray:
    """One-to-one row join, rejecting missing, extra, or duplicated identities."""
    if len(np.unique(rows)) != len(rows) or len(np.unique(source_rows)) != len(
        source_rows
    ):
        raise ValueError("duplicate row identity")
    if set(rows) != set(source_rows):
        raise ValueError("source row population differs")
    positions = {int(r): i for i, r in enumerate(source_rows)}
    return np.asarray([positions[int(r)] for r in rows], dtype=np.int64)


def prepare() -> None:
    """Verify sources and persist aligned observed/context designs and targets."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "prepared.json").exists():
        doc = json.loads((OUT / "prepared.json").read_text())
        for name, sha in doc["outputs_sha256"].items():
            if digest(OUT / name) != sha:
                raise ValueError(f"prepared Matryoshka output changed: {name}")
        log("[matryoshka] verified prepared checkpoint")
        return
    manifest = json.loads((RUN / "matryoshka_stage_manifest.json").read_text())
    for entry in manifest["files"]:
        if digest(Path(entry["local"])) != entry["sha256"]:
            raise ValueError(f"source hash mismatch: {entry['path']}")
    with np.load(BANK / "split_indices_matryoshka.npz") as z:
        rows = np.concatenate([z["s_fit"], z["s_score"]]).astype(np.int64)
        if len(z["s_fit"]) != 24000 or len(z["s_score"]) != 6000:
            raise ValueError("original split size changed")
        corpus = np.concatenate([z["prov_fit"], z["prov_score"]])
    join_index(rows, rows)
    with np.load(PARENT / "perfeature_m_lmsys_default.npz") as z:
        ids, tiers, activity = z["feat_ids"], z["tier"], z["activity"]
    if len(ids) != 16384 or len(np.unique(ids)) != len(ids):
        raise ValueError("original tier panel changed")
    perm = np.random.default_rng(14825).permutation(24000)
    original_tr, va, te = perm[2000:], perm[:2000], np.arange(24000, 30000)
    parts = []
    for path in sorted(BANK.glob("pooled_m_lmsys_g*.npz")):
        with np.load(path) as z:
            parts.append(
                {
                    k: z[k]
                    for k in ("row_idx", "ans_idx", "idx_off", "ans_mean", "n_ans")
                }
            )
    original_rows = np.concatenate([p["row_idx"] for p in parts])
    ji = join_index(rows, original_rows)
    n_ans = np.concatenate([p["n_ans"] for p in parts])[ji]
    pos = {int(r): i for i, r in enumerate(rows)}
    targets = EA._densify(parts, "ans_idx", "idx_off", "ans_mean", ids, len(rows), pos)
    if not np.isfinite(targets).all():
        raise ValueError("nonfinite original token-SAE means")
    atomic_npy(OUT / "targets.npy", targets)
    del targets, parts
    gc.collect()
    dense_parts, dense_rows = [], []
    for path in sorted(BANK.glob("dense_l20_g*.npz")):
        with np.load(path) as z:
            dense_parts.append(z["c20"])
            dense_rows.append(z["row_idx"])
    context = np.concatenate(dense_parts)[join_index(rows, np.concatenate(dense_rows))]
    with np.load(VBAR) as z:
        rec_j = join_index(rows, z["row_idx"])
        observed_n_ans = z["n_ans"][rec_j]
        mismatch = n_ans != observed_n_ans
        # Explicit pre-fit audit found exactly two training-only mismatches.
        # Do not replace their targets, infer a cause, or relax the identity check.
        mismatch_rows = rows[mismatch]
        if set(mismatch_rows.tolist()) != {656068, 656678}:
            raise ValueError("recapture token-count mismatch population changed")
        if not np.all(n_ans[mismatch] == 1024) or not np.all(
            observed_n_ans[mismatch] == 1014
        ):
            raise ValueError("recapture token-count discrepancy changed")
        if np.any(mismatch[va]) or np.any(mismatch[te]):
            raise ValueError("unanticipated validation/test token-count mismatch")
        tr = original_tr[~mismatch[original_tr]]
        if len(tr) != 21998:
            raise ValueError("matched training population changed")
        observed = z["vbar20"][rec_j]
        cosine = z["c20_cos"][rec_j].astype(np.float64)
    if not np.isfinite(cosine).all():
        raise ValueError("nonfinite recapture identity cosine")
    # Reuse #2476 G2a v2: do not resurrect its superseded flat 0.999 gate.
    gate = {
        "min": float(cosine.min()),
        "p001": float(np.quantile(cosine, 0.001)),
        "median": float(np.median(cosine)),
    }
    if gate["min"] < 0.995 or gate["p001"] < 0.999 or gate["median"] < 0.9995:
        raise ValueError(f"recapture identity gate failed: {gate}")
    source_gate_path = REPO / "eval_results/issue_2476/turnavg/gates_p2.json"
    source_gates = json.loads(source_gate_path.read_text())
    if source_gates["g2b"]["verdict"] != "PASS" or not source_gates["g2b"]["max_at_20"]:
        raise ValueError("original recapture hook gate failed")
    for name, x in (("observed_answer", observed), ("context", context)):
        if x.shape != (30000, 3584) or not np.isfinite(x).all():
            raise ValueError(f"invalid {name} design")
        atomic_npy(OUT / f"{name}.npy", x)
    atomic_npz(
        OUT / "registry.npz",
        rows=rows,
        tr=tr,
        original_tr=original_tr,
        va=va,
        te=te,
        feat_ids=ids,
        tier=tiers,
        activity=activity,
        n_ans=n_ans,
        observed_n_ans=observed_n_ans,
        excluded_train_rows=mismatch_rows,
        corpus=corpus,
    )
    write_json_atomic(
        OUT / "prepared.json",
        {
            "layer": 20,
            "target": "mean_t(SAE(h_t))",
            "input": "mean_t(h_t)",
            "carve_seed": 14825,
            "n_train": 21998,
            "original_n_train": 22000,
            "excluded_train_rows": mismatch_rows.tolist(),
            "exclusion": "original target averages 1024 answer tokens, recapture averages 1014; both new matched fits exclude these two rows; validation/test unchanged; cause unresolved",
            "n_val": 2000,
            "n_test": 6000,
            "panel_width": 16384,
            "dictionary_width": 65536,
            "source_manifest_sha256": digest(RUN / "matryoshka_stage_manifest.json"),
            "vbar_sha256": digest(VBAR),
            "source_gates": source_gates,
            "source_gates_sha256": digest(source_gate_path),
            "recapture_g2a_v2": gate,
            "outputs_sha256": {
                name: digest(OUT / name)
                for name in (
                    "targets.npy",
                    "observed_answer.npy",
                    "context.npy",
                    "registry.npz",
                )
            },
            "metadata": provenance("matryoshka-prepare"),
        },
    )
    log(
        "[matryoshka] 30000 rows joined; two mismatched training rows excluded from both new fits; all validation/test rows agree"
    )


def fit(source: str) -> None:
    """Exact parent shared ridge fit with a persisted validation curve and outputs."""
    destination = OUT / source
    destination.mkdir(exist_ok=True)
    identity = {
        "prepared_sha256": digest(OUT / "prepared.json"),
        "script_sha256": digest(Path(__file__)),
        "source": source,
    }
    sentinel = destination / "complete.json"
    if sentinel.exists():
        old = json.loads(sentinel.read_text())
        if old["identity"] != identity:
            raise ValueError("Matryoshka fit identity changed")
        for name, sha in old["outputs_sha256"].items():
            if digest(destination / name) != sha:
                raise ValueError(f"Matryoshka fit payload changed: {name}")
        return
    started = time.monotonic()
    input_name = "context" if source == "context_original" else source
    x = np.load(OUT / f"{input_name}.npy", mmap_mode="r")
    y = np.load(OUT / "targets.npy", mmap_mode="r")
    with np.load(OUT / "registry.npz") as z:
        tr, va, te, ids = (z[k] for k in ("tr", "va", "te", "feat_ids"))
        if source == "context_original":
            tr = z["original_tr"]
    log(f"[matryoshka:{source}] factorization begins")
    fac = BR.N1M._ridge_factorize(x, y, tr, "cpu", 1024)
    parity = BR._assert_predict_equivalence(x, va, fac, "cpu", 1024)
    eva, ete = (BR._project(x, idx, fac, "cpu", 1024) for idx in (va, te))
    yval = np.asarray(y[va], dtype=np.float64)
    curve = []
    for lam in LAMBDA_GRID:
        p = BR._predict_projected(eva, fac, lam)
        curve.append(float(np.square(yval - p).sum()))
    best = int(np.argmin(curve))
    if best in (0, len(LAMBDA_GRID) - 1):
        raise ValueError(f"selected penalty at grid edge: {best}")
    lam = float(LAMBDA_GRID[best])
    atomic_npy(
        destination / "predictions.npy",
        BR._predict_projected(ete, fac, lam).astype(np.float32),
    )
    atomic_npz(
        destination / "factor.npz", **{k: v.cpu().numpy() for k, v in fac.items()}
    )
    pred = np.load(destination / "predictions.npy", mmap_mode="r")
    true = y[te]
    metrics = score_arrays(true, pred)
    atomic_npz(destination / "perfeature.npz", feat_ids=ids, **metrics)
    # Required retrieval diagnostic, fixed original test order and full 6000 pool.
    retrieval = {
        m: knn_retrieval(pred, true, ks=(1, 5, 10), metric=m)
        for m in ("euclidean", "cosine")
    }
    parent_parity = None
    if source == "context_original":
        with np.load(PARENT / "perfeature_m_lmsys_dense_in.npz") as z:
            if not np.array_equal(ids, z["feat_ids"]):
                raise ValueError("parent feature order mismatch")
            finite = np.isfinite(metrics["r2"]) & np.isfinite(z["r2"])
            parent_parity = {
                "max_abs_r2": float(
                    np.max(np.abs(metrics["r2"][finite] - z["r2"][finite]))
                ),
                "n_compared": int(finite.sum()),
            }
        if parent_parity["max_abs_r2"] > 0.002:
            raise ValueError(
                f"recomputed original dense-context baseline differs: {parent_parity}"
            )
    write_json_atomic(
        sentinel,
        {
            "identity": identity,
            "n_train": len(tr),
            "selected_lambda": lam,
            "lambda_grid": LAMBDA_GRID.tolist(),
            "validation_sse": curve,
            "pooled_r2": float(1 - metrics["ss_res"].sum() / metrics["ss_tot"].sum()),
            "n_defined": int(np.isfinite(metrics["r2"]).sum()),
            "retrieval": retrieval,
            "identity_plus_bias": "inapplicable: 3584 dense coordinates versus 16384 SAE features",
            "projected_prediction_parity": parity,
            "parent_dense_context_parity": parent_parity,
            "wall_seconds": time.monotonic() - started,
            "outputs_sha256": {
                name: digest(destination / name)
                for name in (
                    "predictions.npy",
                    "factor.npz",
                    "perfeature.npz",
                )
            },
            "metadata": provenance(f"matryoshka-fit-{source}"),
        },
    )
    log(
        f"[matryoshka:{source}] finished lambda={lam:g} seconds={time.monotonic() - started:.1f}"
    )
    del fac, eva, ete, yval, pred, true
    gc.collect()


if __name__ == "__main__":
    torch.set_num_threads(8)
    prepare()
    for arm in ("observed_answer", "context", "context_original"):
        fit(arm)
