"""Qwen3 context/CoT-end map geometry from existing issue2546 captures only.

Reconstruct the exact production Ridge recipe, verify an existing fold's predictions,
then compare all-row descriptive operators. No generation, penalty selection,
necessity-only refit, or changes to the existing held-out performance estimates.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

CORPORA = ("math", "gsm8k_train", "contexthub", "mmlu", "arc_challenge", "csqa", "piqa")
CELLS = {"context": ("p7_A", "cx_last"), "end_of_thought": ("p7_D", "cot_boundary")}
KS = (10, 50, 200)
DIM, N_ROWS = 4096, 33810
NULL_DRAWS = 5  # Source: issue2546_cx_eot_prepost_diffs.stage_a3a.


def sha256(path: Path) -> str:
    """Hash the actual consumed file without loading all bytes at once."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, payload: dict) -> None:
    """Atomically checkpoint each completed phase, with no NaN placeholders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as temporary:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Persist uncompressed reusable tensors atomically, outside eval_results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as temporary, temporary.open("wb") as handle:
        np.savez(handle, **arrays)


def production_ridge_class(path: Path) -> type:
    """Extract just the production class, avoiding module argv/log side effects."""
    tree = ast.parse(path.read_text(), filename=str(path))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Ridge"]
    if len(classes) != 1:
        raise ValueError(f"Expected one production Ridge class: {path}")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Ridge"]


def raw_operator(model) -> torch.Tensor:
    """Return column-vector M in y=Mx+b, undoing train-input standardization."""
    return (model.B / model.xsd[:, None]).T.contiguous()


def overlap(first: torch.Tensor, second: torch.Tensor) -> dict:
    """Compute principal cosines and basis-invariant mean/squared-share summaries."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError((first.shape, second.shape))
    identity = torch.eye(first.shape[1], dtype=first.dtype)
    for basis in (first, second):
        torch.testing.assert_close(basis.T @ basis, identity, rtol=1e-10, atol=1e-10)
    values = torch.linalg.svdvals(first.T @ second)
    return {
        "principal_cosines": values.tolist(),
        "mean_principal_cos": float(values.mean()),
        "sq_projection_share": float(values.square().mean()),
    }


def random_reference(dimension: int, k: int, draws: int, seed: int) -> dict:
    """Batch Gaussian-QR Haar subspaces against a fixed basis by rotational symmetry."""
    generator = torch.Generator().manual_seed(seed)
    gaussian = torch.randn(draws, dimension, k, generator=generator, dtype=torch.float64)
    basis = torch.linalg.qr(gaussian, mode="reduced").Q
    values = torch.linalg.svdvals(basis[:, :k, :])
    means, shares = values.mean(1), values.square().mean(1)
    return {
        "seed": seed,
        "n_draws": draws,
        "principal_cosines_per_draw": values.tolist(),
        "mean_principal_cos_per_draw": means.tolist(),
        "sq_projection_share_per_draw": shares.tolist(),
        "mean_principal_cos": float(means.mean()),
        "mean_principal_cos_min_max": [float(means.min()), float(means.max())],
        "sq_projection_share": float(shares.mean()),
        "analytic_expected_sq_projection_share": k / dimension,
        "achievable_ceiling": 1.0,
        "note": "Isotropic geometric reference, not a refitted-map significance test.",
    }


def load_state(root: Path, kind: str, ids: np.ndarray, provenance: dict) -> np.ndarray:
    """Align all original evaluation rows to cached states; reject missing/duplicate IDs."""
    positions = {row: index for index, row in enumerate(ids.tolist())}
    if len(positions) != len(ids):
        raise ValueError("Duplicate original row IDs")
    result, filled, seen = np.empty((len(ids), DIM), np.float32), np.zeros(len(ids), bool), set()
    for corpus in CORPORA:
        path = root / "hf/targets" / f"{kind}__arm3__think_on__{corpus}__l24.npz"
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"row_ids", kind}:
                raise ValueError((path, archive.files))
            row_ids, values = archive["row_ids"].astype(str), archive[kind]
        if values.shape != (len(row_ids), DIM) or not np.isfinite(values).all():
            raise ValueError(f"Malformed states: {path}")
        if len(set(row_ids)) != len(row_ids) or seen.intersection(row_ids):
            raise ValueError(f"Duplicate state IDs: {path}")
        seen.update(row_ids)
        keep = np.array([row in positions for row in row_ids], bool)
        selected = np.array([positions[row] for row in row_ids[keep]], np.int64)
        result[selected], filled[selected] = values[keep], True
        provenance[str(path)] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "source_rows": len(row_ids),
            "consumed_rows": len(selected),
            "realized_keys": ["row_ids", kind],
        }
        print(f"[load] {kind}/{corpus} rows={len(selected)}", flush=True)
    if not filled.all():
        raise ValueError(f"Missing {int((~filled).sum())} rows for {kind}")
    return result


def run(args: argparse.Namespace) -> None:
    """Run four fixed-recipe fits, two SVDs, and the predeclared subspace comparisons."""
    started = time.perf_counter()
    if args.out.exists() or args.tensors.exists():
        raise FileExistsError("Fresh output directories required; no silent overwrite/resume")
    args.out.mkdir(parents=True)
    args.tensors.mkdir(parents=True)
    source = args.repo / "scripts/issue2546_allfit_necessity.py"
    ridge = production_ridge_class(source)
    provenance, bases, validations = {}, {}, {}
    metadata = as_metadata_dict(git_provenance(cwd=args.repo), phase="qwen3-map-geometry")
    metadata.update(
        {
            "started_utc": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "script_sha256": sha256(Path(__file__)),
        }
    )
    recipe = {
        "model": "Qwen/Qwen3-8B",
        "arm": 3,
        "mode": "think_on",
        "layer": 24,
        "dimension": DIM,
        "n_rows": N_ROWS,
        "target": "ans_mean",
        "fit_scope": "All-row descriptive reconstruction; not an original held-out-fold fit",
        "rowset": "Original allfit intersection, no necessity filtering",
        "penalty_selection": "Existing production JSON lambda, no new selection",
        "input_standardization": "Train mean, sample std (correction=1) plus1e-9",
        "output_preprocessing": "Train-mean centering only, no whitening",
        "dtype": "float64 fit and SVD",
        "orientation": "Column y=Mx+b; right=input/read; left=output/write",
        "weight_status": "Weights not saved originally; exact recipe reconstructed",
        "production_source": str(source),
        "production_source_sha256": sha256(source),
        "historical_openthinker_difference": "Same geometry definitions, but historical "
        "OpenThinker reconstruction used float32 GEMMs/population std, not exact "
        "production float64/sample std retained here.",
    }
    ids_common = folds_common = answer = None
    for name, (cell, kind) in CELLS.items():
        metric_path = args.repo / "eval_results/issue_2546/allfit" / f"{cell}__a3.json"
        metrics = json.loads(metric_path.read_text())
        provenance[str(metric_path)] = {"sha256": sha256(metric_path)}
        if (
            metrics["arm"] != 3
            or metrics["layer"] != 24
            or metrics["cell"] != cell
            or metrics["x"] != ["think_on", kind]
            or metrics["y"] != ["think_on", "ans_mean"]
            or metrics["n_rows"] != N_ROWS
        ):
            raise ValueError("Wrong production metric metadata")
        penalty = metrics["lambda"]
        pred_path = args.data_root / "allfit/preds" / f"{cell}__all__a3.npz"
        provenance[str(pred_path)] = {"sha256": sha256(pred_path)}
        with np.load(pred_path, allow_pickle=False) as archive:
            ids, folds, saved = (
                archive["conv_ids"].astype(str),
                archive["folds"],
                archive["pred_l24"],
            )
            if not archive["fitted_mask"].all():
                raise ValueError("Incomplete production predictions")
        if len(ids) != N_ROWS or set(folds) != set(range(5)):
            raise ValueError((len(ids), np.unique(folds)))
        if saved.shape != (N_ROWS, DIM) or not np.isfinite(saved).all():
            raise ValueError("Malformed original predictions")
        if ids_common is None:
            ids_common, folds_common = ids, folds
            answer = load_state(args.data_root, "ans_mean", ids, provenance)
        else:
            np.testing.assert_array_equal(ids, ids_common)
            np.testing.assert_array_equal(folds, folds_common)
        inputs = load_state(args.data_root, kind, ids, provenance)
        test = folds == 0
        if int((~test).sum()) <= DIM:
            raise ValueError("Underdetermined parity fit")
        tick = time.perf_counter()
        model = ridge(
            torch.from_numpy(inputs[~test]).double(),
            torch.from_numpy(answer[~test]).double(),
            penalty,
        )
        pilot_seconds = time.perf_counter() - tick
        predicted = model.predict(torch.from_numpy(inputs[test]).double()).float().numpy()
        difference = predicted.astype(np.float64) - saved[test].astype(np.float64)
        # Cached predictions are float32: ~17 machine-eps absolute/relative tolerance.
        np.testing.assert_allclose(predicted, saved[test], rtol=2e-6, atol=2e-6)
        validations[name] = {
            "fold": 0,
            "n_train": int((~test).sum()),
            "n_test": int(test.sum()),
            "lambda": penalty,
            "fit_seconds": pilot_seconds,
            "pass": True,
            "max_abs_difference": float(np.abs(difference).max()),
            "relative_prediction_l2_error": float(
                np.linalg.norm(difference) / np.linalg.norm(saved[test])
            ),
            "rtol": 2e-6,
            "atol": 2e-6,
            "banked_heldout_metrics": metrics["subsets"],
        }
        write_json(args.out / f"parity_{name}.json", validations[name])
        print(
            f"[parity] {name} fit_seconds={pilot_seconds:.2f} "
            f"max_abs={validations[name]['max_abs_difference']:.3g}",
            flush=True,
        )
        del model, saved, predicted, difference
        gc.collect()
        tick = time.perf_counter()
        model = ridge(torch.from_numpy(inputs).double(), torch.from_numpy(answer).double(), penalty)
        operator = raw_operator(model)
        intercept = model.ymu - model.xmu @ operator.T
        probe = torch.from_numpy(inputs[:32]).double()
        torch.testing.assert_close(
            probe @ operator.T + intercept, model.predict(probe), atol=1e-9, rtol=1e-9
        )
        write_npz(
            args.tensors / f"{name}_operator.npz",
            operator=operator.numpy(),
            intercept=intercept.numpy(),
            x_mean=model.xmu.numpy(),
            x_std=model.xsd.numpy(),
            y_mean=model.ymu.numpy(),
            row_ids=ids,
            original_folds=folds,
        )
        del model, inputs, probe
        gc.collect()
        fit_seconds = time.perf_counter() - tick
        print(f"[full_fit] {name} seconds={fit_seconds:.2f}", flush=True)
        tick = time.perf_counter()
        left, singular, right_t = torch.linalg.svd(operator, full_matrices=False)
        relative_error = float(
            torch.linalg.norm(operator @ right_t[:200].T - left[:, :200] * singular[:200])
            / torch.linalg.norm(operator @ right_t[:200].T)
        )
        if relative_error > 1e-10:
            raise ValueError(f"SVD error: {relative_error}")
        bases[name] = {
            "right_input": right_t[: max(KS)].T.contiguous(),
            "left_output": left[:, : max(KS)].contiguous(),
        }
        write_npz(
            args.tensors / f"{name}_bases.npz",
            singular_values=singular.numpy(),
            input_basis=bases[name]["right_input"].numpy(),
            output_basis=bases[name]["left_output"].numpy(),
        )
        write_json(
            args.out / f"fit_{name}.json",
            {
                "lambda": penalty,
                "n_train": len(ids),
                "dimension": DIM,
                "fit_seconds": fit_seconds,
                "svd_seconds": time.perf_counter() - tick,
                "svd_relative_error": relative_error,
                "singular_values": singular.tolist(),
            },
        )
        print(f"[svd] {name} seconds={time.perf_counter() - tick:.2f}", flush=True)
        del operator, left, singular, right_t, intercept
        gc.collect()
    overlaps = {}
    for k in KS:
        overlaps[str(k)] = {
            direction: overlap(
                bases["context"][direction][:, :k], bases["end_of_thought"][direction][:, :k]
            )
            for direction in ("right_input", "left_output")
        }
        overlaps[str(k)]["random_reference"] = random_reference(DIM, k, NULL_DRAWS, 2546 + k)
        write_json(args.out / f"overlap_k{k}.json", overlaps[str(k)])
        print(
            f"[overlap] k={k} input={overlaps[str(k)]['right_input']['mean_principal_cos']:.6f} "
            f"output={overlaps[str(k)]['left_output']['mean_principal_cos']:.6f}",
            flush=True,
        )
    tensor_files = {
        str(p): {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(args.tensors.glob("*.npz"))
    }
    write_json(
        args.out / "results.json",
        {
            "metadata": metadata,
            "recipe": recipe,
            "provenance": provenance,
            "validation": validations,
            "subspace_overlaps": overlaps,
            "tensor_artifacts": tensor_files,
            "elapsed_seconds": time.perf_counter() - started,
            "scope": "Descriptive fitted-map geometry, not model causal computation",
        },
    )
    write_json(
        args.out / "completed.json",
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "exit_code": 0,
            "results_sha256": sha256(args.out / "results.json"),
        },
    )
    print(f"[complete] seconds={time.perf_counter() - started:.2f} out={args.out}", flush=True)


def main() -> None:
    """Parse paths for this analysis-only driver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
