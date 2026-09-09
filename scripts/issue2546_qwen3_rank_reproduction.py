"""CPU-only, validation-selected CoT rank reproduction from cached issue2546 states.

Freeze production ridge penalties/preprocessing. Outer test fold k, inner validation
fold (k+1)%5, fit the remaining three folds; select minimum rank allowing 10% extra
validation SSE (issue2588). Refit on four outer training folds and test the selected
rank. Share raw fold sufficient statistics; one output eigensystem gives all ranks.
No model inference, rank-by-rank refits, new penalty search, or cross-model p-values.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import resource
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

from explore_persona_space.analysis.null_battery import (  # noqa: E402
    PRIMARY_LAMBDA,
    shrunk_cholesky_from_cov,
)
from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

CORPORA = ("math", "gsm8k_train", "contexthub", "mmlu", "arc_challenge", "csqa", "piqa")
STATES = {"context": ("p7_A", "cx_last"), "end_of_thought": ("p7_D", "cot_boundary")}
DIM, N_ROWS = 4096, 33810
TOLERANCES = (0.05, 0.10, 0.20)  # 10% primary; fixed cheap sensitivity checks.


def sha256(path: Path) -> str:
    """Hash consumed source bytes without materializing large files."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, payload: dict) -> None:
    """Persist a completed phase atomically; non-finite JSON is forbidden."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as temporary:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def load_state(
    root: Path, kind: str, ids: np.ndarray, manifest: dict, side: str = "think_on"
) -> np.ndarray:
    """Align cached states exactly to original OOF IDs, rejecting missing/duplicate rows."""
    positions = {row: i for i, row in enumerate(ids.tolist())}
    if len(positions) != len(ids):
        raise ValueError("Duplicate evaluation IDs")
    result = np.empty((len(ids), DIM), np.float32)
    filled, seen = np.zeros(len(ids), bool), set()
    for corpus in CORPORA:
        if side not in ("think_on", "think_off"):
            raise ValueError(f"Unknown thinking mode: {side}")
        path = root / "hf/targets" / f"{kind}__arm3__{side}__{corpus}__l24.npz"
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"row_ids", kind}:
                raise ValueError((path, archive.files))
            rows, values = archive["row_ids"].astype(str), archive[kind]
        if values.shape != (len(rows), DIM) or not np.isfinite(values).all():
            raise ValueError(f"Invalid activation matrix: {path}")
        if len(set(rows)) != len(rows) or seen.intersection(rows):
            raise ValueError(f"Duplicate cached IDs: {path}")
        seen.update(rows)
        keep = np.array([r in positions for r in rows], bool)
        selected = np.array([positions[r] for r in rows[keep]], np.int64)
        result[selected], filled[selected] = values[keep], True
        manifest[str(path)] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "source_rows": len(rows),
            "consumed_rows": len(selected),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "finite": True,
            "duplicate_ids": 0,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
        print(f"[load] {kind}/{corpus} {len(selected)} rows", flush=True)
    if not filled.all():
        raise ValueError(f"Missing {int((~filled).sum())} rows for {kind}")
    return result


@dataclass
class Moments:
    """Additive raw sufficient statistics, kept in float64 throughout."""

    n: int
    sx: torch.Tensor
    sy: torch.Tensor
    xx: torch.Tensor
    xy: torch.Tensor


def moments(x: torch.Tensor, y: torch.Tensor) -> Moments:
    """Reduce a row block once, reusable by all outer/inner train splits."""
    return Moments(len(x), x.sum(0), y.sum(0), x.T @ x, x.T @ y)


def combine(blocks: list[Moments]) -> Moments:
    """Combine disjoint blocks without re-reading tall activation matrices."""
    if not blocks:
        raise ValueError("Cannot combine an empty training set")
    return Moments(
        sum(b.n for b in blocks),
        *(
            torch.stack([getattr(b, key) for b in blocks]).sum(0)
            for key in ("sx", "sy", "xx", "xy")
        ),
    )


def participation_ratio(block: Moments) -> float:
    """Exact raw centered-covariance PR, trace(C)^2/||C||F^2, without eigenvectors."""
    scatter = block.xx - torch.outer(block.sx, block.sx) / block.n
    return float(scatter.trace().square() / scatter.square().sum())


def fit(block: Moments, penalty: float) -> dict:
    """Production sample-standardized ridge and TRAIN-only fitted-output PCA basis."""
    xmu, ymu = block.sx / block.n, block.sy / block.n
    centered = block.xx - torch.outer(block.sx, block.sx) / block.n
    variance = centered.diagonal() / (block.n - 1)
    if torch.any(variance < 0):
        raise ValueError("Negative input variance from sufficient statistics")
    xsd = variance.sqrt() + 1e-9  # Exact issue2546 production convention.
    gram = centered / xsd[:, None] / xsd[None, :]
    cross = (block.xy - torch.outer(block.sx, block.sy) / block.n) / xsd[:, None]
    regularized = gram.clone()
    regularized.diagonal().add_(penalty)
    coef = torch.cholesky_solve(cross, torch.linalg.cholesky(regularized))
    output_gram = coef.T @ gram @ coef
    values, vectors = torch.linalg.eigh((output_gram + output_gram.T) / 2)
    if float(values.min()) < -1e-10 * float(values.max()):
        raise ValueError("Fitted-output Gram is not positive semidefinite")
    return {
        "xmu": xmu,
        "ymu": ymu,
        "xsd": xsd,
        "coef": coef,
        "vectors": vectors.flip(1),
        "eigenvalues": values.flip(0).clamp_min(0),
    }


def predict(model: dict, x: torch.Tensor) -> torch.Tensor:
    """Evaluate the frozen affine map in its original answer coordinate system."""
    return ((x - model["xmu"]) / model["xsd"]) @ model["coef"] + model["ymu"]


def sse_curve(
    prediction: torch.Tensor, target: torch.Tensor, intercept: torch.Tensor, vectors: torch.Tensor
) -> np.ndarray:
    """Evaluate every output-projection rank by cumulative SSE, including rank zero."""
    pc = (prediction - intercept) @ vectors
    yc = (target - intercept) @ vectors
    increments = (pc.square() - 2 * pc * yc).sum(0)
    zero = (target - intercept).square().sum()
    curve = torch.cat((zero[None], zero + increments.cumsum(0))).numpy()
    direct = float((target - prediction).square().sum())
    np.testing.assert_allclose(curve[-1], direct, rtol=1e-10, atol=1e-7)
    if not np.isfinite(curve).all() or np.any(curve < 0):
        raise ValueError("Invalid rank SSE curve")
    return curve


def select_rank(curve: np.ndarray, tolerance: float) -> int:
    """Smallest rank allowing the specified extra error relative to the full map."""
    passing = np.flatnonzero(curve <= (1 + tolerance) * curve[-1])
    if len(passing) == 0:
        raise ValueError("Full rank must meet its own tolerance")
    return int(passing[0])


def corpus_sst(y: torch.Tensor, train: np.ndarray, test: np.ndarray, corpus: np.ndarray) -> float:
    """Paper denominator: test error of TRAIN-fold, per-corpus answer means."""
    total = 0.0
    for name in np.unique(corpus[test]):
        tr, te = train & (corpus == name), test & (corpus == name)
        if not tr.any():
            raise ValueError(f"Corpus {name} absent from training")
        total += float((y[te] - y[tr].mean(0)).square().sum())
    return total


def retrieval(
    prediction: torch.Tensor, target: torch.Tensor, train_mean: torch.Tensor, ell: np.ndarray
) -> np.ndarray:
    """Original train-whitened cosine/CSLS k=10 retrieval over the whole test fold."""

    def whiten_unit(values: torch.Tensor) -> np.ndarray:
        """Whiten using training-only parameters then normalize each row."""
        z = solve_triangular(ell, (values - train_mean).numpy().T, lower=True).T
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)

    similarity = whiten_unit(prediction) @ whiten_unit(target).T
    n = len(similarity)
    if n <= 10:
        raise ValueError("CSLS k=10 requires a larger pool")
    rq = np.partition(similarity, n - 10, axis=1)[:, n - 10 :].mean(1)
    rp = np.partition(similarity, n - 10, axis=0)[n - 10 :, :].mean(0)
    similarity *= 2
    similarity -= rq[:, None]
    similarity -= rp[None, :]
    return similarity.argmax(1) == np.arange(n)


def validate_production_class(
    source: Path, x: torch.Tensor, y: torch.Tensor, block: Moments, penalty: float
) -> None:
    """Compare sufficient-stat ridge against the safely extracted production class."""
    nodes = [
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Ridge"
    ]
    if len(nodes) != 1:
        raise ValueError("Expected one original Ridge class")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    old = namespace["Ridge"](x, y, penalty)
    new = fit(block, penalty)
    torch.testing.assert_close(predict(new, x), old.predict(x), rtol=1e-9, atol=1e-9)


def run(args: argparse.Namespace) -> None:
    """Run fixed nested rank selection, checkpointing each complete outer state/fold."""
    torch.set_num_threads(8)
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    recipe = {
        "model": "Qwen/Qwen3-8B",
        "layer": 24,
        "dimension": DIM,
        "n_rows": N_ROWS,
        "mode": "think_on",
        "answer_target": "ans_mean",
        "outer_test": "original fold k",
        "inner_validation": "original fold (k+1)%5",
        "inner_training": "remaining three folds",
        "outer_refit": "all four non-test folds",
        "selection": "minimum TRAIN-output-PCA projection rank with <=10% extra validation SSE",
        "sensitivity_extra_sse": list(TOLERANCES),
        "penalties": "frozen original production values; no new selection",
        "r2_baseline": "outer TRAIN-fold per-corpus answer means",
        "input_dimension": "raw centered training-input covariance participation ratio",
        "inference": "one model; fold ranges are descriptive, no model-population p-value",
        "independence_limit": (
            "rank selection excludes outer test; penalties/layer inherited from original study, "
            "not newly nested"
        ),
        "dtype": "float64",
        "gpu_hours": 0,
        "script_sha256": sha256(Path(__file__)),
        "production_script_sha256": sha256(args.production_source),
    }
    manifest, records = {}, []
    with np.load(args.data_root / "allfit/preds/p7_A__all__a3.npz", allow_pickle=False) as a:
        ids, folds, labels = a["conv_ids"].astype(str), a["folds"], a["labels"]
    if len(ids) != N_ROWS or not np.array_equal(np.bincount(folds), [6762] * 5):
        raise ValueError("Wrong original row/fold universe")
    corpus = np.array([row.split(":")[0] for row in ids])
    y = torch.from_numpy(load_state(args.data_root, "ans_mean", ids, manifest)).double()
    yy_blocks = [moments(y[folds == k], y[folds == k]) for k in range(5)]
    baseline, whiteners = {}, {}
    for k in range(5):
        tr, te = folds != k, folds == k
        b = combine([yy_blocks[j] for j in range(5) if j != k])
        cov = (b.xx - torch.outer(b.sx, b.sx) / b.n) / (b.n - 1)
        whiteners[k] = shrunk_cholesky_from_cov(cov.numpy(), PRIMARY_LAMBDA)
        baseline[k] = {
            "sst_corpus": corpus_sst(y, tr, te, corpus),
            "sst_global": float((y[te] - b.sy / b.n).square().sum()),
        }
    del yy_blocks
    for state, (cell, kind) in STATES.items():
        pred_path = args.data_root / "allfit/preds" / f"{cell}__all__a3.npz"
        with np.load(pred_path, allow_pickle=False) as archive:
            for key, expected in (("conv_ids", ids), ("folds", folds), ("labels", labels)):
                np.testing.assert_array_equal(archive[key], expected)
            saved = archive["pred_l24"]
            if not archive["fitted_mask"].all() or not np.isfinite(saved).all():
                raise ValueError("Incomplete cached predictions")
        metric_path = args.data_root / "allfit/results" / f"{cell}__a3.json"
        original = json.loads(metric_path.read_text())
        if (
            original["x"] != ["think_on", kind]
            or original["y"] != ["think_on", "ans_mean"]
            or original["n_rows"] != N_ROWS
            or original["layer"] != 24
        ):
            raise ValueError("Wrong production result metadata")
        penalty = original["lambda"]
        manifest[str(pred_path)] = {"sha256": sha256(pred_path), "shape": list(saved.shape)}
        manifest[str(metric_path)] = {"sha256": sha256(metric_path)}
        hit_path = args.data_root / "allfit/preds" / f"hits__{cell}__all__a3.npz"
        with np.load(hit_path, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["row_ids"], ids)
            np.testing.assert_array_equal(archive["folds"], folds)
            original_hits = archive["hit_whitened_csls"]
        manifest[str(hit_path)] = {"sha256": sha256(hit_path)}
        x = torch.from_numpy(load_state(args.data_root, kind, ids, manifest)).double()
        blocks = []
        for k in range(5):
            tick = time.perf_counter()
            blocks.append(moments(x[folds == k], y[folds == k]))
            print(f"[moments] {state}/{k} seconds={time.perf_counter() - tick:.1f}", flush=True)
        all_pr = participation_ratio(combine(blocks))
        for k in args.folds:
            destination = args.out / f"{state}__fold{k}.json"
            key = {
                "recipe": recipe,
                "state": state,
                "fold": k,
                "lambda": penalty,
                "source_hashes": {p: m["sha256"] for p, m in manifest.items()},
            }
            if destination.exists():
                existing = json.loads(destination.read_text())
                if existing["cache_key"] != key or existing["status"] != "complete":
                    raise ValueError(f"Stale/incompatible checkpoint: {destination}")
                records.append(existing)
                print(f"[resume] {state}/{k}", flush=True)
                continue
            tick = time.perf_counter()
            te, va = folds == k, folds == (k + 1) % 5
            tr = ~(te | va)
            inner = fit(
                combine([blocks[j] for j in range(5) if j not in (k, (k + 1) % 5)]), penalty
            )
            val_curve = sse_curve(predict(inner, x[va]), y[va], inner["ymu"], inner["vectors"])
            ranks = {str(t): select_rank(val_curve, t) for t in TOLERANCES}
            del inner
            outer_block = combine([blocks[j] for j in range(5) if j != k])
            outer = fit(outer_block, penalty)
            full = predict(outer, x[te])
            cast = full.float().numpy()
            difference = cast.astype(np.float64) - saved[te].astype(np.float64)
            # Same gate as the prior issue2546 Qwen3 geometry reconstruction.
            np.testing.assert_allclose(cast, saved[te], rtol=2e-6, atol=2e-6)
            test_curve = sse_curve(full, y[te], outer["ymu"], outer["vectors"])
            rank = ranks["0.1"]
            basis = outer["vectors"][:, :rank]
            reduced = ((full - outer["ymu"]) @ basis) @ basis.T + outer["ymu"]
            reduced_hits = retrieval(reduced, y[te], outer["ymu"], whiteners[k])
            identity = x[te] + (outer["ymu"] - outer["xmu"])
            identity_hits = retrieval(identity, y[te], outer["ymu"], whiteners[k])
            selected = {
                t: {
                    "rank": r,
                    "test_sse": float(test_curve[r]),
                    "test_r2_corpus": 1 - float(test_curve[r]) / baseline[k]["sst_corpus"],
                    "test_extra_sse": float(test_curve[r] / test_curve[-1] - 1),
                }
                for t, r in ranks.items()
            }
            record = {
                "status": "complete",
                "cache_key": key,
                "state": state,
                "fold": k,
                "n_inner_train": int(tr.sum()),
                "n_validation": int(va.sum()),
                "n_test": int(te.sum()),
                "n_outer_train": outer_block.n,
                **baseline[k],
                "selected": selected,
                "full_test_sse": float(test_curve[-1]),
                "full_test_r2_corpus": 1 - float(test_curve[-1]) / baseline[k]["sst_corpus"],
                "full_retrieval_hits": int(original_hits[te].sum()),
                "rank10_retrieval_hits": int(reduced_hits.sum()),
                "identity_bias_sse": float((identity - y[te]).square().sum()),
                "identity_bias_retrieval_hits": int(identity_hits.sum()),
                "retrieval_pool": int(te.sum()),
                "retrieval_chance": 1 / int(te.sum()),
                "input_pr_train": participation_ratio(outer_block),
                "input_pr_all_rows": all_pr,
                "validation_sse_by_rank": val_curve.tolist(),
                "test_sse_by_rank": test_curve.tolist(),
                "prediction_parity": {
                    "max_abs": float(np.abs(difference).max()),
                    "rmse": float(np.sqrt(np.mean(difference**2))),
                    "float32_exact_fraction": float(np.mean(cast == saved[te])),
                },
                "seconds": time.perf_counter() - tick,
                "finished_utc": datetime.now(UTC).isoformat(),
            }
            write_json(destination, record)
            records.append(record)
            print(
                f"[complete] {state}/{k} rank={rank} R2={record['full_test_r2_corpus']:.6f} "
                f"PR={record['input_pr_train']:.3f} seconds={record['seconds']:.1f}",
                flush=True,
            )
        del x, blocks, saved
    write_json(
        args.out / "manifest.json",
        {
            "recipe": recipe,
            "sources": manifest,
            "corpus_counts": {c: int((corpus == c).sum()) for c in CORPORA},
            "runtime": {
                **as_metadata_dict(
                    git_provenance(cwd=Path(__file__).resolve().parents[1]),
                    phase="qwen3-cot-rank-reproduction",
                ),
                "seconds": time.perf_counter() - started,
                "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                "pid": os.getpid(),
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
        },
    )
    if len(records) == 10:
        summary = {}
        for state in STATES:
            rows = [r for r in records if r["state"] == state]
            denom = sum(r["sst_corpus"] for r in rows)
            total = sum(r["n_test"] for r in rows)
            summary[state] = {
                "r2_full": 1 - sum(r["full_test_sse"] for r in rows) / denom,
                "r2_rank10": 1 - sum(r["selected"]["0.1"]["test_sse"] for r in rows) / denom,
                "r2_identity_bias": 1 - sum(r["identity_bias_sse"] for r in rows) / denom,
                "rank10_by_fold": [r["selected"]["0.1"]["rank"] for r in rows],
                "rank_sensitivity": {
                    str(t): [r["selected"][str(t)]["rank"] for r in rows] for t in TOLERANCES
                },
                "input_pr_by_fold": [r["input_pr_train"] for r in rows],
                "input_pr_all_rows": rows[0]["input_pr_all_rows"],
                "retrieval_full": sum(r["full_retrieval_hits"] for r in rows) / total,
                "retrieval_rank10": sum(r["rank10_retrieval_hits"] for r in rows) / total,
                "retrieval_identity_bias": sum(r["identity_bias_retrieval_hits"] for r in rows)
                / total,
                "test_extra_sse_rank10": sum(r["selected"]["0.1"]["test_sse"] for r in rows)
                / sum(r["full_test_sse"] for r in rows)
                - 1,
            }
            original = json.loads(
                (args.data_root / "allfit/results" / f"{STATES[state][0]}__a3.json").read_text()
            )
            np.testing.assert_allclose(
                summary[state]["r2_full"],
                original["subsets"]["all"]["r2_corpus"],
                atol=1e-9,
                rtol=0,
            )
            np.testing.assert_allclose(
                summary[state]["retrieval_full"],
                original["subsets"]["all"]["acc1"],
                atol=1e-12,
                rtol=0,
            )
        write_json(
            args.out / "summary.json",
            {
                "status": "complete",
                "recipe": recipe,
                "states": summary,
                "n_models": 1,
                "n_rows": N_ROWS,
                "n_folds": 5,
                "completed_cells": len(records),
            },
        )
        print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    """Parse only explicit existing-artifact paths; CPU-only execution is intentional."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--production-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds) or not set(args.folds).issubset(range(5)):
        parser.error("folds must be distinct values in 0..4")
    run(args)


if __name__ == "__main__":
    main()
