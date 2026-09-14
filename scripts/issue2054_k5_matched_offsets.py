"""Direct held-out constant-shift tests on conversation-paired K5 representations."""

# ruff: noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import issue2054_k5_loso_calibration as base


def ranks_from_distances(distance):
    """Use the repository retrieval convention, including tolerance-based midranks."""
    truth = np.diag(distance)
    tolerance = 1e-9 * np.maximum(np.abs(truth)[:, None], 1e-12)
    closer = (distance < truth[:, None] - tolerance).sum(1)
    tied = (np.abs(distance - truth[:, None]) <= tolerance).sum(1) - 1
    return 1.0 + closer + 0.5 * tied


def evaluate_pair(source, target, membership):
    """Learn each shift without the test fold, retaining per-query error and rank."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or len(source) != len(membership):
        raise ValueError("invalid matched arrays")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite matched arrays")
    if set(membership) != set(range(5)):
        raise ValueError("expected all five global folds")
    delta = target - source
    energy = np.einsum("ij,ij->i", delta, delta)
    if energy.sum() <= 0:
        raise ValueError("zero displacement makes explained fraction undefined")
    residual = np.zeros(len(source))
    records, biases = [], []
    ranks = {name: np.zeros(len(source)) for name in ("identity", "bias")}
    for fold in range(5):
        test, train = membership == fold, membership != fold
        if min(test.sum(), train.sum()) < 2:
            raise ValueError("insufficient paired fold coverage")
        bias = delta[train].mean(0)
        a, b = source[test], target[test]
        error = delta[test] - bias
        residual[test] = np.einsum("ij,ij->i", error, error)
        total = float(np.square(b - b.mean(0)).sum())
        if total <= 0 or energy[test].sum() <= 0:
            raise ValueError("undefined fold score")
        # Reuse the expensive cross-Gram matrix for identity and identity+bias.
        distance = np.square(a).sum(1)[:, None] + np.square(b).sum(1)[None, :] - 2 * a @ b.T
        distance = np.maximum(distance, 0)
        distances = {
            "identity": distance,
            "bias": np.maximum(
                distance + 2 * (a @ bias)[:, None] - 2 * (b @ bias)[None, :] + bias @ bias, 0
            ),
        }
        metrics = {}
        for name, matrix in distances.items():
            rr = ranks_from_distances(matrix)
            ranks[name][test] = rr
            sse = float((energy if name == "identity" else residual)[test].sum())
            metrics[name] = {
                "r2": 1 - sse / total,
                "euclidean_top1": float((rr <= 1).mean()),
                "euclidean_top5": float((rr <= 5).mean()),
                "median_rank": float(np.median(rr)),
            }
        records.append(
            {
                "fold": fold,
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "retrieval_pool": int(test.sum()),
                "chance_top1": 1 / int(test.sum()),
                "displacement_energy": float(energy[test].sum()),
                "residual_energy": float(residual[test].sum()),
                "constant_fraction": 1 - float(residual[test].sum() / energy[test].sum()),
                "shift_norm": float(np.linalg.norm(bias)),
                "metrics": metrics,
            }
        )
        biases.append(bias)
    return records, {
        "bias": np.stack(biases),
        "fold": membership,
        "displacement_energy": energy,
        "residual_energy": residual,
        "identity_rank": ranks["identity"],
        "bias_rank": ranks["bias"],
    }


def fit(out, inputs, model):
    """Evaluate all fifteen pairs, separately for contexts and K5 mean answers."""
    source_list = json.loads(inputs.read_text())
    sources = {r["path"]: r for r in source_list if "/production_v1/k5/" in r["path"]}
    panel = base.load_panel(model, sources)
    prefixes = [prefix for _, prefix in base.SETTINGS]
    out.mkdir(parents=True, exist_ok=True)
    script_sha = base.sha(__file__)
    for i, j in itertools.combinations(range(6), 2):
        ca, cb = [f"{prefixes[k]}__{model}" for k in (i, j)]
        a, b = panel[ca], panel[cb]
        ai, bi = (
            {cid: n for n, cid in enumerate(a["ids"])},
            {cid: n for n, cid in enumerate(b["ids"])},
        )
        ids = sorted(ai.keys() & bi.keys())
        if len(ids) < 1000:
            raise RuntimeError("unexpectedly small paired intersection")
        ia, ib = np.array([ai[c] for c in ids]), np.array([bi[c] for c in ids])
        folds = a["membership"][ia]
        if not np.array_equal(folds, b["membership"][ib]):
            raise RuntimeError("conversation folds differ between settings")
        for arm, key in [("context", "x"), ("answer", "y")]:
            path = out / "pairs" / f"{model}__{i}_{j}__{arm}.json"
            if path.exists():
                old = json.loads(path.read_text())
                if (
                    old["script_sha256"] != script_sha
                    or base.sha(path.with_suffix(".npz")) != old["array_sha256"]
                ):
                    raise RuntimeError("stale or damaged pair checkpoint")
                print(f"[phase=verified_resume] {path.name}", flush=True)
                continue
            records, arrays = evaluate_pair(a[key][ia], b[key][ib], folds)
            arrays["conv_id"] = np.array(ids)
            arrays["source_cap_mask"], arrays["target_cap_mask"] = a["caps"][ia], b["caps"][ib]
            path.parent.mkdir(parents=True, exist_ok=True)
            base.save_npz(path.with_suffix(".npz"), arrays)
            record = {
                "model": model,
                "source": ca,
                "target": cb,
                "source_index": i,
                "target_index": j,
                "arm": arm,
                "n_source": len(a["ids"]),
                "n_target": len(b["ids"]),
                "n_paired": len(ids),
                "constant_fraction": 1
                - float(arrays["residual_energy"].sum() / arrays["displacement_energy"].sum()),
                "r2_mean": {
                    name: float(np.mean([f["metrics"][name]["r2"] for f in records]))
                    for name in ("identity", "bias")
                },
                "top1_mean": {
                    name: float(np.mean([f["metrics"][name]["euclidean_top1"] for f in records]))
                    for name in ("identity", "bias")
                },
                "folds": records,
                "script_sha256": script_sha,
                "array_sha256": base.sha(path.with_suffix(".npz")),
                "input_sha256": [
                    sources[f"{base.PREFIX}/production_v1/k5/{c}.npz"]["sha256"] for c in (ca, cb)
                ],
            }
            base.atomic_json(path, record)
            print(
                f"[phase=pair] {model} {i}->{j} {arm} n={len(ids)} fraction={record['constant_fraction']:.4f}",
                flush=True,
            )


def collect(out, inputs):
    """Verify complete realized coverage and pin all inference artifacts."""
    rows = []
    for model in base.MODELS:
        for i, j in itertools.combinations(range(6), 2):
            for arm in ("context", "answer"):
                path = out / "pairs" / f"{model}__{i}_{j}__{arm}.json"
                row = json.loads(path.read_text())
                if (
                    row["script_sha256"] != base.sha(__file__)
                    or base.sha(path.with_suffix(".npz")) != row["array_sha256"]
                ):
                    raise RuntimeError("pair inference code or array changed")
                rows.append(row)
    sources = [r for r in json.loads(inputs.read_text()) if "/production_v1/k5/" in r["path"]]
    base.atomic_json(out / "inputs.json", sources)
    report = {
        "status": "complete",
        "pairs": rows,
        "coverage": {"setting_pairs": 30, "representation_panels": 60, "fold_evaluations": 300},
        "settings": base.SETTINGS,
        "k5_revision": base.K5_REV,
        "fold_map_git_sha": base.SOURCE_SHA,
        "folds": 5,
        "seed": 137,
        "method": "For matched conversations, fit b=mean(target-source) on four global folds and predict held-out target as source+b. Context and K5 mean answer arms scored independently. No map predictions are calibrated.",
        "constant_fraction_definition": "1 - sum_heldout ||target-source-b_train||^2 / sum_heldout ||target-source||^2. Pooled across disjoint held-out folds; 1 is perfectly constant, 0 means no improvement over zero shift. This is not centered R2.",
        "limitations": [
            "Training requires paired target representations.",
            "Story scaffolds differ with speaker and are not persona system prompts.",
            "Answer residuals include finite-five-rollout sampling variation; this test is not noise corrected.",
            "R2 and retrieval are source-to-target in displayed order; displacement fraction is invariant to reversing a pair.",
        ],
    }
    base.atomic_json(out / "results.json", report)
    return report


def main():
    """Run a model's paired analysis or collect its verified checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["fit", "collect"], required=True)
    parser.add_argument("--model", choices=base.MODELS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "fit":
        fit(args.out, args.inputs, args.model)
    else:
        collect(args.out, args.inputs)


if __name__ == "__main__":
    main()
