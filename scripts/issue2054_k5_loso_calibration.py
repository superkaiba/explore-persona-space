"""Calibrate the completed K5 LOSO maps using target-training conversations.

Restore the exact source-only estimator; verify its held-out predictions against
the immutable parent. Only a vector intercept and, optionally, one scalar are
estimated on the excluded setting's NON-TEST folds. No new model inference.
"""

# ruff: noqa: E402
# Load the shared-VM thread caps before importing numpy/torch-backed helpers.

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from explore_persona_space.orchestrate.hub import retry_transient
from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance
from scripts import issue2054_k5_loso as loso
from scripts.issue2054_k3 import atomic_json
from scripts.issue2054_k3_fit import save_npz, score
from scripts.issue2054_k5_paper_figures import SETTINGS

HF_REPO = "superkaiba1/explore-persona-space-data"
K5_REV = "9de026f872c19b2ca4fd3e4539de820e08038ee3"
LOSO_REV = "534f9b67838897671bb35f5fb2a73059bd0f9fa7"
PREFIX = "issue2054_section44_k5_gcp"
SOURCE_SHA = "957c454a8ec9a2f520bb754543e7494904636e20"
MODELS = ("qwen2.5-7b", "qwen2.5-7b-instruct")


def sha(path):
    """Return a streaming SHA-256 for a local artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def references():
    """Read and validate the immutable completed LOSO report."""
    path = REPO / "eval_results/issue_2054/section44_k5/loso_results.json"
    if sha(path) != "56beabc6855f4420b61150c5aad7dad550c7a5e1c1f5b8f18eba7f8df1b94179":
        raise RuntimeError("LOSO reference changed")
    data = json.loads(path.read_text())
    if data["metadata"]["status"] != "complete" or len(data["panels"]) != 12:
        raise RuntimeError("incomplete LOSO reference")
    return data


def stage(out):
    """Fetch only K5 aggregates and original test predictions; verify each hash."""
    api = HfApi()
    known = {}
    for revision, folder in [
        (K5_REV, "production_v1/k5"),
        (LOSO_REV, "leave_one_setting_out_v1/fold_checkpoints"),
    ]:
        entries = retry_transient(
            lambda revision=revision, folder=folder: list(
                api.list_repo_tree(
                    HF_REPO,
                    repo_type="dataset",
                    revision=revision,
                    path_in_repo=f"{PREFIX}/{folder}",
                    recursive=False,
                )
            ),
            what=f"list calibration inputs {folder}",
        )
        if not entries:
            raise RuntimeError(f"empty input prefix: {folder}")
        known[revision] = {entry.path for entry in entries}
    jobs = []
    for panel in references()["panels"]:
        cell = panel["cell"]
        hashes = {fold["input_sha256"] for fold in panel["folds"]}
        if len(hashes) != 1:
            raise RuntimeError("folds disagree on K5 input")
        jobs.append((K5_REV, f"{PREFIX}/production_v1/k5/{cell}.npz", hashes.pop()))
        for fold in panel["folds"]:
            path = (
                f"{PREFIX}/leave_one_setting_out_v1/fold_checkpoints/{cell}__fold{fold['fold']}.npz"
            )
            jobs.append((LOSO_REV, path, fold["prediction_sha256"]))
    if len(jobs) != 72:
        raise RuntimeError("expected 12 input banks and 60 original prediction banks")

    def fetch(job):
        revision, path, expected = job
        if path not in known[revision]:
            raise RuntimeError(f"missing pinned input: {path}")
        local = retry_transient(
            lambda: hf_hub_download(HF_REPO, path, repo_type="dataset", revision=revision),
            what=f"stage {path}",
        )
        if sha(local) != expected:
            raise RuntimeError(f"input hash differs: {path}")
        return {"revision": revision, "path": path, "local": local, "sha256": expected}

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, record in enumerate(pool.map(fetch, jobs)):
            records.append(record)
            atomic_json(out / "inputs.partial.json", records)
            print(f"[phase=stage] verified={index + 1}/{len(jobs)}", flush=True)
    atomic_json(out / "inputs.json", records)


def load_panel(model, sources):
    """Load only the K5 targets; inherit and audit the global fold mapping."""
    content = subprocess.check_output(
        ["git", "show", f"{SOURCE_SHA}:eval_results/issue_2054/shared_fold_map.json"], cwd=REPO
    )
    fold_map = json.loads(content)
    if fold_map["k"] != 5 or fold_map["seed"] != 137:
        raise RuntimeError("unexpected source fold map")
    panel = {}
    for ref in references()["panels"]:
        cell = ref["cell"]
        if not cell.endswith("__" + model):
            continue
        source = sources[f"{PREFIX}/production_v1/k5/{cell}.npz"]
        if sha(source["local"]) != source["sha256"]:
            raise RuntimeError("staged K5 input changed")
        with np.load(source["local"], allow_pickle=False) as z:
            ids = list(map(str, z["conv_id"]))
            x, y = z["v_C"].astype(np.float64), z["v_A"].astype(np.float64)
            caps = z["cap_mask"]
        if x.shape != y.shape or x.shape[1] != 3584 or len(set(ids)) != len(ids):
            raise RuntimeError("invalid K5 activation layout")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or caps.shape != (len(ids), 5):
            raise RuntimeError("invalid K5 values or rollout count")
        panel[cell] = {
            "x": x,
            "y": y,
            "ids": ids,
            "caps": caps,
            "membership": np.array([fold_map["fold_of"][cid] for cid in ids]),
        }
    if len(panel) != 6:
        raise RuntimeError("expected six settings per checkpoint")
    return panel


def calibrate(train_prediction, train_target):
    """Least-squares vector intercept, and one scalar plus vector intercept."""
    p, y = (
        np.asarray(train_prediction, dtype=np.float64),
        np.asarray(train_target, dtype=np.float64),
    )
    if p.shape != y.shape or p.ndim != 2 or len(p) < 2:
        raise ValueError("invalid calibration arrays")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite calibration values")
    pm, ym = p.mean(0), y.mean(0)
    pc, yc = p - pm, y - ym
    denominator = float(np.einsum("ij,ij->", pc, pc))
    if denominator <= 0:
        raise ValueError("constant predictions cannot identify scalar scaling")
    gain = float(np.einsum("ij,ij->", pc, yc) / denominator)
    return {"bias": ym - pm, "gain": gain, "prediction_mean": pm, "target_mean": ym}


def adapted_predictions(prediction, coefficients):
    """Apply calibration learned without any held-out target labels."""
    c = coefficients
    return {
        "bias": prediction + c["bias"],
        "bias_scale": c["gain"] * (prediction - c["prediction_mean"]) + c["target_mean"],
    }


def fit_fingerprint():
    """Bind inference code and constants, allowing presentation-only revisions."""
    functions = [sha, references, load_panel, calibrate, adapted_predictions, fit]
    payload = {
        "code": [inspect.getsource(f) for f in functions],
        "parents": {
            str(Path(path).relative_to(REPO)): sha(path)
            for path in [
                loso.__file__,
                inspect.getfile(loso.PooledMomentRidge),
                inspect.getfile(loso.combine),
                inspect.getfile(score),
                inspect.getfile(loso.SharedEighRidge),
            ]
        },
        "source_sha": SOURCE_SHA,
        "k5_revision": K5_REV,
        "loso_revision": LOSO_REV,
        "models": MODELS,
        "hf_repo": HF_REPO,
        "prefix": PREFIX,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def fit(out, model, limit):
    """Restore each source map, check parity, and calibrate on non-test folds."""
    sources = {r["path"]: r for r in json.loads((out / "inputs.json").read_text())}
    refs = {p["cell"]: p for p in references()["panels"]}
    panel = load_panel(model, sources)
    bank = loso.moments(panel, "cpu")
    fingerprint = fit_fingerprint()
    completed = 0
    for cell, p in panel.items():
        for fold in range(5):
            path = out / "folds" / f"{cell}__fold{fold}.json"
            cp = path.with_suffix(".npz")
            if path.exists():
                previous = json.loads(path.read_text())
                if (
                    previous["fit_fingerprint"] != fingerprint
                    or sha(cp) != previous["coefficient_sha256"]
                ):
                    raise RuntimeError("checkpoint fingerprint changed")
                continue
            started = time.monotonic()
            expected = refs[cell]["folds"][fold]
            audit = loso.source_audit(panel, cell, fold)
            if audit != expected["source_audit"]:
                raise RuntimeError("source folds differ from parent")
            training = loso.combine(bank, 5, drop_speaker=cell, drop_fold=fold)
            if training["n"] != audit["n_train"]:
                raise RuntimeError("source moment count differs from audit")
            frozen = loso.PooledMomentRidge(**training)
            if frozen.info()["best_lambda"] != expected["ridge"]["best_lambda"]:
                raise RuntimeError("restored source-only lambda differs")
            train, test = p["membership"] != fold, p["membership"] == fold
            pred_train = frozen.predict_np(p["x"][train])
            pred_test = frozen.predict_np(p["x"][test])
            original = sources[
                f"{PREFIX}/leave_one_setting_out_v1/fold_checkpoints/{cell}__fold{fold}.npz"
            ]
            if sha(original["local"]) != original["sha256"]:
                raise RuntimeError("original prediction hash changed")
            with np.load(original["local"], allow_pickle=False) as z:
                original_prediction = z["prediction"]
                if list(z["conv_id"]) != list(np.asarray(p["ids"])[test]):
                    raise RuntimeError("test prediction IDs differ")
            relative = float(
                np.linalg.norm(pred_test - original_prediction)
                / np.linalg.norm(original_prediction)
            )
            if relative > 1e-6:
                raise RuntimeError(f"restored-map prediction parity failed: {relative}")
            coefficients = calibrate(pred_train, p["y"][train])
            # Apply calibration to the exact PUBLISHED frozen test prediction.
            predictions = adapted_predictions(original_prediction, coefficients)
            predictions["identity_bias_target"] = p["x"][test] + (
                p["y"][train] - p["x"][train]
            ).mean(0)
            metrics = {key: score(value, p["y"][test]) for key, value in predictions.items()}
            metrics["frozen"] = score(original_prediction, p["y"][test])
            reference = expected["cohorts"]["all"]
            if (
                abs(metrics["frozen"]["r2"] - reference["metrics"]["leave_one_setting_out"]["r2"])
                > 1e-10
            ):
                raise RuntimeError("frozen R2 parity failed")
            save_npz(cp, coefficients)
            result = {
                "status": "complete",
                "cell": cell,
                "fold": fold,
                "n_calibration": int(train.sum()),
                "n_test": int(test.sum()),
                "gain": coefficients["gain"],
                "metrics": metrics,
                "own": reference["reference"]["own"],
                "six_setting_pool": reference["reference"]["six_setting_pool"],
                "source_identity_bias": reference["metrics"]["source_identity_bias"],
                "source_audit": audit,
                "calibration_ids_sha256": loso.digest_ids(list(np.asarray(p["ids"])[train])),
                "test_ids_sha256": audit["test_ids_sha256"],
                "test_overlap_calibration": 0,
                "source_input_sha256": expected["input_sha256"],
                "source_prediction": original,
                "restored_prediction_relative_l2": relative,
                "restored_ridge": frozen.info(),
                "coefficient_sha256": sha(cp),
                "script_sha256": sha(__file__),
                "fit_fingerprint": fingerprint,
                "seconds": time.monotonic() - started,
                "metadata": as_metadata_dict(git_provenance(cwd=REPO), phase="calibrate"),
            }
            atomic_json(path, result)
            print(
                f"[phase=calibrate] {cell} fold={fold} seconds={result['seconds']:.1f} "
                f"r2={metrics['frozen']['r2']:.4f}/{metrics['bias']['r2']:.4f}/{metrics['bias_scale']['r2']:.4f}",
                flush=True,
            )
            del frozen, training, pred_train, pred_test, predictions, original_prediction
            completed += 1
            if limit and completed >= limit:
                return


def collect(out):
    """Require all five folds and valid coefficient hashes for every setting."""
    panels = []
    for ref in references()["panels"]:
        cell = ref["cell"]
        rows = [json.loads((out / "folds" / f"{cell}__fold{f}.json").read_text()) for f in range(5)]
        if any(
            r["status"] != "complete" or r["cell"] != cell or r["fold"] != f
            for f, r in enumerate(rows)
        ):
            raise RuntimeError("incomplete calibration coverage")
        for row in rows:
            cp = out / "folds" / f"{cell}__fold{row['fold']}.npz"
            if row["fit_fingerprint"] != fit_fingerprint() or sha(cp) != row["coefficient_sha256"]:
                raise RuntimeError("stale or corrupted calibration checkpoint")
        panels.append(
            {
                "cell": cell,
                "folds": rows,
                "r2_mean": {
                    key: float(np.mean([r["metrics"][key]["r2"] for r in rows]))
                    for key in ["frozen", "bias", "bias_scale", "identity_bias_target"]
                },
                "own_r2": float(np.mean([r["own"]["r2"] for r in rows])),
            }
        )
    data = {
        "status": "complete",
        "panels": panels,
        "input_manifest_sha256": sha(out / "inputs.json"),
        "method": (
            "K5 LOSO source-only map; target non-test-fold vector bias / "
            "vector bias plus one scalar; fixed test sets"
        ),
        "script_sha256": sha(__file__),
        "metadata": as_metadata_dict(git_provenance(cwd=REPO), phase="collect"),
    }
    atomic_json(out / "results.json", data)
    return data


def plot(out, fig_dir):
    """Render the exact frozen and calibrated transfer means and five folds."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from explore_persona_space.analysis.c2a_plot_style import (
        INK,
        MUTED,
        ROLES,
        better_label,
        c2a_figure,
        panel_header,
        save_c2a_figure,
        set_c2a_style,
        style_axis,
    )

    data = collect(out)
    by_cell = {p["cell"]: p for p in data["panels"]}
    set_c2a_style()
    fig, fraction = c2a_figure("full", aspect=0.65)
    axes = fig.subplots(1, 2, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.22, top=0.78, wspace=0.14)
    styles = [
        ("frozen", "Frozen transfer", MUTED, "o"),
        ("bias", "+ Bias", ROLES["linear"].color, "s"),
        ("bias_scale", "+ Bias + scaling", ROLES["base_model"].color, "D"),
        ("own", "Separate map", INK, "_"),
    ]
    for ax, model, letter, title in zip(
        axes, MODELS, ["A", "B"], ["Base", "Instruction-tuned"], strict=True
    ):
        for i, (_label, prefix) in enumerate(SETTINGS):
            row = by_cell[f"{prefix}__{model}"]
            for offset, (key, _, color, marker) in zip([-0.23, 0, 0.23, 0], styles, strict=True):
                if key == "own":
                    ax.plot(
                        [i - 0.34, i + 0.34], [row["own_r2"]] * 2, color=color, lw=1.6, zorder=1
                    )
                    continue
                values = np.array([f["metrics"][key]["r2"] for f in row["folds"]])
                ax.scatter(
                    i + offset + np.linspace(-0.045, 0.045, 5),
                    values,
                    s=15,
                    marker=marker,
                    color=color,
                    alpha=0.35,
                    zorder=2,
                )
                ax.scatter(
                    i + offset,
                    row["r2_mean"][key],
                    s=78,
                    marker=marker,
                    color=color,
                    edgecolors="white",
                    linewidths=0.7,
                    zorder=3,
                )
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(np.arange(6), [s[0] for s in SETTINGS], rotation=40, ha="right")
        ax.set_xlim(-0.55, 5.55)
        ax.set_ylim(-0.52, 0.76)
        ax.set_xlabel("Held-out setting")
        style_axis(ax)
        panel_header(ax, letter, "Five source settings", title=title)
    axes[0].set_ylabel(better_label("Held-out $R^2$"))
    handles = [
        Line2D(
            [],
            [],
            marker=m,
            color=c,
            linestyle="-" if k == "own" else "None",
            markersize=9,
            label=label,
        )
        for k, label, c, m in styles
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.98), ncol=2, frameon=False
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = fig_dir / "leave_one_setting_out_calibrated"
    saved = save_c2a_figure(
        fig,
        stem,
        title="Calibration of held-out-setting transfer",
        subject=data["method"],
        creator=Path(__file__).name,
        include_width=fraction,
    )
    atomic_json(stem.with_suffix(".data.json"), data)
    atomic_json(
        stem.with_suffix(".meta.json"),
        {
            "render": saved["record"],
            "source": str(out / "results.json"),
            "source_sha256": sha(out / "results.json"),
            "code_sha256": sha(__file__),
            "fold_marks": "Five fold estimates; no confidence interval implied",
            "method": data["method"],
        },
    )
    plt.close(fig)
    print(f"[phase=plot] {saved['png']}", flush=True)


def main():
    """Dispatch input staging, fitting, or rendering as a resumable phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["stage", "fit", "plot"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--fig-dir", type=Path, default=REPO / "figures/issue_2054/k5_loso_calibration"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "stage":
        stage(args.out)
    elif args.stage == "fit":
        if args.model is None:
            parser.error("--model is required for fit")
        fit(args.out, args.model, args.limit)
    else:
        plot(args.out, args.fig_dir)
    print("[phase=done] requested stage complete", flush=True)


if __name__ == "__main__":
    main()
