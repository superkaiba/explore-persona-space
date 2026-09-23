"""Recompute the published decoder-direction analysis using complete saved labels.

This is a data-loading correction to #1482, with no new model or judge calls.
The original 39 candidates, target, matching, correlation retirement and stopping
rules are retained. Three complete runs separate the published convention, the
label-loading fix alone (diagnostic), and a common resolved-label population.
An additional marginal read uses axis-specific resolved populations.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

BASE = Path("eval_results/issue_1482")
TARGET = BASE / "decoder_direction/decoder_direction_r2_fullwidth_common_universe.npy"
VOTES = BASE / "label_agreement/agreement_votes.npz"
PUBLISHED = Path(
    "figures/issue_1482/concordance_decoder_direction_common_noside_noratio/"
    "writeup_stepwise.meta.json"
)
AXIS_CLASSES = {
    "interpretable": ("yes", "no"),
    "abstraction": ("token_surface", "lexical_semantic", "abstract_contextual"),
    "content_type": ("topic", "task_format", "entity", "syntax", "operation"),
    "speaker_property": ("none", "language", "register_style", "identity_disposition"),
    "functional_role": ("input_side", "output_promoting", "mixed"),
}
EXCLUSIONS = (
    "Fires on BOTH context and answer side",
    "Fires on the answer side only",
    "Side ratio (answer-side firing fraction)",
)


def log(message: str) -> None:
    """Emit a timestamped progress observation."""
    print(f"[{datetime.now(UTC).isoformat()}] {message}", flush=True)


def digest(path: Path) -> str:
    """Hash a local input or source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> None:
    """Atomically persist a strict JSON checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stepwise(vecs: dict, y: np.ndarray, families: dict, sw, path: Path) -> dict:
    """Run the inherited selection algorithm and checkpoint every completed round."""
    controls, rounds = [], []
    candidates = list(vecs)
    result = {"n": len(y), "rounds": rounds, "status": "running"}
    for k in range(sw.MAX_ROUNDS):
        cells, bins = sw.crossed_strata([vecs[c] for c in controls], len(y))
        frac = sw.pair_fraction(cells, len(y))
        scores = []
        for name in candidates:
            c = float(sw.concordance(vecs[name], y, cells))
            if np.isfinite(c):
                scores.append({"name": name, "family": families[name], "c": c})
        if not scores:
            result["stop_reason"] = "no estimable candidates"
            break
        winner = max(scores, key=lambda row: abs(row["c"] - 0.5))
        sizes = [len(cell) for cell in cells]
        row = {
            "round": k,
            "controls": list(controls),
            "n": len(y),
            "bins": bins,
            "n_cells": len(cells),
            "median_cell": int(np.median(sizes)),
            "min_cell": min(sizes),
            "pair_frac": frac,
            "scores": scores,
            "winner": winner["name"],
            "winner_c": winner["c"],
            "retired_as_siblings": [],
        }
        rounds.append(row)
        log(f"{path.stem} round {k}: {winner['name']}, c-0.5={winner['c'] - 0.5:+.6f}")
        if abs(winner["c"] - 0.5) < sw.STOP_EFFECT:
            row["winner"] = ""
            result["stop_reason"] = "effect below inherited threshold"
        elif frac < sw.STOP_MIN_PAIR_FRAC:
            row["winner"] = ""
            result["stop_reason"] = "pair fraction below inherited threshold"
        else:
            retired = []
            for name in candidates:
                if name == winner["name"]:
                    continue
                rho = float(spearmanr(vecs[winner["name"]], vecs[name]).statistic)
                if np.isfinite(rho) and abs(rho) >= sw.DECLUSTER_RHO:
                    row["retired_as_siblings"].append({"name": name, "rho": rho})
                    retired.append(name)
            controls.append(winner["name"])
            candidates = [c for c in candidates if c != winner["name"] and c not in retired]
        save(path, result)
        if not row["winner"]:
            break
    result.setdefault("stop_reason", "inherited maximum of 14 rounds")
    result["status"] = "complete"
    save(path, result)
    return result


def verify_published(result: dict, published: dict) -> dict:
    """Require full numerical reproduction, including all conditional rounds."""
    assert len(result["rounds"]) == len(published["rounds"])
    max_delta = 0.0
    for actual, expected in zip(result["rounds"], published["rounds"], strict=True):
        for key in ("winner", "controls", "n", "bins", "n_cells", "median_cell", "min_cell"):
            assert actual[key] == expected[key], (key, actual[key], expected[key])
        assert abs(actual["pair_frac"] - expected["pair_frac"]) < 1e-12
        aa = {row["name"]: row["c"] for row in actual["scores"]}
        ee = {row["name"]: row["c"] for row in expected["scores"]}
        assert aa.keys() == ee.keys()
        for name in aa:
            delta = abs(aa[name] - ee[name])
            assert delta < 1e-12, (actual["round"], name, delta)
            max_delta = max(max_delta, delta)
        assert [r["name"] for r in actual["retired_as_siblings"]] == [
            r["name"] for r in expected["retired_as_siblings"]
        ]
    return {"rounds_reproduced": len(result["rounds"]), "max_absolute_score_difference": max_delta}


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    """Validate saved inputs, reproduce the reference, and run the label corrections."""
    source_root = Path(cfg.source_root).resolve()
    out = Path(cfg.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source_root / "scripts"))
    sw = importlib.import_module("issue1482_concordance_stepwise")
    wf = sw.WF
    # Old helper modules carry historical absolute paths. Bind them explicitly.
    wf.REPO = source_root
    wf.SB.PROJECT_ROOT = source_root
    wf.TARGET_R2 = source_root / TARGET
    published = json.loads((source_root / PUBLISHED).read_text())
    input_paths = [
        TARGET,
        VOTES,
        PUBLISHED,
        BASE / "predictor_battery/fullwidth_covariates_v2.npz",
        BASE / "predictor_battery/fullwidth_discrete_covariates.npz",
        BASE / "predictor_battery/fullwidth_matrix.npz",
        BASE / "predictor_battery/shapley_blocks_densesae_ridge_k24.json",
        BASE / "run_length/run_length_perfeature.npz",
        Path("scripts/issue1482_concordance_stepwise.py"),
        Path("scripts/issue1482_concordance_writeup_figs.py"),
        Path("scripts/issue1482_concordance_fig.py"),
        Path("scripts/issue1482_shapley_blocks.py"),
    ]
    manifest = {
        "analysis_script_sha256": digest(Path(__file__)),
        "source_root": str(source_root),
        "inputs": {str(p): digest(source_root / p) for p in input_paths},
        "source_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip(),
        "analysis_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip(),
        "selection": {
            "max_rounds": sw.MAX_ROUNDS,
            "stop_effect": sw.STOP_EFFECT,
            "stop_pair_fraction": sw.STOP_MIN_PAIR_FRAC,
            "retire_rho": sw.DECLUSTER_RHO,
            "target_cells": sw.TARGET_CELLS,
            "min_bins": sw.MIN_BINS,
            "max_bins": sw.MAX_BINS,
            "minimum_cell_size": 120,
        },
        "corrected_primary": "common population resolved on all five axes",
        "sensitivity": "marginal concordance on each candidate's available resolved population",
        "unknown_labels": ["unlabeled", "unresolved", "unclear"],
        "exclusions": list(EXCLUSIONS),
        "scope": "existing decoder-direction R2, no refitting or model/judge calls",
        "uncertainty": "descriptive point estimates; no new intervals or significance claims",
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Changed inputs: use a new output"
    else:
        save(manifest_path, manifest)
    log("Loading the original covariates and reproducing the published population")
    battery = wf.battery()
    ok, y = battery["ok"], battery["y"]
    assert len(y) == published["n_rows"] == 120716
    names = [r["name"] for r in battery["rows"] if r["name"] not in EXCLUSIONS]
    assert names == [r["name"] for r in published["rounds"][0]["scores"]]
    assert len(names) == 39
    old_vectors = {name: battery["vecs"][name] for name in names}
    families = {r["name"]: r["family"] for r in battery["rows"]}
    votes = np.load(source_root / VOTES, allow_pickle=True)
    old_matrix = np.load(
        source_root / BASE / "predictor_battery/fullwidth_matrix.npz", allow_pickle=True
    )
    full_vectors = {name: value.copy() for name, value in old_vectors.items()}
    masks = {}
    coverage = {"base_population": len(y), "axes": {}}
    common = np.ones(len(y), dtype=bool)
    for axis, classes in AXIS_CLASSES.items():
        all_labels = votes[f"{axis}__label"].astype(str)
        assert all_labels.shape == (131072,)
        assert set(np.unique(all_labels)) <= set(classes) | {"unlabeled", "unresolved", "unclear"}
        prior_labels = old_matrix[f"label__{axis}"].astype(str)
        assert np.array_equal(prior_labels, all_labels[old_matrix["feat_ids"]])
        resolved = votes[f"{axis}__resolved"]
        best = votes[f"{axis}__best"]
        survived = votes[f"{axis}__n_surv"]
        assert np.all(best <= survived) and np.all(survived <= 5)
        assert np.array_equal(resolved, best >= 3)
        labels = all_labels[ok]
        known = np.isin(labels, classes) & resolved[ok]
        masks[axis] = known
        common &= known
        coverage["axes"][axis] = {
            "resolved_usable": int(known.sum()),
            "unresolved": int(np.sum(labels == "unresolved")),
            "unlabeled": int(np.sum(labels == "unlabeled")),
            "unclear": int(np.sum(labels == "unclear")),
            "unanimous_five_of_five": int(np.sum((best[ok] == 5) & (survived[ok] == 5))),
            "class_counts": {level: int(np.sum(labels == level)) for level in classes},
        }
        for key, name in wf.JUDGED.items():
            ax, level = key.split(":", 1)
            if ax == axis and name in full_vectors:
                full_vectors[name] = (labels == level).astype(float)
    coverage["common_resolved_population"] = int(common.sum())
    coverage["excluded_from_common_population"] = int((~common).sum())
    save(out / "coverage.json", coverage)
    np.savez_compressed(
        out / "populations.npz",
        base_feature_ids=np.flatnonzero(ok),
        common_feature_ids=np.flatnonzero(ok)[common],
        **{f"resolved__{ax}": mask for ax, mask in masks.items()},
    )
    runs = {}
    for key, vecs, targets in [
        ("published_reproduction", old_vectors, y),
        ("full_labels_unknown_zero_diagnostic", full_vectors, y),
        ("full_labels_common_resolved", {n: v[common] for n, v in full_vectors.items()}, y[common]),
    ]:
        path = out / f"{key}.json"
        if path.exists() and json.loads(path.read_text()).get("status") == "complete":
            runs[key] = json.loads(path.read_text())
            log(f"Reusing completed {key}, with identical manifest")
        else:
            runs[key] = stepwise(vecs, targets, families, sw, path)
        if key == "published_reproduction":
            verification = verify_published(runs[key], published)
            save(out / "verification.json", verification)
            log(f"Reference reproduction passed: {verification}")
    axis_by_name = {name: key.split(":", 1)[0] for key, name in wf.JUDGED.items()}
    available = []
    for name, vector in full_vectors.items():
        mask = masks[axis_by_name[name]] if name in axis_by_name else np.ones(len(y), dtype=bool)
        score = float(sw.concordance(vector[mask], y[mask], [np.arange(int(mask.sum()))]))
        assert np.isfinite(score), name
        available.append({"name": name, "n": int(mask.sum()), "c": score})
    save(out / "available_resolved_marginal.json", {"scores": available})
    comparison = []
    marginals = {
        key: {r["name"]: r["c"] - 0.5 for r in run["rounds"][0]["scores"]}
        for key, run in runs.items()
    }
    selected = {
        key: {
            r["winner"]: {"round": r["round"], "score": r["winner_c"] - 0.5}
            for r in run["rounds"]
            if r["winner"]
        }
        for key, run in runs.items()
    }
    available_by_name = {r["name"]: r for r in available}
    for name in names:
        row = {"name": name, "family": families[name]}
        for key in runs:
            row[f"{key}_marginal"] = marginals[key][name]
            row[f"{key}_selected"] = selected[key].get(name)
        row["available_resolved_marginal"] = available_by_name[name]["c"] - 0.5
        row["available_resolved_n"] = available_by_name[name]["n"]
        comparison.append(row)
    save(out / "comparison.json", {"properties": comparison})
    with (out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    for path in input_paths:
        assert digest(source_root / path) == manifest["inputs"][str(path)], f"Input changed: {path}"
    save(
        out / "completion.json",
        {
            "status": "complete",
            "finished_utc": datetime.now(UTC).isoformat(),
            "manifest_sha256": digest(manifest_path),
            "verification": verification,
            "coverage": coverage,
            "runs": {
                key: {
                    "n": run["n"],
                    "stop_reason": run["stop_reason"],
                    "selection": [
                        {"name": r["winner"], "round": r["round"], "score": r["winner_c"] - 0.5}
                        for r in run["rounds"]
                        if r["winner"]
                    ],
                }
                for key, run in runs.items()
            },
        },
    )
    log(f"All runs complete, verified inputs unchanged. Results: {out}")


if __name__ == "__main__":
    main()
