#!/usr/bin/env python
# ruff: noqa: RUF003
# Intentional Unicode (ρ, σ, λ, Σ, Δ, ×) in scientific docstrings + logs.
"""Task #653 OFF-POD cluster bootstrap (plan §9 "(off-pod, VM, CPU)").

Runs on the VM AFTER ``i653_dispatch.py`` Provision-2 upload lands the Δx
tensors (the pod is terminated first — this is CPU-only, no GPU dependency).
For each per-cell Δx cloud it computes the full 10 000-resample cluster
bootstrap CI on every registered spectral DV (top-share λ, PR_λ) and writes
``bootstrap_ci_<cell>.json`` alongside the dx geometry JSONs, then refreshes
the per-cell ``deciding_ci`` ambiguity flag in ``cross_arm_verdict.json`` at the
full bootstrap depth (the on-pod analyze used a small n_boot=200 flag).

The cluster unit is the (context-persona, question) row (plan §6 "resampling
the rows of the Δx matrix and recomputing the SVD"); each Δx row IS one
(persona, question), so this is the standard row bootstrap. n_resamples=10000,
seed=653. Source: plan §6 statistical plan + §10 reproducibility card.

Inputs (read order): the uploaded HF ``analysis_tensors/<cell>.npz`` if
``--from-hf`` is passed, else the local ``eval_results/issue_653/armB/
dx_tensors/<cell>.npz``. The orchestrator runs this at the Step 9a interpretation
phase:

    uv run python scripts/issue_653/i653_postpod_bootstrap.py \\
        --out-root eval_results/issue_653

    # or pull tensors from HF first (if the local copies were cleaned):
    uv run python scripts/issue_653/i653_postpod_bootstrap.py \\
        --out-root eval_results/issue_653 --from-hf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from explore_persona_space.experiments import issue_653 as i653
from explore_persona_space.experiments.issue_653 import spectral

DECIDING_DV = "top_share_lambda"  # the §3.4 deciding quantity for the ambiguity flag
BOOTSTRAP_DVS = ("top_share_lambda", "pr_lambda")


def _pull_tensors_from_hf(out_root: Path) -> Path:
    """Download the uploaded Δx tensors from the HF data repo into the local
    tensors dir (used when the pod's local copies were cleaned). Fail-loud."""
    from huggingface_hub import hf_hub_download, list_repo_files

    prefix = f"issue653_{i653.HF_UPLOAD_PREFIX}/analysis_tensors"
    files = [
        f
        for f in list_repo_files(i653.HF_DATA_REPO, repo_type="dataset", revision="main")
        if f.startswith(prefix) and f.endswith(".npz")
    ]
    if not files:
        raise FileNotFoundError(
            f"--from-hf: no analysis_tensors/*.npz under {prefix} on {i653.HF_DATA_REPO}. "
            "Did Provision-2 upload run?"
        )
    dest = out_root / "armB" / "dx_tensors"
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        local = hf_hub_download(i653.HF_DATA_REPO, f, repo_type="dataset", revision="main")
        (dest / Path(f).name).write_bytes(Path(local).read_bytes())
    print(f"  [bootstrap] pulled {len(files)} Δx tensors from HF -> {dest}", flush=True)
    return dest


def bootstrap_cell(npz_path: Path) -> dict:
    """Full 10k cluster-bootstrap CI for one cell's Δx cloud, all spectral DVs."""
    npz = np.load(npz_path)
    cloud = npz["cloud"].astype(np.float64)
    cluster_ids = np.arange(cloud.shape[0])  # row bootstrap (§6 clustering unit)
    out: dict[str, dict] = {}
    for dv in BOOTSTRAP_DVS:
        out[dv] = spectral.cluster_bootstrap_dv(
            cloud,
            cluster_ids,
            dv,
            n_boot=i653.BOOTSTRAP_B,  # 10000; Source: plan §6 / §10
            seed=i653.BOOTSTRAP_SEED,  # 653
        )
    out["n_rows"] = int(cloud.shape[0])
    return out


def _refresh_ambiguity_flags(out_root: Path, per_cell_ci: dict[str, dict]) -> None:
    """Re-classify the cross_arm_verdict.json deciding_ci ambiguity flag at the
    full bootstrap depth (the on-pod analyze used a shallow n_boot=200)."""
    verdict_path = out_root / "cross_arm_verdict.json"
    if not verdict_path.exists():
        print(f"  [bootstrap] no {verdict_path}; skipping ambiguity-flag refresh", flush=True)
        return
    grid = json.loads(verdict_path.read_text())
    for vd in grid.get("verdicts", []):
        cid = vd.get("cell_id")
        ci = per_cell_ci.get(cid, {}).get(DECIDING_DV)
        if ci is None or vd.get("spectrum_underdetermined"):
            continue
        lo, hi = ci["ci_low"], ci["ci_high"]
        vd["deciding_ci"] = [lo, hi]
        # The most load-bearing thresholds for the label (mirror classify_cell).
        ambiguous = any(
            lo <= thr <= hi
            for thr in (i653.TOP_SHARE_LOWRANK, i653.PR_LAMBDA_LOWRANK, i653.PR_LAMBDA_H3)
        )
        vd["ambiguous"] = bool(ambiguous)
        vd["deciding_ci_n_boot"] = i653.BOOTSTRAP_B
    grid["bootstrap_refreshed_n_boot"] = i653.BOOTSTRAP_B
    verdict_path.write_text(json.dumps(grid, indent=1))
    print(f"  [bootstrap] refreshed ambiguity flags in {verdict_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task #653 off-pod cluster bootstrap.")
    parser.add_argument("--out-root", default="eval_results/issue_653", help="output root")
    parser.add_argument(
        "--from-hf",
        action="store_true",
        help="pull the Δx tensors from the HF data repo first (if local copies were cleaned)",
    )
    args = parser.parse_args(argv)
    out_root = Path(args.out_root)

    tensors_dir = out_root / "armB" / "dx_tensors"
    if args.from_hf:
        tensors_dir = _pull_tensors_from_hf(out_root)
    if not tensors_dir.exists():
        raise FileNotFoundError(
            f"no Δx tensors under {tensors_dir}. Run the dispatcher dx phase (uploads "
            f"to HF analysis_tensors/) and pass --from-hf, or run on the VM with the "
            f"local eval_results present."
        )

    npz_files = sorted(tensors_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"no *.npz Δx tensors in {tensors_dir}")

    boot_dir = out_root / "armB"
    per_cell_ci: dict[str, dict] = {}
    written: list[str] = []
    for npz_path in npz_files:
        cell_id = npz_path.stem
        ci = bootstrap_cell(npz_path)
        per_cell_ci[cell_id] = ci
        payload = {
            "cell_id": cell_id,
            "n_boot": i653.BOOTSTRAP_B,
            "seed": i653.BOOTSTRAP_SEED,
            "cluster_unit": "(context-persona, question) row (§6)",
            "bootstrap": ci,
        }
        out_path = boot_dir / f"bootstrap_ci_{cell_id}.json"
        out_path.write_text(json.dumps(payload, indent=1))
        written.append(str(out_path))
        ts = ci[DECIDING_DV]
        print(
            f"  [bootstrap] {cell_id}: top_share CI "
            f"[{ts['ci_low']:.3f}, {ts['ci_high']:.3f}] (n_boot={ts['n_boot']})",
            flush=True,
        )

    _refresh_ambiguity_flags(out_root, per_cell_ci)
    print(f"  [bootstrap] wrote {len(written)} bootstrap_ci_*.json under {boot_dir}", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
