#!/usr/bin/env python3
"""#603 V1 validation gate (CPU, VM) — BLOCKS pod provisioning.

Downloads the 18 #551 shift tensors from the private HF data repo
(``issue551_shift_reextract/analysis_tensors/shifts/``), recomputes the
published #551 control statistics with THIS branch's code (the ported
``svd_direction_constancy`` math + the new ``write_decomposition``
loader path), and asserts agreement with the committed
``eval_results/issue_551/controls/{mean_resp,norm_alignment}.json`` to
``<= 1e-3``:

- per-cell ``mean_cos_to_U1_mean_resp``, ``s_top1_frac_mean_resp`` and
  the full per-persona ``cos_to_U1_mean_resp`` profile (6 same-variant
  cells);
- per-cell per-persona slot-read ``norms`` + ``cos_to_U1`` and the
  deterministic ``spearman_rho_norm_vs_cos`` (Monte-Carlo p NOT
  re-checked — rng-path dependent, not a published headline).

Also records (calibration observation, non-binding): the marker-arm
``medical_doctor`` CMF from :func:`decompose_write` at the
mean-over-response read — expected LOW, ballpark of the published
0.24-0.32 cos-to-top-direction band.

Exit 0 = gate PASS (pod may be provisioned). Exit 3 = gate FAIL.

Run (VM)::

    uv run python scripts/issue603_validate_on_551.py \
        --out eval_results/issue_603/v1_gate.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="i603_v1_gate")

from explore_persona_space.analysis.svd_direction_constancy import (  # noqa: E402
    assemble_M,
    spearman_rho,
    svd_summary,
)
from explore_persona_space.analysis.write_decomposition import decompose_write  # noqa: E402

PRIVATE_REPO = "superkaiba1/explore-persona-space-data-private"
TENSOR_PREFIX = "issue551_shift_reextract/analysis_tensors/shifts"
CONTROLS_DIR = PROJECT_ROOT / "eval_results" / "issue_551" / "controls"
PARENT_SVD_DIR = PROJECT_ROOT / "eval_results" / "issue_521" / "svd"
TOL = 1e-3
SOURCE_PERSONA = "medical_doctor"

VARIANTS = ("same", "base", "on_policy")
ARMS = ("marker", "em")
SEEDS = (42, 137, 256)
SAME_CELLS = [f"same_{arm}_seed{seed}" for arm in ARMS for seed in SEEDS]
ALL_CELLS = [f"{v}_{a}_seed{s}" for v in VARIANTS for a in ARMS for s in SEEDS]


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _download_tensors(dest: Path) -> Path:
    """Per-file hf_hub_download of the 18 .pt files (mirrors issue551_controls)."""
    import shutil

    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    for cell in ALL_CELLS:
        fname = f"{cell}.pt"
        target = dest / fname
        if target.exists() and target.stat().st_size > 0:
            logger.info("[skip] %s already downloaded", fname)
            continue
        local = hf_hub_download(
            repo_id=PRIVATE_REPO,
            filename=f"{TENSOR_PREFIX}/{fname}",
            repo_type="dataset",
        )
        shutil.copy2(local, target)
        logger.info("[downloaded] %s (%.2f MB)", fname, target.stat().st_size / 1e6)
    return dest


def _load_shifts(shifts_dir: Path, cell: str) -> dict[str, dict[str, torch.Tensor]]:
    payload = torch.load(shifts_dir / f"{cell}.pt", map_location="cpu", weights_only=False)
    return payload["shifts"]


def _parent_persona_order(cell: str) -> list[str]:
    path = PARENT_SVD_DIR / f"{cell}.json"
    with path.open() as f:
        return list(json.load(f)["persona_order"])


def main() -> int:
    """Run the V1 gate; write the gate JSON; exit 0 on PASS, 3 on FAIL."""
    ap = argparse.ArgumentParser(description="#603 V1 gate vs #551 published controls")
    ap.add_argument("--out", default="eval_results/issue_603/v1_gate.json")
    ap.add_argument(
        "--tensors-dir",
        default="eval_results/issue_603/_v1_tensors",
        help="Local cache dir for the downloaded #551 .pt files (gitignored bulk).",
    )
    args = ap.parse_args()

    shifts_dir = _download_tensors(Path(args.tensors_dir))

    ref_mean_resp = json.loads((CONTROLS_DIR / "mean_resp.json").read_text())["per_cell"]
    ref_norm_align = json.loads((CONTROLS_DIR / "norm_alignment.json").read_text())["per_cell"]

    deviations: list[dict] = []
    per_cell: dict[str, dict] = {}
    calibration: dict[str, dict] = {}

    def _check(cell: str, name: str, got: float, want: float) -> None:
        dev = abs(got - want)
        if dev > TOL:
            deviations.append({"cell": cell, "stat": name, "got": got, "want": want, "dev": dev})

    for cell in SAME_CELLS:
        shifts = _load_shifts(shifts_dir, cell)
        persona_order = _parent_persona_order(cell)

        # Slot read (parent delta_v key) — norms + cos profile + rho.
        m_slot, order = assemble_M(shifts, persona_order=persona_order)
        assert order == persona_order, (order, persona_order)
        svd_slot = svd_summary(m_slot)
        norms = np.linalg.norm(m_slot, axis=0)
        ref_na = ref_norm_align[cell]
        for i, p in enumerate(persona_order):
            _check(cell, f"norm[{p}]", float(norms[i]), float(ref_na["norms"][p]))
            _check(
                cell,
                f"cos_to_U1[{p}]",
                float(svd_slot["cos_to_U1"][i]),
                float(ref_na["cos_to_U1"][p]),
            )
        rho = spearman_rho(norms, np.asarray(svd_slot["cos_to_U1"], dtype=np.float64))
        _check(
            cell, "spearman_rho_norm_vs_cos", float(rho), float(ref_na["spearman_rho_norm_vs_cos"])
        )

        # Mean-over-response read — top-share + cos profile.
        m_mr, _ = assemble_M(shifts, persona_order=persona_order, use_mean_resp=True)
        svd_mr = svd_summary(m_mr)
        ref_mr = ref_mean_resp[cell]
        _check(
            cell,
            "mean_cos_to_U1_mean_resp",
            float(np.mean(svd_mr["cos_to_U1"])),
            float(ref_mr["mean_cos_to_U1_mean_resp"]),
        )
        _check(
            cell,
            "s_top1_frac_mean_resp",
            float(svd_mr["s_top1_frac"]),
            float(ref_mr["s_top1_frac_mean_resp"]),
        )
        for i, p in enumerate(persona_order):
            _check(
                cell,
                f"cos_to_U1_mean_resp[{p}]",
                float(svd_mr["cos_to_U1"][i]),
                float(ref_mr["cos_to_U1_mean_resp"][i]),
            )

        per_cell[cell] = {
            "n_personas": len(persona_order),
            "mean_cos_to_U1_mean_resp": float(np.mean(svd_mr["cos_to_U1"])),
            "s_top1_frac_mean_resp": float(svd_mr["s_top1_frac"]),
            "spearman_rho_norm_vs_cos": float(rho),
        }

        # Calibration observation: medical_doctor CMF via the NEW
        # decompose_write (non-binding; sanity context for the analyzer).
        dec = decompose_write(shifts, source=SOURCE_PERSONA, key="delta_v_mean_resp")
        calibration[cell] = {
            "source": SOURCE_PERSONA,
            "cmf_mean_resp": dec["cmf"],
            "norm": dec["norm"],
            "shared_norm": dec["shared_norm"],
            "residual_norm": dec["residual_norm"],
            "cmf_svd": dec["cmf_svd"],
            "cmf_svd_unitnorm": dec["cmf_svd_unitnorm"],
            "n_bystanders": dec["n_bystanders"],
        }
        logger.info(
            "[cell %s] checks ok so far; calib cmf(medical_doctor)=%.4f norm=%.3f",
            cell,
            dec["cmf"],
            dec["norm"],
        )

    gate_pass = len(deviations) == 0
    payload = {
        "meta": {
            "issue": 603,
            "git_commit": _git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tol": TOL,
            "tensors_source": f"hf://{PRIVATE_REPO}/{TENSOR_PREFIX}",
            "controls_reference": str(CONTROLS_DIR),
            "env_versions": {
                pkg: __import__("importlib.metadata", fromlist=["version"]).version(pkg)
                for pkg in ("torch", "numpy")
            },
        },
        "gate_pass": gate_pass,
        "n_deviations": len(deviations),
        "deviations": deviations[:50],
        "per_cell": per_cell,
        "calibration_medical_doctor_cmf": calibration,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("[wrote] %s", out_path)

    if not gate_pass:
        logger.error(
            "[V1 GATE FAIL] %d deviations > %.0e — DO NOT provision the pod", len(deviations), TOL
        )
        return 3
    logger.info("[V1 GATE PASS] all recomputed statistics within %.0e of published values", TOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
