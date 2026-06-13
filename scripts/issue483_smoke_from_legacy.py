#!/usr/bin/env python
"""CPU smoke for the #483 build pipeline, driven from the legacy 111-persona centroids.

Synthesizes centroid bundles (L10 + L20) in the build-script schema from the
untracked legacy tensors at
``eval_results/single_token_100_persona/centroids/centroids_layer{10,20}.pt``
(whose layer-20 1-cosine reproduces the committed legacy matrix to ~7e-7,
which also pins the tensor's persona order to the matrix's persona_names),
writes a 111-persona smoke roster, then drives the REAL CLI
(``build_canonical_persona_pool.py --build-matrices --audit``) end-to-end on
CPU twice:

1. PASS branch - clean bundles: gates must pass, exit 0, pool_v1/pool_meta
   written with the empirical-quantile centered edges.
2. FAIL branch - the L20 bundle perturbed (noise on 5 personas): the K1
   stability gate must fail (diagnosis unavailable on CPU) and the CLI must
   exit non-zero.

Afterwards point the acceptance tests at the PASS artifacts:

  EPM_CANONICAL_POOL_DIR=<workdir>/pass/data uv run pytest tests/test_persona_pool.py -v

Usage:
  uv run python scripts/issue483_smoke_from_legacy.py --workdir /tmp/i483_smoke \
      [--legacy-dir <repo>/eval_results/single_token_100_persona]
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SMOKE_LAYERS = (10, 20)


def _make_bundle(tensor: torch.Tensor, names: list[str], layer: int, out: Path) -> None:
    torch.save(
        {
            "centroids": tensor.float(),
            "persona_names": names,
            "layer": layer,
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "questions_sha256": "smoke-legacy (per-question forwards not retained)",
            "recipe": "last_prompt_token (legacy one-shot build, smoke)",
            "built_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "commit": "smoke",
        },
        out,
    )


def _setup_case(case_dir: Path, legacy_dir: Path, *, perturb: bool) -> tuple[Path, Path]:
    """Create data/ + staging/ for one smoke case; returns (data_dir, staging_dir)."""
    from run_100_persona_leakage import ALL_EVAL_PERSONAS

    data_dir, staging = case_dir / "data", case_dir / "staging"
    (data_dir / "legacy").mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)

    legacy_matrix = legacy_dir / "cosine_distance_matrix_layer20.json"
    shutil.copy(legacy_matrix, data_dir / "legacy" / "cosine_distance_matrix_layer20_478.json")
    names: list[str] = json.loads(legacy_matrix.read_text())["persona_names"]

    for layer in SMOKE_LAYERS:
        tensor = torch.load(
            legacy_dir / "centroids" / f"centroids_layer{layer}.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert tensor.shape[0] == len(names), (tensor.shape, len(names))
        if perturb and layer == 20:
            g = torch.Generator().manual_seed(483)
            noise = torch.randn(5, tensor.shape[1], generator=g) * tensor.std() * 0.5
            tensor = tensor.clone()
            tensor[:5] += noise  # blow 5 personas out of tolerance -> K1 must fire
        _make_bundle(tensor, names, layer, staging / f"centroids_v1_L{layer}.pt")

    roster = {
        name: {
            "prompt": ALL_EVAL_PERSONAS[name]["prompt"],
            "origin": "personas_100",
            "category": ALL_EVAL_PERSONAS[name].get("category", "unknown"),
            "synthetic": False,
            "sentinel": False,
        }
        for name in names
    }
    (data_dir / "roster_v1.json").write_text(
        json.dumps(
            {
                "schema_version": "cpp_v1",
                "version": "v1",
                "built_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "counts_by_origin": {"personas_100": len(roster)},
                "duplicates_resolved_first_wins": [],
                "personas": roster,
            }
        )
    )
    return data_dir, staging


def _run_cli(data_dir: Path, staging: Path) -> int:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "build_canonical_persona_pool.py"),
        "--build-matrices",
        "--audit",
        "--device",
        "cpu",
        "--data-dir",
        str(data_dir),
        "--staging-dir",
        str(staging),
        "--no-upload",
        "--allow-partial-layers",
    ]
    print(f"[smoke] $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", required=True)
    ap.add_argument(
        "--legacy-dir",
        default=str(REPO_ROOT / "eval_results" / "single_token_100_persona"),
        help="dir holding cosine_distance_matrix_layer20.json + centroids/*.pt",
    )
    args = ap.parse_args()
    workdir, legacy_dir = Path(args.workdir), Path(args.legacy_dir)

    # ── PASS branch ──────────────────────────────────────────────────────────
    data_dir, staging = _setup_case(workdir / "pass", legacy_dir, perturb=False)
    rc = _run_cli(data_dir, staging)
    assert rc == 0, f"PASS branch exited {rc} (expected 0)"
    meta = json.loads((data_dir / "pool_meta_v1.json").read_text())
    assert meta["gates"]["stability"]["pass"] is True, meta["gates"]["stability"]
    assert meta["gates"]["regression_478"]["pass"] is True, meta["gates"]["regression_478"]
    pool = json.loads((data_dir / "pool_v1.json").read_text())
    assert len(pool["personas"]) == 111, len(pool["personas"])
    edges = meta["band_presets"]["centered_v1_L20"]["edges"]
    print(
        f"[smoke] PASS branch OK: exit 0, gates pass, 111-persona pool, "
        f"centered edges {[round(e, 4) for e in edges]}"
    )

    # ── FAIL branch (forced K1) ──────────────────────────────────────────────
    data_dir_f, staging_f = _setup_case(workdir / "fail", legacy_dir, perturb=True)
    rc = _run_cli(data_dir_f, staging_f)
    assert rc != 0, "FAIL branch exited 0 (K1 stability gate should have failed)"
    meta_f = json.loads((data_dir_f / "pool_meta_v1.json").read_text())
    assert meta_f["gates"]["stability"]["pass"] is False, meta_f["gates"]["stability"]
    print(
        f"[smoke] FAIL branch OK: exit {rc}, stability gate failed as forced "
        f"(p95={meta_f['gates']['stability']['p95_abs_delta']:.4f}, "
        f"cause={meta_f['gates']['stability']['diagnosed_cause']!r})"
    )
    print(
        f"\n[smoke] now run:\n  EPM_CANONICAL_POOL_DIR={workdir / 'pass' / 'data'} "
        f"uv run pytest tests/test_persona_pool.py -v"
    )


if __name__ == "__main__":
    main()
