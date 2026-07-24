"""#1586 fu caveat-fix round: seed-pooled Δnorm lattice + marker transfer fractions.

Analyzer round `caveat-fix-marker-dosematch-impolite-lr-deconfound`. Computes the
plan-v7 §3 registered SEED-POOLED con-regime paired mean-shift-norm differences
(question-cluster bootstrap, paired indices, seed-stratified pooling — the
executed run's `issue1586_pooled_lattice.py` convention verbatim, adapted to the
fu cell names) from the round's own capture stores, plus the po-regime marker
pooled reads and the install-normalized marker transfer fractions at the
matched-dose arms.

Inputs
------
- capture stores (own-text + shared-text) staged from the HF data repo under
  ``data/issue_1586/hf_dl/fu_caveatfix/issue1586_methodgen/fu_caveatfix/analysis_tensors/``
- marker panel slot reads staged under ``/tmp/i1586fu/.../marker_panel/``

Output: ``eval_results/issue_1586/caveat-fix-marker-dosematch-impolite-lr-deconfound/
geometry/pooled_lattice_fu.json``

Batched draws only (``_mu_norm_draws`` — the subset-sum GEMM helper); no
per-draw Python loop. ~14 store loads + 20 reads, VM CPU, minutes.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue1586_geometry import _mu_norm_draws, bootstrap_index_matrix  # noqa: E402

from explore_persona_space.experiments.issue_1112 import geometry as geo  # noqa: E402

WT = Path(__file__).resolve().parents[1]
STAGE = WT / "data/issue_1586/hf_dl/p11_json_fu/issue1586_methodgen/fu_caveatfix/analysis_tensors"
OUT = WT / "eval_results/issue_1586/caveat-fix-marker-dosematch-impolite-lr-deconfound/geometry"
SLOT_READS = Path("/tmp/i1586fu/issue1586_methodgen/fu_caveatfix/marker_panel")
N_BOOT = 2000

# fu pair map: (behavior, regime) -> [(ft_cell, lora_cell) per seed]
PAIRS = {
    ("mk", "con"): [
        ("mk-pers-ft2e6-con-s42", "mk-pers-lora-con-s42"),
        ("mk-pers-ft2e6-con-s137", "mk-pers-lora-con-s137"),
    ],
    ("mk", "po"): [
        ("mk-pers-ft2e6-po-s42", "mk-pers-lora-po-s42"),
        ("mk-pers-ft2e6-po-s137", "mk-pers-lora-po-s137"),
    ],
    ("imp", "con"): [
        ("imp-pers-ft-con-s42", "imp-pers-lora5e6-con-s42"),
        ("imp-pers-ft-con-s137", "imp-pers-lora5e6-con-s137"),
    ],
}
LAYER = {"mk": 25, "imp": 14}


def pooled_read(tree: Path, base_tree: Path, beh: str, regime: str, arm: str) -> dict:
    """Seed-pooled paired Δnorm (FT − LoRA): per seed, per-draw diffs on one
    shared question-cluster index matrix; pooled draw = mean over seeds."""
    layer = LAYER[beh]
    base = geo.load_store(base_tree / f"base_{beh}" / "pooled.pt")
    cluster_ids = [f"{c}__{q}" for c, q in geo._row_keys(base)]
    idx = bootstrap_index_matrix(cluster_ids, n_boot=N_BOOT, seed=geo.BOOT_SEED)
    per_seed_draws, per_seed_points = [], []
    for ft_cell, lora_cell in PAIRS[(beh, regime)]:
        ft = geo.load_store(tree / ft_cell / "pooled.pt")
        lo = geo.load_store(tree / lora_cell / "pooled.pt")
        cloud_ft = geo.delta_cloud(ft, base, arm, layer)
        cloud_lo = geo.delta_cloud(lo, base, arm, layer)
        per_seed_draws.append(_mu_norm_draws(cloud_ft, idx) - _mu_norm_draws(cloud_lo, idx))
        per_seed_points.append(
            float(np.linalg.norm(cloud_ft.mean(axis=0)))
            - float(np.linalg.norm(cloud_lo.mean(axis=0)))
        )
    pooled = np.mean(per_seed_draws, axis=0)
    return {
        "point": float(np.mean(per_seed_points)),
        "ci_low": float(np.nanquantile(pooled, 0.025)),
        "ci_high": float(np.nanquantile(pooled, 0.975)),
        "n_boot": N_BOOT,
        "resampling": "paired, seed-stratified pooled",
        "per_seed_points": per_seed_points,
    }


def main() -> None:
    own = STAGE / "capture"
    tf = STAGE / "capture_tf"
    out: dict = {"norm": {}, "n_boot": N_BOOT, "boot_seed": geo.BOOT_SEED}
    for beh, regime in PAIRS:
        for arm in ("prefix", "context", "response"):
            key = f"own/{beh}/{regime}/{arm}/L{LAYER[beh]}"
            out["norm"][key] = pooled_read(own, own, beh, regime, arm)
            print(key, json.dumps(out["norm"][key]), flush=True)
        # shared-text control: response arm only (tf tree has no base_*; the
        # own-text base store is the shared base — identical base generations)
        key = f"tf/{beh}/{regime}/response/L{LAYER[beh]}"
        out["norm"][key] = pooled_read(tf, own, beh, regime, "response")
        print(key, json.dumps(out["norm"][key]), flush=True)

    # marker install-normalized transfer fractions at the matched arms
    frac = {}
    if SLOT_READS.exists():
        for cell_dir in sorted(SLOT_READS.iterdir()):
            s = json.loads((cell_dir / "slot_reads.json").read_text())
            src = s["by_context"]["persona_software_engineer"]
            frac[cell_dir.name] = {
                "margin_fraction": s["pooled_nonsource_delta_margin"] / src["delta_margin_mean"],
                "logp_fraction": s["pooled_nonsource_delta_logp"] / src["delta_logp_mean"],
                "source_delta_margin": src["delta_margin_mean"],
                "source_delta_logp": src["delta_logp_mean"],
                "pooled_nonsource_delta_margin": s["pooled_nonsource_delta_margin"],
                "pooled_nonsource_delta_logp": s["pooled_nonsource_delta_logp"],
            }
    out["marker_transfer_fractions"] = frac

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pooled_lattice_fu.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT / 'pooled_lattice_fu.json'}")


if __name__ == "__main__":
    main()
