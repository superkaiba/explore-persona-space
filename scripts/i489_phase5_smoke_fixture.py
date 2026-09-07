"""Build a tiny CPU smoke fixture for scripts/i489_phase5_analyze.py.

Writes:
  eval_results/issue_489/phase1/cosine_per_layer.json   (24x24 dummy)
  eval_results/issue_489/phase4/per_cell/G_<i>__<j>_frac0.25.json (552 off-diag)

Each Phase 4 cell has delta_g sampled to be roughly correlated with the
cosine distance so the H3 bootstrap has a nontrivial signal to push
against. After running this fixture builder, the smoke command is:

    uv run python scripts/i489_phase5_analyze.py --seed 4242 --fracs 0.25 \\
        --bootstrap-n 30 --smoke

It writes analysis.json into eval_results/issue_489/phase5/.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from explore_persona_space.experiments.i489_contexts import (
    ICL_CONTEXTS,
    SP_CONTEXTS,
    UNION_CONTEXTS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE1_DIR = REPO_ROOT / "eval_results" / "issue_489" / "phase1"
PHASE4_DIR = REPO_ROOT / "eval_results" / "issue_489" / "phase4" / "per_cell"
PHASE5_DIR = REPO_ROOT / "eval_results" / "issue_489" / "phase5"
HEADLINE_LAYER = 21

SEED = 4242
FRAC = 0.25

random.seed(SEED)
cids = [c.cid for c in UNION_CONTEXTS]
icl_cids = {c.cid for c in ICL_CONTEXTS}
sp_cids = {c.cid for c in SP_CONTEXTS}

# Phase 1: random cosine-sim matrix in [0.4, 0.95]; symmetric, diag=1.
cos = {}
for i, ci in enumerate(cids):
    cos[ci] = {}
    for j, cj in enumerate(cids):
        if i == j:
            cos[ci][cj] = 1.0
        elif cj in cos and ci in cos[cj]:
            cos[ci][cj] = cos[cj][ci]
        else:
            cos[ci][cj] = round(random.uniform(0.4, 0.95), 4)

PHASE1_DIR.mkdir(parents=True, exist_ok=True)
(PHASE1_DIR / "cosine_per_layer.json").write_text(
    json.dumps({"cos_sim_per_layer": {str(HEADLINE_LAYER): cos}}, indent=2)
)

# Phase 4: 552 off-diagonal cells. delta_g = a + b * cos_dist + noise (per arm).
PHASE4_DIR.mkdir(parents=True, exist_ok=True)
n_written = 0
for ti in cids:
    for tj in cids:
        if ti == tj:
            continue
        cos_dist = 1.0 - cos[ti][tj]
        # Inject ICL vs SP arm-specific slope so the resampler has work to do.
        if tj in icl_cids:
            base = 0.3 * cos_dist + 0.05
        elif tj in sp_cids:
            base = 0.6 * cos_dist - 0.02
        else:
            base = 0.0
        noise = random.gauss(0.0, 0.05)
        delta_g = base + noise
        # `length_fn` reads cell["L_R"] (or similar) — provide a few common
        # keys phase5 may read so partialling-on-length works.
        payload = {
            "frac": FRAC,
            "seed": SEED,
            "T_i": ti,
            "T_j": tj,
            "delta_g": float(delta_g),
            "L_R": 100 + random.randint(0, 50),
            "n_responses_T_i": 8,
            "n_responses_T_j": 8,
        }
        fname = f"G_{ti}__{tj}_frac{FRAC}.json"
        (PHASE4_DIR / fname).write_text(json.dumps(payload))
        n_written += 1

assert n_written == 24 * 23, f"expected 552 off-diag cells, wrote {n_written}"

print(f"wrote phase1 fixture -> {PHASE1_DIR / 'cosine_per_layer.json'}")
print(f"wrote {n_written} phase4 cells -> {PHASE4_DIR}")
print(f"phase5 outputs will land in {PHASE5_DIR}")
print()
print("smoke command:")
print(
    f"  uv run python scripts/i489_phase5_analyze.py --seed {SEED} "
    f"--fracs {FRAC} --bootstrap-n 30 --smoke"
)
