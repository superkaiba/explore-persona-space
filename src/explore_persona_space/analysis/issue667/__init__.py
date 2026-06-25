"""Issue #667 — forward-pass preview of the leakage-predictor gate chain.

Analysis-only (no training). Tests the five trained-model gate-chain
assumptions A3.6-A3.10 of the project's leakage-predictor theory on #537's
existing contrastive LoRA adapters, via a forward-pass activation-extraction
sweep benchmarked against #537's measured cross-context leakage matrix ``G``.

The importable, GPU-free heart lives in :mod:`gate_chain` (the whitened gate,
Sigma_c inverse + lambda sweep, the realized activation gate, the B3 reduction
unit test, and the A3.6-A3.10 statistics). The GPU forward-pass extraction +
the per-assumption CPU runner live in the ``scripts/issue667_*`` entrypoints.
"""

from __future__ import annotations

# In-scope behaviors (plan §3): marker EXCLUDED (saturated, R3-4 degrades the
# write direction); refusal EXCLUDED (#537 noise-limited, 0.7x floor). marker
# is run only as a caveated supplement.
IN_SCOPE_BEHAVIORS: tuple[str, ...] = ("em", "sycophancy", "fact")
SUPPLEMENT_BEHAVIORS: tuple[str, ...] = ("marker",)

# Read layers (plan §11): L14 primary, {7, 21} the free depth supplement.
PRIMARY_LAYER: int = 14
SUPPLEMENT_LAYERS: tuple[int, ...] = (7, 21)
ALL_LAYERS: tuple[int, ...] = (7, 14, 21)

# Qwen-2.5-7B dims (asserted at extraction time).
HIDDEN_SIZE: int = 3584
N_LAYERS: int = 28

BASE_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"

# HF repos (read #537 adapters + G, #658 store; write #667 analysis tensors).
HF_MODEL_REPO: str = "superkaiba1/explore-persona-space"
HF_DATA_REPO: str = "superkaiba1/explore-persona-space-data"

# Reused-artifact paths (Hub-verified at implementation time, plan §5/§10).
G_TENSOR_PATH: str = "issue537_context_generalization/G_tensor/G_tensor.npz"
G_META_LOCAL: str = "eval_results/issue_537/G_tensor/G_meta.json"
SIGMA_C_PATH: str = "issue658_theory_assumptions/store/sigma_c.pt"
R_B_PATH: str = "issue658_theory_assumptions/store/r_b.pt"
STORE_MANIFEST_PATH: str = "issue658_theory_assumptions/store/store_manifest.json"

# The load-bearing #537 + #658 pins (plan §5 (f); verified this session).
EXPECTED_G_META_GIT_COMMIT: str = "34f2502c656cd804524f2a3d4d5231270aaf0664"
EXPECTED_REGISTRY_HASH: str = "f12061d6c2f6c0b2969d900bda45c1ca23d77dffda29ddb58d95c835859efd39"
EXPECTED_STORE_PROBE_POOL_HASH: str = (
    "ad687becec266286549aaaa1af3b35e246d593e012e233564e58ff75fb015dd7"
)

# r_b column map: in-scope behavior -> the #658 r_b.pt key (plan §5). fact is
# ABSENT from #658's r_b.pt (4 cols) -> re-extracted fresh in Phase 1.
RB_COLUMN_FOR_BEHAVIOR: dict[str, str | None] = {
    "em": "broad_em",
    "sycophancy": "sycophancy",
    "fact": None,  # not in #658 r_b.pt; re-extract via the #594 diff-in-means recipe
    "marker": None,  # supplement; r_B not used for the marker companion
}

# r_b recipe (plan §10/§11): diff-in-means (Persona Vectors, arXiv 2507.21509).
RB_RECIPE: str = "diffmeans"

# Sigma_c regularization default (plan §11): ridge fraction-of-mean-eigenvalue.
SIGMA_C_LAMBDA_FRACTION: float = 1e-2

# The HF write prefix for #667 per-cell tensors (Upload Policy: analysis tensors).
HF_ANALYSIS_TENSORS_PREFIX: str = "issue667_gate_chain_preview/analysis_tensors"

# Output dir for the per-assumption JSONs (eval_results, JSON/text only).
LOCAL_OUT_DIR: str = "eval_results/issue_667"

WANDB_PROJECT: str = "issue667"
