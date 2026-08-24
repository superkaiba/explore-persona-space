"""Issue #2333 constants — pins, pair universe, arms, and model specs.

Everything a POD-side driver needs that would otherwise live in committed
``eval_results/`` artifacts is hardcoded here with provenance comments: pods
run partial clones whose sparse cones exclude other issues' ``eval_results/``
(gotchas.md, #1739/#2211), so the driver must never read those files pod-side.
VM-side analysis re-verifies these constants against the committed artifacts.
"""

from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# HF data-repo revision pins (plan §10; resolved 2026-08-16 on this session).
# ---------------------------------------------------------------------------
DATA_REPO = "superkaiba1/explore-persona-space-data"
# #2162 grid/bank artifacts (issue2162_ladder/ prefix) — plan §10 pin.
PIN_2162 = "dc8108ab84f33695bbc769da0e6e8e2327f51eeb"
# #2094 bank artifacts (issue2094_fmetric/ prefix) — plan §10 pin.
PIN_2094 = "cfedd60af9e061e9efa42a1573ebeab9ec790eca"
# Post-fu1 revision carrying BOTH issue2094_fmetric/fu1_conf1/ (30 files) and
# issue2094_fmetric/judge_raw_fu1/ (77 files); component last_commits
# 6b832224d594 (21:47) and c5d7de56d182 (23:20) on 2026-08-06 — the later one
# resolves both stems (verified via list_repo_tree at this revision).
PIN_FU1 = "c5d7de56d182a0eb6ea1b36f22851fedf5969d8c"

HF_PREFIX = "issue2333_snowball"
# Write-repo override (parent pattern: overflow repos on file-count fallback).
DATA_WRITE_REPO_ENV = "EPM_2333_DATA_WRITE_REPO"

# Realized remote paths (probed via list_repo_tree at the pins, 2026-08-16).
R2162_BANK_JSON = "issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json"  # @ PIN_2162
R2162_VC_BANK = "issue2162_ctxinfo/analysis_tensors/vc_bank/vc_bank.pt"  # @ PIN_2162
R2162_GRID_ROLLOUTS = "issue2162_ctxinfo/raw_completions/grid"  # @ PIN_2162
R2094_VC_BANK = "issue2094_singlepos/analysis_tensors/vc_bank/vc_bank.pt"  # @ PIN_2094
FU1_CONF1_PREFIX = "issue2094_singlepos/raw_completions/fu1_regen_confirm/conf1"  # @ PIN_FU1
FU1_JUDGE_RAW_PREFIX = "issue2094_singlepos/raw_completions/judge_raw_fu1"  # @ PIN_FU1

# ---------------------------------------------------------------------------
# Models (plan §4.1). q35 decoder blocks nest under .model.language_model
# (multimodal wrapper) — resolved additively by extraction._resolve_decoder_blocks.
# ---------------------------------------------------------------------------
MODELS = {
    "q25": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "n_layers": 28,
        "hidden": 3584,
        "read_layer": 26,  # #2162 READ_LAYER (issue2162_analysis.py)
    },
    "q35": {
        "model_id": "Qwen/Qwen3.5-9B",
        "n_layers": 32,
        "hidden": 4096,
        "read_layer": 30,  # plan §6: ~0.94 depth-matched analogue of 26/28
    },
}

# ---------------------------------------------------------------------------
# Pair universe.
# S1: the 5 surviving #2162 cells (eval_results/issue_2162/f_metrics/
# best_cells.json, key "best_cells", verified this session) x 36 pairs each.
# ---------------------------------------------------------------------------
S1_CELLS = (
    "load_instr_format_l5",
    "load_instr_format_l3",
    "conflict_format_fwd",
    "instr_format",
    "conflict_format_rev",
)
S1_PAIRS_PER_CELL = 36
S2_CELL = "s2_matched_query"  # the single S2 grid cell (15 matched-query pairs)

# The 10 well-separated S2 pair ids (plan §5: fu1 conf1 wellsep set) and the 5
# excluded ones — re-derived and cross-checked by issue2333_analysis.py
# --phase s2-ce-derive against the vendored fu1_conf1_confirmation.json.
S2_WELLSEP_N = 10
S2_EXCLUDED_N = 5

# ---------------------------------------------------------------------------
# S2 shuffled-donor map (fallback derangement, seed 23330 — plan §4.2 / A15).
# Recovery of the parent's realized derangement from
# eval_results/issue_2094/f_metrics/null_cells.jsonl (the plan §4.2 PRIMARY)
# is AMBIGUOUS — measured 2026-08-16 (r2 re-probe): restricting donor_pair_id
# to pair-type ("mq--*") donors, 5 of 15 matched-query pairs carry >1 distinct
# donor across the parent's null blocks (all 15 are multi-donor once centroid
# nulls are included), so no single derangement reproduces the parent. Per
# plan §4.2's NAMED FALLBACK the realized map is a fresh seeded derangement
# over the 15 matched-query pair ids, seed 23330 (`seeded_derangement` below —
# the map `issue2333_run.build_donor_maps` actually installs).
# ---------------------------------------------------------------------------
S2_DERANGEMENT_SEED = 23330
BOOTSTRAP_SEED = 23330  # analysis bootstrap (plan §10)

# q35 environment resolution (plan §4.4 B-1): #2329's REALIZED transformers
# pin — origin/issue-2329 scripts/issue2329_run.py gate 0b asserts ==5.15.0
# and issue2329_dispatch.sh installs it pod-side (`TRANSFORMERS_PIN`); the
# repo pin (4.57.6) FAILS AutoConfig for qwen3_5 (plan §12 A8). envcheck
# (q35) asserts the RUNTIME version equals this pin; every later q35 phase
# asserts env identity with the envcheck that passed.
Q35_TRANSFORMERS_PIN = "5.15.0"

# #2329 artifact prefix (plan §4.4 B-1 fitness probe): the reuse-or-selfgen
# decision probes these classes at ONE resolved data-repo revision.
HF_PREFIX_2329 = "issue2329_q35rerun"
FITNESS_2329_CLASSES = (
    f"{HF_PREFIX_2329}/analysis_tensors/vc_bank",  # bank slice (q35 re-tokenized)
    f"{HF_PREFIX_2329}/raw_completions/anchors",  # fresh q35 anchors
    f"{HF_PREFIX_2329}/raw_completions/ce_control",  # stage-1 ce-control rows
)


def seeded_derangement(items: list[str], seed: int) -> dict[str, str]:
    """Seeded derangement (no fixed point) over ``items`` — the plan §4.2 S2
    fallback (seed 23330). Mirrors bank2094._seeded_derangement's rejection
    sampling; fails loud if no derangement is found in 10000 attempts."""
    assert len(items) >= 2, items
    rng = random.Random(seed)
    for _ in range(10000):
        perm = list(items)
        rng.shuffle(perm)
        if all(a != b for a, b in zip(items, perm, strict=True)):
            return dict(zip(items, perm, strict=True))
    raise RuntimeError(f"no derangement of {len(items)} items in 10000 attempts")


# ---------------------------------------------------------------------------
# Arms (plan §4.2): 12 = {patch, prefill} x k in {1,2,3} x scheme {med, bstart};
# each run in variants {steered, null}. Slug shape: <kind><k>_<scheme>.
# ---------------------------------------------------------------------------
ARM_KINDS = ("patch", "prefill")
ARM_KS = (1, 2, 3)
ARM_SCHEMES = ("med", "bstart")
VARIANTS = ("steered", "null")

ARM_SLUGS = tuple(
    f"{kind}{k}_{scheme}" for kind in ARM_KINDS for k in ARM_KS for scheme in ARM_SCHEMES
)


def parse_arm(slug: str) -> tuple[str, int, str]:
    """``patch2_med`` -> ("patch", 2, "med"). Fails loud on unknown slugs."""
    head, scheme = slug.rsplit("_", 1)
    for kind in ARM_KINDS:
        if head.startswith(kind):
            k = int(head[len(kind) :])
            assert k in ARM_KS and scheme in ARM_SCHEMES, slug
            return kind, k, scheme
    raise ValueError(f"unknown arm slug: {slug}")


def expected_grid_slugs() -> set[str]:
    """The 144 production grid-block shard slugs (byte-parity with
    ``issue2162_run.block_slug(f"{cell}|{arm}|{variant}")`` — test-pinned).

    Torch-free so the VM-side judge/analysis completeness gate (pre-spend,
    plan §7 / code-review r1 Major 3) can enumerate without the pod driver.
    """
    cells = (*S1_CELLS, S2_CELL)
    slugs = {
        f"{cell}|{arm}|{variant}".replace("|", "__").replace(".", "p")
        for cell in cells
        for arm in ARM_SLUGS
        for variant in VARIANTS
    }
    assert len(slugs) == len(cells) * len(ARM_SLUGS) * len(VARIANTS), len(slugs)
    return slugs


def expected_ce_control_slugs() -> set[str]:
    """The 12 q35 ce_control shard slugs ((5 S1 cells + 1 S2 cell) x 2 variants)."""
    return {
        f"{cell}|ce_replace|{variant}".replace("|", "__").replace(".", "p")
        for cell in (*S1_CELLS, S2_CELL)
        for variant in VARIANTS
    }


# ---------------------------------------------------------------------------
# Generation / capture knobs (plan §4.2, §10).
# ---------------------------------------------------------------------------
GRID_DRAWS = 5  # K draws per (pair, arm, variant)
GRID_TEMPERATURE = 1.0
MAX_NEW_TOKENS = 2048
GEN_BATCH = 16
SEED_BASE = 42
DONOR_MAX_NEW_TOKENS = 8  # greedy donor openings (plan §4.2)
DONOR_K_MAX = 3  # capture states/token ids for answer positions 1..3
ANCHOR_DRAWS = 10  # q35 anchors (floors/ceilings), temp 1.0
CE_CONTROL_DRAWS = 5  # q35 banked-ce control (plan §4.4 B1)

# Capture-parity two-bar (plan §7, gotchas.md bf16 calibration, #779):
PARITY_EARLY_LAYERS = (0, 1, 2, 3)
PARITY_EARLY_COS_MIN = 0.999
PARITY_FLAT_COS_MIN = 0.995
