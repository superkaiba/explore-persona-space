"""P-Fit driver for #823 follow-up `inconsistent-origin-persona-ladder`.

Fits the registered 7-arm ladder (k in {1,2,4,8,16} + own/plain anchors) under
two protocols — P1 parent-parity 5-fold pooled-GCV ridge (headline) and P2
single-split val-selected ridge (n-ladder + per-persona control) — computes the
paired per-context bootstrap, evaluates the registered verdict lattice, and
persists every registered output (plan v10 sections 3/4.3/6/6.5/7).

Requirement mapping (plan v10 + the binding adversarial reconciliation):
  R1 split integrity — `checked_fixed_split` REFUSES any n_ctx != 5000 at the
     call site: `fixed_split` slices `perm[n_test+n_val : n_test+n_val+n_train]`,
     so a 4998-row call silently truncates train to 3598 (no exception) and
     shares only 502/1000 test rows with the banked permutation. The banked
     convention is split-over-5000 THEN per-subset intersection; newly-invalid
     rows drop AFTER the split from whichever subset they land in.
  R2 drop accounting — expected drops are ~12 total (the parent's 2 + ~10 new),
     NOT <=2; per-subset drop counts are reported. Registered check
     `realized_train >= 3584` at split materialization; on failure the
     d-boundary rung is reported UNREALIZABLE and labeled as such in
     `ladder_singlesplit_p2.json` — never silently coerced to a smaller subset.
  R3 banked-split assert at the PRE-DROP level — `predrop_banked_split_check`
     asserts the pre-drop permutation slices against the banked
     `single_split_protocol.json` record (parent counts 3600/400/1000; the
     4998-mask intersection reproduces 3599/399/1000 exactly). A POST-drop hash
     comparison fails loudly on a HEALTHY run (a newly-invalid row landing in
     test makes realized test a proper subset of the banked test set).
  R4 verdict lattice — BOTH interval endpoints (`ci_low_delta_mean`,
     `ci_high_delta_mean`) are persisted in `ladder_r2_p1.json`;
     `lattice_verdict` implements the registered predicate VERBATIM (point
     conjuncts retained AND interval conjuncts added, so disjointness holds by
     construction). Boundary fixture (committed unit test): point 0.02 with
     CI [-0.02, 0.06] MUST return Intermediate, not Flat.
  R5 one primary estimator — `primary_estimator: "gcv-pure-parent-parity"` is
     encoded in `ladder_r2_p1.json`; the uncapped parent-parity pure-GCV
     selection (`issue779_fitter_fair_comparison._gcv_solve`, which takes NO cap
     parameter) is the unconditional primary for every arm/layer, the bootstrap
     interval, and the lattice. The 0.9-dof-capped re-selection is computed
     ONLY as a labeled sensitivity (sensitivity-only keys) when the realized
     mask makes the cap bindable. Variants are never mixed within a contrast.
  R6 conditioning — Degrades, Flat, AND Intermediate are each gated on the
     registered joint distinctness predicate (M1 AND M2); predicate failure
     maps EVERY outcome to "manipulation failure" with the numeric lattice
     retained as descriptive only (`conditioned_interpretation`).

Also registered (same reconciliation):
  - absolute-error (sum ss_res) ladder + fixed-reference-denominator re-read
    (`R2_fixed(k) = 1 - sum ss_res(k) / sum ss_tot(k=1)`) reported alongside
    R2 — pure re-reductions of the persisted per-context arrays;
  - G1 reduction-convention pin: the banked `ridge_r2_by_arm.json` stores
    per-FOLD R2 arrays, so its quoted values are fold-MEANS; G1 compares
    LIKE-FOR-LIKE (fold-mean vs the embedded fold-mean constants) and a
    PRE-CHECK confirms the fold-mean-vs-pooled gap is far under tolerance
    BEFORE the gate verdict (a near-tolerance gap is a convention mismatch to
    fix, never a reproduce failure);
  - k=8 is the PRIMARY matched-capacity contrast for G_mix (~50 val rows vs
    ~25 at k=16); k=16 reported alongside with per-fit lambda + edge flags;
  - identity+learned-bias baseline AND kNN retrieval (euclidean + cosine,
    chance stated) on EVERY fitted cell, per-persona / matched-n cells included.

Vectorization: ONE shared per-(layer, fold) float64 Gram eigh
(`FFC._factorize`) reused across all 7 targets and both lambda grids; the
bootstrap is a batched re-reduction over persisted per-context (ss_res, ss_tot)
arrays (chunked fancy-indexed sums — no per-draw Python loop over the pool).
Checkpoints: per-layer (P1) and per-rung / per-k (P2) npz+sidecar chunks in the
durable out-root, resume keyed on generating parameters (never recomputed float
bytes). Cells with n_train < 3585 are labeled estimator-degenerate for absolute
reads (valid as matched-regime contrasts).

Smoke blind-spot enumeration (plan-sanctioned downgrades, disclosed):
  - G1 reproduce gate is SKIPPED at smoke n (reproducing banked full-n R2 is
    impossible at n=10) — the smoke PASS does not certify bundle/mask/solver
    numerical reproduction; G1 runs at production n.
  - the pre-drop banked-split check (predrop_banked_split_check: membership
    shas + counts) is SKIPPED at smoke n — it quantifies over the FULL 4998-id
    parent mask, unreachable at n=10; at production n it runs immediately after
    mask materialization, BEFORE the gates and the P1 fits.
  - P2 val-selection is DEGENERATE at smoke n (n_val ~ 1 => every pooled-R2
    score non-finite): smoke cells keep grid[0] and are LABELED
    `val_selection_degenerate`; production fails loud instead
    (val_select_lambda degenerate_ok). >=2 validation contexts would need a
    larger upstream capture smoke slice (only context 4 of 0-9 lands in val).
  - P1 GCV fits at smoke n (n_train ~ 8 << d) are estimator-degenerate — the
    smoke asserts shapes/finiteness/pipeline only; gate verdicts (G2 tolerance,
    drop-rule abort, wall abort, d-boundary) are INFORMATIONAL at smoke scale
    (production-n-calibrated gates must not kill the smoke leg).
  - no substituted implementations (smoke runs the production solvers, the
    production staging, and the production upload path to the `_smoke` prefix)
    and no production-only third-party imports.

Kill criteria implemented here (designed halts, distinct rcs, report JSONs —
never a bare rc=1): mask-integrity (rc=5), fits-wall (rc=6), solver-parity
after CPU-eigh fallback + contingency (rc=7).

Usage:
  uv run python scripts/issue823_ladder_fits.py --import-check
  uv run python scripts/issue823_ladder_fits.py --list-arms
  uv run python scripts/issue823_ladder_fits.py --smoke          # pod, 10 ctx
  uv run python scripts/issue823_ladder_fits.py                  # pod, full
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # creds + shared-VM thread caps BEFORE torch import (sibling pattern)

import argparse
import dataclasses
import hashlib
import json
import logging
import pathlib
import sys
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.model_selection import KFold

# Repo root on sys.path so `scripts.*` sibling imports resolve in script mode
# (the parent's `_ensure_repo_root_on_syspath` pattern; sys.path[0] is scripts/).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.experiments.issue_779.fit_h import ridge_fit_predict  # noqa: E402
from explore_persona_space.experiments.issue_823.run_823 import (  # noqa: E402
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    log_phase,
    write_sentinel,
)
from explore_persona_space.orchestrate.hub import (  # noqa: E402
    _upload_folder_filtered,
    retry_transient,
)
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)
from scripts import issue779_fitter_fair_comparison as FFC  # noqa: E402
from scripts import issue779_percontext_recon as PR  # noqa: E402
from scripts import issue823_single_split_protocol as SSP  # noqa: E402
from scripts.issue823_ladder_capture import (  # noqa: E402
    FOLLOWUP_LABEL,
    fetch_gen_inputs,
    group_paths,
    load_pair_rows,
    resolve_dataset_revision,
    verify_gen_sentinel,
)
from scripts.issue823_ladder_gen import (  # noqa: E402
    DATA_REPO,
    HF_PREFIX,
    K_ARMS,
    MASK_GATE_SCHEMA_ID,
    N_CONTEXTS_FULL,
    N_PERSONAS,
    PARENT_PREFIX,
    PARENT_REV,
    _require_canonical_upload,
    write_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue823_ladder_fits")

# ── Registered constants (plan v10 sections 3 / 4.3 / 6 / 7 / 9) ─────────────
PRIMARY_ESTIMATOR = "gcv-pure-parent-parity"  # R5: the ONE registered P1 primary
READ_OUT_LAYERS = (14, 26, 17)  # section-3 delta_mean layer set (registered order)
TRAIT_LAYERS = {"evil": 14, "sycophancy": 26, "hallucination": 17}  # descriptive labels
P2_LAYERS = (14, 17, 19, 26)
LADDER_KS = (1, 2, 4, 8, 16)
ARM_NAMES = ("k1", "k2", "k4", "k8", "k16", "own", "plain")
LADDER_ARM_NAMES = ("k1", "k2", "k4", "k8", "k16")
P1_N_FOLDS = 5
P1_FOLD_SEED = 0  # KFold(5, shuffle=True, random_state=0) — parent parity (run_823 phase 4)
P1_GRID_PARAMS = ("logspace", -2, 4, 13)  # == FFC.LAMBDAS; hash generating params, not bytes
P2_GRID_PARAMS = ("logspace", -2, 8, 21)  # == SSP.LAMBDAS_WIDE
BOOT_N = 10_000
BOOT_SEED = 0
BOOT_CHUNK = 1_000
N_CTX = 5000
HIDDEN = 3584
D_BOUNDARY = 3584  # interpolation boundary; n_train < 3585 => estimator-degenerate
SPLIT_SEED = 42
SPLIT_NOMINAL = (3600, 400, 1000)  # fixed_split(n_ctx=5000, ...) nominal sizes
# R3: banked split record (eval_results/issue_823/single_split_protocol/
# single_split_protocol.json `split_realized`) — PRE-DROP-level constants.
BANKED_SPLIT_PARENT = {"n_train": 3600, "n_val": 400, "n_test": 1000}
BANKED_SPLIT_ARMS_MASKED = {"n_valid": 4998, "n_train": 3599, "n_val": 399, "n_test": 1000}
# MEMBERSHIP pins for the pre-drop permutation slices (counts alone pass a
# count-preserving membership swap): sha256 of each sorted int64 id array from
# fixed_split(5000, 3600, 400, 1000, seed=42) — bit-exact integer inputs, safe
# to hash (machine-stable; NOT a recomputed-float hash).
BANKED_SPLIT_PREDROP_SHA256 = {
    # SHA_PIN_DOMAIN: INDEX — sha256 of each sorted-int64 id array (_ids_sha)
    "train": "f1f133f4f7565c4accb009143a405abfd97302c1717c52d1fd6cbdc75e6b4928",
    # SHA_PIN_DOMAIN: INDEX
    "val": "2e307fb2d1b74c82752d9460d131a3c1949860e9f0eefe6a82d15cee9f1e0613",
    # SHA_PIN_DOMAIN: INDEX
    "test": "b9377786b24bc9c1c360303fdb8fac86c0097d264479de1dca3c23dd1047d31d",
}
# plan v13 section-7 kill 1 (L619-623, L1175): the per-arm abort counts
# INTEGRITY-CLASS (non-refusal) NEW invalid rows only; refusal-attributed drops
# are governed by the separate refusal-attrition budget (disposition 2, <=500
# total) and never trip this gate. Threshold value unchanged (1% of corpus).
NEW_DROPS_ABORT_PER_ARM = 50
P2_RUNGS_BELOW_TOP = (3584, 2400, 1800, 900, 450, 225)
PER_PERSONA_KS = (2, 4, 8, 16)
GMIX_PRIMARY_K = 8  # PRIMARY matched-capacity contrast (~50 val rows vs ~25 at k=16)
SMALL_CELL_TEST_FLOOR = 100  # test pools under this carry the `small-cell` label (section 6)
PLANNED_FITS_WALL_H = 1.5  # plan section-9 P-Fit row
CONTINGENCY_EXTRA_WALL_H = 2.0  # plan section-8 priced canonical-solver contingency
FITS_WALL_ABORT_FACTOR = 2.0
H1_DELTA_THRESHOLD = 0.05
H2_BAND = 0.03
G1_TOL = 0.005
G2_MAX_REL_TOL = 1e-4  # DV-level (parent's 1e-8 bit-parity gate FAILS healthy at ~1.7e-5)
G2_DELTA_R2_TOL = 1e-4
G2_SLICES = ((14, 0), (26, 2), (17, 4))  # >=3 (layer, fold) production-shape slices
DOF_CAP = 0.9  # sensitivity-only re-selection cap (NEVER the primary)
SMOKE_N_CONTEXTS = 10
SMOKE_BOOT_N = 200
MIN_CELL_FLOORS = {"train": 2, "val": 1, "test": 1}
# Designed-abort rcs (distinct; gen halt=3, capture wall=4)
RC_MASK_ABORT = 5
RC_FITS_WALL_ABORT = 6
RC_SOLVER_PARITY_ABORT = 7

# Production pod out-root (run_823 phase convention). A smoke out-root at or
# under this path would alias the production run's artifacts/sentinel — refused
# at argparse time (smoke_root_aliases_production).
PROD_POD_OUT_ROOT = pathlib.Path("/workspace/eps/out/issue823_ladder")

# G1 embedded reference constants — fold-MEAN R2 recomputed from the banked
# `eval_results/issue_823/ridge_r2_by_arm.json` per-fold arrays at
# implementation time (r2_by_layer is trait-independent within an arm;
# banked arm keys: own=A_prime, plain=B2). LIKE-FOR-LIKE: the banked JSON
# stores per-FOLD R2, so these are fold-means, compared against the refit's
# fold-mean — never against the pooled statistic (reduction-convention pin).
G1_BANKED_FOLD_MEAN = {
    ("own", 14): 0.5988290884118461,
    ("own", 26): 0.6080161650744824,
    ("own", 17): 0.6260118332433705,
    ("plain", 14): 0.5845818410435927,
    ("plain", 26): 0.5559560884817063,
    ("plain", 17): 0.5911089915529909,
}
G1_BANKED_ARM_KEYS = {"own": "A_prime", "plain": "B2"}
BANKED_R2_JSON = _REPO_ROOT / "eval_results/issue_823/ridge_r2_by_arm.json"
BANKED_SPLIT_JSON = (
    _REPO_ROOT / "eval_results/issue_823/single_split_protocol/single_split_protocol.json"
)

# Parent input pins (staged local-first, else HF at PARENT_REV / BUNDLE_REV)
BUNDLE_REV = SSP.BUNDLE_REV
BUNDLE_PATH_IN_REPO = "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
PARENT_ARM_FILES = {"own": "v_a_prime.pt", "plain": "v_b2.pt"}
PARENT_MASK_PATH_IN_REPO = f"{PARENT_PREFIX}/raw_completions/phase1/common_valid_idx.json"
M1_TFIDF_BAR = 0.8  # parent arm-identity bar: flag if mean within-context cosine > 0.8
M2_FLOOR_MULT = 2.0
M2_MIN_PERSONAS = 12
M2_MIN_LAYERS = 2

assert tuple(K_ARMS) == LADDER_KS, (K_ARMS, LADDER_KS)
assert HIDDEN == EXPECTED_HIDDEN and N_CTX == N_CONTEXTS_FULL


# ── R1/R3: split construction + banked pre-drop assert ───────────────────────


def checked_fixed_split(n_ctx: int = N_CTX) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R1: the banked split is over the FULL 5000-context space — refuse anything else.

    `fixed_split` slices `perm[n_test+n_val : n_test+n_val+n_train]`; on a
    4998-element permutation numpy SILENTLY truncates train to 3598 (no
    exception) and the permutation shares only 502/1000 test rows with the
    banked one. Returns sorted disjoint (train, val, test) index arrays.
    """
    if n_ctx != N_CTX:
        raise ValueError(
            f"checked_fixed_split: n_ctx={n_ctx} != {N_CTX}. The banked convention splits "
            "over the FULL 5000-context space THEN intersects each subset with the valid "
            "set; calling fixed_split on a masked count silently truncates the train slice "
            "(numpy slice-past-end) and destroys row-identity with the banked permutation."
        )
    tr, va, te = SSP.fixed_split(
        n_ctx=n_ctx,
        n_train=SPLIT_NOMINAL[0],
        n_val=SPLIT_NOMINAL[1],
        n_test=SPLIT_NOMINAL[2],
        seed=SPLIT_SEED,
    )
    sizes = (len(tr), len(va), len(te))
    assert sizes == SPLIT_NOMINAL, f"pre-drop split sizes {sizes} != nominal {SPLIT_NOMINAL}"
    union = np.concatenate([tr, va, te])
    assert len(np.unique(union)) == sum(SPLIT_NOMINAL), "split subsets overlap"
    return tr, va, te


def _ids_sha(ids: np.ndarray) -> str:
    """sha256 of a sorted int64 id array (bit-exact integer input — safe to hash)."""
    return hashlib.sha256(np.sort(np.asarray(ids, dtype=np.int64)).tobytes()).hexdigest()


def predrop_banked_split_check(
    pre_split: tuple[np.ndarray, np.ndarray, np.ndarray],
    parent_valid_ids: np.ndarray,
) -> dict:
    """R3: assert the PRE-DROP permutation slices against the banked record.

    The banked run recorded parent counts 3600/400/1000 over 5000 and, after
    intersecting each subset with its 4998-row valid set, 3599/399/1000. A
    post-drop comparison against the NEW mask fails loudly on a HEALTHY run
    (newly-invalid rows make realized subsets proper subsets of the banked
    ones), so the assert is pinned here — pre-NEW-drop, banked-mask level.
    """
    tr, va, te = pre_split
    parent_set = set(np.asarray(parent_valid_ids).tolist())
    assert len(parent_set) == BANKED_SPLIT_ARMS_MASKED["n_valid"], (
        f"parent valid set has {len(parent_set)} ids != banked "
        f"{BANKED_SPLIT_ARMS_MASKED['n_valid']}"
    )
    got_parent = {"n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    if got_parent != BANKED_SPLIT_PARENT:
        raise RuntimeError(
            f"pre-drop split counts {got_parent} != banked parent record {BANKED_SPLIT_PARENT}"
        )
    got_sha = {"train": _ids_sha(tr), "val": _ids_sha(va), "test": _ids_sha(te)}
    if got_sha != BANKED_SPLIT_PREDROP_SHA256:
        bad = sorted(k for k in got_sha if got_sha[k] != BANKED_SPLIT_PREDROP_SHA256[k])
        raise RuntimeError(
            f"pre-drop split MEMBERSHIP drifted on subset(s) {bad}: sorted-id sha256 "
            f"{got_sha} != banked BANKED_SPLIT_PREDROP_SHA256 — counts match but the "
            "permutation's subset membership differs from the banked run's"
        )
    got_masked = {
        "n_valid": len(parent_set),
        "n_train": int(sum(1 for i in tr if i in parent_set)),
        "n_val": int(sum(1 for i in va if i in parent_set)),
        "n_test": int(sum(1 for i in te if i in parent_set)),
    }
    if got_masked != BANKED_SPLIT_ARMS_MASKED:
        raise RuntimeError(
            f"pre-drop slices intersected with the banked 4998 mask give {got_masked} != "
            f"banked arms_masked record {BANKED_SPLIT_ARMS_MASKED} — the permutation does "
            "not reproduce the banked run's own convention (split-over-5000 then intersect)"
        )
    return {
        "banked_parent": BANKED_SPLIT_PARENT,
        "banked_arms_masked": BANKED_SPLIT_ARMS_MASKED,
        "predrop_sha256": {
            "train": _ids_sha(tr),
            "val": _ids_sha(va),
            "test": _ids_sha(te),
        },
        "check": "PASS (pre-drop level)",
    }


def realized_split_with_drops(
    pre_split: tuple[np.ndarray, np.ndarray, np.ndarray],
    parent_valid_ids: np.ndarray,
    new_valid_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    """R2: realized subset = pre-drop permutation slice INTERSECT new mask.

    Newly-invalid rows drop AFTER the split from whichever subset they land in.
    Per-subset drop counts are reported, split into parent-era drops (the 2
    rows outside the parent 4998 mask) and NEW drops (~10 expected).
    """
    parent_set = set(np.asarray(parent_valid_ids).tolist())
    new_set = set(np.asarray(new_valid_ids).tolist())
    assert new_set <= parent_set, "new mask must be a subset of the parent mask"
    subsets: dict[str, np.ndarray] = {}
    drops: dict[str, dict] = {}
    for name, ids in zip(("train", "val", "test"), pre_split):
        keep = np.array([i for i in ids if i in new_set], dtype=int)
        parent_dropped = [int(i) for i in ids if i not in parent_set]
        new_dropped = [int(i) for i in ids if i in parent_set and i not in new_set]
        subsets[name] = keep
        drops[name] = {
            "pre_drop_n": int(len(ids)),
            "realized_n": int(len(keep)),
            "parent_drops": len(parent_dropped),
            "new_drops": len(new_dropped),
            "new_dropped_ids": new_dropped,
        }
    drops["total_drops"] = int(
        sum(d["parent_drops"] + d["new_drops"] for d in drops.values() if isinstance(d, dict))
    )
    return subsets, drops


def d_boundary_disposition(realized_train_n: int) -> dict:
    """R2: registered check `realized_train >= 3584` at split materialization.

    Failure => the d-boundary rung is UNREALIZABLE (labeled, remaining rungs
    proceed) — never silently coerced to a smaller subset.
    """
    ok = realized_train_n >= D_BOUNDARY
    return {
        "check": "realized_train >= 3584",
        "realized_train": int(realized_train_n),
        "d_boundary": D_BOUNDARY,
        "pass": bool(ok),
        "d_rung_status": "realizable" if ok else "UNREALIZABLE",
    }


def p2_rung_table(realized_train_n: int) -> list[dict]:
    """n-ladder rungs: TOP = realized post-drop train size (never a hardcoded 3600)."""
    rungs = [{"n_train": int(realized_train_n), "status": "top-rung (realized_train)"}]
    for r in P2_RUNGS_BELOW_TOP:
        if r == realized_train_n:
            continue  # top rung already covers it
        if r > realized_train_n:
            rungs.append({"n_train": int(r), "status": "UNREALIZABLE"})
        else:
            rungs.append({"n_train": int(r), "status": "realizable"})
    return rungs


def estimator_degenerate(n_train: int) -> bool:
    """n_train < 3585 (at or below d = 3584) => degenerate for absolute reads."""
    return n_train < D_BOUNDARY + 1


# ── R4/R6: verdict lattice + conditioning ────────────────────────────────────


def lattice_verdict(delta_mean: float, ci_low: float, ci_high: float) -> str:
    """Registered lattice, VERBATIM (plan section 3 — disjoint and exhaustive).

    Degrades <=> delta_mean > 0.05 AND ci_low_delta_mean > 0;
    Flat     <=> -0.03 <= delta_mean <= 0.03 AND ci_low >= -0.03 AND ci_high <= 0.03;
    Intermediate <=> otherwise. The interval conjuncts only SHRINK Flat — a point
    inside the band whose interval reaches past +0.05 is Intermediate, not Flat.
    """
    for v in (delta_mean, ci_low, ci_high):
        if not np.isfinite(v):
            raise ValueError(f"non-finite lattice input ({delta_mean}, {ci_low}, {ci_high})")
    if delta_mean > H1_DELTA_THRESHOLD and ci_low > 0:
        return "Degrades"
    if -H2_BAND <= delta_mean <= H2_BAND and ci_low >= -H2_BAND and ci_high <= H2_BAND:
        return "Flat"
    return "Intermediate"


def conditioned_interpretation(verdict: str, m1_pass: bool, m2_pass: bool) -> dict:
    """R6: EVERY lattice outcome is conditioned on DISTINCT = M1-pass AND M2-pass.

    Predicate failure maps Degrades, Flat, AND Intermediate to "manipulation
    failure"; the numeric lattice result is retained as DESCRIPTIVE ONLY.
    """
    assert verdict in ("Degrades", "Flat", "Intermediate"), verdict
    distinct = bool(m1_pass) and bool(m2_pass)
    if distinct:
        interp = verdict
        note = "joint distinctness predicate PASS — lattice label is interpretable"
    else:
        interp = "manipulation failure"
        note = (
            f"joint distinctness predicate FAIL (m1_pass={bool(m1_pass)}, "
            f"m2_pass={bool(m2_pass)}) — the personas never produced distinct answer "
            f"distributions; numeric lattice label '{verdict}' retained as DESCRIPTIVE "
            "ONLY (no origin-(in)consistency headline publishes off a failed manipulation)"
        )
    return {
        "distinct": distinct,
        "m1_pass": bool(m1_pass),
        "m2_pass": bool(m2_pass),
        "lattice_label_numeric": verdict,
        "interpretation": interp,
        "note": note,
    }


# ── G1 reduction-convention pin ──────────────────────────────────────────────


def fold_mean_r2(fold_components: list[tuple[float, float]]) -> float:
    """Banked convention: per-fold R2 = 1 - ss_res/(ss_tot + 1e-12), then mean."""
    return float(np.mean([1.0 - sr / (st + 1e-12) for sr, st in fold_components]))


def pooled_r2_from_components(fold_components: list[tuple[float, float]]) -> float:
    """Pooled statistic: 1 - sum(ss_res)/sum(ss_tot) over the fold components."""
    sres = float(sum(sr for sr, _ in fold_components))
    stot = float(sum(st for _, st in fold_components))
    return 1.0 - sres / (stot + 1e-12)


def reduction_convention_precheck(
    fold_components: list[tuple[float, float]], tol: float = G1_TOL
) -> dict:
    """Pre-check BEFORE the G1 verdict: fold-mean vs pooled gap must be far under tol.

    Fold-mean != pooled in general; an unpinned convention gap would eat into a
    +-0.005 tolerance derived from a realized <=0.0015 delta. A gap approaching
    tolerance is a CONVENTION MISMATCH to fix (fix-and-retry), never a
    reproduce failure — hence the raise names the convention, not the data.
    """
    fm = fold_mean_r2(fold_components)
    pooled = pooled_r2_from_components(fold_components)
    gap = abs(fm - pooled)
    if gap >= tol / 2.0:
        raise RuntimeError(
            f"reduction-convention pre-check: fold-mean vs pooled gap {gap:.6f} is not far "
            f"under the G1 tolerance {tol} — convention mismatch to fix before any G1 "
            "verdict is read (compare LIKE-FOR-LIKE in the banked fold-mean convention)"
        )
    return {"fold_mean": fm, "pooled": pooled, "gap": gap, "tol": tol, "pass": True}


def load_banked_fold_means() -> dict:
    """Re-derive the embedded G1 constants from the banked JSON (drift guard)."""
    banked = json.loads(BANKED_R2_JSON.read_text())
    out = {}
    for arm, banked_key in G1_BANKED_ARM_KEYS.items():
        r2_by_layer = banked["refit"][banked_key]["evil"]["r2_by_layer"]  # trait-independent
        for layer in READ_OUT_LAYERS:
            got = float(np.mean(r2_by_layer[layer]))
            want = G1_BANKED_FOLD_MEAN[(arm, layer)]
            if abs(got - want) > 1e-12:
                raise RuntimeError(
                    f"banked ridge_r2_by_arm.json fold-mean for ({arm}, L{layer}) = {got!r} "
                    f"!= embedded constant {want!r} — the banked artifact drifted from the "
                    "constants embedded at implementation time; re-derive before trusting G1"
                )
            out[(arm, layer)] = got
    return out


# ── Bootstrap (batched re-reduction over persisted per-context arrays) ───────


def bootstrap_paired(
    ss_res: dict[tuple[str, int], np.ndarray],
    ss_tot: dict[tuple[str, int], np.ndarray],
    n_draws: int,
    seed: int,
    delta_layers: tuple[int, ...] = READ_OUT_LAYERS,
    chunk: int = BOOT_CHUNK,
) -> dict:
    """Paired per-context bootstrap: ONE shared context resample per draw.

    Pooled R2 of a resample = 1 - sum ss_res[idx] / sum ss_tot[idx] (rank-space
    re-reduction of the persisted arrays; ss_tot per context is FIXED at its
    fold-mean-referenced value — parent parity). delta draw = mean over
    `delta_layers` of (R2_k1 - R2_k16). Both CI endpoints come off the same
    persisted draws (ci_high adds ZERO compute). Vectorized: chunked
    fancy-indexed batched sums, no per-draw Python loop over the pool.
    """
    cells = sorted(ss_res.keys())
    n = len(next(iter(ss_res.values())))
    for key in cells:
        assert ss_res[key].shape == (n,) and ss_tot[key].shape == (n,), key
    rng = np.random.default_rng(seed)
    draws = {key: np.empty(n_draws) for key in cells}
    done = 0
    while done < n_draws:
        c = min(chunk, n_draws - done)
        idx = rng.integers(0, n, size=(c, n))
        for key in cells:
            sres = ss_res[key][idx].sum(axis=1)
            stot = ss_tot[key][idx].sum(axis=1)
            draws[key][done : done + c] = 1.0 - sres / (stot + 1e-12)
        done += c
    delta_draws = np.mean(
        [draws[("k1", layer)] - draws[("k16", layer)] for layer in delta_layers], axis=0
    )
    per_cell_ci = {
        f"{arm}:L{layer}": {
            "ci_low": float(np.quantile(draws[(arm, layer)], 0.025)),
            "ci_high": float(np.quantile(draws[(arm, layer)], 0.975)),
        }
        for (arm, layer) in cells
    }
    return {
        "n_draws": int(n_draws),
        "seed": int(seed),
        "ci_low_delta_mean": float(np.quantile(delta_draws, 0.025)),
        "ci_high_delta_mean": float(np.quantile(delta_draws, 0.975)),
        "per_cell_ci": per_cell_ci,
    }


# ── Mixture-floor re-reads (pure re-reductions; zero extra compute) ──────────


def fixed_reference_r2(ss_res_k: np.ndarray, ss_tot_k1: np.ndarray) -> float:
    """R2_fixed(k) = 1 - sum ss_res(k) / sum ss_tot(k=1) — same k=1 denominator."""
    return float(1.0 - ss_res_k.sum() / (ss_tot_k1.sum() + 1e-12))


# ── Solver seams (thin wrappers over the reused primitives — no clones) ──────


def factorize_robust(x_tr: np.ndarray, dev: torch.device) -> dict:
    """FFC._factorize with the cuSOLVER non-convergence CPU fallback (gotcha)."""
    try:
        return FFC._factorize(x_tr, dev)
    except torch.linalg.LinAlgError:
        logger.warning("[eigh] cuda eigh failed to converge (n=%d) — CPU fallback", len(x_tr))
        return FFC._factorize(x_tr, torch.device("cpu"))


def val_select_lambda(
    fact: dict,
    vty: torch.Tensor,
    ymu: torch.Tensor,
    kval_v: torch.Tensor,
    y_val: np.ndarray,
    grid: np.ndarray,
    *,
    degenerate_ok: bool = False,
) -> tuple[float, float]:
    """Val-selected lambda over an arbitrary grid off a shared factorization.

    The wide-grid variant of FFC.gram_fit_apply's val branch (that helper pins
    the module-level 13-point LAMBDAS; P2 registers logspace(-2, 8, 21)).

    When EVERY grid point scores non-finite pooled R2 (degenerate val pool —
    ss_tot ~ 0, e.g. a single val row at smoke n), selection is meaningless:
    production (degenerate_ok=False) raises; smoke (degenerate_ok=True) logs
    and returns (grid[0], nan) so the caller can LABEL the cell degenerate.
    """
    best_lam, best_r2 = float(grid[0]), -np.inf
    any_finite = False
    for lam in grid:
        pred = FFC._apply(fact, float(lam), vty, ymu, kval_v)
        r2 = PR._pooled_r2(pred, y_val)
        if np.isfinite(r2):
            any_finite = True
            if r2 > best_r2:
                best_r2, best_lam = float(r2), float(lam)
    if not any_finite:
        if not degenerate_ok:
            raise RuntimeError(
                "val_select_lambda: every grid point scored non-finite pooled R2 — the "
                "validation pool is degenerate (ss_tot ~ 0, e.g. a single val row); "
                "refusing to silently keep grid[0] in production"
            )
        logger.warning("[p2] val selection DEGENERATE (all scores non-finite) — smoke label")
        return float(grid[0]), float("nan")
    return best_lam, best_r2


def gcv_solve_dof_capped(fact: dict, y_tr: np.ndarray, cap_frac: float = DOF_CAP):
    """SENSITIVITY-ONLY dof-capped GCV re-selection (never the primary; R5).

    Same GCV criterion as FFC._gcv_solve restricted to grid lambdas whose
    dof(lambda) = sum w/(w+lambda) <= cap_frac * n_train. Reported under
    sensitivity-only keys when the realized mask makes the cap bindable.
    """
    y = torch.as_tensor(np.asarray(y_tr), dtype=torch.float64, device=fact["dev"])
    if y.ndim == 1:
        y = y[:, None]
    ymu = y.mean(0)
    y_c = y - ymu
    vty = fact["V"].T @ y_c
    sq_vty = (vty**2).sum(1)
    tot = float((y_c**2).sum())
    w, ntr = fact["w"], fact["ntr"]
    cap = cap_frac * ntr
    best_lam, best_gcv, best_dof = None, float("inf"), float("nan")
    for lam in FFC.LAMBDAS:
        filt = w / (w + lam)
        dof = float(filt.sum())
        if dof > cap:
            continue
        rss = tot - float(((2 * filt - filt**2) * sq_vty).sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv, best_lam, best_dof = gcv, float(lam), dof
    if best_lam is None:
        raise RuntimeError(f"dof cap {cap_frac} excludes every grid lambda (n_train={ntr})")
    return best_lam, vty, ymu, best_dof


def dof_cap_sensitivity(
    inputs,
    gathers: dict,
    mask_ids: np.ndarray,
    folds: list,
    dev: torch.device,
    ckpt_dir: pathlib.Path,
    fingerprint: dict,
    layers: tuple[int, ...] = READ_OUT_LAYERS,
    arms: tuple[str, ...] = ARM_NAMES,
) -> dict[str, dict]:
    """Capped-sensitivity re-selection: ONE shared factorization per (layer, fold).

    The arm loop solves every capped target against the SHARED eigh — 15
    factorizations at production shape (3 layers x 5 folds), never 105 (the
    arm-encloses-fold shape). Per-layer checkpoints resume without recomputing
    any factorization; per-cell progress lines; per-fold selected lambda /
    lambda_edge / dof / n_train persisted under sensitivity keys.
    """
    cells: dict[str, dict] = {}
    unit, total = 0, len(layers) * len(folds)
    for layer in layers:
        name = f"p1_sens_L{layer:02d}"
        if chunk_done(ckpt_dir, name, fingerprint):
            z = np.load(ckpt_dir / f"{name}.npz", allow_pickle=True)
            cells.update(json.loads(str(z["cells"])))
            unit += len(folds)
            logger.info("[p1-sens] resume: layer %d loaded from checkpoint", layer)
            continue
        x_full = inputs.input_col(layer, mask_ids)
        y_full = {arm: arm_target(inputs, gathers, arm, layer, mask_ids, mask_ids) for arm in arms}
        comps: dict[str, list] = {arm: [] for arm in arms}
        details: dict[str, list] = {arm: [] for arm in arms}
        for f_idx, (tr, te) in enumerate(folds):
            t_cell = time.monotonic()
            fact = factorize_robust(x_full[tr], dev)  # ONE eigh shared by all capped targets
            kev = FFC._cross_kernel(fact, x_full[te])
            for arm in arms:
                lam, vty, ymu, dof = gcv_solve_dof_capped(fact, y_full[arm][tr])
                pred = FFC._apply(fact, lam, vty, ymu, kev)
                sres, stot = per_context_ss(pred, y_full[arm][te])
                comps[arm].append((float(sres.sum()), float(stot.sum())))
                details[arm].append(
                    {
                        "fold": f_idx,
                        "lambda": lam,
                        "lambda_edge": SSP.lambda_edge(lam, FFC.LAMBDAS),
                        "dof": dof,
                        "n_train": int(len(tr)),
                    }
                )
            unit += 1
            print(
                f"[p1-sens] unit {unit}/{total} L={layer} fold={f_idx} "
                f"elapsed={time.monotonic() - t_cell:.1f}s",
                flush=True,
            )
        layer_cells = {
            f"{arm}:L{layer}": {
                "pooled_r2_dof_capped": pooled_r2_from_components(comps[arm]),
                "cap": DOF_CAP,
                "folds": details[arm],
            }
            for arm in arms
        }
        cells.update(layer_cells)
        save_chunk(ckpt_dir, name, {"cells": np.array(json.dumps(layer_cells))}, fingerprint)
    return cells


def svd_gcv_lambda(x_tr: np.ndarray, y_tr: np.ndarray, lambdas: np.ndarray) -> float:
    """DIAGNOSTIC-ONLY mirror of fit_h.ridge_fit_predict's GCV selection loop.

    Used ONLY to report the canonical path's selected lambda in the G2 record
    (ridge_fit_predict does not expose it); never used for predictions.
    """
    xtr = np.asarray(x_tr, dtype=np.float64)
    ytr = np.asarray(y_tr, dtype=np.float64)
    n = xtr.shape[0]
    xmu, xsd = xtr.mean(0), xtr.std(0) + 1e-9
    xn = (xtr - xmu) / xsd
    yc = ytr - ytr.mean(0)
    u, s, _vt = np.linalg.svd(xn, full_matrices=False)
    s2 = s**2
    uty = u.T @ yc
    best_lam, best_gcv = float(lambdas[0]), np.inf
    for lam in lambdas:
        filt = s2 / (s2 + lam)
        yhat = u @ (filt[:, None] * uty)
        rss = float(np.sum((yc - yhat) ** 2))
        dof = float(np.sum(filt))
        denom = (n - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else np.inf
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, float(lam)
    return best_lam


def per_context_ss(pred: np.ndarray, true: np.ndarray, ref_mean: np.ndarray | None = None):
    """Per-context (ss_res, ss_tot); ss_tot vs `ref_mean` (default: this set's mean)."""
    mu = true.mean(0) if ref_mean is None else ref_mean
    sres = ((true - pred) ** 2).sum(axis=1)
    stot = ((true - mu) ** 2).sum(axis=1)
    return sres, stot


# ── Staging ──────────────────────────────────────────────────────────────────


def _hf_fetch(path_in_repo: str, revision: str, dl_dir: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(
        retry_transient(
            lambda: hf_hub_download(
                DATA_REPO,
                path_in_repo,
                repo_type="dataset",
                revision=revision,
                local_dir=dl_dir,
            ),
            what=f"hf_hub_download({path_in_repo})",
        )
    )


def stage_parent_inputs(root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Bundle (cx_last), parent arm tensors (own/plain), parent common-valid mask."""
    dl = root / "parent_inputs"
    out = {}
    out["bundle"] = _hf_fetch(BUNDLE_PATH_IN_REPO, BUNDLE_REV, dl)
    for arm, fn in PARENT_ARM_FILES.items():
        out[arm] = _hf_fetch(f"{PARENT_PREFIX}/analysis_tensors/{fn}", PARENT_REV, dl)
    out["mask"] = _hf_fetch(PARENT_MASK_PATH_IN_REPO, PARENT_REV, dl)
    return out


def expected_store_rows(p: int, n_contexts: int) -> int:
    """Registered pair arithmetic: persona p serves context i iff i % k == p for some arm k > p.

    Per-persona expected store row counts at n=5000: p0=5000, p1=2500, p2/p3=1250,
    p4-7=625, p8-15=312 (total 14996 == gen.registered_pair_total(5000)).
    """
    return sum(1 for i in range(n_contexts) if any(i % k == p for k in LADDER_KS if p < k))


def stage_pair_store(
    root: pathlib.Path, store_prefix: str, revision: str | None, n_contexts: int
) -> tuple[dict[int, pathlib.Path], dict, pathlib.Path]:
    """Pair store: local-first WITH producer completion evidence, else HF at ONE pinned sha.

    The local branch accepts the store ONLY on the producer's own completion
    predicate: per ACTIVE persona (expected_store_rows > 0 at n_contexts) the
    `.done.json` sidecar must exist AND its fingerprint must match this run's
    regime (n_contexts / n_layers / hidden) — a tensor file alone can be a
    stale or partial capture generation. capture_digest.json (the manipulation
    accounting's coverage record) is REQUIRED on both branches; the remote
    branch stages it alongside the tensors. Returns (paths, store_identity,
    capture_digest_path); store_identity is bound into the checkpoint
    fingerprints so a resume against a different store generation is refused.
    """
    tensors_dir = root / "analysis_tensors"
    active = [p for p in range(N_PERSONAS) if expected_store_rows(p, n_contexts) > 0]
    paths: dict[int, pathlib.Path] = {}
    local_ok = all(group_paths(tensors_dir, p)[0].exists() for p in active)
    if local_ok:
        sidecars: dict[int, dict] = {}
        for p in active:
            t_path, done_path = group_paths(tensors_dir, p)
            if not done_path.exists():
                raise RuntimeError(
                    f"local pair store at {tensors_dir} has {t_path.name} but no producer "
                    f"sidecar {done_path.name} — the capture group is not verifiably "
                    "complete (stale/partial capture generation)"
                )
            side = json.loads(done_path.read_text())
            fp = side.get("fingerprint", {})
            got = {k: fp.get(k) for k in ("n_contexts", "n_layers", "hidden")}
            want = {"n_contexts": n_contexts, "n_layers": EXPECTED_LAYERS, "hidden": HIDDEN}
            if got != want:
                raise RuntimeError(
                    f"local pair store sidecar for persona {p} is stale: fingerprint "
                    f"regime {got} != this run's {want}"
                )
            sidecars[p] = side
            paths[p] = t_path
        digest_path = tensors_dir / "capture_digest.json"
        if not digest_path.exists():
            raise RuntimeError(
                f"local pair store at {tensors_dir} lacks capture_digest.json — the "
                "manipulation accounting cannot verify capture coverage without it"
            )
        identity = {
            "source": "local-sidecars",
            "sidecar_fingerprint_sha256": hashlib.sha256(
                json.dumps(
                    {str(p): sidecars[p].get("fingerprint", {}) for p in active},
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        }
        logger.info(
            "pair store staged locally at %s (%d sidecars verified)", tensors_dir, len(active)
        )
        return paths, identity, digest_path
    resolved = resolve_dataset_revision(revision)
    logger.info("fetching pair store from %s/%s @ %s", DATA_REPO, store_prefix, resolved)
    for p in active:
        paths[p] = _hf_fetch(
            f"{store_prefix}/analysis_tensors/v_pairs_p{p:02d}.pt", resolved, root / "store_dl"
        )
    digest_path = _hf_fetch(
        f"{store_prefix}/analysis_tensors/capture_digest.json", resolved, root / "store_dl"
    )
    identity = {"source": "hf", "revision": resolved, "prefix": store_prefix}
    return paths, identity, digest_path


@dataclasses.dataclass
class LadderInputs:
    """Everything the fit core consumes (assembled by staging, or synthetically in tests)."""

    cx_last: object  # (n_ctx, n_layers, hidden) tensor-like; layer col via layers_map
    layers_map: list[int]
    parent_arm: dict[str, np.ndarray]  # own/plain -> (n_ctx, n_layers, hidden) float32
    parent_valid_ids: np.ndarray
    store_v: dict[int, np.ndarray]  # persona -> (n_p, n_layers, hidden) float32
    store_ctx: dict[int, np.ndarray]  # persona -> (n_p,) context ids
    store_span: dict[int, np.ndarray]  # persona -> (n_p,) span lengths (0 => invalid)
    n_contexts: int
    n_layers: int = EXPECTED_LAYERS
    hidden: int = HIDDEN
    store_identity: dict | None = None  # producer completion evidence (stage_pair_store)
    capture_digest: dict | None = None  # capture coverage record (manipulation accounting)

    def input_col(self, layer: int, ids: np.ndarray) -> np.ndarray:
        col = self.layers_map.index(layer)
        x = self.cx_last[ids][:, col, :]
        if isinstance(x, torch.Tensor):
            return x.to(torch.float64).numpy()
        return np.asarray(x, dtype=np.float64)


def load_inputs(
    root: pathlib.Path, store_prefix: str, store_revision: str | None, n_contexts: int
) -> LadderInputs:
    """Stage + materialize all fit inputs (plan section-12 RAM budget ~12-14 GB)."""
    parent = stage_parent_inputs(root)
    bundle = torch.load(str(parent["bundle"]), map_location="cpu", mmap=True, weights_only=False)
    for fld in ("cx_last", "layers"):
        assert fld in bundle, f"pass_b bundle missing {fld}"
    cx_last = bundle["cx_last"]
    layers_map = list(bundle["layers"])
    assert tuple(cx_last.shape) == (N_CTX, EXPECTED_LAYERS, HIDDEN), cx_last.shape

    parent_arm = {}
    for arm in ("own", "plain"):
        t = torch.load(str(parent[arm]), map_location="cpu", mmap=True, weights_only=False)
        assert tuple(t.shape) == (N_CTX, EXPECTED_LAYERS, HIDDEN), (arm, t.shape)
        parent_arm[arm] = t.numpy()

    mask_d = json.loads(parent["mask"].read_text())
    parent_valid = np.array(sorted(mask_d["common_valid_idx"]), dtype=int)
    parent_valid = parent_valid[parent_valid < n_contexts]

    store_paths, store_identity, digest_path = stage_pair_store(
        root, store_prefix, store_revision, n_contexts
    )
    capture_digest = json.loads(digest_path.read_text())
    store_v, store_ctx, store_span = {}, {}, {}
    for p, path in sorted(store_paths.items()):
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
        v = payload["v"]
        assert v.ndim == 3 and v.shape[1:] == (EXPECTED_LAYERS, HIDDEN), (p, v.shape)
        want_rows = expected_store_rows(p, n_contexts)
        assert v.shape[0] == want_rows, (
            f"persona {p}: store has {v.shape[0]} rows != expected_store_rows(p, "
            f"{n_contexts}) = {want_rows} (registered pair arithmetic) — stale/partial "
            "capture generation"
        )
        store_v[p] = v.numpy()
        store_ctx[p] = payload["context_ids"].numpy()
        store_span[p] = payload["span_lengths"].numpy()
    return LadderInputs(
        cx_last=cx_last,
        layers_map=layers_map,
        parent_arm=parent_arm,
        parent_valid_ids=parent_valid,
        store_v=store_v,
        store_ctx=store_ctx,
        store_span=store_span,
        n_contexts=n_contexts,
        store_identity=store_identity,
        capture_digest=capture_digest,
    )


# ── Arm assembly + mask (equalize-down) ──────────────────────────────────────


def assert_mask_gate_schema(sentinel: dict) -> None:
    """Fail-loud consumer-side pin of the P-Gen v11 mask-gate label schema.

    The section-7 kill-1 gate classifies dropped pairs via P-Gen's persisted
    per-record `validity` labels (stop_reason-keyed, refusal-precedence — plan
    v13 L603-617, schema `issue823_mask_gate_v11_stop_reason_precedence`).
    Consuming labels produced under any OTHER schema would silently change the
    gate's semantics, so a mismatched/absent schema id in the gen sentinel's
    `generation_config_fingerprint.fields` raises — the superseded v10
    span/emptiness proxy (the 2026-08-19 rc=5 false abort) must be impossible
    to fall back to silently.
    """
    fp = sentinel.get("generation_config_fingerprint")
    fields = fp.get("fields") if isinstance(fp, dict) else None
    got = fields.get("mask_gate_schema_id") if isinstance(fields, dict) else None
    if got != MASK_GATE_SCHEMA_ID:
        raise RuntimeError(
            f"P-Gen sentinel mask_gate_schema_id {got!r} != required "
            f"{MASK_GATE_SCHEMA_ID!r} — refusing to classify mask drops under an "
            "unknown validity-label schema (plan v13 kill 1 consumes the v11 "
            "stop_reason-keyed class split; no silent span-proxy fallback)"
        )


def build_mask_and_gathers(
    inputs: LadderInputs, by_persona: dict[int, list[dict]]
) -> tuple[np.ndarray, dict, dict]:
    """New common-valid mask + per-arm gather plans + per-arm NEW-drop accounting.

    A context drops if ANY of its <=5 distinct ladder pair rows is invalid
    (span==0 or row absent) — equalize-down: every arm fits on identical
    contexts (mask construction unchanged, plan v13 L587). Returns
    (mask_ids, gathers, drop_record); gathers[arm] is a list of
    (persona, positions_in_mask, store_row_indices).

    Kill-1 classification (plan v13 L603-617, v11 schema): each dropped
    (context, persona) pair is classified from P-Gen's persisted `validity`
    label — "refusal" => refusal-attributed (P0 prompt integrity passed
    upstream: the gen sentinel is written only after P-Gen's own gates), and
    everything else is integrity-class, sub-bucketed as `missing_record` (no
    P-Gen record for the pair, L615), the gen-side label verbatim
    ("empty" / "error:<category>"), or `capture_zero_span` (gen-"ok" but the
    store row is absent/zero-span, L616). A record PRESENT without a
    `validity` key is a v11 schema break => RuntimeError, never a quiet class
    vote (`by_persona` is scanned in full so an unlabeled producer fails loud
    regardless of which pairs dropped).
    """
    rowmap = {p: {int(c): j for j, c in enumerate(inputs.store_ctx[p])} for p in inputs.store_ctx}
    parent_ids = [int(i) for i in inputs.parent_valid_ids if i < inputs.n_contexts]

    label: dict[tuple[int, int], str] = {}
    for p, rows in by_persona.items():
        for r in rows:
            if "validity" not in r:
                raise RuntimeError(
                    f"P-Gen record (context_id={r.get('context_id')}, persona={p}) has "
                    "no 'validity' label — v11 mask-gate schema break; refusing to "
                    "classify mask drops"
                )
            label[(int(r["context_id"]), int(p))] = str(r["validity"])

    def drop_class(i: int, p: int) -> tuple[str, str]:
        """(class, subclass) for a dropped pair — class in {refusal, integrity}."""
        v = label.get((i, p))
        if v is None:
            return "integrity", "missing_record"  # plan v13 L615: missing/unparseable record
        if v == "refusal":
            return "refusal", "refusal"
        if v == "ok":
            return "integrity", "capture_zero_span"  # gen-valid, store row absent/zero (L616)
        return "integrity", v  # "empty" | "error:<category>" | any unrecognized label

    def pair_ok(i: int, p: int) -> bool:
        j = rowmap.get(p, {}).get(i)
        return j is not None and inputs.store_span[p][j] > 0

    new_drops_per_arm = {}
    for k in LADDER_KS:
        new_drops_per_arm[f"k{k}"] = [i for i in parent_ids if not pair_ok(i, i % k)]
    mask_ids = np.array(
        [i for i in parent_ids if all(pair_ok(i, i % k) for k in LADDER_KS)], dtype=int
    )
    by_class_per_arm: dict[str, dict[str, int]] = {}
    subclasses_per_arm: dict[str, dict[str, int]] = {}
    for a, ids in new_drops_per_arm.items():
        k = int(a[1:])
        cls_n = {"refusal": 0, "integrity": 0}
        sub_n: dict[str, int] = {}
        for i in ids:
            cls, sub = drop_class(i, i % k)
            cls_n[cls] += 1
            sub_n[sub] = sub_n.get(sub, 0) + 1
        by_class_per_arm[a] = cls_n
        subclasses_per_arm[a] = dict(sorted(sub_n.items()))
    drop_record = {
        "parent_valid_n": len(parent_ids),
        "mask_n": int(len(mask_ids)),
        "new_drops_per_arm": {a: len(v) for a, v in new_drops_per_arm.items()},
        "new_drops_per_arm_by_class": by_class_per_arm,
        "new_drop_subclasses_per_arm": subclasses_per_arm,
        "new_dropped_ids_union": sorted({i for v in new_drops_per_arm.values() for i in v}),
        "abort_threshold_per_arm": NEW_DROPS_ABORT_PER_ARM,
        "abort_class": "integrity",
        "mask_gate_schema_id": MASK_GATE_SCHEMA_ID,
    }
    gathers: dict[str, list] = {}
    for k in LADDER_KS:
        plan = []
        for p in sorted({int(i) % k for i in mask_ids}):
            pos = np.array([j for j, i in enumerate(mask_ids) if int(i) % k == p], dtype=int)
            rows = np.array([rowmap[p][int(mask_ids[j])] for j in pos], dtype=int)
            plan.append((p, pos, rows))
        gathers[f"k{k}"] = plan
    return mask_ids, gathers, drop_record


def mask_integrity_verdict(drop_record: dict) -> tuple[str, int]:
    """(arm, integrity_count) of the worst arm for the section-7 kill-1 gate.

    v11 semantics (plan v13 L619-623, L1175): the pipeline-integrity kill
    counts INTEGRITY-CLASS (non-refusal) NEW invalid rows per arm vs
    NEW_DROPS_ABORT_PER_ARM; refusal-attributed drops are governed by the
    separate refusal-attrition budget (disposition 2) and never trip this
    gate.
    """
    by_class = drop_record["new_drops_per_arm_by_class"]
    return max(((a, c["integrity"]) for a, c in by_class.items()), key=lambda kv: kv[1])


def arm_target(
    inputs: LadderInputs, gathers: dict, arm: str, layer: int, ids: np.ndarray, mask_ids: np.ndarray
) -> np.ndarray:
    """(len(ids), hidden) float64 target for `arm` at `layer`, rows aligned to `ids`."""
    if arm in ("own", "plain"):
        return np.asarray(inputs.parent_arm[arm][ids, layer, :], dtype=np.float64)
    pos_of = {int(c): j for j, c in enumerate(mask_ids)}
    out = np.empty((len(ids), inputs.hidden), dtype=np.float64)
    want = {int(i) for i in ids}
    id_to_out = {int(c): j for j, c in enumerate(ids)}
    filled = 0
    for p, pos, rows in gathers[arm]:
        sel = [(int(mask_ids[q]), r) for q, r in zip(pos, rows) if int(mask_ids[q]) in want]
        if not sel:
            continue
        ctxs, rws = zip(*sel)
        block = inputs.store_v[p][np.array(rws), layer, :]
        for c, vec in zip(ctxs, block):
            out[id_to_out[c]] = vec
        filled += len(ctxs)
    assert filled == len(ids), (arm, layer, filled, len(ids), len(pos_of))
    return out


# ── Retrieval + identity per fitted cell (house-rule pair) ───────────────────


def cell_baselines(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, pred: np.ndarray
) -> dict:
    """identity+learned-bias R2 + kNN retrieval (euclid + cosine) for one cell."""
    id_pred = identity_bias_predict(x_tr, y_tr, x_te)
    out = {
        "identity_bias_r2": PR._pooled_r2(id_pred, y_te),
        "n_pool": int(len(y_te)),
        "small_cell": bool(len(y_te) < SMALL_CELL_TEST_FLOOR),
    }
    for metric in ("euclidean", "cosine"):
        out[f"knn_{metric}"] = knn_retrieval(pred, y_te, metric=metric)
    return out


# ── Checkpoint helpers (resume keyed on generating parameters) ───────────────


def checkpoint_fingerprint(mask_ids: np.ndarray, extra: dict) -> dict:
    """Machine-stable resume key: int-id sha + generating parameters (never float bytes)."""
    return {
        "mask_sha": _ids_sha(mask_ids),
        "estimator": PRIMARY_ESTIMATOR,
        "p1_grid": list(P1_GRID_PARAMS),
        "p2_grid": list(P2_GRID_PARAMS),
        "fold_seed": P1_FOLD_SEED,
        "n_folds": P1_N_FOLDS,
        "arms": list(ARM_NAMES),
        **extra,
    }


def chunk_done(ckpt_dir: pathlib.Path, name: str, fingerprint: dict) -> bool:
    """True iff checkpoint `name` exists with a MATCHING fingerprint (else fail loud)."""
    sidecar = ckpt_dir / f"{name}.json"
    if not sidecar.exists():
        return False
    d = json.loads(sidecar.read_text())
    if d.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"{sidecar} exists with a DIFFERENT fingerprint (stale regime) — refusing "
            "silent reuse; clear the out-root checkpoints or resolve the drift first"
        )
    if not (ckpt_dir / f"{name}.npz").exists():
        raise RuntimeError(f"{sidecar} present but {name}.npz missing — partial checkpoint")
    return True


def save_chunk(ckpt_dir: pathlib.Path, name: str, arrays: dict, fingerprint: dict) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_dir / f"{name}.tmp.npz"  # suffix stays .npz (np.savez appends it otherwise)
    np.savez(tmp, **arrays)
    tmp.replace(ckpt_dir / f"{name}.npz")
    write_json(ckpt_dir / f"{name}.json", {"fingerprint": fingerprint, "ts": time.time()})


def restore_pp_arrays(z, p2_pc: dict) -> int:
    """Restore per-context `pp_*` arrays from a per-k checkpoint npz into p2_pc.

    Returns the number of arrays restored. The pp_ keys are the four-layer
    per-context (ss_res, ss_tot) inputs the final percontext_ladder_p2.npz
    archive persists (registered artifact inputs) — without restoring them a
    RESUMED per-persona block would silently drop its arrays from the archive.
    """
    n = 0
    for key in z.files:
        if key.startswith("pp_"):
            p2_pc[key] = np.asarray(z[key])
            n += 1
    return n


def assert_p2_percontext_complete(
    per_persona: dict, p2_pc: dict, layers: tuple[int, ...] = P2_LAYERS
) -> None:
    """Set-check: every non-skipped per-persona cell has its four pp_ arrays.

    Runs BEFORE the percontext_ladder_p2.npz write, so a resume/restore bug can
    never ship a silently-partial archive. `layers` mirrors the fit loop's
    per-cell layer keys (cells carry `L<layer>` sub-dicts).
    """
    missing: list[str] = []
    for k_name, k_cells in per_persona.items():
        for p_name, cell in k_cells.items():
            if not isinstance(cell, dict) or str(cell.get("status", "")).startswith("skipped"):
                continue
            for layer in layers:
                if f"L{layer}" not in cell:
                    continue
                for suffix in ("a_sres", "a_stot", "c_sres", "c_stot"):
                    key = f"pp_{k_name}_{p_name}_L{layer}_{suffix}"
                    if key not in p2_pc:
                        missing.append(key)
    if missing:
        raise RuntimeError(
            f"percontext_ladder_p2 archive would be missing {len(missing)} per-context "
            f"arrays (first: {missing[:6]}) — a resumed per-persona block lost its "
            "registered pp_ inputs (restore_pp_arrays not applied?)"
        )


def p2_withheld_result(metadata: dict, split_block: dict) -> dict:
    """P2 record under the solver-parity contingency: fast-path fits WITHHELD.

    P2's single-split fits run on the shared gram fast-path primitives
    (_factorize/_gcv_solve/_apply); when the G2 gate rejected that path, those
    fits are unverified — withheld rather than reported (the priced canonical
    contingency covers P1 only; plan sections 4.3/8).
    """
    return {
        "status": "WITHHELD — solver-parity contingency engaged",
        "reason": (
            "P2 single-split fits use the gram fast-path primitives that FAILED the G2 "
            "parity gate; their fits are unverified under contingency and are withheld "
            "rather than reported (the canonical-solver contingency prices P1 only)"
        ),
        "metadata": metadata,
        "split": split_block,
        "full_arms": {},
        "n_ladder": {},
        "per_persona": {},
        "g_mix": {},
    }


# ── Smoke/production isolation + sentinel naming ─────────────────────────────


def smoke_root_aliases_production(root: pathlib.Path | str) -> bool:
    """True iff a smoke out-root resolves AT or UNDER the production pod out-root.

    A smoke run scheduled before the production run on the same pod must never
    write artifacts/checkpoints/sentinels into the production tree.
    """
    return pathlib.Path(root).resolve().is_relative_to(PROD_POD_OUT_ROOT)


def sentinel_filename(smoke: bool) -> str:
    """DISTINCT completion-sentinel filenames per mode.

    Both modes writing the same sentinel path on the pod would let a smoke run
    scheduled BEFORE production satisfy the production poller.
    """
    return "issue-823-ladder-fits-smoke-done.json" if smoke else "issue-823-ladder-fits-done.json"


# ── Abort helper (designed halts, never bare rc=1) ───────────────────────────


def designed_abort(eval_dir: pathlib.Path, kind: str, rc: int, payload: dict, smoke: bool) -> None:
    """Write the abort report + exit rc (production) / log informational (smoke)."""
    report = {"abort_kind": kind, "rc": rc, "smoke": smoke, **payload}
    write_json(eval_dir / "fits_abort_report.json", report)
    if smoke:
        logger.warning("[smoke-informational] %s would abort in production: %s", kind, payload)
        return
    log_phase(f"pfit_abort_{kind}")
    logger.error("designed abort %s (rc=%d): %s", kind, rc, payload)
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(rc)


# ── Import check ─────────────────────────────────────────────────────────────


def run_import_check() -> None:
    """Execute deferred imports + signature-bind every reused seam + argcheck."""
    import inspect

    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    inspect.signature(SSP.fixed_split).bind(
        n_ctx=5000, n_train=3600, n_val=400, n_test=1000, seed=42
    )
    inspect.signature(FFC._factorize).bind(np.zeros((4, 3)), torch.device("cpu"))
    fact = FFC._factorize(np.random.default_rng(0).normal(size=(6, 4)), torch.device("cpu"))
    lam, vty, ymu = FFC._gcv_solve(fact, np.random.default_rng(1).normal(size=(6, 4)))
    kev = FFC._cross_kernel(fact, np.zeros((2, 4)))
    FFC._apply(fact, lam, vty, ymu, kev)
    FFC._vty_ymu(fact, np.zeros((6, 4)))
    inspect.signature(ridge_fit_predict).bind(np.zeros((4, 3)), np.zeros((4, 3)), np.zeros((2, 3)))
    inspect.signature(identity_bias_predict).bind(
        np.zeros((4, 3)), np.zeros((4, 3)), np.zeros((2, 3))
    )
    inspect.signature(knn_retrieval).bind(np.zeros((4, 3)), np.zeros((4, 3)), metric="cosine")
    inspect.signature(fetch_gen_inputs).bind(pathlib.Path("."), "prefix", None, None)
    inspect.signature(load_pair_rows).bind({}, 10)
    inspect.signature(verify_gen_sentinel).bind({})
    inspect.signature(resolve_dataset_revision).bind(None)
    inspect.signature(_upload_folder_filtered).bind(
        local_dir=pathlib.Path("."),
        repo_id=DATA_REPO,
        repo_type="dataset",
        path_in_repo="x/logs",
        allow_patterns=["*.json"],
        expected_repo_paths=["x/logs/y.json"],
    )
    from sklearn.feature_extraction.text import TfidfVectorizer  # deferred in m1_tfidf

    print(
        json.dumps(
            {
                "import_check": "ok",
                "deferred_imports": ["sklearn TfidfVectorizer (m1_tfidf)"],
                "signature_bound": [
                    "SSP.fixed_split",
                    "FFC._factorize/_gcv_solve/_cross_kernel/_apply/_vty_ymu (executed)",
                    "fit_h.ridge_fit_predict",
                    "mapping_baselines.identity_bias_predict/knn_retrieval",
                    "capture.fetch_gen_inputs/load_pair_rows/verify_gen_sentinel",
                    "capture.resolve_dataset_revision",
                    "hub._upload_folder_filtered",
                ],
                "constants": {
                    "primary_estimator": PRIMARY_ESTIMATOR,
                    "arms": list(ARM_NAMES),
                    "read_out_layers": list(READ_OUT_LAYERS),
                    "p2_layers": list(P2_LAYERS),
                    "tfidf_cls": TfidfVectorizer.__name__,
                },
            }
        )
    )


# ── Manipulation checks ──────────────────────────────────────────────────────


def m1_tfidf(by_persona: dict[int, list[dict]], mask_ids: np.ndarray) -> dict:
    """M1: within-context cross-persona TF-IDF cosine (contexts with >=2 personas)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    mask_set = {int(i) for i in mask_ids}
    texts_by_ctx: dict[int, list[str]] = {}
    corpus, keys = [], []
    for p, rows in by_persona.items():
        for r in rows:
            cid = int(r["context_id"])
            if cid in mask_set and r.get("filled") and r.get("answer_text"):
                texts_by_ctx.setdefault(cid, []).append(r["answer_text"])
                corpus.append(r["answer_text"])
                keys.append((cid, p))
    vec = TfidfVectorizer()
    mat = vec.fit_transform(corpus)
    rows_by_ctx: dict[int, list[int]] = {}
    for j, (cid, _p) in enumerate(keys):
        rows_by_ctx.setdefault(cid, []).append(j)
    per_ctx_means = []
    for cid, texts in texts_by_ctx.items():
        if len(texts) < 2:
            continue
        rows = rows_by_ctx[cid]
        sub = mat[rows]
        sims = (sub @ sub.T).toarray()
        norms = np.sqrt(np.asarray(sub.multiply(sub).sum(axis=1))).ravel()
        denom = np.outer(norms, norms) + 1e-12
        cos = sims / denom
        iu = np.triu_indices(len(rows), k=1)
        if len(iu[0]):
            per_ctx_means.append(float(cos[iu].mean()))
    mean_cos = float(np.mean(per_ctx_means)) if per_ctx_means else float("nan")
    return {
        "mean_within_context_cross_persona_tfidf_cosine": mean_cos,
        "n_contexts_scored": len(per_ctx_means),
        "bar": M1_TFIDF_BAR,
        "m1_pass": bool(np.isfinite(mean_cos) and mean_cos <= M1_TFIDF_BAR),
    }


def m2_paired_separation(inputs: LadderInputs, mask_ids: np.ndarray) -> dict:
    """M2: within-context paired activation separation vs the persona-0 reference.

    delta_p(i) = v_p(i) - v_0(i) over persona-p pair rows in the mask; pass <=>
    ||m_p|| > 2 x sqrt(tr(Sigma_p)/n_p) for >=12/15 personas at >=2/3 read-out
    layers. Paired form removes the context main effect (residue-class
    assignment makes raw centroids differ by context composition alone).
    """
    mask_set = {int(i) for i in mask_ids}
    row0 = {int(c): j for j, c in enumerate(inputs.store_ctx[0])}
    per_persona: dict[str, dict] = {}
    pass_counts = {}
    for p in range(1, N_PERSONAS):
        if p not in inputs.store_ctx:
            # No store rows at this n_contexts (expected_store_rows == 0 — smoke slice);
            # counts as a non-passing persona, never a KeyError.
            per_persona[f"p{p:02d}"] = {"absent": True}
            pass_counts[p] = 0
            continue
        ctxs = [
            int(c)
            for j, c in enumerate(inputs.store_ctx[p])
            if int(c) in mask_set and inputs.store_span[p][j] > 0 and int(c) in row0
        ]
        rows_p = {int(c): j for j, c in enumerate(inputs.store_ctx[p])}
        layer_stats = {}
        n_layers_passed = 0
        for layer in READ_OUT_LAYERS:
            if not ctxs:
                layer_stats[f"L{layer}"] = {"n": 0, "pass": False}
                continue
            vp = inputs.store_v[p][np.array([rows_p[c] for c in ctxs]), layer, :].astype(np.float64)
            v0 = inputs.store_v[0][np.array([row0[c] for c in ctxs]), layer, :].astype(np.float64)
            delta = vp - v0
            m_p = delta.mean(axis=0)
            tr_sigma = float(((delta - m_p) ** 2).sum(axis=1).mean())
            floor = float(np.sqrt(tr_sigma / max(len(ctxs), 1)))
            norm = float(np.linalg.norm(m_p))
            passed = norm > M2_FLOOR_MULT * floor
            n_layers_passed += int(passed)
            layer_stats[f"L{layer}"] = {
                "n": len(ctxs),
                "shift_norm": norm,
                "noise_floor": floor,
                "pass": bool(passed),
            }
        per_persona[f"p{p:02d}"] = layer_stats
        pass_counts[p] = n_layers_passed
    n_pass = sum(1 for p, c in pass_counts.items() if c >= M2_MIN_LAYERS)
    return {
        "per_persona": per_persona,
        "n_personas_passing": int(n_pass),
        "criterion": (
            f"||m_p|| > {M2_FLOOR_MULT} x noise floor for >= {M2_MIN_PERSONAS} of "
            f"{N_PERSONAS - 1} personas at >= {M2_MIN_LAYERS} of {len(READ_OUT_LAYERS)} layers"
        ),
        "m2_pass": bool(n_pass >= M2_MIN_PERSONAS),
    }


def m3_accounting(by_persona: dict[int, list[dict]], capture_digest: dict | None) -> dict:
    """M3: lengths, refusal/empty counts, cap-hit fractions, truncation, batch waves."""
    out: dict[str, dict] = {"per_persona": {}}
    for p, rows in sorted(by_persona.items()):
        lens = [len(r["answer_text"]) for r in rows if r.get("filled") and r.get("answer_text")]
        waves: dict[str, int] = {}
        for r in rows:
            b = r.get("batch_id")
            if b:
                waves[b] = waves.get(b, 0) + 1
        q = np.quantile(lens, [0.05, 0.5, 0.95]).tolist() if lens else [None] * 3
        out["per_persona"][f"p{p:02d}"] = {
            "n_rows": len(rows),
            "n_filled": sum(1 for r in rows if r.get("filled")),
            "n_refusal": sum(1 for r in rows if r.get("stop_reason") == "refusal"),
            "n_empty": sum(1 for r in rows if not r.get("answer_text")),
            "cap_hit_fraction": float(np.mean([bool(r.get("cap_hit")) for r in rows]))
            if rows
            else None,
            "answer_len_chars_p5_p50_p95": q,
            "batch_wave_counts": waves,
        }
    if capture_digest is not None:
        out["truncation_by_arm_persona"] = capture_digest.get("truncation_by_arm_persona")
        out["capture_cap_hit"] = capture_digest.get("cap_hit")
    return out


# ── Main ─────────────────────────────────────────────────────────────────────


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "P-Fit for #823 inconsistent-origin-persona-ladder: 7-arm P1 pooled-GCV ridge "
            "+ P2 single-split n-ladder / per-persona control + bootstrap + verdict lattice."
        )
    )
    parser.add_argument("--smoke", action="store_true", help="first 10 contexts; _smoke prefix")
    parser.add_argument("--n-contexts", type=int, default=None, help="smoke-only override")
    parser.add_argument("--out-root", type=pathlib.Path, default=None, help="durable out-root")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--store-prefix", default=None, help="HF prefix of the pair store")
    parser.add_argument("--store-revision", default=None, help="data-repo sha for store fetch")
    parser.add_argument("--gen-prefix", default=HF_PREFIX, help="HF prefix of the P-Gen outputs")
    parser.add_argument("--gen-revision", default=None)
    parser.add_argument(
        "--gen-local-dir",
        type=pathlib.Path,
        default=None,
        help="local P-Gen dir (sentinel-verified)",
    )
    parser.add_argument("--boot-draws", type=int, default=None, help="bootstrap draw override")
    parser.add_argument(
        "--planned-wall-hours", type=float, default=PLANNED_FITS_WALL_H, help="plan section-9 row"
    )
    parser.add_argument(
        "--force-canonical",
        action="store_true",
        help="skip the gram fast path: canonical solver at layers {14,17,19,26} (contingency)",
    )
    parser.add_argument("--import-check", action="store_true")
    parser.add_argument("--list-arms", action="store_true")
    args = parser.parse_args(argv)

    if args.list_arms:
        print(
            json.dumps(
                {
                    "arms": list(ARM_NAMES),
                    "ladder_ks": list(LADDER_KS),
                    "read_out_layers": list(READ_OUT_LAYERS),
                    "p2_layers": list(P2_LAYERS),
                    "primary_estimator": PRIMARY_ESTIMATOR,
                }
            )
        )
        return
    if args.import_check:
        run_import_check()
        return

    t_start = time.monotonic()
    dev = _resolve_device(args.device)
    if args.smoke:
        n_contexts = args.n_contexts if args.n_contexts is not None else SMOKE_N_CONTEXTS
        assert 0 < n_contexts <= N_CTX
        root = args.out_root or pathlib.Path("/tmp/issue-823-smoke/ladder_fits")
        if smoke_root_aliases_production(root):
            parser.error(
                f"--smoke out-root {root} resolves at/under the production out-root "
                f"{PROD_POD_OUT_ROOT} — a smoke run must never write artifacts, "
                "checkpoints, or sentinels into the production tree"
            )
        out_prefix = HF_PREFIX + "_smoke"
        boot_n = args.boot_draws or SMOKE_BOOT_N
    else:
        if args.n_contexts is not None and args.n_contexts != N_CTX:
            parser.error("--n-contexts is smoke-only; production runs the full 5000 contexts")
        n_contexts = N_CTX
        if args.out_root is not None:
            root = args.out_root
        elif pathlib.Path("/workspace").exists():
            root = pathlib.Path("/workspace/eps/out/issue823_ladder")
        else:
            parser.error("production off-pod requires an explicit --out-root")
        out_prefix = HF_PREFIX
        boot_n = args.boot_draws or BOOT_N
    store_prefix = args.store_prefix or out_prefix
    eval_dir = root / "eval_results"
    ckpt_dir = root / "fit_checkpoints"
    eval_dir.mkdir(parents=True, exist_ok=True)
    solver_mode = "canonical-contingency" if args.force_canonical else "gram-fast-path"
    logger.info(
        "P-Fit: n_contexts=%d smoke=%s dev=%s root=%s solver=%s",
        n_contexts,
        args.smoke,
        dev,
        root,
        solver_mode,
    )

    # ── stage inputs ──
    log_phase("pfit_stage")
    inputs = load_inputs(root, store_prefix, args.store_revision, n_contexts)
    gen_paths, gen_revision = fetch_gen_inputs(
        root / "gen_inputs", args.gen_prefix, args.gen_revision, args.gen_local_dir
    )
    sentinel = verify_gen_sentinel(gen_paths)
    assert_mask_gate_schema(sentinel)
    by_persona = load_pair_rows(gen_paths, n_contexts)
    # capture_digest is REQUIRED by stage_pair_store on both branches (local +
    # remote), so the manipulation accounting always sees it — never None here.
    capture_digest = inputs.capture_digest

    metadata = {
        "script": "scripts/issue823_ladder_fits.py",
        "task": 823,
        "followup_label": FOLLOWUP_LABEL,
        **as_metadata_dict(git_provenance(), phase="pfit"),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "n_contexts": n_contexts,
        "smoke": args.smoke,
        "device": str(dev),
        "solver_mode": solver_mode,
        "primary_estimator": PRIMARY_ESTIMATOR,
        "gen_revision": gen_revision,
        "parent_rev": PARENT_REV,
        "bundle_rev": BUNDLE_REV,
    }

    # ── mask + drop-rule (kill criterion 1, v11 class-split semantics) ──
    log_phase("pfit_mask")
    mask_ids, gathers, drop_record = build_mask_and_gathers(inputs, by_persona)
    logger.info(
        "mask: %d contexts; drops per arm: %s; integrity per arm: %s",
        len(mask_ids),
        drop_record["new_drops_per_arm"],
        {a: c["integrity"] for a, c in drop_record["new_drops_per_arm_by_class"].items()},
    )
    worst_arm = mask_integrity_verdict(drop_record)
    if worst_arm[1] > NEW_DROPS_ABORT_PER_ARM:
        designed_abort(
            eval_dir,
            "mask_integrity",
            RC_MASK_ABORT,
            {"drop_record": drop_record, "worst_arm_integrity": worst_arm},
            args.smoke,
        )

    # ── split integrity (R1/R2/R3) — immediately after mask materialization,
    # BEFORE G2/G1/P1, so a membership or count drift halts the round before
    # any gate runs or any expensive artifact is written ──
    log_phase("pfit_split_integrity")
    pre_split = checked_fixed_split(N_CTX)
    if n_contexts == N_CTX:
        predrop = predrop_banked_split_check(pre_split, inputs.parent_valid_ids)
    else:
        # Smoke blind-spot (enumerated in the module docstring): the banked-split
        # assert quantifies over the FULL 4998-id parent mask, unreachable at smoke n.
        predrop = {"skipped_smoke": True}
    subsets, split_drops = realized_split_with_drops(
        pre_split,
        inputs.parent_valid_ids,
        mask_ids,
    )
    realized_train = len(subsets["train"])
    d_disp = d_boundary_disposition(realized_train)
    rung_table = p2_rung_table(realized_train)
    logger.info(
        "split integrity: realized %d/%d/%d; drops %s; d-rung %s",
        realized_train,
        len(subsets["val"]),
        len(subsets["test"]),
        {k: v for k, v in split_drops.items() if k != "total_drops"},
        d_disp["d_rung_status"],
    )

    # ── manipulation checks (before the lattice: R6 conditioning inputs) ──
    log_phase("pfit_manipulation_checks")
    m1 = m1_tfidf(by_persona, mask_ids)
    m2 = m2_paired_separation(inputs, mask_ids)
    m3 = m3_accounting(by_persona, capture_digest)
    manip = {
        "metadata": metadata,
        "m1_tfidf": m1,
        "m2_paired_separation": m2,
        "m3_accounting": m3,
        "distinct": bool(m1["m1_pass"] and m2["m2_pass"]),
    }
    write_json(eval_dir / "ladder_manipulation_checks.json", manip)

    # ── folds (parent parity: KFold(5, shuffle, seed 0); depends only on n) ──
    folds = list(
        KFold(n_splits=P1_N_FOLDS, shuffle=True, random_state=P1_FOLD_SEED).split(
            np.zeros(len(mask_ids))
        )
    )
    fold_ns = [len(tr) for tr, _ in folds]
    cap_bindable = any(DOF_CAP * n_tr < D_BOUNDARY for n_tr in fold_ns)
    if not args.smoke:
        assert min(fold_ns) > HIDDEN, (
            f"P1 GCV requires n_train > d ({HIDDEN}); realized min fold n_train "
            f"{min(fold_ns)} — unreachable under the drop-rule abort (plan section 4.3)"
        )

    # ── G2 solver parity (kill criterion 4) ──
    g2_record: dict = {
        "slices": [],
        "tolerances": {"max_rel": G2_MAX_REL_TOL, "delta_r2": G2_DELTA_R2_TOL},
    }
    contingency = args.force_canonical
    if not contingency:
        log_phase("pfit_g2")
        for pass_dev in (dev, torch.device("cpu")):
            slice_records = []
            for layer, fold_idx in G2_SLICES:
                tr, te = folds[fold_idx % len(folds)]
                ids_tr, ids_te = mask_ids[tr], mask_ids[te]
                x_tr = inputs.input_col(layer, ids_tr)
                x_te = inputs.input_col(layer, ids_te)
                y_tr = arm_target(inputs, gathers, "own", layer, ids_tr, mask_ids)
                y_te = arm_target(inputs, gathers, "own", layer, ids_te, mask_ids)
                fact = factorize_robust(x_tr, pass_dev)
                lam_fast, vty, ymu = FFC._gcv_solve(fact, y_tr)
                pred_fast = FFC._apply(fact, lam_fast, vty, ymu, FFC._cross_kernel(fact, x_te))
                pred_slow = ridge_fit_predict(x_tr, y_tr, x_te)
                lam_slow = svd_gcv_lambda(x_tr, y_tr, FFC.LAMBDAS)
                scale = float(np.abs(pred_slow).max()) + 1e-12
                max_rel = float(np.abs(pred_fast - pred_slow).max()) / scale
                d_r2 = abs(PR._pooled_r2(pred_fast, y_te) - PR._pooled_r2(pred_slow, y_te))
                slice_records.append(
                    {
                        "layer": layer,
                        "fold": fold_idx,
                        "device": str(pass_dev),
                        "max_rel": max_rel,
                        "delta_r2": float(d_r2),
                        "lambda_gram": lam_fast,
                        "lambda_canonical": lam_slow,
                        "lambda_agree": bool(lam_fast == lam_slow),
                        "pass": bool(max_rel <= G2_MAX_REL_TOL and d_r2 <= G2_DELTA_R2_TOL),
                    }
                )
            g2_record["slices"].extend(slice_records)
            g2_pass = all(s["pass"] for s in slice_records)
            g2_record["pass"] = g2_pass
            if g2_pass:
                break
            logger.warning(
                "G2 FAIL on %s; %s",
                pass_dev,
                "CPU-float64 eigh fallback"
                if pass_dev == dev and pass_dev.type != "cpu"
                else "fallback exhausted",
            )
            if pass_dev.type == "cpu" or dev.type == "cpu":
                break
        if not g2_record.get("pass", False):
            if args.smoke:
                logger.warning("[smoke-informational] G2 verdict FAIL at smoke n (expected)")
            else:
                contingency = True
                logger.error("G2 FAIL after CPU fallback — canonical contingency engaged")
    g2_record["contingency_engaged"] = contingency
    # EFFECTIVE post-gate solver mode: a G2 FAIL flips gram-fast-path ->
    # contingency ABOVE, so the arg-time value is stale here. Recomputed before
    # any checkpoint fingerprint / metadata consumer — a resume across the flip
    # must be refused (mixed-solver headline), never silently mixed.
    solver_mode = "canonical-contingency" if contingency else "gram-fast-path"
    metadata["solver_mode"] = solver_mode
    fit_layers = list(P2_LAYERS) if contingency else list(range(inputs.n_layers))
    planned_wall_h = args.planned_wall_hours + (CONTINGENCY_EXTRA_WALL_H if contingency else 0.0)

    # ── G1 reproduce gate (SKIPPED at smoke n — blind-spot enumerated) ──
    g1_record: dict = {"skipped_smoke": args.smoke}
    if not args.smoke:
        log_phase("pfit_g1")
        banked = load_banked_fold_means()
        parent_ids = inputs.parent_valid_ids
        g1_folds = list(
            KFold(n_splits=P1_N_FOLDS, shuffle=True, random_state=P1_FOLD_SEED).split(
                np.zeros(len(parent_ids))
            )
        )
        g1_cells = {}
        precheck = None
        g1_record["solver"] = solver_mode
        for layer in READ_OUT_LAYERS:
            comps: dict[str, list[tuple[float, float]]] = {"own": [], "plain": []}
            for tr, te in g1_folds:
                x_tr = inputs.input_col(layer, parent_ids[tr])
                x_te = inputs.input_col(layer, parent_ids[te])
                if contingency:
                    fact, kev = None, None
                else:
                    fact = factorize_robust(x_tr, dev)
                    kev = FFC._cross_kernel(fact, x_te)
                for arm in ("own", "plain"):
                    y_tr = arm_target(inputs, gathers, arm, layer, parent_ids[tr], mask_ids)
                    y_te = arm_target(inputs, gathers, arm, layer, parent_ids[te], mask_ids)
                    if contingency:
                        # Contingency parity verification (plan kill 4): the banked
                        # constants came from the canonical solver, so reproducing
                        # them through ridge_fit_predict — the EFFECTIVE headline
                        # solver under contingency — verifies the contingency path
                        # end-to-end; a G1 FAIL below takes the rc=7 designed abort.
                        pred = ridge_fit_predict(x_tr, y_tr, x_te)
                    else:
                        lam, vty, ymu = FFC._gcv_solve(fact, y_tr)
                        pred = FFC._apply(fact, lam, vty, ymu, kev)
                    comps[arm].append(
                        (
                            float(((y_te - pred) ** 2).sum()),
                            float(((y_te - y_te.mean(0)) ** 2).sum()),
                        )
                    )
            for arm in ("own", "plain"):
                if precheck is None:
                    precheck = reduction_convention_precheck(comps[arm])
                fm = fold_mean_r2(comps[arm])
                want = banked[(arm, layer)]
                g1_cells[f"{arm}:L{layer}"] = {
                    "refit_fold_mean_r2": fm,
                    "refit_pooled_r2": pooled_r2_from_components(comps[arm]),
                    "banked_fold_mean_r2": want,
                    "abs_diff": abs(fm - want),
                    "pass": bool(abs(fm - want) <= G1_TOL),
                }
        g1_record.update(
            {
                "cells": g1_cells,
                "reduction_convention_precheck": precheck,
                "tolerance": G1_TOL,
                "convention": "fold-mean vs banked fold-mean (LIKE-FOR-LIKE; banked JSON stores per-fold arrays)",
                "pass": all(c["pass"] for c in g1_cells.values()),
            }
        )
        if not g1_record["pass"]:
            designed_abort(
                eval_dir, "g1_reproduce", RC_SOLVER_PARITY_ABORT, {"g1": g1_record}, args.smoke
            )

    # ── P1 headline fits (G3 pilot on the first cell; per-layer checkpoints) ──
    log_phase("pfit_p1")
    n_mask = len(mask_ids)
    fp_p1 = checkpoint_fingerprint(
        mask_ids,
        {
            "chunk": "p1",
            "solver": solver_mode,  # EFFECTIVE post-gate mode (recomputed above)
            "n_contexts": n_contexts,
            "store_identity": inputs.store_identity,  # consumed pair-store generation
        },
    )
    n_arms = len(ARM_NAMES)
    p1_sres = np.full((n_arms, inputs.n_layers, n_mask), np.nan)
    p1_stot = np.full((n_arms, inputs.n_layers, n_mask), np.nan)
    p1_id_sres = np.full((n_arms, inputs.n_layers, n_mask), np.nan)
    p1_id_stot = np.full((n_arms, inputs.n_layers, n_mask), np.nan)
    p1_cells: dict[str, dict] = {}
    p1_baselines: dict[str, dict] = {}
    g3_record: dict = {}
    unit, total_units = 0, len(fit_layers) * P1_N_FOLDS
    for layer in fit_layers:
        name = f"p1_L{layer:02d}"
        if chunk_done(ckpt_dir, name, fp_p1):
            z = np.load(ckpt_dir / f"{name}.npz", allow_pickle=True)
            p1_sres[:, layer, :] = z["sres"]
            p1_stot[:, layer, :] = z["stot"]
            p1_id_sres[:, layer, :] = z["id_sres"]
            p1_id_stot[:, layer, :] = z["id_stot"]
            p1_cells.update(json.loads(str(z["cells"])))
            p1_baselines.update(json.loads(str(z["baselines"])))
            unit += P1_N_FOLDS
            logger.info("[p1] resume: layer %d loaded from checkpoint", layer)
            continue
        x_full = inputs.input_col(layer, mask_ids)
        y_full = {
            arm: arm_target(inputs, gathers, arm, layer, mask_ids, mask_ids) for arm in ARM_NAMES
        }
        layer_cells: dict[str, dict] = {}
        layer_base: dict[str, dict] = {}
        acc: dict[str, list] = {arm: [] for arm in ARM_NAMES}
        for f_idx, (tr, te) in enumerate(folds):
            t_cell = time.monotonic()
            x_tr, x_te = x_full[tr], x_full[te]
            if contingency:
                fact = None
            else:
                fact = factorize_robust(x_tr, dev)
                kev = FFC._cross_kernel(fact, x_te)
            for a_idx, arm in enumerate(ARM_NAMES):
                y_tr, y_te = y_full[arm][tr], y_full[arm][te]
                if contingency:
                    pred = ridge_fit_predict(x_tr, y_tr, x_te)
                    lam, dof = None, None
                else:
                    lam, vty, ymu = FFC._gcv_solve(fact, y_tr)
                    pred = FFC._apply(fact, lam, vty, ymu, kev)
                    filt = fact["w"] / (fact["w"] + lam)
                    dof = float(filt.sum())
                sres, stot = per_context_ss(pred, y_te)
                p1_sres[a_idx, layer, te] = sres
                p1_stot[a_idx, layer, te] = stot
                id_pred = identity_bias_predict(x_tr, y_tr, x_te)
                id_sres, id_stot = per_context_ss(id_pred, y_te)
                p1_id_sres[a_idx, layer, te] = id_sres
                p1_id_stot[a_idx, layer, te] = id_stot
                acc[arm].append(
                    {
                        "fold": f_idx,
                        "ss_res": float(sres.sum()),
                        "ss_tot": float(stot.sum()),
                        "r2": 1.0 - float(sres.sum()) / (float(stot.sum()) + 1e-12),
                        "lambda": lam,
                        "lambda_edge": SSP.lambda_edge(lam, FFC.LAMBDAS) if lam else None,
                        "dof": dof,
                        "n_train": int(len(tr)),
                    }
                )
                # House-rule pair on EVERY fitted cell: predictions are not
                # persisted, so a layer gate here would make the retrieval read
                # unrecoverable for ungated layers without refits.
                key = f"{arm}:L{layer}:fold{f_idx}"
                layer_base[key] = cell_baselines(x_tr, y_tr, x_te, y_te, pred)
            unit += 1
            elapsed = time.monotonic() - t_cell
            print(
                f"[p1] unit {unit}/{total_units} L={layer} fold={f_idx} elapsed={elapsed:.1f}s",
                flush=True,
            )
            if unit == 1:
                per_cell_s = elapsed
                projected_h = per_cell_s * total_units / 3600.0
                g3_record = {
                    "per_cell_s": per_cell_s,
                    "total_units": total_units,
                    "projected_wall_h": projected_h,
                    "planned_wall_h": planned_wall_h,
                    "abort_factor": FITS_WALL_ABORT_FACTOR,
                    "pass": bool(projected_h <= FITS_WALL_ABORT_FACTOR * planned_wall_h),
                }
                logger.info("[G3] %s", g3_record)
                if not g3_record["pass"]:
                    designed_abort(
                        eval_dir, "fits_wall", RC_FITS_WALL_ABORT, {"g3": g3_record}, args.smoke
                    )
        for arm in ARM_NAMES:
            comps = [(c["ss_res"], c["ss_tot"]) for c in acc[arm]]
            layer_cells[f"{arm}:L{layer}"] = {
                "pooled_r2": pooled_r2_from_components(comps),
                "fold_mean_r2": fold_mean_r2(comps),
                "folds": acc[arm],
                "identity_bias_pooled_r2": 1.0
                - float(np.nansum(p1_id_sres[ARM_NAMES.index(arm), layer]))
                / (float(np.nansum(p1_id_stot[ARM_NAMES.index(arm), layer])) + 1e-12),
                "estimator_degenerate": estimator_degenerate(min(fold_ns)),
            }
        p1_cells.update(layer_cells)
        p1_baselines.update(layer_base)
        save_chunk(
            ckpt_dir,
            name,
            {
                "sres": p1_sres[:, layer, :],
                "stot": p1_stot[:, layer, :],
                "id_sres": p1_id_sres[:, layer, :],
                "id_stot": p1_id_stot[:, layer, :],
                "cells": np.array(json.dumps(layer_cells)),
                "baselines": np.array(json.dumps(layer_base)),
            },
            fp_p1,
        )

    # Row-coverage set-check (plan section 3): every registered (context x arm x layer)
    # row present + finite BEFORE any bootstrap or contrast is computed.
    for a_idx, arm in enumerate(ARM_NAMES):
        for layer in fit_layers:
            row = p1_sres[a_idx, layer, :]
            assert np.isfinite(row).all(), (
                f"row-coverage: missing per-context rows for arm={arm} L={layer} — refusing "
                "to compute any bootstrap/contrast on an incomplete (context x arm) key set"
            )

    # persist per-context arrays (bootstrap + every registered re-read consumes these)
    np.savez(
        eval_dir / "percontext_ladder.npz",
        p1_ss_res=p1_sres,
        p1_ss_tot=p1_stot,
        p1_identity_ss_res=p1_id_sres,
        p1_identity_ss_tot=p1_id_stot,
        context_ids=mask_ids,
        arm_names=np.array(ARM_NAMES),
        layers=np.array(fit_layers),
    )

    # ── bootstrap + lattice (R4) + mixture-floor re-reads ──
    log_phase("pfit_bootstrap")
    ss_res_cells = {
        (arm, layer): p1_sres[ARM_NAMES.index(arm), layer, :]
        for arm in ARM_NAMES
        for layer in READ_OUT_LAYERS
    }
    ss_tot_cells = {
        (arm, layer): p1_stot[ARM_NAMES.index(arm), layer, :]
        for arm in ARM_NAMES
        for layer in READ_OUT_LAYERS
    }
    boot = bootstrap_paired(ss_res_cells, ss_tot_cells, boot_n, BOOT_SEED)
    pooled = {
        (arm, layer): 1.0
        - ss_res_cells[(arm, layer)].sum() / (ss_tot_cells[(arm, layer)].sum() + 1e-12)
        for (arm, layer) in ss_res_cells
    }
    delta_mean = float(np.mean([pooled[("k1", L)] - pooled[("k16", L)] for L in READ_OUT_LAYERS]))
    verdict = lattice_verdict(delta_mean, boot["ci_low_delta_mean"], boot["ci_high_delta_mean"])
    conditioning = conditioned_interpretation(verdict, m1["m1_pass"], m2["m2_pass"])
    from scipy.stats import spearmanr

    k_vals = [1, 2, 4, 8, 16]
    r2_by_k = {L: [pooled[(f"k{k}", L)] for k in k_vals] for L in READ_OUT_LAYERS}
    spearman = {f"L{L}": float(spearmanr(k_vals, r2_by_k[L]).statistic) for L in READ_OUT_LAYERS}
    # Mixture-floor decomposition (registered reads; pure re-reductions)
    ssres_ladder = {
        f"{arm}:L{L}": float(ss_res_cells[(arm, L)].sum())
        for arm in ARM_NAMES
        for L in READ_OUT_LAYERS
    }
    fixed_ref = {
        f"k{k}:L{L}": fixed_reference_r2(ss_res_cells[(f"k{k}", L)], ss_tot_cells[("k1", L)])
        for k in k_vals
        for L in READ_OUT_LAYERS
    }
    implied = {}
    for k in (2, 4, 8, 16):
        for L in READ_OUT_LAYERS:
            between, n_tot = 0.0, 0
            row0 = {int(c): j for j, c in enumerate(inputs.store_ctx[0])}
            for p, pos, rows in gathers[f"k{k}"]:
                if p == 0:
                    n_tot += len(pos)
                    continue
                ctxs = [int(mask_ids[q]) for q in pos]
                vp = inputs.store_v[p][rows, L, :].astype(np.float64)
                v0 = inputs.store_v[0][np.array([row0[c] for c in ctxs]), L, :].astype(np.float64)
                m_p = (vp - v0).mean(axis=0)
                between += len(pos) * float(m_p @ m_p)
                n_tot += len(pos)
            mean_sstot = float(ss_tot_cells[(f"k{k}", L)].mean())
            implied[f"k{k}:L{L}"] = {
                "between_persona_mean_shift_energy": between / max(n_tot, 1),
                "implied_r2_penalty": (between / max(n_tot, 1)) / (mean_sstot + 1e-12),
                "observed_delta_vs_k1": float(pooled[("k1", L)] - pooled[(f"k{k}", L)]),
            }

    # Sensitivity-only dof-capped re-selection (R5: never the primary)
    sensitivity = None
    if cap_bindable and not contingency:
        log_phase("pfit_sensitivity_dof_cap")
        fp_sens = checkpoint_fingerprint(
            mask_ids,
            {
                "chunk": "p1_sens",
                "solver": solver_mode,
                "n_contexts": n_contexts,
                "dof_cap": DOF_CAP,
                "store_identity": inputs.store_identity,
            },
        )
        # ONE shared factorization per (layer, fold) — 15 eighs, never 105
        # (arm loop solves all capped targets against the shared eigh).
        sens_cells = dof_cap_sensitivity(inputs, gathers, mask_ids, folds, dev, ckpt_dir, fp_sens)
        sensitivity = {
            "note": "SENSITIVITY ONLY — the registered primary is gcv-pure-parent-parity; "
            "variants are never mixed within a contrast",
            "cells": sens_cells,
        }

    p1_result = {
        "metadata": metadata,
        "primary_estimator": PRIMARY_ESTIMATOR,
        "delta_mean": delta_mean,
        "ci_low_delta_mean": boot["ci_low_delta_mean"],
        "ci_high_delta_mean": boot["ci_high_delta_mean"],
        "lattice": conditioning,
        "lattice_thresholds": {"h1_delta": H1_DELTA_THRESHOLD, "h2_band": H2_BAND},
        "read_out_layers": list(READ_OUT_LAYERS),
        "trait_layers": TRAIT_LAYERS,
        "delta_per_layer": {
            f"L{L}": float(pooled[("k1", L)] - pooled[("k16", L)]) for L in READ_OUT_LAYERS
        },
        "spearman_k_r2_descriptive": spearman,
        "pooled_r2": {f"{a}:L{L}": float(v) for (a, L), v in pooled.items()},
        "bootstrap": boot,
        "cells": p1_cells,
        "dof_cap_bindable": bool(cap_bindable),
        "sensitivity_dof_capped": sensitivity,
        "mixture_floor": {
            "ss_res_ladder": ssres_ladder,
            "fixed_reference_denominator_r2": fixed_ref,
            "implied_mixture_penalty": implied,
        },
        "drop_accounting": drop_record,
        "gates": {"g1": g1_record, "g2": g2_record, "g3": g3_record},
        "estimator_degeneracy_rule": "n_train < 3585 => degenerate for absolute reads",
    }
    write_json(eval_dir / "ladder_r2_p1.json", p1_result)
    logger.info(
        "P1: delta_mean=%.4f CI=[%.4f, %.4f] label=%s distinct=%s",
        delta_mean,
        boot["ci_low_delta_mean"],
        boot["ci_high_delta_mean"],
        verdict,
        conditioning["distinct"],
    )

    # ── P2: single-split protocol (split integrity ran up top, pre-gates) ──
    log_phase("pfit_p2")
    fp_p2 = checkpoint_fingerprint(
        mask_ids,
        {
            "chunk": "p2",
            "split_seed": SPLIT_SEED,
            "n_contexts": n_contexts,
            "solver": solver_mode,
            "store_identity": inputs.store_identity,
        },
    )
    split_block = {
        "call": "fixed_split(n_ctx=5000, n_train=3600, n_val=400, n_test=1000, seed=42)",
        "predrop_banked_check": predrop,
        "drops": split_drops,
        "realized": {s: int(len(subsets[s])) for s in subsets},
        "d_boundary": d_disp,
    }
    p2_full: dict[str, dict] = {}
    p2_pc: dict[str, np.ndarray] = {}
    n_ladder: dict[str, dict] = {}
    per_persona: dict[str, dict] = {}
    gmix: dict[str, dict] = {}
    if contingency:
        # Item-10 contract: P2's single-split fits run on the gram fast-path
        # primitives the G2 gate rejected — WITHHELD (no unverified fast-path
        # fits reported; no percontext_ladder_p2.npz written). The priced
        # canonical contingency covers P1 only.
        p2_result = p2_withheld_result(metadata, split_block)
        logger.warning("P2 WITHHELD — solver-parity contingency engaged (fast path unverified)")
    else:
        tr_ids, va_ids, te_ids = subsets["train"], subsets["val"], subsets["test"]
        wide = SSP.LAMBDAS_WIDE
        p2_pc["test_ids"] = te_ids
        for layer in P2_LAYERS:
            x_tr = inputs.input_col(layer, tr_ids)
            x_va = inputs.input_col(layer, va_ids)
            x_te = inputs.input_col(layer, te_ids)
            fact = factorize_robust(x_tr, dev)
            kva = FFC._cross_kernel(fact, x_va)
            kte = FFC._cross_kernel(fact, x_te)
            for arm in ARM_NAMES:
                y_tr = arm_target(inputs, gathers, arm, layer, tr_ids, mask_ids)
                y_va = arm_target(inputs, gathers, arm, layer, va_ids, mask_ids)
                y_te = arm_target(inputs, gathers, arm, layer, te_ids, mask_ids)
                vty, ymu = FFC._vty_ymu(fact, y_tr)
                lam, val_r2 = val_select_lambda(
                    fact, vty, ymu, kva, y_va, wide, degenerate_ok=args.smoke
                )
                pred = FFC._apply(fact, lam, vty, ymu, kte)
                sres, stot = per_context_ss(pred, y_te)
                p2_pc[f"full_{arm}_L{layer}_sres"] = sres
                p2_pc[f"full_{arm}_L{layer}_stot"] = stot
                p2_full[f"{arm}:L{layer}"] = {
                    "test_r2": 1.0 - float(sres.sum()) / (float(stot.sum()) + 1e-12),
                    "val_r2": val_r2,
                    "val_selection_degenerate": bool(not np.isfinite(val_r2)),
                    "lambda": lam,
                    "lambda_edge": SSP.lambda_edge(lam, wide),
                    "n_train": int(realized_train),
                    "estimator_degenerate": estimator_degenerate(realized_train),
                    "baselines": cell_baselines(x_tr, y_tr, x_te, y_te, pred),
                }
            print(f"[p2-full] layer {layer} done", flush=True)

        # n-ladder: nested seeded subsets of the realized train rows
        ladder_perm = np.random.default_rng(SPLIT_SEED).permutation(realized_train)
        n_ladder: dict[str, dict] = {}
        for rung in rung_table:
            if rung["status"] == "UNREALIZABLE":
                n_ladder[f"n{rung['n_train']}"] = {"status": "UNREALIZABLE", **rung}
                continue
            n_tr = rung["n_train"]
            name = f"p2_rung{n_tr}"
            if chunk_done(ckpt_dir, name, fp_p2):
                z = np.load(ckpt_dir / f"{name}.npz", allow_pickle=True)
                n_ladder[f"n{n_tr}"] = json.loads(str(z["cells"]))
                logger.info("[p2] resume: rung %d loaded", n_tr)
                continue
            rows = tr_ids[np.sort(ladder_perm[:n_tr])]
            rung_cells: dict[str, dict] = {"status": rung["status"], "n_train": n_tr}
            for layer in P2_LAYERS:
                x_tr = inputs.input_col(layer, rows)
                x_va = inputs.input_col(layer, va_ids)
                x_te = inputs.input_col(layer, te_ids)
                fact = factorize_robust(x_tr, dev)
                kva = FFC._cross_kernel(fact, x_va)
                kte = FFC._cross_kernel(fact, x_te)
                for arm in ARM_NAMES:
                    y_tr = arm_target(inputs, gathers, arm, layer, rows, mask_ids)
                    y_va = arm_target(inputs, gathers, arm, layer, va_ids, mask_ids)
                    y_te = arm_target(inputs, gathers, arm, layer, te_ids, mask_ids)
                    vty, ymu = FFC._vty_ymu(fact, y_tr)
                    lam, val_r2 = val_select_lambda(
                        fact, vty, ymu, kva, y_va, wide, degenerate_ok=args.smoke
                    )
                    pred = FFC._apply(fact, lam, vty, ymu, kte)
                    rung_cells[f"{arm}:L{layer}"] = {
                        "test_r2": PR._pooled_r2(pred, y_te),
                        "val_r2": val_r2,
                        "val_selection_degenerate": bool(not np.isfinite(val_r2)),
                        "lambda": lam,
                        "lambda_edge": SSP.lambda_edge(lam, wide),
                        "estimator_degenerate": estimator_degenerate(n_tr),
                        "baselines": cell_baselines(x_tr, y_tr, x_te, y_te, pred),
                    }
            n_ladder[f"n{n_tr}"] = rung_cells
            save_chunk(ckpt_dir, name, {"cells": np.array(json.dumps(rung_cells))}, fp_p2)
            print(f"[p2-ladder] rung n_train={n_tr} done", flush=True)

        # per-persona control (a/b/c/d) + G_mix
        per_persona: dict[str, dict] = {}
        gmix_inputs: dict[int, list] = {k: [] for k in PER_PERSONA_KS}
        for k in PER_PERSONA_KS:
            name = f"p2_pp_k{k}"
            if chunk_done(ckpt_dir, name, fp_p2):
                z = np.load(ckpt_dir / f"{name}.npz", allow_pickle=True)
                per_persona[f"k{k}"] = json.loads(str(z["cells"]))
                gmix_inputs[k] = json.loads(str(z["gmix"]))
                n_restored = restore_pp_arrays(z, p2_pc)
                logger.info(
                    "[p2] resume: per-persona k=%d loaded (%d pp_ arrays restored)",
                    k,
                    n_restored,
                )
                continue
            k_cells: dict[str, dict] = {}
            k_gmix: list = []
            for p in range(k):
                rows = {
                    s: np.array([i for i in subsets[s] if int(i) % k == p], dtype=int)
                    for s in ("train", "val", "test")
                }
                floors_ok = all(len(rows[s]) >= MIN_CELL_FLOORS[s] for s in MIN_CELL_FLOORS)
                if not floors_ok:
                    if not args.smoke:
                        raise RuntimeError(
                            f"per-persona cell k={k} p={p} under floor "
                            f"{ {s: len(rows[s]) for s in rows} } — data bug at production n"
                        )
                    k_cells[f"p{p}"] = {"status": "skipped_small_cell (smoke)"}
                    continue
                mixed_rng = np.random.default_rng(1000 * k + p)
                mix_tr = np.sort(mixed_rng.choice(tr_ids, size=len(rows["train"]), replace=False))
                mix_va = np.sort(mixed_rng.choice(va_ids, size=len(rows["val"]), replace=False))
                cell: dict[str, dict] = {}
                for layer in P2_LAYERS:
                    x_tr = inputs.input_col(layer, rows["train"])
                    x_va = inputs.input_col(layer, rows["val"])
                    x_te = inputs.input_col(layer, rows["test"])
                    arm = f"k{k}"
                    fact = factorize_robust(x_tr, dev)
                    kva = FFC._cross_kernel(fact, x_va)
                    kte = FFC._cross_kernel(fact, x_te)
                    out_layer: dict[str, dict] = {}
                    y_te_full_mean = None
                    for sub_name, target_arm in (("a_within", arm), ("b_same_ctx_k1", "k1")):
                        y_tr = arm_target(
                            inputs, gathers, target_arm, layer, rows["train"], mask_ids
                        )
                        y_va = arm_target(inputs, gathers, target_arm, layer, rows["val"], mask_ids)
                        y_te = arm_target(
                            inputs, gathers, target_arm, layer, rows["test"], mask_ids
                        )
                        vty, ymu = FFC._vty_ymu(fact, y_tr)
                        lam, val_r2 = val_select_lambda(
                            fact, vty, ymu, kva, y_va, wide, degenerate_ok=args.smoke
                        )
                        pred = FFC._apply(fact, lam, vty, ymu, kte)
                        sres, stot = per_context_ss(pred, y_te)
                        out_layer[sub_name] = {
                            "test_r2": 1.0 - float(sres.sum()) / (float(stot.sum()) + 1e-12),
                            "val_selection_degenerate": bool(not np.isfinite(val_r2)),
                            "lambda": lam,
                            "lambda_edge": SSP.lambda_edge(lam, wide),
                            "estimator_degenerate": True,
                            "baselines": cell_baselines(x_tr, y_tr, x_te, y_te, pred),
                        }
                        if sub_name == "a_within":
                            pc_key = f"pp_k{k}_p{p}_L{layer}"
                            p2_pc[f"{pc_key}_a_sres"] = sres
                            p2_pc[f"{pc_key}_a_stot"] = stot
                    # (c) matched-n mixed comparator (own factorization; mixed rows)
                    xm_tr = inputs.input_col(layer, mix_tr)
                    xm_va = inputs.input_col(layer, mix_va)
                    fact_m = factorize_robust(xm_tr, dev)
                    km_va = FFC._cross_kernel(fact_m, xm_va)
                    km_te = FFC._cross_kernel(fact_m, x_te)
                    ym_tr = arm_target(inputs, gathers, arm, layer, mix_tr, mask_ids)
                    ym_va = arm_target(inputs, gathers, arm, layer, mix_va, mask_ids)
                    y_te = arm_target(inputs, gathers, arm, layer, rows["test"], mask_ids)
                    vty, ymu = FFC._vty_ymu(fact_m, ym_tr)
                    lam_c, val_r2_c = val_select_lambda(
                        fact_m, vty, ymu, km_va, ym_va, wide, degenerate_ok=args.smoke
                    )
                    pred_c = FFC._apply(fact_m, lam_c, vty, ymu, km_te)
                    sres_c, stot_c = per_context_ss(pred_c, y_te)
                    out_layer["c_matched_n_mixed"] = {
                        "test_r2": 1.0 - float(sres_c.sum()) / (float(stot_c.sum()) + 1e-12),
                        "val_selection_degenerate": bool(not np.isfinite(val_r2_c)),
                        "lambda": lam_c,
                        "lambda_edge": SSP.lambda_edge(lam_c, wide),
                        "estimator_degenerate": True,
                        "baselines": cell_baselines(xm_tr, ym_tr, x_te, y_te, pred_c),
                    }
                    p2_pc[f"pp_k{k}_p{p}_L{layer}_c_sres"] = sres_c
                    p2_pc[f"pp_k{k}_p{p}_L{layer}_c_stot"] = stot_c
                    # (d) full-mixed triangulation: re-reduce the FULL arm-k P2 fit's
                    # persisted per-context test components on the persona-p test rows
                    # (ss_tot vs the FULL test-pool mean — fixed-denominator convention).
                    te_pos = np.array(
                        [j for j, i in enumerate(te_ids) if int(i) % k == p], dtype=int
                    )
                    d_sres = p2_pc[f"full_{arm}_L{layer}_sres"][te_pos]
                    d_stot = p2_pc[f"full_{arm}_L{layer}_stot"][te_pos]
                    out_layer["d_full_mixed_triangulation"] = {
                        "test_r2_fullpool_denominator": 1.0
                        - float(d_sres.sum()) / (float(d_stot.sum()) + 1e-12),
                        "note": "pure re-reduction of the persisted full-arm per-context "
                        "components; ss_tot vs the full test-pool mean",
                    }
                    cell[f"L{layer}"] = out_layer
                    if layer == P2_LAYERS[0]:
                        k_gmix.append(
                            {
                                "p": p,
                                "sres_a": p2_pc[f"pp_k{k}_p{p}_L{layer}_a_sres"].tolist(),
                                "stot_a": p2_pc[f"pp_k{k}_p{p}_L{layer}_a_stot"].tolist(),
                                "sres_c": sres_c.tolist(),
                                "stot_c": stot_c.tolist(),
                            }
                        )
                k_cells[f"p{p}"] = cell
                print(f"[p2-pp] k={k} p={p} done", flush=True)
            per_persona[f"k{k}"] = k_cells
            gmix_inputs[k] = k_gmix
            save_chunk(
                ckpt_dir,
                name,
                {
                    "cells": np.array(json.dumps(k_cells)),
                    "gmix": np.array(json.dumps(k_gmix)),
                    # Registered per-context artifact inputs (four-layer a/c ss
                    # arrays) ride the per-k checkpoint so a RESUMED block can
                    # restore them into the final archive (restore_pp_arrays).
                    **{key: val for key, val in p2_pc.items() if key.startswith(f"pp_k{k}_")},
                },
                fp_p2,
            )

        # G_mix (reported decomposition with a CI; k=8 PRIMARY; never a recovery fraction)
        gmix: dict[str, dict] = {}
        rng = np.random.default_rng(BOOT_SEED + 1)
        for k in PER_PERSONA_KS:
            cells = gmix_inputs[k]
            if not cells:
                gmix[f"k{k}"] = {"status": "no cells (smoke floors)"}
                continue
            gaps = []
            draws_per_p = []
            for c in cells:
                sa, ta = np.array(c["sres_a"]), np.array(c["stot_a"])
                sc, tc = np.array(c["sres_c"]), np.array(c["stot_c"])
                r2a = 1.0 - sa.sum() / (ta.sum() + 1e-12)
                r2c = 1.0 - sc.sum() / (tc.sum() + 1e-12)
                gaps.append(r2a - r2c)
                n_p = len(sa)
                idx = rng.integers(0, n_p, size=(boot_n, n_p))
                da = 1.0 - sa[idx].sum(1) / (ta[idx].sum(1) + 1e-12)
                dc = 1.0 - sc[idx].sum(1) / (tc[idx].sum(1) + 1e-12)
                draws_per_p.append(da - dc)
            g_draws = np.mean(draws_per_p, axis=0)
            gmix[f"k{k}"] = {
                "g_mix": float(np.mean(gaps)),
                "ci_low": float(np.quantile(g_draws, 0.025)),
                "ci_high": float(np.quantile(g_draws, 0.975)),
                "per_persona_gaps": [float(g) for g in gaps],
                "primary": bool(k == GMIX_PRIMARY_K),
                "layer": int(P2_LAYERS[0]),
                "note": "standalone P2 read; never divided by / narrated as a fraction of "
                "the P1 delta_mean",
            }

        assert_p2_percontext_complete(per_persona, p2_pc)
        np.savez(eval_dir / "percontext_ladder_p2.npz", **p2_pc)
        p2_result = {
            "status": "complete",
            "metadata": metadata,
            "split": split_block,
            "lambda_grid": "logspace(-2, 8, 21) val-selected",
            "full_arms": p2_full,
            "n_ladder": n_ladder,
            "per_persona": per_persona,
            "g_mix": gmix,
            "gmix_primary_k": GMIX_PRIMARY_K,
            "estimator_degeneracy_rule": "n_train < 3585 => degenerate for absolute reads; "
            "per-persona cells are matched-regime contrasts",
        }
    write_json(eval_dir / "ladder_singlesplit_p2.json", p2_result)

    # ── baselines JSON (house-rule pair, every fitted cell) ──
    baselines = {
        "metadata": metadata,
        "p1": p1_baselines,
        "p2_full": {key: cell["baselines"] for key, cell in p2_full.items()},
        "p2_n_ladder": {
            rk: {
                ck: c["baselines"]
                for ck, c in rung.items()
                if isinstance(c, dict) and "baselines" in c
            }
            for rk, rung in n_ladder.items()
        },
        "p2_per_persona": {
            kk: {
                pk: {
                    lk: {
                        sk: sc["baselines"]
                        for sk, sc in lc.items()
                        if isinstance(sc, dict) and "baselines" in sc
                    }
                    for lk, lc in pc.items()
                    if isinstance(pc, dict) and isinstance(lc, dict)
                }
                for pk, pc in kc.items()
                if isinstance(pc, dict)
            }
            for kk, kc in per_persona.items()
        },
        "chance_note": "chance_at_k = k / n_pool, stated per pool by knn_retrieval",
    }
    write_json(eval_dir / "ladder_baselines.json", baselines)

    # ── upload (text/JSON uploads ALWAYS; smoke -> _smoke prefix) ──
    log_phase("pfit_upload")
    expected = sorted(
        p.name for p in eval_dir.iterdir() if p.suffix in (".json", ".npz") and p.is_file()
    )
    path_in_repo = f"{out_prefix}/logs/fits"
    url = _upload_folder_filtered(
        local_dir=eval_dir,
        repo_id=DATA_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        allow_patterns=["*.json", "*.npz"],
        expected_repo_paths=[f"{path_in_repo}/{fn}" for fn in expected],
    )
    if not url:
        raise RuntimeError(
            f"P-Fit upload of {len(expected)} files to {DATA_REPO}/{path_in_repo} failed or "
            "verified incomplete — refusing to report P-Fit complete"
        )
    # Canonical-destination gate (shared gen gate, not a second copy): a
    # truthy OVERFLOW-repo URL must never satisfy the completion condition.
    _require_canonical_upload(url, f"{DATA_REPO}/{path_in_repo}")

    sentinel_dir = (
        pathlib.Path("/workspace/logs") if pathlib.Path("/workspace").exists() else root / "logs"
    )
    write_sentinel(
        sentinel_dir / sentinel_filename(args.smoke),
        {
            "kind": "epm:progress",
            "version": 1,
            "note": "P-Fit complete (inconsistent-origin-persona-ladder)",
            "phase": "pfit",
            "complete": True,
            "gates": {
                "g1_pass": g1_record.get("pass"),
                "g2_pass": g2_record.get("pass"),
                "g3_pass": g3_record.get("pass"),
                "contingency_engaged": contingency,
            },
            "p2_status": p2_result.get("status", "complete"),
            "delta_mean": delta_mean,
            "ci_low_delta_mean": boot["ci_low_delta_mean"],
            "ci_high_delta_mean": boot["ci_high_delta_mean"],
            "lattice_label_numeric": verdict,
            "distinct": conditioning["distinct"],
            "interpretation": conditioning["interpretation"],
            "primary_estimator": PRIMARY_ESTIMATOR,
            "hf_path_in_repo": path_in_repo,
            "elapsed_h": (time.monotonic() - t_start) / 3600.0,
            "metadata": metadata,
            "ts": time.time(),
        },
    )
    log_phase("done")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
