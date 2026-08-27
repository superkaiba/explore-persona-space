"""Issue #2569 leg 6 — denoised reduced-rank shift regression (plan v4 §4 leg 6; blockers B2/B3).

Per content arm (arms.json @ 3bb20debe2, ``kind == "content"``), regress the 16,400-row
matched-base-text realized-write matrix Delta (trained - base answer-span shift) on the base
context matrix C, with:

- **B3 row-pairing guard** (PRECONDITION of any subtraction): the exact ORDERED composite
  row keys ``(row_question_idx, row_sha)`` of ``pooled.pt`` (base), ``pooled_tf.pt``
  (trained) and ``lasttoken.pt`` (context) must match; on an order mismatch the stores are
  explicitly joined by composite ID; on MISSING or DUPLICATE composite keys the arm HALTs
  (siblings proceed). Key-source ladder: (i) id fields inside the .pt payloads (OBSERVED
  present: ``row_question_idx`` + ``row_sha``, schema probed 2026-08-25 at the pinned
  revisions), (ii) manifest row-id lists (observed ABSENT — manifests carry counts only),
  (iii) ``delta_tf/<arm>/pos.jsonl`` (observed NOT row-keyed for the corpus: 20 training
  rows, no ids — recorded unusable), (iv) the arm's ``raw_rows_*.jsonl`` id columns via
  ``--raw-rows-dir``. An arm with no usable key source HALTs — never a silent order
  assumption.
- **tbar VALUE check (DEMOTED, B3):** the computed Delta column mean is compared against the
  banked ``delta_tf/<arm>/tbar.pt``. MEASURED at the pin (2026-08-25): the banked tbar is the
  mean displacement over the arm's 20 TRAINING-mix rows (``meta.n_rows == 20``,
  ``meta.pos_path == issue1434_.../pos.jsonl``), NOT the 16,400-row corpus mean the plan's
  fp16-tolerance premise assumed — so the strict reproduction check is applied ONLY when the
  banked ``meta.n_rows`` equals the corpus row count, and otherwise the check records the
  per-layer cosine + norms with ``basis_mismatch: true`` (corroboration, never gating; a
  permutation of rows leaves every recorded value unchanged — the committed unit test
  asserts exactly that).
- **B2 split-half denoised-rank estimator:** rows split into two disjoint halves by
  conversation id (= ``row_sha``; seed 0; rows sharing a sha never straddle halves). WITHIN
  each half h, C_h and Delta_h are centered at the half means and the half operator is the
  ridge map M_h = (C_h^T C_h + lambda * n_h * I)^-1 C_h^T Delta_h from MATCHED-ROW
  within-half covariances. Cross-half covariance products ``C_h^T Delta_h'`` are BANNED:
  ``half_moments`` takes ONE row-index set and slices BOTH matrices with it, and
  ``fit_split_half`` asserts the two halves are disjoint — there is no API path that pairs
  one half's contexts with the other half's shifts. Held-out EVALUATION (fit M_1 on half 1,
  predict row-wise on half 2) is matched-row and stays. Denoised rank = the number of
  LEADING half-1 factors (descending sigma) that (a) greedily match a half-2 factor with
  factor |cos| >= 0.5 (factor cos = min of the input-side and output-side |cos| — both
  sides must agree), and (b) whose singular values exceed the CALIBRATED per-half noise
  threshold: p95 over 20 within-half row-shuffle draws (shuffle Delta rows within the half)
  of the null map's TOP singular value at the SAME lambda. Gavish-Donoho 4/sqrt(3)
  (arXiv 1305.5870) is quoted as an ANALYTIC REFERENCE only.
- **Mapping baselines (per plan leg 6 step 5):** each half map reports identity+learned-bias
  (same-dim 3584->3584; expected to fail) and kNN retrieval (chance = k/n_pool stated),
  scored on the opposite half.
- **Pooled cross-arm RRR:** fit on the 3 same-behavior sibling content arms, target arm held
  out (the #1979 fit-on-3-siblings construction; realized row counts recorded).
- **Factor-vector persistence (closes concern leg6-cross-arm-factor-vectors-unpersisted):**
  every unit ALSO writes ``L{layer}_{conv}_factors.npz`` — the per-half factor VECTORS with
  role-named arrays (``context_factors_half{1,2}`` = input-side singular vectors in the base
  context basis; ``shift_factors_half{1,2}`` = output-side singular vectors in the
  answer-span shift basis; columns ordered by descending sigma) plus an embedded
  ``schema_json`` carrying the orientation and the explicit basis bookkeeping. Each arm dir
  additionally gets ``operator_factors.pt`` in the atlas sidecar shape (keys ``u``/``s``/
  ``v``/``split_half_floor`` — the EXISTING consumer contract in issue2569_atlas.py step (5),
  concern leg7-atlas-writemap-operators-unpersisted; role-annotation keys ride alongside):
  an ALL-ROWS ridge refit at the primary unit's selected lambda, truncated to the top 64
  factors, floor = the raw operator vec-cosine between the two independent half fits.
- **Cross-arm shared-factor phase (``--cross-arm``; closes concern
  leg6-cross-arm-shared-factor-no-producer — plan leg 6 step 4 / H6 / the leg-6 success
  criterion):** per (layer, convention) cell, pairwise cross-arm factor cosines over each
  arm's DENOISED half-1 factors, greedily matched by min(|cos_context|, |cos_shift|) (the
  within-arm/unit-7c |cos| convention). The CRITERION band is the SELECTION-SYMMETRIC
  max-matched rotation null (option 1 of selection-symmetric-nulls.md, the leg-5
  dv3_max_matched_null shape): every null draw undergoes the IDENTICAL greedy matching over
  the same (r_a, r_b) factor grid via the SHARED ``_greedy_match_stat``, then the same
  max-over-matched reduction — p95 of the max distribution controls FWER at 0.05 per pair at
  ANY denoised rank k (the per-comparison band's false-positive rate is 1-(0.95)^k instead:
  0.79 at the registered range's k=30). The per-draw x per-match-slot matrix persists in the
  cell JSON so the band is recomputable post-hoc. The per-comparison #1345 band (REUSED
  ``issue1345_operator_comparison.raw_cosine_with_rotation_null``, computed ONCE per vector
  dimension on canonical basis-vector probes — a Haar rotation maps any unit vector to a
  uniform direction, so that null depends only on d) rides labeled UNCORRECTED for the
  record, and the within-arm split-half agreement rides as the NOISE FLOOR. A pair whose
  recorded factor bases differ is REFUSED with a recorded reason — never a fabricated
  cosine. Emits ``cross_arm/L{layer}_{conv}.json`` + ``cross_arm/summary.json`` carrying the
  registered-criterion verdict input: the count of cross-arm shared factors above the
  SYMMETRIC null over same-behavior arm pairs, with the pairs named.
- **Terminal HF upload (``run_upload``, LAST — after ``summary.json``):** the FULL leg6
  tree (per-arm unit JSONs + factor ``.npz`` + ``operator_factors.pt``, ``pooled/``,
  ``cross_arm/``, ``summary.json``) uploads per leaf dir to
  ``<--hf-prefix>/leg6/...`` on the HF data repo via ``upload_dir_sharded`` with fail-loud
  exact-set ``hub.verify_repo_paths_uploaded`` per leaf (leaves enumerated FROM DISK —
  every prefix the run wrote, never only the current phase's, #1773). Tensors cannot ride
  ``eval_results/`` (JSON/text only; ``*.npz`` is gitignored repo-wide) and
  ``operator_factors.pt`` is a plan-referenced downstream input (the atlas globs it) — an
  unuploaded sidecar makes leg 7's atlas rows drop on every cross-machine read (#521
  class). ``--skip-upload`` skips LOUDLY for smoke / local runs.

Context conventions: PRIMARY ``last_prompt`` (the plan-wide v_C = last-prompt-token state,
plan §6 pooling-convention row); companions ``last_ctx`` (the context-segment last token) and
``span_mean`` (pooled.pt's span-mean context object — the plan-ordered convention twin).

Store pins (HF dataset repo ``superkaiba1/explore-persona-space-data``): ``corpus_capture`` /
``corpus_capture_tf`` / ``delta_tf`` @ ``c07267285d``; ``lasttoken_ctx`` @ ``89bc6145``.
This driver reads a LOCAL staged mirror (``--staged-root``) laid out as
``<root>/issue1768_mapshift/<store>/<unit>/<file>`` — staging is the P-A dispatcher's job.

``torch.load(..., weights_only=False)`` is deliberate: these are self-produced,
revision-pinned project artifacts (#1900 precedent).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE numpy/torch: shared-VM thread caps freeze at import (#847/#891)

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("issue2569_leg6")

ISSUE = 2569
SPLIT_SEED = 0  # plan §4 leg 6 step 3: "two disjoint halves by conversation id (seed 0)"
N_SHUFFLE_DRAWS = 20
COS_FLOOR = 0.5
# Cross-arm shared-factor phase (plan leg 6 step 4 / H6): rotation-null draws + seed for the
# REUSED issue1345_operator_comparison.raw_cosine_with_rotation_null band. Measured basis
# (2026-08-25, this VM, thread caps 8): ONE ma._random_orthogonal(3584) draw = 2.47 s, so the
# one-time production band at 200 draws is ~8.3 min per unique vector dim (cached per d).
CROSS_NULL_SEED = 2569
N_CROSS_NULL_DRAWS = 200
FACTOR_K_PERSIST = 32  # per-half factor vectors persisted per unit (matches the [:32] JSON caps)
OPERATOR_K_PERSIST = 64  # all-rows operator sidecar truncation (the _svd_factors k_cap)
# Lambda grid GENERATING PARAMETERS (machine-stable resume key — never hash float bytes).
LAMBDA_GRID_PARAMS = ("logspace", -6.0, 2.0, 17)
GAVISH_DONOHO_COEF = 4.0 / (3.0**0.5)  # arXiv 1305.5870 — analytic reference ONLY

# Statistic CLASS labels (unit-5 atlas convention: every similarity statistic states its
# class IN the artifact, next to the number — direction-aware vs rotation-invariant-only).
STAT_CLASS_CROSS_COS = (
    "direction-aware (raw factor cosine, |.| for SVD sign ambiguity; NOT rotation-invariant/"
    "spectrum-only; issue1345_operator_comparison.raw_cosine_with_rotation_null)"
)
STAT_CLASS_ROTATION_NULL = (
    "two-sided random-rotation chance band on the raw cosine (issue1345_operator_comparison."
    "raw_cosine_with_rotation_null; by symmetry the signed p97.5 equals the p95 of |cos|)"
)
STAT_CLASS_SPLITHALF_FLOOR = (
    "direction-aware within-arm split-half factor agreement (NOISE FLOOR — bounds how well "
    "any cross-arm estimate of the factor can agree; not itself a cross-arm similarity)"
)
STAT_CLASS_OPERATOR_FLOOR = (
    "direction-aware (raw operator vec-cosine between the two independent half-fit maps; "
    "raw read of issue1345_operator_comparison.raw_cosine_with_rotation_null at n_draws=0)"
)
# Selection-symmetric max-matched null (.claude/rules/selection-symmetric-nulls.md option 1;
# mirrors leg 5's dv3_max_matched_null in issue650_analyze.py / issue2569_dw_fleet.py):
# every draw undergoes the IDENTICAL greedy matching over the same (r_a, r_b) factor grid,
# then the same max-over-matched reduction, so observed and null get the same number of
# chances. The p95 of the max-matched null distribution is the FWER-controlling band —
# under H0, P(any match above the band) = 0.05 per pair REGARDLESS of the denoised rank k
# (the per-comparison band's false-positive rate grows as 1-(0.95)^k instead: 0.40 at k=10,
# 0.79 at k=30 — the registered rank range's upper end).
CROSS_NULL_AGGREGATION = "greedy_min_side_match_then_max_over_matched"
STAT_CLASS_SYMMETRIC_NULL = (
    "selection-symmetric max-matched rotation null (per draw: independent Haar frames per "
    "side, the IDENTICAL greedy min-side matching over the same (r_a, r_b) grid, then max "
    "over matched pairs; p95 of the max distribution is the FWER-controlling band; the "
    "matched point estimate stays winner's-curse-inflated — never a corrected magnitude)"
)
FACTOR_ORIENTATION = (
    "row-vector map: pred = (context - mu_c) @ M + mu_d; M = U diag(s) V^T; context factors "
    "= columns of U (input side), shift factors = columns of V (output side); factor sign is "
    "arbitrary under SVD"
)

MAPSHIFT_PREFIX = "issue1768_mapshift"
STORE_REVISIONS = {
    "corpus_capture": "c07267285d",
    "corpus_capture_tf": "c07267285d",
    "delta_tf": "c07267285d",
    "lasttoken_ctx": "89bc6145",
}
CONTEXT_CONVENTIONS = ("last_prompt", "last_ctx", "span_mean")


# ──────────────────────────────────────────────────────────────────────────
# B3 — row-pairing guard
# ──────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class GuardResult:
    """Outcome of the B3 pairing guard for one arm.

    ``action`` is one of ``"exact"`` (ordered keys identical — identity permutations),
    ``"joined"`` (equal key SETS, different order — explicit ID join; ``perm_*`` reorder
    trained/context rows onto the base order), or ``"halt"`` (missing/duplicate keys or no
    key source; ``reason`` names why). Only ``action != "halt"`` may proceed to subtraction.
    """

    action: str
    ordered_match: bool
    reason: str
    key_source: str
    n_rows: int
    perm_trained: np.ndarray | None = None
    perm_ctx: np.ndarray | None = None


def payload_row_keys(payload: dict) -> list[tuple[int, str]] | None:
    """Key-source rung (i): composite ``(row_question_idx, row_sha)`` keys from a payload.

    Returns None when either id field is absent (the ladder then advances). ``row_sha``
    alone is NOT unique (82 duplicate-prompt rows measured in the pinned corpus), so the
    composite key is the unit of uniqueness.
    """
    qidx = payload.get("row_question_idx")
    shas = payload.get("row_sha")
    if qidx is None or shas is None:
        return None
    if len(qidx) != len(shas):
        raise RuntimeError(f"key fields disagree in length: {len(qidx)} vs {len(shas)}")
    return [(int(q), str(s)) for q, s in zip(qidx, shas)]


def raw_rows_keys(raw_rows_dir: Path) -> list[tuple[int, str]] | None:
    """Key-source rung (iv): composite keys from an arm's ``raw_rows_*.jsonl`` id columns.

    Reads via text-mode iteration (never ``splitlines()`` — U+2028 shred class, #950).
    Returns None when the directory holds no ``raw_rows_*.jsonl`` or rows lack id fields.
    """
    files = sorted(raw_rows_dir.glob("raw_rows_*.jsonl"))
    if not files:
        return None
    keys: list[tuple[int, str]] = []
    for fp in files:
        with fp.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "question_idx" not in row or "sha" not in row:
                    return None
                keys.append((int(row["question_idx"]), str(row["sha"])))
    return keys or None


def resolve_keys(
    payload: dict, *, raw_rows_dir: Path | None = None, store_name: str = "?"
) -> tuple[list[tuple[int, str]], str]:
    """Walk the B3 key-source ladder for one store payload; raise on no usable source.

    Rungs: (i) payload id fields; (ii) manifest row-id lists — measured ABSENT in the pinned
    stores (manifests carry ``n_rows`` counts only), recorded here for ladder completeness;
    (iii) ``delta_tf/<arm>/pos.jsonl`` — measured NOT row-keyed for the corpus (20 training
    rows, no id fields), recorded unusable; (iv) ``raw_rows_*.jsonl`` id columns.
    """
    keys = payload_row_keys(payload)
    if keys is not None:
        return keys, "payload-id-fields"
    # Rung (ii)/(iii) measured unusable at the pins (see module docstring). Rung (iv):
    if raw_rows_dir is not None:
        keys = raw_rows_keys(raw_rows_dir)
        if keys is not None:
            return keys, "raw-rows-id-columns"
    raise RuntimeError(
        f"[B3] no usable key source for store {store_name!r}: payload id fields absent, "
        "manifest carries no row-id list, pos.jsonl is not corpus-row-keyed, and no "
        "raw_rows id columns were staged — HALT (never a silent order assumption)."
    )


def _key_diag(keys: list[tuple[int, str]]) -> tuple[bool, int]:
    """Return (has_duplicates, n_unique) for one store's composite key list."""
    uniq = set(keys)
    return len(uniq) != len(keys), len(uniq)


def pairing_guard(
    base_keys: list[tuple[int, str]],
    trained_keys: list[tuple[int, str]],
    ctx_keys: list[tuple[int, str]],
) -> GuardResult:
    """B3: assert exact ordered key match across the three stores, join by ID, or HALT.

    HALT conditions: duplicate composite keys in ANY store, or unequal key SETS (missing
    rows). Order mismatch with equal sets returns ``action="joined"`` with permutations
    mapping the trained/context stores onto the BASE row order.
    """
    n = len(base_keys)
    for name, keys in (("base", base_keys), ("trained", trained_keys), ("ctx", ctx_keys)):
        dup, _ = _key_diag(keys)
        if dup:
            return GuardResult(
                action="halt",
                ordered_match=False,
                reason=f"duplicate composite keys in {name} store",
                key_source="",
                n_rows=n,
            )
    if not (len(trained_keys) == n and len(ctx_keys) == n):
        return GuardResult(
            action="halt",
            ordered_match=False,
            reason=(
                f"row-count mismatch: base={n} trained={len(trained_keys)} ctx={len(ctx_keys)}"
            ),
            key_source="",
            n_rows=n,
        )
    if base_keys == trained_keys and base_keys == ctx_keys:
        return GuardResult(
            action="exact",
            ordered_match=True,
            reason="ordered keys identical",
            key_source="",
            n_rows=n,
        )
    base_set = set(base_keys)
    if set(trained_keys) != base_set or set(ctx_keys) != base_set:
        return GuardResult(
            action="halt",
            ordered_match=False,
            reason="key SETS differ across stores (missing keys)",
            key_source="",
            n_rows=n,
        )
    pos_t = {k: i for i, k in enumerate(trained_keys)}
    pos_c = {k: i for i, k in enumerate(ctx_keys)}
    perm_t = np.asarray([pos_t[k] for k in base_keys], dtype=np.int64)
    perm_c = np.asarray([pos_c[k] for k in base_keys], dtype=np.int64)
    return GuardResult(
        action="joined",
        ordered_match=False,
        reason="order mismatch — explicit composite-ID join applied",
        key_source="",
        n_rows=n,
        perm_trained=perm_t,
        perm_ctx=perm_c,
    )


def tbar_value_check(
    delta_mean_by_layer: dict[int, np.ndarray], tbar_payload: dict, *, n_rows_corpus: int
) -> dict:
    """DEMOTED corroborating VALUE check of the computed Delta column mean vs banked tbar.

    Permutation-invariant BY CONSTRUCTION (a column mean is unchanged under row
    permutation) — it can never certify row order (B3). The strict fp16-tolerance
    reproduction check applies ONLY when the banked mean was computed over the same row
    basis (``meta.n_rows == n_rows_corpus``); the pinned artifacts carry training-row means
    (n_rows=20), so on the real stores this records cosine/norm corroboration with
    ``basis_mismatch: true`` instead (see module docstring).
    """
    meta = tbar_payload.get("meta", {})
    banked_n = int(tbar_payload.get("n_rows", meta.get("n_rows", -1)))
    basis_match = banked_n == n_rows_corpus
    out: dict = {
        "banked_n_rows": banked_n,
        "corpus_n_rows": n_rows_corpus,
        "basis_mismatch": not basis_match,
        "banked_pos_path": str(meta.get("pos_path", "")),
        "per_layer": {},
        "strict_pass": None,
    }
    strict_ok = True
    for layer, mean_vec in sorted(delta_mean_by_layer.items()):
        banked = tbar_payload["tbar"].get(layer)
        if banked is None:
            out["per_layer"][int(layer)] = {"banked": "absent"}
            continue
        b = np.asarray(banked.numpy(), dtype=np.float64)
        m = np.asarray(mean_vec, dtype=np.float64)
        denom = float(np.linalg.norm(b) * np.linalg.norm(m))
        cos = float(b @ m / denom) if denom > 0 else float("nan")
        max_abs = float(np.max(np.abs(b - m)))
        rec = {
            "cosine": cos,
            "max_abs_diff": max_abs,
            "banked_norm": float(np.linalg.norm(b)),
            "computed_norm": float(np.linalg.norm(m)),
        }
        if basis_match:
            # fp16 tolerance: the pooled states are fp16; the mean inherits ~2^-10 grain.
            rec["strict_pass"] = bool(max_abs <= 2e-3 * max(1.0, float(np.max(np.abs(b)))))
            strict_ok = strict_ok and rec["strict_pass"]
        out["per_layer"][int(layer)] = rec
    out["strict_pass"] = bool(strict_ok) if basis_match else None
    return out


# ──────────────────────────────────────────────────────────────────────────
# B2 — split-half denoised-rank estimator
# ──────────────────────────────────────────────────────────────────────────


def split_halves_by_conversation(
    row_shas: list[str], *, seed: int = SPLIT_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices into two disjoint halves by conversation id (= row_sha), seed 0.

    Rows sharing a sha are assigned together (duplicate-prompt rows never straddle halves).
    Deterministic: unique shas sorted, permuted by ``default_rng(seed)``, greedily assigned
    to the lighter half by row count. Returns (idx_half1, idx_half2), each sorted ascending.
    """
    by_sha: dict[str, list[int]] = {}
    for i, s in enumerate(row_shas):
        by_sha.setdefault(s, []).append(i)
    uniq = sorted(by_sha)
    order = np.random.default_rng(seed).permutation(len(uniq))
    halves: tuple[list[int], list[int]] = ([], [])
    counts = [0, 0]
    for j in order:
        rows = by_sha[uniq[int(j)]]
        h = 0 if counts[0] <= counts[1] else 1
        halves[h].extend(rows)
        counts[h] += len(rows)
    idx1 = np.asarray(sorted(halves[0]), dtype=np.int64)
    idx2 = np.asarray(sorted(halves[1]), dtype=np.int64)
    assert len(np.intersect1d(idx1, idx2)) == 0, "halves must be disjoint"
    return idx1, idx2


def half_moments(
    c_all: torch.Tensor, d_all: torch.Tensor, idx: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Matched-row within-half moments: mu_c, mu_d, Scc, Scd from ONE shared row-index set.

    THE B2 BAN LIVES HERE: both C and Delta are sliced with the SAME ``idx``, so a
    cross-half covariance product (C_h^T Delta_h', h != h') is unrepresentable through this
    API. Returns fp64 (mu_c, mu_d, Scc, Scd, n_h); Scc/Scd are centered Gram products.
    """
    rows = torch.as_tensor(idx, dtype=torch.long)
    c_h = c_all.index_select(0, rows).to(torch.float64)
    d_h = d_all.index_select(0, rows).to(torch.float64)
    mu_c = c_h.mean(dim=0)
    mu_d = d_h.mean(dim=0)
    c_c = c_h - mu_c
    d_c = d_h - mu_d
    return mu_c, mu_d, c_c.T @ c_c, c_c.T @ d_c, int(rows.numel())


def ridge_map(scc: torch.Tensor, scd: torch.Tensor, lam_abs: float) -> torch.Tensor:
    """Solve M = (Scc + lam_abs I)^-1 Scd with a pinv fallback on a singular system.

    Healthy path is one batched LAPACK solve (bit-identical across resumes); on
    ``LinAlgError`` the min-norm least-squares pinv solution is used and logged — never a
    silent placeholder (gotchas: batched-solve singular-slice rule).
    """
    d = scc.shape[0]
    reg = scc + lam_abs * torch.eye(d, dtype=scc.dtype)
    try:
        return torch.linalg.solve(reg, scd)
    except torch.linalg.LinAlgError:
        log.warning("[leg6] ridge solve singular at lam=%.3e — pinv fallback", lam_abs)
        return torch.linalg.pinv(reg) @ scd


def heldout_r2(
    m_fit: torch.Tensor,
    mu_c_fit: torch.Tensor,
    mu_d_fit: torch.Tensor,
    c_eval: torch.Tensor,
    d_eval: torch.Tensor,
) -> float:
    """Matched-row held-out R^2: predict d_hat = (c - mu_c_fit) M + mu_d_fit on the other half."""
    c64 = c_eval.to(torch.float64)
    d64 = d_eval.to(torch.float64)
    pred = (c64 - mu_c_fit) @ m_fit + mu_d_fit
    resid = float(((d64 - pred) ** 2).sum())
    total = float(((d64 - d64.mean(dim=0)) ** 2).sum())
    return 1.0 - resid / total if total > 0 else float("nan")


def lambda_grid(params: tuple = LAMBDA_GRID_PARAMS) -> np.ndarray:
    """Relative lambda grid from generating parameters (kind, lo, hi, count)."""
    kind, lo, hi, count = params
    assert kind == "logspace", kind
    return np.logspace(float(lo), float(hi), int(count))


def top_singular_batched(mats: torch.Tensor, *, iters: int = 60, seed: int = 0) -> torch.Tensor:
    """Top singular value of each matrix in a (B, d, d) stack via batched power iteration.

    Power iteration on M^T M with a seeded start vector; 60 iterations gives ~1e-6 relative
    accuracy on well-separated spectra (verified against exact svdvals in the unit tests).
    """
    b, d, _ = mats.shape
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(b, d, 1, generator=gen, dtype=mats.dtype)
    v = v / v.norm(dim=1, keepdim=True)
    for _ in range(iters):
        w = torch.bmm(mats, v)
        v = torch.bmm(mats.transpose(1, 2), w)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-300)
    w = torch.bmm(mats, v)
    return w.norm(dim=1).squeeze(-1)


def shuffle_threshold(
    c_all: torch.Tensor,
    d_all: torch.Tensor,
    idx: np.ndarray,
    lam_abs: float,
    *,
    n_draws: int = N_SHUFFLE_DRAWS,
    seed: int = SPLIT_SEED,
) -> tuple[float, list[float]]:
    """Calibrated noise threshold: p95 of the null map's top singular value over shuffles.

    Each draw permutes Delta rows WITHIN the half (breaking the pairing, preserving both
    marginals), refits the ridge at the SAME lambda, and reads the top singular value. The
    Scd products for all draws are computed as ONE batched GEMM family (vectorize-first).
    """
    rows = torch.as_tensor(idx, dtype=torch.long)
    c_h = c_all.index_select(0, rows).to(torch.float64)
    d_h = d_all.index_select(0, rows).to(torch.float64)
    c_c = c_h - c_h.mean(dim=0)
    d_c = d_h - d_h.mean(dim=0)
    n_h, d_dim = c_c.shape
    rng = np.random.default_rng(seed + 1)
    perms = np.stack([rng.permutation(n_h) for _ in range(n_draws)])
    # (n_h, n_draws * d): gather permuted Delta copies side by side -> one GEMM.
    d_stack = torch.cat([d_c[torch.as_tensor(p, dtype=torch.long)] for p in perms], dim=1)
    scd_all = (c_c.T @ d_stack).reshape(d_dim, n_draws, d_c.shape[1]).permute(1, 0, 2)
    scc = c_c.T @ c_c
    reg = scc + lam_abs * torch.eye(d_dim, dtype=torch.float64)
    try:
        m_null = torch.linalg.solve(reg.unsqueeze(0).expand(n_draws, -1, -1), scd_all)
    except torch.linalg.LinAlgError:
        log.warning("[leg6] batched null solve singular — pinv fallback")
        m_null = torch.linalg.pinv(reg).unsqueeze(0) @ scd_all
    tops = top_singular_batched(m_null.contiguous(), seed=seed + 2)
    vals = [float(x) for x in tops]
    return float(np.percentile(np.asarray(vals), 95.0)), vals


@dataclasses.dataclass
class HalfFit:
    """One half's fitted operator + its SVD factors (fp64)."""

    m: torch.Tensor
    mu_c: torch.Tensor
    mu_d: torch.Tensor
    n_rows: int
    svals: np.ndarray
    u: np.ndarray  # input-side (left) singular vectors, columns
    v: np.ndarray  # output-side (right) singular vectors, columns


def _svd_factors(m: torch.Tensor, *, k_cap: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full SVD of the fitted map, truncated to the leading ``k_cap`` factors."""
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    k = min(k_cap, s.numel())
    return (
        s[:k].numpy().astype(np.float64),
        u[:, :k].numpy().astype(np.float64),
        vh[:k].T.numpy().astype(np.float64),
    )


def greedy_factor_match(
    fit1: HalfFit, fit2: HalfFit, *, cos_floor: float = COS_FLOOR
) -> list[dict]:
    """Greedily match half-1 factors (descending sigma) to half-2 factors by factor cosine.

    Factor cosine = min(|cos(u1_i, u2_j)|, |cos(v1_i, v2_j)|) — BOTH the input-side and
    output-side directions must agree. Each half-1 factor takes the best unmatched half-2
    factor; a match below ``cos_floor`` records ``matched: False``.
    """
    cos_u = np.abs(fit1.u.T @ fit2.u)
    cos_v = np.abs(fit1.v.T @ fit2.v)
    factor_cos = np.minimum(cos_u, cos_v)
    taken: set[int] = set()
    out: list[dict] = []
    for i in range(factor_cos.shape[0]):
        order = np.argsort(-factor_cos[i])
        j = next((int(jj) for jj in order if int(jj) not in taken), None)
        c = float(factor_cos[i, j]) if j is not None else 0.0
        matched = j is not None and c >= cos_floor
        if matched:
            taken.add(int(j))
        out.append(
            {
                "i1": int(i),
                "j2": int(j) if j is not None else -1,
                "factor_cos": c,
                "matched": bool(matched),
                "s1": float(fit1.svals[i]),
                "s2": float(fit2.svals[j]) if j is not None else 0.0,
            }
        )
    return out


def denoised_rank(matches: list[dict], thr1: float, thr2: float) -> int:
    """Count the maximal LEADING prefix of half-1 factors that agree and clear both thresholds."""
    rank = 0
    for m in matches:
        if m["matched"] and m["s1"] > thr1 and m["s2"] > thr2:
            rank += 1
        else:
            break
    return rank


def _oc():
    """Deferred import of the REUSED reference module (heavy sibling-import chain, ~5 s).

    ``scripts/`` is sys.path[0] when this file runs as a script and is inserted by the test
    file under pytest; this helper self-inserts the script dir so a module-mode importer
    resolves the sibling too. Returns ``issue1345_operator_comparison``.
    """
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import issue1345_operator_comparison as oc

    return oc


def fit_split_half(
    c_all: torch.Tensor,
    d_all: torch.Tensor,
    row_shas: list[str],
    *,
    seed: int = SPLIT_SEED,
    lam_params: tuple = LAMBDA_GRID_PARAMS,
    n_shuffle: int = N_SHUFFLE_DRAWS,
    cos_floor: float = COS_FLOOR,
    knn_ks: tuple[int, ...] = (1, 5, 10),
    return_factors: bool = False,
) -> dict | tuple[dict, dict]:
    """The full B2 estimator for one (arm, layer, context-convention) unit.

    Returns a JSON-serializable record: selected lambda (+ edge flag), per-half held-out
    R^2, shuffle-calibrated thresholds, factor-match table, denoised rank, the
    Gavish-Donoho reference count, the operator-level split-half self-agreement (raw
    vec-cosine between the two half maps — the atlas floor convention), and mapping
    baselines (identity+bias R^2 + kNN retrieval, scored on the opposite half).

    With ``return_factors=True`` additionally returns the factor-VECTOR arrays (role-named:
    ``context_factors_half{1,2}`` / ``shift_factors_half{1,2}`` / ``singular_values_half
    {1,2}``, truncated to ``FACTOR_K_PERSIST``; see ``FACTOR_ORIENTATION``) plus the two
    fitted half maps (``map_half1``/``map_half2`` — NOT persisted; for tests/diagnostics).
    """
    idx1, idx2 = split_halves_by_conversation(row_shas, seed=seed)
    assert len(np.intersect1d(idx1, idx2)) == 0, "B2: halves must be disjoint"
    mu_c1, mu_d1, scc1, scd1, n1 = half_moments(c_all, d_all, idx1)
    mu_c2, mu_d2, scc2, scd2, n2 = half_moments(c_all, d_all, idx2)

    rows1 = torch.as_tensor(idx1, dtype=torch.long)
    rows2 = torch.as_tensor(idx2, dtype=torch.long)
    c1, d1 = c_all.index_select(0, rows1), d_all.index_select(0, rows1)
    c2, d2 = c_all.index_select(0, rows2), d_all.index_select(0, rows2)

    # Scale-free lambda: lam_abs = lam_rel * tr(Scc_h)/d  (per-half trace scale).
    d_dim = int(scc1.shape[0])
    grid = lambda_grid(lam_params)
    scale1 = float(torch.diagonal(scc1).mean())
    scale2 = float(torch.diagonal(scc2).mean())

    def _score(lam_rel: float) -> tuple[float, float, float]:
        m1 = ridge_map(scc1, scd1, lam_rel * scale1)
        m2 = ridge_map(scc2, scd2, lam_rel * scale2)
        r2_12 = heldout_r2(m1, mu_c1, mu_d1, c2, d2)
        r2_21 = heldout_r2(m2, mu_c2, mu_d2, c1, d1)
        return (r2_12 + r2_21) / 2.0, r2_12, r2_21

    scores = [_score(float(lr))[0] for lr in grid]
    best = int(np.argmax(scores))
    edge = best in (0, len(grid) - 1)
    if edge:  # widen once by 2 decades on the selected side and reselect (plan C4 shape)
        kind, lo, hi, count = lam_params
        lo2 = float(lo) - (2.0 if best == 0 else 0.0)
        hi2 = float(hi) + (2.0 if best == len(grid) - 1 else 0.0)
        grid = lambda_grid((kind, lo2, hi2, int(count) + 4))
        scores = [_score(float(lr))[0] for lr in grid]
        best = int(np.argmax(scores))
        edge = best in (0, len(grid) - 1)
    lam_rel = float(grid[best])
    _, r2_12, r2_21 = _score(lam_rel)

    m1 = ridge_map(scc1, scd1, lam_rel * scale1)
    m2 = ridge_map(scc2, scd2, lam_rel * scale2)
    s1, u1, v1 = _svd_factors(m1)
    s2, u2, v2 = _svd_factors(m2)
    fit1 = HalfFit(m1, mu_c1, mu_d1, n1, s1, u1, v1)
    fit2 = HalfFit(m2, mu_c2, mu_d2, n2, s2, u2, v2)

    thr1, draws1 = shuffle_threshold(
        c_all, d_all, idx1, lam_rel * scale1, n_draws=n_shuffle, seed=seed
    )
    thr2, draws2 = shuffle_threshold(
        c_all, d_all, idx2, lam_rel * scale2, n_draws=n_shuffle, seed=seed + 100
    )
    matches = greedy_factor_match(fit1, fit2, cos_floor=cos_floor)
    rank = denoised_rank(matches, thr1, thr2)

    # Operator-level split-half self-agreement (the atlas floor convention): raw vec-cosine
    # between the two independently fitted half maps, via the REUSED reference formula.
    op_cos = float(_oc().raw_cosine_with_rotation_null(m1, m2, n_draws=0, seed=0)["raw_cosine"])

    # Gavish-Donoho 4/sqrt(3) count — ANALYTIC REFERENCE ONLY (premise does not hold here).
    med1 = float(np.median(s1)) if len(s1) else 0.0
    gd_count = int(np.sum(s1 > GAVISH_DONOHO_COEF * med1)) if med1 > 0 else 0

    # Mapping baselines on the opposite half (identity+learned-bias + kNN retrieval).
    from explore_persona_space.analysis import mapping_baselines as mb

    c1n, d1n = c1.to(torch.float64).numpy(), d1.to(torch.float64).numpy()
    c2n, d2n = c2.to(torch.float64).numpy(), d2.to(torch.float64).numpy()
    idb_pred = mb.identity_bias_predict(c1n, d1n, c2n)
    idb_resid = float(((d2n - idb_pred) ** 2).sum())
    idb_total = float(((d2n - d2n.mean(axis=0)) ** 2).sum())
    idb_r2 = 1.0 - idb_resid / idb_total if idb_total > 0 else float("nan")
    ridge_pred2 = ((c2.to(torch.float64) - mu_c1) @ m1 + mu_d1).numpy()
    knn = mb.knn_retrieval(ridge_pred2.astype(np.float32), d2n.astype(np.float32), ks=knn_ks)
    knn["chance"] = {int(k): float(k) / float(len(d2n)) for k in knn_ks}

    rec = {
        "n_rows": int(c_all.shape[0]),
        "n_half": [n1, n2],
        "d": d_dim,
        "lambda_rel_selected": lam_rel,
        "lambda_abs": [lam_rel * scale1, lam_rel * scale2],
        "lambda_grid_params": list(LAMBDA_GRID_PARAMS),
        "lambda_grid_edge": bool(edge),
        "heldout_r2": {"fit1_eval2": r2_12, "fit2_eval1": r2_21},
        "shuffle_threshold_p95": [thr1, thr2],
        "shuffle_top_sv_draws": [draws1, draws2],
        "n_shuffle_draws": int(n_shuffle),
        "cos_floor": float(cos_floor),
        "factor_matches": matches[:32],
        "denoised_rank": int(rank),
        "gavish_donoho_reference_count": gd_count,
        "singular_values_half1": [float(x) for x in s1[:32]],
        "singular_values_half2": [float(x) for x in s2[:32]],
        "identity_bias_r2": idb_r2,
        "knn_retrieval": knn,
        "operator_splithalf_cosine": op_cos,
        "operator_splithalf_cosine_class": STAT_CLASS_OPERATOR_FLOOR,
        "factor_orientation": FACTOR_ORIENTATION,
    }
    if not return_factors:
        return rec
    k = int(min(FACTOR_K_PERSIST, len(s1), len(s2)))
    factors = {
        "context_factors_half1": u1[:, :k].astype(np.float32),
        "shift_factors_half1": v1[:, :k].astype(np.float32),
        "singular_values_half1": s1[:k].astype(np.float64),
        "context_factors_half2": u2[:, :k].astype(np.float32),
        "shift_factors_half2": v2[:, :k].astype(np.float32),
        "singular_values_half2": s2[:k].astype(np.float64),
        # NOT persisted — returned for tests/diagnostics (reconstruction + role checks).
        "map_half1": m1,
        "map_half2": m2,
    }
    return rec, factors


# ──────────────────────────────────────────────────────────────────────────
# Store loading + per-arm driver
# ──────────────────────────────────────────────────────────────────────────


def load_store_file(path: Path) -> dict:
    """torch.load a pinned self-produced store payload (weights_only=False is deliberate)."""
    if not path.is_file():
        raise FileNotFoundError(f"staged store file missing: {path}")
    return torch.load(path, weights_only=False, map_location="cpu")


def context_matrix(convention: str, lasttoken: dict, pooled_base: dict, layer: int) -> torch.Tensor:
    """Resolve the context matrix C for one convention (see module docstring)."""
    if convention == "last_prompt":
        return lasttoken["arms"]["last_prompt"][layer]
    if convention == "last_ctx":
        return lasttoken["arms"]["last_ctx"][layer]
    if convention == "span_mean":
        return pooled_base["arms"]["context"][layer]
    raise ValueError(f"unknown context convention {convention!r}")


def _atomic_json(path: Path, payload: dict) -> None:
    """Atomic JSON write through the shared process-unique atomic-replace primitive
    (#2336: a fixed ``.tmp`` sibling name is a concurrent-writer clobber)."""
    with atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))


def _meta() -> dict:
    """Reproducibility metadata block (git provenance + versions + timestamp)."""
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    md = as_metadata_dict(git_provenance(), phase="leg6")
    md.update(
        {
            "issue": ISSUE,
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "store_revisions": dict(STORE_REVISIONS),
        }
    )
    return md


def regime_key(*, layer: int, convention: str, seed: int, n_shuffle: int, cos_floor: float) -> str:
    """Machine-stable resume key from GENERATING PARAMETERS (never float-array bytes)."""
    blob = json.dumps(
        {
            "layer": layer,
            "convention": convention,
            "seed": seed,
            "n_shuffle": n_shuffle,
            "cos_floor": cos_floor,
            "lambda_grid_params": list(LAMBDA_GRID_PARAMS),
            "store_revisions": dict(sorted(STORE_REVISIONS.items())),
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def arm_store_paths(staged_root: Path, arm: dict) -> dict[str, Path]:
    """Resolve the four staged store files for one arm (base unit + trained arm)."""
    root = staged_root / MAPSHIFT_PREFIX
    base_unit = arm.get("base_unit", "base_content")
    aid = arm["arm_id"]
    return {
        "pooled_base": root / "corpus_capture" / base_unit / "pooled.pt",
        "pooled_tf": root / "corpus_capture_tf" / aid / "pooled_tf.pt",
        "lasttoken": root / "lasttoken_ctx" / base_unit / "lasttoken.pt",
        "tbar": root / "delta_tf" / aid / "tbar.pt",
    }


def factor_bases_for(layer: int, convention: str, d_context: int, d_shift: int) -> dict:
    """Explicit basis bookkeeping for one unit's factor vectors (the cross-arm admission key).

    Two arms' factors are comparable ONLY when these dicts are EQUAL: the input side lives
    in the BASE model's residual stream at ``layer`` under ``convention`` pooling (shared
    across arms by construction — one base store), the output side in the answer-span shift
    space at ``layer`` (trained pooled_tf minus base pooled, both span-mean). Store pins
    ride along so a re-pinned producer fails the equality loudly instead of comparing
    vectors from different snapshots.
    """
    ctx_store = (
        f"lasttoken_ctx@{STORE_REVISIONS['lasttoken_ctx']}"
        if convention in ("last_prompt", "last_ctx")
        else f"corpus_capture@{STORE_REVISIONS['corpus_capture']}"
    )
    return {
        "context": {
            "space": "base-model residual stream",
            "layer": int(layer),
            "summary": convention,
            "store": ctx_store,
            "dim": int(d_context),
        },
        "shift": {
            "space": "answer-span shift (trained pooled_tf minus base pooled)",
            "layer": int(layer),
            "summary": "span_mean",
            "store": (
                f"corpus_capture_tf@{STORE_REVISIONS['corpus_capture_tf']}"
                f" - corpus_capture@{STORE_REVISIONS['corpus_capture']}"
            ),
            "dim": int(d_shift),
        },
    }


def write_factor_sidecar(
    npz_path: Path,
    factors: dict,
    *,
    bases: dict,
    arm: str,
    layer: int,
    convention: str,
    rk: str,
    unit_kind: str,
) -> None:
    """Atomically persist one unit's factor VECTORS as a role-named ``.npz`` sidecar.

    Arrays: ``context_factors_half{1,2}`` / ``shift_factors_half{1,2}`` (columns = factors,
    descending sigma; fp32) + ``singular_values_half{1,2}`` (fp64); ``schema_json`` embeds
    the orientation, the basis bookkeeping, and provenance so the sidecar is
    self-describing. Written via ``np.savez`` on an OPEN file object — a path argument
    would append ``.npz`` to the temp name and break the atomic replace.
    """
    schema = {
        "schema_version": 1,
        "orientation": FACTOR_ORIENTATION,
        "factor_bases": bases,
        "arm": arm,
        "layer": int(layer),
        "context_convention": convention,
        "unit_kind": unit_kind,
        "regime_key": rk,
        "k": int(factors["singular_values_half1"].shape[0]),
        "halves": (
            "factors from the two independent per-half ridge fits at the shared selected lambda"
        ),
    }
    with atomic_replace(npz_path) as tmp, open(tmp, "wb") as fh:
        np.savez(
            fh,
            schema_json=np.asarray(json.dumps(schema, sort_keys=True)),
            **{k: v for k, v in factors.items() if not k.startswith("map_half")},
        )


def write_operator_factors(
    arm_dir: Path,
    c_mat: torch.Tensor,
    delta: torch.Tensor,
    unit_rec: dict,
    *,
    arm: str,
    layer: int,
    convention: str,
    rk: str,
    resume: bool,
) -> None:
    """Persist the per-arm ``operator_factors.pt`` atlas sidecar (atlas step (5) contract).

    Keys ``u``/``s``/``v``/``split_half_floor`` are the EXISTING consumer contract
    (``issue2569_atlas.py`` globs ``<leg6_dir>/*/operator_factors.pt`` and reconstructs
    ``A = (u * s) @ v.T``; concern leg7-atlas-writemap-operators-unpersisted) — role
    annotations + basis bookkeeping ride alongside. Factors are the top
    ``OPERATOR_K_PERSIST`` of an ALL-ROWS ridge refit at the unit-selected lambda (the
    ``run_pooled`` m_all pattern); the floor is the unit's operator-level split-half
    self-agreement. Atomic write; resume keyed on the embedded ``regime_key``.
    """
    path = arm_dir / "operator_factors.pt"
    if resume and path.is_file():
        prior = torch.load(path, weights_only=False, map_location="cpu")
        if isinstance(prior, dict) and prior.get("regime_key") == rk:
            log.info("[leg6] %s operator_factors.pt resume-skip", arm)
            return
    lam_rel = float(unit_rec["lambda_rel_selected"])
    all_idx = np.arange(c_mat.shape[0], dtype=np.int64)
    mu_c, mu_d, scc, scd, _n = half_moments(c_mat, delta, all_idx)
    del mu_c, mu_d
    m_all = ridge_map(scc, scd, lam_rel * float(torch.diagonal(scc).mean()))
    s, u, v = _svd_factors(m_all, k_cap=OPERATOR_K_PERSIST)
    payload = {
        "u": torch.as_tensor(u, dtype=torch.float32),
        "s": torch.as_tensor(s, dtype=torch.float64),
        "v": torch.as_tensor(v, dtype=torch.float32),
        "split_half_floor": float(unit_rec["operator_splithalf_cosine"]),
        "split_half_floor_class": STAT_CLASS_OPERATOR_FLOOR,
        "u_role": "context (read/input) directions — columns, descending sigma",
        "v_role": "shift (write/output) directions — columns, descending sigma",
        "orientation": FACTOR_ORIENTATION,
        "factor_bases": unit_rec["factor_bases"],
        "arm": arm,
        "layer": int(layer),
        "context_convention": convention,
        "fit": "all-rows ridge refit at the unit-selected lambda (row-vector convention)",
        "k": int(len(s)),
        "regime_key": rk,
        "metadata": _meta(),
    }
    with atomic_replace(path) as tmp:
        torch.save(payload, tmp)
    log.info(
        "[leg6] %s operator_factors.pt written (k=%d, split_half_floor=%.4f)",
        arm,
        len(s),
        payload["split_half_floor"],
    )


def rotation_null_band(d: int, *, n_draws: int, seed: int, cache: dict) -> dict:
    """Rotation-null chance band for d-dim FACTOR (vector) cosines, cached per dimension.

    REUSES ``issue1345_operator_comparison.raw_cosine_with_rotation_null`` verbatim on
    canonical basis-vector probes (e_1 both sides, shape (d, 1)): a Haar rotation maps any
    unit vector to a uniform direction, so for vector-shaped inputs the null distribution
    depends only on ``d`` — ONE band per dimension is shared across pairs (recorded in the
    artifact). Matching uses |cos|, so the decision threshold is the SIGNED p97.5 the
    reference emits, which equals the p95 of |cos| by symmetry.
    """
    key = (int(d), int(n_draws), int(seed))
    if key in cache:
        return cache[key]
    e1 = torch.zeros(d, 1, dtype=torch.float64)
    e1[0, 0] = 1.0
    t0 = time.time()
    out = _oc().raw_cosine_with_rotation_null(e1, e1, n_draws=n_draws, seed=seed)
    band = dict(out["rotation_null"])
    band.update(
        {
            "dim": int(d),
            "seed": int(seed),
            "probe": "canonical basis vector e_1 both sides (probe raw_cosine ignored)",
            "shared_across_pairs": True,
            "sharing_justification": (
                "for vector-shaped inputs a Haar rotation maps any unit vector to a "
                "uniform direction; the null distribution depends only on the dimension"
            ),
            "abs_threshold_note": (
                "matching uses |cos|; the signed p97.5 band equals the p95 of |cos| by "
                "symmetry and is the decision threshold"
            ),
            "statistic_class": STAT_CLASS_ROTATION_NULL,
            "wall_s": round(time.time() - t0, 2),
        }
    )
    cache[key] = band
    return band


def _greedy_match_stat(
    cos_ctx_mat: np.ndarray, cos_shf_mat: np.ndarray
) -> tuple[list[tuple[int, int, float]], float]:
    """The SHARED cross-arm selection: greedy min-side matching, then max over matches.

    Used by BOTH the observed read and every symmetric-null draw (option 1 of
    .claude/rules/selection-symmetric-nulls.md: the null inherits the IDENTICAL
    selection), so ``null_aggregation_matches_observed`` holds by construction — the
    same function computes both sides. Inputs are |cos| matrices (r_a x r_b); rows are
    iterated in descending-sigma order (the SVD order), each taking the best unmatched
    column by min(|cos_context|, |cos_shift|). Returns (matches [(i, j, comb)], t_max).
    """
    comb = np.minimum(cos_ctx_mat, cos_shf_mat)
    taken: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for i in range(comb.shape[0]):
        order = np.argsort(-comb[i])
        j = next((int(x) for x in order if int(x) not in taken), None)
        if j is None:
            break
        taken.add(j)
        matches.append((i, int(j), float(comb[i, j])))
    t_max = max((v for _, _, v in matches), default=0.0)
    return matches, t_max


def symmetric_null_band(
    d_ctx: int, d_shf: int, r_a: int, r_b: int, *, n_draws: int, seed: int, cache: dict
) -> dict:
    """Selection-symmetric (max-matched) null band, cached per (d_ctx, d_shf, r_a, r_b).

    Per draw: sample ONE independent uniform orthonormal r_b-frame per side (batched
    torch QR of Gaussian (n_draws, d, r_b); Haar-rotating arm-b's factor frame U_b by R
    gives R @ U_b == a uniform Stiefel frame, and for orthonormal U_a the read
    |U_a^T F| =d= |F[:r_a, :]| — complete U_a^T to an orthogonal matrix and use the
    frame distribution's left-invariance — so the band depends only on
    (d_ctx, d_shf, r_a, r_b) and is SHARED across pairs; QR column-sign non-uniformity
    is inert because the statistic uses |cos|), then the IDENTICAL ``_greedy_match_stat``
    selection and max-over-matched reduction the observed read uses. Persists the full
    per-draw x per-match-slot matrix (``draws_matched_cos``) so the band is recomputable
    post-hoc (the rule's persistence clause). ``n_draws=0`` returns a NaN band (every
    ``above_symmetric_null`` comparison is then False — no fabricated verdict).
    """
    key = (int(d_ctx), int(d_shf), int(r_a), int(r_b), int(n_draws))
    if key in cache:
        return cache[key]
    blob = json.dumps({"seed": seed, "key": list(key)}, sort_keys=True)
    derived_seed = int(hashlib.sha256(blob.encode()).hexdigest()[:8], 16)
    t0 = time.time()
    draws_max: list[float] = []
    draws_matrix: list[list[float]] = []
    n_slots = min(r_a, r_b)
    if n_draws > 0:
        gen = torch.Generator().manual_seed(derived_seed)
        g_ctx = torch.randn(n_draws, d_ctx, r_b, generator=gen, dtype=torch.float64)
        g_shf = torch.randn(n_draws, d_shf, r_b, generator=gen, dtype=torch.float64)
        f_ctx = torch.linalg.qr(g_ctx, mode="reduced")[0]  # (B, d_ctx, r_b) uniform frames
        f_shf = torch.linalg.qr(g_shf, mode="reduced")[0]
        cos_ctx = f_ctx[:, :r_a, :].abs().numpy()  # |U_a^T F| == |F[:r_a, :]| in law
        cos_shf = f_shf[:, :r_a, :].abs().numpy()
        for b in range(n_draws):
            matches, t_max = _greedy_match_stat(cos_ctx[b], cos_shf[b])
            draws_max.append(float(t_max))
            row = [float("nan")] * n_slots
            for slot, (_, _, v) in enumerate(matches):
                row[slot] = float(v)
            draws_matrix.append(row)
    band = {
        "d_ctx": int(d_ctx),
        "d_shf": int(d_shf),
        "r_a": int(r_a),
        "r_b": int(r_b),
        "n_draws": int(n_draws),
        "seed": int(seed),
        "derived_seed": int(derived_seed),
        "p95_max_matched": (
            float(np.percentile(np.asarray(draws_max), 95.0)) if draws_max else float("nan")
        ),
        "mean_max_matched": float(np.mean(draws_max)) if draws_max else float("nan"),
        "draws_max_matched": draws_max,
        # Per-draw x per-match-slot |cos| matrix — the honest band (or any alternative
        # aggregation) is a pure re-reduction of this, no re-run needed.
        "draws_matched_cos": draws_matrix,
        "null_aggregation": CROSS_NULL_AGGREGATION,
        "statistic_class": STAT_CLASS_SYMMETRIC_NULL,
        "sharing_justification": (
            "for orthonormal factor frames the null |cos| matrices depend only on "
            "(d_ctx, d_shf, r_a, r_b) — Haar-rotated frames are uniform Stiefel frames and "
            "|U_a^T F| =d= |F[:r_a, :]| — so one band per shape tuple is shared across pairs"
        ),
        "wall_s": round(time.time() - t0, 2),
    }
    cache[key] = band
    return band


def cross_regime_key(
    *, layer: int, convention: str, n_null_draws: int, null_seed: int, arm_unit_keys: dict
) -> str:
    """Machine-stable cross-arm resume key: generating parameters + the consumed units'
    own regime keys (an upstream unit re-fit under a new regime invalidates the cell)."""
    blob = json.dumps(
        {
            "layer": layer,
            "convention": convention,
            "n_null_draws": n_null_draws,
            "null_seed": null_seed,
            "matching": "greedy-min-both-sides-abs-v2-symmetric-null",
            "null_aggregation": CROSS_NULL_AGGREGATION,
            "arm_unit_keys": dict(sorted(arm_unit_keys.items())),
            "store_revisions": dict(sorted(STORE_REVISIONS.items())),
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run_arm(
    arm: dict,
    staged_root: Path,
    out_root: Path,
    *,
    layers: list[int],
    conventions: list[str],
    raw_rows_dir: Path | None,
    resume: bool,
) -> dict:
    """Run the B3 guard + tbar check + B2 estimator for one arm; checkpoint per unit."""
    aid = arm["arm_id"]
    arm_dir = out_root / "leg6" / aid
    paths = arm_store_paths(staged_root, arm)
    t0 = time.time()

    pooled_base = load_store_file(paths["pooled_base"])
    pooled_tf = load_store_file(paths["pooled_tf"])
    lasttoken = load_store_file(paths["lasttoken"])

    keys_b, src_b = resolve_keys(pooled_base, raw_rows_dir=raw_rows_dir, store_name="pooled_base")
    keys_t, src_t = resolve_keys(pooled_tf, raw_rows_dir=raw_rows_dir, store_name="pooled_tf")
    keys_c, src_c = resolve_keys(lasttoken, raw_rows_dir=raw_rows_dir, store_name="lasttoken")
    guard = pairing_guard(keys_b, keys_t, keys_c)
    guard_rec = {
        "action": guard.action,
        "ordered_match": guard.ordered_match,
        "reason": guard.reason,
        "key_sources": {"base": src_b, "trained": src_t, "ctx": src_c},
        "n_rows": guard.n_rows,
    }
    if guard.action == "halt":
        _atomic_json(arm_dir / "guard.json", {"arm": aid, "guard": guard_rec, "metadata": _meta()})
        log.error("[leg6] arm %s HALT: %s", aid, guard.reason)
        return {"arm": aid, "halt": True, "reason": guard.reason}

    # Align trained/ctx rows onto the base order when the guard joined by ID.
    perm_t = guard.perm_trained if guard.perm_trained is not None else None
    perm_c = guard.perm_ctx if guard.perm_ctx is not None else None

    layers_avail = sorted(pooled_base["arms"]["response"].keys())
    delta_by_layer: dict[int, torch.Tensor] = {}
    for layer in layers:
        if layer not in layers_avail:
            raise RuntimeError(f"layer {layer} absent in store (available: {layers_avail})")
        base_resp = pooled_base["arms"]["response"][layer]
        tf_resp = pooled_tf["arms"]["response"][layer]
        if perm_t is not None:
            tf_resp = tf_resp[torch.as_tensor(perm_t, dtype=torch.long)]
        delta_by_layer[layer] = tf_resp.to(torch.float32) - base_resp.to(torch.float32)

    tbar_payload = load_store_file(paths["tbar"])
    delta_means = {
        layer: dv.to(torch.float64).mean(dim=0).numpy() for layer, dv in delta_by_layer.items()
    }
    tbar_rec = tbar_value_check(delta_means, tbar_payload, n_rows_corpus=guard.n_rows)
    _atomic_json(
        arm_dir / "guard.json",
        {"arm": aid, "guard": guard_rec, "tbar_check": tbar_rec, "metadata": _meta()},
    )

    row_shas = [k[1] for k in keys_b]
    units = [(layer, conv) for layer in layers for conv in conventions]
    unit_recs: dict[tuple[int, str], dict] = {}
    done = 0
    for k, (layer, conv) in enumerate(units, start=1):
        rk = regime_key(
            layer=layer,
            convention=conv,
            seed=SPLIT_SEED,
            n_shuffle=N_SHUFFLE_DRAWS,
            cos_floor=COS_FLOOR,
        )
        unit_path = arm_dir / f"L{layer}_{conv}.json"
        npz_path = arm_dir / f"L{layer}_{conv}_factors.npz"
        # Resume ALSO requires the factor sidecar: a pre-vector-format unit JSON with a
        # matching regime key would otherwise skip the fit and never gain its vectors.
        if resume and unit_path.is_file() and npz_path.is_file():
            try:
                prior = json.loads(unit_path.read_text())
            except json.JSONDecodeError:
                prior = {}
            if prior.get("regime_key") == rk:
                log.info("[leg6] unit %d/%d %s/L%d/%s resume-skip", k, len(units), aid, layer, conv)
                unit_recs[(layer, conv)] = prior
                done += 1
                continue
        c_mat = context_matrix(conv, lasttoken, pooled_base, layer)
        if conv in ("last_prompt", "last_ctx") and perm_c is not None:
            c_mat = c_mat[torch.as_tensor(perm_c, dtype=torch.long)]
        c_mat = c_mat.to(torch.float32)
        rec, factors = fit_split_half(c_mat, delta_by_layer[layer], row_shas, return_factors=True)
        bases = factor_bases_for(
            layer, conv, int(c_mat.shape[1]), int(delta_by_layer[layer].shape[1])
        )
        write_factor_sidecar(
            npz_path,
            factors,
            bases=bases,
            arm=aid,
            layer=layer,
            convention=conv,
            rk=rk,
            unit_kind="per_arm",
        )
        rec.update(
            {
                "arm": aid,
                "layer": int(layer),
                "context_convention": conv,
                "regime_key": rk,
                "factor_vectors_file": npz_path.name,
                "factor_vectors_k": int(factors["singular_values_half1"].shape[0]),
                "factor_bases": bases,
                "metadata": _meta(),
            }
        )
        _atomic_json(unit_path, rec)
        unit_recs[(layer, conv)] = rec
        done += 1
        print(
            f"[leg6] unit {k}/{len(units)} {aid}/L{layer}/{conv} "
            f"rank={rec['denoised_rank']} r2={rec['heldout_r2']['fit1_eval2']:.4f} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    # Atlas operator sidecar (issue2569_atlas.py step (5) glob contract) from the PRIMARY
    # unit's regime — layer 19 / last_prompt where present (the plan's headline unit).
    p_layer = 19 if 19 in layers else layers[0]
    p_conv = "last_prompt" if "last_prompt" in conventions else conventions[0]
    p_rec = unit_recs[(p_layer, p_conv)]
    p_c = context_matrix(p_conv, lasttoken, pooled_base, p_layer)
    if p_conv in ("last_prompt", "last_ctx") and perm_c is not None:
        p_c = p_c[torch.as_tensor(perm_c, dtype=torch.long)]
    write_operator_factors(
        arm_dir,
        p_c.to(torch.float32),
        delta_by_layer[p_layer],
        p_rec,
        arm=aid,
        layer=p_layer,
        convention=p_conv,
        rk=p_rec["regime_key"],
        resume=resume,
    )
    return {"arm": aid, "halt": False, "units_done": done}


def run_pooled(
    target: dict,
    siblings: list[dict],
    staged_root: Path,
    out_root: Path,
    *,
    layers: list[int],
    conventions: list[str],
    raw_rows_dir: Path | None,
    resume: bool,
) -> dict:
    """Pooled cross-arm RRR: fit on 3 same-behavior sibling arms, target arm held out.

    Reuses the B2 split-half machinery on the pooled sibling rows (denoised rank of the
    pooled operator); target-arm evaluation reports held-out R^2 + identity+bias + kNN.
    Every consumed arm passes its own B3 guard first; a halted sibling is dropped (named),
    a halted target halts the pooled unit.
    """
    aid = target["arm_id"]
    pooled_dir = out_root / "leg6" / "pooled" / aid
    t0 = time.time()

    def _arm_cd(arm: dict) -> tuple[torch.Tensor, dict[int, torch.Tensor], list[str]] | None:
        paths = arm_store_paths(staged_root, arm)
        pb = load_store_file(paths["pooled_base"])
        pt = load_store_file(paths["pooled_tf"])
        lt = load_store_file(paths["lasttoken"])
        kb, _ = resolve_keys(pb, raw_rows_dir=raw_rows_dir, store_name="pooled_base")
        kt, _ = resolve_keys(pt, raw_rows_dir=raw_rows_dir, store_name="pooled_tf")
        kc, _ = resolve_keys(lt, raw_rows_dir=raw_rows_dir, store_name="lasttoken")
        g = pairing_guard(kb, kt, kc)
        if g.action == "halt":
            log.error("[leg6-pooled] arm %s guard HALT: %s", arm["arm_id"], g.reason)
            return None
        deltas: dict[int, torch.Tensor] = {}
        ctxs: dict[int, dict[str, torch.Tensor]] = {}
        for layer in layers:
            tf_r = pt["arms"]["response"][layer]
            if g.perm_trained is not None:
                tf_r = tf_r[torch.as_tensor(g.perm_trained, dtype=torch.long)]
            deltas[layer] = tf_r.to(torch.float32) - pb["arms"]["response"][layer].to(torch.float32)
            ctxs[layer] = {}
            for conv in conventions:
                cm = context_matrix(conv, lt, pb, layer)
                if conv in ("last_prompt", "last_ctx") and g.perm_ctx is not None:
                    cm = cm[torch.as_tensor(g.perm_ctx, dtype=torch.long)]
                ctxs[layer][conv] = cm.to(torch.float32)
        return ctxs, deltas, [k[1] for k in kb]  # type: ignore[return-value]

    target_data = _arm_cd(target)
    if target_data is None:
        _atomic_json(pooled_dir / "halt.json", {"arm": aid, "reason": "target guard halt"})
        return {"arm": aid, "halt": True}
    sib_data = []
    for s in siblings:
        d = _arm_cd(s)
        if d is None:
            log.warning("[leg6-pooled] sibling %s dropped (guard halt)", s["arm_id"])
        else:
            sib_data.append((s["arm_id"], d))
    if not sib_data:
        raise RuntimeError(f"[leg6-pooled] no usable siblings for target {aid}")

    for layer in layers:
        for conv in conventions:
            rk = regime_key(
                layer=layer,
                convention=conv,
                seed=SPLIT_SEED,
                n_shuffle=N_SHUFFLE_DRAWS,
                cos_floor=COS_FLOOR,
            )
            unit_path = pooled_dir / f"L{layer}_{conv}.json"
            if resume and unit_path.is_file():
                try:
                    if json.loads(unit_path.read_text()).get("regime_key") == rk:
                        continue
                except json.JSONDecodeError:
                    pass
            c_pool = torch.cat([d[0][layer][conv] for _, d in sib_data], dim=0)
            d_pool = torch.cat([d[1][layer] for _, d in sib_data], dim=0)
            # Conversation ids prefixed per sibling arm keep cross-arm shas distinct.
            shas = [f"{name}:{s}" for name, d in sib_data for s in d[2]]
            rec, factors = fit_split_half(c_pool, d_pool, shas, return_factors=True)
            bases = factor_bases_for(layer, conv, int(c_pool.shape[1]), int(d_pool.shape[1]))
            npz_path = pooled_dir / f"L{layer}_{conv}_factors.npz"
            write_factor_sidecar(
                npz_path,
                factors,
                bases=bases,
                arm=aid,
                layer=layer,
                convention=conv,
                rk=rk,
                unit_kind="pooled",
            )
            rec["factor_vectors_file"] = npz_path.name
            rec["factor_vectors_k"] = int(factors["singular_values_half1"].shape[0])
            rec["factor_bases"] = bases
            # Target-arm evaluation with a pooled-rows refit at the selected lambda.
            lam_rel = rec["lambda_rel_selected"]
            all_idx = np.arange(c_pool.shape[0], dtype=np.int64)
            mu_c, mu_d, scc, scd, _ = half_moments(c_pool, d_pool, all_idx)
            m_all = ridge_map(scc, scd, lam_rel * float(torch.diagonal(scc).mean()))
            ct, dt = target_data[0][layer][conv], target_data[1][layer]
            rec["target_arm"] = aid
            rec["siblings"] = [name for name, _ in sib_data]
            rec["pooled_train_rows"] = int(c_pool.shape[0])
            rec["target_r2"] = heldout_r2(m_all, mu_c, mu_d, ct, dt)
            rec.update(
                {
                    "layer": int(layer),
                    "context_convention": conv,
                    "regime_key": rk,
                    "metadata": _meta(),
                }
            )
            _atomic_json(unit_path, rec)
            print(
                f"[leg6-pooled] {aid}/L{layer}/{conv} rank={rec['denoised_rank']} "
                f"target_r2={rec['target_r2']:.4f} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    return {"arm": aid, "halt": False}


def _load_cross_arm_entry(arm_dir: Path, layer: int, conv: str) -> dict | tuple[None, str]:
    """Load one arm's denoised half-1 factors + bookkeeping for the cross-arm phase.

    Returns the entry dict, or ``(None, reason)`` when the arm must be skipped (unit not
    run / halted, factor vectors not persisted, or zero denoised factors) — recorded, never
    silently dropped.
    """
    unit_path = arm_dir / f"L{layer}_{conv}.json"
    if not unit_path.is_file():
        return None, "unit JSON missing (arm halted or unit not run)"
    unit = json.loads(unit_path.read_text())
    fname = unit.get("factor_vectors_file")
    if not fname or not (arm_dir / fname).is_file():
        return None, "factor vectors not persisted (pre-vector unit format)"
    rank = int(unit["denoised_rank"])
    if rank <= 0:
        return None, "denoised_rank=0 — no denoised factors to compare"
    npz = np.load(arm_dir / fname, allow_pickle=False)
    r = int(min(rank, npz["context_factors_half1"].shape[1]))
    return {
        "ctx": npz["context_factors_half1"][:, :r].astype(np.float64),
        "shf": npz["shift_factors_half1"][:, :r].astype(np.float64),
        "sig": np.asarray(npz["singular_values_half1"][:r], dtype=np.float64),
        # Within-arm split-half factor agreement per half-1 index (the noise floor).
        "within": {int(m["i1"]): float(m["factor_cos"]) for m in unit["factor_matches"]},
        "bases": unit["factor_bases"],
        "rank": rank,
        "r": r,
        "unit_rk": unit["regime_key"],
    }


def run_cross_arm(
    arms: list[dict],
    out_root: Path,
    *,
    layers: list[int],
    conventions: list[str],
    n_null_draws: int = N_CROSS_NULL_DRAWS,
    resume: bool = True,
) -> dict:
    """Cross-arm shared-factor phase (plan leg 6 step 4 / H6 / the leg-6 criterion input).

    Per (layer, convention) cell: load each arm's per-arm unit JSON + factor sidecar, keep
    the leading DENOISED half-1 factors, refuse basis-mismatched pairs (recorded reason,
    never a fabricated cosine), greedily match factors across every arm pair by
    min(|cos_context|, |cos_shift|), and read each match against THREE statistics — the
    SELECTION-SYMMETRIC max-matched rotation null (the criterion band: every draw runs the
    IDENTICAL greedy matching, then max-over-matched; FWER 0.05 per pair at any rank k),
    the per-comparison rotation null (reused #1345 band, kept labeled UNCORRECTED for the
    record), and the within-arm split-half factor agreement as the NOISE FLOOR. The
    criterion input counts same-behavior pairs' matches above the SYMMETRIC band (H6
    scopes the shared-factor claim to same-behavior arms); all-pairs counts ride as
    labeled companions.
    """
    cross_dir = out_root / "leg6" / "cross_arm"
    beh = {a["arm_id"]: a.get("beh_key", "") for a in arms}
    band_cache: dict = {}
    oc = _oc()
    cells: list[dict] = []
    t0 = time.time()
    n_cells = len(layers) * len(conventions)
    cell_i = 0
    for layer in layers:
        for conv in conventions:
            cell_i += 1
            entries: dict[str, dict] = {}
            skipped: list[dict] = []
            for a in sorted(beh):
                loaded = _load_cross_arm_entry(out_root / "leg6" / a, layer, conv)
                if isinstance(loaded, tuple):
                    skipped.append({"arm": a, "reason": loaded[1]})
                else:
                    entries[a] = loaded
            rk = cross_regime_key(
                layer=layer,
                convention=conv,
                n_null_draws=n_null_draws,
                null_seed=CROSS_NULL_SEED,
                arm_unit_keys={a: e["unit_rk"] for a, e in entries.items()},
            )
            cell_path = cross_dir / f"L{layer}_{conv}.json"
            if resume and cell_path.is_file():
                try:
                    prior = json.loads(cell_path.read_text())
                except json.JSONDecodeError:
                    prior = {}
                if prior.get("regime_key") == rk:
                    cells.append(prior)
                    log.info("[leg6-crossarm] cell L%d/%s resume-skip", layer, conv)
                    continue
            pairs: list[dict] = []
            n_same_sym = n_all_sym = n_same_pc = n_all_pc = 0
            n_pairs_same_beh = 0
            shared_same_sym: list[str] = []
            bands_used: dict[str, dict] = {}
            sym_bands_used: dict[str, dict] = {}
            for a, b in combinations(sorted(entries), 2):
                ea, eb = entries[a], entries[b]
                same_beh = beh[a] == beh[b]
                pair: dict = {"arm_a": a, "arm_b": b, "same_behavior": bool(same_beh)}
                if ea["bases"] != eb["bases"]:
                    diff = [
                        side
                        for side in ("context", "shift")
                        if ea["bases"].get(side) != eb["bases"].get(side)
                    ]
                    pair["admissible"] = False
                    pair["refusal_reason"] = (
                        f"factor_bases mismatch on {diff or ['<top-level>']} — a cosine "
                        "between vectors in different bases is not a similarity "
                        "(recorded skip, no number fabricated)"
                    )
                    pairs.append(pair)
                    continue
                pair["admissible"] = True
                if same_beh:
                    n_pairs_same_beh += 1
                d_ctx = int(ea["ctx"].shape[0])
                d_shf = int(ea["shf"].shape[0])
                band_ctx = rotation_null_band(
                    d_ctx, n_draws=n_null_draws, seed=CROSS_NULL_SEED, cache=band_cache
                )
                band_shf = rotation_null_band(
                    d_shf, n_draws=n_null_draws, seed=CROSS_NULL_SEED, cache=band_cache
                )
                bands_used[str(d_ctx)] = band_ctx
                bands_used[str(d_shf)] = band_shf
                sym_band = symmetric_null_band(
                    d_ctx,
                    d_shf,
                    int(ea["r"]),
                    int(eb["r"]),
                    n_draws=n_null_draws,
                    seed=CROSS_NULL_SEED,
                    cache=band_cache,
                )
                sym_key = f"{d_ctx}x{d_shf}|ra{ea['r']}|rb{eb['r']}"
                sym_bands_used[sym_key] = sym_band
                sym_p95 = float(sym_band["p95_max_matched"])
                # |cos| matrices; the SHARED selection function computes the matching for
                # the observed read exactly as it does inside every null draw; recorded
                # per-match values come from the reused reference fn.
                cos_ctx_mat = np.abs(ea["ctx"].T @ eb["ctx"])
                cos_shf_mat = np.abs(ea["shf"].T @ eb["shf"])
                sel_matches, t_obs = _greedy_match_stat(cos_ctx_mat, cos_shf_mat)
                matches: list[dict] = []
                for i, j, _comb in sel_matches:
                    rc = float(
                        oc.raw_cosine_with_rotation_null(
                            torch.as_tensor(ea["ctx"][:, i : i + 1]),
                            torch.as_tensor(eb["ctx"][:, j : j + 1]),
                            n_draws=0,
                            seed=0,
                        )["raw_cosine"]
                    )
                    rs = float(
                        oc.raw_cosine_with_rotation_null(
                            torch.as_tensor(ea["shf"][:, i : i + 1]),
                            torch.as_tensor(eb["shf"][:, j : j + 1]),
                            n_draws=0,
                            seed=0,
                        )["raw_cosine"]
                    )
                    fcos = float(min(abs(rc), abs(rs)))
                    above_pc = bool(
                        abs(rc) > float(band_ctx["null_p975"])
                        and abs(rs) > float(band_shf["null_p975"])
                    )
                    # Max-statistic band: any match above the p95 of the MAX-matched null
                    # is FWER-controlled at 0.05 per pair regardless of the rank k.
                    above_sym = bool(fcos > sym_p95)
                    wa = ea["within"].get(i)
                    wb = eb["within"].get(j)
                    present = [x for x in (wa, wb) if x is not None]
                    floor = min(present) if present else None
                    matches.append(
                        {
                            "factor_a": int(i),
                            "factor_b": int(j),
                            "cos_context": rc,
                            "cos_shift": rs,
                            "factor_cos": fcos,
                            "above_symmetric_null": above_sym,
                            "above_rotation_null_percomparison": above_pc,
                            "within_agreement_a": wa,
                            "within_agreement_b": wb,
                            "splithalf_floor": floor,
                            "above_splithalf_floor": bool(floor is not None and fcos >= floor),
                            "sigma_a": float(ea["sig"][i]),
                            "sigma_b": float(eb["sig"][j]),
                        }
                    )
                    if above_sym:
                        n_all_sym += 1
                        if same_beh:
                            n_same_sym += 1
                            shared_same_sym.append(f"{a}~{b}:f{i}~f{j}")
                    if above_pc:
                        n_all_pc += 1
                        if same_beh:
                            n_same_pc += 1
                # Selection-symmetry assertion (the leg-5 dv3 shape): the observed max the
                # criterion compares to the band must equal the max over the recorded
                # matches — both sides of the comparison go through _greedy_match_stat.
                # Tolerance = fp32 factor-storage grain: the reference fn norm-divides
                # while the persisted columns are unit only to fp32 (~1e-7); a genuine
                # aggregation divergence is orders of magnitude larger.
                t_recorded = max((m["factor_cos"] for m in matches), default=0.0)
                if abs(t_obs - t_recorded) > 1e-6:
                    raise AssertionError(
                        f"cross-arm {a}~{b}: observed aggregation diverged from the null's "
                        f"({t_obs} vs {t_recorded}) — selection symmetry broken"
                    )
                pair["matches"] = matches
                pair["max_matched_cos"] = float(t_obs)
                pair["max_matched_cos_note"] = (
                    "winner's-curse-inflated point estimate (max over greedy matches); the "
                    "symmetric band equalizes the CHANCES, not the magnitude — never a "
                    "corrected estimate"
                )
                pair["symmetric_null_key"] = sym_key
                pair["above_symmetric_null_any"] = bool(t_obs > sym_p95)
                pair["null_aggregation_matches_observed"] = True
                pairs.append(pair)
            cell = {
                "layer": int(layer),
                "context_convention": conv,
                "factor_half": "half1 (leading denoised prefix per arm)",
                "matching_rule": (
                    "greedy by min(|cos_context|, |cos_shift|): arm-a factors in "
                    "descending-sigma order each take the best unmatched arm-b factor "
                    "(the within-arm greedy_factor_match |cos| convention); only factors "
                    "inside each arm's denoised rank are compared; the SAME selection runs "
                    "inside every symmetric-null draw (_greedy_match_stat)"
                ),
                "factor_orientation": FACTOR_ORIENTATION,
                "statistic_classes": {
                    "cross_arm_factor_cosine": STAT_CLASS_CROSS_COS,
                    "symmetric_null": STAT_CLASS_SYMMETRIC_NULL,
                    "rotation_null_percomparison": STAT_CLASS_ROTATION_NULL,
                    "splithalf_floor": STAT_CLASS_SPLITHALF_FLOOR,
                },
                "rotation_null_bands_percomparison": bands_used,
                "symmetric_null_bands": sym_bands_used,
                "assertions": {"null_aggregation_matches_observed": True},
                "n_null_draws": int(n_null_draws),
                "null_seed": CROSS_NULL_SEED,
                "arms": {
                    a: {
                        "denoised_rank": e["rank"],
                        "n_factors_compared": e["r"],
                        "unit_regime_key": e["unit_rk"],
                    }
                    for a, e in entries.items()
                },
                "skipped_arms": skipped,
                "pairs": pairs,
                "criterion": {
                    "registered": (
                        "leg 6: >=1 cross-arm shared factor above the rotation null "
                        "(H6 scopes the shared-factor claim to same-behavior arms)"
                    ),
                    "shared_factor_definition": (
                        "a greedily matched cross-arm factor pair whose "
                        "min(|cos_context|, |cos_shift|) exceeds the SELECTION-SYMMETRIC "
                        "max-matched rotation-null band (p95 of the per-draw same-selection "
                        "max; FWER 0.05 per pair at any rank k)"
                    ),
                    "n_shared_above_null_same_behavior": int(n_same_sym),
                    "n_shared_above_null_all_pairs": int(n_all_sym),
                    "pairs_above_null_same_behavior": shared_same_sym,
                    "met": bool(n_same_sym >= 1),
                    "n_same_behavior_pairs_tested": int(n_pairs_same_beh),
                    "pair_multiplicity_note": (
                        "the band controls false positives PER PAIR at 0.05; across "
                        f"{n_pairs_same_beh} same-behavior pairs the any-pair criterion "
                        "retains pair-level multiplicity (reader may Bonferroni)"
                    ),
                    "per_comparison_uncorrected": {
                        "label": (
                            "UNCORRECTED per-comparison read (single-comparison p97.5 band "
                            "vs k greedy matches — multiplicity-inflated: P(>=1|H0) = "
                            "1-(0.95)^k, e.g. 0.79 at k=30; kept for the record, NOT the "
                            "criterion input)"
                        ),
                        "n_shared_above_percomparison_null_same_behavior": int(n_same_pc),
                        "n_shared_above_percomparison_null_all_pairs": int(n_all_pc),
                    },
                },
                "regime_key": rk,
                "metadata": _meta(),
            }
            _atomic_json(cell_path, cell)
            cells.append(cell)
            print(
                f"[leg6-crossarm] cell {cell_i}/{n_cells} L{layer}/{conv} "
                f"arms={len(entries)} pairs={len(pairs)} "
                f"shared_above_sym_null_same_beh={n_same_sym} "
                f"(percomparison={n_same_pc}) elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    summary = {
        "cells": [
            {
                "layer": c["layer"],
                "context_convention": c["context_convention"],
                "n_shared_above_null_same_behavior": (
                    c["criterion"]["n_shared_above_null_same_behavior"]
                ),
                "n_shared_above_null_all_pairs": c["criterion"]["n_shared_above_null_all_pairs"],
                "met": c["criterion"]["met"],
            }
            for c in cells
        ],
        "criterion_met_any_cell": bool(any(c["criterion"]["met"] for c in cells)),
        "metadata": _meta(),
    }
    _atomic_json(cross_dir / "summary.json", summary)
    return summary


def leg6_upload_leaves(out_root: Path) -> list[tuple[Path, str]]:
    """Enumerate EVERY leaf directory the leg-6 battery writes (fail-loud on none).

    A leaf = any directory under ``<out_root>/leg6`` (the root itself included, for
    ``summary.json``) holding >=1 direct file: per-arm dirs (unit JSONs + factor .npz +
    operator_factors.pt), ``pooled/<arm>/`` dirs, and ``cross_arm/``. Enumerated FROM DISK
    at upload time so a per-issue verify covers every prefix the run wrote — never only
    the current phase's (#1773). Returns ``(local_dir, rel_prefix)`` with ``rel_prefix``
    relative to ``leg6`` ("" for the root), sorted for determinism.
    """
    leg6 = out_root / "leg6"
    if not leg6.is_dir():
        raise RuntimeError(f"[leg6-upload] nothing to upload: {leg6} missing — run phases first")
    leaves: list[tuple[Path, str]] = []
    for d in sorted([leg6, *(p for p in leg6.rglob("*") if p.is_dir())]):
        if any(f.is_file() for f in d.iterdir()):
            leaves.append((d, "" if d == leg6 else d.relative_to(leg6).as_posix()))
    if not leaves:
        raise RuntimeError(f"[leg6-upload] no files under {leg6} — run the phases first")
    return leaves


def run_upload(out_root: Path, *, hf_prefix: str, skip: bool) -> None:
    """Production HF upload of the FULL leg6 tree with fail-loud exact-set verify.

    Mirrors ``issue2569_weights.phase_upload``: ``--skip-upload`` SKIPS LOUDLY (smoke /
    local runs must never clobber the production prefix); one ``upload_dir_sharded``
    call per leaf (hub-routed, overflow-aware, non-recursive by design) preserving the
    ``leg6/...`` relative layout under ``<hf_prefix>/leg6/...`` — one staging download of
    that prefix reproduces the exact local tree, so the atlas step-(5)
    ``*/operator_factors.pt`` glob works on the staged mirror unchanged. Tensors (.npz /
    .pt) land under the issue-owned ``analysis_tensors`` prefix per the Upload Policy;
    the JSONs ride the same tree (non-LFS) for pod-side durability — the VM-side harvest
    commits them to ``eval_results/`` in git. When not rerouted to the overflow repo,
    every leaf is verified by exact per-file set via ``hub.verify_repo_paths_uploaded``.
    """
    if skip:
        log.warning("[leg6-upload] --skip-upload: HF upload SKIPPED (loud)")
        return
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import issue779_common as i779

    api = HfApi()
    leaves = leg6_upload_leaves(out_root)
    for local, rel in leaves:
        prefix = f"{hf_prefix}/leg6/{rel}".rstrip("/")
        files = sorted(p.name for p in local.iterdir() if p.is_file())
        res = upload_dir_sharded(
            local,
            i779.HF_DATA_REPO,
            prefix,
            repo_type="dataset",
            shard_glob="*",
            verify=True,
            delete_local=False,
            resume_skip=False,
        )
        if not res.rerouted:
            expected = [f"{prefix}/{n}" for n in files]
            missing = hub.verify_repo_paths_uploaded(
                api, i779.HF_DATA_REPO, expected, path_in_repo=prefix
            )
            assert not missing, f"[leg6-upload] verify FAILED — missing on Hub: {missing}"
        log.info(
            "[leg6-upload] %s -> %s (%d files, rerouted=%s)",
            local,
            prefix,
            len(files),
            res.rerouted,
        )
    print(
        f"[leg6-upload] uploaded+verified {len(leaves)} leaves under {hf_prefix}/leg6", flush=True
    )


def load_arms(arms_json: Path, *, kind: str = "content") -> list[dict]:
    """Load arm records from a staged arms.json (fail-loud on an empty selection)."""
    payload = json.loads(arms_json.read_text())
    rows = [a for a in payload["arms"] if a.get("kind") == kind]
    if not rows:
        raise RuntimeError(f"empty arm selection: no kind={kind!r} arms in {arms_json}")
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI: run the leg-6 battery over the staged #1768 stores (P-A pod phase)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--staged-root", default="/workspace/eps2569", help="staged store mirror root")
    ap.add_argument("--out-root", default="/workspace/eps2569/out", help="output root")
    ap.add_argument(
        "--arms-json",
        default=None,
        help="staged issue1900_leakrace/config/arms.json (default: under staged root)",
    )
    ap.add_argument("--arms", default=None, help="comma list of arm_ids (default: all content)")
    ap.add_argument("--layers", default="19", help="comma list of layers (default 19)")
    ap.add_argument("--context-conventions", default=",".join(CONTEXT_CONVENTIONS))
    ap.add_argument("--raw-rows-dir", default=None, help="key-source rung (iv) staging dir")
    ap.add_argument("--pooled", action="store_true", help="also run the pooled cross-arm fits")
    ap.add_argument(
        "--cross-arm",
        action="store_true",
        help="also run the cross-arm shared-factor phase (plan leg 6 step 4 / H6)",
    )
    ap.add_argument(
        "--null-draws",
        type=int,
        default=N_CROSS_NULL_DRAWS,
        help="rotation-null draws per unique dimension for the cross-arm band",
    )
    ap.add_argument("--no-resume", action="store_true", help="recompute even if units exist")
    ap.add_argument(
        "--skip-upload",
        action="store_true",
        help="skip the terminal HF upload phase LOUDLY (smoke / local runs)",
    )
    ap.add_argument(
        "--hf-prefix",
        default="issue2569_theory/analysis_tensors",
        help="HF data-repo destination prefix (issue-owned; never a parent's prefix)",
    )
    ap.add_argument("--import-check", action="store_true", help="static arg/bind check, exit 0")
    args = ap.parse_args(argv)

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)

    staged_root = Path(args.staged_root)
    out_root = Path(args.out_root)
    arms_json = (
        Path(args.arms_json)
        if args.arms_json
        else (staged_root / "issue1900_leakrace" / "config" / "arms.json")
    )
    arms = load_arms(arms_json, kind="content")
    if args.arms:
        want = {a.strip() for a in args.arms.split(",") if a.strip()}
        arms = [a for a in arms if a["arm_id"] in want]
        if not arms:
            raise RuntimeError(f"empty arm selection after --arms filter: {sorted(want)}")
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    convs = [c.strip() for c in args.context_conventions.split(",") if c.strip()]
    for c in convs:
        if c not in CONTEXT_CONVENTIONS:
            raise ValueError(f"unknown context convention {c!r} (known: {CONTEXT_CONVENTIONS})")
    raw_rows_dir = Path(args.raw_rows_dir) if args.raw_rows_dir else None
    resume = not args.no_resume

    results = []
    for k, arm in enumerate(arms, start=1):
        t0 = time.time()
        res = run_arm(
            arm,
            staged_root,
            out_root,
            layers=layers,
            conventions=convs,
            raw_rows_dir=raw_rows_dir,
            resume=resume,
        )
        results.append(res)
        print(
            f"[leg6] arm {k}/{len(arms)} {arm['arm_id']} halt={res.get('halt')} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    if args.pooled:
        by_beh: dict[str, list[dict]] = {}
        for a in arms:
            by_beh.setdefault(a["beh_key"], []).append(a)
        for beh, group in sorted(by_beh.items()):
            for target in group:
                siblings = [a for a in group if a["arm_id"] != target["arm_id"]]
                run_pooled(
                    target,
                    siblings,
                    staged_root,
                    out_root,
                    layers=layers,
                    conventions=convs,
                    raw_rows_dir=raw_rows_dir,
                    resume=resume,
                )
    cross_summary = None
    if args.cross_arm:
        cross_summary = run_cross_arm(
            arms,
            out_root,
            layers=layers,
            conventions=convs,
            n_null_draws=args.null_draws,
            resume=resume,
        )
    halted = [r["arm"] for r in results if r.get("halt")]
    _atomic_json(
        out_root / "leg6" / "summary.json",
        {
            "arms_run": [r["arm"] for r in results],
            "arms_halted": halted,
            "cross_arm": cross_summary,
            "metadata": _meta(),
        },
    )
    # Terminal phase LAST (after summary.json so the summary itself uploads): full-tree
    # HF persistence with fail-loud exact-set verify (mirrors issue2569_weights).
    run_upload(out_root, hf_prefix=args.hf_prefix, skip=args.skip_upload)
    log.info("[leg6] done: %d arms, %d halted", len(results), len(halted))
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
