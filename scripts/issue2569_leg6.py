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
# Lambda grid GENERATING PARAMETERS (machine-stable resume key — never hash float bytes).
LAMBDA_GRID_PARAMS = ("logspace", -6.0, 2.0, 17)
GAVISH_DONOHO_COEF = 4.0 / (3.0**0.5)  # arXiv 1305.5870 — analytic reference ONLY

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
) -> dict:
    """The full B2 estimator for one (arm, layer, context-convention) unit.

    Returns a JSON-serializable record: selected lambda (+ edge flag), per-half held-out
    R^2, shuffle-calibrated thresholds, factor-match table, denoised rank, the
    Gavish-Donoho reference count, and mapping baselines (identity+bias R^2 + kNN
    retrieval, scored on the opposite half).
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

    return {
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
    }


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
        if resume and unit_path.is_file():
            try:
                prior = json.loads(unit_path.read_text())
            except json.JSONDecodeError:
                prior = {}
            if prior.get("regime_key") == rk:
                log.info("[leg6] unit %d/%d %s/L%d/%s resume-skip", k, len(units), aid, layer, conv)
                done += 1
                continue
        c_mat = context_matrix(conv, lasttoken, pooled_base, layer)
        if conv in ("last_prompt", "last_ctx") and perm_c is not None:
            c_mat = c_mat[torch.as_tensor(perm_c, dtype=torch.long)]
        c_mat = c_mat.to(torch.float32)
        rec = fit_split_half(c_mat, delta_by_layer[layer], row_shas)
        rec.update(
            {
                "arm": aid,
                "layer": int(layer),
                "context_convention": conv,
                "regime_key": rk,
                "metadata": _meta(),
            }
        )
        _atomic_json(unit_path, rec)
        done += 1
        print(
            f"[leg6] unit {k}/{len(units)} {aid}/L{layer}/{conv} "
            f"rank={rec['denoised_rank']} r2={rec['heldout_r2']['fit1_eval2']:.4f} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
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
            rec = fit_split_half(c_pool, d_pool, shas)
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
    ap.add_argument("--no-resume", action="store_true", help="recompute even if units exist")
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
    halted = [r["arm"] for r in results if r.get("halt")]
    _atomic_json(
        out_root / "leg6" / "summary.json",
        {"arms_run": [r["arm"] for r in results], "arms_halted": halted, "metadata": _meta()},
    )
    log.info("[leg6] done: %d arms, %d halted", len(results), len(halted))
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
