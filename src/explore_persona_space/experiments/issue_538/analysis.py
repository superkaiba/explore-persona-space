"""DV1-DV5 + GD1 / GD2 / GD3 analyses for issue #527.

Plan §6 (Evaluation). Pure numpy — no model calls, no GPU. Operates on the
per-context shift matrices and ΔG arrays the ``shift_extract`` step produces
(plan §4 Step 7 / Assumption #16).

Per cell (= per (pair, arm, seed)) the analysis takes:
- ``M_A`` (n_contexts × HIDDEN_SIZE) — singleton A-only shifts.
- ``M_B`` (n_contexts × HIDDEN_SIZE) — singleton B-only shifts.
- ``M_joint`` (n_contexts × HIDDEN_SIZE) — joint A+B shifts.
- ``delta_logp_A``, ``delta_logp_B``, ``delta_logp_joint`` per context.

Per-pair × per-seed the cell-level outputs are stacked for the
``H1``/``H2``/``H3`` PASS rules.
"""

# ruff: noqa: RUF002  # math/scientific notation in docstrings

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GD1_TOP1_SV_GATE: float = 0.75
GD1_EFFECTIVE_RANK_GATE: float = 2.0
GD2_SINGLETON_COSINE_GATE: float = 0.6
GD3_EFFECTIVE_RANK_GATE: float = 2.0

# Hypothesis thresholds (plan §6 Thresholds).
H1_DV1_MEDIAN: float = 0.85
H1_DV1_COVERAGE: float = 0.80
H2_RESIDUAL_NAT_MAX: float = 1.0
DV4_EMISSION_GATE: float = 0.5


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity between two 1-D vectors (numpy)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _svd_spectrum(M: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Return (top-1 SV share, effective rank via participation ratio, singular values).

    M shape: (n_rows, hidden). Effective rank = (Σ s_i^2)^2 / Σ s_i^4
    (participation ratio of the squared spectrum). Top-1 share =
    s_1^2 / Σ s_i^2.
    """
    if M.ndim != 2:
        raise ValueError(f"_svd_spectrum expects 2D, got shape={M.shape}")
    # economy SVD — only singular values needed.
    s = np.linalg.svd(M, compute_uv=False)
    s2 = s.astype(np.float64) ** 2
    denom = float(s2.sum())
    if denom <= 0.0:
        return 0.0, 0.0, s
    top1_share = float(s2[0] / denom)
    eff_rank = float((s2.sum() ** 2) / (s2**2).sum())
    return top1_share, eff_rank, s


@dataclass
class CellAnalysis:
    """All DVs + gating diagnostics for ONE (pair, seed) — joint vs singletons."""

    pair_id: str
    seed: int
    n_contexts: int

    # Plan §6 DV1-5
    dv1_cosines: np.ndarray = field(default_factory=lambda: np.zeros(0))  # per context
    dv1_median: float = 0.0
    dv1_coverage_at_threshold: float = 0.0  # frac of contexts with cos >= H1_DV1_MEDIAN

    dv2_residual_raw: np.ndarray = field(default_factory=lambda: np.zeros(0))  # per context
    dv2_residual_norm: np.ndarray = field(default_factory=lambda: np.zeros(0))  # normalized
    dv2_residual_norm_median: float = 0.0

    dv3_magnitude_residual: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dv3_residual_median: float = 0.0

    # DV4 = emission gate, plan §6 (precondition)
    dv4_source_emission_a: float = 0.0
    dv4_source_emission_b: float = 0.0
    dv4_source_emission_joint_a: float = 0.0
    dv4_source_emission_joint_b: float = 0.0

    # DV5 = singleton-vs-joint strength match
    dv5_glogprob_gap_a: float = 0.0
    dv5_glogprob_gap_b: float = 0.0
    dv5_emission_gap_a: float = 0.0
    dv5_emission_gap_b: float = 0.0

    # Gating diagnostics (plan §6)
    gd1_top1_sv_share: float = 0.0
    gd1_effective_rank: float = 0.0
    gd2_singleton_cosine_median: float = 0.0
    gd3_a_top1_sv_share: float = 0.0
    gd3_a_effective_rank: float = 0.0
    gd3_b_top1_sv_share: float = 0.0
    gd3_b_effective_rank: float = 0.0

    # Resolved gate booleans
    gd1_pass: bool = False
    gd2_pass: bool = False
    gd3_pass: bool = False
    dv4_pass: bool = False
    dv1_diagnostic: bool = False  # gd1 & gd2 & gd3 & dv4 (the headline read condition)

    h1_pass: bool = False
    h2_pass: bool = False

    base_cos_a_b: float = 0.0  # base-model centered L20 cos(A,B) — manipulation check


def analyze_cell(
    *,
    pair_id: str,
    seed: int,
    pair_a: str,
    pair_b: str,
    contexts: list[str],
    shift_a: dict[str, np.ndarray],  # context_name -> shift vector
    shift_b: dict[str, np.ndarray],
    shift_joint: dict[str, np.ndarray],
    delta_logp_a: dict[str, float],
    delta_logp_b: dict[str, float],
    delta_logp_joint: dict[str, float],
    source_emission_a: dict[str, float] | None = None,  # at source-self only
    source_emission_b: dict[str, float] | None = None,
    base_cos_a_b: float = 0.0,
) -> CellAnalysis:
    """Compute every DV + gating diagnostic for ONE (pair, seed) cell.

    ``contexts`` is the held-out eval panel (typically the 19 #311
    personas + default assistant + the 2 sources of the pair, dedup'd).
    All four ``shift_*`` dicts MUST agree on the context set.
    """
    out = CellAnalysis(pair_id=pair_id, seed=seed, n_contexts=len(contexts))
    out.base_cos_a_b = float(base_cos_a_b)

    n = len(contexts)
    hidden_dim = next(iter(shift_a.values())).shape[0]
    M_a = np.zeros((n, hidden_dim), dtype=np.float64)
    M_b = np.zeros((n, hidden_dim), dtype=np.float64)
    M_joint = np.zeros((n, hidden_dim), dtype=np.float64)
    for i, c in enumerate(contexts):
        for label, mat, src in (
            ("A_only", M_a, shift_a),
            ("B_only", M_b, shift_b),
            ("joint", M_joint, shift_joint),
        ):
            if c not in src:
                raise AssertionError(f"shift_{label!s} missing context={c!r}")
            mat[i] = src[c].astype(np.float64)

    # ── DV1: per-context cos(shift_joint, shift_a + shift_b) ─────────────
    sums = M_a + M_b
    dv1 = np.array([_cosine(M_joint[i], sums[i]) for i in range(n)], dtype=np.float64)
    out.dv1_cosines = dv1
    out.dv1_median = float(np.median(dv1))
    out.dv1_coverage_at_threshold = float((dv1 >= H1_DV1_MEDIAN).mean())

    # ── DV2: residual + normalized residual ───────────────────────────────
    residual_vec = M_joint - sums  # (n, hidden)
    res_raw = np.linalg.norm(residual_vec, axis=1)
    joint_norm = np.linalg.norm(M_joint, axis=1)
    # Avoid div-by-zero; if joint norm is 0 the cell is degenerate
    res_norm = np.where(joint_norm > 1e-12, res_raw / np.maximum(joint_norm, 1e-12), 0.0)
    out.dv2_residual_raw = res_raw
    out.dv2_residual_norm = res_norm
    out.dv2_residual_norm_median = float(np.median(res_norm))

    # ── DV3: magnitude additivity in log P(marker) ────────────────────────
    da = np.array([delta_logp_a[c] for c in contexts], dtype=np.float64)
    db = np.array([delta_logp_b[c] for c in contexts], dtype=np.float64)
    dj = np.array([delta_logp_joint[c] for c in contexts], dtype=np.float64)
    dv3 = dj - (da + db)
    out.dv3_magnitude_residual = dv3
    out.dv3_residual_median = float(np.median(dv3))

    # ── DV4: source-self emission (precondition gate) ─────────────────────
    # Source-self means "evaluate the trained source's adapter under its
    # own persona context." Stored separately because the source IS a
    # context too but the emission rate at source-self is the gate.
    if source_emission_a is not None and pair_a in source_emission_a:
        out.dv4_source_emission_a = float(source_emission_a[pair_a])
    if source_emission_b is not None and pair_b in source_emission_b:
        out.dv4_source_emission_b = float(source_emission_b[pair_b])
    # Joint arm — both sources need to clear the emission floor for the
    # H1 read to be a superposition test of the joint TRAINING (not a
    # half-floored implant).
    if source_emission_a is not None:
        # Use the joint-arm emission probe under pair_a's persona; the
        # caller is expected to populate this key from the joint adapter's
        # source-self eval.
        joint_key = f"joint_{pair_a}"
        if joint_key in source_emission_a:
            out.dv4_source_emission_joint_a = float(source_emission_a[joint_key])
    if source_emission_b is not None:
        joint_key = f"joint_{pair_b}"
        if joint_key in source_emission_b:
            out.dv4_source_emission_joint_b = float(source_emission_b[joint_key])

    out.dv4_pass = (
        out.dv4_source_emission_a >= DV4_EMISSION_GATE
        and out.dv4_source_emission_b >= DV4_EMISSION_GATE
        and out.dv4_source_emission_joint_a >= DV4_EMISSION_GATE
        and out.dv4_source_emission_joint_b >= DV4_EMISSION_GATE
    )

    # ── DV5: strength match (singleton vs joint at the source) ────────────
    out.dv5_glogprob_gap_a = (
        float(da[contexts.index(pair_a)] - dj[contexts.index(pair_a)])
        if pair_a in contexts
        else 0.0
    )
    out.dv5_glogprob_gap_b = (
        float(db[contexts.index(pair_b)] - dj[contexts.index(pair_b)])
        if pair_b in contexts
        else 0.0
    )
    if (
        source_emission_a is not None
        and pair_a in source_emission_a
        and f"joint_{pair_a}" in source_emission_a
    ):
        out.dv5_emission_gap_a = float(
            source_emission_a[pair_a] - source_emission_a[f"joint_{pair_a}"]
        )
    if (
        source_emission_b is not None
        and pair_b in source_emission_b
        and f"joint_{pair_b}" in source_emission_b
    ):
        out.dv5_emission_gap_b = float(
            source_emission_b[pair_b] - source_emission_b[f"joint_{pair_b}"]
        )

    # ── GD1: joint-shift SVD ───────────────────────────────────────────────
    top1_j, eff_j, _s_j = _svd_spectrum(M_joint)
    out.gd1_top1_sv_share = top1_j
    out.gd1_effective_rank = eff_j
    out.gd1_pass = (top1_j <= GD1_TOP1_SV_GATE) and (eff_j >= GD1_EFFECTIVE_RANK_GATE)

    # ── GD2: median cosine(shift_A(c), shift_B(c)) ─────────────────────────
    singleton_cos = np.array([_cosine(M_a[i], M_b[i]) for i in range(n)], dtype=np.float64)
    out.gd2_singleton_cosine_median = float(np.median(singleton_cos))
    out.gd2_pass = float(np.median(singleton_cos)) <= GD2_SINGLETON_COSINE_GATE

    # ── GD3: per-singleton SVDs ────────────────────────────────────────────
    top1_a, eff_a, _ = _svd_spectrum(M_a)
    top1_b, eff_b, _ = _svd_spectrum(M_b)
    out.gd3_a_top1_sv_share = top1_a
    out.gd3_a_effective_rank = eff_a
    out.gd3_b_top1_sv_share = top1_b
    out.gd3_b_effective_rank = eff_b
    out.gd3_pass = (eff_a >= GD3_EFFECTIVE_RANK_GATE) and (eff_b >= GD3_EFFECTIVE_RANK_GATE)

    # ── Headline read condition ────────────────────────────────────────────
    out.dv1_diagnostic = bool(out.gd1_pass and out.gd2_pass and out.gd3_pass and out.dv4_pass)
    out.h1_pass = bool(
        out.dv1_diagnostic
        and out.dv1_median >= H1_DV1_MEDIAN
        and out.dv1_coverage_at_threshold >= H1_DV1_COVERAGE
    )
    out.h2_pass = bool(abs(out.dv3_residual_median) < H2_RESIDUAL_NAT_MAX)

    return out


def cell_to_dict(cell: CellAnalysis) -> dict:
    """Serialize CellAnalysis to a JSON-safe dict.

    Per-context arrays are kept (per-cell artifact is small — n_contexts ≤ 21).
    """
    return {
        "pair_id": cell.pair_id,
        "seed": cell.seed,
        "n_contexts": cell.n_contexts,
        "base_cos_a_b": cell.base_cos_a_b,
        "dv1": {
            "per_context_cosines": cell.dv1_cosines.tolist(),
            "median": cell.dv1_median,
            "coverage_at_threshold": cell.dv1_coverage_at_threshold,
        },
        "dv2": {
            "residual_raw": cell.dv2_residual_raw.tolist(),
            "residual_norm": cell.dv2_residual_norm.tolist(),
            "residual_norm_median": cell.dv2_residual_norm_median,
        },
        "dv3": {
            "magnitude_residual": cell.dv3_magnitude_residual.tolist(),
            "residual_median": cell.dv3_residual_median,
        },
        "dv4": {
            "source_emission_a": cell.dv4_source_emission_a,
            "source_emission_b": cell.dv4_source_emission_b,
            "source_emission_joint_a": cell.dv4_source_emission_joint_a,
            "source_emission_joint_b": cell.dv4_source_emission_joint_b,
            "pass": cell.dv4_pass,
        },
        "dv5": {
            "glogprob_gap_a": cell.dv5_glogprob_gap_a,
            "glogprob_gap_b": cell.dv5_glogprob_gap_b,
            "emission_gap_a": cell.dv5_emission_gap_a,
            "emission_gap_b": cell.dv5_emission_gap_b,
        },
        "gating_diagnostics": {
            "gd1_top1_sv_share": cell.gd1_top1_sv_share,
            "gd1_effective_rank": cell.gd1_effective_rank,
            "gd1_pass": cell.gd1_pass,
            "gd2_singleton_cosine_median": cell.gd2_singleton_cosine_median,
            "gd2_pass": cell.gd2_pass,
            "gd3_a_top1_sv_share": cell.gd3_a_top1_sv_share,
            "gd3_a_effective_rank": cell.gd3_a_effective_rank,
            "gd3_b_top1_sv_share": cell.gd3_b_top1_sv_share,
            "gd3_b_effective_rank": cell.gd3_b_effective_rank,
            "gd3_pass": cell.gd3_pass,
        },
        "dv1_diagnostic": cell.dv1_diagnostic,
        "h1_pass": cell.h1_pass,
        "h2_pass": cell.h2_pass,
    }
