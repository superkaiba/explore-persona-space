# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, Δ, ×, ≥, ≫) in scientific docstrings.
"""Option-A nonlinear-benefit tests for issue #763 `neutral-contrast-and-cofit`.

PORT of the #742 plan-v9 "Option A" Stage-1 machinery from the UNMERGED sibling
branch ``origin/issue-742`` (``src/explore_persona_space/analysis/
issue_742_decoding_ceiling.py`` at commit ``86334f3d2c``; the module is NOT on
``main`` — verified ``git ls-tree origin/main`` empty), per the artifact-reuse
"porting from an unmerged sibling branch" protocol. Ported VERBATIM (drift list
below): ``PCABasis`` / ``fit_pca_basis`` (single full-sample basis),
``LeaceEraser`` / ``_symmetric_inv_sqrt`` / ``fit_leace`` (closed-form Belrose
2306.03819), ``_pairwise_euclidean`` / ``_double_center`` /
``distance_correlation`` (Székely 0803.4101), ``_planted_nonlinear_dataset``
(the synthetic power harness), ``HeldOutLeakageResult`` / ``_linear_probe_rho``
/ ``held_out_linear_leakage`` (the §4.3.2(e) linear-leakage diagnostic), and
``Stage1Verdict`` / ``classify_stage1_verdict`` (the three-part selectivity
rule, Δ_sel=0.10, Hewitt-Liang 1909.03368).

Port DRIFT (each deliberate, named for the implementer report):

1. ``option_a_permutation_test`` REPLACES ``dcor_permutation_test``: the same
   refit-the-FULL-PCA→LEACE-pipeline-per-draw loop, but each draw now computes
   BOTH dCor AND HSIC (Gretton 2005, RBF median-heuristic — the plan §4.3.2(d)
   robustness pair the #742 module lacked) on the identical freshly-erased
   points, so both statistics share one refit budget and one procedurally-
   identical null.
2. NO adaptive d_eff selection: #763's plan FIXES d=10 for the verdict (the
   must-ask bans changing PCA d) with an exploratory d=20 sensitivity leg; the
   #742 ``select_d_eff_for_power`` candidate scan is NOT ported. The power
   pre-check runs at the fixed d through the same pipeline and the
   power-limited verdict overwrite (below) carries #742's ``variance_limited``
   semantics.
3. ``paired_signflip_test`` + ``signflip_min_detectable_delta_rho`` are NEW
   (#763 §4.3.1 / §4.3.3) — the kernel-vs-linear paired-fold exact test and its
   minimal-detectable-|Δρ| simulation bound.
4. ``selectivity_branch`` vocabulary keeps the #742 implementation's
   ``{control-fails-null, effect-size-margin, non-selective, not-applicable}``
   (its documented refinement of plan v9's literal ``failed``) — carried
   verbatim so the two tasks' verdict records stay join-compatible.

OPTION-A CONTRACT (assertable, ``assert_option_a_contract``): ONE full-sample
PCA basis + ONE LEACE eraser per statistic evaluation; the observed statistic
and every one of the ``n_perm`` null draws are produced by the literally
identical PCA→LEACE→{dCor, HSIC} procedure (fit hooks are called exactly
``n_perm + 1`` times each); the ``nonlinear-yes`` verdict is gated on the
three-part selectivity rule.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# ── PCA (single full-sample basis — Option A, one commensurable frame) ────────


@dataclass
class PCABasis:
    """A fitted PCA basis: mean + the top-``d_eff`` principal-component loadings.

    ``transform(X)`` projects ``(n, d)`` onto ``(n, d_eff)`` in the fitted basis.
    Single full-sample fit (Option A) — no per-fold rotation.
    """

    mean: np.ndarray  # (d,)
    components: np.ndarray  # (d_eff, d) — rows are the top principal directions
    explained_variance_ratio: np.ndarray  # (d_eff,)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=float) - self.mean[None, :]
        return Xc @ self.components.T


def fit_pca_basis(X: np.ndarray, d_eff: int) -> PCABasis:
    """Fit a single full-sample PCA basis reducing ``(n, d)`` to ``d_eff`` dims.

    Centered SVD of the full sample (Option A — the single commensurable frame).
    ``d_eff`` is clamped to ``min(d_eff, n, d)``.
    """
    Xm = np.asarray(X, dtype=float)
    n, d = Xm.shape
    k = int(min(d_eff, n, d))
    mean = Xm.mean(axis=0)
    Xc = Xm - mean[None, :]
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k]
    total_var = float((S**2).sum())
    evr = (S[:k] ** 2) / total_var if total_var > 0 else np.zeros(k)
    return PCABasis(mean=mean, components=components, explained_variance_ratio=evr)


# ── LEACE (closed-form, Belrose 2306.03819) ───────────────────────────────────


@dataclass
class LeaceEraser:
    """A fitted closed-form LEACE eraser for a (1-D continuous) concept E0.

    ``transform(v0)`` returns the residual (mean re-added, minimal change);
    ``P`` is the (d, d) whitened oblique projection applied to centered data.
    """

    mean_x: np.ndarray  # (d,)
    P: np.ndarray  # (d, d)

    def transform(self, v0: np.ndarray) -> np.ndarray:
        X = np.asarray(v0, dtype=float)
        Xc = X - self.mean_x[None, :]
        return (Xc @ self.P.T) + self.mean_x[None, :]


def _symmetric_inv_sqrt(M: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Return (M^{1/2}, M^{-1/2}) for a symmetric PSD matrix (eigh, floored evals)."""
    Msym = 0.5 * (M + M.T)
    evals, evecs = np.linalg.eigh(Msym)
    evals_clipped = np.clip(evals, eps, None)
    sqrt = (evecs * np.sqrt(evals_clipped)) @ evecs.T
    inv_sqrt = (evecs * (1.0 / np.sqrt(evals_clipped))) @ evecs.T
    return sqrt, inv_sqrt


def fit_leace(v0: np.ndarray, E0: np.ndarray) -> LeaceEraser:
    """Fit a single full-sample closed-form LEACE eraser for the concept E0.

    Guarantee: ``cov(E0, residual) ≈ 0`` along every coordinate on the FIT
    sample (unit-tested by ``assert_option_a_contract``); orthogonal directions
    minimally changed.
    """
    X = np.asarray(v0, dtype=float)
    z = np.asarray(E0, dtype=float)
    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    zc = z - z.mean()
    sigma_xx = (Xc.T @ Xc) / n
    sigma_xz = (Xc.T @ zc) / n
    sqrt, inv_sqrt = _symmetric_inv_sqrt(sigma_xx)
    w_white = inv_sqrt @ sigma_xz
    norm = float(np.linalg.norm(w_white))
    if norm < 1e-12:
        return LeaceEraser(mean_x=X.mean(axis=0), P=np.eye(d))
    w_hat = w_white / norm
    P = np.eye(d) - sqrt @ np.outer(w_hat, w_hat) @ inv_sqrt
    return LeaceEraser(mean_x=X.mean(axis=0), P=P)


# ── dCor (Székely 0803.4101) + HSIC (Gretton 2005) ───────────────────────────


def _pairwise_euclidean(A: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix for ``(n, d)`` (or ``(n,)``) input."""
    Am = np.asarray(A, dtype=float)
    if Am.ndim == 1:
        Am = Am[:, None]
    sq = np.sum(Am**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Am @ Am.T)
    d2 = np.clip(d2, 0.0, None)
    return np.sqrt(d2)


def _double_center(D: np.ndarray) -> np.ndarray:
    """Double-centering of a distance matrix (Székely 2007)."""
    row = D.mean(axis=1, keepdims=True)
    col = D.mean(axis=0, keepdims=True)
    grand = D.mean()
    return D - row - col + grand


def distance_correlation(X: np.ndarray, y: np.ndarray) -> float:
    """Distance correlation ``dCor(X, y)`` in [0, 1] (0 iff independent)."""
    A = _double_center(_pairwise_euclidean(X))
    B = _double_center(_pairwise_euclidean(y))
    dcov2 = float((A * B).mean())
    dvarx2 = float((A * A).mean())
    dvary2 = float((B * B).mean())
    denom = np.sqrt(dvarx2 * dvary2)
    if denom < 1e-12:
        return 0.0
    dcor2 = dcov2 / denom
    return float(np.sqrt(max(0.0, dcor2)))


def _median_heuristic_sigma(D: np.ndarray) -> float:
    """RBF bandwidth σ = median of the strictly-positive pairwise distances."""
    off = D[np.triu_indices_from(D, k=1)]
    pos = off[off > 0]
    return float(np.median(pos)) if pos.size else 1.0


def hsic_statistic(X: np.ndarray, y: np.ndarray) -> float:
    """Biased-V-statistic HSIC with RBF kernels at the median-heuristic bandwidth.

    ``HSIC = (1/n²) tr(K H L H)`` with ``H = I − 11ᵀ/n`` (Gretton et al. 2005).
    The plan §4.3.2(d) robustness pair to dCor — computed on the SAME erased
    points inside the same refit-per-draw permutation loop.
    """
    Dx = _pairwise_euclidean(X)
    Dy = _pairwise_euclidean(y)
    sx = _median_heuristic_sigma(Dx)
    sy = _median_heuristic_sigma(Dy)
    K = np.exp(-(Dx**2) / (2.0 * sx**2))
    L = np.exp(-(Dy**2) / (2.0 * sy**2))
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(K @ H @ L @ H) / (n**2))


# ── the Option-A refit-per-draw permutation test (dCor + HSIC jointly) ────────


@dataclass
class OptionAPermutationResult:
    """Refit-per-draw permutation result for ONE statistic (dCor or HSIC)."""

    stat: float
    null: np.ndarray
    p_value: float
    d_eff: int
    n_perm: int

    # back-compat aliases (the #742 verdict classifier reads .dcor / .p_value)
    @property
    def dcor(self) -> float:
        return self.stat


def _pipeline_stats(
    v0: np.ndarray,
    E0: np.ndarray,
    d_eff: int,
    pca_fit_fn: Callable[..., PCABasis],
    leace_fit_fn: Callable[..., LeaceEraser],
) -> tuple[float, float]:
    """One pass of the full PCA→LEACE→{dCor, HSIC} pipeline (single frame).

    Fits a fresh single-full-sample PCA basis on ``v0``, reduces, fits a fresh
    LEACE eraser against ``E0`` on the reduced points, erases, and returns
    ``(dCor, HSIC)`` of the residual vs ``E0``. Both fits route through the
    injected hooks so a counting wrapper can prove the refit-per-draw contract.
    """
    basis = pca_fit_fn(v0, d_eff)
    reduced = basis.transform(v0)
    eraser = leace_fit_fn(reduced, E0)
    residual = eraser.transform(reduced)
    return distance_correlation(residual, E0), hsic_statistic(residual, E0)


def option_a_permutation_test(
    v0: np.ndarray,
    E0: np.ndarray,
    *,
    d_eff: int = 10,
    n_perm: int = 1000,
    rng: np.random.Generator | None = None,
    pca_fit_fn: Callable[..., PCABasis] | None = None,
    leace_fit_fn: Callable[..., LeaceEraser] | None = None,
) -> tuple[OptionAPermutationResult, OptionAPermutationResult]:
    """Option-A permutation test: refit the FULL pipeline per draw; dCor + HSIC.

    For each of ``n_perm`` permutations the ``E0`` labels are permuted, then PCA
    + LEACE are RE-FIT (with the permuted E0) and dCor AND HSIC are recomputed
    on the freshly-erased points — the observed statistic and every null draw
    are produced by the identical procedure, so the label-permutation null
    absorbs the fit-after-looking-at-labels selection effect (the #742 plan-v9
    MF3 fix; a per-fold-frame pooled dCor is the named bug this replaces).

    Returns ``(dcor_result, hsic_result)`` with right-tail +1-corrected p's.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if pca_fit_fn is None:
        pca_fit_fn = fit_pca_basis
    if leace_fit_fn is None:
        leace_fit_fn = fit_leace
    v = np.asarray(v0, dtype=float)
    z = np.asarray(E0, dtype=float)

    obs_dcor, obs_hsic = _pipeline_stats(v, z, d_eff, pca_fit_fn, leace_fit_fn)
    null_dcor = np.empty(n_perm, dtype=float)
    null_hsic = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        z_perm = z[rng.permutation(len(z))]
        null_dcor[i], null_hsic[i] = _pipeline_stats(v, z_perm, d_eff, pca_fit_fn, leace_fit_fn)
    p_dcor = float((1.0 + np.sum(null_dcor >= obs_dcor)) / (1.0 + n_perm))
    p_hsic = float((1.0 + np.sum(null_hsic >= obs_hsic)) / (1.0 + n_perm))
    return (
        OptionAPermutationResult(obs_dcor, null_dcor, p_dcor, int(d_eff), int(n_perm)),
        OptionAPermutationResult(obs_hsic, null_hsic, p_hsic, int(d_eff), int(n_perm)),
    )


# ── synthetic power pre-check THROUGH the same pipeline (plan §4.3.3) ─────────


def _planted_nonlinear_dataset(
    *, d_eff: int, n: int, effect: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (v0, E0) with a planted nonlinear residual at ~``effect`` partial-corr.

    ``E0`` blends a NONLINEAR function of v0 (standardized squared-norm of the
    leading dims — its linear component is ~0) with pure noise at blend weight
    ``effect`` (ported verbatim from #742).
    """
    v0 = rng.normal(0.0, 1.0, size=(n, d_eff))
    r2 = (v0[:, : max(1, d_eff // 2)] ** 2).sum(axis=1)
    r2_std = (r2 - r2.mean()) / (r2.std() + 1e-12)
    noise = rng.normal(0.0, 1.0, size=n)
    signal = 1.0 / (1.0 + np.exp(-r2_std))
    signal = (signal - signal.mean()) / (signal.std() + 1e-12)
    E0 = effect * signal + np.sqrt(max(0.0, 1.0 - effect**2)) * noise
    return v0, E0


def dcor_power_check(
    *,
    d_eff: int = 10,
    n: int = 50,
    n_perm: int = 1000,
    effect: float = 0.10,
    n_trials: int = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """Realized power of the Option-A dCor test at the stated effect floor.

    Each trial draws a fresh planted-nonlinear ``(v0, E0)`` and runs the
    IDENTICAL single-full-sample refit-per-draw pipeline (``option_a_
    permutation_test``); detection = dCor p < 0.05. Returns the detection
    fraction. Realized power < 0.8 ⇒ any production null is reported
    "indistinguishable from null given variance", never "no nonlinear signal".
    """
    if rng is None:
        rng = np.random.default_rng(0)
    detections = 0
    for _ in range(n_trials):
        v0, E0 = _planted_nonlinear_dataset(d_eff=d_eff, n=n, effect=effect, rng=rng)
        dcor_res, _ = option_a_permutation_test(v0, E0, d_eff=d_eff, n_perm=n_perm, rng=rng)
        if dcor_res.p_value < 0.05:
            detections += 1
    return float(detections / n_trials)


# ── held-out post-LEACE linear-leakage diagnostic (§4.3.2(e)) ────────────────


@dataclass
class HeldOutLeakageResult:
    """Held-out post-LEACE linear-leakage diagnostic result (ported from #742)."""

    rho: float
    null_ci: tuple[float, float]
    post_leace_linear_pass: bool


def _linear_probe_rho(features: np.ndarray, target: np.ndarray) -> float:
    """Best-fit linear-probe correlation of ``target`` on ``features`` (with bias)."""
    F = np.asarray(features, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    y = np.asarray(target, dtype=float)
    design = np.column_stack([np.ones(len(y)), F])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    y_hat = design @ beta
    if np.std(y_hat) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(y_hat, y)[0, 1])


def held_out_linear_leakage(
    v0_train: np.ndarray,
    E0_train: np.ndarray,
    v0_held: np.ndarray,
    E0_held: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    n_perm: int = 1000,
) -> HeldOutLeakageResult:
    """Diagnose residual LINEAR leakage in the held-out LEACE residual.

    LEACE's closed-form guarantee is train-sample-only; a held-out residual can
    still carry linear E0-correlation from sampling variance, which would
    masquerade as a nonlinear residual. ``post_leace_linear_pass`` is True iff
    the held-out probe ρ sits within its own permutation-null CI.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    eraser = fit_leace(v0_train, E0_train)
    resid_held = eraser.transform(v0_held)
    rho = _linear_probe_rho(resid_held, E0_held)
    z = np.asarray(E0_held, dtype=float)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = _linear_probe_rho(resid_held, z[rng.permutation(len(z))])
    lo = float(np.percentile(null, 2.5))
    hi = float(np.percentile(null, 97.5))
    return HeldOutLeakageResult(rho=rho, null_ci=(lo, hi), post_leace_linear_pass=lo <= rho <= hi)


# ── the three-part selectivity-gated verdict (plan §4.3.2(f)/(g)) ─────────────

STAGE1_VERDICTS = frozenset(
    {
        "nonlinear-yes",
        "non-selective",
        "linear-erasure-leakage-unresolved",
        "ceiling-limited",
        "indistinguishable-from-null-given-variance",
    }
)
STAGE1_SELECTIVITY_BRANCHES = frozenset(
    {"control-fails-null", "effect-size-margin", "non-selective", "not-applicable"}
)


@dataclass(frozen=True)
class Stage1Verdict:
    """Verdict + selectivity branch; compares equal to its bare verdict string."""

    verdict: str
    selectivity_branch: str = "not-applicable"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Stage1Verdict):
            return (self.verdict, self.selectivity_branch) == (
                other.verdict,
                other.selectivity_branch,
            )
        if isinstance(other, str):
            return self.verdict == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.verdict)

    def __str__(self) -> str:
        return self.verdict


def classify_stage1_verdict(
    *,
    dcor_pass: bool,
    linear_pass: bool,
    control_res: OptionAPermutationResult | None = None,
    observed_dcor: float | None = None,
    alpha: float = 0.05,
    delta_sel: float = 0.10,
) -> Stage1Verdict:
    """The #742 plan-v9 three-part selectivity rule (Δ_sel = 0.10), ported verbatim.

    ``nonlinear-yes`` iff dcor_pass AND linear_pass AND selective, where
    selective = control fails its own null (p ≥ α) OR the true task beats the
    control by ≥ Δ_sel. dCor null → ``ceiling-limited``; linear leakage →
    ``linear-erasure-leakage-unresolved``; unselective → ``non-selective``
    (d ≫ n probe memorization, NEVER reported as yes).
    """
    if not dcor_pass:
        return Stage1Verdict("ceiling-limited", "not-applicable")
    if not linear_pass:
        return Stage1Verdict("linear-erasure-leakage-unresolved", "not-applicable")
    if control_res is None:
        return Stage1Verdict("nonlinear-yes", "not-applicable")
    if control_res.p_value >= alpha:
        return Stage1Verdict("nonlinear-yes", "control-fails-null")
    if observed_dcor is None:
        return Stage1Verdict("non-selective", "non-selective")
    if observed_dcor - control_res.stat >= delta_sel:
        return Stage1Verdict("nonlinear-yes", "effect-size-margin")
    return Stage1Verdict("non-selective", "non-selective")


# ── the per-(behavior, d) Option-A cell driver (#763 §4.3.2) ─────────────────


def run_option_a_cell(
    v0_layer: np.ndarray,
    E0: np.ndarray,
    *,
    behavior: str,
    layer: int,
    d_eff: int,
    n_perm: int,
    rng: np.random.Generator,
    realized_power: float | None = None,
    power_floor: float = 0.8,
) -> dict:
    """One Option-A cell: PCA→LEACE→{dCor, HSIC} + control task + held-out + verdict.

    ``realized_power`` is the SHARED synthetic power pre-check at this (d, n)
    config (computed once by the driver — it is behavior-independent); when it
    is below ``power_floor`` a null dCor read is overwritten to
    ``indistinguishable-from-null-given-variance`` (never "no nonlinear
    signal"), carrying #742's variance-limited semantics at the FIXED d.
    """
    n = v0_layer.shape[0]
    dcor_res, hsic_res = option_a_permutation_test(
        v0_layer, E0, d_eff=d_eff, n_perm=n_perm, rng=rng
    )
    dcor_pass = bool(dcor_res.p_value < 0.05)

    # control-task / shuffled-E0 null (Hewitt-Liang): one fixed shuffle, then the
    # same refit-per-draw procedure on the shuffled labels — BINDING on the verdict.
    shuffled = E0[rng.permutation(n)]
    control_res, _ = option_a_permutation_test(
        v0_layer, shuffled, d_eff=d_eff, n_perm=n_perm, rng=rng
    )

    # held-out post-LEACE linear-leakage diagnostic (70/30 split on the reduced pts)
    basis = fit_pca_basis(v0_layer, d_eff)
    reduced = basis.transform(v0_layer)
    perm = rng.permutation(n)
    n_train = round(0.7 * n)
    tr, he = perm[:n_train], perm[n_train:]
    held = held_out_linear_leakage(reduced[tr], E0[tr], reduced[he], E0[he], rng=rng)
    linear_pass = bool(held.post_leace_linear_pass)

    verdict_obj = classify_stage1_verdict(
        dcor_pass=dcor_pass,
        linear_pass=linear_pass,
        control_res=control_res,
        observed_dcor=dcor_res.stat,
    )
    verdict = verdict_obj.verdict
    selectivity_branch = verdict_obj.selectivity_branch
    selectivity_rule_passed = verdict == "nonlinear-yes"
    power_limited = realized_power is not None and realized_power < power_floor
    if power_limited and not dcor_pass:
        verdict = "indistinguishable-from-null-given-variance"
        selectivity_branch = "not-applicable"
        selectivity_rule_passed = False

    if verdict == "nonlinear-yes":
        # defense-in-depth: re-derive the selectivity rule from the emitted reads
        _selective = control_res.p_value >= 0.05 or (dcor_res.stat - control_res.stat) >= 0.10
        assert dcor_pass and linear_pass and _selective, (
            "nonlinear-yes emitted but the selectivity rule does not hold "
            f"(dcor_pass={dcor_pass}, linear_pass={linear_pass}, "
            f"observed={dcor_res.stat}, control={control_res.stat}, "
            f"control_p={control_res.p_value})"
        )

    return {
        "behavior": behavior,
        "layer": layer,
        "d_eff": int(d_eff),
        "coordinate_scheme": (
            "single_full_sample_pca_basis+single_full_sample_leace; "
            "dCor+HSIC pipeline refit per permutation (Option A)"
        ),
        "dcor_observed": dcor_res.stat,
        "dcor_p_value": dcor_res.p_value,
        "dcor_null_median": float(np.median(dcor_res.null)),
        "dcor_pass": dcor_pass,
        "hsic_observed": hsic_res.stat,
        "hsic_p_value": hsic_res.p_value,
        "hsic_null_median": float(np.median(hsic_res.null)),
        "control_task_dcor": control_res.stat,
        "control_task_p_value": control_res.p_value,
        "held_out_linear_rho": held.rho,
        "held_out_null_ci": list(held.null_ci),
        "post_leace_linear_pass": linear_pass,
        "selectivity_rule_passed": selectivity_rule_passed,
        "selectivity_branch": selectivity_branch,
        "verdict": verdict,
        "realized_power": realized_power,
        "power_limited": bool(power_limited),
        "n_perm": int(n_perm),
        "dcor_null": [float(x) for x in dcor_res.null],
        "control_dcor_null": [float(x) for x in control_res.null],
    }


# ── kernel-vs-linear paired sign-flip test (#763 §4.3.1) ──────────────────────


def paired_signflip_test(
    err_lin: np.ndarray,
    err_krr: np.ndarray,
    *,
    n_flips: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Exact paired sign-flip permutation test on per-context error differences.

    ``statistic = mean(err_lin − err_krr)`` (positive ⇒ the kernel is better);
    the null flips the sign of each paired difference independently (B =
    ``n_flips``); two-sided p with the +1 correction. Vectorized: one (B, n)
    Rademacher draw × one GEMV.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    d = np.asarray(err_lin, dtype=float) - np.asarray(err_krr, dtype=float)
    n = d.shape[0]
    obs = float(d.mean())
    signs = rng.integers(0, 2, size=(n_flips, n)) * 2 - 1  # ±1
    null = (signs * d[None, :]).mean(axis=1)
    p = float((1.0 + np.sum(np.abs(null) >= abs(obs))) / (1.0 + n_flips))
    return {"statistic_mean_err_diff": obs, "p_value": p, "n_flips": int(n_flips), "n": int(n)}


def signflip_min_detectable_delta_rho(
    *,
    n: int = 50,
    base_rho: float = 0.5,
    shared_noise: float = 0.5,
    delta_grid: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30),
    n_trials: int = 200,
    n_flips: int = 2000,
    target_power: float = 0.8,
    rng: np.random.Generator | None = None,
) -> dict:
    """Minimal detectable |Δρ| for the paired sign-flip test, by simulation.

    Simulation harness (registered construction — the plan mandates a same-
    harness bound without pinning the generator): per trial, draw a latent
    target ``y`` and two predictions whose POPULATION correlations with ``y``
    are ``base_rho`` and ``base_rho + δ``, with a ``shared_noise`` fraction of
    their noise shared (the paired structure the folds induce); score both on
    the rank scale, form per-context squared-error differences, run the
    sign-flip test, and count detections (p < 0.05). The minimal detectable
    |Δρ| is the smallest grid δ whose realized power ≥ ``target_power``
    (``None`` when no grid point clears it). ``n_flips`` is reduced inside the
    simulation (power estimation only; the production test uses B=10,000).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    from scipy.stats import rankdata

    power_per_delta: dict[float, float] = {}
    for delta in delta_grid:
        rho2 = min(0.995, base_rho + delta)
        detections = 0
        for _ in range(n_trials):
            y = rng.normal(size=n)
            eps_shared = rng.normal(size=n)
            eps1 = shared_noise * eps_shared + np.sqrt(1 - shared_noise**2) * rng.normal(size=n)
            eps2 = shared_noise * eps_shared + np.sqrt(1 - shared_noise**2) * rng.normal(size=n)
            pred_lin = base_rho * y + np.sqrt(1 - base_rho**2) * eps1
            pred_krr = rho2 * y + np.sqrt(1 - rho2**2) * eps2
            y_r = (rankdata(y) - 1) / (n - 1)
            e_lin = ((rankdata(pred_lin) - 1) / (n - 1) - y_r) ** 2
            e_krr = ((rankdata(pred_krr) - 1) / (n - 1) - y_r) ** 2
            res = paired_signflip_test(e_lin, e_krr, n_flips=n_flips, rng=rng)
            if res["p_value"] < 0.05:
                detections += 1
        power_per_delta[delta] = detections / n_trials
    clearing = [d for d in delta_grid if power_per_delta[d] >= target_power]
    return {
        "min_detectable_delta_rho": min(clearing) if clearing else None,
        "power_per_delta": {str(k): v for k, v in power_per_delta.items()},
        "base_rho": base_rho,
        "shared_noise": shared_noise,
        "n_trials": n_trials,
        "n_flips_sim": n_flips,
        "target_power": target_power,
    }


# ── Option-A conformance self-check (plan §4.3.2 code note) ───────────────────


@dataclass
class _CountingHooks:
    """Counting wrappers proving the refit-per-draw contract (n_perm+1 fits each)."""

    pca_calls: int = 0
    leace_calls: int = 0
    _pca: Callable[..., PCABasis] = field(default=fit_pca_basis)
    _leace: Callable[..., LeaceEraser] = field(default=fit_leace)

    def pca(self, X: np.ndarray, d_eff: int) -> PCABasis:
        self.pca_calls += 1
        return self._pca(X, d_eff)

    def leace(self, v0: np.ndarray, E0: np.ndarray) -> LeaceEraser:
        self.leace_calls += 1
        return self._leace(v0, E0)


def assert_option_a_contract(*, n: int = 24, d: int = 40, d_eff: int = 5, n_perm: int = 7) -> dict:
    """Fail-loud Option-A conformance assert (run by the smoke before any cell).

    Proves on a tiny synthetic problem that (a) PCA and LEACE are each REFIT
    exactly ``n_perm + 1`` times per permutation test (once observed + once per
    draw — the refit-per-draw contract), and (b) after a full-sample LEACE
    erasure NO linear probe recovers E0 from the erased embedding (the §4.3.2(b)
    unit test: in-sample post-erasure probe ρ ≈ 0).
    """
    rng = np.random.default_rng(763)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 1.5 + rng.normal(size=n) * 0.3
    hooks = _CountingHooks()
    option_a_permutation_test(
        X, y, d_eff=d_eff, n_perm=n_perm, rng=rng, pca_fit_fn=hooks.pca, leace_fit_fn=hooks.leace
    )
    assert hooks.pca_calls == n_perm + 1, (hooks.pca_calls, n_perm + 1)
    assert hooks.leace_calls == n_perm + 1, (hooks.leace_calls, n_perm + 1)
    basis = fit_pca_basis(X, d_eff)
    reduced = basis.transform(X)
    erased = fit_leace(reduced, y).transform(reduced)
    probe_rho = abs(_linear_probe_rho(erased, y))
    assert probe_rho < 0.05, f"post-LEACE in-sample linear probe recovered E0 (ρ={probe_rho:.3f})"
    return {
        "pca_calls": hooks.pca_calls,
        "leace_calls": hooks.leace_calls,
        "post_leace_probe_rho": probe_rho,
    }
