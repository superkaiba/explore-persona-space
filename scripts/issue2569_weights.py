"""Issue #2569 P-A weights battery, parts 1+2 (leg 1 core + leg 8 steps 1/3/4).

Phase driver over the banked layer-19 (primary; L14/L26 replicates) n1m ridge
map's ROW operator ``x |-> x @ A`` (plan v4 SS4 leg 1 steps 1-3/5/6 + leg 8
steps 1/3/4). Weights-only by default: no rows are staged (P-B refinements —
data-weighted mass fractions ``u_i^T Sigma_c u_i``, split-half spectrum error
bars, fixed-point nearest banked answers — are the rowbattery driver's job and
are NOT computed here from assumed inputs); the ONE rows-consuming exception is
the leg-8 step-4 certificates phase's probe/held-out legs, which run ONLY when
``--rows-dir`` points at the P-B P1 assemble dir (layer 19 only) and otherwise
record an EXPLICIT deferral (see the certificates phase docstring). One later
unit extends THIS file with leg 3 (wiring/receipts/attribution) plus the
two-sided SAE dashboards — including the SAE NAMING of the leg-8 step-3
monitor gradients (``sae_naming`` keys read ``deferred``) and the fixed
point's #2476-encoder decode (they need the shared SAE dictionary machinery) —
this file marks those keys ``deferred``, never stubs them.

Orientation (plan SS4 orientation dictionary — B1; verbatim, load-bearing):
all reads are on A under the ROW action ``x @ A``. INPUT/read singular
directions = LEFT singular vectors ``u_i`` (``u_i @ A = sigma_i v_i``);
OUTPUT/write directions = RIGHT singular vectors ``v_i``. Eigen dashboards
READ along RIGHT eigenvectors and WRITE along LEFT eigenvectors
(``x @ A = sum_i lam_i (x . v_i^e) u_i^e^T``) — the u/v letter assignment
FLIPS between the eigen and singular decompositions, so this module only
ever uses the landed field names ``OP.SingularTriplets.read_input_u`` /
``.write_output_v`` and ``OP.EigenPairs.read_right_v`` / ``.write_left_rows``.

Eigen normalization convention (non-normal A — the general eigenproblem):
right eigenvectors are the LAPACK-unit-norm columns of ``V`` from
``scipy.linalg.eig`` (via ``OP.eigen_read_write``); LEFT eigenvectors are
the rows of ``inv(V)`` (the plan's leg-1 step-2 convention), so the pair is
biorthonormal BY CONSTRUCTION (``write_left_rows @ read_right_v == I`` up to
inversion roundoff — persisted as ``biortho_max_err``) and the expansion
``A = sum_i lam_i v_i^e u_i^e^T`` needs no extra scaling. Left rows are NOT
unit-norm (their norms grow with kappa(V)). Complex conjugate pairs are
expected and KEPT (both members enter the |lambda|-sorted arrays); the
summary reports ``n_complex_pairs`` / ``n_real_eigs``.

Self-alignment transpose-invariance note (plan SS4 leg 1 step 3): the SCALAR
``c_i = cos(u_i @ A, u_i)`` equals ``cos(v_i, u_i)`` algebraically (sigma_i
> 0), so its VALUE is transpose-invariant (#1774 measured identical both
ways: +0.084/+0.104/+0.136 on the top three) and #1774 comparability is
unaffected. What B1 fixes is which direction FAMILY the labels attach to
(INPUT = LEFT singular vectors) — do not "correct" the c_i computation.

Anatomy classes (plan SS4 leg 1 step 3, verbatim thresholds; tau_read is
RETIRED — C6 — and deliberately absent). First-match precedence
ignored -> copied -> damped -> transcoded -> rotated_scaled:
  ignored:        sigma_i < tau_kernel
  copied:         sigma_i in [0.8, 1.25] and |c_i| >= 0.8
  damped:         sigma_i in [tau_kernel, 0.8) and |c_i| >= 0.8
  transcoded:     sigma_i >= 0.8 and |c_i| < 0.2
  rotated_scaled: everything else
tau_kernel = the singular value at which cumulative sigma^2 mass reaches 99%
(the #1768 ``operator_kv_deep`` convention; the k90 = 90%-mass twin is also
reported, #1774). Mass fractions are sigma^2-weighted with unweighted counts
alongside; the DATA-weighted variant is P-B's (key ``data_weighted_mass``
reads ``deferred-to-P-B``). If tau_kernel >= 0.8 the damped band is empty
and ``ignored`` (checked first) absorbs the overlap — documented, not a bug.

Effective kernel (leg 8 step 1): the bottom LEFT-singular subspace — INPUT
directions with sigma_i < tau_kernel. Ridge has NO exact kernel: every key,
docstring, and phrasing string says "directions the map reads at < X% of
typical gain", never "null space".

Fixed point (leg 1 step 6): solve ``x* (I - A) = b`` in the ROW form; rho(A)
is computed FRESH from THIS map's own eigenvalues (never #1774's 0.910 —
a prior from a different fit). rho >= 1 drops the iterated-map reading and
reports x* as the affine-consistency point only.

Phases (each loops the layer list; per-(phase, layer) resume predicate keyed
on the generating-parameter regime dict — never a recomputed-float hash):

  factor        B1 entry asserts already ran in main(); full fp64 SVD +
                full non-symmetric eig (OP.eigen_read_write), svds top-8
                cross-check, kappa(V), rho, biorthogonality residual ->
                leg1/factor_L<L>.pt + .json. THE expensive phase (dense
                d=3584 fp64 SVD + eig + complex inv + complex-SVD cond:
                minutes-to-tens-of-minutes per layer on cpu-bigmem; keep
                out of VM test paths — tests use d<=48 synthetics).
  anatomy       tau_kernel/k99 (+ k90 twin), per-direction classes, sigma^2
                mass fractions + counts -> leg1/anatomy_L<L>.json.
  alpha-lowrank alpha = tr(A)/d; spectrum of A - alpha*I vs A; residual
                top-k variance (k in 1/8/32/128) -> leg1/alpha_lowrank_L<L>.json.
  fixed-point   x*, ||x*||, residual check, rho branch ->
                leg1/fixed_point_L<L>.pt + .json (sae_decode deferred).
  kernel        effective-kernel basis + stats -> leg8/effective_kernel_L<L>.pt
                + .json.
  monitor-geometry  leg 8 step 3: per-trait monitor decision geometry over the
                #779 monitoring r_B set (evil/sycophancy/hallucination @
                037fcbb2, unit-normalized) — flip gradient A r (B1), minimal
                context change per unit read, least-norm pre-images
                y @ A^+ (effective-rank truncated at tau_kernel + full-pinv
                companion) WITH the coset ambiguity + gain-ratio context
                stated in every artifact -> leg8/monitor_geometry_L<L>.pt
                + .json. SAE naming of the gradient: deferred (next unit).
  certificates  leg 8 step 4: sensitivity certificates for the monitor family
                — (i) direct r^T v_C, (ii) mapped r.(v_C @ A + b), (iii) a
                fitted 1-D context probe w^T v_C -> r^T v_A. Weights-only
                sensitivities (grad norms; worst case = eps * grad_norm,
                SINGLE-application only — measured rho(A) >= 1 forecloses any
                geometric-series/iterated bound, stated as UNAVAILABLE) always
                compute; the probe fit + held-out signal normalization need
                the P-B row store and run ONLY at layer 19 with --rows-dir
                (else an explicit deferral is recorded)
                -> leg8/certificates_L<L>.json (+ certificates_probe_L19.pt
                on rows-attached runs).
  upload        production-only HF upload of leg1/ + leg8/ (fail-loud
                exact-set verify; smoke/skip is LOUD).

Output schema (all under --out-root; <L> = layer; every .pt is a torch.save
dict loaded with ``weights_only=False`` — a self-produced, pinned project
artifact, #1900 convention; every .json carries ``regime`` + ``metadata``
(git provenance + phase identity)):

  leg1/entry_asserts_L<L>.json:
    entry_asserts: {gram: {max_rel_err, n_probes},
                    singular_orientation: {max_row_form_rel_err,
                      row_form_rel_err[k], wrong_column_form_rel_err[k],
                      sigma[k], k},
                    apply_path: {max_abs_diff, max_rel_diff, n_probes}}
    selected_lambda: float
  leg1/factor_L<L>.pt:
    sigma: float64 [d] (descending)
    self_alignment_c: float64 [d] (SIGNED c_i, fp64, computed pre-downcast)
    read_input_u_fp32: float32 [d, d] (columns = INPUT/read dirs u_i)
    write_output_v_fp32: float32 [d, d] (columns = OUTPUT/write dirs v_i)
    eig_lambda: complex128 [d] (|lambda|-descending)
    eig_read_right_v_top: complex64 [d, K] (columns = READ-side right eigvecs)
    eig_write_left_rows_top: complex64 [K, d] (rows = WRITE-side left eigvecs)
    stats: {rho, kappa_v, biortho_max_err, n_complex_pairs, n_real_eigs,
            svds_sigma_max_rel_diff, d, top_k}
    regime, metadata
  leg1/factor_L<L>.json: stats + sigma/|lambda| quantiles + top-K tables
    (singular: rank/sigma/c; eigen: rank/abs_lambda/re/im/is_complex).
  leg1/anatomy_L<L>.json:
    tau_kernel, k99, k90, tau_k90, sigma_max, sigma_median,
    classes: {<label>: {count, frac_count, sigma2_mass_frac}},
    labels: [d], sigma: [d], c: [d],
    top_directions: [{rank, sigma, c, abs_c, label}] (top_k rows),
    thresholds: {...verbatim...}, precedence: "...",
    data_weighted_mass: "deferred-to-P-B (rowbattery moments)"
  leg1/alpha_lowrank_L<L>.json:
    alpha, d, fro_A, fro_residual, sigma_A: [d], sigma_residual: [d],
    var_explained_topk: {"1": f, "8": f, "32": f, "128": f}
  leg1/fixed_point_L<L>.pt: {x_star: float64 [d], regime, metadata}
  leg1/fixed_point_L<L>.json:
    rho, iterated_map_reading: bool, x_star_norm, b_norm, residual_rel,
    sae_decode: "deferred-to-sae-dashboards-unit",
    nearest_banked_answers: "deferred-to-P-B (needs rows)"
  leg8/effective_kernel_L<L>.pt:
    kernel_basis_fp32: float32 [d, m] (columns = kernel INPUT dirs, sigma<tau)
    sigma_kernel: float64 [m]; regime, metadata
  leg8/effective_kernel_L<L>.json:
    tau_kernel, k99, k90, kernel_dim, tau_over_median_gain, tau_over_max_gain,
    pct_of_typical_gain, sigma_kernel_max, sigma_kernel_mean, claims_phrasing
  leg8/monitor_geometry_L<L>.pt:
    traits: {<trait>: {r_hat: float64 [d] (unit-normalized layer-<L> r_B row),
                       gradient: float64 [d] (A @ r_hat — the B1 row-space
                         monitor gradient),
                       preimage_unit_level: float64 [d] (least-norm pre-image
                         of y = r_hat under the ROW action, effective-rank
                         truncated: retained sigma_i >= tau_kernel),
                       preimage_unit_level_fullpinv: float64 [d] (untruncated
                         full-pinv companion — every sigma_i > 0 kept)}}
    tau_kernel, caveats: [str, str], regime
  leg8/monitor_geometry_L<L>.json:
    kernel_gain_context: {tau_kernel, k99, kernel_dim, sigma_median,
      tau_over_median_gain, note (the gain-ratio caveat: tau_kernel EXCEEDS the
      median gain on the banked L19 map, so a majority of directions sit below
      tau — 'below-tau' is a RELATIVE-gain statement, never 'ignored')}
    traits: {<trait>: {grad_norm, min_context_change_per_unit_read (= 1/|A r|),
      read_at_zero_context (= r.b), gradient_mass_below_tau_frac,
      preimage_norm, preimage_fullpinv_norm, target_mass_below_tau_frac,
      achieved_level_fraction (measured r.(v_ln @ A)),
      achieved_level_fraction_algebra (= 1 - target_mass_below_tau_frac),
      preimage_orientation_residual, n_retained, kernel_dim}}
    coset_ambiguity (per-layer string: every pre-image = particular solution +
      arbitrary effective-kernel component, kernel_dim free directions + the
      gain ratio attached), affine_note, sae_naming ("deferred-..."),
    caveats: [activation-space caveat (a), map-level caveat (b)] — verbatim
  leg8/certificates_L<L>.json:
    monitors: {<trait>: {direct_projection: {gradient_is, grad_norm,
      worst_case_score_movement}, mapped_read: {same keys},
      mapped_over_direct_grad_ratio,
      fitted_probe: {status: computed|deferred..., and when computed:
        grad_norm (= |w|), selected_lambda, selector, lambda_grid_edge,
        val_r2, heldout_r2, n_train, d, n_val, n_test},
      heldout: {<monitor>: {std, corr_with_target, eps_to_move_one_heldout_sd,
        signal_to_sensitivity}} | deferral string}}
    rows: {n_train, d, n_val, n_test, ridge_convention, lambda_grid
      {kind, lo, hi, num, widen_rounds_used}} | deferral string
    formulas, bound_scope (single-application; rho >= 1 forecloses iterated
      bounds — stated UNAVAILABLE, never computed under a false premise),
    baselines: {identity_bias: "inapplicable — ...", knn_retrieval:
      "inapplicable — ..."} (scalar target: dimension mismatch, stated not
      skipped), caveats (verbatim), regime, metadata
  leg8/certificates_probe_L19.pt (rows-attached runs only):
    w: float64 [d, n_traits] (probe weights, train-centered raw-feature ridge)
    x_mean: float64 [d], t_mean: float64 [n_traits], traits: [str], regime

Smoke seam (plan SS4 Smoke run, P-A line): ``--smoke`` = L19-only layer list,
top-8 per-direction reporting width, and the 100-draw budget recorded in
``regime.n_draws`` (the empirical random-unit-direction null floor that
CONSUMES it lands with the SAE-dashboard unit; threading it now keeps the
P-A seam stable). The dense d=3584 factorizations themselves are the
operator's intrinsic size and run at full d under smoke — the smoke narrows
layers/width/draws, not the matrix (same chain, tiny n on every axis that
has an n). The upload phase is production-fenced (smoke/skip logs LOUDLY);
its live branch is exercised pod-side, never by a VM unit test.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM smoke)

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402

import issue779_common as C  # noqa: E402
import issue2569_operator as OP  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2569.weights")

TASK_ID = 2569
LAYERS_PRODUCTION = (19, 14, 26)  # L19 primary + replicates (plan SS4 leg 1 step 1)
LAYERS_SMOKE = (19,)  # plan SS4 Smoke run: P-A --smoke = L19-only
TOP_K_PRODUCTION = 32  # per-direction reporting width (plan SS4 leg 1 step 4 top-32)
TOP_K_SMOKE = 8  # plan SS4 Smoke run: top-8 directions
N_DRAWS_PRODUCTION = 1_000  # dashboard-unit null-floor draws (plan SS4 leg 1 step 4)
N_DRAWS_SMOKE = 100  # plan SS4 Smoke run: 100 bootstrap draws
MASS_KERNEL = 0.99  # tau_kernel = 99% sigma^2-mass rank (#1768 convention)
MASS_K90 = 0.90  # the #1774 k90 twin
COPIED_GAIN_LO, COPIED_GAIN_HI = 0.8, 1.25  # plan SS4 leg 1 step 3 (verbatim)
ALIGN_HI = 0.8  # |c| floor for copied/damped
ALIGN_LO = 0.2  # |c| ceiling for transcoded
TRANSCODED_GAIN_FLOOR = 0.8  # = the copied-class gain floor (C6: tau_read retired)
ALPHA_RESIDUAL_KS = (1, 8, 32, 128)  # plan SS4 leg 1 step 5
REGIME_VERSION = 1  # bump on any output-affecting logic change (resume key member)

# ── leg 8 steps 3+4 (monitor geometry + certificates) ─────────────────────────────
# #779 monitoring r_B set: HF data repo, full-revision pin (plan SS10: "@ 037fcbb2
# (verified)"; schema probed from the real evil.pt blob 2026-08-24 — keys
# ['counts', 'layers', 'metadata', 'r_b', 'smoke', 'trait'], r_b (28, 3584) fp32).
RB_HF_PREFIX = "issue779_monitoring/r_b"
RB_HF_REVISION = "037fcbb210bc52c459959b0746cc268fe08bae96"
RB_TRAITS = ("evil", "sycophancy", "hallucination")  # issue1482_early_layer.py order
# Certificate (iii) ridge probe: validation-split lambda selection (NEVER GCV —
# CLAUDE.md GCV ban context) over the plan-C4 widened grid, generating params only
# (machine-stable resume keys, #1336).
CERT_LAMBDA_GRID = ("logspace", -5.0, 8.0, 27)
CERT_GRID_WIDEN_MAX = 3  # widen-on-edge rounds (2 decades/edge each); residual edge REPORTED
CERT_ROWS_LAYER = 19  # the only layer with a P-B row store (X19/Y19)
CERT_CHUNK_DEFAULT = 65_536  # moment-accumulation chunk (issue2569_rowbattery.py parity)
# Binding caveats — plan SS4 leg 8 step 4 requires BOTH verbatim in every output
# artifact of these phases.
CAVEAT_ACTIVATION = (
    "activation-space perturbations are not established to correspond to realizable "
    "text perturbations — this ships as sensitivity analysis and decision-geometry "
    "characterization, NEVER a security guarantee"
)
CAVEAT_MAP_LEVEL = (
    'all "the map cannot distinguish" claims are map-level (#1776), and the '
    "kernel-pair validation is precisely the test of the stronger reading."
)
CERT_FORMULAS = {
    "direct_projection": "s(v_C) = r_hat . v_C ; gradient = r_hat",
    "mapped_read": "s(v_C) = r_hat . (v_C @ A + b) ; gradient = A @ r_hat (B1 row action)",
    "fitted_probe": "s(v_C) = w . (v_C - x_mean) + t_mean ; gradient = w",
    "worst_case": "max_{|dv|<=eps} |s(v_C+dv) - s(v_C)| = eps * |gradient| (Cauchy-Schwarz; "
    "exact for a fixed linear read, SINGLE application of the map only)",
    "eps_to_move_one_heldout_sd": "std_heldout(s) / |gradient|  (context-space L2 units)",
    "signal_to_sensitivity": "corr_heldout(s, t) * std_heldout(s) / |gradient| "
    "(target-signal movement per unit context-space perturbation budget)",
}


# ── small writers (atomic; provenance-stamped) ────────────────────────────────────


def _atomic_torch_save(obj: dict, path: Path) -> None:
    """torch.save through atomic_replace (write-tmp + os.replace; same dir)."""
    with atomic_replace(path) as tmp:
        torch.save(obj, tmp)


def _write_json(path: Path, obj: dict, *, phase: str) -> None:
    """Atomic JSON write with reproducibility metadata (git provenance + dirty
    flag + card phase identity, per code-style.md SS Reproducibility metadata)."""
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    obj = dict(obj)
    md = C.reproducibility_metadata()
    md.update(as_metadata_dict(git_provenance(), phase=phase))
    obj.setdefault("metadata", md)
    C.write_json_atomic(path, obj)


def _sentinel(name: str, note: str, extra: dict | None = None) -> None:
    """Non-blocking phase sentinel (poller-parseable; never kills the run on OSError)."""
    payload = {"blocks_pipeline": False}
    if extra:
        payload.update(extra)
    try:
        C.write_sentinel(f"phase-{name}", note, task_id=TASK_ID, extra=payload)
    except OSError as exc:  # sentinel is telemetry, not a result artifact
        logger.warning("[%s] sentinel write failed (non-blocking): %s", name, exc)


def _headroom(out_root: Path, need_gb: float, phase: str) -> None:
    """Fail-loud disk headroom at the mount the out-root RESOLVES to (#1333)."""
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    assert_out_root_headroom(out_root, need_gb, phase=phase)


# ── pure computation core (unit-tested on tiny synthetic matrices) ────────────────


def full_svd_row(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full fp64 SVD of A oriented for the ROW action.

    Returns ``(read_input_u, sigma, write_output_v)``: columns of
    ``read_input_u`` are the INPUT/read directions u_i (LEFT singular
    vectors), columns of ``write_output_v`` are the OUTPUT/write directions
    v_i (RIGHT singular vectors), sigma descending, and
    ``u_i @ A == sigma_i v_i`` (the B1 row identity). Dense LAPACK — keep
    d=3584 calls on the pod, never in VM test paths.
    """
    A64 = np.asarray(A, dtype=np.float64)
    u, s, vt = np.linalg.svd(A64)
    return u, s, vt.T


def self_alignment(read_input_u: np.ndarray, write_output_v: np.ndarray) -> np.ndarray:
    """Signed self-alignment ``c_i = cos(v_i, u_i)`` per singular triplet (fp64).

    Algebraically equals ``cos(u_i @ A, u_i)`` since ``u_i @ A = sigma_i v_i``
    with sigma_i > 0 — the SCALAR is transpose-invariant (module docstring);
    invariant under the SVD's joint sign flips (u_i, v_i) -> (-u_i, -v_i).
    """
    u = np.asarray(read_input_u, dtype=np.float64)
    v = np.asarray(write_output_v, dtype=np.float64)
    num = np.einsum("ij,ij->j", u, v)
    den = np.linalg.norm(u, axis=0) * np.linalg.norm(v, axis=0)
    return num / den


def eigen_summary(A: np.ndarray, n_top: int) -> dict:
    """Full non-symmetric eigendecomposition summary, |lambda|-sorted (row action).

    Wraps ``OP.eigen_read_write`` (right eigenvectors from ``scipy.linalg.eig``,
    LEFT eigenvectors = rows of ``inv(V)`` — biorthonormal by construction;
    see the module docstring's normalization convention). Returns a dict with
    ``lam`` (complex128 [d], |lambda| desc), ``read_right_top`` (complex128
    [d, K]), ``write_left_top`` (complex128 [K, d]), ``rho``, ``kappa_v``
    (2-norm cond of the right-eigenvector matrix), ``biortho_max_err``
    (max |W V - I|), ``n_complex_pairs``, ``n_real_eigs``. COST at d=3584:
    dense eig + inv + complex GEMM + complex-SVD cond — pod territory.
    """
    pairs = OP.eigen_read_write(A)
    lam = pairs.lam
    d = lam.size
    order = np.argsort(-np.abs(lam), kind="stable")
    k = min(int(n_top), d)
    top = order[:k]
    resid = pairs.write_left_rows @ pairs.read_right_v - np.eye(d)
    tol = 1e-12 * max(float(np.abs(lam).max()), 1.0)
    n_complex = int(np.sum(np.abs(lam.imag) > tol))
    return {
        "lam": lam[order],
        "read_right_top": pairs.read_right_v[:, top],
        "write_left_top": pairs.write_left_rows[top, :],
        "rho": float(pairs.spectral_radius),
        "kappa_v": float(np.linalg.cond(pairs.read_right_v)),
        "biortho_max_err": float(np.abs(resid).max()),
        "n_complex_pairs": n_complex // 2,
        "n_real_eigs": d - n_complex,
    }


def classify_anatomy(sigma: np.ndarray, c: np.ndarray, tau_kernel: float) -> np.ndarray:
    """Per-direction anatomy labels (plan SS4 leg 1 step 3; first-match precedence).

    Order: ignored -> copied -> damped -> transcoded -> rotated_scaled (module
    docstring). Operates on the INPUT directions u_i via their (sigma_i, c_i);
    returns an object array of label strings, one per direction.
    """
    s = np.asarray(sigma, dtype=np.float64)
    abs_c = np.abs(np.asarray(c, dtype=np.float64))
    conds = [
        s < tau_kernel,
        (s >= COPIED_GAIN_LO) & (s <= COPIED_GAIN_HI) & (abs_c >= ALIGN_HI),
        (s >= tau_kernel) & (s < COPIED_GAIN_LO) & (abs_c >= ALIGN_HI),
        (s >= TRANSCODED_GAIN_FLOOR) & (abs_c < ALIGN_LO),
    ]
    choices = ["ignored", "copied", "damped", "transcoded"]
    return np.select(conds, choices, default="rotated_scaled")


def anatomy_stats(sigma: np.ndarray, c: np.ndarray) -> dict:
    """Anatomy classification + sigma^2-weighted mass fractions and counts.

    Computes tau_kernel (99% sigma^2-mass, #1768 convention) and the k90 twin
    via ``OP.tau_kernel_threshold``, classifies every direction, and returns
    {tau_kernel, k99, k90, tau_k90, labels, classes: {label: {count,
    frac_count, sigma2_mass_frac}}} — P-A mass is sigma^2-weighted; the
    DATA-weighted variant is P-B's (never computed here).
    """
    s = np.asarray(sigma, dtype=np.float64)
    tau, k99 = OP.tau_kernel_threshold(s, mass=MASS_KERNEL)
    tau90, k90 = OP.tau_kernel_threshold(s, mass=MASS_K90)
    labels = classify_anatomy(s, c, tau)
    total_mass = float(np.sum(s**2))
    classes = {}
    for lab in ("ignored", "copied", "damped", "transcoded", "rotated_scaled"):
        m = labels == lab
        classes[lab] = {
            "count": int(m.sum()),
            "frac_count": float(m.sum() / s.size),
            "sigma2_mass_frac": float(np.sum(s[m] ** 2) / total_mass),
        }
    return {
        "tau_kernel": float(tau),
        "k99": int(k99),
        "k90": int(k90),
        "tau_k90": float(tau90),
        "labels": labels,
        "classes": classes,
    }


def alpha_low_rank_stats(A: np.ndarray, ks: tuple[int, ...] = ALPHA_RESIDUAL_KS) -> dict:
    """The ``A ~= alpha*I + low-rank`` test (plan SS4 leg 1 step 5).

    alpha = tr(A)/d; residual R = A - alpha*I; returns both full singular
    spectra (values-only SVDs) + the top-k variance-explained fractions of R
    for k in ``ks`` (each min'd against d). Values-only dgesdd — far cheaper
    than the factor phase's full-vector SVD.
    """
    A64 = np.asarray(A, dtype=np.float64)
    d = A64.shape[0]
    alpha = float(np.trace(A64) / d)
    R = A64 - alpha * np.eye(d)
    s_a = np.linalg.svd(A64, compute_uv=False)
    s_r = np.linalg.svd(R, compute_uv=False)
    mass = np.cumsum(s_r**2) / np.sum(s_r**2)
    var_explained = {str(k): float(mass[min(k, d) - 1]) for k in ks}
    return {
        "alpha": alpha,
        "d": d,
        "fro_A": float(np.linalg.norm(A64)),
        "fro_residual": float(np.linalg.norm(R)),
        "sigma_A": s_a,
        "sigma_residual": s_r,
        "var_explained_topk": var_explained,
    }


def fixed_point_stats(A: np.ndarray, b: np.ndarray, rho: float) -> tuple[np.ndarray, dict]:
    """Row-convention fixed point ``x* (I - A) = b`` + the rho(A) guard branch.

    ``rho`` MUST be this map's own fresh spectral radius (the factor phase's
    eig read — never #1774's prior). rho >= 1 DROPS the iterated-map reading
    (``iterated_map_reading = False``) and x* is reported as the affine-
    consistency point only. A singular ``I - A`` raises LinAlgError — the
    crash IS the signal (no fixed point exists to report).
    """
    x_star = OP.fixed_point(A, b)
    b64 = np.asarray(b, dtype=np.float64)
    resid = float(np.linalg.norm(x_star - x_star @ np.asarray(A, dtype=np.float64) - b64))
    return x_star, {
        "rho": float(rho),
        "iterated_map_reading": bool(rho < 1.0),
        "x_star_norm": float(np.linalg.norm(x_star)),
        "b_norm": float(np.linalg.norm(b64)),
        "residual_rel": resid / float(np.linalg.norm(b64)),
    }


def effective_kernel_stats(read_input_u: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, dict]:
    """Leg-8 step 1: the effective kernel = bottom LEFT-singular subspace.

    INPUT directions with sigma_i < tau_kernel (strict — a boundary value
    sigma == tau stays OUT of the kernel; via ``OP.kernel_read_directions``).
    Ridge has NO exact kernel: the returned ``claims_phrasing`` states the
    sanctioned "reads at < X% of typical (median) gain" form, and no key here
    says "null space". Returns ``(basis [d, m], stats)``.
    """
    s = np.asarray(sigma, dtype=np.float64)
    tau, k99 = OP.tau_kernel_threshold(s, mass=MASS_KERNEL)
    _tau90, k90 = OP.tau_kernel_threshold(s, mass=MASS_K90)
    basis = OP.kernel_read_directions(read_input_u, s, tau)
    sig_k = s[s < tau]
    med = float(np.median(s))
    pct = 100.0 * tau / med if med > 0 else float("inf")
    pct_max = 100.0 * tau / s[0] if s[0] > 0 else float("inf")
    stats = {
        "tau_kernel": float(tau),
        "k99": int(k99),
        "k90": int(k90),
        "kernel_dim": int(basis.shape[1]),
        "tau_over_median_gain": float(tau / med) if med > 0 else None,
        "tau_over_max_gain": float(tau / s[0]) if s[0] > 0 else None,
        "pct_of_typical_gain": pct,
        "sigma_kernel_max": float(sig_k.max()) if sig_k.size else None,
        "sigma_kernel_mean": float(sig_k.mean()) if sig_k.size else None,
        "claims_phrasing": (
            f"effective kernel = the {int(basis.shape[1])} input directions the map reads at "
            f"< {pct:.3f}% of typical (median) gain and < {pct_max:.3f}% of peak gain "
            f"(sigma_i < tau_kernel = {tau:.6g}); ridge has no exact kernel — this is a "
            "low-gain read subspace, never an exact null space"
        ),
    }
    return basis, stats


# ── pure computation core, leg 8 steps 3+4 (unit-tested on tiny synthetics) ───────


def monitor_flip_geometry(A: np.ndarray, b: np.ndarray, r_hat: np.ndarray) -> dict:
    """Leg-8 step 3 flip geometry for one unit-normalized monitor readout.

    B1 ROW action: the mapped read is ``s(v) = r_hat . (v @ A + b)
    = v . (A @ r_hat) + r_hat . b``, so the context-space gradient is
    ``g = A @ r_hat`` and the minimal-norm context change moving the read by
    one unit lies ALONG g with norm ``1/|g|``. Returns the gradient (fp64)
    plus the scalar geometry; asserts a non-degenerate gradient.
    """
    g = OP.monitor_gradient(A, r_hat)
    gn = float(np.linalg.norm(g))
    assert gn > 0.0, "degenerate monitor: A @ r_hat == 0"
    b64 = np.asarray(b, dtype=np.float64)
    r64 = np.asarray(r_hat, dtype=np.float64)
    return {
        "gradient": g,
        "grad_norm": gn,
        "min_context_change_per_unit_read": 1.0 / gn,
        "read_at_zero_context": float(b64 @ r64),
    }


def least_norm_preimage(
    read_input_u: np.ndarray,
    sigma: np.ndarray,
    write_output_v: np.ndarray,
    y: np.ndarray,
    tau: float,
) -> dict:
    """Least-norm pre-image of the target output level ``y`` under the ROW action.

    With ``A = sum_i sigma_i u_i v_i^T`` (row action ``x @ A = sum_i sigma_i
    (x . u_i) v_i``; B1 orientation — u = INPUT/read, v = OUTPUT/write), the
    pseudoinverse pre-image is ``v = y @ A^+ = sum_i sigma_i^-1 (y . v_i) u_i``.
    Returns BOTH companions: the EFFECTIVE-RANK form (retain ``sigma_i >= tau``
    — the strict complement of the ``sigma < tau`` kernel of
    ``OP.kernel_read_directions``) and the FULL-pinv form (every
    ``sigma_i > 0``). Coset ambiguity is quantified, never dropped: any
    pre-image = this particular solution + an arbitrary effective-kernel
    component (``kernel_dim`` free directions). Mass fractions are of
    ``|y|^2``; ``achieved_level_fraction_algebra`` is exact when the v basis
    is complete (full SVD of a square A).
    """
    u = np.asarray(read_input_u, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    v = np.asarray(write_output_v, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    y_norm2 = float(y64 @ y64)
    assert y_norm2 > 0.0, "degenerate target level y == 0"
    proj = v.T @ y64  # components y . v_i in the write basis
    retained = s >= float(tau)
    positive = s > 0.0
    coeff_ret = np.where(retained, proj / np.where(retained, s, 1.0), 0.0)
    coeff_full = np.where(positive, proj / np.where(positive, s, 1.0), 0.0)
    v_ln = u @ coeff_ret
    v_full = u @ coeff_full
    below = proj[~retained]
    mass_below = float(below @ below) / y_norm2
    return {
        "preimage": v_ln,
        "preimage_fullpinv": v_full,
        "n_retained": int(retained.sum()),
        "kernel_dim": int((~retained).sum()),
        "target_mass_below_tau_frac": mass_below,
        "achieved_level_fraction_algebra": 1.0 - mass_below,
        "preimage_norm": float(np.linalg.norm(v_ln)),
        "preimage_fullpinv_norm": float(np.linalg.norm(v_full)),
    }


def select_lambda_with_widening(
    val_r2_fn, grid_params: tuple = CERT_LAMBDA_GRID, widen_max: int = CERT_GRID_WIDEN_MAX
) -> dict:
    """Validation-split lambda selection with widen-on-edge (plan C4; NEVER GCV).

    ``val_r2_fn(lams) -> np.ndarray`` returns the validation R^2 per lambda.
    The grid travels as GENERATING PARAMETERS ``("logspace", lo, hi, num)`` —
    never a materialized-float-array hash (machine-stable keys, #1336). An
    argmax on a grid EDGE widens that edge by 2 decades (per-decade density
    preserved) up to ``widen_max`` rounds; a residual edge after exhaustion
    is REPORTED via ``lambda_grid_edge``, never silently accepted.
    """
    kind, lo, hi, num = grid_params
    assert kind == "logspace", grid_params
    lo, hi, num = float(lo), float(hi), int(num)
    rounds = 0
    while True:
        lams = np.logspace(lo, hi, num)
        r2 = np.asarray(val_r2_fn(lams), dtype=np.float64)
        assert r2.shape == lams.shape, (r2.shape, lams.shape)
        j = int(np.argmax(r2))
        at_lo, at_hi = j == 0, j == lams.size - 1
        if not (at_lo or at_hi) or rounds >= widen_max:
            return {
                "selected_lambda": float(lams[j]),
                "val_r2": float(r2[j]),
                "lambda_grid_edge": "low" if at_lo else ("high" if at_hi else "none"),
                "widen_rounds_used": rounds,
                "grid": {"kind": kind, "lo": lo, "hi": hi, "num": num},
            }
        per_decade = (num - 1) / max(hi - lo, 1e-9)
        if at_lo:
            lo -= 2.0
        else:
            hi += 2.0
        num = int(round(per_decade * (hi - lo))) + 1
        rounds += 1


def _eigh_robust(g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.linalg.eigh with the canonical cuda->CPU LinAlgError fallback.

    Exact numerical-backend swap, NO Gram jitter (#1335 convention; canonical
    impl scripts/issue825_fit_cells.py::_eigh_robust). A Gram that fails on
    CPU too is genuinely pathological input — let it raise.
    """
    try:
        return torch.linalg.eigh(g)
    except torch.linalg.LinAlgError:
        print(
            f"[eigh-robust] eigh failed on {g.device} (n={g.shape[0]}); CPU LAPACK retry",
            flush=True,
        )
        w, vv = torch.linalg.eigh(g.cpu())
        return w.to(g.device), vv.to(g.device)


def _accumulate_probe_moments(
    x_mm, y_mm, r_mat: np.ndarray, positions: np.ndarray, *, chunk: int, dev, tag: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Chunked fp64 raw-moment accumulation for the certificate ridge probe.

    ``x_mm``/``y_mm`` are (N, d) fp16 np memmaps (the P-B P1 row store);
    ``r_mat`` (d, T) stacks the unit monitor readouts; targets ``t = Y @ R``
    are formed row-chunked (never materializing Y). Returns UNCENTERED torch
    fp64 sums ``(gxx [d,d], gxt [d,T], sx [d], st_ [T], n)`` — centered by
    the caller (issue2569_rowbattery.py moment-accumulation parity).
    """
    d = int(x_mm.shape[1])
    n_traits = int(r_mat.shape[1])
    r64 = torch.as_tensor(np.asarray(r_mat, dtype=np.float64), device=dev)
    gxx = torch.zeros((d, d), dtype=torch.float64, device=dev)
    gxt = torch.zeros((d, n_traits), dtype=torch.float64, device=dev)
    sx = torch.zeros(d, dtype=torch.float64, device=dev)
    st_ = torch.zeros(n_traits, dtype=torch.float64, device=dev)
    n = int(positions.size)
    t0 = time.time()
    for k0 in range(0, n, chunk):
        pos = positions[k0 : k0 + chunk]
        xb = torch.as_tensor(np.asarray(x_mm[pos], dtype=np.float64), device=dev)
        yb = torch.as_tensor(np.asarray(y_mm[pos], dtype=np.float64), device=dev)
        tb = yb @ r64
        gxx += xb.T @ xb
        gxt += xb.T @ tb
        sx += xb.sum(dim=0)
        st_ += tb.sum(dim=0)
        logger.info(
            "[%s] moments rows %d..%d/%d elapsed=%.1fs",
            tag,
            k0,
            min(k0 + chunk, n),
            n,
            time.time() - t0,
        )
    return gxx, gxt, sx, st_, n


def _heldout_monitor_stats(scores: np.ndarray, target: np.ndarray, grad_norm: float) -> dict:
    """Held-out normalization for one monitor: std, target corr, and the two
    scale-invariant certificate reads (eps-to-move-one-SD; signal-to-sensitivity,
    both in context-space L2 units)."""
    sd = float(np.std(scores, ddof=1))
    corr = (
        float(np.corrcoef(scores, target)[0, 1])
        if sd > 0.0 and float(np.std(target, ddof=1)) > 0.0
        else 0.0
    )
    return {
        "std": sd,
        "corr_with_target": corr,
        "eps_to_move_one_heldout_sd": sd / grad_norm,
        "signal_to_sensitivity": corr * sd / grad_norm,
    }


def certificate_rows_core(
    x_mm,
    y_mm,
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    test_pos: np.ndarray,
    r_mat: np.ndarray,
    trait_names: tuple[str, ...],
    a_mat: np.ndarray,
    b_vec: np.ndarray,
    *,
    chunk: int = CERT_CHUNK_DEFAULT,
    dev: str = "cpu",
    grid_params: tuple = CERT_LAMBDA_GRID,
    widen_max: int = CERT_GRID_WIDEN_MAX,
) -> dict:
    """Certificate (iii) fitted 1-D context probes + held-out normalization.

    Fits, per trait, a train-centered raw-feature ridge probe ``w`` from
    context states v_C to the trait target ``t = r_hat . v_A`` (= ``Y @
    r_hat``), lambda selected on the pinned VALIDATION split (never GCV) with
    widen-on-edge, then evaluates ALL THREE monitor forms on the held-out
    TEST split. HARD ESTIMATOR-VALIDITY GATE: ``n_train < d`` is REFUSED
    (RuntimeError) — every held-out R^2 in that regime is estimator-degenerate
    (#1701). Returns ``{"probes", "heldout", "rows", "w", "x_mean", "t_mean"}``.
    """
    d = int(x_mm.shape[1])
    n_train = int(train_pos.size)
    if n_train < d:
        raise RuntimeError(
            f"certificate probe REFUSED: n_train={n_train} < d={d} — estimator-degenerate "
            "regime (every held-out R^2 is a ceiling artifact, #1701); attach a row store "
            "with n_train >= d or leave the probe leg deferred"
        )
    assert val_pos.size > 1 and test_pos.size > 1, (val_pos.size, test_pos.size)
    dev_t = torch.device(dev)
    r64 = np.asarray(r_mat, dtype=np.float64)
    gxx, gxt, sx, st_, n = _accumulate_probe_moments(
        x_mm, y_mm, r64, train_pos, chunk=chunk, dev=dev_t, tag="certificates"
    )
    sxx_c = gxx - torch.outer(sx, sx) / n
    sxt_c = gxt - torch.outer(sx, st_) / n
    evals_t, vecs_t = _eigh_robust(sxx_c / n)
    evals = evals_t.cpu().numpy()
    vecs = vecs_t.cpu().numpy()
    bt = (vecs_t.T @ (sxt_c / n)).cpu().numpy()  # (d, T) in the eigenbasis
    x_mean = (sx / n).cpu().numpy()
    t_mean = (st_ / n).cpu().numpy()
    xv_c = np.asarray(x_mm[val_pos], dtype=np.float64) - x_mean
    tv = np.asarray(y_mm[val_pos], dtype=np.float64) @ r64
    xt_raw = np.asarray(x_mm[test_pos], dtype=np.float64)
    xt_c = xt_raw - x_mean
    tt = np.asarray(y_mm[test_pos], dtype=np.float64) @ r64
    vv_val = xv_c @ vecs  # precompute: validation rows in the eigenbasis

    def _r2_cols(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Columnwise R^2 of ``pred`` (n, k) against ``target`` (n,)."""
        denom = float(((target - target.mean()) ** 2).sum())
        assert denom > 0.0, "degenerate validation target (zero variance)"
        return 1.0 - ((target[:, None] - pred) ** 2).sum(axis=0) / denom

    probes: dict = {}
    heldout: dict = {}
    w_cols = []
    b64 = np.asarray(b_vec, dtype=np.float64)
    for j, trait in enumerate(trait_names):

        def _val_r2(lams: np.ndarray, j: int = j) -> np.ndarray:
            coef = bt[:, j][:, None] / (evals[:, None] + lams[None, :])
            return _r2_cols(vv_val @ coef + t_mean[j], tv[:, j])

        sel = select_lambda_with_widening(_val_r2, grid_params, widen_max)
        w_j = vecs @ (bt[:, j] / (evals + sel["selected_lambda"]))
        w_cols.append(w_j)
        s_probe = xt_c @ w_j + t_mean[j]
        heldout_r2 = float(_r2_cols(s_probe[:, None], tt[:, j])[0])
        wn = float(np.linalg.norm(w_j))
        probes[trait] = {
            "status": "computed",
            "selected_lambda": sel["selected_lambda"],
            "selector": "validation-split R^2 on the pinned val split (GCV REFUSED)",
            "lambda_grid_edge": sel["lambda_grid_edge"],
            "widen_rounds_used": sel["widen_rounds_used"],
            "realized_grid": sel["grid"],
            "val_r2": sel["val_r2"],
            "heldout_r2": heldout_r2,
            "grad_norm": wn,
            "n_train": n_train,
            "d": d,
            "n_val": int(val_pos.size),
            "n_test": int(test_pos.size),
        }
        g_mapped = OP.monitor_gradient(a_mat, r64[:, j])
        heldout[trait] = {
            "direct_projection": _heldout_monitor_stats(xt_raw @ r64[:, j], tt[:, j], 1.0),
            "mapped_read": _heldout_monitor_stats(
                xt_raw @ g_mapped + float(b64 @ r64[:, j]),
                tt[:, j],
                float(np.linalg.norm(g_mapped)),
            ),
            "fitted_probe": _heldout_monitor_stats(s_probe, tt[:, j], wn),
        }
    return {
        "probes": probes,
        "heldout": heldout,
        "rows": {
            "n_train": n_train,
            "d": d,
            "n_val": int(val_pos.size),
            "n_test": int(test_pos.size),
            "ridge_convention": "(Sxx_c/n + lambda I) w = Sxt_c/n; train-centered raw "
            "features; probe s(v) = w . (v - x_mean) + t_mean",
            "lambda_grid": list(grid_params),
        },
        "w": np.stack(w_cols, axis=1),
        "x_mean": x_mean,
        "t_mean": t_mean,
    }


# ── driver plumbing (layers, regime keys, resume) ─────────────────────────────────


def _layers(args) -> tuple[int, ...]:
    """Resolve the layer list: explicit --layers > smoke L19-only > production."""
    if args.layers:
        out = tuple(int(x) for x in str(args.layers).split(","))
        bad = [x for x in out if x not in OP.N1M_LAYERS]
        assert not bad, f"--layers {bad} not banked (banked: {list(OP.N1M_LAYERS)})"
        return out
    return LAYERS_SMOKE if args.smoke else LAYERS_PRODUCTION


def _top_k(args) -> int:
    """Per-direction reporting width: explicit --top-k > smoke 8 > production 32."""
    return int(args.top_k) if args.top_k > 0 else (TOP_K_SMOKE if args.smoke else TOP_K_PRODUCTION)


def _n_draws(args) -> int:
    """Null-floor draw budget (recorded in regime; consumed by the dashboard unit)."""
    return (
        int(args.n_draws)
        if args.n_draws > 0
        else (N_DRAWS_SMOKE if args.smoke else N_DRAWS_PRODUCTION)
    )


def _regime(args, layer: int) -> dict:
    """Resume/regime key: GENERATING PARAMETERS only (never recomputed floats).

    Every output-affecting knob is a member (#722 r3); literals are stable
    constants, so the key is machine-independent (gotchas: never hash a
    recomputed float array).
    """
    return {
        "regime_version": REGIME_VERSION,
        "layer": int(layer),
        "smoke": bool(args.smoke),
        "top_k": _top_k(args),
        "n_draws": _n_draws(args),
        "mass_kernel": MASS_KERNEL,
        "mass_k90": MASS_K90,
        "class_thresholds": {
            "copied_gain": [COPIED_GAIN_LO, COPIED_GAIN_HI],
            "align_hi": ALIGN_HI,
            "align_lo": ALIGN_LO,
            "transcoded_gain_floor": TRANSCODED_GAIN_FLOOR,
        },
        "alpha_residual_ks": list(ALPHA_RESIDUAL_KS),
    }


def _unit_done(json_path: Path, regime: dict, fresh: bool) -> bool:
    """Resume predicate: unit complete iff its JSON exists with an EQUAL regime."""
    if fresh or not json_path.exists():
        return False
    try:
        prior = json.loads(json_path.read_text()).get("regime")
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        # encoding-corrupt / truncated / unreadable prior JSON => recompute the unit
        return False
    return prior == json.loads(json.dumps(regime))


def _leg1(args) -> Path:
    """leg1/ output dir under the out-root (created on demand)."""
    p = args.out_root / "leg1"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _leg8(args) -> Path:
    """leg8/ output dir under the out-root (created on demand)."""
    p = args.out_root / "leg8"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_payload(args, layer: int) -> OP.MapPayload:
    """Load + validate the banked map for ``layer`` (fail-loud vendored loader)."""
    return OP.load_banked_map(layer, root=args.map_root)


def _load_factor(args, layer: int) -> dict:
    """Load the factor phase's .pt for ``layer`` (fail-loud when absent/stale).

    ``weights_only=False`` is deliberate: a self-produced project artifact
    written by this driver's own factor phase (#1900 convention).
    """
    path = _leg1(args) / f"factor_L{layer}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} absent — run `--phase factor` first (downstream phases consume it)"
        )
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("regime") != _regime(args, layer):
        raise RuntimeError(
            f"{path}: stale factor artifact (regime mismatch) — re-run `--phase factor` "
            f"(or pass --fresh) before downstream phases"
        )
    return obj


# ── leg 8 steps 3+4 plumbing (r_B staging + phase-specific regime keys) ───────────


def _stage_rb(args) -> dict[str, Path]:
    """Stage the #779 monitoring r_B blobs and return ``{trait: path}``.

    Local ``--rb-dir`` first (tests / pre-staged pods); else HF data repo at
    the FULL pinned revision (plan SS10 line: `issue779_monitoring/r_b/...
    @ 037fcbb2 (verified)`), via the retried/atomic/idempotent
    ``hub.stage_hub_file``. Fail-loud: a missing trait blob raises — never a
    substituted or fabricated readout.
    """
    out: dict[str, Path] = {}
    if args.rb_dir is not None:
        for trait in RB_TRAITS:
            p = Path(args.rb_dir) / f"{trait}.pt"
            if not p.exists():
                raise FileNotFoundError(f"--rb-dir given but {p} is absent")
            out[trait] = p
        return out
    from explore_persona_space.orchestrate import hub

    dest_dir = args.out_root / "r_b"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for trait in RB_TRAITS:
        out[trait] = hub.stage_hub_file(
            C.HF_DATA_REPO,
            f"{RB_HF_PREFIX}/{trait}.pt",
            dest_dir / f"{trait}.pt",
            repo_type="dataset",
            revision=RB_HF_REVISION,
        )
    return out


def _rb_layer_vector(path: Path, trait: str, layer: int, d: int) -> np.ndarray:
    """Load one r_B blob and return the UNIT-NORMALIZED fp64 layer-``layer`` row.

    Schema asserts mirror the OBSERVED artifact (probed real blob
    ``issue779_monitoring/r_b/evil.pt @ 037fcbb2``: keys counts/layers/
    metadata/r_b/smoke/trait; ``r_b`` (28, 3584) fp32) + the
    issue1482_early_layer.py consumer contract (trait match, ``smoke is
    False``, per-layer indexability). ``weights_only=False`` is deliberate:
    a revision-pinned project-produced artifact (#1900 convention).
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload.get("trait") == trait, (str(path), payload.get("trait"), trait)
    assert payload.get("smoke") is False, f"{path}: smoke r_B blob refused"
    arr = payload["r_b"]
    assert arr.ndim == 2 and int(arr.shape[1]) == d, (tuple(arr.shape), d)
    layers = list(payload.get("layers", range(int(arr.shape[0]))))
    assert layers[layer] == layer, (str(path), layer, layers[:4])
    r = np.asarray(arr[layer], dtype=np.float64)
    rn = float(np.linalg.norm(r))
    assert rn > 0.0, f"{path}: zero r_B row at layer {layer}"
    return r / rn


def _rb_source(args) -> str:
    """Regime token for where r_B came from (generating parameter, not a hash)."""
    return "local-dir" if args.rb_dir is not None else f"hf:{RB_HF_REVISION}"


def _mg_regime(args, layer: int) -> dict:
    """monitor-geometry resume key: base regime + this phase's generating params
    under a SUB-KEY (the shared ``_regime`` stays untouched — ``_load_factor``
    compares against it, so extending it in place would stale every banked
    factor artifact)."""
    return {
        **_regime(args, layer),
        "monitor_geometry": {"rb_source": _rb_source(args), "traits": list(RB_TRAITS)},
    }


def _cert_regime(args, layer: int) -> dict:
    """certificates resume key: base regime + probe generating params.

    ``rows_attached`` is a member so a later rows-attached re-run RECOMPUTES
    the unit that previously recorded a deferral (never a stale skip);
    ``cert_chunk``/``device`` are members because fp64 accumulation order is
    output-affecting in last bits.
    """
    rows_attached = bool(args.rows_dir) and int(layer) == CERT_ROWS_LAYER
    return {
        **_regime(args, layer),
        "certificates": {
            "rb_source": _rb_source(args),
            "traits": list(RB_TRAITS),
            "lambda_grid": list(CERT_LAMBDA_GRID),
            "grid_widen_max": CERT_GRID_WIDEN_MAX,
            "rows_attached": rows_attached,
            "cert_chunk": int(args.cert_chunk),
            "device": str(args.device),
        },
    }


def _load_rows_store(args) -> tuple:
    """Open the P-B P1 assemble dir (``--rows-dir``) and derive the pinned splits.

    Contract = the LANDED issue2569_rowbattery.py P1 outputs: ``X19.fp16.npy``
    / ``Y19.fp16.npy`` ((N, 3584) fp16 np memmaps), ``rows_present.npy``
    (sorted int64), ``split_meta.json``. Splits re-derive through the pinned
    committed-split helpers (``T24._committed_split`` +
    ``T24._assert_pinned_valtest``); the train pool = rows_present minus
    val/test, positions via ``RB._positions_in_present`` — REUSED, never a
    re-invented schema. Returns ``(x_mm, y_mm, train_pos, val_pos, test_pos)``.
    """
    import issue2476_turnavg_sae as T24
    import issue2569_rowbattery as RB

    rows_dir = Path(args.rows_dir)
    need = ["X19.fp16.npy", "Y19.fp16.npy", "rows_present.npy", "split_meta.json"]
    missing = [f for f in need if not (rows_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"--rows-dir {rows_dir} missing required files: {missing}")
    x_mm = np.load(rows_dir / "X19.fp16.npy", mmap_mode="r")
    y_mm = np.load(rows_dir / "Y19.fp16.npy", mmap_mode="r")
    rows_present = np.load(rows_dir / "rows_present.npy")
    assert x_mm.shape == y_mm.shape and x_mm.ndim == 2, (x_mm.shape, y_mm.shape)
    assert int(x_mm.shape[0]) == int(rows_present.size), (x_mm.shape, rows_present.size)
    committed = T24._committed_split()
    _r1_train, val_ids, test_ids = T24._assert_pinned_valtest(committed)
    pool_ids = np.setdiff1d(rows_present, np.union1d(val_ids, test_ids))
    train_pos = np.searchsorted(rows_present, pool_ids)
    val_pos = RB._positions_in_present(rows_present, val_ids, "val")
    test_pos = RB._positions_in_present(rows_present, test_ids, "test")
    return x_mm, y_mm, train_pos, val_pos, test_pos


# ── phases ────────────────────────────────────────────────────────────────────────


def phase_factor(args) -> None:
    """Full SVD + full non-symmetric eig per layer -> leg1/factor_L<L>.pt/.json."""
    _headroom(args.out_root, 1.0 if args.smoke else 2.0, "pa-factor")
    layers = _layers(args)
    for i, layer in enumerate(layers, 1):
        t0 = time.time()
        regime = _regime(args, layer)
        jpath = _leg1(args) / f"factor_L{layer}.json"
        if _unit_done(jpath, regime, args.fresh):
            logger.info("[factor] unit %d/%d L%d SKIP (done)", i, len(layers), layer)
            continue
        payload = _load_payload(args, layer)
        A, _b = OP.row_operator(payload)
        d = payload.d
        k = _top_k(args)
        u, s, v = full_svd_row(A)
        c = self_alignment(u, v)
        # B1 row-identity spot check on the FULL SVD's top-8 (fp64)
        kk = min(8, d - 2)
        row_err = np.linalg.norm(u[:, :kk].T @ A - s[:kk, None] * v[:, :kk].T, axis=1) / s[:kk]
        assert row_err.max() < 1e-6, f"[factor] L{layer} row identity breached: {row_err}"
        # cross-check the sparse svds path (the module's top_singular_triplets)
        trip = OP.top_singular_triplets(A, k=kk)
        svds_rel = float(np.max(np.abs(trip.sigma - s[:kk]) / s[:kk]))
        assert svds_rel < 1e-6, f"[factor] L{layer} svds/full-SVD sigma mismatch: {svds_rel:.3e}"
        eig = eigen_summary(A, n_top=k)
        stats = {
            "rho": eig["rho"],
            "kappa_v": eig["kappa_v"],
            "biortho_max_err": eig["biortho_max_err"],
            "n_complex_pairs": eig["n_complex_pairs"],
            "n_real_eigs": eig["n_real_eigs"],
            "svds_sigma_max_rel_diff": svds_rel,
            "d": int(d),
            "top_k": int(k),
        }
        _atomic_torch_save(
            {
                "sigma": torch.from_numpy(s),
                "self_alignment_c": torch.from_numpy(c),
                "read_input_u_fp32": torch.from_numpy(u.astype(np.float32)),
                "write_output_v_fp32": torch.from_numpy(v.astype(np.float32)),
                "eig_lambda": torch.from_numpy(eig["lam"]),
                "eig_read_right_v_top": torch.from_numpy(
                    eig["read_right_top"].astype(np.complex64)
                ),
                "eig_write_left_rows_top": torch.from_numpy(
                    eig["write_left_top"].astype(np.complex64)
                ),
                "stats": stats,
                "regime": regime,
            },
            _leg1(args) / f"factor_L{layer}.pt",
        )
        abs_lam = np.abs(eig["lam"])
        qs = [0.0, 0.25, 0.5, 0.75, 1.0]
        _write_json(
            jpath,
            {
                "regime": regime,
                "stats": stats,
                "sigma_quantiles": dict(
                    zip(map(str, qs), np.quantile(s, qs).tolist(), strict=True)
                ),
                "abs_lambda_quantiles": dict(
                    zip(map(str, qs), np.quantile(abs_lam, qs).tolist(), strict=True)
                ),
                "top_singular": [
                    {"rank": j + 1, "sigma": float(s[j]), "c": float(c[j])}
                    for j in range(min(k, d))
                ],
                "top_eigen": [
                    {
                        "rank": j + 1,
                        "abs_lambda": float(abs_lam[j]),
                        "re": float(eig["lam"][j].real),
                        "im": float(eig["lam"][j].imag),
                        "is_complex": bool(abs(eig["lam"][j].imag) > 1e-12 * max(abs_lam[0], 1.0)),
                    }
                    for j in range(min(k, d))
                ],
            },
            phase="pa-factor",
        )
        logger.info(
            "[factor] unit %d/%d L%d elapsed=%.1fs rho=%.4f kappa_v=%.3e",
            i,
            len(layers),
            layer,
            time.time() - t0,
            eig["rho"],
            eig["kappa_v"],
        )
    _sentinel("pa-factor", f"factor done layers={list(layers)}")


def phase_anatomy(args) -> None:
    """Functional anatomy on the INPUT directions -> leg1/anatomy_L<L>.json."""
    layers = _layers(args)
    for i, layer in enumerate(layers, 1):
        t0 = time.time()
        regime = _regime(args, layer)
        jpath = _leg1(args) / f"anatomy_L{layer}.json"
        if _unit_done(jpath, regime, args.fresh):
            logger.info("[anatomy] unit %d/%d L%d SKIP (done)", i, len(layers), layer)
            continue
        fac = _load_factor(args, layer)
        s = fac["sigma"].numpy()
        c = fac["self_alignment_c"].numpy()
        st = anatomy_stats(s, c)
        labels = st.pop("labels")
        k = _top_k(args)
        _write_json(
            jpath,
            {
                "regime": regime,
                **st,
                "sigma_max": float(s[0]),
                "sigma_median": float(np.median(s)),
                "labels": labels.tolist(),
                "sigma": s.tolist(),
                "c": c.tolist(),
                "top_directions": [
                    {
                        "rank": j + 1,
                        "sigma": float(s[j]),
                        "c": float(c[j]),
                        "abs_c": float(abs(c[j])),
                        "label": str(labels[j]),
                    }
                    for j in range(min(k, s.size))
                ],
                "thresholds": regime["class_thresholds"],
                "precedence": "ignored -> copied -> damped -> transcoded -> rotated_scaled "
                "(first match; tau_read RETIRED per C6)",
                "data_weighted_mass": "deferred-to-P-B (rowbattery moments: u_i^T Sigma_c u_i)",
            },
            phase="pa-anatomy",
        )
        logger.info(
            "[anatomy] unit %d/%d L%d elapsed=%.1fs tau=%.4g k99=%d k90=%d",
            i,
            len(layers),
            layer,
            time.time() - t0,
            st["tau_kernel"],
            st["k99"],
            st["k90"],
        )
    _sentinel("pa-anatomy", f"anatomy done layers={list(layers)}")


def phase_alpha_lowrank(args) -> None:
    """alpha*I + low-rank test -> leg1/alpha_lowrank_L<L>.json (values-only SVDs)."""
    layers = _layers(args)
    for i, layer in enumerate(layers, 1):
        t0 = time.time()
        regime = _regime(args, layer)
        jpath = _leg1(args) / f"alpha_lowrank_L{layer}.json"
        if _unit_done(jpath, regime, args.fresh):
            logger.info("[alpha-lowrank] unit %d/%d L%d SKIP (done)", i, len(layers), layer)
            continue
        payload = _load_payload(args, layer)
        A, _b = OP.row_operator(payload)
        st = alpha_low_rank_stats(A)
        _write_json(
            jpath,
            {
                "regime": regime,
                "alpha": st["alpha"],
                "d": st["d"],
                "fro_A": st["fro_A"],
                "fro_residual": st["fro_residual"],
                "var_explained_topk": st["var_explained_topk"],
                "sigma_A": st["sigma_A"].tolist(),
                "sigma_residual": st["sigma_residual"].tolist(),
            },
            phase="pa-alpha-lowrank",
        )
        logger.info(
            "[alpha-lowrank] unit %d/%d L%d elapsed=%.1fs alpha=%.5f top1=%.4f",
            i,
            len(layers),
            layer,
            time.time() - t0,
            st["alpha"],
            st["var_explained_topk"]["1"],
        )
    _sentinel("pa-alpha-lowrank", f"alpha-lowrank done layers={list(layers)}")


def phase_fixed_point(args) -> None:
    """Fixed point x* (I - A) = b with the fresh-rho branch -> leg1/fixed_point_L<L>."""
    layers = _layers(args)
    for i, layer in enumerate(layers, 1):
        t0 = time.time()
        regime = _regime(args, layer)
        jpath = _leg1(args) / f"fixed_point_L{layer}.json"
        if _unit_done(jpath, regime, args.fresh):
            logger.info("[fixed-point] unit %d/%d L%d SKIP (done)", i, len(layers), layer)
            continue
        payload = _load_payload(args, layer)
        A, b = OP.row_operator(payload)
        rho = float(_load_factor(args, layer)["stats"]["rho"])  # THIS map's fresh eig read
        x_star, st = fixed_point_stats(A, b, rho)
        assert st["residual_rel"] < 1e-8, f"[fixed-point] L{layer} solve residual: {st}"
        _atomic_torch_save(
            {"x_star": torch.from_numpy(x_star), "regime": regime},
            _leg1(args) / f"fixed_point_L{layer}.pt",
        )
        _write_json(
            jpath,
            {
                "regime": regime,
                **st,
                "sae_decode": "deferred-to-sae-dashboards-unit (needs the #2476 encoder)",
                "nearest_banked_answers": "deferred-to-P-B (needs rows)",
            },
            phase="pa-fixed-point",
        )
        logger.info(
            "[fixed-point] unit %d/%d L%d elapsed=%.1fs rho=%.4f iterated=%s |x*|=%.3f",
            i,
            len(layers),
            layer,
            time.time() - t0,
            st["rho"],
            st["iterated_map_reading"],
            st["x_star_norm"],
        )
    _sentinel("pa-fixed-point", f"fixed-point done layers={list(layers)}")


def phase_kernel(args) -> None:
    """Leg-8 step 1 effective kernel -> leg8/effective_kernel_L<L>.pt/.json."""
    layers = _layers(args)
    for i, layer in enumerate(layers, 1):
        t0 = time.time()
        regime = _regime(args, layer)
        jpath = _leg8(args) / f"effective_kernel_L{layer}.json"
        if _unit_done(jpath, regime, args.fresh):
            logger.info("[kernel] unit %d/%d L%d SKIP (done)", i, len(layers), layer)
            continue
        fac = _load_factor(args, layer)
        s = fac["sigma"].numpy()
        u = fac["read_input_u_fp32"].numpy().astype(np.float64)
        basis, st = effective_kernel_stats(u, s)
        _atomic_torch_save(
            {
                "kernel_basis_fp32": torch.from_numpy(basis.astype(np.float32)),
                "sigma_kernel": torch.from_numpy(s[s < st["tau_kernel"]]),
                "regime": regime,
            },
            _leg8(args) / f"effective_kernel_L{layer}.pt",
        )
        _write_json(jpath, {"regime": regime, **st}, phase="pa-kernel")
        logger.info(
            "[kernel] unit %d/%d L%d elapsed=%.1fs dim=%d tau=%.4g (%.3f%% of median gain)",
            i,
            len(layers),
            layer,
            time.time() - t0,
            st["kernel_dim"],
            st["tau_kernel"],
            st["pct_of_typical_gain"],
        )
    _sentinel("pa-kernel", f"kernel done layers={list(layers)}")


def phase_monitor_geometry(args) -> None:
    """Leg-8 step 3 monitor decision geometry -> leg8/monitor_geometry_L<L>.pt/.json.

    Per trait readout r from the #779 monitoring r_B set (unit-normalized):
    the B1 flip gradient A r, the minimal context change per unit read, and
    least-norm pre-images of a unit read level (effective-rank truncated at
    tau_kernel + full-pinv companion) — WITH the coset ambiguity + gain-ratio
    context stated in every artifact. An orientation guard re-multiplies the
    pre-image through the REAL A: a U/V letter flip (the B1 confusion class)
    collapses the achieved level and raises.
    """
    layers = _layers(args)
    rb_paths = _stage_rb(args)
    for i, layer in enumerate(layers, 1):
        t0 = time.time()
        regime = _mg_regime(args, layer)
        jpath = _leg8(args) / f"monitor_geometry_L{layer}.json"
        if _unit_done(jpath, regime, args.fresh):
            logger.info("[monitor-geometry] unit %d/%d L%d SKIP (done)", i, len(layers), layer)
            continue
        payload = _load_payload(args, layer)
        a_mat, b_vec = OP.row_operator(payload)
        fac = _load_factor(args, layer)
        s = fac["sigma"].numpy().astype(np.float64)
        u = fac["read_input_u_fp32"].numpy().astype(np.float64)
        v = fac["write_output_v_fp32"].numpy().astype(np.float64)
        tau, k99 = OP.tau_kernel_threshold(s, mass=MASS_KERNEL)
        kernel_dim = int((s < tau).sum())
        med = float(np.median(s))
        d = int(payload.d)
        gain_ratio = tau / med if med > 0 else float("inf")
        traits_pt: dict = {}
        traits_json: dict = {}
        for trait in RB_TRAITS:
            r_hat = _rb_layer_vector(rb_paths[trait], trait, layer, d)
            flip = monitor_flip_geometry(a_mat, b_vec, r_hat)
            pre = least_norm_preimage(u, s, v, r_hat, tau)
            # Orientation guard (B1): the measured achieved level through the
            # REAL A must match the algebraic 1 - mass_below (fp32-factor
            # noise tolerance 1e-2); a U/V letter flip collapses it.
            achieved_meas = float(r_hat @ (pre["preimage"] @ a_mat))
            resid = abs(achieved_meas - pre["achieved_level_fraction_algebra"])
            if resid > 1e-2:
                raise RuntimeError(
                    f"[monitor-geometry] L{layer} {trait}: orientation guard breached — "
                    f"measured achieved level {achieved_meas:.6f} vs algebraic "
                    f"{pre['achieved_level_fraction_algebra']:.6f} (resid {resid:.3e}); "
                    "U/V letter-flip suspected (B1 orientation dictionary)"
                )
            g = flip["gradient"]
            gu = u.T @ g
            below = gu[s < tau]
            gmass = float(below @ below) / float(gu @ gu)
            traits_pt[trait] = {
                "r_hat": torch.from_numpy(r_hat),
                "gradient": torch.from_numpy(g),
                "preimage_unit_level": torch.from_numpy(pre["preimage"]),
                "preimage_unit_level_fullpinv": torch.from_numpy(pre["preimage_fullpinv"]),
            }
            traits_json[trait] = {
                "grad_norm": flip["grad_norm"],
                "min_context_change_per_unit_read": flip["min_context_change_per_unit_read"],
                "read_at_zero_context": flip["read_at_zero_context"],
                "gradient_mass_below_tau_frac": gmass,
                "preimage_norm": pre["preimage_norm"],
                "preimage_fullpinv_norm": pre["preimage_fullpinv_norm"],
                "target_mass_below_tau_frac": pre["target_mass_below_tau_frac"],
                "achieved_level_fraction": achieved_meas,
                "achieved_level_fraction_algebra": pre["achieved_level_fraction_algebra"],
                "preimage_orientation_residual": resid,
                "n_retained": pre["n_retained"],
                "kernel_dim": pre["kernel_dim"],
            }
        caveats = [CAVEAT_ACTIVATION, CAVEAT_MAP_LEVEL]
        coset = (
            f"every pre-image is one particular solution only: any of the {kernel_dim} "
            f"effective-kernel directions (sigma_i < tau_kernel = {tau:.6g}, i.e. reads at "
            f"< {100.0 * gain_ratio:.1f}% of median gain) can be added while moving the "
            "mapped read by < tau per unit norm — a least-norm pre-image is never 'the' "
            "context (open concern effective-kernel-tau-above-median-gain)"
        )
        gain_note = (
            (
                f"tau_kernel={tau:.6g} EXCEEDS the median gain sigma_median={med:.6g} "
                f"(ratio {gain_ratio:.3f}): a MAJORITY of input directions sit below tau, "
                "so 'below-tau' is a RELATIVE-gain statement (low-gain read), NEVER "
                "'ignored'"
            )
            if tau > med
            else (
                f"tau_kernel={tau:.6g} <= median gain sigma_median={med:.6g} "
                f"(ratio {gain_ratio:.3f}) on this layer's map"
            )
        )
        _atomic_torch_save(
            {
                "traits": traits_pt,
                "tau_kernel": float(tau),
                "caveats": caveats,
                "regime": regime,
            },
            _leg8(args) / f"monitor_geometry_L{layer}.pt",
        )
        _write_json(
            jpath,
            {
                "regime": regime,
                "kernel_gain_context": {
                    "tau_kernel": float(tau),
                    "k99": int(k99),
                    "kernel_dim": kernel_dim,
                    "sigma_median": med,
                    "tau_over_median_gain": gain_ratio,
                    "note": gain_note,
                },
                "traits": traits_json,
                "coset_ambiguity": coset,
                "affine_note": "read_at_zero_context = r_hat . b is the affine offset — the "
                "flip DISTANCE along the gradient depends on the current context's read, "
                "not only on the geometry",
                "sae_naming": "deferred-to-SAE-dashboard-unit (leg 1 step 4 machinery; the "
                "next unit in this file names the gradient + pre-image directions)",
                "caveats": caveats,
            },
            phase="pa-monitor-geometry",
        )
        logger.info(
            "[monitor-geometry] unit %d/%d L%d elapsed=%.1fs kernel_dim=%d grad_norms=%s",
            i,
            len(layers),
            layer,
            time.time() - t0,
            kernel_dim,
            {t: round(traits_json[t]["grad_norm"], 4) for t in RB_TRAITS},
        )
    _sentinel("pa-monitor-geometry", f"monitor geometry done layers={list(layers)}")


def phase_certificates(args) -> None:
    """Leg-8 step 4 sensitivity certificates -> leg8/certificates_L<L>.json
    (+ leg8/certificates_probe_L19.pt on rows-attached runs).

    Weights-only sensitivities (gradient norms; worst case = eps * |gradient|,
    SINGLE application only — the measured rho(A) >= 1 forecloses iterated
    bounds, stated UNAVAILABLE) always compute. The fitted probe (iii) + the
    held-out normalization need the P-B P1 row store: they run ONLY at layer
    19 with ``--rows-dir`` (P-A and P-B run concurrently on different pods) —
    otherwise an EXPLICIT deferral is recorded, never a silent skip.
    """
    layers = _layers(args)
    rb_paths = _stage_rb(args)
    for i, layer in enumerate(layers, 1):
        t0 = time.time()
        regime = _cert_regime(args, layer)
        rows_attached = regime["certificates"]["rows_attached"]
        jpath = _leg8(args) / f"certificates_L{layer}.json"
        if _unit_done(jpath, regime, args.fresh):
            logger.info("[certificates] unit %d/%d L%d SKIP (done)", i, len(layers), layer)
            continue
        payload = _load_payload(args, layer)
        a_mat, b_vec = OP.row_operator(payload)
        fac = _load_factor(args, layer)
        rho = float(fac["stats"]["rho"])
        d = int(payload.d)
        r_mat = np.stack([_rb_layer_vector(rb_paths[t], t, layer, d) for t in RB_TRAITS], axis=1)
        deferral = (
            "deferred — the P-B P1 assemble dir (X19/Y19 fp16 row store) was not attached "
            "via --rows-dir (P-A and P-B run concurrently on different pods; re-run "
            "`--phase certificates --rows-dir <assemble-dir>` at layer 19 post-P-B; "
            "concern leg8-cert-heldout-needs-pb-rows)"
        )
        core: dict | None = None
        if rows_attached:
            x_mm, y_mm, train_pos, val_pos, test_pos = _load_rows_store(args)
            core = certificate_rows_core(
                x_mm,
                y_mm,
                train_pos,
                val_pos,
                test_pos,
                r_mat,
                RB_TRAITS,
                a_mat,
                b_vec,
                chunk=int(args.cert_chunk),
                dev=str(args.device),
                grid_params=CERT_LAMBDA_GRID,
                widen_max=CERT_GRID_WIDEN_MAX,
            )
            _atomic_torch_save(
                {
                    "w": torch.from_numpy(core["w"]),
                    "x_mean": torch.from_numpy(core["x_mean"]),
                    "t_mean": torch.from_numpy(core["t_mean"]),
                    "traits": list(RB_TRAITS),
                    "regime": regime,
                },
                _leg8(args) / f"certificates_probe_L{layer}.pt",
            )
        monitors: dict = {}
        for j, trait in enumerate(RB_TRAITS):
            g_mapped = OP.monitor_gradient(a_mat, r_mat[:, j])
            gn_m = float(np.linalg.norm(g_mapped))
            monitors[trait] = {
                "direct_projection": {
                    "gradient_is": "r_hat (the unit monitor readout, applied in context space)",
                    "grad_norm": 1.0,
                    "worst_case_score_movement": "eps * 1.0",
                },
                "mapped_read": {
                    "gradient_is": "A @ r_hat (B1 row action)",
                    "grad_norm": gn_m,
                    "worst_case_score_movement": f"eps * {gn_m:.6g}",
                },
                "mapped_over_direct_grad_ratio": gn_m,
                "fitted_probe": core["probes"][trait] if core else {"status": deferral},
                "heldout": core["heldout"][trait] if core else deferral,
            }
        bound_scope = (
            (
                f"worst-case movement eps*|gradient| holds for a SINGLE application of the "
                f"map only; measured rho(A) = {rho:.4f} >= 1 on this layer FORECLOSES any "
                "geometric-series / iterated-map bound in (I - A)^-1 — such bounds are "
                "UNAVAILABLE and are not computed"
            )
            if rho >= 1.0
            else (
                f"worst-case movement eps*|gradient| stated for a single application; "
                f"measured rho(A) = {rho:.4f} < 1 on this layer (iterated bounds still "
                "not computed in this unit)"
            )
        )
        _write_json(
            jpath,
            {
                "regime": regime,
                "monitors": monitors,
                "rows": core["rows"] if core else deferral,
                "formulas": CERT_FORMULAS,
                "bound_scope": bound_scope,
                "baselines": {
                    "identity_bias": "inapplicable — scalar-target probe (d -> 1): the "
                    "identity+learned-bias baseline requires matched input/output "
                    "dimension (stated, never silently skipped)",
                    "knn_retrieval": "inapplicable — scalar-target probe: kNN retrieval "
                    "among held-out target VECTORS has no scalar-read analogue here "
                    "(stated, never silently skipped)",
                },
                "caveats": [CAVEAT_ACTIVATION, CAVEAT_MAP_LEVEL],
            },
            phase="pa-certificates",
        )
        logger.info(
            "[certificates] unit %d/%d L%d elapsed=%.1fs rows_attached=%s probe=%s",
            i,
            len(layers),
            layer,
            time.time() - t0,
            rows_attached,
            "computed" if core else "deferred",
        )
    _sentinel("pa-certificates", f"certificates done layers={list(layers)}")


def phase_upload(args) -> None:
    """Production-only HF upload of leg1/ + leg8/ with fail-loud exact-set verify.

    Smoke or --skip-upload SKIPS LOUDLY (a smoke's artifacts must never
    clobber the production prefix). Mirrors the rowbattery `_upload_leaf`
    shape: `upload_dir_sharded` (hub-routed, overflow-aware) + the exact-set
    `hub.verify_repo_paths_uploaded` when not rerouted.
    """
    if args.skip_upload or args.smoke:
        logger.warning("[upload] skip_upload/smoke: HF upload SKIPPED (loud)")
        return
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    for leaf, local in (("weights/leg1", _leg1(args)), ("weights/leg8", _leg8(args))):
        files = sorted(p for p in local.iterdir() if p.is_file())
        assert files, f"[upload] nothing to upload under {local} — run the phases first"
        prefix = f"{args.hf_prefix}/{leaf}"
        res = upload_dir_sharded(
            local,
            C.HF_DATA_REPO,
            prefix,
            repo_type="dataset",
            shard_glob="*",
            verify=True,
            delete_local=False,
            resume_skip=False,
        )
        if not res.rerouted:
            expected = [f"{prefix}/{p.name}" for p in files]
            missing = hub.verify_repo_paths_uploaded(
                HfApi(), C.HF_DATA_REPO, expected, path_in_repo=prefix
            )
            assert not missing, f"[upload] verify FAILED — missing on Hub: {missing}"
        logger.info("[upload] %s -> %s (rerouted=%s)", local, prefix, res.rerouted)
    _sentinel("pa-upload", "weights-battery leg1+leg8 uploaded")


PHASE_ORDER = (
    "factor",
    "anatomy",
    "alpha-lowrank",
    "fixed-point",
    "kernel",
    "monitor-geometry",
    "certificates",
    "upload",
)
PHASES = {
    "factor": phase_factor,
    "anatomy": phase_anatomy,
    "alpha-lowrank": phase_alpha_lowrank,
    "fixed-point": phase_fixed_point,
    "kernel": phase_kernel,
    "monitor-geometry": phase_monitor_geometry,
    "certificates": phase_certificates,
    "upload": phase_upload,
}


# ── CLI ───────────────────────────────────────────────────────────────────────────


def _parse_args(argv=None):
    """Argparse CLI for the P-A weights-battery driver (phase-dispatch shape)."""
    ap = argparse.ArgumentParser(
        description="Issue #2569 P-A weights battery, part 1 (see module docstring)"
    )
    ap.add_argument("--phase", default="all", choices=["all", *PHASE_ORDER])
    ap.add_argument("--out-root", type=Path, default=Path("/workspace/eps2569"))
    ap.add_argument(
        "--hf-prefix",
        default="issue2569_theory/analysis_tensors",
        help="HF data-repo destination prefix (issue-owned; never a parent's prefix)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="L19-only + top-8 + 100-draw seam of the SAME pipeline (plan SS4 Smoke run)",
    )
    ap.add_argument(
        "--layers",
        default=None,
        help="comma ints (default: smoke -> 19; production -> 19,14,26)",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="per-direction reporting width (0 = auto: smoke 8 / production 32)",
    )
    ap.add_argument(
        "--n-draws",
        type=int,
        default=0,
        help="null-floor draw budget recorded in the regime (0 = auto: smoke 100 / "
        "production 1000; consumed by the SAE-dashboard unit's empirical null)",
    )
    ap.add_argument(
        "--map-root",
        type=Path,
        default=None,
        help="banked-map root override (else EPS2569_MAP_ROOT env / repo root)",
    )
    ap.add_argument(
        "--rb-dir",
        type=Path,
        default=None,
        help="local #779 r_B blob dir override ({evil,sycophancy,hallucination}.pt; "
        "else staged from HF at the pinned revision)",
    )
    ap.add_argument(
        "--rows-dir",
        type=Path,
        default=None,
        help="P-B P1 assemble dir (X19/Y19 fp16 row store) — arms the certificate "
        "probe (iii) + held-out legs at layer 19 (post-P-B; else an explicit "
        "deferral is recorded)",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        help="probe moment-accumulation device (cpu|cuda; regime member — last-bit "
        "output-affecting)",
    )
    ap.add_argument(
        "--cert-chunk",
        type=int,
        default=CERT_CHUNK_DEFAULT,
        help="row chunk for probe moment accumulation (rowbattery parity; regime member)",
    )
    ap.add_argument("--fresh", action="store_true", help="ignore the per-unit resume predicate")
    ap.add_argument("--skip-upload", action="store_true", help="local-only run (loud)")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="argparse-attribute completeness + call-arity bind + deferred-import resolution",
    )
    return ap.parse_args(argv)


def main() -> None:
    """Driver entry: import-check seam, B1 entry asserts per layer, phase loop."""
    args = _parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Deferred-import resolution (smoke-architecture Axis 1): execute every
        # function-body import of this driver so a missing symbol fails HERE.
        import issue2476_turnavg_sae as T24  # noqa: F401
        import issue2569_rowbattery as RB  # noqa: F401
        from huggingface_hub import HfApi  # noqa: F401

        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.preflight import (
            assert_out_root_headroom,  # noqa: F401
        )
        from explore_persona_space.orchestrate.provenance import (  # noqa: F401
            as_metadata_dict,
            git_provenance,
        )
        from explore_persona_space.orchestrate.upload_sharded import (
            upload_dir_sharded,  # noqa: F401
        )

        assert callable(OP.load_apply_map) and callable(OP.run_driver_identity_asserts)
        assert callable(hub.verify_repo_paths_uploaded)
        assert callable(hub.stage_hub_file)
        # leg 8 steps 3+4 deferred symbols: the rows-store contract helpers
        assert callable(T24._committed_split) and callable(T24._assert_pinned_valtest)
        assert callable(RB._positions_in_present)
        print("[import-check] OK", flush=True)
        raise SystemExit(0)
    args.out_root.mkdir(parents=True, exist_ok=True)
    layers = _layers(args)
    # B1 driver-entry identity asserts per layer (a raise HALTS — apply-path
    # breakage class; the apply_map reference loads ONCE, ~20 s torch chain).
    apply_map = OP.load_apply_map()
    for layer in layers:
        payload = _load_payload(args, layer)
        entry = OP.run_driver_identity_asserts(payload, apply_map=apply_map)
        _write_json(
            _leg1(args) / f"entry_asserts_L{layer}.json",
            {
                "regime": _regime(args, layer),
                "entry_asserts": entry,
                "selected_lambda": float(payload.selected_lambda),
            },
            phase="pa-entry",
        )
    logger.info(
        "[main] phase=%s out_root=%s layers=%s smoke=%s top_k=%d n_draws=%d",
        args.phase,
        args.out_root,
        list(layers),
        args.smoke,
        _top_k(args),
        _n_draws(args),
    )
    seq = PHASE_ORDER if args.phase == "all" else (args.phase,)
    for name in seq:
        PHASES[name](args)
    # poller terminal line (pod-side-reporting.md req 1): single reserved emission
    # at the driver's own graceful exit, AFTER the phases' sentinel writes.
    print("[phase=done]", flush=True)
    # explicit exit: heavy C-extension teardown must not rewrite the rc (gotchas.md)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
