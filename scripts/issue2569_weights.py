"""Issue #2569 P-A weights battery, part 1 (leg 1 core + leg 8 step 1).

Phase driver over the banked layer-19 (primary; L14/L26 replicates) n1m ridge
map's ROW operator ``x |-> x @ A`` (plan v4 SS4 leg 1 steps 1-3/5/6 + leg 8
step 1). Weights-only: no rows are staged (P-B refinements — data-weighted
mass fractions ``u_i^T Sigma_c u_i``, split-half spectrum error bars, fixed-
point nearest banked answers — are the rowbattery driver's job and are NOT
computed here from assumed inputs). Two later units extend THIS file with
leg 3 (wiring/receipts/attribution) and leg 8 steps 3/4 (monitor decision
geometry, sensitivity certificates); leg-1 step 4's two-sided SAE dashboards
and the fixed point's #2476-encoder decode ALSO land with that dashboard
unit (they need the SAE dictionary machinery shared with leg-8 step 3) —
part 1 marks those keys ``deferred``, never stubs them.

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
    except (OSError, json.JSONDecodeError):
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


PHASE_ORDER = ("factor", "anatomy", "alpha-lowrank", "fixed-point", "kernel", "upload")
PHASES = {
    "factor": phase_factor,
    "anatomy": phase_anatomy,
    "alpha-lowrank": phase_alpha_lowrank,
    "fixed-point": phase_fixed_point,
    "kernel": phase_kernel,
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
