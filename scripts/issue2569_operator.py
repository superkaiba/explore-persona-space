#!/usr/bin/env python3
"""Task #2569 shared operator module: the banked n1m ridge map as a ROW-action operator.

Every #2569 leg (gate ladder, row battery, dW fleet, cross-model atlas, kernel mining)
imports THIS module instead of re-deriving products from the raw payload — the
re-derivation is exactly what produced the v3 orientation error (plan blocker B1).

Vendored payload contract (plan blocker B6)
-------------------------------------------
The banked map payload is ``{W, xmu, xsd, ymu, selected_lambda}`` (fp32 tensors plus
``kind == fitter == 'ridge'`` and ``layer``) at::

    data/issue_2094/joint_transport/banked_maps/issue779_monitoring/n1m_readout/
        weights/L{14,19,26}/ridge.pt        (51,425,703 B each; L19 primary)

Registered prediction path (the ONLY sanctioned apply form)::

    vhat = ((v - xmu) / xsd) @ W + ymu

Identity+bias offset: ``ymu - xmu``. This contract is VENDORED here from the
working-tree provenance module ``scripts/issue2474_n1m_map.py::load_n1m_comp``
(L159-184) + ``identity_bias_offset`` (L185) — that module exists in ZERO git refs,
so pods must never import it; it is cited as provenance only. The contract is
verified against the artifact itself (observed top-level keys on the real L19 file:
``['W', 'fitter', 'kind', 'layer', 'selected_lambda', 'xmu', 'xsd', 'ymu']``).

Row-action operator (plan blocker B1)
-------------------------------------
Raw-space affine endomorphism on layer-19 residual space, row-vector convention::

    A = diag(1/xsd) @ W          # (3584, 3584); vhat = v @ A + b   (ROW action)
    b = ymu - (xmu / xsd) @ W

ALL eigen / singular / kernel / fixed-point reads are on A UNDER THE ROW ACTION
``x -> x @ A``. The transpose ``A.T`` appears only where a solver's API needs it,
never as a change of convention.

Orientation dictionary (used verbatim by every leg)
---------------------------------------------------
- INPUT/read singular directions  = LEFT  singular vectors u_i (``u_i @ A = s_i v_i``).
- OUTPUT/write singular directions = RIGHT singular vectors v_i.
- Eigen READS along RIGHT eigenvectors and WRITES along LEFT eigenvectors
  (``x @ A = sum_i lam_i (x . v_i^e) u_i^e^T`` under the biorthogonal expansion).
  NOTE the u/v letter assignment FLIPS between the eigen and singular
  decompositions — encoded here in field NAMES (``read_*`` / ``write_*``) so a
  caller cannot get it backwards.
- Through-map context similarity = ``c (A A^T) c'^T`` (standardized form ``W W^T``).
- Mapped displacement = ``delta_c @ A`` (affine terms cancel in differences).
- Monitor gradient = ``A r`` (gradient of ``r . (v @ A + b)`` w.r.t. ``v``).
- Wiring edges = ``E_f A^T D``.
- Effective kernel = the bottom LEFT-singular subspace (input directions read at
  low gain; ridge has no exact kernel).
- Fixed point: solve ``x* (I - A) = b`` (equivalently ``x*^T = (I - A^T)^{-1} b^T``).

Driver identity asserts (fp64, run at P-A/P-B entry — plan B1):
(i)   ``||x @ A||^2 == x (A A^T) x^T`` on 64 random probes;
(ii)  ``u_i @ A ~= s_i v_i`` (relative error < 1e-6) for the top-8 singular
      triplets — the row form holds to ~1.3e-15 on the real L19 map while the v3
      column form ``A v_i ~= s_i v_i`` fails at relative error ~1.35;
(iii) the vendored prediction path equals the main-resident
      ``scripts/issue779_ffc_n1m_fits.py::apply_map`` (L882) on a probe batch.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE any heavy import: binds the shared-VM thread caps in-process (#847)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.linalg import eig as scipy_eig  # noqa: E402
from scipy.sparse.linalg import svds  # noqa: E402

__all__ = [
    "BANKED_MAP_RELPATH",
    "D_MODEL",
    "N1M_LAYERS",
    "EigenPairs",
    "MapPayload",
    "SingularTriplets",
    "assert_prediction_matches_apply_map",
    "assert_row_action_gram",
    "assert_singular_orientation",
    "banked_map_path",
    "context_similarity",
    "eigen_read_write",
    "fixed_point",
    "identity_bias_offset",
    "kernel_read_directions",
    "load_apply_map",
    "load_banked_map",
    "mapped_displacement",
    "monitor_gradient",
    "predict",
    "prediction_difference",
    "row_operator",
    "run_driver_identity_asserts",
    "spectral_radius",
    "tau_kernel_threshold",
    "through_map_gram",
    "top_singular_triplets",
    "wiring_in_edges",
]

N1M_LAYERS: tuple[int, ...] = (14, 19, 26)
D_MODEL: int = 3584
BANKED_MAP_RELPATH: str = (
    "data/issue_2094/joint_transport/banked_maps/issue779_monitoring/"
    "n1m_readout/weights/L{layer}/ridge.pt"
)
_MAP_ROOT_ENV = "EPS2569_MAP_ROOT"
_PAYLOAD_KEYS = ("W", "xmu", "xsd", "ymu")


def _repo_root() -> Path:
    """Repo root inferred from this file's location (``scripts/`` sits under the root)."""
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class MapPayload:
    """A banked n1m ridge map, upcast fp64, plus the raw fp32 torch payload.

    ``W/xmu/xsd/ymu`` are float64 numpy arrays in the vendored contract shape;
    ``raw`` is the untouched ``torch.load`` dict (fp32 tensors + metadata keys) so
    assert (iii) can hand it verbatim to ``issue779_ffc_n1m_fits.apply_map``.
    """

    layer: int
    path: Path
    W: np.ndarray
    xmu: np.ndarray
    xsd: np.ndarray
    ymu: np.ndarray
    selected_lambda: float
    raw: dict = field(repr=False)

    @property
    def d(self) -> int:
        """Residual-space dimension (3584 for the banked Qwen2.5-7B maps)."""
        return int(self.W.shape[0])


def banked_map_path(layer: int, root: Path | str | None = None) -> Path:
    """Absolute path of the banked ridge payload for ``layer``.

    Root resolution precedence: explicit ``root`` argument > the
    ``EPS2569_MAP_ROOT`` environment variable > the repo root inferred from this
    file. Pods that stage the payload elsewhere (e.g. ``/workspace/eps2569/``)
    pass ``root`` explicitly or set the env var.
    """
    if root is None:
        env = os.environ.get(_MAP_ROOT_ENV)
        root = Path(env) if env else _repo_root()
    return Path(root) / BANKED_MAP_RELPATH.format(layer=layer)


def load_banked_map(layer: int = 19, root: Path | str | None = None) -> MapPayload:
    """Load + validate a banked n1m ridge payload (the vendored B6 contract).

    Validations (all fail loud): file exists; ``kind == fitter == 'ridge'``;
    payload ``layer`` matches the request; the four component keys are present;
    ``W`` is square ``(d, d)`` and the three vectors are ``(d,)``; every value is
    finite; ``xsd`` is strictly positive (it divides). Components are upcast to
    float64 (a rename-free upcast — the persisted payload is fp32).

    ``weights_only=False`` is deliberate: the payload is a self-produced,
    revision-pinned project artifact (plain tensors + scalars), matching the
    provenance loader's own call.
    """
    path = banked_map_path(layer, root=root)
    if not path.exists():
        raise FileNotFoundError(
            f"banked n1m ridge absent: {path} — banked only at layers {list(N1M_LAYERS)}; "
            f"on a pod, stage the payload and pass root= (or set {_MAP_ROOT_ENV})"
        )
    p = torch.load(path, map_location="cpu", weights_only=False)
    if p.get("kind") != "ridge" or p.get("fitter") != "ridge":
        raise RuntimeError(
            f"{path}: expected the ridge fitter, got kind={p.get('kind')!r} "
            f"fitter={p.get('fitter')!r}"
        )
    if int(p.get("layer", -1)) != int(layer):
        raise RuntimeError(f"{path}: payload layer {p.get('layer')} != requested {layer}")
    missing = [k for k in _PAYLOAD_KEYS if k not in p]
    if missing:
        raise RuntimeError(f"{path}: payload missing contract keys {missing}")
    comp = {k: np.asarray(p[k], dtype=np.float64) for k in _PAYLOAD_KEYS}
    d = comp["W"].shape[0]
    if comp["W"].shape != (d, d):
        raise RuntimeError(f"{path}: W shape {comp['W'].shape} is not square")
    for k in ("xmu", "xsd", "ymu"):
        if comp[k].shape != (d,):
            raise RuntimeError(f"{path}: {k} shape {comp[k].shape} != ({d},)")
    for k in _PAYLOAD_KEYS:
        if not np.isfinite(comp[k]).all():
            raise RuntimeError(f"{path}: {k} contains non-finite values")
    if not (comp["xsd"] > 0).all():
        raise RuntimeError(f"{path}: xsd must be strictly positive (it divides)")
    return MapPayload(
        layer=int(layer),
        path=path,
        W=comp["W"],
        xmu=comp["xmu"],
        xsd=comp["xsd"],
        ymu=comp["ymu"],
        selected_lambda=float(p["selected_lambda"]),
        raw=p,
    )


def predict(payload: MapPayload, v: np.ndarray) -> np.ndarray:
    """The REGISTERED prediction path: ``vhat = ((v - xmu)/xsd) @ W + ymu`` (fp64).

    ``v`` is ``(d,)`` or ``(n, d)`` of RAW (unstandardized) layer states; returns
    the same leading shape. Every leg's gate score / mining statistic must reduce
    to differences of THIS path (assert iii) — never a re-derived product.
    """
    v64 = np.asarray(v, dtype=np.float64)
    return ((v64 - payload.xmu) / payload.xsd) @ payload.W + payload.ymu


def prediction_difference(payload: MapPayload, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Registered prediction DIFFERENCE ``predict(v1) - predict(v2)``.

    Affine terms cancel: this equals ``(v1 - v2) @ A`` (= ``delta_std @ W``); the
    explicit two-predict form is kept as the REFERENCE the B1 assert-(iii) probe
    batches compare against.
    """
    return predict(payload, v1) - predict(payload, v2)


def identity_bias_offset(payload: MapPayload) -> np.ndarray:
    """The map's OWN identity-plus-bias offset ``ymu - xmu`` (vendored from B6 provenance).

    In a standardized ridge, ``xmu`` is the input mean and ``ymu`` the target mean
    added back at predict time, so their difference is exactly the constant shift
    an identity+bias predictor would learn on the same rows.
    """
    return payload.ymu - payload.xmu


def row_operator(payload: MapPayload) -> tuple[np.ndarray, np.ndarray]:
    """Form ``(A, b)`` with ``A = diag(1/xsd) @ W`` and ``b = ymu - (xmu/xsd) @ W`` (fp64).

    ``diag(1/xsd) @ W`` scales the ROWS of ``W`` by ``1/xsd``, so
    ``v @ A + b == predict(payload, v)`` exactly (the ROW action is the only
    sanctioned convention; see the module docstring's orientation dictionary).
    """
    A = payload.W / payload.xsd[:, None]
    b = payload.ymu - (payload.xmu / payload.xsd) @ payload.W
    return A, b


# --------------------------------------------------------------------------------------
# Orientation dictionary helpers (row action throughout)
# --------------------------------------------------------------------------------------


def through_map_gram(A: np.ndarray) -> np.ndarray:
    """Through-map context-similarity Gram ``A @ A.T``.

    Under the row action the image inner product is
    ``<c @ A, c' @ A> = c (A A^T) c'^T`` — so the CONTEXT-side similarity metric
    is ``A A^T`` (in standardized input coordinates, ``W W^T``), NOT ``A^T A``.
    """
    return A @ A.T


def context_similarity(A: np.ndarray, c: np.ndarray, c2: np.ndarray | None = None) -> np.ndarray:
    """Through-map context similarity ``<c @ A, c2 @ A>`` for row vectors/batches.

    Computed as the inner product of the mapped IMAGES (one GEMM per side), which
    equals ``c (A A^T) c2^T`` (assert i) without materializing the Gram — prefer
    this form for large batches. ``c2`` defaults to ``c``.
    """
    ci = np.atleast_2d(np.asarray(c, dtype=np.float64)) @ A
    cj = ci if c2 is None else np.atleast_2d(np.asarray(c2, dtype=np.float64)) @ A
    out = ci @ cj.T
    return out


@dataclass(frozen=True)
class SingularTriplets:
    """Top singular triplets of A under the ROW action, sorted by descending sigma.

    ``read_input_u[:, i]`` = LEFT singular vector u_i = the i-th INPUT/read
    direction; ``write_output_v[:, i]`` = RIGHT singular vector v_i = the i-th
    OUTPUT/write direction; ``read_input_u[:, i] @ A == sigma[i] * write_output_v[:, i]``
    (the B1 row identity). NOTE: in the EIGEN decomposition the letters flip —
    see :class:`EigenPairs`.
    """

    sigma: np.ndarray
    read_input_u: np.ndarray
    write_output_v: np.ndarray


def top_singular_triplets(A: np.ndarray, k: int = 8, seed: int = 0) -> SingularTriplets:
    """Top-``k`` singular triplets via ``scipy.sparse.linalg.svds`` (descending sigma).

    Uses a seeded deterministic start vector. ``svds`` is the sanctioned path for
    a FEW leading triplets (a dense fp64 3584^2 SVD takes >15 min on the shared
    VM); the full factorization belongs to the pod drivers.
    """
    A64 = np.asarray(A, dtype=np.float64)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(min(A64.shape))
    u, s, vt = svds(A64, k=k, v0=v0)
    order = np.argsort(s)[::-1]
    return SingularTriplets(
        sigma=s[order],
        read_input_u=u[:, order],
        write_output_v=vt[order].T,
    )


@dataclass(frozen=True)
class EigenPairs:
    """Eigendecomposition of A oriented for the ROW action (biorthogonal expansion).

    ``A = read_right_v @ diag(lam) @ write_left_rows`` with
    ``write_left_rows @ read_right_v == I`` (rows of ``inv(V_right)`` are the LEFT
    eigenvectors, biorthonormal by construction). Under the row action::

        x @ A = sum_i lam[i] * (x . read_right_v[:, i]) * write_left_rows[i]

    so eigen dashboards READ along the RIGHT eigenvectors (``read_right_v``
    columns) and WRITE along the LEFT eigenvectors (``write_left_rows`` rows).
    NOTE the u/v letter FLIP vs the singular decomposition: there the READ
    directions are the LEFT singular vectors — which is exactly the confusion
    class B1 closes; use the ``read_*``/``write_*`` field names, never the letters.
    Arrays are complex (conjugate pairs for real A); ``spectral_radius`` is
    ``max |lam|``.
    """

    lam: np.ndarray
    read_right_v: np.ndarray
    write_left_rows: np.ndarray
    spectral_radius: float


def eigen_read_write(A: np.ndarray) -> EigenPairs:
    """Full non-symmetric eigendecomposition of A, oriented for the row action.

    Right eigenvectors from ``scipy.linalg.eig`` (columns); LEFT eigenvectors as
    the rows of ``inv(V_right)`` (the plan's leg-1 convention), giving the
    biorthogonal expansion documented on :class:`EigenPairs`. COST: dense
    ``eig`` + ``inv`` at (d, d) — minutes at d=3584; pod-driver territory, keep
    it out of VM test paths (tests use small synthetic matrices).
    """
    A64 = np.asarray(A, dtype=np.float64)
    lam, v_right = scipy_eig(A64)
    write_left_rows = np.linalg.inv(v_right)
    return EigenPairs(
        lam=lam,
        read_right_v=v_right,
        write_left_rows=write_left_rows,
        spectral_radius=float(np.abs(lam).max()),
    )


def spectral_radius(A: np.ndarray) -> float:
    """``rho(A) = max |eigenvalue|`` (eigenvalues only — cheaper than full eig).

    The plan's fixed-point guard computes this FRESH: if ``rho(A) >= 1`` the
    iterated-map reading is dropped and the fixed point is reported as the
    affine-consistency point only.
    """
    return float(np.abs(np.linalg.eigvals(np.asarray(A, dtype=np.float64))).max())


def mapped_displacement(delta: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Row-mapped displacement ``delta @ A`` (= ``delta_std @ W``; affine terms cancel).

    ``delta`` is a ``(d,)`` vector or ``(n, d)`` batch of RAW-space context
    displacements ``c1 - c2``; equals ``prediction_difference(payload, c1, c2)``
    (assert iii checks any mining statistic against that registered form).
    """
    return np.asarray(delta, dtype=np.float64) @ A


def monitor_gradient(A: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Gradient of the mapped monitor read ``r . (v @ A + b)`` w.r.t. ``v``: ``A @ r``.

    The minimal context-space change flipping the mapped read is along this
    direction (leg 8's decision geometry; leg 5's gate-direction comparator).
    """
    return A @ np.asarray(r, dtype=np.float64)


def wiring_in_edges(e_f: np.ndarray, A: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Wiring in-edges ``E_f @ A.T @ D`` for answer-encoder row(s) ``e_f``.

    ``e_f``: ``(d,)`` or ``(m, d)`` answer-SAE encoder rows; ``D``: ``(d, n_feat)``
    context-SAE decoder matrix. Computed as one ``(m, d)`` GEMV/GEMM against
    ``A.T`` then one GEMM against ``D`` — never materialize the dense
    feature-by-feature wiring matrix. The transpose here is the row-convention
    edge FORM (plan B1), not a change of convention.
    """
    ef = np.atleast_2d(np.asarray(e_f, dtype=np.float64))
    out = (ef @ A.T) @ D
    return out[0] if np.asarray(e_f).ndim == 1 else out


def tau_kernel_threshold(singular_values: np.ndarray, mass: float = 0.99) -> tuple[float, int]:
    """``(tau_kernel, rank)`` from a FULL singular spectrum: the sigma^2-mass convention.

    ``rank`` = the smallest number of leading singular values whose cumulative
    sigma^2 mass reaches ``mass`` (default 99% — the #1768 ``operator_kv_deep``
    convention; pass 0.90 for #1774's k90 twin); ``tau_kernel`` = the ``rank``-th
    singular value (the value AT which the mass is reached). Input must be the
    full descending spectrum — a truncated ``svds`` spectrum has the wrong total
    mass.
    """
    s = np.sort(np.asarray(singular_values, dtype=np.float64))[::-1]
    if s.ndim != 1 or s.size == 0:
        raise ValueError("singular_values must be a non-empty 1-D spectrum")
    cum = np.cumsum(s**2) / np.sum(s**2)
    rank = int(np.searchsorted(cum, mass) + 1)
    rank = min(rank, s.size)
    return float(s[rank - 1]), rank


def kernel_read_directions(read_input_u: np.ndarray, sigma: np.ndarray, tau: float) -> np.ndarray:
    """Effective-kernel INPUT directions: LEFT singular vectors with ``sigma < tau``.

    Pure filter over a precomputed decomposition (columns of ``read_input_u``
    paired with ``sigma``): the bottom LEFT-singular subspace — the directions the
    map READS at low gain (``u @ A`` is small). Ridge has no exact kernel; phrase
    claims as "read at < X% of typical gain".
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    return np.asarray(read_input_u)[:, sigma < tau]


def fixed_point(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the ROW-convention fixed point ``x* (I - A) = b``.

    Equivalently ``x*^T = (I - A^T)^{-1} b^T`` — the transpose enters only because
    the LAPACK solver wants column form. Callers must guard with a FRESH
    ``spectral_radius(A)`` read: at ``rho(A) >= 1`` the iterated-map reading is
    dropped and x* is reported as the affine-consistency point only.
    """
    A64 = np.asarray(A, dtype=np.float64)
    d = A64.shape[0]
    return np.linalg.solve((np.eye(d) - A64).T, np.asarray(b, dtype=np.float64))


# --------------------------------------------------------------------------------------
# B1 driver identity asserts (fp64; run at P-A/P-B entry AND as committed unit tests)
# --------------------------------------------------------------------------------------


def assert_row_action_gram(
    A: np.ndarray, n_probes: int = 64, seed: int = 0, rtol: float = 1e-8
) -> dict:
    """B1 assert (i): ``||x @ A||^2 == x (A A^T) x^T`` on random probes (fp64).

    Ties the image-norm read to the through-map Gram: if the Gram were oriented
    ``A^T A`` (the column-action similarity) this fails on any non-normal A.
    Returns ``{"max_rel_err": ...}``; raises ``AssertionError`` past ``rtol``.
    """
    A64 = np.asarray(A, dtype=np.float64)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_probes, A64.shape[0]))
    lhs = np.einsum("ij,ij->i", X @ A64, X @ A64)
    G = through_map_gram(A64)
    rhs = np.einsum("ij,jk,ik->i", X, G, X)
    max_rel = float(np.max(np.abs(lhs - rhs) / np.abs(rhs)))
    if max_rel >= rtol:
        raise AssertionError(
            f"[b1-gram-assert] ||x@A||^2 vs x(AA^T)x^T diverge: max rel err {max_rel:.3e} "
            f">= {rtol:.1e} on {n_probes} probes"
        )
    return {"max_rel_err": max_rel, "n_probes": n_probes}


def assert_singular_orientation(
    A: np.ndarray, k: int = 8, rtol: float = 1e-6, seed: int = 0
) -> dict:
    """B1 assert (ii): ``u_i @ A ~= sigma_i v_i`` (rel err < ``rtol``) for top-k triplets.

    The ROW form (LEFT singular vector in, sigma x RIGHT singular vector out) must
    hold to ~fp64 precision (measured 1.3e-15 on the real L19 map). Also reports
    the v3 COLUMN-form misread ``A @ v_i ~= sigma_i v_i`` (diagnostic only, NOT
    asserted here — it is legitimately small for a symmetric A; the real map
    fails it at ~1.35). Returns per-triplet stats; raises on a row-form breach.
    """
    A64 = np.asarray(A, dtype=np.float64)
    trip = top_singular_triplets(A64, k=k, seed=seed)
    row_err = (
        np.linalg.norm(
            trip.read_input_u.T @ A64 - trip.sigma[:, None] * trip.write_output_v.T, axis=1
        )
        / trip.sigma
    )
    col_err = (
        np.linalg.norm(
            (A64 @ trip.write_output_v).T - trip.sigma[:, None] * trip.write_output_v.T, axis=1
        )
        / trip.sigma
    )
    max_row = float(row_err.max())
    if max_row >= rtol:
        raise AssertionError(
            f"[b1-singular-orientation-assert] row identity u_i @ A = sigma_i v_i breached: "
            f"max rel err {max_row:.3e} >= {rtol:.1e} over top-{k} triplets"
        )
    return {
        "max_row_form_rel_err": max_row,
        "row_form_rel_err": row_err.tolist(),
        "wrong_column_form_rel_err": col_err.tolist(),
        "sigma": trip.sigma.tolist(),
        "k": k,
    }


def load_apply_map():
    """Load ``apply_map`` from the main-resident ``scripts/issue779_ffc_n1m_fits.py``.

    Deferred importlib-by-file-path load (never a ``scripts.*`` package import —
    those crash pod-side in script mode, #823). The module self-inserts
    ``scripts/`` + ``src/`` into ``sys.path`` for its own sibling imports; import
    wall is ~20 s (torch + the #779 sibling chain), so this stays out of module
    top level and is paid only by assert (iii).
    """
    path = Path(__file__).resolve().parent / "issue779_ffc_n1m_fits.py"
    spec = importlib.util.spec_from_file_location("issue779_ffc_n1m_fits", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load apply_map reference from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.apply_map


def assert_prediction_matches_apply_map(
    payload: MapPayload,
    n_probes: int = 8,
    seed: int = 0,
    rtol: float = 1e-9,
    atol: float = 1e-8,
    apply_map=None,
) -> dict:
    """B1 assert (iii): the vendored path equals ``issue779_ffc_n1m_fits.apply_map``.

    Runs both prediction paths on a random probe batch (fp64, CPU) and asserts
    element-wise agreement. ``apply_map`` may be injected (tests); by default it
    is loaded from the main-resident module via :func:`load_apply_map`. Returns
    ``{"max_abs_diff": ..., "max_rel_diff": ...}``; raises on divergence.
    """
    if apply_map is None:
        apply_map = load_apply_map()
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_probes, payload.d))
    ours = predict(payload, X)
    ref = apply_map(payload.raw, X, torch.device("cpu"))
    abs_diff = np.abs(ours - ref)
    max_abs = float(abs_diff.max())
    denom = np.maximum(np.abs(ref), atol)
    max_rel = float((abs_diff / denom).max())
    if not np.allclose(ours, ref, rtol=rtol, atol=atol):
        raise AssertionError(
            f"[b1-apply-path-assert] vendored predict diverges from apply_map: "
            f"max abs diff {max_abs:.3e}, max rel diff {max_rel:.3e} "
            f"(rtol={rtol:.1e}, atol={atol:.1e})"
        )
    return {"max_abs_diff": max_abs, "max_rel_diff": max_rel, "n_probes": n_probes}


def run_driver_identity_asserts(
    payload: MapPayload,
    n_probes: int = 64,
    k: int = 8,
    seed: int = 0,
    apply_map=None,
) -> dict:
    """Run all three B1 identity asserts on a loaded payload (P-A/P-B entry gate).

    Forms ``(A, b)`` via :func:`row_operator`, then runs asserts (i)-(iii).
    Returns the merged per-assert stats dict (persist it in the phase's entry
    record); raises ``AssertionError`` on any breach — a breach is apply-path
    breakage class and HALTS the phase.
    """
    A, _b = row_operator(payload)
    return {
        "gram": assert_row_action_gram(A, n_probes=n_probes, seed=seed),
        "singular_orientation": assert_singular_orientation(A, k=k, seed=seed),
        "apply_path": assert_prediction_matches_apply_map(
            payload, n_probes=min(n_probes, 8), seed=seed, apply_map=apply_map
        ),
    }
