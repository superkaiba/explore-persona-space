"""#603 P3' write decomposition — shared vs source-specific components.

Decomposes a source persona's measured activation write ``w`` (the
trained-minus-base shift at the source context) against the SHARED
direction ``u`` = the unit-normalized MEAN of the bystander shifts
(leave-source-out by construction: the source's own column is excluded
from the mean).

PRIMARY DV: ``cmf`` — the signed cosine of ``w`` against ``u``
(scale-free, in [-1, 1]; an anti-aligned write is information, not
noise). CONTRAST DV: ``norm`` = ||w||. Robustness variants carried in
the same call (plan #603 §4 step 3): the norm-weighted SVD top
direction, the unit-norm-SVD top direction (the #551 norm-weighting
lesson), and the leave-one-bystander-out jackknife of ``cmf``.

Pure CPU torch/numpy over tensors produced by
``activation_shift.extract_per_context_shifts``; deterministic.
"""

from __future__ import annotations

import hashlib

import torch

DEFAULT_KEY = "delta_v_mean_resp"


def _sha256_tensor(t: torch.Tensor) -> str:
    """Stable content hash of a tensor (fp64 little-endian bytes)."""
    return hashlib.sha256(t.detach().to(torch.float64).cpu().numpy().tobytes()).hexdigest()


def _unit(v: torch.Tensor) -> torch.Tensor:
    n = v.norm()
    if n.item() == 0.0:
        raise ValueError("cannot unit-normalize a zero vector")
    return v / n


def _signed_cos(w: torch.Tensor, u: torch.Tensor) -> float:
    """Signed cosine of w against unit vector u (w must be nonzero)."""
    wn = w.norm()
    if wn.item() == 0.0:
        raise ValueError("zero-norm write — cosine undefined")
    return float(torch.dot(w, u) / wn)


def _top_left_singular(m: torch.Tensor) -> torch.Tensor:
    """Top LEFT singular vector of M (n_bys, H) -> (H,), sign-oriented.

    Sign convention matches ``svd_direction_constancy.svd_summary``:
    orient so the mean bystander shift has nonnegative projection.
    """
    assert m.dim() == 2, m.shape
    # torch.linalg.svd on (n, H): Vh rows are right-singular vectors of m,
    # i.e. directions in H-space (m = U S Vh). Top H-space direction = Vh[0].
    _, _, vh = torch.linalg.svd(m, full_matrices=False)
    u1 = vh[0]
    if torch.dot(m.mean(dim=0), u1).item() < 0:
        u1 = -u1
    return u1


def decompose_write(
    shifts: dict[str, dict[str, torch.Tensor]],
    source: str,
    key: str = DEFAULT_KEY,
) -> dict[str, object]:
    """Decompose the source write against the mean-bystander direction.

    Parameters
    ----------
    shifts
        ``{persona: entry}`` from ``extract_per_context_shifts`` (or a
        loaded ``.pt`` payload's ``"shifts"``); every entry must carry
        ``key`` as an (H,) tensor.
    source
        The cell's source persona (must be a key of ``shifts``; its
        column is EXCLUDED from the shared-direction estimate).
    key
        Which per-persona tensor to decompose (default: layer-primary
        mean-over-response shift).

    Returns
    -------
    dict with keys ``cmf`` (PRIMARY, signed cosine), ``norm``
    (CONTRAST, ||w||), ``shared_norm`` (signed projection), and
    ``residual_norm``, plus the robustness variants ``cmf_svd``,
    ``cmf_svd_unitnorm``, ``cmf_jackknife`` (list, one re-estimate per
    dropped bystander), ``u_vector_sha``, ``n_bystanders``,
    ``bystander_order``.
    """
    if source not in shifts:
        raise KeyError(f"source {source!r} not in shifts ({sorted(shifts)[:5]}...)")
    w = shifts[source][key].detach().double().cpu()  # (H,)
    assert w.dim() == 1, f"expected (H,) source write, got {tuple(w.shape)}"

    bys = sorted(p for p in shifts if p != source)
    if len(bys) < 2:
        raise ValueError(f"need >= 2 bystanders, got {len(bys)}")
    m = torch.stack([shifts[p][key].detach().double().cpu() for p in bys])  # (n_bys, H)
    assert m.shape == (len(bys), w.shape[0]), m.shape

    mean_b = m.mean(dim=0)
    u = _unit(mean_b)  # shared / mean-bystander direction
    proj = float(torch.dot(w, u))  # signed shared-component magnitude
    residual = w - proj * u

    # Robustness: SVD top direction (norm-weighted) + unit-norm SVD (#551
    # lesson: norm weighting can let one large column own the direction).
    u1_svd = _top_left_singular(m)
    row_norms = m.norm(dim=1, keepdim=True)
    if (row_norms == 0).any().item():
        raise ValueError("zero-norm bystander shift — unit-norm SVD undefined")
    u1_unitnorm = _top_left_singular(m / row_norms)

    # Leave-one-bystander-out jackknife of the PRIMARY cmf.
    cmf_jack: list[float] = []
    for i in range(len(bys)):
        keep = [j for j in range(len(bys)) if j != i]
        u_i = _unit(m[keep].mean(dim=0))
        cmf_jack.append(_signed_cos(w, u_i))

    return {
        "cmf": _signed_cos(w, u),  # PRIMARY DV: signed cosine in [-1, 1]
        "norm": float(w.norm()),  # CONTRAST DV
        "shared_norm": proj,  # signed ||c_a||
        "residual_norm": float(residual.norm()),
        "cmf_svd": _signed_cos(w, u1_svd),
        "cmf_svd_unitnorm": _signed_cos(w, u1_unitnorm),
        "cmf_jackknife": cmf_jack,
        "u_vector_sha": _sha256_tensor(u),
        "n_bystanders": len(bys),
        "bystander_order": bys,
        "key": key,
        "source": source,
    }


def split_half_reliability(
    per_q: torch.Tensor,
    *,
    n_random_splits: int = 50,
    seed: int = 42,
) -> dict[str, object]:
    """Split-half direction reliability of a per-question shift stack.

    ``per_q``: (n_q, H). Returns the deterministic even/odd-question
    half cosine plus the mean over ``n_random_splits`` random 50/50
    splits (#551 recipe; rng seed 42). Reliability r is the cosine
    between the two half-mean directions.
    """
    assert per_q.dim() == 2, per_q.shape
    n_q = per_q.shape[0]
    if n_q < 2:
        return {"r_even_odd": None, "r_random_mean": None, "n_q": int(n_q)}
    x = per_q.detach().double().cpu()

    def _half_cos(idx_a: torch.Tensor, idx_b: torch.Tensor) -> float:
        a = x[idx_a].mean(dim=0)
        b = x[idx_b].mean(dim=0)
        na, nb = a.norm(), b.norm()
        if na.item() == 0.0 or nb.item() == 0.0:
            return 0.0
        return float(torch.dot(a, b) / (na * nb))

    even = torch.arange(0, n_q, 2)
    odd = torch.arange(1, n_q, 2)
    r_eo = _half_cos(even, odd)

    gen = torch.Generator().manual_seed(seed)
    rs: list[float] = []
    half = n_q // 2
    for _ in range(n_random_splits):
        perm = torch.randperm(n_q, generator=gen)
        rs.append(_half_cos(perm[:half], perm[half:]))
    return {
        "r_even_odd": r_eo,
        "r_random_mean": float(sum(rs) / len(rs)),
        "r_random_all": rs,
        "n_q": int(n_q),
        "n_random_splits": n_random_splits,
        "rng_seed": seed,
    }
