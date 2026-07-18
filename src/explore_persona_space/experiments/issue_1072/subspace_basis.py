"""Low-dimensional token-subspace bases (issue #1072 amendment, plan §4.1).

The one new piece of math for the ``lowdim-token-subspace`` round: per answer
position, an orthonormal basis of span{gamma ⊙ W_U[y] : y ∈ S_b(t)} for the three
registered basis families (top-8 / top-32 next-token candidates, realized
lookahead-8), plus the chunked logits read that produces the candidate ids.

Key facts baked in (plan §4.1 + §12):

- **Candidate ids are exact from the PRE-norm final hidden state.** Under
  RMSNorm, ``logits = W_U · RMSNorm(h) = U_dir · (h / rms(h))`` with
  ``U_dir = gamma ⊙ W_U`` and ``rms(h) = sqrt(mean(h²) + eps) > 0`` a per-position
  positive scalar — so ``topk(U_dir · h / rms) == topk(logits)`` exactly, and
  log-probs need only the per-chunk logsumexp of the SAME scaled GEMM. Full
  logits never leave a (chunk x V) buffer (§8 OOM row).
- **Top-8 ⊆ top-32 via ONE unpivoted QR** (Householder leading-column
  nesting): ``Q[:, :8]`` of the top-32 QR spans the leading 8 input columns
  exactly while ``|R_ii|`` stays above the rank tolerance (assumption A4; the
  smoke asserts span equality numerically, g7 guards the full-rank premise).
- **g7 rank guard:** after order-preserving dedupe, require
  ``min_i |R_ii| / ‖A_i‖ > RANK_TOL``; a failing position is recomputed by SVD
  with a rank cutoff at the same relative tolerance (span-preserving) and
  COUNTED — the driver aborts if the rank-reduced fraction exceeds 1% in any
  arm x basis (a systematic degeneracy is an ids bug, not numerics).

All batched APIs take COMPACTED padded id tensors (valid ids in the leading
``eff_k`` columns) so the ~14M-position capture pass never runs a per-position
Python loop (vectorize-first rule) — the rare SVD fallback is the only
per-row path.
"""

from __future__ import annotations

import numpy as np

RANK_TOL = 1e-4  # plan §11 — `ungrounded — needs smoke-test` (fp32-QR scale)
TOPK = 32
NESTED_K = 8
LOOKAHEAD = 8


def topk_ids_from_final_hidden(
    h_final,
    u_dir_t,
    rms_eps: float,
    k: int = TOPK,
    chunk: int = 512,
):
    """Top-k candidate ids + log-probs from the PRE-norm final hidden states.

    Args:
        h_final: (P, H) torch tensor (any float dtype) — the LAST decoder
            layer's output (pre ``model.norm``) at the positions of interest.
        u_dir_t: (V, H) fp32 torch tensor, gamma-folded unembedding rows
            (``lm_head.weight * norm.weight``), on the compute device.
        rms_eps: the model's ``rms_norm_eps`` (variance epsilon).
        k: number of candidates to keep.
        chunk: positions per (chunk x V) logits buffer (memory bound).

    Returns:
        (ids (P, k) int32 torch, logprobs (P, k) fp16 torch) — logit-rank
        ordered (descending), on the input device. Full-vocab logits are never
        materialized beyond one chunk (plan §10 ``discarded_artifacts``).
    """
    import torch

    assert h_final.ndim == 2 and u_dir_t.ndim == 2, (h_final.shape, u_dir_t.shape)
    assert h_final.shape[1] == u_dir_t.shape[1], (h_final.shape, u_dir_t.shape)
    h = h_final.float()
    # RMSNorm scale, fp32 (matches Qwen2RMSNorm's fp32 variance path).
    rms = torch.sqrt(h.pow(2).mean(dim=1, keepdim=True) + rms_eps)
    hn = h / rms
    ids_out = torch.empty((h.shape[0], k), dtype=torch.int32, device=h.device)
    lp_out = torch.empty((h.shape[0], k), dtype=torch.float16, device=h.device)
    with torch.no_grad():
        for c0 in range(0, h.shape[0], chunk):
            logits = hn[c0 : c0 + chunk] @ u_dir_t.T  # (c, V) fp32 — the ONLY full-row buffer
            lse = torch.logsumexp(logits, dim=1, keepdim=True)
            vals, ids = torch.topk(logits, k, dim=1)  # descending — logit-rank order
            ids_out[c0 : c0 + chunk] = ids.to(torch.int32)
            lp_out[c0 : c0 + chunk] = (vals - lse).to(torch.float16)
            del logits
    return ids_out, lp_out


def compact_dedupe_windows(window, valid):
    """Order-preserving dedupe + left-compaction of padded id windows (batched).

    Args:
        window: (B, m) int64 torch tensor of ids (pad rows arbitrary).
        valid: (B, m) bool torch tensor — True where the window entry exists.

    Returns:
        (ids_compact (B, m) int64 — each row's kept ids left-compacted in
        original order, tail entries arbitrary; eff_k (B,) int64 — kept count
        per row). Span is unchanged by dedupe (plan §4.1 basis (b)).
    """
    import torch

    b, m = window.shape
    dup = torch.zeros((b, m), dtype=torch.bool, device=window.device)
    for j in range(1, m):
        dup[:, j] = (window[:, j : j + 1] == window[:, :j]).any(dim=1) & valid[:, j]
    keep = valid & ~dup
    eff_k = keep.sum(dim=1)
    assert bool((eff_k >= 1).all()), "empty window after dedupe/validity"
    # Stable keep-first ordering: argsort(~keep) keeps original order within keeps.
    order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)
    ids_compact = torch.gather(window, 1, order)
    return ids_compact, eff_k


def orthonormal_bases(
    u_dir_t,
    ids_pad,
    eff_k=None,
    rank_tol: float = RANK_TOL,
):
    """Orthonormal bases of span{U_dir[y] : y ∈ ids} per position, batched.

    Args:
        u_dir_t: (V, H) fp32 torch tensor on the compute device.
        ids_pad: (B, k_max) int64 torch tensor with each row's valid ids
            COMPACTED into the leading ``eff_k[i]`` columns (deduped —
            :func:`compact_dedupe_windows`; top-k ids are distinct by
            construction of ``topk``). Pad entries are ignored.
        eff_k: (B,) int64 torch tensor (None ⇒ all rows use k_max columns).
        rank_tol: g7 relative tolerance on ``|R_ii| / ‖A_i‖``.

    Returns:
        (q (B, H, k_max) fp32 torch — zero-padded beyond the effective rank so
        the projection ``Q Qᵀ z`` is unaffected; eff_k_out (B,) int64 torch —
        the SVD numerical rank where the fallback fired, else the input
        ``eff_k``; reduced_rows list[int] — g7 fallback positions).

    Batched unpivoted QR per eff-k group (``torch.linalg.qr`` mode="reduced"
    is Householder — leading-column nesting holds, plan §11 "Top-8 ⊆ top-32").
    """
    import torch

    assert ids_pad.ndim == 2, ids_pad.shape
    b, k_max = ids_pad.shape
    hid = u_dir_t.shape[1]
    assert 1 <= k_max <= hid, (k_max, hid)
    if eff_k is None:
        eff_k = torch.full((b,), k_max, dtype=torch.int64, device=ids_pad.device)
    assert bool((eff_k >= 1).all()) and bool((eff_k <= k_max).all()), (
        int(eff_k.min()),
        int(eff_k.max()),
    )
    q_out = torch.zeros((b, hid, k_max), dtype=torch.float32, device=u_dir_t.device)
    eff_out = eff_k.clone()
    reduced_rows: list[int] = []
    with torch.no_grad():
        for k in torch.unique(eff_k).tolist():
            rows = torch.nonzero(eff_k == int(k)).reshape(-1)
            a = u_dir_t[ids_pad[rows, : int(k)].to(u_dir_t.device)]  # (b_k, k, H)
            a = a.transpose(1, 2).contiguous()  # (b_k, H, k)
            col_norms = a.norm(dim=1)  # (b_k, k)
            q, r = torch.linalg.qr(a, mode="reduced")
            ratio = r.diagonal(dim1=1, dim2=2).abs() / (col_norms + 1e-30)
            bad = (ratio <= rank_tol).any(dim=1)
            good = ~bad
            if good.any():
                q_out[rows.to(q_out.device)[good], :, : int(k)] = q[good]
            for bi in torch.nonzero(bad).reshape(-1).tolist():
                # g7 span-preserving SVD fallback (counted; rank cutoff at the
                # SAME relative tolerance — plan §4.1 rank guard).
                u_svd, s_svd, _vh = torch.linalg.svd(a[bi], full_matrices=False)
                r_eff = int((s_svd > rank_tol * s_svd[0]).sum())
                assert r_eff >= 1, "degenerate basis: zero numerical rank"
                pos = int(rows[bi])
                q_out[pos, :, :r_eff] = u_svd[:, :r_eff]
                eff_out[pos] = r_eff
                reduced_rows.append(pos)
    return q_out, eff_out, reduced_rows


def nested_leading_bases(u_dir_t, q_full, ids_pad, reduced_rows: list[int], k_lead: int = NESTED_K):
    """Leading-``k_lead`` sub-bases from a full-k QR (top-8 ⊆ top-32 nesting).

    For non-reduced rows the leading ``k_lead`` Q-columns of the full QR span
    the leading ``k_lead`` input columns EXACTLY (unpivoted Householder QR);
    rows that took the g7 SVD fallback lose the nesting property, so their
    leading sub-basis is recomputed independently (rare path).

    Returns (q_lead (B, H, k_lead) fp32, eff_lead (B,) int64 torch,
    reduced_rows_lead list[int]).
    """
    import torch

    b = q_full.shape[0]
    q_lead = q_full[:, :, :k_lead].clone()
    eff_lead = torch.full((b,), k_lead, dtype=torch.int64, device=q_full.device)
    reduced_lead: list[int] = []
    if reduced_rows:
        rows = torch.tensor(sorted(reduced_rows), dtype=torch.int64)
        q_r, eff_r, red_r = orthonormal_bases(u_dir_t, ids_pad[rows, :k_lead])
        q_lead[rows.to(q_lead.device)] = q_r
        eff_lead[rows.to(eff_lead.device)] = eff_r.to(eff_lead.device)
        reduced_lead = [int(rows[i]) for i in red_r]
    return q_lead, eff_lead, reduced_lead


def project_rows(z, q):
    """Batched subspace projection ``z_par = Q (Qᵀ z)``.

    Args:
        z: (B, H) torch tensor (float); q: (B, H, k) fp32 (zero-padded cols OK).

    Returns:
        (B, H) fp32 torch tensor.
    """
    import torch

    with torch.no_grad():
        coeff = torch.einsum("bhk,bh->bk", q, z.float())
        return torch.einsum("bhk,bk->bh", q, coeff)


def nesting_check(u_dir_t, top32_ids: np.ndarray, k_lead: int = NESTED_K) -> float:
    """Numerical top-8-nesting assert helper (plan §8 risk row 2; smoke duty).

    Compares the projector from the leading ``k_lead`` Q-columns of the top-32
    QR against the projector from an INDEPENDENT QR of the leading ``k_lead``
    input columns. Returns the max Frobenius gap over positions (caller
    asserts < 1e-4 — fp32 projector scale).
    """
    import torch

    ids_t = torch.from_numpy(np.asarray(top32_ids, dtype=np.int64)).to(u_dir_t.device)
    q32, _eff, red = orthonormal_bases(u_dir_t, ids_t)
    assert not red, f"nesting check requires full-rank top-32 bases (got {len(red)} reduced)"
    q8, _eff8, red8 = orthonormal_bases(u_dir_t, ids_t[:, :k_lead])
    assert not red8, red8
    with torch.no_grad():
        p_a = q32[:, :, :k_lead] @ q32[:, :, :k_lead].transpose(1, 2)
        p_b = q8 @ q8.transpose(1, 2)
        gap = (p_a - p_b).norm(dim=(1, 2))
    return float(gap.max())
