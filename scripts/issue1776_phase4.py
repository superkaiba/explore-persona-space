"""#1776 Phase 4: J-space workspace mediation (plan §4 Phase 4, §6 nulls).

Sub-commands:
  build-dict    P0.2 J-space token dictionary at ONE layer: lens vector for
                token w = unit-normalized J_ell^T (g ⊙ W_U[w]) — the
                J_ell^T-pulled-back unembedding row under the vendored repo's
                readout ``lm_head(final_norm(J h))`` with the RMSNorm folded as
                its diagonal weight g (the scale 1/rms is row-constant and
                drops under the unit-norm the pursuit consumes; ``row_norms``
                are persisted so raw-logit decode ranking stays recoverable).
  energy        Read-side (L14 dict): pursuit + top-k-span energies of M''s
                top-20 right-singular directions, J_last's top-20 row-space
                directions, and the r_B layer-14 rows. Write-side (L19 dict):
                M''s top-20 column-space directions + the measured Phase-3
                shifts Δv̄ per (direction × α). Each vs THREE registered null
                families (plan §6) with the top-k selection re-run INSIDE every
                draw, 100 draws/family; null mean + p97.5 + band-to-ceiling
                margin (energy <= 1) next to every energy.
  refit-split   Ridge c_last(19)→{P_J·v, (I−P_J)·v, full v} at n=50k on the
                EXTENDED 28-pt grid (the Phase-0.5 Gram machinery), plus a
                dim/spectrum-matched random-subspace refit reference; the
                J-space R² fraction reported RELATIVE to the variance fraction
                P_J^{(19)} captures. P_J = top-r right-singular subspace of the
                L19 dictionary (r by --pj-energy cumulative spectral energy,
                default 0.95, or --pj-rank override — both recorded).
  jdelta-split  Causal split: δ_J = P_J^{(14)}Δ vs δ_⊥ for every Phase-3
                direction; per-cell cos(Δv̄, J_last·αδ_J) vs cos(Δv̄, J_last·αδ_⊥)
                from the persisted phase-3 summaries (pure re-reduction).

Null equivalence used for family (i): pursuit over a ROTATED dictionary D·Q is
selection- and energy-identical to pursuit of the Q^T-rotated probe over D
(corr x·(DQ)^T = (xQ^T)·D^T; atom Gram invariant), so all three families batch
as probe-row constructions through ONE batched pursuit call (vectorize rule).

CPU smoke: ``--smoke`` runs every sub-command body on synthetic tiny dims —
energies in [0,1] asserted, null bands sane, per-draw selection exercised,
refit round-trip (P + perp + full R² recovered on a planted subspace map),
jdelta round-trip on a synthetic phase-3 root.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import issue1776_common as C76
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # bind shared-VM thread caps BEFORE numpy/torch import (#847 gate)

import numpy as np  # noqa: E402
import torch  # noqa: E402

# ── dictionary construction (P0.2) ───────────────────────────────────────────


def build_dictionary(lens_path: Path, model_name: str, layer: int, *, device: str, tiny: bool):
    """(V, H) unit-row dictionary + row_norms at ``layer`` from the fitted lens."""
    C76.add_jlens_path()
    from jlens.lens import JacobianLens

    lens = JacobianLens.load(str(lens_path))
    assert layer in lens.jacobians, (layer, lens.source_layers)
    J = lens.jacobians[layer].to(device, torch.float32)  # (H, H): maps layer-l -> final basis
    if tiny:
        import issue1776_jlens_fit as JF

        _lens_model, hf, _tok = JF.load_lens_model(model_name, device=device, tiny=True)
    else:
        from transformers import AutoModelForCausalLM

        hf = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    w_u = hf.lm_head.weight.detach().to(device, torch.float32)  # (V, H)
    g = hf.model.norm.weight.detach().to(device, torch.float32)  # (H,)
    v, h = w_u.shape
    assert J.shape == (h, h), (J.shape, w_u.shape)
    rows = torch.empty((v, h), dtype=torch.float16)
    norms = torch.empty(v, dtype=torch.float32)
    for s in range(0, v, 16384):
        blk = (w_u[s : s + 16384] * g) @ J  # (b, H): J^T pullback as row-vector form
        n = blk.norm(dim=1).clamp_min(1e-12)
        rows[s : s + 16384] = (blk / n[:, None]).to(torch.float16).cpu()
        norms[s : s + 16384] = n.cpu()
    return {
        "rows_unit": rows,
        "row_norms": norms,
        "layer": layer,
        "model": model_name,
        "lens_path": str(lens_path),
        "convention": "unit-norm J^T (g*W_U[w]); raw-logit rank = (rows_unit@x)*row_norms",
    }


def cmd_build_dict(args) -> int:
    t0 = time.time()
    payload = build_dictionary(
        args.lens, args.model, args.layer, device=args.device, tiny=args.tiny
    )
    payload["repro"] = C76.repro_meta()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(
        f"[phase4] [phase=dict_done layer={args.layer}] shape={tuple(payload['rows_unit'].shape)}"
        f" -> {args.out} elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    return 0


def load_dict(path: Path, device: str) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    d["rows_unit"] = d["rows_unit"].to(device)
    return d


# ── batched nonneg gradient pursuit + top-k-span projection ──────────────────


def _refit_coeffs(atoms: torch.Tensor, x: torch.Tensor, *, nonneg: bool) -> torch.Tensor:
    """Batched LS refit on selected atoms; nonneg clamps coefficients >= 0.
    atoms (B, t, H) fp32, x (B, H) fp32 -> coeffs (B, t)."""
    gram = atoms @ atoms.transpose(1, 2)  # (B, t, t)
    eye = torch.eye(gram.shape[-1], device=gram.device) * 1e-6
    rhs = (atoms @ x.unsqueeze(-1)).squeeze(-1)  # (B, t)
    coeffs = torch.linalg.solve(gram + eye, rhs)
    if nonneg:
        coeffs = coeffs.clamp_min(0.0)
    return coeffs


def pursuit_energies(
    X: torch.Tensor, D: torch.Tensor, k: int, *, chunk: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Batched greedy pursuit over unit-row dictionary D (V, H).

    Returns (pursuit_energy, span_energy) per row of X — pursuit = nonneg
    sparse coding via greedy max-positive-correlation atom selection + clamped
    LS refit (the honest cone read); span = orthogonal projection onto the
    span of the top-k atoms by |correlation| (the cheap upper bound). Both
    energies = ||x̂||²/||x||² ∈ [0, 1]. Selection runs per ROW, so a null draw's
    rows re-select their own atoms (selection-symmetric by construction).
    """
    dev = D.device
    n = X.shape[0]
    e_pur = np.empty(n)
    e_span = np.empty(n)
    for s in range(0, n, chunk):
        x = X[s : s + chunk].to(dev, torch.float32)
        b = x.shape[0]
        xn2 = (x * x).sum(1).clamp_min(1e-24)
        corr0 = x.to(D.dtype) @ D.T  # (b, V)
        # top-k-span (signed): top-k atoms by |corr|, full LS projection.
        top = corr0.abs().topk(k, dim=1).indices  # (b, k)
        atoms = D[top].to(torch.float32)  # (b, k, H)
        c = _refit_coeffs(atoms, x, nonneg=False)
        xhat = (c.unsqueeze(1) @ atoms).squeeze(1)
        e_span[s : s + b] = ((xhat * xhat).sum(1) / xn2).cpu().numpy()
        # greedy nonneg pursuit.
        resid = x.clone()
        sel = torch.empty((b, 0), dtype=torch.long, device=dev)
        taken = torch.zeros((b, D.shape[0]), dtype=torch.bool, device=dev)
        xhat = torch.zeros_like(x)
        for _t in range(k):
            corr = (resid.to(D.dtype) @ D.T).float().masked_fill(taken, -torch.inf)
            idx = corr.argmax(1)  # nonneg: max positive corr; clamp handles <=0 picks
            taken[torch.arange(b, device=dev), idx] = True
            sel = torch.cat([sel, idx[:, None]], dim=1)
            atoms = D[sel].to(torch.float32)  # (b, t, H)
            c = _refit_coeffs(atoms, x, nonneg=True)
            xhat = (c.unsqueeze(1) @ atoms).squeeze(1)
            resid = x - xhat
        e_pur[s : s + b] = ((xhat * xhat).sum(1) / xn2).cpu().numpy()
    assert (e_span >= -1e-4).all() and (e_span <= 1.0 + 1e-4).all(), (
        e_span.min(),
        e_span.max(),
    )
    assert (e_pur >= -1e-4).all() and (e_pur <= 1.0 + 1e-4).all(), (e_pur.min(), e_pur.max())
    return e_pur.clip(0, 1), e_span.clip(0, 1)


# ── null families (plan §6: 3 registered families, selection rides per draw) ─


def _rand_orthogonal(h: int, gen: torch.Generator, device: str) -> torch.Tensor:
    a = torch.randn(h, h, generator=gen, device="cpu").to(device)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r))  # unique sign convention


def _cov_factor(acts: torch.Tensor, device: str) -> torch.Tensor:
    """Σ̂^{1/2} of the empirical layer covariance (eigh; cuSOLVER CPU fallback)."""
    a = acts.to(device, torch.float32)
    a = a - a.mean(0)
    cov = (a.T @ a) / max(1, a.shape[0] - 1)
    try:
        w, v = torch.linalg.eigh(cov)
    except torch.linalg.LinAlgError:
        print("[phase4] eigh non-convergence on device — CPU fallback (gotchas recipe)")
        w, v = torch.linalg.eigh(cov.cpu())
        w, v = w.to(device), v.to(device)
    return v @ torch.diag(w.clamp_min(0).sqrt()) @ v.T


def null_probe_rows(
    X: torch.Tensor, family: str, n_draws: int, *, cov_half: torch.Tensor | None, seed: int
) -> torch.Tensor:
    """(n_draws * n, H) null probe rows; per-draw re-selection happens inside
    pursuit (per-row selection). Families: rotation | isotropic | cov."""
    h = X.shape[1]
    dev = X.device
    norms = X.norm(dim=1, keepdim=True)
    out = []
    gen = torch.Generator().manual_seed(seed)
    for _d in range(n_draws):
        if family == "rotation":
            q = _rand_orthogonal(h, gen, str(dev))
            out.append(X @ q.T)  # pursuit(D@Q, x) == pursuit(D, x@Q^T... see docstring)
        elif family == "isotropic":
            z = torch.randn(X.shape, generator=gen).to(dev)
            out.append(z / z.norm(dim=1, keepdim=True) * norms)
        elif family == "cov":
            assert cov_half is not None, "cov family requires --acts"
            z = torch.randn(X.shape, generator=gen).to(dev) @ cov_half.T
            out.append(z / z.norm(dim=1, keepdim=True).clamp_min(1e-12) * norms)
        else:
            raise ValueError(family)
    return torch.cat(out, 0)


def energy_read(
    name: str, X: torch.Tensor, D: torch.Tensor, args, cov_half: torch.Tensor | None
) -> dict:
    """Observed + 3-family null bands for one probe set (both projectors)."""
    # crash-fix r11: named-tensor same-device guard immediately ahead of the mm
    # chain (pursuit + the cov null's ``z @ cov_half.T`` in null_probe_rows) —
    # any residual mismatch fail-louds HERE naming the offending tensor, not
    # deep in torch mm internals ("mat2 is on cuda:0 ... on cpu").
    devices = {"x_probe": X.device, "dict_rows": D.device}
    if cov_half is not None:
        devices["cov_half"] = cov_half.device
    assert len({str(d) for d in devices.values()}) == 1, f"energy_read device mismatch: {devices}"
    n = X.shape[0]
    obs_pur, obs_span = pursuit_energies(X, D, args.k, chunk=args.chunk)
    rec: dict = {
        "probe_set": name,
        "n_vectors": int(n),
        "k": args.k,
        "pursuit": {"per_vector": obs_pur.tolist(), "mean": float(obs_pur.mean())},
        "topk_span": {"per_vector": obs_span.tolist(), "mean": float(obs_span.mean())},
        "nulls": {},
    }
    for fam in ("rotation", "isotropic", "cov"):
        rows = null_probe_rows(X, fam, args.n_draws, cov_half=cov_half, seed=args.seed)
        pur, span = pursuit_energies(rows, D, args.k, chunk=args.chunk)
        draws_pur = pur.reshape(args.n_draws, n).mean(1)
        draws_span = span.reshape(args.n_draws, n).mean(1)
        rec["nulls"][fam] = {
            "n_draws": args.n_draws,
            "pursuit_mean": float(draws_pur.mean()),
            "pursuit_p975": float(np.quantile(draws_pur, 0.975)),
            "topk_span_mean": float(draws_span.mean()),
            "topk_span_p975": float(np.quantile(draws_span, 0.975)),
            # band-vs-ceiling: energy is bounded <= 1 (selection-symmetric-nulls rule)
            "pursuit_band_to_ceiling": float(1.0 - np.quantile(draws_pur, 0.975)),
            "topk_span_band_to_ceiling": float(1.0 - np.quantile(draws_span, 0.975)),
        }
        print(
            f"[phase4] [energy] {name} fam={fam} obs_pur={obs_pur.mean():.4f} "
            f"null_mean={draws_pur.mean():.4f} p975={np.quantile(draws_pur, 0.975):.4f}",
            flush=True,
        )
    return rec


def _svd_w(payload_path: Path, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(right, left) top-k singular directions of a persisted ridge W."""
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    w = payload["W"].to(torch.float32)
    u, _s, vh = torch.linalg.svd(w, full_matrices=False)
    # W maps standardized input (rows: input dim) -> output: x @ W. Right/input
    # singular directions are U's columns; output/column space is Vh's rows.
    return u[:, :topk].T.contiguous(), vh[:topk].contiguous()


def measured_shifts(phase3_root: Path) -> dict[str, torch.Tensor]:
    """Δv̄ per steered stratum from the persisted phase-3 summaries (mean over
    contexts of per-context mean steered v − baseline mean v)."""
    import issue1776_phase3 as P3

    dirs = {"summaries": phase3_root / "summaries"}
    base = P3.load_base_means(dirs)
    out: dict[str, torch.Tensor] = {}
    for p in sorted(dirs["summaries"].glob("*.pt")):
        if p.stem == "baseline_a0":
            continue
        st = torch.load(p, map_location="cpu", weights_only=True)
        dvs = [
            st["v19"][cid].to(torch.float64).mean(0) - base[cid]
            for cid in st["v19"]
            if cid in base and st["v19"][cid].shape[0] > 0
        ]
        if dvs:
            out[p.stem] = torch.stack(dvs).mean(0).to(torch.float32)
    assert out, f"no steered summaries under {phase3_root}"
    return out


def _to_device_sets(sets: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    """TOTAL device placement for probe sets (crash-fix r11, att-20260729-060640).

    Every matmul participant of ``energy_read`` joins the dictionaries + cov
    factors on the ONE device resolved at ``cmd_energy`` entry (``args.device``).
    All four probe producers load on CPU by design (``map_location="cpu"``:
    ``_svd_w``, the ``_load_j`` SVD, ``P3.load_directions``,
    ``measured_shifts``); ``pursuit_energies`` tolerates that (it chunks probes
    onto ``D.device``), but the cov null family matmuls X-keyed draws against
    ``cov_half`` (``null_probe_rows``: ``z @ cov_half.T`` with ``z`` on
    ``X.device``), so a CPU probe set against a cuda ``cov_half`` dies in
    wrapper_CUDA_mm — a branch structurally unexercisable on the CPU-only VM
    smoke, where every tensor is cpu by construction. float32 matches what
    ``pursuit_energies`` casts per chunk anyway.
    """
    return {k: v.to(device, torch.float32) for k, v in sets.items()}


def cmd_energy(args) -> int:
    dev = args.device
    d14 = load_dict(args.dict14, dev)
    d19 = load_dict(args.dict19, dev)
    h = d14["rows_unit"].shape[1]
    cov14 = cov19 = None
    if args.acts14:
        cov14 = _cov_factor(_load_acts(args.acts14, h), dev)
    if args.acts19:
        cov19 = _cov_factor(_load_acts(args.acts19, h), dev)

    read_sets: dict[str, torch.Tensor] = {}
    write_sets: dict[str, torch.Tensor] = {}
    if args.mprime_weights:
        right, left = _svd_w(args.mprime_weights, args.topk)
        read_sets["mprime_right_top20"] = right
        write_sets["mprime_column_top20"] = left
    if args.jlast:
        j = _load_j(args.jlast)
        _u, _s, vh = torch.linalg.svd(j.to(torch.float32), full_matrices=False)
        read_sets["jlast_rowspace_top20"] = vh[: args.topk].contiguous()
    if args.rb_dir:
        import issue1776_phase3 as P3

        # P3.load_directions contract (crash-fix r10): the TRAIT branch row-selects
        # the (L, H) stack via args.source_layer (phase3.py L136) and a `random`
        # stem would read args.random_seed — supply EVERY attr the callee reads,
        # matching phase3's own parser defaults (C76.SOURCE_LAYER / 1776).
        ns = argparse.Namespace(
            rb_dir=args.rb_dir,
            mprime_weights=None,
            directions=list(_rb_traits(args.rb_dir)),
            source_layer=C76.SOURCE_LAYER,
            random_seed=1776,
        )
        bank, _prov = P3.load_directions(ns, h)
        read_sets["rb_layer14_rows"] = torch.stack([bank[t] for t in sorted(bank)])
    if args.phase3_root:
        shifts = measured_shifts(args.phase3_root)
        write_sets["measured_shifts"] = torch.stack([shifts[k] for k in sorted(shifts)])
        write_shift_names = sorted(shifts)
    else:
        write_shift_names = []

    assert read_sets or write_sets, "nothing to read — pass probe-set inputs"
    # crash-fix r11: the SINGLE move-to-device site for every probe tensor —
    # see _to_device_sets. Placed after ALL producers, before any energy_read.
    read_sets = _to_device_sets(read_sets, dev)
    write_sets = _to_device_sets(write_sets, dev)
    report: dict = {
        "k": args.k,
        "n_draws": args.n_draws,
        "read_side_layer": d14["layer"],
        "write_side_layer": d19["layer"],
        "measured_shift_strata": write_shift_names,
        "read_side": [
            energy_read(nm, x, d14["rows_unit"], args, cov14) for nm, x in read_sets.items()
        ],
        "write_side": [
            energy_read(nm, x, d19["rows_unit"], args, cov19) for nm, x in write_sets.items()
        ],
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(args.out, report)
    print(f"[phase4] [phase=energy_done] -> {args.out}", flush=True)
    return 0


def _rb_traits(rb_dir: Path) -> list[str]:
    return sorted(p.stem for p in Path(rb_dir).glob("*.pt"))


def _load_j(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj
    for key in ("J", "j", "matrix"):
        if key in obj:
            return obj[key]
    raise KeyError(f"no J tensor in {path}: keys={sorted(obj)}")


def _load_acts(path: Path, h: int) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    t = (
        obj
        if isinstance(obj, torch.Tensor)
        else obj[next(k for k in ("acts", "cx", "x") if k in obj)]
    )
    assert t.ndim == 2 and t.shape[1] == h, (t.shape, h)
    return t.to(torch.float32)


# ── projector from the dictionary (subspace read) ────────────────────────────


def dict_projector(d: dict, *, pj_energy: float, pj_rank: int | None, device: str):
    """Top-r right-singular subspace of the dictionary (Gram-space eigh)."""
    rows = d["rows_unit"].to(device, torch.float32)
    gram = rows.T @ rows  # (H, H)
    try:
        w, v = torch.linalg.eigh(gram)
    except torch.linalg.LinAlgError:
        print("[phase4] eigh non-convergence on device — CPU fallback (gotchas recipe)")
        w, v = torch.linalg.eigh(gram.cpu())
        w, v = w.to(device), v.to(device)
    w = w.flip(0).clamp_min(0)
    v = v.flip(1)
    if pj_rank is None:
        frac = torch.cumsum(w, 0) / w.sum()
        pj_rank = int((frac < pj_energy).sum().item() + 1)
    basis = v[:, :pj_rank].contiguous()  # (H, r)
    spectrum = w[:pj_rank].cpu()
    return basis, pj_rank, spectrum, w.cpu()


def cmd_refit_split(args) -> int:
    import issue1776_comparator_fit as CF

    import issue779_common as C
    import issue779_ffc_n1m_fits as N1M

    dev = args.device
    d19 = load_dict(args.dict19, dev)
    basis, r, spectrum, full_w = dict_projector(
        d19, pj_energy=args.pj_energy, pj_rank=args.pj_rank, device=dev
    )
    h = basis.shape[0]

    if args.smoke_synthetic:
        rng = np.random.default_rng(args.seed)
        n = 2600
        x = rng.standard_normal((n, h)).astype(np.float32)
        b_np = basis.cpu().numpy().astype(np.float64)
        # planted map: y lives ENTIRELY in the P_J subspace -> split must show it
        w_true = rng.standard_normal((h, r)) / np.sqrt(h)
        y = ((x @ w_true) @ b_np.T + 0.05 * rng.standard_normal((n, h))).astype(np.float32)
        tr, val, te = np.arange(2000), np.arange(2000, 2300), np.arange(2300, 2600)
    else:
        ns = argparse.Namespace(
            # N1M contract (N1G._load_pass_b_bundle): pass_b must be a real Path
            # (None -> AttributeError at .exists()). Crash-fix r7: resolve the
            # CLI's default=None to the reused module's own constant.
            pass_b=args.pass_b if args.pass_b is not None else N1M.N1G.PASS_B_LOCAL,
            out_dir=args.assemble_out_dir,
            manifest_from_hf=True,
            manifest_hf_prefix="issue779_monitoring/fitter-fair-comparison-n1m",
            # N1M contract (issue779_ffc_n1m_fits.py L949-953): ns.hf_prefix is the
            # CAPTURE prefix (the chunk stream flat-joins <hf_prefix>/<name>.pt);
            # only manifest_hf_prefix is the round root. Crash-fix r6 — same
            # wrong-prefix class as the comparator p0 404 (att-20260729-082617).
            hf_prefix=f"{N1M.N1G.HF_PREFIX}/final_token_capture",
            n1m_capture_dir=None,
            mm_dir=args.mm_dir,
            # N1M contract (N50._pinned_original_shas): orig_dir must be a real dir
            # holding the original round's fair_comparison.json. Crash-fix r7
            # (att-20260729-060640): None crashed `None / "fair_comparison.json"`.
            orig_dir=N1M.DEFAULT_ORIG_DIR,
            fresh_stream=False,
            prefetch=2,
            max_chunks=None,
        )
        # BOTH layers, matching the comparator's completed memmap cursor
        # (layers==[14,19]): a [19]-only request MISMATCHES the cursor
        # (issue779_ffc_n1m_fits.py L632-641) and silently WIPES + re-streams
        # all 1920 chunks (~hours), destroying the shared memmap phase5 then
        # re-streams AGAIN. Crash-fix r7 assemble-ns audit.
        layers = [C76.SOURCE_LAYER, C76.READOUT_LAYER]
        per_layer, prov, _orig, val, te, _split = N1M.assemble_multilayer(ns, layers)
        x, y = per_layer[C76.READOUT_LAYER]  # cx_last(19) -> v(19): the shipped-M family
        assert x.shape[1] == C.EXPECTED_HIDDEN == h, (x.shape, h)
        tr = CF.select_train_rows(prov, val, te, args.n_train, args.seed, False)

    p_np = (basis @ basis.T).cpu().numpy().astype(np.float32)
    gen = torch.Generator().manual_seed(args.seed + 1)
    rand_basis = torch.linalg.qr(torch.randn(h, r, generator=gen))[0]
    pr_np = (rand_basis @ rand_basis.T).numpy().astype(np.float32)

    y64 = np.asarray(y, dtype=np.float64)
    targets = {
        "full": np.asarray(y, dtype=np.float32),
        "pj": (y64 @ p_np.T).astype(np.float32),
        "perp": (y64 @ (np.eye(h, dtype=np.float64) - p_np.T)).astype(np.float32),
        "random_subspace_ref": (y64 @ pr_np.T).astype(np.float32),
    }
    y_te_c = y64[np.asarray(te)] - y64[np.asarray(te)].mean(0)
    var_frac_pj = float(((y_te_c @ p_np.T) ** 2).sum() / (y_te_c**2).sum())
    var_frac_rand = float(((y_te_c @ pr_np.T) ** 2).sum() / (y_te_c**2).sum())

    fits = {}
    for tag, yk in targets.items():
        pred, meta, _payload = N1M.fit_ridge_with_weights(
            np.asarray(x, dtype=np.float32),
            yk,
            tr,
            val,
            te,
            C76.EXTENDED_LAMBDA_GRID,
            dev,
            args.ridge_block,
        )
        yk_te = np.asarray(yk[np.asarray(te)], dtype=np.float64)
        ss_res = float(((yk_te - pred) ** 2).sum())
        ss_tot = float(((yk_te - yk_te.mean(0)) ** 2).sum())
        fits[tag] = {"test_pooled_r2": 1.0 - ss_res / ss_tot, "meta": meta}
        print(f"[phase4] [refit] {tag}: r2={fits[tag]['test_pooled_r2']:.4f}", flush=True)

    report = {
        "pj_rank": int(r),
        "pj_energy_threshold": args.pj_energy,
        "pj_spectrum_head": spectrum[:10].tolist(),
        "dict_spectrum_total": float(full_w.sum()),
        "n_train": int(len(tr)),
        "d": int(h),
        "variance_fraction_pj": var_frac_pj,
        "variance_fraction_random_subspace": var_frac_rand,
        "fits": fits,
        "r2_fraction_pj_over_full": (
            fits["pj"]["test_pooled_r2"] / fits["full"]["test_pooled_r2"]
            if fits["full"]["test_pooled_r2"] > 0
            else None
        ),
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(args.out, report)
    print(f"[phase4] [phase=refit_split_done] -> {args.out}", flush=True)
    return 0


def cmd_jdelta_split(args) -> int:
    """Per-cell steering-prediction accuracy of J·δ_J vs J·δ_⊥ (plan §4 P4 iii)."""
    import issue1776_phase3 as P3

    dev = args.device
    d14 = load_dict(args.dict14, dev)
    basis, r, _spec, _w = dict_projector(
        d14, pj_energy=args.pj_energy, pj_rank=args.pj_rank, device=dev
    )
    proj = basis @ basis.T
    root = args.phase3_root
    bank = torch.load(root / "directions.pt", map_location="cpu", weights_only=False)["bank"]
    j = _load_j(args.jlast).to(dev, torch.float32)
    base = P3.load_base_means({"summaries": root / "summaries"})

    rows = []
    for p in sorted((root / "summaries").glob("*.pt")):
        if p.stem == "baseline_a0":
            continue
        st = torch.load(p, map_location="cpu", weights_only=True)
        name, alpha = _parse_stratum(p.stem, bank)
        if name is None:
            continue
        delta = bank[name].to(dev, torch.float32)
        d_j = proj @ delta
        d_p = delta - d_j
        pred_j = (j @ (alpha * d_j)).cpu().to(torch.float64)
        pred_p = (j @ (alpha * d_p)).cpu().to(torch.float64)
        for cid, per in st["v19"].items():
            if cid not in base or per.shape[0] == 0:
                continue
            dv = per.to(torch.float64).mean(0) - base[cid]
            rows.append(
                {
                    "stratum": p.stem,
                    "direction": name,
                    "alpha": alpha,
                    "context_id": cid,
                    "dv_norm": float(dv.norm()),
                    "cos_pred_j_component": _cos64(dv, pred_j),
                    "cos_pred_perp_component": _cos64(dv, pred_p),
                    "delta_j_norm_frac": float(d_j.norm() / delta.norm()),
                }
            )
    assert rows, f"no steered cells under {root}"
    per_stratum: dict[str, dict] = {}
    for s in sorted({r_["stratum"] for r_ in rows}):
        sub = [r_ for r_ in rows if r_["stratum"] == s]
        per_stratum[s] = {
            "n_cells": len(sub),
            "mean_cos_j": float(np.mean([r_["cos_pred_j_component"] for r_ in sub])),
            "mean_cos_perp": float(np.mean([r_["cos_pred_perp_component"] for r_ in sub])),
            "delta_j_norm_frac": sub[0]["delta_j_norm_frac"],
        }
    report = {
        "pj_rank": int(r),
        "pj_energy_threshold": args.pj_energy,
        "per_stratum": per_stratum,
        "per_cell": rows,
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(args.out, report)
    print(f"[phase4] [phase=jdelta_done] strata={len(per_stratum)} -> {args.out}", flush=True)
    return 0


def _cos64(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = float(a.norm()), float(b.norm())
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((a @ b) / (na * nb))


def _parse_stratum(stem: str, bank: dict) -> tuple[str | None, float]:
    """'evil_a2' -> ('evil', 2.0); all_positions strata ('*_allpos') skipped."""
    if stem.endswith("_allpos") or "_a" not in stem:
        return None, 0.0
    name, _, a = stem.rpartition("_a")
    if name not in bank:
        return None, 0.0
    return name, float(a)


# ── smoke ─────────────────────────────────────────────────────────────────────


def smoke(args) -> int:
    """Tiny synthetic CPU e2e over every sub-command body (plan smoke parity)."""
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    h, v, k, draws = 32, 200, 4, 8
    # seed 99 != ns.seed: a shared seed makes null_probe_rows' isotropic draws
    # REPLAY the dictionary's own randn stream (null rows == dict atoms 0..n,
    # null energy pinned at 1.0) — a smoke-only RNG collision, asserted below.
    gen = torch.Generator().manual_seed(99)
    rows = torch.randn(v, h, generator=gen)
    rows = rows / rows.norm(dim=1, keepdim=True)
    for layer, name in ((3, "dict14"), (5, "dict19")):
        torch.save(
            {"rows_unit": rows.to(torch.float16), "row_norms": torch.ones(v), "layer": layer},
            out / f"{name}.pt",
        )
    # in-dictionary probes must reconstruct ~fully; random probes near null band.
    x_in = rows[:6] * 3.0
    e_pur, e_span = pursuit_energies(x_in, rows.to(args.device), k)
    assert e_span.min() > 0.99, e_span
    assert e_pur.mean() > 0.9, e_pur

    # rb bank leg (crash-fix r10 regression pin): production-shaped #779 per-trait
    # (L, H) stack so the smoke EXERCISES cmd_energy's rb_dir branch — the
    # P3.load_directions cross-script namespace seam the r9 launch crashed on
    # (AttributeError: source_layer). layers must include C76.SOURCE_LAYER.
    rb_dir = out / "rb"
    rb_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "r_b": torch.randn(2, h, generator=gen),
            "layers": [C76.SOURCE_LAYER, C76.READOUT_LAYER],
        },
        rb_dir / "evil.pt",
    )
    ns = argparse.Namespace(
        k=k,
        n_draws=draws,
        chunk=64,
        seed=0,
        device=args.device,
        dict14=out / "dict14.pt",
        dict19=out / "dict19.pt",
        acts14=out / "acts.pt",  # cov null binds on BOTH sides (read L14 + write L19)
        acts19=out / "acts.pt",
        mprime_weights=str(out / "mprime.pt"),
        jlast=str(out / "jlast.pt"),
        rb_dir=rb_dir,
        phase3_root=None,
        topk=5,
        out=out / "jspace_energy.json",
    )
    torch.save(torch.randn(300, h, generator=gen), out / "acts.pt")
    torch.save({"W": torch.randn(h, h, generator=gen), "xmu": torch.zeros(h)}, out / "mprime.pt")
    torch.save(torch.randn(h, h, generator=gen), out / "jlast.pt")
    rc = cmd_energy(ns)
    assert rc == 0
    rep = json.loads((out / "jspace_energy.json").read_text())
    for side in ("read_side", "write_side"):
        for rec in rep[side]:
            for fam, nl in rec["nulls"].items():
                assert 0.0 <= nl["pursuit_p975"] <= 1.0, (fam, nl)
                assert nl["pursuit_band_to_ceiling"] >= 0.0, (fam, nl)
                # a null band pinned at the ceiling = degenerate band (or the
                # smoke RNG collision above): uninformative by construction.
                assert nl["pursuit_p975"] < 0.99, (rec["probe_set"], fam, nl)
    assert {r["probe_set"] for r in rep["read_side"]} == {
        "mprime_right_top20",
        "jlast_rowspace_top20",
        "rb_layer14_rows",
    }

    # refit-split round-trip on a planted P_J-subspace map.
    ns2 = argparse.Namespace(
        dict19=out / "dict19.pt",
        pj_energy=0.8,
        pj_rank=8,
        device=args.device,
        smoke_synthetic=True,
        seed=0,
        ridge_block=512,
        n_train=2000,
        pass_b=None,
        assemble_out_dir=None,
        mm_dir=None,
        out=out / "refit_split.json",
    )
    assert cmd_refit_split(ns2) == 0
    rs = json.loads((out / "refit_split.json").read_text())
    assert rs["fits"]["pj"]["test_pooled_r2"] > 0.8, rs["fits"]
    assert rs["fits"]["pj"]["test_pooled_r2"] > rs["fits"]["perp"]["test_pooled_r2"] + 0.3, rs[
        "fits"
    ]

    # jdelta-split round-trip on a synthetic phase-3 root.
    root = out / "p3root"
    (root / "summaries").mkdir(parents=True, exist_ok=True)
    bank = {"evil": torch.randn(h, generator=gen)}
    bank["evil"] /= bank["evil"].norm()
    torch.save({"bank": bank, "provenance": {}}, root / "directions.pt")
    base = {"c0": torch.randn(2, h, generator=gen)}
    torch.save({"v19": base, "stratum": "baseline_a0"}, root / "summaries" / "baseline_a0.pt")
    torch.save(
        {"v19": {"c0": base["c0"] + 0.5}, "stratum": "evil_a2"},
        root / "summaries" / "evil_a2.pt",
    )
    ns3 = argparse.Namespace(
        dict14=out / "dict14.pt",
        pj_energy=0.8,
        pj_rank=8,
        device=args.device,
        phase3_root=root,
        jlast=str(out / "jlast.pt"),
        out=out / "jdelta_split.json",
    )
    assert cmd_jdelta_split(ns3) == 0
    js = json.loads((out / "jdelta_split.json").read_text())
    assert js["per_stratum"]["evil_a2"]["n_cells"] == 1
    frac = js["per_stratum"]["evil_a2"]["delta_j_norm_frac"]
    assert 0.0 < frac < 1.0, frac
    print("[phase4] [phase=smoke_done] PASS", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-dict")
    b.add_argument("--lens", type=Path, required=True)
    b.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    b.add_argument("--layer", type=int, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    b.add_argument("--tiny", action="store_true")

    e = sub.add_parser("energy")
    e.add_argument("--dict14", type=Path, required=True)
    e.add_argument("--dict19", type=Path, required=True)
    e.add_argument("--mprime-weights", type=Path)
    e.add_argument("--jlast", type=Path)
    e.add_argument("--rb-dir", type=Path)
    e.add_argument("--phase3-root", type=Path)
    e.add_argument("--acts14", type=Path, help="(n,H) layer-14 acts for the cov null")
    e.add_argument("--acts19", type=Path, help="(n,H) layer-19 acts for the cov null")
    e.add_argument("--k", type=int, default=25)
    e.add_argument("--topk", type=int, default=20)
    e.add_argument("--n-draws", type=int, default=100)
    e.add_argument("--chunk", type=int, default=256)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    e.add_argument("--out", type=Path, required=True)

    r = sub.add_parser("refit-split")
    r.add_argument("--dict19", type=Path, required=True)
    r.add_argument("--pj-energy", type=float, default=0.95)
    r.add_argument("--pj-rank", type=int, default=None)
    r.add_argument("--n-train", type=int, default=50_000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--ridge-block", type=int, default=8192)
    r.add_argument("--pass-b", type=Path, default=None)
    r.add_argument("--assemble-out-dir", type=Path, default=C76.DATA_DIR / "ffc_n1m")
    r.add_argument("--mm-dir", type=Path, default=C76.DATA_DIR / "n1m_mm")
    r.add_argument("--smoke-synthetic", action="store_true")
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    r.add_argument("--out", type=Path, required=True)

    j = sub.add_parser("jdelta-split")
    j.add_argument("--dict14", type=Path, required=True)
    j.add_argument("--jlast", type=Path, required=True)
    j.add_argument("--phase3-root", type=Path, required=True)
    j.add_argument("--pj-energy", type=float, default=0.95)
    j.add_argument("--pj-rank", type=int, default=None)
    j.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    j.add_argument("--out", type=Path, required=True)

    s = sub.add_parser("smoke")
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--device", default="cpu")

    args = ap.parse_args(argv)
    if args.cmd == "build-dict":
        rc = cmd_build_dict(args)
    elif args.cmd == "energy":
        rc = cmd_energy(args)
    elif args.cmd == "refit-split":
        if not args.smoke_synthetic:
            assert args.pass_b is not None, "--pass-b required outside --smoke-synthetic"
        rc = cmd_refit_split(args)
    elif args.cmd == "jdelta-split":
        rc = cmd_jdelta_split(args)
    else:
        rc = smoke(args)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit: PyGILState atexit race (gotchas)


if __name__ == "__main__":
    main()
