# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, σ, λ, ×, →, —) in scientific docstrings + logs.
"""Arm A — the training-free write→read map ρ through the token bottleneck.

Pipeline (plan §4 Phase A):

* **A0** base-model residual covariance Σ + per-layer RMS at ℓ over the prompt
  set 𝒬 (one forward pass).
* **A1** random-write steering: add a bias ``m·w`` to the layer-ℓ residual
  during generation, sample A_w (vLLM), coherence-filter.
* **A2** read: unsteered teacher-force on A_w → response-mean pool at ℓ' → ρ(w).
* **A3** fit J (ridge, CV-λ) with ``ρ(w) ≈ J w``; SVD(J) + SVD of the stacked
  ρ(w_i) cloud; round-trip cos(w, ρ(w)).
* **A4** structured-write probe: ρ(d_B) per behavior → cos to r_B vs the #503
  random CI.

The GPU steering + generation live in :func:`steer_and_sample` /
:func:`read_unsteered` (thin HF/vLLM harnesses); the fit + geometry
(:func:`fit_ridge_jacobian`, :func:`round_trip_cosines`, :func:`covariance_rms`)
are pure-numpy and CPU-smoke-testable.

The random-write distributions (§5): ``iso`` = isotropic Gaussian; ``cov`` =
residual-covariance-matched (A7 isotropy control). The covariance is estimated
from base-model residuals at ℓ over 𝒬 in A0.
"""

from __future__ import annotations

import numpy as np

from . import ARM_A_DISTRIBUTIONS


def covariance_rms(residuals: np.ndarray) -> dict:
    """A0: per-layer residual covariance Σ + RMS from a (n, d) residual matrix.

    Args:
        residuals: (n_samples, d_model) base-model residual vectors at layer ℓ
            over the prompt set (last-token or response-mean — caller's choice;
            the write distributions are built from whatever pool is passed).

    Returns:
        ``{rms, cov, mean}`` where rms is the scalar root-mean-square residual
        norm component (used to scale write magnitudes), cov is the (d, d)
        covariance, mean is the (d,) mean residual.
    """
    X = np.asarray(residuals, dtype=np.float64)
    assert X.ndim == 2, X.shape
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    # per-coordinate RMS magnitude — the layer-comparable scale (#623 convention).
    rms = float(np.sqrt((X**2).sum(axis=1).mean()))
    return {"rms": rms, "cov": cov, "mean": mean}


def sample_write_directions(
    *,
    d_model: int,
    n: int,
    distribution: str,
    cov: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Sample ``n`` unit write directions from ``distribution`` (iso | cov).

    iso = isotropic Gaussian normalized to unit norm. cov =
    residual-covariance-matched (sampled from N(0, Σ), normalized) — the A7
    isotropy control. Magnitudes are applied separately (m·RMS), so these are
    unit vectors.
    """
    if distribution not in ARM_A_DISTRIBUTIONS:
        raise ValueError(f"unknown distribution {distribution!r}; want {ARM_A_DISTRIBUTIONS}")
    rng = np.random.default_rng(seed)
    if distribution == "iso":
        V = rng.standard_normal(size=(n, d_model))
    else:  # cov
        if cov is None:
            raise ValueError("distribution='cov' needs the residual covariance Σ (from A0)")
        # Sample from N(0, Σ) via Cholesky (jitter for PSD safety).
        d = cov.shape[0]
        L = np.linalg.cholesky(cov + 1e-6 * np.eye(d))
        V = rng.standard_normal(size=(n, d)) @ L.T
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    return V


def fit_ridge_jacobian(
    W: np.ndarray,
    Rho: np.ndarray,
    *,
    lambdas: np.ndarray | None = None,
    n_folds: int = 5,
    seed: int = 0,
) -> dict:
    """A3: fit ρ(w) ≈ J w via ridge regression with CV-picked λ.

    W: (n, d) write vectors (already magnitude-scaled). Rho: (n, d) read-back
    response-mean shifts ρ(w). Solves min_J ||Rho − W Jᵀ||² + λ||J||² per
    output coordinate jointly (closed-form ridge), choosing λ by ``n_folds``
    CV on held-out reconstruction MSE.

    Returns ``{J, lambda, cv_mse, r2}`` — J is (d_out, d_in), r2 the
    variance-explained on the full fit.
    """
    W = np.asarray(W, dtype=np.float64)
    Rho = np.asarray(Rho, dtype=np.float64)
    assert W.ndim == 2 and Rho.ndim == 2 and W.shape[0] == Rho.shape[0], (W.shape, Rho.shape)
    n = W.shape[0]
    if lambdas is None:
        lambdas = np.logspace(-2, 4, 13)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)

    def _solve(Wtr: np.ndarray, Rtr: np.ndarray, lam: float) -> np.ndarray:
        # Closed-form ridge: J = (Wᵀ W + λI)⁻¹ Wᵀ R  → Jᵀ has shape (d_in, d_out)
        A = Wtr.T @ Wtr + lam * np.eye(Wtr.shape[1])
        return np.linalg.solve(A, Wtr.T @ Rtr)  # (d_in, d_out)

    best_lam, best_mse = lambdas[0], np.inf
    for lam in lambdas:
        mses = []
        for fi in range(n_folds):
            te = folds[fi]
            tr = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
            if tr.size == 0 or te.size == 0:
                continue
            Jt = _solve(W[tr], Rho[tr], lam)
            pred = W[te] @ Jt
            mses.append(float(((pred - Rho[te]) ** 2).mean()))
        mse = float(np.mean(mses)) if mses else np.inf
        if mse < best_mse:
            best_mse, best_lam = mse, lam

    Jt = _solve(W, Rho, best_lam)  # (d_in, d_out)
    J = Jt.T  # (d_out, d_in)
    pred = W @ Jt
    ss_res = float(((pred - Rho) ** 2).sum())
    ss_tot = float(((Rho - Rho.mean(axis=0)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {"J": J, "lambda": float(best_lam), "cv_mse": float(best_mse), "r2": float(r2)}


def round_trip_cosines(W: np.ndarray, Rho: np.ndarray) -> np.ndarray:
    """A3: per-write cos(w, ρ(w)) — does a write re-read as itself?

    Returns a (n,) array of cosines. Compared against the #503 norm-matched
    random-direction CI by the caller (see spectral.norm_matched_random_cos_ci).
    """
    W = np.asarray(W, dtype=np.float64)
    Rho = np.asarray(Rho, dtype=np.float64)
    assert W.shape == Rho.shape, (W.shape, Rho.shape)
    wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    rn = Rho / (np.linalg.norm(Rho, axis=1, keepdims=True) + 1e-12)
    return (wn * rn).sum(axis=1)


def apply_jacobian(J: np.ndarray, d_B: np.ndarray) -> np.ndarray:
    """A4: ρ(d_B) ≈ J d_B — push a structured write through the fitted map."""
    J = np.asarray(J, dtype=np.float64)
    d = np.asarray(d_B, dtype=np.float64).ravel()
    assert J.shape[1] == d.shape[0], (J.shape, d.shape)
    return J @ d


# ── GPU harnesses (thin; exercised only on the pod / GPU smoke) ──────────────


def steer_and_sample(
    model_path: str,
    personas: dict[str, str | None],
    questions: list[str],
    *,
    layer: int,
    write_unit: np.ndarray,
    magnitude_abs: float,
    max_new_tokens: int = 512,
    gpu_memory_utilization: float = 0.85,
    device: str = "cuda:0",
):
    """A1: HF generation with a constant residual-stream bias ``magnitude_abs *
    write_unit`` added at decoder block ``layer`` during the forward pass.

    Uses an HF forward hook (vLLM has no clean residual-injection hook), greedy
    decode. Returns one row dict per (persona, question): ``{persona,
    question_idx, prompt_token_ids, response_token_ids, finish_reason}`` — the
    SAME row shape ``representation_shift._teacher_forced_response_mean``
    consumes, so the A2 read reuses that engine verbatim.

    GPU-only; not exercised by the CPU smoke. Kept thin so the fit / geometry
    above carry the testable logic.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": device}, trust_remote_code=True
    )
    model.eval()

    bias = torch.tensor(
        magnitude_abs * np.asarray(write_unit, dtype=np.float32),
        dtype=torch.bfloat16,
        device=device,
    )

    def hook_fn(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs + bias  # broadcast over (B, T, d)
        if isinstance(out, tuple):
            return (hs, *out[1:])
        return hs

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    rows: list[dict] = []
    try:
        for p_name, p_prompt in personas.items():
            for q_idx, question in enumerate(questions):
                msgs = ([{"role": "system", "content": p_prompt}] if p_prompt else []) + [
                    {"role": "user", "content": question}
                ]
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                prompt_len = inputs["input_ids"].shape[1]
                resp_ids = gen[0, prompt_len:].tolist()
                eos = tokenizer.eos_token_id
                finish = "stop" if (resp_ids and resp_ids[-1] == eos) else "length"
                if resp_ids and resp_ids[-1] == eos:
                    resp_ids = resp_ids[:-1]
                rows.append(
                    {
                        "persona": p_name,
                        "question_idx": q_idx,
                        "prompt_token_ids": inputs["input_ids"][0].tolist(),
                        "response_token_ids": resp_ids,
                        "finish_reason": finish,
                    }
                )
    finally:
        handle.remove()
        del model
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    return rows


def coherence_pass_rate(
    rows: list[dict],
    base_logprob_floor: float,
    *,
    max_3gram_repeat_frac: float = 0.5,
) -> dict:
    """§4 coherence filter (code-not-judge): keep rows whose mean base log-prob ≥
    the 5th-percentile unsteered floor AND whose max 3-gram repetition fraction
    < ``max_3gram_repeat_frac``.

    Rows must carry ``mean_base_logprob`` (computed by the A2 unsteered read).
    Returns ``{pass_rate, n, kept_mask}``. The §7 gate fires on pass_rate ≥ 0.5.
    """

    def _rep_frac(ids: list[int]) -> float:
        if len(ids) < 3:
            return 0.0
        grams = [tuple(ids[i : i + 3]) for i in range(len(ids) - 2)]
        if not grams:
            return 0.0
        from collections import Counter

        most = Counter(grams).most_common(1)[0][1]
        return most / len(grams)

    kept = []
    for r in rows:
        lp_ok = r.get("mean_base_logprob", float("-inf")) >= base_logprob_floor
        rep_ok = _rep_frac(r["response_token_ids"]) < max_3gram_repeat_frac
        kept.append(bool(lp_ok and rep_ok))
    n = len(rows)
    return {
        "pass_rate": (sum(kept) / n) if n else 0.0,
        "n": n,
        "kept_mask": kept,
    }
