"""Residual-stream activation extraction + pooled-LOPO logistic-regression probes.

Used by ``scripts/run_issue_358_extract.py`` (activation extraction) and
``scripts/analyze_issue_358_probe.py`` (probe + null distributions). Kept
deliberately small — anything else (PCA, UMAP, plotting) lives in the
matching analysis scripts.

Conventions
-----------
* Activations are extracted at the **last input token** of the formatted
  ChatML prompt — the position the model is about to generate from. This
  matches the read-out used in ``scripts/run_issue_276_pre_poison_similarity.py``
  and Anthropic's "Simple probes" sleeper-agent setup.
* The forward pass uses ``output_hidden_states=True``. The returned tuple
  has length ``num_hidden_layers + 1``: index 0 is the embedding output and
  indices 1..L are the residual stream after each transformer block. The
  helper returns ``out[1:]`` re-indexed so layer ``L`` (0-indexed) is at
  position ``L`` in the returned tensor.
* All cached activations are stored as ``torch.float32`` on CPU. The
  forward pass runs in the model's native dtype (bf16 for Qwen3-4B); we
  upcast at the very last step to make downstream sklearn / numpy work
  numerically stable.

The pooled-LOPO probe trains a class-balanced L2 logistic regression on
``n_pool - 1`` prompts, predicts the held-out one's decision-function
score, then computes a single AUROC on the pooled (score, y) pairs.
A 1000-resample prompt-level bootstrap CI is reported alongside.
**Per-fold StandardScaler — never reuse a global scaler fit on the full
panel, because that leaks held-out variance into the training fold's
normalisation.** This is distinct from any PCA-global StandardScaler in
``scripts/analyze_issue_358_pca.py`` — these two scalers must stay
separate.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Activation extraction
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def extract_residual_stream_activations(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    layers: Sequence[int] | None = None,
    device: str | torch.device | None = None,
    position: int = -1,
) -> torch.Tensor:
    """Extract residual-stream activations at one input position per prompt.

    Each prompt is tokenised (no padding, batch size 1), forwarded with
    ``output_hidden_states=True``, and the activation at ``position``
    (default ``-1`` = last input token) is read off every requested layer
    and stacked into a ``(n_prompts, n_layers, hidden_size)`` fp32 tensor
    on CPU.

    Parameters
    ----------
    model
        A causal-LM with ``config.num_hidden_layers`` and
        ``config.hidden_size`` set. Must already be on ``device`` and in
        eval mode; this helper does not move or set the mode.
    tokenizer
        The matching ``AutoTokenizer``. Used with ``return_tensors="pt"``
        and ``add_special_tokens=False`` (callers are expected to bake
        their own ChatML special tokens into the prompt string).
    prompts
        Iterable of formatted prompt strings.
    layers
        Which 0-indexed transformer-block layers to keep. ``None`` =
        every layer in the model.
    device
        Device to move input tensors to. Defaults to ``model``'s device.
    position
        Token position to read at. ``-1`` (default) = last input token.

    Returns
    -------
    torch.Tensor
        Shape ``(len(prompts), len(layers), hidden_size)``, fp32, on CPU.
    """
    n_layers_total = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    if layers is None:
        layers = list(range(n_layers_total))
    layers = list(layers)
    for L in layers:
        if not (0 <= L < n_layers_total):
            raise ValueError(f"layer index {L} out of range for model with {n_layers_total} layers")

    if device is None:
        device = next(model.parameters()).device

    out = torch.zeros(len(prompts), len(layers), hidden, dtype=torch.float32)
    for i, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        ids = ids.to(device)
        hs = model(ids, output_hidden_states=True).hidden_states
        # hs is a tuple of length (num_hidden_layers + 1).
        # hs[0] = embedding output; hs[L+1] = output of transformer block L.
        for j, L in enumerate(layers):
            out[i, j] = hs[L + 1][0, position].float().cpu()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Probes + null distributions
# ─────────────────────────────────────────────────────────────────────────────


def _fit_lr(Xz_tr: np.ndarray, y_tr: np.ndarray, C: float, seed: int) -> LogisticRegression:
    """Fit one L2-regularised logistic regression with class-balanced weights."""
    clf = LogisticRegression(
        C=C,
        penalty="l2",
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(Xz_tr, y_tr)
    return clf


def pooled_lopo_probe(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    seed: int = 42,
    n_bootstrap: int = 1000,
) -> dict:
    """Pooled-prediction Leave-One-Prompt-Out probe.

    Train L2-regularised logistic regression on ``n-1`` prompts, predict
    the held-out prompt's decision-function score; accumulate scores into
    a single ``n``-length vector and compute AUROC once on the pooled
    (score, y) pairs. Bootstrap a 95% prompt-level CI by resampling
    indices with replacement ``n_bootstrap`` times.

    Per-fold StandardScaler — fit on the training fold only, never on
    the full panel. (The PCA-global scaler in
    ``scripts/analyze_issue_358_pca.py`` is a *different* scaler; do not
    pass it in here.)

    Parameters
    ----------
    X
        Activation matrix, shape ``(n_pool, hidden)``.
    y
        Binary labels in ``{0, 1}``, shape ``(n_pool,)``.
    C
        Inverse regularisation strength. Default 1.0 (sklearn default,
        matches MacDiarmid 2024 / Anthropic blog).
    seed
        RNG seed for both the LR optimiser and the bootstrap resampler.
    n_bootstrap
        Number of bootstrap resamples for the CI. Draws that lose a class
        are silently dropped (counted in ``n_bootstrap_dropped``).

    Returns
    -------
    dict
        ``pooled_auroc``, ``ci_95`` ([lo, hi]), ``train_auroc``,
        ``n_pool``, ``n_pos``, ``n_neg``, ``n_bootstrap_dropped``,
        ``fold_scores`` (per-prompt held-out decision-function score).
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    n = X.shape[0]
    if n != y.shape[0]:
        raise ValueError(f"X and y must have matching length; got {X.shape[0]} vs {y.shape[0]}")
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError(f"y must be binary 0/1, got unique values {np.unique(y).tolist()}")
    if len(np.unique(y)) < 2:
        raise ValueError(f"y must contain both classes; got {np.unique(y).tolist()}")

    # ─── Pooled-LOPO held-out scores ────────────────────────────────
    scores = np.zeros(n)
    for tr, te in LeaveOneOut().split(X):
        scaler = StandardScaler().fit(X[tr])  # per-fold, not global
        Xz_tr = scaler.transform(X[tr])
        Xz_te = scaler.transform(X[te])
        clf = _fit_lr(Xz_tr, y[tr], C=C, seed=seed)
        scores[te] = clf.decision_function(Xz_te)

    pooled_auroc = float(roc_auc_score(y, scores))

    # ─── Prompt-level bootstrap CI ──────────────────────────────────
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ys, ss = y[idx], scores[idx]
        if len(np.unique(ys)) < 2:
            continue  # bootstrap draw lost a class — drop, don't crash
        boot.append(roc_auc_score(ys, ss))
    if not boot:
        raise RuntimeError(
            f"every one of {n_bootstrap} bootstrap draws lost a class — "
            f"can't compute CI (n_pos={int(y.sum())}, n_neg={int((1 - y).sum())})"
        )
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    # ─── Train-AUROC regime-confirmation pass ───────────────────────
    full_scaler = StandardScaler().fit(X)
    full_clf = _fit_lr(full_scaler.transform(X), y, C=C, seed=seed)
    train_auroc = float(roc_auc_score(y, full_clf.decision_function(full_scaler.transform(X))))

    return {
        "pooled_auroc": pooled_auroc,
        "ci_95": [float(ci_lo), float(ci_hi)],
        "train_auroc": train_auroc,
        "n_pool": int(n),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "n_bootstrap_dropped": int(n_bootstrap - len(boot)),
        "fold_scores": scores.tolist(),
    }


def shuffled_label_null(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_perm: int = 200,
    C: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Null distribution of pooled-LOPO AUROC under label shuffles.

    Returns
    -------
    np.ndarray
        Shape ``(n_perm,)`` — pooled-LOPO AUROC under each label permutation.
        Permutations that produce a degenerate pool are dropped, so the
        returned length may be less than ``n_perm``.
    """
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_perm):
        y_perm = rng.permutation(y)
        try:
            r = pooled_lopo_probe(X, y_perm, C=C, seed=seed + k + 1, n_bootstrap=1)
        except (ValueError, RuntimeError):
            continue  # degenerate permutation lost a class
        out.append(r["pooled_auroc"])
    return np.asarray(out, dtype=float)


def random_projection_null(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_proj: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """AUROC of random unit hyperplanes — probe-class baseline.

    Replaces the trained probe with a random unit vector ``w``; reports
    ``roc_auc_score(y, X @ w)`` for each draw. No training, no fold
    structure — this isolates "is any linear function high-AUROC by
    chance?" from "does the trained probe find structure?".

    Returns
    -------
    np.ndarray
        Shape ``(n_proj,)``.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(n_proj)
    for i in range(n_proj):
        w = rng.standard_normal(X.shape[1])
        w /= np.linalg.norm(w)
        out[i] = roc_auc_score(y, X @ w)
    return out
