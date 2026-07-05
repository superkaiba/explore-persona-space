#!/usr/bin/env python3
"""Issue #923 shared constants + helpers (context/query decomposition of v̄(c,q)).

Plan: tasks/running/923/plans/plan.md (v3). No training anywhere; ridge-only
read-outs. This module carries the pieces every #923 phase script shares:

- HF prefixes for the reused #658 stores + this issue's own upload bucket;
- prompt rendering for the four feature presentations (full / context-prefix /
  empty-system query / no-system-block query) with the fail-loud template
  asserts (Qwen silently inserts a default system prompt when none is given);
- the context-prefix token arithmetic (F_ctx = last token of the context block,
  an exact causal-mask identity with the same position inside the full prompt);
- the custom 4D attention mask for the masked-context query presentation
  (``arm_qry_iii``): RIGHT-padded batch, causal, query rows blocked from
  context columns, padded rows self-attend (0-weight x finite value, never NaN);
- fold assignment (7 LOFO context families x 4 stratified query folds);
- weighted exact PCA for few-distinct-row fold designs (the §8 rank-deficiency
  mitigation) mirroring ``vectorized_mlp_skill.robust_pca_basis``;
- pack IO (fp16 tensors + metadata) shared by capture / reduce / fit.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
V0_MAX_NEW_TOKENS = 512  # #658 G1 recipe (temperature 0.0), matched exactly
SEED = 42
N_QUERY_FOLDS = 4
HEADLINE_LAYER = 18  # frozen pre-registered headline layer (#722 peak)

DATA_DIR = PROJECT_ROOT / "data" / "issue923"
I594_DATA_DIR = PROJECT_ROOT / "data" / "issue594"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_923"
FIGURES_DIR = PROJECT_ROOT / "figures" / "issue_923"

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX_923 = "issue923_ctx_query_decomposition"
# Reused #658 stores (plan §10; Hub-verified at plan time). Keyed by genre.
STORE_PREFIXES: dict[str, str] = {
    "betley": "issue658_theory_assumptions/store",
    "uc": "issue658_theory_assumptions/store_genre-generalization-ultrachat",
}
GENRES: tuple[str, ...] = ("betley", "uc")

# Fixed family order (battery §594): used for LOFO fold order + blend rotation.
FAMILY_ORDER: tuple[str, ...] = (
    "persona",
    "wildchat",
    "icl",
    "rephrase",
    "format",
    "behavior",
    "default",
)

# Qwen2.5 chat template inserts this system prompt when messages carry no system
# turn. Presentation (i) must SUPPRESS it via an explicit empty system turn; the
# rendered-string assert below fails loud if the template's behavior drifts.
QWEN_DEFAULT_SYSTEM_SNIPPET = "You are Qwen, created by Alibaba Cloud."

ARMS_SINGLE: tuple[str, ...] = ("arm_ctx", "arm_qry_i", "arm_qry_ii", "arm_qry_iii")
ARMS_CONCAT: tuple[str, ...] = ("arm_concat_i", "arm_concat_ii", "arm_concat_iii")
ARM_FULL = "arm_full"
ALL_RIDGE_ARMS: tuple[str, ...] = ARMS_SINGLE + ARMS_CONCAT + (ARM_FULL,)

# Phase-0 output files (committed to the issue branch; see .gitignore whitelist).
UC_EXT_PROBES_PATH = DATA_DIR / "probes_uc_ext.json"
DOLLY_PROBES_PATH = DATA_DIR / "probes_dolly.json"
BETLEY_PROBES_PATH = DATA_DIR / "probes_betley.json"
FOLDS_PATH = DATA_DIR / "fold_assignments.json"
STORE_PINS_PATH = DATA_DIR / "store_pins.json"


# ── small IO helpers ──────────────────────────────────────────────────────────


def load_json(path: Path | str):
    """Read one JSON file."""
    with open(path) as f:
        return json.load(f)


def dump_json(obj, path: Path | str) -> None:
    """Atomic-ish JSON write (tmp + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def sha256_file(path: Path | str) -> str:
    """Whole-file sha256 hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def texts_hash(texts: list[str]) -> str:
    """Stable sha256 over an ordered text pool (matches issue594 probes_hash)."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


HF_PINS_PATH = DATA_DIR / "hf_pins.json"


def hf_revision(kind: str, repo_id: str) -> str:
    """Pinned HF revision for a model/dataset load (committed ``hf_pins.json``).

    Every issue923 HF model/dataset load passes ``revision=hf_revision(...)``
    so production cannot silently drift when the upstream repo moves (r1
    blocker ``hf-revision-pinning-missing``; store span files are pinned
    separately in ``store_pins.json``). An unpinned repo id fails loud —
    resolve its sha and add it to the pin file rather than loading unpinned.
    """
    assert kind in ("models", "datasets"), kind
    pins = load_json(HF_PINS_PATH)
    rev = pins.get(kind, {}).get(repo_id)
    assert rev, f"no pinned revision for {kind}:{repo_id} — add it to {HF_PINS_PATH}"
    return rev


# ── prompt rendering / prefix arithmetic ──────────────────────────────────────


def user_turn_suffix(q: str) -> str:
    """The final-user-turn + assistant-header suffix of a Qwen2.5 chat render."""
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def render_full_prompt(tokenizer, instance: dict, q: str) -> str:
    """Full (context, query) prompt text — the #658-locked c_C rendering."""
    from issue594_common import messages_for_instance

    return tokenizer.apply_chat_template(
        messages_for_instance(instance, q), tokenize=False, add_generation_prompt=True
    )


def context_prefix_split(tokenizer, instance: dict, q: str) -> tuple[list[int], list[int]]:
    """(prefix_ids, full_ids): the context-block prefix of the full-prompt tokens.

    The prefix is the full render minus the trailing ``user_turn_suffix(q)``;
    asserts (a) the render ends with that suffix and (b) the prefix tokenization
    is an exact token-id prefix of the full tokenization (special-token
    boundaries make this exact). For a ``default``-family instance with no
    system prompt the prefix is the template's auto-inserted default system
    block (§8 fallback — caller logs it). F_ctx read at ``prefix_ids[-1]`` in a
    prefix-only forward equals the same position inside the full prompt by
    causal masking (verified numerically by the capture identity check).
    """
    full_text = render_full_prompt(tokenizer, instance, q)
    suffix = user_turn_suffix(q)
    assert full_text.endswith(suffix), (
        f"full prompt for {instance['id']} does not end with the user-turn suffix; "
        f"tail={full_text[-120:]!r}"
    )
    prefix_text = full_text[: -len(suffix)]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    assert len(prefix_ids) >= 1, f"empty context prefix for {instance['id']}"
    assert full_ids[: len(prefix_ids)] == prefix_ids, (
        f"context prefix is not a token-prefix of the full prompt for {instance['id']} "
        f"(BPE boundary drift)"
    )
    return prefix_ids, full_ids


def render_qry_empty_system(tokenizer, q: str) -> str:
    """Presentation (i): explicit EMPTY system turn (suppresses the Qwen default).

    Fail-loud asserts: the default-system substring must NOT appear, and the
    render must end with the assistant header (plan §4.1 / §8).
    """
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}, {"role": "user", "content": q}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert QWEN_DEFAULT_SYSTEM_SNIPPET not in text, (
        "presentation (i) contaminated: the Qwen default system prompt was inserted "
        "despite the explicit empty system turn"
    )
    assert text.endswith(user_turn_suffix(q)), f"unexpected (i) render tail: {text[-80:]!r}"
    return text


def render_qry_no_system_block(q: str) -> str:
    """Presentation (ii): hand-rendered, no system block at all (template-free)."""
    return user_turn_suffix(q)


# ── 4D attention mask for the masked-context presentation (iii) ───────────────


def build_masked_context_4d_mask(
    ctx_lens: list[int],
    seq_lens: list[int],
    max_len: int,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Additive 4D attention mask for a RIGHT-padded masked-context batch.

    Row b holds a full (c,q) prompt of ``seq_lens[b]`` real tokens at positions
    [0, L_b) whose first ``ctx_lens[b]`` tokens are the context block. Rules
    (0 = attend, finfo.min = blocked):

    - causal everywhere (j <= i);
    - query rows (i >= c_b) blocked from context columns (j < c_b) — the
      masked-context intervention; positions preserved (right-pad keeps real
      tokens at their unpadded absolute positions, so default position_ids are
      correct);
    - context rows keep plain causal attention among themselves (their outputs
      are never attended by query rows, so they cannot leak);
    - padded rows (i >= L_b) attend ONLY to themselves — keeps their outputs
      finite so 0-weight contributions stay 0 (0 x NaN would poison real rows);
    - every real row keeps j == i (self) allowed, so no row is fully masked.
    """
    b = len(seq_lens)
    assert len(ctx_lens) == b
    neg = torch.finfo(dtype).min
    i_idx = torch.arange(max_len, device=device).view(1, max_len, 1)
    j_idx = torch.arange(max_len, device=device).view(1, 1, max_len)
    c = torch.tensor(ctx_lens, device=device).view(b, 1, 1)
    ell = torch.tensor(seq_lens, device=device).view(b, 1, 1)
    causal = j_idx <= i_idx
    real_j = j_idx < ell
    real_i = i_idx < ell
    not_cut = (i_idx < c) | (j_idx >= c)  # query rows may not see context cols
    allowed = causal & real_j & real_i & not_cut
    allowed = allowed | (i_idx == j_idx)  # padded rows self-attend (finite outputs)
    mask = torch.where(allowed, torch.zeros((), dtype=dtype), torch.full((), neg, dtype=dtype))
    return mask.unsqueeze(1)  # (B, 1, T, T)


# ── fold assignment ───────────────────────────────────────────────────────────


def assign_stratified_folds(lengths: list[int], n_folds: int, seed: int) -> list[int]:
    """Fold index per query, stratified by length decile (plan §4.2).

    Deciles over the matched-Betley token lengths; within each decile the
    queries are seed-shuffled; the concatenated decile order is dealt
    round-robin into ``n_folds`` folds → exactly equal fold sizes when
    ``len(lengths) % n_folds == 0``.
    """
    n = len(lengths)
    arr = np.asarray(lengths, dtype=np.float64)
    edges = np.percentile(arr, np.linspace(0, 100, 11))
    decile = np.clip(np.searchsorted(edges, arr, side="right") - 1, 0, 9)
    rng = np.random.default_rng(seed)
    order: list[int] = []
    for d in range(10):
        idx = np.where(decile == d)[0]
        order.extend(idx[rng.permutation(len(idx))].tolist())
    folds = [0] * n
    for pos, qi in enumerate(order):
        folds[qi] = pos % n_folds
    return folds


def load_folds() -> dict:
    """Load + sanity-check fold_assignments.json (Phase-0 output)."""
    payload = load_json(FOLDS_PATH)
    for genre, nq in (
        ("uc", payload["n_queries"]["uc"]),
        ("betley", payload["n_queries"]["betley"]),
    ):
        folds = payload["query_folds"][genre]
        assert len(folds) == nq, (genre, len(folds), nq)
        counts = [folds.count(k) for k in range(N_QUERY_FOLDS)]
        assert max(counts) - min(counts) <= 1, (genre, counts)
    return payload


# ── weighted exact PCA (few-distinct-row fold designs) ────────────────────────


def weighted_pca_basis(
    Xd: np.ndarray, weights: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted PCA mean + top-k components over DISTINCT rows with multiplicity.

    ``Xd`` (n_distinct, H) distinct feature rows, ``weights`` (n_distinct,)
    multiplicities (train-cell counts). Exactly the PCA of the EXPANDED train
    design (each row repeated ``weights[i]`` times): weighted mean, then SVD of
    ``sqrt(w) * (Xd - mu)``. Mirrors ``robust_pca_basis``'s gesdd→gesvd
    fallback. Returns (mu (H,), comps (k', H)) with k' = min(k, n_distinct).
    Projected coordinates are deliberately NOT re-standardized downstream
    (``press_fit_predict(standardize=False)`` — the no-post-PCA-whitening
    choice); near-zero-variance projected dims stay in place and are benign
    under PRESS-ridge regularization.
    """
    w = np.asarray(weights, dtype=np.float64)
    assert (w > 0).all(), "weighted_pca_basis: nonpositive multiplicity"
    mu = (w[:, None] * Xd).sum(axis=0) / w.sum()
    Xc = (Xd - mu) * np.sqrt(w)[:, None]
    try:
        _, _s, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        _, _s_t, Vh = torch.linalg.svd(torch.from_numpy(Xc), full_matrices=False)
        Vt = Vh.numpy()
    kk = min(k, Vt.shape[0])
    return mu, Vt[:kk]


# ── pack IO ───────────────────────────────────────────────────────────────────


def save_pack(path: Path, tensors: dict[str, torch.Tensor], meta: dict) -> None:
    """Save a tensor pack (fp16 tensors + JSON-able metadata) atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"tensors": tensors, "meta": meta}, tmp)
    tmp.replace(path)


def load_pack(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Load a tensor pack; returns (tensors, meta)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["tensors"], blob["meta"]


def cell_row(ctx_idx: int, q_idx: int, n_q: int) -> int:
    """Flat row index for cell (ctx_idx, q_idx) in an (n_ctx * n_q, ...) pack."""
    return ctx_idx * n_q + q_idx
