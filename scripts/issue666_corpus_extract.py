#!/usr/bin/env python
# ruff: noqa: RUF003
# Intentional scientific Unicode (Σ, λ, κ, ×, ⁻¹, ᵀ) in docstrings/comments.
"""issue #666 Phase 4 — broad-corpus Σ_c extraction (the ONE GPU step; plan §4c, Must-Fix 1).

``Σ_c = E[ccᵀ]`` MUST be estimated off a BROAD BACKGROUND CORPUS (≥2-5k contexts),
NOT the n=50 battery (design-doc §7.4 / §352-356 / §829 — rank ≤ 49 at d=3584
manufactures a spurious "whitening wins"). The corpus SOURCE TEXT is reused from
the project's existing builder (``project_corpus_v2.download_fineweb`` — FineWeb-Edu
natural text); the context vectors are RE-EXTRACTED through the SAME #664
``last_input``-slot recipe at layer 14 on Qwen-2.5-7B-Instruct base (NOT the
builder's native layer-32/Qwen3 or mean-over-probes output, which is incommensurate
with the store's ``c_C_base``).

Each corpus context is its text wrapped as a single user turn → the chat template
(``add_generation_prompt=True``) → the residual-stream hidden state at the LAST
prompt token (the "last-input" slot), all 28 layers. The (N, 28, d) tensor is
saved + the layer-14 Σ_c / Σc⁻¹ computed via the shared
``leakage_predictor.estimate_sigma_inv`` (CV-λ, conditioning report), then uploaded
to the HF data repo (plan §10 — a plan-referenced downstream input).

SMOKE-VS-PRODUCTION BOUNDARY (by design): ``--slice`` runs ≤8 SYNTHETIC contexts
on CPU (no GPU, no network). The synthetic ``_synthetic_vectors`` exercises the
POST-EXTRACTION ``Σ_c → CV-λ-inverse`` code path (``compute_sigma_c`` →
``leakage_predictor.estimate_sigma_inv``) on production-shaped (N, 28, 3584)
tensors, but NOT the Qwen forward. The real Qwen-2.5-7B layer-14 chat-template +
last-input-slot extraction (``_last_input_vectors``) is first-validated at
PRODUCTION launch (the experimenter step), which fails loud on any extraction
defect (shape assert + fail-loud HF upload). The contract split:
  • smoke verifies  : the Σ_c estimator handles real-shape (N, 28, 3584) tensors;
  • production verifies: the Qwen forward + chat-template + slot-extraction produce
    those tensors correctly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
PRIMARY_LAYER = 14
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_CORPUS_PREFIX = "issue666_phase4/sigma_c_corpus"
DATA_DIR = REPO / "data" / "issue_666"
DEFAULT_N_CONTEXTS = 3000  # ≥2-5k broad-corpus contexts (plan §4c)
N_SLICE = 8  # smoke ceiling: --slice runs ≤8 synthetic CPU contexts (no GPU/network)


def _fineweb_texts(n: int) -> list[str]:
    """N natural-text contexts from FineWeb-Edu (reuse project_corpus_v2 source).

    The corpus SOURCE TEXT only — the extraction goes through the #664 last_input
    recipe below, NOT the builder's native projection.
    """
    import project_corpus_v2 as pc

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fw_path = DATA_DIR / "fineweb_corpus.jsonl"
    pc.download_fineweb(fw_path, max_docs=n)
    texts: list[str] = []
    with open(fw_path) as f:
        for line in f:
            if len(texts) >= n:
                break
            doc = json.loads(line)
            t = doc.get("text", "").strip()
            if t:
                texts.append(t[:2000])  # cap per-context length (assistant-axis recipe)
    return texts


def _last_input_vectors(texts: list[str], *, device: str, tf_batch_size: int = 8) -> np.ndarray:
    """Per-context last-input-slot residual, all 28 layers, on the base model.

    Each text → one user turn → chat template (add_generation_prompt=True) →
    teacher-forced forward → the hidden state at the LAST prompt token, layers
    1..28. Returns (N, 28, d) float32. Left-pads each batch with explicit
    position_ids (RoPE faithfulness — the same recipe as
    ``issue664_extract_store._answer_side_means``).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
    on_cuda = device.startswith("cuda") and torch.cuda.is_available()
    dtype = torch.bfloat16 if on_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, dtype=dtype, trust_remote_code=True
    ).eval()
    if on_cuda:
        model = model.to(device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    # Tokenize each context's prompt.
    id_lists: list[list[int]] = []
    for t in texts:
        msgs = [{"role": "user", "content": t}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        id_lists.append(tok.encode(text, add_special_tokens=False))

    out_vecs: list[torch.Tensor] = []
    for start in range(0, len(id_lists), tf_batch_size):
        chunk = id_lists[start : start + tf_batch_size]
        max_len = max(len(ids) for ids in chunk)
        input_ids, attn, pads = [], [], []
        for ids in chunk:
            pad_len = max_len - len(ids)
            input_ids.append([pad] * pad_len + ids)
            attn.append([0] * pad_len + [1] * len(ids))
            pads.append(pad_len)
        input_ids_t = torch.tensor(input_ids, device=device)
        attn_t = torch.tensor(attn, device=device)
        pos_ids = (attn_t.long().cumsum(dim=1) - 1).clamp(min=0)
        with torch.no_grad():
            res = model(
                input_ids=input_ids_t,
                attention_mask=attn_t,
                position_ids=pos_ids,
                output_hidden_states=True,
            )
        hs = torch.stack(res.hidden_states[1:], dim=1).float()  # (B, 28, T, d)
        for i, ids in enumerate(chunk):
            li = pads[i] + len(ids) - 1  # last prompt token slot
            out_vecs.append(hs[i, :, li, :].cpu())
        del res, hs
    del model
    return torch.stack(out_vecs, dim=0).numpy().astype(np.float32)  # (N, 28, d)


def _synthetic_vectors(n: int, nl: int = EXPECTED_LAYERS, d: int = EXPECTED_HIDDEN) -> np.ndarray:
    """A tiny SYNTHETIC (n, nl, d) corpus-vector tensor for the CPU smoke (no GPU).

    The smoke uses these in place of the real Qwen ``_last_input_vectors`` output so
    the post-extraction ``Σ_c → CV-λ-inverse`` path (``compute_sigma_c``) runs on
    PRODUCTION-shaped (n, 28, 3584) tensors with no GPU/network. The Qwen forward
    itself is NOT exercised here — it is first-validated at production launch (the
    experimenter step). See the module docstring's SMOKE-VS-PRODUCTION BOUNDARY.
    """
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, nl, d)).astype(np.float32)


def _resolve_slice_n(n_contexts: int) -> int:
    """The number of synthetic contexts ``--slice`` actually runs: ``min(n, N_SLICE)``.

    ``--slice`` alone (the smoke caller passes no ``--n-contexts``) inherits
    ``DEFAULT_N_CONTEXTS`` (3000); the smoke spirit is N≤8 synthetic CPU contexts,
    so the slice is capped at ``N_SLICE``. A caller MAY pass ``--n-contexts`` to run
    a smaller heavier smoke, but never above the cap. Returns ≤ ``N_SLICE`` always.
    """
    return min(n_contexts, N_SLICE)


def compute_sigma_c(corpus_vectors: np.ndarray, *, layer: int) -> dict:
    """Σ_c / Σc⁻¹ at the layer off the broad-corpus context vectors (plan §4c).

    ``corpus_vectors`` : (N, 28, d). Uses the shared
    ``leakage_predictor.estimate_sigma_inv`` (CV-λ over the registered grid +
    conditioning report). Returns a dict with the inverse, chosen λ, condition
    number, and the headline-eligibility flag (broad corpus → eligible).
    """
    from explore_persona_space.analysis.leakage_predictor import estimate_sigma_inv

    C = corpus_vectors[:, layer, :].astype(np.float64)  # (N, d)
    res = estimate_sigma_inv(C, seed=0, corpus_kind="broad")
    return {
        "Sigma_inv": res.Sigma_inv,
        "Sigma_c": res.Sigma_c,
        "lam": res.lam,
        "cond_number": res.cond_number,
        "rank_deficient": res.rank_deficient,
        "headline_eligible": res.headline_eligible,
        "n_contexts": res.n_contexts,
        "dim": res.dim,
        "layer": layer,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="issue 666 broad-corpus Σ_c extraction.")
    ap.add_argument(
        "--n-contexts",
        type=int,
        default=DEFAULT_N_CONTEXTS,
        help=(
            "broad-corpus size for the full run; under --slice it is CLAMPED to "
            f"N_SLICE={N_SLICE} (may lower the slice, never raise it above the cap)"
        ),
    )
    ap.add_argument("--layer", type=int, default=PRIMARY_LAYER)
    ap.add_argument(
        "--slice",
        action="store_true",
        help=f"smoke: ≤{N_SLICE} synthetic CPU contexts (no GPU/network)",
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--no-upload", action="store_true", help="skip HF upload (smoke)")
    args = ap.parse_args()

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.slice:
        # Smoke is CPU-only by design — _synthetic_vectors exercises the
        # post-extraction Σ_c → CV-λ-inverse code path with the right production
        # shape (N, 28, 3584), but NOT the Qwen forward. The real Qwen-2.5-7B
        # layer-14 chat-template + last-input-slot extraction is first-validated at
        # production launch (the experimenter step), which fails loud on any
        # extraction defect. Cap the slice at N_SLICE so `--slice` alone (no
        # explicit --n-contexts) runs the small CPU smoke; --n-contexts can still
        # lower the slice size for a heavier smoke (but never above the cap).
        n_slice = _resolve_slice_n(args.n_contexts)
        print(f"[corpus] SMOKE: {n_slice} synthetic CPU contexts (no GPU/network)")
        vecs = _synthetic_vectors(n_slice)
    else:
        device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        texts = _fineweb_texts(args.n_contexts)
        print(f"[corpus] extracting {len(texts)} contexts on {device} (layer all-28, last_input)")
        vecs = _last_input_vectors(texts, device=device)

    assert vecs.ndim == 3, f"corpus vectors must be (N, n_layer, d), got {vecs.shape}"
    layer = min(args.layer, vecs.shape[1] - 1)
    cache = DATA_DIR / "sigma_c_corpus_vectors.pt"
    torch.save({"corpus_vectors": torch.from_numpy(vecs), "layer": layer}, cache)

    sig = compute_sigma_c(vecs, layer=layer)
    sig_cache = DATA_DIR / "sigma_c_inv.pt"
    torch.save(
        {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in sig.items()},
        sig_cache,
    )
    digest = {k: v for k, v in sig.items() if not isinstance(v, np.ndarray)}
    print(f"[corpus] Σ_c digest: {json.dumps(digest)}")
    print(f"[corpus] vectors {vecs.shape} -> {cache}; Σc⁻¹ -> {sig_cache}")

    if not args.slice and not args.no_upload:
        # Plan-referenced downstream input (Upload Policy: analysis tensors before
        # any pod teardown). upload_dataset_directory globs by pattern and lands
        # each file at <bucket>/<file.name>; pass the *.pt pattern (the default is
        # *.jsonl) and the positional bucket prefix. Fail-loud (raises on any miss).
        from explore_persona_space.orchestrate.hub import upload_dataset_directory  # type: ignore

        upload_dataset_directory(DATA_DIR, HF_CORPUS_PREFIX, pattern="*.pt")
        print(f"[corpus] uploaded *.pt to {DATA_REPO}/{HF_CORPUS_PREFIX}")

    print("[phase=corpus_extract] done OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
