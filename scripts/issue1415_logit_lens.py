"""Issue #1415 logit-lens descriptive companion (round B deliverable 4).

For each pair's V_c (both extraction arms), V_a(c), and Delta (both arms) at
the primary layer 20 — plus the top-3 #922 slow modes (block 14, |lambda| >=
0.98, from the committed ``eval_results/issue_922/fixed_point_slow_modes_
topvecs.npz``) and the block-14 fixed point h* — compute the top-10 promoted
tokens under the model's unembedding::

    logit_lens(v) = model.lm_head(model.model.norm(v))

Descriptive only (plan v5 §4.11): no null comparison; a single batched
unembedding matmul over all vectors. ``model.model.norm`` is the final
RMSNorm, so the read is scale-invariant (vector norms do not matter); the
slow-mode eigenvectors are complex — the REAL part is used and the imaginary
mass fraction is recorded (sign is arbitrary for an eigenvector; both the +
and - readouts are emitted for the modes).

Output: ``eval_results/issue_1415/logit_lens_tokens.json``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE numpy/torch — the #847 thread-cap hook binds at import time

import numpy as np  # noqa: E402
import torch  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1415_analysis_common as common  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1415_logit_lens")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SLOW_MODES_NPZ = (
    common.REPO_ROOT / "eval_results" / "issue_922" / "fixed_point_slow_modes_topvecs.npz"
)
SLOW_MODE_BLOCK = 14  # plan v5 §4.11: top-3 modes with |lambda| >= 0.98 at layer 14
N_SLOW_MODES = 3
TOP_K = 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--activations",
        type=Path,
        default=common.REPO_ROOT / "data" / "issue_1415" / "phase1" / "activations",
    )
    ap.add_argument("--slow-modes-npz", type=Path, default=SLOW_MODES_NPZ)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=common.REPO_ROOT / "eval_results" / "issue_1415" / "logit_lens_tokens.json",
    )
    ap.add_argument("--model-id", type=str, default=MODEL_ID)
    ap.add_argument("--layer", type=int, default=common.PRIMARY_LAYER)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--device", type=str, default=None, help="default: cuda if available")
    return ap.parse_args(argv)


def compute_lens(
    model, vectors: dict[str, torch.Tensor], top_k: int, tokenizer=None
) -> dict[str, dict]:
    """Batched logit-lens: one stacked ``lm_head(norm(V))`` over all vectors.

    ``model`` needs ``model.model.norm`` (final RMSNorm) + ``model.lm_head``
    (the HF Qwen2 causal-LM structure — the from-config tiny model in tests
    has the identical attribute path). Returns per-name top-k token ids,
    logits, softmax probs, and decoded token strings when a tokenizer is
    given.
    """
    assert vectors, "no vectors to lens"
    names = list(vectors)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    stacked = torch.stack([vectors[n] for n in names]).to(device=device, dtype=dtype)
    assert stacked.dim() == 2, stacked.shape
    with torch.no_grad():
        logits = model.lm_head(model.model.norm(stacked)).float()  # (N, V)
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(logits, k=top_k, dim=-1)
    out: dict[str, dict] = {}
    for i, name in enumerate(names):
        ids = top.indices[i].tolist()
        rec = {
            "token_ids": ids,
            "logits": [float(x) for x in top.values[i]],
            "probs": [float(probs[i, j]) for j in ids],
        }
        if tokenizer is not None:
            rec["tokens"] = [tokenizer.decode([j]) for j in ids]
        out[name] = rec
    return out


def load_slow_mode_vectors(npz_path: Path, block: int, n_modes: int) -> tuple[dict, dict]:
    """Top-n slow-mode eigenvectors (real part, +/- sign) + block fixed point.

    Returns (vectors, provenance). Eigvals in the npz are sorted descending by
    |lambda| (verified against the committed #922 json: 0.9896/0.9850/0.9823
    at block 14).
    """
    z = np.load(npz_path)
    eigvals = z[f"block{block}_eigvals"]
    eigvecs = z[f"block{block}_eigvecs"]  # (H, n_modes_stored), complex
    order = np.argsort(-np.abs(eigvals))[:n_modes]
    vectors: dict[str, torch.Tensor] = {}
    prov: dict = {"npz": str(npz_path), "block": block, "modes": {}}
    for rank, idx in enumerate(order):
        vec = eigvecs[:, idx]
        real = np.real(vec)
        imag_frac = float(np.linalg.norm(np.imag(vec)) / np.linalg.norm(vec))
        v = torch.from_numpy(real.astype(np.float32))
        assert v.norm() > 0, (block, int(idx))
        vectors[f"i922/block{block}_slow_mode{rank}_plus"] = v
        vectors[f"i922/block{block}_slow_mode{rank}_minus"] = -v
        prov["modes"][f"mode{rank}"] = {
            "eig_index": int(idx),
            "abs_eigval": float(np.abs(eigvals[idx])),
            "imag_mass_fraction": imag_frac,
        }
    h_star = torch.from_numpy(np.asarray(z[f"block{block}_h_star"], dtype=np.float32))
    vectors[f"i922/block{block}_h_star"] = h_star
    return vectors, prov


def build_pair_vectors(pairs: list[common.PairTensors], layer: int) -> dict[str, torch.Tensor]:
    vectors: dict[str, torch.Tensor] = {}
    li = pairs[0].layers.index(layer)
    for p in pairs:
        vectors[f"{p.pair_id}/v_c_context"] = p.v_c["context"][li]
        vectors[f"{p.pair_id}/v_c_prefix"] = p.v_c["prefix"][li]
        vectors[f"{p.pair_id}/v_a_c"] = p.v_a_c[li]
        vectors[f"{p.pair_id}/delta_context"] = p.delta["context"][li]
        vectors[f"{p.pair_id}/delta_prefix"] = p.delta["prefix"][li]
    return vectors


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    pairs = common.load_all_pairs(args.activations)
    assert args.layer in pairs[0].layers, (args.layer, pairs[0].layers)
    vectors = build_pair_vectors(pairs, args.layer)
    mode_vectors, mode_prov = load_slow_mode_vectors(
        args.slow_modes_npz, SLOW_MODE_BLOCK, N_SLOW_MODES
    )
    vectors.update(mode_vectors)
    logger.info("lensing %d vectors (%d pairs + slow modes)", len(vectors), len(pairs))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype="auto", device_map=device
    )
    model.eval()

    lens = compute_lens(model, vectors, args.top_k, tokenizer=tokenizer)
    out = {
        "layer": args.layer,
        "top_k": args.top_k,
        "model_id": args.model_id,
        "read": "lm_head(model.model.norm(v)) — descriptive companion, no null (plan §4.11)",
        "slow_modes": mode_prov,
        "vectors": lens,
        "repro": common.repro_meta("issue1415_logit_lens"),
    }
    common.write_json_atomic(args.out_json, out)
    logger.info("wrote %s (%d vectors)", args.out_json, len(lens))


if __name__ == "__main__":
    main()
