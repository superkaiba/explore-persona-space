"""Text-embedding baseline predictor for evil-OOD-spread rungs.

Task #1739 evil-ood-spread-round unit 4b (plan v16 §4-figures + §6 baseline).

Off-registry helper: computes per-context predictions from a sentence-embedding
backbone and emits them in the SAME schema as arm predictors so downstream
figures/tables can compare arm16_surface_feat + map-family arms against this
non-representation baseline. NOT arm17; NOT a registered ARM; NEVER
participates in the 16-arm rho / permutation-null-max headline.

Design:
    - `AutoModel + mean-pooling` on `BAAI/bge-small-en-v1.5` (BERT-family
      384-dim embedder; the field-standard cheap sentence embedder as of 2026).
      No `sentence-transformers` dep — bare `transformers` + a mean-pool head
      reproduces the reference embedding within cosine ~0.9999 of the ST client
      (the client wraps the same forward with the same pooling).
    - Predictor: per-context L2-normalized embedding of the user query text;
      the score for each context is `mean(embedding)` (a single scalar per
      context — simple, deterministic, avoids fitting a linear head that would
      re-derive the map-family setup this baseline is meant to CONTRAST with).
    - `--embedder fake` for smoke: deterministic sha-derived pseudo-embedding,
      no model download.

Inputs (per rung `<R>`):
    --contexts eval_results/issue_1739/evil_ood_spread/contexts/<R>.json
        {"order": [ctx_id, ...],
         "texts": {"<ctx_id>": "<user query text>"}}
    OR the flat shape (order determined by iteration):
        {"contexts": [{"context_id": "...", "text": "..."}, ...]}

Outputs (per rung `<R>`):
    <out>/<R>.json
        {"rung": str,
         "arm_id": "text_embed_baseline",
         "embedder": str,
         "predictions": [{"context_id": str, "arm_id": str, "pred_score": float,
                          "fold_idx": null}, ...]}

Smoke:
    --smoke runs ONE rung x 20 contexts with `--embedder fake`, asserts the
    output JSON schema, exits rc=0.

Grounded on:
    - plan v16 §4 (text-embedding baseline as figure companion to arm16).
    - Absence of an in-repo canonical sentence embedder (grepped, none found).
    - .claude/rules/data-realism.md (real corpora → real text embeddings).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np  # noqa: F401


def _ensure_repo_root_on_syspath() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    sentinel = repo_root / "scripts" / "issue1739_rescore_ood.py"
    assert sentinel.exists(), f"repo-root sentinel missing: {sentinel}"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


_REPO_ROOT = _ensure_repo_root_on_syspath()

DEFAULT_EMBEDDER = "BAAI/bge-small-en-v1.5"
FAKE_EMBEDDER_DIM = 64
DEFAULT_MAX_TOKENS = 512  # BGE-small context; long queries truncate.
ARM_ID = "text_embed_baseline"


def _log(msg: str) -> None:
    print(f"[textembed] {msg}", flush=True)


# ---------------------------------------------------------------------------
# fake embedder (smoke)
# ---------------------------------------------------------------------------
def _fake_embed(texts: list[str]) -> "np.ndarray":  # type: ignore[name-defined]
    """Deterministic sha-derived pseudo-embeddings.

    Each text is hashed to 64 bytes -> 16 fp32 values -> tiled/truncated to
    FAKE_EMBEDDER_DIM. L2-normalized. NOT semantically meaningful — smoke-only.
    """
    import numpy as np

    out = np.zeros((len(texts), FAKE_EMBEDDER_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hashlib.sha256(t.encode("utf-8")).digest()  # 32 bytes
        # 8 fp32 from 32 bytes
        vals = np.array(struct.unpack("8f", h), dtype=np.float32)
        # Tile to FAKE_EMBEDDER_DIM
        rep = (FAKE_EMBEDDER_DIM + 7) // 8
        tiled = np.tile(vals, rep)[:FAKE_EMBEDDER_DIM]
        norm = float(np.linalg.norm(tiled))
        if norm > 0:
            out[i] = tiled / norm
    return out


# ---------------------------------------------------------------------------
# real embedder (BGE via bare transformers)
# ---------------------------------------------------------------------------
def _bge_embed(texts: list[str], model_name: str, batch_size: int = 32) -> "np.ndarray":  # type: ignore[name-defined]
    """Mean-pooled + L2-normalized embeddings via AutoModel.

    Matches the standard BGE inference recipe: forward the tokenized query
    through the encoder, mean-pool over the attention-masked sequence, then
    L2-normalize. Runs on CPU (float32) by default; move to GPU only if
    a large embed-time bottleneck surfaces.
    """
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    outputs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=DEFAULT_MAX_TOKENS,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        # Mean-pool over attended tokens.
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        outputs.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _embed(texts: list[str], embedder: str) -> "np.ndarray":  # type: ignore[name-defined]
    if embedder == "fake":
        return _fake_embed(texts)
    return _bge_embed(texts, embedder)


# ---------------------------------------------------------------------------
# per-context prediction
# ---------------------------------------------------------------------------
def _predictions_from_embeddings(
    ctx_order: list[str],
    embeddings: "np.ndarray",  # type: ignore[name-defined]
) -> list[dict[str, Any]]:
    """One scalar per context: mean of the L2-normalized embedding.

    This is deliberately SIMPLE (no fitted head): the baseline exists to show
    what a raw text-embedding signal alone predicts about the DV without any
    activation-space geometry — a lower bar the map-family arms must clear
    to justify the extraction cost.
    """

    scores = embeddings.mean(axis=1)
    predictions: list[dict[str, Any]] = []
    for i, cid in enumerate(ctx_order):
        predictions.append(
            {
                "context_id": cid,
                "arm_id": ARM_ID,
                "pred_score": float(scores[i]),
                "fold_idx": None,  # OOD transfer — no fold structure on new rungs.
            }
        )
    return predictions


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _load_contexts(path: Path) -> tuple[list[str], list[str]]:
    """Return (ctx_order, texts_in_order). Supports two layouts."""
    payload = json.loads(path.read_text())
    if "order" in payload and "texts" in payload:
        order = list(payload["order"])
        texts = [payload["texts"][cid] for cid in order]
    elif "contexts" in payload:
        order = [rec["context_id"] for rec in payload["contexts"]]
        texts = [rec.get("text", "") for rec in payload["contexts"]]
    else:
        raise ValueError(f"unsupported contexts JSON layout: {path}")
    if len(order) != len(texts):
        raise ValueError("order/texts length mismatch")
    return order, texts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    tmp.replace(path)


def _repro_metadata() -> dict[str, Any]:
    import platform
    import subprocess

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True
        ).strip()
    except Exception:
        sha = "unavailable-no-git"
    return {
        "git_commit": sha,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rungs", nargs="+", required=True, help="Rung slugs to embed.")
    p.add_argument(
        "--embedder",
        default=DEFAULT_EMBEDDER,
        help=(
            "HF model id (default BAAI/bge-small-en-v1.5) OR the literal 'fake' "
            "for smoke (deterministic sha-derived pseudo-embeddings, no download)."
        ),
    )
    p.add_argument(
        "--input-root",
        default="eval_results/issue_1739/evil_ood_spread",
        help="Root dir for contexts/ inputs.",
    )
    p.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        default="eval_results/issue_1739/evil_ood_spread/text_embed_baseline/",
        help="Output dir for per-rung JSON.",
    )
    p.add_argument(
        "--dv-labels", default=None, help="Reserved (unused): DV labeling path for provenance."
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--smoke", action="store_true", help="Tiny slice: 1 rung, fake embedder OK.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    embedder = args.embedder
    if args.smoke and embedder != "fake":
        _log(f"smoke mode: forcing embedder=fake (was {embedder})")
        embedder = "fake"

    rungs = list(args.rungs)
    if args.smoke:
        rungs = rungs[:1]

    input_root = Path(args.input_root)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rung in rungs:
        ctx_path = input_root / "contexts" / f"{rung}.json"
        _log(f"[{rung}] loading contexts={ctx_path}")
        ctx_order, texts = _load_contexts(ctx_path)
        _log(f"[{rung}] n_ctx={len(ctx_order)} embedder={embedder}")
        embeddings = _embed(texts, embedder)
        assert embeddings.shape[0] == len(ctx_order), (
            f"embeddings/ctx count mismatch: {embeddings.shape[0]} vs {len(ctx_order)}"
        )
        predictions = _predictions_from_embeddings(ctx_order, embeddings)
        payload = {
            "rung": rung,
            "arm_id": ARM_ID,
            "embedder": embedder,
            "n_ctx": len(ctx_order),
            "embedding_dim": int(embeddings.shape[1]),
            "predictions": predictions,
            "provenance": {
                "batch_size": args.batch_size,
                "dv_labels": args.dv_labels,
                **_repro_metadata(),
            },
        }
        out_path = out_dir / f"{rung}.json"
        _write_json(out_path, payload)
        _log(f"[{rung}] wrote {out_path} (n_preds={len(predictions)})")

    _log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
