#!/usr/bin/env python3
"""Issue #617 Step 2: embed first-user-turns + cluster sweep (VM CPU / 1xL4).

Per plan §4 step 2. Embeds each conversation's FIRST USER TURN with
``BAAI/bge-large-en-v1.5`` via ``transformers.AutoModel`` (no
``sentence-transformers`` dependency — §12 Assumption 2), CLS-token pooled
(``last_hidden_state[:, 0]``) + L2-normalized (BGE v1.5's documented pooling
per the model card + ``1_Pooling/config.json: pooling_mode_cls_token: true``
— NOT mean-pooling, M3), then clusters over K in {5,10,20} (KMeans) plus one
HDBSCAN(min_cluster_size=max(30, N/200)).

POOLING-PATH ACCEPTANCE CRITERION (M3): ``embed_first_user_turns`` pools via
``last_hidden_state[:, 0]`` (CLS) and MUST NOT compute an attention-masked
mean. ``--selftest-pooling`` runs the assertion directly.

Usage::

    uv run python scripts/issue617_cluster.py                       # full sweep on the 20k slice
    uv run python scripts/issue617_cluster.py --ks 5 --device cpu   # smoke (K=5 only)
    uv run python scripts/issue617_cluster.py --selftest-pooling    # M3 pooling-path check only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue617_common import (  # noqa: E402
    CLUSTER_KS,
    CLUSTER_PATH,
    DATA_DIR,
    EMBED_MAX_TOKENS,
    EMBEDDER_DIM,
    EMBEDDER_MODEL,
    SEED,
    SLICE_PATH,
)
from sklearn.cluster import HDBSCAN, KMeans  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue617_cluster")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Pooling-mode provenance string carried into cluster_assignments.json +
# separability.json (M3 guard).
POOLING_MODE = "cls"


def embed_first_user_turns(
    texts: list[str],
    model_id: str = EMBEDDER_MODEL,
    batch: int = 64,
    device: str = "auto",
    max_tokens: int = EMBED_MAX_TOKENS,
) -> np.ndarray:
    """CLS-token pooled (``last_hidden_state[:, 0]``) + L2-normalized
    bge-large-en-v1.5 embeddings (N, 1024).

    M3: pooling is CLS (the [CLS] token at position 0), NOT an
    attention-masked mean over the token axis. BGE v1.5's documented pooling
    is CLS + L2-norm.
    """
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    if use_cuda:
        model = model.to("cuda")

    out = np.empty((len(texts), EMBEDDER_DIM), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            chunk = texts[start : start + batch]
            enc = tok(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            )
            if use_cuda:
                enc = {k: v.to("cuda") for k, v in enc.items()}
            model_output = model(**enc)
            # M3: CLS pooling = last_hidden_state[:, 0]. Explicitly NOT a
            # mean over the token axis weighted by attention_mask.
            cls = model_output.last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            out[start : start + len(chunk)] = cls.float().cpu().numpy()
            if start % (batch * 20) == 0:
                logger.info("embedded %d/%d", start + len(chunk), len(texts))
    assert out.shape == (len(texts), EMBEDDER_DIM), out.shape
    return out


def selftest_pooling(model_id: str = EMBEDDER_MODEL) -> None:
    """M3 acceptance assertion: the embedder uses CLS pooling, not a mean.

    Compares ``embed_first_user_turns`` output for two short fixed inputs
    against (a) a reference CLS-pooled + L2-normalized vector computed inline,
    and (b) the attention-mask-weighted MEAN-pooled + L2-normalized vector,
    asserting it MATCHES (a) and DIFFERS from (b). This is the load-bearing
    guard: mean-vs-CLS pooling on BGE v1.5 produces materially different
    embeddings, and cluster quality is the sole input to the headline metric.
    """
    from transformers import AutoModel, AutoTokenizer

    fixed = ["how do I fix a flat bike tire?", "explain the French Revolution"]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    enc = tok(
        fixed, padding=True, truncation=True, max_length=EMBED_MAX_TOKENS, return_tensors="pt"
    )
    with torch.no_grad():
        hs = model(**enc).last_hidden_state  # (B, T, H)
    # (a) reference CLS pool + L2-norm.
    ref_cls = torch.nn.functional.normalize(hs[:, 0], p=2, dim=1).numpy()
    # (b) attention-masked mean pool + L2-norm (the WRONG pooling for v1.5).
    mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
    summed = (hs * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    ref_mean = torch.nn.functional.normalize(summed / counts, p=2, dim=1).numpy()

    got = embed_first_user_turns(fixed, model_id=model_id, device="cpu")
    assert got.shape == ref_cls.shape == ref_mean.shape, (got.shape, ref_cls.shape)
    cls_cos = float((got * ref_cls).sum(axis=1).min())
    mean_cos = float((got * ref_mean).sum(axis=1).max())
    assert cls_cos > 0.999, (
        f"pooling-path assert failed: embed_first_user_turns does NOT match CLS pooling "
        f"(min cosine vs reference CLS = {cls_cos:.6f}); expected > 0.999"
    )
    assert mean_cos < 0.999, (
        f"pooling-path assert failed: embed_first_user_turns matches MEAN pooling "
        f"(max cosine vs reference mean = {mean_cos:.6f}); v1.5 must use CLS, not mean"
    )
    logger.info(
        "M3 pooling-path OK: matches CLS (cos %.6f), differs from mean (cos %.6f)",
        cls_cos,
        mean_cos,
    )


def cluster_sweep(
    emb: np.ndarray,
    conv_ids: list[str],
    first_users: list[str],
    ks=CLUSTER_KS,
    seed: int = SEED,
    target: int | None = None,
) -> dict:
    """KMeans per K + one HDBSCAN. Returns per-config {labels, sizes, examples}.

    HDBSCAN noise (label -1) is recorded but EXCLUDED from the pair pool at
    scoring time (the cluster id ``-1`` is dropped). ``min_cluster_size`` =
    max(30, target//200) per plan §4 step 2 (target = the slice size).
    """
    n = emb.shape[0]
    target = target or n
    configs: dict[str, dict] = {}

    def summarize(algo: str, labels: np.ndarray) -> dict:
        sizes: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
        for lab in sorted(set(labels.tolist())):
            cid = f"{algo}_c{int(lab):02d}" if lab >= 0 else f"{algo}_noise"
            members = np.where(labels == lab)[0]
            sizes[cid] = len(members)
            examples[cid] = [first_users[i][:200] for i in members[:5]]
        return {
            "labels": {conv_ids[i]: int(labels[i]) for i in range(n)},
            "sizes": sizes,
            "examples": examples,
        }

    for k in ks:
        algo = f"kmeans{k}"
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(emb)
        configs[algo] = summarize(algo, labels)
        logger.info("%s: %d clusters, sizes=%s", algo, k, configs[algo]["sizes"])

    min_cluster_size = max(30, target // 200)
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    hdb_labels = hdb.fit_predict(emb)
    n_noise = int((hdb_labels == -1).sum())
    n_clusters = len({lab for lab in hdb_labels.tolist() if lab >= 0})
    configs["hdbscan"] = summarize("hdbscan", hdb_labels)
    configs["hdbscan"]["min_cluster_size"] = min_cluster_size
    configs["hdbscan"]["n_noise"] = n_noise
    configs["hdbscan"]["n_clusters_non_noise"] = n_clusters
    logger.info(
        "hdbscan(min_cluster_size=%d): %d non-noise clusters, %d noise points",
        min_cluster_size,
        n_clusters,
        n_noise,
    )
    if n_clusters < 2:
        logger.warning(
            "HDBSCAN yielded %d non-noise clusters (<2); the winner can still come "
            "from a KMeans config (plan §8 risk row).",
            n_clusters,
        )
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #617 Step 2: embed + cluster sweep.")
    parser.add_argument("--slice", type=Path, default=SLICE_PATH)
    parser.add_argument("--out", type=Path, default=CLUSTER_PATH)
    parser.add_argument("--model", default=EMBEDDER_MODEL)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=list(CLUSTER_KS),
        help="KMeans K values (smoke: --ks 5)",
    )
    parser.add_argument(
        "--selftest-pooling",
        action="store_true",
        help="run the M3 CLS-vs-mean pooling-path assertion and exit",
    )
    args = parser.parse_args()

    if args.selftest_pooling:
        selftest_pooling(args.model)
        return 0

    # M3 guard always runs before the real embedding (cheap, CPU).
    selftest_pooling(args.model)

    with open(args.slice) as f:
        slice_payload = json.load(f)
    convs = slice_payload["conversations"]
    conv_ids = [c["conv_id"] for c in convs]
    first_users = [c["first_user"] for c in convs]
    target = slice_payload["meta"].get("target", len(convs))
    logger.info("Embedding %d first-user-turns with %s (CLS pool)", len(convs), args.model)
    emb = embed_first_user_turns(
        first_users, model_id=args.model, batch=args.batch, device=args.device
    )

    configs = cluster_sweep(emb, conv_ids, first_users, ks=tuple(args.ks), target=target)

    payload = {
        "meta": {
            "embedder": args.model,
            "pooling_mode": POOLING_MODE,
            "embedder_dim": EMBEDDER_DIM,
            "ks": list(args.ks),
            "n_conversations": len(convs),
            "seed": SEED,
            "slice_path": str(args.slice),
            "metadata": reproducibility_metadata({"script": "issue617_cluster"}),
        },
        "configs": configs,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    logger.info("Wrote %s: %d configs", args.out, len(configs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
