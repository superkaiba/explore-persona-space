#!/usr/bin/env python3
"""Exact-token follow-up to the #1901 WikiText boundary-token control.

The parent experiment pooled sentence-final punctuation types (and, for the
headline period arm, two distinct tokenizer IDs).  This driver holds the data
protocol fixed but treats each observed Qwen token ID as its own arm:

    659 ``Ġ.``   13 ``.``   937 ``Ġ?``   753 ``Ġ!``

For every exact ID it fits the layer-19 residual at that one token to the mean
layer-19 residual of the following text span.  It also fits (i) a pooled map
with the same total n as one exact-ID map and (ii) a pooled map using all four
arms, then evaluates both per identity.  Finally, every exact-ID map is
evaluated on every target identity, with and without target-mean recentering.

Phases:
  prepare  select balanced rows from the banked #1901 manifest under one
           article-disjoint split, then re-run exact/near-duplicate screens;
  capture  capture x_boundary and next-span y for layer 19 (layer 2 in smoke);
  fit      fit exact, pooled-matched, and pooled-all ridge maps and transfer;
  figure   render the per-ID comparison and cross-ID transfer matrix;
  capture_fit  run capture then fit (the production GPU-pod entry point).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_fitter_fair_comparison as F79  # noqa: E402
import issue931_common as common  # noqa: E402
import issue931_extract_store as ES  # noqa: E402
import issue1901_boundary_token_control as PARENT  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue779_ffc_n1m_generate_capture import NearDupeGate, _norm  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue1901_individual_boundary_tokens")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)

TOKEN_SPECS = {
    659: {"token": "Ġ.", "decoded": " .", "label": "space + period"},
    13: {"token": ".", "decoded": ".", "label": "period"},
    937: {"token": "Ġ?", "decoded": " ?", "label": "space + question"},
    753: {"token": "Ġ!", "decoded": " !", "label": "space + exclamation"},
}
TOKEN_IDS = tuple(TOKEN_SPECS)
SPLIT_SEED = 190101
FIT_SEED = 190102
DEFAULT_PARENT_MANIFEST = (
    PROJECT_ROOT
    / "data/issue_1901/individual_boundary_parent/issue1901_boundary_ctl/manifest"
)
DEFAULT_OUT_ROOT = Path(os.environ.get("WORKLOAD_ROOT", "/workspace")) / (
    "eps-issue-1901-individual-boundary"
)
DEFAULT_RESULT = PROJECT_ROOT / "eval_results/issue_1901/individual_boundary_tokens.json"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figures/issue_1901"
PROD_LAYER = 19
SMOKE_LAYER = 2


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def _stable_int(*parts: object) -> int:
    msg = "\x1f".join(str(v) for v in parts).encode()
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "big")


def article_split(article_id: str, seed: int = SPLIT_SEED) -> str:
    """One global, deterministic 70/10/20 article split for every token ID."""
    u = _stable_int("article-split", seed, article_id) / 2**64
    if u < 0.20:
        return "test"
    if u < 0.30:
        return "val"
    return "train"


def _load_parent_articles(parent_dir: Path, meta: dict) -> dict[str, torch.Tensor]:
    ids_by_art: dict[str, torch.Tensor] = {}
    for name in meta["article_shards"]:
        payload = torch.load(parent_dir / name, map_location="cpu", weights_only=True)
        for article_id, ids in zip(payload["window_ids"], payload["input_ids"], strict=True):
            ids_by_art[str(article_id)] = ids
    return ids_by_art


def _push_lowest(heap: list, limit: int, key: int, row: dict) -> None:
    """Retain the ``limit`` rows with smallest deterministic keys."""
    item = (-key, str(row["row_id"]), row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif key < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _candidate_limits(args) -> dict[str, int]:
    return {
        "train": max(args.n_train * args.candidate_multiplier, args.n_train),
        "val": max(args.n_val * args.candidate_multiplier, args.n_val),
        "test": max(args.n_test * args.candidate_multiplier, args.n_test),
    }


def scan_parent_candidates(
    parent_dir: Path, ids_by_art: dict[str, torch.Tensor], meta: dict, args
) -> tuple[dict, dict]:
    """Stream the 1.28M-row parent manifest without retaining its common rows."""
    limits = _candidate_limits(args)
    heaps = {(tid, split): [] for tid in TOKEN_IDS for split in limits}
    raw = {tid: Counter() for tid in TOKEN_IDS}
    seen_rows = 0
    for name in meta["manifest_shards"]:
        with (parent_dir / name).open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                seen_rows += 1
                article_id = str(row["article_id"])
                anchor = int(row["anchor_pos"])
                token_id = int(ids_by_art[article_id][anchor])
                if token_id not in TOKEN_SPECS:
                    continue
                split = article_split(article_id, args.split_seed)
                raw[token_id][split] += 1
                kept = {
                    "row_id": row["row_id"],
                    "article_id": article_id,
                    "sep_char": row["sep_char"],
                    "anchor_pos": anchor,
                    "c_span": [int(v) for v in row["c_span"]],
                    "t_span": [int(v) for v in row["t_span"]],
                    "n_span_tokens": int(row.get("n_span_tokens", row["t_span"][1] - row["t_span"][0])),
                    "split": split,
                    "boundary_token_id": token_id,
                    "boundary_token": TOKEN_SPECS[token_id]["token"],
                    "parent_split": row.get("split"),
                }
                key = _stable_int("candidate", args.split_seed, row["row_id"])
                _push_lowest(heaps[(token_id, split)], limits[split], key, kept)
    assert seen_rows == int(meta["n_manifest_rows"]), (seen_rows, meta["n_manifest_rows"])
    out = {}
    for arm, heap in heaps.items():
        out[arm] = [v[2] for v in sorted(heap, key=lambda x: (-x[0], x[1]))]
    return out, {str(tid): dict(raw[tid]) for tid in TOKEN_IDS}


def _target_text(tokenizer, row: dict, ids_by_art: dict[str, torch.Tensor]) -> str:
    lo, hi = row["t_span"]
    return tokenizer.decode(ids_by_art[row["article_id"]][lo:hi].tolist())


def select_rows(
    candidates: list[dict],
    target: int,
    tokenizer,
    ids_by_art: dict[str, torch.Tensor],
    *,
    article_cap: int,
    forbidden_norms: set[str],
    near_gate: NearDupeGate | None,
) -> tuple[list[dict], list[str], dict]:
    """Select after exact-dedup, optional eval-near-dupe, and article cap."""
    selected: list[dict] = []
    texts: list[str] = []
    per_article: Counter = Counter()
    drops: Counter = Counter()
    for row in candidates:
        article_id = row["article_id"]
        if per_article[article_id] >= article_cap:
            drops["article_cap"] += 1
            continue
        text = _target_text(tokenizer, row, ids_by_art)
        norm = _norm(text)
        if not norm:
            drops["empty"] += 1
            continue
        if norm in forbidden_norms:
            drops["exact"] += 1
            continue
        if near_gate is not None and near_gate.is_dupe(text):
            drops["near_or_exact_eval"] += 1
            continue
        out = dict(row)
        out["selection_order"] = len(selected)
        selected.append(out)
        texts.append(text)
        forbidden_norms.add(norm)
        per_article[article_id] += 1
        if len(selected) == target:
            break
    return selected, texts, {"drops": dict(drops), "n_scanned": sum(drops.values()) + len(selected)}


def phase_prepare(args) -> int:  # noqa: C901 - one linear audited selection ladder
    parent_dir = args.parent_manifest
    parent_meta = json.loads((parent_dir / "meta.json").read_text())
    ids_by_art = _load_parent_articles(parent_dir, parent_meta)
    candidates, raw_counts = scan_parent_candidates(parent_dir, ids_by_art, parent_meta, args)
    tokenizer = common.get_tokenizer()
    for token_id, spec in TOKEN_SPECS.items():
        got = tokenizer.decode([token_id])
        assert got == spec["decoded"], (token_id, got, spec["decoded"])

    quotas = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    selected: dict[tuple[int, str], list[dict]] = {}
    screens: dict[str, dict] = {}
    eval_norms: set[str] = set()
    eval_texts: list[str] = []
    # Rarest identities claim duplicate-free eval rows first.
    arm_order = sorted(TOKEN_IDS, key=lambda tid: sum(raw_counts[str(tid)].values()))
    for split in ("test", "val"):
        for token_id in arm_order:
            rows, texts, stats = select_rows(
                candidates[(token_id, split)],
                quotas[split],
                tokenizer,
                ids_by_art,
                article_cap=args.max_per_article,
                forbidden_norms=eval_norms,
                near_gate=None,
            )
            assert len(rows) == quotas[split], (
                f"token {token_id} {split}: selected {len(rows)} < {quotas[split]}; "
                f"raw={raw_counts[str(token_id)].get(split, 0)} screens={stats}"
            )
            selected[(token_id, split)] = rows
            eval_texts.extend(texts)
            screens[f"{token_id}:{split}"] = stats

    near_gate = NearDupeGate(eval_texts, ngram=5, thresh=0.8)
    train_norms: set[str] = set()
    for token_id in arm_order:
        rows, _texts, stats = select_rows(
            candidates[(token_id, "train")],
            quotas["train"],
            tokenizer,
            ids_by_art,
            article_cap=args.max_per_article,
            forbidden_norms=train_norms,
            near_gate=near_gate,
        )
        assert len(rows) == quotas["train"], (
            f"token {token_id} train: selected {len(rows)} < {quotas['train']}; "
            f"raw={raw_counts[str(token_id)].get('train', 0)} screens={stats}"
        )
        selected[(token_id, "train")] = rows
        screens[f"{token_id}:train"] = stats

    rows_out = [
        row
        for token_id in TOKEN_IDS
        for split in ("train", "val", "test")
        for row in selected[(token_id, split)]
    ]
    assert len({r["row_id"] for r in rows_out}) == len(rows_out)
    split_articles = {
        split: {r["article_id"] for r in rows_out if r["split"] == split}
        for split in quotas
    }
    assert not (split_articles["train"] & split_articles["val"])
    assert not (split_articles["train"] & split_articles["test"])
    assert not (split_articles["val"] & split_articles["test"])
    chosen_articles = sorted({r["article_id"] for r in rows_out})
    selected_dir = args.out_root / "manifest"
    selected_dir.mkdir(parents=True, exist_ok=True)
    rows_name = "manifest_shard000.jsonl"
    articles_name = "articles_shard000.pt"
    _write_jsonl(selected_dir / rows_name, rows_out)
    torch.save(
        {
            "window_ids": chosen_articles,
            "input_ids": [ids_by_art[a].to(torch.int32) for a in chosen_articles],
        },
        selected_dir / articles_name,
    )
    row_sha = hashlib.sha256("\n".join(r["row_id"] for r in rows_out).encode()).hexdigest()
    meta = {
        "experiment": "issue1901_individual_boundary_tokens",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "parent_dir": str(parent_dir.resolve()),
            "parent_n_manifest_rows": int(parent_meta["n_manifest_rows"]),
            "parent_git_commit": parent_meta.get("metadata", {}).get("git_commit"),
        },
        "token_specs": {str(k): v for k, v in TOKEN_SPECS.items()},
        "split": {
            "grain": "article_id",
            "seed": int(args.split_seed),
            "fractions": {"train": 0.7, "val": 0.1, "test": 0.2},
        },
        "quotas_per_token_id": quotas,
        "max_per_article_per_token_split": int(args.max_per_article),
        "candidate_multiplier": int(args.candidate_multiplier),
        "raw_counts": raw_counts,
        "screening": {
            "exact_normalized_unique_across_eval": True,
            "exact_normalized_unique_across_train": True,
            "train_vs_eval_near_dupe": near_gate.stats(),
            "per_arm": screens,
        },
        "n_manifest_rows": len(rows_out),
        "n_articles": len(chosen_articles),
        "article_counts_by_split": {k: len(v) for k, v in split_articles.items()},
        "manifest_shards": [rows_name],
        "article_shards": [articles_name],
        "selected_row_ids_sha256": row_sha,
    }
    _write_json(selected_dir / "meta.json", meta)
    logger.info(
        "[prepare] selected %d rows (%d/token) over %d articles; row sha=%s",
        len(rows_out),
        sum(quotas.values()),
        len(chosen_articles),
        row_sha[:12],
    )
    return 0


def _load_selected(manifest_dir: Path) -> tuple[list[dict], dict[str, list[int]], dict]:
    meta = json.loads((manifest_dir / "meta.json").read_text())
    rows: list[dict] = []
    for name in meta["manifest_shards"]:
        rows.extend(ES._read_jsonl(manifest_dir / name))
    ids_by_art: dict[str, list[int]] = {}
    for name in meta["article_shards"]:
        payload = torch.load(manifest_dir / name, map_location="cpu", weights_only=True)
        for article_id, ids in zip(payload["window_ids"], payload["input_ids"], strict=True):
            ids_by_art[str(article_id)] = [int(v) for v in ids.tolist()]
    assert len(rows) == meta["n_manifest_rows"]
    for row in rows:
        assert int(ids_by_art[row["article_id"]][int(row["anchor_pos"])]) == int(
            row["boundary_token_id"]
        )
    return rows, ids_by_art, meta


def phase_capture(args) -> int:
    manifest_dir = args.out_root / "manifest"
    rows, ids_by_art, meta = _load_selected(manifest_dir)
    layer = SMOKE_LAYER if args.smoke else args.layer
    persist = (layer,)
    store = args.out_root / "store"
    store.mkdir(parents=True, exist_ok=True)
    done, next_idx = PARENT._scan_store_resume(store, persist)
    pending = [r for r in rows if r["row_id"] not in done]
    items = PARENT._items_from_manifest(pending, ids_by_art)
    tiny_dir = args.tiny_model_dir
    if args.smoke:
        tiny_dir = tiny_dir or str(args.out_root / "tiny_model")
        if not (Path(tiny_dir) / "config.json").exists():
            ES.make_tiny_model(Path(tiny_dir), layers=max(args.tiny_layers, layer + 1))
    model = ES.load_model(tiny_dir)
    tokenizer = common.get_tokenizer(tiny_dir or common.MODEL_ID)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    assert pad_id is not None
    logger.info(
        "[capture] %d pending rows / %d articles at exact layer %d",
        len(pending),
        len(items),
        layer,
    )
    buf: list[dict] = []
    shard_idx = next_idx

    def write(records: list[dict]) -> None:
        nonlocal shard_idx
        ES.write_shard(records, store, shard_idx, "armC", layers=persist)
        (store / f"armC_shard{shard_idx:03d}.pt").replace(
            store / f"pairs_shard{shard_idx:03d}.pt"
        )
        (store / f"armC_shard{shard_idx:03d}.json").replace(
            store / f"pairs_shard{shard_idx:03d}.json"
        )
        shard_idx += 1

    for records in ES.run_extraction(
        model, items, pad_id, args.batch_size, "armC", layers=persist
    ):
        buf.extend(records)
        while len(buf) >= args.shard_pairs:
            write(buf[: args.shard_pairs])
            buf = buf[args.shard_pairs :]
    if buf:
        write(buf)
    sidecars = sorted(store.glob("pairs_shard*.json"))
    n_captured = sum(json.loads(p.read_text())["n_rows"] for p in sidecars)
    assert n_captured == len(rows), (n_captured, len(rows))
    _write_json(
        args.out_root / "capture_meta.json",
        {
            "layer": layer,
            "n_rows": n_captured,
            "n_shards": len(sidecars),
            "manifest_row_ids_sha256": meta["selected_row_ids_sha256"],
        },
    )
    return 0


def _indices(rows: list[dict], row_pos: dict[str, int], token_id: int, split: str) -> np.ndarray:
    arm = [r for r in rows if int(r["boundary_token_id"]) == token_id and r["split"] == split]
    arm.sort(key=lambda r: int(r["selection_order"]))
    return np.asarray([row_pos[r["row_id"]] for r in arm], dtype=np.int64)


def _score(pred: np.ndarray, true: np.ndarray, articles: list[str], n_boot: int, seed: int) -> dict:
    r2, cosine = F79._recon_point(pred, true)
    ci = PARENT.article_cluster_boot(pred, true, articles, n_boot, seed)
    return {
        "r2": float(r2),
        "mean_cosine": float(cosine),
        "article_bootstrap_r2": ci,
        "n_test": int(len(true)),
    }


def _shuffle_null_r2(pred: np.ndarray, true: np.ndarray, n_draws: int, seed: int) -> dict:
    """Test-pair permutation null for R2, without materializing permuted tensors."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(len(true)) for _ in range(n_draws)])
    cross = pred @ true.T
    dots = cross[np.arange(len(true))[None, :], perms].sum(axis=1)
    sse = (pred**2).sum() + (true**2).sum() - 2.0 * dots
    sst = ((true - true.mean(0)) ** 2).sum()
    draws = 1.0 - sse / sst
    return {
        "n_draws": int(n_draws),
        "mean": float(draws.mean()),
        "lo": float(np.quantile(draws, 0.025)),
        "hi": float(np.quantile(draws, 0.975)),
    }


def paired_article_delta(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    true: np.ndarray,
    articles: list[str],
    n_boot: int,
    seed: int,
) -> dict:
    """Article-cluster bootstrap of R2(A)-R2(B) on the same test rows."""
    true = np.asarray(true, dtype=np.float64)
    res_a = ((true - pred_a) ** 2).sum(axis=1)
    res_b = ((true - pred_b) ** 2).sum(axis=1)
    unique = sorted(set(articles))
    lookup = {article: i for i, article in enumerate(unique)}
    labels = np.asarray([lookup[a] for a in articles])
    n_article = len(unique)
    counts = np.bincount(labels, minlength=n_article).astype(np.float64)
    sse_a = np.bincount(labels, weights=res_a, minlength=n_article)
    sse_b = np.bincount(labels, weights=res_b, minlength=n_article)
    sum_y = np.zeros((n_article, true.shape[1]), dtype=np.float64)
    np.add.at(sum_y, labels, true)
    sum_yy = np.bincount(labels, weights=(true**2).sum(1), minlength=n_article)
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, n_article, size=(n_boot, n_article))
    mult = np.zeros((n_boot, n_article), dtype=np.float64)
    np.add.at(mult, (np.arange(n_boot)[:, None], boot), 1.0)
    n_star = mult @ counts
    mean_star = (mult @ sum_y) / n_star[:, None]
    sst = mult @ sum_yy - n_star * (mean_star**2).sum(1)
    numerator = (mult @ sse_b) - (mult @ sse_a)
    delta = np.full_like(numerator, np.nan)
    np.divide(numerator, sst, out=delta, where=sst > 1e-12)
    delta = delta[np.isfinite(delta)]
    assert delta.size, "all paired bootstrap draws had degenerate target variance"
    point = float((res_b.sum() - res_a.sum()) / ((true - true.mean(0)) ** 2).sum())
    return {
        "point": point,
        "lo": float(np.quantile(delta, 0.025)),
        "hi": float(np.quantile(delta, 0.975)),
        "n_articles": n_article,
        "n_boot": int(n_boot),
    }


def _apply(payload: dict, X: torch.Tensor, idx: np.ndarray, device: torch.device) -> np.ndarray:
    x = X[torch.as_tensor(idx, dtype=torch.long)].to(torch.float64).numpy()
    return N1M.apply_map(payload, x, device)


def _apply_moment_aligned(
    payload: dict,
    X: torch.Tensor,
    idx: np.ndarray,
    target_xmu: np.ndarray,
    target_ymu: np.ndarray,
    device: torch.device,
    *,
    target_xsd: np.ndarray | None = None,
) -> np.ndarray:
    """Transfer a source map after aligning target input/output moments.

    ``target_xsd=None`` removes the target-vs-source input mean offset while
    retaining the source fit's scale. Supplying ``target_xsd`` additionally
    puts target inputs into their own train-arm z-score coordinates. Both are
    reported because a token-embedding mean shift must not masquerade as a
    failure of the learned linear relation.
    """
    xe = X[torch.as_tensor(idx, dtype=torch.long)].to(torch.float64).to(device)
    xmu = torch.as_tensor(target_xmu, dtype=torch.float64, device=device)
    scale_np = payload["xsd"].numpy() if target_xsd is None else target_xsd
    scale = torch.as_tensor(scale_np, dtype=torch.float64, device=device)
    ymu = torch.as_tensor(target_ymu, dtype=torch.float64, device=device)
    W = payload["W"].to(device, torch.float64)
    return (((xe - xmu) / scale) @ W + ymu).cpu().numpy()


def _fit_map(X, Y, tr, val, te, device, args):
    pred, meta, payload = N1M.fit_ridge_with_weights(
        X,
        Y,
        tr,
        val,
        te,
        PARENT._lambdas_for(len(tr)),
        device,
        args.ridge_block,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return pred, meta, payload


def phase_fit(args) -> int:  # noqa: C901 - explicit arm/cross-eval matrix is clearer inline
    rows, _ids_by_art, manifest_meta = _load_selected(args.out_root / "manifest")
    capture_meta = json.loads((args.out_root / "capture_meta.json").read_text())
    layer = int(capture_meta["layer"])
    X, Y, row_ids, article_ids = PARENT._load_layer_arrays(args.out_root / "store", layer, (layer,))
    assert len(set(row_ids)) == len(row_ids) == len(rows)
    row_pos = {row_id: i for i, row_id in enumerate(row_ids)}
    by_arm = {
        tid: {split: _indices(rows, row_pos, tid, split) for split in ("train", "val", "test")}
        for tid in TOKEN_IDS
    }
    for token_id, splits in by_arm.items():
        for split, idx in splits.items():
            want = int(manifest_meta["quotas_per_token_id"][split])
            assert len(idx) == want, (token_id, split, len(idx), want)
    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "--device cuda requested but CUDA is unavailable"

    result = {
        "experiment": "issue1901_individual_boundary_tokens",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": (
            "Does boundary-token-to-next-span mapping hold for each exact tokenizer ID, "
            "rather than only after pooling boundary identities?"
        ),
        "model": common.MODEL_ID if not args.smoke else str(args.tiny_model_dir or "tiny-random"),
        "layer": layer,
        "token_specs": {str(k): v for k, v in TOKEN_SPECS.items()},
        "manifest": manifest_meta,
        "fit": {
            "kind": "linear ridge",
            "lambda_selection": "held-out validation R2",
            "ridge_block": int(args.ridge_block),
            "n_article_boot": int(args.n_boot),
            "n_shuffle_null": int(args.n_null),
            "seed": int(args.fit_seed),
        },
        "individual": {},
        "pooled": {},
        "cross_id_transfer": {
            "raw_r2": {},
            "output_mean_recentered_r2": {},
            "input_output_mean_recentered_r2": {},
            "target_zscore_aligned_r2": {},
        },
    }
    payloads: dict[int, dict] = {}
    own_predictions: dict[int, np.ndarray] = {}
    true_by_arm: dict[int, np.ndarray] = {}
    articles_by_arm: dict[int, list[str]] = {}

    for arm_i, token_id in enumerate(TOKEN_IDS):
        idx = by_arm[token_id]
        pred, fit_meta, payload = _fit_map(
            X, Y, idx["train"], idx["val"], idx["test"], device, args
        )
        true = PARENT._to_f64_np(Y, idx["test"])
        articles = [article_ids[i] for i in idx["test"]]
        own_predictions[token_id] = pred
        true_by_arm[token_id] = true
        articles_by_arm[token_id] = articles
        payloads[token_id] = payload
        identity = PARENT._identity_bias_chunked(X, Y, idx["train"], idx["test"])
        train_mean = PARENT._to_f64_np(Y, idx["train"]).mean(0)
        constant = np.broadcast_to(train_mean, true.shape).copy()
        score = _score(pred, true, articles, args.n_boot, args.fit_seed + arm_i)
        result["individual"][str(token_id)] = {
            "fit_meta": fit_meta,
            "score": score,
            "identity_bias": _score(
                identity, true, articles, args.n_boot, args.fit_seed + 100 + arm_i
            ),
            "constant_train_mean": _score(
                constant, true, articles, args.n_boot, args.fit_seed + 200 + arm_i
            ),
            "shuffled_pair_null_r2": _shuffle_null_r2(
                pred, true, args.n_null, args.fit_seed + 300 + arm_i
            ),
        }
        logger.info("[fit] exact token %d R2=%.4f", token_id, score["r2"])

    target_moments = {}
    for target in TOKEN_IDS:
        x_train = PARENT._to_f64_np(X, by_arm[target]["train"])
        target_moments[target] = {
            "xmu": x_train.mean(0),
            "xsd": x_train.std(0, ddof=1) + 1e-9,
            "ymu": PARENT._to_f64_np(Y, by_arm[target]["train"]).mean(0),
        }

    # Cross-identity evaluation. Report raw transfer, the original output-only
    # recentering, input+output mean alignment (the primary transfer control),
    # and the stricter target-z-score alignment as a sensitivity analysis.
    for source in TOKEN_IDS:
        raw_row, output_row, mean_aligned_row, z_aligned_row = {}, {}, {}, {}
        source_mean = payloads[source]["ymu"].to(torch.float64).numpy()
        for target in TOKEN_IDS:
            te = by_arm[target]["test"]
            pred = _apply(payloads[source], X, te, device)
            moments = target_moments[target]
            raw_row[str(target)] = float(F79._recon_point(pred, true_by_arm[target])[0])
            output_row[str(target)] = float(
                F79._recon_point(pred + moments["ymu"] - source_mean, true_by_arm[target])[0]
            )
            pred_mean_aligned = _apply_moment_aligned(
                payloads[source], X, te, moments["xmu"], moments["ymu"], device
            )
            mean_aligned_row[str(target)] = float(
                F79._recon_point(pred_mean_aligned, true_by_arm[target])[0]
            )
            pred_z_aligned = _apply_moment_aligned(
                payloads[source],
                X,
                te,
                moments["xmu"],
                moments["ymu"],
                device,
                target_xsd=moments["xsd"],
            )
            z_aligned_row[str(target)] = float(
                F79._recon_point(pred_z_aligned, true_by_arm[target])[0]
            )
        result["cross_id_transfer"]["raw_r2"][str(source)] = raw_row
        result["cross_id_transfer"]["output_mean_recentered_r2"][str(source)] = output_row
        result["cross_id_transfer"]["input_output_mean_recentered_r2"][str(source)] = (
            mean_aligned_row
        )
        result["cross_id_transfer"]["target_zscore_aligned_r2"][str(source)] = z_aligned_row

    n_per_arm = len(by_arm[TOKEN_IDS[0]]["train"])
    v_per_arm = len(by_arm[TOKEN_IDS[0]]["val"])
    assert n_per_arm % len(TOKEN_IDS) == 0 and v_per_arm % len(TOKEN_IDS) == 0
    pooled_defs = {
        "matched_n": {
            "train": np.concatenate(
                [by_arm[t]["train"][: n_per_arm // len(TOKEN_IDS)] for t in TOKEN_IDS]
            ),
            "val": np.concatenate(
                [by_arm[t]["val"][: v_per_arm // len(TOKEN_IDS)] for t in TOKEN_IDS]
            ),
        },
        "all_data": {
            "train": np.concatenate([by_arm[t]["train"] for t in TOKEN_IDS]),
            "val": np.concatenate([by_arm[t]["val"] for t in TOKEN_IDS]),
        },
    }
    union_test = np.concatenate([by_arm[t]["test"] for t in TOKEN_IDS])
    pooled_predictions: dict[str, dict[int, np.ndarray]] = {}
    for pool_i, (name, pool) in enumerate(pooled_defs.items()):
        _pred_union, fit_meta, payload = _fit_map(
            X, Y, pool["train"], pool["val"], union_test, device, args
        )
        pooled_predictions[name] = {}
        per_token = {}
        for arm_i, token_id in enumerate(TOKEN_IDS):
            pred = _apply(payload, X, by_arm[token_id]["test"], device)
            pooled_predictions[name][token_id] = pred
            per_token[str(token_id)] = _score(
                pred,
                true_by_arm[token_id],
                articles_by_arm[token_id],
                args.n_boot,
                args.fit_seed + 400 + 10 * pool_i + arm_i,
            )
        result["pooled"][name] = {
            "n_train": int(len(pool["train"])),
            "n_val": int(len(pool["val"])),
            "per_token": per_token,
            "fit_meta": fit_meta,
        }
        del payload
        gc.collect()

    result["paired_individual_minus_pooled_r2"] = {}
    for token_i, token_id in enumerate(TOKEN_IDS):
        result["paired_individual_minus_pooled_r2"][str(token_id)] = {}
        for pool_i, name in enumerate(pooled_defs):
            result["paired_individual_minus_pooled_r2"][str(token_id)][name] = (
                paired_article_delta(
                    own_predictions[token_id],
                    pooled_predictions[name][token_id],
                    true_by_arm[token_id],
                    articles_by_arm[token_id],
                    args.n_boot,
                    args.fit_seed + 500 + 10 * token_i + pool_i,
                )
            )

    own = np.asarray([result["individual"][str(t)]["score"]["r2"] for t in TOKEN_IDS])
    pooled_matched = np.asarray(
        [result["pooled"]["matched_n"]["per_token"][str(t)]["r2"] for t in TOKEN_IDS]
    )
    pooled_all = np.asarray(
        [result["pooled"]["all_data"]["per_token"][str(t)]["r2"] for t in TOKEN_IDS]
    )
    mean_aligned = result["cross_id_transfer"]["input_output_mean_recentered_r2"]
    z_aligned = result["cross_id_transfer"]["target_zscore_aligned_r2"]
    off_diag_mean = np.asarray(
        [mean_aligned[str(s)][str(t)] for s in TOKEN_IDS for t in TOKEN_IDS if s != t]
    )
    off_diag_z = np.asarray(
        [z_aligned[str(s)][str(t)] for s in TOKEN_IDS for t in TOKEN_IDS if s != t]
    )
    result["summary"] = {
        "mean_individual_r2": float(own.mean()),
        "min_individual_r2": float(own.min()),
        "mean_pooled_matched_r2": float(pooled_matched.mean()),
        "mean_pooled_all_r2": float(pooled_all.mean()),
        "individual_minus_pooled_matched_mean_r2": float((own - pooled_matched).mean()),
        "individual_minus_pooled_all_mean_r2": float((own - pooled_all).mean()),
        "mean_input_output_recentered_off_diagonal_transfer_r2": float(off_diag_mean.mean()),
        "mean_target_zscore_aligned_off_diagonal_transfer_r2": float(off_diag_z.mean()),
    }
    _write_json(args.result, result)
    logger.info("[fit] wrote %s", args.result)
    return 0


def phase_figure(args) -> int:
    result = json.loads(args.result.read_text())
    set_paper_style("blog")
    import matplotlib.pyplot as plt

    colors = paper_palette(4)
    labels = [
        (f"space + {TOKEN_SPECS[t]['decoded'].strip()}\nID {t}" if TOKEN_SPECS[t]["decoded"].startswith(" ") else f"{TOKEN_SPECS[t]['decoded']}\nID {t}")
        for t in TOKEN_IDS
    ]
    x = np.arange(len(TOKEN_IDS))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    series = [
        (
            "individual map",
            [result["individual"][str(t)]["score"] for t in TOKEN_IDS],
            colors[0],
        ),
        (
            "pooled, matched n",
            [result["pooled"]["matched_n"]["per_token"][str(t)] for t in TOKEN_IDS],
            colors[1],
        ),
        (
            "pooled, all data",
            [result["pooled"]["all_data"]["per_token"][str(t)] for t in TOKEN_IDS],
            colors[2],
        ),
    ]
    for j, (name, scores, color) in enumerate(series):
        vals = np.asarray([v["r2"] for v in scores])
        lo = np.asarray([v["article_bootstrap_r2"]["lo"] for v in scores])
        hi = np.asarray([v["article_bootstrap_r2"]["hi"] for v in scores])
        axes[0].bar(x + (j - 1) * width, vals, width, label=name, color=color, alpha=0.9)
        xpos = x + (j - 1) * width
        # Draw percentile intervals directly: unlike a symmetric-error API,
        # this remains valid when a small-sample bootstrap interval happens not
        # to contain the full-sample point.
        axes[0].vlines(xpos, lo, hi, color="black", linewidth=1, zorder=4)
        axes[0].hlines(lo, xpos - 0.025, xpos + 0.025, color="black", linewidth=1, zorder=4)
        axes[0].hlines(hi, xpos - 0.025, xpos + 0.025, color="black", linewidth=1, zorder=4)
    axes[0].axhline(0, color="0.4", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Held-out next-span R²")
    axes[0].set_title("Exact boundary identities vs pooled maps")
    axes[0].legend(frameon=False, fontsize=9)

    matrix = np.asarray(
        [
            [
                result["cross_id_transfer"]["input_output_mean_recentered_r2"][str(s)][str(t)]
                for t in TOKEN_IDS
            ]
            for s in TOKEN_IDS
        ]
    )
    vmax = float(np.nanmax(np.abs(matrix)))
    im = axes[1].imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_xticks(x, labels)
    axes[1].set_yticks(x, labels)
    axes[1].set_xlabel("evaluated on target token")
    axes[1].set_ylabel("map trained on source token")
    axes[1].set_title("Cross-ID transfer (input/output means aligned)")
    for i in range(len(TOKEN_IDS)):
        for j in range(len(TOKEN_IDS)):
            color = "white" if abs(matrix[i, j]) > 0.55 * vmax else "black"
            axes[1].text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color)
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="R²")
    written = savefig_paper(fig, args.figure_stem, dir=args.figure_dir)
    logger.info("[figure] wrote %s", written)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--phase", required=True, choices=("prepare", "capture", "fit", "figure", "capture_fit")
    )
    ap.add_argument("--parent-manifest", type=Path, default=DEFAULT_PARENT_MANIFEST)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    ap.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    ap.add_argument("--figure-stem", default="individual_boundary_tokens")
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-val", type=int, default=160)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--max-per-article", type=int, default=6)
    ap.add_argument("--candidate-multiplier", type=int, default=20)
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--fit-seed", type=int, default=FIT_SEED)
    ap.add_argument("--layer", type=int, default=PROD_LAYER)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard-pairs", type=int, default=1000)
    ap.add_argument("--ridge-block", type=int, default=N1M.RIDGE_BLOCK)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tiny-model-dir", type=str, default=None)
    ap.add_argument("--tiny-layers", type=int, default=4)
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    assert args.n_train > 0 and args.n_val > 0 and args.n_test > 0
    assert args.n_train % len(TOKEN_IDS) == 0, "n_train must be divisible by four"
    assert args.n_val % len(TOKEN_IDS) == 0, "n_val must be divisible by four"
    if args.smoke:
        args.n_train = min(args.n_train, 8)
        args.n_val = min(args.n_val, 4)
        args.n_test = min(args.n_test, 4)
        args.n_boot = min(args.n_boot, 30)
        args.n_null = min(args.n_null, 20)
        args.shard_pairs = min(args.shard_pairs, 32)
        if args.device == "cuda" and not torch.cuda.is_available():
            args.device = "cpu"
    if args.phase == "prepare":
        return phase_prepare(args)
    if args.phase == "capture":
        return phase_capture(args)
    if args.phase == "fit":
        return phase_fit(args)
    if args.phase == "figure":
        return phase_figure(args)
    if args.phase == "capture_fit":
        phase_capture(args)
        return phase_fit(args)
    raise AssertionError(args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
