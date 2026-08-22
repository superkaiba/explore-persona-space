"""Issue #2224 same-issue follow-up round 1 driver (proposer-9b-cheap).

Three legs over EXISTING #2224 artifacts (no new training, no new generations):

- **Leg 1 (``refit``, pod GPU)** — E1 per-corpus context->answer map REFIT at
  sample level over the banked P0c predictor summaries (HF
  ``issue2224_screening/analysis_tensors/predictor_summaries/{lmsys,ultrachat}``),
  BOTH mapping arms (context-end AND prefix-end -> response-avg), 5-fold
  held-out R^2 + identity+learned-bias baseline + kNN retrieval (standing
  rules), then the calibration read that failed for the #2222 frozen map
  (r vs actual base-gen projections) and top/bottom-500 selection Jaccard vs
  exact dP, re-derived from the refit maps on held-out rows.
- **Leg 2 (``transport``, same pod run)** — cross-corpus probe transport:
  ridge probe (the Form-A convention: answer-space activations -> graded judge
  label) trained on corpus A, scored on corpus B, both directions, all 3
  traits; same-corpus 5-fold held-out AUC as the reference row.
- **Leg 3 (``rejudge-*``, VM-side, API-only)** — the 17 post-finetune
  trait-score cells below the 0.95 judge-completeness floor: drop-class
  triage from the persisted judge raw outputs, surgical re-judge of
  RECOVERABLE classes only (transport-lost + api-refusal re-issue; bounded
  same-instrument re-draw of stochastic-malformed unscored items — the
  ``issue2224_select.judge_with_redraw`` convention), then updated cell
  means + completeness + headline-contrast movement.

Content hygiene: this driver handles LMSYS/UltraChat-derived text internally
but NEVER logs or persists prompt/response text beyond what the parent
pipeline already persisted; all new JSON outputs carry ids/counts/digests
only (the rejudge save_raw files carry parsed score dicts, no text).

Stream-reduce staging (code-style rule): capture shards are downloaded ONE at
a time, sliced to the needed (kind, layer) pairs, and deleted — the 40 GB per
corpus shard set is never materialized (~5.7 GB of fp16 slices per corpus).

Smoke blind-spot enumeration (what the VM smoke's PASS does NOT certify):

- ``--device cuda`` GPU routing unexercised on the CPU-only VM host (the
  vendored ridge cores carry the #825 CPU-eigh fallback); first exercised on
  the pod.
- Production fit shape d=3584 first runs on the pod: ``--smoke-dim`` is a
  smoke-only dial REFUSED without ``--limit-shards`` (cannot leak into
  production).
- ``phase_transport``'s <10-judged-rows cell check is DOWNGRADED under smoke
  (warn + skip the cell) but RAISES in production.
- The stage phase's full-corpus coverage assert (rows == manifest
  ``n_samples``) is SKIPPED under ``--limit-shards``.
- Worktree-vs-fresh-clone input gap: VM-local/untracked inputs (screening
  score tables, eval questions, generations, judge raws) are staged
  local-first/HF-fallback (``ensure_screening_tables`` /
  ``_stage_local_or_hf``); a VM smoke where the local copy exists cannot
  detect a broken HF fallback path.
- The R2 redraw loop's live-call body is not reached by the 1-cell smoke
  (that cell has ``r2_items=0``); it is the same ``judge_graded`` call the R1
  smoke exercises.
- The upload smoke probes a scratch ``--prefix-root``; production differs
  only by the prefix string.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from issue2224_common import (
    PROJECT_ROOT,
    atomic_write_json,
    load_jsonl,
    repro_meta,
    sha256_file,
    stable_seed,
)
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE numpy/torch imports: shared-VM thread caps + HF token (#847)

import numpy as np  # noqa: E402

logger = logging.getLogger("issue2224_followup_r1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Constants ────────────────────────────────────────────────────────────────────

CORPORA = ("lmsys", "ultrachat")
TRAITS = ("evil", "hallucination", "sycophancy")
KINDS = ("resp_avg_dataset", "resp_avg_natural", "last_prompt", "prefix_end")
ARM_INPUT_KIND = {"context": "last_prompt", "prefix": "prefix_end"}
MAP_ARMS = ("context", "prefix")

DATA_REPO = "superkaiba1/explore-persona-space-data"
SUMMARIES_PREFIX = "issue2224_screening/analysis_tensors/predictor_summaries"
RB_PREFIX = "issue778_persona_vectors/analysis_tensors_v2/rb_v2"  # suite_slice.RB_PREFIX
POSTFT_PREFIX = "issue2224_screening/raw_completions/postft_eval"
EVAL_Q_PREFIX = "issue2224_screening/eval_questions"
PACKED_PREFIX = "issue2224_screening/judge_postft_packed"
SCREENING_HF_PREFIX = "issue2224_screening/screening_scores"
HF_FU_PREFIX = "issue2224_screening/followup_r1"

# Fit conventions (issue2224_probe_refit reference regime: 17-point log grid
# 1e-2..1e6, dof_cap 0.9 — the #1887 dof-capped GCV convention).
LAMBDAS = np.logspace(-2.0, 6.0, 17)
DOF_CAP = 0.9
N_FOLDS = 5
TOP_N = 500  # issue2224_select.TOP_N (paper top/bottom-500)
KNN_N = 5000  # seeded retrieval subsample (full 50k x 50k distance = 20 GB, too big)
KNN_KS = (1, 5, 10)
COMPLETENESS_FLOOR = 0.95
REJUDGE_MAX_REDRAW_ROUNDS = 2  # issue2224_select.REDRAW_MAX_ROUNDS (<=3 attempts/item)

RESULTS_DIR_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_2224" / "followup_r1"
OUT_ROOT_DEFAULT = PROJECT_ROOT / "data" / "issue_2224" / "followup_r1"
SCREENING_SCORES_DIR = PROJECT_ROOT / "eval_results" / "issue_2224" / "screening_scores"
SELECTIONS_DIR = PROJECT_ROOT / "eval_results" / "issue_2224" / "selections"
TRAIT_SCORES_DIR = PROJECT_ROOT / "eval_results" / "issue_2224" / "selection_finetune"
ANALYSIS_4B_DIR = PROJECT_ROOT / "eval_results" / "issue_2224" / "analysis_4b"
JUDGE_POSTFT_LOCAL = PROJECT_ROOT / "data" / "issue_2224" / "judge_postft"
EVAL_Q_LOCAL = PROJECT_ROOT / "data" / "issue_2224" / "eval_questions"
GEN_LOCAL_ROOT = PROJECT_ROOT / "data" / "issue_2224" / "screening_ft" / "postft_eval"
RB_LOCAL = PROJECT_ROOT / "data" / "issue_2224" / "hf_dl" / "rb_v2"


# ── Shared helpers ───────────────────────────────────────────────────────────────


def ensure_screening_tables(screening_dir: Path) -> Path:
    """Stage the 6 per-corpus screening score tables local-first / HF-fallback.

    The tables are worktree-local UNTRACKED on the VM (not committed on
    ``issue-2224``), so a fresh pod clone has none of them — every pod-side
    phase stages them from ``issue2224_screening/screening_scores/`` before
    any read (fu-r1 code-review r1 blocker: FileNotFoundError at phase 1 on a
    fresh clone; the #779/#1773 lane-input-staging class).
    """
    for corpus in CORPORA:
        for trait in TRAITS:
            _stage_local_or_hf(
                screening_dir / corpus / f"{trait}.json",
                f"{SCREENING_HF_PREFIX}/{corpus}/{trait}.json",
            )
    return screening_dir


def resolve_trait_layers(screening_dir: Path) -> dict[str, int]:
    """Read-out layer per trait from the committed score tables' meta.

    The realized ``readout_layer`` is part of each corpus score table; the two
    corpora must agree per trait (same #2222 selection) — fail loud otherwise.
    """
    layers: dict[str, int] = {}
    for trait in TRAITS:
        per_corpus = set()
        for corpus in CORPORA:
            p = screening_dir / corpus / f"{trait}.json"
            meta = json.loads(p.read_text())["meta"]
            per_corpus.add(int(meta["readout_layer"]))
        if len(per_corpus) != 1:
            raise RuntimeError(f"{trait}: readout_layer differs across corpora: {per_corpus}")
        layers[trait] = per_corpus.pop()
    return layers


def _pooled_r2(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Pooled held-out R^2 per target column ((n, T) arrays), mirroring the
    vendored ``dof_capped_ridge_multi_y`` tail."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_res = ((y - pred) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)


def _r2_summary(r2: np.ndarray) -> dict:
    r2 = np.asarray(r2, dtype=np.float64)
    finite = r2[np.isfinite(r2)]
    return {
        "mean_over_dims": float(finite.mean()),
        "median_over_dims": float(np.median(finite)),
        "p5_over_dims": float(np.percentile(finite, 5)),
        "p95_over_dims": float(np.percentile(finite, 95)),
        "n_dims": int(finite.size),
    }


def _affine_calibration(exact: np.ndarray, standin: np.ndarray) -> dict:
    """Affine-fit calibration block, field-for-field the shape of
    ``issue2224_free_analysis.run_calibration`` (r, slopes, quantiles, scale
    ratios) so refit numbers read side-by-side with the frozen-map probe."""
    x = np.asarray(exact, dtype=np.float64)
    y = np.asarray(standin, dtype=np.float64)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise RuntimeError("non-finite projections in calibration inputs")
    r = float(np.corrcoef(x, y)[0, 1])
    cov = float(np.cov(x, y, ddof=1)[0, 1])
    qs = (5.0, 50.0, 95.0)
    qx = {f"p{int(q)}": float(np.percentile(x, q)) for q in qs}
    qy = {f"p{int(q)}": float(np.percentile(y, q)) for q in qs}
    spread_x = qx["p95"] - qx["p5"]
    spread_y = qy["p95"] - qy["p5"]
    return {
        "pearson_r": r,
        "r2": r * r,
        "slope_standin_on_exact": cov / float(np.var(x, ddof=1)),
        "slope_exact_on_standin": cov / float(np.var(y, ddof=1)),
        "exact_quantiles": qx,
        "standin_quantiles": qy,
        "scale_ratio_p95_minus_p5": float(spread_y / spread_x) if spread_x > 0 else None,
        "scale_ratio_std": float(y.std() / x.std()) if x.std() > 0 else None,
        "n": int(x.size),
    }


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def _tail_sets(ids: list[str], scores: np.ndarray, k: int) -> tuple[set, set]:
    """Top-k / bottom-k id sets under the ``issue2224_select.ranked_ids``
    convention (score desc, id asc tie-break — #1946 stable-tie lesson)."""
    from issue2224_select import ranked_ids

    table = {sid: {"s": float(v)} for sid, v in zip(ids, scores)}
    order = ranked_ids(table, "s")
    return set(order[:k]), set(order[-k:])


def _fold_ids(n: int, n_folds: int, seed_key: str) -> np.ndarray:
    """Deterministic near-equal random folds (permutation mod n_folds)."""
    rng = np.random.default_rng(stable_seed("fu-r1-folds", seed_key))
    return rng.permutation(n) % n_folds


# ── Phase: stage (stream-reduce the capture shards) ──────────────────────────────


def _shard_census(corpus: str, scratch: Path) -> tuple[dict, list[dict]]:
    """(manifest, ordered shard rows) for one corpus's HF summaries prefix.

    Mirrors ``issue2224_predictor_scores.load_layer_slices``: concatenate every
    ``shards_s*.jsonl``, dedup append-log rows by FILE (last write wins), order
    by file name; manifest must read status=complete.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import list_hf_files_under_path, stage_hub_file

    prefix = f"{SUMMARIES_PREFIX}/{corpus}"
    remote = list_hf_files_under_path(HfApi(), DATA_REPO, prefix, repo_type="dataset")
    names = {p.rsplit("/", 1)[1] for p in remote}
    meta_dir = scratch / corpus / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = meta_dir / "manifest.json"
    if not manifest_path.exists():
        stage_hub_file(DATA_REPO, f"{prefix}/manifest.json", manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"{corpus}: manifest status={manifest.get('status')!r} — incomplete")
    shard_rows: list[dict] = []
    for name in sorted(n for n in names if n.startswith("shards_s") and n.endswith(".jsonl")):
        local = meta_dir / name
        if not local.exists():
            stage_hub_file(DATA_REPO, f"{prefix}/{name}", local)
        shard_rows.extend(load_jsonl(local))
    if not shard_rows:
        raise RuntimeError(f"{corpus}: no shards_s*.jsonl rows under {prefix}")
    shard_rows = sorted({r["file"]: r for r in shard_rows}.values(), key=lambda r: r["file"])
    return manifest, shard_rows


def _part_path(out_root: Path, corpus: str, shard_file: str) -> Path:
    return out_root / "slices" / corpus / "parts" / f"{shard_file}.npz"


def _part_regime(layers: list[int], shard_file: str) -> dict:
    return {"kinds": list(KINDS), "layers": [int(x) for x in layers], "shard_file": shard_file}


def _part_ok(path: Path, regime: dict) -> bool:
    """Resume predicate: part exists AND its embedded regime matches (#722 r3)."""
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            prior = json.loads(str(z["regime"]))
    except Exception:
        return False
    return prior == regime


def phase_stage(args) -> int:
    """Stream-reduce capture shards -> per-(kind, layer) fp16 slice parts.

    One shard at a time: ``stage_hub_file`` -> slice needed layers -> atomic
    part npz -> delete the shard .pt (peak transient ~= one shard, ~0.4 GB).
    Per-unit persistence + resume (code-style T2: 100 shards/corpus > ~50).
    """
    import torch

    layers = sorted(
        set(resolve_trait_layers(ensure_screening_tables(Path(args.screening_dir))).values())
    )
    out_root = Path(args.out_root)
    corpora = [c.strip() for c in args.corpora.split(",") if c.strip()]
    for corpus in corpora:
        manifest, shard_rows = _shard_census(corpus, out_root / "slices")
        if args.limit_shards is not None:
            shard_rows = shard_rows[: args.limit_shards]
        dl_dir = out_root / "slices" / corpus / "dl"
        dl_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for i, row in enumerate(shard_rows):
            part = _part_path(out_root, corpus, row["file"])
            regime = _part_regime(layers, row["file"])
            if _part_ok(part, regime):
                print(
                    f"[stage] unit {i + 1}/{len(shard_rows)} {corpus}/{row['file']} "
                    f"resume-skip elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
                continue
            local_pt = dl_dir / row["file"]
            from explore_persona_space.orchestrate.hub import stage_hub_file

            stage_hub_file(
                DATA_REPO, f"{SUMMARIES_PREFIX}/{corpus}/{row['file']}", local_pt, overwrite=True
            )
            # Self-produced sha-pinned shard bundles (dict of tensors + python
            # lists) — weights_only=False is the sanctioned load for these
            # (mirrors load_layer_slices; torch>=2.6 weights_only lesson).
            payload = torch.load(local_pt, map_location="cpu", weights_only=False)
            arrays: dict[str, np.ndarray] = {}
            for kind in KINDS:
                if kind not in payload["kinds"]:
                    raise RuntimeError(f"{corpus}/{row['file']}: kind {kind!r} missing")
                tens = payload["kinds"][kind]
                for layer in layers:
                    arrays[f"{kind}__L{layer}"] = tens[:, layer, :].numpy()
            sample_ids = np.array([str(s) for s in payload["sample_ids"]])
            part.parent.mkdir(parents=True, exist_ok=True)
            tmp = part.with_name(f"{part.stem}.tmp.npz")  # np.savez appends .npz otherwise
            np.savez(
                tmp,
                sample_ids=sample_ids,
                regime=np.array(json.dumps(regime)),
                **arrays,
            )
            os.replace(tmp, part)
            local_pt.unlink()  # stream-reduce: never accumulate the 40 GB shard set
            print(
                f"[stage] unit {i + 1}/{len(shard_rows)} {corpus}/{row['file']} "
                f"n={len(sample_ids)} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        # Coverage assert (skipped under an explicit shard limit — smoke slices).
        if args.limit_shards is None:
            ids, _, _ = load_slices(args, corpus, layers, kinds=("last_prompt",))
            if len(ids) != int(manifest.get("n_samples", -1)):
                raise RuntimeError(
                    f"{corpus}: {len(ids)} staged rows != manifest n_samples "
                    f"{manifest.get('n_samples')}"
                )
    # rb_v2 persona vectors: local-first, HF fallback (lane-staging rule).
    rb_dir = Path(args.rb_dir)
    rb_dir.mkdir(parents=True, exist_ok=True)
    for trait in TRAITS:
        target = rb_dir / f"{trait}.pt"
        if not target.exists():
            from explore_persona_space.orchestrate.hub import stage_hub_file

            stage_hub_file(DATA_REPO, f"{RB_PREFIX}/{trait}.pt", target)
    print(f"[stage] complete corpora={','.join(corpora)} layers={layers}", flush=True)
    return 0


def load_slices(
    args, corpus: str, layers: list[int], kinds: tuple[str, ...] = KINDS
) -> tuple[list[str], dict[tuple[str, int], np.ndarray], dict]:
    """Concatenate staged parts in census order; duplicate-id assert (fp16 out)."""
    out_root = Path(args.out_root)
    manifest_path = out_root / "slices" / corpus / "meta" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    parts_dir = out_root / "slices" / corpus / "parts"
    part_files = sorted(parts_dir.glob("*.pt.npz"))
    if not part_files:
        raise RuntimeError(f"{corpus}: no staged parts under {parts_dir} — run --phase stage")
    ids: list[str] = []
    buf: dict[tuple[str, int], list[np.ndarray]] = {}
    for pf in part_files:
        with np.load(pf, allow_pickle=False) as z:
            ids.extend(str(s) for s in z["sample_ids"])
            for kind in kinds:
                for layer in layers:
                    buf.setdefault((kind, layer), []).append(z[f"{kind}__L{layer}"])
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{corpus}: duplicate sample_ids across staged parts")
    slices = {k: np.concatenate(v, axis=0) for k, v in buf.items()}
    return ids, slices, manifest


# ── Phase: refit (leg 1 — E1 per-corpus map refit) ───────────────────────────────


def _load_rb(rb_dir: Path, trait: str, n_layers: int, hidden: int) -> np.ndarray:
    from issue2224_predictor_scores import load_rb

    return load_rb(rb_dir / f"{trait}.pt", (n_layers, hidden))


def _identity_bias_heldout(x: np.ndarray, y: np.ndarray, fold_ids: np.ndarray) -> np.ndarray:
    """Held-out identity+learned-bias predictions under the SAME folds."""
    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    pred = np.full(y.shape, np.nan)
    for f in np.unique(fold_ids):
        hold = fold_ids == f
        pred[hold] = identity_bias_predict(x[~hold], y[~hold], x[hold])
    return pred


def _knn_block(pred: np.ndarray, true: np.ndarray, seed_key: str, knn_n: int) -> dict:
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    n = pred.shape[0]
    take = min(knn_n, n)
    idx = np.random.default_rng(stable_seed("fu-r1-knn", seed_key)).choice(n, take, replace=False)
    out = {}
    for metric in ("euclidean", "cosine"):
        out[metric] = knn_retrieval(pred[idx], true[idx], ks=KNN_KS, metric=metric)
    out["note"] = f"seeded {take}-row held-out subsample; pool == true targets; chance = k/n_pool"
    return out


def _save_map_npz(path: Path, fits: dict[int, dict], layers: list[int], meta: dict) -> None:
    """Persist the refit map in the #1739 ``_save_map`` npz contract so
    ``issue2224_predictor_scores.load_linear_map`` + ``apply_map_at_layer``
    (and ``check_map_pooling`` via meta['variant']) consume it unchanged.

    ``pred = ((x - x_mu)/x_sd) @ w + y_mu`` with x_mu=0, x_sd=1, y_mu=b0
    reproduces the vendored ridge's ``x @ w + b0`` exactly.
    """
    d = fits[layers[0]]["w"].shape[0]
    w = np.stack([fits[layer]["w"] for layer in layers]).astype(np.float64)  # (Ly, d, T)
    y_mu = np.stack([fits[layer]["b0"][None, :] for layer in layers])  # (Ly, 1, T)
    x_mu = np.zeros((len(layers), 1, d))
    x_sd = np.ones((len(layers), 1, d))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp.npz")
    np.savez(
        tmp,
        w=w,
        x_mu=x_mu,
        x_sd=x_sd,
        y_mu=y_mu,
        layers=np.array(layers, dtype=np.int64),
        meta=np.array(json.dumps(meta)),
    )
    os.replace(tmp, path)


def phase_refit(args) -> int:
    """Leg 1: per-corpus map refit, both mapping arms, held-out reads."""
    from issue2224_vendored_ridge import (
        dof_capped_ridge_fit_all,
        dof_capped_ridge_multi_y,
    )

    trait_layers = resolve_trait_layers(ensure_screening_tables(Path(args.screening_dir)))
    layers = sorted(set(trait_layers.values()))
    out_root = Path(args.out_root)
    results_dir = Path(args.results_dir)
    corpora = [c.strip() for c in args.corpora.split(",") if c.strip()]
    smoke = args.smoke_dim is not None
    if smoke and args.limit_shards is None:
        raise RuntimeError("--smoke-dim is a smoke-only dial: it requires --limit-shards")

    for corpus in corpora:
        out_json = results_dir / f"refit_{corpus}.json"
        regime = {
            "layers": layers,
            "n_folds": N_FOLDS,
            "lambda_grid": [float(x) for x in LAMBDAS],
            "dof_cap": DOF_CAP,
            "smoke_dim": args.smoke_dim,
            "limit_shards": args.limit_shards,
            "knn_n": args.knn_n,
        }
        if out_json.exists():
            prior = json.loads(out_json.read_text())
            if prior.get("regime") == regime:
                print(f"[refit] {corpus} resume-skip (regime match)", flush=True)
                continue
        ids, slices, manifest = load_slices(args, corpus, layers)
        n = len(ids)
        n_layers_m, hidden = int(manifest["n_layers"]), int(manifest["hidden"])
        fold_ids = _fold_ids(n, N_FOLDS, f"refit-{corpus}")
        min_ntr = int(min((fold_ids != f).sum() for f in np.unique(fold_ids)))

        # Smoke-only dimension cut (production d=3584 needs n_tr > d; a
        # limited-shard smoke slice cannot satisfy that — #1345 gate-calibration
        # class, disclosed in the runner/report smoke blind-spot enumeration).
        d_full = hidden
        if smoke:
            d_use = min(args.smoke_dim, max(4, min_ntr - 4))
            col_idx = np.random.default_rng(stable_seed("fu-r1-smokedim", corpus)).choice(
                hidden, d_use, replace=False
            )
        else:
            d_use = d_full
            col_idx = None

        # Frozen-map screening scores (committed tables) for refit-vs-frozen r.
        frozen_scores: dict[str, dict] = {}
        for trait in TRAITS:
            tbl = json.loads((Path(args.screening_dir) / corpus / f"{trait}.json").read_text())
            frozen_scores[trait] = tbl["scores"]

        corpus_out: dict = {
            "regime": regime,
            "meta": {
                **repro_meta("issue2224_followup_r1.refit"),
                "corpus": corpus,
                "n_samples": n,
                "d": d_use,
                "d_full": d_full,
                "n_train_min_across_folds": min_ntr,
                "n_train_vs_d": f"n_train_min={min_ntr} > d={d_use} (well-posed primal ridge)",
                "pooling": (
                    "context arm x=last_prompt (context_end), prefix arm x=prefix_end; "
                    "y=resp_avg_natural (response-avg of the base greedy generation) — "
                    "matches the frozen map's response_avg <- context_end convention"
                ),
                "shared_term_caveat": (
                    "score-level r (mapped_dp vs exact_dp) shares the a_ds@v term in both "
                    "scores (the #383 shared-term class); the standin-level r "
                    "(M(v_C)@v vs a_nat@v) is the clean read — both reported"
                ),
                "jaccard_chance": (
                    f"two random {TOP_N}-of-{n} sets: E[J] ~= {TOP_N / max(1, 2 * n - TOP_N):.4f}"
                ),
            },
            "arms": {},
        }
        for arm in MAP_ARMS:
            x_kind = ARM_INPUT_KIND[arm]
            arm_out: dict = {"layers": {}}
            final_fits: dict[int, dict] = {}
            for layer in layers:
                x = slices[(x_kind, layer)].astype(np.float32)
                y = slices[("resp_avg_natural", layer)].astype(np.float32)
                a_ds = slices[("resp_avg_dataset", layer)].astype(np.float64)
                if col_idx is not None:
                    x, y, a_ds = x[:, col_idx], y[:, col_idx], a_ds[:, col_idx]
                t0 = time.time()
                fit = dof_capped_ridge_multi_y(
                    x,
                    y.astype(np.float64),
                    fold_ids,
                    lambdas=LAMBDAS,
                    dof_cap=DOF_CAP,
                    device=args.device,
                )
                heldout = fit["heldout_pred"]  # (n, d) — every row from its fold's fit
                ident = _identity_bias_heldout(x.astype(np.float64), y.astype(np.float64), fold_ids)
                lam_sel = np.asarray(fit["gcv_lambda"], dtype=np.float64)
                layer_out: dict = {
                    "heldout_r2": _r2_summary(fit["heldout_r2"]),
                    "identity_bias_heldout_r2": _r2_summary(
                        _pooled_r2(y.astype(np.float64), ident)
                    ),
                    "selected_lambda": {
                        "median": float(np.median(lam_sel)),
                        "p5": float(np.percentile(lam_sel, 5)),
                        "p95": float(np.percentile(lam_sel, 95)),
                        "grid_lo": float(LAMBDAS[0]),
                        "grid_hi": float(LAMBDAS[-1]),
                    },
                    "knn_retrieval": {
                        "refit": _knn_block(
                            heldout, y.astype(np.float64), f"{corpus}-{arm}-{layer}", args.knn_n
                        ),
                        "identity_bias": _knn_block(
                            ident, y.astype(np.float64), f"{corpus}-{arm}-{layer}-id", args.knn_n
                        ),
                    },
                    "fit_seconds": round(time.time() - t0, 1),
                    "traits": {},
                }
                # Per-trait projection reads at this layer.
                a_nat = y.astype(np.float64)
                for trait, t_layer in trait_layers.items():
                    if t_layer != layer:
                        continue
                    rb = _load_rb(Path(args.rb_dir), trait, n_layers_m, d_full)
                    v = rb[layer]
                    if col_idx is not None:
                        v = v[col_idx]
                    v_hat = v / np.linalg.norm(v)
                    exact = (a_ds - a_nat) @ v_hat
                    refit_score = (a_ds - heldout) @ v_hat
                    ident_score = (a_ds - ident) @ v_hat
                    standin_refit = heldout @ v_hat
                    standin_nat = a_nat @ v_hat
                    frozen_arm = f"mapped_dp_{arm}"
                    frozen = np.array([float(frozen_scores[trait][sid][frozen_arm]) for sid in ids])
                    k = TOP_N if not smoke else max(5, n // 10)
                    top_e, bot_e = _tail_sets(ids, exact, k)
                    top_r, bot_r = _tail_sets(ids, refit_score, k)
                    top_f, bot_f = _tail_sets(ids, frozen, k)
                    tr_out = {
                        "readout_layer": layer,
                        "score_level_calibration_vs_exact": _affine_calibration(exact, refit_score),
                        "standin_level_calibration": _affine_calibration(
                            standin_nat, standin_refit
                        ),
                        "frozen_map_score_r_vs_exact": float(np.corrcoef(exact, frozen)[0, 1]),
                        "refit_vs_frozen_score_r": float(np.corrcoef(refit_score, frozen)[0, 1]),
                        "identity_bias_score_r_vs_exact": float(
                            np.corrcoef(exact, ident_score)[0, 1]
                        ),
                        "projection_r2_refit": float(
                            _pooled_r2(standin_nat[:, None], standin_refit[:, None])[0]
                        ),
                        "jaccard_k": k,
                        "jaccard_top_refit_vs_exact": _jaccard(top_r, top_e),
                        "jaccard_bottom_refit_vs_exact": _jaccard(bot_r, bot_e),
                        "jaccard_top_frozen_vs_exact": _jaccard(top_f, top_e),
                        "jaccard_bottom_frozen_vs_exact": _jaccard(bot_f, bot_e),
                    }
                    layer_out["traits"][trait] = tr_out
                    # Per-sample derived scores (durable, regen-cheap): npz.
                    sc_path = out_root / "refit" / f"{corpus}__{arm}__L{layer}__{trait}_scores.npz"
                    sc_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = sc_path.with_name(f"{sc_path.stem}.tmp.npz")
                    np.savez(
                        tmp,
                        sample_ids=np.array(ids),
                        exact=exact.astype(np.float32),
                        refit_score=refit_score.astype(np.float32),
                        frozen_score=frozen.astype(np.float32),
                        identity_score=ident_score.astype(np.float32),
                        standin_refit=standin_refit.astype(np.float32),
                        standin_nat=standin_nat.astype(np.float32),
                    )
                    os.replace(tmp, sc_path)
                arm_out["layers"][str(layer)] = layer_out
                # Final deployable map on ALL rows (the E1 artifact).
                final_fits[layer] = dof_capped_ridge_fit_all(
                    x.astype(np.float64),
                    y.astype(np.float64),
                    lambdas=LAMBDAS,
                    dof_cap=DOF_CAP,
                    device=args.device,
                )
                print(
                    f"[refit] unit {corpus}/{arm}/L{layer} n={n} d={d_use} "
                    f"r2_mean={layer_out['heldout_r2']['mean_over_dims']:.4f} "
                    f"elapsed={layer_out['fit_seconds']}s",
                    flush=True,
                )
            map_meta = {
                "issue": 2224,
                "round": "followup_r1_E1_refit",
                "corpus": corpus,
                "variant": {"context": "context_end", "prefix": "prefix_end"}[arm],
                "fit": "dof_capped_ridge_fit_all (vendored #2222 cores)",
                "n_train": n,
                "d": d_use,
                "smoke": smoke,
                "selected_lambda_median_by_layer": {
                    str(layer): float(np.median(final_fits[layer]["lam"])) for layer in layers
                },
            }
            _save_map_npz(
                out_root / "refit" / f"{corpus}__{arm}__map.npz", final_fits, layers, map_meta
            )
            corpus_out["arms"][arm] = arm_out
        atomic_write_json(corpus_out, out_json)
        print(f"[refit] wrote {out_json}", flush=True)
    return 0


# ── Phase: transport (leg 2 — cross-corpus probe transport) ──────────────────────


def _auc_with_ci(x: np.ndarray, graded: np.ndarray, seed_key: str) -> dict:
    """Point AUC + vectorized bootstrap CI per binarization (free_analysis
    conventions: LABEL_THRESHOLDS / N_BOOT / chunked shared-index bootstrap)."""
    from scipy.stats import rankdata

    from issue2224_free_analysis import (
        BOOT_CHUNK,
        DEGENERATE_POS_FLOOR,
        LABEL_THRESHOLDS,
        N_BOOT,
        _auc_from_ranks,
    )

    n = x.size
    out = {}
    rng = np.random.default_rng(stable_seed("fu-r1-auc", seed_key))
    for tname, (op, thr) in LABEL_THRESHOLDS.items():
        y = ((graded >= thr) if op == "ge" else (graded > thr)).astype(np.float64)
        n_pos = int(y.sum())
        if n_pos == 0 or n_pos == n:
            # Single-class binarization (tiny smoke slices; a saturated cell):
            # AUC undefined — report the degeneracy, never crash (#1415
            # companion-stat drop-class semantics).
            out[tname] = {
                "auc": None,
                "ci95": None,
                "ci_method": "undefined — single-class labels",
                "n": int(n),
                "n_pos": n_pos,
                "degenerate": True,
            }
            continue
        point = float(_auc_from_ranks(rankdata(x, method="average")[None, :], y[None, :])[0])
        draws = []
        done = 0
        while done < N_BOOT:
            b = min(BOOT_CHUNK, N_BOOT - done)
            idx = rng.integers(0, n, size=(b, n))
            ranks = rankdata(x[idx], method="average", axis=1)
            draws.append(_auc_from_ranks(ranks, y[idx]))
            done += b
        valid = np.concatenate(draws)
        valid = valid[np.isfinite(valid)]  # single-class resamples yield NaN draws
        out[tname] = {
            "auc": point,
            "ci95": (
                [float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))]
                if valid.size > 0
                else None
            ),
            "ci_method": f"bootstrap percentile, {N_BOOT} draws, {int(valid.size)} valid",
            "n": int(n),
            "n_pos": n_pos,
            "degenerate": n_pos < DEGENERATE_POS_FLOOR,
        }
    return out


def phase_transport(args) -> int:
    """Leg 2: probe transport A->B (both directions) + same-corpus held-out AUC."""
    from issue2224_vendored_ridge import (
        dof_capped_ridge_fit_all,
        dof_capped_ridge_multi_y,
        ridge_predict,
    )

    trait_layers = resolve_trait_layers(ensure_screening_tables(Path(args.screening_dir)))
    results_dir = Path(args.results_dir)
    smoke = args.smoke_dim is not None
    if smoke and args.limit_shards is None:
        raise RuntimeError("--smoke-dim is a smoke-only dial: it requires --limit-shards")
    corpora = [c.strip() for c in args.corpora.split(",") if c.strip()]

    # Judged labels + features per (corpus, trait).
    cells: dict[tuple[str, str], dict] = {}
    layers = sorted(set(trait_layers.values()))
    for corpus in corpora:
        ids, slices, manifest = load_slices(args, corpus, layers, kinds=("resp_avg_dataset",))
        row_of = {sid: i for i, sid in enumerate(ids)}
        for trait in TRAITS:
            lp = Path(args.selections_dir) / corpus / trait / "filter_scores.json"
            labels_doc = json.loads(lp.read_text())
            judged = {
                k: float(v)
                for k, v in labels_doc["scores"].items()
                if v is not None and k in row_of
            }
            jids = sorted(judged)
            if len(jids) < 10:
                if smoke:
                    logger.warning(
                        "[transport] %s/%s: only %d judged rows in the smoke slice — cell "
                        "skipped (smoke-only outcome)",
                        corpus,
                        trait,
                        len(jids),
                    )
                    continue
                raise RuntimeError(f"{corpus}/{trait}: only {len(jids)} judged rows")
            rows = np.array([row_of[k] for k in jids])
            layer = trait_layers[trait]
            cells[(corpus, trait)] = {
                "ids": jids,
                "x": slices[("resp_avg_dataset", layer)][rows].astype(np.float64),
                "y": np.array([judged[k] for k in jids], dtype=np.float64),
                "n_labels_null": sum(1 for v in labels_doc["scores"].values() if v is None),
                "label_sha256": sha256_file(lp),
            }

    out: dict = {
        "meta": {
            **repro_meta("issue2224_followup_r1.transport"),
            "probe_convention": (
                "Form-A ridge (issue2224_predictor_scores.probe_score shape): answer-space "
                "resp_avg_dataset at the trait read-out layer -> graded judge label; "
                "dof-capped GCV ridge (vendored #2222 cores), NOT logistic"
            ),
            "subset_note": (
                "judged subset only (union of the selection arms' top-k candidates per "
                "cell — selection-conditioned, not a random pool draw; matches the "
                "free-analysis AUC read)"
            ),
            "trait_layers": trait_layers,
            "smoke": smoke,
        },
        "cells": {},
    }
    for trait in TRAITS:
        for corpus in corpora:
            if (corpus, trait) not in cells:
                continue
            cell = cells[(corpus, trait)]
            x, y, n = cell["x"], cell["y"], len(cell["ids"])
            fold_ids = _fold_ids(n, N_FOLDS, f"transport-{corpus}-{trait}")
            min_ntr = int(min((fold_ids != f).sum() for f in np.unique(fold_ids)))
            if smoke:
                d_use = min(args.smoke_dim, max(4, min_ntr - 4))
                col_idx = np.random.default_rng(
                    stable_seed("fu-r1-smokedim-t", corpus, trait)
                ).choice(x.shape[1], d_use, replace=False)
                x = x[:, col_idx]
            d_use = x.shape[1]
            if min_ntr <= d_use:
                raise RuntimeError(
                    f"{corpus}/{trait}: n_train_min={min_ntr} <= d={d_use} — refuse "
                    f"(estimator-degenerate regime, #1701)"
                )
            fit = dof_capped_ridge_multi_y(
                x.astype(np.float32),
                y[:, None],
                fold_ids,
                lambdas=LAMBDAS,
                dof_cap=DOF_CAP,
                device=args.device,
            )
            same = _auc_with_ci(fit["heldout_pred"][:, 0], y, f"{corpus}-{trait}-same")
            cell_out = {
                "n_judged": n,
                "n_labels_null_dropped": cell["n_labels_null"],
                "d": d_use,
                "n_train_min_across_folds": min_ntr,
                "n_train_vs_d": f"n_train_min={min_ntr} > d={d_use}",
                "heldout_label_r2": float(fit["heldout_r2"][0]),
                "same_corpus_heldout_auc": same,
            }
            # Cross-corpus: train on THIS corpus (all judged rows), score the other.
            for other in corpora:
                if other == corpus or (other, trait) not in cells:
                    continue
                oc = cells[(other, trait)]
                xo = oc["x"]
                if smoke:
                    xo = xo[:, col_idx]
                full = dof_capped_ridge_fit_all(
                    x.astype(np.float64),
                    y[:, None],
                    lambdas=LAMBDAS,
                    dof_cap=DOF_CAP,
                    device=args.device,
                )
                pred_o = ridge_predict(full, xo)[:, 0]
                cell_out[f"transport_to_{other}"] = {
                    "n_train": n,
                    "n_eval": len(oc["ids"]),
                    "auc": _auc_with_ci(pred_o, oc["y"], f"{corpus}->{other}-{trait}"),
                    "selected_lambda_median": float(np.median(full["lam"])),
                }
            out["cells"][f"{corpus}/{trait}"] = cell_out
            auc_ge1 = same["trait_bearing_ge1"]["auc"]
            print(
                f"[transport] unit {corpus}/{trait} n={n} d={d_use} "
                f"same_auc_ge1={'NA' if auc_ge1 is None else f'{auc_ge1:.4f}'}",
                flush=True,
            )
    # Reference row: the banked probe_diff_context AUC (free-analysis committed).
    banked = Path(args.free_analysis_dir) / "auc_by_arm.json"
    if banked.exists():
        doc = json.loads(banked.read_text())
        out["banked_probe_diff_reference"] = {
            key: {
                arm: doc["results"][key]["arms"][arm]
                for arm in ("probe_diff_context", "probe_diff_prefix")
            }
            for key in doc["results"]
        }
    results_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, results_dir / "transport.json")
    print(f"[transport] wrote {results_dir / 'transport.json'}", flush=True)
    return 0


# ── Phase: rejudge-triage (leg 3a) ───────────────────────────────────────────────


def subfloor_cells(trait_scores_dir: Path) -> list[str]:
    """Cell ids whose trait-score completeness sits below the 0.95 floor."""
    out = []
    for d in sorted(Path(trait_scores_dir).iterdir()):
        p = d / "trait_scores.json"
        if not p.exists():
            continue
        te = json.loads(p.read_text())["trait_expression"]
        if te["n_scored_items"] / te["n_items"] < COMPLETENESS_FLOOR:
            out.append(d.name)
    return out


def _cell_raw_path(args, cid: str) -> Path:
    """Round-0 trait save_raw for one cell: local-first, packed-HF fallback."""
    local = Path(args.judge_root) / "raw" / f"trait_{cid}.json"
    if local.exists():
        return local
    # Fallback: unpack from the packed tree (upload-policy pack contract:
    # one line per SOURCE file, {"src": <rel path>, "doc": <original JSON>}).
    unpack_dir = Path(args.out_root) / "rejudge" / "unpacked_raw"
    target = unpack_dir / f"trait_{cid}.json"
    if target.exists():
        return target
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import (
        list_hf_files_under_path,
        stage_hub_file,
    )

    unpack_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(args.out_root) / "rejudge" / "packed_dl"
    shard_dir.mkdir(parents=True, exist_ok=True)
    remote = list_hf_files_under_path(HfApi(), DATA_REPO, PACKED_PREFIX, repo_type="dataset")
    for rp in sorted(p for p in remote if p.rsplit("/", 1)[1].startswith("raw.shard")):
        local_shard = shard_dir / rp.rsplit("/", 1)[1]
        if not local_shard.exists():
            stage_hub_file(DATA_REPO, rp, local_shard)
        with open(local_shard) as f:
            for line in f:  # text-mode iteration (never splitlines — #950)
                row = json.loads(line)
                if row["src"] == f"raw/trait_{cid}.json":
                    atomic_write_json(row["doc"], target)
                    return target
    raise RuntimeError(f"{cid}: trait save_raw not found locally nor in {PACKED_PREFIX}")


def classify_cell_draws(save_raw: Path, item_ids: list[str]) -> dict:
    """Per-item per-draw drop-class census for one cell.

    Precedence mirrors ``graded_judge.judge_result_from_save_raw`` /
    ``issue2224_select.malformed_drop_counts``: kept -> transport ->
    api-refusal -> refusal -> truncation -> malformed.
    """
    from explore_persona_space.eval import batch_judge as _bj
    from explore_persona_space.eval import graded_judge as _gj

    with open(save_raw) as f:
        all_scores: dict[str, object] = json.load(f).get("all_scores", {})
    classes = ("kept", "transport", "api_refusal", "refusal", "truncation", "malformed")
    per_item = {iid: dict.fromkeys(classes, 0) for iid in item_ids}
    for cid, parsed in all_scores.items():
        item_id = cid.rsplit("__", 2)[0]
        if item_id not in per_item:
            continue
        s = _gj._score_from_parsed(parsed)
        if s is not None:
            cls = "kept"
        elif _bj.is_transport_error_dict(parsed):
            cls = "transport"
        elif _bj.is_api_refusal_error_dict(parsed):
            cls = "api_refusal"
        elif _gj._is_refusal_parsed(parsed):
            cls = "refusal"
        else:
            stop_reason = parsed.get("stop_reason") if isinstance(parsed, dict) else None
            cls = "truncation" if _bj.is_truncation_stop_reason(stop_reason) else "malformed"
        per_item[item_id][cls] += 1
    totals = {c: sum(v[c] for v in per_item.values()) for c in classes}
    return {"per_item": per_item, "totals": totals}


def phase_rejudge_triage(args) -> int:
    """Leg 3a: classify every dropped draw in the sub-floor cells; write the
    recovery plan (counts + item ids only — never text)."""
    cids = (
        [c.strip() for c in args.cells.split(",") if c.strip()]
        if args.cells
        else subfloor_cells(Path(args.trait_scores_dir))
    )
    out: dict = {
        "meta": {
            **repro_meta("issue2224_followup_r1.rejudge_triage"),
            "completeness_floor": COMPLETENESS_FLOOR,
            "class_precedence": "kept > transport > api_refusal > refusal > truncation > malformed",
            "recoverable_classes": (
                "transport + api_refusal re-issued per lost draw (rules 24(ii)/28: "
                "instrument losses, never content drops); stochastic-malformed unscored "
                "items get the bounded same-instrument re-draw "
                "(issue2224_select.judge_with_redraw convention, <=1+"
                f"{REJUDGE_MAX_REDRAW_ROUNDS} attempts/item); instructed judge-REFUSAL is "
                "a produced verdict and stays dropped (rule 9)"
            ),
        },
        "cells": {},
    }
    total_r1 = 0
    total_r2_items = 0
    for cid in cids:
        ts = json.loads((Path(args.trait_scores_dir) / cid / "trait_scores.json").read_text())
        te = ts["trait_expression"]
        item_ids = list(te["per_item_scores"].keys())
        raw_path = _cell_raw_path(args, cid)
        census = classify_cell_draws(raw_path, item_ids)
        per_item = census["per_item"]
        r1_plan = {
            iid: v["transport"] + v["api_refusal"]
            for iid, v in per_item.items()
            if v["transport"] + v["api_refusal"] > 0
        }
        unscored = [iid for iid, v in per_item.items() if v["kept"] == 0]
        r2_items = [iid for iid in unscored if per_item[iid]["malformed"] > 0]
        unrecoverable = [
            iid for iid in unscored if per_item[iid]["malformed"] == 0 and iid not in r1_plan
        ]
        total_r1 += sum(r1_plan.values())
        total_r2_items += len(r2_items)
        out["cells"][cid] = {
            "n_items": te["n_items"],
            "n_scored_items": te["n_scored_items"],
            "completeness": round(te["n_scored_items"] / te["n_items"], 4),
            "judge": ts["judge"],
            "raw_sha256": sha256_file(raw_path),
            "class_totals": census["totals"],
            "n_unscored_items": len(unscored),
            "r1_reissue_draws": sum(r1_plan.values()),
            "r1_items": r1_plan,
            "r2_redraw_items": r2_items,
            "unrecoverable_items": unrecoverable,
            "unrecoverable_note": (
                "all drops on these items are instructed judge-REFUSAL/truncation "
                "verdicts — deterministic content class, stays dropped"
            ),
        }
        print(
            f"[rejudge-triage] {cid} totals={census['totals']} r1={sum(r1_plan.values())} "
            f"r2_items={len(r2_items)} unrecoverable={len(unrecoverable)}",
            flush=True,
        )
    out["meta"]["projected_calls"] = {
        "r1_reissue_draws": total_r1,
        "r2_items_round1": total_r2_items,
        "r2_max_total": total_r2_items * (1 + REJUDGE_MAX_REDRAW_ROUNDS),
        "routing": (
            "sync (forced via threshold_base=REDRAW_SYNC_THRESHOLD) — total well under "
            "the ~5k Batch/pilot-gate band (llm-judging rules 23/26)"
        ),
    }
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, results_dir / "rejudge_triage.json")
    print(f"[rejudge-triage] wrote {results_dir / 'rejudge_triage.json'}", flush=True)
    return 0


# ── Phase: rejudge-run (leg 3b) ──────────────────────────────────────────────────


def _stage_local_or_hf(local: Path, hf_path: str) -> Path:
    if local.exists():
        return local
    from explore_persona_space.orchestrate.hub import stage_hub_file

    return stage_hub_file(DATA_REPO, hf_path, local)


def cell_items(args, cid: str) -> dict[str, tuple[str, str]]:
    """{item_id: (question, answer)} for one cell (mirrors
    ``issue2224_finetune_sweep.judge_items_for_cell``: item_id = f"{qid}-g{draw}")."""
    corpus, trait = cid.split("__")[0], cid.split("__")[1]
    qpath = _stage_local_or_hf(
        Path(args.eval_questions_dir) / f"{corpus}__{trait}.jsonl",
        f"{EVAL_Q_PREFIX}/{corpus}__{trait}.jsonl",
    )
    gpath = _stage_local_or_hf(
        Path(args.gen_root) / cid / "generations.jsonl",
        f"{POSTFT_PREFIX}/{cid}/generations.jsonl",
    )
    qs = {str(q["qid"]): str(q["question"]) for q in load_jsonl(qpath)}
    items: dict[str, tuple[str, str]] = {}
    for r in load_jsonl(gpath):
        qid = str(r["qid"])
        if qid not in qs:
            raise RuntimeError(f"{cid}: generation row qid {qid!r} not in questions")
        items[f"{qid}-g{int(r['draw'])}"] = (qs[qid], str(r["response"]))
    return items


def load_trait_rubric_checked(args, cid: str, recorded_sha: str) -> str:
    """Trait rubric from the persona_vectors clone, sha-asserted against the
    cell's recorded ``trait_rubric_sha256`` (instrument identity, fail loud)."""
    import hashlib

    from issue778_lib import load_trait_data

    if args.pv_root is None:
        raise RuntimeError("--pv-root required (persona_vectors clone for the trait rubrics)")
    trait = cid.split("__")[1]
    text = load_trait_data(Path(args.pv_root), trait).eval_prompt
    got = hashlib.sha256(text.encode()).hexdigest()
    if got != recorded_sha:
        raise RuntimeError(
            f"{cid}: rubric sha mismatch — pv clone {got[:16]} != recorded "
            f"{recorded_sha[:16]} (NOT the parent instrument; refuse to re-judge)"
        )
    return text


def _rejudge_one_cell(args, cid: str, plan: dict) -> dict:
    """R1 re-issue (transport + api-refusal draws) + R2 bounded re-draw for one
    cell. Returns the per-cell recovery record (parsed scores only, no text)."""
    from issue2224_select import REDRAW_SYNC_THRESHOLD, malformed_drop_counts
    from explore_persona_space.eval.graded_judge import DEFAULT_JUDGE_MODEL, judge_graded

    ts = json.loads((Path(args.trait_scores_dir) / cid / "trait_scores.json").read_text())
    rubric = load_trait_rubric_checked(args, cid, ts["judge"]["trait_rubric_sha256"])
    max_tokens = int(ts["judge"]["max_tokens"])  # parent instrument (1024)
    qa = cell_items(args, cid)
    out_root = Path(args.out_root) / "rejudge"
    rec: dict = {
        "cell_id": cid,
        # Provenance (code-review r1 item 3): judge_graded defaults to
        # DEFAULT_JUDGE_MODEL; record the realized id in the durable record.
        "judge_model": DEFAULT_JUDGE_MODEL,
        "max_tokens": max_tokens,
        "r1": {},
        "r2": {},
        "recovered_scores": {},
    }

    # R1 — re-issue instrument-lost draws, grouped by per-item lost count.
    r1_items = dict(plan["r1_items"])
    if args.rejudge_limit_items is not None:
        r1_items = dict(sorted(r1_items.items())[: args.rejudge_limit_items])
    by_k: dict[int, list[str]] = {}
    for iid, k in r1_items.items():
        by_k.setdefault(int(k), []).append(iid)
    for k, iids in sorted(by_k.items()):
        triples = [(iid, *qa[iid]) for iid in sorted(iids)]
        res = judge_graded(
            triples,
            rubric,
            n_draws=k,
            cache_dir=out_root / "cache" / cid / f"r1_k{k}",
            save_raw=out_root / "raw" / f"r1_{cid}__k{k}.json",
            max_tokens=max_tokens,
            threshold_base=REDRAW_SYNC_THRESHOLD,  # force the sync route
        )
        for iid, draws in res.per_item_scores.items():
            if draws:
                rec["recovered_scores"].setdefault(iid, []).extend(float(s) for s in draws)
        rec["r1"][f"k{k}"] = {
            "n_items": len(iids),
            "n_total_draws": res.n_total_draws,
            "n_dropped_draws": res.n_dropped_draws,
            "n_refusal_draws": res.n_refusal_draws,
            "n_transport_lost_draws": res.n_transport_lost_draws,
            "n_api_refusal_draws": res.n_api_refusal_draws,
        }

    # R2 — bounded same-instrument re-draw for stochastic-malformed unscored
    # items (first parsed draw restores the item; still-malformed continue).
    pending = [iid for iid in plan["r2_redraw_items"] if iid not in rec["recovered_scores"]]
    if args.rejudge_limit_items is not None:
        pending = sorted(pending)[: args.rejudge_limit_items]
    for rnd in range(1, REJUDGE_MAX_REDRAW_ROUNDS + 2):  # 1 + REDRAW rounds attempts
        if not pending:
            break
        triples = [(iid, *qa[iid]) for iid in sorted(pending)]
        save_raw = out_root / "raw" / f"r2_{cid}__round{rnd}.json"
        res = judge_graded(
            triples,
            rubric,
            n_draws=1,
            cache_dir=out_root / "cache" / cid / f"r2_round{rnd}",
            save_raw=save_raw,
            max_tokens=max_tokens,
            threshold_base=REDRAW_SYNC_THRESHOLD,
        )
        n_rec = 0
        for iid in list(pending):
            s = res.scores.get(iid)
            if s is not None:
                rec["recovered_scores"].setdefault(iid, []).append(float(s))
                n_rec += 1
        malformed_r = malformed_drop_counts(save_raw, set(pending))
        pending = sorted(
            iid
            for iid in pending
            if iid not in rec["recovered_scores"] and malformed_r.get(iid, 0) > 0
        )
        rec["r2"][f"round{rnd}"] = {
            "n_items": len(triples),
            "n_recovered": n_rec,
            "n_still_malformed": len(pending),
            "n_refusal_draws": res.n_refusal_draws,
            "n_transport_lost_draws": res.n_transport_lost_draws,
        }
    rec["r2_residual_items"] = pending
    return rec


def phase_rejudge_run(args) -> int:
    """Leg 3b: surgical re-judge of the triage plan's recoverable classes."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    triage_path = Path(args.results_dir) / "rejudge_triage.json"
    if not triage_path.exists():
        raise RuntimeError(f"{triage_path} missing — run --phase rejudge-triage first")
    triage = json.loads(triage_path.read_text())
    triage_sha = sha256_file(triage_path)
    cids = sorted(triage["cells"])
    if args.rejudge_limit_cells is not None:
        cids = cids[: args.rejudge_limit_cells]
    cells_dir = Path(args.out_root) / "rejudge" / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    regime = {
        "triage_sha256": triage_sha,
        "rejudge_limit_items": args.rejudge_limit_items,
        "max_redraw_rounds": REJUDGE_MAX_REDRAW_ROUNDS,
    }
    pending = []
    for cid in cids:
        done = cells_dir / f"{cid}.json"
        if done.exists() and json.loads(done.read_text()).get("regime") == regime:
            print(f"[rejudge-run] {cid} resume-skip", flush=True)
            continue
        pending.append(cid)
    failures: list[tuple[str, str]] = []
    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.rejudge_concurrency)) as ex:
        futs = {
            ex.submit(_rejudge_one_cell, args, cid, triage["cells"][cid]): cid for cid in pending
        }
        for fut in as_completed(futs):
            cid = futs[fut]
            n_done += 1
            try:
                rec = fut.result()
            except Exception as e:  # per-cell checkpointing; collected + re-raised loud
                failures.append((cid, f"{type(e).__name__}: {e}"))
                logger.error("[rejudge-run] FAILED %s: %s", cid, e)
                continue
            rec["regime"] = regime
            rec["meta"] = repro_meta("issue2224_followup_r1.rejudge_run")
            atomic_write_json(rec, cells_dir / f"{cid}.json")
            print(
                f"[rejudge-run] unit {n_done}/{len(pending)} {cid} "
                f"recovered_items={len(rec['recovered_scores'])} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    if failures:
        raise RuntimeError(
            f"[rejudge-run] {len(failures)} cell(s) failed: {sorted(c for c, _ in failures)} "
            f"— completed cells are checkpointed; first error: {failures[0][1]}"
        )
    return 0


# ── Phase: rejudge-aggregate (leg 3c) ────────────────────────────────────────────


def phase_rejudge_aggregate(args) -> int:
    """Leg 3c: merged per-item pools -> updated cell means + completeness table
    + headline-contrast point movement."""
    from explore_persona_space.eval.graded_judge import judge_result_from_save_raw

    triage = json.loads((Path(args.results_dir) / "rejudge_triage.json").read_text())
    cells_dir = Path(args.out_root) / "rejudge" / "cells"
    updated: dict = {}
    for cid in sorted(triage["cells"]):
        rec_path = cells_dir / f"{cid}.json"
        if not rec_path.exists():
            logger.warning("[rejudge-aggregate] %s: no recovery record — skipped", cid)
            continue
        rec = json.loads(rec_path.read_text())
        ts_path = Path(args.trait_scores_dir) / cid / "trait_scores.json"
        ts = json.loads(ts_path.read_text())
        te = ts["trait_expression"]
        item_ids = list(te["per_item_scores"].keys())
        raw_path = _cell_raw_path(args, cid)
        # Round-0 per-item kept-draw pools via the production reduce.
        res0 = judge_result_from_save_raw(raw_path, [(iid, "", "") for iid in item_ids])
        pools = {iid: list(res0.per_item_scores.get(iid, [])) for iid in item_ids}
        for iid, extra in rec["recovered_scores"].items():
            pools[iid].extend(float(s) for s in extra)
        kept = {iid: p for iid, p in pools.items() if p}
        new_scores = {iid: sum(p) / len(p) for iid, p in kept.items()}
        new_mean = sum(new_scores.values()) / len(new_scores) if new_scores else None
        new_rate = (
            sum(v > 50 for v in new_scores.values()) / len(new_scores) if new_scores else None
        )
        updated[cid] = {
            "before": {
                "graded_mean": te["graded_mean"],
                "rate_gt50": te["rate_gt50"],
                "n_scored_items": te["n_scored_items"],
                "completeness": round(te["n_scored_items"] / te["n_items"], 4),
            },
            "after": {
                "graded_mean": round(float(new_mean), 4) if new_mean is not None else None,
                "rate_gt50": round(float(new_rate), 4) if new_rate is not None else None,
                "n_scored_items": len(new_scores),
                "completeness": round(len(new_scores) / te["n_items"], 4),
            },
            "n_items": te["n_items"],
            "n_items_recovered": len(
                [iid for iid in rec["recovered_scores"] if te["per_item_scores"][iid] is None]
            ),
            "per_item_scores_after": {
                iid: round(float(v), 4) for iid, v in sorted(new_scores.items())
            },
            "still_above_floor": len(new_scores) / te["n_items"] >= COMPLETENESS_FLOOR,
        }
        print(
            f"[rejudge-aggregate] {cid} completeness "
            f"{updated[cid]['before']['completeness']} -> "
            f"{updated[cid]['after']['completeness']} mean "
            f"{updated[cid]['before']['graded_mean']} -> "
            f"{updated[cid]['after']['graded_mean']}",
            flush=True,
        )

    # Headline-contrast point movement (paired on (question, draw) slots scored
    # non-None in BOTH cells — the analysis_4b pairing convention). Bootstrap
    # CIs are NOT recomputed here; movement is read against the ORIGINAL CI.
    contrast_moves = []
    for cf in sorted(Path(args.analysis_4b_dir).glob("contrasts_*.json")):
        doc = json.loads(cf.read_text())
        for con in doc.get("contrasts", []):
            ca, cb = con["cell_a"], con["cell_b"]
            if ca not in updated and cb not in updated:
                continue

            def _scores_for(cid: str) -> dict[str, float]:
                if cid in updated:
                    return {k: float(v) for k, v in updated[cid]["per_item_scores_after"].items()}
                ts = json.loads(
                    (Path(args.trait_scores_dir) / cid / "trait_scores.json").read_text()
                )
                return {
                    k: float(v)
                    for k, v in ts["trait_expression"]["per_item_scores"].items()
                    if v is not None
                }

            sa, sb = _scores_for(ca), _scores_for(cb)
            paired = sorted(set(sa) & set(sb))
            if not paired:
                continue
            new_delta = float(np.mean([sa[k] - sb[k] for k in paired]))
            old = con["response_level"]
            contrast_moves.append(
                {
                    "file": cf.name,
                    "contrast": con.get("contrast"),
                    "cell_a": ca,
                    "cell_b": cb,
                    "old_mean": old["mean"],
                    "old_ci": [old["ci_lo"], old["ci_hi"]],
                    "old_n_paired": con["n_paired"],
                    "new_mean": round(new_delta, 4),
                    "new_n_paired": len(paired),
                    "moved_outside_old_ci": not (old["ci_lo"] <= new_delta <= old["ci_hi"]),
                    "sign_flip": (new_delta > 0) != (old["mean"] > 0),
                }
            )
    out = {
        "meta": {
            **repro_meta("issue2224_followup_r1.rejudge_aggregate"),
            "merge_rule": (
                "item score = mean over round-0 kept draws + recovered parsed draws "
                "(drop-never-coerce unchanged); completeness = items with >=1 kept draw"
            ),
            "contrast_note": (
                "point movement only — bootstrap CIs are the ORIGINAL analysis_4b "
                "intervals; moved_outside_old_ci flags a point that left them"
            ),
        },
        "cells": {
            cid: {k: v for k, v in rec.items() if k != "per_item_scores_after"}
            for cid, rec in updated.items()
        },
        "contrast_moves": contrast_moves,
        "n_cells_restored_above_floor": sum(1 for r in updated.values() if r["still_above_floor"]),
    }
    results_dir = Path(args.results_dir)
    atomic_write_json(out, results_dir / "rejudge_updated_cells.json")
    print(f"[rejudge-aggregate] wrote {results_dir / 'rejudge_updated_cells.json'}", flush=True)
    return 0


# ── Phase: aggregate + upload ────────────────────────────────────────────────────


def phase_aggregate(args) -> int:
    """Join legs 1+2 (+3 when present) into one summary JSON."""
    results_dir = Path(args.results_dir)
    corpora = [c.strip() for c in args.corpora.split(",") if c.strip()]
    summary: dict = {"meta": repro_meta("issue2224_followup_r1.aggregate"), "legs": {}}
    refit = {}
    for corpus in corpora:
        p = results_dir / f"refit_{corpus}.json"
        if not p.exists():
            raise RuntimeError(f"{p} missing — run --phase refit")
        doc = json.loads(p.read_text())
        refit[corpus] = {
            "n": doc["meta"]["n_samples"],
            "n_train_vs_d": doc["meta"]["n_train_vs_d"],
            "arms": {
                arm: {
                    layer: {
                        "heldout_r2_mean": lo["heldout_r2"]["mean_over_dims"],
                        "identity_r2_mean": lo["identity_bias_heldout_r2"]["mean_over_dims"],
                        "traits": {
                            t: {
                                "refit_score_r_vs_exact": tv["score_level_calibration_vs_exact"][
                                    "pearson_r"
                                ],
                                "frozen_score_r_vs_exact": tv["frozen_map_score_r_vs_exact"],
                                "standin_r": tv["standin_level_calibration"]["pearson_r"],
                                "jaccard_top_refit": tv["jaccard_top_refit_vs_exact"],
                                "jaccard_top_frozen": tv["jaccard_top_frozen_vs_exact"],
                            }
                            for t, tv in lo["traits"].items()
                        },
                    }
                    for layer, lo in doc["arms"][arm]["layers"].items()
                }
                for arm in doc["arms"]
            },
        }
    summary["legs"]["refit"] = refit
    tp = results_dir / "transport.json"
    if not tp.exists():
        raise RuntimeError(f"{tp} missing — run --phase transport")
    summary["legs"]["transport"] = {
        key: {
            "same_auc_ge1": cell["same_corpus_heldout_auc"]["trait_bearing_ge1"]["auc"],
            **{
                k: v["auc"]["trait_bearing_ge1"]["auc"]
                for k, v in cell.items()
                if k.startswith("transport_to_")
            },
        }
        for key, cell in json.loads(tp.read_text())["cells"].items()
    }
    for name in ("rejudge_triage.json", "rejudge_updated_cells.json"):
        p = results_dir / name
        if p.exists():
            summary["legs"][name.removesuffix(".json")] = {"present": True, "path": str(p)}
    atomic_write_json(summary, results_dir / "followup_r1_summary.json")
    print(f"[aggregate] wrote {results_dir / 'followup_r1_summary.json'}", flush=True)
    return 0


def _upload_leg(local_dir: Path, allow: list[str], path_in_repo: str) -> None:
    """ONE bulk fail-loud upload_folder commit + exact-set verify (the
    issue2224_suite_slice._upload_leg shape, copied verbatim)."""
    import fnmatch

    from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, _upload_folder_filtered

    rels = sorted(
        str(p.relative_to(local_dir))
        for p in local_dir.rglob("*")
        if p.is_file() and any(fnmatch.fnmatch(str(p.relative_to(local_dir)), a) for a in allow)
    )
    if not rels:
        raise RuntimeError(f"[upload] nothing matches {allow} under {local_dir}")
    expected = [f"{path_in_repo}/{rel}" for rel in rels]
    url = _upload_folder_filtered(
        local_dir=local_dir,
        repo_id=DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        allow_patterns=allow,
        expected_repo_paths=expected,
    )
    if not url:
        raise RuntimeError(
            f"[upload] bulk upload {local_dir} -> {path_in_repo} FAILED or verified incomplete"
        )
    logger.info("[upload] verified %d files at %s", len(expected), path_in_repo)


def phase_upload(args) -> int:
    """Upload the durable out-root legs to HF (staged slices are deliberately
    NOT uploaded: regenerable from the banked predictor_summaries via
    ``--phase stage`` — the recorded regen recipe)."""
    out_root = Path(args.out_root)
    prefix_root = args.prefix_root.rstrip("/")
    legs = {
        "refit": (out_root / "refit", ["*.npz", "*.json"], f"{prefix_root}/refit"),
        "rejudge_raw": (
            out_root / "rejudge" / "raw",
            ["*.json"],
            f"{prefix_root}/rejudge/raw",
        ),
        "rejudge_cells": (
            out_root / "rejudge" / "cells",
            ["*.json"],
            f"{prefix_root}/rejudge/cells",
        ),
    }
    requested = [x.strip() for x in args.legs.split(",") if x.strip()]
    unknown = sorted(set(requested) - set(legs))
    if unknown:
        raise RuntimeError(f"[upload] unknown legs {unknown}; valid: {sorted(legs)}")
    for leg in requested:
        local_dir, allow, dest = legs[leg]
        if not local_dir.is_dir():
            raise RuntimeError(f"[upload] leg {leg}: local dir missing: {local_dir}")
        _upload_leg(local_dir, allow, dest)
    print(f"[fu-r1:upload:complete] legs={','.join(requested)}", flush=True)
    return 0


# ── Entry point ──────────────────────────────────────────────────────────────────

PHASES = {
    "stage": phase_stage,
    "refit": phase_refit,
    "transport": phase_transport,
    "rejudge-triage": phase_rejudge_triage,
    "rejudge-run": phase_rejudge_run,
    "rejudge-aggregate": phase_rejudge_aggregate,
    "aggregate": phase_aggregate,
    "upload": phase_upload,
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue #2224 follow-up round 1 (E1 refit + probe transport + re-judge)."
    )
    parser.add_argument("--phase", choices=sorted(PHASES), default=None)
    parser.add_argument("--list-phases", action="store_true")
    parser.add_argument("--import-check", action="store_true")
    parser.add_argument("--corpora", default=",".join(CORPORA))
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR_DEFAULT)
    parser.add_argument("--screening-dir", type=Path, default=SCREENING_SCORES_DIR)
    parser.add_argument("--selections-dir", type=Path, default=SELECTIONS_DIR)
    parser.add_argument("--trait-scores-dir", type=Path, default=TRAIT_SCORES_DIR)
    parser.add_argument("--analysis-4b-dir", type=Path, default=ANALYSIS_4B_DIR)
    parser.add_argument(
        "--free-analysis-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_2224" / "free_analysis",
    )
    parser.add_argument("--rb-dir", type=Path, default=RB_LOCAL)
    parser.add_argument("--judge-root", type=Path, default=JUDGE_POSTFT_LOCAL)
    parser.add_argument("--eval-questions-dir", type=Path, default=EVAL_Q_LOCAL)
    parser.add_argument("--gen-root", type=Path, default=GEN_LOCAL_ROOT)
    parser.add_argument("--pv-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu", help="cuda on the pod lane")
    parser.add_argument("--knn-n", type=int, default=KNN_N)
    parser.add_argument("--cells", default=None, help="comma cell_id filter (triage)")
    parser.add_argument(
        "--limit-shards", type=int, default=None, help="smoke: cap staged shards per corpus"
    )
    parser.add_argument(
        "--smoke-dim",
        type=int,
        default=None,
        help="smoke-only feature-dim cut (requires --limit-shards; recorded in outputs)",
    )
    parser.add_argument("--rejudge-limit-cells", type=int, default=None, help="smoke cap")
    parser.add_argument("--rejudge-limit-items", type=int, default=None, help="smoke cap")
    parser.add_argument("--rejudge-concurrency", type=int, default=4)
    parser.add_argument("--legs", default="refit", help="upload legs (comma list)")
    parser.add_argument(
        "--prefix-root", default=HF_FU_PREFIX, help="HF prefix root (scratch-prefix smoke)"
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if args.list_phases:
        print(json.dumps(sorted(PHASES)))
        return 0
    if args.import_check:
        import importlib

        for mod in ("numpy", "torch", "scipy"):
            importlib.import_module(mod)
        from issue2224_free_analysis import (  # noqa: F401
            BOOT_CHUNK,
            DEGENERATE_POS_FLOOR,
            LABEL_THRESHOLDS,
            N_BOOT,
            _auc_from_ranks,
        )
        from issue2224_predictor_scores import load_rb  # noqa: F401
        from issue2224_select import (  # noqa: F401
            REDRAW_SYNC_THRESHOLD,
            malformed_drop_counts,
            ranked_ids,
        )
        from issue2224_vendored_ridge import (  # noqa: F401
            dof_capped_ridge_fit_all,
            dof_capped_ridge_multi_y,
            ridge_predict,
        )
        from issue778_lib import load_trait_data  # noqa: F401
        from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
            identity_bias_predict,
            knn_retrieval,
        )
        from explore_persona_space.eval.batch_judge import (  # noqa: F401
            is_api_refusal_error_dict,
            is_transport_error_dict,
            is_truncation_stop_reason,
        )
        from explore_persona_space.eval.graded_judge import (  # noqa: F401
            judge_graded,
            judge_result_from_save_raw,
        )
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined
        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            DEFAULT_DATASET_REPO,
            _upload_folder_filtered,
            list_hf_files_under_path,
            stage_hub_file,
        )

        assert_args_attributes_defined(__file__)
        print("[import-check] OK issue2224_followup_r1")
        return 0
    if args.phase is None:
        raise SystemExit("--phase required (see --list-phases)")
    # Smoke-overwrite guard (code-review r1 item 2): a smoke-dialed run must
    # never rewrite the committed production JSONs under the DEFAULT
    # results-dir (phase_transport always rewrites transport.json). An
    # explicit non-default --results-dir (a scratch dir) is respected.
    if (args.smoke_dim is not None or args.limit_shards is not None) and Path(
        args.results_dir
    ).resolve() == RESULTS_DIR_DEFAULT.resolve():
        args.results_dir = Path(args.out_root) / "smoke_results"
        print(
            f"[smoke-guard] smoke dials set — results-dir rebound to {args.results_dir}",
            flush=True,
        )
    return PHASES[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
