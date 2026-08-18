#!/usr/bin/env python
"""Issue #2356 fits driver — P4 groups/splits (VM) + P6 maps + P7 probes/battery/transfer/stats.

Phases (``--phase groups|maps|probes|battery|transfer|stats``):

- ``groups``   P4 (VM CPU): judge labels -> per-arm grouping (Arm A: base_id;
               Arm B: TF-IDF (word 1-2 ∪ char 3-5) cosine >= 0.55 ∪ v_C-cosine
               >= 0.92 @ hs layer 14, connected components + degeneracy guard)
               -> balanced sets -> PRE-LABEL all-group 5-fold assignment (M1)
               -> ``splits.json`` + judge-interface files under ``splits/``.
- ``maps``     P6 (fits pod): primal ridge v_C -> v_A_greedy per hs layer 1..28
               (``ridge_fit_predict_primal_layer_batched``, f64, GCV
               logspace(-2,4,13), layer_chunk=4). Conditions: 3a generic-only
               (1 fit) + 3b generic + in-domain train-group rows per arm x fold
               (M1 fail-loud set-equality assert; the builder reads ONLY the
               corpus manifest + splits.json train_groups). SVD factor bundles
               (U/Vt fp32 + S f64 + xmu/xsd/ymu) persisted + uploaded per
               condition (M3); held-out generic R2 / identity+bias / kNN
               diagnostics -> results/map_diagnostics.json.
- ``probes``   P7: all-LINEAR predictor arms per arm (judge read #1, ctx dual
               ridge + DiM #2, mapped z_r probes #3a/#3b over the rank ladder,
               answer probes #4, matched-rank PCA control, F2 text-surface),
               nested inner-group-CV selection on train groups only, OOF
               scores, LODO (Arm B), limited-label ladder ->
               results/predictor_scores_arm{A,B}.json.
- ``battery``  P7: #2202 retrieval battery (whitened-cosine acc@1 PRIMARY, own
               shrunk-Cholesky whiten stats lam=0.1 from THIS run's train
               answers; raw-euclidean, r2_cand_norm, pearson; K=4 draw-averaged
               targets; behavior split + NN-behavior-match; S2 one-sided gate
               CI_lower > 1/n_pool) -> results/map_discrimination.json.
- ``transfer`` P7 (report-only): cross-regime A<->B transfer of the ctx ridge
               probe (+ DiM) -> results/transfer.json.
- ``stats``    P7: pooled OOF AUROC + balanced acc on ONE common per-arm row
               mask, paired group bootstrap (2000 draws, seed 1234), advisory
               group-label permutation (1000 draws, seed 5678, frozen-lambda
               dual-ridge hats with per-draw layer re-selection), H1 3-way
               lattice -> results/stats.json.

Checkpoint-per-phase with fingerprint-keyed done-sentinels (manifest shas +
git sha + output-affecting flags; bare file existence never skips). Per-unit
JSONL persistence + progress lines for >50-unit loops. ``--selftest`` runs all
six phases in-process on tiny synthetic data (no pod / GPU / network).

Argcheck note (accepted false negative, per code-style.md): ``args.x += 1``
AugAssign would enter the DEFINED set; this module does not use that form.

Content hygiene: this script PROCESSES Arm-A prompt text at runtime (splits
files for the judge) but never prints prompt text to stdout/logs; manifests
and result JSONs carry shas/row_ids only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402  (after load_dotenv: thread-cap discipline)

logger = logging.getLogger("issue2356_fits")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

ISSUE = 2356
SLUG = "refusalpred"
HF_PREFIX = f"issue{ISSUE}_{SLUG}"
HF_PREFIX_SMOKE = f"issue{ISSUE}_{SLUG}_smoke"

ARMS = ("armA", "armB")
CORPORA = ("armA", "armB", "generic")
GENERIC_KEYS = ("v_C", "v_A_greedy")
ARM_KEYS = ("v_C", "v_A_greedy", "v_A_rollout_mean")  # 2-D (L, d) consolidated keys
ARM_PRESENCE_KEYS = ("v_A_sample_k",)  # 3-D (K, L, d): presence-asserted, read lazily

# Predictor column names (P7). Judge (#1) is read from the judge script's
# predictor_scores.json; everything else is fitted here. All oriented P(REFUSE).
PRED_JUDGE = "judge_fewshot"
PRED_CTX = "ctx_ridge"
PRED_DIM = "ctx_dim"
PRED_3A = "map3a_zr"
PRED_3B = "map3b_zr"
PRED_PCA = "pca_ctx"
PRED_ANS = "ans_greedy"
PRED_ANS_RM = "ans_rollout"
PRED_TEXT = "text_surface"
PRED_TEXT_NOIND = "text_surface_noind"
PRED_ISREW = "is_rewrite"
PRED_LODO = "ctx_ridge_lodo"

HEADLINE_PREDS = (PRED_JUDGE, PRED_CTX, PRED_DIM, PRED_3A, PRED_3B, PRED_PCA, PRED_ANS, PRED_TEXT)


def _hf_prefix(args: argparse.Namespace) -> str:
    return HF_PREFIX_SMOKE if args.smoke else HF_PREFIX


def _eval_root(args: argparse.Namespace) -> Path:
    return Path(args.eval_root).resolve()


def _out_root(args: argparse.Namespace) -> Path:
    p = Path(args.out_root).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stores_dir(args: argparse.Namespace) -> Path:
    if args.stores_dir:
        return Path(args.stores_dir).resolve()
    return _out_root(args) / "summary_stores"


def _results_dir(args: argparse.Namespace) -> Path:
    p = _eval_root(args) / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Small utilities: atomic JSON, fingerprints, sentinels, progress lines
# ---------------------------------------------------------------------------


def _atomic_json(path: Path, obj: Any) -> None:
    """Write JSON atomically (tmp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _provenance() -> dict[str, Any]:
    """Reproducibility metadata (git sha + dirty flag + timestamps)."""
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    meta = dict(as_metadata_dict(git_provenance()))
    meta["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["numpy_version"] = np.__version__
    meta["issue"] = ISSUE
    return meta


def _phase_fingerprint(args: argparse.Namespace, phase: str, inputs: dict[str, Any]) -> str:
    """Fingerprint = output-affecting flags + input identities + code sha (c24)."""
    keyed = {
        "phase": phase,
        "inputs": inputs,
        "flags": {
            "smoke": bool(args.smoke),
            "seed": args.seed,
            "n_folds": args.n_folds,
            "inner_folds": args.inner_folds,
            "vc_layer": args.vc_layer,
            "tfidf_tau": args.tfidf_tau,
            "vc_tau": args.vc_tau,
            "label_hi": args.label_hi,
            "label_lo": args.label_lo,
            "min_valid": args.min_valid,
            "rank_ladder": args.rank_ladder,
            "ladder_sizes": args.ladder_sizes,
            "ladder_seeds": args.ladder_seeds,
            "n_boot": args.n_boot,
            "boot_seed": args.boot_seed,
            "n_perm": args.n_perm,
            "perm_seed": args.perm_seed,
            "gcv_dof_cap": args.gcv_dof_cap,
            "generic_heldout": args.generic_heldout,
            "k_draw_avg": args.k_draw_avg,
            "floor_a": args.floor_a,
            "floor_b": args.floor_b,
            "allow_degraded_folds": bool(args.allow_degraded_folds),
        },
    }
    try:
        from explore_persona_space.orchestrate.provenance import git_provenance

        keyed["git_sha"] = git_provenance().sha
    except Exception:  # git-less scratch tree (SLURM lane): degrade, never crash
        keyed["git_sha"] = "unavailable-no-git-checkout"
    return _sha256_obj(keyed)


def _sentinel_path(args: argparse.Namespace, phase: str) -> Path:
    d = _out_root(args) / "sentinels"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{phase}-done.json"


def _sentinel_ok(sent: Path, fingerprint: str, *, resume: bool) -> bool:
    """Skip only on an exact fingerprint match; bare existence never skips."""
    if not resume or not sent.exists():
        return False
    try:
        data = json.loads(sent.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("fingerprint") == fingerprint


def _write_sentinel(sent: Path, fingerprint: str, extra: dict[str, Any]) -> None:
    _atomic_json(sent, {"fingerprint": fingerprint, **extra, "meta": _provenance()})


def _progress(phase: str, k: int, n: int, key: str, t0: float) -> None:
    print(f"[{phase}] unit {k}/{n} {key} elapsed={time.time() - t0:.1f}s", flush=True)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Single-line O_APPEND write (per-unit persistence)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True))
        fh.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:  # text-mode iteration, never splitlines()
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Hub helpers (lazy imports; upload wiring per canonical paths)
# ---------------------------------------------------------------------------


def _hub():
    from explore_persona_space.orchestrate import hub

    return hub


def _upload_file(args: argparse.Namespace, local: Path, rel: str) -> None:
    """Upload one file to {prefix}/{rel} on the data repo (fail-loud)."""
    if args.no_upload:
        logger.info("[upload] skipped (--no-upload): %s", rel)
        return
    hub = _hub()
    hub._upload(
        local_path=local,
        repo_id=hub.DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=f"{_hf_prefix(args)}/{rel}/{local.name}",
        raise_on_error=True,
        upload_as_file=True,
    )
    logger.info("[upload] %s -> %s/%s/%s", local.name, _hf_prefix(args), rel, local.name)


def _upload_dir(args: argparse.Namespace, local_dir: Path, rel: str, expected: list[str]) -> None:
    """Upload a directory (one bulk commit) + verify an expected-paths subset."""
    if args.no_upload:
        logger.info("[upload] skipped (--no-upload): %s/", rel)
        return
    from huggingface_hub import HfApi

    hub = _hub()
    # HUB_DIR_FILECOUNT_EXEMPT: map bundles are <=29 files per condition dir.
    hub._upload(
        local_path=local_dir,
        repo_id=hub.DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=f"{_hf_prefix(args)}/{rel}",
        raise_on_error=True,
    )
    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        hub.DEFAULT_DATASET_REPO,
        expected,
        path_in_repo=f"{_hf_prefix(args)}/{rel}",
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(f"upload incomplete under {rel}: missing {missing}")
    logger.info("[upload] %s/ verified (%d expected files)", rel, len(expected))


def _stage_summary_stores(args: argparse.Namespace, shas: list[str]) -> Path:
    """Ensure per-sha store npzs exist locally; stage missing ones from the Hub.

    Files live at {prefix}/summary_stores/<sha>.npz (unit-1 upload layout).
    Resume-safe: existing files are skipped. Fail-loud on any missing sha
    after staging.
    """
    dest = _stores_dir(args)
    dest.mkdir(parents=True, exist_ok=True)
    missing = [s for s in shas if not (dest / f"{s}.npz").exists()]
    if missing and not args.stage_from_hub:
        raise RuntimeError(
            f"{len(missing)} store npz files missing under {dest} "
            f"(first: {missing[0]}); pass --stage-from-hub to fetch them"
        )
    if missing:
        from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

        need_gb = max(1.0, len(missing) * 3.5e-3)  # ~3.5 MB/file upper bound
        assert_out_root_headroom(dest, need_gb * 1.5, phase="stage_summary_stores")
        hub = _hub()
        t0 = time.time()
        if len(missing) > 500:
            # Bulk path: stage_hub_prefix mirrors at dest_dir/<repo-rel path>
            # (verbatim prefix mirror — gotchas.md), then files move into dest.
            mirror = _out_root(args) / "hf_mirror"
            hub.stage_hub_prefix(
                hub.DEFAULT_DATASET_REPO,
                f"{_hf_prefix(args)}/summary_stores",
                mirror,
                repo_type="dataset",
            )
            staged_root = mirror / _hf_prefix(args) / "summary_stores"
            assert staged_root.is_dir(), staged_root  # mirror-root arithmetic check
            for i, sha in enumerate(missing):
                src = staged_root / f"{sha}.npz"
                if src.exists():
                    os.replace(src, dest / f"{sha}.npz")
                if (i + 1) % 500 == 0 or i + 1 == len(missing):
                    _progress("stage_stores", i + 1, len(missing), sha[:12], t0)
        else:
            from huggingface_hub import hf_hub_download

            for i, sha in enumerate(missing):
                hub.retry_transient(
                    hf_hub_download,
                    repo_id=hub.DEFAULT_DATASET_REPO,
                    repo_type="dataset",
                    filename=f"{_hf_prefix(args)}/summary_stores/{sha}.npz",
                    local_dir=dest / "_hfstage",
                )
                src = dest / "_hfstage" / _hf_prefix(args) / "summary_stores" / f"{sha}.npz"
                os.replace(src, dest / f"{sha}.npz")
                if (i + 1) % 100 == 0 or i + 1 == len(missing):
                    _progress("stage_stores", i + 1, len(missing), sha[:12], t0)
    still = [s for s in shas if not (dest / f"{s}.npz").exists()]
    if still:
        raise RuntimeError(f"stores still missing after staging: {len(still)} (first {still[0]})")
    return dest


# ---------------------------------------------------------------------------
# Input loaders: manifests, judge labels, judge predictor scores
# ---------------------------------------------------------------------------


def load_manifest(args: argparse.Namespace, corpus: str) -> list[dict[str, Any]]:
    """Load a corpus manifest (unit-1 realized path). Rows carry prompt_sha (+
    base_id/axis for armA; source/category for armB). No prompt text."""
    p = _eval_root(args) / "corpus" / f"{corpus}_manifest.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = data["rows"]
    if not rows:
        raise RuntimeError(f"empty manifest {p}")
    return rows


def manifest_sha(args: argparse.Namespace, corpus: str) -> str:
    return _sha256_file(_eval_root(args) / "corpus" / f"{corpus}_manifest.json")


def load_corpus_text(args: argparse.Namespace, corpus: str) -> dict[str, str]:
    """prompt_sha -> prompt text, from the local text JSONL (runtime-only;
    never printed). Stages from the Hub when absent and --stage-from-hub."""
    p = _eval_root(args) / "corpus" / "text" / f"{corpus}.jsonl"
    if not p.exists():
        if not args.stage_from_hub:
            raise RuntimeError(f"corpus text missing: {p} (pass --stage-from-hub)")
        from huggingface_hub import hf_hub_download

        hub = _hub()
        got = hub.retry_transient(
            hf_hub_download,
            repo_id=hub.DEFAULT_DATASET_REPO,
            repo_type="dataset",
            filename=f"{_hf_prefix(args)}/corpus/{corpus}.jsonl",
            local_dir=p.parent / "_hfstage",
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        os.replace(got, p)
    out: dict[str, str] = {}
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["prompt_sha"]] = row["prompt"]
    if not out:
        raise RuntimeError(f"no rows parsed from {p}")
    return out


def load_judge_labels(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """item_id -> {score, label} from labeling/labels.json, with rejudge
    rescued_scores merged over missing/None scores (rule-28 sync re-issue)."""
    base = _eval_root(args) / "judge"
    labels_p = base / "labeling" / "labels.json"
    data = json.loads(labels_p.read_text(encoding="utf-8"))
    labels: dict[str, dict[str, Any]] = dict(data["labels"])
    rejudge_p = base / "rejudge" / "rejudge.json"
    n_rescued = 0
    if rejudge_p.exists():
        rescued = json.loads(rejudge_p.read_text(encoding="utf-8")).get("rescued_scores", {})
        for item_id, score in rescued.items():
            cur = labels.get(item_id)
            if cur is None or cur.get("score") is None:
                labels[item_id] = {"score": score, "label": _label_from_score(score)}
                n_rescued += 1
    logger.info("[labels] %d judged items (%d rejudge-rescued)", len(labels), n_rescued)
    return labels


def _label_from_score(score: Any) -> str:
    """Mirror of issue2356_judge.py semantics: None -> UNCLEAR; >=50 engage."""
    if score is None:
        return "UNCLEAR"
    return "engage" if float(score) >= 50 else "refuse"


def row_label_from_draws(
    labels: dict[str, dict[str, Any]],
    sha: str,
    *,
    k_samples: int,
    min_valid: int,
    hi: float,
    lo: float,
) -> dict[str, Any]:
    """Aggregate the N sampled-draw judge labels into the row-level label.

    positive class = engage (answer/comply); rate = engage fraction over valid
    draws; label engage iff rate >= hi, refuse iff rate <= lo, else None
    (middle dropped); rows with < min_valid valid draws are label=None with
    reason recorded. Greedy label carried separately (battery pool labels).
    """
    draw_labels = []
    for k in range(k_samples):
        item = labels.get(f"{sha}.s{k:02d}")
        if item is None:
            continue
        lab = item.get("label") or _label_from_score(item.get("score"))
        if lab in ("engage", "refuse"):
            draw_labels.append(lab)
    n_valid = len(draw_labels)
    rate = (sum(1 for x in draw_labels if x == "engage") / n_valid) if n_valid else None
    label: str | None = None
    reason = "ok"
    if n_valid < min_valid:
        reason = "below_min_valid"
    elif rate >= hi:
        label = "engage"
    elif rate <= lo:
        label = "refuse"
    else:
        reason = "middle_band"
    greedy_item = labels.get(f"{sha}.greedy")
    greedy_label = None
    if greedy_item is not None:
        gl = greedy_item.get("label") or _label_from_score(greedy_item.get("score"))
        if gl in ("engage", "refuse"):
            greedy_label = gl
    return {
        "rate": rate,
        "n_valid": n_valid,
        "label": label,
        "drop_reason": None if label else reason,
        "greedy_label": greedy_label,
    }


def load_predictor_scores(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """row_id -> {p_answer, p_refuse, arm, fold} from the judge predictor."""
    p = _eval_root(args) / "judge" / "predictor" / "predictor_scores.json"
    if not p.exists():
        logger.warning("[judge] predictor_scores.json absent at %s (judge #1 will be NaN)", p)
        return {}
    return json.loads(p.read_text(encoding="utf-8"))["scores"]


def load_splits(args: argparse.Namespace, arm: str) -> dict[str, Any]:
    p = _eval_root(args) / arm / "splits.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_arm_labels(args: argparse.Namespace, arm: str) -> dict[str, Any]:
    p = _eval_root(args) / arm / "labels.json"
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Phase groups (P4): labels -> grouping -> balanced sets -> pre-label folds
# ---------------------------------------------------------------------------


def _armb_vc_matrix(args: argparse.Namespace, shas: list[str]) -> np.ndarray:
    """(n, d) fp32 v_C at hs layer --vc-layer for the Arm-B pool, cached.

    Consumer-loader KEY ASSERT: every store npz must carry the ``v_C`` key
    with >= vc_layer+1 hidden states.
    """
    cache_dir = _out_root(args) / "consolidated"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"armB__v_C_l{args.vc_layer}.npy"
    sidecar = cache.with_suffix(".rows.json")
    if cache.exists() and sidecar.exists():
        rows = json.loads(sidecar.read_text(encoding="utf-8"))
        if rows == shas:
            return np.load(cache).astype(np.float32)
        logger.info("[groups] vc cache row-order mismatch -> rebuild")
    dest = _stage_summary_stores(args, shas)
    mats = []
    t0 = time.time()
    for i, sha in enumerate(shas):
        with np.load(dest / f"{sha}.npz") as data:
            assert "v_C" in data.files, (sha, sorted(data.files))
            v = data["v_C"]
            assert v.ndim == 2 and v.shape[0] > args.vc_layer, (sha, v.shape, args.vc_layer)
            mats.append(v[args.vc_layer].astype(np.float32))
        if (i + 1) % 250 == 0 or i + 1 == len(shas):
            _progress("groups_vc", i + 1, len(shas), sha[:12], t0)
    mat = np.stack(mats, axis=0)
    np.save(cache, mat)
    _atomic_json(sidecar, shas)
    return mat


def _cosine_matrix(mat: np.ndarray) -> np.ndarray:
    z = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return z @ z.T


def _tfidf_sims(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(word 1-2-gram, char 3-5-gram) TF-IDF cosine-similarity matrices."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    sims = []
    for kwargs in (
        {"analyzer": "word", "ngram_range": (1, 2)},
        {"analyzer": "char", "ngram_range": (3, 5)},
    ):
        vec = TfidfVectorizer(**kwargs)
        x = vec.fit_transform(texts)  # rows l2-normalized -> dot = cosine
        sims.append(np.asarray((x @ x.T).todense(), dtype=np.float32))
    return sims[0], sims[1]


def _components_at(
    s_word: np.ndarray, s_char: np.ndarray, s_vc: np.ndarray | None, tfidf_tau: float, vc_tau: float
) -> np.ndarray:
    """Connected-component labels for the union graph at the given thresholds."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    adj = (s_word >= tfidf_tau) | (s_char >= tfidf_tau)
    if s_vc is not None:
        adj = adj | (s_vc >= vc_tau)
    np.fill_diagonal(adj, False)
    _, labels = connected_components(csr_matrix(adj), directed=False)
    return labels


def _size_histogram(group_of: dict[str, str]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    counts: dict[str, int] = {}
    for gid in group_of.values():
        counts[gid] = counts.get(gid, 0) + 1
    for n in counts.values():
        sizes[str(n)] = sizes.get(str(n), 0) + 1
    return sizes


def _assign_folds(
    balanced_strata: dict[str, str],
    rest_strata: dict[str, str],
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    """Pre-label ALL-group fold assignment (M1).

    Balanced-set groups are dealt round-robin within stratum (class/dataset
    stratified) first; remaining groups are dealt round-robin within their
    source/axis stratum (NEVER by label). Every group gets a fold.
    """
    rng = np.random.default_rng(seed)
    fold_of: dict[str, int] = {}
    counter = 0
    for strata in (balanced_strata, rest_strata):
        by_stratum: dict[str, list[str]] = {}
        for gid, st in strata.items():
            if gid in fold_of:
                continue
            by_stratum.setdefault(st, []).append(gid)
        for st in sorted(by_stratum):
            gids = sorted(by_stratum[st])
            rng.shuffle(gids)
            for gid in gids:
                fold_of[gid] = counter % n_folds
                counter += 1
    return fold_of


def _balanced_arm_a(
    rows: list[dict[str, Any]],
    labels_by_sha: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[list[str], list[str]]:
    """Arm-A matched-pair balanced set. Returns (balanced_shas, flip_group_ids).

    Flip-group = base_id with >=1 engage(COMPLY)-labeled row. Within each:
    keep m = min(n_engage, n_refuse) rows per class; refuse picks base first,
    then rng-sampled refuse variants (seed 42). Exact 1:1 within groups.
    """
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_group.setdefault(r["base_id"], []).append(r)
    balanced: list[str] = []
    flip_groups: list[str] = []
    for gid in sorted(by_group):
        grows = by_group[gid]
        eng = [r for r in grows if labels_by_sha[r["prompt_sha"]]["label"] == "engage"]
        ref = [r for r in grows if labels_by_sha[r["prompt_sha"]]["label"] == "refuse"]
        if not eng:
            continue
        flip_groups.append(gid)
        m = min(len(eng), len(ref))
        if m == 0:
            continue  # flip-group with no refuse rows: excluded from balanced
        eng_keep = eng if len(eng) == m else list(rng.choice(eng, size=m, replace=False))
        ref_base = [r for r in ref if r.get("axis") == "base"]
        ref_var = [r for r in ref if r.get("axis") != "base"]
        rng.shuffle(ref_var)
        ref_keep = (ref_base + ref_var)[:m]
        balanced.extend(r["prompt_sha"] for r in eng_keep)
        balanced.extend(r["prompt_sha"] for r in ref_keep)
    return balanced, flip_groups


def _balanced_arm_b(
    rows: list[dict[str, Any]],
    labels_by_sha: dict[str, dict[str, Any]],
    group_of: dict[str, str],
    seed: int,
) -> tuple[list[str], list[str], str]:
    """Arm-B group-level majority subsample to ~1:1.

    Keeps every group containing >=1 minority-labeled row (all its labeled
    rows), then adds majority-only groups greedily (seed-42 shuffle) while
    the majority row total stays <= the minority row total. Returns
    (balanced_shas, minority_group_ids, minority_label).
    """
    rng = np.random.default_rng(seed)
    labeled = [r for r in rows if labels_by_sha[r["prompt_sha"]]["label"] is not None]
    n_ref = sum(1 for r in labeled if labels_by_sha[r["prompt_sha"]]["label"] == "refuse")
    n_eng = len(labeled) - n_ref
    minority = "refuse" if n_ref <= n_eng else "engage"
    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in labeled:
        by_group.setdefault(group_of[r["prompt_sha"]], []).append(r)
    minority_groups = sorted(
        g
        for g, grows in by_group.items()
        if any(labels_by_sha[r["prompt_sha"]]["label"] == minority for r in grows)
    )
    balanced: list[str] = []
    maj_kept = 0
    min_kept = 0
    for g in minority_groups:
        for r in by_group[g]:
            balanced.append(r["prompt_sha"])
            if labels_by_sha[r["prompt_sha"]]["label"] == minority:
                min_kept += 1
            else:
                maj_kept += 1
    maj_only = sorted(g for g in by_group if g not in set(minority_groups))
    rng.shuffle(maj_only)
    for g in maj_only:
        add = len(by_group[g])
        if maj_kept + add > min_kept:
            continue
        maj_kept += add
        balanced.extend(r["prompt_sha"] for r in by_group[g])
    logger.info(
        "[groups] armB balanced: minority=%s rows min=%d maj=%d groups_min=%d",
        minority,
        min_kept,
        maj_kept,
        len(minority_groups),
    )
    return balanced, minority_groups, minority


def phase_groups(args: argparse.Namespace) -> None:
    """P4: labels -> grouping -> balanced sets -> pre-label all-group folds."""
    inputs = {
        "armA_manifest": manifest_sha(args, "armA"),
        "armB_manifest": manifest_sha(args, "armB"),
        "labels": _sha256_file(_eval_root(args) / "judge" / "labeling" / "labels.json"),
    }
    fp = _phase_fingerprint(args, "groups", inputs)
    sent = _sentinel_path(args, "groups")
    if _sentinel_ok(sent, fp, resume=not args.no_resume):
        logger.info("[groups] resume-skip (fingerprint match)")
        return

    judge_labels = load_judge_labels(args)
    arm_rows = {arm: load_manifest(args, arm) for arm in ARMS}
    texts = {arm: load_corpus_text(args, arm) for arm in ARMS}

    # --- row labels per arm -------------------------------------------------
    labels_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        out: dict[str, dict[str, Any]] = {}
        for r in arm_rows[arm]:
            sha = r["prompt_sha"]
            out[sha] = row_label_from_draws(
                judge_labels,
                sha,
                k_samples=args.k_samples,
                min_valid=args.min_valid,
                hi=args.label_hi,
                lo=args.label_lo,
            )
        labels_by_arm[arm] = out
        n_lab = sum(1 for v in out.values() if v["label"] is not None)
        logger.info("[groups] %s: %d/%d rows labeled", arm, n_lab, len(out))

    # --- Arm A grouping (natural base_id) ------------------------------------
    group_a = {r["prompt_sha"]: r["base_id"] for r in arm_rows["armA"]}

    # --- Arm B grouping (TF-IDF union v_C-cosine components + guards) --------
    shas_b = [r["prompt_sha"] for r in arm_rows["armB"]]
    texts_b = [texts["armB"][s] for s in shas_b]
    s_word, s_char = _tfidf_sims(texts_b)
    vc = _armb_vc_matrix(args, shas_b)
    s_vc = _cosine_matrix(vc)

    tfidf_tau, vc_tau, steps = args.tfidf_tau, args.vc_tau, 0
    while True:
        comp = _components_at(s_word, s_char, s_vc, tfidf_tau, vc_tau)
        sizes = np.bincount(comp)
        largest_frac = float(sizes.max()) / len(shas_b)
        if largest_frac <= args.largest_comp_frac or steps >= 5:
            break
        steps += 1
        tfidf_tau += 0.05
        vc_tau = min(vc_tau + 0.05, 0.999)
        logger.info(
            "[groups] armB degeneracy step %d: largest=%.3f -> taus %.2f/%.3f",
            steps,
            largest_frac,
            tfidf_tau,
            vc_tau,
        )
    if largest_frac > args.largest_comp_frac:
        raise RuntimeError(
            f"armB grouping degenerate after {steps} threshold steps "
            f"(largest component {largest_frac:.3f} > {args.largest_comp_frac})"
        )
    group_b = {s: f"B_g{int(c):05d}" for s, c in zip(shas_b, comp)}
    sensitivity = {}
    for tau in (0.45, 0.65):
        c = _components_at(s_word, s_char, s_vc, tau, vc_tau)
        sensitivity[str(tau)] = {
            "n_groups": int(c.max()) + 1,
            "largest_frac": float(np.bincount(c).max()) / len(shas_b),
        }

    group_of = {"armA": group_a, "armB": group_b}

    # --- balanced sets + yield floors ----------------------------------------
    bal_a, flip_groups = _balanced_arm_a(arm_rows["armA"], labels_by_arm["armA"], args.seed)
    bal_b, minority_groups, minority_b = _balanced_arm_b(
        arm_rows["armB"], labels_by_arm["armB"], group_b, args.seed
    )
    balanced = {"armA": bal_a, "armB": bal_b}
    floors = {
        "armA": {"flip_groups": len(flip_groups), "floor": args.floor_a},
        "armB": {"minority_groups": len(minority_groups), "floor": args.floor_b},
    }
    degraded: dict[str, bool] = {}
    for arm, count, floor in (
        ("armA", len(flip_groups), args.floor_a),
        ("armB", len(minority_groups), args.floor_b),
    ):
        degraded[arm] = False
        if count < floor:
            if not args.allow_degraded_folds:
                raise RuntimeError(
                    f"{arm} yield floor not met ({count} < {floor}): pre-registered branch = "
                    "extension wave (plan section 4 two-tier), else re-run this phase with "
                    "--allow-degraded-folds for the 2-fold degrade. Nothing aborts silently."
                )
            degraded[arm] = True
            logger.warning("[groups] %s below floor (%d < %d): 2-fold degrade", arm, count, floor)

    # --- fold assignment (pre-label, ALL groups — M1) -------------------------
    splits_by_arm: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        n_folds = 2 if degraded[arm] else args.n_folds
        rows = arm_rows[arm]
        gmap = group_of[arm]
        bal_set = set(balanced[arm])
        bal_groups = sorted({gmap[s] for s in bal_set})
        # stratum for balanced groups: class-mix (+ dataset source for armB)
        bal_strata: dict[str, str] = {}
        for g in bal_groups:
            g_shas = [s for s in bal_set if gmap[s] == g]
            labs = {labels_by_arm[arm][s]["label"] for s in g_shas}
            mix = "mixed" if len(labs) > 1 else next(iter(labs))
            if arm == "armB":
                srcs = {r["source"] for r in rows if r["prompt_sha"] in set(g_shas)}
                st = f"{sorted(srcs)[0]}|{mix}"
            else:
                st = mix
            bal_strata[g] = st
        rest_strata: dict[str, str] = {}
        for r in rows:
            g = gmap[r["prompt_sha"]]
            if g in bal_strata or g in rest_strata:
                continue
            rest_strata[g] = str(r.get("source") or r.get("axis") or "na")
        fold_of = _assign_folds(bal_strata, rest_strata, n_folds, args.seed)
        all_groups = sorted({gmap[s] for s in gmap})
        assert set(fold_of) == set(all_groups), "fold assignment must cover ALL groups (M1)"
        folds: dict[str, Any] = {}
        for k in range(n_folds):
            eval_groups = sorted(g for g in all_groups if fold_of[g] == k)
            train_groups = sorted(g for g in all_groups if fold_of[g] != k)
            eval_rows = sorted(s for s in bal_set if fold_of[gmap[s]] == k)
            train_rows = sorted(s for s in bal_set if fold_of[gmap[s]] != k)
            folds[str(k)] = {
                "train_row_ids": train_rows,
                "eval_row_ids": eval_rows,
                "train_groups": train_groups,
                "eval_groups": eval_groups,
            }
        splits_by_arm[arm] = {
            "arm": arm,
            "n_folds": n_folds,
            "degraded": degraded[arm],
            "seed": args.seed,
            "balanced_row_ids": sorted(bal_set),
            "group_fold": fold_of,
            "folds": folds,
        }

    # --- persist P4 outputs (no prompt text in the git-committed files) ------
    for arm in ARMS:
        arm_dir = _eval_root(args) / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        gmap = group_of[arm]
        lab_rows = {sha: {**labels_by_arm[arm][sha], "group_id": gmap[sha]} for sha in gmap}
        n_lab = sum(1 for v in lab_rows.values() if v["label"] is not None)
        _atomic_json(
            arm_dir / "labels.json",
            {
                "rows": lab_rows,
                "counts": {
                    "n_rows": len(lab_rows),
                    "n_labeled": n_lab,
                    "n_refuse": sum(1 for v in lab_rows.values() if v["label"] == "refuse"),
                    "n_engage": sum(1 for v in lab_rows.values() if v["label"] == "engage"),
                },
                "thresholds": {
                    "hi": args.label_hi,
                    "lo": args.label_lo,
                    "min_valid": args.min_valid,
                },
                "meta": _provenance(),
            },
        )
        groups_payload: dict[str, Any] = {
            "group_of": gmap,
            "size_histogram": _size_histogram(gmap),
            "floors": floors[arm],
            "degraded": degraded[arm],
            "meta": _provenance(),
        }
        if arm == "armA":
            groups_payload["flip_groups"] = flip_groups
        else:
            groups_payload.update(
                {
                    "minority_label": minority_b,
                    "minority_groups": minority_groups,
                    "thresholds_realized": {"tfidf_tau": tfidf_tau, "vc_tau": vc_tau},
                    "degeneracy_steps": steps,
                    "largest_component_frac": largest_frac,
                    "sensitivity_tfidf_tau": sensitivity,
                }
            )
        _atomic_json(arm_dir / "groups.json", groups_payload)
        _atomic_json(arm_dir / "splits.json", {**splits_by_arm[arm], "meta": _provenance()})
        for name in ("labels.json", "groups.json", "splits.json"):
            _upload_file(args, arm_dir / name, arm)

    # --- judge-interface files (carry prompt TEXT; HF-only, never git) -------
    splits_dir = _eval_root(args) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    bal_eval_rows = []
    for arm in ARMS:
        sp = splits_by_arm[arm]
        gmap = group_of[arm]
        for sha in sp["balanced_row_ids"]:
            bal_eval_rows.append(
                {
                    "row_id": sha,
                    "arm": arm,
                    "fold": sp["group_fold"][gmap[sha]],
                    "prompt": texts[arm][sha],
                }
            )
    _atomic_json(splits_dir / "balanced_eval_rows.json", bal_eval_rows)
    _upload_file(args, splits_dir / "balanced_eval_rows.json", "splits")
    for arm in ARMS:
        sp = splits_by_arm[arm]
        for k in range(sp["n_folds"]):
            rows_k = [
                {
                    "row_id": sha,
                    "prompt": texts[arm][sha],
                    "label": labels_by_arm[arm][sha]["label"],
                    "group_id": group_of[arm][sha],
                }
                for sha in sp["folds"][str(k)]["train_row_ids"]
            ]
            p = splits_dir / f"train_rows_{arm}_fold{k}.json"
            _atomic_json(p, rows_k)
            _upload_file(args, p, "splits")

    _write_sentinel(
        sent,
        fp,
        {
            "phase": "groups",
            "floors": floors,
            "degraded": degraded,
            "n_balanced": {a: len(balanced[a]) for a in ARMS},
        },
    )
    logger.info("[phase=groups done]")


# ---------------------------------------------------------------------------
# Store consolidation (per-corpus memmapped arrays; KEY ASSERTS at entry)
# ---------------------------------------------------------------------------


def _consolidate(args: argparse.Namespace, corpus: str, keys: tuple[str, ...]) -> Path:
    """Build (n, L, d) fp16 memmap .npy per key in manifest row order.

    Cached by manifest sha; consumer-loader KEY ASSERT on every store npz.
    Returns the consolidated dir.
    """
    cons = _out_root(args) / "consolidated"
    cons.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args, corpus)
    shas = [r["prompt_sha"] for r in rows]
    sidecar = cons / f"{corpus}.rows.json"
    want = {"manifest_sha": manifest_sha(args, corpus), "keys": sorted(keys), "rows": shas}
    if sidecar.exists():
        have = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            have.get("manifest_sha") == want["manifest_sha"]
            and set(have.get("keys", [])) >= set(keys)
            and have.get("rows") == shas
            and all((cons / f"{corpus}__{k}.npy").exists() for k in keys)
        ):
            return cons
    dest = _stage_summary_stores(args, shas)
    presence = ARM_PRESENCE_KEYS if corpus in ARMS else ()
    with np.load(dest / f"{shas[0]}.npz") as first:
        for k in (*keys, *presence):
            assert k in first.files, (corpus, k, sorted(first.files))
        n_hs, d_model = first["v_C"].shape
    mems = {
        k: np.lib.format.open_memmap(
            cons / f"{corpus}__{k}.npy.tmp",
            mode="w+",
            dtype=np.float16,
            shape=(len(shas), n_hs, d_model),
        )
        for k in keys
    }
    t0 = time.time()
    for i, sha in enumerate(shas):
        with np.load(dest / f"{sha}.npz") as data:
            for k in keys:
                assert k in data.files, (corpus, sha, k, sorted(data.files))
                v = data[k]
                assert v.shape == (n_hs, d_model), (sha, k, v.shape)
                mems[k][i] = v
        if (i + 1) % 250 == 0 or i + 1 == len(shas):
            _progress(f"consolidate_{corpus}", i + 1, len(shas), sha[:12], t0)
    for k in keys:
        mems[k].flush()
        del mems[k]
        os.replace(cons / f"{corpus}__{k}.npy.tmp", cons / f"{corpus}__{k}.npy")
    _atomic_json(sidecar, want)
    return cons


def _load_cons(args: argparse.Namespace, corpus: str, key: str) -> tuple[np.ndarray, list[str]]:
    """Memmapped consolidated array + row order (manifest order)."""
    cons = _out_root(args) / "consolidated"
    arr = np.load(cons / f"{corpus}__{key}.npy", mmap_mode="r")
    rows = json.loads((cons / f"{corpus}.rows.json").read_text(encoding="utf-8"))["rows"]
    assert arr.shape[0] == len(rows), (corpus, key, arr.shape, len(rows))
    return arr, rows


# ---------------------------------------------------------------------------
# Phase maps (P6): primal ridge v_C -> v_A_greedy, SVD bundles, diagnostics
# ---------------------------------------------------------------------------


def build_map_inputs_3b(
    manifest_rows: list[dict[str, Any]],
    group_of: dict[str, str],
    train_groups: list[str],
) -> list[str]:
    """M1 map-input builder — PRE-LABEL train-group membership ONLY.

    Inputs are exactly: the corpus manifest rows, the pre-label group map
    (groups.json ``group_of``), and splits.json ``train_groups``. This
    builder opens NEITHER labels.json NOR any balanced-row file (enforced by
    construction — it takes no path arguments); the mechanical set-equality
    assert below runs on every fold (fail-loud).
    """
    tg = set(train_groups)
    ids = sorted(r["prompt_sha"] for r in manifest_rows if group_of[r["prompt_sha"]] in tg)
    expected = {r["prompt_sha"] for r in manifest_rows if group_of[r["prompt_sha"]] in tg}
    if set(ids) != expected:  # pragma: no cover - structural fail-loud (M1)
        raise AssertionError("M1: map_train_row_ids != {row : group_id in train_groups}")
    return ids


def _svd_robust(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD with gesvd fallback on gesdd non-convergence (#722 r3 class)."""
    try:
        return np.linalg.svd(w, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as scipy_svd

        logger.warning("[maps] gesdd non-convergence -> gesvd fallback")
        return scipy_svd(w, full_matrices=False, lapack_driver="gesvd")


def _std_stats(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train-stat standardization params matching the primal core exactly:
    train mean, POPULATION std + 1e-9 (fit_h twin parity), y train mean."""
    xmu = x.mean(axis=0)
    xsd = x.std(axis=0) + 1e-9
    ymu = y.mean(axis=0)
    return xmu, xsd, ymu


def _map_bundle_path(args: argparse.Namespace, cond: str, layer: int) -> Path:
    d = _out_root(args) / "maps" / cond
    d.mkdir(parents=True, exist_ok=True)
    return d / f"layer{layer:02d}.npz"


def load_map_bundle(args: argparse.Namespace, cond: str, layer: int) -> dict[str, np.ndarray]:
    p = _map_bundle_path(args, cond, layer)
    with np.load(p) as data:
        for k in ("U", "S", "Vt", "xmu", "xsd", "ymu"):
            assert k in data.files, (cond, layer, k, sorted(data.files))
        return {k: data[k] for k in data.files}


def map_predict(
    bundle: dict[str, np.ndarray], x: np.ndarray, rank: int | None = None
) -> np.ndarray:
    """Reconstruct map predictions from the SVD bundle (optionally rank-r)."""
    xs = (x.astype(np.float64) - bundle["xmu"]) / bundle["xsd"]
    u, s, vt = bundle["U"].astype(np.float64), bundle["S"], bundle["Vt"].astype(np.float64)
    if rank is not None:
        u, s, vt = u[:, :rank], s[:rank], vt[:rank]
    return (xs @ (u * s)) @ vt + bundle["ymu"]


def map_z(bundle: dict[str, np.ndarray], x: np.ndarray, rank: int) -> np.ndarray:
    """Rank-r latent z_r = S_r^{1/2} U_r^T x_std (plan section 5 convention)."""
    xs = (x.astype(np.float64) - bundle["xmu"]) / bundle["xsd"]
    return (xs @ bundle["U"][:, :rank].astype(np.float64)) * np.sqrt(bundle["S"][:rank])


def _map_conditions(args: argparse.Namespace) -> list[tuple[str, str | None, int | None]]:
    conds: list[tuple[str, str | None, int | None]] = [("3a_generic", None, None)]
    for arm in ARMS:
        n_folds = load_splits(args, arm)["n_folds"]
        conds.extend((f"3b_{arm}_fold{k}", arm, k) for k in range(n_folds))
    return conds


def phase_maps(args: argparse.Namespace) -> None:
    """P6: 11 map-fit conditions x hs layers 1..28, SVD bundles + diagnostics."""
    inputs = {c: manifest_sha(args, c) for c in CORPORA}
    for arm in ARMS:
        inputs[f"{arm}_splits"] = _sha256_file(_eval_root(args) / arm / "splits.json")
    fp = _phase_fingerprint(args, "maps", inputs)
    sent = _sentinel_path(args, "maps")
    if _sentinel_ok(sent, fp, resume=not args.no_resume):
        logger.info("[maps] resume-skip (fingerprint match)")
        return

    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )
    from explore_persona_space.experiments.issue_1739.fits import (
        ridge_fit_predict_primal_layer_batched,
    )

    _consolidate(args, "generic", GENERIC_KEYS)
    for arm in ARMS:
        _consolidate(args, arm, ARM_KEYS)

    gx, g_rows = _load_cons(args, "generic", "v_C")
    gy, _ = _load_cons(args, "generic", "v_A_greedy")
    n_g, n_hs, d_model = gx.shape
    layers = list(range(1, n_hs))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_g)
    n_held = min(args.generic_heldout, max(1, n_g // 10))
    held_idx = np.sort(perm[:n_held])
    fit_idx = np.sort(perm[n_held:])
    logger.info(
        "[maps] generic fit=%d held=%d layers=%s d=%d", len(fit_idx), n_held, layers, d_model
    )

    group_of = {
        arm: json.loads((_eval_root(args) / arm / "groups.json").read_text())["group_of"]
        for arm in ARMS
    }
    arm_cons = {arm: _load_cons(args, arm, "v_C") for arm in ARMS}
    arm_ans = {arm: _load_cons(args, arm, "v_A_greedy")[0] for arm in ARMS}
    arm_row_pos = {arm: {s: i for i, s in enumerate(arm_cons[arm][1])} for arm in ARMS}
    manifests = {arm: load_manifest(args, arm) for arm in ARMS}

    diagnostics: dict[str, Any] = {}
    conds = _map_conditions(args)
    t0 = time.time()
    for ci, (cond, arm, fold) in enumerate(conds):
        cond_sent = _out_root(args) / "maps" / cond / "done.json"
        extra_ids: list[str] = []
        if arm is not None:
            sp = load_splits(args, arm)
            extra_ids = build_map_inputs_3b(
                manifests[arm], group_of[arm], sp["folds"][str(fold)]["train_groups"]
            )
        cond_fp = _sha256_obj({"fp": fp, "cond": cond, "extra_ids": extra_ids})
        if _sentinel_ok(cond_sent, cond_fp, resume=not args.no_resume):
            diagnostics[cond] = json.loads(
                (_out_root(args) / "maps" / cond / "diagnostics.json").read_text()
            )
            logger.info("[maps] %s resume-skip", cond)
            continue

        extra_pos = [arm_row_pos[arm][s] for s in extra_ids] if arm is not None else []
        per_layer: list[dict[str, Any]] = []
        for lo in range(0, len(layers), args.layer_chunk):
            chunk = layers[lo : lo + args.layer_chunk]
            xtr = np.stack(
                [
                    np.concatenate(
                        [gx[fit_idx, ell].astype(np.float32)]
                        + (
                            [arm_cons[arm][0][extra_pos, ell].astype(np.float32)]
                            if extra_pos
                            else []
                        )
                    )
                    for ell in chunk
                ]
            )
            ytr = np.stack(
                [
                    np.concatenate(
                        [gy[fit_idx, ell].astype(np.float32)]
                        + ([arm_ans[arm][extra_pos, ell].astype(np.float32)] if extra_pos else [])
                    )
                    for ell in chunk
                ]
            )
            xev = np.stack([gx[held_idx, ell].astype(np.float32) for ell in chunk])
            yev = np.stack([gy[held_idx, ell].astype(np.float32) for ell in chunk])
            assert xtr.shape[1] > d_model, (
                f"primal map fit requires n_train > d ({xtr.shape[1]} <= {d_model})"
            )
            preds, weights = ridge_fit_predict_primal_layer_batched(
                xtr, ytr, xev, device=args.device, return_weights=True, layer_chunk=len(chunk)
            )
            for j, ell in enumerate(chunk):
                xmu, xsd, ymu = _std_stats(xtr[j].astype(np.float64), ytr[j].astype(np.float64))
                u, s, vt = _svd_robust(weights[j])
                out = _map_bundle_path(args, cond, ell)
                tmp = out.with_name(f"layer{ell:02d}.tmp.npz")  # keep .npz suffix (np.savez)
                np.savez(
                    tmp,
                    U=u.astype(np.float32),
                    S=s.astype(np.float64),
                    Vt=vt.astype(np.float32),
                    xmu=xmu,
                    xsd=xsd,
                    ymu=ymu,
                )
                os.replace(tmp, out)
                ib_pred = identity_bias_predict(xtr[j], ytr[j], xev[j])
                sr = s / max(float(s.sum()), 1e-30)
                row = {
                    "layer": ell,
                    "r2_map": _r2_pooled(preds[j], yev[j]),
                    "r2_identity_bias": _r2_pooled(ib_pred, yev[j]),
                    "knn": {
                        m: knn_retrieval(preds[j], yev[j], ks=(1, 5, 10), metric=m)
                        for m in ("euclidean", "cosine")
                    },
                    "spectrum_top16": s[:16].tolist(),
                    "participation_ratio": float(1.0 / np.maximum((sr**2).sum(), 1e-30)),
                    "n_train": int(xtr.shape[1]),
                }
                per_layer.append(row)
                _progress(
                    "maps",
                    ci * len(layers) + len(per_layer),
                    len(conds) * len(layers),
                    f"{cond}/l{ell}",
                    t0,
                )
        best = max(per_layer, key=lambda r: r["r2_map"])
        cond_diag = {
            "condition": cond,
            "arm": arm,
            "fold": fold,
            "n_extra_indomain": len(extra_ids),
            "per_layer": per_layer,
            "best_layer_by_generic_r2": best["layer"],
            "best_r2": best["r2_map"],
        }
        _atomic_json(_out_root(args) / "maps" / cond / "diagnostics.json", cond_diag)
        diagnostics[cond] = cond_diag
        _upload_dir(
            args,
            _out_root(args) / "maps" / cond,
            f"analysis_tensors/maps/{cond}",
            [f"layer{ell:02d}.npz" for ell in layers] + ["diagnostics.json"],
        )
        _write_sentinel(cond_sent, cond_fp, {"phase": "maps", "condition": cond})

    out = {
        "conditions": diagnostics,
        "layers": layers,
        "generic": {"n_fit": int(len(fit_idx)), "n_heldout": int(n_held)},
        "notes": {
            "lambda_grid": "logspace(-2,4,13) GCV per slice (primal core default)",
            "selected_lambda": "not exposed by the primal core (n>d regime); "
            "dual-fit lambda diagnostics are reported in the probes phase",
        },
        "meta": _provenance(),
    }
    p = _results_dir(args) / "map_diagnostics.json"
    _atomic_json(p, out)
    _upload_file(args, p, "results")
    _write_sentinel(sent, fp, {"phase": "maps", "n_conditions": len(conds)})
    logger.info("[phase=maps done] %d conditions", len(conds))


def _r2_pooled(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-30)


# ---------------------------------------------------------------------------
# Phase probes (P7): all-linear predictor arms, nested selection, OOF scores
# ---------------------------------------------------------------------------


def _auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUROC with midrank ties; y in {0,1}, higher score -> y=1."""
    from scipy.stats import rankdata

    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    mask = np.isfinite(s)
    s, y = s[mask], y[mask]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


_WIDE_LAMBDAS = np.logspace(-2, 6, 17)


def _dual_ridge(
    x_tr: np.ndarray, y_tr: np.ndarray, x_ev: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray]:
    """Dual-Gram ridge over a leading slice axis; returns (scores, best_lambda).

    x_tr (L, n, d), y_tr (n,), x_ev (L, m, d) -> scores (L, m). gcv_dof_cap
    per plan (0.9); on the all-lambda-capped RuntimeError, retries once on
    the registered remedy (grid widened upward to 1e6).
    """
    from explore_persona_space.experiments.issue_779.fit_h import (
        ridge_fit_predict_fast_layer_batched,
    )

    n_slices = x_tr.shape[0]
    ytr = np.broadcast_to(
        np.asarray(y_tr, dtype=np.float64)[None, :, None], (n_slices, len(y_tr), 1)
    ).copy()
    try:
        preds, info = ridge_fit_predict_fast_layer_batched(
            x_tr, ytr, x_ev, device=args.device, return_info=True, gcv_dof_cap=args.gcv_dof_cap
        )
    except RuntimeError as exc:
        logger.warning("[probes] dof-cap exhausted grid (%s) -> widened grid retry", exc)
        preds, info = ridge_fit_predict_fast_layer_batched(
            x_tr,
            ytr,
            x_ev,
            lambdas=_WIDE_LAMBDAS,
            device=args.device,
            return_info=True,
            gcv_dof_cap=args.gcv_dof_cap,
        )
    return preds[:, :, 0], np.asarray(info["best_lambda"], dtype=np.float64)


def _dim_scores(x_tr: np.ndarray, y_tr: np.ndarray, x_ev: np.ndarray) -> np.ndarray:
    """Difference-in-means direction score per slice: (L, m)."""
    y = np.asarray(y_tr, dtype=bool)
    mu1 = x_tr[:, y].astype(np.float64).mean(axis=1, keepdims=True)
    mu0 = x_tr[:, ~y].astype(np.float64).mean(axis=1, keepdims=True)
    w = (mu1 - mu0).transpose(0, 2, 1)  # (L, d, 1)
    return (x_ev.astype(np.float64) @ w)[:, :, 0]


def _inner_splits(groups: list[str], k: int, seed: int) -> list[tuple[set[str], set[str]]]:
    """Deterministic k-way group split -> [(inner_train, inner_val)] pairs."""
    rng = np.random.default_rng(seed)
    g = sorted(set(groups))
    rng.shuffle(g)
    folds = [set(g[i::k]) for i in range(k)]
    return [(set(g) - f, f) for f in folds if f]


class _ArmData:
    """Loaded per-arm state shared across the probes/battery/transfer phases."""

    def __init__(self, args: argparse.Namespace, arm: str):
        self.arm = arm
        self.splits = load_splits(args, arm)
        self.labels = load_arm_labels(args, arm)["rows"]
        self.group_of = {sha: v["group_id"] for sha, v in self.labels.items()}
        self.vc, self.rows = _load_cons(args, arm, "v_C")
        self.ans = _load_cons(args, arm, "v_A_greedy")[0]
        self.ans_rm = _load_cons(args, arm, "v_A_rollout_mean")[0]
        self.pos = {sha: i for i, sha in enumerate(self.rows)}
        self.manifest = {r["prompt_sha"]: r for r in load_manifest(args, arm)}
        self.n_hs = self.vc.shape[1]
        self.layers = list(range(1, self.n_hs))
        self.bal = list(self.splits["balanced_row_ids"])
        self.y = {sha: 1 if self.labels[sha]["label"] == "refuse" else 0 for sha in self.bal}

    def feats(self, source: np.ndarray, shas: list[str]) -> np.ndarray:
        """(L, n, d) fp32 layer-leading features for the given rows."""
        idx = [self.pos[s] for s in shas]
        block = np.asarray(source[idx][:, self.layers], dtype=np.float32)  # (n, L, d)
        return np.ascontiguousarray(block.transpose(1, 0, 2))


def _layer_select_oof(
    args: argparse.Namespace,
    ad: _ArmData,
    source: np.ndarray,
    tr: list[str],
    ev: list[str],
    seed: int,
    *,
    engine: str = "ridge",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Engine A: inner-group-CV layer selection + final fit.

    Returns (eval scores at the selected layer, meta). engine in
    {"ridge", "dim"}; selection metric = pooled inner-OOF AUROC per layer on
    TRAIN groups only.
    """
    y_tr = np.array([ad.y[s] for s in tr])
    inner = _inner_splits([ad.group_of[s] for s in tr], args.inner_folds, seed)
    inner_scores = [[] for _ in ad.layers]
    inner_y: list[int] = []
    for itr_g, ival_g in inner:
        itr = [s for s in tr if ad.group_of[s] in itr_g]
        ival = [s for s in tr if ad.group_of[s] in ival_g]
        if not ival or len({ad.y[s] for s in itr}) < 2:
            continue
        x_itr = ad.feats(source, itr)
        x_ival = ad.feats(source, ival)
        y_itr = np.array([ad.y[s] for s in itr])
        if engine == "ridge":
            sc, _ = _dual_ridge(x_itr, y_itr, x_ival, args)
        else:
            sc = _dim_scores(x_itr, y_itr, x_ival)
        for li in range(len(ad.layers)):
            inner_scores[li].extend(sc[li].tolist())
        inner_y.extend(ad.y[s] for s in ival)
    inner_auroc = [
        _auroc(np.array(inner_scores[li]), np.array(inner_y)) for li in range(len(ad.layers))
    ]
    best_li = int(np.nanargmax(inner_auroc))
    x_tr = ad.feats(source, tr)
    x_ev = ad.feats(source, ev)
    lam = None
    if engine == "ridge":
        sc, lams = _dual_ridge(x_tr, y_tr, x_ev, args)
        lam = float(lams[best_li])
    else:
        sc = _dim_scores(x_tr, y_tr, x_ev)
    meta = {
        "layer": int(ad.layers[best_li]),
        "inner_auroc": float(inner_auroc[best_li]),
        "lambda": lam,
        "inner_auroc_by_layer": {str(ad.layers[i]): inner_auroc[i] for i in range(len(ad.layers))},
    }
    return sc[best_li], meta


def _rank_ladder(args: argparse.Namespace, d_model: int) -> list[int | str]:
    out: list[int | str] = []
    for tok in str(args.rank_ladder).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append("full" if tok == "full" else int(tok))
    return [r for r in out if r == "full" or r < d_model]


def _latents_for(
    args: argparse.Namespace,
    ad: _ArmData,
    cond: str,
    max_rank: int,
    want_full: bool,
) -> tuple[dict[int, np.ndarray], np.ndarray | None]:
    """Per-layer z latents for ALL balanced rows of the arm under one map
    condition. Returns ({layer: (n_bal, max_rank) f64}, {layer: full} or None)."""
    z_max: dict[int, np.ndarray] = {}
    z_full: dict[int, np.ndarray] = {}
    idx = [ad.pos[s] for s in ad.bal]
    for ell in ad.layers:
        bundle = load_map_bundle(args, cond, ell)
        x = np.asarray(ad.vc[idx, ell], dtype=np.float32)
        z_max[ell] = map_z(bundle, x, max_rank)
        if want_full:
            z_full[ell] = map_z(bundle, x, int(bundle["S"].shape[0]))
        del bundle
    return z_max, (z_full if want_full else None)


def _layer_rank_select_oof(
    args: argparse.Namespace,
    ad: _ArmData,
    z_by_layer: dict[int, np.ndarray],
    z_full_by_layer: dict[int, np.ndarray] | None,
    ladder: list[int | str],
    tr: list[str],
    ev: list[str],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Engine B: joint (layer, rank) selection by inner-group-CV AUROC.

    z_by_layer maps layer -> (n_bal, max_finite_rank) latents for ALL
    balanced rows (row order = ad.bal); rank-r features are the first r
    columns (z_r is a prefix of z_rmax by construction).
    """
    bal_pos = {s: i for i, s in enumerate(ad.bal)}
    y_tr = np.array([ad.y[s] for s in tr])
    inner = _inner_splits([ad.group_of[s] for s in tr], args.inner_folds, seed)
    layers = sorted(z_by_layer)

    def stack(shas: list[str], r: int | str) -> np.ndarray:
        rows = [bal_pos[s] for s in shas]
        if r == "full":
            assert z_full_by_layer is not None
            return np.stack([z_full_by_layer[ell][rows] for ell in layers])
        return np.stack([z_by_layer[ell][rows, :r] for ell in layers])

    pooled: dict[tuple[int, int | str], list[float]] = {}
    pooled_y: list[int] = []
    for itr_g, ival_g in inner:
        itr = [s for s in tr if ad.group_of[s] in itr_g]
        ival = [s for s in tr if ad.group_of[s] in ival_g]
        if not ival or len({ad.y[s] for s in itr}) < 2:
            continue
        y_itr = np.array([ad.y[s] for s in itr])
        for r in ladder:
            sc, _ = _dual_ridge(stack(itr, r), y_itr, stack(ival, r), args)
            for li, ell in enumerate(layers):
                pooled.setdefault((ell, r), []).extend(sc[li].tolist())
        pooled_y.extend(ad.y[s] for s in ival)
    y_arr = np.array(pooled_y)
    aurocs = {k: _auroc(np.array(v), y_arr) for k, v in pooled.items()}
    best_key = max(aurocs, key=lambda k: np.nan_to_num(aurocs[k], nan=-1.0))
    best_layer, best_rank = best_key
    sc, lams = _dual_ridge(stack(tr, best_rank), y_tr, stack(ev, best_rank), args)
    li = layers.index(best_layer)
    meta = {
        "layer": int(best_layer),
        "rank": best_rank,
        "inner_auroc": float(aurocs[best_key]),
        "lambda": float(lams[li]),
    }
    return sc[li], meta


def _pca_latents(args: argparse.Namespace, ad: _ArmData, max_rank: int) -> dict[int, np.ndarray]:
    """Matched-rank PCA control: PCs fit on the GENERIC corpus per layer
    (torch.pca_lowrank, seed-pinned), scores for the arm's balanced rows."""
    import torch

    cache_dir = _out_root(args) / "pca"
    cache_dir.mkdir(parents=True, exist_ok=True)
    gx, _ = _load_cons(args, "generic", "v_C")
    idx = [ad.pos[s] for s in ad.bal]
    out: dict[int, np.ndarray] = {}
    for ell in ad.layers:
        cache = cache_dir / f"generic_l{ell:02d}_q{max_rank}.npz"
        if not cache.exists():
            torch.manual_seed(args.seed)
            x = torch.as_tensor(np.asarray(gx[:, ell], dtype=np.float32))
            q = min(max_rank, x.shape[0] - 1, x.shape[1])
            _, _, v = torch.pca_lowrank(x, q=q, center=True)
            tmp = cache.with_name(cache.stem + ".tmp.npz")
            np.savez(tmp, mean=x.mean(dim=0).numpy(), V=v.numpy())
            os.replace(tmp, cache)
        with np.load(cache) as data:
            mean, v = data["mean"], data["V"]
        xa = np.asarray(ad.vc[idx, ell], dtype=np.float64)
        out[ell] = (xa - mean.astype(np.float64)) @ v.astype(np.float64)
    return out


def _text_features(
    ad: _ArmData,
    texts: dict[str, str],
    tr: list[str],
    ev: list[str],
    *,
    with_indicators: bool,
):
    """F2 text-surface features: TF-IDF (word 1-2 ∪ char 3-5) + length +
    token count + axis (armA) / source+category (armB) indicators.

    Vectorizers/one-hots are fit on TRAIN rows only (no eval leakage)."""
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer

    def dense_block(shas: list[str]) -> np.ndarray:
        cols = []
        for s in shas:
            t = texts[s]
            cols.append([np.log1p(len(t)), np.log1p(len(t.split()))])
        return np.asarray(cols, dtype=np.float64)

    blocks_tr, blocks_ev = [], []
    for kwargs in (
        {"analyzer": "word", "ngram_range": (1, 2), "max_features": 20000},
        {"analyzer": "char", "ngram_range": (3, 5), "max_features": 20000},
    ):
        vec = TfidfVectorizer(**kwargs)
        blocks_tr.append(vec.fit_transform([texts[s] for s in tr]))
        blocks_ev.append(vec.transform([texts[s] for s in ev]))
    blocks_tr.append(csr_matrix(dense_block(tr)))
    blocks_ev.append(csr_matrix(dense_block(ev)))
    if with_indicators:
        keys = ("axis",) if ad.arm == "armA" else ("source", "category")
        vocab = sorted({str(ad.manifest[s].get(k)) for s in tr for k in keys})
        v_ix = {v: i for i, v in enumerate(vocab)}

        def ind_block(shas: list[str]) -> np.ndarray:
            m = np.zeros((len(shas), len(vocab)))
            for i, s in enumerate(shas):
                for k in keys:
                    j = v_ix.get(str(ad.manifest[s].get(k)))
                    if j is not None:
                        m[i, j] = 1.0
            return m

        blocks_tr.append(csr_matrix(ind_block(tr)))
        blocks_ev.append(csr_matrix(ind_block(ev)))
    return hstack(blocks_tr).tocsr(), hstack(blocks_ev).tocsr()


def _text_surface_scores(
    ad: _ArmData,
    texts: dict[str, str],
    tr: list[str],
    ev: list[str],
    seed: int,
    *,
    with_indicators: bool,
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    x_tr, x_ev = _text_features(ad, texts, tr, ev, with_indicators=with_indicators)
    y_tr = np.array([ad.y[s] for s in tr])
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(x_tr, y_tr)
    refuse_col = list(clf.classes_).index(1)
    return clf.predict_proba(x_ev)[:, refuse_col]


def _ladder_class_groups(ad: _ArmData, tr: list[str]) -> dict[int, list[str]]:
    """Per-class candidate group lists for the limited-label ladder."""
    out: dict[int, set[str]] = {0: set(), 1: set()}
    for s in tr:
        out[ad.y[s]].add(ad.group_of[s])
    return {c: sorted(v) for c, v in out.items()}


def _ladder_sizes(args: argparse.Namespace) -> list[int | str]:
    out: list[int | str] = []
    for tok in str(args.ladder_sizes).split(","):
        tok = tok.strip()
        if tok:
            out.append("all" if tok == "all" else int(tok))
    return out


def phase_probes(args: argparse.Namespace) -> None:
    """P7 probes: OOF scores per predictor per arm + ladder + LODO."""
    inputs = {c: manifest_sha(args, c) for c in CORPORA}
    for arm in ARMS:
        inputs[f"{arm}_splits"] = _sha256_file(_eval_root(args) / arm / "splits.json")
        inputs[f"{arm}_labels"] = _sha256_file(_eval_root(args) / arm / "labels.json")
    fp = _phase_fingerprint(args, "probes", inputs)
    sent = _sentinel_path(args, "probes")
    if _sentinel_ok(sent, fp, resume=not args.no_resume):
        logger.info("[probes] resume-skip (fingerprint match)")
        return

    _consolidate(args, "generic", GENERIC_KEYS)
    for arm in ARMS:
        _consolidate(args, arm, ARM_KEYS)
    judge_scores = load_predictor_scores(args)
    ladder = None
    t0 = time.time()
    unit_dir = _out_root(args) / "probes"
    unit_dir.mkdir(parents=True, exist_ok=True)

    for arm in ARMS:
        ad = _ArmData(args, arm)
        texts = load_corpus_text(args, arm)
        if ladder is None:
            ladder = _rank_ladder(args, ad.vc.shape[2])
        finite_ranks = [r for r in ladder if r != "full"]
        max_rank = max(finite_ranks) if finite_ranks else 8
        pca_lat = _pca_latents(args, ad, max_rank)
        n_folds = ad.splits["n_folds"]
        z3a, z3a_full = _latents_for(args, ad, "3a_generic", max_rank, "full" in ladder)

        for k in range(n_folds):
            unit_key = f"{arm}_fold{k}"
            unit_path = unit_dir / f"{unit_key}.json"
            unit_fp = _sha256_obj({"fp": fp, "unit": unit_key})
            if unit_path.exists():
                have = json.loads(unit_path.read_text(encoding="utf-8"))
                if have.get("fingerprint") == unit_fp:
                    _progress("probes", k + 1, n_folds, f"{unit_key} (resume-skip)", t0)
                    continue
            tr = ad.splits["folds"][str(k)]["train_row_ids"]
            ev = ad.splits["folds"][str(k)]["eval_row_ids"]
            if not tr or not ev:
                raise RuntimeError(f"empty balanced split for {unit_key}")
            seed_k = args.seed + 1000 * k
            unit: dict[str, Any] = {
                "fingerprint": unit_fp,
                "arm": arm,
                "fold": k,
                "eval_row_ids": ev,
                "scores": {},
                "meta": {},
                "ladder": [],
            }

            sc, meta = _layer_select_oof(args, ad, ad.vc, tr, ev, seed_k, engine="ridge")
            unit["scores"][PRED_CTX] = sc.tolist()
            unit["meta"][PRED_CTX] = meta
            ctx_layer = meta["layer"]

            sc, meta = _layer_select_oof(args, ad, ad.vc, tr, ev, seed_k, engine="dim")
            unit["scores"][PRED_DIM] = sc.tolist()
            unit["meta"][PRED_DIM] = meta

            sc, meta = _layer_select_oof(args, ad, ad.ans, tr, ev, seed_k, engine="ridge")
            unit["scores"][PRED_ANS] = sc.tolist()
            unit["meta"][PRED_ANS] = meta

            sc, meta = _layer_select_oof(args, ad, ad.ans_rm, tr, ev, seed_k, engine="ridge")
            unit["scores"][PRED_ANS_RM] = sc.tolist()
            unit["meta"][PRED_ANS_RM] = meta

            sc, meta = _layer_rank_select_oof(args, ad, z3a, z3a_full, ladder, tr, ev, seed_k)
            unit["scores"][PRED_3A] = sc.tolist()
            unit["meta"][PRED_3A] = meta
            sel_3a = (meta["layer"], meta["rank"])

            z3b, z3b_full = _latents_for(args, ad, f"3b_{arm}_fold{k}", max_rank, "full" in ladder)
            sc, meta = _layer_rank_select_oof(args, ad, z3b, z3b_full, ladder, tr, ev, seed_k)
            unit["scores"][PRED_3B] = sc.tolist()
            unit["meta"][PRED_3B] = meta
            del z3b, z3b_full

            pca_ladder = [r for r in ladder if r != "full"]
            sc, meta = _layer_rank_select_oof(args, ad, pca_lat, None, pca_ladder, tr, ev, seed_k)
            unit["scores"][PRED_PCA] = sc.tolist()
            unit["meta"][PRED_PCA] = meta
            sel_pca = (meta["layer"], meta["rank"])

            unit["scores"][PRED_TEXT] = _text_surface_scores(
                ad, texts, tr, ev, seed_k, with_indicators=True
            ).tolist()
            unit["scores"][PRED_TEXT_NOIND] = _text_surface_scores(
                ad, texts, tr, ev, seed_k, with_indicators=False
            ).tolist()
            if arm == "armA":
                unit["scores"][PRED_ISREW] = [
                    float(ad.manifest[s].get("axis") != "base") for s in ev
                ]

            # limited-label ladder at the fold's selected configurations
            bal_pos = {s: i for i, s in enumerate(ad.bal)}
            cls_groups = _ladder_class_groups(ad, tr)
            for n_lab in _ladder_sizes(args):
                for seed_i in range(args.ladder_seeds if n_lab != "all" else 1):
                    rng = np.random.default_rng(seed_i)
                    if n_lab == "all":
                        sub = list(tr)
                    else:
                        keep_groups: dict[int, set[str]] = {}
                        for c, glist in cls_groups.items():
                            m = min(int(n_lab), len(glist))
                            keep_groups[c] = set(rng.choice(glist, size=m, replace=False).tolist())
                        sub = [s for s in tr if ad.group_of[s] in keep_groups[ad.y[s]]]
                    if len({ad.y[s] for s in sub}) < 2:
                        continue
                    y_ev = np.array([ad.y[s] for s in ev])
                    row: dict[str, Any] = {
                        "n_lab": n_lab,
                        "seed": seed_i,
                        "n_rows": len(sub),
                        "auroc": {},
                    }
                    x_tr = ad.feats(ad.vc, sub)[[ctx_layer - 1]]
                    x_ev = ad.feats(ad.vc, ev)[[ctx_layer - 1]]
                    sc, _ = _dual_ridge(x_tr, np.array([ad.y[s] for s in sub]), x_ev, args)
                    row["auroc"][PRED_CTX] = _auroc(sc[0], y_ev)
                    for name, lat, lat_full, sel in (
                        (PRED_3A, z3a, z3a_full, sel_3a),
                        (PRED_PCA, pca_lat, None, sel_pca),
                    ):
                        ell, r = sel
                        if r == "full":
                            feats_all = lat_full[ell]
                        else:
                            feats_all = lat[ell][:, :r]
                        ztr = feats_all[[bal_pos[s] for s in sub]][None]
                        zev = feats_all[[bal_pos[s] for s in ev]][None]
                        sc, _ = _dual_ridge(ztr, np.array([ad.y[s] for s in sub]), zev, args)
                        row["auroc"][name] = _auroc(sc[0], y_ev)
                    unit["ladder"].append(row)

            _atomic_json(unit_path, unit)
            _progress("probes", k + 1, n_folds, unit_key, t0)

        # ------- assemble the per-arm score file -----------------------------
        scores: dict[str, dict[str, Any]] = {}
        meta_by_fold: dict[str, Any] = {}
        ladder_rows: list[dict[str, Any]] = []
        for k in range(n_folds):
            unit = json.loads((unit_dir / f"{arm}_fold{k}.json").read_text(encoding="utf-8"))
            meta_by_fold[str(k)] = unit["meta"]
            for j, sha in enumerate(unit["eval_row_ids"]):
                rec = scores.setdefault(
                    sha,
                    {"y": ad.y[sha], "group_id": ad.group_of[sha], "fold": k},
                )
                for pred, vals in unit["scores"].items():
                    rec[pred] = vals[j]
            for row in unit["ladder"]:
                ladder_rows.append({"fold": k, **row})
        for sha, rec in scores.items():
            js = judge_scores.get(sha)
            if js is not None and js.get("p_refuse") is not None:
                rec[PRED_JUDGE] = float(js["p_refuse"])

        # ------- LODO (Arm B only): leave-one-dataset-out ctx ridge ----------
        if arm == "armB":
            sources = sorted({r["source"] for r in ad.manifest.values()})
            if len(sources) >= 2:
                for src in sources:
                    tr = [s for s in ad.bal if ad.manifest[s]["source"] != src]
                    ev = [s for s in ad.bal if ad.manifest[s]["source"] == src]
                    if not ev or len({ad.y[s] for s in tr}) < 2:
                        continue
                    sc, meta = _layer_select_oof(args, ad, ad.vc, tr, ev, args.seed, engine="ridge")
                    meta_by_fold[f"lodo_{src}"] = meta
                    for j, sha in enumerate(ev):
                        scores.setdefault(
                            sha, {"y": ad.y[sha], "group_id": ad.group_of[sha], "fold": -1}
                        )[PRED_LODO] = float(sc[j])

        out = {
            "arm": arm,
            "n_folds": n_folds,
            "scores": scores,
            "ladder": ladder_rows,
            "selection": meta_by_fold,
            "notes": {
                "orientation": "all scores oriented as P(REFUSE); y=1 is refuse",
                "pca_control": "finite ladder ranks only (matched-rank control)",
                "ladder": "fits at the fold's selected layer/(layer,rank)",
            },
            "meta": _provenance(),
        }
        p = _results_dir(args) / f"predictor_scores_{arm}.json"
        _atomic_json(p, out)
        _upload_file(args, p, "results")
        logger.info("[probes] %s: %d scored rows", arm, len(scores))

    _write_sentinel(sent, fp, {"phase": "probes"})
    logger.info("[phase=probes done]")


# ---------------------------------------------------------------------------
# Phase battery (P7): #2202 retrieval battery as the map-quality gate
# ---------------------------------------------------------------------------


def _ensure_scripts_on_syspath() -> None:
    p = str(_REPO_ROOT / "scripts")
    assert (_REPO_ROOT / "scripts" / "issue2356_fits.py").exists(), _REPO_ROOT
    if p not in sys.path:
        sys.path.insert(0, p)


def _whiten(x: np.ndarray, mu: np.ndarray, chol_l: np.ndarray) -> np.ndarray:
    from scipy.linalg import solve_triangular

    return solve_triangular(chol_l, (np.asarray(x, np.float64) - mu).T, lower=True).T


def _top1_indices(pred: np.ndarray, pool: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Chunked top-1 pool index under cosine similarity (for NN-behavior-match)."""
    pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
    qn = pool / (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-12)
    out = np.empty(len(pn), dtype=np.int64)
    for lo in range(0, len(pn), chunk):
        sim = pn[lo : lo + chunk] @ qn.T
        out[lo : lo + chunk] = sim.argmax(axis=1)
    return out


def _battery_metrics(
    pred: np.ndarray,
    pool: np.ndarray,
    true_idx: np.ndarray,
    *,
    mu_a: np.ndarray,
    chol_l: np.ndarray,
    n_boot: int,
    boot_seed: int,
) -> dict[str, Any]:
    """One battery read: 4 metric spaces + S2 bootstrap gate on the primary."""
    _ensure_scripts_on_syspath()
    import issue2202_failchar as fc
    import issue2202_freshwhiten_avg as fw

    n_pool = pool.shape[0]
    chance = 1.0 / n_pool
    out: dict[str, Any] = {"n_targets": int(len(true_idx)), "n_pool": int(n_pool), "chance": chance}
    spaces: dict[str, tuple[np.ndarray, np.ndarray, str]] = {
        "whitened_cosine": (_whiten(pred, mu_a, chol_l), _whiten(pool, mu_a, chol_l), "cosine"),
        "raw_euclidean": (pred, pool, "euclidean"),
        "pearson": (fw._row_demean(pred), fw._row_demean(pool), "cosine"),
    }
    acc1_flags: dict[str, np.ndarray] = {}
    for name, (p, q, metric) in spaces.items():
        ranks, _, n_closer = fc.ranks_of_targets(p, q, true_idx, metric, phase=f"battery_{name}")
        acc1 = np.asarray(n_closer) == 0
        acc1_flags[name] = acc1
        out[name] = {
            "acc_at_1": float(acc1.mean()),
            "median_rank": float(np.median(ranks)),
            "mrr": float(np.mean(1.0 / np.asarray(ranks, dtype=np.float64))),
        }
    r2ranks = fw.ranks_r2_cand_norm(pred, pool, true_idx, phase="battery_r2cn")
    r2ranks = np.asarray(r2ranks[0] if isinstance(r2ranks, tuple) else r2ranks, dtype=np.float64)
    out["r2_cand_norm"] = {
        "acc_at_1": float((r2ranks <= 1.0).mean()),
        "median_rank": float(np.median(r2ranks)),
        "mrr": float(np.mean(1.0 / r2ranks)),
    }
    # S2 gate: one-sided bootstrap over targets on the PRIMARY read
    rng = np.random.default_rng(boot_seed)
    flags = acc1_flags["whitened_cosine"].astype(np.float64)
    if n_boot > 0 and len(flags) > 1:
        draws = rng.integers(0, len(flags), size=(n_boot, len(flags)))
        boot = flags[draws].mean(axis=1)
        ci_lower = float(np.percentile(boot, 5.0))
    else:
        ci_lower = float("nan")
    out["s2_gate"] = {
        "primary": "whitened_cosine.acc_at_1",
        "ci_lower_5pct": ci_lower,
        "chance": chance,
        "pass": bool(ci_lower > chance) if np.isfinite(ci_lower) else None,
    }
    out["_acc1_flags_whitened"] = acc1_flags["whitened_cosine"].tolist()
    return out


def _draw_avg_targets(
    args: argparse.Namespace,
    ad: _ArmData,
    judge_labels: dict[str, dict[str, Any]],
    target_shas: list[str],
    layer: int,
) -> tuple[dict[str, np.ndarray], int]:
    """Mean of the first K SAME-LABEL sampled-draw v_A vectors per target
    (deterministic by draw index; #2202 K_DRAWS convention). Returns
    (sha -> (d,) f64, n_fallback_greedy)."""
    dest = _stage_summary_stores(args, target_shas)
    out: dict[str, np.ndarray] = {}
    n_fallback = 0
    for sha in target_shas:
        row_label = ad.labels[sha]["label"]
        with np.load(dest / f"{sha}.npz") as data:
            assert "v_A_sample_k" in data.files, (sha, sorted(data.files))
            vk = data["v_A_sample_k"]  # (K, L, d)
        picks = []
        for k in range(vk.shape[0]):
            item = judge_labels.get(f"{sha}.s{k:02d}")
            lab = (
                None
                if item is None
                else (item.get("label") or _label_from_score(item.get("score")))
            )
            if lab == row_label:
                picks.append(k)
            if len(picks) >= args.k_draw_avg:
                break
        if picks:
            out[sha] = vk[picks, layer].astype(np.float64).mean(axis=0)
        else:
            n_fallback += 1
            out[sha] = np.asarray(ad.ans[ad.pos[sha], layer], dtype=np.float64)
    return out, n_fallback


def phase_battery(args: argparse.Namespace) -> None:
    """P7 battery: per (arm x map condition x target-variant) retrieval reads."""
    inputs = {c: manifest_sha(args, c) for c in CORPORA}
    for arm in ARMS:
        inputs[f"{arm}_splits"] = _sha256_file(_eval_root(args) / arm / "splits.json")
    fp = _phase_fingerprint(args, "battery", inputs)
    sent = _sentinel_path(args, "battery")
    if _sentinel_ok(sent, fp, resume=not args.no_resume):
        logger.info("[battery] resume-skip (fingerprint match)")
        return

    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict
    from explore_persona_space.analysis.null_battery import (
        PRIMARY_LAMBDA,
        shrunk_cholesky_from_cov,
    )

    diag = json.loads((_results_dir(args) / "map_diagnostics.json").read_text(encoding="utf-8"))
    layer = int(diag["conditions"]["3a_generic"]["best_layer_by_generic_r2"])
    judge_labels = load_judge_labels(args)
    gx, _ = _load_cons(args, "generic", "v_C")
    gy, _ = _load_cons(args, "generic", "v_A_greedy")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(gx.shape[0])
    n_held = min(args.generic_heldout, max(1, gx.shape[0] // 10))
    fit_idx = np.sort(perm[n_held:])

    units_path = _out_root(args) / "battery" / "units.jsonl"
    done_units = {r["unit"]: r for r in _read_jsonl(units_path) if r.get("fingerprint") == fp}
    whiten_dir = _out_root(args) / "whiten"
    whiten_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    manifests = {arm: load_manifest(args, arm) for arm in ARMS}
    group_of_all = {
        arm: json.loads((_eval_root(args) / arm / "groups.json").read_text())["group_of"]
        for arm in ARMS
    }
    t0 = time.time()
    n_units_total = 0
    unit_i = 0
    for arm in ARMS:
        ad = _ArmData(args, arm)
        pool = np.asarray(ad.ans[:, layer], dtype=np.float64)  # ALL greedy answers
        pool_greedy_label = [ad.labels[s]["greedy_label"] for s in ad.rows]
        labeled = [s for s in ad.rows if ad.labels[s]["label"] is not None]
        n_folds = ad.splits["n_folds"]
        conds = ["3a_generic"] + [f"3b_{arm}_fold{k}" for k in range(n_folds)]
        n_units_total += len(conds) * 2
        gen_xtr = np.asarray(gx[fit_idx, layer], dtype=np.float64)
        gen_ytr = np.asarray(gy[fit_idx, layer], dtype=np.float64)

        for cond in conds:
            fold = None if cond == "3a_generic" else int(cond.rsplit("fold", 1)[1])
            if fold is None:
                targets = list(labeled)
                extra_ids: list[str] = []
            else:
                eval_groups = set(ad.splits["folds"][str(fold)]["eval_groups"])
                targets = [s for s in labeled if ad.group_of[s] in eval_groups]
                extra_ids = build_map_inputs_3b(
                    manifests[arm],
                    group_of_all[arm],
                    ad.splits["folds"][str(fold)]["train_groups"],
                )
            if not targets:
                logger.warning("[battery] %s/%s: no targets, skipped", arm, cond)
                continue
            # whiten stats: THIS run's map-fit train answers (generic fit +
            # in-domain train-group answers for 3b), at the battery layer.
            extra_pos = [ad.pos[s] for s in extra_ids]
            ytr = (
                np.concatenate([gen_ytr, np.asarray(ad.ans[extra_pos, layer], np.float64)])
                if extra_pos
                else gen_ytr
            )
            xtr = (
                np.concatenate([gen_xtr, np.asarray(ad.vc[extra_pos, layer], np.float64)])
                if extra_pos
                else gen_xtr
            )
            mu_a = ytr.mean(axis=0)
            centered = ytr - mu_a
            cov = (centered.T @ centered) / len(ytr)
            chol_l = shrunk_cholesky_from_cov(cov, PRIMARY_LAMBDA)
            wpath = whiten_dir / f"{arm}__{cond}.npz"
            if not wpath.exists():
                tmp = wpath.with_name(wpath.stem + ".tmp.npz")
                np.savez(
                    tmp,
                    mu_A=mu_a,
                    mu_C=xtr.mean(axis=0),
                    L=chol_l,
                    lam=np.float64(PRIMARY_LAMBDA),
                    n_train=np.int64(len(ytr)),
                )
                os.replace(tmp, wpath)
                _upload_file(args, wpath, "analysis_tensors/whiten")

            bundle = load_map_bundle(args, cond, layer)
            x_t = np.asarray(ad.vc[[ad.pos[s] for s in targets], layer], dtype=np.float32)
            preds = map_predict(bundle, x_t)
            ib_pred = identity_bias_predict(xtr, ytr, x_t.astype(np.float64))
            true_idx = np.array([ad.pos[s] for s in targets], dtype=np.int64)

            for variant in ("greedy", "draw_avg"):
                unit_i += 1
                unit_key = f"{arm}|{cond}|{variant}"
                if unit_key in done_units:
                    results.setdefault(arm, {}).setdefault(cond, {})[variant] = done_units[
                        unit_key
                    ]["result"]
                    _progress("battery", unit_i, n_units_total, f"{unit_key} (resume-skip)", t0)
                    continue
                pool_v = pool
                n_fb = 0
                if variant == "draw_avg":
                    da, n_fb = _draw_avg_targets(args, ad, judge_labels, targets, layer)
                    pool_v = pool.copy()
                    for s in targets:
                        pool_v[ad.pos[s]] = da[s]
                res = _battery_metrics(
                    preds,
                    pool_v,
                    true_idx,
                    mu_a=mu_a,
                    chol_l=chol_l,
                    n_boot=args.n_boot,
                    boot_seed=args.boot_seed,
                )
                acc1_flags = np.array(res.pop("_acc1_flags_whitened"), dtype=bool)
                y_t = np.array([ad.y[s] if s in ad.y else 0 for s in targets])
                lab_t = [ad.labels[s]["label"] for s in targets]
                res["behavior_split_acc1_whitened"] = {
                    lab: float(acc1_flags[[x == lab for x in lab_t]].mean())
                    for lab in ("refuse", "engage")
                    if any(x == lab for x in lab_t)
                }
                top1 = _top1_indices(_whiten(preds, mu_a, chol_l), _whiten(pool_v, mu_a, chol_l))
                match_flags = [
                    pool_greedy_label[j] == lab
                    for j, lab in zip(top1, lab_t)
                    if pool_greedy_label[j] is not None
                ]
                res["nn_behavior_match_rate"] = float(np.mean(match_flags)) if match_flags else None
                res["n_draw_avg_fallback_greedy"] = int(n_fb)
                del y_t
                # identity+bias baseline pushed through the SAME battery
                ib_res = _battery_metrics(
                    ib_pred,
                    pool_v,
                    true_idx,
                    mu_a=mu_a,
                    chol_l=chol_l,
                    n_boot=args.n_boot,
                    boot_seed=args.boot_seed,
                )
                ib_res.pop("_acc1_flags_whitened", None)
                res["identity_bias_baseline"] = ib_res
                results.setdefault(arm, {}).setdefault(cond, {})[variant] = res
                _append_jsonl(units_path, {"unit": unit_key, "fingerprint": fp, "result": res})
                _progress("battery", unit_i, n_units_total, unit_key, t0)

    out = {
        "battery_layer": layer,
        "results": results,
        "notes": {
            "primary": "whitened_cosine acc@1 vs chance=1/n_pool (one-sided 5th-pct gate)",
            "whiten": "shrunk Cholesky lam=0.1 from THIS run's map-fit train answers",
            "draw_avg": f"first {args.k_draw_avg} same-label sampled draws (deterministic)",
            "pool": "ALL captured greedy answers of the arm",
        },
        "meta": _provenance(),
    }
    p = _results_dir(args) / "map_discrimination.json"
    _atomic_json(p, out)
    _upload_file(args, p, "results")
    _write_sentinel(sent, fp, {"phase": "battery"})
    logger.info("[phase=battery done]")


# ---------------------------------------------------------------------------
# Phase transfer (P7, report-only): cross-regime A<->B
# ---------------------------------------------------------------------------


def phase_transfer(args: argparse.Namespace) -> None:
    """Report-only cross-regime transfer: ctx ridge (+ DiM) trained on ALL
    balanced rows of one arm, evaluated on the other arm's balanced rows."""
    inputs = {arm: _sha256_file(_eval_root(args) / arm / "splits.json") for arm in ARMS}
    fp = _phase_fingerprint(args, "transfer", inputs)
    sent = _sentinel_path(args, "transfer")
    if _sentinel_ok(sent, fp, resume=not args.no_resume):
        logger.info("[transfer] resume-skip (fingerprint match)")
        return

    for arm in ARMS:
        _consolidate(args, arm, ARM_KEYS)
    ads = {arm: _ArmData(args, arm) for arm in ARMS}
    out: dict[str, Any] = {"directions": {}, "notes": {"role": "report-only (plan Step G)"}}
    for src, dst in (("armA", "armB"), ("armB", "armA")):
        a_src, a_dst = ads[src], ads[dst]
        tr = list(a_src.bal)
        for engine in ("ridge", "dim"):
            sc_meta = _layer_select_oof(args, a_src, a_src.vc, tr, tr[:1], args.seed, engine=engine)
            layer = sc_meta[1]["layer"]
            x_tr = a_src.feats(a_src.vc, tr)[[layer - 1]]
            x_ev = a_dst.feats(a_dst.vc, list(a_dst.bal))[[layer - 1]]
            y_tr = np.array([a_src.y[s] for s in tr])
            if engine == "ridge":
                sc, _ = _dual_ridge(x_tr, y_tr, x_ev, args)
            else:
                sc = _dim_scores(x_tr, y_tr, x_ev)
            y_dst = np.array([a_dst.y[s] for s in a_dst.bal])
            key = f"{src}->{dst}|{engine}"
            out["directions"][key] = {
                "layer": layer,
                "n_train": len(tr),
                "n_eval": len(a_dst.bal),
                "auroc": _auroc(sc[0], y_dst),
                "scores": {s: float(v) for s, v in zip(a_dst.bal, sc[0])},
            }
            logger.info(
                "[transfer] %s auroc=%.3f layer=%d", key, out["directions"][key]["auroc"], layer
            )

    out["meta"] = _provenance()
    p = _results_dir(args) / "transfer.json"
    _atomic_json(p, out)
    _upload_file(args, p, "results")
    _write_sentinel(sent, fp, {"phase": "transfer"})
    logger.info("[phase=transfer done]")


# ---------------------------------------------------------------------------
# Phase stats (P7): pooled AUROC/balanced-acc, paired group bootstrap,
# advisory permutation, H1 lattice
# ---------------------------------------------------------------------------


def _weighted_auroc_draws(scores: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted Mann-Whitney AUROC per draw (midrank ties), vectorized.

    scores (n,), y (n,) in {0,1}, weights (n_draws, n) -> (n_draws,).
    """
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    yy = y[order].astype(np.float64)
    w = weights[:, order]
    starts = np.flatnonzero(np.concatenate(([True], s[1:] != s[:-1])))
    w_pos = w * yy
    w_neg = w * (1.0 - yy)
    seg_pos = np.add.reduceat(w_pos, starts, axis=1)
    seg_neg = np.add.reduceat(w_neg, starts, axis=1)
    cum_before = np.cumsum(seg_neg, axis=1) - seg_neg
    num = (seg_pos * (cum_before + 0.5 * seg_neg)).sum(axis=1)
    wp = w_pos.sum(axis=1)
    wn = w_neg.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / (wp * wn)
    out[(wp <= 0) | (wn <= 0)] = np.nan
    return out


def _group_bootstrap_weights(
    groups: list[str], n_draws: int, seed: int
) -> tuple[np.ndarray, list[str]]:
    """(n_draws, n_rows) group-multiplicity weights (resample groups w/ repl.)."""
    uniq = sorted(set(groups))
    g_ix = {g: i for i, g in enumerate(uniq)}
    row_g = np.array([g_ix[g] for g in groups])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_draws, len(uniq)))
    flat = draws + (np.arange(n_draws)[:, None] * len(uniq))
    counts = np.bincount(flat.ravel(), minlength=n_draws * len(uniq)).reshape(n_draws, len(uniq))
    return counts[:, row_g].astype(np.float64), uniq


def _balanced_acc(scores: np.ndarray, y: np.ndarray, thresh: float) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    pred = (s > thresh).astype(np.int64)
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    tnr = float((1 - pred)[y == 0].mean()) if (y == 0).any() else float("nan")
    return 0.5 * (tpr + tnr)


def _auroc_cols(scores_mat: np.ndarray, y_mat: np.ndarray) -> np.ndarray:
    """Column-wise AUROC (ordinal ranks; continuous scores so ties ~measure-0).

    scores_mat (n, n_draws), y_mat (n, n_draws) in {0,1} -> (n_draws,).
    """
    ranks = scores_mat.argsort(axis=0).argsort(axis=0).astype(np.float64) + 1.0
    n1 = y_mat.sum(axis=0).astype(np.float64)
    n0 = y_mat.shape[0] - n1
    num = (ranks * y_mat).sum(axis=0) - n1 * (n1 + 1.0) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / (n1 * n0)
    out[(n1 <= 0) | (n0 <= 0)] = np.nan
    return out


def _dual_hat(x_tr: np.ndarray, x_ev: np.ndarray, lam: float) -> np.ndarray:
    """Frozen-lambda dual-ridge hat H (n_ev, n_tr): scores = H @ y + const.

    Same standardization convention as the fit cores (train mean, population
    std + 1e-9); the additive constant from y-centering is AUROC-invariant.
    """
    xmu = x_tr.mean(axis=0)
    xsd = x_tr.std(axis=0) + 1e-9
    xn = (np.asarray(x_tr, np.float64) - xmu) / xsd
    xe = (np.asarray(x_ev, np.float64) - xmu) / xsd
    k = xn @ xn.T
    k_ev = xe @ xn.T
    return k_ev @ np.linalg.solve(k + lam * np.eye(k.shape[0]), np.eye(k.shape[0]))


def _perm_null_arm(
    args: argparse.Namespace, ad: _ArmData, meta_by_fold: dict[str, Any]
) -> dict[str, Any]:
    """Advisory group-label permutation null for the ctx probe (#2).

    Per draw: size-stratified permutation of per-group label vectors, dual
    hats at the OBSERVED per-(fold, layer) GCV lambda (frozen across draws),
    per-draw inner layer re-selection, pooled outer AUROC vs the permuted
    labels. Deviations from the registered fit core (frozen lambda; this
    module's own hat convention) are advisory-only and recorded in the note.
    """
    n_perm = args.n_perm
    if n_perm <= 0:
        return {"skipped": True}
    n_folds = ad.splits["n_folds"]
    bal = ad.bal
    bal_pos = {s: i for i, s in enumerate(bal)}
    y_obs = np.array([ad.y[s] for s in bal], dtype=np.float64)

    # size-stratified permutation of per-group label vectors
    rows_of: dict[str, list[int]] = {}
    for s in bal:
        rows_of.setdefault(ad.group_of[s], []).append(bal_pos[s])
    gids = sorted(rows_of)
    rng = np.random.default_rng(args.perm_seed)
    yp = np.empty((len(bal), n_perm), dtype=np.float64)
    buckets: dict[int, list[str]] = {}
    for g in gids:
        buckets.setdefault(len(rows_of[g]), []).append(g)
    for d in range(n_perm):
        for _, bucket in sorted(buckets.items()):
            perm = rng.permutation(len(bucket))
            for gi, g in enumerate(bucket):
                src = bucket[perm[gi]]
                for r_dst, r_src in zip(rows_of[g], rows_of[src]):
                    yp[r_dst, d] = y_obs[r_src]
    assert np.allclose(yp.sum(axis=0), y_obs.sum()), "group permutation must preserve counts"

    # frozen-lambda hats per (fold, layer) + inner hats
    null_by_fold: list[tuple] = []
    for k in range(n_folds):
        tr = ad.splits["folds"][str(k)]["train_row_ids"]
        ev = ad.splits["folds"][str(k)]["eval_row_ids"]
        x_tr_all = ad.feats(ad.vc, tr)
        x_ev_all = ad.feats(ad.vc, ev)
        _, lams = _dual_ridge(x_tr_all, np.array([ad.y[s] for s in tr]), x_ev_all, args)
        hats = {
            ell: _dual_hat(x_tr_all[li], x_ev_all[li], float(lams[li]))
            for li, ell in enumerate(ad.layers)
        }
        inner = _inner_splits([ad.group_of[s] for s in tr], args.inner_folds, args.seed + 1000 * k)
        inner_hats = []
        for itr_g, ival_g in inner:
            itr = [s for s in tr if ad.group_of[s] in itr_g]
            ival = [s for s in tr if ad.group_of[s] in ival_g]
            if not ival or not itr:
                continue
            xi_tr = ad.feats(ad.vc, itr)
            xi_ev = ad.feats(ad.vc, ival)
            h = {
                ell: _dual_hat(xi_tr[li], xi_ev[li], float(lams[li]))
                for li, ell in enumerate(ad.layers)
            }
            inner_hats.append(([bal_pos[s] for s in itr], [bal_pos[s] for s in ival], h))
        null_by_fold.append(([bal_pos[s] for s in tr], [bal_pos[s] for s in ev], hats, inner_hats))

    null_scores = np.full((len(bal), n_perm), np.nan)
    for tr_ix, ev_ix, hats, inner_hats in null_by_fold:
        # inner: pooled scores per layer over all draws -> per-draw layer pick
        pooled_rows: list[int] = []
        pooled_by_layer = {ell: [] for ell in ad.layers}
        for itr_ix, ival_ix, h in inner_hats:
            for ell in ad.layers:
                pooled_by_layer[ell].append(h[ell] @ yp[itr_ix])
            pooled_rows.extend(ival_ix)
        y_val = yp[pooled_rows]
        auroc_by_layer = np.stack(
            [_auroc_cols(np.concatenate(pooled_by_layer[ell], axis=0), y_val) for ell in ad.layers]
        )  # (L, n_perm)
        best_li = np.nanargmax(np.nan_to_num(auroc_by_layer, nan=-1.0), axis=0)  # (n_perm,)
        outer = np.stack([hats[ell] @ yp[tr_ix] for ell in ad.layers])  # (L, n_ev, n_perm)
        sel = outer[best_li, :, np.arange(n_perm)].T  # (n_ev, n_perm)
        null_scores[ev_ix] = sel
    null_auroc = _auroc_cols(null_scores, yp)
    obs = meta_by_fold.get("observed_ctx_auroc")
    return {
        "n_perm": int(n_perm),
        "null_mean": float(np.nanmean(null_auroc)),
        "null_q50": float(np.nanpercentile(null_auroc, 50)),
        "null_q95": float(np.nanpercentile(null_auroc, 95)),
        "observed": obs,
        "p_value_advisory": (
            float((1 + np.nansum(null_auroc >= obs)) / (n_perm + 1)) if obs is not None else None
        ),
        "note": "advisory only: frozen per-(fold,layer) GCV lambda from the observed fit; "
        "this module's dual-hat convention; per-draw inner layer re-selection",
    }


def phase_stats(args: argparse.Namespace) -> None:
    """P7 stats: pooled metrics + paired group bootstrap + H1 lattice."""
    inputs = {
        arm: _sha256_file(_results_dir(args) / f"predictor_scores_{arm}.json") for arm in ARMS
    }
    fp = _phase_fingerprint(args, "stats", inputs)
    sent = _sentinel_path(args, "stats")
    if _sentinel_ok(sent, fp, resume=not args.no_resume):
        logger.info("[stats] resume-skip (fingerprint match)")
        return

    out: dict[str, Any] = {"arms": {}, "h1": {}, "notes": {}}
    delta_int_ci: dict[str, tuple[float, float] | None] = {}
    for arm in ARMS:
        data = json.loads(
            (_results_dir(args) / f"predictor_scores_{arm}.json").read_text(encoding="utf-8")
        )
        rows = {sha: rec for sha, rec in data["scores"].items() if rec.get("fold", -1) >= 0}
        preds_present = [p for p in HEADLINE_PREDS if any(p in rec for rec in rows.values())]
        mask = [
            sha
            for sha, rec in sorted(rows.items())
            if all(p in rec and rec[p] is not None and np.isfinite(rec[p]) for p in preds_present)
        ]
        n_excluded = len(rows) - len(mask)
        if not mask:
            raise RuntimeError(f"{arm}: empty common row mask across {preds_present}")
        y = np.array([rows[s]["y"] for s in mask], dtype=np.int64)
        groups = [rows[s]["group_id"] for s in mask]
        weights, uniq_groups = _group_bootstrap_weights(groups, args.n_boot, args.boot_seed)

        table: dict[str, Any] = {}
        boot_aurocs: dict[str, np.ndarray] = {}
        for p in preds_present + [
            q for q in (PRED_ANS_RM, PRED_TEXT_NOIND, PRED_ISREW) if any(q in rows[s] for s in mask)
        ]:
            sc = np.array([rows[s].get(p, np.nan) for s in mask], dtype=np.float64)
            valid = np.isfinite(sc)
            point = _auroc(sc[valid], y[valid])
            thresh = float(np.median(sc[valid]))
            bacc = _balanced_acc(sc[valid], y[valid], thresh)
            entry: dict[str, Any] = {
                "auroc": point,
                "balanced_acc": bacc,
                "threshold": thresh,
                "n": int(valid.sum()),
            }
            if p in preds_present:
                boots = _weighted_auroc_draws(sc, y, weights)
                boot_aurocs[p] = boots
                entry["auroc_ci95"] = [
                    float(np.nanpercentile(boots, 2.5)),
                    float(np.nanpercentile(boots, 97.5)),
                ]
            table[p] = entry

        contrasts: dict[str, Any] = {}
        for name, a, b in (
            ("delta_int", PRED_CTX, PRED_JUDGE),
            ("ctx_minus_text_surface", PRED_CTX, PRED_TEXT),
            ("map3a_minus_pca", PRED_3A, PRED_PCA),
            ("map3b_minus_map3a", PRED_3B, PRED_3A),
            ("ans_minus_ctx", PRED_ANS, PRED_CTX),
        ):
            if a in boot_aurocs and b in boot_aurocs:
                diff = boot_aurocs[a] - boot_aurocs[b]
                ci = (float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5)))
                contrasts[name] = {
                    "point": table[a]["auroc"] - table[b]["auroc"],
                    "ci95": list(ci),
                    "n_draws": int(args.n_boot),
                }
                if name == "delta_int":
                    delta_int_ci[arm] = ci
            elif name == "delta_int":
                delta_int_ci[arm] = None

        # ladder summary: per (pred, n_lab) mean over folds per seed -> mean+-sd
        ladder_summary: dict[str, dict[str, Any]] = {}
        for row in data.get("ladder", []):
            for p, v in row["auroc"].items():
                ladder_summary.setdefault(p, {}).setdefault(str(row["n_lab"]), {}).setdefault(
                    str(row["seed"]), []
                ).append(v)
        ladder_out: dict[str, Any] = {}
        for p, by_n in ladder_summary.items():
            ladder_out[p] = {}
            for n_lab, by_seed in by_n.items():
                per_seed = [float(np.nanmean(v)) for v in by_seed.values()]
                ladder_out[p][n_lab] = {
                    "mean": float(np.nanmean(per_seed)),
                    "sd": float(np.nanstd(per_seed)),
                    "n_seeds": len(per_seed),
                }

        # LODO summary (arm B)
        lodo: dict[str, Any] = {}
        if arm == "armB":
            lodo_rows = [
                (sha, rec) for sha, rec in data["scores"].items() if rec.get(PRED_LODO) is not None
            ]
            if lodo_rows:
                sc = np.array([r[PRED_LODO] for _, r in lodo_rows])
                yy = np.array([r["y"] for _, r in lodo_rows])
                lodo["pooled_auroc"] = _auroc(sc, yy)
                lodo["n"] = len(lodo_rows)

        ad = _ArmData(args, arm)
        obs_meta = {"observed_ctx_auroc": table.get(PRED_CTX, {}).get("auroc")}
        perm = _perm_null_arm(args, ad, obs_meta)

        out["arms"][arm] = {
            "n_common_rows": len(mask),
            "n_excluded_from_mask": n_excluded,
            "n_groups": len(uniq_groups),
            "predictors": table,
            "contrasts": contrasts,
            "ladder": ladder_out,
            "lodo": lodo,
            "permutation_advisory": perm,
        }

    # H1 3-way lattice on delta_int (ctx - judge), both arms
    cis = [delta_int_ci.get(arm) for arm in ARMS]
    if any(c is None for c in cis):
        verdict = "not-computable (judge scores missing in >=1 arm)"
    elif all(c[0] > 0 for c in cis):
        verdict = "Confirmed"
    elif any(c[1] < 0 for c in cis):
        verdict = "Falsified"
    else:
        verdict = "Inconclusive"
    out["h1"] = {
        "definition": "delta_int = AUROC(ctx_ridge) - AUROC(judge_fewshot) per arm; "
        "Confirmed iff CI>0 both arms; Falsified iff CI wholly <0 in either; else Inconclusive",
        "delta_int_ci_by_arm": {a: (list(c) if c else None) for a, c in delta_int_ci.items()},
        "verdict": verdict,
    }
    transfer_p = _results_dir(args) / "transfer.json"
    if transfer_p.exists():
        tr = json.loads(transfer_p.read_text(encoding="utf-8"))
        out["transfer_aurocs"] = {k: v["auroc"] for k, v in tr.get("directions", {}).items()}
    out["notes"] = {
        "common_mask": "rows with ALL headline predictor scores present; rows with a "
        "missing judge score are EXCLUDED, never imputed to 50 (registered delta_int)",
        "balanced_acc_threshold": "pooled per-predictor score median (label-free; "
        "balanced eval sets put the prior at 0.5)",
        "bootstrap": f"paired group bootstrap, {args.n_boot} draws, seed {args.boot_seed}, "
        "weighted Mann-Whitney AUROC over ONE common per-arm row mask",
    }
    out["meta"] = _provenance()
    p = _results_dir(args) / "stats.json"
    _atomic_json(p, out)
    _upload_file(args, p, "results")
    _write_sentinel(sent, fp, {"phase": "stats", "h1": out["h1"]["verdict"]})
    logger.info("[phase=stats done] H1=%s", out["h1"]["verdict"])


# ---------------------------------------------------------------------------
# Import-check (module-level fn: avoids the in-main import-shadowing trap),
# signature binds for every reused-helper call site, selftest, CLI
# ---------------------------------------------------------------------------


def _import_check() -> None:
    """Execute every deferred import + signature-BIND every reused-helper call
    site (shapes as placeholders; nothing is executed). Exits loudly on any
    missing symbol or call-shape mismatch."""
    import inspect

    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)

    import torch  # noqa: F401
    from huggingface_hub import HfApi, hf_hub_download  # noqa: F401
    from scipy.linalg import solve_triangular, svd  # noqa: F401
    from scipy.sparse import csr_matrix, hstack  # noqa: F401
    from scipy.sparse.csgraph import connected_components  # noqa: F401
    from scipy.stats import rankdata  # noqa: F401
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
    from sklearn.linear_model import LogisticRegression  # noqa: F401

    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )
    from explore_persona_space.analysis.null_battery import (
        PRIMARY_LAMBDA,
        shrunk_cholesky_from_cov,
    )
    from explore_persona_space.experiments.issue_779.fit_h import (
        ridge_fit_predict_fast_layer_batched,
    )
    from explore_persona_space.experiments.issue_1739.fits import (
        ridge_fit_predict_primal_layer_batched,
    )
    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    _ensure_scripts_on_syspath()
    import issue2202_failchar as fc
    import issue2202_freshwhiten_avg as fw

    a = np.zeros((2, 4, 3))
    y3 = np.zeros((2, 4, 1))
    e3 = np.zeros((2, 2, 3))
    m2 = np.zeros((4, 3))
    v1 = np.zeros(4)
    idx = np.zeros(2, dtype=np.int64)
    binds: list[tuple[Any, tuple, dict]] = [
        (
            ridge_fit_predict_fast_layer_batched,
            (a, y3, e3),
            {"device": "cpu", "return_info": True, "gcv_dof_cap": 0.9},
        ),
        (
            ridge_fit_predict_fast_layer_batched,
            (a, y3, e3),
            {"lambdas": _WIDE_LAMBDAS, "device": "cpu", "return_info": True, "gcv_dof_cap": 0.9},
        ),
        (
            ridge_fit_predict_primal_layer_batched,
            (a, a, e3),
            {"device": "cpu", "return_weights": True, "layer_chunk": 2},
        ),
        (identity_bias_predict, (m2, m2, m2), {}),
        (knn_retrieval, (m2, m2), {"ks": (1, 5, 10), "metric": "euclidean"}),
        (shrunk_cholesky_from_cov, (np.eye(3), PRIMARY_LAMBDA), {}),
        (fc.ranks_of_targets, (m2, m2, idx, "cosine"), {"phase": "bind"}),
        (fc.ranks_of_targets, (m2, m2, idx, "euclidean"), {"phase": "bind"}),
        (fw.ranks_r2_cand_norm, (m2, m2, idx), {"phase": "bind"}),
        (fw._row_demean, (m2,), {}),
        (hub.stage_hub_prefix, ("repo/x", "prefix", Path("/tmp/x")), {"repo_type": "dataset"}),
        (
            hub._upload,
            (),
            {
                "local_path": Path("/tmp/x"),
                "repo_id": "repo/x",
                "repo_type": "dataset",
                "path_in_repo": "p/x",
                "raise_on_error": True,
                "upload_as_file": True,
            },
        ),
        (
            hub.verify_repo_paths_uploaded,
            (None, "repo/x", ["a"]),
            {"path_in_repo": "p", "repo_type": "dataset"},
        ),
        (assert_out_root_headroom, (Path("/tmp"), 1.0), {"phase": "bind"}),
        (as_metadata_dict, (git_provenance(),), {}),
        (solve_triangular, (np.eye(3), np.zeros((3, 2))), {"lower": True}),
    ]
    n_bound = 0
    for fn, pos, kw in binds:
        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            print(f"[import-check] skip-with-note: no signature for {fn!r}", flush=True)
            continue
        sig.bind(*pos, **kw)
        n_bound += 1
    try:
        inspect.signature(hub.retry_transient).bind_partial(hf_hub_download)
        n_bound += 1
    except (ValueError, TypeError):
        print("[import-check] skip-with-note: retry_transient signature unavailable", flush=True)
    assert callable(HfApi)
    print(f"[import-check] OK: deferred imports executed; {n_bound} call sites bound", flush=True)


# ---------------------------------------------------------------------------
# Selftest: all six phases in-process on tiny synthetic data
# ---------------------------------------------------------------------------


def _selftest(_: argparse.Namespace) -> int:
    """End-to-end shape smoke on synthetic arrays (no pod / GPU / network).

    Tiny dims deliberately distinct (d=16, n_hs=5, K=6) so shape bugs cannot
    hide behind square shapes; strong planted signal so AUROC/battery asserts
    are meaningful at this n.
    """
    import shutil
    import tempfile

    base = Path(tempfile.mkdtemp(prefix="i2356-selftest-"))
    eval_root = base / "eval_results"
    out_root = base / "out"
    stores = out_root / "summary_stores"
    stores.mkdir(parents=True)
    (eval_root / "corpus" / "text").mkdir(parents=True)
    (eval_root / "judge" / "labeling").mkdir(parents=True)
    (eval_root / "judge" / "predictor").mkdir(parents=True)
    rng = np.random.default_rng(0)
    d, n_hs, k_samp = 16, 5, 6
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)
    w_true = rng.normal(size=(d, d)) * 0.2 + np.eye(d)
    vocab = [f"w{i:03d}" for i in range(300)]

    def mk_text() -> str:
        return " ".join(rng.choice(vocab, size=10, replace=False).tolist())

    def mk_store(sha: str, y: int | None, with_samples: bool) -> None:
        vc = rng.normal(scale=0.4, size=(n_hs, d))
        if y is not None:
            vc[1:] += (2.5 if y else -2.5) * np.linspace(0.5, 1.0, n_hs - 1)[:, None] * u
        va = vc @ w_true + rng.normal(scale=0.05, size=(n_hs, d))
        payload = {"v_C": vc.astype(np.float16), "v_A_greedy": va.astype(np.float16)}
        if with_samples:
            vk = np.stack([va + rng.normal(scale=0.05, size=va.shape) for _ in range(k_samp)])
            payload["v_A_sample_k"] = vk.astype(np.float16)
            payload["v_A_rollout_mean"] = vk.mean(0).astype(np.float16)
        np.savez(stores / f"{sha}.npz", **payload)

    label_items: dict[str, dict[str, Any]] = {}
    pred_scores: dict[str, dict[str, Any]] = {}
    manifests: dict[str, list[dict[str, Any]]] = {"armA": [], "armB": [], "generic": []}
    texts: dict[str, list[dict[str, str]]] = {"armA": [], "armB": [], "generic": []}

    def add_labels(sha: str, y: int) -> None:
        score = 10.0 if y else 90.0  # y=1 refuse -> low engage score
        label_items[f"{sha}.greedy"] = {"score": score, "label": _label_from_score(score)}
        for k in range(k_samp):
            label_items[f"{sha}.s{k:02d}"] = {"score": score, "label": _label_from_score(score)}
        p_ref = float(np.clip(0.8 * y + 0.1 + rng.normal(scale=0.03), 0, 1))
        pred_scores[sha] = {"p_answer": 1 - p_ref, "p_refuse": p_ref, "arm": "", "fold": 0}

    for i in range(60):
        sha = f"g{i:04d}"
        manifests["generic"].append({"prompt_sha": sha})
        texts["generic"].append({"prompt_sha": sha, "prompt": mk_text()})
        mk_store(sha, int(rng.integers(0, 2)), False)
    axes = ("base", "past_tense", "passive_voice", "formal_register")
    for b in range(12):
        for j, axis in enumerate(axes):
            sha = f"a{b:02d}{j}"
            y = 1 if j in (0, 1) else 0  # base+1 variant refuse, 2 variants comply
            manifests["armA"].append(
                {"prompt_sha": sha, "base_id": f"base{b:02d}", "axis": axis, "source": "bank"}
            )
            texts["armA"].append({"prompt_sha": sha, "prompt": mk_text()})
            mk_store(sha, y, True)
            add_labels(sha, y)
    dup_text = mk_text()
    for i in range(40):
        sha = f"b{i:04d}"
        y = 1 if i < 16 else 0
        manifests["armB"].append(
            {
                "prompt_sha": sha,
                "source": "orb" if i % 2 == 0 else "pht",
                "category": f"c{i % 3}",
                "n_tok": 10,
            }
        )
        t = dup_text + " please" if i == 39 else (dup_text if i == 38 else mk_text())
        texts["armB"].append({"prompt_sha": sha, "prompt": t})
        mk_store(sha, y, True)
        add_labels(sha, y)

    for corpus in CORPORA:
        _atomic_json(
            eval_root / "corpus" / f"{corpus}_manifest.json",
            {"rows": manifests[corpus], "meta": {"selftest": True}},
        )
        with open(eval_root / "corpus" / "text" / f"{corpus}.jsonl", "w", encoding="utf-8") as fh:
            for row in texts[corpus]:
                fh.write(json.dumps(row) + "\n")
    _atomic_json(eval_root / "judge" / "labeling" / "labels.json", {"labels": label_items})
    _atomic_json(
        eval_root / "judge" / "predictor" / "predictor_scores.json", {"scores": pred_scores}
    )

    argv = [
        "--eval-root",
        str(eval_root),
        "--out-root",
        str(out_root),
        "--stores-dir",
        str(stores),
        "--no-upload",
        "--n-folds",
        "3",
        "--inner-folds",
        "2",
        "--vc-layer",
        "2",
        "--k-samples",
        "6",
        "--min-valid",
        "5",
        "--floor-a",
        "3",
        "--floor-b",
        "3",
        "--rank-ladder",
        "2,4,full",
        "--ladder-sizes",
        "2,all",
        "--ladder-seeds",
        "2",
        "--n-boot",
        "50",
        "--n-perm",
        "20",
        "--generic-heldout",
        "12",
        "--k-draw-avg",
        "2",
        "--phase",
        "groups,maps,probes,battery,transfer,stats",
    ]
    args = build_argparser().parse_args(argv)
    for ph in args.phase.split(","):
        PHASES[ph](args)

    # --- asserts -------------------------------------------------------------
    for arm in ARMS:
        sp = load_splits(args, arm)
        gmap = json.loads((eval_root / arm / "groups.json").read_text())["group_of"]
        assert set(sp["group_fold"]) == set(gmap.values()), f"{arm}: fold coverage != all groups"
    gb = json.loads((eval_root / "armB" / "groups.json").read_text())["group_of"]
    assert gb["b0038"] == gb["b0039"], "near-dup pair must land in one group"
    diag = json.loads((eval_root / "results" / "map_diagnostics.json").read_text())
    best_r2 = diag["conditions"]["3a_generic"]["best_r2"]
    assert best_r2 > 0.5, f"map r2 too low for a planted linear map: {best_r2}"
    for arm in ARMS:
        ps = json.loads((eval_root / "results" / f"predictor_scores_{arm}.json").read_text())
        n_bal = len(load_splits(args, arm)["balanced_row_ids"])
        n_scored = sum(1 for r in ps["scores"].values() if r.get("fold", -1) >= 0)
        assert n_scored == n_bal, (arm, n_scored, n_bal)
    st = json.loads((eval_root / "results" / "stats.json").read_text())
    assert st["h1"]["verdict"] in ("Confirmed", "Falsified", "Inconclusive"), st["h1"]
    for arm in ARMS:
        ctx = st["arms"][arm]["predictors"][PRED_CTX]["auroc"]
        assert np.isfinite(ctx) and ctx > 0.6, (arm, ctx)
    bt = json.loads((eval_root / "results" / "map_discrimination.json").read_text())
    acc = bt["results"]["armA"]["3a_generic"]["greedy"]["whitened_cosine"]["acc_at_1"]
    assert acc > bt["results"]["armA"]["3a_generic"]["greedy"]["chance"], acc
    assert (eval_root / "results" / "transfer.json").exists()
    print(f"[selftest] PASS (all six phases; scratch={base})", flush=True)
    shutil.rmtree(base, ignore_errors=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PHASES = {
    "groups": phase_groups,
    "maps": phase_maps,
    "probes": phase_probes,
    "battery": phase_battery,
    "transfer": phase_transfer,
    "stats": phase_stats,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #2356 fits driver (P4/P6/P7).")
    ap.add_argument("--phase", default="", help="comma-separated: " + "|".join(sorted(PHASES)))
    ap.add_argument("--list-phases", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="use the *_smoke HF prefix")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--stage-from-hub", action="store_true")
    ap.add_argument("--eval-root", default=str(_REPO_ROOT / "eval_results" / f"issue_{ISSUE}"))
    ap.add_argument("--out-root", default=str(_REPO_ROOT / "data" / f"issue_{ISSUE}" / "fits"))
    ap.add_argument("--stores-dir", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=4)
    ap.add_argument("--vc-layer", type=int, default=14)
    ap.add_argument("--tfidf-tau", type=float, default=0.55)
    ap.add_argument("--vc-tau", type=float, default=0.92)
    ap.add_argument("--largest-comp-frac", type=float, default=0.10)
    ap.add_argument("--label-hi", type=float, default=0.7)
    ap.add_argument("--label-lo", type=float, default=0.3)
    ap.add_argument("--min-valid", type=int, default=7)
    ap.add_argument("--k-samples", type=int, default=10)
    ap.add_argument("--floor-a", type=int, default=25)
    ap.add_argument("--floor-b", type=int, default=40)
    ap.add_argument("--allow-degraded-folds", action="store_true")
    ap.add_argument("--rank-ladder", default="4,8,16,32,64,128,256,full")
    ap.add_argument("--ladder-sizes", default="25,50,100,200,all")
    ap.add_argument("--ladder-seeds", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=1234)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--perm-seed", type=int, default=5678)
    ap.add_argument("--gcv-dof-cap", type=float, default=0.9)
    ap.add_argument("--layer-chunk", type=int, default=4)
    ap.add_argument("--generic-heldout", type=int, default=800)
    ap.add_argument("--k-draw-avg", type=int, default=4)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.list_phases:
        print(",".join(sorted(PHASES)), flush=True)
        return 0
    if args.import_check:
        _import_check()
        return 0
    if args.selftest:
        return _selftest(args)
    if not args.phase:
        build_argparser().error("--phase is required (or --selftest / --import-check)")
    for ph in args.phase.split(","):
        ph = ph.strip()
        if ph not in PHASES:
            raise SystemExit(f"unknown phase {ph!r}; choose from {sorted(PHASES)}")
        PHASES[ph](args)
    return 0


if __name__ == "__main__":
    rc = main()
    # Explicit flush + exit: heavy C-extension imports (torch/scipy/sklearn)
    # can hit the PyGILState_Release atexit race on bare interpreter exit
    # (gotchas.md phased-dispatcher entry).
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
