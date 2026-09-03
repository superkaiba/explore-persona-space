"""Issue #2661 — flat Der-recipe CONTEXT SAE + full-dictionary context-feature ->
answer-feature map (odd-correlation edge read): pod-side phase-dispatch driver.

Lean run (no /issue cycle; tasks/*/2661/body.md ## Provenance is the decision
record). Reuse basis, cited per function where verbatim:

  - scripts/issue2552_turnsae_der.py @ branch issue-2552 cb39df3ce1c (assemble
    delegation, flat-SAE construction + train loop, census pass, mining heap,
    pinned-chunk text iteration, sharded-JSONL eval lists, upload staging).
  - scripts/vendored_2476/ @ d8e9f8bdd4 (assemble kernel, MatryoshkaBatchTopKSAE,
    split/pool plumbing, FVE/R2/null/kNN kernels) — vendored files byte-identical.
  - scripts/issue2569_rowbattery.py (leg 4: per-feature hurdle metrics, 27-value
    lambda grid protocol, index-aligned null convention).
  - scripts/issue1482_map_coefficients.py (edge recipe: split-half top-M
    replication + label-shuffle per-column null threshold + structure moments).

Phases (PHASE_ORDER; every phase --smoke-able, CPU-runnable at tiny N):

  assemble        vendored parent kernel VERBATIM: 1,920 banked capture chunks @
                  89cfa76cdc -> X19/Y19 fp16 memmaps, splits sha-asserted.
  sae_train_ctx   flat BatchTopK on X19 train rows (933,444): width 32,768,
                  k=128, lr 2e-4, batch 256, 3 epochs, Adam(0.9,0.999),
                  threshold EMA 0.999, init seed 2661. Constructed DIRECTLY with
                  tier_bounds=(32768,) — never through _sae_tier_bounds (the
                  #2552 plan §4 P1.2 trap). HALT rc=25 when the 20k-holdout
                  variance-FVE < 0.5 (the task-body floor; SAE-val FVE logged
                  beside it for #2476/#2552 parity).
  sae_metrics     BOTH SAEs (fresh ctx on X19; REUSED #2552 answer SAE @
                  fdcec4c823e2 on Y19), on the 20k holdout AND the 10k SAE-val
                  rows: Der nMSE (raw = Der's; mean-centered variant beside it),
                  variance-FVE, realized L0, dead census (120k fit rows + 20k
                  holdout), log-binned firing-fraction histogram, per-feature
                  activation-when-active mean. Der reference numbers logged.
  encode_full     ALL 32,768 features per side on the 120k fit + 400 val + 20k
                  holdout rows -> CSR npz (+ row ids); zero-variance-on-fit
                  columns reported per side (dropped mechanically downstream).
  map_ridge_full  ridge nonzero-variance ctx features -> ALL 32,768 answer
                  features on the 120k fit rows; standardized inputs (xmu/xsd),
                  intercept via ymu; lambda by whole-map val R^2 (400 rows) over
                  the #2569 27-value grid 1e-5..1e8 (edge hits reported); GPU
                  Gram + eigendecomposition, fp64 accumulation; SAVES B (fp16,
                  reindexed to 32,768 rows) + xmu/xsd/ymu/lambda + holdout preds.
  map_mlp         2-layer MLP (hidden 4,096, GELU), Adam, batch 1,024, early
                  stop on val pooled R^2 (patience 3, max 30 epochs), seed 2661;
                  holdout preds + checkpoint saved.
  controls        composed zero-fit route (banked dense map @ 89cfa76cdc ->
                  answer-SAE encode), dense-input ridge (3,584-d state -> answer
                  features, same lambda protocol), index-aligned identity+bias
                  (labelled NULL), train-mean null, 20-draw row-shuffle null
                  (lambda pinned; seeds 2661100-2661119), kNN retrieval
                  (euclidean+cosine, k in {1,5,10}, chance k/20,000).
  perfeature_reads per answer feature x route: unconditional held-out R^2,
                  firing AUROC, conditional-magnitude R^2 (fired rows ONLY,
                  never mixed), fit/holdout firing counts, per-feature
                  shuffle-null band (2.5/97.5 pct over 20 draws); census_only
                  when holdout firing count == 0. NO alive floor anywhere.
  edges           from B in standardized units: top-32 in-edges per answer
                  feature + top-32 out-edges per ctx feature; split-half refit
                  (60k/60k, FULL-train standardizer) + label-shuffle null
                  (5 draws x 2,048 answer columns, seeds 2661000+); an edge
                  survives iff top-20,000 |coef| of BOTH halves AND
                  sign-consistent with the full fit AND above its column's null
                  threshold (#1482 recipe). wiring_edges.npz + top_pairs.json +
                  coef_structure.json + per-column edge-mass curves + the
                  receipts answer-feature sets (regex over the committed #2552
                  descriptions copy).
  eval_lists      ctx-SAE lists for the SAME 2,000 eval rows #2552 judged
                  (pinned eval_lists artifact @ fdcec4c823e2): all-active +
                  judged top-100, persisted like feature_lists_rep_ta.*.jsonl.
  mining          description-need set = eval-list union  U  ctx features in
                  top_pairs  U  top-32 in-edges of the receipts answer set; per
                  feature: top-25 activating USER PROMPTS over the 120k fit rows
                  + 20 non-activating negatives (kind=negative rows; the #2552
                  top25_*.jsonl shape otherwise). Context-side evidence only.
  upload          HF issue2661_flatsae/{analysis_tensors,raw_completions}/...;
                  small JSONs copied under eval_results/issue_2661/; manifest
                  with byte sizes printed.

Pod-side contract: sentinels via issue779_common.write_sentinel(task_id=2661),
[phase=...] log lines, terminal [phase=done]. LMSYS/WildChat text handled
DIGEST-ONLY in logs. Resume is regime-keyed via the vendored
_enter_phase_regime; phase outputs additionally guarded by local checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "vendored_2476"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue1482_early_layer as EL  # noqa: E402
import issue1482_error_analysis as EA  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
import turnavg_sae as T  # noqa: E402  (vendored @ d8e9f8bdd4)

from explore_persona_space.atomic_io import savez_atomic  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2661")

TASK_ID = 2661

# ── vendoring-depth fixups (the #2552 driver convention, VERBATIM) ───────────────
T.PROJECT_ROOT = PROJECT_ROOT
T.COMMITTED_SPLIT_1482 = PROJECT_ROOT / "eval_results" / "issue_1482" / "split_1482.json"
T.COMMITTED_M_SPLIT = (
    PROJECT_ROOT / "eval_results" / "issue_1482" / "matryoshka_tier" / "m_split.json"
)
T.TASK_ID = TASK_ID  # sentinels carry issue-2661 naming (pod-side contract)

# ── context-SAE recipe (task body: Der recipe, arXiv 2606.28548 App. A) ──────────
CTX_DICT = 32_768
CTX_K = 128
CTX_TIER_BOUNDS = (32_768,)  # FLAT — never through _sae_tier_bounds (#2552 trap)
CTX_SEED = 2661  # SAE init; the SAE-val carve stays @ 2476 (parent parity)
RC_G1 = 25
G1_FVE_FLOOR = 0.5  # task-body halt floor on the 20k-holdout variance-FVE
DER_REFERENCE_NMSE = {"turn_averaged": 0.097, "per_token": 0.162}  # paper numbers
I2552_ANSWER_NMSE_HOLDOUT = 0.078  # #2552 rep SAE holdout nMSE (task-body record)

# ── map / controls constants (task body + #2569/#1482 conventions) ───────────────
LAMBDA_GRID_27 = tuple(np.logspace(-5.0, 8.0, 27))  # #2569 issue2569_gateladder VERBATIM
MLP_HIDDEN = 4_096
MLP_BATCH = 1_024
MLP_LR = 1e-3  # unspecified by the brief — recorded in deviations.md
MLP_MAX_EPOCHS = 30
MLP_PATIENCE = 3
MLP_SEED = 2661
SHUFFLE_SEEDS = tuple(range(2_661_100, 2_661_120))  # 20-draw row-shuffle null (task body)
KNN_KS = (1, 5, 10)
ZERO_VAR_EPS = 1e-12

# ── edge-gate constants (#1482 map_coefficients recipe, seeds re-based to 2661) ──
TOP_EDGES = 32
TOPM = 20_000  # candidate pool + per-half top-K set size (#1482 VERBATIM)
SPLITHALF_SEED = 2661
EDGE_NULL_SEED_BASE = 2_661_000
N_NULL_DRAWS = 5
NULL_COLS = 2_048
EDGE_MASS_RANKS = (1, 2, 4, 8, 16, 32, 64, 128)

# ── mining constants (#2552 conventions + task-body negatives) ───────────────────
MINING_TOP = 25
MINING_NEG = 20
MINING_NEG_CAND = 4_096  # shared seeded candidate pool for negatives
EXAMPLE_TEXT_CAP = 1_500
EVAL_TURNS_N = 2_000

# receipts: answer-side behavior families resolved by regex over the committed
# #2552 description copy (task body "Receipts set"); ids recorded, never assumed
RECEIPTS_PATTERNS = {
    "refusal": r"refus|declin|cannot assist|won't help|unable to (?:help|assist|comply)|"
    r"policy|safety guideline|content (?:policy|guideline)|ethic|inappropriate|harmful request",
    "ccp_government_position": r"chinese government|ccp|communist party|official (?:chinese|prc)|"
    r"one[- ]china|taiwan.*(?:province|part of china)|state council|beijing'?s (?:position|stance)",
    "qwen_identity": r"qwen|alibaba|tongyi|created by alibaba|developed by alibaba",
    "sycophancy_adjacent": r"agree(?:ment|ing|s) with the user|prais|flatter|complimen|"
    r"validat\w* the user|affirm\w* the user|enthusiastic agreement",
    "harmful_content": r"violen|weapon|explosiv|illegal drug|malware|exploit code|"
    r"self[- ]harm|suicid|abuse|hate speech|slur",
}

# ── pins (eval_results/issue_2661/regime_pins.json is the committed record) ──────
LINEAGE_REVISION = T.DATA_REPO_REVISION  # 89cfa76cdc… (capture chunks + banked refit)
I2552_REVISION = "fdcec4c823e2638ae8661ccafca8f30f84ac6233"  # answer SAE + eval lists
ANSWER_SAE_PREFIX = "issue2552_turnsae/analysis_tensors/sae_rep"
EVAL_LISTS_PREFIX = "issue2552_turnsae/analysis_tensors/eval_lists"
REFIT_HOLDOUT_PATH = (
    "issue1482_error_analysis/analysis_tensors/percontext/refit_holdout__ridge__seed0.npz"
)
REGIME_PINS_PATH = PROJECT_ROOT / "eval_results" / "issue_2661" / "regime_pins.json"
DESCRIPTIONS_PATH = (
    PROJECT_ROOT / "eval_results" / "issue_2661" / "inputs" / ("descriptions_rep_ta.json")
)
LAYER = 19


def _regime_pins() -> dict:
    assert REGIME_PINS_PATH.exists(), (
        f"regime_pins.json missing at {REGIME_PINS_PATH} — commit the pin record on "
        "issue-2661 before any pod dispatch"
    )
    doc = json.loads(REGIME_PINS_PATH.read_text())
    assert doc["capture_chunks_revision"] == LINEAGE_REVISION, (
        doc["capture_chunks_revision"],
        LINEAGE_REVISION,
    )
    assert doc["issue2552_revision"] == I2552_REVISION
    return doc


def _resolve_repo_revision(revision: str | None, what: str) -> str:
    """Resolve a ref/short-sha (None = current HEAD) on the data repo to the full
    40-hex sha through the transient-retry envelope (VERBATIM from the #2552
    driver @ cb39df3ce1c). The judge driver's pod-artifact fetch MUST resolve
    HEAD at fetch time: pod uploads land AFTER the pin record was committed, so
    the committed pins structurally cannot see them (#2552 r2
    p4-future-revision). The judge imports THIS module for it (review r1 Major 1:
    the helper was referenced but never ported)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    info = hub.retry_transient(
        lambda: HfApi().repo_info(C.HF_DATA_REPO, repo_type="dataset", revision=revision),
        what=what,
    )
    return str(info.sha)


# ── small shared helpers (issue2552_turnsae_der.py @ cb39df3ce1c, VERBATIM
#    except the 2661 dir names) ───────────────────────────────────────────────────


def _eval_dir(args) -> Path:
    return args.out_root / "eval"


def _mining_dir(args) -> Path:
    return args.out_root / "mining"


def _lists_dir(args) -> Path:
    return args.out_root / "eval_lists"


def _sae_ctx_dir(args) -> Path:
    return args.out_root / "sae_ctx"


def _metrics_dir(args) -> Path:
    return args.out_root / "sae_metrics"


def _encodes_dir(args) -> Path:
    return args.out_root / "encodes"


def _ridge_dir(args) -> Path:
    return args.out_root / "map_ridge"


def _mlp_dir(args) -> Path:
    return args.out_root / "map_mlp"


def _controls_dir(args) -> Path:
    return args.out_root / "controls"


def _perfeature_dir(args) -> Path:
    return args.out_root / "perfeature"


def _edges_dir(args) -> Path:
    return args.out_root / "edges"


def _ctx_width(args) -> int:
    return int(args.ctx_width) if int(args.ctx_width) > 0 else CTX_DICT


def _ctx_k(args) -> int:
    return int(args.ctx_k) if int(args.ctx_k) > 0 else CTX_K


def _production(args) -> bool:
    """True only when EVERY smoke knob is off (#2552 g3-M5 convention)."""
    return (
        args.max_chunks == 0
        and args.smoke_rows == 0
        and not args.smoke
        and args.sae_steps == 0
        and args.n_eval_turns == 0
        and not args.tiny_model
        and int(args.ctx_width) == 0
        and int(args.ctx_k) == 0
    )


def _hf_fetch(path_in_repo: str, dest_dir: Path, revision: str) -> Path:
    """Revision-pinned single-file fetch through the canonical transient-retry
    envelope (VERBATIM from the #2552 driver; never snapshot_download)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    dest_dir.mkdir(parents=True, exist_ok=True)
    got = hub.retry_transient(
        lambda: hf_hub_download(
            C.HF_DATA_REPO,
            filename=path_in_repo,
            repo_type="dataset",
            revision=revision,
            local_dir=str(dest_dir),
        ),
        what=f"pinned fetch ({path_in_repo}@{revision[:8]})",
    )
    return Path(got)


def _positions_of(args, ids: np.ndarray, what: str) -> np.ndarray:
    """Map assembled row ids -> memmap positions (VERBATIM #2552). Production:
    every id MUST be present; smoke slices drop absent ids with a WARN."""
    a_dir = T._assemble_dir(args)
    rows_present = np.load(a_dir / "rows_present.npy")
    ids = np.asarray(ids, np.int64)
    pos = np.searchsorted(rows_present, ids)
    ok = (pos < len(rows_present)) & (rows_present[np.minimum(pos, len(rows_present) - 1)] == ids)
    if _production(args):
        assert bool(ok.all()), (
            f"[{what}] {int((~ok).sum())} of {len(ids)} rows absent from the assembled "
            "memmap — irreconcilable in production"
        )
    elif not bool(ok.all()):
        logger.warning(
            "[%s] smoke: %d/%d rows absent from slice; dropped", what, int((~ok).sum()), len(ids)
        )
    return pos[ok].astype(np.int64)


def _measured_update(args, **kv) -> None:
    path = _eval_dir(args) / "regime_measured.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(path.read_text()) if path.exists() else {}
    doc.update({k: v for k, v in kv.items()})
    T._write_json(path, doc, phase="regime_measured")


def _sha_ids(ids: np.ndarray) -> str:
    return EL._sha_ids(np.asarray(ids, np.int64))


def _jsonl_write_sharded(base: Path, rows: list[dict], cap_bytes: int = 9_000_000) -> list[Path]:
    """VERBATIM from the #2552 driver: single file < cap, else .shardNNN parts
    (< 9 MB each — the non-LFS upload rule; UTF-8 BYTE sizing)."""
    payloads = [json.dumps(r, ensure_ascii=False) for r in rows]
    sizes = [len(p.encode("utf-8")) for p in payloads]
    total = sum(s + 1 for s in sizes)
    base.parent.mkdir(parents=True, exist_ok=True)
    if total < cap_bytes:
        base.write_text("\n".join(payloads) + ("\n" if payloads else ""))
        return [base]
    parts: list[Path] = []
    buf: list[str] = []
    size = 0
    idx = 0

    def _flush():
        nonlocal buf, size, idx
        p = base.with_name(f"{base.stem}.shard{idx:03d}{base.suffix}")
        p.write_text("\n".join(buf) + "\n")
        parts.append(p)
        idx += 1
        buf, size = [], 0

    for p, nbytes in zip(payloads, sizes, strict=True):
        if size + nbytes + 1 > cap_bytes and buf:
            _flush()
        buf.append(p)
        size += nbytes + 1
    if buf:
        _flush()
    return parts


def _upload_leaf(args, local_dir: Path, leaf: str, *, resume_skip: bool) -> None:
    """Production HF upload of one artifact leaf with exact-set verify
    (VERBATIM #2552 _upload_leaf; skip-loud under --skip-upload / non-production)."""
    if args.skip_upload or not _production(args):
        logger.warning("[upload] skip_upload/non-production: %s upload SKIPPED (loud)", leaf)
        return
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    prefix = f"{args.hf_prefix}/{leaf}"
    res = upload_dir_sharded(
        local_dir,
        C.HF_DATA_REPO,
        prefix,
        repo_type="dataset",
        shard_glob="*",
        verify=True,
        delete_local=False,
        resume_skip=resume_skip,
    )
    if not res.rerouted:
        expected = [
            f"{prefix}/{p.relative_to(local_dir)}"
            for p in sorted(local_dir.rglob("*"))
            if p.is_file()
        ]
        missing = hub.verify_repo_paths_uploaded(
            HfApi(), C.HF_DATA_REPO, expected, path_in_repo=prefix
        )
        assert not missing, f"[upload] verify FAILED — missing on Hub under {prefix}: {missing}"
    logger.info("[upload] %s -> %s (rerouted=%s)", local_dir.name, prefix, res.rerouted)


def _stage_upload_files(args, files: list[Path], leaf: str, *, resume_skip: bool) -> None:
    """Stage an explicit FILE list into a scratch dir and upload it as one leaf."""
    if args.skip_upload or not _production(args):
        logger.warning("[upload] skip_upload/non-production: %s upload SKIPPED (loud)", leaf)
        return
    stage = T._stage_dir(args) / f"upload_{leaf.replace('/', '_')}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    for f in files:
        assert f.exists(), f"[upload] staged file missing: {f}"
        shutil.copy2(f, stage / f.name)
    _upload_leaf(args, stage, leaf, resume_skip=resume_skip)


# ── raw-chunk iteration with persistent cache + ci index (VERBATIM #2552) ─────────


def _chunk_index_path(args) -> Path:
    return T._stage_dir(args) / "chunk_ci_index.json"


def _raw_cache_dir(args) -> Path:
    d = T._stage_dir(args) / "raw_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _iter_rows_pinned(args, needed_ci: dict[int, int], *, tag: str):
    """Yield (row_idx, ci, prompt, response) for every needed ci at the LINEAGE
    pin (VERBATIM #2552 _iter_rows_pinned). Text is never logged."""
    dns = SimpleNamespace(max_chunks=args.max_chunks, scratch=T._stage_dir(args))
    names = EA._raw_chunk_names(dns)
    idx_path = _chunk_index_path(args)
    index: dict[str, list[int]] = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    have_index = set(index) >= set(names)
    cache = _raw_cache_dir(args)
    n_found = 0
    t0 = time.time()
    for j, name in enumerate(names):
        if have_index and not (set(index.get(name, ())) & set(needed_ci)):
            continue
        local = cache / name
        if not local.exists():
            got = Path(
                N1M._download_chunk_with_retry(
                    C.HF_DATA_REPO,
                    f"{EA.RAW_PREFIX}/{name}",
                    cache,
                    revision=LINEAGE_REVISION,
                )
            )
            if got != local and got.exists():
                shutil.move(str(got), str(local))
        rows = json.loads(local.read_text())["rows"]
        if name not in index:
            index[name] = sorted(int(r["ci"]) for r in rows)
        for r in rows:
            ci = int(r["ci"])
            if ci in needed_ci:
                n_found += 1
                yield needed_ci[ci], ci, r["prompt"], r["response"]
        if (j + 1) % 100 == 0:
            print(
                f"[{tag}] chunk sweep {j + 1}/{len(names)} found={n_found} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    C.write_json_atomic(idx_path, index)


# ── eval rows: the SAME 2,000 holdout rows #2552 judged (pinned artifact) ─────────


def _eval_ids(args) -> np.ndarray:
    """Row ids of the 2,000 judged eval turns, read from the PINNED #2552
    eval_lists artifact (task-body reuse row: the two sides stay comparable).
    Cached under the stage dir; production asserts n == 2,000."""
    cache = T._stage_dir(args) / "eval_ids_2552.json"
    if cache.exists():
        ids = np.asarray(json.loads(cache.read_text())["eval_ids"], np.int64)
    else:
        stage = T._stage_dir(args) / "i2552_eval_lists"
        idx = json.loads(
            _hf_fetch(
                f"{EVAL_LISTS_PREFIX}/feature_lists_2000turns.json", stage, I2552_REVISION
            ).read_text()
        )
        cfg = idx["configs"]["rep_ta"]
        rows: list[int] = []
        for fname in cfg["files"]:
            fp = _hf_fetch(f"{EVAL_LISTS_PREFIX}/{fname}", stage, I2552_REVISION)
            with fp.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rows.append(int(json.loads(line)["row_id"]))
        assert len(rows) == int(cfg["n_turns"]), (len(rows), cfg["n_turns"])
        ids = np.sort(np.asarray(sorted(set(rows)), np.int64))
        assert len(ids) == len(rows), "duplicate row_id in the #2552 eval lists"
        C.write_json_atomic(
            cache, {"eval_ids": [int(x) for x in ids], "eval_ids_sha256": _sha_ids(ids)}
        )
    if _production(args):
        assert len(ids) == EVAL_TURNS_N, f"pinned eval lists hold {len(ids)} != {EVAL_TURNS_N}"
    return ids


def _eval_ids_in_slice(args) -> tuple[np.ndarray, str]:
    """Eval ids restricted to the assembled slice. Production: identity (all
    2,000 present, asserted downstream). Smoke: intersect with rows_present; an
    empty intersection falls back to the first holdout-present rows (LOUD) so
    the phase still exercises its real path."""
    ids = _eval_ids(args)
    if _production(args):
        return ids, "pinned-2552"
    rows_present = np.load(T._assemble_dir(args) / "rows_present.npy")
    keep = ids[np.isin(ids, rows_present)]
    n_want = max(2, int(args.n_eval_turns) or 2)
    if len(keep) >= 2:
        return keep[:n_want], "pinned-2552-slice"
    _row_ci, _prov, pools = T._load_scratch_meta(args)
    hold = np.asarray(pools["holdout"], np.int64)
    fb = hold[np.isin(hold, rows_present)][:n_want]
    assert len(fb) >= 2, f"smoke slice holds {len(fb)} holdout rows (<2)"
    logger.warning("[eval_ids] smoke: 0 pinned eval rows in slice; holdout fallback n=%d", len(fb))
    return fb, "smoke-holdout-fallback"


# ── P1: assemble (delegated verbatim) ─────────────────────────────────────────────


def phase_assemble(args) -> None:
    T.phase_assemble(args)


# ── P2: flat context-SAE training (Der recipe; #2552 sae_train shape) ────────────


def _build_ctx_sae(args, device: str) -> "T.MatryoshkaBatchTopKSAE":
    """Flat 1-tier construction, DIRECT — never through _sae_tier_bounds (the
    #2552 plan §4 P1.2 trap). Smoke narrows width/k via --ctx-width/--ctx-k."""
    width, k = _ctx_width(args), _ctx_k(args)
    sae = T.MatryoshkaBatchTopKSAE(
        act_dim=int(C.EXPECTED_HIDDEN),
        dict_size=width,
        k=k,
        tier_bounds=(width,),
        seed=CTX_SEED,
    ).to(device)
    assert sae.tier_bounds == (width,), (
        f"flat ctx SAE must be 1-tier, got {sae.tier_bounds} (#2552 plan §4 P1.2 trap)"
    )
    return sae


def _load_ctx_sae(args) -> "T.MatryoshkaBatchTopKSAE":
    sae = T.MatryoshkaBatchTopKSAE.load_local(_sae_ctx_dir(args), device=args.device)
    assert sae.tier_bounds == (sae.dict_size,), (
        f"loaded ctx SAE is not flat 1-tier: {sae.tier_bounds}"
    )
    return sae


def _load_answer_sae(args) -> "T.MatryoshkaBatchTopKSAE":
    """The REUSED #2552 flat answer SAE @ the pinned revision. Smoke substitutes
    a deterministic tiny stand-in (SMOKE STAND-IN — the 940 MB banked fetch is a
    production-path leg; recorded in deviations.md)."""
    if not _production(args):
        width, k = _ctx_width(args), _ctx_k(args)
        logger.warning(
            "[answer_sae] SMOKE STAND-IN: tiny flat SAE (width=%d k=%d seed=%d) substitutes "
            "the banked #2552 answer SAE — production fetches %s @ %s",
            width,
            k,
            CTX_SEED + 7,
            ANSWER_SAE_PREFIX,
            I2552_REVISION[:12],
        )
        sae = T.MatryoshkaBatchTopKSAE(
            act_dim=int(C.EXPECTED_HIDDEN),
            dict_size=width,
            k=k,
            tier_bounds=(width,),
            seed=CTX_SEED + 7,
        ).to(args.device)
        with torch.no_grad():
            sae.threshold.fill_(0.05)  # nonzero gate so the smoke census has dead features
        return sae.eval()
    stage = T._stage_dir(args) / "banked_answer_sae"
    for name in ("cfg.json", "sae_weights.safetensors"):
        _hf_fetch(f"{ANSWER_SAE_PREFIX}/{name}", stage, I2552_REVISION)
    d = stage / ANSWER_SAE_PREFIX
    sae = T.MatryoshkaBatchTopKSAE.load_local(d, device=args.device)
    assert sae.dict_size == CTX_DICT and sae.k == CTX_K and sae.tier_bounds == (CTX_DICT,), (
        sae.dict_size,
        sae.k,
        sae.tier_bounds,
    )
    return sae


def phase_sae_train_ctx(args) -> None:
    """Flat ctx SAE on X19 train rows — the #2552 phase_sae_train loop shape
    VERBATIM, retargeted X19 + seed 2661 + the task-body 20k-holdout FVE halt."""
    C.phase("sae_train_ctx")
    out = _sae_ctx_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    w_path = out / "sae_weights.safetensors"
    log_path = out / "train_log.json"
    gates_path = out / "gates_g1.json"
    regime, resume_ok = T._enter_phase_regime(
        out,
        args,
        "sae_train_ctx",
        stale_paths=[w_path, log_path, gates_path, out / "cfg.json", out / "ckpt_last.pt"],
    )
    if resume_ok and w_path.exists() and log_path.exists() and gates_path.exists():
        gates = json.loads(gates_path.read_text())
        if _production(args) and gates["g1"]["verdict"] == "FAIL":
            logger.error("[sae_train_ctx] resume: recorded G1 FAIL re-applied")
            sys.exit(RC_G1)
        _stage_upload_files(
            args,
            [w_path, out / "cfg.json", log_path, gates_path],
            "analysis_tensors/sae_ctx",
            resume_skip=True,
        )
        logger.info("[sae_train_ctx] resume: weights+log+gates present; skip")
        return
    EA._headroom(args.out_root, 1 if args.smoke else 4, "sae-train-ctx")
    a_dir = T._assemble_dir(args)
    assert (a_dir / "split_meta.json").exists(), "sae_train_ctx needs assemble outputs"
    x_mm = np.load(a_dir / "X19.fp16.npy", mmap_mode="r")
    tr_pos, val_pos, pool_doc = T._sae_row_positions(args)
    print(f"[sae_train_ctx] pools re-measured: {json.dumps(pool_doc)}", flush=True)
    dev = args.device
    model = _build_ctx_sae(args, dev)
    # b_dec init: seeded train-subsample mean (#2552 convention, ctx seed)
    rng0 = np.random.default_rng(CTX_SEED + 1)
    sub = np.sort(rng0.choice(tr_pos, size=min(65_536, len(tr_pos)), replace=False))
    mu = np.zeros(model.act_dim, dtype=np.float64)
    for s in range(0, len(sub), 8192):
        mu += np.asarray(x_mm[sub[s : s + 8192]], np.float64).sum(0)
    with torch.no_grad():
        model.b_dec.copy_(torch.as_tensor(mu / len(sub), dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=T.SAE_LR, betas=T.SAE_ADAM_BETAS)
    ckpt_path = out / "ckpt_last.pt"
    start_epoch, step = 0, 0
    epoch_rows: list[dict] = []
    steps_cap = int(args.sae_steps)
    if resume_ok and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, step = int(ck["epoch_done"]), int(ck["step"])
        epoch_rows = list(ck["log_rows"])
        if bool(ck.get("steps_capped")) and steps_cap and step >= steps_cap:
            start_epoch = T.SAE_EPOCHS
        logger.info("[sae_train_ctx] RESUMED at epoch %d (step %d)", start_epoch, step)
    t0 = time.time()
    stop = False
    for epoch in range(start_epoch, T.SAE_EPOCHS):
        rng_e = np.random.default_rng(CTX_SEED * 1000 + epoch)
        run_loss, run_n = 0.0, 0
        diags: dict = {"l0_train": float("nan")}
        for xb in T._block_batches(x_mm, tr_pos, T.SAE_BATCH, rng_e):
            x = torch.as_tensor(np.asarray(xb, np.float32), device=dev)
            loss, diags, _fired = model.train_step_losses(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += diags["loss"]
            run_n += 1
            step += 1
            if step % 200 == 0:
                print(
                    f"[sae_train_ctx] epoch {epoch + 1}/{T.SAE_EPOCHS} step {step} "
                    f"loss={run_loss / max(1, run_n):.1f} thr={float(model.threshold):.4f} "
                    f"l0={diags['l0_train']:.0f} elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
            if steps_cap and step >= steps_cap:
                stop = True
                break
        fve_val, l0_val = T._recon_fve(model, x_mm, val_pos)
        row = {
            "epoch": epoch + 1,
            "steps": step,
            "mean_loss": round(run_loss / max(1, run_n), 3),
            "saeval_var_fve": round(fve_val, 6),
            "saeval_nmse_centered": round(1.0 - fve_val, 6),
            "saeval_l0": round(l0_val, 2),
            "threshold": float(model.threshold),
            "elapsed_s": round(time.time() - t0, 1),
        }
        epoch_rows.append(row)
        print(
            f"[sae_train_ctx] unit {epoch + 1}/{T.SAE_EPOCHS} epoch-done {json.dumps(row)}",
            flush=True,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch_done": epoch if stop else epoch + 1,
                "steps_capped": bool(stop),
                "step": step,
                "log_rows": epoch_rows,
            },
            ckpt_path,
        )
        if stop:
            break
    assert epoch_rows, "sae_train_ctx produced no epoch rows"
    # G1: the task-body halt floor binds on the 20k-HOLDOUT variance-FVE (the
    # SAE-val FVE above is logged for #2476/#2552 parity, not gated)
    _row_ci, _prov, pools = T._load_scratch_meta(args)
    hold_pos = _positions_of(args, pools["holdout"], "sae_train_ctx/holdout")
    fve_hold, l0_hold = T._recon_fve(model, x_mm, hold_pos)
    g1_pass = fve_hold >= G1_FVE_FLOOR
    gates = {
        "g1": {
            "holdout_var_fve": float(fve_hold),
            "holdout_nmse_centered": float(1.0 - fve_hold),
            "holdout_l0": float(l0_hold),
            "saeval_var_fve": float(epoch_rows[-1]["saeval_var_fve"]),
            "floor": G1_FVE_FLOOR,
            "floor_basis": "20k-holdout variance-FVE (task-body decision record)",
            "n_holdout": int(len(hold_pos)),
            "verdict": "PASS"
            if g1_pass
            else ("FAIL" if _production(args) else "INFORMATIONAL-smoke"),
        }
    }
    model.save_dir(out)
    T._write_json(
        log_path,
        {
            "pools": pool_doc,
            "epochs": epoch_rows,
            "steps": step,
            "steps_cap": steps_cap,
            "cfg": model.cfg_dict(),
            "init_seed": CTX_SEED,
            "input": "X19 (last-prompt-token context state, layer 19)",
        },
        phase="sae_train_ctx",
    )
    T._write_json(gates_path, gates, phase="sae_train_ctx")
    _measured_update(
        args, ctx_sae_holdout_var_fve=float(fve_hold), ctx_sae_holdout_l0=float(l0_hold)
    )
    if _production(args) and not g1_pass:
        T._sentinel("sae_train_ctx", "G1 FVE below floor (gates_g1.json written)", {"rc": RC_G1})
        logger.error("[sae_train_ctx] G1 FAIL: %s", gates["g1"])
        sys.exit(RC_G1)
    _stage_upload_files(
        args,
        [w_path, out / "cfg.json", log_path, gates_path],
        "analysis_tensors/sae_ctx",
        resume_skip=False,
    )
    if ckpt_path.exists():
        ckpt_path.unlink()  # optimizer state is a discard artifact (#2552 convention)
    T._sentinel("sae_train_ctx", f"done (holdout fve={fve_hold:.4f} l0={l0_hold:.1f})")


# ── P3: Der metrics for BOTH SAEs ────────────────────────────────────────────────


@torch.no_grad()
def _der_metrics_pass(sae, mm, positions: np.ndarray, chunk: int = 2048) -> dict:
    """One streaming pass: Der's RAW nMSE (E||x-xhat||^2 / E||x||^2 — the paper
    metric), the mean-centered variant (1 - variance-FVE), variance-FVE, and
    realized L0 under inference threshold gating. fp64 accumulators (the
    T._recon_fve kernel extended with the raw second moment)."""
    pos = np.asarray(positions, np.int64)
    n = int(len(pos))
    assert n >= 2, f"metrics need >= 2 rows, got {n}"
    x_sum = torch.zeros(sae.act_dim, dtype=torch.float64, device=sae.device)
    x_sq = torch.zeros_like(x_sum)
    r_sq = torch.zeros_like(x_sum)
    r_sum = torch.zeros_like(x_sum)
    l0 = 0.0
    for s in range(0, n, chunk):
        x = torch.as_tensor(np.asarray(mm[pos[s : s + chunk]], np.float32), device=sae.device)
        f = sae.encode(x, chunk=chunk)
        r = x - sae.decode(f)
        x_sum += x.sum(0, dtype=torch.float64)
        x_sq += (x * x).sum(0, dtype=torch.float64)
        r_sum += r.sum(0, dtype=torch.float64)
        r_sq += (r * r).sum(0, dtype=torch.float64)
        l0 += float((f > 0).sum())
    e_x2 = float(x_sq.sum()) / n
    e_r2 = float(r_sq.sum()) / n
    nmse_raw = e_r2 / max(e_x2, 1e-30)

    def _var_sum(ssum, ssq):
        return float(((ssq - ssum * ssum / n) / (n - 1)).sum())

    ss_tot = _var_sum(x_sum, x_sq)
    fve = float("nan") if ss_tot < 1e-12 else 1.0 - _var_sum(r_sum, r_sq) / ss_tot
    nmse_centered = float("nan") if ss_tot < 1e-12 else _var_sum(r_sum, r_sq) / ss_tot
    return {
        "n_rows": n,
        "nmse_raw": nmse_raw,
        "nmse_raw_is_ders_metric": True,
        "nmse_mean_centered": nmse_centered,
        "variance_fve": fve,
        "realized_l0": l0 / n,
    }


@torch.no_grad()
def _firing_census(sae, mm, positions: np.ndarray, chunk: int = 4096) -> dict:
    """Streaming per-feature firing counts + activation sum (when active) over
    the given rows (the #2552 _census_pass kernel, corpus split dropped)."""
    W = sae.dict_size
    counts = torch.zeros(W, dtype=torch.int64, device=sae.device)
    a_sum = torch.zeros(W, dtype=torch.float64, device=sae.device)
    pos = np.asarray(positions, np.int64)
    for s in range(0, len(pos), chunk):
        x = torch.as_tensor(np.asarray(mm[pos[s : s + chunk]], np.float32), device=sae.device)
        f = sae.encode(x, chunk=chunk)
        counts += (f > 0).sum(0)
        a_sum += f.sum(0, dtype=torch.float64)
    return {
        "counts": counts.cpu().numpy(),
        "act_sum": a_sum.cpu().numpy(),
        "n_rows": int(len(pos)),
    }


def _log_binned_hist(counts: np.ndarray, n_rows: int, n_bins: int = 30) -> dict:
    """Log-binned firing-fraction histogram (dead features counted separately)."""
    frac = counts.astype(np.float64) / max(1, n_rows)
    alive = frac[frac > 0]
    n_dead = int((frac == 0).sum())
    if len(alive) == 0:
        return {"n_dead": n_dead, "bin_edges": [], "bin_counts": []}
    lo = math.floor(math.log10(float(alive.min())))
    edges = np.logspace(lo, 0, n_bins + 1)
    hist, _ = np.histogram(alive, bins=edges)
    return {
        "n_dead": n_dead,
        "bin_edges": [float(e) for e in edges],
        "bin_counts": [int(h) for h in hist],
    }


def phase_sae_metrics(args) -> None:
    """Der's simple metrics for BOTH dictionaries on the 20k holdout + 10k
    SAE-val rows; dead census on the 120k fit rows and the holdout."""
    C.phase("sae_metrics")
    out = _metrics_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    finals = [out / "sae_metrics_ctx.json", out / "sae_metrics_answer.json"]
    regime, resume_ok = T._enter_phase_regime(
        out,
        args,
        "sae_metrics",
        stale_paths=[*finals, out / "perfeature_census_ctx.npz", out / "perfeature_census_ans.npz"],
    )
    if resume_ok and all(p.exists() for p in finals):
        logger.info("[sae_metrics] resume: outputs present; skip")
        return
    a_dir = T._assemble_dir(args)
    x_mm = np.load(a_dir / "X19.fp16.npy", mmap_mode="r")
    y_mm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")
    _row_ci, _prov, pools = T._load_scratch_meta(args)
    hold_pos = _positions_of(args, pools["holdout"], "sae_metrics/holdout")
    fit_pos = _positions_of(args, pools["sae_fit"], "sae_metrics/fit")
    _tr_pos, saeval_pos, _pool_doc = T._sae_row_positions(args)
    if not _production(args):
        hold_pos, fit_pos, saeval_pos = hold_pos[:512], fit_pos[:1024], saeval_pos[:256]
        for name, p in (("holdout", hold_pos), ("fit", fit_pos), ("sae_val", saeval_pos)):
            assert len(p) >= 2, f"smoke slice holds {len(p)} {name} rows (<2)"
    for side, sae, mm, tag in (
        ("ctx", _load_ctx_sae(args), x_mm, "X19 last-prompt-token context state"),
        ("answer", _load_answer_sae(args), y_mm, "Y19 whole-answer mean state"),
    ):
        rows = {
            "holdout": _der_metrics_pass(sae, mm, hold_pos),
            "sae_val": _der_metrics_pass(sae, mm, saeval_pos),
        }
        cen_fit = _firing_census(sae, mm, fit_pos)
        cen_hold = _firing_census(sae, mm, hold_pos)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_when_active = np.where(
                cen_fit["counts"] > 0, cen_fit["act_sum"] / np.maximum(cen_fit["counts"], 1), 0.0
            )
        npz_path = out / f"perfeature_census_{'ctx' if side == 'ctx' else 'ans'}.npz"
        savez_atomic(
            npz_path,
            counts_fit=cen_fit["counts"],
            counts_holdout=cen_hold["counts"],
            act_sum_fit=cen_fit["act_sum"],
            mean_when_active_fit=mean_when_active.astype(np.float32),
            n_fit_rows=np.int64(cen_fit["n_rows"]),
            n_holdout_rows=np.int64(cen_hold["n_rows"]),
        )
        doc = {
            "side": side,
            "input": tag,
            "dict_size": int(sae.dict_size),
            "k": int(sae.k),
            "tier_bounds": list(sae.tier_bounds),
            "threshold": float(sae.threshold),
            "splits": rows,
            "dead_features": {
                "n_dead_on_fit_rows": int((cen_fit["counts"] == 0).sum()),
                "n_dead_on_holdout": int((cen_hold["counts"] == 0).sum()),
                "n_fit_rows": cen_fit["n_rows"],
                "n_holdout_rows": cen_hold["n_rows"],
                "definition": "never fires (inference threshold gating)",
            },
            "firing_fraction_hist_fit": _log_binned_hist(cen_fit["counts"], cen_fit["n_rows"]),
            "reference_numbers": {
                "der_paper_nmse": DER_REFERENCE_NMSE,
                "i2552_answer_sae_holdout_nmse": I2552_ANSWER_NMSE_HOLDOUT,
                "note": "Der et al. arXiv 2606.28548 (turn-averaged 0.097, per-token 0.162)",
            },
            "sae_provenance": (
                "fresh #2661 ctx SAE (seed 2661)"
                if side == "ctx"
                else f"REUSED #2552 sae_rep @ {I2552_REVISION[:12]}"
                + (" [SMOKE STAND-IN]" if not _production(args) else "")
            ),
            **_meta(args, phase="sae_metrics"),
        }
        T._write_json(out / f"sae_metrics_{side}.json", doc, phase="sae_metrics")
        print(
            f"[sae_metrics] unit {side} done "
            + json.dumps({k: round(v, 5) for k, v in rows["holdout"].items() if k != "n_rows"}),
            flush=True,
        )
        del sae
        if args.device == "cuda":
            torch.cuda.empty_cache()
    _stage_upload_files(
        args,
        sorted(out.glob("sae_metrics_*.json")) + sorted(out.glob("perfeature_census_*.npz")),
        "analysis_tensors/sae_metrics",
        resume_skip=False,
    )
    T._sentinel("sae_metrics", "Der metrics done (both SAEs)")


def _meta(args, *, phase: str) -> dict:
    """Reproducibility metadata block (git commit + timestamp + pins) for every
    result JSON this driver emits (CLAUDE.md Reproducibility Requirements)."""
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return {
        "metadata": {
            **as_metadata_dict(git_provenance(), phase=phase),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pins": {
                "capture_chunks_revision": LINEAGE_REVISION,
                "issue2552_revision": I2552_REVISION,
            },
        }
    }


# ── P4: encode_full — ALL features, both sides, CSR storage ───────────────────────


@torch.no_grad()
def _encode_csr(sae, mm, positions: np.ndarray, chunk: int = 4096, tag: str = "") -> sp.csr_matrix:
    """Encode the given rows through the sae's inference gate -> scipy CSR
    (fp32 data). Row order follows `positions` verbatim (the _encode_restricted
    contract); only nonzeros are stored (k=128 -> ~100 MB per matrix)."""
    pos = np.asarray(positions, np.int64)
    rows_l: list[np.ndarray] = []
    cols_l: list[np.ndarray] = []
    vals_l: list[np.ndarray] = []
    t0 = time.time()
    n_chunks = math.ceil(len(pos) / chunk)
    for i, s in enumerate(range(0, len(pos), chunk)):
        x = torch.as_tensor(np.asarray(mm[pos[s : s + chunk]], np.float32), device=sae.device)
        f = sae.encode(x, chunk=chunk)
        nz = torch.nonzero(f > 0, as_tuple=False)
        rows_l.append((nz[:, 0].cpu().numpy() + s).astype(np.int64))
        cols_l.append(nz[:, 1].cpu().numpy().astype(np.int32))
        vals_l.append(f[nz[:, 0], nz[:, 1]].cpu().numpy().astype(np.float32))
        if (i + 1) % 10 == 0 or i + 1 == n_chunks:
            print(
                f"[encode{tag}] chunk {i + 1}/{n_chunks} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    rows = np.concatenate(rows_l) if rows_l else np.empty(0, np.int64)
    cols = np.concatenate(cols_l) if cols_l else np.empty(0, np.int32)
    vals = np.concatenate(vals_l) if vals_l else np.empty(0, np.float32)
    return sp.csr_matrix(
        (vals, (rows, cols.astype(np.int64))), shape=(len(pos), sae.dict_size), dtype=np.float32
    )


def _save_csr(path: Path, X: sp.csr_matrix) -> None:
    savez_atomic(
        path,
        data=X.data.astype(np.float32),
        indices=X.indices.astype(np.int32),
        indptr=X.indptr.astype(np.int64),
        shape=np.asarray(X.shape, np.int64),
    )


def _load_csr(path: Path) -> sp.csr_matrix:
    z = np.load(path)
    return sp.csr_matrix(
        (z["data"], z["indices"].astype(np.int64), z["indptr"]), shape=tuple(z["shape"])
    )


def _csr_colsum64(X: sp.csr_matrix, *, power: int = 1) -> np.ndarray:
    """EXACT fp64 per-column sum of a CSR (bincount over indices with fp64
    weights). scipy's own ``sum``/``mean`` accumulate at the MATRIX dtype —
    float32 here — which broke the pod-side ib parity assert at a measured
    max|delta| of 6.6e-7 against the fp64 canonical helper (r2 pod smoke fix;
    every parity/moment read now accumulates fp64 on CPU)."""
    w = X.data.astype(np.float64)
    if power == 2:
        w = w * w
    return np.bincount(X.indices, weights=w, minlength=X.shape[1]).astype(np.float64)


def _col_moments_csr(X: sp.csr_matrix, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fp64 per-column (mean, variance) over the given row subset (population
    var; EXACT fp64 accumulation via _csr_colsum64 — r2 pod smoke fix)."""
    sub = X[np.asarray(rows, np.int64)]
    n = max(1, sub.shape[0])
    s1 = _csr_colsum64(sub)
    s2 = _csr_colsum64(sub, power=2)
    mu = s1 / n
    var = np.maximum(s2 / n - mu * mu, 0.0)
    return mu, var


def phase_encode_full(args) -> None:
    """Encode ALL features per side on the 120k fit + 400 val + 20k holdout rows
    (CSR + row ids); report zero-variance-on-fit columns per side."""
    C.phase("encode_full")
    out = _encodes_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    finals = [
        out / "ctx_full_csr.npz",
        out / "ans_full_csr.npz",
        out / "rows.npz",
        out / "zero_variance.json",
    ]
    regime, resume_ok = T._enter_phase_regime(out, args, "encode_full", stale_paths=finals)
    if resume_ok and all(p.exists() for p in finals):
        logger.info("[encode_full] resume: outputs present; skip")
        return
    EA._headroom(args.out_root, 1 if args.smoke else 6, "encode-full")
    a_dir = T._assemble_dir(args)
    x_mm = np.load(a_dir / "X19.fp16.npy", mmap_mode="r")
    y_mm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")
    rows_present = np.load(a_dir / "rows_present.npy")
    _row_ci, _prov, pools = T._load_scratch_meta(args)
    _r1, val_ids, _test = T._assert_pinned_valtest(T._committed_split())
    fit_pos = _positions_of(args, pools["sae_fit"], "encode_full/fit")
    val_pos = _positions_of(args, val_ids, "encode_full/val")
    te_pos = _positions_of(args, pools["holdout"], "encode_full/te")
    if _production(args):
        assert (len(fit_pos), len(val_pos), len(te_pos)) == (120_000, 400, 20_000), (
            len(fit_pos),
            len(val_pos),
            len(te_pos),
        )
    else:  # smoke caps (deterministic heads) + nonzero-yield floors (#2569 shape)
        fit_pos, val_pos, te_pos = fit_pos[:2000], val_pos[:64], te_pos[:256]
        assert len(fit_pos) >= 32 and len(val_pos) >= 4 and len(te_pos) >= 8, (
            len(fit_pos),
            len(val_pos),
            len(te_pos),
        )
    rows_cat = np.concatenate([fit_pos, val_pos, te_pos])
    zv: dict = {}
    for side, loader, mm in (("ctx", _load_ctx_sae, x_mm), ("ans", _load_answer_sae, y_mm)):
        sae = loader(args)
        Xc = _encode_csr(sae, mm, rows_cat, tag=f"/{side}")
        _save_csr(out / f"{side}_full_csr.npz", Xc)
        mu, var = _col_moments_csr(Xc, np.arange(len(fit_pos)))
        zero_cols = np.flatnonzero(var <= ZERO_VAR_EPS)
        savez_atomic(out / f"{side}_fit_moments.npz", mu=mu, var=var)
        zv[side] = {
            "dict_size": int(sae.dict_size),
            "n_zero_variance_on_fit": int(len(zero_cols)),
            "zero_variance_ids": [int(i) for i in zero_cols],
            "note": "dropped mechanically from the ridge design; NO alive floor",
        }
        print(f"[encode_full] unit {side}: nnz={Xc.nnz} zero_var={len(zero_cols)}", flush=True)
        del sae, Xc
        if args.device == "cuda":
            torch.cuda.empty_cache()
    savez_atomic(
        out / "rows.npz",
        fit_ids=rows_present[fit_pos],
        val_ids=rows_present[val_pos],
        te_ids=rows_present[te_pos],
        fit_pos=fit_pos,
        val_pos=val_pos,
        te_pos=te_pos,
    )
    T._write_json(
        out / "zero_variance.json", {**zv, **_meta(args, phase="encode_full")}, phase="encode_full"
    )
    _stage_upload_files(
        args,
        [*finals, out / "ctx_fit_moments.npz", out / "ans_fit_moments.npz"],
        "analysis_tensors/encodes",
        resume_skip=False,
    )
    T._sentinel("encode_full", f"done (rows={len(rows_cat)})")


# ── shared fp64 Gram/XtY accumulation off CSR (sparse-side GEMMs) ────────────────


def _torch_dev(args) -> torch.device:
    return torch.device(args.device)


def _sparse_xtd(Xc: sp.csr_matrix, dense: torch.Tensor, device) -> torch.Tensor:
    """Xc^T @ dense via a sparse COO left operand (flops ~ nnz * d_y, not
    n * d_x * d_y). Returns fp32 (d_x, d_y) on `device`."""
    coo = Xc.tocoo()
    idx = torch.stack(
        [torch.as_tensor(coo.col, dtype=torch.int64), torch.as_tensor(coo.row, dtype=torch.int64)]
    )
    xt = (
        torch.sparse_coo_tensor(
            idx,
            torch.as_tensor(coo.data, dtype=torch.float32),
            size=(Xc.shape[1], Xc.shape[0]),
        )
        .to(device)
        .coalesce()
    )
    return torch.sparse.mm(xt, dense)


def _accumulate_raw_products(
    X: sp.csr_matrix,
    Y: sp.csr_matrix | None,
    rows: np.ndarray,
    device,
    chunk: int = 8192,
    tag: str = "",
) -> dict:
    """One pass over `rows`: raw XtX (and XtY when Y given), column sums, in
    fp64 cross-chunk accumulators (fp32 sparse GEMM inside a chunk)."""
    rows = np.asarray(rows, np.int64)
    d_x = X.shape[1]
    xtx = torch.zeros((d_x, d_x), dtype=torch.float64, device=device)
    xty = None
    if Y is not None:
        xty = torch.zeros((d_x, Y.shape[1]), dtype=torch.float64, device=device)
    colsum_x = np.zeros(d_x, np.float64)
    ysum = np.zeros(Y.shape[1], np.float64) if Y is not None else None
    t0 = time.time()
    n_chunks = math.ceil(len(rows) / chunk)
    for i, s in enumerate(range(0, len(rows), chunk)):
        r = rows[s : s + chunk]
        Xc = X[r]
        xd = torch.as_tensor(Xc.toarray(), dtype=torch.float32, device=device)
        xtx += _sparse_xtd(Xc, xd, device).double()
        colsum_x += _csr_colsum64(Xc)
        if Y is not None:
            Yc = Y[r]
            yd = torch.as_tensor(Yc.toarray(), dtype=torch.float32, device=device)
            xty += _sparse_xtd(Xc, yd, device).double()
            ysum += _csr_colsum64(Yc)
            del yd
        del xd
        if (i + 1) % 4 == 0 or i + 1 == n_chunks:
            print(
                f"[gram{tag}] chunk {i + 1}/{n_chunks} elapsed={time.time() - t0:.0f}s", flush=True
            )
    return {"xtx_raw": xtx, "xty_raw": xty, "colsum_x": colsum_x, "ysum": ysum, "n": int(len(rows))}


def _standardized_gram(acc: dict, xmu: np.ndarray, xsd: np.ndarray, device) -> torch.Tensor:
    """Gs = D^-1 (XtX_raw - colsum xmu^T - xmu colsum^T + n xmu xmu^T) D^-1, fp64.
    Exact closed form; xmu/xsd may come from a DIFFERENT (full-train) row set
    than the accumulated rows (the #1482 split-half standardizer convention)."""
    mu = torch.as_tensor(xmu, dtype=torch.float64, device=device)
    cs = torch.as_tensor(acc["colsum_x"], dtype=torch.float64, device=device)
    sd = torch.as_tensor(xsd, dtype=torch.float64, device=device)
    g = acc["xtx_raw"] - torch.outer(cs, mu) - torch.outer(mu, cs) + acc["n"] * torch.outer(mu, mu)
    g /= torch.outer(sd, sd)
    return g


def _standardized_xty(
    acc: dict, xmu: np.ndarray, xsd: np.ndarray, ymu: np.ndarray, device
) -> torch.Tensor:
    """Xs^T (Y - ymu) = D^-1 (XtY_raw - colsum_x ymu^T - xmu (n ymu)^T
    + n xmu ymu^T) = D^-1 (XtY_raw - colsum_x ymu^T)  [X-centering term vanishes
    against Y-centering]. fp64."""
    del xmu  # the X-centering term vanishes against Y-centering (docstring)
    cs = torch.as_tensor(acc["colsum_x"], dtype=torch.float64, device=device)
    ym = torch.as_tensor(ymu, dtype=torch.float64, device=device)
    sd = torch.as_tensor(xsd, dtype=torch.float64, device=device)
    out = acc["xty_raw"] - torch.outer(cs, ym)
    out /= sd[:, None]
    return out


def _std_rows_dense(
    X: sp.csr_matrix, rows: np.ndarray, xmu: np.ndarray, xsd: np.ndarray, device
) -> torch.Tensor:
    """(X[rows] - xmu) / xsd as a dense fp32 tensor on `device`."""
    xd = torch.as_tensor(
        X[np.asarray(rows, np.int64)].toarray(), dtype=torch.float32, device=device
    )
    xd -= torch.as_tensor(xmu, dtype=torch.float32, device=device)
    xd /= torch.as_tensor(xsd, dtype=torch.float32, device=device)
    return xd


def _dense_rows(Y: sp.csr_matrix, rows: np.ndarray, device) -> torch.Tensor:
    return torch.as_tensor(
        Y[np.asarray(rows, np.int64)].toarray(), dtype=torch.float32, device=device
    )


def _pooled_r2(pred: np.ndarray, true: np.ndarray) -> float:
    """Whole-map pooled R^2 (fp64; SS_tot about the scored sample's own mean)."""
    p = np.asarray(pred, np.float64)
    t = np.asarray(true, np.float64)
    mu = t.mean(0)
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - mu) ** 2).sum())
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def _eigh_ridge_fit(
    args,
    X: sp.csr_matrix,
    Y: sp.csr_matrix,
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    live: np.ndarray,
    lambdas,
    *,
    tag: str,
    out_dir: Path,
    b_full_rows: int,
    col_block: int = 4096,
) -> dict:
    """Standardized-input ridge live-ctx-cols -> ALL Y columns via ONE fp64 Gram
    eigendecomposition + whole-map val-R^2 lambda selection (#2569 grid).
    Persists B (fp16, reindexed to `b_full_rows` rows), xmu/xsd/ymu/lambda, the
    val curve, and the holdout predictions (fp16). Returns the fit doc."""
    dev = _torch_dev(args)
    Xl = X[:, live].tocsr()
    d = int(len(live))
    d_y = int(Y.shape[1])
    xmu_l, xvar_l = _col_moments_csr(Xl, tr)
    xsd_l = np.sqrt(xvar_l)
    assert (xsd_l > 0).all(), "zero-variance column reached the ridge design (drop upstream)"
    ymu, _yvar = _col_moments_csr(Y, tr)
    acc = _accumulate_raw_products(Xl, Y, tr, dev, tag=f"/{tag}")
    gs = _standardized_gram(acc, xmu_l, xsd_l, dev)
    xty_c = _standardized_xty(acc, xmu_l, xsd_l, ymu, dev)
    del acc

    # fp64 eigendecomposition with the cuSOLVER CPU fallback (#2476 convention)
    def _eig(device_str, g64=gs):
        return torch.linalg.eigh(g64.to(torch.device(device_str)))

    s_eig, v = T._eigh_fallback(_eig, args.device)
    s_eig, v = s_eig.to(dev), v.to(dev)
    del gs
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    w = v.t() @ xty_c  # fp64 (v and xty_c are both fp64)
    del xty_c
    # lambda by whole-map val R^2 (400 rows; #2569 27-value grid; edge hit reported)
    xs_va = _std_rows_dense(Xl, va, xmu_l, xsd_l, dev).double()
    a_va = xs_va @ v
    y_va = Y[va].toarray().astype(np.float64)
    curve = []
    best = (float(lambdas[0]), -np.inf)
    for lam in lambdas:
        inv = 1.0 / (s_eig + float(lam))
        pv = ((a_va * inv) @ w).cpu().numpy() + ymu
        r2 = _pooled_r2(pv, y_va)
        curve.append({"lambda": float(lam), "val_pooled_r2": float(r2)})
        if np.isfinite(r2) and r2 > best[1]:
            best = (float(lam), float(r2))
    lam_star, val_r2 = best
    edge = bool(lam_star in (float(lambdas[0]), float(lambdas[-1])))
    if edge:
        logger.warning("[%s] selected lambda %.3g sits at the grid edge (reported)", tag, lam_star)
    inv = 1.0 / (s_eig + lam_star)
    # B + holdout predictions, column-blocked; B reindexed to b_full_rows rows
    b_path = out_dir / f"B_{tag}.fp16.npy"
    b_mm = np.lib.format.open_memmap(
        str(b_path), mode="w+", dtype=np.float16, shape=(b_full_rows, d_y)
    )
    b_mm[:] = 0
    xs_te = _std_rows_dense(Xl, te, xmu_l, xsd_l, dev)
    pred_te = torch.zeros((len(te), d_y), dtype=torch.float32, device=dev)
    v32 = v.float()
    del v
    for c0 in range(0, d_y, col_block):
        c1 = min(c0 + col_block, d_y)
        b_blk = v32 @ (inv.float()[:, None] * w[:, c0:c1].float())
        pred_te[:, c0:c1] = xs_te @ b_blk
        b_mm[live, c0:c1] = b_blk.cpu().numpy().astype(np.float16)
    b_mm.flush()
    pred = pred_te.cpu().numpy() + ymu[None, :].astype(np.float32)
    savez_atomic(
        out_dir / f"standardizer_{tag}.npz",
        live_cols=live,
        xmu_live=xmu_l,
        xsd_live=xsd_l,
        ymu=ymu,
        lambda_star=np.float64(lam_star),
        eigs=s_eig.cpu().numpy(),
    )
    np.save(out_dir / f"pred_te_{tag}.fp32.npy", pred.astype(np.float32))
    y_te = Y[te].toarray().astype(np.float64)
    doc = {
        "tag": tag,
        "d_live_inputs": d,
        "d_targets": d_y,
        "n_fit": int(len(tr)),
        "n_val": int(len(va)),
        "n_te": int(len(te)),
        "selected_lambda": lam_star,
        "val_pooled_r2_at_selected": val_r2,
        "lambda_grid": {"lo": float(lambdas[0]), "hi": float(lambdas[-1]), "n": len(lambdas)},
        "lambda_grid_edge_hit": edge,
        "val_curve": curve,
        "holdout_pooled_r2": _pooled_r2(pred, y_te),
        "gram_precision": "fp64 accumulation (fp32 sparse GEMM within row chunks); fp64 eigh",
        "b_dtype": "fp16 on disk, reindexed to full input width (zero rows = dropped columns)",
    }
    del pred_te, xs_te, w, v32
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return doc


def phase_map_ridge_full(args) -> None:
    """Ridge from all nonzero-variance ctx features to ALL 32,768 answer
    features (task headline map). Saves B — the #2569 leg-4 gap this closes."""
    C.phase("map_ridge_full")
    out = _ridge_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    finals = [
        out / "map_ridge_metrics.json",
        out / "B_ridge.fp16.npy",
        out / "standardizer_ridge.npz",
        out / "pred_te_ridge.fp32.npy",
    ]
    regime, resume_ok = T._enter_phase_regime(out, args, "map_ridge_full", stale_paths=finals)
    if resume_ok and all(p.exists() for p in finals):
        logger.info("[map_ridge_full] resume: outputs present; skip")
        return
    EA._headroom(args.out_root, 2 if args.smoke else 12, "map-ridge-full")
    enc = _encodes_dir(args)
    X = _load_csr(enc / "ctx_full_csr.npz")
    Y = _load_csr(enc / "ans_full_csr.npz")
    rows = np.load(enc / "rows.npz")
    n_fit, n_val, n_te = len(rows["fit_pos"]), len(rows["val_pos"]), len(rows["te_pos"])
    tr = np.arange(n_fit)
    va = n_fit + np.arange(n_val)
    te = n_fit + n_val + np.arange(n_te)
    var_ctx = np.load(enc / "ctx_fit_moments.npz")["var"]
    live = np.flatnonzero(var_ctx > ZERO_VAR_EPS)
    assert len(live) >= 2, f"only {len(live)} nonzero-variance ctx features"
    t0 = time.time()
    doc = _eigh_ridge_fit(
        args,
        X,
        Y,
        tr,
        va,
        te,
        live,
        LAMBDA_GRID_27,
        tag="ridge",
        out_dir=out,
        b_full_rows=int(X.shape[1]),
    )
    doc["wall_s"] = round(time.time() - t0, 1)
    doc["inputs"] = "all nonzero-variance context-SAE features (standardized; xmu/xsd)"
    doc["targets"] = "ALL answer-SAE features (intercept via ymu); NO alive floor"
    T._write_json(
        out / "map_ridge_metrics.json",
        {**doc, **_meta(args, phase="map_ridge_full")},
        phase="map_ridge_full",
    )
    _measured_update(
        args,
        ridge_lambda=doc["selected_lambda"],
        ridge_val_r2=doc["val_pooled_r2_at_selected"],
        ridge_holdout_pooled_r2=doc["holdout_pooled_r2"],
        ridge_d_live=doc["d_live_inputs"],
    )
    _stage_upload_files(args, finals, "analysis_tensors/map_ridge", resume_skip=False)
    T._sentinel(
        "map_ridge_full",
        f"done (lambda={doc['selected_lambda']:.3g} val_r2={doc['val_pooled_r2_at_selected']:.4f})",
    )


# ── P6: MLP companion map ────────────────────────────────────────────────────────


def phase_map_mlp(args) -> None:
    """2-layer MLP (hidden 4,096, GELU) on the SAME standardized inputs/targets
    as the ridge; early stop on val pooled R^2 (patience 3, max 30), seed 2661."""
    C.phase("map_mlp")
    out = _mlp_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    finals = [
        out / "map_mlp_metrics.json",
        out / "pred_te_mlp.fp32.npy",
        out / "mlp_ckpt.safetensors",
    ]
    regime, resume_ok = T._enter_phase_regime(out, args, "map_mlp", stale_paths=finals)
    if resume_ok and all(p.exists() for p in finals):
        logger.info("[map_mlp] resume: outputs present; skip")
        return
    EA._headroom(args.out_root, 1 if args.smoke else 6, "map-mlp")
    enc = _encodes_dir(args)
    X = _load_csr(enc / "ctx_full_csr.npz")
    Y = _load_csr(enc / "ans_full_csr.npz")
    rows = np.load(enc / "rows.npz")
    n_fit, n_val, n_te = len(rows["fit_pos"]), len(rows["val_pos"]), len(rows["te_pos"])
    tr = np.arange(n_fit)
    va = n_fit + np.arange(n_val)
    te = n_fit + n_val + np.arange(n_te)
    std = np.load(_ridge_dir(args) / "standardizer_ridge.npz")
    live = np.asarray(std["live_cols"], np.int64)
    xmu_l, xsd_l = np.asarray(std["xmu_live"]), np.asarray(std["xsd_live"])
    Xl = X[:, live].tocsr()
    d, d_y = int(len(live)), int(Y.shape[1])
    dev = _torch_dev(args)
    hidden = MLP_HIDDEN if _production(args) else min(MLP_HIDDEN, 128)
    torch.manual_seed(MLP_SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, d_y)
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    xs_va = _std_rows_dense(Xl, va, xmu_l, xsd_l, dev)
    y_va = Y[va].toarray().astype(np.float64)
    max_epochs = MLP_MAX_EPOCHS if _production(args) else 2

    @torch.no_grad()
    def _val_r2() -> float:
        model.eval()
        pv = model(xs_va).cpu().numpy().astype(np.float64)
        model.train()
        return _pooled_r2(pv, y_va)

    best = (-np.inf, None, 0)  # (val_r2, state, epoch)
    epochs_log: list[dict] = []
    t0 = time.time()
    for epoch in range(max_epochs):
        rng = np.random.default_rng(MLP_SEED * 1000 + epoch)
        perm = rng.permutation(len(tr))
        run_loss, run_n = 0.0, 0
        for s in range(0, len(perm), MLP_BATCH):
            idx = tr[perm[s : s + MLP_BATCH]]
            xb = _std_rows_dense(Xl, idx, xmu_l, xsd_l, dev)
            yb = _dense_rows(Y, idx, dev)
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += float(loss.detach())
            run_n += 1
        r2 = _val_r2()
        epochs_log.append(
            {
                "epoch": epoch + 1,
                "mean_mse": run_loss / max(1, run_n),
                "val_pooled_r2": r2,
                "elapsed_s": round(time.time() - t0, 1),
            }
        )
        print(
            f"[map_mlp] unit epoch {epoch + 1}/{max_epochs} {json.dumps(epochs_log[-1])}",
            flush=True,
        )
        if r2 > best[0]:
            best = (
                r2,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                epoch + 1,
            )
        elif epoch + 1 - best[2] >= MLP_PATIENCE:
            print(f"[map_mlp] early stop at epoch {epoch + 1} (best {best[2]})", flush=True)
            break
    assert best[1] is not None
    model.load_state_dict(best[1])
    model.eval()
    from safetensors.torch import save_file

    save_file({k: v for k, v in model.state_dict().items()}, str(out / "mlp_ckpt.safetensors"))
    preds = np.empty((n_te, d_y), np.float32)
    with torch.no_grad():
        for s in range(0, n_te, 2048):
            xb = _std_rows_dense(Xl, te[s : s + 2048], xmu_l, xsd_l, dev)
            preds[s : s + xb.shape[0]] = model(xb).cpu().numpy()
    np.save(out / "pred_te_mlp.fp32.npy", preds)
    y_te = Y[te].toarray().astype(np.float64)
    doc = {
        "hidden": hidden,
        "activation": "GELU",
        "optimizer": "Adam",
        "lr": MLP_LR,
        "batch": MLP_BATCH,
        "seed": MLP_SEED,
        "max_epochs": max_epochs,
        "patience": MLP_PATIENCE,
        "best_epoch": best[2],
        "best_val_pooled_r2": best[0],
        "holdout_pooled_r2": _pooled_r2(preds, y_te),
        "epochs": epochs_log,
        "inputs": "standardized live ctx features (ridge standardizer reused)",
        **_meta(args, phase="map_mlp"),
    }
    T._write_json(out / "map_mlp_metrics.json", doc, phase="map_mlp")
    _measured_update(args, mlp_best_val_r2=best[0], mlp_holdout_pooled_r2=doc["holdout_pooled_r2"])
    _stage_upload_files(args, finals, "analysis_tensors/map_mlp", resume_skip=False)
    T._sentinel("map_mlp", f"done (best epoch {best[2]} val_r2={best[0]:.4f})")


# ── P7: controls ─────────────────────────────────────────────────────────────────


def _fetch_banked_refit(args, te_ids: np.ndarray) -> np.ndarray:
    """The banked dense-map holdout predictions (composed zero-fit route input),
    row-aligned to te_ids. Production: pinned HF fetch + schema assert (#2552
    cached-artifact-schema). Smoke: a shape-faithful synthesized stand-in
    (SMOKE STAND-IN — recorded in deviations.md)."""
    if not _production(args):
        logger.warning(
            "[controls] SMOKE STAND-IN: synthesized banked refit npz (production fetches %s @ %s)",
            REFIT_HOLDOUT_PATH,
            LINEAGE_REVISION[:12],
        )
        rng = np.random.default_rng(2661)
        return rng.standard_normal((len(te_ids), int(C.EXPECTED_HIDDEN))).astype(np.float16)
    hz_path = _hf_fetch(REFIT_HOLDOUT_PATH, T._stage_dir(args) / "refit", LINEAGE_REVISION)
    hz = np.load(hz_path)
    assert {"holdout_pred16", "holdout_rows"} <= set(hz.files), (
        f"banked refit npz missing required keys — have {sorted(hz.files)}"
    )
    vhat = np.asarray(hz["holdout_pred16"], np.float16)
    hold_rows = np.asarray(hz["holdout_rows"], np.int64)
    assert vhat.ndim == 2 and len(vhat) == len(hold_rows), (vhat.shape, hold_rows.shape)
    order = np.argsort(hold_rows)
    pos = np.searchsorted(hold_rows[order], te_ids)
    ok = (pos < len(hold_rows)) & (hold_rows[order][np.minimum(pos, len(hold_rows) - 1)] == te_ids)
    assert bool(ok.all()), f"banked refit missing {int((~ok).sum())} holdout rows"
    return vhat[order[pos]]


def _shuffle_null_r2_blocked(
    pred, true_csc: sp.csc_matrix, seeds, *, col_block: int = 4096, tag: str = ""
) -> np.ndarray:
    """(n_draws, d_y) fp16 per-feature R^2 under row-shuffled predictions — the
    T._shuffle_null_r2 kernel (VERBATIM semantics), column-blocked so the fp64
    temporaries stay bounded at full width."""
    n, d_y = true_csc.shape
    out = np.zeros((len(seeds), d_y), np.float16)
    perms = [np.random.default_rng(s).permutation(n) for s in seeds]
    for c0 in range(0, d_y, col_block):
        c1 = min(c0 + col_block, d_y)
        t = true_csc[:, c0:c1].toarray().astype(np.float64)
        p = np.asarray(pred[:, c0:c1], np.float64)
        mu = t.mean(0)
        ss_tot = ((t - mu) ** 2).sum(0)
        for i, perm in enumerate(perms):
            ss_res = ((t - p[perm]) ** 2).sum(0)
            r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)
            out[i, c0:c1] = r2.astype(np.float16)
    print(f"[controls] shuffle-null{tag}: {len(seeds)} draws done", flush=True)
    return out


def phase_controls(args) -> None:
    """Composed zero-fit route, dense-input ridge, index-aligned identity+bias
    (labelled NULL), train-mean null, 20-draw row-shuffle nulls, kNN retrieval."""
    C.phase("controls")
    out = _controls_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    finals = [
        out / "controls.json",
        out / "pred_te_composed.fp16.npy",
        out / "pred_te_densein.fp32.npy",
        out / "pred_te_ib.fp16.npy",
        out / "shuffle_nulls.npz",
        out / "knn_retrieval.json",
    ]
    regime, resume_ok = T._enter_phase_regime(out, args, "controls", stale_paths=finals)
    if resume_ok and all(p.exists() for p in finals):
        logger.info("[controls] resume: outputs present; skip")
        return
    EA._headroom(args.out_root, 1 if args.smoke else 10, "controls")
    enc = _encodes_dir(args)
    X = _load_csr(enc / "ctx_full_csr.npz")
    Y = _load_csr(enc / "ans_full_csr.npz")
    rows = np.load(enc / "rows.npz")
    n_fit, n_val, n_te = len(rows["fit_pos"]), len(rows["val_pos"]), len(rows["te_pos"])
    tr = np.arange(n_fit)
    va = n_fit + np.arange(n_val)
    te = n_fit + n_val + np.arange(n_te)
    d_y = int(Y.shape[1])
    doc: dict = {"routes": {}}

    # composed zero-fit: banked dense prediction -> answer-SAE encode
    vhat = _fetch_banked_refit(args, np.asarray(rows["te_ids"], np.int64))
    sae_ans = _load_answer_sae(args)
    f_comp = T._encode_restricted(sae_ans, vhat, np.arange(len(vhat)), np.arange(d_y))
    np.save(out / "pred_te_composed.fp16.npy", np.asarray(f_comp, np.float16))
    doc["routes"]["composed"] = {
        "recipe": "banked #1482 dense ridge holdout predictions -> #2552 answer-SAE encode",
        "zero_fit": True,
        "banked": REFIT_HOLDOUT_PATH,
        "smoke_stand_in": not _production(args),
    }
    del sae_ans, vhat
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # dense-input ridge: 3,584-d context STATE -> answer features (same protocol)
    a_dir = T._assemble_dir(args)
    x_mm = np.load(a_dir / "X19.fp16.npy", mmap_mode="r")
    cat_pos = np.concatenate([rows["fit_pos"], rows["val_pos"], rows["te_pos"]])
    x_dense = np.asarray(x_mm[cat_pos], np.float32)
    Xd = sp.csr_matrix(x_dense)
    var_d = x_dense[tr].var(axis=0, dtype=np.float64)
    live_d = np.flatnonzero(var_d > ZERO_VAR_EPS)
    t0 = time.time()
    dd = _eigh_ridge_fit(
        args,
        Xd,
        Y,
        tr,
        va,
        te,
        live_d,
        LAMBDA_GRID_27,
        tag="densein",
        out_dir=out,
        b_full_rows=int(Xd.shape[1]),
    )
    dd["wall_s"] = round(time.time() - t0, 1)
    doc["routes"]["dense_input_ridge"] = dd
    del x_dense, Xd

    # index-aligned identity+bias (labelled NULL: indices unrelated across dicts)
    assert int(X.shape[1]) == d_y, (X.shape, d_y)
    xmu_full, _xvar = _col_moments_csr(X, tr)
    ymu_full, yvar_full = _col_moments_csr(Y, tr)
    bias = ymu_full - xmu_full
    pred_ib = X[te].toarray().astype(np.float32) + bias.astype(np.float32)
    # canonical-helper parity on a subsample (#2552 _ib_arm_c convention)
    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    sub = tr[: min(2_000, len(tr))]
    ib_sub = identity_bias_predict(
        X[sub].toarray().astype(np.float64),
        Y[sub].toarray().astype(np.float64),
        np.zeros((1, d_y), np.float64),
    )[0]
    # BOTH sides fp64 on CPU (r2 pod smoke fix: scipy sparse mean accumulated
    # fp32 and sat 6.6e-7 from the fp64 helper; _col_moments_csr is now exact
    # fp64, so the residual is fp64 summation-order only, ~1e-13)
    mu_x_sub, _vx = _col_moments_csr(X, sub)
    mu_y_sub, _vy = _col_moments_csr(Y, sub)
    bias_sub = mu_y_sub - mu_x_sub
    ib_delta = float(np.abs(ib_sub - bias_sub).max())
    print(f"[controls] ib parity max|delta|={ib_delta:.3e} (fp64 both sides)", flush=True)
    assert np.allclose(ib_sub, bias_sub, rtol=1e-9, atol=1e-8), (
        f"ib bias parity failed vs canonical helper (max|delta|={ib_delta:.3e})"
    )
    np.save(out / "pred_te_ib.fp16.npy", pred_ib.astype(np.float16))
    doc["routes"]["index_aligned_ib"] = {
        "label": "index-aligned NULL — feature indices are unrelated across dictionaries",
        "widths_match": True,
    }

    # train-mean null
    np.save(out / "pred_trainmean.fp32.npy", ymu_full.astype(np.float32))
    doc["routes"]["train_mean_null"] = {"vector": "pred_trainmean.fp32.npy"}

    # 20-draw row-shuffle nulls (lambda pinned upstream; task-body seeds)
    y_te_csc = Y[te].tocsc()
    pred_ridge = np.load(_ridge_dir(args) / "pred_te_ridge.fp32.npy", mmap_mode="r")
    pred_mlp = np.load(_mlp_dir(args) / "pred_te_mlp.fp32.npy", mmap_mode="r")
    pred_densein = np.load(out / "pred_te_densein.fp32.npy", mmap_mode="r")
    nulls = {}
    for name, pr in (
        ("ridge", pred_ridge),
        ("mlp", pred_mlp),
        ("composed", f_comp),
        ("densein", pred_densein),
    ):
        nulls[name] = _shuffle_null_r2_blocked(pr, y_te_csc, SHUFFLE_SEEDS, tag=f"/{name}")
    savez_atomic(
        out / "shuffle_nulls.npz",
        seeds=np.asarray(SHUFFLE_SEEDS, np.int64),
        **{f"null_r2_{k}": v for k, v in nulls.items()},
    )
    doc["shuffle_null"] = {
        "n_draws": len(SHUFFLE_SEEDS),
        "seeds": [int(s) for s in SHUFFLE_SEEDS],
        "convention": "prediction rows permuted; #2476 fp64 R2 kernel",
    }

    # kNN retrieval in answer-feature space (chance = k/n_pool)
    y_te_dense = y_te_csc.toarray().astype(np.float32)
    knn: dict = {
        "chance": {str(k): k / max(1, n_te) for k in KNN_KS},
        "n_pool": int(n_te),
        "routes": {},
    }
    for name, pr in (
        ("ridge", pred_ridge),
        ("mlp", pred_mlp),
        ("composed", f_comp),
        ("densein", pred_densein),
    ):
        pc = {}
        for metric in ("euclidean", "cosine"):
            r = T._knn_retrieval_chunked(
                np.asarray(pr, np.float32),
                y_te_dense,
                ks=KNN_KS,
                metric=metric,
                device=args.device,
            )
            r["wilson_ci_acc"] = {str(k): T._wilson(r["acc_at_k"][k], r["n"]) for k in KNN_KS}
            pc[metric] = r
        knn["routes"][name] = pc
    T._write_json(out / "knn_retrieval.json", knn, phase="controls")
    T._write_json(out / "controls.json", {**doc, **_meta(args, phase="controls")}, phase="controls")
    _stage_upload_files(
        args,
        [
            *finals,
            out / "pred_trainmean.fp32.npy",
            out / "B_densein.fp16.npy",
            out / "standardizer_densein.npz",
        ],
        "analysis_tensors/controls",
        resume_skip=False,
    )
    T._sentinel("controls", "done (4 routes + nulls + kNN)")


# ── P8: per-feature reads (hurdle metrics per route; NO alive floor) ──────────────


def _perfeature_r2(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature UNCONDITIONAL holdout R2 (VERBATIM issue2569_rowbattery.py;
    fp64). Degenerate-variance features -> NaN."""
    t = np.asarray(true, np.float64)
    mu = t.mean(0)
    ss_tot = ((t - mu) ** 2).sum(0)
    ss_res = ((t - np.asarray(pred, np.float64)) ** 2).sum(0)
    r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)
    return r2, ss_tot


def _firing_auroc(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-feature AUROC of the prediction as a score for the FIRING event
    (VERBATIM issue2569_rowbattery.py; midrank Mann-Whitney; all/none-fired -> NaN)."""
    from scipy.stats import rankdata

    p = np.asarray(pred, np.float64)
    t = np.asarray(true, np.float64)
    n = t.shape[0]
    pos = t > 0
    n_pos = pos.sum(0).astype(np.float64)
    n_neg = n - n_pos
    r = rankdata(p, axis=0, method="average")
    rank_pos_sum = np.where(pos, r, 0.0).sum(0)
    auc = (rank_pos_sum - n_pos * (n_pos + 1) / 2.0) / np.maximum(n_pos * n_neg, 1.0)
    return np.where((n_pos > 0) & (n_neg > 0), auc, np.nan)


def _conditional_magnitude_r2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-feature CONDITIONAL-magnitude R2, fired rows ONLY (VERBATIM
    issue2569_rowbattery.py; never mixed with the unconditional R2)."""
    p = np.asarray(pred, np.float64)
    t = np.asarray(true, np.float64)
    m = t > 0
    n_f = m.sum(0).astype(np.float64)
    mu = np.where(n_f > 0, (t * m).sum(0) / np.maximum(n_f, 1.0), 0.0)
    ss_tot = (((t - mu) * m) ** 2).sum(0)
    ss_res = (((t - p) * m) ** 2).sum(0)
    ok = (n_f >= 2) & (ss_tot > 1e-12)
    return np.where(ok, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)


def _routes_for_reads(args) -> list[tuple[str, Path]]:
    return [
        ("ridge", _ridge_dir(args) / "pred_te_ridge.fp32.npy"),
        ("mlp", _mlp_dir(args) / "pred_te_mlp.fp32.npy"),
        ("composed", _controls_dir(args) / "pred_te_composed.fp16.npy"),
        ("densein", _controls_dir(args) / "pred_te_densein.fp32.npy"),
        ("index_ib", _controls_dir(args) / "pred_te_ib.fp16.npy"),
    ]


def phase_perfeature_reads(args) -> None:
    """Per answer feature x route: unconditional R^2, firing AUROC, conditional-
    magnitude R^2 (#2569 hurdle kernels VERBATIM), firing counts, shuffle-null
    bands; census_only when the holdout firing count is 0."""
    C.phase("perfeature_reads")
    out = _perfeature_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    finals = [out / "perfeature_reads.npz", out / "perfeature_summary.json"]
    regime, resume_ok = T._enter_phase_regime(out, args, "perfeature_reads", stale_paths=finals)
    if resume_ok and all(p.exists() for p in finals):
        logger.info("[perfeature_reads] resume: outputs present; skip")
        return
    enc = _encodes_dir(args)
    Y = _load_csr(enc / "ans_full_csr.npz")
    rows = np.load(enc / "rows.npz")
    n_fit, n_val, n_te = len(rows["fit_pos"]), len(rows["val_pos"]), len(rows["te_pos"])
    te = n_fit + n_val + np.arange(n_te)
    d_y = int(Y.shape[1])
    y_te_csc = Y[te].tocsc()
    counts_fit = np.asarray(Y[:n_fit].getnnz(axis=0), np.int64)  # values are > 0 by gating
    counts_te = np.asarray(y_te_csc.getnnz(axis=0), np.int64)
    census_only = counts_te == 0
    nz = np.load(_controls_dir(args) / "shuffle_nulls.npz")
    routes = _routes_for_reads(args)
    preds = {name: np.load(p, mmap_mode="r") for name, p in routes}
    ymu_full, _ = _col_moments_csr(Y, np.arange(n_fit))
    arrays: dict[str, np.ndarray] = {
        "counts_fit": counts_fit,
        "counts_holdout": counts_te,
        "census_only": census_only,
    }
    col_block = 4096
    for name in [*preds, "train_mean"]:
        arrays[f"r2_{name}"] = np.full(d_y, np.nan, np.float32)
        arrays[f"auroc_{name}"] = np.full(d_y, np.nan, np.float32)
        arrays[f"cond_r2_{name}"] = np.full(d_y, np.nan, np.float32)
    for c0 in range(0, d_y, col_block):
        c1 = min(c0 + col_block, d_y)
        t_blk = y_te_csc[:, c0:c1].toarray().astype(np.float64)
        for name, pr in preds.items():
            p_blk = np.asarray(pr[:, c0:c1], np.float64)
            r2, _sst = _perfeature_r2(p_blk, t_blk)
            arrays[f"r2_{name}"][c0:c1] = r2.astype(np.float32)
            arrays[f"auroc_{name}"][c0:c1] = _firing_auroc(p_blk, t_blk).astype(np.float32)
            arrays[f"cond_r2_{name}"][c0:c1] = _conditional_magnitude_r2(p_blk, t_blk).astype(
                np.float32
            )
        tm_blk = np.broadcast_to(ymu_full[c0:c1], t_blk.shape)
        r2_tm, _ = _perfeature_r2(tm_blk, t_blk)
        arrays["r2_train_mean"][c0:c1] = r2_tm.astype(np.float32)
        arrays["auroc_train_mean"][c0:c1] = _firing_auroc(tm_blk, t_blk).astype(np.float32)
        arrays["cond_r2_train_mean"][c0:c1] = _conditional_magnitude_r2(tm_blk, t_blk).astype(
            np.float32
        )
    for k in nz.files:
        if k.startswith("null_r2_"):
            band = np.nanpercentile(np.asarray(nz[k], np.float32), [2.5, 97.5], axis=0)
            arrays[f"{k}_p025"] = band[0].astype(np.float32)
            arrays[f"{k}_p975"] = band[1].astype(np.float32)
    savez_atomic(out / "perfeature_reads.npz", **arrays)

    # summary: medians by fit-firing-count decile, per route (census_only excluded)
    scored = ~census_only
    deciles = np.zeros(d_y, np.int64)
    if int(scored.sum()) >= 10:
        qs = np.quantile(counts_fit[scored].astype(np.float64), np.linspace(0.1, 0.9, 9))
        deciles[scored] = np.digitize(counts_fit[scored].astype(np.float64), qs)
    summary: dict = {
        "n_features": int(d_y),
        "n_census_only": int(census_only.sum()),
        "alive_floor": "NONE (task-body decision: no alive floor anywhere)",
        "routes": {},
    }
    for name in [*preds, "train_mean"]:
        r2 = arrays[f"r2_{name}"]
        summary["routes"][name] = {
            "r2_median": float(np.nanmedian(r2[scored])) if scored.any() else None,
            "auroc_median": float(np.nanmedian(arrays[f"auroc_{name}"][scored]))
            if scored.any()
            else None,
            "cond_r2_median": float(np.nanmedian(arrays[f"cond_r2_{name}"][scored]))
            if scored.any()
            else None,
            "r2_median_by_fit_count_decile": [
                float(np.nanmedian(r2[scored & (deciles == q)]))
                if int((scored & (deciles == q)).sum())
                else None
                for q in range(10)
            ],
        }
    T._write_json(
        out / "perfeature_summary.json",
        {**summary, **_meta(args, phase="perfeature_reads")},
        phase="perfeature_reads",
    )
    _stage_upload_files(args, finals, "analysis_tensors/perfeature", resume_skip=False)
    T._sentinel("perfeature_reads", f"done (census_only={int(census_only.sum())})")


# ── P9: edges (the #1482 map_coefficients recipe on the full-dictionary B) ────────


def _topm_flat_blocked(B_live: np.ndarray, m: int, col_block: int = 4096) -> np.ndarray:
    """Flat indices (into the LIVE-row matrix) of the m largest |entries| —
    #1482 _topk_flat, column-blocked so no fp32 copy of the full matrix lives."""
    d, d_y = B_live.shape
    m = min(int(m), d * d_y)
    best_v = np.empty(0, np.float32)
    best_i = np.empty(0, np.int64)
    for c0 in range(0, d_y, col_block):
        c1 = min(c0 + col_block, d_y)
        blk = np.abs(np.asarray(B_live[:, c0:c1], np.float32))
        flat = blk.ravel()
        take = min(m, flat.size)
        idx = np.argpartition(flat, flat.size - take)[flat.size - take :]
        rows_l, cols_l = np.unravel_index(idx, blk.shape)
        gi = rows_l.astype(np.int64) * d_y + (cols_l + c0)
        best_v = np.concatenate([best_v, flat[idx]])
        best_i = np.concatenate([best_i, gi])
        if len(best_v) > m:
            keep = np.argpartition(best_v, len(best_v) - m)[len(best_v) - m :]
            best_v, best_i = best_v[keep], best_i[keep]
    return best_i


def _edge_survival(
    B_live, ba_m, bb_m, null_sd_all: np.ndarray, tau: np.ndarray, *, topm: int = TOPM
) -> dict:
    """The #1482 survival gate, pure: an edge survives iff it sits in the top-m
    |coef| of BOTH halves AND is sign-consistent with the full fit AND clears
    its column's null threshold. Index space = LIVE rows of B."""
    d, d_y = B_live.shape
    cand = _topm_flat_blocked(B_live, topm)
    ci, cj = np.unravel_index(cand, (d, d_y))
    b_cand = np.asarray(B_live[ci, cj], np.float32)
    set_a = set(_topm_flat_blocked(ba_m, topm).tolist())
    set_b = set(_topm_flat_blocked(bb_m, topm).tolist())
    in_a = np.fromiter((int(c) in set_a for c in cand), bool, len(cand))
    in_b = np.fromiter((int(c) in set_b for c in cand), bool, len(cand))
    ba_c = np.asarray(ba_m[ci, cj], np.float32)
    bb_c = np.asarray(bb_m[ci, cj], np.float32)
    sgn = np.sign(b_cand)
    sign_ok = (np.sign(ba_c) == sgn) & (np.sign(bb_c) == sgn)
    replicated = in_a & in_b & sign_ok
    zval = np.abs(b_cand) / np.maximum(null_sd_all[cj], 1e-30)
    null_ok = np.abs(b_cand) > tau[cj]
    return {
        "cand": cand,
        "ci": ci,
        "cj": cj,
        "b_cand": b_cand,
        "ba": ba_c,
        "bb": bb_c,
        "in_a": in_a,
        "in_b": in_b,
        "sign_ok": sign_ok,
        "replicated": replicated,
        "zval": zval,
        "null_ok": null_ok,
        "surviving": replicated & null_ok,
    }


def _half_done_path(hp: Path) -> Path:
    """Done-marker beside a split-half memmap (written ONLY after the fill)."""
    return hp.with_name(hp.stem.replace(".fp16", "") + ".done.json")


def _half_reusable(hp: Path, d: int, d_y: int) -> bool:
    """A split-half memmap is reusable ONLY with its done-marker present and
    shape-matched (review r1 Major 2: open_memmap pre-allocates the full file,
    so bare existence never proves a completed fill)."""
    dp = _half_done_path(hp)
    if not (hp.exists() and dp.exists()):
        return False
    doc = json.loads(dp.read_text())
    if doc.get("shape") != [int(d), int(d_y)]:
        logger.warning(
            "[edges] %s done-marker shape %s != (%d, %d) — recompute",
            hp.name,
            doc.get("shape"),
            d,
            d_y,
        )
        return False
    return True


def _resolve_receipts() -> dict:
    """Receipts answer-feature sets: regex families over the COMMITTED #2552
    description copy (ids recorded — task body 'record the ids you chose')."""
    import re

    assert DESCRIPTIONS_PATH.exists(), (
        f"committed answer descriptions missing at {DESCRIPTIONS_PATH}"
    )
    doc = json.loads(DESCRIPTIONS_PATH.read_text())
    descs = {int(k): str(v) for k, v in doc["descriptions"].items()}
    out: dict = {
        "source": str(DESCRIPTIONS_PATH.relative_to(PROJECT_ROOT)),
        "n_descriptions": len(descs),
        "families": {},
    }
    for fam, pat in RECEIPTS_PATTERNS.items():
        rx = re.compile(pat, re.IGNORECASE)
        hits = {fid: d for fid, d in descs.items() if rx.search(d)}
        out["families"][fam] = {
            "pattern": pat,
            "n_features": len(hits),
            "feature_ids": sorted(hits),
            "descriptions": {str(k): hits[k] for k in sorted(hits)},
        }
    return out


def phase_edges(args) -> None:
    """Standardized-unit edge read on B: top-32 in/out edges, split-half refit
    (FULL-train standardizer), label-shuffle null threshold, survival gate,
    structure moments, per-column edge-mass curves, receipts resolution."""
    C.phase("edges")
    out = _edges_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    finals = [
        out / "wiring_edges.npz",
        out / "top_pairs.json",
        out / "coef_structure.json",
        out / "receipts_answer_features.json",
        out / "edge_mass_curves.npz",
    ]
    # intermediates JOIN the stale list (review r1 Major 2: a code-SHA recompute
    # must never reuse old-code halves / null calibration)
    intermediates = [
        out / "B_half_a.fp16.npy",
        out / "B_half_b.fp16.npy",
        out / "B_half_a.done.json",
        out / "B_half_b.done.json",
        out / "null_calibration.npz",
    ]
    regime, resume_ok = T._enter_phase_regime(
        out, args, "edges", stale_paths=[*finals, *intermediates]
    )
    if resume_ok and all(p.exists() for p in finals):
        logger.info("[edges] resume: outputs present; skip")
        return
    EA._headroom(args.out_root, 2 if args.smoke else 14, "edges")
    dev = _torch_dev(args)
    enc = _encodes_dir(args)
    X = _load_csr(enc / "ctx_full_csr.npz")
    Y = _load_csr(enc / "ans_full_csr.npz")
    rows = np.load(enc / "rows.npz")
    n_fit = len(rows["fit_pos"])
    tr = np.arange(n_fit)
    std = np.load(_ridge_dir(args) / "standardizer_ridge.npz")
    live = np.asarray(std["live_cols"], np.int64)
    xmu_l, xsd_l = np.asarray(std["xmu_live"]), np.asarray(std["xsd_live"])
    lam_star = float(std["lambda_star"])
    Xl = X[:, live].tocsr()
    d, d_y = int(len(live)), int(Y.shape[1])
    b_full = np.load(_ridge_dir(args) / "B_ridge.fp16.npy", mmap_mode="r")
    B_live = np.asarray(b_full[live], np.float16)

    # ── split-half refits at lambda*, FULL-train standardizer (#1482 phase_split) ──
    perm = np.random.default_rng(SPLITHALF_SEED).permutation(len(tr))
    halves = {"a": np.sort(tr[perm[: len(tr) // 2]]), "b": np.sort(tr[perm[len(tr) // 2 :]])}
    half_B: dict[str, np.ndarray] = {}
    for hname, hrows in halves.items():
        hp = out / f"B_half_{hname}.fp16.npy"
        if _half_reusable(hp, d, d_y):
            half_B[hname] = np.load(hp, mmap_mode="r")
            continue
        acc = _accumulate_raw_products(Xl, Y, hrows, dev, tag=f"/half_{hname}")
        gs_h = _standardized_gram(acc, xmu_l, xsd_l, dev)
        ymu_h, _ = _col_moments_csr(Y, hrows)
        xty_h = _standardized_xty(acc, xmu_l, xsd_l, ymu_h, dev)
        del acc
        gs_h += lam_star * torch.eye(d, dtype=torch.float64, device=gs_h.device)

        def _chol(device_str, g=gs_h):
            return torch.linalg.cholesky(g.to(torch.device(device_str)))

        lfac = T._eigh_fallback(_chol, args.device).to(dev)
        del gs_h
        # ATOMIC write (review r1 Major 2): fill a tmp memmap, fsync-close, then
        # os.replace to the final name + a done-marker written ONLY after the
        # fill completes — a crash mid-fill can never leave a valid-looking,
        # partially-zero half at the final path
        tmp = out / f".tmp_B_half_{hname}.fp16.npy"
        bh = np.lib.format.open_memmap(str(tmp), mode="w+", dtype=np.float16, shape=(d, d_y))
        for c0 in range(0, d_y, 4096):
            c1 = min(c0 + 4096, d_y)
            sol = torch.cholesky_solve(xty_h[:, c0:c1].to(lfac.device), lfac)
            bh[:, c0:c1] = sol.cpu().numpy().astype(np.float16)
        bh.flush()
        del bh
        os.replace(tmp, hp)
        T._write_json(
            _half_done_path(hp),
            {
                "half": hname,
                "shape": [int(d), int(d_y)],
                "n_rows": int(len(hrows)),
                "lambda_star": lam_star,
            },
            phase="edges",
        )
        half_B[hname] = np.load(hp, mmap_mode="r")
        del xty_h, lfac
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[edges] unit split-half {hname} done (n={len(hrows)})", flush=True)

    # ── label-shuffle null (5 draws x 2,048 answer cols; #1482 phase_null) ─────────
    null_path = out / "null_calibration.npz"
    if not null_path.exists():
        rng = np.random.default_rng(EDGE_NULL_SEED_BASE)
        n_cols = min(NULL_COLS, d_y)
        cols = np.sort(rng.choice(d_y, size=n_cols, replace=False))
        Ysub = Y[:, cols].tocsr()
        acc_full = _accumulate_raw_products(Xl, None, tr, dev, tag="/null_gram")
        gs = _standardized_gram(acc_full, xmu_l, xsd_l, dev)
        colsum_x_tr = acc_full["colsum_x"].copy()
        del acc_full
        gs += lam_star * torch.eye(d, dtype=torch.float64, device=gs.device)

        def _chol_f(device_str, g=gs):
            return torch.linalg.cholesky(g.to(torch.device(device_str)))

        lfac = T._eigh_fallback(_chol_f, args.device).to(dev)
        del gs
        draws = np.empty((N_NULL_DRAWS, d, n_cols), np.float32)
        for k in range(N_NULL_DRAWS):
            pk = np.random.default_rng(EDGE_NULL_SEED_BASE + k).permutation(len(tr))
            # X^T Y_perm accumulated chunkwise (rows of X against permuted Ysub rows)
            xty_p = torch.zeros((d, n_cols), dtype=torch.float64, device=dev)
            for s in range(0, len(tr), 8192):
                r = tr[s : s + 8192]
                Xc = Xl[r]
                yd = torch.as_tensor(
                    Ysub[pk[s : s + 8192]].toarray(), dtype=torch.float32, device=dev
                )
                xty_p += _sparse_xtd(Xc, yd, dev).double()
            ymu_k = _csr_colsum64(Ysub[pk]) / max(1, len(tr))
            cs = torch.as_tensor(colsum_x_tr, dtype=torch.float64, device=dev)
            xty_p -= torch.outer(cs, torch.as_tensor(ymu_k, dtype=torch.float64, device=dev))
            xty_p /= torch.as_tensor(xsd_l, dtype=torch.float64, device=dev)[:, None]
            draws[k] = torch.cholesky_solve(xty_p.to(lfac.device), lfac).float().cpu().numpy()
            print(f"[edges] null draw {k + 1}/{N_NULL_DRAWS} done", flush=True)
        del lfac
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        _ymu_s, yvar_sub = _col_moments_csr(Ysub, tr)
        y_sd_sub = np.sqrt(yvar_sub)
        null_sd_col = draws.std(axis=1, dtype=np.float64).mean(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(y_sd_sub > 1e-12, null_sd_col / np.maximum(y_sd_sub, 1e-12), np.nan)
        z_abs_q = np.nanquantile(
            np.abs(draws) / np.maximum(null_sd_col[None, None, :], 1e-30),
            [0.99, 0.999, 0.9999, 0.99999, 1.0],
        )
        savez_atomic(
            null_path,
            cols=cols,
            null_sd_col=null_sd_col,
            y_sd_col=y_sd_sub,
            ratio=ratio,
            z_abs_q=z_abs_q,
            seeds=np.arange(EDGE_NULL_SEED_BASE, EDGE_NULL_SEED_BASE + N_NULL_DRAWS),
        )
        del draws
    nz = np.load(null_path)

    # per-column threshold for EVERY column via the calibrated scale model (#1482)
    ratio = np.asarray(nz["ratio"], np.float64)
    ratio_med = float(np.nanmedian(ratio[np.isfinite(ratio)]))
    z_thresh = float(np.asarray(nz["z_abs_q"])[3])  # 99.999th pct of null |z|
    _ymu_f, yvar_f = _col_moments_csr(Y, tr)
    y_sd_all = np.sqrt(yvar_f)
    null_sd_all = ratio_med * y_sd_all
    tau = z_thresh * null_sd_all

    # ── candidates + survival gate (#1482 phase_analyze, index space = live rows) ──
    gate = _edge_survival(B_live, half_B["a"], half_B["b"], null_sd_all, tau, topm=TOPM)
    cand, ci, cj = gate["cand"], gate["ci"], gate["cj"]
    b_cand, ba_c, bb_c = gate["b_cand"], gate["ba"], gate["bb"]
    in_a, in_b, sign_ok = gate["in_a"], gate["in_b"], gate["sign_ok"]
    replicated, zval, surviving = gate["replicated"], gate["zval"], gate["surviving"]
    order = np.argsort(-zval[surviving])
    sidx = np.nonzero(surviving)[0][order]
    pairs = []
    for p in sidx[: int(args.max_pairs)]:
        i, j = int(ci[p]), int(cj[p])
        pairs.append(
            {
                "ctx_feat_id": int(live[i]),
                "ans_feat_id": int(j),
                "coef_std_units": float(b_cand[p]),
                "coef_half_a": float(ba_c[p]),
                "coef_half_b": float(bb_c[p]),
                "split_half_sign_agree": bool(sign_ok[p]),
                "split_half_both_topk": bool(in_a[p] and in_b[p]),
                "null_z": float(zval[p]),
                "null_threshold_coef": float(tau[j]),
                "is_index_aligned": bool(int(live[i]) == int(j)),
            }
        )

    # ── top-32 in/out edges + per-column mass curves (blocked) ────────────────────
    n_in = min(TOP_EDGES, d)
    in_ids = np.empty((d_y, n_in), np.int32)
    in_coefs = np.empty((d_y, n_in), np.float16)
    mass_curves = np.empty((d_y, len(EDGE_MASS_RANKS)), np.float16)
    for c0 in range(0, d_y, 4096):
        c1 = min(c0 + 4096, d_y)
        blk = np.abs(np.asarray(B_live[:, c0:c1], np.float32))
        sgn_blk = np.asarray(B_live[:, c0:c1], np.float32)
        top = np.argpartition(-blk, n_in - 1, axis=0)[:n_in]
        vals = np.take_along_axis(blk, top, axis=0)
        ordv = np.argsort(-vals, axis=0)
        top_sorted = np.take_along_axis(top, ordv, axis=0)
        in_ids[c0:c1] = live[top_sorted].T.astype(np.int32)
        in_coefs[c0:c1] = np.take_along_axis(sgn_blk, top_sorted, axis=0).T.astype(np.float16)
        col_sum = blk.sum(0)
        vals_sorted = np.take_along_axis(vals, ordv, axis=0)
        n_rank = min(max(EDGE_MASS_RANKS), n_in)
        csum = np.cumsum(vals_sorted[:n_rank], axis=0)
        for ri, r in enumerate(EDGE_MASS_RANKS):
            rr = min(r, n_rank) - 1
            with np.errstate(invalid="ignore", divide="ignore"):
                mass_curves[c0:c1, ri] = np.where(
                    col_sum > 0, csum[rr] / np.maximum(col_sum, 1e-30), np.nan
                ).astype(np.float16)
    n_out = min(TOP_EDGES, d_y)
    out_ids = np.empty((d, n_out), np.int32)
    out_coefs = np.empty((d, n_out), np.float16)
    for r0 in range(0, d, 4096):
        r1 = min(r0 + 4096, d)
        blk = np.asarray(B_live[r0:r1], np.float32)
        ab = np.abs(blk)
        top = np.argpartition(-ab, n_out - 1, axis=1)[:, :n_out]
        vals = np.take_along_axis(ab, top, axis=1)
        ordv = np.argsort(-vals, axis=1)
        top_sorted = np.take_along_axis(top, ordv, axis=1)
        out_ids[r0:r1] = top_sorted.astype(np.int32)
        out_coefs[r0:r1] = np.take_along_axis(blk, top_sorted, axis=1).astype(np.float16)

    # ── structure moments (#1482 _spectrum adapted to torch on-device) ────────────
    bt = torch.as_tensor(np.asarray(B_live, np.float32))
    if dev.type == "cuda":
        bt = bt.to(dev)
    fro2 = float((bt.double() ** 2).sum())
    g_small = (bt.t() @ bt) if d <= d_y else (bt @ bt.t())
    fro4 = float((g_small.double() ** 2).sum())
    rng_s = np.random.default_rng(0)
    om = torch.as_tensor(rng_s.standard_normal((d_y, 256)).astype(np.float32), device=bt.device)
    q = torch.linalg.qr(bt @ om)[0]
    for _ in range(2):
        q = torch.linalg.qr(bt @ (bt.t() @ q))[0]
    sv = torch.linalg.svdvals(q.t() @ bt).cpu().numpy()
    del g_small, q, om
    # per-column concentration (blocked; #1482 phase_analyze quantile summaries)
    col_sum_all = np.zeros(d_y, np.float64)
    pr_col = np.full(d_y, np.nan, np.float64)
    top1_share = np.full(d_y, np.nan, np.float64)
    top10_share = np.full(d_y, np.nan, np.float64)
    for c0 in range(0, d_y, 4096):
        c1 = min(c0 + 4096, d_y)
        blk = np.abs(np.asarray(B_live[:, c0:c1], np.float64))
        cs = blk.sum(0)
        col_sum_all[c0:c1] = cs
        s2 = (blk**2).sum(0)
        s4 = (blk**4).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            pr_col[c0:c1] = np.where(s4 > 0, s2**2 / np.maximum(s4, 1e-300), np.nan)
            top1_share[c0:c1] = np.where(cs > 0, blk.max(0) / np.maximum(cs, 1e-30), np.nan)
            t10 = np.sort(blk, axis=0)[-min(10, d) :].sum(0)
            top10_share[c0:c1] = np.where(cs > 0, t10 / np.maximum(cs, 1e-30), np.nan)
    del bt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    # index-aligned "diagonal" mass — labelled NULL (indices unrelated across dicts)
    live_in_targets = live[live < d_y]
    li = np.searchsorted(live, live_in_targets)
    diag_abs = np.abs(np.asarray(B_live[li, live_in_targets], np.float64))
    diag_mass = float(diag_abs.sum())
    total_mass = float(col_sum_all.sum())

    receipts = _resolve_receipts()
    q_levels = (0.1, 0.5, 0.9)
    structure = {
        "coefficient_units": "standardized input (per 1 SD of the context feature)",
        "lambda_star": lam_star,
        "d_live_ctx": d,
        "d_answer": d_y,
        "spectrum": {
            "frobenius_sq": fro2,
            "sum_sigma4": fro4,
            "participation_ratio_spectrum": fro2**2 / fro4 if fro4 > 0 else float("nan"),
            "sigma_top": [float(x) for x in sv[:64]],
            "stable_rank": fro2 / float(sv[0] ** 2) if sv.size and sv[0] > 0 else float("nan"),
            "method": "#1482 _spectrum: exact Frobenius moments + randomized subspace "
            "(2 power iterations, 256 probes)",
        },
        "column_concentration": {
            "top1_share": {
                f"q{int(q * 100)}": float(np.nanquantile(top1_share, q)) for q in q_levels
            },
            "top10_share": {
                f"q{int(q * 100)}": float(np.nanquantile(top10_share, q)) for q in q_levels
            },
            "participation_ratio_per_column": {
                f"q{int(q * 100)}": float(np.nanquantile(pr_col, q)) for q in q_levels
            },
        },
        "index_aligned_diagonal": {
            "label": "NULL — feature indices are unrelated across dictionaries",
            "abs_mass_share": diag_mass / total_mass if total_mass > 0 else float("nan"),
            "n_entries": int(len(live_in_targets)),
        },
        "off_diagonal_mass_share": 1.0 - (diag_mass / total_mass) if total_mass > 0 else None,
        "null_calibration": {
            "n_draws": N_NULL_DRAWS,
            "n_cols_sampled": int(len(np.asarray(nz["cols"]))),
            "seed_base": EDGE_NULL_SEED_BASE,
            "ratio_median": ratio_med,
            "z_thresh_p99999": z_thresh,
        },
        "gate": {
            "top_m": TOPM,
            "splithalf_seed": SPLITHALF_SEED,
            "n_candidates": int(len(cand)),
            "n_replicated": int(replicated.sum()),
            "n_surviving": int(surviving.sum()),
            "rule": "top-20,000 |coef| of BOTH halves AND sign-consistent AND above the "
            "column's null threshold (#1482 recipe)",
        },
        **_meta(args, phase="edges"),
    }
    savez_atomic(
        out / "wiring_edges.npz",
        live_cols=live,
        in_edge_ids=in_ids,
        in_edge_coefs=in_coefs,
        out_edge_ids=out_ids,
        out_edge_coefs=out_coefs,
        out_edge_ctx_ids=live.astype(np.int32),
        cand_flat=cand,
        cand_surviving=surviving,
        cand_replicated=replicated,
        tau=tau.astype(np.float32),
        null_sd_all=null_sd_all.astype(np.float32),
    )
    savez_atomic(
        out / "edge_mass_curves.npz",
        ranks=np.asarray(EDGE_MASS_RANKS, np.int64),
        curves=mass_curves,
    )
    T._write_json(
        out / "top_pairs.json",
        {
            "n_surviving": int(surviving.sum()),
            "pairs": pairs,
            "max_pairs_kept": int(args.max_pairs),
            **_meta(args, phase="edges"),
        },
        phase="edges",
    )
    T._write_json(out / "coef_structure.json", structure, phase="edges")
    T._write_json(
        out / "receipts_answer_features.json",
        {**receipts, **_meta(args, phase="edges")},
        phase="edges",
    )
    _measured_update(
        args, edges_n_surviving=int(surviving.sum()), edges_n_candidates=int(len(cand))
    )
    _stage_upload_files(
        args, [*finals, out / "null_calibration.npz"], "analysis_tensors/edges", resume_skip=False
    )
    T._sentinel("edges", f"done (surviving={int(surviving.sum())}/{len(cand)})")


# ── P10: eval lists (ctx SAE on the SAME 2,000 judged rows) ───────────────────────


@torch.no_grad()
def _ctx_lists(args, sae, eval_ids: np.ndarray, eval_pos: np.ndarray, x_mm) -> list[dict]:
    """All-active + judged top-100 per turn (the #2552 _ta_lists shape VERBATIM,
    pointed at the ctx SAE over X19)."""
    turns = []
    for s in range(0, len(eval_pos), 256):
        x = torch.as_tensor(np.asarray(x_mm[eval_pos[s : s + 256]], np.float32), device=sae.device)
        f = sae.encode(x)
        for j in range(f.shape[0]):
            row = f[j]
            nzt = torch.nonzero(row > 0, as_tuple=False).squeeze(-1)
            order = torch.argsort(row[nzt], descending=True)
            ids = nzt[order].cpu().numpy()
            vals = row[nzt][order].cpu().numpy()
            full = [[int(i), float(v)] for i, v in zip(ids, vals, strict=True)]
            turns.append(
                {
                    "row_id": int(eval_ids[s + j]),
                    "pre_truncation": full,  # all-active (turn-averaged native list)
                    "judged_top100": full[:100],
                }
            )
    return turns


def phase_eval_lists(args) -> None:
    """Per-turn ctx-SAE feature lists for the pinned #2552 eval rows; persisted
    like feature_lists_rep_ta.*.jsonl (sharded + index)."""
    C.phase("eval_lists")
    out = _lists_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    finals = [out / "lists_ctx.json", out / "feature_lists_2000turns.json"]
    regime, resume_ok = T._enter_phase_regime(
        out,
        args,
        "eval_lists",
        stale_paths=[*finals, *out.glob("feature_lists_ctx*.jsonl")],
    )
    if resume_ok and all(p.exists() for p in finals):
        logger.info("[eval_lists] resume: outputs present; skip")
        return
    eval_ids, source = _eval_ids_in_slice(args)
    eval_pos = _positions_of(args, eval_ids, "eval_lists")
    if _production(args):
        assert len(eval_pos) == EVAL_TURNS_N, (
            f"[eval_lists] join assert FAILED: {len(eval_pos)}/{EVAL_TURNS_N} pinned eval "
            "rows present in the assembled memmap"
        )
    a_dir = T._assemble_dir(args)
    x_mm = np.load(a_dir / "X19.fp16.npy", mmap_mode="r")
    sae = _load_ctx_sae(args)
    turns = _ctx_lists(args, sae, eval_ids, eval_pos, x_mm)
    meta = {
        "config": "ctx",
        "eval_row_source": source,
        "eval_ids_sha256": _sha_ids(eval_ids),
        "list_convention": "all-active turn-averaged ctx codes, desc by activation; "
        "judged list = top-100",
    }
    T._write_json(out / "lists_ctx.json", {**meta, "turns": turns}, phase="eval_lists")
    parts = _jsonl_write_sharded(out / "feature_lists_ctx.jsonl", turns)
    index = {
        "schema": "sharded-jsonl-v1",
        "configs": {"ctx": {"meta": meta, "files": [p.name for p in parts], "n_turns": len(turns)}},
    }
    (out / "feature_lists_2000turns.json").write_text(json.dumps(index))
    union = set()
    for t_ in turns:
        union.update(int(f) for f, _v in t_["judged_top100"])
    _measured_update(args, ctx_eval_list_union_n=len(union), n_eval_rows=len(turns))
    _stage_upload_files(
        args,
        [out / "feature_lists_2000turns.json", *parts],
        "analysis_tensors/eval_lists",
        resume_skip=False,
    )
    del sae
    if args.device == "cuda":
        torch.cuda.empty_cache()
    T._sentinel("eval_lists", f"done (n={len(turns)} union={len(union)})")


# ── P11: mining (context-side evidence for W1) ────────────────────────────────────


@torch.no_grad()
def _mine_top_rows(sae, mm, positions, row_ids, cols, device, chunk=4096):
    """Streaming per-feature top-25 activating ROWS (VERBATIM #2552 _mine_top_rows)."""
    cols_t = torch.as_tensor(np.asarray(cols, np.int64), device=device)
    n_f = len(cols)
    top_v = torch.full((MINING_TOP, n_f), -1.0, dtype=torch.float32, device=device)
    top_r = torch.full((MINING_TOP, n_f), -1, dtype=torch.int64, device=device)
    pos = np.asarray(positions, np.int64)
    rid = torch.as_tensor(np.asarray(row_ids, np.int64), device=device)
    t0 = time.time()
    n_chunks = math.ceil(len(pos) / chunk)
    for i, s in enumerate(range(0, len(pos), chunk)):
        x = torch.as_tensor(np.asarray(mm[pos[s : s + chunk]], np.float32), device=device)
        f = sae.encode(x, chunk=chunk)[:, cols_t]
        cat_v = torch.cat([top_v, f], dim=0)
        cat_r = torch.cat([top_r, rid[s : s + x.shape[0]].unsqueeze(1).expand(-1, n_f)], dim=0)
        v, idx = torch.topk(cat_v, MINING_TOP, dim=0)
        top_v = v
        top_r = torch.gather(cat_r, 0, idx)
        if (i + 1) % 20 == 0 or i + 1 == n_chunks:
            print(f"[mine] chunk {i + 1}/{n_chunks} elapsed={time.time() - t0:.0f}s", flush=True)
    top_r[top_v <= 0] = -1  # never present a zero-activation "example"
    return top_v.cpu().numpy(), top_r.cpu().numpy()


def _need_set(args) -> tuple[np.ndarray, dict]:
    """Description-need set: eval-list union U ctx features in top_pairs U top-32
    in-edges of the receipts answer features (task-body mining rule)."""
    lists = json.loads((_lists_dir(args) / "lists_ctx.json").read_text())
    eval_union: set[int] = set()
    for t_ in lists["turns"]:
        eval_union.update(int(f) for f, _v in t_["judged_top100"])
    wz = np.load(_edges_dir(args) / "wiring_edges.npz")
    in_ids = np.asarray(wz["in_edge_ids"], np.int64)
    # FULL surviving mask, never the display-capped top_pairs.json (review r1
    # Minor 4: --max-pairs caps the DISPLAY artifact only)
    live_w = np.asarray(wz["live_cols"], np.int64)
    d_y_w = int(in_ids.shape[0])
    ci_w = np.asarray(wz["cand_flat"], np.int64) // d_y_w
    surv_w = np.asarray(wz["cand_surviving"], bool)
    pair_ctx = {int(x) for x in live_w[ci_w[surv_w]]}
    receipts = json.loads((_edges_dir(args) / "receipts_answer_features.json").read_text())
    receipt_ctx: set[int] = set()
    for fam in receipts["families"].values():
        for aid in fam["feature_ids"]:
            if 0 <= int(aid) < in_ids.shape[0]:
                receipt_ctx.update(int(x) for x in in_ids[int(aid)])
    need = np.asarray(sorted(eval_union | pair_ctx | receipt_ctx), np.int64)
    doc = {
        "n_eval_list_union": len(eval_union),
        "n_top_pairs_ctx": len(pair_ctx),
        "n_receipts_inedge_ctx": len(receipt_ctx),
        "n_need_total": int(len(need)),
        "rule": "eval-list union U FULL surviving-edge ctx side (wiring_edges.npz mask; "
        "top_pairs.json is the display-capped artifact) U top-32 in-edges of receipts answers",
    }
    return need, doc


def phase_mining(args) -> None:
    """Top-25 activating USER PROMPTS + 20 non-activating negatives per need-set
    feature, over the 120k map-fit rows (eval-disjoint by split construction,
    hard-asserted). #2552 top25 jsonl shape + a `kind` field for negatives."""
    C.phase("mining")
    out = _mining_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    heap_path = out / "top25_ctx.npz"
    done_marker = out / ".text_join_done.json"
    regime, resume_ok = T._enter_phase_regime(
        out,
        args,
        "mining",
        stale_paths=[heap_path, done_marker, out / "need_set.json", *out.glob("top25_ctx*.jsonl")],
    )
    if resume_ok and heap_path.exists() and done_marker.exists():
        logger.info("[mining] resume: heap + text join present; skip")
        return
    enc = _encodes_dir(args)
    rows = np.load(enc / "rows.npz")
    fit_ids = np.asarray(rows["fit_ids"], np.int64)
    fit_pos = np.asarray(rows["fit_pos"], np.int64)
    row_ci = np.load(T._stage_dir(args) / "row_ci.npy")
    text_ok = row_ci[fit_ids] >= 0
    pool_ids, pool_pos = fit_ids[text_ok], fit_pos[text_ok]
    eval_ids, _src = _eval_ids_in_slice(args)
    assert set(int(x) for x in eval_ids).isdisjoint(int(x) for x in pool_ids), (
        "MF-A VIOLATION: mining pool intersects the judged eval rows"
    )
    need, need_doc = _need_set(args)
    need_doc["n_pool_rows"] = int(len(pool_ids))
    need_doc["n_excluded_non_text"] = int((~text_ok).sum())
    if len(need) == 0:
        logger.warning("[mining] empty need set — writing empty artifacts (smoke shape)")
        savez_atomic(heap_path, feat_ids=np.empty(0, np.int64))
        T._write_json(
            out / "need_set.json", {**need_doc, **_meta(args, phase="mining")}, phase="mining"
        )
        T._write_json(done_marker, {"written": {}, "n_unique_rows": 0}, phase="mining")
        T._sentinel("mining", "done (empty need set)")
        return
    a_dir = T._assemble_dir(args)
    x_mm = np.load(a_dir / "X19.fp16.npy", mmap_mode="r")
    sae = _load_ctx_sae(args)
    vals, hrows = _mine_top_rows(sae, x_mm, pool_pos, pool_ids, need, args.device)

    # negatives: one shared seeded candidate pool, encoded once at the need cols;
    # per feature the first MINING_NEG non-firing candidates in a per-feature
    # seeded order (fewer than MINING_NEG non-firing -> lowest-activation fill)
    rng = np.random.default_rng(CTX_SEED)
    n_cand = min(MINING_NEG_CAND, len(pool_ids))
    cand_sel = rng.choice(len(pool_ids), size=n_cand, replace=False)
    cand_pos, cand_ids = pool_pos[cand_sel], pool_ids[cand_sel]
    f_cand = T._encode_restricted(sae, x_mm, cand_pos, need)  # (n_cand, n_need) fp16
    negs: dict[int, list[tuple[int, str]]] = {}
    for fi, feat in enumerate(need):
        order = np.random.default_rng([CTX_SEED, 9, int(feat)]).permutation(n_cand)
        acts = np.asarray(f_cand[:, fi], np.float32)
        zero = order[acts[order] <= 0][:MINING_NEG]
        picked = [(int(cand_ids[i]), "negative") for i in zero]
        if len(picked) < MINING_NEG:
            rest = order[acts[order] > 0]
            low = rest[np.argsort(acts[rest], kind="stable")][: MINING_NEG - len(picked)]
            picked += [(int(cand_ids[i]), "negative_lowest_activation") for i in low]
        negs[int(feat)] = picked
    savez_atomic(heap_path, feat_ids=need, top_vals=vals, top_rows=hrows, neg_cand_ids=cand_ids)
    del sae, f_cand
    if args.device == "cuda":
        torch.cuda.empty_cache()

    needed_rows: set[int] = set(int(r) for r in hrows.ravel() if r >= 0)
    for picked in negs.values():
        needed_rows.update(r for r, _k in picked)
    needed_ci = {int(row_ci[r]): int(r) for r in sorted(needed_rows)}
    assert all(ci >= 0 for ci in needed_ci), "mining pool leaked a non-text-resolvable row"
    texts: dict[int, str] = {}
    for row_idx, _ci, prompt, _response in _iter_rows_pinned(args, needed_ci, tag="mining"):
        texts[int(row_idx)] = prompt[:EXAMPLE_TEXT_CAP]  # USER PROMPT (context side)
    missing = needed_rows - set(texts)
    assert not missing, f"text join missing {len(missing)} mining rows"
    recs = []
    for fi, feat in enumerate(need):
        for rank in range(MINING_TOP):
            r = int(hrows[rank, fi])
            if r < 0:
                continue
            recs.append(
                {
                    "family": "ctx",
                    "feat_id": int(feat),
                    "rank": rank,
                    "row_id": r,
                    "activation": float(vals[rank, fi]),
                    "kind": "positive",
                    "text": texts[r],
                }
            )
        for rank, (r, kind) in enumerate(negs[int(feat)]):
            recs.append(
                {
                    "family": "ctx",
                    "feat_id": int(feat),
                    "rank": rank,
                    "row_id": r,
                    "activation": 0.0,
                    "kind": kind,
                    "text": texts[r],
                }
            )
    parts = _jsonl_write_sharded(out / "top25_ctx.jsonl", recs)
    T._write_json(
        out / "need_set.json",
        {
            **need_doc,
            "need_ids": [int(x) for x in need],
            "mining_ids_sha256": _sha_ids(np.sort(pool_ids)),
            "eval_disjoint_assert": "PASS",
            **_meta(args, phase="mining"),
        },
        phase="mining",
    )
    T._write_json(
        done_marker,
        {"written": [p.name for p in parts], "n_unique_rows": len(texts)},
        phase="mining",
    )
    print(
        f"[mining] unit done: need={len(need)} examples={len(recs)} files={len(parts)}", flush=True
    )
    _stage_upload_files(args, [*parts, heap_path], "raw_completions/mining", resume_skip=False)
    _stage_upload_files(args, [out / "need_set.json"], "analysis_tensors", resume_skip=False)
    T._sentinel("mining", f"done (need={len(need)}, {len(texts)} unique rows joined)")


# ── P12: upload ───────────────────────────────────────────────────────────────────

REPO_JSON_COPIES = (
    ("sae_metrics", "sae_metrics_ctx.json"),
    ("sae_metrics", "sae_metrics_answer.json"),
    ("encodes", "zero_variance.json"),
    ("map_ridge", "map_ridge_metrics.json"),
    ("map_mlp", "map_mlp_metrics.json"),
    ("controls", "controls.json"),
    ("controls", "knn_retrieval.json"),
    ("perfeature", "perfeature_summary.json"),
    ("edges", "top_pairs.json"),
    ("edges", "coef_structure.json"),
    ("edges", "receipts_answer_features.json"),
    ("mining", "need_set.json"),
)


def phase_upload(args) -> None:
    """Re-upload every leaf (resume_skip=False: a fixed-artifact re-run must
    re-upload) + copy the small JSONs under eval_results/issue_2661/ (smoke
    diverts under out_root) + print a byte-size manifest."""
    C.phase("upload")
    leaves = [
        (
            _sae_ctx_dir(args),
            ["sae_weights.safetensors", "cfg.json", "train_log.json", "gates_g1.json"],
            "analysis_tensors/sae_ctx",
        ),
        (_metrics_dir(args), None, "analysis_tensors/sae_metrics"),
        (_encodes_dir(args), None, "analysis_tensors/encodes"),
        (_ridge_dir(args), None, "analysis_tensors/map_ridge"),
        (_mlp_dir(args), None, "analysis_tensors/map_mlp"),
        (_controls_dir(args), None, "analysis_tensors/controls"),
        (_perfeature_dir(args), None, "analysis_tensors/perfeature"),
        (_edges_dir(args), None, "analysis_tensors/edges"),
        (_lists_dir(args), None, "analysis_tensors/eval_lists"),
        (_mining_dir(args), None, "raw_completions/mining"),
    ]
    manifest: dict[str, dict] = {}
    for d, names, leaf in leaves:
        if not d.exists():
            logger.warning("[upload] leaf dir missing (phase not run?): %s", d)
            continue
        files = (
            [d / n for n in names if (d / n).exists()]
            if names
            else [p for p in sorted(d.iterdir()) if p.is_file() and p.name != "regime.json"]
        )
        manifest[leaf] = {p.name: p.stat().st_size for p in files}
        _stage_upload_files(args, files, leaf, resume_skip=False)
    repo_dst = (
        PROJECT_ROOT / "eval_results" / "issue_2661"
        if _production(args)
        else args.out_root / "repo_stage" / "eval_results" / "issue_2661"
    )
    repo_dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for sub, name in REPO_JSON_COPIES:
        src = args.out_root / sub / name
        if src.exists():
            shutil.copy2(src, repo_dst / name)
            copied.append(name)
    total = sum(sum(v.values()) for v in manifest.values())
    print(
        "[upload] manifest "
        + json.dumps(
            {
                "total_bytes": total,
                "leaves": {
                    k: {"n_files": len(v), "bytes": sum(v.values())} for k, v in manifest.items()
                },
                "repo_json_copies": copied,
            }
        ),
        flush=True,
    )
    T._write_json(
        args.out_root / "upload_manifest.json",
        {
            "leaves": manifest,
            "repo_json_copies": copied,
            "total_bytes": total,
            **_meta(args, phase="upload"),
        },
        phase="upload",
    )
    sent_dir = args.out_root / "sentinels"
    sent_dir.mkdir(parents=True, exist_ok=True)
    T._write_json(sent_dir / "p1_done.json", {"phase": "p1", "status": "done"}, phase="upload")
    T._sentinel("upload", f"done ({total} bytes across {len(manifest)} leaves)")


# ── composed smoke (same phase functions, tiny slice, CPU-minutes) ───────────────


def phase_smoke(args) -> None:
    """Tiny-N composed run of the SAME phase functions: 2 capture chunks,
    <=2,000 rows, ctx width 1,024 / k 16, ~1 epoch, uploads skipped, under
    out_root/smoke. Asserts the flat tier-bounds trap guard + per-leg outputs,
    then re-runs two phases to exercise the resume/skip legs."""
    s = argparse.Namespace(**vars(args))
    s.smoke = True
    s.out_root = args.out_root / "smoke"
    s.max_chunks = args.max_chunks or 2
    s.smoke_rows = args.smoke_rows or 2_000
    s.sae_steps = args.sae_steps or 8  # ~1 epoch at 2,000 rows / batch 256
    s.ctx_width = int(args.ctx_width) or 1_024
    s.ctx_k = int(args.ctx_k) or 16
    s.n_eval_turns = int(args.n_eval_turns) or 4
    s.skip_upload = True
    s.out_root.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}
    for name in PHASE_ORDER:
        t0 = time.time()
        PHASES[name](s)
        timing[name] = round(time.time() - t0, 1)
        print(f"[smoke] leg {name} done in {timing[name]}s", flush=True)
    checks = {
        "assemble": T._assemble_dir(s) / "split_meta.json",
        "sae_train_ctx": _sae_ctx_dir(s) / "sae_weights.safetensors",
        "sae_metrics": _metrics_dir(s) / "sae_metrics_ctx.json",
        "encode_full": _encodes_dir(s) / "ctx_full_csr.npz",
        "map_ridge_full": _ridge_dir(s) / "B_ridge.fp16.npy",
        "map_mlp": _mlp_dir(s) / "pred_te_mlp.fp32.npy",
        "controls": _controls_dir(s) / "shuffle_nulls.npz",
        "perfeature_reads": _perfeature_dir(s) / "perfeature_reads.npz",
        "edges": _edges_dir(s) / "top_pairs.json",
        "eval_lists": _lists_dir(s) / "feature_lists_2000turns.json",
        "mining": _mining_dir(s) / "need_set.json",
        "upload": s.out_root / "upload_manifest.json",
    }
    missing = {k: str(p) for k, p in checks.items() if not p.exists()}
    assert not missing, f"[smoke] legs missing outputs: {missing}"
    sae = _load_ctx_sae(s)
    assert sae.tier_bounds == (sae.dict_size,), "smoke tier-bounds assert (flat trap guard)"
    # resume-matrix light: a second invocation of two phases must SKIP
    for name in ("sae_metrics", "map_ridge_full"):
        t0 = time.time()
        PHASES[name](s)
        took = time.time() - t0
        assert took < max(30.0, timing[name]), f"[smoke] {name} re-run did not fast-skip"
        print(f"[smoke] resume leg {name} skipped in {took:.1f}s", flush=True)
    T._write_json(s.out_root / "smoke_timing.json", {"legs_s": timing}, phase="smoke")
    T._sentinel("smoke", f"composed smoke PASS ({json.dumps(timing)})")
    logger.info("[smoke] PASS: %s", timing)


# ── CLI ──────────────────────────────────────────────────────────────────────────

PHASE_ORDER = (
    "assemble",
    "sae_train_ctx",
    "sae_metrics",
    "encode_full",
    "map_ridge_full",
    "map_mlp",
    "controls",
    "perfeature_reads",
    "edges",
    "eval_lists",
    "mining",
    "upload",
)

PHASES = {
    "smoke": phase_smoke,
    "assemble": phase_assemble,
    "sae_train_ctx": phase_sae_train_ctx,
    "sae_metrics": phase_sae_metrics,
    "encode_full": phase_encode_full,
    "map_ridge_full": phase_map_ridge_full,
    "map_mlp": phase_map_mlp,
    "controls": phase_controls,
    "perfeature_reads": phase_perfeature_reads,
    "edges": phase_edges,
    "eval_lists": phase_eval_lists,
    "mining": phase_mining,
    "upload": phase_upload,
}


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Issue #2661 driver (flat ctx SAE + full-dictionary feature map)"
    )
    ap.add_argument("--phase", default="all", choices=["all", *PHASES])
    ap.add_argument("--out-root", type=Path, default=Path("/workspace/eps-issue-2661"))
    ap.add_argument("--hf-prefix", default="issue2661_flatsae")
    ap.add_argument("--smoke", action="store_true", help="tiny-N run of the SAME pipeline")
    ap.add_argument("--smoke-rows", type=int, default=0, help="0 = production")
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = all 1,920 chunks")
    ap.add_argument("--sae-steps", type=int, default=0, help="0 = full 3-epoch train")
    ap.add_argument("--ctx-width", type=int, default=0, help="0 = production 32,768")
    ap.add_argument("--ctx-k", type=int, default=0, help="0 = production 128")
    ap.add_argument("--n-eval-turns", type=int, default=0, help="0 = 2,000 (production)")
    ap.add_argument("--max-pairs", type=int, default=500, help="top_pairs.json cap")
    ap.add_argument("--gen-batch", type=int, default=16, help="vendored-kernel passthrough")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--skip-upload", action="store_true", help="local-only run (loud)")
    ap.add_argument(
        "--fresh-stream", action="store_true", help="assemble: ignore the stream cursor"
    )
    ap.add_argument("--gpu-id", type=int, default=-1, help="informational; CVD pins the device")
    ap.add_argument("--production", action="store_true", help="assert-guard: refuse smoke knobs")
    ap.add_argument(
        "--resume-across-code-sha",
        action="store_true",
        help="crash-fix relaunch escape (vendored _enter_phase_regime contract)",
    )
    # vendored-kernel passthrough (regime hash inputs; #2552 driver convention)
    ap.add_argument("--tiny-model", action="store_true", help="smoke-only from-config model")
    ap.add_argument("--sae-dict", type=int, default=0, help="parent passthrough (unused here)")
    ap.add_argument("--sae-k", type=int, default=0, help="parent passthrough (unused here)")
    ap.add_argument("--fit-n", type=int, default=0, help="parent passthrough (regime hash)")
    ap.add_argument("--n-perm", type=int, default=10_000, help="parent passthrough (regime hash)")
    ap.add_argument("--n-boot", type=int, default=10_000, help="parent passthrough (regime hash)")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="argparse-attribute completeness + deferred-import resolution",
    )
    ap.add_argument("--list-phases", action="store_true", help="print the phase registry and exit")
    return ap.parse_args(argv)


def main() -> None:
    args = _parse_args()
    if args.list_phases:
        print(json.dumps({"order": list(PHASE_ORDER), "registry": sorted(PHASES)}))
        raise SystemExit(0)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Deferred-import resolution (smoke-architecture Axis 1): execute every
        # function-body import of this driver so a missing symbol fails HERE.
        from scipy.stats import rankdata  # noqa: F401

        from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
            identity_bias_predict,
        )
        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.provenance import (  # noqa: F401
            as_metadata_dict,
            git_provenance,
        )
        from explore_persona_space.orchestrate.upload_sharded import (  # noqa: F401
            upload_dir_sharded,
        )
        from huggingface_hub import HfApi, hf_hub_download  # noqa: F401
        from safetensors.torch import load_file, save_file  # noqa: F401

        print("[import-check] OK", flush=True)
        raise SystemExit(0)
    if args.production:
        assert _production(args), (
            "--production forbids smoke knobs (--smoke/--smoke-rows/--max-chunks/"
            "--sae-steps/--ctx-width/--ctx-k/--n-eval-turns/--tiny-model)"
        )
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda":
        # fp32 GEMMs (B/pred blocks, MLP) run under TF32 for wall-clock; every
        # fp64 path (Gram accumulation, eigh, Cholesky) is unaffected. Recorded
        # in deviations.md; the Gram itself accumulates fp64 either way.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("[main] TF32 enabled for fp32 GEMMs (fp64 paths unaffected)")
    args.out_root.mkdir(parents=True, exist_ok=True)
    _regime_pins()  # fail fast when the committed pin record is absent/drifted
    logger.info(
        "[main] phase=%s out_root=%s device=%s smoke=%s",
        args.phase,
        args.out_root,
        args.device,
        args.smoke,
    )
    seq = PHASE_ORDER if args.phase == "all" else (args.phase,)
    for name in seq:
        PHASES[name](args)
    print("[phase=done]", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
