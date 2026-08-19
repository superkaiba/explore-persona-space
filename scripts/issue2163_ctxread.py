#!/usr/bin/env python
"""issue 2163 — which SAE features are READ at the context vector by the context->answer map.

Phased driver (plan v4, tasks #2163). Reuses the parent #1482 machinery via import
(`issue1482_densesae_fullwidth` fit path, `issue1738_sae_arm._GramFactor`,
`issue1482_early_layer._stage_scratch_meta`, `analysis.mapping_baselines`) and adds ONLY the
new reads: the U/C/A read ladder on the W and M maps, Carried_j, the sparse-input B refit,
the last-token census, and the Phase-5 activity-matched partial regression with
selection-symmetric stratified-permutation nulls.

Phases (CLI `--phase`):
  upload-inputs   Phase U (VM only): push dense targets + covariates v2 to the HF inputs prefix.
  stage           Phase 0a: stage store/inputs/meta/SAE onto the work root (pod side).
  census          Phase 0b: one-pass shard sweep -> Psi_last CSR + answer-mean CSR + census
                  (+ corpus-join pins, prefix-degeneracy assert).
  fit-maps        Phase 1: W + M refits with parity gates; bundles persisted + uploaded.
  read-ladder     Phase 2: U/U*sd/C/A on W and M, per-half + split-half A, recon share,
                  A_j row-pairing null (200 draws).
  carried         Phase 3: Carried_j + paired contrast.
  answer-matchedn Phase 4: answer-side pooled-vs-conditional + matched-n control.
  partials        Phase 5: activity-matched partials + per-DV stratified permutation nulls.
  confirm-b       Phase 6: census-restricted sparse-input B refit (CPU; venue switch to GPU).
  confirm-b-gpu   Phase 6-alt: the fit leg on the GPU cell (re-stages inputs + re-runs census;
                  uploads results + the panel sub-block itself — see the phase docstring).
  upload-verify   Phase 7: upload results + bundles, exact-set verify.
  harvest         Phase 8a (VM): download small results into eval_results/issue_2163/.
  figures         Phase 8b (VM): render figures into figures/issue_2163/.

The thin pod wrapper `scripts/issue2163_pod_workload.sh` runs phases 0-7 and owns the single
terminal done token; this module never emits it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE numpy/torch: thread caps + HF/W&B credentials (code-style.md #847)

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

import issue1482_densesae_fullwidth as DSF  # noqa: E402  (parent driver, reused via import)
from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", force=True
)
logger = logging.getLogger("i2163.ctxread")

# ── constants ─────────────────────────────────────────────────────────────────

HF_DATA_REPO = DSF.HF_DATA_REPO
STORE_PREFIX = DSF.STORE_PREFIX  # issue1482_error_analysis/analysis_tensors/sae_pooled
INPUTS_1482_PREFIX = DSF.INPUTS_PREFIX  # issue1482_densesae_fullwidth/inputs
SCRATCH_META_PREFIX = "issue1482_error_analysis/analysis_tensors/scratch_meta"
PERFEATURE_1482_PATH = "issue1482_densesae_fullwidth/perfeature/ridge__mean_perfeature.npz"
OUT_PREFIX = "issue2163_ctxread"  # NEVER the parent's prefixes (#1005 clobber class)
INPUTS_PREFIX = f"{OUT_PREFIX}/inputs"

DICT_SIZE = DSF.DICT_SIZE  # 131_072
H_DIM = DSF.H_DIM  # 3_584
LAMBDAS = DSF.LAMBDAS  # np.logspace(-3, 8, 23)
OUT_BLOCK = DSF.OUT_BLOCK  # 16_384
SEED = 2163

# Phase-1 parity targets (banked in git; plan section 4 Phase 1).
# eval_results/issue_1482/densesae_fullwidth/cells/ridge__mean.json
W_BANKED_LAMBDA = 3162.2776601683795  # grid index 13
W_BANKED_POOLED_R2_PANEL = 0.7216031048274674
# eval_results/issue_1482/sae_dense_bridge/sae_dense_bridge.json (dense_ctx__to__dense, n=120k)
M_BANKED_LAMBDA = 1000.0  # grid index 12
M_BANKED_DENSE_R2 = 0.7166207983746851
M_BANKED_IDENTITY_BIAS_R2 = -1.0229822863871925

FLOOR_LAST_DEFAULT = 20  # census restriction floor (active train rows)
CAP_DB_DEFAULT = 49_152  # census restriction cap (fp64 eigh RAM/wall, plan section 9)
KNN_N_PRED = 5_000  # bridge convention (issue1482_sae_dense_bridge.KNN_N_PRED)
KNN_KS = (1, 5, 10)
N_DRAWS_PAIRING = 200
N_DRAWS_STRAT = 1_000
MATCHEDN_FEATURES = 2_000
MATCHEDN_DRAWS = 200
SMOKE_UPLOAD_CAP_BYTES = 100 * 1024 * 1024

# Selection-set constants (plan section 4 Phase 5).
DROPPED_ALWAYS = ("dec_norm",)  # constant column (std 3.92e-07): rank-degenerate
LASTTOKEN_COVS = ("lasttoken_count", "mean_act_when_active_lasttoken", "span_last_ratio")
MATCH_COV = "lasttoken_count"
MATCH_COV_2 = "firing_freq_per_token"  # robustness second matching variable
GEOMETRY_DEFINITIONAL = ("massive_dim_mass", "write_norm", "proj_var")

# Parent-cited references (reported, never recomputed here).
PARENT_DECILE_GRADIENT = (0.091, 0.266)
PARENT_SHUFFLE_BAND = {"median": -0.07, "rare_decile_q2_5": -0.139, "K": 20}

# Prefix-degeneracy invariant (CORRECTED after the full-grain Phase-0 measurement, plan v6 §12 A11).
# The earlier byte-identity form (`n_distinct == 1`) was falsified at full grain: the store carries
# 4 distinct h_prefix vectors over 142,000 rows (258 rows / 0.18% differ from the reference; only 3
# distinct vectors among them), mutually within cosine 0.99989, under a UNIFORM prefix_end of 23.
# That residual is capture-time numerical noise, not context-driven variation, so the arm stays
# degenerate — but the assert must encode the invariant that is actually true while still failing
# loud on a store whose prefixes genuinely vary with context.
PREFIX_MAX_VARIANTS = 8  # distinct h_prefix vectors tolerated store-wide (measured: 4)
PREFIX_MIN_COS = 0.999  # min cosine of any variant to the reference (measured: 0.99988980)

VM_DENSE_DIR = Path("/mnt/eps-data/thomasjiralerspong/issue1482_saedense/dense")
# Phase U runs on the VM against the MAIN repo root (the worktree is sparse and the npz is
# gitignored there) — absolute path per the plan's Phase U spec.
VM_COVARIATES_DIR = Path(
    "/home/thomasjiralerspong/explore-persona-space/eval_results/issue_1482/predictor_battery"
)
DENSE_INPUT_FILES = (
    "Y_L19.f32.mm",
    "row_ids.npy",
    "row_ci.npy",
    "filled.npy",
    "dense_targets_meta.json",
)
COVARIATE_INPUT_FILES = (
    "fullwidth_covariates_v2.npz",
    "fullwidth_covariates_v2.provenance.json",
    "projvar_fix_impact.json",
    "fullwidth_covariates_legend.json",
)

PLANNED_H = {  # plan section 9 per-component planned_wall_h (pilot-gate references)
    "census": 0.5,
    "fit_maps": 1.2,
    "read_ladder": 0.75,
    "knn": 0.5,
    "carried": 0.5,
    "answer_matchedn": 0.25,
    "partials": 0.25,
    "confirm_b": 1.0,
    "confirm_b_gpu": 1.5,
}
EIGH_VENUE_SWITCH_H = 2.0  # projected CPU eigh wall above this -> GPU cell (plan section 4 P6)


# ── small utilities ───────────────────────────────────────────────────────────


def _now_iso() -> str:
    """UTC ISO-8601 timestamp for result metadata."""
    return datetime.now(UTC).isoformat()


def _meta() -> dict:
    """Reproducibility metadata block (git provenance + env versions + timestamp)."""
    prov = git_provenance(cwd=PROJECT_ROOT)
    return {
        **as_metadata_dict(prov),
        "timestamp_utc": _now_iso(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "torch": torch.__version__,
        "seed": SEED,
    }


def _write_json(path: Path, obj: dict) -> None:
    """Atomic JSON write (tmp + replace) with parent-dir creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    tmp.replace(path)


def _read_json(path: Path) -> dict:
    """Load a JSON file (fail-loud on absence)."""
    return json.loads(path.read_text())


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    """Streaming sha256 of a file (content-identity pin for Phase U, reuse check (f))."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def _device(args) -> torch.device:
    """Resolve --device (auto prefers CUDA when available)."""
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def _out(args) -> Path:
    """Phase output root (<work>/out)."""
    return Path(args.work) / "out"


def _results(args) -> Path:
    """Result JSON/npz directory (<work>/out/results)."""
    return _out(args) / "results"


def _assembled(args) -> Path:
    """Census-assembled CSR + registry directory (<work>/assembled)."""
    return Path(args.work) / "assembled"


def _stage_root(args) -> Path:
    """Mirror root for stage_hub_prefix (files land at <root>/<repo-relative path>)."""
    return Path(args.work) / "staged"


def _staged(args, prefix: str) -> Path:
    """Where a staged HF prefix lands under the mirror root (the parent's arithmetic)."""
    root = _stage_root(args)
    out = root / prefix
    assert out.is_relative_to(root), (out, root)
    return out


def _store_dir(args) -> Path:
    """Pooled-shard store directory (staged or --local-store)."""
    return Path(args.local_store) if args.local_store else _staged(args, STORE_PREFIX)


def _inputs_1482_dir(args) -> Path:
    """Parent #1482 inputs directory (X_dense/order/which/f_out; staged or --local-inputs)."""
    return Path(args.local_inputs) if args.local_inputs else _staged(args, INPUTS_1482_PREFIX)


def _meta_dir(args) -> Path:
    """Scratch-meta directory (split_indices/row_ci/prov; staged or --local-meta)."""
    return Path(args.local_meta) if args.local_meta else _staged(args, SCRATCH_META_PREFIX)


def _dense_dir(args) -> Path:
    """Dense-targets directory (Y_L19 + meta; staged or --local-dense)."""
    return Path(args.local_dense) if args.local_dense else _staged(args, INPUTS_PREFIX)


def _cov_dir(args) -> Path:
    """Covariates-v2 directory (staged or --local-covariates)."""
    return Path(args.local_covariates) if args.local_covariates else _staged(args, INPUTS_PREFIX)


def _hf_base(args) -> str:
    """HF output prefix base (smoke runs divert to a *_smoke sibling; inputs never divert)."""
    if args.hf_out_prefix:
        return str(args.hf_out_prefix)
    return f"{OUT_PREFIX}_smoke" if args.smoke else OUT_PREFIX


def _regime(args, extra: dict | None = None) -> dict:
    """Every output-affecting regime key, for resume-sentinel pinning (#722 r3)."""
    reg = {
        "seed": SEED,
        "smoke": bool(args.smoke),
        "smoke_train": int(args.smoke_train),
        "smoke_val": int(args.smoke_val),
        "smoke_holdout": int(args.smoke_holdout),
        "max_shards": int(args.max_shards),
        "floor_last": int(args.floor_last),
        "cap_db": int(args.cap_db),
        "n_draws_pairing": int(args.n_draws_pairing),
        "n_draws_strat": int(args.n_draws_strat),
        "matchedn_features": int(args.matchedn_features),
        "matchedn_draws": int(args.matchedn_draws),
        "knn_probes": int(args.knn_probes),
    }
    if extra:
        reg.update(extra)
    return reg


def _resume_ok(args, sentinel: Path, regime: dict, phase: str) -> bool:
    """True when a done-sentinel exists AND its regime matches (mismatch fails loud)."""
    if args.no_resume or not sentinel.exists():
        return False
    prior = _read_json(sentinel).get("regime", {})
    if prior != regime:
        raise SystemExit(
            f"[{phase}] resume regime mismatch at {sentinel}:\n prior={prior}\n now ={regime}\n"
            "-> rerun with --no-resume or a fresh --work"
        )
    logger.info("[%s] resume: %s matches regime — skipping", phase, sentinel.name)
    return True


def _pilot_gate(
    args, phase: str, unit_s: float, multiplier: float, planned_h: float, parallelism: float = 1.0
) -> float:
    """Shared pilot helper (plan section 9): extrapolate ONE production-shape unit and abort.

    Logs the canonical `pilot:` line; when the projection exceeds 2x the planned wall on a
    NON-smoke run, writes pilot_gate_report.json, best-effort uploads it, and exits rc=7 (the
    designed artifact-routed halt, #1415). Returns projected hours.
    """
    projected_h = unit_s * multiplier / max(parallelism, 1e-9) / 3600.0
    logger.info(
        "pilot: %s unit_s=%.2f projected_h=%.3f planned_h=%.2f",
        phase,
        unit_s,
        projected_h,
        planned_h,
    )
    if projected_h > 2.0 * planned_h and not args.smoke:
        report = {
            "phase": phase,
            "unit_s": unit_s,
            "multiplier": multiplier,
            "parallelism": parallelism,
            "projected_h": projected_h,
            "planned_h": planned_h,
            "meta": _meta(),
        }
        path = _out(args) / "pilot_gate_report.json"
        _write_json(path, report)
        if not args.skip_upload:
            # UPLOAD_RETURN_DISCARD_EXEMPT: best-effort report push; rc=7 abort proceeds anyway
            hub._upload(
                path,
                HF_DATA_REPO,
                "dataset",
                f"{_hf_base(args)}/results/pilot_gate_report.json",
                upload_as_file=True,
            )
        raise SystemExit(7)
    return projected_h


def _upload_tree(args, local_dir: Path, repo_prefix: str, what: str) -> list[str]:
    """Upload a directory tree file-by-set to HF (one folder commit) + exact-set verify.

    Smoke runs skip files above SMOKE_UPLOAD_CAP_BYTES (logged). Returns uploaded repo paths.
    Fail-loud: a failed upload or a missing verified path raises.
    """
    from huggingface_hub import HfApi

    files = sorted(p for p in local_dir.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    skipped = []
    if args.smoke:
        keep = []
        for p in files:
            if p.stat().st_size > SMOKE_UPLOAD_CAP_BYTES:
                skipped.append(str(p.relative_to(local_dir)))
            else:
                keep.append(p)
        files = keep
        if skipped:
            logger.info("[upload:%s] smoke cap skips %d files: %s", what, len(skipped), skipped)
    if not files:
        logger.info("[upload:%s] nothing to upload from %s", what, local_dir)
        return []
    expected = [f"{repo_prefix}/{p.relative_to(local_dir)}" for p in files]
    api = HfApi()
    missing = hub.verify_repo_paths_uploaded(
        api, HF_DATA_REPO, expected, path_in_repo=repo_prefix, repo_type="dataset"
    )
    if not missing:
        logger.info("[upload:%s] all %d files already on HF at %s", what, len(files), repo_prefix)
        return expected
    if skipped:
        # A smoke cap must not silently mask a partial folder upload of the kept set.
        logger.info("[upload:%s] uploading kept set only (smoke cap active)", what)
        for p, rp in zip(files, expected, strict=True):
            # UPLOAD_LOOP_EXEMPT: smoke-only kept-set of small files; fail-loud per file
            out = hub._upload(p, HF_DATA_REPO, "dataset", rp, upload_as_file=True)
            if not out:
                raise SystemExit(f"[upload:{what}] upload returned no path for {rp}")
    else:
        out = hub._upload(local_dir, HF_DATA_REPO, "dataset", repo_prefix)
        if not out:
            raise SystemExit(f"[upload:{what}] folder upload returned no path for {repo_prefix}")
    still = hub.verify_repo_paths_uploaded(
        api, HF_DATA_REPO, expected, path_in_repo=repo_prefix, repo_type="dataset"
    )
    if still:
        raise SystemExit(f"[upload:{what}] {len(still)} paths missing after upload: {still[:5]}")
    logger.info("[upload:%s] %d files verified at %s", what, len(files), repo_prefix)
    return expected


def _upload_phase_results(args, what: str) -> None:
    """Idempotent per-phase results upload (plan section 9 'result files upload as produced').

    Verify-first via _upload_tree (already-landed files are skipped), so a pod loss between
    fit-maps and upload-verify forfeits at most the in-flight phase's recompute — not every
    result file since fit-maps (round-1 Minor: per-phase result uploads).
    """
    if args.skip_upload:
        return
    resd = _results(args)
    if not resd.exists():
        return
    _upload_tree(args, resd, f"{_hf_base(args)}/results", what)


def _upload_panel_block(args) -> list[str]:
    """Upload the plan-declared 16,384-panel B sub-block (plan section 4 Phase 6 persistence).

    Shared by phase_upload_verify (CPU path), phase_confirm_b (post-fit), and
    phase_confirm_b_gpu (venue-switch path) so the panel block cannot be lost when the GPU
    cell terminates without ever running upload-verify (round-1 Issue:
    gpu-cell-panel-block-not-uploaded). Verify-first + fail-loud; smoke size cap honored.
    """
    pb = _out(args) / "B_panel_block.f32.npy"
    if not pb.exists():
        return []
    if args.smoke and pb.stat().st_size > SMOKE_UPLOAD_CAP_BYTES:
        logger.info("[upload:B_panel] smoke cap skips %s", pb.name)
        return []
    from huggingface_hub import HfApi

    rp = f"{_hf_base(args)}/analysis_tensors/B_panel/B_panel_block.f32.npy"
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), HF_DATA_REPO, [rp], path_in_repo=rp.rsplit("/", 1)[0], repo_type="dataset"
    )
    if not missing:
        logger.info("[upload:B_panel] already on HF at %s", rp)
        return [rp]
    out = hub._upload(pb, HF_DATA_REPO, "dataset", rp, upload_as_file=True)
    if not out:
        raise SystemExit(f"[upload:B_panel] upload returned no path for {rp}")
    logger.info("[upload:B_panel] verified at %s", rp)
    return [rp]


# ── phase U: VM-side input uploads ────────────────────────────────────────────


def phase_upload_inputs(args) -> int:
    """Phase U (VM only): upload dense targets + covariates v2 to the HF inputs prefix.

    Idempotent (verify-first). Pins content identity of the banked X_dense.f32.mm against the
    VM-local X_L19.f32.mm via sha256 vs the Hub LFS sha (artifact-reuse check (f)).
    """
    from huggingface_hub import HfApi

    api = HfApi()
    dense_src = Path(args.upload_dense_src) if args.upload_dense_src else VM_DENSE_DIR
    cov_src = Path(args.upload_cov_src) if args.upload_cov_src else VM_COVARIATES_DIR
    plan = [(dense_src / n, f"{INPUTS_PREFIX}/{n}") for n in DENSE_INPUT_FILES]
    plan += [(cov_src / n, f"{INPUTS_PREFIX}/{n}") for n in COVARIATE_INPUT_FILES]
    for local, _ in plan:
        if not local.exists():
            raise SystemExit(f"[upload-inputs] missing source file: {local}")

    expected = [rp for _, rp in plan]
    missing = set(
        hub.verify_repo_paths_uploaded(
            api, HF_DATA_REPO, expected, path_in_repo=INPUTS_PREFIX, repo_type="dataset"
        )
    )
    for local, rp in plan:
        if rp not in missing:
            logger.info("[upload-inputs] already on HF: %s", rp)
            continue
        logger.info("[upload-inputs] uploading %s (%.1f MB)", rp, local.stat().st_size / 1e6)
        # UPLOAD_LOOP_EXEMPT: fixed ~9-file Phase-U inputs list; verify-first idempotent
        out = hub._upload(local, HF_DATA_REPO, "dataset", rp, upload_as_file=True)
        if not out:
            raise SystemExit(f"[upload-inputs] upload returned no path for {rp}")
    still = hub.verify_repo_paths_uploaded(
        api, HF_DATA_REPO, expected, path_in_repo=INPUTS_PREFIX, repo_type="dataset"
    )
    if still:
        raise SystemExit(f"[upload-inputs] missing after upload: {still}")

    # Content-identity pin: local X_L19.f32.mm == HF-banked inputs/X_dense.f32.mm (check (f)).
    x_local = dense_src / "X_L19.f32.mm"
    info = hub.retry_transient(
        lambda: api.get_paths_info(
            HF_DATA_REPO, [f"{INPUTS_1482_PREFIX}/X_dense.f32.mm"], repo_type="dataset"
        ),
        what="X_dense.f32.mm paths_info",
    )
    assert len(info) == 1 and info[0].lfs is not None, info
    hub_sha = info[0].lfs.sha256
    local_sha = _sha256(x_local)
    if hub_sha != local_sha:
        raise SystemExit(
            f"[upload-inputs] X content pin FAILED: local {x_local} sha256={local_sha} != "
            f"HF {INPUTS_1482_PREFIX}/X_dense.f32.mm sha256={hub_sha}"
        )
    logger.info("[upload-inputs] X content pin OK (sha256=%s)", local_sha[:16])
    _write_json(
        _out(args) / "upload_inputs_report.json",
        {
            "uploaded": expected,
            "x_content_pin_sha256": local_sha,
            "meta": _meta(),
        },
    )
    return 0


# ── phase stage ───────────────────────────────────────────────────────────────


def phase_stage(args) -> int:
    """Phase 0a: stage store + parent inputs + scratch meta + own inputs + SAE + perfeature.

    Runs the (h)(iv) 1-file staging probes (entry file per source family opened by its
    production consumer) BEFORE the bulk pulls. Local --local-* overrides skip that family.
    """
    assert_out_root_headroom(Path(args.work), 4 if args.smoke else 25, phase="stage")
    root = _stage_root(args)
    counts: dict[str, int] = {}

    # (h)(iv) probes: one KB-scale entry file per HF-staged family, opened by its consumer.
    if not args.local_inputs:
        p = hub.stage_hub_file(
            HF_DATA_REPO,
            f"{INPUTS_1482_PREFIX}/densesae_inputs_meta.json",
            _staged(args, INPUTS_1482_PREFIX) / "densesae_inputs_meta.json",
            repo_type="dataset",
        )
        json.loads(Path(p).read_text())
    if not args.local_dense:
        p = hub.stage_hub_file(
            HF_DATA_REPO,
            f"{INPUTS_PREFIX}/dense_targets_meta.json",
            _staged(args, INPUTS_PREFIX) / "dense_targets_meta.json",
            repo_type="dataset",
        )
        json.loads(Path(p).read_text())

    # Bulk prefixes (mirror-root layout; consumer opens the exact fetch destinations — no
    # staging transformation, so (h)(iv) is satisfied by the probes above + np.load below).
    if args.local_store:
        counts["store"] = -1
        logger.info("[stage] store: using --local-store %s", args.local_store)
    else:
        paths = hub.stage_hub_prefix(HF_DATA_REPO, STORE_PREFIX, root, repo_type="dataset")
        counts["store"] = len(paths)
    if args.local_inputs:
        counts["inputs_1482"] = -1
    else:
        paths = hub.stage_hub_prefix(HF_DATA_REPO, INPUTS_1482_PREFIX, root, repo_type="dataset")
        counts["inputs_1482"] = len(paths)
    if args.local_dense and args.local_covariates:
        counts["inputs_2163"] = -1
    else:
        paths = hub.stage_hub_prefix(HF_DATA_REPO, INPUTS_PREFIX, root, repo_type="dataset")
        counts["inputs_2163"] = len(paths)

    # Scratch meta via the parent helper (reused, not rewritten — plan Phase 0).
    if args.local_meta:
        counts["scratch_meta"] = -1
    else:
        from issue1482_early_layer import _stage_scratch_meta

        _stage_scratch_meta(SimpleNamespace(scratch=_meta_dir(args)))
        counts["scratch_meta"] = 3

    # Parent per-feature parity npz (Phase-4 join + Phase-1 per-feature gate).
    dest = _staged(args, PERFEATURE_1482_PATH.rsplit("/", 1)[0]) / "ridge__mean_perfeature.npz"
    hub.stage_hub_file(HF_DATA_REPO, PERFEATURE_1482_PATH, dest, repo_type="dataset")
    np.load(dest)  # consumer-open probe
    counts["perfeature_1482"] = 1

    # SAE weights (only W_dec/b_dec consumed) via the pinned loader.
    import issue1482_sae as SAE

    sae_cache = Path(args.work) / "sae_cache"
    SAE.BatchTopKSAE.ensure_downloaded(64, sae_cache, layer=19)
    counts["sae"] = 1

    # One-shard consumer probe (store family): open the first staged/local shard.
    shard0 = sorted(_store_dir(args).glob("pooled_*.npz"))
    if not shard0:
        raise SystemExit(f"[stage] no pooled shards under {_store_dir(args)}")
    with np.load(shard0[0]) as z:
        assert "psil_idx" in z.files and "row_idx" in z.files, sorted(z.files)

    _write_json(_out(args) / "stage_report.json", {"counts": counts, "root": str(root)})
    logger.info("[stage] done: %s", counts)
    return 0


# ── registry + design loading ─────────────────────────────────────────────────


def _registry(args) -> dict:
    """Row registry (order/which/f_out) with coverage-aware smoke selection.

    Production: all 142,000 rows. Smoke: deterministic per-split sample RESTRICTED to rows
    covered by the staged shard subset (assembled/coverage_rows.npy, written by census), so the
    smoke fit sees real nonzero targets. Same output shape as the parent's `_load_registry`.
    """
    d = _inputs_1482_dir(args)
    order = np.load(d / "order.npy")
    which = np.load(d / "which.npy")
    f_out = np.load(d / "f_out.npy")
    assert order.shape == which.shape, (order.shape, which.shape)
    n_full = int(order.shape[0])
    keep = np.arange(n_full)
    if args.smoke:
        cov_path = _assembled(args) / "coverage_rows.npy"
        if not cov_path.exists():
            raise SystemExit("[registry] smoke needs census first (coverage_rows.npy missing)")
        covered = np.zeros(n_full, dtype=bool)
        covered[np.load(cov_path)] = True
        rng = np.random.default_rng(SEED)
        want = {0: args.smoke_holdout, 1: args.smoke_train, 2: args.smoke_val}
        picks = []
        for code, k in want.items():
            idx = np.where((which == code) & covered)[0]
            take = min(int(k), len(idx))
            if take < 2:
                raise SystemExit(f"[registry] smoke split {code}: {len(idx)} covered rows (<2)")
            picks.append(rng.choice(idx, size=take, replace=False))
        keep = np.sort(np.concatenate(picks))
    order_k, which_k = order[keep], which[keep]
    reg = {
        "n_full": n_full,
        "n": int(len(keep)),
        "keep": keep,
        "order": order_k,
        "which": which_k,
        "ho": np.where(which_k == 0)[0].astype(np.int64),
        "tr": np.where(which_k == 1)[0].astype(np.int64),
        "va": np.where(which_k == 2)[0].astype(np.int64),
        "f_out": f_out,
    }
    for k in ("tr", "va", "ho"):
        if len(reg[k]) == 0:
            raise SystemExit(f"[registry] empty split '{k}' (n={reg['n']})")
    return reg


def _load_design(args, reg: dict) -> np.ndarray:
    """X design memmap (parent semantics: full memmap, or contiguous keep-subset in smoke)."""
    path = _inputs_1482_dir(args) / "X_dense.f32.mm"
    expect = reg["n_full"] * H_DIM * 4
    got = path.stat().st_size
    if got != expect:
        raise SystemExit(f"[design] {path} is {got} bytes, expected {expect}")
    full = np.memmap(path, dtype=np.float32, mode="r", shape=(reg["n_full"], H_DIM))
    if reg["n"] == reg["n_full"]:
        return full
    return np.ascontiguousarray(full[reg["keep"]])


def _load_dense_targets(args, reg: dict) -> np.ndarray:
    """Y_L19 dense targets aligned to the registry rows.

    row_ids.npy is asserted EXACTLY equal to order.npy (verified on the real artifact), so the
    dense matrices share the CSR row order; smoke subsets via reg["keep"].
    """
    d = _dense_dir(args)
    row_ids = np.load(d / "row_ids.npy")
    filled = np.load(d / "filled.npy")
    order_full = np.load(_inputs_1482_dir(args) / "order.npy")
    if not np.array_equal(row_ids, order_full):
        raise SystemExit("[dense] row_ids.npy != order.npy — dense targets misaligned")
    if not bool(filled.all()):
        raise SystemExit(f"[dense] {int((~filled).sum())} unfilled dense rows")
    meta = _read_json(d / "dense_targets_meta.json")
    assert meta["fields"]["X_L19.f32.mm"] == "cx_last@L19", meta["fields"]
    assert meta["fields"]["Y_L19.f32.mm"] == "v_x@L19 (mean-response)", meta["fields"]
    path = d / "Y_L19.f32.mm"
    expect = reg["n_full"] * H_DIM * 4
    if path.stat().st_size != expect:
        raise SystemExit(f"[dense] {path} is {path.stat().st_size} bytes, expected {expect}")
    full = np.memmap(path, dtype=np.float32, mode="r", shape=(reg["n_full"], H_DIM))
    if reg["n"] == reg["n_full"]:
        return full
    return np.ascontiguousarray(full[reg["keep"]])


def _prov_per_row(args, reg: dict) -> np.ndarray:
    """Per-registry-row corpus label (0 = LMSYS, 1 = WildChat) via prov[order] fancy-indexing."""
    prov = np.load(_meta_dir(args) / "prov.npy")
    return np.asarray(prov[reg["order"]], dtype=np.int64)


# ── phase census ──────────────────────────────────────────────────────────────


def _shard_paths(args) -> list[Path]:
    """Sorted pooled shard paths, optionally truncated by --max-shards (smoke)."""
    paths = sorted(_store_dir(args).glob("pooled_*.npz"))
    if not paths:
        raise SystemExit(f"[census] no pooled shards under {_store_dir(args)}")
    return paths[: args.max_shards] if args.max_shards else paths


def phase_census(args) -> int:
    """Phase 0b: one-pass (x2) shard sweep -> Psi_last CSR + answer-mean CSR + census.

    Also: corpus-join pins (full grain, independent of shard subset), prefix-degeneracy
    re-assert over the staged shards, last-token covariates, and the census restriction
    (floor + cap) for Phase 6.
    """
    asm = _assembled(args)
    asm.mkdir(parents=True, exist_ok=True)
    sentinel = _out(args) / "census.done.json"
    regime = _regime(args)
    if _resume_ok(args, sentinel, regime, "census"):
        return 0
    assert_out_root_headroom(Path(args.work), 2 if args.smoke else 10, phase="census")

    d = _inputs_1482_dir(args)
    order = np.load(d / "order.npy")
    which = np.load(d / "which.npy")
    n_rows = int(order.shape[0])

    # Corpus-join pins at FULL grain (plan Phase 0; drift guards, pre-verified at plan time).
    meta_dir = _meta_dir(args)
    prov = np.load(meta_dir / "prov.npy")
    row_ci = np.load(meta_dir / "row_ci.npy")
    with np.load(meta_dir / "split_indices.npz") as si:
        cat = np.concatenate([si["holdout"], si["sae_fit"], si["sae_val"]])
    if not np.array_equal(order, cat):
        raise SystemExit("[census] order != concat(split_indices[holdout,sae_fit,sae_val])")
    joined_ci = row_ci[order]
    if int((joined_ci < 0).sum()) != 0:
        raise SystemExit(f"[census] {int((joined_ci < 0).sum())} joined rows hit sentinel row_ci")
    ho_mask_full = which == 0
    n_ho_full = int(ho_mask_full.sum())
    if not args.smoke and n_ho_full != 20_000:
        raise SystemExit(f"[census] joined holdout {n_ho_full} != 20000")
    lmsys_frac_ho = float((prov[order[ho_mask_full]] == 0).mean())
    if abs(lmsys_frac_ho - 0.5498) > 0.02:
        raise SystemExit(f"[census] holdout lmsys_frac {lmsys_frac_ho:.4f} off 0.5498 by >0.02")
    logger.info("[census] corpus-join pins OK (holdout lmsys_frac=%.4f)", lmsys_frac_ho)

    # Row-id -> CSR-row position lookup over the 964,844-row universe.
    pos = np.full(int(prov.shape[0]), -1, dtype=np.int64)
    pos[order] = np.arange(n_rows, dtype=np.int64)

    shards = _shard_paths(args)
    n_shards = len(shards)

    # Pass 1: per-row lengths for both CSRs + coverage.
    psil_len = np.zeros(n_rows, dtype=np.int64)
    ans_len = np.zeros(n_rows, dtype=np.int64)
    covered = np.zeros(n_rows, dtype=bool)
    t0 = time.time()
    for si_, path in enumerate(shards):
        with np.load(path) as z:
            rp = pos[z["row_idx"]]
            assert (rp >= 0).all(), f"[census] {path.name}: shard row ids outside the registry"
            assert not covered[rp].any(), f"[census] {path.name}: duplicate rows across shards"
            covered[rp] = True
            psil_len[rp] = z["psil_off"]
            ans_len[rp] = z["idx_off"]
        if si_ == min(49, n_shards - 1):
            unit_s = time.time() - t0
            # Pass-1 is roughly half the per-shard work of pass-2; x2 covers both passes.
            _pilot_gate(args, "census", unit_s, 2.0 * n_shards / (si_ + 1), PLANNED_H["census"])
        if (si_ + 1) % 200 == 0 or si_ + 1 == n_shards:
            logger.info(
                "[census] pass1 shard %d/%d elapsed=%.0fs", si_ + 1, n_shards, time.time() - t0
            )
    np.save(asm / "coverage_rows.npy", np.where(covered)[0])
    n_cov = int(covered.sum())
    if not args.max_shards and n_cov != n_rows:
        raise SystemExit(f"[census] full store covers {n_cov}/{n_rows} rows")

    psil_indptr = np.concatenate(([0], np.cumsum(psil_len))).astype(np.int64)
    ans_indptr = np.concatenate(([0], np.cumsum(ans_len))).astype(np.int64)
    nnz_l, nnz_a = int(psil_indptr[-1]), int(ans_indptr[-1])
    logger.info("[census] nnz: psil=%d (%.1f/row) ans=%d", nnz_l, nnz_l / max(n_cov, 1), nnz_a)

    psil_idx = np.lib.format.open_memmap(
        asm / "psil_indices.i32.npy", mode="w+", dtype=np.int32, shape=(nnz_l,)
    )
    psil_val = np.lib.format.open_memmap(
        asm / "psil_val.f32.npy", mode="w+", dtype=np.float32, shape=(nnz_l,)
    )
    y_indices = np.memmap(asm / "y_indices.i32", mode="w+", dtype=np.int32, shape=(nnz_a,))
    y_val = np.memmap(asm / "y_val_mean.f16", mode="w+", dtype=np.float16, shape=(nnz_a,))

    # Census accumulators (per feature).
    tr_cnt = np.zeros(DICT_SIZE, dtype=np.int64)  # last-token active counts, train rows
    ho_cnt = np.zeros(DICT_SIZE, dtype=np.int64)
    all_cnt = np.zeros(DICT_SIZE, dtype=np.int64)
    tr_sum = np.zeros(DICT_SIZE, dtype=np.float64)  # last-token value sums, train rows
    tr_sum2 = np.zeros(DICT_SIZE, dtype=np.float64)
    ho_sum2 = np.zeros(DICT_SIZE, dtype=np.float64)  # for C_j: mean(psi^2 | active, holdout)
    psi_cnt_tr = np.zeros(DICT_SIZE, dtype=np.int64)  # pooled (psi) counts, train rows

    which_by_pos = which  # split code per CSR row
    href_ref: np.ndarray | None = None
    href_ref64: np.ndarray | None = None
    href_ref_norm = 0.0
    href_variant_keys: set[bytes] = set()  # distinct h_prefix vectors seen (incl. the reference)
    href_n_deviating = 0  # rows not byte-identical to the reference
    href_min_cos = 1.0  # min cosine of any deviating row to the reference
    prefix_end_values: set[int] = set()
    t0 = time.time()
    for si_, path in enumerate(shards):
        with np.load(path) as z:
            rp = pos[z["row_idx"]]
            w = which_by_pos[rp]
            # Prefix-degeneracy re-assert (CORRECTED invariant, plan v6 §12 A11): the store may
            # carry a few numerically-indistinguishable h_prefix variants (capture-time noise);
            # it must NOT carry prefixes that vary with context. Measure, never assume.
            hp = np.asarray(z["h_prefix"], dtype=np.float16)
            if href_ref is None:
                href_ref = hp[0].copy()
                href_ref64 = href_ref.astype(np.float64)
                href_ref_norm = float(np.linalg.norm(href_ref64))
                href_variant_keys.add(href_ref.tobytes())
            prefix_end_values.update(int(v) for v in np.unique(z["prefix_end"]))
            neq = ~np.all(hp == href_ref[None, :], axis=1)
            n_neq = int(neq.sum())
            if n_neq:
                href_n_deviating += n_neq
                # Dedupe first, then measure per VARIANT — bounded work even if many rows deviate.
                for variant in np.unique(hp[neq], axis=0):
                    href_variant_keys.add(variant.tobytes())
                    row = variant.astype(np.float64)
                    denom = float(np.linalg.norm(row)) * href_ref_norm
                    cos = float(row @ href_ref64) / denom if denom > 0 else 0.0
                    href_min_cos = min(href_min_cos, cos)
                if len(href_variant_keys) > PREFIX_MAX_VARIANTS:
                    raise SystemExit(
                        "[census] prefix degeneracy violated: "
                        f"{len(href_variant_keys)} distinct h_prefix vectors "
                        f"exceeds PREFIX_MAX_VARIANTS={PREFIX_MAX_VARIANTS} "
                        f"(shard {si_ + 1}/{n_shards}) — prefixes appear to vary with context, "
                        "so the plan's stated prefix-arm deviation no longer holds"
                    )
            # Last-token CSR fill (per-row offsets from lengths).
            off = np.concatenate(([0], np.cumsum(z["psil_off"]))).astype(np.int64)
            idx_flat = np.asarray(z["psil_idx"], dtype=np.int32)
            val_flat = np.asarray(z["psil_val"], dtype=np.float32)
            starts = psil_indptr[rp]
            lens = np.asarray(z["psil_off"], dtype=np.int64)
            take_dst = np.repeat(starts, lens) + (
                np.arange(len(idx_flat), dtype=np.int64) - np.repeat(off[:-1], lens)
            )
            psil_idx[take_dst] = idx_flat
            psil_val[take_dst] = val_flat
            # Census counts.
            rep_w = np.repeat(w, lens)
            v64 = val_flat.astype(np.float64)
            np.add.at(all_cnt, idx_flat, 1)
            tr_m = rep_w == 1
            ho_m = rep_w == 0
            np.add.at(tr_cnt, idx_flat[tr_m], 1)
            np.add.at(ho_cnt, idx_flat[ho_m], 1)
            np.add.at(tr_sum, idx_flat[tr_m], v64[tr_m])
            np.add.at(tr_sum2, idx_flat[tr_m], v64[tr_m] ** 2)
            np.add.at(ho_sum2, idx_flat[ho_m], v64[ho_m] ** 2)
            # Pooled (psi) train counts for span_last_ratio.
            plens = np.asarray(z["psi_off"], dtype=np.int64)
            p_idx = np.asarray(z["psi_idx"], dtype=np.int32)
            p_tr = np.repeat(w, plens) == 1
            np.add.at(psi_cnt_tr, p_idx[p_tr], 1)
            # Answer-mean CSR fill.
            a_off = np.asarray(z["idx_off"], dtype=np.int64)
            a_idx = np.asarray(z["ans_idx"], dtype=np.int32)
            a_val = np.asarray(z["ans_mean"], dtype=np.float16)
            a_cum = np.concatenate(([0], np.cumsum(a_off))).astype(np.int64)
            a_dst = np.repeat(ans_indptr[rp], a_off) + (
                np.arange(len(a_idx), dtype=np.int64) - np.repeat(a_cum[:-1], a_off)
            )
            y_indices[a_dst] = a_idx
            y_val[a_dst] = a_val
        if (si_ + 1) % 200 == 0 or si_ + 1 == n_shards:
            logger.info(
                "[census] pass2 shard %d/%d elapsed=%.0fs", si_ + 1, n_shards, time.time() - t0
            )
    n_prefix_variants = len(href_variant_keys)
    if n_prefix_variants > PREFIX_MAX_VARIANTS or href_min_cos < PREFIX_MIN_COS:
        raise SystemExit(
            f"[census] prefix degeneracy violated: {n_prefix_variants} distinct h_prefix vectors "
            f"(cap {PREFIX_MAX_VARIANTS}), min cosine to reference {href_min_cos:.8f} "
            f"(floor {PREFIX_MIN_COS}) over {href_n_deviating} deviating rows — the prefix appears "
            "to vary with context, so the plan's stated prefix-arm deviation no longer holds"
        )
    if len(prefix_end_values) != 1:
        raise SystemExit(
            f"[census] prefix degeneracy violated: prefix_end is not uniform "
            f"({sorted(prefix_end_values)}) — the prefix TEXT differs across rows, which the "
            "stated prefix-arm deviation assumes it does not"
        )
    logger.info(
        "[census] prefix degeneracy OK: %d distinct h_prefix vectors, %d deviating rows, "
        "min cosine %.8f, prefix_end=%s",
        n_prefix_variants,
        href_n_deviating,
        href_min_cos,
        sorted(prefix_end_values),
    )

    np.save(asm / "psil_indptr.npy", psil_indptr)
    np.save(asm / "y_indptr.npy", ans_indptr)
    y_indices.flush()
    y_val.flush()
    _write_json(
        asm / "ystore_meta.json",
        {"n_rows": n_rows, "nnz": nnz_a, "poolings": ["mean"], "source_shards": n_shards},
    )

    # Row-nnz distribution + measured-constant asserts (mean ~33, max 61; NOT 64).
    lens_cov = psil_len[covered]
    row_stats = {
        "mean": float(lens_cov.mean()),
        "median": float(np.median(lens_cov)),
        "p5": float(np.percentile(lens_cov, 5)),
        "p95": float(np.percentile(lens_cov, 95)),
        "max": int(lens_cov.max()),
        "min": int(lens_cov.min()),
    }
    if not (10.0 <= row_stats["mean"] <= 64.0):
        raise SystemExit(f"[census] psil mean nnz/row {row_stats['mean']:.1f} outside [10, 64]")
    if row_stats["max"] > 128:
        raise SystemExit(f"[census] psil max nnz/row {row_stats['max']} > 128")

    # Last-token covariates (Phase 5). Never-active features are ZERO-imputed on the
    # conditional-mean + ratio columns (documented; keeps the complete-case set v2-driven).
    lasttoken_count = tr_cnt.astype(np.float64)
    mean_act = np.zeros(DICT_SIZE, dtype=np.float64)
    m = tr_cnt > 0
    mean_act[m] = tr_sum[m] / tr_cnt[m]
    span_last_ratio = np.zeros(DICT_SIZE, dtype=np.float64)
    mm = psi_cnt_tr > 0
    span_last_ratio[mm] = tr_cnt[mm] / psi_cnt_tr[mm]

    # Census restriction (Phase 6): floor on train-active counts, cap by descending count,
    # deterministic tie-break by feature id (lexsort; argsort tie order is machine-dependent).
    floor = 2 if args.smoke else args.floor_last
    eligible = np.where(tr_cnt >= floor)[0]
    order_sel = np.lexsort((eligible, -tr_cnt[eligible]))
    restrict = eligible[order_sel][: args.cap_db]
    restrict_sorted = np.sort(restrict)
    np.save(asm / "restrict_ids.npy", restrict_sorted)

    np.savez(
        asm / "census.npz",
        feat_ids=np.arange(DICT_SIZE, dtype=np.int32),
        tr_count=tr_cnt,
        ho_count=ho_cnt,
        all_count=all_cnt,
        tr_sum=tr_sum,
        tr_sum2=tr_sum2,
        ho_sum2=ho_sum2,
        psi_count_tr=psi_cnt_tr,
        lasttoken_count=lasttoken_count,
        mean_act_when_active_lasttoken=mean_act,
        span_last_ratio=span_last_ratio,
    )
    census = {
        "n_rows": n_rows,
        "n_covered_rows": n_cov,
        "n_shards": n_shards,
        "nnz_psil": nnz_l,
        "nnz_ans_mean": nnz_a,
        "row_nnz_stats_psil": row_stats,
        "coverage_frac_rows_with_code": float((psil_len[covered] > 0).mean()),
        "n_distinct_features_lasttoken": int((all_cnt > 0).sum()),
        "n_distinct_features_lasttoken_train": int((tr_cnt > 0).sum()),
        "restriction": {
            "floor_last": floor,
            "cap": int(args.cap_db),
            "n_eligible": int(len(eligible)),
            "d_B": int(len(restrict_sorted)),
            "excluded_count": int(DICT_SIZE - len(restrict_sorted)),
            "excluded_active_mass_frac": float(
                1.0 - tr_cnt[restrict_sorted].sum() / max(tr_cnt.sum(), 1)
            ),
        },
        "corpus_join": {
            "holdout_n": n_ho_full,
            "holdout_lmsys_frac": lmsys_frac_ho,
            "holdout_lmsys_n": int((prov[order[ho_mask_full]] == 0).sum()),
            "holdout_wildchat_n": int((prov[order[ho_mask_full]] == 1).sum()),
            "universe_lmsys_frac_ref": 0.5498,
            "n_sentinel_row_ci_hits": 0,
        },
        # REALIZED values, never a hardcoded constant: the clean-result must state the measured
        # prefix-arm degeneracy (plan v6 §4 stated deviation + §12 A11), not an assumed "1".
        "prefix_degeneracy": {
            "n_distinct_h_prefix_vectors": n_prefix_variants,
            "n_rows_deviating_from_reference": href_n_deviating,
            "min_cosine_to_reference": href_min_cos,
            "prefix_end_values": sorted(prefix_end_values),
            "max_variants_allowed": PREFIX_MAX_VARIANTS,
            "min_cosine_floor": PREFIX_MIN_COS,
        },
        "lasttoken_covariates": {
            "mean_act_zero_imputed_features": int((~m).sum()),
            "span_ratio_zero_imputed_features": int((~mm).sum()),
            "definitions": {
                "lasttoken_count": "train-row last-token active count",
                "mean_act_when_active_lasttoken": "train conditional mean (0 if never active)",
                "span_last_ratio": "train last-token count / train pooled(psi) count (0 if 0/0)",
            },
        },
        "meta": _meta(),
    }
    _write_json(_out(args) / "census.json", census)
    # Plan section 9: census uploads right after Phase 0 (per-phase persistence). Copies
    # mirror phase_upload_verify's results-tree layout; upload precedes the done-sentinel so
    # a failed upload re-runs the (cheap, resumable) phase instead of silently skipping.
    if not args.skip_upload:
        resd = _results(args)
        resd.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_out(args) / "census.json", resd / "census.json")
        if (_assembled(args) / "census.npz").exists():
            shutil.copy2(_assembled(args) / "census.npz", resd / "census.npz")
        _upload_phase_results(args, "results-census")
    _write_json(sentinel, {"regime": regime, "written": _now_iso()})
    logger.info(
        "[census] done: d_B=%d, distinct lasttoken features=%d",
        census["restriction"]["d_B"],
        census["n_distinct_features_lasttoken"],
    )
    return 0


def _ystore(args, reg: dict) -> DSF.YStore:
    """Answer-mean YStore over the assembled CSR arrays (parent class, poolings=("mean",))."""
    meta = _read_json(_assembled(args) / "ystore_meta.json")
    assert meta["n_rows"] == reg["n_full"], (meta["n_rows"], reg["n_full"])
    return DSF.YStore(_assembled(args), meta["n_rows"], meta["nnz"], poolings=("mean",))


def _psil_csr(args, reg: dict) -> sp.csr_matrix:
    """Last-token Psi CSR restricted to the registry rows (fp32 values)."""
    asm = _assembled(args)
    indptr = np.load(asm / "psil_indptr.npy")
    indices = np.load(asm / "psil_indices.i32.npy", mmap_mode="r")
    values = np.load(asm / "psil_val.f32.npy", mmap_mode="r")
    keep = reg["keep"]
    lens = (indptr[keep + 1] - indptr[keep]).astype(np.int64)
    out_indptr = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
    total = int(out_indptr[-1])
    take = np.repeat(indptr[keep], lens) + (
        np.arange(total, dtype=np.int64)
        - np.repeat(np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
    )
    return sp.csr_matrix(
        (np.asarray(values[take], dtype=np.float32), np.asarray(indices[take]), out_indptr),
        shape=(reg["n"], DICT_SIZE),
    )


# ── phase fit-maps ────────────────────────────────────────────────────────────


def _gate(args, name: str, ok: bool, detail: str, gates: list[dict]) -> None:
    """Record a parity gate; HALT on failure in production, demote to report-only at smoke."""
    gates.append({"gate": name, "pass": bool(ok), "detail": detail, "enforced": not args.smoke})
    lvl = logging.INFO if ok else logging.ERROR
    logger.log(lvl, "[gate] %s: %s (%s)", name, "PASS" if ok else "FAIL", detail)
    if not ok and not args.smoke:
        raise SystemExit(f"[gate] {name} FAILED: {detail}")


def _dense_ho_pool(ystore: DSF.YStore, ho: np.ndarray) -> np.ndarray:
    """Dense fp32 (n_ho, DICT_SIZE) holdout answer-target pool (kNN pool; ~10.5 GB at prod)."""
    csc = ystore.csc_rows(ho, "mean")
    out = np.empty((len(ho), DICT_SIZE), dtype=np.float32)
    for c0 in range(0, DICT_SIZE, OUT_BLOCK):
        c1 = min(c0 + OUT_BLOCK, DICT_SIZE)
        out[:, c0:c1] = DSF._dense_cols(csc, c0, c1)
    return out


def _knn_block(pred: np.ndarray, pool: np.ndarray, n_probe: int) -> dict:
    """kNN retrieval (both metrics) on the deterministic head probes vs the full pool."""
    k = min(n_probe, pred.shape[0], pool.shape[0])
    out = {}
    p64 = np.asarray(pred[:k], dtype=np.float64)
    pool64 = np.asarray(pool, dtype=np.float64)
    for metric in ("euclidean", "cosine"):
        out[metric] = knn_retrieval(
            p64,
            pool64[:k],
            ks=KNN_KS,
            metric=metric,
            pool=pool64,
            true_pool_idx=np.arange(k),
        )
    out["n_probes"] = int(k)
    out["chance_at_1"] = float(1.0 / pool.shape[0])
    return out


def phase_fit_maps(args) -> int:
    """Phase 1: refit W (dense->feature) + M (dense->dense) with parity gates; persist bundles.

    Mirrors `fit_ridge`'s exact flow through the SAME imported helpers (`_GramFactor`, `_xty`,
    `_val_block_ss`, `_ridge_holdout`, `_score`) but keeps the factorization to materialize
    P = U diag(1/(s+lambda)) B, shares ONE Gram/eigh across W and M, and gates against the
    banked parity values instead of the parent's `_banked` JSON.
    """
    from issue1738_sae_arm import _GramFactor  # reused VERBATIM (plan section 10)

    outd = _out(args)
    sentinel = outd / "fit_maps.done.json"
    regime = _regime(args)
    if _resume_ok(args, sentinel, regime, "fit-maps"):
        return 0
    assert_out_root_headroom(Path(args.work), 3 if args.smoke else 12, phase="fit-maps")

    dev = _device(args)
    reg = _registry(args)
    tr, va, ho = reg["tr"], reg["va"], reg["ho"]
    if len(tr) < H_DIM and not args.smoke:
        raise SystemExit(f"n_train {len(tr)} < d_in {H_DIM}: estimator-degenerate, refusing")
    logger.info("[fit-maps] n=%d (tr=%d va=%d ho=%d)", reg["n"], len(tr), len(va), len(ho))

    X = _load_design(args, reg)
    ystore = _ystore(args, reg)
    gates: list[dict] = []

    t0 = time.time()
    fac = _GramFactor(X, tr, dev, DSF.GRAM_BLOCK)
    t_gram = time.time() - t0
    logger.info("[fit-maps] Gram+eigh %.0fs (shared across W and M)", t_gram)

    # ---- W fit (mirror of fit_ridge) -----------------------------------------
    s1, _ = ystore.col_stats(tr, "mean")
    ymu_np = s1 / len(tr)
    ymu = torch.as_tensor(ymu_np, dtype=torch.float64, device=dev)

    t0 = time.time()
    n_pilot = min(4096, len(tr))
    Xstd_pilot = fac.std_rows(tr[:n_pilot])
    Ytr_pilot = ystore.csr_rows(tr[:n_pilot], "mean")
    _, _ = DSF._xty(Ytr_pilot, Xstd_pilot, dev, args.xty_device)
    unit_s = time.time() - t0
    del Xstd_pilot, Ytr_pilot
    _pilot_gate(args, "fit_maps", unit_s, len(tr) / n_pilot, PLANNED_H["fit_maps"])

    t0 = time.time()
    Xstd_t = fac.std_rows(tr)
    Ytr = ystore.csr_rows(tr, "mean")
    xty, xty_backend = DSF._xty(Ytr, Xstd_t, dev, args.xty_device)
    del Ytr, Xstd_t
    xty -= torch.outer(fac.colsum, ymu)
    B_W = fac.U.T @ xty
    del xty
    t_xty = time.time() - t0
    logger.info("[fit-maps] W X^T Y (%s) %.0fs", xty_backend, t_xty)

    ev = fac.std_rows(va) @ fac.U
    eh = fac.std_rows(ho) @ fac.U
    y_val = torch.as_tensor(ystore.csr_rows(va, "mean").toarray(), dtype=torch.float64, device=dev)
    ssr = np.zeros(len(LAMBDAS))
    sst = 0.0
    for c0 in range(0, DICT_SIZE, OUT_BLOCK):
        r, t = DSF._val_block_ss(y_val, ev, B_W, ymu, fac.s_eig, c0, min(c0 + OUT_BLOCK, DICT_SIZE))
        ssr += r
        sst += t
    del y_val
    val_r2 = 1.0 - ssr / max(sst, 1e-30)
    best = int(np.nanargmax(val_r2))
    sel_lam_w = float(LAMBDAS[best])
    _gate(
        args,
        "W_selected_lambda",
        sel_lam_w == W_BANKED_LAMBDA and best == 13,
        f"selected lambda={sel_lam_w:.10g} (grid idx {best}); banked {W_BANKED_LAMBDA:.10g}",
        gates,
    )
    lam_w = W_BANKED_LAMBDA if args.smoke else sel_lam_w  # smoke scores at the banked lambda

    s1h, s2h = ystore.col_stats(ho, "mean")
    ss_tot_w = s2h - (s1h**2) / len(ho)
    y_ho_csc = ystore.csc_rows(ho, "mean")
    ss_res_w = DSF._ridge_holdout(y_ho_csc, eh, B_W, ymu, fac.s_eig, lam_w, dev)
    score_w = DSF._score(ss_res_w, ss_tot_w, reg["f_out"], lam_w)
    perfeat_w = DSF._score_perfeature(ss_res_w, ss_tot_w)
    _gate(
        args,
        "W_pooled_r2_panel",
        abs(score_w["pooled_r2_panel"] - W_BANKED_POOLED_R2_PANEL) < DSF.REPRO_TOL_POOLED,
        f"panel R2={score_w['pooled_r2_panel']:.10f} banked {W_BANKED_POOLED_R2_PANEL:.10f} "
        f"tol {DSF.REPRO_TOL_POOLED}",
        gates,
    )
    banked_pf = np.load(
        _staged(args, PERFEATURE_1482_PATH.rsplit("/", 1)[0]) / "ridge__mean_perfeature.npz"
    )
    r2_banked = np.asarray(banked_pf["r2"], dtype=np.float64)
    pane = reg["f_out"]
    both = np.isfinite(perfeat_w["r2"][pane]) & np.isfinite(r2_banked[pane])
    max_delta = float(np.abs(perfeat_w["r2"][pane][both] - r2_banked[pane][both]).max())
    n_nan_mismatch = int((np.isfinite(perfeat_w["r2"][pane]) != np.isfinite(r2_banked[pane])).sum())
    _gate(
        args,
        "W_perfeature_panel",
        max_delta < DSF.REPRO_TOL_PERFEAT and n_nan_mismatch == 0,
        f"max |dR2| on panel={max_delta:.3e} (tol {DSF.REPRO_TOL_PERFEAT}); "
        f"nan mismatches={n_nan_mismatch}",
        gates,
    )

    # W bundle: P fp32 + standardizer + spectra provenance.
    inv_w = 1.0 / (fac.s_eig + lam_w)
    P_W = (fac.U @ (inv_w[:, None] * B_W)).cpu().numpy()
    wdir = outd / "W_bundle"
    wdir.mkdir(parents=True, exist_ok=True)
    np.save(wdir / "P_W.f32.npy", P_W.astype(np.float32))
    np.save(wdir / "xmu.f64.npy", fac.xmu.cpu().numpy())
    np.save(wdir / "xsd.f64.npy", fac.xsd.cpu().numpy())
    np.save(wdir / "ymu_W.f64.npy", ymu_np)
    np.save(wdir / "s_eig.f64.npy", fac.s_eig.cpu().numpy())
    np.save(wdir / "ss_tot_W.f64.npy", ss_tot_w)
    np.save(wdir / "ss_res_W.f64.npy", ss_res_w)
    _write_json(
        wdir / "W_bundle_meta.json",
        {
            "lambda": lam_w,
            "selected_lambda": sel_lam_w,
            "val_r2_by_lambda": {str(float(a)): float(b) for a, b in zip(LAMBDAS, val_r2)},
            "score": score_w,
            "selector": "val-carve sweep (parent convention, no GCV)",
            "n_train": int(len(tr)),
            "d_in": H_DIM,
            "meta": _meta(),
        },
    )
    del P_W

    # Per-feature R2 for Phase 4 (fp32 vectors, results dir).
    resd = _results(args)
    resd.mkdir(parents=True, exist_ok=True)
    np.savez(
        resd / "perfeature_W.npz",
        feat_ids=np.arange(DICT_SIZE, dtype=np.int32),
        r2=perfeat_w["r2"].astype(np.float32),
        ss_tot=ss_tot_w.astype(np.float64),
        ss_res=ss_res_w.astype(np.float64),
        scored=perfeat_w["scored"],
        panel_ids=pane.astype(np.int32),
    )

    # ---- M fit (dense -> dense; SAME fac) ------------------------------------
    Yd = _load_dense_targets(args, reg)
    ymu_m_np = np.asarray(Yd[tr], dtype=np.float64).mean(axis=0)
    ymu_m = torch.as_tensor(ymu_m_np, dtype=torch.float64, device=dev)
    xty_m = fac.xty_centered(Yd, tr, ymu_m)
    B_M = fac.U.T @ xty_m
    del xty_m
    ev_m = ev  # same X rows: reuse the rotated val/holdout designs
    y_val_m = torch.as_tensor(np.asarray(Yd[va], dtype=np.float64), device=dev)
    ssr_m = np.empty(len(LAMBDAS))
    for i, lam in enumerate(LAMBDAS):
        pred = (ev_m * (1.0 / (fac.s_eig + float(lam)))) @ B_M + ymu_m
        ssr_m[i] = float(((y_val_m - pred) ** 2).sum())
    best_m = int(np.argmin(ssr_m))
    sel_lam_m = float(LAMBDAS[best_m])
    _gate(
        args,
        "M_selected_lambda",
        sel_lam_m == M_BANKED_LAMBDA and best_m == 12,
        f"selected lambda={sel_lam_m:.10g} (grid idx {best_m}); banked {M_BANKED_LAMBDA:.10g}",
        gates,
    )
    lam_m = M_BANKED_LAMBDA if args.smoke else sel_lam_m
    inv_m = 1.0 / (fac.s_eig + lam_m)
    y_ho_m = np.asarray(Yd[ho], dtype=np.float64)
    pred_ho_m = ((eh * inv_m) @ B_M + ymu_m).cpu().numpy()
    mu_ho = y_ho_m.mean(axis=0)
    sse_m = float(((y_ho_m - pred_ho_m) ** 2).sum())
    sst_m = float(((y_ho_m - mu_ho) ** 2).sum())
    r2_m = 1.0 - sse_m / max(sst_m, 1e-30)  # PR._pooled_r2 convention (holdout own mean)
    _gate(
        args,
        "M_holdout_dense_r2",
        abs(r2_m - M_BANKED_DENSE_R2) < 1e-3,
        f"dense R2={r2_m:.10f} banked {M_BANKED_DENSE_R2:.10f} tol 1e-3",
        gates,
    )

    # Identity+learned-bias baseline for M (same-dimension map) — REPORT-ONLY (never a halt).
    # Plan section 6 requires the baseline be REPORTED; section 7 registers only the two
    # parity gates. Round 1 HALTed this at an uncalibrated 1e-6, which a benign
    # reduction-order drift could false-trip mid-pod; the recompute is now checked at the
    # parity-class tol (1e-3) as a WARN, persisted in repro_gates.json with enforced=False.
    ib_pred = identity_bias_predict(
        np.asarray(X[tr], dtype=np.float64),
        np.asarray(Yd[tr], dtype=np.float64),
        np.asarray(X[ho], dtype=np.float64),
    )
    ib_r2 = 1.0 - float(((y_ho_m - ib_pred) ** 2).sum()) / max(sst_m, 1e-30)
    ib_match = abs(ib_r2 - M_BANKED_IDENTITY_BIAS_R2) < 1e-3
    gates.append(
        {
            "gate": "M_identity_bias_r2",
            "pass": bool(ib_match),
            "detail": f"identity+bias R2={ib_r2:.10f} banked {M_BANKED_IDENTITY_BIAS_R2:.10f} "
            "tol 1e-3 (report-only, never enforced)",
            "enforced": False,
        }
    )
    logger.log(
        logging.INFO if ib_match else logging.WARNING,
        "[gate] M_identity_bias_r2: %s (report-only) identity+bias R2=%.10f banked %.10f",
        "MATCH" if ib_match else "DRIFT",
        ib_r2,
        M_BANKED_IDENTITY_BIAS_R2,
    )
    del ib_pred

    P_M = (fac.U @ (inv_m[:, None] * B_M)).cpu().numpy()
    mdir = outd / "M_bundle"
    mdir.mkdir(parents=True, exist_ok=True)
    np.save(mdir / "P_M.f64.npy", P_M)
    np.save(mdir / "ymu_M.f64.npy", ymu_m_np)
    _write_json(
        mdir / "M_bundle_meta.json",
        {
            "lambda": lam_m,
            "selected_lambda": sel_lam_m,
            "val_sse_by_lambda": {str(float(a)): float(b) for a, b in zip(LAMBDAS, ssr_m)},
            "holdout_dense_r2": r2_m,
            "identity_bias_r2": ib_r2,
            "sst_holdout": sst_m,
            "selector": "val-carve sweep (parent convention, no GCV)",
            "standardizer": "shared W_bundle xmu/xsd (same X, same train rows)",
            "meta": _meta(),
        },
    )

    # ---- kNN retrieval (mandatory mapping-baselines read; identity-bias N/A for W) ------
    t0 = time.time()
    n_probe = min(args.knn_probes, len(ho))
    n_unit = min(1024, n_probe)
    pool_w = _dense_ho_pool(ystore, ho)
    pred_probe_w = np.empty((n_probe, DICT_SIZE), dtype=np.float32)
    Xstd_ho_np = (
        np.asarray(X[ho[:n_probe]], dtype=np.float64) - fac.xmu.cpu().numpy()
    ) / fac.xsd.cpu().numpy()
    P_W32 = np.load(wdir / "P_W.f32.npy", mmap_mode="r")
    for c0 in range(0, DICT_SIZE, OUT_BLOCK):
        c1 = min(c0 + OUT_BLOCK, DICT_SIZE)
        pred_probe_w[:, c0:c1] = (
            Xstd_ho_np @ np.asarray(P_W32[:, c0:c1], dtype=np.float64) + ymu_np[c0:c1]
        ).astype(np.float32)
    t_unit = time.time() - t0
    _pilot_gate(args, "knn", t_unit, max(n_probe / n_unit, 1.0), PLANNED_H["knn"])
    knn_w = _knn_block(pred_probe_w, pool_w, n_probe)
    del pred_probe_w, pool_w
    pool_m = np.asarray(Yd[ho], dtype=np.float32)
    knn_m = _knn_block(pred_ho_m[:n_probe].astype(np.float32), pool_m, n_probe)
    del pool_m, pred_ho_m, y_ho_m

    _write_json(
        resd / "knn_baselines.json",
        {
            "W": {**knn_w, "identity_bias": "inapplicable — 3584 -> 131072 dim mismatch"},
            "M": {**knn_m, "identity_bias_r2": ib_r2},
            "convention": "5000 deterministic head probes, pool = full holdout, chance=k/n_pool",
            "meta": _meta(),
        },
    )
    _write_json(
        outd / "repro_gates.json",
        {"gates": gates, "wall_s": {"gram_eigh": t_gram, "xty": t_xty}, "meta": _meta()},
    )
    del fac, B_W, B_M, ev, eh

    # Expensive-store-before-long-read ordering: bundles upload NOW (plan section 9).
    if not args.skip_upload:
        base = _hf_base(args)
        _upload_tree(args, wdir, f"{base}/analysis_tensors/W_bundle", "W_bundle")
        _upload_tree(args, mdir, f"{base}/analysis_tensors/M_bundle", "M_bundle")
        _upload_phase_results(args, "results-fit-maps")
    _write_json(sentinel, {"regime": regime, "written": _now_iso()})
    return 0


# ── phase read-ladder ─────────────────────────────────────────────────────────


def _load_sae_dec(args) -> tuple[np.ndarray, np.ndarray]:
    """(W_dec (3584, 131072) fp32, b_dec (3584,) fp32) from the pinned SAE checkpoint."""
    import issue1482_sae as SAE

    sae_cache = Path(args.work) / "sae_cache"
    sae = SAE.BatchTopKSAE.load(64, "cpu", sae_cache, layer=19)
    w_dec = sae.w_dec.detach().cpu().numpy().astype(np.float32)
    b_dec = sae.b_dec.detach().cpu().numpy().astype(np.float32)
    assert w_dec.shape == (H_DIM, DICT_SIZE), w_dec.shape
    return w_dec, b_dec


def _bundles(args) -> dict:
    """Load the persisted W/M bundles (P, standardizer, ymu, metadata)."""
    outd = _out(args)
    wdir, mdir = outd / "W_bundle", outd / "M_bundle"
    return {
        "P_W": np.load(wdir / "P_W.f32.npy", mmap_mode="r"),
        "xmu": np.load(wdir / "xmu.f64.npy"),
        "xsd": np.load(wdir / "xsd.f64.npy"),
        "ymu_W": np.load(wdir / "ymu_W.f64.npy"),
        "ss_tot_W": np.load(wdir / "ss_tot_W.f64.npy"),
        "P_M": np.load(mdir / "P_M.f64.npy"),
        "ymu_M": np.load(mdir / "ymu_M.f64.npy"),
        "meta_W": _read_json(wdir / "W_bundle_meta.json"),
        "meta_M": _read_json(mdir / "M_bundle_meta.json"),
    }


FEAT_BLOCK = 16_384  # feature-row chunk for E-space reductions


def _u_vector(w_dec: np.ndarray, xsd: np.ndarray, K: np.ndarray) -> np.ndarray:
    """U_j = sqrt(((E K) o E).sum(1)) computed in feature chunks (E = (W_dec/xsd[:,None])^T)."""
    u2 = np.empty(DICT_SIZE, dtype=np.float64)
    for j0 in range(0, DICT_SIZE, FEAT_BLOCK):
        j1 = min(j0 + FEAT_BLOCK, DICT_SIZE)
        e = (w_dec[:, j0:j1].astype(np.float64) / xsd[:, None]).T
        u2[j0:j1] = np.einsum("jh,jh->j", e @ K, e)
    return np.sqrt(np.clip(u2, 0.0, None))


def _e_dot_rows(w_dec: np.ndarray, xsd: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Rowwise <E[j], M[j]> over feature chunks (term-1 core of the exact ablation dSSE)."""
    out = np.empty(DICT_SIZE, dtype=np.float64)
    for j0 in range(0, DICT_SIZE, FEAT_BLOCK):
        j1 = min(j0 + FEAT_BLOCK, DICT_SIZE)
        e = (w_dec[:, j0:j1].astype(np.float64) / xsd[:, None]).T
        out[j0:j1] = np.einsum("jh,jh->j", e, M[j0:j1])
    return out


def _residual_pass_w(args, reg, ystore, X, b) -> tuple[np.ndarray, np.ndarray, float]:
    """Holdout residual pass for W: returns (Q = R P_W^T (n_ho, H), per-col ss_res, SST)."""
    ho = reg["ho"]
    xmu, xsd, ymu = b["xmu"], b["xsd"], b["ymu_W"]
    Xstd_ho = (np.asarray(X[ho], dtype=np.float64) - xmu) / xsd
    P = b["P_W"]
    y_csc = ystore.csc_rows(ho, "mean")
    Q = np.zeros((len(ho), H_DIM), dtype=np.float64)
    ss_res = np.zeros(DICT_SIZE, dtype=np.float64)
    n_blocks = (DICT_SIZE + OUT_BLOCK - 1) // OUT_BLOCK
    t0 = time.time()
    for bi, c0 in enumerate(range(0, DICT_SIZE, OUT_BLOCK)):
        c1 = min(c0 + OUT_BLOCK, DICT_SIZE)
        Pb = np.asarray(P[:, c0:c1], dtype=np.float64)
        pred = Xstd_ho @ Pb + ymu[c0:c1]
        rb = DSF._dense_cols(y_csc, c0, c1).astype(np.float64) - pred
        Q += rb @ Pb.T
        ss_res[c0:c1] = (rb**2).sum(axis=0)
        if bi == 0:
            _pilot_gate(args, "read_ladder", time.time() - t0, n_blocks, PLANNED_H["read_ladder"])
        logger.info(
            "[read-ladder] W residual block %d/%d %.0fs", bi + 1, n_blocks, time.time() - t0
        )
    sst = float(b["ss_tot_W"].sum())
    return Q, ss_res, sst


def _ablation_read(
    psil_ho: sp.csr_matrix,
    Q: np.ndarray,
    w_dec,
    xsd,
    u: np.ndarray,
    sst: float,
    row_mask: np.ndarray | None = None,
    sst_sub: float | None = None,
) -> dict:
    """Exact ablation dSSE per feature over a holdout row subset.

    dSSE_j = 2 * <E[j], (Psi^T Q)[j]> + U_j^2 * sum(psi_j^2); A_j = dSSE_j / SST. The
    cross-term CAN be negative and is never clipped.
    """
    if row_mask is not None:
        psil = psil_ho[row_mask]
        q = Q[row_mask]
        sst_use = float(sst_sub)
    else:
        psil, q, sst_use = psil_ho, Q, float(sst)
    m = np.asarray((psil.T @ q), dtype=np.float64)  # (DICT, H)
    term1 = _e_dot_rows(w_dec, xsd, m)
    psi2 = np.asarray(psil.power(2).sum(axis=0), dtype=np.float64).ravel()
    dsse = 2.0 * term1 + (u**2) * psi2
    return {
        "dsse": dsse,
        "a": dsse / max(sst_use, 1e-30),
        "term1": term1,
        "psi2_sum": psi2,
        "sst": sst_use,
        "n_rows": int(psil.shape[0]),
    }


def _pairing_null(
    args, psil_ho: sp.csr_matrix, Q: np.ndarray, w_dec, xsd, u: np.ndarray, sst: float, tag: str
) -> Path:
    """A_j row-pairing permutation null (n draws): permute holdout row pairing Psi<->Q.

    Uses D = E Q^T (fp32 gathers per draw; term2 is pairing-invariant). Persists the
    per-draw x per-feature dSSE matrix (fp32) + the observed row (selection-symmetric
    persistence contract). Returns the npz path.
    """
    n_draws = args.n_draws_pairing
    n_ho = Q.shape[0]
    d_mat = np.empty((DICT_SIZE, n_ho), dtype=np.float32)
    qt = Q.T.astype(np.float64)
    for j0 in range(0, DICT_SIZE, FEAT_BLOCK):
        j1 = min(j0 + FEAT_BLOCK, DICT_SIZE)
        e = (w_dec[:, j0:j1].astype(np.float64) / xsd[:, None]).T
        d_mat[j0:j1] = (e @ qt).astype(np.float32)
    coo = psil_ho.tocoo()
    rows_n = coo.row.astype(np.int64)
    cols_n = coo.col.astype(np.int64)
    vals_n = coo.data.astype(np.float64)
    psi2 = np.asarray(psil_ho.power(2).sum(axis=0), dtype=np.float64).ravel()
    term2 = (u**2) * psi2
    rng = np.random.default_rng(SEED + 1)
    draws = np.empty((n_draws, DICT_SIZE), dtype=np.float32)
    t0 = time.time()
    for d in range(n_draws):
        perm = rng.permutation(n_ho)
        contrib = vals_n * d_mat[cols_n, perm[rows_n]].astype(np.float64)
        t1 = np.bincount(cols_n, weights=contrib, minlength=DICT_SIZE)
        draws[d] = (2.0 * t1 + term2).astype(np.float32)
        if (d + 1) % 50 == 0 or d + 1 == n_draws:
            logger.info("[pairing-null:%s] draw %d/%d %.0fs", tag, d + 1, n_draws, time.time() - t0)
    del d_mat
    observed = _ablation_read(psil_ho, Q, w_dec, xsd, u, sst)
    path = _results(args) / "nulls" / f"pairing_null__{tag}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        observed_dsse=observed["dsse"].astype(np.float32),
        draws_dsse=draws,
        sst=np.float64(sst),
        n_draws=np.int64(n_draws),
        dtype_note="draws fp32 (fp16 subnormal-flushes A~1e-9); dSSE units, divide by sst for A",
    )
    return path


def phase_read_ladder(args) -> int:
    """Phase 2: U/U*sd/C/A on W and M + per-half/split-half A + recon share + pairing nulls."""
    resd = _results(args)
    sentinel = _out(args) / "read_ladder.done.json"
    regime = _regime(args)
    if _resume_ok(args, sentinel, regime, "read-ladder"):
        return 0
    assert_out_root_headroom(Path(args.work), 3 if args.smoke else 15, phase="read-ladder")

    reg = _registry(args)
    ho = reg["ho"]
    X = _load_design(args, reg)
    ystore = _ystore(args, reg)
    b = _bundles(args)
    w_dec, b_dec = _load_sae_dec(args)
    census = np.load(_assembled(args) / "census.npz")
    psil = _psil_csr(args, reg)
    psil_ho = psil[ho]
    prov_row = _prov_per_row(args, reg)
    xsd = b["xsd"]

    # sd(psi_last) over train rows (zeros included, population sd) — census sufficient stats.
    # Production census covers ALL rows; the train count for the sd denominator is the
    # registered train-split size of the census build (120,000 in production).
    n_tr_full = int((np.load(_inputs_1482_dir(args) / "which.npy") == 1).sum())
    s1, s2 = census["tr_sum"], census["tr_sum2"]
    var_tr = np.clip(s2 / n_tr_full - (s1 / n_tr_full) ** 2, 0.0, None)
    sd_tr = np.sqrt(var_tr)

    ladders = {}
    for tag in ("W", "M"):
        if tag == "W":
            P = np.asarray(b["P_W"], dtype=np.float64)
            k_mat = P @ P.T
            q, ss_res, sst = _residual_pass_w(args, reg, ystore, X, b)
            del ss_res
        else:
            P = b["P_M"]
            k_mat = P @ P.T
            Yd = _load_dense_targets(args, reg)
            xstd_ho = (np.asarray(X[ho], dtype=np.float64) - b["xmu"]) / xsd
            pred = xstd_ho @ P + b["ymu_M"]
            y_ho = np.asarray(Yd[ho], dtype=np.float64)
            r = y_ho - pred
            q = r @ P.T
            sst = float(((y_ho - y_ho.mean(axis=0)) ** 2).sum())
            del Yd, y_ho, pred, r, xstd_ho
        u = _u_vector(w_dec, xsd, k_mat)
        if tag == "W":
            del P
        del k_mat
        # C_j = U^2 * mean(psi^2 | active, holdout) / (SST / n_ho).
        ho_cnt = np.asarray(census["ho_count"], dtype=np.float64)
        mean_psi2_act = np.zeros(DICT_SIZE, dtype=np.float64)
        act = ho_cnt > 0
        mean_psi2_act[act] = np.asarray(census["ho_sum2"], dtype=np.float64)[act] / ho_cnt[act]
        pooled_var = sst / max(len(ho), 1)
        c_vec = (u**2) * mean_psi2_act / max(pooled_var, 1e-30)

        obs = _ablation_read(psil_ho, q, w_dec, xsd, u, sst)
        halves = {}
        lm_mask = prov_row[ho] == 0
        for name, mask in (
            ("lmsys", lm_mask),
            ("wildchat", ~lm_mask),
            ("even", np.arange(len(ho)) % 2 == 0),
            ("odd", np.arange(len(ho)) % 2 == 1),
        ):
            if int(mask.sum()) < 2:
                halves[name] = None
                continue
            s1h_sub, s2h_sub = ystore.col_stats(ho[mask], "mean")
            sst_sub = float((s2h_sub - (s1h_sub**2) / max(int(mask.sum()), 1)).sum())
            halves[name] = _ablation_read(
                psil_ho, q, w_dec, xsd, u, sst, row_mask=mask, sst_sub=sst_sub
            )
        np.savez(
            resd / f"read_ladder__{tag}.npz",
            feat_ids=np.arange(DICT_SIZE, dtype=np.int32),
            u=u.astype(np.float64),
            u_sd=(u * sd_tr).astype(np.float64),
            c=c_vec.astype(np.float64),
            a=obs["a"].astype(np.float64),
            dsse=obs["dsse"].astype(np.float64),
            dsse_term1=obs["term1"].astype(np.float64),
            psi2_sum_ho=obs["psi2_sum"].astype(np.float64),
            sst=np.float64(sst),
            a_lmsys=(halves["lmsys"]["a"] if halves["lmsys"] else np.full(DICT_SIZE, np.nan)),
            a_wildchat=(
                halves["wildchat"]["a"] if halves["wildchat"] else np.full(DICT_SIZE, np.nan)
            ),
            a_even=(halves["even"]["a"] if halves["even"] else np.full(DICT_SIZE, np.nan)),
            a_odd=(halves["odd"]["a"] if halves["odd"] else np.full(DICT_SIZE, np.nan)),
            n_ho=np.int64(len(ho)),
            sd_train_psil=sd_tr.astype(np.float64),
            formula=np.str_(
                "U=sqrt(((E K)oE).sum(1)); C=U^2*mean(psi^2|active,ho)/(SST/n_ho); "
                "A=dSSE/SST, dSSE=2<E,Psi^T Q>+U^2*sum(psi^2); SST=holdout own-mean pooled"
            ),
        )
        _pairing_null(args, psil_ho, q, w_dec, xsd, u, sst, tag)
        ladders[tag] = {"u": u, "sst": sst}
        del q

    # SAE reconstruction residual share of v_C on holdout (U/A inherit reconstruction error).
    x_ho = np.asarray(X[ho], dtype=np.float64)
    recon = np.asarray(psil_ho @ w_dec.T.astype(np.float64)) + b_dec.astype(np.float64)
    sse_rec = float(((x_ho - recon) ** 2).sum())
    sst_rec = float(((x_ho - x_ho.mean(axis=0)) ** 2).sum())
    _write_json(
        resd / "recon_share.json",
        {
            "holdout_recon_residual_share": sse_rec / max(sst_rec, 1e-30),
            "holdout_recon_fve": 1.0 - sse_rec / max(sst_rec, 1e-30),
            "n_ho": int(len(ho)),
            "meta": _meta(),
        },
    )
    _upload_phase_results(args, "results-read-ladder")
    _write_json(sentinel, {"regime": regime, "written": _now_iso()})
    return 0


# ── phase carried ─────────────────────────────────────────────────────────────


def phase_carried(args) -> int:
    """Phase 3: Carried_j via fully-expanded sufficient statistics (exact, blockwise)."""
    resd = _results(args)
    sentinel = _out(args) / "carried.done.json"
    regime = _regime(args)
    if _resume_ok(args, sentinel, regime, "carried"):
        return 0
    assert_out_root_headroom(Path(args.work), 2 if args.smoke else 10, phase="carried")

    reg = _registry(args)
    tr, ho = reg["tr"], reg["ho"]
    ystore = _ystore(args, reg)
    psil = _psil_csr(args, reg)
    psil_tr, psil_ho = psil[tr], psil[ho]
    prov_row = _prov_per_row(args, reg)

    n_tr = len(tr)
    s_psi = np.asarray(psil_tr.sum(axis=0), dtype=np.float64).ravel()
    s_psi2 = np.asarray(psil_tr.power(2).sum(axis=0), dtype=np.float64).ravel()
    mu_psi = s_psi / n_tr
    sxx = s_psi2 - (s_psi**2) / n_tr
    ok = sxx > 1e-12
    s1_tr, _ = ystore.col_stats(tr, "mean")
    ymu = s1_tr / n_tr

    subsets: dict[str, np.ndarray | None] = {
        "full": None,
        "lmsys": prov_row[ho] == 0,
        "wildchat": prov_row[ho] == 1,
    }
    st: dict[str, dict] = {}
    for name, mask in subsets.items():
        rows = ho if mask is None else ho[mask]
        if len(rows) < 2:
            st[name] = {}
            continue
        p_sub = psil_ho if mask is None else psil_ho[mask]
        s1h, s2h = ystore.col_stats(rows, "mean")
        st[name] = {
            "mask": mask,
            "n": len(rows),
            "s1h": s1h,
            "s2h": s2h,
            "sst": float((s2h - (s1h**2) / len(rows)).sum()),
            "s_psi_ho": np.asarray(p_sub.sum(axis=0), dtype=np.float64).ravel(),
            "s_psi2_ho": np.asarray(p_sub.power(2).sum(axis=0), dtype=np.float64).ravel(),
            "csc": ystore.csc_rows(rows, "mean"),
            "dot_a_u": np.zeros(DICT_SIZE, dtype=np.float64),
            "dot_ymu_u": np.zeros(DICT_SIZE, dtype=np.float64),
            "t0": 0.0,
        }

    dot_a_a = np.zeros(DICT_SIZE, dtype=np.float64)
    dot_a_ymu = np.zeros(DICT_SIZE, dtype=np.float64)
    norm_ymu2 = 0.0
    # <A_j, s1h> / <ymu, s1h> per subset (one extra GEMV per block; the middle-term expansion
    # needs <beta, s1h> — see the derivation in the results formula string below).
    dot_a_s1h = {name: np.zeros(DICT_SIZE, dtype=np.float64) for name in subsets if st[name]}
    dot_ymu_s1h = {name: 0.0 for name in subsets if st[name]}
    csc_tr = ystore.csc_rows(tr, "mean")
    block = args.carried_block
    n_blocks = (DICT_SIZE + block - 1) // block
    t0 = time.time()
    for bi, c0 in enumerate(range(0, DICT_SIZE, block)):
        c1 = min(c0 + block, DICT_SIZE)
        mu_b = ymu[c0:c1]
        yb_tr = sp.csr_matrix(csc_tr[:, c0:c1])
        a_blk = np.asarray((psil_tr.T @ yb_tr).todense(), dtype=np.float64)
        dot_a_a += np.einsum("jc,jc->j", a_blk, a_blk)
        dot_a_ymu += a_blk @ mu_b
        norm_ymu2 += float(mu_b @ mu_b)
        for name in subsets:
            s = st[name]
            if not s:
                continue
            p_sub = psil_ho if s["mask"] is None else psil_ho[s["mask"]]
            yb_ho = sp.csr_matrix(s["csc"][:, c0:c1])
            u_blk = np.asarray((p_sub.T @ yb_ho).todense(), dtype=np.float64)
            s["dot_a_u"] += np.einsum("jc,jc->j", a_blk, u_blk)
            s["dot_ymu_u"] += u_blk @ mu_b
            s1h_b = s["s1h"][c0:c1]
            dot_a_s1h[name] += a_blk @ s1h_b
            dot_ymu_s1h[name] += float(mu_b @ s1h_b)
            s["t0"] += float((s["s2h"][c0:c1] - 2.0 * mu_b * s1h_b + s["n"] * mu_b**2).sum())
        del a_blk
        if bi == 0:
            _pilot_gate(args, "carried", time.time() - t0, n_blocks, PLANNED_H["carried"])
        logger.info("[carried] block %d/%d %.0fs", bi + 1, n_blocks, time.time() - t0)

    results = {}
    for name in subsets:
        s = st[name]
        if not s:
            results[name] = np.full(DICT_SIZE, np.nan)
            continue
        n_sub = s["n"]
        with np.errstate(divide="ignore", invalid="ignore"):
            b_u = (s["dot_a_u"] - s_psi * s["dot_ymu_u"]) / sxx
            b_ymu = (dot_a_ymu - s_psi * norm_ymu2) / sxx
            b_s1h = (dot_a_s1h[name] - s_psi * dot_ymu_s1h[name]) / sxx
            b_norm2 = (dot_a_a - 2.0 * s_psi * dot_a_ymu + (s_psi**2) * norm_ymu2) / (sxx**2)
        b_t = b_s1h - n_sub * b_ymu
        middle = b_u - s["s_psi_ho"] * b_ymu - mu_psi * b_t
        third = (s["s_psi2_ho"] - 2.0 * mu_psi * s["s_psi_ho"] + n_sub * mu_psi**2) * b_norm2
        sse = s["t0"] - 2.0 * middle + third
        r2 = np.full(DICT_SIZE, np.nan)
        r2[ok] = 1.0 - sse[ok] / max(s["sst"], 1e-30)
        results[name] = r2

    # Paired contrast vs the read (U and A axes), overall + within activity deciles.
    from scipy.stats import spearmanr

    ladder = np.load(resd / "read_ladder__W.npz")
    u_w = np.asarray(ladder["u"], dtype=np.float64)
    a_w = np.asarray(ladder["a"], dtype=np.float64)
    census = np.load(_assembled(args) / "census.npz")
    ltc = np.asarray(census["lasttoken_count"], dtype=np.float64)
    car = results["full"]
    mask = np.isfinite(car) & np.isfinite(u_w) & np.isfinite(a_w)

    def _dec_bins(v: np.ndarray) -> np.ndarray:
        """Deterministic decile bins by (value, id)-lexsorted rank."""
        idx = np.arange(len(v))
        order_ = np.lexsort((idx, v))
        bins = np.empty(len(v), dtype=np.int64)
        bins[order_] = np.minimum((np.arange(len(v)) * 10) // max(len(v), 1), 9)
        return bins

    contrast: dict = {"overall": {}, "per_decile": {}, "tail": {}}
    for axis, vec in (("u", u_w), ("a", a_w)):
        rho, _p = spearmanr(vec[mask], car[mask])
        contrast["overall"][axis] = {"spearman": float(rho), "n": int(mask.sum())}
    bins = _dec_bins(ltc[mask])
    for axis, vec in (("u", u_w), ("a", a_w)):
        per = []
        for d in range(10):
            sel = bins == d
            if int(sel.sum()) >= 10:
                rho, _p = spearmanr(vec[mask][sel], car[mask][sel])
                per.append({"decile": d, "spearman": float(rho), "n": int(sel.sum())})
            else:
                per.append({"decile": d, "spearman": None, "n": int(sel.sum())})
        contrast["per_decile"][axis] = per
    # High-carried / low-read residual tail: top-decile Carried, bottom-half U.
    car_m, u_m = car[mask], u_w[mask]
    ids_m = np.where(mask)[0]
    car_bins = _dec_bins(car_m)
    u_med = np.median(u_m)
    tail_ids = ids_m[(car_bins == 9) & (u_m < u_med)]
    contrast["tail"] = {
        "definition": "top-decile Carried AND below-median U (W map)",
        "n": int(len(tail_ids)),
        "feature_ids_head": [int(i) for i in tail_ids[:200]],
    }

    np.savez(
        resd / "carried.npz",
        feat_ids=np.arange(DICT_SIZE, dtype=np.int32),
        carried=results["full"].astype(np.float64),
        carried_lmsys=results["lmsys"].astype(np.float64),
        carried_wildchat=results["wildchat"].astype(np.float64),
        sxx_train=sxx.astype(np.float64),
        n_undefined=np.int64(int((~ok).sum())),
        formula=np.str_(
            "Carried_j = 1 - SSE_j/SST; y_hat = ymu_tr + beta_j (psi_j - mu_psi_j); "
            "beta_j = Cov_tr(psi_j, y)/Var_tr(psi_j); SST = holdout own-mean pooled"
        ),
    )
    _write_json(resd / "paired_contrast.json", {**contrast, "meta": _meta()})
    _upload_phase_results(args, "results-carried")
    _write_json(sentinel, {"regime": regime, "written": _now_iso()})
    return 0


# ── phase answer-matchedn ─────────────────────────────────────────────────────


def phase_answer_matchedn(args) -> int:
    """Phase 4: answer-side pooled-vs-conditional R2 + matched-n down-sampling control."""
    resd = _results(args)
    sentinel = _out(args) / "answer_matchedn.done.json"
    regime = _regime(args)
    if _resume_ok(args, sentinel, regime, "answer-matchedn"):
        return 0

    reg = _registry(args)
    ho = reg["ho"]
    pane = np.asarray(reg["f_out"], dtype=np.int64)
    X = _load_design(args, reg)
    ystore = _ystore(args, reg)
    b = _bundles(args)
    y_csc = ystore.csc_rows(ho, "mean")
    perfeat = np.load(resd / "perfeature_W.npz")
    r2_pooled = np.asarray(perfeat["r2"], dtype=np.float64)

    xmu, xsd, ymu = b["xmu"], b["xsd"], b["ymu_W"]
    xstd_ho = (np.asarray(X[ho], dtype=np.float64) - xmu) / xsd
    P = b["P_W"]

    n_pane = len(pane)
    n_act = np.zeros(n_pane, dtype=np.int64)
    cond_ss_res = np.zeros(n_pane, dtype=np.float64)
    cond_sum_y = np.zeros(n_pane, dtype=np.float64)
    cond_sum_y2 = np.zeros(n_pane, dtype=np.float64)

    # Selection for the matched-n control is decided AFTER counts; the value/residual gather
    # runs in a second pass restricted to selected features.
    blk = 2_048
    t0 = time.time()
    n_blocks = (n_pane + blk - 1) // blk
    for bi, p0 in enumerate(range(0, n_pane, blk)):
        p1 = min(p0 + blk, n_pane)
        cols = pane[p0:p1]
        yb = DSF._dense_cols(y_csc[:, cols], 0, len(cols)).astype(np.float64)
        pb = np.asarray(P[:, cols], dtype=np.float64)
        pred = xstd_ho @ pb + ymu[cols]
        act = yb > 0
        n_act[p0:p1] = act.sum(axis=0)
        r2_ = (yb - pred) ** 2
        cond_ss_res[p0:p1] = (r2_ * act).sum(axis=0)
        cond_sum_y[p0:p1] = (yb * act).sum(axis=0)
        cond_sum_y2[p0:p1] = ((yb**2) * act).sum(axis=0)
        if bi == 0:
            _pilot_gate(
                args, "answer_matchedn", time.time() - t0, n_blocks, PLANNED_H["answer_matchedn"]
            )
        if (bi + 1) % 2 == 0 or bi + 1 == n_blocks:
            logger.info("[matchedn] cond block %d/%d %.0fs", bi + 1, n_blocks, time.time() - t0)

    with np.errstate(divide="ignore", invalid="ignore"):
        cond_sst = cond_sum_y2 - (cond_sum_y**2) / np.maximum(n_act, 1)
        r2_cond = np.where((n_act >= 2) & (cond_sst > 1e-12), 1.0 - cond_ss_res / cond_sst, np.nan)

    # Activity deciles over PANEL features by holdout answer-side active count (deterministic).
    idx = np.arange(n_pane)
    order_ = np.lexsort((idx, n_act))
    bins = np.empty(n_pane, dtype=np.int64)
    bins[order_] = np.minimum((np.arange(n_pane) * 10) // max(n_pane, 1), 9)
    n_rare = int(np.median(n_act[bins == 0]))
    n_rare = max(n_rare, 2)

    rng = np.random.default_rng(SEED + 4)
    per_dec = max(args.matchedn_features // 10, 1)
    selected = []
    for d in range(10):
        members = np.where(bins == d)[0]
        take = min(per_dec, len(members))
        selected.append(np.sort(rng.choice(members, size=take, replace=False)))
    selected = np.concatenate(selected)
    sel_set = {int(s) for s in selected}

    # Second pass: gather active-row (y, residual) per selected feature; matched-n draws.
    ybar_full = np.asarray(ystore.col_stats(ho, "mean")[0], dtype=np.float64) / len(ho)
    draw_mean_r2 = np.full(n_pane, np.nan)
    skipped_below_rare = 0
    t0 = time.time()
    done = 0
    for p0 in range(0, n_pane, blk):
        p1 = min(p0 + blk, n_pane)
        local = [i for i in range(p0, p1) if i in sel_set]
        if not local:
            continue
        cols = pane[np.asarray(local)]
        yb = DSF._dense_cols(y_csc[:, cols], 0, len(cols)).astype(np.float64)
        pb = np.asarray(P[:, cols], dtype=np.float64)
        pred = xstd_ho @ pb + ymu[cols]
        res2 = (yb - pred) ** 2
        dev2 = (yb - ybar_full[cols]) ** 2
        act = yb > 0
        for k, i in enumerate(local):
            na = int(act[:, k].sum())
            if na < n_rare:
                skipped_below_rare += 1
                continue
            rows_k = np.where(act[:, k])[0]
            r2v = res2[rows_k, k]
            d2v = dev2[rows_k, k]
            picks = np.argsort(rng.random((args.matchedn_draws, na)), axis=1)[:, :n_rare]
            num = r2v[picks].sum(axis=1)
            den = d2v[picks].sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                r2d = np.where(den > 1e-12, 1.0 - num / den, np.nan)
            draw_mean_r2[i] = float(np.nanmean(r2d))
            done += 1
            if done % 500 == 0:
                logger.info("[matchedn] feature %d/%d %.0fs", done, len(selected), time.time() - t0)

    per_decile = []
    for d in range(10):
        sel_d = bins == d
        sel_dm = sel_d & np.isfinite(draw_mean_r2)
        per_decile.append(
            {
                "decile": d,
                "n_features": int(sel_d.sum()),
                "median_pooled_r2": float(np.nanmedian(r2_pooled[pane][sel_d])),
                "median_conditional_r2": float(np.nanmedian(r2_cond[sel_d])),
                "median_matchedn_r2": (
                    float(np.nanmedian(draw_mean_r2[sel_dm])) if sel_dm.any() else None
                ),
                "n_matchedn_features": int(sel_dm.sum()),
                "median_n_active": float(np.median(n_act[sel_d])),
            }
        )
    _write_json(
        resd / "answer_matchedn.json",
        {
            "n_rare": n_rare,
            "n_selected": int(len(selected)),
            "n_skipped_below_rare": skipped_below_rare,
            "n_draws": int(args.matchedn_draws),
            "per_decile": per_decile,
            "parent_reference": {
                "decile_gradient_pooled_r2": PARENT_DECILE_GRADIENT,
                "label_shuffle_band": PARENT_SHUFFLE_BAND,
                "note": "cited from the parent #1482, not recomputed",
            },
            "formulas": {
                "conditional_r2": "1 - sum_act (y-pred)^2 / (sum_act y^2 - (sum_act y)^2/n_act)",
                "matchedn_r2": "per draw: 1 - sum_sub res^2 / sum_sub (y - ybar_full_ho)^2, "
                "subsets of size n_rare drawn from ACTIVE holdout rows without replacement",
            },
            "meta": _meta(),
        },
    )
    np.savez(
        resd / "answer_matchedn_perfeature.npz",
        panel_ids=pane.astype(np.int32),
        n_active_ho=n_act,
        r2_pooled=r2_pooled[pane].astype(np.float64),
        r2_conditional=r2_cond.astype(np.float64),
        matchedn_mean_r2=draw_mean_r2.astype(np.float64),
        decile=bins,
    )
    _upload_phase_results(args, "results-answer-matchedn")
    _write_json(sentinel, {"regime": regime, "written": _now_iso()})
    return 0


# ── phase partials ────────────────────────────────────────────────────────────


def _rank(v: np.ndarray) -> np.ndarray:
    """Average-tie ranks (scipy rankdata) as fp64."""
    from scipy.stats import rankdata

    return rankdata(v).astype(np.float64)


def _residualize(y: np.ndarray, m_cols: np.ndarray) -> np.ndarray:
    """OLS residual of y on [1, m_cols] (least-squares; y and m_cols are rank vectors)."""
    design = np.column_stack([np.ones(len(y))] + [m_cols[:, k] for k in range(m_cols.shape[1])])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def _load_selection(args) -> tuple[dict[str, np.ndarray], list[str], list[dict]]:
    """Selection-set covariates: v2 varying floats + census last-token covariates.

    Returns (name -> values (DICT_SIZE,)), the sorted column list, and the dropped report.
    Zero-variance columns are dropped and reported (dec_norm asserted among them).
    """
    cov = np.load(_cov_dir(args) / "fullwidth_covariates_v2.npz")
    feat_ids = np.asarray(cov["feat_ids"], dtype=np.int64)
    assert np.array_equal(feat_ids, np.arange(DICT_SIZE)), "covariates feat_ids not aligned"
    census = np.load(_assembled(args) / "census.npz")
    cols: dict[str, np.ndarray] = {}
    dropped: list[dict] = []
    for name in sorted(cov.files):
        if name == "feat_ids":
            continue
        v = np.asarray(cov[name], dtype=np.float64)
        fin = np.isfinite(v)
        std = float(np.std(v[fin])) if fin.any() else 0.0
        if std < 1e-6:
            dropped.append({"column": name, "reason": "zero-variance", "std": std})
            continue
        cols[name] = v
    assert any(d["column"] == "dec_norm" for d in dropped), (
        "dec_norm expected constant/dropped; got dropped=" + str(dropped)
    )
    for name in LASTTOKEN_COVS:
        cols[name] = np.asarray(census[name], dtype=np.float64)
    columns = sorted(cols)
    return cols, columns, dropped


def _partial_row(
    dv_rank: np.ndarray,
    cov_ranks: np.ndarray,
    resid_c: np.ndarray,
    norm_c: np.ndarray,
    match_idx: list[int],
    m_design: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Partial Spearman of dv vs every covariate given the match design (vectorized).

    Returns (partials (n_cov,), degenerate flags). A covariate identical to a matching
    variable has a zero-variance residual: its partial is defined as 0.0 and flagged.
    """
    resid_d = _residualize(dv_rank, m_design)
    nd = float(np.linalg.norm(resid_d))
    n_cov = cov_ranks.shape[0]
    out = np.zeros(n_cov)
    degen = np.zeros(n_cov, dtype=bool)
    for k in range(n_cov):
        if k in match_idx or norm_c[k] < 1e-9 or nd < 1e-9:
            degen[k] = True
            continue
        out[k] = float(resid_d @ resid_c[k]) / (nd * norm_c[k])
    return out, degen


def phase_partials(args) -> int:
    """Phase 5: activity-matched partials + per-DV stratified permutation nulls (+ controls)."""
    resd = _results(args)
    sentinel = _out(args) / "partials.done.json"
    regime = _regime(args)
    if _resume_ok(args, sentinel, regime, "partials"):
        return 0

    from scipy.stats import spearmanr

    cols, columns, dropped = _load_selection(args)
    n_cov = len(columns)
    census = np.load(_assembled(args) / "census.npz")
    lad_w = np.load(resd / "read_ladder__W.npz")
    lad_m = np.load(resd / "read_ladder__M.npz")
    carried = np.load(resd / "carried.npz")
    perfeat = np.load(resd / "perfeature_W.npz")

    feat_ids = np.arange(DICT_SIZE, dtype=np.int64)
    for arr, nm in (
        (lad_w["feat_ids"], "read_ladder_W"),
        (lad_m["feat_ids"], "read_ladder_M"),
        (carried["feat_ids"], "carried"),
        (perfeat["feat_ids"], "perfeature_W"),
        (np.load(_cov_dir(args) / "fullwidth_covariates_v2.npz")["feat_ids"], "covariates"),
        (census["feat_ids"], "census"),
    ):
        assert np.array_equal(np.asarray(arr, dtype=np.int64), feat_ids), (
            f"feature-id join broken for {nm}"
        )

    u_w = np.asarray(lad_w["u"], dtype=np.float64)
    dvs = {
        "logU_W": np.where(u_w > 0, np.log10(np.clip(u_w, 1e-300, None)), np.nan),
        "A_W": np.asarray(lad_w["a"], dtype=np.float64),
        "carried": np.asarray(carried["carried"], dtype=np.float64),
        "U_M": np.asarray(lad_m["u"], dtype=np.float64),
    }
    dv_roles = {"logU_W": "primary", "A_W": "secondary", "carried": "secondary", "U_M": "secondary"}
    outside_col = np.asarray(perfeat["r2"], dtype=np.float64)  # answer-side per-feature R2

    cov_mat = np.stack([cols[c] for c in columns])  # (n_cov, DICT)
    cov_finite = np.isfinite(cov_mat).all(axis=0)
    match_idx = [columns.index(MATCH_COV)]
    match2_idx = sorted({columns.index(MATCH_COV), columns.index(MATCH_COV_2)})

    report: dict = {
        "selection_columns": columns,
        "realized_breadth": n_cov,
        "dropped_columns": dropped,
        "match_covariate": MATCH_COV,
        "robustness_match_covariates": [MATCH_COV, MATCH_COV_2],
        "geometry_definitional": list(GEOMETRY_DEFINITIONAL),
        "dv_roles": dv_roles,
        "per_dv": {},
        "meta": _meta(),
    }

    rng = np.random.default_rng(SEED + 5)
    n_draws = args.n_draws_strat
    t_pilot: float | None = None
    for dv_name, dv in dvs.items():
        mask = cov_finite & np.isfinite(dv)
        n = int(mask.sum())
        if n < 50:
            report["per_dv"][dv_name] = {"n_complete_case": n, "skipped": "n < 50"}
            logger.info("[partials] %s: n=%d < 50 — skipped", dv_name, n)
            continue
        dv_r = _rank(dv[mask])
        cov_r = np.stack([_rank(cov_mat[k][mask]) for k in range(n_cov)])
        m_design = cov_r[match_idx].T  # (n, 1)
        m2_design = cov_r[match2_idx].T
        resid_c = np.stack([_residualize(cov_r[k], m_design) for k in range(n_cov)])
        norm_c = np.linalg.norm(resid_c, axis=1)
        obs, degen = _partial_row(dv_r, cov_r, resid_c, norm_c, match_idx, m_design)

        # Robustness table (observed only): both matching variables partialled.
        resid_c2 = np.stack([_residualize(cov_r[k], m2_design) for k in range(n_cov)])
        norm_c2 = np.linalg.norm(resid_c2, axis=1)
        obs2, degen2 = _partial_row(dv_r, cov_r, resid_c2, norm_c2, match2_idx, m2_design)

        # Outside-selection column (single-column, never in the max). In PRODUCTION om is a
        # strict subset of mask (the parent's ~9,961 zero-variance features carry NaN
        # answer-side R2), so the null band gets its OWN strata + permutations over om rows —
        # the mask-row draws below cannot be reused (round-1 Issue:
        # outside-selection-band-missing-in-production).
        om = mask & np.isfinite(outside_col)
        ocol_val = None
        out_null: dict | None = None
        if int(om.sum()) >= 50:
            n_o = int(om.sum())
            dv_o = _rank(dv[om])
            m_o = np.stack([_rank(cov_mat[k][om]) for k in match_idx]).T
            oc = _rank(outside_col[om])
            r_d = _residualize(dv_o, m_o)
            r_c = _residualize(oc, m_o)
            # Degeneracy floors are RELATIVE to the rank-vector scale: a CONSTANT dv (or oc)
            # has an exactly-zero centered null (rankdata ties -> one repeated value) while its
            # lstsq residual is float noise GROWING with n — an absolute 1e-12 floor then
            # reports a noise "partial" no null draw can accompany (observed at smoke scale:
            # A_W all-zero over 128,450 features -> |r_d| ~ 3.6e-5 of pure round-off).
            nrd_o = float(np.linalg.norm(r_d))
            nrc_o = float(np.linalg.norm(r_c))
            ok_d = nrd_o > 1e-9 * max(1.0, float(np.linalg.norm(dv_o)))
            ok_c = nrc_o > 1e-9 * max(1.0, float(np.linalg.norm(oc)))
            if ok_d and ok_c:
                ocol_val = float(r_d @ r_c) / (nrd_o * nrc_o)
                mo_rank = m_o[:, 0]
                idx_o = np.arange(n_o)
                order_o = np.lexsort((idx_o, mo_rank))
                strata_o = np.empty(n_o, dtype=np.int64)
                strata_o[order_o] = np.minimum((np.arange(n_o) * 10) // n_o, 9)
                out_null = {
                    "dv": dv_o,
                    "m_rank": mo_rank,
                    "r_c": r_c,
                    "nrc": nrc_o,
                    "idx": idx_o,
                    "rows": [np.where(strata_o == s)[0] for s in range(10)],
                }

        # Stratified permutation null: deciles of the match rank; permute dv WITHIN strata.
        m_rank = cov_r[match_idx[0]]
        idx = np.arange(n)
        order_ = np.lexsort((idx, m_rank))
        strata = np.empty(n, dtype=np.int64)
        strata[order_] = np.minimum((np.arange(n) * 10) // n, 9)
        stratum_rows = [np.where(strata == s)[0] for s in range(10)]

        draws = np.empty((n_draws, n_cov), dtype=np.float32)
        draws_outside = np.full(n_draws, np.nan, dtype=np.float32)
        chunk = 100
        t0 = time.time()
        for d0 in range(0, n_draws, chunk):
            d1 = min(d0 + chunk, n_draws)
            nb = d1 - d0
            perm = np.tile(idx, (nb, 1))
            for rows_s in stratum_rows:
                sub = np.argsort(rng.random((nb, len(rows_s))), axis=1)
                perm[:, rows_s] = rows_s[sub]
            dvp = dv_r[perm]  # (nb, n)
            rd = dvp - dvp.mean(axis=1, keepdims=True)
            mr = m_rank - m_rank.mean()
            beta = (rd @ mr) / max(float(mr @ mr), 1e-30)
            rd = rd - beta[:, None] * mr[None, :]
            nd = np.linalg.norm(rd, axis=1)
            num = rd @ resid_c.T  # (nb, n_cov)
            with np.errstate(divide="ignore", invalid="ignore"):
                part = num / (nd[:, None] * norm_c[None, :])
            part[:, np.asarray(match_idx)] = 0.0
            part[:, norm_c < 1e-9] = 0.0
            draws[d0:d1] = part.astype(np.float32)
            if out_null is not None:
                # Same stratified-permutation scheme, run over the om subset's own rows.
                perm_o = np.tile(out_null["idx"], (nb, 1))
                for rows_s in out_null["rows"]:
                    sub_o = np.argsort(rng.random((nb, len(rows_s))), axis=1)
                    perm_o[:, rows_s] = rows_s[sub_o]
                dvp_o = out_null["dv"][perm_o]  # (nb, n_om)
                rd_o = dvp_o - dvp_o.mean(axis=1, keepdims=True)
                mr_o = out_null["m_rank"] - out_null["m_rank"].mean()
                beta_o = (rd_o @ mr_o) / max(float(mr_o @ mr_o), 1e-30)
                rd_o = rd_o - beta_o[:, None] * mr_o[None, :]
                nd_o = np.linalg.norm(rd_o, axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    draws_outside[d0:d1] = (
                        rd_o @ out_null["r_c"] / (nd_o * out_null["nrc"])
                    ).astype(np.float32)
            if d0 == 0:
                t_unit = time.time() - t0
                if t_pilot is None:
                    t_pilot = t_unit
                    _pilot_gate(
                        args,
                        "partials",
                        t_unit,
                        (n_draws / chunk) * len(dvs),
                        PLANNED_H["partials"],
                    )
            if (d1 % 500 == 0) or d1 == n_draws:
                logger.info(
                    "[partials] %s draws %d/%d %.0fs", dv_name, d1, n_draws, time.time() - t0
                )

        # delta_sel both-sides identity (plan section 5): the observed max and every
        # per-draw null max reduce over the IDENTICAL registered column list — obs and
        # each draws row come from the same resid_c/norm_c stack built from `columns`.
        assert (
            obs.shape == (n_cov,) and draws.shape == (n_draws, n_cov) and n_cov == len(columns)
        ), (
            f"delta_sel column-list mismatch: obs={obs.shape} draws={draws.shape} "
            f"n_cov={n_cov} columns={len(columns)}"
        )
        draw_max = np.abs(draws).max(axis=1)
        band = float(np.quantile(draw_max, 0.975))
        obs_abs = np.abs(obs)
        obs_max = float(obs_abs.max())
        delta_sel = obs_max - band
        np.savez(
            resd / "nulls" / f"stratperm__{dv_name}.npz",
            observed=obs.astype(np.float32),
            observed_robust2=obs2.astype(np.float32),
            draws=draws,
            draws_outside=draws_outside,
            columns=np.asarray(columns),
            degenerate=degen,
            degenerate_robust2=degen2,
            n_complete_case=np.int64(n),
            match=np.str_(MATCH_COV),
        )
        n_outside_band = int((obs_abs > band).sum())
        report["per_dv"][dv_name] = {
            "n_complete_case": n,
            "observed_partials": {c: float(v) for c, v in zip(columns, obs)},
            "observed_partials_robust2": {c: float(v) for c, v in zip(columns, obs2)},
            "degenerate_partial": [c for c, g in zip(columns, degen) if g],
            "max_abs_partial": obs_max,
            "argmax_column": columns[int(obs_abs.argmax())],
            "null_band_p97_5_of_max": band,
            "delta_sel": delta_sel,
            "n_columns_outside_band": n_outside_band,
            "achievable_ceiling": 1.0,
            "band_to_ceiling_margin": 1.0 - band,
            "outside_selection_answer_r2": {
                "observed_partial": ocol_val,
                "band_p97_5": (
                    float(np.nanquantile(np.abs(draws_outside), 0.975))
                    if np.isfinite(draws_outside).any()
                    else None
                ),
            },
        }
        ob = report["per_dv"][dv_name]["outside_selection_answer_r2"]
        assert ob["observed_partial"] is None or ob["band_p97_5"] is not None, (
            f"[partials] {dv_name}: outside-selection band missing while the observed partial "
            f"is reported — a band-less partial invites exactly the selection-symmetry "
            f"misread the plan registers this null against"
        )
        logger.info(
            "[partials] %s: max|partial|=%.4f (%s) band=%.4f delta_sel=%+.4f",
            dv_name,
            obs_max,
            report["per_dv"][dv_name]["argmax_column"],
            band,
            delta_sel,
        )

    # Instrument positive controls (narration gate only — computation proceeds regardless).
    a_w = dvs["A_W"]
    ltc = np.asarray(census["lasttoken_count"], dtype=np.float64)
    act_v2 = cols["activity"]
    m1 = np.isfinite(a_w) & np.isfinite(ltc)
    rho_a, _ = spearmanr(a_w[m1], ltc[m1])
    m2 = np.isfinite(act_v2) & np.isfinite(ltc)
    rho_act, _ = spearmanr(act_v2[m2], ltc[m2])
    report["instrument_positive_controls"] = {
        "spearman_A_W_vs_lasttoken_count": float(rho_a),
        "threshold_A": 0.3,
        "spearman_activity_vs_lasttoken_count": float(rho_act),
        "threshold_activity": 0.2,
        "narration_gate_pass": bool(rho_a >= 0.3 and rho_act >= 0.2),
        "note": "gates NARRATION of the kill branch only; computation proceeds regardless",
    }

    # Corpus-fold + split-half stability of the read.
    def _pair_rho(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) < 50:
            return None, int(m.sum())
        r, _ = spearmanr(x[m], y[m])
        return float(r), int(m.sum())

    a_lm = np.asarray(lad_w["a_lmsys"], dtype=np.float64)
    a_wc = np.asarray(lad_w["a_wildchat"], dtype=np.float64)
    a_ev = np.asarray(lad_w["a_even"], dtype=np.float64)
    a_od = np.asarray(lad_w["a_odd"], dtype=np.float64)
    car_lm = np.asarray(carried["carried_lmsys"], dtype=np.float64)
    car_wc = np.asarray(carried["carried_wildchat"], dtype=np.float64)
    r1, n1 = _pair_rho(a_lm, a_wc)
    r2_, n2 = _pair_rho(a_ev, a_od)
    r3, n3 = _pair_rho(car_lm, car_wc)
    report["stability"] = {
        "A_W_corpus_half_spearman": {"rho": r1, "n": n1},
        "A_W_split_half_even_odd_spearman": {"rho": r2_, "n": n2},
        "carried_corpus_half_spearman": {"rho": r3, "n": n3},
    }

    # Per-half partials for the holdout-dependent DVs (observed only).
    per_half = {}
    for nm, vec in (
        ("A_W_lmsys", a_lm),
        ("A_W_wildchat", a_wc),
        ("carried_lmsys", car_lm),
        ("carried_wildchat", car_wc),
    ):
        m = cov_finite & np.isfinite(vec)
        if int(m.sum()) < 50:
            per_half[nm] = None
            continue
        dv_r = _rank(vec[m])
        cov_rr = np.stack([_rank(cov_mat[k][m]) for k in range(n_cov)])
        md = cov_rr[match_idx].T
        rc = np.stack([_residualize(cov_rr[k], md) for k in range(n_cov)])
        ncrm = np.linalg.norm(rc, axis=1)
        oh, _dg = _partial_row(dv_r, cov_rr, rc, ncrm, match_idx, md)
        per_half[nm] = {c: float(v) for c, v in zip(columns, oh)}
    report["per_half_partials"] = per_half

    _write_json(resd / "predictor_partials.json", report)
    _upload_phase_results(args, "results-partials")
    _write_json(sentinel, {"regime": regime, "written": _now_iso()})
    return 0


# ── phase confirm-b ───────────────────────────────────────────────────────────


def _confirm_b_gram(args, reg: dict) -> dict:
    """Assemble the census-restricted standardized Gram + XtY inputs for the B refit.

    Returns handles; persists gram + standardizer to <out>/B_gram/ for the GPU-cell path.
    """
    asm = _assembled(args)
    restrict = np.load(asm / "restrict_ids.npy")
    d_b = int(len(restrict))
    psil = _psil_csr(args, reg)
    tr = reg["tr"]
    psil_r = psil[:, restrict].tocsr()
    psil_tr = psil_r[tr]
    n_tr = len(tr)
    mu = np.asarray(psil_tr.sum(axis=0), dtype=np.float64).ravel() / n_tr
    s2 = np.asarray(psil_tr.power(2).sum(axis=0), dtype=np.float64).ravel()
    var = np.clip((s2 - n_tr * mu**2) / max(n_tr - 1, 1), 0.0, None)
    sd = np.sqrt(var) + 1e-9  # _GramFactor convention
    g_raw = np.asarray((psil_tr.T @ psil_tr).todense(), dtype=np.float64)
    a_std = (g_raw - n_tr * np.outer(mu, mu)) / np.outer(sd, sd)
    del g_raw
    bdir = _out(args) / "B_gram"
    bdir.mkdir(parents=True, exist_ok=True)
    np.save(bdir / "gram_std.f64.npy", a_std)
    np.save(bdir / "mu.f64.npy", mu)
    np.save(bdir / "sd.f64.npy", sd)
    np.save(bdir / "restrict_ids.npy", restrict)
    _write_json(bdir / "B_gram_meta.json", {"d_B": d_b, "n_train": n_tr, "meta": _meta()})
    return {"a_std": a_std, "mu": mu, "sd": sd, "restrict": restrict, "psil_r": psil_r, "d_b": d_b}


def _confirm_b_val_block(y_val, ev, rot_b, ymu, s_eig, c0, c1, dev):
    """One output-block validation-SSE contribution for the confirm-B lambda sweep.

    `DSF._val_block_ss` slices ALL THREE of its data args INTERNALLY by its (c0, c1) window,
    so every arg here reaches it sliced exactly once by the SAME window: y_val and ymu are
    sliced HERE to the block's columns, rot_b already IS the block's columns, and the helper
    window is the identity (0, c1 - c0). Passing y_val FULL-WIDTH with a (0, width) window
    scores every block against block 0's targets and 8x-counts block 0's SST (the round-1
    Critical: confirm-b-val-target-misalignment). Pinned by
    tests/test_issue2163_ctxread.py::test_confirm_b_blockwise_val_sweep_matches_whole.
    """
    return DSF._val_block_ss(
        y_val[:, c0:c1], ev, torch.as_tensor(rot_b, device=dev), ymu[c0:c1], s_eig, 0, c1 - c0
    )


def _confirm_b_fit(args, reg: dict, g: dict, dev: torch.device) -> int:
    """The B fit leg: eigh + two-pass blockwise solve + scoring + bnorm + kNN + contrast."""
    resd = _results(args)
    ystore = _ystore(args, reg)
    tr, va, ho = reg["tr"], reg["va"], reg["ho"]
    n_tr = len(tr)
    d_b = g["d_b"]
    restrict = g["restrict"]
    mu_t = torch.as_tensor(g["mu"], dtype=torch.float64, device=dev)
    sd_t = torch.as_tensor(g["sd"], dtype=torch.float64, device=dev)

    # eigh pilot (venue switch handled by the caller for the CPU path).
    a_t = torch.as_tensor(g["a_std"], dtype=torch.float64, device=dev)
    t0 = time.time()
    s_eig, u_mat = torch.linalg.eigh(a_t)
    s_eig = torch.clamp(s_eig, min=0.0)
    logger.info("[confirm-b] eigh(d_B=%d) %.0fs on %s", d_b, time.time() - t0, dev)
    del a_t

    def _std_rows(rows: np.ndarray) -> torch.Tensor:
        """Standardized dense psi block for a row subset (torch fp64 on dev)."""
        dense = np.asarray(g["psil_r"][rows].todense(), dtype=np.float64)
        return (torch.as_tensor(dense, device=dev) - mu_t) / sd_t

    s1, _ = ystore.col_stats(tr, "mean")
    ymu_np = s1 / n_tr
    ymu = torch.as_tensor(ymu_np, dtype=torch.float64, device=dev)
    ev = _std_rows(va) @ u_mat
    eh_rows = _std_rows(ho)
    eh = eh_rows @ u_mat
    del eh_rows
    y_val = torch.as_tensor(ystore.csr_rows(va, "mean").toarray(), dtype=torch.float64, device=dev)
    y_ho_csc = ystore.csc_rows(ho, "mean")
    psil_tr_t = g["psil_r"][tr].T.tocsr()  # (d_B, n_tr) CSR of psi^T

    rot_dir = _out(args) / "B_rot"
    rot_dir.mkdir(parents=True, exist_ok=True)
    ssr = np.zeros(len(LAMBDAS))
    sst = 0.0
    n_blocks = (DICT_SIZE + OUT_BLOCK - 1) // OUT_BLOCK
    t0 = time.time()
    for bi, c0 in enumerate(range(0, DICT_SIZE, OUT_BLOCK)):
        c1 = min(c0 + OUT_BLOCK, DICT_SIZE)
        # XtY_std block: D_s^-1 [Psi^T Y - mu (col sums of Y)]  (centering exact: colsum_std=0)
        yb_tr = sp.csr_matrix(ystore.csc_rows(tr, "mean")[:, c0:c1])
        xty_b = np.asarray((psil_tr_t @ yb_tr).todense(), dtype=np.float64)
        ysum_b = np.asarray(yb_tr.sum(axis=0), dtype=np.float64).ravel()
        xty_b = (xty_b - np.outer(g["mu"], ysum_b)) / g["sd"][:, None]
        rot_b = (u_mat.T @ torch.as_tensor(xty_b, device=dev)).cpu().numpy()
        np.save(rot_dir / f"rot_{bi:02d}.npy", rot_b.astype(np.float32))
        r, t = _confirm_b_val_block(y_val, ev, rot_b, ymu, s_eig, c0, c1, dev)
        ssr += r
        sst += t
        if bi == 0:
            _pilot_gate(
                args,
                "confirm_b" if dev.type != "cuda" else "confirm_b_gpu",
                time.time() - t0,
                2.0 * n_blocks,
                PLANNED_H["confirm_b" if dev.type != "cuda" else "confirm_b_gpu"],
            )
        logger.info("[confirm-b] val block %d/%d %.0fs", bi + 1, n_blocks, time.time() - t0)
    del y_val
    val_r2 = 1.0 - ssr / max(sst, 1e-30)
    best = int(np.nanargmax(val_r2))
    lam = float(LAMBDAS[best])
    logger.info("[confirm-b] selected lambda=%.6g val_R2=%.6f", lam, val_r2[best])

    inv = torch.as_tensor(1.0 / (np.asarray(s_eig.cpu()) + lam), device=dev)
    s1h, s2h = ystore.col_stats(ho, "mean")
    ss_tot = s2h - (s1h**2) / len(ho)
    ss_res = np.zeros(DICT_SIZE, dtype=np.float64)
    bnorm2 = np.zeros(d_b, dtype=np.float64)
    pane = np.asarray(reg["f_out"], dtype=np.int64)
    panel_block = np.lib.format.open_memmap(
        _out(args) / "B_panel_block.f32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(d_b, len(pane)),
    )
    n_probe = min(args.knn_probes, len(ho))
    pred_probe = np.empty((n_probe, DICT_SIZE), dtype=np.float32)
    eh_inv = eh * inv
    t0 = time.time()
    for bi, c0 in enumerate(range(0, DICT_SIZE, OUT_BLOCK)):
        c1 = min(c0 + OUT_BLOCK, DICT_SIZE)
        rot_b = torch.as_tensor(
            np.load(rot_dir / f"rot_{bi:02d}.npy").astype(np.float64), device=dev
        )
        b_b = (u_mat @ (inv[:, None] * rot_b)).cpu().numpy()
        bnorm2 += (b_b**2).sum(axis=1)
        in_pane = (pane >= c0) & (pane < c1)
        if in_pane.any():
            panel_block[:, np.where(in_pane)[0]] = b_b[:, pane[in_pane] - c0].astype(np.float32)
        pred = (eh_inv @ rot_b).cpu().numpy() + ymu_np[c0:c1]
        yb = DSF._dense_cols(y_ho_csc, c0, c1).astype(np.float64)
        ss_res[c0:c1] = ((yb - pred) ** 2).sum(axis=0)
        pred_probe[:, c0:c1] = pred[:n_probe].astype(np.float32)
        del rot_b, b_b, pred, yb
        logger.info("[confirm-b] holdout block %d/%d %.0fs", bi + 1, n_blocks, time.time() - t0)
    panel_block.flush()
    score = DSF._score(ss_res, ss_tot, reg["f_out"], lam)
    bnorm = np.sqrt(bnorm2)
    np.savez(
        resd / "bnorm.npz",
        restrict_ids=restrict.astype(np.int32),
        bnorm=bnorm.astype(np.float64),
        selected_lambda=np.float64(lam),
    )

    # kNN retrieval on the B map's holdout predictions (identity-bias N/A: dim mismatch).
    pool = _dense_ho_pool(ystore, ho)
    knn_b = _knn_block(pred_probe, pool, n_probe)
    del pool, pred_probe

    # Primary check: Spearman(||B[j]||, U_j) on the restriction, overall + activity deciles.
    from scipy.stats import spearmanr

    lad_w = np.load(resd / "read_ladder__W.npz")
    u_w = np.asarray(lad_w["u"], dtype=np.float64)[restrict]
    census = np.load(_assembled(args) / "census.npz")
    ltc = np.asarray(census["lasttoken_count"], dtype=np.float64)[restrict]
    rho, _p = spearmanr(bnorm, u_w)
    idx = np.arange(d_b)
    order_ = np.lexsort((idx, ltc))
    bins = np.empty(d_b, dtype=np.int64)
    bins[order_] = np.minimum((np.arange(d_b) * 10) // max(d_b, 1), 9)
    per_dec = []
    for dd in range(10):
        sel = bins == dd
        if int(sel.sum()) >= 10:
            r_, _ = spearmanr(bnorm[sel], u_w[sel])
            per_dec.append({"decile": dd, "spearman": float(r_), "n": int(sel.sum())})
        else:
            per_dec.append({"decile": dd, "spearman": None, "n": int(sel.sum())})

    _write_json(
        resd / "confirm_B.json",
        {
            "d_B": d_b,
            "floor_last": int(2 if args.smoke else args.floor_last),
            "cap": int(args.cap_db),
            "n_train": n_tr,
            "n_train_over_d_B": n_tr / max(d_b, 1),
            "selected_lambda": lam,
            "selector": "val-carve sweep over the 23-lambda grid (no GCV)",
            "val_r2_by_lambda": {str(float(a)): float(b) for a, b in zip(LAMBDAS, val_r2)},
            "holdout_score": score,
            "spearman_bnorm_vs_U_W": float(rho),
            "spearman_bnorm_vs_U_W_per_activity_decile": per_dec,
            "knn": {**knn_b, "identity_bias": "inapplicable — d_B -> 131072 dim mismatch"},
            "device": str(dev),
            "meta": _meta(),
        },
    )
    shutil.rmtree(rot_dir, ignore_errors=True)
    return 0


def phase_confirm_b(args) -> int:
    """Phase 6 (CPU): Gram assembly + eigh-pilot venue switch + the fit leg."""
    sentinel = _out(args) / "confirm_b.done.json"
    regime = _regime(args)
    if _resume_ok(args, sentinel, regime, "confirm-b"):
        return 0
    assert_out_root_headroom(Path(args.work), 3 if args.smoke else 25, phase="confirm-b")

    dev = _device(args)
    reg = _registry(args)
    g = _confirm_b_gram(args, reg)
    d_b = g["d_b"]

    # eigh venue pilot: time eigh at min(8192, d_B), cubic-extrapolate to d_B.
    d_pilot = min(8_192, d_b)
    a_sub = torch.as_tensor(g["a_std"][:d_pilot, :d_pilot], dtype=torch.float64)
    t0 = time.time()
    torch.linalg.eigh(a_sub)
    t_eigh = time.time() - t0
    del a_sub
    projected_h = t_eigh * (d_b / max(d_pilot, 1)) ** 3 / 3600.0
    logger.info(
        "pilot: confirm_b_eigh unit_s=%.2f projected_h=%.3f venue_switch_h=%.1f (d_B=%d)",
        t_eigh,
        projected_h,
        EIGH_VENUE_SWITCH_H,
        d_b,
    )
    if projected_h > EIGH_VENUE_SWITCH_H and dev.type != "cuda" and not args.smoke:
        # Venue switch: upload the Gram bundle + defer the fit leg to the GPU cell.
        base = _hf_base(args)
        _upload_tree(args, _out(args) / "B_gram", f"{base}/analysis_tensors/B_gram", "B_gram")
        _write_json(
            _results(args) / "confirm_B_venue.json",
            {
                "deferred_to": "confirm-b-gpu",
                "projected_eigh_h_cpu": projected_h,
                "d_B": d_b,
                "meta": _meta(),
            },
        )
        logger.info("[confirm-b] venue switch fired — fit leg deferred to the GPU cell")
        _write_json(sentinel, {"regime": regime, "written": _now_iso(), "deferred": True})
        return 0
    rc = _confirm_b_fit(args, reg, g, dev)
    if not args.skip_upload:
        _upload_phase_results(args, "results-confirm-b")
        _upload_panel_block(args)
    _write_json(sentinel, {"regime": regime, "written": _now_iso()})
    return rc


def phase_confirm_b_gpu(args) -> int:
    """Phase 6-alt (GPU cell): re-stage inputs + re-run census, then the fit leg on CUDA.

    Deliberate deviation from plan section 9 `off_pod_phases` (which sketches staging only the
    B_gram bundle from HF): the fit leg ALSO needs the census-derived Y/psi CSR stores, which
    are NOT persisted to HF — the answer-mean CSR is a byte-derivable re-pack of the staged
    store, and upload-verify deliberately re-derives it rather than re-uploading ~2 GB — so a
    bundle-only staging cannot feed `_confirm_b_fit`. The cell therefore re-runs phase_stage +
    phase_census (deterministic; the census resume sentinel makes re-runs cheap) and recomputes
    the small restricted Gram from the restaged inputs: ~1 h extra GPU-pod wall, inside the
    plan's 4 GPU-h allowance.

    Uploads its results AND the plan-declared 16,384-panel sub-block itself — the CPU path's
    phase_upload_verify never runs on this pod, so without the panel upload here the
    plan-section-4 persisted artifact would be LOST at pod termination (round-1 Issue:
    gpu-cell-panel-block-not-uploaded).
    """
    if not torch.cuda.is_available():
        raise SystemExit("[confirm-b-gpu] CUDA required for the GPU cell")
    args.device = "cuda"
    phase_stage(args)
    phase_census(args)
    reg = _registry(args)
    g = _confirm_b_gram(args, reg)
    rc = _confirm_b_fit(args, reg, g, torch.device("cuda"))
    if not args.skip_upload:
        base = _hf_base(args)
        _upload_tree(args, _results(args), f"{base}/results", "results-gpu-cell")
        _upload_panel_block(args)
    return rc


# ── phase upload-verify ───────────────────────────────────────────────────────


def phase_upload_verify(args) -> int:
    """Phase 7: upload results + bundles + census artifacts; exact-set verify per prefix."""
    outd = _out(args)
    resd = _results(args)
    resd.mkdir(parents=True, exist_ok=True)
    # Small out-root JSONs ride the results prefix.
    for name in ("census.json", "repro_gates.json", "stage_report.json"):
        src = outd / name
        if src.exists():
            shutil.copy2(src, resd / name)
    src = _assembled(args) / "census.npz"
    if src.exists():
        shutil.copy2(src, resd / "census.npz")

    base = _hf_base(args)
    uploaded: dict[str, list[str]] = {}
    uploaded["results"] = _upload_tree(args, resd, f"{base}/results", "results")
    for bundle in ("W_bundle", "M_bundle", "B_gram"):
        d = outd / bundle
        if d.exists():
            uploaded[bundle] = _upload_tree(args, d, f"{base}/analysis_tensors/{bundle}", bundle)
    panel_paths = _upload_panel_block(args)
    if panel_paths:
        uploaded["B_panel"] = panel_paths
    # psil CSR (small; persist-by-default). The answer-mean CSR is a byte-derivable re-pack of
    # the already-uploaded HF store (regen: this driver's census phase) — not re-uploaded.
    asm = _assembled(args)
    psil_files = [
        "psil_indptr.npy",
        "psil_indices.i32.npy",
        "psil_val.f32.npy",
        "restrict_ids.npy",
        "coverage_rows.npy",
    ]
    psil_dir = outd / "_psil_upload"
    psil_dir.mkdir(exist_ok=True)
    for n in psil_files:
        if (asm / n).exists():
            shutil.copy2(asm / n, psil_dir / n)
    uploaded["psil_csr"] = _upload_tree(
        args, psil_dir, f"{base}/analysis_tensors/psil_csr", "psil_csr"
    )

    verdict = {
        "prefixes": sorted({f"{base}/results", f"{base}/analysis_tensors"}),
        "uploaded_counts": {k: len(v) for k, v in uploaded.items()},
        "smoke": bool(args.smoke),
        "meta": _meta(),
    }
    _write_json(outd / "upload_verified.json", verdict)
    logger.info("[upload-verify] done: %s", verdict["uploaded_counts"])
    return 0


# ── phase harvest + figures ───────────────────────────────────────────────────


def phase_harvest(args) -> int:
    """Phase 8a (VM): download the results prefix into the harvest dir (small files)."""
    base = _hf_base(args)
    dest = (
        Path(args.harvest_out)
        if args.harvest_out
        else (PROJECT_ROOT / "eval_results" / "issue_2163")
    )
    dest.mkdir(parents=True, exist_ok=True)
    staged = hub.stage_hub_prefix(HF_DATA_REPO, f"{base}/results", dest, repo_type="dataset")
    # Mirror-root layout: flatten <dest>/<base>/results/* -> <dest>/*.
    mirror = dest / base / "results"
    if mirror.exists():
        for p in sorted(mirror.rglob("*")):
            if p.is_file():
                rel = p.relative_to(mirror)
                tgt = dest / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, tgt)
        shutil.rmtree(dest / base.split("/")[0])
    logger.info("[harvest] %d files -> %s", len(staged), dest)
    _write_json(dest / "harvest_meta.json", {"n_files": len(staged), "meta": _meta()})
    return 0


def _fig_dir(args) -> Path:
    """Figures output directory."""
    return Path(args.figures_out) if args.figures_out else (PROJECT_ROOT / "figures" / "issue_2163")


def phase_figures(args) -> int:
    """Phase 8b (VM): render headline + exploratory figures from the harvested results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    src = (
        Path(args.harvest_out)
        if args.harvest_out
        else (PROJECT_ROOT / "eval_results" / "issue_2163")
    )
    fdir = _fig_dir(args)
    fdir.mkdir(parents=True, exist_ok=True)
    pal = paper_palette(6)

    partials = _read_json(src / "predictor_partials.json")
    lad_w = np.load(src / "read_ladder__W.npz")
    carried = np.load(src / "carried.npz")
    census = np.load(src / "census.npz")
    ltc = np.asarray(census["lasttoken_count"], dtype=np.float64)
    a_w = np.asarray(lad_w["a"], dtype=np.float64)
    u_w = np.asarray(lad_w["u"], dtype=np.float64)
    car = np.asarray(carried["carried"], dtype=np.float64)

    # 1) Hero: partials forest (primary DV) with the selection-symmetric band + ceiling.
    dv = "logU_W"
    if dv in partials["per_dv"] and "observed_partials" in partials["per_dv"][dv]:
        blk = partials["per_dv"][dv]
        cols_ = list(blk["observed_partials"])
        vals = np.array([blk["observed_partials"][c] for c in cols_])
        order_ = np.argsort(np.abs(vals))
        fig, ax = plt.subplots(figsize=(7, 8))
        geo = set(partials["geometry_definitional"])
        colors = [pal[1] if cols_[i] in geo else pal[0] for i in order_]
        ax.barh(np.arange(len(cols_)), vals[order_], color=colors)
        ax.set_yticks(np.arange(len(cols_)), [cols_[i] for i in order_], fontsize=7)
        band = blk["null_band_p97_5_of_max"]
        ax.axvline(band, color="k", linestyle="--", linewidth=1)
        ax.axvline(-band, color="k", linestyle="--", linewidth=1)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel(f"partial Spearman (| {partials['match_covariate']})")
        ax.set_title(f"{dv}: activity-matched partials vs stratified-permutation band")
        savefig_paper(fig, "fig_partials_forest_logU_W", dir=fdir)
        plt.close(fig)

    # 2) Read vs carried scatter — round 2: TWO panels, so the section's headline quantity
    #    (per-unit U vs carried, Spearman ~ -0.05) is plotted, not only the frequency-weighted
    #    |A| vs carried (+0.59) that round 1 showed alone. y is linear in both panels.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    mu = np.isfinite(u_w) & (u_w > 0) & np.isfinite(car)
    dec_u = np.minimum((np.argsort(np.argsort(ltc[mu])) * 10) // max(int(mu.sum()), 1), 9)
    axes[0].scatter(u_w[mu], np.clip(car[mu], -1, 1), s=2, c=dec_u, cmap="viridis", alpha=0.3)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("U_j — per-unit read (W map, log)")
    axes[0].set_ylabel("Carried_j (univariate held-out R2)")
    axes[0].set_title("per-unit read vs carried")
    m = np.isfinite(a_w) & np.isfinite(car)
    dec = np.minimum((np.argsort(np.argsort(ltc[m])) * 10) // max(int(m.sum()), 1), 9)
    sc = axes[1].scatter(
        np.abs(a_w[m]) + 1e-12, np.clip(car[m], -1, 1), s=2, c=dec, cmap="viridis", alpha=0.3
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("|A_j| — frequency-weighted read (log)")
    axes[1].set_title("frequency-weighted read vs carried")
    fig.colorbar(sc, ax=list(axes), label="last-token activity decile")
    savefig_paper(fig, "fig_read_vs_carried", dir=fdir)
    plt.close(fig)

    # 3) Positive-control scatters (raw).
    for name, vec, thr in (
        ("fig_A_vs_activity", a_w, 0.3),
        ("fig_U_vs_activity", u_w, None),
    ):
        mm = np.isfinite(vec) & np.isfinite(ltc)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(ltc[mm] + 1, np.abs(vec[mm]) + 1e-12, s=2, alpha=0.2, color=pal[0])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("last-token active count (train) + 1")
        ax.set_ylabel("|value|")
        ax.set_title(name.replace("fig_", "").replace("_", " "))
        savefig_paper(fig, name, dir=fdir)
        plt.close(fig)

    # 4) Matched-n gradient (+ raw per-feature companion).
    amn = _read_json(src / "answer_matchedn.json")
    dec_rows = amn["per_decile"]
    xs = [r["decile"] for r in dec_rows]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for key, lbl, c in (
        ("median_pooled_r2", "pooled R2", pal[0]),
        ("median_conditional_r2", "conditional R2", pal[1]),
        ("median_matchedn_r2", "matched-n R2", pal[2]),
    ):
        ys = [r[key] for r in dec_rows]
        ax.plot(xs, [np.nan if y is None else y for y in ys], marker="o", label=lbl, color=c)
    ax.set_xlabel("answer-side activity decile (holdout active count)")
    ax.set_ylabel("median per-feature R2")
    ax.legend()
    savefig_paper(fig, "fig_matchedn_gradient", dir=fdir)
    plt.close(fig)
    pfm = np.load(src / "answer_matchedn_perfeature.npz")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(pfm["n_active_ho"] + 1, pfm["r2_pooled"], s=2, alpha=0.15, color=pal[0])
    ax.set_xscale("log")
    ax.set_xlabel("holdout answer-side active count + 1")
    ax.set_ylabel("per-feature pooled R2")
    savefig_paper(fig, "fig_matchedn_raw_perfeature", dir=fdir)
    plt.close(fig)

    # 5) Signed dSSE histogram (negative cross-terms reported, never clipped).
    dsse = np.asarray(lad_w["dsse"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fin = np.isfinite(dsse) & (dsse != 0)
    ax.hist(np.sign(dsse[fin]) * np.log10(np.abs(dsse[fin]) + 1e-30), bins=120, color=pal[0])
    ax.set_xlabel("sign(dSSE) * log10|dSSE| (W map)")
    ax.set_ylabel("features")
    savefig_paper(fig, "fig_dsse_sign", dir=fdir)
    plt.close(fig)

    # 6) kNN bars.
    knn = _read_json(src / "knn_baselines.json")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels, vals, colors = [], [], []
    entries = [("W", knn["W"]), ("M", knn["M"])]
    cb = src / "confirm_B.json"
    if cb.exists():
        entries.append(("B", _read_json(cb)["knn"]))
    map_names = {"W": "dense->SAE", "M": "dense->dense", "B": "last-token refit"}
    for i, (nm, blk) in enumerate(entries):
        for j, metric in enumerate(("euclidean", "cosine")):
            labels.append(f"{map_names.get(nm, nm)}\n{metric}")
            vals.append(blk[metric]["acc_at_k"]["1"] if "acc_at_k" in blk[metric] else np.nan)
            colors.append(pal[i])
    ax.bar(np.arange(len(vals)), vals, color=colors)
    ax.set_xticks(np.arange(len(vals)), labels, rotation=45, fontsize=8)
    ax.set_ylabel("retrieval acc@1 (chance = 1/n_pool)")
    savefig_paper(fig, "fig_knn_bars", dir=fdir)
    plt.close(fig)

    # 7) Corpus-half stability scatter.
    a_lm = np.asarray(lad_w["a_lmsys"], dtype=np.float64)
    a_wc = np.asarray(lad_w["a_wildchat"], dtype=np.float64)
    mm = np.isfinite(a_lm) & np.isfinite(a_wc)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(np.abs(a_lm[mm]) + 1e-12, np.abs(a_wc[mm]) + 1e-12, s=2, alpha=0.15, color=pal[0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|A_j| LMSYS half")
    ax.set_ylabel("|A_j| WildChat half")
    savefig_paper(fig, "fig_corpus_half_A", dir=fdir)
    plt.close(fig)

    # 8) Null-band histograms per DV.
    for dv_name in ("logU_W", "A_W", "carried", "U_M"):
        npz_path = src / "nulls" / f"stratperm__{dv_name}.npz"
        if not npz_path.exists():
            continue
        z = np.load(npz_path)
        draws = np.abs(np.asarray(z["draws"], dtype=np.float64)).max(axis=1)
        obs = float(np.abs(np.asarray(z["observed"], dtype=np.float64)).max())
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(draws, bins=60, color=pal[3], alpha=0.8)
        ax.axvline(obs, color=pal[1], linewidth=2, label="observed max |partial|")
        ax.axvline(float(np.quantile(draws, 0.975)), color="k", linestyle="--", label="null p97.5")
        ax.set_xlabel("max |partial| over the selection set")
        ax.set_title(dv_name)
        ax.legend()
        savefig_paper(fig, f"fig_stratperm_band_{dv_name}", dir=fdir)
        plt.close(fig)

    # 9) bnorm vs U (confirmatory), when present.
    bn_path = src / "bnorm.npz"
    if bn_path.exists():
        bn = np.load(bn_path)
        rid = np.asarray(bn["restrict_ids"], dtype=np.int64)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(u_w[rid] + 1e-12, np.asarray(bn["bnorm"]) + 1e-12, s=2, alpha=0.2, color=pal[0])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("U_j (W map)")
        ax.set_ylabel("||B[j]|| (census restriction)")
        savefig_paper(fig, "fig_bnorm_vs_U", dir=fdir)
        plt.close(fig)

    # 10) Round-2 population split (issue2163_population_partials.py output, when present):
    #     paired forest of the activity-matched partials within the train-active vs the
    #     never-active populations, each against its OWN stratified-permutation band, plus a
    #     raw per-feature view of the composition (U_j vs proj_var, colored by population).
    pp_path = src / "population_partials.json"
    if pp_path.exists():
        pp = _read_json(pp_path)
        act = pp["populations"]["train_active"]
        nev = pp["populations"]["never_active"]
        degen_p = set(act["degenerate_partial"]) | set(nev["degenerate_partial"])
        cols_p = [c for c in pp["selection_columns"] if c not in degen_p]
        av = np.array([act["observed_partials"][c] for c in cols_p])
        nv = np.array([nev["observed_partials"][c] for c in cols_p])
        order_p = np.argsort(np.abs(nv))
        y = np.arange(len(cols_p))
        fig, ax = plt.subplots(figsize=(7.5, 8))
        ax.barh(
            y + 0.2,
            nv[order_p],
            height=0.38,
            color=pal[1],
            label=f"never-active at last token (n={nev['n']:,})",
        )
        ax.barh(
            y - 0.2,
            av[order_p],
            height=0.38,
            color=pal[0],
            label=f"train-active at last token (n={act['n']:,})",
        )
        ax.set_yticks(y, [cols_p[i] for i in order_p], fontsize=7)
        for b, c in ((nev["band_p97_5_of_max"], pal[1]), (act["band_p97_5_of_max"], pal[0])):
            ax.axvline(b, color=c, linestyle="--", linewidth=1)
            ax.axvline(-b, color=c, linestyle="--", linewidth=1)
        ax.set_xlabel("partial Spearman vs log U_j (given lasttoken_count)")
        ax.legend(loc="lower left", fontsize=8)
        savefig_paper(fig, "fig_population_partials", dir=fdir)
        plt.close(fig)
        cov_p = (
            Path(args.local_covariates) / "fullwidth_covariates_v2.npz"
            if args.local_covariates
            else None
        )
        if cov_p is not None and cov_p.exists():
            pv = np.asarray(np.load(cov_p)["proj_var"], dtype=np.float64)
            mm2 = np.isfinite(pv) & (pv > 0) & np.isfinite(u_w) & (u_w > 0)
            nevm = mm2 & (ltc == 0)
            actm = mm2 & (ltc > 0)
            fig, ax = plt.subplots(figsize=(6.5, 5))
            ax.scatter(
                pv[nevm],
                u_w[nevm],
                s=2,
                alpha=0.08,
                color=pal[1],
                label=f"never-active (n={int(nevm.sum()):,})",
            )
            ax.scatter(
                pv[actm],
                u_w[actm],
                s=2,
                alpha=0.3,
                color=pal[0],
                label=f"train-active (n={int(actm.sum()):,})",
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("proj_var — residual-state variance along the decoder direction (log)")
            ax.set_ylabel("U_j — per-unit read (log)")
            ax.legend(loc="lower left", fontsize=8, markerscale=4)
            savefig_paper(fig, "fig_population_raw_projvar", dir=fdir)
            plt.close(fig)

    logger.info("[figures] rendered into %s", fdir)
    return 0


# ── import-check mode ─────────────────────────────────────────────────────────


def run_import_check() -> int:
    """Execute every deferred import + signature-bind the cross-module call shapes.

    The deferred (function-body) imports of this module are executed here explicitly, and the
    imported helpers' signatures are bound with each call site's statically-known shape.
    """
    import inspect

    from huggingface_hub import HfApi  # deferred in upload paths

    import issue1482_sae as SAE
    from issue1482_early_layer import _stage_scratch_meta
    from issue1738_sae_arm import _GramFactor

    # Signature binds (call-shape checks; placeholder values only).
    inspect.signature(_GramFactor.__init__).bind(object(), object(), object(), object(), 1)
    inspect.signature(_GramFactor.xty_centered).bind(object(), object(), object(), object())
    inspect.signature(_stage_scratch_meta).bind(SimpleNamespace(scratch=Path("/tmp")))
    inspect.signature(SAE.BatchTopKSAE.load).bind(64, "cpu", Path("/tmp"), layer=19)
    inspect.signature(SAE.BatchTopKSAE.ensure_downloaded).bind(64, Path("/tmp"), layer=19)
    inspect.signature(hub.stage_hub_file).bind("r", "p", Path("/tmp/x"), repo_type="dataset")
    inspect.signature(hub.stage_hub_prefix).bind("r", "p", Path("/tmp"), repo_type="dataset")
    inspect.signature(hub.verify_repo_paths_uploaded).bind(
        HfApi(), "r", ["a"], path_in_repo="p", repo_type="dataset"
    )
    inspect.signature(hub._upload).bind(Path("/tmp/x"), "r", "dataset", "p", upload_as_file=True)
    inspect.signature(assert_out_root_headroom).bind(Path("/tmp"), 1, phase="x")
    inspect.signature(DSF._xty).bind(object(), object(), object(), "auto")
    inspect.signature(DSF._val_block_ss).bind(*([object()] * 5), 0, 1)
    inspect.signature(DSF._ridge_holdout).bind(*([object()] * 6), object())
    inspect.signature(DSF._score).bind(object(), object(), object(), 1.0)
    inspect.signature(DSF._score_perfeature).bind(object(), object())
    inspect.signature(DSF._dense_cols).bind(object(), 0, 1)
    inspect.signature(DSF.YStore.__init__).bind(object(), Path("/tmp"), 1, 1, poolings=("mean",))
    inspect.signature(identity_bias_predict).bind(object(), object(), object())
    inspect.signature(knn_retrieval).bind(
        object(),
        object(),
        ks=(1,),
        metric="euclidean",
        pool=object(),
        true_pool_idx=object(),
    )
    inspect.signature(hub.retry_transient).bind(lambda: None, what="x")

    # Figures-phase deferred imports.
    import matplotlib  # noqa: F401

    from explore_persona_space.analysis.paper_plots import (  # noqa: F401
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    _check_phase_args_defined()

    print(
        "import-check OK: all deferred imports resolved, call shapes bound, "
        "and every phase's args.<attr> is argparser-defined"
    )
    return 0


def _check_phase_args_defined() -> None:
    """Assert every `args.<attr>` referenced ANYWHERE in this module is argparser-defined.

    An undefined attribute is an AttributeError that fires only when the referencing code RUNS —
    invisible to a smoke exercising a subset of phases, and invisible to the deferred-import checks
    above (an `args.X` reference is not an import). Three shipped here: both VM-side phases
    (`harvest`, `figures`) read an undefined `args.harvest_out`, and `_fig_dir` read an undefined
    `args.figures_out`.

    The scan is WHOLE-MODULE, deliberately. A first version scanned only the `PHASES` function
    bodies and missed `args.figures_out` because it lives in `_fig_dir` — a helper the phase calls.
    Any per-function scope is escapable by moving the reference one call deeper, so the only
    non-escapable scope is the file. Raises SystemExit naming every gap.
    """
    import re

    src = Path(__file__).read_text(encoding="utf-8")
    defined = {
        m.group(1).replace("-", "_")
        for m in re.finditer(r'ap\.add_argument\(\s*"--([a-z0-9-]+)"', src)
    }
    defined |= {"phase", "import_check"}  # add_argument dests not matching the --flag pattern

    missing = sorted({a for a in re.findall(r"args\.([a-z_][a-z0-9_]*)", src) if a not in defined})
    if missing:
        raise SystemExit(
            "import-check FAILED: module references argparser-undefined args attributes "
            f"({', '.join(missing)}) — each is an AttributeError that fires only when the "
            "referencing code path runs"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

PHASES = {
    "upload-inputs": phase_upload_inputs,
    "stage": phase_stage,
    "census": phase_census,
    "fit-maps": phase_fit_maps,
    "read-ladder": phase_read_ladder,
    "carried": phase_carried,
    "answer-matchedn": phase_answer_matchedn,
    "partials": phase_partials,
    "confirm-b": phase_confirm_b,
    "confirm-b-gpu": phase_confirm_b_gpu,
    "upload-verify": phase_upload_verify,
    "harvest": phase_harvest,
    "figures": phase_figures,
}


def build_argparser() -> argparse.ArgumentParser:
    """CLI (Hydra is not used here: this mirrors the parent phased-driver convention)."""
    ap = argparse.ArgumentParser(description="issue 2163 context-read ladder driver")
    ap.add_argument("--phase", choices=sorted(PHASES), help="phase to run")
    ap.add_argument("--work", default="/root/issue2163_work", help="work root (out-root)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--xty-device", default="auto", choices=("auto", "scipy", "cusparse"))
    ap.add_argument("--smoke", action="store_true", help="tiny deterministic slice")
    ap.add_argument("--smoke-train", type=int, default=600)
    ap.add_argument("--smoke-val", type=int, default=64)
    ap.add_argument("--smoke-holdout", type=int, default=128)
    ap.add_argument("--max-shards", type=int, default=0, help="census shard cap (smoke)")
    ap.add_argument("--local-store", default=None, help="local pooled-shard dir (skips staging)")
    ap.add_argument("--local-inputs", default=None, help="local #1482 inputs dir")
    ap.add_argument("--local-meta", default=None, help="local scratch-meta dir")
    ap.add_argument("--local-dense", default=None, help="local dense-targets dir")
    ap.add_argument("--local-covariates", default=None, help="local covariates dir")
    ap.add_argument("--upload-dense-src", default=None, help="phase U dense source dir")
    ap.add_argument("--upload-cov-src", default=None, help="phase U covariates source dir")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--hf-out-prefix", default=None, help="override the HF output prefix base")
    ap.add_argument("--floor-last", type=int, default=FLOOR_LAST_DEFAULT)
    ap.add_argument("--cap-db", type=int, default=CAP_DB_DEFAULT)
    ap.add_argument("--n-draws-pairing", type=int, default=N_DRAWS_PAIRING)
    ap.add_argument("--n-draws-strat", type=int, default=N_DRAWS_STRAT)
    ap.add_argument("--matchedn-features", type=int, default=MATCHEDN_FEATURES)
    ap.add_argument("--matchedn-draws", type=int, default=MATCHEDN_DRAWS)
    ap.add_argument("--knn-probes", type=int, default=KNN_N_PRED)
    # VM-side phases 8a/8b. Both phase_harvest and phase_figures read this; it was referenced but
    # never defined, so both were dead on arrival (AttributeError on first run).
    ap.add_argument(
        "--harvest-out",
        default=None,
        help="VM harvest destination root (default: <repo>/eval_results/issue_2163)",
    )
    ap.add_argument(
        "--figures-out",
        default=None,
        help="VM figures destination root (default: <repo>/figures/issue_2163)",
    )
    ap.add_argument(
        "--carried-block",
        type=int,
        default=0,
        help="carried output-column block (0 = 8192 prod / 1024 smoke)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="execute deferred imports + signature binds, then exit",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point: dispatch one phase (the wrapper sequences phases + owns the done token)."""
    args = build_argparser().parse_args(argv)
    if args.import_check:
        return run_import_check()
    if not args.phase:
        raise SystemExit("--phase is required (or --import-check)")
    if args.carried_block == 0:
        args.carried_block = 1_024 if args.smoke else 8_192
    _out(args).mkdir(parents=True, exist_ok=True)
    logger.info(
        "[main] phase=%s work=%s smoke=%s device=%s",
        args.phase,
        args.work,
        args.smoke,
        args.device,
    )
    t0 = time.time()
    rc = PHASES[args.phase](args)
    logger.info("[main] phase=%s rc=%s wall=%.0fs", args.phase, rc, time.time() - t0)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
