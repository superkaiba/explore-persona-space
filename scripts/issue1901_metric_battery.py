#!/usr/bin/env python
# ruff: noqa: RUF002, RUF003
"""#1901 mapping-quality metric battery: 1M-context maps vs baselines on every metric.

Estimator-ladder x metric-battery grid over the banked #779 context->answer maps
(weights REUSED from HF `issue779_monitoring/n1m_readout/weights`, applied via
`issue779_ffc_n1m_fits.apply_map`) plus the baseline ladder (constant train-mean,
identity copy, identity + learned bias, scaled/diagonal identity), scored on
pooled/per-dim R2, mean cosine, kNN retrieval (euclid/cosine/CSLS), median
rank/MRR, hubness diagnostics, pool-size sensitivity — in the context arm
(pinned 3600/400/1000 split, seed 42) and the prefix arm (#722 50-context
battery, LOFO). Plan: tasks #1901 plans/plan.md (v4).

Phases (p0-p3 resume-by-sentinel; p4 always re-runs — cheap figures-only pass;
per-phase JSON outputs):
  p0_stage       stage the 12 weight payloads (revision-pinned; mirror-root
                 arithmetic asserted; 1-file probe + consumer-open first;
                 realized-keys check per payload)
  p1_distractors seeded partial stream of capture chunks -> 100k-row
                 distractor pool npz (+ manifest, dedup stats, HF upload)
  p2_context     context-arm battery (ladder x metrics x pools x nulls x CIs)
  p3_prefix      prefix-arm battery (LOFO ladder incl. batched MLP)
  p4_figures     figures + metric_characterization.json (no resume sentinel)
  all            p0 -> p4

Follow-up round `wildchat-target-battery` (plan v7; sentinels keyed on
wc_regime() so parent sentinels stay valid):
  w0_wc_candidates  VM: fresh WildChat region (streamed PAST the parent n1m
                    consumption point, revision-pinned) -> exclusion-set +
                    transposed near-dupe screen -> mini-manifest on HF
  w1_wc_capture     pod GPU: the reused N1G capture rig on the mini-manifest
                    (--smoke runs the CPU-reachable surface only — carve-out)
  w2_wc_battery     VM: estimator-identity control, then the IDENTICAL ladder x
                    metric battery on wc targets + transfer_comparison.json
                    (--with-intrain-companion adds the in-train ladder read)
  w3_wc_figures     VM: transfer figures + metric_characterization
                    wildchat_transfer fields (no resume sentinel)
  wc_all            w0 -> w3

`--smoke`: SAME code paths, reduced knobs (20-chunk stream, L19 only, pools
{1000, 5000, 3000-with-distractors}, n_boot=50, K=20), outputs diverted to
<staging-root>/smoke/ (never the committed eval_results/figures trees), HF
uploads diverted to the issue1901_metrics/smoke_probe/ prefix.

Launch (VM, detached per the >15-min rule; thread caps REQUIRED):
  env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
      NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \\
      uv run python scripts/issue1901_metric_battery.py --phase all \\
      --staging-root /mnt/eps-data/$USER/issue1901_metrics
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps BEFORE numpy/torch (shared-VM rule)

import argparse
import datetime
import hashlib
import json
import logging
import multiprocessing
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue779_common as C  # noqa: E402
import issue779_ffc_baselines as B  # noqa: E402  (_fit_scale / _fit_diag)
import issue779_ffc_n1m_fits as N1M  # noqa: E402  (apply_map)
import issue779_ffc_n1m_generate_capture as N1G  # noqa: E402  (HF_PREFIX, pass_b)
import issue779_ffc_n50k_fits as N50F  # noqa: E402  (_slice_layer)
import issue779_ffc_n50k_generate_capture as N50G  # noqa: E402  (_remote_index)
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402  (_pooled_r2 / _per_context_cosine)

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.orchestrate import hub  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1901")

# ── constants (values copied verbatim from the plan / ground truth) ──────────────

WEIGHTS_PREFIX = "issue779_monitoring/n1m_readout/weights"
CAPTURE_PREFIX = f"{N1G.HF_PREFIX}/final_token_capture"
MANIFEST_PREFIX = f"{N1G.HF_PREFIX}/{N1G.MANIFEST_SUBDIR}"
# Known-good data-repo listing revision (#1776 Repro; plan §10).
KNOWN_GOOD_REVISION = "687eb8b42cd01e1279fd857655e895e284440524"
PASS_B_SIZE_BYTES = 6_021_122_751  # fair_comparison.json data_provenance.pass_b.size_bytes

BANKED_MULTILAYER = (
    PROJECT_ROOT
    / "eval_results/issue_779/n1m-nonlinear-map-behavior-readout/n1m_multilayer_fits.json"
)
BANKED_IDBIAS = PROJECT_ROOT / "eval_results/issue_779/identity_bias_knn/results.json"
BANKED_722 = PROJECT_ROOT / "eval_results/issue_722/identity_bias_knn/results.json"
# residual_skip banked summary (L19 mixed_1m only; NO persisted weights on HF — plan §4
# stated exclusion; concern residual-skip-exclusion-row-missing).
BANKED_N1M = PROJECT_ROOT / "eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json"

FITTERS = ("ridge", "mlp_w8192", "mlp_w32768", "krr_nystrom")
FITTER_KIND = {
    "ridge": "ridge",
    "mlp_w8192": "mlp",
    "mlp_w32768": "mlp",
    "krr_nystrom": "krr_nystrom",
}
# apply_map contract keys per payload kind (issue779_ffc_n1m_fits.py:882-919, read at plan time).
CONTRACT_KEYS = {
    "ridge": {"kind", "xmu", "xsd", "ymu", "W"},
    "mlp": {"kind", "state_dict", "width", "xmu", "xsd", "ymu"},
    "krr_nystrom": {"kind", "landmarks", "inv_sqrt", "W_dual", "ymu", "gamma"},
}
LAYERS_PROD = (19, 14, 26)  # 19 = headline (pre-registered); 14/26 exploratory
KS_CONTEXT = (1, 5, 10)
KS_PREFIX = (1, 3, 5)
K_CSLS = 10  # Conneau et al. 2018 (arXiv 1710.04087)
RIDGE_REPRO_TOL = 1e-6  # kill criterion 1 (plan §7; #1776 precedent 8e-11)
APPLIED_BANKED_FLAG = 1e-3  # reported-delta loud-flag threshold (plan §4 critic-fold a)
RECIPE_VERSION = "i1901-v1"

# ── wildchat-target-battery round constants (plan v7; follow-up wildchat-target-battery) ──
WC_RECIPE_VERSION = "issue1901-wc-heldout-v1"  # plan v7 §4 w0 recipe tag
WC_HF_ROOT = "issue1901_wildchat"  # ROUND ROOT on the HF data repo (module appends subdirs)
WC_STREAM_TAG = "wildchat_heldout_1901"  # _stream_corpus cache tag for the fresh region
WILDCHAT_DATASET = N1G.WILDCHAT_REPO  # allenai/WildChat-1M (consumer's own constant)
WC_N_TARGETS = 1_000  # chance parity with the parent test-1000 headline pool (plan §11)
WC_TARGET_FLOOR = 500  # kill criterion 2: below this, HALT (plan §7)
WC_CANDIDATE_TARGET = 2_500  # fresh-region candidate overprovision (plan §11)
WC_MANIFEST_KEEP = 1_300  # mini-manifest keep (slack for over-length/empty losses)
WC_SMOKE_CANDIDATE_TARGET = 30  # real-corpus structural probe target (plan §4 smoke)
WC_SMOKE_MANIFEST_KEEP = 20
WC_SMOKE_N_TARGETS = 12  # >= K_CSLS + 2 = 12 (csls asserts k < n_pool AND k <= n_query)
WC_SCREEN_PILOT_ROWS = 20_000  # transposed-screen pilot rows (plan §9 w0 basis)
WC_SCREEN_BUDGET_S = 3_600.0  # abort threshold: projected screen wall > 2x this => halt
WC_SCREEN_WORKERS = 8  # screen fan-out width (env override EPM_WC_SCREEN_WORKERS)
WC_ARMS_SMOKE = ("const_mean", "identity_copy", "identity_bias", "ridge")  # plan smoke clause
WC_DRAWS_SEED_OFFSET = 1919  # wc battery Draws seed = cfg.seed + this (recorded)

ARMS_963K = ("const_mean", "identity_copy", "identity_bias") + FITTERS
ARMS_3600 = (
    "const_mean_3600",
    "identity_bias_3600",
    "scaled_identity_3600",
    "diagonal_only_3600",
    "ridge_3600",
)
ARMS_N50 = ("ridge_n50_fixedlam", "identity_bias_n50")
# Arms evaluated at the big distractor pools (L19): the 963k ladder + the
# dissociation star identity_bias_3600 (bounds the P2 GEMM budget to ~plan §9).
BIG_POOL_ARMS = set(ARMS_963K) | {"identity_bias_3600"}

ARM_LABELS = {
    "const_mean": "Constant train-mean (963k)",
    "identity_copy": "Identity copy",
    "identity_bias": "Identity + learned bias (963k)",
    "ridge": "Linear map (ridge, 963k)",
    "mlp_w8192": "Small neural map (w8192)",
    "mlp_w32768": "Wide neural map (w32768)",
    "krr_nystrom": "Kernel map (Nystrom RBF)",
    "const_mean_3600": "Constant train-mean (3600)",
    "identity_bias_3600": "Identity + learned bias (3600)",
    "scaled_identity_3600": "Scaled identity (3600)",
    "diagonal_only_3600": "Per-dim rescale (3600)",
    "ridge_3600": "Linear map (ridge, 3600)",
    "ridge_n50_fixedlam": "Linear map (ridge, n=50, fixed lambda)",
    "identity_bias_n50": "Identity + learned bias (n=50)",
}


# ── config ───────────────────────────────────────────────────────────────────────


@dataclass
class Cfg:
    phase: str
    staging_root: Path
    smoke: bool
    revision: str
    seed: int
    force: bool
    intrain: bool = False  # wc round: optional in-train memorization companion (plan v7 branch c)

    @property
    def layers(self) -> tuple[int, ...]:
        return (19,) if self.smoke else LAYERS_PROD

    @property
    def n_boot(self) -> int:
        return 50 if self.smoke else int(F.BOOT_N)  # 1000 (F.BOOT_N, the banked convention)

    @property
    def k_perm(self) -> int:
        return 20 if self.smoke else 200  # K=200 (#1776 convention)

    @property
    def n_chunks(self) -> int:
        return 20 if self.smoke else 210

    @property
    def n_distractors(self) -> int:
        return 2_000 if self.smoke else 100_000

    @property
    def distractor_pools(self) -> tuple[int, ...]:
        # total pool sizes (test 1000 + distractors); smoke exercises the
        # concat/corpus-label path at a tiny pool.
        return (3_000,) if self.smoke else (20_000, 100_000)

    @property
    def out_root(self) -> Path:
        return (self.staging_root / "smoke") if self.smoke else self.staging_root

    @property
    def eval_dir(self) -> Path:
        if self.smoke:
            return self.out_root / "eval_results"
        return PROJECT_ROOT / "eval_results" / "issue_1901" / "metric_battery"

    @property
    def fig_dir(self) -> Path:
        if self.smoke:
            return self.out_root / "figures"
        return PROJECT_ROOT / "figures" / "issue_1901"

    @property
    def hf_out_prefix(self) -> str:
        base = "issue1901_metrics"
        return f"{base}/smoke_probe/analysis_tensors" if self.smoke else f"{base}/analysis_tensors"

    @property
    def distractor_npz(self) -> Path:
        return self.out_root / "distractors_L19.npz"

    @property
    def p1_sentinel(self) -> Path:
        return self.out_root / "distractors_L19.done.json"

    def regime(self) -> dict:
        return {
            "recipe_version": RECIPE_VERSION,
            "smoke": self.smoke,
            "seed": self.seed,
            "revision": self.revision,
            "layers": list(self.layers),
            "n_boot": self.n_boot,
            "k_perm": self.k_perm,
            "n_chunks": self.n_chunks,
            "n_distractors": self.n_distractors,
            "distractor_pools": list(self.distractor_pools),
            "fitters": list(FITTERS),
        }

    # ── wildchat-target-battery round (plan v7 phases w0-w3) ────────────────────
    # wc knobs live in a SEPARATE wc_regime() so the parent phases' sentinels
    # (written under regime()) stay valid — adding wc keys to regime() would
    # regime-mismatch every completed parent sentinel on the shared root.

    @property
    def wc_hf_root(self) -> str:
        # ROUND ROOT (N1G semantics — the reused capture module appends
        # final_token_capture/ + raw_completions/ itself; gotchas #1776).
        return f"{WC_HF_ROOT}/smoke_probe" if self.smoke else WC_HF_ROOT

    @property
    def wc_dir(self) -> Path:
        return self.out_root / "wildchat"  # generated wc outputs (rebinds under smoke)

    @property
    def wc_manifest_dir(self) -> Path:
        return self.wc_dir / "manifest"

    @property
    def wc_candidate_target(self) -> int:
        return WC_SMOKE_CANDIDATE_TARGET if self.smoke else WC_CANDIDATE_TARGET

    @property
    def wc_manifest_keep(self) -> int:
        return WC_SMOKE_MANIFEST_KEEP if self.smoke else WC_MANIFEST_KEEP

    @property
    def wc_n_targets(self) -> int:
        # smoke 12 (plan said 8; RESIZED UP against the csls_scores floor
        # K_CSLS=10 < n_pool and <= n_query => n >= 11 — smoke-slice-arithmetic duty)
        return WC_SMOKE_N_TARGETS if self.smoke else WC_N_TARGETS

    @property
    def wc_target_floor(self) -> int:
        # production kill criterion 2 floor; smoke: any nonzero proceeds (#1345
        # gate-calibration rule — production-n floors are demoted at smoke n)
        return 1 if self.smoke else WC_TARGET_FLOOR

    @property
    def wc_pool_totals(self) -> tuple[int, ...]:
        # extra wc pools beyond targets-only: {5k, 20k, 100k} with seeded parent
        # distractors (plan v7 §4); smoke = targets-only pool per the plan.
        return () if self.smoke else (5_000, 20_000, 100_000)

    @property
    def wc_w0_sentinel(self) -> Path:
        return self.out_root / "wc_w0_done.json"

    @property
    def wc_exclusion_npz(self) -> Path:
        # deterministic derived content (960k manifest + 5k round1 sha1 digests);
        # shared across smoke/prod legs (identical bytes), so NON-rebinding.
        return self.staging_root / "wc_exclusion_fps.npz"

    def wc_regime(self) -> dict:
        return {
            "recipe_version": WC_RECIPE_VERSION,
            "smoke": self.smoke,
            "seed": self.seed,
            "revision": self.revision,
            "layer": 19,
            "n_boot": self.n_boot,
            "k_perm": self.k_perm,
            "wc_candidate_target": self.wc_candidate_target,
            "wc_manifest_keep": self.wc_manifest_keep,
            "wc_n_targets": self.wc_n_targets,
            "wc_pool_totals": list(self.wc_pool_totals),
            "near_dupe": {"ngram": N1G.NEAR_DUPE_NGRAM, "thresh": N1G.NEAR_DUPE_JACCARD},
        }


# ── small utilities ──────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
    ).stdout.strip()


def _ru_maxrss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # linux: KB -> GB


def _meta(cfg: Cfg, phase: str, t0: float) -> dict:
    return {
        "script": "issue1901_metric_battery",
        "phase": phase,
        "git_commit": _git_commit(),
        "timestamp_utc": _now(),
        "wall_s": round(time.time() - t0, 1),
        "ru_maxrss_gb": round(_ru_maxrss_gb(), 2),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "thread_env": {
            k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "MALLOC_ARENA_MAX")
        },
        "regime": cfg.regime(),
    }


def _atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)}")


def _atomic_npz(path: Path, **arrays) -> None:
    """np.savez APPENDS .npz to non-.npz names — tmp is '<stem>.tmp.npz' (gotchas)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _round_list(a: np.ndarray, sig: int = 6) -> list[float]:
    return [float(f"{x:.{sig}g}") for x in np.asarray(a, dtype=np.float64).ravel()]


def _resume_skip(cfg: Cfg, out_path: Path, phase: str) -> dict | None:
    """Return the existing phase output when it exists with a MATCHING regime.

    A mismatched regime on the same out-root FAILS LOUD (per-leg out-roots
    already separate smoke/prod; a mismatch here means a deliberate knob change
    on the same root -> require --force). --force redoes the phase.
    """
    if cfg.force or not out_path.exists():
        return None
    prior = json.loads(out_path.read_text())
    prior_regime = (prior.get("metadata") or {}).get("regime") or prior.get("regime")
    if prior_regime == cfg.regime():
        logger.info("[%s] output %s exists with matching regime; skipping", phase, out_path)
        return prior
    raise RuntimeError(
        f"[{phase}] {out_path} exists under a DIFFERENT regime "
        f"(stored != current). Re-run with --force to redo, or use a different root.\n"
        f"stored:  {prior_regime}\ncurrent: {cfg.regime()}"
    )


# ═══════════════════════════════ Phase 0 — stage ═══════════════════════════════


def _weights_files(cfg: Cfg) -> list[str]:
    layers = cfg.layers if cfg.smoke else LAYERS_PROD
    return [f"{WEIGHTS_PREFIX}/L{li}/{name}.pt" for li in layers for name in FITTERS]


def staged_path(cfg: Cfg, repo_rel: str) -> Path:
    """Consumed path for a staged repo file. stage_hub_prefix lands files at
    dest_dir/<repo-relative path> (verbatim prefix mirror — gotchas: dest is a
    MIRROR ROOT), so consumed == staging_root / repo_rel by construction."""
    return cfg.staging_root / repo_rel


def _assert_mirror_root_arithmetic(cfg: Cfg) -> None:
    """Plan Phase 0 mirror-root caveat: dest_root/<hub prefix> == consumed path."""
    probe_rel = f"{WEIGHTS_PREFIX}/L19/ridge.pt"
    got = staged_path(cfg, probe_rel)
    want = cfg.staging_root / WEIGHTS_PREFIX / "L19" / "ridge.pt"
    assert got == want, (got, want)
    assert str(got.relative_to(cfg.staging_root)) == probe_rel, got


def _realized_keys_check(path: Path, kind: str) -> list[str]:
    """#1073 duty: the payload's REALIZED key set must cover the apply_map
    contract keys — mmap read, fail loud on any miss (kill criterion 3)."""
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    realized = set(payload.keys())
    missing = CONTRACT_KEYS[kind] - realized
    if missing:
        raise RuntimeError(
            f"realized-keys check FAILED for {path} (kind={kind}): missing {sorted(missing)} "
            f"from realized {sorted(realized)} — do not substitute a refit; halting (plan §7)."
        )
    del payload
    return sorted(realized)


def _ensure_pass_b() -> Path:
    """Ensure the local pass_b bundle exists (worktree symlink to the MAIN
    checkout's verified copy, else HF fetch) and matches the provenance size.
    Extracted verbatim from phase_p0 so w0/w2 share the identical path."""
    pass_b = N1G.PASS_B_LOCAL
    if not pass_b.exists():
        # From a WORKTREE, PROJECT_ROOT-relative data/ is empty — prefer a symlink
        # to the MAIN checkout's verified copy over a duplicate 6 GB download onto
        # the shared boot disk (plan §9 disk row: pass_b = 0 new bytes).
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        ).stdout.strip()
        main_copy = Path(common).parent / "data" / "issue_779" / "pass_b" / pass_b.name
        if main_copy.exists() and main_copy.stat().st_size == PASS_B_SIZE_BYTES:
            pass_b.parent.mkdir(parents=True, exist_ok=True)
            pass_b.symlink_to(main_copy)
            logger.info("[pass_b] symlinked to main-checkout copy %s", main_copy)
        else:
            logger.info("[pass_b] absent at %s; fetching via N1G._load_pass_b_bundle", pass_b)
            N1G._load_pass_b_bundle(pass_b)  # fetches from HF (PASS_B_HF_PATH) on miss
    size = pass_b.stat().st_size
    assert size == PASS_B_SIZE_BYTES, (
        f"pass_b size {size} != provenance-recorded {PASS_B_SIZE_BYTES} (fair_comparison.json)"
    )
    return pass_b


def phase_p0(cfg: Cfg) -> dict:
    t0 = time.time()
    out_path = cfg.out_root / "p0_done.json"
    prior = _resume_skip(cfg, out_path, "p0")
    if prior is not None:
        return prior

    # Preamble asserts (plan Phase 0): data-disk headroom + pass_b presence/size.
    st = os.statvfs(cfg.staging_root.parent if not cfg.staging_root.exists() else cfg.staging_root)
    avail_gb = st.f_bavail * st.f_frsize / 1e9
    need_gb = 5.0 if cfg.smoke else 30.0
    assert avail_gb >= need_gb, f"staging disk headroom {avail_gb:.1f} GB < {need_gb} GB"
    cfg.out_root.mkdir(parents=True, exist_ok=True)

    pass_b = _ensure_pass_b()
    size = pass_b.stat().st_size

    _assert_mirror_root_arithmetic(cfg)

    # (h)(iv) 1-file staging probe + consumer-open BEFORE bulk staging: the SAME
    # production helper (stage_hub_prefix; exact-file prefixes resolve via its
    # file_exists fallback, hub.py list_hf_files_under_path docstring).
    probe_rel = f"{WEIGHTS_PREFIX}/L19/ridge.pt"
    probe_paths = hub.stage_hub_prefix(
        C.HF_DATA_REPO, probe_rel, cfg.staging_root, repo_type="dataset", revision=cfg.revision
    )
    assert probe_paths == [staged_path(cfg, probe_rel)], probe_paths
    ridge_keys = _realized_keys_check(probe_paths[0], "ridge")
    logger.info("[p0] 1-file probe + consumer-open PASS: %s keys=%s", probe_paths[0], ridge_keys)

    # Bulk staging through the SAME helper. Smoke stages the L19 dir only
    # (~2.8 GB — all four fitted-arm classes REAL in smoke); production stages
    # the full weights tree (12 files; already-staged files skip idempotently).
    bulk_prefix = f"{WEIGHTS_PREFIX}/L19" if cfg.smoke else WEIGHTS_PREFIX
    staged = hub.stage_hub_prefix(
        C.HF_DATA_REPO, bulk_prefix, cfg.staging_root, repo_type="dataset", revision=cfg.revision
    )
    logger.info("[p0] bulk staged %d files under %s", len(staged), bulk_prefix)

    checks = {}
    for rel in _weights_files(cfg):
        p = staged_path(cfg, rel)
        assert p.exists(), f"expected staged payload missing: {p}"
        name = p.stem
        checks[rel] = {
            "size_bytes": p.stat().st_size,
            "realized_keys": _realized_keys_check(p, FITTER_KIND[name]),
        }

    out = {
        "revision": cfg.revision,
        "staging_root": str(cfg.staging_root),
        "avail_gb_at_start": round(avail_gb, 1),
        "pass_b": {"path": str(pass_b), "size_bytes": size},
        "mirror_root_assert": "PASS",
        "one_file_probe": {"path": str(probe_paths[0]), "consumer_open": "PASS"},
        "payload_checks": checks,
        "metadata": _meta(cfg, "p0_stage", t0),
    }
    _atomic_json(out_path, out)
    logger.info("[p0] done in %.1fs (ru_maxrss %.1f GB)", time.time() - t0, _ru_maxrss_gb())
    return out


# ═══════════════════════════ Phase 1 — distractor pool ══════════════════════════


def _stream_jsonl_fields(path: Path):
    """Text-mode line iteration (NEVER splitlines — U+2028 JSONL gotcha), yielding
    (i, corpus) ONLY. Manifest rows carry raw real-user prompt text — content
    hygiene: no other field is read, held, or logged."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            yield int(r["i"]), str(r["corpus"])


def _ci_corpus_threshold(cfg: Cfg) -> dict:
    """ci -> corpus via the manifest's lmsys-block-then-wildchat-block ordering.

    Grounding: build_new_pool concatenates `pool = lmsys_pool + wildchat_pool`
    (issue779_ffc_n1m_generate_capture.py:491) and read_manifest_pool asserts
    contiguous global-index order — so ci < n_lmsys ⇔ lmsys. Verified at run
    time on BOTH ends: the first manifest part must be pure lmsys ascending
    from i=0; the last part pure wildchat with indices >= n_lmsys.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    files = hub.list_hf_files_under_path(
        api, C.HF_DATA_REPO, MANIFEST_PREFIX, repo_type="dataset", revision=cfg.revision
    )
    parts = sorted(f for f in files if f.rsplit("/", 1)[-1].startswith("part_"))
    meta_rel = f"{MANIFEST_PREFIX}/meta.json"
    assert meta_rel in files and parts, (len(files), len(parts))
    scratch = cfg.out_root / "manifest_probe"
    meta_p = hub.stage_hub_file(
        C.HF_DATA_REPO, meta_rel, scratch / "meta.json", repo_type="dataset", revision=cfg.revision
    )
    meta = json.loads(meta_p.read_text())
    n_lmsys, n_wildchat, n_new = int(meta["n_lmsys"]), int(meta["n_wildchat"]), int(meta["n_new"])
    assert n_lmsys + n_wildchat == n_new, (n_lmsys, n_wildchat, n_new)

    first_p = hub.stage_hub_file(
        C.HF_DATA_REPO,
        parts[0],
        scratch / "first.jsonl",
        repo_type="dataset",
        revision=cfg.revision,
    )
    last_p = hub.stage_hub_file(
        C.HF_DATA_REPO,
        parts[-1],
        scratch / "last.jsonl",
        repo_type="dataset",
        revision=cfg.revision,
    )
    expect = 0
    for i, corpus in _stream_jsonl_fields(first_p):
        assert i == expect and corpus == "lmsys", (i, expect, corpus)
        expect += 1
    last_rows = list(_stream_jsonl_fields(last_p))
    assert last_rows and all(c == "wildchat" for _i, c in last_rows), "last part not pure wildchat"
    assert all(i >= n_lmsys for i, _c in last_rows), "last-part indices below n_lmsys"
    logger.info(
        "[p1] corpus threshold verified: ci<%d => lmsys (first part %d rows, last part %d rows)",
        n_lmsys,
        expect,
        len(last_rows),
    )
    return {
        "n_lmsys": n_lmsys,
        "n_wildchat": n_wildchat,
        "n_new": n_new,
        "boundary_check": "PASS (first part pure lmsys ascending; last part pure wildchat)",
    }


def _row_hashes(a: np.ndarray) -> np.ndarray:
    """blake2b digest per row (fp32 bytes) for exact-duplicate counting."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    return np.array([hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in a])


def phase_p1(cfg: Cfg) -> dict:
    t0 = time.time()
    manifest_out = cfg.eval_dir / "distractor_manifest.json"
    if not cfg.force and cfg.p1_sentinel.exists() and cfg.distractor_npz.exists():
        prior = json.loads(cfg.p1_sentinel.read_text())
        if (prior.get("metadata") or {}).get("regime") == cfg.regime():
            logger.info("[p1] sentinel matches regime; skipping")
            return prior
        raise RuntimeError(f"[p1] sentinel {cfg.p1_sentinel} regime mismatch; use --force")

    corpus_info = _ci_corpus_threshold(cfg)
    n_lmsys = corpus_info["n_lmsys"]

    # Chunk universe (scoped listing) + seeded subset.
    remote = N50G._remote_index(CAPTURE_PREFIX)  # {basename: {size, sha256}}
    names = sorted(remote)
    assert names, f"no capture chunks under {CAPTURE_PREFIX}"
    universe_sha = hashlib.sha256("\n".join(names).encode()).hexdigest()
    rng = np.random.default_rng(cfg.seed)
    sel = rng.choice(len(names), size=min(cfg.n_chunks, len(names)), replace=False)
    chunk_names = [names[i] for i in sorted(sel)]
    fp = {
        "seed": cfg.seed,
        "layer": 19,
        "universe_sha256": universe_sha,
        "n_chunks": len(chunk_names),
        "revision": cfg.revision,
        "recipe_version": RECIPE_VERSION,
    }

    # Checkpoint/resume (every 25 chunks; keyed on the full fingerprint).
    ckpt_npz = cfg.out_root / "p1_ckpt.npz"
    ckpt_meta = cfg.out_root / "p1_ckpt.json"
    start, vx_parts, ci_parts = 0, [], []
    if ckpt_npz.exists() and ckpt_meta.exists() and not cfg.force:
        m = json.loads(ckpt_meta.read_text())
        if m.get("fingerprint") == fp:
            blob = np.load(ckpt_npz)
            vx_parts, ci_parts = [blob["vx"]], [blob["ci"]]
            start = int(m["cursor"])
            logger.info("[p1] resuming at chunk cursor %d/%d", start, len(chunk_names))
        else:
            logger.info("[p1] checkpoint fingerprint mismatch; restarting stream")

    scratch = cfg.out_root / "chunk_scratch"
    rows_per_chunk: list[int] = []
    dtype_seen: set[str] = set()
    for k in range(start, len(chunk_names)):
        name = chunk_names[k]
        local = hub.stage_hub_file(
            C.HF_DATA_REPO,
            f"{CAPTURE_PREFIX}/{name}",
            scratch / name,
            repo_type="dataset",
            revision=cfg.revision,
        )
        b = F._mmap_load(local)
        # Real-corpus structural probe at full consumed grain (plan §12 A7):
        for fld in ("cx_last", "v_x", "ci", "layers"):
            assert fld in b, f"chunk {name} missing field {fld}"
        assert list(b["layers"]) == list(N1G.CAPTURE_LAYERS), (name, b["layers"])
        n_r = int(b["v_x"].shape[0])
        assert b["v_x"].shape == (n_r, len(N1G.CAPTURE_LAYERS), C.EXPECTED_HIDDEN), b["v_x"].shape
        assert b["cx_last"].shape == b["v_x"].shape, (name, b["cx_last"].shape)
        dtype_seen.add(str(b["v_x"].dtype))
        vx = N50F._slice_layer(b, "v_x", 19)  # (n_r, H) fp32
        ci = np.array([int(x) for x in b["ci"]], dtype=np.int64)
        assert (ci >= 0).all(), f"chunk {name}: ci must be >= 0 (pass_b rows carry ci=-1)"
        assert ci.shape[0] == n_r, (ci.shape, n_r)
        rows_per_chunk.append(n_r)
        vx_parts.append(vx)
        ci_parts.append(ci)
        del b
        local.unlink()  # stream-reduce: peak ~one chunk
        if (k + 1) % 25 == 0 or (k + 1) == len(chunk_names):
            _atomic_npz(ckpt_npz, vx=np.concatenate(vx_parts), ci=np.concatenate(ci_parts))
            _atomic_json(ckpt_meta, {"fingerprint": fp, "cursor": k + 1})
            vx_parts = [np.load(ckpt_npz)["vx"]]
            ci_parts = [np.load(ckpt_npz)["ci"]]
        logger.info(
            "[p1] chunk %d/%d %s rows=%d elapsed=%.0fs",
            k + 1,
            len(chunk_names),
            name,
            n_r,
            time.time() - t0,
        )

    vx = np.concatenate(vx_parts).astype(np.float32)
    ci = np.concatenate(ci_parts)
    n_streamed = vx.shape[0]
    if n_streamed > cfg.n_distractors:
        keep = np.sort(
            np.random.default_rng(cfg.seed + 1).choice(
                n_streamed, size=cfg.n_distractors, replace=False
            )
        )
        vx, ci = vx[keep], ci[keep]
    elif not cfg.smoke:
        raise RuntimeError(
            f"[p1] streamed {n_streamed} rows < target {cfg.n_distractors}; raise the "
            f"Cfg.n_chunks property (fixed at {cfg.n_chunks} of {len(names)} available "
            "chunks; there is no CLI flag for it — it is a pinned regime key)"
        )
    corpus = np.where(ci < n_lmsys, "lmsys", "wildchat")

    # Duplicate diagnostics (measured + reported; ties handled by mid-ranks downstream).
    pool_h = _row_hashes(vx)
    uniq, counts = np.unique(pool_h, return_counts=True)
    n_dup_rows = int((counts > 1).sum()), int(counts[counts > 1].sum() - (counts > 1).sum())
    dup_stats = {
        "n_rows": int(vx.shape[0]),
        "n_unique_vectors": int(len(uniq)),
        "n_duplicate_groups": n_dup_rows[0],
        "n_excess_duplicate_rows": n_dup_rows[1],
    }
    _atomic_npz(cfg.distractor_npz, vx=vx, ci=ci, corpus=corpus.astype("U8"))
    logger.info(
        "[p1] persisted %s (%d rows, dup groups=%d)",
        cfg.distractor_npz,
        vx.shape[0],
        dup_stats["n_duplicate_groups"],
    )

    # Upload BEFORE the downstream battery consumes it (plan §9 phase-order
    # persistence; upload-policy expensive-store-before-long-fit). Smoke uploads
    # to the smoke_probe prefix — the fenced branch executes for real (#1769).
    dest = f"{cfg.hf_out_prefix}/distractors_L19.npz"
    url = ""
    for attempt in range(2):
        # UPLOAD_LOOP_EXEMPT: bounded 2-attempt outer retry around ONE npz file (#1315 recipe)
        url = hub._upload(
            cfg.distractor_npz,
            C.HF_DATA_REPO,
            "dataset",
            dest,
            upload_as_file=True,
            raise_on_error=(attempt == 1),
        )
        if url:
            break
        logger.warning("[p1] upload returned no path (attempt %d); bounded outer retry", attempt)
        time.sleep(30)
    if not url:
        raise RuntimeError(f"[p1] upload returned no path for {cfg.distractor_npz} -> {dest}")
    from huggingface_hub import HfApi

    missing = hub.verify_repo_paths_uploaded(
        HfApi(), C.HF_DATA_REPO, [dest], path_in_repo=cfg.hf_out_prefix, repo_type="dataset"
    )
    assert not missing, f"[p1] uploaded npz not visible on Hub: {missing}"

    manifest = {
        "chunk_names": chunk_names,
        "chunk_universe": {"n_total": len(names), "sha256": universe_sha},
        "rows_per_chunk": {
            "n_streamed": n_streamed,
            "min": int(np.min(rows_per_chunk)) if rows_per_chunk else None,
            "median": float(np.median(rows_per_chunk)) if rows_per_chunk else None,
            "max": int(np.max(rows_per_chunk)) if rows_per_chunk else None,
            "dtype_seen": sorted(dtype_seen),
        },
        "ci_range": [int(ci.min()), int(ci.max())],
        "corpus_counts": {k: int((corpus == k).sum()) for k in ("lmsys", "wildchat")},
        "corpus_threshold": corpus_info,
        "dup_stats": dup_stats,
        "hf_upload": {"dest": dest, "url": url},
        "metadata": _meta(cfg, "p1_distractors", t0),
    }
    _atomic_json(manifest_out, manifest)
    _atomic_json(cfg.p1_sentinel, manifest)
    logger.info("[p1] done in %.1fs (ru_maxrss %.1f GB)", time.time() - t0, _ru_maxrss_gb())
    return manifest


# ══════════════════════════ shared battery machinery ═══════════════════════════


def csls_scores(S: np.ndarray, k: int = K_CSLS) -> np.ndarray:
    """Cross-domain CSLS (Conneau et al. 2018, arXiv 1710.04087).

    score[i, j] = 2*S[i, j] - r_query[i] - r_pool[j], where S is the
    (n_query, n_pool) CROSS-domain cosine-similarity matrix, r_query[i] is the
    mean of the k largest S[i, :] (row-wise) and r_pool[j] the mean of the k
    largest S[:, j] (column-wise). PINNED to the cross-domain formulation:
    both neighborhoods come from THIS matrix — NEVER a pool-internal
    (n_pool, n_pool) neighborhood (plan §4 PIN; 40 GB fp32 at the 100k pool).
    """
    n_q, n_p = S.shape
    assert 0 < k < n_p, (k, S.shape)
    assert k <= n_q, f"CSLS column neighborhood needs k <= n_query ({k} > {n_q})"
    top_q = np.partition(S, n_p - k, axis=1)[:, n_p - k :]
    r_q = top_q.mean(axis=1)
    top_p = np.partition(S, n_q - k, axis=0)[n_q - k :, :]
    r_p = top_p.mean(axis=0)
    return 2.0 * S - r_q[:, None] - r_p[None, :]


def rank_matrix_for_cols(d: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Mid-ranks (the knn_retrieval formula: 1 + #closer + 0.5*#tied-others,
    tolerance-based ties) of pool column cols[j] within each row i of the
    distance matrix d — R[i, j]. R[i, i-th true col] reproduces the helper's
    observed ranks bitwise (parity-asserted per cell at pools <= 5000);
    permutation-null draws are pure re-index reads of R (zero extra GEMMs).
    Sort + searchsorted per row: an n<=1000-iteration prep loop, NOT a
    per-draw loop (vectorize-many-cell-fits compliance)."""
    n = d.shape[0]
    cols = np.asarray(cols)
    R = np.empty((n, cols.shape[0]), dtype=np.float64)
    for i in range(n):
        row = np.sort(d[i])
        v = d[i, cols]
        tol = 1e-9 * np.maximum(np.abs(v), 1e-12)
        lo = np.searchsorted(row, v - tol, side="left")
        hi = np.searchsorted(row, v + tol, side="right")
        R[i] = 1.0 + lo + 0.5 * (hi - lo - 1)
    return R


def _ranks_summary(ranks: np.ndarray, ks: tuple[int, ...], n_pool: int) -> dict:
    return {
        "acc_at_k": {int(k): float((ranks <= k).mean()) for k in ks},
        "chance_at_k": {int(k): float(k / n_pool) for k in ks},
        "median_rank": float(np.median(ranks)),
        "mrr": float((1.0 / ranks).mean()),
        "n": int(ranks.shape[0]),
        "n_pool": int(n_pool),
    }


@dataclass
class Draws:
    """Shared draw indices (plan critic-fold c: shared across estimators so
    paired-difference CIs are pure re-reductions of the persisted matrices)."""

    boot_idx: np.ndarray  # (n_boot, n) int
    perms: np.ndarray  # (k_perm, n) int
    cnt: np.ndarray  # (n_boot, n) f64 multiplicity matrix

    @classmethod
    def make(cls, n: int, n_boot: int, k_perm: int, seed: int) -> "Draws":
        rng = np.random.default_rng(seed)
        boot_idx = rng.integers(0, n, size=(n_boot, n))
        perms = np.stack([rng.permutation(n) for _ in range(k_perm)])
        cnt = np.zeros((n_boot, n), dtype=np.float64)
        np.add.at(cnt, (np.repeat(np.arange(n_boot), n), boot_idx.ravel()), 1.0)
        return cls(boot_idx, perms, cnt)


@dataclass
class ReconContext:
    """Per-(layer) target-side precomputation shared across all arms."""

    Yte: np.ndarray  # (n, d) f64
    s_y: np.ndarray  # (n,) row sq-norms
    ss_tot: float
    ss_tot_dim: np.ndarray  # (d,)
    ss_tot_draws: np.ndarray  # (n_boot,) bootstrap SS_tot per draw (shared)
    y_norm: np.ndarray  # (n,) row norms

    @classmethod
    def make(cls, Yte: np.ndarray, draws: Draws) -> "ReconContext":
        Yte = np.asarray(Yte, dtype=np.float64)
        n = Yte.shape[0]
        mu = Yte.mean(0)
        s_y = (Yte**2).sum(1)
        ss_tot_dim = ((Yte - mu) ** 2).sum(0)
        ss_tot = float(ss_tot_dim.sum())
        sum_y = draws.cnt @ Yte  # (n_boot, d) — ONE shared GEMM per layer
        sumsq = draws.cnt @ s_y
        ss_tot_draws = sumsq - (sum_y**2).sum(1) / n
        return cls(Yte, s_y, ss_tot, ss_tot_dim, ss_tot_draws, np.linalg.norm(Yte, axis=1))


def eval_recon_cell(pred: np.ndarray, rc: ReconContext, draws: Draws) -> tuple[dict, dict]:
    """Pooled R2 (banked whole_map_r2 formula, via PR._pooled_r2 parity), mean
    cosine, per-dim R2 summary, batched bootstrap draws + shuffled-pair nulls.
    Returns (summary, draw_arrays)."""
    pred = np.asarray(pred, dtype=np.float64)
    Yte, n = rc.Yte, rc.Yte.shape[0]
    res_i = ((Yte - pred) ** 2).sum(1)
    cos_i = PR._per_context_cosine(pred, Yte)
    assert np.isfinite(cos_i).all(), "non-finite per-row cosine"
    r2_point = 1.0 - res_i.sum() / rc.ss_tot
    helper_r2 = PR._pooled_r2(pred, Yte)
    assert abs(r2_point - helper_r2) < 1e-10, (r2_point, helper_r2)
    cos_point = float(cos_i.mean())

    # batched bootstrap (one gather + matvecs on the shared count matrix)
    ss_res_d = draws.cnt @ res_i
    r2_d = 1.0 - ss_res_d / rc.ss_tot_draws
    cos_d = (draws.cnt @ cos_i) / n

    # batched shuffled-pair nulls (re-indexed cross GEMM; SS_tot invariant
    # under permutation — same target set, same mean)
    s_p = (pred**2).sum(1)
    cross = pred @ Yte.T
    ar = np.arange(n)[None, :]
    g = cross[ar, draws.perms]  # (K, n)
    ss_res_null = rc.s_y[draws.perms].sum(1) + s_p.sum() - 2.0 * g.sum(1)
    r2_null = 1.0 - ss_res_null / rc.ss_tot
    p_norm = np.linalg.norm(pred, axis=1)
    crossn = cross / ((p_norm[:, None] + 1e-12) * (rc.y_norm[None, :] + 1e-12))
    cos_null = crossn[ar, draws.perms].mean(1)

    # per-dim R2 summary (plan: median, frac>0, variance-weighting decomposition)
    ss_res_dim = ((Yte - pred) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_dim = 1.0 - ss_res_dim / rc.ss_tot_dim
    order = np.argsort(rc.ss_tot_dim)[::-1]  # by target variance, desc
    d_dim = r2_dim.shape[0]
    top10 = order[: max(1, d_dim // 10)]
    bot50 = order[d_dim // 2 :]
    hist, _edges = np.histogram(np.clip(r2_dim, -2, 1), bins=48, range=(-2.0, 1.0))
    perdim = {
        "median": float(np.median(r2_dim)),
        "frac_positive": float((r2_dim > 0).mean()),
        "mean_unweighted": float(r2_dim.mean()),
        "r2_pooled_top10pct_var_dims": float(
            1.0 - ss_res_dim[top10].sum() / rc.ss_tot_dim[top10].sum()
        ),
        "r2_pooled_bottom50pct_var_dims": float(
            1.0 - ss_res_dim[bot50].sum() / rc.ss_tot_dim[bot50].sum()
        ),
        "deciles": _round_list(np.quantile(r2_dim, np.linspace(0.1, 0.9, 9))),
        "hist_counts_minus2_to_1_48bins": [int(x) for x in hist],
    }

    summary = {
        "r2": {
            "point": float(r2_point),
            "lo": float(np.quantile(r2_d, 0.025)),
            "hi": float(np.quantile(r2_d, 0.975)),
        },
        "mean_cosine": {
            "point": cos_point,
            "lo": float(np.quantile(cos_d, 0.025)),
            "hi": float(np.quantile(cos_d, 0.975)),
        },
        "perdim_r2": perdim,
        "null": {
            "r2": {"mean": float(r2_null.mean()), "p975": float(np.quantile(r2_null, 0.975))},
            "mean_cosine": {
                "mean": float(cos_null.mean()),
                "p975": float(np.quantile(cos_null, 0.975)),
            },
        },
    }
    draw_arrays = {
        "r2_boot": r2_d,
        "cos_boot": cos_d,
        "r2_null": r2_null,
        "cos_null": cos_null,
    }
    return summary, draw_arrays


@dataclass
class PoolSpec:
    """One retrieval candidate pool: f64 pool + normalized copy + provenance."""

    name: str
    pool64: np.ndarray  # (n_pool, d) f64
    pool_n: np.ndarray  # row-normalized copy (helper cosine convention)
    q2: np.ndarray  # (n_pool,) sq-norms (helper euclid convention)
    true_idx: np.ndarray  # (n_test,) pool row of each test row's true target
    labels: np.ndarray  # (n_pool,) corpus/provenance labels
    composition: dict

    @classmethod
    def make(
        cls, name: str, pool_f32: np.ndarray, true_idx: np.ndarray, labels: np.ndarray
    ) -> "PoolSpec":
        pool64 = np.asarray(pool_f32, dtype=np.float64)
        q2 = (pool64**2).sum(1)
        pool_n = pool64 / (np.linalg.norm(pool64, axis=1, keepdims=True) + 1e-12)
        h = _row_hashes(pool_f32)
        uniq, counts = np.unique(h, return_counts=True)
        comp = {
            "n_pool": int(pool64.shape[0]),
            "labels": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
            "n_unique_vectors": int(len(uniq)),
            "n_excess_duplicate_rows": int(counts.sum() - len(uniq)),
        }
        return cls(name, pool64, pool_n, q2, np.asarray(true_idx), labels, comp)


def _dist_euclid(pred64: np.ndarray, spec: PoolSpec) -> np.ndarray:
    """Squared euclid via GEMM — byte-identical formula to mapping_baselines
    _pairwise_dist('euclidean') with the pool reductions precomputed."""
    p2 = (pred64**2).sum(1)[:, None]
    return p2 + spec.q2[None, :] - 2.0 * (pred64 @ spec.pool64.T)


def _sim_cosine(pred64: np.ndarray, spec: PoolSpec) -> np.ndarray:
    pn = pred64 / (np.linalg.norm(pred64, axis=1, keepdims=True) + 1e-12)
    return pn @ spec.pool_n.T


def eval_retrieval_cell(
    pred: np.ndarray,
    spec: PoolSpec,
    ks: tuple[int, ...],
    draws: Draws,
    *,
    helper_parity: bool,
    hub_diag: bool = False,
) -> tuple[dict, dict]:
    """Retrieval battery for one (arm, pool): euclid + cosine + CSLS ranks,
    acc@k / median rank / MRR, batched bootstrap + shuffled-pair nulls, hubness
    diagnostic. helper_parity: additionally run the canonical knn_retrieval and
    assert bitwise-equal summaries (pools <= 5000 — plan §9 GEMM budget)."""
    pred64 = np.asarray(pred, dtype=np.float64)
    n, n_pool = pred64.shape[0], spec.pool64.shape[0]
    ar = np.arange(n)
    out: dict = {}
    draw_arrays: dict = {}
    S_cos = None
    for metric in ("euclidean", "cosine", "csls"):
        if metric == "euclidean":
            d = _dist_euclid(pred64, spec)
        elif metric == "cosine":
            S_cos = _sim_cosine(pred64, spec)
            d = 1.0 - S_cos
        else:
            assert S_cos is not None
            d = -csls_scores(S_cos, K_CSLS)
        R = rank_matrix_for_cols(d, spec.true_idx)
        obs_ranks = R[ar, ar]
        summary = _ranks_summary(obs_ranks, ks, n_pool)
        if helper_parity and metric in ("euclidean", "cosine"):
            ref = knn_retrieval(
                pred64,
                spec.pool64 if spec.name == "test" else pred64,  # `true` unused when pool given
                ks=ks,
                metric=metric,
                pool=spec.pool64,
                true_pool_idx=spec.true_idx,
            )
            for k in ks:
                assert np.isclose(summary["acc_at_k"][k], ref["acc_at_k"][k], atol=1e-12)
            assert np.isclose(summary["median_rank"], ref["median_rank"], atol=1e-9)
            assert np.isclose(summary["mrr"], ref["mrr"], atol=1e-12)
            summary["helper_parity"] = "PASS"
        # shuffled-pair null: re-index the SAME rank matrix (zero extra GEMMs)
        null_ranks = R[ar[None, :], draws.perms]  # (K, n)
        null_acc1 = (null_ranks <= 1).mean(1)
        summary["null"] = {
            "acc1_mean": float(null_acc1.mean()),
            "acc1_p975": float(np.quantile(null_acc1, 0.975)),
            "mrr_mean": float((1.0 / null_ranks).mean()),
            "median_rank_mean": float(np.median(null_ranks, axis=1).mean()),
        }
        # bootstrap over test rows: gather of observed ranks
        boot_ranks = obs_ranks[draws.boot_idx]
        acc1_d = (boot_ranks <= 1).mean(1)
        mrr_d = (1.0 / boot_ranks).mean(1)
        summary["acc1_ci"] = {
            "lo": float(np.quantile(acc1_d, 0.025)),
            "hi": float(np.quantile(acc1_d, 0.975)),
        }
        summary["mrr_ci"] = {
            "lo": float(np.quantile(mrr_d, 0.025)),
            "hi": float(np.quantile(mrr_d, 0.975)),
        }
        if hub_diag and metric in ("euclidean", "cosine"):
            summary["hubness"] = _hubness(d, spec)
        out[metric] = summary
        draw_arrays[metric] = {
            "acc1_boot": acc1_d,
            "mrr_boot": mrr_d,
            "acc1_null": null_acc1,
            # per-row observed mid-ranks (<= n_pool; half-integers on ties — 6-sig-digit
            # rounding in _round_list is lossless below ~1e6 for CDF purposes): persisted
            # so the plan §6 rank-CDF figure is re-renderable from the JSONs alone.
            "obs_ranks": obs_ranks,
        }
        del d, R
    return out, draw_arrays


def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = x.mean()
    s = x.std()
    return float(((x - m) ** 3).mean() / (s**3 + 1e-30))


def _hubness(d: np.ndarray, spec: PoolSpec) -> dict:
    """N_10 k-occurrence diagnostic + hub corpus composition (plan critic-fold e)."""
    n_pool = d.shape[1]
    k = min(10, n_pool - 1)
    top = np.argpartition(d, k, axis=1)[:, :k]
    counts = np.bincount(top.ravel(), minlength=n_pool)
    hub_cut = max(1, int(np.ceil(n_pool * 0.001)))
    hub_items = np.argsort(counts)[::-1][:hub_cut]
    hub_labels = spec.labels[hub_items]
    comp = {str(kk): int(v) for kk, v in zip(*np.unique(hub_labels, return_counts=True))}
    return {
        "n10_skewness": _skew(counts),
        "n10_max": int(counts.max()),
        "n10_frac_zero": float((counts == 0).mean()),
        "top_hub_corpus_composition": comp,
        "pool_corpus_composition": spec.composition["labels"],
        "n_top_hubs": int(hub_cut),
    }


def gaussian_hubness_reference(n: int, n_pool: int, d_dim: int, seed: int) -> dict:
    """Matched-shape Gaussian-pool N_10 skewness reference (H4 comparator)."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, d_dim), dtype=np.float32)
    p = rng.standard_normal((n_pool, d_dim), dtype=np.float32)
    out = {}
    q2 = (p.astype(np.float64) ** 2).sum(1)
    p64 = p.astype(np.float64)
    qn64 = q.astype(np.float64)
    d_e = (qn64**2).sum(1)[:, None] + q2[None, :] - 2.0 * (qn64 @ p64.T)
    pn = p64 / (np.linalg.norm(p64, axis=1, keepdims=True) + 1e-12)
    qn = qn64 / (np.linalg.norm(qn64, axis=1, keepdims=True) + 1e-12)
    d_c = 1.0 - qn @ pn.T
    for metric, dm in (("euclidean", d_e), ("cosine", d_c)):
        k = min(10, n_pool - 1)
        top = np.argpartition(dm, k, axis=1)[:, :k]
        counts = np.bincount(top.ravel(), minlength=n_pool)
        out[metric] = {"n10_skewness": _skew(counts)}
    return out


def _check_null_collapse(arm: str, summary: dict, dup_note: str) -> None:
    """Kill criterion 2 (plan §7, FITTED maps only): a shuffled-pair null that
    does not collapse to floor means the metric implementation reads structure
    that is not there. Cosine EXCLUDED (its high anisotropy floor is the H3
    FINDING). Retrieval weighed against duplicate counts first (message)."""
    r2_null_p975 = summary["null"]["r2"]["p975"]
    if r2_null_p975 > 0.05:
        raise RuntimeError(
            f"KILL (H3 null non-collapse): {arm} shuffled-pair R2 null p97.5="
            f"{r2_null_p975:.4f} > 0.05 — halting before any science claim (plan §7)."
        )


def _check_retrieval_null_collapse(arm: str, metric: str, pool: str, s: dict, dup_note: str):
    chance = s["chance_at_k"][1]
    null_mean = s["null"]["acc1_mean"]
    if null_mean > 5.0 * chance + 0.005:
        raise RuntimeError(
            f"KILL (H3 retrieval null non-collapse): {arm}/{metric}/pool={pool} null acc@1 "
            f"mean={null_mean:.5f} vs chance={chance:.5f} — weigh against duplicate counts "
            f"first ({dup_note}); halting (plan §7)."
        )


# ═══════════════════════════ Phase 2 — context arm ═════════════════════════════


def _load_payload(cfg: Cfg, li: int, name: str) -> dict:
    return torch.load(
        staged_path(cfg, f"{WEIGHTS_PREFIX}/L{li}/{name}.pt"),
        map_location="cpu",
        weights_only=False,
    )


def _assert_reproduction(applied_r2: float, banked_r2: float) -> None:
    """Kill criterion 1 (plan §7): applied L19 ridge pooled R2 must match the
    banked 0.7541708417500046 within 1e-6 (#1776 precedent: 8e-11)."""
    diff = abs(applied_r2 - banked_r2)
    if diff > RIDGE_REPRO_TOL:
        raise RuntimeError(
            f"KILL (reproduction): applied L19 ridge R2 {applied_r2!r} deviates from banked "
            f"{banked_r2!r} by {diff:.3e} > {RIDGE_REPRO_TOL} — staged-weights reuse premise "
            "broken; halting, file the defect (plan §7)."
        )


def _fixed_lambda_ridge(
    xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, lam: float
) -> np.ndarray:
    """Gram/dual ridge at a FIXED lambda (F's shared-factorization helpers) —
    the H5 n=50 rung (no val set exists at n=50; lambda inherited from the
    3600-arm val selection, recorded)."""
    dev = torch.device("cpu")
    fact = F._factorize(xtr, dev)
    VtY, ymu = F._vty_ymu(fact, ytr)
    return F._apply(fact, float(lam), VtY, ymu, F._cross_kernel(fact, xte))


def _small_n_preds(
    X: np.ndarray,
    Y: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    Xte64: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], float]:
    """3600-arm + n50 companion rungs (parent P2 block extracted verbatim so the
    wc round refits on the IDENTICAL pass_b train/val inputs — plan v7 §4 w2:
    'fit inputs unchanged, only the evaluation targets swap'). Returns
    (preds, lam_3600). RNG usage identical to the original inline block."""
    xtr, xva = X[train].astype(np.float64), X[val].astype(np.float64)
    ytr, yva = Y[train].astype(np.float64), Y[val].astype(np.float64)
    preds: dict[str, np.ndarray] = {}
    preds["const_mean_3600"] = np.tile(ytr.mean(0), (Xte64.shape[0], 1))
    preds["identity_bias_3600"] = identity_bias_predict(xtr, ytr, Xte64)
    preds["scaled_identity_3600"] = B._fit_scale(xtr, ytr) * Xte64
    preds["diagonal_only_3600"] = Xte64 * B._fit_diag(xtr, ytr)
    (ridge_3600,), lam_3600 = F.gram_fit_apply(
        xtr, ytr, [Xte64], torch.device("cpu"), val=(xva, yva)
    )
    preds["ridge_3600"] = np.asarray(ridge_3600)
    # H5 attribution rung: n_train=50 context subsample (plan §4 Phase 3)
    rng50 = np.random.default_rng(seed)
    sub50 = np.sort(rng50.choice(train, size=50, replace=False))
    x50, y50 = X[sub50].astype(np.float64), Y[sub50].astype(np.float64)
    preds["ridge_n50_fixedlam"] = _fixed_lambda_ridge(x50, y50, Xte64, lam_3600)
    preds["identity_bias_n50"] = identity_bias_predict(x50, y50, Xte64)
    return preds, float(lam_3600)


def _pilot_battery_timing(cfg: Cfg, d_dim: int) -> dict:
    """Production-shape pilot (plan §9 P2 basis 'pilot-gated'): time ONE
    estimator x full retrieval battery at the 100k-pool production shape on a
    synthetic pool (compute-only; no download), extrapolate to the arm count."""
    rng = np.random.default_rng(0)
    n = 1000
    n_pool = 100_000
    pred = rng.standard_normal((n, d_dim)).astype(np.float64)
    pool = rng.standard_normal((n_pool, d_dim), dtype=np.float32)
    spec = PoolSpec.make("pilot", pool, np.arange(n), np.array(["synthetic"] * n_pool))
    draws = Draws.make(n, 50, 20, 0)
    t0 = time.time()
    eval_retrieval_cell(pred, spec, KS_CONTEXT, draws, helper_parity=False)
    per_cell_s = time.time() - t0
    n_cells = len(BIG_POOL_ARMS) * len(cfg.distractor_pools)
    projected_h = per_cell_s * n_cells / 3600.0
    logger.info(
        "[pilot] 1 arm x full battery @ pool=100k: %.1fs -> projected %.2fh over %d big-pool cells",
        per_cell_s,
        projected_h,
        n_cells,
    )
    return {
        "per_cell_s_at_100k": round(per_cell_s, 1),
        "n_big_pool_cells": n_cells,
        "projected_wall_h_big_pools": round(projected_h, 3),
        "planned_wall_h_p2": 0.5,
    }


def _build_pools_for_layer(
    cfg: Cfg, li: int, Yte32: np.ndarray, Y_all32: np.ndarray, test: np.ndarray
) -> list[PoolSpec]:
    """Pool grid (plan §4): test-1000 headline everywhere; pass_b-5000 +
    distractor pools at the headline layer only."""
    n = Yte32.shape[0]
    pools = [PoolSpec.make("test", Yte32, np.arange(n), np.array(["lmsys(test)"] * n))]
    if li != 19:
        return pools
    labels_5k = np.array(["lmsys(pass_b)"] * Y_all32.shape[0], dtype=object)
    labels_5k[test] = "lmsys(test)"
    pools.append(PoolSpec.make("passb_5000", Y_all32, test, labels_5k))
    if cfg.distractor_pools:
        blob = np.load(cfg.distractor_npz)
        dvx, dcorpus = blob["vx"], blob["corpus"]
        for total in cfg.distractor_pools:
            n_d = total - n
            assert dvx.shape[0] >= n_d, (
                f"distractor pool has {dvx.shape[0]} rows < {n_d} needed for pool {total}"
            )
            pool32 = np.concatenate([Yte32, dvx[:n_d]])
            labels = np.concatenate(
                [np.array(["lmsys(test)"] * n, dtype=object), dcorpus[:n_d].astype(object)]
            )
            pools.append(PoolSpec.make(f"distr_{total}", pool32, np.arange(n), labels))
    return pools


def phase_p2(cfg: Cfg) -> dict:
    t0 = time.time()
    out_path = cfg.eval_dir / "context_arm.json"
    draws_path = cfg.eval_dir / "boot_draws_context.json"
    prior = _resume_skip(cfg, out_path, "p2")
    if prior is not None and set(str(li) for li in cfg.layers) <= set(prior.get("per_layer", {})):
        return prior

    banked = json.loads(BANKED_MULTILAYER.read_text())
    banked_idbias = json.loads(BANKED_IDBIAS.read_text())
    bundle = F.load_pass_b()
    n_ctx = int(bundle["cx_last"].shape[0])
    assert n_ctx == F.N_PASS_B == 5000, n_ctx
    train, val, test = F.fixed_split(n_ctx, n_ctx - 400 - 1000, 400, 1000, F.SPLIT_SEED)
    # split-pin assert against the banked (git-resident) shas — byte-identical split
    assert F._sha_ids(val) == banked["split"]["val_sha256"], "val sha != banked pinned sha"
    assert F._sha_ids(test) == banked["split"]["test_sha256"], "test sha != banked pinned sha"

    dup_note = "n/a"
    if (cfg.eval_dir / "distractor_manifest.json").exists():
        man = json.loads((cfg.eval_dir / "distractor_manifest.json").read_text())
        dup_note = f"distractor dup_stats={man['dup_stats']}"

    result = prior if prior is not None else {"per_layer": {}}
    result.setdefault("per_layer", {})
    all_draws: dict = (
        json.loads(draws_path.read_text()) if draws_path.exists() and prior is not None else {}
    )
    pilot = _pilot_battery_timing(cfg, C.EXPECTED_HIDDEN) if cfg.smoke else None

    for li in cfg.layers:
        if str(li) in result["per_layer"] and not cfg.force:
            logger.info("[p2] L%d already done; skipping", li)
            continue
        t_layer = time.time()
        X = F.input_layer(bundle, "last", li)
        Y = F.target_vx(bundle, li)
        Xte = X[test].astype(np.float64)
        Yte32 = Y[test]
        rng_seed = cfg.seed + li
        draws = Draws.make(len(test), cfg.n_boot, cfg.k_perm, rng_seed)
        rc = ReconContext.make(Yte32, draws)

        # ── build the arm ladder ────────────────────────────────────────────
        preds: dict[str, np.ndarray] = {}
        pay_r = _load_payload(cfg, li, "ridge")
        xmu_full = pay_r["xmu"].to(torch.float64).numpy()
        ymu_full = pay_r["ymu"].to(torch.float64).numpy()
        del pay_r
        preds["const_mean"] = np.broadcast_to(ymu_full, Xte.shape)
        preds["identity_copy"] = Xte
        preds["identity_bias"] = Xte + (ymu_full - xmu_full)
        applied_vs_banked: dict = {}
        banked_preds = banked["per_layer"][str(li)]["per_point"]["mixed_1m"]["predictors"]
        for name in FITTERS:
            payload = _load_payload(cfg, li, name)
            preds[name] = N1M.apply_map(payload, X[test], torch.device("cpu"))
            del payload
            applied_r2 = PR._pooled_r2(preds[name], rc.Yte)
            banked_r2 = float(banked_preds[name]["whole_map_r2"])
            delta = applied_r2 - banked_r2
            applied_vs_banked[name] = {
                "applied_r2": float(applied_r2),
                "banked_r2": banked_r2,
                "delta": float(delta),
                "flag_over_1e3": bool(abs(delta) > APPLIED_BANKED_FLAG),
            }
            if abs(delta) > APPLIED_BANKED_FLAG:
                logger.warning(
                    "[p2] L%d %s applied-vs-banked |delta|=%.2e > 1e-3 (reported, not a halt "
                    "— the HALT assert is ridge-only, plan §4a)",
                    li,
                    name,
                    abs(delta),
                )
            logger.info(
                "[p2] L%d %s applied R2=%.6f (banked %.6f)", li, name, applied_r2, banked_r2
            )
        if li == 19:
            _assert_reproduction(
                applied_vs_banked["ridge"]["applied_r2"], applied_vs_banked["ridge"]["banked_r2"]
            )
            logger.info("[p2] L19 ridge reproduction assert PASS (tol %.0e)", RIDGE_REPRO_TOL)

        small_n_meta: dict = {}
        if li == 19:
            sn_preds, lam_3600 = _small_n_preds(X, Y, train, val, Xte, cfg.seed)
            preds.update(sn_preds)
            banked_lam = None
            banked_last = banked_idbias["inputs"].get("last", {})
            if int(banked_last.get("layer", -1)) == li:
                banked_lam = float(banked_last["ridge_lambda"])
            small_n_meta = {
                "ridge_3600_lambda": float(lam_3600),
                "banked_idbias_lambda": banked_lam,
                "lambda_matches_banked": (banked_lam is not None and banked_lam == float(lam_3600)),
            }
            small_n_meta["n50_lambda_source"] = (
                "3600-arm val-selected lambda (banked identity_bias_knn convention)"
            )
            small_n_meta["n50_caveat"] = (
                "n_train=50 << d=3584: deliberately under-determined, regularization-limited "
                "regime (H5 n-isolating rung; plan §4/#1701 statement) — R2 here is "
                "estimator-degenerate, never compared against large-n R2"
            )

        # residual_skip stated-exclusion row (plan §4; concern
        # residual-skip-exclusion-row-missing): the fitter has NO persisted weights
        # on HF, so it cannot be applied to test rows — banked-R2/cosine-only row.
        stated_exclusions: dict = {}
        if li == 19:
            rs = json.loads(BANKED_N1M.read_text())["per_point"]["mixed_1m"]["predictors"][
                "residual_skip"
            ]
            stated_exclusions["residual_skip"] = {
                "label": "Residual-skip (963k, banked-only)",
                "banked_r2": float(rs["whole_map_r2"]),
                "banked_mean_cosine": float(rs["mean_cosine"]),
                "exclusion": "no persisted weights — retrieval not evaluable "
                "(stated exclusion, plan §4)",
                "source": (
                    "eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json "
                    "per_point.mixed_1m.predictors.residual_skip"
                ),
            }

        pools = _build_pools_for_layer(cfg, li, Yte32, Y, test)
        gauss_ref = {}
        for spec in (pools[0], pools[-1]) if len(pools) > 1 else (pools[0],):
            gauss_ref[spec.name] = gaussian_hubness_reference(
                len(test), spec.pool64.shape[0], C.EXPECTED_HIDDEN, cfg.seed
            )

        arms_out: dict = {}
        layer_draws: dict = {}
        arm_list = list(preds)
        for a_i, arm in enumerate(arm_list):
            t_arm = time.time()
            recon, recon_draws = eval_recon_cell(preds[arm], rc, draws)
            if arm in FITTERS:
                _check_null_collapse(arm, recon, dup_note)
            retrieval: dict = {}
            arm_retr_draws: dict = {}
            for spec in pools:
                if spec.name.startswith("distr_") and arm not in BIG_POOL_ARMS:
                    continue
                r_out, r_draws = eval_retrieval_cell(
                    preds[arm],
                    spec,
                    KS_CONTEXT,
                    draws,
                    helper_parity=(spec.pool64.shape[0] <= 5000),
                    hub_diag=True,
                )
                for metric in ("euclidean", "csls"):
                    if arm in FITTERS:
                        _check_retrieval_null_collapse(
                            arm, metric, spec.name, r_out[metric], dup_note
                        )
                retrieval[spec.name] = r_out
                arm_retr_draws[spec.name] = r_draws
            arms_out[arm] = {
                "label": ARM_LABELS[arm],
                **recon,
                "retrieval": retrieval,
            }
            layer_draws[arm] = {
                "r2_boot": _round_list(recon_draws["r2_boot"]),
                "cos_boot": _round_list(recon_draws["cos_boot"]),
                "r2_null": _round_list(recon_draws["r2_null"]),
                "cos_null": _round_list(recon_draws["cos_null"]),
                "retrieval": {
                    pn: {m: {kk: _round_list(vv) for kk, vv in dd.items()} for m, dd in pd.items()}
                    for pn, pd in arm_retr_draws.items()
                },
            }
            logger.info(
                "[p2] L%d arm %d/%d %s r2=%.4f acc1(test,eucl)=%.3f elapsed=%.1fs",
                li,
                a_i + 1,
                len(arm_list),
                arm,
                arms_out[arm]["r2"]["point"],
                arms_out[arm]["retrieval"]["test"]["euclidean"]["acc_at_k"][1],
                time.time() - t_arm,
            )

        paired = _paired_contrasts(layer_draws, arms_out)
        result["per_layer"][str(li)] = {
            "applied_vs_banked": applied_vs_banked,
            "small_n_meta": small_n_meta,
            "stated_exclusions": stated_exclusions,
            "pools": {s.name: s.composition for s in pools},
            "pool_scope_note": (
                "distractor pools evaluated for the 963k ladder + identity_bias_3600 "
                "(BIG_POOL_ARMS; bounds P2 to the plan §9 GEMM budget); other small-n "
                "arms at test/passb pools"
            ),
            "gaussian_hubness_reference": gauss_ref,
            "arms": arms_out,
            "paired_contrasts": paired,
            "per_corpus_test_rows": {
                "lmsys": len(test),
                "wildchat": 0,
                "note": (
                    "test rows index the pass_b half by split construction (assemble() "
                    "assert) — all round-1 lmsys; the groupby degenerates to one group "
                    "(plan critic-fold b, answered honestly)"
                ),
            },
            "wall_s": round(time.time() - t_layer, 1),
        }
        all_draws[str(li)] = layer_draws
        result["split"] = {
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "seed": int(F.SPLIT_SEED),
            "val_sha256": banked["split"]["val_sha256"],
            "test_sha256": banked["split"]["test_sha256"],
            "byte_identical_banked": True,
        }
        if pilot:
            result["pilot"] = pilot
        result["metadata"] = _meta(cfg, "p2_context", t0)
        _atomic_json(out_path, result)  # per-layer incremental persistence
        _atomic_json(draws_path, all_draws)
        logger.info("[p2] L%d done in %.1fs", li, time.time() - t_layer)

    logger.info("[p2] done in %.1fs (ru_maxrss %.1f GB)", time.time() - t0, _ru_maxrss_gb())
    return result


def _paired_contrasts(layer_draws: dict, arms_out: dict) -> dict:
    """Paired-difference CIs on SHARED draws (plan critic-fold c): H2 nonlinear
    gaps (mlp/krr - ridge) on R2 + acc@1; H4 CSLS - plain cosine on acc@1."""
    out: dict = {}

    def _ci_of_diff(a: list, b: list) -> dict:
        d = np.asarray(a) - np.asarray(b)
        return {
            "mean": float(d.mean()),
            "lo": float(np.quantile(d, 0.025)),
            "hi": float(np.quantile(d, 0.975)),
        }

    if "ridge" in layer_draws:
        for name in ("mlp_w8192", "mlp_w32768", "krr_nystrom"):
            if name not in layer_draws:
                continue
            entry = {
                "r2": _ci_of_diff(layer_draws[name]["r2_boot"], layer_draws["ridge"]["r2_boot"])
            }
            for pool_name in layer_draws[name]["retrieval"]:
                if pool_name in layer_draws["ridge"]["retrieval"]:
                    entry[f"acc1_euclid_{pool_name}"] = _ci_of_diff(
                        layer_draws[name]["retrieval"][pool_name]["euclidean"]["acc1_boot"],
                        layer_draws["ridge"]["retrieval"][pool_name]["euclidean"]["acc1_boot"],
                    )
            out[f"{name}_minus_ridge"] = entry
    for arm, ad in layer_draws.items():
        for pool_name, pd in ad["retrieval"].items():
            if "csls" in pd and "cosine" in pd:
                out.setdefault("csls_minus_cosine_acc1", {}).setdefault(arm, {})[pool_name] = (
                    _ci_of_diff(pd["csls"]["acc1_boot"], pd["cosine"]["acc1_boot"])
                )
    return out


# ═══════════════════════════ Phase 3 — prefix arm ══════════════════════════════


def phase_p3(cfg: Cfg) -> dict:
    """Prefix-arm battery: the #722 50-context x 7-family battery (LOFO),
    loaders + fold structure inherited verbatim from issue722_identity_bias_knn,
    arm set extended to the full ladder + a batched LOFO MLP rung."""
    t0 = time.time()
    out_path = cfg.eval_dir / "prefix_arm.json"
    prior = _resume_skip(cfg, out_path, "p3")
    if prior is not None:
        return prior

    from issue810_common import battery_family_map
    from issue810_fit_reconstruction import _load_cc, _load_free_summaries

    from explore_persona_space.analysis.vectorized_mlp_skill import (
        MLPGroup,
        fit_batched_loco_mlp_multihead,
    )
    from explore_persona_space.experiments.issue_779.fit_h import ridge_fit_predict

    summaries, capture_layers = _load_free_summaries("betley")
    mean_summ = summaries["mean"]  # {ctx_id: (Lc, H)}
    ctx_ids = sorted(mean_summ)
    fam_map = battery_family_map(PROJECT_ROOT / "data" / "issue594" / "battery.json")
    fams = sorted({fam_map[c] for c in ctx_ids})
    assert len(ctx_ids) == 50 and len(fams) == 7, (len(ctx_ids), fams)
    cc = _load_cc(ctx_ids, list(range(len(capture_layers))))
    fam_of = np.array([fam_map[c] for c in ctx_ids])
    fam_labels = np.array([fams.index(f) for f in fam_of], dtype=np.int64)
    folds = [np.where(fam_of == f)[0] for f in fams]
    n = len(ctx_ids)

    banked722 = json.loads(BANKED_722.read_text())
    headline_li = int(banked722["best_ridge_layer"])
    if cfg.smoke:
        pick = sorted({0, list(capture_layers).index(headline_li), len(capture_layers) - 1})
        layer_cols = pick
    else:
        layer_cols = list(range(len(capture_layers)))
    logger.info(
        "[p3] %d layers (headline L%d, banked #722 val-selected), folds=%s",
        len(layer_cols),
        headline_li,
        [len(f) for f in folds],
    )

    # Batched LOFO MLP across ALL selected layers in ONE call (canonical helper;
    # row_groups = family labels -> group-LOFO folds, #928 extension; width 512 /
    # 300 epochs / lr 1e-3 / wd 1e-4 = the helper's #722-recipe defaults).
    xs, ys = {}, {}
    for lc in layer_cols:
        xs[lc] = np.stack([cc[c][lc] for c in ctx_ids]).astype(np.float64)
        ys[lc] = np.stack([np.asarray(mean_summ[c][lc].float(), dtype=np.float64) for c in ctx_ids])
    groups = [
        MLPGroup(key=(lc,), X=xs[lc].astype(np.float32), Y=ys[lc].astype(np.float32))
        for lc in layer_cols
    ]
    t_mlp = time.time()
    mlp_res = fit_batched_loco_mlp_multihead(
        groups, device="cpu", row_groups=fam_labels, standardization="per_fold"
    )
    logger.info(
        "[p3] batched LOFO MLP: %d members (%d layers x %d folds) in %.1fs",
        mlp_res.n_members,
        len(layer_cols),
        len(folds),
        time.time() - t_mlp,
    )

    arm_names = (
        "ridge_lofo",
        "identity_copy",
        "identity_bias",
        "scaled_identity",
        "diagonal_only",
        "const_fold_mean",
        "mlp_lofo",
    )
    p3_labels = {
        "ridge_lofo": "Linear map (ridge, LOFO)",
        "identity_copy": "Identity copy",
        "identity_bias": "Identity + learned bias (fold)",
        "scaled_identity": "Scaled identity (fold)",
        "diagonal_only": "Per-dim rescale (fold)",
        "const_fold_mean": "Constant fold-train-mean",
        "mlp_lofo": "Neural map (w512, LOFO, batched)",
    }
    per_layer: dict = {}
    draws = Draws.make(n, cfg.n_boot, cfg.k_perm, cfg.seed + 7000)
    headline_draws: dict = {}
    for j, lc in enumerate(layer_cols):
        li = int(capture_layers[lc])
        t_layer = time.time()
        x, y = xs[lc], ys[lc]
        oof = {name: np.zeros_like(y) for name in arm_names}
        oof["mlp_lofo"] = np.asarray(mlp_res.preds_by_key[(lc,)], dtype=np.float64)
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            oof["ridge_lofo"][te] = ridge_fit_predict(x[tr], y[tr], x[te])
            oof["identity_copy"][te] = x[te]
            oof["identity_bias"][te] = identity_bias_predict(x[tr], y[tr], x[te])
            oof["scaled_identity"][te] = B._fit_scale(x[tr], y[tr]) * x[te]
            oof["diagonal_only"][te] = x[te] * B._fit_diag(x[tr], y[tr])
            oof["const_fold_mean"][te] = np.tile(y[tr].mean(0), (len(te), 1))
        rc = ReconContext.make(y, draws)
        spec = PoolSpec.make("battery50", y.astype(np.float32), np.arange(n), fam_of.astype(object))
        arms_out: dict = {}
        for arm in arm_names:
            recon, recon_draws = eval_recon_cell(oof[arm], rc, draws)
            retrieval, retr_draws = eval_retrieval_cell(
                oof[arm], spec, KS_PREFIX, draws, helper_parity=True, hub_diag=(li == headline_li)
            )
            if arm in ("ridge_lofo", "mlp_lofo"):
                _check_null_collapse(f"prefix/{arm}", recon, "50-target pool, no distractors")
                for metric in ("euclidean", "csls"):
                    _check_retrieval_null_collapse(
                        f"prefix/{arm}",
                        metric,
                        "battery50",
                        retrieval[metric],
                        "50-target pool, no distractors",
                    )
            arms_out[arm] = {
                "label": p3_labels[arm],
                **recon,
                "retrieval": {"battery50": retrieval},
            }
            if li == headline_li:
                headline_draws[arm] = {
                    "r2_boot": _round_list(recon_draws["r2_boot"]),
                    "r2_null": _round_list(recon_draws["r2_null"]),
                    "retrieval": {
                        "battery50": {
                            m: {kk: _round_list(vv) for kk, vv in dd.items()}
                            for m, dd in retr_draws.items()
                        }
                    },
                }
        # per-family LOFO breakdown (headline layer): per held-out family
        if li == headline_li:
            fam_break: dict = {}
            for fi, f in enumerate(fams):
                te = folds[fi]
                fam_break[f] = {
                    "n": int(len(te)),
                    "r2_ridge": PR._pooled_r2(oof["ridge_lofo"][te], y[te]),
                    "r2_identity_bias": PR._pooled_r2(oof["identity_bias"][te], y[te]),
                    "acc1_ridge_euclid": float(
                        (
                            rank_matrix_for_cols(
                                _dist_euclid(oof["ridge_lofo"][te].astype(np.float64), spec), te
                            )[np.arange(len(te)), np.arange(len(te))]
                            <= 1
                        ).mean()
                    ),
                }
            per_layer_break = fam_break
        else:
            per_layer_break = None
        per_layer[str(li)] = {
            "arms": arms_out,
            **({"per_family": per_layer_break} if per_layer_break else {}),
            "wall_s": round(time.time() - t_layer, 1),
        }
        logger.info(
            "[p3] layer %d/%d (L%d) done in %.1fs — ridge r2=%.3f id+bias acc1=%.2f",
            j + 1,
            len(layer_cols),
            li,
            time.time() - t_layer,
            arms_out["ridge_lofo"]["r2"]["point"],
            arms_out["identity_bias"]["retrieval"]["battery50"]["euclidean"]["acc_at_k"][1],
        )

    out = {
        "design": {
            "n_contexts": 50,
            "families": fams,
            "folds": "leave-one-family-out (verbatim #722 fold structure)",
            "input": "c_C last-input-token, query-averaged (#594 store)",
            "target": "v_A mean answer summary, query-averaged (#658 store/v0_summaries.pt)",
            "ridge": "fit_h.ridge_fit_predict (numpy-SVD, GCV) — the #722 ridge path",
            "mlp": (
                "fit_batched_loco_mlp_multihead, row_groups=family labels (group-LOFO), "
                "hidden=512, max_epochs=300, lr=1e-3, wd=1e-4, seed=658 (helper defaults "
                "= the #722 skill recipe)"
            ),
            "ks": list(KS_PREFIX),
            "headline_layer": headline_li,
            "headline_layer_source": "banked #722 best_ridge_layer (val-selected; results.json)",
            "layers_evaluated": [int(capture_layers[lc]) for lc in layer_cols],
        },
        "n_train_ll_d_caveat": (
            "every prefix-arm fit has n_train ~= 43 << d = 3584 — a deliberately "
            "under-determined, regularization-limited regime (the banked #722 battery's own "
            "regime and the single-variable comparison target; the regime IS part of the "
            "object of study, H5). All prefix-arm R2 values are estimator-degenerate reads "
            "and are NEVER compared numerically against context-arm R2 (plan §4 / #1701)."
        ),
        "estimator_degenerate_regime": True,
        "per_layer": per_layer,
        "metadata": _meta(cfg, "p3_prefix", t0),
    }
    _atomic_json(out_path, out)
    _atomic_json(cfg.eval_dir / "boot_draws_prefix.json", {str(headline_li): headline_draws})
    logger.info("[p3] done in %.1fs (ru_maxrss %.1f GB)", time.time() - t0, _ru_maxrss_gb())
    return out


# ═══════════════════ Phase 4 — figures + characterization ══════════════════════

HERO_METRICS = (
    ("r2", "R2 (pooled)"),
    ("mean_cosine", "mean cosine"),
    ("acc1_euclid", "acc@1 (euclid)"),
    ("acc1_csls", "acc@1 (CSLS)"),
    ("mrr", "MRR"),
    ("median_rank", "median rank"),
)


def _arm_metric_value(arm: dict, key: str, pool: str) -> float:
    if key == "r2":
        return arm["r2"]["point"]
    if key == "mean_cosine":
        return arm["mean_cosine"]["point"]
    r = arm["retrieval"][pool]
    if key == "acc1_euclid":
        return (
            r["euclidean"]["acc_at_k"][1]
            if 1 in r["euclidean"]["acc_at_k"]
            else r["euclidean"]["acc_at_k"]["1"]
        )
    if key == "acc1_csls":
        return (
            r["csls"]["acc_at_k"][1] if 1 in r["csls"]["acc_at_k"] else r["csls"]["acc_at_k"]["1"]
        )
    if key == "mrr":
        return r["euclidean"]["mrr"]
    if key == "median_rank":
        return r["euclidean"]["median_rank"]
    raise KeyError(key)


def _err_offsets(point: float, lo: float, hi: float) -> tuple[float, float]:
    """Non-negative per-point offsets (matplotlib xerr/yerr contract — gotchas)."""
    return (max(0.0, point - lo), max(0.0, hi - point))


def _ladder_heatmap(ax, arms_dict: dict, pool: str, title: str, exclusions: dict | None = None):
    """Arms x HERO_METRICS annotated heatmap (column-normalized colors; median
    rank inverted). Hoisted from phase_p4 so the wc round's w3 hero side-by-side
    reuses the identical renderer (one live copy — code-style supersede rule)."""
    rows = list(arms_dict)
    M = np.array(
        [[_arm_metric_value(arms_dict[a], k, pool) for k, _lbl in HERO_METRICS] for a in rows]
    )
    ylabels = [arms_dict[a]["label"] for a in rows]
    # stated-exclusion rows (plan §4: residual_skip appears in the summary table
    # as a banked-R2/cosine-only row — retrieval cells render "n/a").
    for ex in (exclusions or {}).values():
        M = np.vstack(
            [
                M,
                [
                    ex["banked_r2"]
                    if k == "r2"
                    else (ex["banked_mean_cosine"] if k == "mean_cosine" else np.nan)
                    for k, _lbl in HERO_METRICS
                ],
            ]
        )
        ylabels.append(f"{ex['label']}\nno weights — retrieval n/a")
    norm = (M - np.nanmin(M, 0)) / (np.nanmax(M, 0) - np.nanmin(M, 0) + 1e-12)
    norm[:, -1] = 1 - norm[:, -1]  # median rank: lower is better
    ax.imshow(norm, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(HERO_METRICS)))
    ax.set_xticklabels([lbl for _k, lbl in HERO_METRICS], rotation=30, ha="right")
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(ylabels)
    for i in range(M.shape[0]):
        for j in range(len(HERO_METRICS)):
            v = M[i, j]
            ax.text(
                j,
                i,
                "n/a" if np.isnan(v) else f"{v:.3g}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if norm[i, j] < 0.5 else "black",
            )
    ax.set_title(title)


def phase_p4(cfg: Cfg) -> dict:
    t0 = time.time()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style("generic")
    ctx = json.loads((cfg.eval_dir / "context_arm.json").read_text())
    pfx = json.loads((cfg.eval_dir / "prefix_arm.json").read_text())
    man = json.loads((cfg.eval_dir / "distractor_manifest.json").read_text())
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    L19 = ctx["per_layer"]["19"]
    pfx_headline = str(pfx["design"]["headline_layer"])
    P = pfx["per_layer"][pfx_headline]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    _ladder_heatmap(
        axes[0],
        L19["arms"],
        "test",
        "Context arm (mixed_1m, L19; pool=1000 test)",
        exclusions=L19.get("stated_exclusions"),
    )
    _ladder_heatmap(
        axes[1], P["arms"], "battery50", f"Prefix arm (50-context battery, L{pfx_headline})"
    )
    fig.savefig(cfg.fig_dir / "hero_ladder_by_metric.png", dpi=200)
    plt.close(fig)

    # hero scatter: R2 (x) vs acc@1 (y) per (estimator x arm x layer)
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    pts = []
    for li, ld in ctx["per_layer"].items():
        for a, ad in ld["arms"].items():
            pts.append(
                (
                    ad["r2"]["point"],
                    _arm_metric_value(ad, "acc1_euclid", "test"),
                    f"ctx L{li} {a}",
                    "o",
                )
            )
    for a, ad in P["arms"].items():
        pts.append(
            (ad["r2"]["point"], _arm_metric_value(ad, "acc1_euclid", "battery50"), f"pfx {a}", "s")
        )
    for x, y, lbl, mk in pts:
        ax.scatter(max(x, -8), y, marker=mk, s=22)
        ax.annotate(lbl, (max(x, -8), y), fontsize=4, alpha=0.7)
    ax.set_xlabel("pooled R2 (clipped at -8)")
    ax.set_ylabel("retrieval acc@1 (euclid)")
    ax.set_title("Dissociation: variance explained vs discriminability")
    fig.savefig(cfg.fig_dir / "hero_r2_vs_acc1_scatter.png", dpi=200)
    plt.close(fig)

    # pool-size decay curves (L19, big-pool arms): acc@1 + median rank vs pool
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    pool_sizes = {}
    for a, ad in L19["arms"].items():
        names, accs, meds = [], [], []
        for pn, r in ad["retrieval"].items():
            npool = r["euclidean"]["n_pool"]
            names.append(npool)
            accs.append(_arm_metric_value(ad, "acc1_euclid", pn))
            meds.append(r["euclidean"]["median_rank"])
            pool_sizes[pn] = npool
        if len(names) < 2:
            continue
        order = np.argsort(names)
        axes[0].plot(np.array(names)[order], np.array(accs)[order], "o-", label=ad["label"], lw=1)
        axes[1].plot(np.array(names)[order], np.array(meds)[order], "o-", label=ad["label"], lw=1)
    xs_chance = sorted(pool_sizes.values())
    axes[0].plot(xs_chance, [1.0 / s for s in xs_chance], "k--", lw=0.8, label="chance")
    for ax, ylab in zip(axes, ("acc@1 (euclid)", "median rank")):
        ax.set_xscale("log")
        ax.set_xlabel("pool size")
        ax.set_ylabel(ylab)
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=5)
    fig.savefig(cfg.fig_dir / "pool_size_decay.png", dpi=200)
    plt.close(fig)

    # metric-comparison grouped bars with bootstrap CIs (clamped offsets)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    arms = list(L19["arms"])
    width = 0.27
    for mi, metric in enumerate(("euclidean", "cosine", "csls")):
        vals, los, his = [], [], []
        for a in arms:
            r = L19["arms"][a]["retrieval"]["test"][metric]
            v = r["acc_at_k"][1] if 1 in r["acc_at_k"] else r["acc_at_k"]["1"]
            vals.append(v)
            los.append(r["acc1_ci"]["lo"])
            his.append(r["acc1_ci"]["hi"])
        err = np.array([_err_offsets(v, lo, hi) for v, lo, hi in zip(vals, los, his)]).T
        ax.bar(
            np.arange(len(arms)) + (mi - 1) * width, vals, width, yerr=err, capsize=2, label=metric
        )
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([L19["arms"][a]["label"] for a in arms], rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("acc@1 (pool=1000)")
    ax.legend()
    fig.savefig(cfg.fig_dir / "metric_compare_bars.png", dpi=200)
    plt.close(fig)

    # per-dim R2 histograms
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
    edges = np.linspace(-2, 1, 49)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for ax, a in zip(axes.ravel(), arms[:8]):
        h = L19["arms"][a]["perdim_r2"]["hist_counts_minus2_to_1_48bins"]
        ax.bar(centers, h, width=np.diff(edges), align="center")
        ax.set_title(L19["arms"][a]["label"], fontsize=6)
        ax.axvline(0, color="k", lw=0.5)
    fig.suptitle("per-dim R2 (clipped at -2), context L19")
    fig.savefig(cfg.fig_dir / "perdim_r2_hist.png", dpi=200)
    plt.close(fig)

    # hubness N10 skewness bars + gaussian reference
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    labels, sk = [], []
    for a in ("ridge", "identity_bias", "const_mean"):
        for pn, r in L19["arms"][a]["retrieval"].items():
            for metric in ("euclidean", "cosine"):
                hb = r[metric].get("hubness")
                if hb:
                    labels.append(f"{a}\n{pn}/{metric[:4]}")
                    sk.append(hb["n10_skewness"])
    ax.bar(range(len(labels)), sk)
    ref = L19.get("gaussian_hubness_reference", {})
    for pn, rr in ref.items():
        ax.axhline(
            rr["euclidean"]["n10_skewness"], ls="--", lw=0.8, label=f"gaussian ref ({pn}, eucl)"
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=5, rotation=45, ha="right")
    ax.set_ylabel("N10 skewness")
    ax.legend(fontsize=6)
    fig.savefig(cfg.fig_dir / "hubness_n10.png", dpi=200)
    plt.close(fig)

    # 3600-vs-963k dumbbell (shared arms)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    pairs = [
        ("const_mean", "const_mean_3600"),
        ("identity_bias", "identity_bias_3600"),
        ("ridge", "ridge_3600"),
    ]
    for i, (big, small) in enumerate(pairs):
        if small not in L19["arms"]:
            continue
        v_big = _arm_metric_value(L19["arms"][big], "acc1_euclid", "test")
        v_small = _arm_metric_value(L19["arms"][small], "acc1_euclid", "test")
        ax.plot([v_small, v_big], [i, i], "-", color="grey", lw=1)
        ax.scatter([v_small], [i], color="tab:orange", label="3600-train" if i == 0 else None)
        ax.scatter([v_big], [i], color="tab:blue", label="963k-train" if i == 0 else None)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([p[0] for p in pairs])
    ax.set_xlabel("acc@1 (euclid, pool=1000)")
    ax.legend(fontsize=6)
    ax.set_title("n-dependence of the ladder (context L19)")
    fig.savefig(cfg.fig_dir / "dumbbell_3600_vs_963k.png", dpi=200)
    plt.close(fig)

    # null-distribution violins (headline arms, from persisted draws)
    draws_p = cfg.eval_dir / "boot_draws_context.json"
    if draws_p.exists():
        dd = json.loads(draws_p.read_text()).get("19", {})
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        data, labels = [], []
        for a in ("ridge", "identity_bias", "const_mean"):
            if a in dd:
                data.append(dd[a]["r2_null"])
                labels.append(f"{a}\nR2 null")
                data.append(dd[a]["cos_null"])
                labels.append(f"{a}\ncos null")
        if data:
            ax.violinplot(data, showmedians=True)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=6)
            ax.set_title(
                f"shuffled-pair null distributions (K={ctx['metadata']['regime']['k_perm']})"
            )
            fig.savefig(cfg.fig_dir / "null_violin.png", dpi=200)
        plt.close(fig)

        # rank CDFs (log-x) per estimator (plan §6 exploratory dump; round-2 fix):
        # per-row observed mid-ranks persisted by p2 in boot_draws_context.json.
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        for ax, metric in zip(axes, ("euclidean", "cosine", "csls")):
            for a, ad in dd.items():
                ranks = ad.get("retrieval", {}).get("test", {}).get(metric, {}).get("obs_ranks")
                if ranks is None:
                    raise RuntimeError(
                        "[p4] obs_ranks missing from boot_draws_context.json (pre-round-2 "
                        "p2 output) — re-run p2 with --force to persist per-row ranks"
                    )
                r = np.sort(np.asarray(ranks, dtype=np.float64))
                cdf = np.arange(1, len(r) + 1) / len(r)
                lbl = L19["arms"][a]["label"] if a in L19["arms"] else a
                ax.step(r, cdf, where="post", lw=1, label=lbl)
            ax.set_xscale("log")
            ax.set_xlabel("rank of true target (log)")
            ax.set_title(f"{metric} (pool=1000 test, context L19)", fontsize=8)
        axes[0].set_ylabel("CDF over test rows")
        axes[0].legend(fontsize=5)
        fig.savefig(cfg.fig_dir / "rank_cdf.png", dpi=200)
        plt.close(fig)

    # prefix per-family breakdown
    if "per_family" in P:
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        fams = list(P["per_family"])
        r2r = [P["per_family"][f]["r2_ridge"] for f in fams]
        r2i = [P["per_family"][f]["r2_identity_bias"] for f in fams]
        xpos = np.arange(len(fams))
        ax.bar(xpos - 0.2, r2r, 0.4, label="ridge LOFO")
        ax.bar(xpos + 0.2, r2i, 0.4, label="identity+bias")
        ax.set_xticks(xpos)
        ax.set_xticklabels(fams, rotation=30, ha="right", fontsize=6)
        ax.set_ylabel("per-family pooled R2 (estimator-degenerate regime)")
        ax.legend(fontsize=6)
        fig.savefig(cfg.fig_dir / "prefix_family_breakdown.png", dpi=200)
        plt.close(fig)

    characterization = _metric_characterization(ctx, pfx, man)
    _atomic_json(cfg.eval_dir / "metric_characterization.json", characterization)
    out = {
        "figures": sorted(p.name for p in cfg.fig_dir.glob("*.png")),
        "characterization": str(cfg.eval_dir / "metric_characterization.json"),
        "metadata": _meta(cfg, "p4_figures", t0),
    }
    # NO p4 sentinel by design: p4 is a cheap (~22 s) figures-only pass and ALWAYS
    # re-runs, so figures can never go stale against re-forced p2/p3 JSONs (the
    # round-1 written-never-read p4_done.json is removed rather than wired into
    # resume — code-review round 1 Minor 4).
    logger.info("[p4] done in %.1fs — %d figures", time.time() - t0, len(out["figures"]))
    return out


def _metric_characterization(ctx: dict, pfx: dict, man: dict) -> dict:
    """Per-metric field guide (plan Phase 4): construct / formula / invariances /
    measured constant-predictor score / measured empirical null floor / failure
    modes / which banked dissociation illustrates it."""
    L19 = ctx["per_layer"]["19"]["arms"]
    cm = L19["const_mean"]

    def _acc1(arm, metric):
        r = L19[arm]["retrieval"]["test"][metric]
        return r["acc_at_k"][1] if 1 in r["acc_at_k"] else r["acc_at_k"]["1"]

    ridge_null = L19["ridge"]["null"]
    ridge_retr_null = L19["ridge"]["retrieval"]["test"]
    return {
        "note": (
            "Constructs, invariances and measured floors for every mapping-quality metric "
            "in use; measured values from the context arm (mixed_1m, L19, pool=1000 test)."
        ),
        "metrics": {
            "pooled_r2": {
                "construct": "fraction of held-out target variance the map explains, "
                "variance-weighted across dims",
                "computed_as": "1 - ||Y-Yhat||^2_F / ||Y-Ybar||^2_F, SS_tot on the test set's "
                "own mean (banked whole_map_r2 convention)",
                "invariances": "sensitive to scale AND offset; dominated by high-variance dims; "
                "unbounded below",
                "constant_predictor_score": cm["r2"]["point"],
                "empirical_null_floor": ridge_null["r2"],
                "failure_modes": [
                    "a context-independent shift scores catastrophically negative while "
                    "retrieval stays high (identity+bias)",
                    "variance-weighting hides dims the map never moves",
                ],
                "banked_dissociation": "identity+bias R2 -0.865 at medrank 1 (#779 "
                "identity_bias_knn); acc@1 0.84 at pooled-OOF R2 -6.5 (#722)",
            },
            "perdim_r2": {
                "construct": "where the pooled number comes from (per-dimension fit)",
                "computed_as": "1 - SS_res_j / SS_tot_j per dim j",
                "invariances": "exposes the variance-weighting of pooled R2",
                "constant_predictor_score": cm["perdim_r2"]["median"],
                "empirical_null_floor": None,
                "failure_modes": [
                    "median can be positive while the pooled number is dominated "
                    "by a few high-variance dims (or vice versa)"
                ],
                "banked_dissociation": "diagonal_only: best non-fitted pooled R2 (+0.096) with "
                "near-floor retrieval (#779)",
            },
            "mean_cosine": {
                "construct": "directional agreement per row",
                "computed_as": "mean_i cos(pred_i, y_i) (banked mean_cosine convention)",
                "invariances": "per-row scale-invariant; HIGH anisotropy floor (activation "
                "spaces share a dominant mean direction)",
                "constant_predictor_score": cm["mean_cosine"]["point"],
                "empirical_null_floor": ridge_null["mean_cosine"],
                "failure_modes": [
                    "the anisotropy floor makes raw cosine look impressive for ANY predictor "
                    "near the mean direction (banked predict_the_mean cosine 0.807)",
                    "shuffled-pair null does NOT collapse to 0 — deliberately excluded from "
                    "the H3 kill (its high floor is the finding, plan §7)",
                ],
                "banked_dissociation": "predict_the_mean cosine 0.807 at exactly-chance "
                "retrieval (#779)",
            },
            "knn_acc_euclid": {
                "construct": "discriminability: can the prediction single out its own target "
                "in a candidate pool",
                "computed_as": "P(true target within k nearest pool neighbors), squared-euclid "
                "GEMM, mid-rank ties (mapping_baselines.knn_retrieval)",
                "invariances": "rank-based — invariant to any monotone distance rescale; "
                "floor = k/n_pool exactly for a constant predictor (pool==true)",
                "constant_predictor_score": _acc1("const_mean", "euclidean"),
                "empirical_null_floor": ridge_retr_null["euclidean"]["null"],
                "failure_modes": [
                    "pool-size dependent (chance = k/n_pool) — never compare across pools",
                    "duplicate pool content inflates tie mid-ranks",
                ],
                "banked_dissociation": "identity+bias acc@1 0.503 at R2 -0.865 (#779)",
            },
            "knn_acc_cosine": {
                "construct": "same, angle-only (norm-blind)",
                "computed_as": "1 - cos similarity as the distance",
                "invariances": "per-row norm-invariant on top of rank invariances",
                "constant_predictor_score": _acc1("const_mean", "cosine"),
                "empirical_null_floor": ridge_retr_null["cosine"]["null"],
                "failure_modes": [
                    "hubness in high-d corrupts plain cosine kNN (Radovanovic 2010; Dinu 2015)"
                ],
                "banked_dissociation": "identity+bias 0.553 cosine vs 0.503 euclid (#779)",
            },
            "csls_acc": {
                "construct": "discriminability, hubness-corrected",
                "computed_as": "2*cos(q,t) - r_query(q) - r_pool(t), k=10 cross-domain "
                "neighborhoods (Conneau 2018) — NEVER pool-internal",
                "invariances": "penalizes universal neighbors (hubs)",
                "constant_predictor_score": _acc1("const_mean", "csls"),
                "empirical_null_floor": ridge_retr_null["csls"]["null"],
                "failure_modes": [
                    "k-neighborhood choice; needs k <= n_query for the column neighborhood"
                ],
                "banked_dissociation": "H4: CSLS-vs-plain gap at large pools measures whether "
                "hubness is live at these pool sizes",
            },
            "median_rank": {
                "construct": "robust full-rank-distribution summary",
                "computed_as": "median over test rows of the true target's mid-rank",
                "invariances": "robust to tail rows; insensitive below n_pool/2 shifts",
                "constant_predictor_score": L19["const_mean"]["retrieval"]["test"]["euclidean"][
                    "median_rank"
                ],
                "empirical_null_floor": ridge_retr_null["euclidean"]["null"]["median_rank_mean"],
                "failure_modes": ["saturates at 1 for all decent maps at small pools"],
                "banked_dissociation": "identity+bias medrank 1 at R2 -0.865 (#779)",
            },
            "mrr": {
                "construct": "head-weighted rank summary",
                "computed_as": "mean(1/rank)",
                "invariances": "head-weighted — dominated by top-ranked rows",
                "constant_predictor_score": L19["const_mean"]["retrieval"]["test"]["euclidean"][
                    "mrr"
                ],
                "empirical_null_floor": ridge_retr_null["euclidean"]["null"]["mrr_mean"],
                "failure_modes": ["hides the tail of hard rows"],
                "banked_dissociation": "ridge MRR 0.790 at medrank 1 (#779)",
            },
            "hubness_n10": {
                "construct": "pool pathology diagnostic (NOT a quality metric)",
                "computed_as": "skewness of the 10-occurrence counts over pool items",
                "invariances": "property of the (pred, pool) geometry, not of the map's fit",
                "constant_predictor_score": None,
                "empirical_null_floor": None,
                "failure_modes": [
                    "hubs clustering by corpus mimic generic high-d hubness "
                    "(checked via top-hub corpus composition, plan critic-fold e)"
                ],
                "banked_dissociation": "n/a (diagnostic)",
            },
        },
        # plan §4 stated exclusion (residual_skip: banked-R2/cosine only, no weights);
        # direct index = fail-loud on a pre-round-2 p2 output.
        "stated_exclusions": ctx["per_layer"]["19"]["stated_exclusions"],
        "prefix_arm_caveat": pfx["n_train_ll_d_caveat"],
        "distractor_pool": {"dup_stats": man["dup_stats"], "corpus_counts": man["corpus_counts"]},
    }


# ═══════════ wildchat-target-battery follow-up round (plan v7, w0-w3) ═══════════
#
# Held-out-corpus transfer battery: does the LMSYS-measured arm RANKING survive a
# genuinely held-out real-user corpus region (fresh WildChat, streamed PAST the
# parent n1m consumption point)? All wc phases key their sentinels on
# cfg.wc_regime() (never regime() — parent sentinels stay valid) and reuse the
# parent battery machinery verbatim (Draws/ReconContext/eval_*_cell/PoolSpec).

# Banked parent-round outputs (committed, PROJECT_ROOT paths — deliberately
# NON-rebinding under --smoke: read-only parent inputs, #542 smoke-root rule).
BANKED_CONTEXT_ARM = PROJECT_ROOT / "eval_results/issue_1901/metric_battery/context_arm.json"
BANKED_CONTEXT_DRAWS = (
    PROJECT_ROOT / "eval_results/issue_1901/metric_battery/boot_draws_context.json"
)
BANKED_CHARACTERIZATION = (
    PROJECT_ROOT / "eval_results/issue_1901/metric_battery/metric_characterization.json"
)

_TOKENIZER_CACHE: dict[str, object] = {}


def _get_tokenizer(model_id: str):
    """Module-level tokenizer cache (never from_pretrained in a loop — the #664
    per-load model_info() Hub-429 gotcha)."""
    if model_id not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE[model_id] = AutoTokenizer.from_pretrained(model_id)
    return _TOKENIZER_CACHE[model_id]


def _wc_meta(cfg: Cfg, phase: str, t0: float) -> dict:
    """_meta with the wc regime substituted (wc sentinels compare wc_regime())."""
    m = _meta(cfg, phase, t0)
    m["regime"] = cfg.wc_regime()
    return m


def _wc_resume_skip(cfg: Cfg, out_path: Path, phase: str) -> dict | None:
    """wc-regime twin of _resume_skip (same fail-loud mismatch contract)."""
    if cfg.force or not out_path.exists():
        return None
    prior = json.loads(out_path.read_text())
    prior_regime = (prior.get("metadata") or {}).get("regime") or prior.get("regime")
    if prior_regime == cfg.wc_regime():
        logger.info("[%s] output %s exists with matching wc regime; skipping", phase, out_path)
        return prior
    raise RuntimeError(
        f"[{phase}] {out_path} exists under a DIFFERENT wc regime "
        f"(stored != current). Re-run with --force to redo, or use a different root.\n"
        f"stored:  {prior_regime}\ncurrent: {cfg.wc_regime()}"
    )


def _sha1_norm(prompt: str) -> str:
    """sha1 hex digest of the near-dupe-normalized prompt (the exclusion-set /
    contamination fingerprint — same normalization as the parent NearDupeGate)."""
    return hashlib.sha1(N1G._norm(prompt).encode("utf-8")).hexdigest()


class _CandidateGate(N1G.NearDupeGate):
    """Transposed NearDupeGate (the plan v7 ~10-line extension): the CANDIDATES
    are the indexed targets; train-pool rows stream through matching_targets(),
    which returns WHICH candidate indices the row collides with (exact-normalized
    match or char-ngram Jaccard >= thresh). Refusal-safe: indices/counts only.

    matching_targets() is the VECTORIZED counting path (one ragged posting
    gather + np.bincount => exact |g INTERSECT tg| per candidate — the same
    integers the serial per-candidate ``len(g & tg)`` computes, and the same
    float64 ``inter / union >= thresh`` compare, so survivor sets are IDENTICAL
    by construction; the equivalence gate in tests/test_issue1901_wildchat.py
    pins it). The pre-vectorization serial body is retained as
    matching_targets_serial_reference() ONLY for that gate (vectorize rule
    Supersede contract: contained serial twin). Rationale: the w0 pilot
    measured 12,338.9 us/row -> projected 11,907 s over 965k rows; a 200-row
    profile put 92% of that (11,353 us/row) in the per-candidate Jaccard loop
    (~918 candidates x ~173k C-level set-intersection ops per row) and 859
    us/row in the posting-union loop the bincount replaces."""

    def __init__(
        self,
        candidates: list[str],
        ngram: int = N1G.NEAR_DUPE_NGRAM,
        thresh: float = N1G.NEAR_DUPE_JACCARD,
    ):
        super().__init__(candidates, ngram=ngram, thresh=thresh)
        self._exact_idx: dict[str, list[int]] = {}
        for i, c in enumerate(candidates):
            self._exact_idx.setdefault(N1G._norm(c), []).append(i)
        # CSR posting index over the candidate ngram vocabulary (built once;
        # read-only afterwards => fork-shared across screen workers via COW).
        self._n_cand = len(candidates)
        self._vocab: dict[str, int] = {}
        posting_arrays: list[np.ndarray] = []
        for ng, tis in self.inv.items():
            self._vocab[ng] = len(posting_arrays)
            posting_arrays.append(np.fromiter(tis, dtype=np.int32, count=len(tis)))
        lens = np.array([a.size for a in posting_arrays], dtype=np.int64)
        self._post_offsets = np.zeros(len(posting_arrays) + 1, dtype=np.int64)
        np.cumsum(lens, out=self._post_offsets[1:])
        self._post_flat = (
            np.concatenate(posting_arrays) if posting_arrays else np.zeros(0, dtype=np.int32)
        )
        self._tg_sizes = np.array([len(tg) for tg in self.target_ngrams], dtype=np.int64)

    def matching_targets(self, prompt: str) -> set[int]:
        """Exact vectorized read: inv[ng] holds ti iff ng in tg, so counting
        posting hits over ng in g IS |g INTERSECT tg| per candidate."""
        n = N1G._norm(prompt)
        hits: set[int] = set(self._exact_idx.get(n, ()))
        g = N1G._char_ngrams(n, self.ngram)
        if not g:
            return hits
        vocab = self._vocab
        ids_list = [vocab[ng] for ng in g if ng in vocab]
        if not ids_list:
            return hits
        ids = np.asarray(ids_list, dtype=np.int64)
        starts = self._post_offsets[ids]
        lens = self._post_offsets[ids + 1] - starts
        total = int(lens.sum())
        csum = np.cumsum(lens)
        gather = np.repeat(starts - (csum - lens), lens) + np.arange(total, dtype=np.int64)
        inter = np.bincount(self._post_flat[gather], minlength=self._n_cand)
        # union >= len(g) >= 1 here, so the division is always defined; the
        # float64 divide matches the serial reference's int/int division exactly
        union = len(g) + self._tg_sizes - inter
        matched = np.flatnonzero((inter > 0) & (inter / union >= self.thresh))
        hits.update(int(ti) for ti in matched)
        return hits

    def matching_targets_serial_reference(self, prompt: str) -> set[int]:
        """Pre-vectorization serial body — retained ONLY as the oracle for the
        equivalence gate (unit test + the real-slice check); production code
        paths call matching_targets()."""
        n = N1G._norm(prompt)
        hits: set[int] = set(self._exact_idx.get(n, ()))
        g = N1G._char_ngrams(n, self.ngram)
        if not g:
            return hits
        cand: set[int] = set()
        for ng in g:
            cand |= self.inv.get(ng, set())
        for ti in cand:
            if ti in hits:
                continue
            tg = self.target_ngrams[ti]
            inter = len(g & tg)
            if inter == 0:
                continue
            union = len(g) + len(tg) - inter
            if union and inter / union >= self.thresh:
                hits.add(ti)
        return hits


def _ensure_payload_staged(cfg: Cfg, li: int, name: str) -> Path:
    """Stage one weight payload (revision-pinned, idempotent) + realized-keys
    check (#1073) — w2 must be runnable without a prior p0 on the same root."""
    rel = f"{WEIGHTS_PREFIX}/L{li}/{name}.pt"
    p = staged_path(cfg, rel)
    if not p.exists():
        got = hub.stage_hub_prefix(
            C.HF_DATA_REPO, rel, cfg.staging_root, repo_type="dataset", revision=cfg.revision
        )
        assert got == [p], (got, p)
    _realized_keys_check(p, FITTER_KIND[name])
    return p


def _ensure_distractors(cfg: Cfg) -> Path:
    """Local p1 distractor npz, staged back from its own HF upload on a miss
    (revision=None: the p1 upload postdates the parent-input pin by design)."""
    if not cfg.distractor_npz.exists():
        dest = f"{cfg.hf_out_prefix}/distractors_L19.npz"
        logger.info("[w2] distractor npz absent locally; staging from HF %s", dest)
        hub.stage_hub_file(C.HF_DATA_REPO, dest, cfg.distractor_npz, repo_type="dataset")
    return cfg.distractor_npz


# ─────────────────────────────── w0: candidates ─────────────────────────────────


def _stage_parent_manifest(cfg: Cfg) -> tuple[list[dict], dict]:
    """Stage + read the parent 960k n1m sampling manifest (revision-pinned;
    read_manifest_pool enforces the i==index + n_new invariants). Content
    hygiene: rows carry raw real-user text — never logged, only hashed."""
    staged_dir = staged_path(cfg, MANIFEST_PREFIX)
    if not N1G._manifest_complete_locally(staged_dir):
        files = hub.stage_hub_prefix(
            C.HF_DATA_REPO,
            MANIFEST_PREFIX,
            cfg.staging_root,
            repo_type="dataset",
            revision=cfg.revision,
        )
        logger.info("[w0] staged parent sampling manifest: %d files", len(files))
    pool, meta = N1G.read_manifest_pool(staged_dir)
    logger.info(
        "[w0] parent manifest: n_new=%d (lmsys=%d, wildchat=%d)",
        meta["n_new"],
        meta["n_lmsys"],
        meta["n_wildchat"],
    )
    return pool, meta


def _round1_prompts(cfg: Cfg, expected_sha: str) -> list[str]:
    """The 5,000 round-1 (pass_b) prompts, re-derived via the parent's own
    deterministic LMSYS-stream recovery and SHA-VERIFIED against the parent
    manifest's used_shas.round1 (plan §7: HALT on drift — a partial exclusion
    set is worse than no run). Cached on the shared staging root."""
    cache = cfg.staging_root / "wc_round1_prompts.jsonl"
    if cache.exists():
        round1 = [r["prompt"] for r in N1G._read_jsonl(cache)]
        if N1G.N10._sha_prompts(round1) == expected_sha:
            logger.info(
                "[w0] round-1 prompts loaded from cache (%d rows, sha-verified)", len(round1)
            )
            return round1
        logger.info("[w0] round-1 cache sha mismatch; re-deriving from the LMSYS stream")
    used = N50G.sample_disjoint_n50k(N1G.N_ROUND1, 0, 0)
    round1 = used["round1"]
    if used["round1_prompt_sha256"] != expected_sha:
        raise RuntimeError(
            f"KILL (round-1 recovery): re-derived round-1 prompt sha "
            f"{used['round1_prompt_sha256']} != parent manifest used_shas.round1 "
            f"{expected_sha} — LMSYS stream ordering drifted; the train-side exclusion "
            "set cannot be trusted (plan §7)."
        )
    N1G._atomic_write_jsonl(cache, [{"prompt": p} for p in round1])
    logger.info("[w0] round-1 prompts re-derived + cached (%d rows, sha-verified)", len(round1))
    return round1


def _wc_exclusion_set(
    cfg: Cfg, pool: list[dict], meta: dict, round1: list[str]
) -> tuple[set[str], dict]:
    """sha1(normalized prompt) exclusion fingerprints over the 960k manifest +
    5k round-1 prompts. Deterministic derived content — cached on the SHARED
    staging root (non-rebinding under --smoke: identical bytes both legs)."""
    fp_meta = {
        "recipe": WC_RECIPE_VERSION,
        "round1_sha": meta["used_shas"]["round1"],
        "manifest_sha": meta["new_prompt_sha256"],
        "hash": "sha1(near-dupe-normalized prompt)",
    }
    meta_path = cfg.staging_root / "wc_exclusion_fps.meta.json"
    if cfg.wc_exclusion_npz.exists() and meta_path.exists():
        stored = json.loads(meta_path.read_text())
        if {k: stored.get(k) for k in fp_meta} == fp_meta:
            fps = np.load(cfg.wc_exclusion_npz)["fps"]
            excl = {b.decode() for b in fps.tolist()}
            logger.info("[w0] exclusion set loaded (%d fingerprints, meta-matched)", len(excl))
            return excl, {**fp_meta, "n_fps": len(excl), "rebuilt": False}
    hexes = {_sha1_norm(r["prompt"]) for r in pool}
    hexes.update(_sha1_norm(p) for p in round1)
    fps = np.array(sorted(hexes), dtype="S40")
    _atomic_npz(cfg.wc_exclusion_npz, fps=fps)
    _atomic_json(meta_path, {**fp_meta, "n_fps": int(fps.shape[0])})
    logger.info(
        "[w0] exclusion set built: %d fingerprints (%d manifest + %d round1 rows, deduped)",
        len(hexes),
        len(pool),
        len(round1),
    )
    return hexes, {**fp_meta, "n_fps": len(hexes), "rebuilt": True}


# Fork-shared read-only state for the screen worker pool. Set by
# _wc_screen_candidates immediately before the fork (children inherit the gate
# + texts via copy-on-write; nothing is pickled per task beyond index bounds).
_SCREEN_GATE: _CandidateGate | None = None
_SCREEN_TEXTS: list[str] | None = None


def _screen_chunk_worker(bounds: tuple[int, int]) -> list[int]:
    """Screen train rows [start, end) against the fork-inherited candidate gate;
    returns the matched candidate indices (sorted, small)."""
    start, end = bounds
    assert _SCREEN_GATE is not None and _SCREEN_TEXTS is not None, "fork state unset"
    matched: set[int] = set()
    for j in range(start, end):
        matched |= _SCREEN_GATE.matching_targets(_SCREEN_TEXTS[j])
    return sorted(matched)


def _wc_screen_candidates(
    cfg: Cfg,
    cand_prompts: list[str],
    pool: list[dict],
    round1: list[str],
    workers: int | None = None,
) -> tuple[list[int], dict]:
    """Transposed train-pool near-dupe screen: candidates indexed, 960k+5k train
    rows streamed through the vectorized counting path (per-candidate Jaccard
    via one bincount — _CandidateGate.matching_targets), fanned out across fork
    workers. Pilot-timed SERIALLY on the first WC_SCREEN_PILOT_ROWS rows;
    projected wall (pilot elapsed + per-row x remaining / workers) > 2x
    WC_SCREEN_BUDGET_S HALTs (plan §9 w0 abort threshold, recalibrated to the
    vectorized + parallel execution shape)."""
    if workers is None:
        workers = max(1, int(os.environ.get("EPM_WC_SCREEN_WORKERS", str(WC_SCREEN_WORKERS))))
    gate = _CandidateGate(cand_prompts)
    train_texts = [r["prompt"] for r in pool] + list(round1)
    n_rows = len(train_texts)
    matched: set[int] = set()
    t0 = time.time()
    pilot: dict = {}

    # serial pilot leg (same per-row path the workers run)
    n_pilot = min(WC_SCREEN_PILOT_ROWS, n_rows)
    for j in range(n_pilot):
        matched |= gate.matching_targets(train_texts[j])
    if n_rows > WC_SCREEN_PILOT_ROWS:
        elapsed = time.time() - t0
        per = elapsed / n_pilot
        proj = elapsed + per * (n_rows - n_pilot) / workers
        pilot = {
            "pilot_rows": n_pilot,
            "per_row_us": round(per * 1e6, 2),
            "workers": workers,
            "projected_wall_s": round(proj, 1),
            "budget_s": WC_SCREEN_BUDGET_S,
        }
        logger.info(
            "[w0] screen pilot: %.1fus/row -> projected %.0fs over %d rows (%d workers)",
            per * 1e6,
            proj,
            n_rows,
            workers,
        )
        if proj > 2 * WC_SCREEN_BUDGET_S:
            raise RuntimeError(
                f"[w0] transposed screen projected {proj:.0f}s > 2x budget "
                f"{WC_SCREEN_BUDGET_S:.0f}s — halting before the burn (plan §9 w0 "
                "abort threshold; vectorize or re-scope before rerunning)"
            )

    # remainder: fan out across fork workers (read-only gate/texts via COW)
    remaining = n_rows - n_pilot
    if remaining > 0 and workers > 1:
        global _SCREEN_GATE, _SCREEN_TEXTS
        chunk = max(5_000, -(-remaining // (workers * 4)))
        bounds = [(s, min(s + chunk, n_rows)) for s in range(n_pilot, n_rows, chunk)]
        _SCREEN_GATE, _SCREEN_TEXTS = gate, train_texts
        try:
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(processes=workers) as mp_pool:
                for k, part in enumerate(mp_pool.imap_unordered(_screen_chunk_worker, bounds), 1):
                    matched.update(part)
                    logger.info(
                        "[w0] screen chunk %d/%d done (%d candidates matched) elapsed=%.0fs",
                        k,
                        len(bounds),
                        len(matched),
                        time.time() - t0,
                    )
        finally:
            _SCREEN_GATE = None
            _SCREEN_TEXTS = None
    elif remaining > 0:
        for j in range(n_pilot, n_rows):
            matched |= gate.matching_targets(train_texts[j])
            if (j + 1) % 200_000 == 0:
                logger.info(
                    "[w0] screen %d/%d rows (%d candidates matched) elapsed=%.0fs",
                    j + 1,
                    n_rows,
                    len(matched),
                    time.time() - t0,
                )

    survivors = [i for i in range(len(cand_prompts)) if i not in matched]
    stats = {
        "n_train_rows_screened": n_rows,
        "n_candidates": len(cand_prompts),
        "n_matched_dropped": len(matched),
        "n_survivors": len(survivors),
        "wall_s": round(time.time() - t0, 1),
        "workers": workers,
        "pilot": pilot,
        "near_dupe": {"ngram": gate.ngram, "jaccard_thresh": gate.thresh},
    }
    logger.info(
        "[w0] transposed screen: %d/%d candidates dropped (%.0fs, %d workers)",
        len(matched),
        len(cand_prompts),
        stats["wall_s"],
        workers,
    )
    return survivors, stats


def _upload_wc_manifest(cfg: Cfg, n_parts: int) -> dict:
    """Fail-loud mini-manifest upload (one commit) + exact-set verify (inside
    _upload_folder_filtered) to the ROUND ROOT's manifest/ prefix."""
    prefix = f"{cfg.wc_hf_root}/manifest"
    names = [f"part_{i:05d}.jsonl" for i in range(n_parts)] + ["meta.json"]
    url = hub._upload_folder_filtered(
        cfg.wc_manifest_dir,
        repo_id=C.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=prefix,
        allow_patterns=["part_*.jsonl", "meta.json"],
        expected_repo_paths=[f"{prefix}/{n}" for n in names],
    )
    if not url:
        raise RuntimeError(f"[w0] mini-manifest upload to {prefix} returned no URL")
    logger.info("[w0] mini-manifest uploaded: %d files -> %s", len(names), prefix)
    return {"hf_prefix": prefix, "url": url, "n_files": len(names)}


def phase_w0(cfg: Cfg) -> dict:
    t0 = time.time()
    prior = _wc_resume_skip(cfg, cfg.wc_w0_sentinel, "w0")
    if prior is not None:
        return prior
    cfg.wc_dir.mkdir(parents=True, exist_ok=True)

    # 1. parent 960k sampling manifest (revision-pinned)
    pool, meta = _stage_parent_manifest(cfg)
    wc_rows = [r for r in pool if r.get("corpus") == "wildchat"]
    assert wc_rows, "parent manifest has no wildchat rows — fresh-region start underivable"
    skip_first = max(int(r["stream_pos"]) for r in wc_rows) + 1

    # 2. round-1 (pass_b) prompt recovery — sha-verified (HALT on drift)
    round1 = _round1_prompts(cfg, meta["used_shas"]["round1"])

    # 3. exclusion fingerprints over manifest + round1
    excl, excl_info = _wc_exclusion_set(cfg, pool, meta, round1)

    # 4. WildChat revision pin — resolved ONCE, recorded, threaded to load_dataset
    from huggingface_hub import HfApi

    wc_revision = hub.retry_transient(
        lambda: HfApi().dataset_info(WILDCHAT_DATASET).sha, what="WildChat revision resolve"
    )
    logger.info("[w0] WildChat revision pinned: %s", wc_revision)

    # 5. fresh-region candidate stream (checkpointed, fingerprint-keyed resume;
    #    per-filter reject counters cover LIVE-streamed rows only)
    rejects = {"checked": 0, "exclusion_hit": 0}

    def keep_fresh(p: str) -> bool:
        rejects["checked"] += 1
        if _sha1_norm(p) in excl:
            rejects["exclusion_hit"] += 1
            return False
        return True

    fingerprint = {
        "recipe": WC_RECIPE_VERSION,
        "revision": wc_revision,
        "skip_first": skip_first,
        "target": cfg.wc_candidate_target,
        "round1_sha": meta["used_shas"]["round1"],
        "manifest_sha": meta["new_prompt_sha256"],
    }
    cands = N1G._stream_corpus(
        WILDCHAT_DATASET,
        WC_STREAM_TAG,
        keep_fresh,
        cfg.wc_candidate_target,
        cfg.wc_dir / "stream_cache",
        fingerprint,
        resume=not cfg.force,
        revision=wc_revision,
        skip_first=skip_first,
    )
    assert cands, "fresh WildChat stream kept 0 candidates (see per-filter reject counters)"

    # 6. transposed train-pool near-dupe screen (+ pilot-timed abort)
    survivors_idx, screen = _wc_screen_candidates(cfg, [r["prompt"] for r in cands], pool, round1)
    kept = [cands[i] for i in survivors_idx][: cfg.wc_manifest_keep]
    if len(kept) < cfg.wc_manifest_keep:
        logger.warning(
            "[w0] only %d screen survivors (< keep target %d); proceeding — the binding "
            "floor is w2's %d captured targets (plan §7 criterion 2)",
            len(kept),
            cfg.wc_manifest_keep,
            WC_TARGET_FLOOR,
        )

    # 7. mini-manifest (parent row schema; FRESH id namespace i=0..N-1)
    rows = [
        {"prompt": r["prompt"], "corpus": "wildchat", "stream_pos": int(r["stream_pos"]), "i": i}
        for i, r in enumerate(kept)
    ]
    mini_meta = {
        "n_new": len(rows),
        "n_lmsys": 0,
        "n_wildchat": len(rows),
        "recipe_version": WC_RECIPE_VERSION,
        "wildchat_revision": wc_revision,
        "skip_first": skip_first,
        "candidate_target": cfg.wc_candidate_target,
        "stream_rejects": {
            **rejects,
            "note": "counters cover LIVE-streamed rows only (a fingerprint-matched "
            "resume skips the already-consumed region)",
        },
        "screen": screen,
        "exclusion": excl_info,
        "used_shas": {
            "round1": meta["used_shas"]["round1"],
            "parent_manifest_new": meta["new_prompt_sha256"],
        },
        "new_prompt_sha256": N1G.N10._sha_prompts([r["prompt"] for r in rows]),
        "capture_layers": list(N1G.CAPTURE_LAYERS),
        "model": N1G.DEFAULT_MODEL,
    }
    n_parts = N1G._write_manifest_parts(cfg.wc_manifest_dir, rows, mini_meta)

    # 8. fail-loud upload + exact-set verify
    upload = _upload_wc_manifest(cfg, n_parts)

    out = {
        "n_candidates": len(cands),
        "n_survivors": len(survivors_idx),
        "n_kept": len(rows),
        "skip_first": skip_first,
        "wildchat_revision": wc_revision,
        "stream_rejects": rejects,
        "screen": screen,
        "exclusion": excl_info,
        "manifest": {"dir": str(cfg.wc_manifest_dir), "n_parts": n_parts, **upload},
        "metadata": _wc_meta(cfg, "w0_wc_candidates", t0),
    }
    _atomic_json(cfg.wc_w0_sentinel, out)
    logger.info(
        "[w0] done in %.1fs: %d candidates -> %d kept (digest-only; no prompt text logged)",
        time.time() - t0,
        len(cands),
        len(rows),
    )
    return out


# ─────────────────────────────── w1: capture ────────────────────────────────────


def _stage_wc_manifest(cfg: Cfg, dest: Path) -> Path:
    """Bridge the wc round's Hub manifest home ({wc_hf_root}/manifest) into the
    capture rig's expected local layout (out_dir/sampling_manifest — the rig's
    _resolve_manifest_dir hardcodes the PARENT prefix on its HF path, so we
    stage manually + set manifest_from_hf=False; #1776 flag-semantics rule)."""
    if N1G._manifest_complete_locally(dest):
        logger.info("[stage] manifest staged: already complete at %s", dest)
        return dest
    import shutil

    dest.mkdir(parents=True, exist_ok=True)
    if N1G._manifest_complete_locally(cfg.wc_manifest_dir):
        srcs = sorted(cfg.wc_manifest_dir.glob("part_*.jsonl"))
        srcs.append(cfg.wc_manifest_dir / "meta.json")
        for f in srcs:
            shutil.copy2(f, dest / f.name)
        logger.info("[stage] manifest staged: %d files (local w0 output)", len(srcs))
    else:
        mirror = cfg.wc_dir / "manifest_hf_mirror"
        prefix = f"{cfg.wc_hf_root}/manifest"
        # revision=None: the wc mini-manifest postdates the parent-input pin
        staged = hub.stage_hub_prefix(C.HF_DATA_REPO, prefix, mirror, repo_type="dataset")
        for p in staged:
            shutil.copy2(p, dest / Path(p).name)
        logger.info("[stage] manifest staged: %d files (HF %s)", len(staged), prefix)
    assert N1G._manifest_complete_locally(dest), f"staged wc manifest incomplete at {dest}"
    return dest


def _w1_args_attr_reads() -> set[str]:
    """AST-audit: every `args.<attr>` the reused capture rig actually reads on
    the w1 call path (run_capture + the args-consuming helpers it calls) — the
    hand-built-Namespace field audit (#1776 rule (iv))."""
    import ast
    import inspect
    import textwrap

    attrs: set[str] = set()
    for fn in (N1G.run_capture, N1G._resolve_manifest_dir, N1G._write_skipped_sidecar):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "args"
            ):
                attrs.add(node.attr)
    return attrs


def _w1_namespace(cfg: Cfg, *, device: str, no_upload: bool, out_dir: Path) -> argparse.Namespace:
    """The run_capture Namespace, field-audited against the rig's realized
    `args.` reads. hf_prefix carries N1G's ROUND-ROOT semantics (the module
    appends final_token_capture/ + raw_completions/ itself — #1776 gotcha)."""
    ns = argparse.Namespace(
        model=N1G.DEFAULT_MODEL,
        device=device,
        out_dir=out_dir,
        hf_prefix=cfg.wc_hf_root,
        manifest_from_hf=False,
        num_shards=1,
        shard_index=0,
        shard_size=500,
        no_upload=no_upload,
    )
    missing = _w1_args_attr_reads() - set(vars(ns))
    assert not missing, f"w1 Namespace missing args the capture rig reads: {sorted(missing)}"
    return ns


def _wc_captured_rowcount(cfg: Cfg) -> tuple[int, int]:
    """(n_rows, n_chunks) actually captured on the Hub, counted from the raw
    completion jsons (content read for len() only — never logged/printed)."""
    prefix = f"{cfg.wc_hf_root}/raw_completions"
    remote = N50G._remote_index(prefix)
    names = sorted(n for n in remote if n.endswith(".json") and "_skipped" not in n)
    scratch = cfg.wc_dir / "raw_probe"
    total = 0
    for name in names:
        local = hub.stage_hub_file(
            C.HF_DATA_REPO, f"{prefix}/{name}", scratch / name, repo_type="dataset"
        )
        total += len(json.loads(local.read_text())["rows"])
        local.unlink()
    return total, len(names)


def _w1_cpu_surface(cfg: Cfg) -> dict:
    """CPU-reachable w1 surface (the smoke leg of the GPU-bound capture phase —
    the documented carve-out): Namespace field audit, manifest staging + the
    rig's OWN consumer-open (_resolve_manifest_dir + read_manifest_pool),
    tokenize-only over-length filtering at the production budget, and the
    upload-dry-run sidecar branch. The GPU body (engine + generate + capture)
    is pilot-gated at chunk 1 in production by run_capture's own chunk loop."""
    w1_dir = cfg.wc_dir / "w1_surface"
    ns = _w1_namespace(cfg, device="cpu", no_upload=True, out_dir=w1_dir)
    manifest_dir = _stage_wc_manifest(cfg, w1_dir / N1G.MANIFEST_SUBDIR)
    resolved = N1G._resolve_manifest_dir(ns)
    assert resolved == manifest_dir, (resolved, manifest_dir)
    pool, _meta = N1G.read_manifest_pool(resolved)  # the rig's own consumer-open
    tok = _get_tokenizer(ns.model)
    kept_p, _kept_ci, skipped = N1G._filter_overlength_prompts(
        [r["prompt"] for r in pool],
        [int(r["i"]) for r in pool],
        lambda p: N1G._rendered_prompt_token_len(tok, p),
        N1G.PROMPT_TOKEN_BUDGET,
    )
    scratch = w1_dir / "shards"
    scratch.mkdir(parents=True, exist_ok=True)
    N1G._write_skipped_sidecar(scratch, ns, skipped)  # no_upload=True -> local write only
    out = {
        "mode": "cpu_surface (GPU capture body: documented carve-out)",
        "n_manifest_rows": len(pool),
        "n_kept_after_length_filter": len(kept_p),
        "n_overlength_skipped": len(skipped),
        "prompt_token_budget": N1G.PROMPT_TOKEN_BUDGET,
        "namespace_attrs_audited": sorted(_w1_args_attr_reads()),
        "sidecar": str(scratch / f"shard{ns.shard_index:02d}_skipped.json"),
    }
    logger.info(
        "[w1-surface] namespace audit + consumer-open + length filter + sidecar dry-run "
        "PASS (%d rows, %d kept)",
        len(pool),
        len(kept_p),
    )
    return out


def phase_w1(cfg: Cfg) -> dict:
    t0 = time.time()
    sentinel = cfg.out_root / "wc_w1_done.json"
    prior = _wc_resume_skip(cfg, sentinel, "w1")
    if prior is not None:
        return prior
    cfg.wc_dir.mkdir(parents=True, exist_ok=True)
    if cfg.smoke:
        out = _w1_cpu_surface(cfg)
    else:
        w1_dir = cfg.wc_dir / "w1_capture"
        ns = _w1_namespace(cfg, device="cuda", no_upload=False, out_dir=w1_dir)
        _stage_wc_manifest(cfg, w1_dir / N1G.MANIFEST_SUBDIR)
        rc = N1G.run_capture(ns)  # resumes by Hub chunk presence; uploads batched
        assert rc == 0, rc
        n_rows, n_chunks = _wc_captured_rowcount(cfg)
        if n_rows < WC_TARGET_FLOOR:
            raise RuntimeError(
                f"KILL (yield floor): {n_rows} captured wc rows < floor {WC_TARGET_FLOOR} "
                "(plan §7 kill criterion 2)"
            )
        out = {
            "n_captured_rows": n_rows,
            "n_chunks": n_chunks,
            "floor": WC_TARGET_FLOOR,
            "n_targets_planned": cfg.wc_n_targets,
        }
        logger.info(
            "[w1] captured %d rows across %d chunks (floor %d met)",
            n_rows,
            n_chunks,
            WC_TARGET_FLOOR,
        )
    out["metadata"] = _wc_meta(cfg, "w1_wc_capture", t0)
    _atomic_json(sentinel, out)
    return out


# ─────────────────────────────── w2: battery ────────────────────────────────────


def _wc_capture_targets(cfg: Cfg) -> dict:
    """Held-out wc targets from the w1 capture chunks (Hub, revision=None — the
    wc uploads postdate the parent pin). Rows returned in global-ci order."""
    prefix = f"{cfg.wc_hf_root}/final_token_capture"
    remote = N50G._remote_index(prefix)
    names = sorted(remote)
    if not names:
        raise RuntimeError(f"[w2] no capture chunks under {prefix} — run w1 first")
    scratch = cfg.wc_dir / "chunk_scratch"
    xs, ys, shas, cis = [], [], [], []
    for name in names:
        local = hub.stage_hub_file(
            C.HF_DATA_REPO, f"{prefix}/{name}", scratch / name, repo_type="dataset"
        )
        b = F._mmap_load(local)
        for fld in ("cx_last", "v_x", "ci", "layers", "prompts"):
            assert fld in b, (name, fld)
        assert list(b["layers"]) == list(N1G.CAPTURE_LAYERS), (name, b["layers"])
        xs.append(N50F._slice_layer(b, "cx_last", 19))
        ys.append(N50F._slice_layer(b, "v_x", 19))
        shas.extend(_sha1_norm(p) for p in b["prompts"])
        cis.extend(int(c) for c in b["ci"])
        del b
        local.unlink()
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    order = np.argsort(np.asarray(cis, dtype=np.int64), kind="stable")
    return {
        "X": X[order],
        "Y": Y[order],
        "shas": [shas[int(i)] for i in order],
        "n_chunks": len(names),
        "in_train": False,
        "source": f"HF {prefix} (wc held-out capture, w1)",
    }


def _parent_chunk_targets(cfg: Cfg, n_want: int) -> dict:
    """IN-TRAIN targets from parent n1m capture chunks (revision-pinned): the w2
    smoke stand-in AND the production --with-intrain-companion targets (plan v7
    branch c). Deterministic: chunk names sorted, rows in ci order."""
    remote = N50G._remote_index(CAPTURE_PREFIX)
    names = sorted(remote)
    assert names, f"no parent capture chunks under {CAPTURE_PREFIX}"
    scratch = cfg.wc_dir / "parent_chunk_scratch"
    xs, ys, shas, cis = [], [], [], []
    got = 0
    for name in names:
        local = hub.stage_hub_file(
            C.HF_DATA_REPO,
            f"{CAPTURE_PREFIX}/{name}",
            scratch / name,
            repo_type="dataset",
            revision=cfg.revision,
        )
        b = F._mmap_load(local)
        for fld in ("cx_last", "v_x", "ci", "layers", "prompts"):
            assert fld in b, (name, fld)
        assert list(b["layers"]) == list(N1G.CAPTURE_LAYERS), (name, b["layers"])
        xs.append(N50F._slice_layer(b, "cx_last", 19))
        ys.append(N50F._slice_layer(b, "v_x", 19))
        shas.extend(_sha1_norm(p) for p in b["prompts"])
        cis.extend(int(c) for c in b["ci"])
        got += xs[-1].shape[0]
        del b
        local.unlink()
        if got >= n_want:
            break
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    order = np.argsort(np.asarray(cis, dtype=np.int64), kind="stable")
    return {
        "X": X[order],
        "Y": Y[order],
        "shas": [shas[int(i)] for i in order],
        "n_chunks": len(xs),
        "in_train": True,
        "source": f"parent n1m capture chunks (IN-TRAIN; {CAPTURE_PREFIX} @ {cfg.revision[:12]})",
    }


def _wc_contamination_check(cfg: Cfg, shas: list[str], *, expect_hits: bool) -> dict:
    """Re-check every target row's fingerprint against the w0 exclusion set.
    Held-out targets with ANY hit HALT (plan §7 criterion 1); in-train targets
    (companion / smoke stand-in) EXPECT hits — informational only."""
    assert cfg.wc_exclusion_npz.exists(), (
        f"exclusion npz missing at {cfg.wc_exclusion_npz} — run w0 on this staging root first"
    )
    excl = {b.decode() for b in np.load(cfg.wc_exclusion_npz)["fps"].tolist()}
    hits = sum(1 for s in shas if s in excl)
    info = {"n_rows": len(shas), "n_exclusion_hits": hits, "expect_hits": expect_hits}
    if expect_hits:
        logger.info("[w2] contamination re-check (in-train targets, informational): %s", info)
    elif hits:
        raise RuntimeError(
            f"KILL (contamination): {hits}/{len(shas)} wc target rows fingerprint-match the "
            "train/round-1 exclusion set (plan §7 kill criterion 1)"
        )
    else:
        logger.info("[w2] contamination re-check PASS: 0/%d hits", len(shas))
    return info


def _w2_repro_control(cfg: Cfg, bundle) -> dict:
    """ESTIMATOR-IDENTITY control (plan v7 §4 w2, runs BEFORE any WildChat
    number): re-apply the staged banked L19 ridge to the PINNED lmsys test-1000
    and assert the pooled R2 reproduces the banked value within RIDGE_REPRO_TOL."""
    banked = json.loads(BANKED_CONTEXT_ARM.read_text())
    b_ridge = banked["per_layer"]["19"]["applied_vs_banked"]["ridge"]
    n_ctx = int(bundle["cx_last"].shape[0])
    _train, _val, test = F.fixed_split(n_ctx, n_ctx - 400 - 1000, 400, 1000, F.SPLIT_SEED)
    X = F.input_layer(bundle, "last", 19)
    Y = F.target_vx(bundle, 19)
    payload = torch.load(
        _ensure_payload_staged(cfg, 19, "ridge"), map_location="cpu", weights_only=False
    )
    pred = N1M.apply_map(payload, X[test], torch.device("cpu"))
    del payload
    applied = float(PR._pooled_r2(pred, np.asarray(Y[test], dtype=np.float64)))
    _assert_reproduction(applied, float(b_ridge["banked_r2"]))
    logger.info(
        "[w2] estimator-identity control PASS: applied=%.16g banked=%.16g (tol %.0e)",
        applied,
        b_ridge["banked_r2"],
        RIDGE_REPRO_TOL,
    )
    return {
        "applied_r2": applied,
        "banked_r2": float(b_ridge["banked_r2"]),
        "parent_applied_r2": float(b_ridge["applied_r2"]),
        "tol": RIDGE_REPRO_TOL,
        "note": "banked L19 ridge re-applied to the PINNED lmsys test-1000 BEFORE any "
        "wc number (plan v7 §4 w2)",
    }


def _wc_kill_check(cfg: Cfg, fn, *args) -> None:
    """Run a plan-§7 kill verdict; at smoke n the verdict is DEMOTED to a log
    line (the #1345 gate-calibration rule — production-n-calibrated verdicts
    fire spuriously at n=12) while the computation stays exercised. Production
    keeps the raise byte-identical."""
    if not cfg.smoke:
        fn(*args)
        return
    try:
        fn(*args)
    except RuntimeError as e:
        logger.warning("[w2] smoke-demoted kill verdict (computed, not raised): %s", e)


def _run_wc_battery(
    cfg: Cfg,
    tgt: dict,
    out_path: Path,
    draws_path: Path,
    *,
    repro_control: dict,
    bundle,
    seed_offset: int,
    label: str,
) -> tuple[dict, dict]:
    """The identical ladder x metric battery on wc targets: same estimators
    (banked 963k payloads applied verbatim; 3600/n50 rungs REFIT on the
    UNCHANGED pass_b train/val — only the evaluation targets swap), same
    metrics, same shared-draw CI/null machinery as parent P2."""
    t0 = time.time()
    X_all = np.asarray(tgt["X"])
    Y_all = np.asarray(tgt["Y"])
    n_avail = int(X_all.shape[0])
    if n_avail < cfg.wc_target_floor:
        raise RuntimeError(
            f"KILL (yield floor): {n_avail} target rows < floor {cfg.wc_target_floor} "
            "(plan §7 kill criterion 2)"
        )
    n = min(cfg.wc_n_targets, n_avail)
    if n < cfg.wc_n_targets:
        logger.warning(
            "[w2] only %d targets available (< planned %d); proceeding (floor %d met)",
            n,
            cfg.wc_n_targets,
            cfg.wc_target_floor,
        )
    Yte32 = np.asarray(Y_all[:n], dtype=np.float32)
    Xte32 = np.asarray(X_all[:n], dtype=np.float32)
    Xte64 = Xte32.astype(np.float64)
    contamination = _wc_contamination_check(cfg, tgt["shas"][:n], expect_hits=tgt["in_train"])

    # ── arm ladder (identical to P2 L19; smoke = ridge + the 3 baselines) ────
    arm_names = list(WC_ARMS_SMOKE) if cfg.smoke else [*ARMS_963K, *ARMS_3600, *ARMS_N50]
    preds: dict[str, np.ndarray] = {}
    pay_r = torch.load(
        _ensure_payload_staged(cfg, 19, "ridge"), map_location="cpu", weights_only=False
    )
    xmu = pay_r["xmu"].to(torch.float64).numpy()
    ymu = pay_r["ymu"].to(torch.float64).numpy()
    del pay_r
    if "const_mean" in arm_names:
        preds["const_mean"] = np.broadcast_to(ymu, Xte64.shape)
    if "identity_copy" in arm_names:
        preds["identity_copy"] = Xte64
    if "identity_bias" in arm_names:
        preds["identity_bias"] = Xte64 + (ymu - xmu)
    for name in FITTERS:
        if name not in arm_names:
            continue
        payload = torch.load(
            _ensure_payload_staged(cfg, 19, name), map_location="cpu", weights_only=False
        )
        preds[name] = N1M.apply_map(payload, Xte32, torch.device("cpu"))
        del payload
    small_n_meta: dict = {}
    if any(a in arm_names for a in (*ARMS_3600, *ARMS_N50)):
        n_ctx = int(bundle["cx_last"].shape[0])
        train, val, _test = F.fixed_split(n_ctx, n_ctx - 400 - 1000, 400, 1000, F.SPLIT_SEED)
        Xp = F.input_layer(bundle, "last", 19)
        Yp = F.target_vx(bundle, 19)
        sn_preds, lam_3600 = _small_n_preds(Xp, Yp, train, val, Xte64, cfg.seed)
        preds.update({k: v for k, v in sn_preds.items() if k in arm_names})
        small_n_meta = {
            "ridge_3600_lambda": float(lam_3600),
            "fit_inputs": "pass_b train/val UNCHANGED (identical refit inputs); only the "
            "evaluation targets swap to the wc pool (plan v7 §4 w2)",
        }

    # ── pools: targets-only headline (chance-matched vs parent test-1000) +
    #    parent-distractor pools at production (BIG_POOL_ARMS scoping as P2) ──
    label_t = "parent_intrain" if tgt["in_train"] else "wildchat_heldout"
    labels_t = np.array([label_t] * n, dtype=object)
    pools = [PoolSpec.make("test", Yte32, np.arange(n), labels_t)]
    if cfg.wc_pool_totals:
        blob = np.load(_ensure_distractors(cfg))
        dvx, dcorpus = blob["vx"], blob["corpus"]
        for total in cfg.wc_pool_totals:
            n_d = total - n
            assert dvx.shape[0] >= n_d, (dvx.shape[0], n_d, total)
            pools.append(
                PoolSpec.make(
                    f"distr_{total}",
                    np.concatenate([Yte32, dvx[:n_d]]),
                    np.arange(n),
                    np.concatenate([labels_t, dcorpus[:n_d].astype(object)]),
                )
            )
    gauss_ref = {}
    for spec in (pools[0], pools[-1]) if len(pools) > 1 else (pools[0],):
        gauss_ref[spec.name] = gaussian_hubness_reference(
            n, spec.pool64.shape[0], C.EXPECTED_HIDDEN, cfg.seed
        )

    draws = Draws.make(n, cfg.n_boot, cfg.k_perm, cfg.seed + seed_offset)
    rc = ReconContext.make(Yte32, draws)
    dup_note = f"{label}: {n} targets + parent distractors (dup_stats in distractor_manifest)"
    arms_out: dict = {}
    layer_draws: dict = {}
    arm_list = list(preds)
    for a_i, arm in enumerate(arm_list):
        t_arm = time.time()
        recon, recon_draws = eval_recon_cell(preds[arm], rc, draws)
        if arm in FITTERS:
            _wc_kill_check(cfg, _check_null_collapse, arm, recon, dup_note)
        retrieval: dict = {}
        arm_retr_draws: dict = {}
        for spec in pools:
            if spec.name.startswith("distr_") and arm not in BIG_POOL_ARMS:
                continue
            r_out, r_draws = eval_retrieval_cell(
                preds[arm],
                spec,
                KS_CONTEXT,
                draws,
                helper_parity=(spec.pool64.shape[0] <= 5000),
                hub_diag=True,
            )
            if arm in FITTERS:
                for metric in ("euclidean", "csls"):
                    _wc_kill_check(
                        cfg,
                        _check_retrieval_null_collapse,
                        arm,
                        metric,
                        spec.name,
                        r_out[metric],
                        dup_note,
                    )
            retrieval[spec.name] = r_out
            arm_retr_draws[spec.name] = r_draws
        arms_out[arm] = {"label": ARM_LABELS[arm], **recon, "retrieval": retrieval}
        layer_draws[arm] = {
            "r2_boot": _round_list(recon_draws["r2_boot"]),
            "cos_boot": _round_list(recon_draws["cos_boot"]),
            "r2_null": _round_list(recon_draws["r2_null"]),
            "cos_null": _round_list(recon_draws["cos_null"]),
            "retrieval": {
                pn: {m: {kk: _round_list(vv) for kk, vv in dd.items()} for m, dd in pd.items()}
                for pn, pd in arm_retr_draws.items()
            },
        }
        logger.info(
            "[w2] %s arm %d/%d %s r2=%.4f acc1(test,eucl)=%.3f elapsed=%.1fs",
            label,
            a_i + 1,
            len(arm_list),
            arm,
            arms_out[arm]["r2"]["point"],
            arms_out[arm]["retrieval"]["test"]["euclidean"]["acc_at_k"][1],
            time.time() - t_arm,
        )

    result = {
        "targets": {
            "n": n,
            "n_available": n_avail,
            "in_train": bool(tgt["in_train"]),
            "source": tgt["source"],
            "n_chunks": tgt.get("n_chunks"),
            "corpus_label": label_t,
            "contamination": contamination,
        },
        "repro_control": repro_control,
        "small_n_meta": small_n_meta,
        "pools": {s.name: s.composition for s in pools},
        "pool_scope_note": (
            "distractor pools evaluated for BIG_POOL_ARMS only (parent P2 GEMM-budget "
            "scoping); small-n arms at the targets-only pool"
        ),
        "gaussian_hubness_reference": gauss_ref,
        "arms": arms_out,
        "paired_contrasts": _paired_contrasts(layer_draws, arms_out),
        "wall_s": round(time.time() - t0, 1),
        "metadata": _wc_meta(cfg, label, t0),
    }
    _atomic_json(out_path, result)
    _atomic_json(draws_path, {"19": layer_draws})
    return result, layer_draws


def _transfer_comparison(cfg: Cfg, wc_arms: dict, wc_layer_draws: dict, targets_meta: dict) -> dict:
    """Arm-RANK transfer read vs the banked LMSYS context arm: Kendall tau per
    metric over the common arms + the pairwise-inversion table, with per-corpus
    paired-difference CIs (shared-draw re-reductions WITHIN each corpus) for the
    draw-bearing metrics."""
    from scipy.stats import kendalltau

    banked = json.loads(BANKED_CONTEXT_ARM.read_text())
    lm_arms = banked["per_layer"]["19"]["arms"]
    common = [a for a in lm_arms if a in wc_arms]
    assert len(common) >= 3, f"too few common arms for a rank-transfer read: {common}"

    def _val(arm: dict, key: str) -> float:
        if key == "acc1_cosine":
            r = arm["retrieval"]["test"]["cosine"]["acc_at_k"]
            return r[1] if 1 in r else r["1"]
        return _arm_metric_value(arm, key, "test")

    metrics = (
        "r2",
        "mean_cosine",
        "acc1_euclid",
        "acc1_cosine",
        "acc1_csls",
        "mrr",
        "median_rank",
    )
    tau_out: dict = {}
    inversions: dict = {}
    for key in metrics:
        lm = [_val(lm_arms[a], key) for a in common]
        wc = [_val(wc_arms[a], key) for a in common]
        tau, p = kendalltau(lm, wc)
        tau_out[key] = {
            "tau": float(tau),
            "p": float(p),
            "lmsys": {a: float(v) for a, v in zip(common, lm)},
            "wildchat": {a: float(v) for a, v in zip(common, wc)},
        }
        inv = []
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                if (lm[i] - lm[j]) * (wc[i] - wc[j]) < 0:
                    inv.append(
                        {
                            "pair": [common[i], common[j]],
                            "lmsys": [float(lm[i]), float(lm[j])],
                            "wildchat": [float(wc[i]), float(wc[j])],
                        }
                    )
        inversions[key] = inv

    lm_draws: dict = {}
    if BANKED_CONTEXT_DRAWS.exists():
        lm_draws = json.loads(BANKED_CONTEXT_DRAWS.read_text()).get("19", {})

    def _diff_ci(a_draws, b_draws) -> list[float]:
        d = np.asarray(a_draws, dtype=np.float64) - np.asarray(b_draws, dtype=np.float64)
        return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]

    getters = {
        "r2": lambda ad: ad["r2_boot"],
        "acc1_euclid": lambda ad: ad["retrieval"]["test"]["euclidean"]["acc1_boot"],
    }
    resolved: dict = {}
    for key, get in getters.items():
        rows = []
        for entry in inversions[key]:
            a, b = entry["pair"]
            if not all(x in lm_draws and x in wc_layer_draws for x in (a, b)):
                continue
            lm_ci = _diff_ci(get(lm_draws[a]), get(lm_draws[b]))
            wc_ci = _diff_ci(get(wc_layer_draws[a]), get(wc_layer_draws[b]))
            rows.append(
                {
                    "pair": [a, b],
                    "lmsys_diff_ci95": lm_ci,
                    "wildchat_diff_ci95": wc_ci,
                    "both_cis_exclude_zero": bool(
                        lm_ci[0] * lm_ci[1] > 0 and wc_ci[0] * wc_ci[1] > 0
                    ),
                }
            )
        resolved[key] = rows

    return {
        "setup": {
            "comparison": (
                "arm RANK transfer: banked LMSYS context arm (L19, pool=test-1000) vs the "
                "wc arm (targets-only pool) — each arm is scored against ITS OWN corpus "
                "targets BY DESIGN (the transfer question); per-corpus paired-difference "
                "CIs are within-corpus shared-draw re-reductions, never cross-corpus"
            ),
            "common_arms": common,
            "wc_targets": targets_meta,
        },
        "kendall_tau": tau_out,
        "pairwise_inversions": inversions,
        "inversions_ci_resolved": resolved,
        "regime": cfg.wc_regime(),
    }


def phase_w2(cfg: Cfg) -> dict:
    t0 = time.time()
    out_path = cfg.eval_dir / "wildchat_arm.json"
    draws_path = cfg.eval_dir / "boot_draws_wildchat.json"
    transfer_path = cfg.eval_dir / "transfer_comparison.json"
    companion_path = cfg.eval_dir / "wildchat_intrain_companion.json"
    prior = _wc_resume_skip(cfg, out_path, "w2")
    companion_due = cfg.intrain and not cfg.smoke
    if (
        prior is not None
        and transfer_path.exists()
        and (not companion_due or companion_path.exists())
    ):
        return prior

    assert BANKED_CONTEXT_ARM.exists(), f"banked context arm missing: {BANKED_CONTEXT_ARM}"
    cfg.wc_dir.mkdir(parents=True, exist_ok=True)
    cfg.eval_dir.mkdir(parents=True, exist_ok=True)
    _ensure_pass_b()
    bundle = F.load_pass_b()
    n_ctx = int(bundle["cx_last"].shape[0])
    assert n_ctx == F.N_PASS_B == 5000, n_ctx

    # estimator-identity control FIRST — before any WildChat number (plan v7 §4)
    repro = _w2_repro_control(cfg, bundle)

    # targets: held-out wc captures (production) / parent-chunk IN-TRAIN
    # stand-in (smoke — w1's GPU body cannot run on the VM; the stand-in keeps
    # the identical downstream chain executing on REAL capture-shaped rows)
    if cfg.smoke:
        tgt = _parent_chunk_targets(cfg, cfg.wc_n_targets)
    else:
        tgt = _wc_capture_targets(cfg)
    result, layer_draws = _run_wc_battery(
        cfg,
        tgt,
        out_path,
        draws_path,
        repro_control=repro,
        bundle=bundle,
        seed_offset=WC_DRAWS_SEED_OFFSET,
        label="w2_wc_battery",
    )
    transfer = _transfer_comparison(cfg, result["arms"], layer_draws, result["targets"])
    _atomic_json(transfer_path, transfer)

    if companion_due:
        tgt_in = _parent_chunk_targets(cfg, cfg.wc_n_targets)
        _run_wc_battery(
            cfg,
            tgt_in,
            companion_path,
            cfg.eval_dir / "boot_draws_wildchat_intrain.json",
            repro_control=repro,
            bundle=bundle,
            seed_offset=WC_DRAWS_SEED_OFFSET + 1,
            label="w2_wc_intrain_companion",
        )
    logger.info("[w2] done in %.1fs (ru_maxrss %.1f GB)", time.time() - t0, _ru_maxrss_gb())
    return result


# ─────────────────────────────── w3: figures ────────────────────────────────────


def phase_w3(cfg: Cfg) -> dict:
    """Transfer figures + the metric_characterization per-metric
    wildchat_transfer field. Like p4: cheap, always re-runs, no sentinel."""
    t0 = time.time()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("generic")
    ctx = json.loads(BANKED_CONTEXT_ARM.read_text())
    wc = json.loads((cfg.eval_dir / "wildchat_arm.json").read_text())
    trans = json.loads((cfg.eval_dir / "transfer_comparison.json").read_text())
    L19 = ctx["per_layer"]["19"]["arms"]
    W = wc["arms"]
    n_wc = wc["targets"]["n"]
    wc_label = "WildChat held-out" if not wc["targets"]["in_train"] else "in-train stand-in"
    common = trans["setup"]["common_arms"]
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. hero side-by-side transfer heatmap (identical renderer as the p4 hero)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    _ladder_heatmap(axes[0], L19, "test", "LMSYS test-1000 (banked context arm, L19)")
    _ladder_heatmap(axes[1], W, "test", f"{wc_label} targets (n={n_wc}, L19)")
    savefig_paper(fig, "wc_hero_transfer_heatmap", dir=cfg.fig_dir)
    plt.close(fig)

    # 2. per-arm dumbbells: lmsys -> wc (acc@1 euclid + pooled R2, R2 clipped)
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.45 * len(common) + 2), layout="constrained")
    for ax, key, xlab in (
        (axes[0], "acc1_euclid", "acc@1 (euclid, targets-only pool)"),
        (axes[1], "r2", "pooled R2 (clipped at -2)"),
    ):
        for i, a in enumerate(common):
            v_lm = trans["kendall_tau"][key]["lmsys"][a]
            v_wc = trans["kendall_tau"][key]["wildchat"][a]
            if key == "r2":
                v_lm, v_wc = max(v_lm, -2.0), max(v_wc, -2.0)
            ax.plot([v_lm, v_wc], [i, i], "-", color="grey", lw=1)
            ax.scatter([v_lm], [i], color="tab:blue", label="LMSYS" if i == 0 else None)
            ax.scatter([v_wc], [i], color="tab:orange", label="WildChat" if i == 0 else None)
        ax.set_yticks(range(len(common)))
        ax.set_yticklabels([L19[a]["label"] for a in common], fontsize=6)
        ax.set_xlabel(xlab)
    axes[0].legend(fontsize=6)
    fig.suptitle(
        "Arm transfer LMSYS -> WildChat "
        f"(tau acc@1={trans['kendall_tau']['acc1_euclid']['tau']:.2f}, "
        f"tau R2={trans['kendall_tau']['r2']['tau']:.2f})"
    )
    savefig_paper(fig, "wc_dumbbell_lmsys_vs_wc", dir=cfg.fig_dir)
    plt.close(fig)

    # 3. pool-size decay overlay (solid=lmsys, dashed=wc; headline arms)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    pool_sizes: set[int] = set()
    for src, arms_d, ls in (("lmsys", L19, "-"), ("wildchat", W, "--")):
        for a in ("ridge", "identity_bias"):
            if a not in arms_d:
                continue
            sizes, accs = [], []
            for _pn, r in arms_d[a]["retrieval"].items():
                acc = r["euclidean"]["acc_at_k"]
                sizes.append(int(r["euclidean"]["n_pool"]))
                accs.append(acc[1] if 1 in acc else acc["1"])
                pool_sizes.add(int(r["euclidean"]["n_pool"]))
            order = np.argsort(sizes)
            ax.plot(
                np.array(sizes)[order],
                np.array(accs)[order],
                marker="o",
                ls=ls,
                lw=1,
                label=f"{a} ({src})",
            )
    xs_chance = sorted(pool_sizes)
    ax.plot(xs_chance, [1.0 / s for s in xs_chance], "k:", lw=0.8, label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("pool size")
    ax.set_ylabel("acc@1 (euclid)")
    ax.legend(fontsize=6)
    ax.set_title("Pool-size decay: LMSYS (solid) vs WildChat (dashed)")
    savefig_paper(fig, "wc_pool_size_decay", dir=cfg.fig_dir)
    plt.close(fig)

    # 4. wc null violins (persisted shared draws)
    wcd_path = cfg.eval_dir / "boot_draws_wildchat.json"
    if wcd_path.exists():
        dd = json.loads(wcd_path.read_text()).get("19", {})
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        data, labels = [], []
        for a in ("ridge", "identity_bias", "const_mean"):
            if a in dd:
                data.append(dd[a]["r2_null"])
                labels.append(f"{a}\nR2 null")
                data.append(dd[a]["cos_null"])
                labels.append(f"{a}\ncos null")
        if data:
            ax.violinplot(data, showmedians=True)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=6)
            ax.set_title(
                f"wc shuffled-pair null distributions (K={wc['metadata']['regime']['k_perm']})"
            )
            savefig_paper(fig, "wc_null_violin", dir=cfg.fig_dir)
        plt.close(fig)

    # 5. metric_characterization: per-metric wildchat_transfer field
    src = cfg.eval_dir / "metric_characterization.json"
    if not src.exists():
        src = BANKED_CHARACTERIZATION  # smoke: read the committed parent copy
    ch = json.loads(src.read_text())
    key_map = {
        "pooled_r2": "r2",
        "mean_cosine": "mean_cosine",
        "knn_acc_euclid": "acc1_euclid",
        "knn_acc_cosine": "acc1_cosine",
        "csls_acc": "acc1_csls",
        "median_rank": "median_rank",
        "mrr": "mrr",
    }
    taus = trans["kendall_tau"]
    for mname, entry in ch.get("metrics", {}).items():
        tk = key_map.get(mname)
        if tk and tk in taus:
            entry["wildchat_transfer"] = {
                "kendall_tau_vs_lmsys": taus[tk]["tau"],
                "p": taus[tk]["p"],
                "n_common_arms": len(common),
                "n_inversions": len(trans["pairwise_inversions"][tk]),
                "wc_targets_in_train": bool(wc["targets"]["in_train"]),
                "source": "transfer_comparison.json (w2; targets-only pool, L19)",
            }
        else:
            entry["wildchat_transfer"] = {
                "note": "no rank-transfer read (diagnostic/decomposition metric)"
            }
    ch["wildchat_transfer_note"] = (
        "per-metric wildchat_transfer added by w3_wc_figures (wildchat-target-battery round)"
    )
    _atomic_json(cfg.eval_dir / "metric_characterization.json", ch)

    out = {
        "figures": sorted(p.name for p in cfg.fig_dir.glob("wc_*.png")),
        "characterization": str(cfg.eval_dir / "metric_characterization.json"),
        "metadata": _wc_meta(cfg, "w3_wc_figures", t0),
    }
    logger.info("[w3] done in %.1fs — %d wc figures", time.time() - t0, len(out["figures"]))
    return out


# ═══════════════════════════════════ main ══════════════════════════════════════

PHASES = ("p0_stage", "p1_distractors", "p2_context", "p3_prefix", "p4_figures")
WC_PHASES = ("w0_wc_candidates", "w1_wc_capture", "w2_wc_battery", "w3_wc_figures")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True, choices=(*PHASES, "all", *WC_PHASES, "wc_all"))
    ap.add_argument(
        "--staging-root",
        type=Path,
        required=True,
        help="multi-GB staging root (data disk, e.g. /mnt/eps-data/$USER/"
        "issue1901_metrics — NEVER / or /tmp)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="same code paths, reduced knobs; outputs under <root>/smoke/",
    )
    ap.add_argument(
        "--revision",
        default=KNOWN_GOOD_REVISION,
        help="data-repo revision pin for all staged inputs",
    )
    ap.add_argument("--seed", type=int, default=1901)
    ap.add_argument("--force", action="store_true", help="redo phases despite sentinels")
    ap.add_argument(
        "--with-intrain-companion",
        action="store_true",
        help="w2: ALSO run the ladder on parent IN-TRAIN capture rows -> "
        "wildchat_intrain_companion.json (plan v7 branch c; production only)",
    )
    args = ap.parse_args()

    cfg = Cfg(
        phase=args.phase,
        staging_root=args.staging_root,
        smoke=bool(args.smoke),
        revision=args.revision,
        seed=int(args.seed),
        force=bool(args.force),
        intrain=bool(args.with_intrain_companion),
    )
    cfg.eval_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    logger.info(
        "[main] phase=%s smoke=%s staging_root=%s revision=%s",
        cfg.phase,
        cfg.smoke,
        cfg.staging_root,
        cfg.revision,
    )

    runners = {
        "p0_stage": phase_p0,
        "p1_distractors": phase_p1,
        "p2_context": phase_p2,
        "p3_prefix": phase_p3,
        "p4_figures": phase_p4,
        "w0_wc_candidates": phase_w0,
        "w1_wc_capture": phase_w1,
        "w2_wc_battery": phase_w2,
        "w3_wc_figures": phase_w3,
    }
    if cfg.phase == "all":
        todo: tuple[str, ...] = PHASES
    elif cfg.phase == "wc_all":
        todo = WC_PHASES
    else:
        todo = (cfg.phase,)
    for ph in todo:
        logger.info("[phase=%s] starting", ph)
        runners[ph](cfg)
        logger.info("[phase=%s] done", ph)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit BEFORE C-extension finalize teardown (gotchas)
