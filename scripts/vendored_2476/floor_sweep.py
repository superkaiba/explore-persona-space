"""Issue #2476 same-issue follow-up `floor-sensitivity-sweep` — pod-side driver.

Plan: tasks/<status>/2476/plans/plan.md (v6). ONE variable changes vs the parent
run: the alive-floor convention becomes a swept analysis parameter — fresh arm
{1,200, 600, 300, 240} of 120,000 fit rows (1%/0.5%/0.25%/0.2%), bridge
{240, 120, 60} of 24,000 — and the alive panels, per-tier medians (map and
identity+bias), the within-activity-quintile tier permutation, the endpoint
contrast, per-tier retrieval, and tierwise firing-rate distributions are
recomputed at each floor. Weights, splits, banked dense predictions, score
rows, seeds, and estimators are IDENTICAL to the parent (nothing is refit; no
generation; no judge). The registered decision rule stays registered at the 1%
floor ONLY; swept floors are labeled "reported (registered at 1% only)".

Phases (plan §4; every estimator kernel is the PARENT's, imported by file path
via the issue2476_perrow_views._load_driver pattern — never re-implemented):
  smoke        Q0 composed tiny-slice run of the SAME phase functions under
               out_root/smoke (--max-chunks 2), REAL SAE-c + chanind loads,
               per-leg output verification + wall fence bases. Production-n
               gates (G-C/G-R) demote to informational at smoke n.
  assemble     Q1 = the parent's phase_assemble VERBATIM (X19/Y19 fp16 memmaps
               from the 1,920 #779 capture chunks; split shas re-asserted).
  stage_banked Q2 revision-pinned staging of every banked input + gate G-A
               (row alignment) + the A7/A4 key/shape asserts.
  census       Q3 full-width fit-side counts+sums passes (SAE-c on Y19; chanind
               lmsys AND pile on vbar20) + gate G-C (counts reconciliation,
               +/-3) + gate G-S (split shas) + union-column restricted encodes
               (fresh 2,150 cols; bridge 800; encoded ONCE, subset per floor).
  stats        Q4 per-floor batteries via the parent kernels (_r2_only,
               _tier_stats, _shuffle_null_r2, _retrieval_cells) + gate G-R
               (registered-floor reproduction vs the committed
               eval_results/issue_2476/turnavg/ references, read at run time).
  figures      Q5 hero + exploratory figures, eval JSONs/npzs -> git
               eval_results/issue_2476/floor_sweep/ (force-add + staged-index
               verify + push-verify), census tensors -> HF
               <hf_prefix>/floor_sweep/, terminal results sentinel.

Binding plan-approval additions (epm:plan-approved v2, 2026-08-24):
  [stat-rec-2]     per-floor rows carry `not_evaluable_census_only` (plan §12
                   A13: >50% of a populated tier's alive features undefined-R2
                   on score rows, evaluated on the map read) and a demoted
                   floor gets NO lattice label — the field replaces it.
  [stat-concern-4] tests/test_issue2476_gates.py pins nonzero-exit failure
                   semantics for _gate_counts (G-C) and _gate_repro (G-R).

Pod-side contract: sentinels under /workspace/logs/issue-2476-*.json ONLY
(never task.py); [phase=...] log lines; [phase=done] terminal. LMSYS/WildChat
text is handled DIGEST-ONLY. Resume is REGIME-KEYED (a regime.json beside each
phase's outputs; config mismatch refuses, code-SHA-only mismatch recomputes
loudly); resume keys hash GENERATING PARAMETERS only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM smoke)

import numpy as np  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2476_floor_sweep")

TASK_ID = 2476
DATA_REPO = "superkaiba1/explore-persona-space-data"
# The parent round's post-run pin (issue2476_perrow_views.py:68; plan §10 card header).
DATA_REPO_REVISION = "89cfa76cdcd4207d95c1fec1c3131f36e21beec0"
TA = "issue2476_turnavg/analysis_tensors"
REFIT_HF = "issue1482_error_analysis/analysis_tensors/percontext/refit_holdout__ridge__seed0.npz"

# Floor grids (plan §11: registered 1% + halvings bracketing k/width ~= 0.153% + the
# #1482 absolute 240-row floor; bridge degenerates to 3 distinct floors).
FLOORS_C = (1200, 600, 300, 240)
FLOORS_B = (240, 120, 60)
REGISTERED_FLOOR_C = 1200
REGISTERED_FLOOR_B = 240

# Gate tolerances (plan §7; G-R quantile predicate amended v7 — plan §11 "G-R
# tolerances (amended v7)"). GR_PERFEATURE_TOL mirrors issue2476_perrow_views.ACC_REPRO_TOL
# (:80) — the parent line's MEASURED backend near-tie ceiling for this recompute class;
# v7 binds it as a >= GR_SHARE_MIN coverage bar (not a max) after the 2026-08-24
# production halt (pod-2476 gates/gr_c.json: exactly 1/879 features, feat 56641 tier 2,
# at delta 2.83e-3 while p99 = 2.15e-4 — a cross-pod fp near-tie, not instrument drift).
# GR_MAX_TOL hard-fails any order-of-magnitude excursion; the per-tier median conjunct
# binds only on tiers with n >= GR_MEDIAN_MIN_TIER_N panel features (an n=3 median IS a
# single feature — the incident's failing conjunct was that same feature's value).
# NOTE (regime key): the three v7 constants are deliberately NOT in _regime() — the
# predicate shape is code (covered by code_sha's loud-recompute branch), and adding
# base keys would flip config_hash and REFUSE resume on the halted production out-root.
GC_COUNT_TOL = 3
GR_PERFEATURE_TOL = 2e-3
GR_MAX_TOL = 1e-2
GR_SHARE_MIN = 0.99
GR_MEDIAN_TOL = 1e-3
GR_MEDIAN_MIN_TIER_N = 10
RC_GA = 30  # G-A row-alignment HALT (parent RC class convention: 22-26 taken)
RC_GC = 31  # G-C counts-reconciliation HALT
RC_GR = 32  # G-R registered-floor reproduction HALT
RC_GS = 33  # G-S split-sha HALT

LATTICE_NOTE = "reported (registered at 1% only)"
EV_TURNAVG = ROOT / "eval_results" / "issue_2476" / "turnavg"  # committed G-R references
K_OVER_WIDTH = 100.0 / 65536.0  # instrument mean firing rate (figure annotation)

# Smoke-only bridge row clamps (the store is production-grain; rows are subset).
SMOKE_B_FIT = 256
SMOKE_B_SCORE = 48

STAGE_FILES = (
    f"{TA}/eval/alive_c.npz",
    f"{TA}/eval/alive_b.npz",
    f"{TA}/eval/ib_c.npz",
    f"{TA}/eval/armb_maps.npz",
    f"{TA}/eval/ftrue_b.npz",
    f"{TA}/eval/ftrue_c_all.fp16.npy",
    f"{TA}/sae_c/cfg.json",
    f"{TA}/sae_c/train_log.json",
    f"{TA}/sae_c/sae_weights.safetensors",
    # Consolidated recapture store only (stated deviation from the plan's 68-file
    # row: vbar_store.npz IS the consolidation of the 64 vbar_g*.npz shards —
    # §12 A8's row_idx set-equality verify is unchanged, the shards are redundant).
    f"{TA}/recapture_store/vbar_store.npz",
    f"{TA}/split_meta/split_meta.json",
    REFIT_HF,
)

_DRV = None


def _drv():
    """The parent driver, imported by FILE PATH (issue2476_perrow_views pattern) —
    gives every reused kernel (_r2_only/_tier_stats/_retrieval_cells/...) plus its
    own module aliases (C/S/M/EA). Cached; reuses a pytest-imported module."""
    global _DRV
    if _DRV is None:
        if "issue2476_turnavg_sae" in sys.modules:
            _DRV = sys.modules["issue2476_turnavg_sae"]
            return _DRV
        spec = importlib.util.spec_from_file_location(
            "issue2476_turnavg_sae", ROOT / "scripts" / "issue2476_turnavg_sae.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["issue2476_turnavg_sae"] = mod
        spec.loader.exec_module(mod)
        _DRV = mod
    return _DRV


# ── small utils ──────────────────────────────────────────────────────────────────


def _production(args) -> bool:
    """Mirror of the parent predicate (full grid, no smoke clamps)."""
    return int(args.max_chunks) == 0 and int(args.smoke_rows) == 0 and not args.smoke


def _stage_banked_dir(args) -> Path:
    return args.out_root / "stage_banked"


def _census_dir(args) -> Path:
    return args.out_root / "census"


def _eval_dir(args) -> Path:
    return args.out_root / "eval"


def _gates_dir(args) -> Path:
    return args.out_root / "gates"


def _write_json_atomic(path: Path, obj: dict) -> None:
    """Plain atomic JSON write (gate records + regime manifests; provenance-bearing
    phase docs go through the parent's _write_json)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{path.name}"
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def _record_gate(args, key: str, record: dict) -> None:
    _write_json_atomic(_gates_dir(args) / f"{key}.json", record)


def _sentinel(name: str, note: str, extra: dict | None = None) -> None:
    """Non-blocking phase sentinel through the parent's poller-conformant writer."""
    payload = {"blocks_pipeline": False}
    if extra:
        payload.update(extra)
    try:
        _drv().C.write_sentinel(f"phase-{name}", note, task_id=TASK_ID, extra=payload)
    except OSError as e:
        logger.warning("[sentinel] phase-%s write failed: %s", name, e)


def _parent_args(args) -> SimpleNamespace:
    """The parent driver's full args contract (every field its _regime /
    _enter_phase_regime / phase_assemble / _load_scratch_meta / _stage_m_split
    read), threaded from the sweep's CLI. Fit/train dials are pinned to the
    parent's production defaults — no fit runs in this round."""
    return SimpleNamespace(
        phase="assemble",
        out_root=args.out_root,
        hf_prefix=str(args.hf_prefix),
        smoke=bool(args.smoke),
        tiny_model=False,
        max_chunks=int(args.max_chunks),
        smoke_rows=int(args.smoke_rows),
        gen_batch=16,
        device=str(args.device),
        sae_dir=args.sae_dir,
        fresh_stream=bool(args.fresh_stream),
        skip_upload=bool(args.skip_upload),
        gpu_id=int(args.gpu_id),
        sae_steps=0,
        sae_dict=0,
        sae_k=0,  # production SAE_K=100 (the parent regime now hashes the k budget)
        n_perm=int(args.n_perm),
        n_boot=int(args.n_boot),
        fit_n=0,
        g2a_probe_rows=0,
        import_check=False,
        resume_across_code_sha=bool(args.resume_across_code_sha),
    )


# ── regime-keyed resume (sweep-owned phases; parent convention) ──────────────────


def _regime(args) -> dict:
    """Regime manifest for the sweep-owned phases: floors + tolerances + revision
    pin + every output/destination-affecting dial. GENERATING PARAMETERS only."""
    from explore_persona_space.orchestrate.provenance import git_provenance

    base = {
        "smoke": bool(args.smoke),
        "max_chunks": int(args.max_chunks),
        "smoke_rows": int(args.smoke_rows),
        "device": str(args.device),
        "n_perm": int(args.n_perm),
        "n_boot": int(args.n_boot),
        "hf_prefix": str(args.hf_prefix),
        "skip_upload": bool(args.skip_upload),
        "floors_c": list(FLOORS_C),
        "floors_b": list(FLOORS_B),
        "data_repo_revision": DATA_REPO_REVISION,
        "gc_count_tol": GC_COUNT_TOL,
        "gr_perfeature_tol": GR_PERFEATURE_TOL,
        "gr_median_tol": GR_MEDIAN_TOL,
    }
    cfg_hash = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()[:16]
    prov = git_provenance()
    code_sha = prov.commit_sha_full or prov.commit_sha or "unknown"
    return {**base, "config_hash": cfg_hash, "code_sha": code_sha}


def _enter_regime(out_dir: Path, args, phase: str, stale_paths=()) -> tuple[dict, bool]:
    """Write/verify the phase regime manifest (parent _enter_phase_regime
    semantics: config mismatch refuses; code-SHA-only mismatch wipes stale
    outputs BEFORE the manifest write and recomputes loudly)."""
    regime = _regime(args)
    path = out_dir / "regime.json"
    if path.exists():
        prev = json.loads(path.read_text())
        if prev.get("config_hash") != regime["config_hash"]:
            raise RuntimeError(
                f"[{phase}] out-root {out_dir} holds a run under a DIFFERENT regime "
                f"(config_hash {prev.get('config_hash')} != {regime['config_hash']}); "
                "use a fresh --out-root (never silently mix regimes)"
            )
        if prev.get("code_sha") != regime["code_sha"]:
            if args.resume_across_code_sha:
                logger.warning(
                    "[%s] code SHA changed but --resume-across-code-sha set: outputs RETAINED",
                    phase,
                )
                _write_json_atomic(path, regime)
                return regime, True
            logger.warning(
                "[%s] code SHA changed (%s -> %s): outputs RECOMPUTED, never skipped",
                phase,
                str(prev.get("code_sha"))[:12],
                regime["code_sha"][:12],
            )
            for p in stale_paths:
                if p.exists():
                    logger.warning("[%s] recompute: removing stale %s", phase, p.name)
                    p.unlink()
            _write_json_atomic(path, regime)
            return regime, False
        return regime, True
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in stale_paths:
        if p.exists():
            logger.warning("[%s] fresh manifest: removing stale %s", phase, p.name)
            p.unlink()
    _write_json_atomic(path, regime)
    return regime, False


# ── gates (pure comparison logic — pinned by tests/test_issue2476_gates.py) ──────


def _gate_counts(
    recomputed: np.ndarray,
    banked: np.ndarray,
    floors,
    *,
    arm: str,
    production: bool,
    tol: int = GC_COUNT_TOL,
    out_path: Path | None = None,
    extra: dict | None = None,
) -> dict:
    """G-C (plan §7): recomputed full-width fit-side counts vs the banked census —
    max per-feature |delta| <= tol AND every per-floor alive-set symmetric
    difference confined to features whose banked count sits within +/-tol of that
    floor. FAIL => record written FIRST, then sys.exit(RC_GC) at production;
    informational (no exit) at smoke n. Banked counts stay CANONICAL for panel
    definition either way (plan §11). ``extra``: caller-supplied diagnostic
    fields merged into the record BEFORE the write (k200 r8 M2: the k200 caller
    records float(sae.threshold) so a near-theta boundary flip is one-look
    diagnosable from the persisted gc_ii record — FAIL path included)."""
    rec = np.asarray(recomputed, np.int64)
    ban = np.asarray(banked, np.int64)
    assert rec.shape == ban.shape, (rec.shape, ban.shape)
    delta = rec - ban
    max_abs = int(np.abs(delta).max()) if len(delta) else 0
    worst = np.argsort(-np.abs(delta))[:5]
    per_floor: dict = {}
    offenders: list[int] = []
    for f in floors:
        f = int(f)
        sym = np.where((ban >= f) != (rec >= f))[0]
        bad = sym[np.abs(ban[sym] - f) > tol]
        offenders.extend(int(i) for i in bad[:5])
        per_floor[str(f)] = {
            "n_sym_diff": int(len(sym)),
            "n_off_boundary": int(len(bad)),
        }
    ok = max_abs <= tol and not offenders
    record = {
        "gate": "G-C",
        "arm": arm,
        "tol_rows": int(tol),
        "max_abs_delta": max_abs,
        "worst_features": [
            {"feat": int(i), "banked": int(ban[i]), "recomputed": int(rec[i])} for i in worst
        ],
        "per_floor_sym_diff": per_floor,
        "off_boundary_examples": offenders,
        "verdict": ("PASS" if ok else "FAIL") if production else "INFORMATIONAL-smoke",
        "production": bool(production),
    }
    if extra:
        # k200 r10 NIT: ``extra`` is caller diagnostics ONLY — it must never
        # clobber a core record field (verdict/production/tol_rows/...), which
        # would silently rewrite the gate's persisted semantics.
        overlap = sorted(set(extra) & set(record))
        assert not overlap, f"_gate_counts extra clobbers reserved record keys: {overlap}"
        record.update(extra)
    if out_path is not None:
        _write_json_atomic(out_path, record)
    print(
        f"[gate] G-C arm={arm} max|delta|={max_abs} verdict={record['verdict']}",
        flush=True,
    )
    if production and not ok:
        logger.error("[gate] G-C FAIL (%s): %s", arm, json.dumps(record["per_floor_sym_diff"]))
        sys.exit(RC_GC)
    return record


def _gate_repro(
    got_r2: np.ndarray,
    want_r2: np.ndarray,
    got_medians: dict[str, float | None],
    want_medians: dict[str, float | None],
    *,
    arm: str,
    production: bool,
    feat_ids: np.ndarray,
    tier: np.ndarray,
    fit_counts: np.ndarray,
    perfeature_tol: float = GR_PERFEATURE_TOL,
    max_tol: float = GR_MAX_TOL,
    share_min: float = GR_SHARE_MIN,
    median_tol: float = GR_MEDIAN_TOL,
    median_min_tier_n: int = GR_MEDIAN_MIN_TIER_N,
    out_path: Path | None = None,
) -> dict:
    """G-R (plan §4 Q4 / §7; quantile predicate, amended v7 — plan §11 "G-R
    tolerances (amended v7)"): at the registered 1% floor the sweep must
    REPRODUCE the parent's committed per-feature R2 under (a) finiteness
    pattern EQUAL, (b) >= share_min of finite panel features within
    |delta| <= perfeature_tol, (c) max finite |delta| <= max_tol, and (d)
    committed per-tier medians within median_tol on tiers with
    n >= median_min_tier_n panel features ONLY — a smaller tier contributes NO
    median conjunct and instead has its per-feature rows recorded verbatim
    under small_tier_deltas (an n=3 median IS a single feature). EVERY feature
    over perfeature_tol is recorded verbatim in `violators` (feat id,
    committed, recomputed, delta, banked fit count — the G-C counts array).
    Reference values are READ from the committed artifacts by the caller,
    never retyped. FAIL => record written FIRST, then sys.exit(RC_GR) at
    production; informational at smoke n (subset score rows cannot
    reproduce)."""
    got = np.asarray(got_r2, np.float64)
    want = np.asarray(want_r2, np.float64)
    ids = np.asarray(feat_ids, np.int64)
    tiers = np.asarray(tier, np.int64)
    counts = np.asarray(fit_counts, np.int64)
    assert got.shape == want.shape == ids.shape == tiers.shape == counts.shape, (
        got.shape,
        want.shape,
        ids.shape,
        tiers.shape,
        counts.shape,
    )
    fin_got, fin_want = np.isfinite(got), np.isfinite(want)
    finiteness_ok = bool((fin_got == fin_want).all())
    fin = fin_got & fin_want
    n_fin = int(fin.sum())
    delta = np.abs(got - want)
    max_abs = float(delta[fin].max()) if n_fin else 0.0
    share = float((fin & (delta <= perfeature_tol)).sum() / n_fin) if n_fin else 1.0
    viol_idx = np.where(fin & (delta > perfeature_tol))[0]
    viol_idx = viol_idx[np.argsort(-delta[viol_idx])]
    violators = [
        {
            "feat": int(ids[i]),
            "committed": float(want[i]),
            "recomputed": float(got[i]),
            "delta": float(delta[i]),
            "fit_count": int(counts[i]),
        }
        for i in viol_idx
    ]
    tier_panel_n = {str(t): int((tiers == t).sum()) for t in (0, 1, 2)}
    median_exempt_tiers = sorted(
        t for t in (0, 1, 2) if tier_panel_n[str(t)] < int(median_min_tier_n)
    )
    med_deltas: dict[str, float | None] = {}
    med_ok = True
    for k, want_v in want_medians.items():
        got_v = got_medians.get(k)
        t = int(k.rsplit("/t", 1)[1])  # caller convention: keys are "{read}/t{tier}"
        d = None if (want_v is None or got_v is None) else abs(float(got_v) - float(want_v))
        med_deltas[k] = d
        if t in median_exempt_tiers:
            continue  # v7: n < median_min_tier_n => NO median conjunct for this tier
        if d is None:
            med_ok = med_ok and (want_v is None) == (got_v is None)
            continue
        med_ok = med_ok and d <= median_tol
    small_tier_deltas = {
        str(t): [
            {
                "feat": int(ids[i]),
                "committed": float(want[i]) if fin_want[i] else None,
                "recomputed": float(got[i]) if fin_got[i] else None,
                "delta": float(delta[i]) if fin[i] else None,
            }
            for i in np.where(tiers == t)[0]
        ]
        for t in median_exempt_tiers
    }
    ok = finiteness_ok and share >= share_min and max_abs <= max_tol and med_ok
    record = {
        "gate": "G-R",
        "arm": arm,
        "predicate": "v7-quantile",
        "n_features": int(len(got)),
        "n_finite": n_fin,
        "finiteness_pattern_equal": finiteness_ok,
        "max_abs_delta_r2": max_abs,
        "share_within_2e3": share,
        "n_violators": len(violators),
        "violators": violators,
        "perfeature_tol": float(perfeature_tol),
        "max_tol": float(max_tol),
        "share_min": float(share_min),
        "median_tol": float(median_tol),
        "median_min_tier_n": int(median_min_tier_n),
        "tier_panel_n": tier_panel_n,
        "median_exempt_tiers": median_exempt_tiers,
        "small_tier_deltas": small_tier_deltas,
        "median_deltas": med_deltas,
        "got_medians": got_medians,
        "committed_medians": want_medians,
        "verdict": ("PASS" if ok else "FAIL") if production else "INFORMATIONAL-smoke",
        "production": bool(production),
    }
    if out_path is not None:
        _write_json_atomic(out_path, record)
    print(
        f"[gate] G-R arm={arm} share={share:.5f} n_viol={len(violators)} "
        f"max|dR2|={max_abs:.2e} med_ok={med_ok} verdict={record['verdict']}",
        flush=True,
    )
    if production and not ok:
        logger.error(
            "[gate] G-R FAIL (%s): the sweep is not measuring the parent's instrument", arm
        )
        sys.exit(RC_GR)
    return record


def _undefined_demotion(r2_map: np.ndarray, r2_ib: np.ndarray, tier: np.ndarray) -> dict:
    """Plan §12 A13 + plan-approval [stat-rec-2]: per-tier undefined-R2 census on
    the score rows. Demoted (not_evaluable_census_only=True) when ANY populated
    tier loses >50% of its alive features to undefined MAP-read R2 (the lattice
    input); identity+bias undefined counts are recorded alongside."""
    per_tier: dict = {}
    demoted = False
    for t in (0, 1, 2):
        m = tier == t
        n = int(m.sum())
        n_undef = int((~np.isfinite(np.asarray(r2_map)[m])).sum()) if n else 0
        n_undef_ib = int((~np.isfinite(np.asarray(r2_ib)[m])).sum()) if n else 0
        per_tier[str(t)] = {
            "n_alive": n,
            "n_undefined_r2_map": n_undef,
            "n_undefined_r2_ib": n_undef_ib,
            "frac_undefined_map": (n_undef / n) if n else None,
        }
        if n > 0 and n_undef > 0.5 * n:
            demoted = True
    return {
        "per_tier": per_tier,
        "not_evaluable_census_only": demoted,
        "rule": (
            "plan §12 A13 + plan-approval stat-rec-2: >50% of a populated tier's alive "
            "features undefined-R2 (map read) on score rows => census-only at this floor "
            "(no lattice label; the boolean replaces it)"
        ),
    }


def _finish_floor_row(stats: dict, demotion: dict, *, floor: int, n_fit: int, registered: bool):
    """Assemble one per-floor row from a _tier_stats result + the A13 demotion doc.
    [stat-rec-2]: a demoted floor carries not_evaluable_census_only=True and NO
    lattice label (the parent's lattice_verdict key is stripped); lattice_inputs
    stay persisted as data either way."""
    row = {k: v for k, v in stats.items() if k != "lattice_verdict"}
    row["floor_rows"] = int(floor)
    row["floor_frac_of_fit_rows"] = float(floor) / float(n_fit)
    row["registered_cell"] = bool(registered)
    row["undefined_r2"] = demotion["per_tier"]
    row["not_evaluable_census_only"] = bool(demotion["not_evaluable_census_only"])
    row["demotion_rule"] = demotion["rule"]
    if not demotion["not_evaluable_census_only"]:
        row["lattice_reported"] = stats["lattice_verdict"]
        row["lattice_note"] = LATTICE_NOTE + (
            "; this IS the registered 1% cell" if registered else ""
        )
    return row


# ── census kernel (the plan §4 Q3(a) ~5-line extension of the parent's loop) ─────


def _encode_counts_sums(sae, mm, positions: np.ndarray, chunk: int = 4096, tag: str = ""):
    """The parent _encode_counts streaming loop VERBATIM + a per-feature SUM
    accumulator (full-width counts AND sums in one pass; sums/n = the train-mean
    null for newly-alive features). Counts on TRUE-summary encodes only."""
    import torch

    with torch.no_grad():
        counts = torch.zeros(sae.dict_size, dtype=torch.int64, device=sae.device)
        sums = torch.zeros(sae.dict_size, dtype=torch.float64, device=sae.device)
        pos = np.sort(np.asarray(positions, np.int64))
        t0 = time.time()
        n_chunks = max(1, (len(pos) + chunk - 1) // chunk)
        for i, s in enumerate(range(0, len(pos), chunk)):
            x = torch.as_tensor(np.asarray(mm[pos[s : s + chunk]], np.float32), device=sae.device)
            f = sae.encode(x, chunk=chunk)
            counts += (f > 0).sum(0)
            sums += f.to(torch.float64).sum(0)
            if (i + 1) % 10 == 0 or i + 1 == n_chunks:
                print(
                    f"[fs_census] counts{tag} chunk {i + 1}/{n_chunks} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
        return counts.cpu().numpy(), sums.cpu().numpy()


# ── Q1: assemble (parent phase VERBATIM) ─────────────────────────────────────────


def phase_assemble(args) -> None:
    """Q1: the parent's phase_assemble on the sweep's out-root (X19/Y19 fp16
    memmaps re-assembled from the 1,920 pinned chunks; split shas re-asserted;
    realized counts reconciled — the parent card's regen recipe verbatim)."""
    drv = _drv()
    drv.phase_assemble(_parent_args(args))


# ── Q2: stage banked inputs + gate G-A ───────────────────────────────────────────


def _stage_banked_files(stage: Path) -> None:
    """Idempotent revision-pinned staging (the perrow_views _stage pattern)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    for f in STAGE_FILES:
        tgt = stage / f
        if tgt.exists() and tgt.stat().st_size > 0:
            continue
        hub.retry_transient(
            lambda f=f: hf_hub_download(
                DATA_REPO, f, repo_type="dataset", revision=DATA_REPO_REVISION, local_dir=str(stage)
            ),
            what="hf_hub_download",
        )
        print(f"[fs_stage_banked] staged {f}", flush=True)


def _assert_npz_keys(z, required: tuple[str, ...], what: str) -> None:
    missing = [k for k in required if k not in z.files]
    assert not missing, f"{what}: missing keys {missing} (have {sorted(z.files)})"


def phase_stage_banked(args) -> None:
    """Q2: stage every banked input at revision DATA_REPO_REVISION, assert the
    observed schemas (plan §12 A2/A4/A7/A8), then gate G-A (row alignment):
    refit holdout rows == ib_c rows == the assembled holdout order (EA order);
    armb row_idx_score == the m-split score rows; store row_idx == the m-split
    row universe. Mismatch => sys.exit(RC_GA) — banked arrays are
    production-grain, so G-A binds at smoke too."""
    drv = _drv()
    drv.C.phase("fs_stage_banked")
    sent_dir = args.out_root / "sentinels"
    sent_dir.mkdir(parents=True, exist_ok=True)
    done_path = sent_dir / "stage_banked.done.json"
    regime, resume_ok = _enter_regime(sent_dir, args, "fs_stage_banked", stale_paths=[done_path])
    if resume_ok and done_path.exists():
        logger.info("[fs_stage_banked] resume: done-file present under matching regime; skip")
        return
    assert drv.C.HF_DATA_REPO == DATA_REPO, (drv.C.HF_DATA_REPO, DATA_REPO)
    stage = _stage_banked_dir(args)
    stage.mkdir(parents=True, exist_ok=True)
    _stage_banked_files(stage)

    e = stage / TA / "eval"
    az_c = np.load(e / "alive_c.npz")
    _assert_npz_keys(
        az_c, ("alive_ids", "counts", "floor", "n_fit_rows", "train_mean", "tier"), "alive_c"
    )
    assert az_c["counts"].shape == (65536,), az_c["counts"].shape
    assert int(az_c["n_fit_rows"]) > 0 and int(az_c["floor"]) >= 1
    az_b = np.load(e / "alive_b.npz")
    _assert_npz_keys(
        az_b, ("alive_ids", "counts", "floor", "n_fit_rows", "train_mean", "tier"), "alive_b"
    )
    assert az_b["counts"].shape == (65536,), az_b["counts"].shape
    ibz = np.load(e / "ib_c.npz")
    _assert_npz_keys(ibz, ("rows", "pred16"), "ib_c")
    bz = np.load(e / "armb_maps.npz")
    _assert_npz_keys(
        bz, ("pred16", "ib_pred16", "row_idx_score", "row_idx_fit", "pred16_inlier"), "armb_maps"
    )
    fz = np.load(e / "ftrue_b.npz")
    _assert_npz_keys(fz, ("row_idx", "f_true"), "ftrue_b")
    hz = np.load(stage / REFIT_HF)
    _assert_npz_keys(hz, ("holdout_pred16", "holdout_rows"), "refit_holdout")
    store = np.load(stage / TA / "recapture_store" / "vbar_store.npz")
    _assert_npz_keys(store, ("row_idx", "vbar20"), "vbar_store")
    # A7: the banked f_true file is RESTRICTED to the 879 registered panel — the
    # reason union re-encodes exist (a full-width file would UPGRADE the plan).
    ftall = np.load(e / "ftrue_c_all.fp16.npy", mmap_mode="r")
    assert ftall.shape[1] == len(az_c["alive_ids"]), (ftall.shape, len(az_c["alive_ids"]))
    cfg = json.loads((stage / TA / "sae_c" / "cfg.json").read_text())
    assert int(cfg["dict_size"]) == 65536 and int(cfg["k"]) == 100, cfg

    # ── G-A: row alignment ────────────────────────────────────────────────────────
    pargs = _parent_args(args)
    drv._load_scratch_meta(pargs)  # stages split_indices.npz + prov.npy (sha-asserted)
    idx = np.load(args.out_root / "stage" / "split_indices.npz")
    hold_assembled = np.asarray(idx["holdout"], np.int64)  # EA holdout ORDER (as stored)
    hold_rows = np.asarray(hz["holdout_rows"], np.int64)
    ib_rows = np.asarray(ibz["rows"], np.int64)
    s_fit, s_score = drv._stage_m_split(pargs)  # sha-asserted vs committed m_split.json
    row_idx_score = np.asarray(bz["row_idx_score"], np.int64)
    row_idx_fit = np.asarray(bz["row_idx_fit"], np.int64)
    ridx_store = np.asarray(store["row_idx"], np.int64)
    checks = {
        "refit_rows_eq_ib_rows": bool(np.array_equal(hold_rows, ib_rows)),
        "refit_rows_eq_assembled_holdout": bool(np.array_equal(hold_rows, hold_assembled)),
        "armb_score_rows_eq_m_split": bool(
            np.array_equal(np.sort(row_idx_score), np.sort(np.asarray(s_score, np.int64)))
        ),
        "armb_fit_rows_subset_m_fit": bool(np.isin(row_idx_fit, np.asarray(s_fit, np.int64)).all()),
        "store_rows_eq_m_universe": bool(
            np.array_equal(
                np.sort(ridx_store),
                np.sort(np.concatenate([np.asarray(s_fit), np.asarray(s_score)])),
            )
        ),
        "armb_n_fit_eq_alive_b_n_fit": bool(len(row_idx_fit) == int(az_b["n_fit_rows"])),
    }
    ok = all(checks.values())
    record = {"gate": "G-A", "checks": checks, "verdict": "PASS" if ok else "FAIL"}
    _record_gate(args, "ga", record)
    print(f"[gate] G-A verdict={record['verdict']} {json.dumps(checks)}", flush=True)
    if not ok:
        logger.error("[gate] G-A row-alignment FAIL: %s", json.dumps(checks))
        sys.exit(RC_GA)

    _write_json_atomic(
        done_path,
        {"regime": regime, "staged": list(STAGE_FILES), "ga": record["verdict"]},
    )
    _sentinel("fs-stage-banked", f"Q2 done ({len(STAGE_FILES)} staged inputs; G-A PASS)")
    logger.info("[fs_stage_banked] done")


# ── Q3: census passes + union encodes ────────────────────────────────────────────


def _local_positions(rows_present: np.ndarray, global_ids: np.ndarray, production: bool):
    """Map global row ids -> positions in the assembled memmap. Production
    rows_present is arange (identity); smoke slices keep only present ids."""
    ids = np.asarray(global_ids, np.int64)
    if production:
        return ids, np.ones(len(ids), bool)
    sel = np.isin(ids, rows_present)
    pos = np.searchsorted(rows_present, ids[sel])
    assert (rows_present[pos] == ids[sel]).all()
    return pos, sel


def _tier_quantiles(counts: np.ndarray, n_fit: int, drv) -> dict:
    """Per-tier firing-fraction quantile grid (census instrument; plan §6.5)."""
    bounds = (0,) + tuple(drv.S.MATRYOSHKA_TIER_BOUNDS)
    qs = np.asarray([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    out = {"q_grid": qs.tolist()}
    frac = np.asarray(counts, np.float64) / max(1, n_fit)
    for t in (0, 1, 2):
        out[f"t{t}"] = np.quantile(frac[bounds[t] : bounds[t + 1]], qs).tolist()
    return out


def phase_census(args) -> None:
    """Q3: (a) full-width fit-side counts+sums (SAE-c on Y19; chanind lmsys AND
    pile on the banked vbar20) -> firing_census_*.npz + gate G-C vs the banked
    counts (+/-3; banked stays canonical; pile has no banked reference —
    recomputed canonical there, reported as such); gate G-S (assembled split
    shas == banked split_meta records). (b) score-side restricted encodes at
    the UNION-alive columns (encoded ONCE at full TopK width, then subset per
    floor — value-identical shared columns)."""
    drv = _drv()
    drv.C.phase("fs_census")
    out = _census_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    finals = [
        out / n
        for n in (
            "firing_census_c.npz",
            "firing_census_b.npz",
            "firing_census_b_pile.npz",
            "union_encodes_c.npz",
            "union_encodes_b.npz",
            "union_encodes_b_pile.npz",
            "union_encodes_meta.json",
        )
    ]
    regime, resume_ok = _enter_regime(out, args, "fs_census", stale_paths=finals)
    if resume_ok and all(p.exists() for p in finals):
        logger.info("[fs_census] resume: outputs present under matching regime; skip")
        return
    drv.EA._headroom(args.out_root, 2 if args.smoke else 20, "fs-census")
    production = _production(args)
    pargs = _parent_args(args)
    stage = _stage_banked_dir(args)
    e = stage / TA / "eval"
    a_dir = args.out_root / "assemble"

    # ── G-S: assembled split shas vs the banked split_meta records ────────────────
    assembled = json.loads((a_dir / "split_meta.json").read_text())
    banked_meta = json.loads((stage / TA / "split_meta" / "split_meta.json").read_text())
    gs_ok = assembled["shas"] == banked_meta["shas"]
    gs = {
        "gate": "G-S",
        "assembled_shas": assembled["shas"],
        "banked_shas": banked_meta["shas"],
        "verdict": "PASS" if gs_ok else "FAIL",
    }
    _record_gate(args, "gs", gs)
    print(f"[gate] G-S verdict={gs['verdict']}", flush=True)
    if not gs_ok:
        logger.error("[gate] G-S split-sha FAIL")
        sys.exit(RC_GS)

    _row_ci, prov_u8, pools = drv._load_scratch_meta(pargs)
    rows_present = np.load(a_dir / "rows_present.npy")
    y_mm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")
    timings: dict = {}

    # ── fresh arm: SAE-c counts+sums over the sae_fit rows on Y19 ─────────────────
    sae_c = drv.MatryoshkaBatchTopKSAE.load_local(stage / TA / "sae_c", device=args.device)
    az_c = np.load(e / "alive_c.npz")
    counts_banked_c = np.asarray(az_c["counts"], np.int64)
    sae_fit = np.sort(np.asarray(pools["sae_fit"], np.int64))
    fit_pos_c, _ = _local_positions(rows_present, sae_fit, production)
    n_fit_used_c = int(len(fit_pos_c))
    assert n_fit_used_c >= 2, "census needs >=2 fresh fit rows (smoke slice too small)"
    t0 = time.time()
    counts_c, sums_c = _encode_counts_sums(sae_c, y_mm, fit_pos_c, tag=" c")
    timings["counts_c_s"] = round(time.time() - t0, 1)
    floors_eff_c = [max(1, min(f, n_fit_used_c)) for f in FLOORS_C]
    _gate_counts(
        counts_c,
        counts_banked_c,
        floors_eff_c if not production else FLOORS_C,
        arm="c",
        production=production,
        out_path=_gates_dir(args) / "gc_c.json",
    )
    np.savez(
        out / "firing_census_c.npz",
        counts=counts_c.astype(np.int64),
        sums=sums_c.astype(np.float64),
        counts_banked=counts_banked_c,
        n_fit_rows=np.int64(n_fit_used_c),
        floors=np.asarray(FLOORS_C, np.int64),
        floors_eff=np.asarray(floors_eff_c, np.int64),
        **{
            f"quantiles_{k}": np.asarray(v)
            for k, v in _tier_quantiles(counts_c, n_fit_used_c, drv).items()
        },
    )

    # fresh union encodes (union from the BANKED counts — canonical panel source)
    hz = np.load(stage / REFIT_HF)
    ibz = np.load(e / "ib_c.npz")
    hold_rows = np.asarray(hz["holdout_rows"], np.int64)
    local_hold, sel = _local_positions(rows_present, hold_rows, production)
    n_te_c = int(len(local_hold))
    assert n_te_c >= 2, "census needs >=2 fresh holdout rows in the assembled slice"
    union_c = np.where(counts_banked_c >= min(FLOORS_C))[0]
    vhat = np.asarray(hz["holdout_pred16"], np.float16)[sel]
    ib16 = np.asarray(ibz["pred16"], np.float16)[sel]
    t0 = time.time()
    f_true_c = drv._encode_restricted(sae_c, y_mm, local_hold, union_c)
    f_pred_c = drv._encode_restricted(sae_c, vhat, np.arange(len(vhat)), union_c)
    f_ib_c = drv._encode_restricted(sae_c, ib16, np.arange(len(ib16)), union_c)
    timings["encode_c_s"] = round(time.time() - t0, 1)
    timings["encode_c_s_per_row"] = round((time.time() - t0) / max(1, 3 * n_te_c), 5)
    np.savez(
        out / "union_encodes_c.npz",
        rows=hold_rows[sel],
        cols=union_c,
        f_true=f_true_c,
        f_pred=f_pred_c,
        f_ib=f_ib_c,
        te_prov=np.asarray(prov_u8, np.uint8)[hold_rows[sel]],
    )
    del sae_c, f_true_c, f_pred_c, f_ib_c
    print(f"[fs_census] unit c done n_fit={n_fit_used_c} n_te={n_te_c}", flush=True)

    # ── bridge arm: chanind lmsys + pile on the banked vbar20 store ───────────────
    store = np.load(stage / TA / "recapture_store" / "vbar_store.npz")
    vbar20 = np.asarray(store["vbar20"], np.float16)
    ridx_store = np.asarray(store["row_idx"], np.int64)
    bz = np.load(e / "armb_maps.npz")
    az_b = np.load(e / "alive_b.npz")
    counts_banked_b = np.asarray(az_b["counts"], np.int64)
    fit_b = np.asarray(bz["row_idx_fit"], np.int64)
    score_b = np.asarray(bz["row_idx_score"], np.int64)
    if not production:
        fit_b = fit_b[: min(SMOKE_B_FIT, len(fit_b))]
        score_b = score_b[: min(SMOKE_B_SCORE, len(score_b))]
        assert len(fit_b) >= 2 and len(score_b) >= 2
    fit_store_pos = np.searchsorted(ridx_store, fit_b)
    assert (ridx_store[fit_store_pos] == fit_b).all(), "store fit-row drift"
    te_store_pos = np.searchsorted(ridx_store, score_b)
    assert (ridx_store[te_store_pos] == score_b).all(), "store score-row drift"
    n_fit_used_b = int(len(fit_store_pos))

    sae_lm = drv.S.SAELensJumpReLU.load(
        drv.M.SAE_IDS["lmsys"], device=args.device, cache_dir=args.sae_dir
    )
    t0 = time.time()
    counts_b, sums_b = _encode_counts_sums(sae_lm, vbar20, fit_store_pos, tag=" b")
    timings["counts_b_s"] = round(time.time() - t0, 1)
    floors_eff_b = [max(1, min(f, n_fit_used_b)) for f in FLOORS_B]
    _gate_counts(
        counts_b,
        counts_banked_b,
        floors_eff_b if not production else FLOORS_B,
        arm="b",
        production=production,
        out_path=_gates_dir(args) / "gc_b.json",
    )
    np.savez(
        out / "firing_census_b.npz",
        counts=counts_b.astype(np.int64),
        sums=sums_b.astype(np.float64),
        counts_banked=counts_banked_b,
        n_fit_rows=np.int64(n_fit_used_b),
        floors=np.asarray(FLOORS_B, np.int64),
        floors_eff=np.asarray(floors_eff_b, np.int64),
        **{
            f"quantiles_{k}": np.asarray(v)
            for k, v in _tier_quantiles(counts_b, n_fit_used_b, drv).items()
        },
    )
    union_b = np.where(counts_banked_b >= min(FLOORS_B))[0]
    pred16 = np.asarray(bz["pred16"], np.float16)
    ib_pred16 = np.asarray(bz["ib_pred16"], np.float16)
    if not production:
        pred16, ib_pred16 = pred16[: len(score_b)], ib_pred16[: len(score_b)]
    f_true_b = drv._encode_restricted(sae_lm, vbar20, te_store_pos, union_b)
    f_pred_b = drv._encode_restricted(sae_lm, pred16, np.arange(len(pred16)), union_b)
    f_ib_b = drv._encode_restricted(sae_lm, ib_pred16, np.arange(len(ib_pred16)), union_b)
    np.savez(
        out / "union_encodes_b.npz",
        rows=score_b,
        cols=union_b,
        f_true=f_true_b,
        f_pred=f_pred_b,
        f_ib=f_ib_b,
    )
    del sae_lm, f_true_b, f_pred_b, f_ib_b
    print(f"[fs_census] unit b done n_fit={n_fit_used_b} n_te={len(score_b)}", flush=True)

    # pile twin (exploratory; recomputed counts CANONICAL — no banked reference)
    sae_pile = drv.S.SAELensJumpReLU.load(
        drv.M.SAE_IDS["pile"], device=args.device, cache_dir=args.sae_dir
    )
    t0 = time.time()
    counts_p, sums_p = _encode_counts_sums(sae_pile, vbar20, fit_store_pos, tag=" b_pile")
    timings["counts_b_pile_s"] = round(time.time() - t0, 1)
    np.savez(
        out / "firing_census_b_pile.npz",
        counts=counts_p.astype(np.int64),
        sums=sums_p.astype(np.float64),
        n_fit_rows=np.int64(n_fit_used_b),
        floors=np.asarray(FLOORS_B, np.int64),
        floors_eff=np.asarray(floors_eff_b, np.int64),
        counts_canonical_note="no banked reference: recomputed counts canonical (plan §4 Q3a)",
        **{
            f"quantiles_{k}": np.asarray(v)
            for k, v in _tier_quantiles(counts_p, n_fit_used_b, drv).items()
        },
    )
    union_p = np.where(counts_p >= min(floors_eff_b))[0]
    if len(union_p) == 0:
        logger.warning("[fs_census] pile union EMPTY at min floor (recorded; census-only)")
    f_true_p = drv._encode_restricted(sae_pile, vbar20, te_store_pos, union_p)
    f_pred_p = drv._encode_restricted(sae_pile, pred16, np.arange(len(pred16)), union_p)
    np.savez(
        out / "union_encodes_b_pile.npz",
        rows=score_b,
        cols=union_p,
        f_true=f_true_p,
        f_pred=f_pred_p,
    )
    del sae_pile, f_true_p, f_pred_p
    print(f"[fs_census] unit b_pile done n_union={len(union_p)}", flush=True)

    drv._write_json(
        out / "union_encodes_meta.json",
        {
            "regime": regime,
            "timings_s": timings,
            "n_fit_used": {"c": n_fit_used_c, "b": n_fit_used_b},
            "n_te_used": {"c": n_te_c, "b": int(len(score_b))},
            "n_union": {
                "c": int(len(union_c)),
                "b": int(len(union_b)),
                "b_pile": int(len(union_p)),
            },
            "floors_eff": {"c": floors_eff_c, "b": floors_eff_b},
            "smoke_clamped": not production,
        },
        phase="fs-census",
    )
    _sentinel(
        "fs-census",
        f"Q3 done (union c={len(union_c)} b={len(union_b)} pile={len(union_p)}; G-C/G-S recorded)",
    )
    logger.info("[fs_census] done")


# ── Q4: per-floor stats + retrieval ──────────────────────────────────────────────


def _arm_sweep(
    drv,
    args,
    *,
    tag: str,
    floors,
    registered_floor: int,
    registered_ids: np.ndarray,
    counts_banked: np.ndarray,
    n_fit_banked: int,
    union_ids: np.ndarray,
    f_true: np.ndarray,
    f_pred: np.ndarray,
    f_ib: np.ndarray,
    te_prov: np.ndarray | None,
    train_mean_full: np.ndarray,
    battery_seed: int,
    committed_pf: Path,
    committed_tests: Path,
    n_perm: int,
    n_boot: int,
    production: bool,
) -> tuple[dict, dict, dict]:
    """One arm's floor sweep: per-feature reads computed ONCE at the union grain
    (R2 map/ib/train-mean + K=20 shuffle nulls — per-feature quantities, so a
    union-grain draw subset per floor is EXACT), then per floor: banked-count
    alive set == the registered panel machinery's output (asserted), the parent
    _tier_stats battery with a FRESH rng(battery_seed) (identical draws across
    floors by design), A13 demotion, retrieval cells, and — at the registered
    floor — gate G-R vs the committed references. Returns (sweep_doc,
    retrieval_doc, perfeature_union_arrays)."""
    r2_map_u = drv._r2_only(f_pred, f_true)
    r2_ib_u = drv._r2_only(f_ib, f_true)
    tm_u = np.asarray(train_mean_full, np.float64)[union_ids]
    r2_tm_u = drv._r2_only(np.broadcast_to(tm_u.astype(np.float32), f_true.shape), f_true)
    null_map = drv._shuffle_null_r2(f_pred, f_true, drv.SHUFFLE_SEEDS_2476, what=f" {tag}/map")
    null_ib = drv._shuffle_null_r2(f_ib, f_true, drv.SHUFFLE_SEEDS_2476, what=f" {tag}/ib")
    corpus_r2: dict[str, np.ndarray] = {}
    corpus_n: dict[str, int] = {}
    if te_prov is not None:
        for label, code in (("lmsys", 0), ("wildchat", 1)):
            m = np.asarray(te_prov) == code
            corpus_n[label] = int(m.sum())
            if int(m.sum()) >= 2:
                corpus_r2[label] = drv._r2_only(f_pred[m], f_true[m])
    tier_u = drv.S.tier_of(union_ids)

    rows = []
    retr_rows: dict = {}
    lattice_vector = []
    masks: dict[str, np.ndarray] = {}
    for fl in floors:
        t0 = time.time()
        alive = np.where(np.asarray(counts_banked, np.int64) >= fl)[0]
        panel, doc = drv.M._tier_stratified_panel(
            np.asarray(counts_banked, np.int64),
            100 * fl,
            int(drv.M.PANEL_CAP),
            int(drv.M.PANEL_SEED),
        )
        assert int(doc["floor"]) == fl, (doc["floor"], fl)
        # plan §2/§4: the 16,384 cap + allocation NEVER bind at any swept floor —
        # panel == clearing set exactly (re-asserted here).
        assert len(panel) == len(alive) and np.array_equal(np.asarray(panel, np.int64), alive), (
            f"panel cap/allocation bound unexpectedly at floor {fl} (arm {tag})"
        )
        cols = np.searchsorted(union_ids, alive)
        assert (union_ids[cols] == alive).all(), "alive set escapes the union columns"
        r2m, r2i, r2t = r2_map_u[cols], r2_ib_u[cols], r2_tm_u[cols]
        tier = tier_u[cols]
        act = np.asarray(counts_banked, np.float64)[alive]
        rng = np.random.default_rng(int(battery_seed))
        stats = drv._tier_stats(r2m, r2i, tier, act, n_perm, n_boot, rng)
        demo = _undefined_demotion(r2m, r2i, tier)
        row = _finish_floor_row(
            stats, demo, floor=fl, n_fit=n_fit_banked, registered=(fl == registered_floor)
        )
        row["n_te_rows"] = int(f_true.shape[0])
        row["n_alive"] = int(len(alive))
        row["alive_by_tier"] = {str(t): int((tier == t).sum()) for t in (0, 1, 2)}
        row["panel"] = doc
        row["trainmean_per_tier_median_r2"] = {
            str(t): drv._median_of(r2t[tier == t]) for t in (0, 1, 2)
        }
        shuffle_doc: dict = {
            "n_seeds": len(drv.SHUFFLE_SEEDS_2476),
            "advisory": True,
            "train_mean_note": "constant predictor: row-shuffle null == observed (no draws)",
            "per_read": {},
        }
        for rname, (nu, obs) in {"map": (null_map, r2m), "ib": (null_ib, r2i)}.items():
            sub = nu[:, cols].astype(np.float64)
            hi = float(np.nanpercentile(sub, 97.5)) if np.isfinite(sub).any() else float("nan")
            rr = obs[np.isfinite(obs)]
            shuffle_doc["per_read"][rname] = {
                "p97_5": hi,
                "frac_above": float((rr > hi).mean()) if len(rr) else None,
            }
        row["shuffle_null"] = shuffle_doc
        if corpus_r2:
            row["corpus_split"] = {
                label: {
                    "n_rows": corpus_n[label],
                    "per_tier_median_r2_map": {
                        str(t): drv._median_of(corpus_r2[label][cols][tier == t]) for t in (0, 1, 2)
                    },
                }
                for label in corpus_r2
            }
        if fl == registered_floor:
            assert np.array_equal(alive, np.asarray(registered_ids, np.int64)), (
                "registered-floor alive set != banked alive_ids (banked counts are canonical; "
                "this must be impossible)"
            )
            want_pf = np.asarray(np.load(committed_pf)["r2"], np.float64)
            tests = json.loads(committed_tests.read_text())
            got_med: dict[str, float | None] = {}
            want_med: dict[str, float | None] = {}
            for t in (0, 1, 2):
                pt = tests["per_tier"][str(t)]
                for read, arr in (("map", r2m), ("ib", r2i)):
                    want_med[f"{read}/t{t}"] = pt[f"median_r2_{read}"].get("median")
                    v = arr[tier == t]
                    v = v[np.isfinite(v)]
                    got_med[f"{read}/t{t}"] = float(np.median(v)) if len(v) else None
            gr = _gate_repro(
                r2m,
                want_pf,
                got_med,
                want_med,
                arm=tag,
                production=production,
                feat_ids=alive,
                tier=tier,
                fit_counts=np.asarray(counts_banked, np.int64)[alive],
                out_path=_gates_dir(args) / f"gr_{tag}.json",
            )
            row["registered_reference"] = {
                "committed_perfeature": str(committed_pf.relative_to(ROOT)),
                "committed_tests": str(committed_tests.relative_to(ROOT)),
                "committed_medians": want_med,
                "recomputed_medians": got_med,
                "max_abs_delta_r2": gr["max_abs_delta_r2"],
                "share_within_2e3": gr["share_within_2e3"],
                "n_violators": gr["n_violators"],
                "gr_verdict": gr["verdict"],
            }
        retr_rows[str(fl)] = {
            "n_alive": int(len(alive)),
            "tiers": drv._retrieval_cells(
                np.asarray(f_true[:, cols]),
                {"map": np.asarray(f_pred[:, cols]), "ib": np.asarray(f_ib[:, cols])},
                tier,
                ks=(1, 5, 10),
                device=args.device,
            ),
        }
        masks[f"alive_f{fl}"] = np.isin(union_ids, alive)
        rows.append(row)
        lattice_vector.append(
            {
                "floor_rows": int(fl),
                "label": row.get("lattice_reported", "not-evaluable-census-only"),
                "registered": bool(fl == registered_floor),
            }
        )
        print(
            f"[fs_stats] unit {tag}/f{fl} n_alive={len(alive)} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    sweep_doc = {
        "arm": tag,
        "floors_rows": [int(f) for f in floors],
        "n_fit_rows": int(n_fit_banked),
        "registered_floor_rows": int(registered_floor),
        "lattice_vector": lattice_vector,  # [stat-rec-1]: the complete ordered label vector
        "lattice_note": LATTICE_NOTE,
        "battery_seed": int(battery_seed),
        "seeds_note": "battery_seed read from the committed tier_tests provenance at run time; "
        "identical seeds/draw counts across floors by design",
        "n_perm": int(n_perm),
        "n_boot": int(n_boot),
        "alive_source": "banked counts (canonical; recomputed counts gate-only, plan §11)",
        "rows": rows,
    }
    retrieval_doc = {
        "arm": tag,
        "n_pool": int(f_true.shape[0]),
        "chance_note": "pool = held-out true feature rows; chance_at_k = k / n_pool",
        "rows": retr_rows,
    }
    pf_arrays = {
        "feat_ids": union_ids,
        "tier": tier_u,
        "counts_banked": np.asarray(counts_banked, np.int64)[union_ids],
        "r2_map": r2_map_u,
        "r2_ib": r2_ib_u,
        "r2_trainmean": r2_tm_u,
        "null_r2_map": null_map,
        "null_r2_ib": null_ib,
        "shuffle_seeds": np.asarray(drv.SHUFFLE_SEEDS_2476, np.int64),
        **masks,
    }
    for label, arr in corpus_r2.items():
        pf_arrays[f"r2_{label}"] = arr
    return sweep_doc, retrieval_doc, pf_arrays


def phase_stats(args) -> None:
    """Q4: the per-floor batteries for both arms + the pile twin; writes
    floor_sweep_{c,b}.json, floor_sweep_b_pile.json, floor_retrieval_{c,b}.json,
    perfeature_union_{c,b}.npz and consolidates every gate record into
    eval/gates_floor_sweep.json."""
    drv = _drv()
    drv.C.phase("fs_stats")
    out = _eval_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    finals = [
        out / n
        for n in (
            "floor_sweep_c.json",
            "floor_sweep_b.json",
            "floor_sweep_b_pile.json",
            "floor_retrieval_c.json",
            "floor_retrieval_b.json",
            "perfeature_union_c.npz",
            "perfeature_union_b.npz",
            "gates_floor_sweep.json",
        )
    ]
    stale = [*finals, *out.glob(".tmp_*")]
    regime, resume_ok = _enter_regime(out, args, "fs_stats", stale_paths=stale)
    if resume_ok and all(p.exists() for p in finals):
        logger.info("[fs_stats] resume: outputs present under matching regime; skip")
        return
    production = _production(args)
    n_perm = min(args.n_perm, 200) if args.smoke else args.n_perm
    n_boot = min(args.n_boot, 200) if args.smoke else args.n_boot
    cz = _census_dir(args)
    stage = _stage_banked_dir(args)
    e = stage / TA / "eval"

    # stats seeds READ from the committed provenance at run time (plan §10 Seeds)
    seed_c = int(json.loads((EV_TURNAVG / "tier_tests_c.json").read_text())["battery_seed"])
    seed_b = int(json.loads((EV_TURNAVG / "tier_tests_b.json").read_text())["battery_seed"])

    az_c = np.load(e / "alive_c.npz")
    fc = np.load(cz / "firing_census_c.npz")
    uc = np.load(cz / "union_encodes_c.npz")
    train_mean_c = np.asarray(fc["sums"], np.float64) / max(1, int(fc["n_fit_rows"]))
    sweep_c, retr_c, pf_c = _arm_sweep(
        drv,
        args,
        tag="c",
        floors=FLOORS_C,
        registered_floor=REGISTERED_FLOOR_C,
        registered_ids=np.asarray(az_c["alive_ids"], np.int64),
        counts_banked=np.asarray(az_c["counts"], np.int64),
        n_fit_banked=int(az_c["n_fit_rows"]),
        union_ids=np.asarray(uc["cols"], np.int64),
        f_true=np.asarray(uc["f_true"], np.float16),
        f_pred=np.asarray(uc["f_pred"], np.float16),
        f_ib=np.asarray(uc["f_ib"], np.float16),
        te_prov=np.asarray(uc["te_prov"], np.uint8),
        train_mean_full=train_mean_c,
        battery_seed=seed_c,
        committed_pf=EV_TURNAVG / "perfeature_c_encodepred.npz",
        committed_tests=EV_TURNAVG / "tier_tests_c.json",
        n_perm=n_perm,
        n_boot=n_boot,
        production=production,
    )
    drv._write_json(out / "floor_sweep_c.json", sweep_c, phase="fs-stats")
    drv._write_json(out / "floor_retrieval_c.json", retr_c, phase="fs-stats")
    tmp = out / ".tmp_perfeature_union_c.npz"
    np.savez(tmp, **pf_c)
    tmp.replace(out / "perfeature_union_c.npz")

    az_b = np.load(e / "alive_b.npz")
    fb = np.load(cz / "firing_census_b.npz")
    ub = np.load(cz / "union_encodes_b.npz")
    train_mean_b = np.asarray(fb["sums"], np.float64) / max(1, int(fb["n_fit_rows"]))
    sweep_b, retr_b, pf_b = _arm_sweep(
        drv,
        args,
        tag="b",
        floors=FLOORS_B,
        registered_floor=REGISTERED_FLOOR_B,
        registered_ids=np.asarray(az_b["alive_ids"], np.int64),
        counts_banked=np.asarray(az_b["counts"], np.int64),
        n_fit_banked=int(az_b["n_fit_rows"]),
        union_ids=np.asarray(ub["cols"], np.int64),
        f_true=np.asarray(ub["f_true"], np.float16),
        f_pred=np.asarray(ub["f_pred"], np.float16),
        f_ib=np.asarray(ub["f_ib"], np.float16),
        te_prov=None,
        train_mean_full=train_mean_b,
        battery_seed=seed_b,
        committed_pf=EV_TURNAVG / "perfeature_b_encodepred.npz",
        committed_tests=EV_TURNAVG / "tier_tests_b.json",
        n_perm=n_perm,
        n_boot=n_boot,
        production=production,
    )
    sweep_b["corpus_split_note"] = "bridge score rows carry no per-corpus split read (plan §6)"
    drv._write_json(out / "floor_sweep_b.json", sweep_b, phase="fs-stats")
    drv._write_json(out / "floor_retrieval_b.json", retr_b, phase="fs-stats")
    tmp = out / ".tmp_perfeature_union_b.npz"
    np.savez(tmp, **pf_b)
    tmp.replace(out / "perfeature_union_b.npz")

    # pile twin: census + medians per floor (exploratory; recomputed counts canonical)
    fp = np.load(cz / "firing_census_b_pile.npz")
    up = np.load(cz / "union_encodes_b_pile.npz")
    counts_p = np.asarray(fp["counts"], np.int64)
    floors_eff_b = [int(f) for f in fp["floors_eff"]]
    union_p = np.asarray(up["cols"], np.int64)
    pile_rows = []
    if len(union_p):
        r2_p = drv._r2_only(
            np.asarray(up["f_pred"], np.float16), np.asarray(up["f_true"], np.float16)
        )
        tier_p_u = drv.S.tier_of(union_p)
        for fl, fl_eff in zip(FLOORS_B, floors_eff_b, strict=True):
            panel_p, doc_p = drv.M._tier_stratified_panel(
                counts_p, 100 * fl_eff, int(drv.M.PANEL_CAP), int(drv.M.PANEL_SEED)
            )
            assert int(doc_p["floor"]) == fl_eff
            alive_p = np.asarray(panel_p, np.int64)
            cap_bound = len(alive_p) != int((counts_p >= fl_eff).sum())
            cols = np.searchsorted(union_p, alive_p)
            assert (union_p[cols] == alive_p).all()
            r2f = r2_p[cols]
            tier_f = tier_p_u[cols]
            demo = _undefined_demotion(r2f, r2f, tier_f)
            pile_rows.append(
                {
                    "floor_rows": int(fl),
                    "floor_rows_effective": int(fl_eff),
                    "n_alive": int(len(alive_p)),
                    "alive_by_tier": {str(t): int((tier_f == t).sum()) for t in (0, 1, 2)},
                    "panel": doc_p,
                    "panel_cap_bound": bool(cap_bound),
                    "per_tier_median_r2_map": {
                        str(t): drv._median_of(r2f[tier_f == t]) for t in (0, 1, 2)
                    },
                    "undefined_r2": demo["per_tier"],
                    "not_evaluable_census_only": bool(demo["not_evaluable_census_only"]),
                }
            )
            print(f"[fs_stats] unit b_pile/f{fl} n_alive={len(alive_p)}", flush=True)
    drv._write_json(
        out / "floor_sweep_b_pile.json",
        {
            "arm": "b_pile",
            "exploratory": True,
            "counts_note": "recomputed counts canonical (no banked reference; plan §4 Q3a)",
            "n_fit_rows": int(fp["n_fit_rows"]),
            "rows": pile_rows,
        },
        phase="fs-stats",
    )

    gates = {}
    for p in sorted(_gates_dir(args).glob("*.json")):
        gates[p.stem] = json.loads(p.read_text())
    drv._write_json(out / "gates_floor_sweep.json", {"gates": gates}, phase="fs-stats")
    lat_c = [x["label"] for x in sweep_c["lattice_vector"]]
    lat_b = [x["label"] for x in sweep_b["lattice_vector"]]
    _sentinel("fs-stats", f"Q4 done (lattice_c={lat_c} lattice_b={lat_b})")
    logger.info("[fs_stats] done: c=%s b=%s", lat_c, lat_b)


# ── Q5: figures + git/HF legs + terminal sentinel ────────────────────────────────


def _git(repo: Path, *argv: str, check: bool = True):
    """Explicit-env git call (subprocess env-passthrough rule; parent shape)."""
    import subprocess

    return subprocess.run(
        ["git", "-C", str(repo), *argv],
        check=check,
        env={**os.environ},
        capture_output=True,
        text=True,
    )


def _git_leg(declared: list[Path]) -> None:
    """Commit + push the declared git-destined result files on the issue branch:
    force-add (repo-wide *.npz gitignore, #958) + staged-index verify on the
    floor_sweep dest + rev-list push-verify with ONE fetch+rebase retry (#1880)
    + per-file artifact-presence assert (#1325). Adapted from the parent's
    _p7_git_leg with the sweep's dest path (the parent hardcodes turnavg/ and is
    a frozen driver — #1547)."""
    repo = ROOT
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    assert branch not in ("", "HEAD"), f"git leg needs a named branch checkout, got {branch!r}"
    rel = [str(p.resolve().relative_to(repo.resolve())) for p in declared]
    assert rel, "[fs_figures] empty declared git set on a git-committing round"
    print(f"[fs_figures] push-verify expected set ({len(rel)} files):", flush=True)
    for p in rel:
        print(f"[fs_figures]   {p}", flush=True)
    _git(repo, "add", "-f", "--", *rel)
    leftover = _git(
        repo,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        "eval_results/issue_2476/floor_sweep",
    ).stdout.strip()
    assert not leftover, f"[fs_figures] staged-index verify FAILED — gitignored skips: {leftover}"
    staged = _git(repo, "status", "--porcelain", "--", *rel).stdout.strip()
    if staged:
        _git(
            repo,
            "commit",
            "-m",
            f"task #2476: floor-sweep eval artifacts + figures ({len(rel)} files)",
            "--",
            *rel,
        )
    else:
        logger.info("[fs_figures] nothing to commit (declared set already committed)")
    verified = False
    for attempt in (1, 2):
        push = _git(repo, "push", "origin", f"HEAD:{branch}", check=False)
        if push.returncode == 0:
            behind = _git(repo, "rev-list", "--count", f"origin/{branch}..HEAD").stdout.strip()
            if behind == "0":
                verified = True
                break
        logger.warning(
            "[fs_figures] push attempt %d not verified (rc=%s): %s — fetch+rebase retry",
            attempt,
            push.returncode,
            (push.stderr or "")[-500:],
        )
        _git(repo, "fetch", "origin", branch)
        rb = _git(repo, "rebase", f"origin/{branch}", check=False)
        if rb.returncode != 0:
            _git(repo, "rebase", "--abort", check=False)
            raise RuntimeError(
                f"[fs_figures] rebase onto origin/{branch} conflicted — results committed "
                "locally; failing LOUD (never done with an unpushed result commit)"
            )
    if not verified:
        raise RuntimeError(f"[fs_figures] push to origin/{branch} not verified after 2 attempts")
    missing = [
        p
        for p in rel
        if not _git(
            repo, "ls-tree", "-r", f"origin/{branch}", "--name-only", "--", p
        ).stdout.strip()
    ]
    assert not missing, f"[fs_figures] artifact-presence FAILED — not in pushed tree: {missing}"
    print(f"[fs_figures] push-verify + artifact-presence OK ({len(rel)} files)", flush=True)


def _hf_leg(args) -> dict:
    """Census tensors + per-feature union arrays -> HF <hf_prefix>/floor_sweep/
    (plan §6.5/§10), fail-loud exact-set verify (the parent P3/P7 pattern)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    up = args.out_root / "stage" / "floor_sweep_upload"
    if up.exists():
        shutil.rmtree(up)
    up.mkdir(parents=True, exist_ok=True)
    srcs = [
        _census_dir(args) / n
        for n in ("firing_census_c.npz", "firing_census_b.npz", "firing_census_b_pile.npz")
    ] + [_eval_dir(args) / n for n in ("perfeature_union_c.npz", "perfeature_union_b.npz")]
    for srcp in srcs:
        assert srcp.exists(), f"[fs_figures] HF upload source missing: {srcp}"
        dst = up / srcp.name
        try:
            os.link(srcp, dst)
        except OSError:
            shutil.copy2(srcp, dst)
    prefix = f"{args.hf_prefix}/floor_sweep"
    res = upload_dir_sharded(
        up,
        DATA_REPO,
        prefix,
        repo_type="dataset",
        shard_glob="*",
        verify=True,
        delete_local=False,
        resume_skip=False,
    )
    if not res.rerouted:
        expected = [f"{prefix}/{p.name}" for p in sorted(up.iterdir()) if p.is_file()]
        missing = hub.verify_repo_paths_uploaded(HfApi(), DATA_REPO, expected, path_in_repo=prefix)
        assert not missing, f"[fs_figures] floor_sweep upload verify FAILED: {missing}"
    logger.info("[fs_figures] uploaded census+union -> %s (rerouted=%s)", prefix, res.rerouted)
    return {"prefix": prefix, "n_files": len(srcs), "rerouted": bool(res.rerouted)}


def _floor_x(doc: dict) -> list[float]:
    return [100.0 * r["floor_rows"] / doc["n_fit_rows"] for r in doc["rows"]]


def _row_med_ci(row: dict, read: str, t: int):
    """(median, ci95) of a per-floor per-tier median read; None-safe."""
    pt = row["per_tier"][str(t)]
    med = pt[f"median_r2_{read}"].get("median")
    ci = pt.get(f"ci95_median_{read}")
    return med, ci


def _fig_hero_r2(sweep: dict, fig_dir: Path, drv, stem: str) -> None:
    """Hero: per-tier median held-out R2 vs floor (map solid + CI band, ib
    dashed), one panel per tier. No caption blocks on canvas (§3.8-bis)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_palette,
        savefig_paper,
    )

    colors = paper_palette(3)
    xs = _floor_x(sweep)
    fig, axes = plt.subplots(1, 3, figsize=figsize_iclr_panels(3), sharex=True)
    for t, ax in enumerate(axes):
        med_map, lo, hi, med_ib = [], [], [], []
        for row in sweep["rows"]:
            m, ci = _row_med_ci(row, "map", t)
            med_map.append(np.nan if m is None else m)
            lo.append(np.nan if not ci else ci[0])
            hi.append(np.nan if not ci else ci[1])
            mi, _ = _row_med_ci(row, "ib", t)
            med_ib.append(np.nan if mi is None else mi)
        ax.fill_between(xs, lo, hi, color=colors[t], alpha=0.2, lw=0)
        ax.plot(xs, med_map, "-o", color=colors[t], ms=2.5, lw=1.0, label="map")
        ax.plot(xs, med_ib, "--s", color=colors[t], ms=2.0, lw=0.9, label="identity+bias")
        ax.set_title(drv.TIER_LABELS[t].replace("\n", " "), fontsize=6)
        ax.set_xlabel("alive floor (% of fit rows)")
    axes[0].invert_xaxis()  # sharex: ONE inversion flips all panels (looser -> right)
    axes[0].set_ylabel("median held-out R²")
    axes[0].legend(fontsize=5, loc="best")
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    print(f"[fs_figures] fig {stem}", flush=True)


def _fig_hero_acc1(sweep: dict, retr: dict, fig_dir: Path, drv, stem: str) -> None:
    """Hero: per-tier acc@1 (euclidean; map solid, ib dashed) vs floor + chance."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_palette,
        savefig_paper,
    )

    colors = paper_palette(3)
    xs = _floor_x(sweep)
    chance = 1.0 / max(1, retr["n_pool"])
    fig, axes = plt.subplots(1, 3, figsize=figsize_iclr_panels(3), sharex=True)
    for t, ax in enumerate(axes):
        for pred, ls in (("map", "-o"), ("ib", "--s")):
            ys = []
            for row in sweep["rows"]:
                cell = retr["rows"][str(row["floor_rows"])]["tiers"].get(str(t), {})
                v = cell.get(pred, {}).get("euclidean", {}).get("acc_at_k", {}).get("1")
                if v is None:
                    v = cell.get(pred, {}).get("euclidean", {}).get("acc_at_k", {}).get(1)
                ys.append(np.nan if v is None else v)
            ax.plot(xs, ys, ls, color=colors[t], ms=2.2, lw=0.9, label=pred)
        ax.axhline(chance, ls=":", lw=0.7, color="gray", label="chance")
        ax.set_yscale("log")
        ax.set_title(drv.TIER_LABELS[t].replace("\n", " "), fontsize=6)
        ax.set_xlabel("alive floor (% of fit rows)")
    axes[0].invert_xaxis()  # sharex: ONE inversion flips all panels
    axes[0].set_ylabel("retrieval acc@1 (euclidean)")
    axes[0].legend(fontsize=5, loc="best")
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    print(f"[fs_figures] fig {stem}", flush=True)


def _fig_alive_counts(docs: dict[str, dict], fig_dir: Path, drv) -> None:
    """Companion: per-tier alive-feature counts vs floor, one panel per arm, log-y."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_palette,
        savefig_paper,
    )

    colors = paper_palette(3)
    fig, axes = plt.subplots(1, len(docs), figsize=figsize_iclr_panels(len(docs)))
    titles = {
        "c": drv._ARM_LABELS["c"],
        "b": drv._ARM_LABELS["b"],
        "b_pile": "pile-trained token-level SAE (exploratory)",
    }
    for ax, (tag, doc) in zip(np.atleast_1d(axes), docs.items(), strict=True):
        xs = _floor_x(doc)
        for t in (0, 1, 2):
            ys = [max(row["alive_by_tier"][str(t)], 0.5) for row in doc["rows"]]
            ax.plot(
                xs,
                ys,
                "-o",
                color=colors[t],
                ms=2.2,
                lw=0.9,
                label=drv.TIER_LABELS[t].splitlines()[0],
            )
        ax.set_yscale("log")
        ax.set_title(titles.get(tag, tag), fontsize=6)
        ax.set_xlabel("alive floor (% of fit rows)")
        ax.invert_xaxis()
    np.atleast_1d(axes)[0].set_ylabel("alive features")
    np.atleast_1d(axes)[0].legend(fontsize=5, loc="best")
    savefig_paper(fig, "i2476_floor_alive_counts", dir=fig_dir)
    plt.close(fig)
    print("[fs_figures] fig i2476_floor_alive_counts", flush=True)


def _fig_r2_ecdf(sweep: dict, pf: dict, fig_dir: Path, drv) -> None:
    """Exploratory: per-floor per-tier ECDFs of finite per-feature R2 (map read)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_palette,
        savefig_paper,
    )

    colors = paper_palette(3)
    floors = [row["floor_rows"] for row in sweep["rows"]]
    fig, axes = plt.subplots(1, len(floors), figsize=figsize_iclr_panels(len(floors)), sharey=True)
    r2 = np.asarray(pf["r2_map"], np.float64)
    tier = np.asarray(pf["tier"])
    for ax, fl in zip(np.atleast_1d(axes), floors, strict=True):
        mask = np.asarray(pf[f"alive_f{fl}"], bool)
        for t in (0, 1, 2):
            v = r2[mask & (tier == t)]
            v = np.sort(v[np.isfinite(v)])
            if not len(v):
                continue
            y = np.arange(1, len(v) + 1) / len(v)
            ax.plot(v, y, color=colors[t], lw=0.9, label=drv.TIER_LABELS[t].splitlines()[0])
        ax.set_xlim(-1.0, 1.0)
        ax.set_title(f"floor {fl} rows", fontsize=6)
        ax.set_xlabel("per-feature held-out R²")
    np.atleast_1d(axes)[0].set_ylabel("fraction of alive features")
    np.atleast_1d(axes)[0].legend(fontsize=5, loc="upper left")
    savefig_paper(fig, "i2476_floor_r2_ecdf", dir=fig_dir)
    plt.close(fig)
    print("[fs_figures] fig i2476_floor_r2_ecdf", flush=True)


def _fig_perm_summary(docs: dict[str, dict], fig_dir: Path, drv) -> None:
    """Exploratory: observed within-stratum pooled Spearman vs the permutation
    null band, per floor, one panel per arm."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_palette,
        savefig_paper,
    )

    color = paper_palette(3)[0]
    fig, axes = plt.subplots(1, len(docs), figsize=figsize_iclr_panels(len(docs)))
    for ax, (tag, doc) in zip(np.atleast_1d(axes), docs.items(), strict=True):
        xs = _floor_x(doc)
        lo = [row["permutation"]["perm_band_2p5_97p5"][0] for row in doc["rows"]]
        hi = [row["permutation"]["perm_band_2p5_97p5"][1] for row in doc["rows"]]
        obs = [row["permutation"]["observed_pooled_spearman"] for row in doc["rows"]]
        ax.fill_between(xs, lo, hi, color="gray", alpha=0.3, lw=0, label="null 2.5–97.5%")
        ax.plot(xs, obs, "-o", color=color, ms=2.5, lw=1.0, label="observed")
        ax.axhline(0.0, ls=":", lw=0.6, color="gray")
        ax.set_title(drv._ARM_LABELS.get(tag, tag), fontsize=6)
        ax.set_xlabel("alive floor (% of fit rows)")
        ax.invert_xaxis()
    np.atleast_1d(axes)[0].set_ylabel("pooled Spearman(tier, R²)")
    np.atleast_1d(axes)[0].legend(fontsize=5, loc="best")
    savefig_paper(fig, "i2476_floor_perm_summary", dir=fig_dir)
    plt.close(fig)
    print("[fs_figures] fig i2476_floor_perm_summary", flush=True)


def _fig_firing_ecdf(census: dict[str, Path], sweeps: dict[str, dict], fig_dir: Path, drv):
    """Exploratory: tierwise firing-fraction ECDFs with the swept floors and the
    instrument mean firing rate k/width marked (perrow _fig_firing_ecdf shape)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_palette,
        savefig_paper,
    )

    colors = paper_palette(3)
    bounds = (0,) + tuple(drv.S.MATRYOSHKA_TIER_BOUNDS)
    fig, axes = plt.subplots(1, len(census), figsize=figsize_iclr_panels(len(census)))
    titles = {
        "c": drv._ARM_LABELS["c"],
        "b": drv._ARM_LABELS["b"],
        "b_pile": "pile-trained token-level SAE (exploratory)",
    }
    for ax, (tag, path) in zip(np.atleast_1d(axes), census.items(), strict=True):
        z = np.load(path)
        counts = np.asarray(z["counts"], np.int64)
        n_fit = max(1, int(z["n_fit_rows"]))
        frac = counts / float(n_fit)
        for t in (0, 1, 2):
            fr = np.sort(frac[bounds[t] : bounds[t + 1]])
            nz = fr[fr > 0]
            n_zero = len(fr) - len(nz)
            if not len(nz):
                continue
            y = (n_zero + np.arange(1, len(nz) + 1)) / len(fr)
            ax.plot(nz, y, color=colors[t], lw=0.9, label=drv.TIER_LABELS[t].splitlines()[0])
        for row in sweeps[tag]["rows"]:
            ax.axvline(row["floor_rows"] / n_fit, ls="--", lw=0.5, color="gray")
        ax.axvline(K_OVER_WIDTH, ls="-.", lw=0.7, color="black")
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("firing fraction over fit rows")
        ax.set_title(titles.get(tag, tag), fontsize=6)
    np.atleast_1d(axes)[0].set_ylabel("fraction of candidates")
    np.atleast_1d(axes)[0].legend(fontsize=5, loc="lower right")
    savefig_paper(fig, "i2476_floor_firing_ecdf", dir=fig_dir)
    plt.close(fig)
    print("[fs_figures] fig i2476_floor_firing_ecdf", flush=True)


def _fig_corpus_split(sweep: dict, fig_dir: Path, drv) -> None:
    """Exploratory: LMSYS-only vs WildChat-only per-tier medians (map) vs floor
    (descriptive grouping read on the same holdout — plan §6 fold note)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_palette,
        savefig_paper,
    )

    if not any("corpus_split" in row for row in sweep["rows"]):
        return
    colors = paper_palette(3)
    xs = _floor_x(sweep)
    fig, axes = plt.subplots(1, 3, figsize=figsize_iclr_panels(3), sharex=True)
    for t, ax in enumerate(axes):
        for label, ls in (("lmsys", "-o"), ("wildchat", "--s")):
            ys = []
            for row in sweep["rows"]:
                v = (
                    row.get("corpus_split", {})
                    .get(label, {})
                    .get("per_tier_median_r2_map", {})
                    .get(str(t))
                )
                ys.append(np.nan if v is None else v)
            ax.plot(xs, ys, ls, color=colors[t], ms=2.2, lw=0.9, label=label)
        ax.set_title(drv.TIER_LABELS[t].replace("\n", " "), fontsize=6)
        ax.set_xlabel("alive floor (% of fit rows)")
    axes[0].invert_xaxis()  # sharex: ONE inversion flips all panels
    axes[0].set_ylabel("median held-out R² (map)")
    axes[0].legend(fontsize=5, loc="best")
    savefig_paper(fig, "i2476_floor_corpus_split", dir=fig_dir)
    plt.close(fig)
    print("[fs_figures] fig i2476_floor_corpus_split", flush=True)


def _fig_shuffle_null(docs: dict[str, dict], fig_dir: Path, drv) -> None:
    """Exploratory: K=20 row-shuffle null p97.5 (map read) beside the per-tier
    observed medians, per floor, one panel per arm."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_palette,
        savefig_paper,
    )

    colors = paper_palette(3)
    fig, axes = plt.subplots(1, len(docs), figsize=figsize_iclr_panels(len(docs)))
    for ax, (tag, doc) in zip(np.atleast_1d(axes), docs.items(), strict=True):
        xs = _floor_x(doc)
        null_hi = [row["shuffle_null"]["per_read"]["map"]["p97_5"] for row in doc["rows"]]
        ax.plot(xs, null_hi, "-x", color="gray", ms=3, lw=0.8, label="shuffle-null p97.5")
        for t in (0, 1, 2):
            ys = [
                (np.nan if _row_med_ci(row, "map", t)[0] is None else _row_med_ci(row, "map", t)[0])
                for row in doc["rows"]
            ]
            ax.plot(
                xs,
                ys,
                "-o",
                color=colors[t],
                ms=2.2,
                lw=0.9,
                label=drv.TIER_LABELS[t].splitlines()[0],
            )
        ax.set_title(drv._ARM_LABELS.get(tag, tag), fontsize=6)
        ax.set_xlabel("alive floor (% of fit rows)")
        ax.invert_xaxis()
    np.atleast_1d(axes)[0].set_ylabel("median held-out R² (map)")
    np.atleast_1d(axes)[0].legend(fontsize=5, loc="best")
    savefig_paper(fig, "i2476_floor_shuffle_null", dir=fig_dir)
    plt.close(fig)
    print("[fs_figures] fig i2476_floor_shuffle_null", flush=True)


def phase_figures(args) -> None:
    """Q5: hero + exploratory figures; production-only git leg (eval JSONs/npzs
    -> eval_results/issue_2476/floor_sweep/, figures -> figures/issue_2476/) +
    HF leg (census + union tensors -> <hf_prefix>/floor_sweep/) + terminal
    results sentinel. Smoke diverts every output under out_root."""
    drv = _drv()
    drv.C.phase("fs_figures")
    state = args.out_root / "figures_state"
    state.mkdir(parents=True, exist_ok=True)
    done_path = state / "fs_done.json"
    regime, resume_ok = _enter_regime(state, args, "fs_figures", stale_paths=[done_path])
    production = _production(args)
    if resume_ok and done_path.exists():
        prev = json.loads(done_path.read_text())
        if production and not args.skip_upload and prev.get("hf_upload", {}).get("skipped"):
            logger.warning("[fs_figures] resume: prior run skipped the HF leg; RE-RUNNING Q5")
        else:
            try:
                drv.C.write_sentinel(
                    "epm:results" if production else "epm:smoke-result",
                    json.dumps(prev.get("digest", {})),
                    task_id=TASK_ID,
                    extra={"smoke": not production, "resumed": True, "blocks_pipeline": False},
                )
            except OSError as exc:
                logger.warning("[fs_figures] resume sentinel re-emit failed: %s", exc)
            logger.info("[fs_figures] resume: fs_done present under matching regime; skip")
            return
    ev = _eval_dir(args)
    cz = _census_dir(args)
    required = [
        ev / "floor_sweep_c.json",
        ev / "floor_sweep_b.json",
        ev / "floor_sweep_b_pile.json",
        ev / "floor_retrieval_c.json",
        ev / "floor_retrieval_b.json",
        ev / "perfeature_union_c.npz",
        ev / "perfeature_union_b.npz",
        ev / "gates_floor_sweep.json",
        cz / "firing_census_c.npz",
        cz / "firing_census_b.npz",
        cz / "firing_census_b_pile.npz",
    ]
    missing_in = [str(p) for p in required if not p.exists()]
    assert not missing_in, f"[fs_figures] earlier-phase inputs missing: {missing_in}"

    import matplotlib

    matplotlib.use("Agg")
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style("iclr")
    if production:
        fig_dir = ROOT / "figures" / "issue_2476"
        dest = ROOT / "eval_results" / "issue_2476" / "floor_sweep"
    else:
        fig_dir = args.out_root / "figures" / "issue_2476"
        dest = args.out_root / "eval_results_stage" / "floor_sweep"
        logger.warning("[fs_figures] non-production: outputs DIVERTED under %s", args.out_root)
    fig_dir.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)

    sweep_c = json.loads((ev / "floor_sweep_c.json").read_text())
    sweep_b = json.loads((ev / "floor_sweep_b.json").read_text())
    sweep_p = json.loads((ev / "floor_sweep_b_pile.json").read_text())
    retr_c = json.loads((ev / "floor_retrieval_c.json").read_text())
    pf_c = dict(np.load(ev / "perfeature_union_c.npz"))
    _fig_hero_r2(sweep_c, fig_dir, drv, "i2476_floor_sweep_hero_r2")
    _fig_hero_acc1(sweep_c, retr_c, fig_dir, drv, "i2476_floor_sweep_hero_acc1")
    _fig_hero_r2(sweep_b, fig_dir, drv, "i2476_floor_hero_b_r2")
    _fig_alive_counts({"c": sweep_c, "b": sweep_b, "b_pile": sweep_p}, fig_dir, drv)
    _fig_r2_ecdf(sweep_c, pf_c, fig_dir, drv)
    _fig_perm_summary({"c": sweep_c, "b": sweep_b}, fig_dir, drv)
    _fig_firing_ecdf(
        {
            "c": cz / "firing_census_c.npz",
            "b": cz / "firing_census_b.npz",
            "b_pile": cz / "firing_census_b_pile.npz",
        },
        {"c": sweep_c, "b": sweep_b, "b_pile": sweep_p},
        fig_dir,
        drv,
    )
    _fig_corpus_split(sweep_c, fig_dir, drv)
    _fig_shuffle_null({"c": sweep_c, "b": sweep_b}, fig_dir, drv)

    copied: list[Path] = []
    for src in sorted([*ev.glob("*.json"), *ev.glob("*.npz")]):
        if src.name == "regime.json":
            continue
        shutil.copy2(src, dest / src.name)
        copied.append(dest / src.name)
    fig_files = sorted(
        q
        for stem in (
            "i2476_floor_sweep_hero_r2",
            "i2476_floor_sweep_hero_acc1",
            "i2476_floor_hero_b_r2",
            "i2476_floor_alive_counts",
            "i2476_floor_r2_ecdf",
            "i2476_floor_perm_summary",
            "i2476_floor_firing_ecdf",
            "i2476_floor_corpus_split",
            "i2476_floor_shuffle_null",
        )
        for q in fig_dir.glob(f"{stem}.*")
    )
    hf_doc: dict = {"skipped": True}
    if production:
        _git_leg(copied + fig_files)
        if args.skip_upload:
            logger.warning("[fs_figures] --skip-upload: HF floor_sweep upload SKIPPED (loud)")
        else:
            hf_doc = _hf_leg(args)

    gates = json.loads((ev / "gates_floor_sweep.json").read_text())["gates"]
    digest = {
        "lattice_vector_c": [x["label"] for x in sweep_c["lattice_vector"]],
        "lattice_vector_b": [x["label"] for x in sweep_b["lattice_vector"]],
        "alive_t2_c_by_floor": {
            str(r["floor_rows"]): r["alive_by_tier"]["2"] for r in sweep_c["rows"]
        },
        "gates": {k: v.get("verdict") for k, v in gates.items()},
    }
    doc = {
        "regime": regime,
        "digest": digest,
        "figures": [str(p) for p in fig_files],
        "eval_artifacts": sorted(str(p) for p in copied),
        "hf_upload": hf_doc,
        "production": production,
    }
    if production and args.skip_upload:
        logger.warning(
            "[fs_figures] production + --skip-upload: NOT writing fs_done.json — Q5 stays "
            "incomplete until the HF leg runs (parent P7 convention)"
        )
    else:
        _write_json_atomic(done_path, doc)
    try:
        drv.C.write_sentinel(
            "epm:results" if production else "epm:smoke-result",
            json.dumps(digest),
            task_id=TASK_ID,
            extra={"smoke": not production, "blocks_pipeline": False},
        )
    except OSError as exc:
        logger.warning("[fs_figures] results sentinel write failed: %s", exc)
    logger.info("[fs_figures] done: %s", json.dumps(digest["gates"]))


# ── Q0: composed smoke ───────────────────────────────────────────────────────────


def _smoke_leg_expected(name: str, s) -> list[Path]:
    """Per-leg durable-output verification set for the composed smoke."""
    a = s.out_root / "assemble"
    table = {
        "assemble": [
            a / "X19.fp16.npy",
            a / "Y19.fp16.npy",
            a / "rows_present.npy",
            a / "split_meta.json",
        ],
        "stage_banked": [s.out_root / "sentinels" / "stage_banked.done.json"],
        "census": [
            _census_dir(s) / n
            for n in (
                "firing_census_c.npz",
                "firing_census_b.npz",
                "firing_census_b_pile.npz",
                "union_encodes_c.npz",
                "union_encodes_b.npz",
                "union_encodes_meta.json",
            )
        ],
        "stats": [
            _eval_dir(s) / n
            for n in (
                "floor_sweep_c.json",
                "floor_sweep_b.json",
                "floor_sweep_b_pile.json",
                "floor_retrieval_c.json",
                "floor_retrieval_b.json",
                "perfeature_union_c.npz",
                "perfeature_union_b.npz",
                "gates_floor_sweep.json",
            )
        ],
        "figures": [
            s.out_root / "figures" / "issue_2476" / "i2476_floor_sweep_hero_r2.png",
            s.out_root / "figures" / "issue_2476" / "i2476_floor_sweep_hero_r2.meta.json",
            s.out_root / "figures_state" / "fs_done.json",
        ],
    }
    return table[name]


def phase_smoke(args) -> None:
    """Q0: composed end-to-end smoke — the SAME phase functions Q1->Q5 on a tiny
    slice (2 capture chunks; bridge rows clamped) under out_root/smoke with the
    REAL SAE-c + chanind dictionaries, per-leg output verification + the Q1/Q3
    wall fence bases (smoke_timing.json). Production-n gates G-C/G-R demote to
    informational at smoke n (plan §4 blind-spot enumeration)."""
    drv = _drv()
    drv.C.phase("fs_smoke")
    assert args.out_root.name != "smoke", "phase_smoke must not recurse into its own smoke root"
    s = argparse.Namespace(**vars(args))
    s.out_root = args.out_root / "smoke"
    s.smoke = True
    s.max_chunks = args.max_chunks if args.max_chunks > 0 else 2
    s.smoke_rows = args.smoke_rows if args.smoke_rows > 0 else SMOKE_B_SCORE
    s.skip_upload = True  # repo/Hub legs are production-only under the composed smoke
    s.sae_dir = s.out_root / "sae_cache"
    s.out_root.mkdir(parents=True, exist_ok=True)
    logger.info("[fs_smoke] composed Q1->Q5 under %s (max_chunks=%d)", s.out_root, s.max_chunks)
    timings: dict[str, float] = {}
    for name in ("assemble", "stage_banked", "census", "stats", "figures"):
        t0 = time.time()
        PHASES[name](s)
        timings[name] = round(time.time() - t0, 1)
        missing = [str(p) for p in _smoke_leg_expected(name, s) if not p.exists()]
        assert not missing, f"[fs_smoke] leg {name} completed without expected outputs: {missing}"
        print(f"[fs_smoke] unit {name} ok elapsed={timings[name]}s", flush=True)
    meta = json.loads((_census_dir(s) / "union_encodes_meta.json").read_text())
    doc = {
        "legs_wall_s": timings,
        "per_chunk_stage_extract_s": round(timings["assemble"] / max(1, s.max_chunks), 1),
        "census_timings_s": meta["timings_s"],
        "encode_s_per_row": meta["timings_s"].get("encode_c_s_per_row"),
        "out_root": str(s.out_root),
        "max_chunks": int(s.max_chunks),
        "smoke_rows": int(s.smoke_rows),
        "skip_upload_forced": True,
    }
    _write_json_atomic(s.out_root / "smoke_timing.json", doc)
    _write_json_atomic(args.out_root / "sentinels" / "smoke.done.json", doc)
    time.sleep(1.1)  # distinct epoch-second sentinel filenames (parent convention)
    try:
        drv.C.write_sentinel(
            "epm:smoke-result",
            json.dumps(doc),
            task_id=TASK_ID,
            extra={"smoke": True, "blocks_pipeline": False},
        )
    except OSError as exc:
        logger.warning("[fs_smoke] smoke-result sentinel write failed: %s", exc)
    logger.info("[fs_smoke] done: %s", json.dumps(timings))


# ── CLI ──────────────────────────────────────────────────────────────────────────

PHASE_ORDER = ("smoke", "assemble", "stage_banked", "census", "stats", "figures")
PHASES = {
    "smoke": phase_smoke,
    "assemble": phase_assemble,
    "stage_banked": phase_stage_banked,
    "census": phase_census,
    "stats": phase_stats,
    "figures": phase_figures,
}


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Issue #2476 floor-sensitivity sweep driver (see module docstring)"
    )
    ap.add_argument("--phase", default="all", choices=["all", *PHASE_ORDER])
    ap.add_argument(
        "--out-root", type=Path, default=Path("/workspace/eps_out/issue2476_floor_sweep")
    )
    ap.add_argument(
        "--hf-prefix",
        default="issue2476_turnavg/analysis_tensors",
        help="HF data-repo destination prefix (floor_sweep/ appended for this round)",
    )
    ap.add_argument("--smoke", action="store_true", help="tiny-N run of the SAME pipeline")
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = all 1,920 chunks (production)")
    ap.add_argument("--smoke-rows", type=int, default=0, help="0 = all bridge rows (production)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--sae-dir", type=Path, default=None, help="SAELens weight cache dir")
    ap.add_argument("--fresh-stream", action="store_true", help="Q1: ignore the stream cursor")
    ap.add_argument("--skip-upload", action="store_true", help="Q5: local-only run (loud)")
    ap.add_argument("--gpu-id", type=int, default=-1, help="informational; CVD pins the device")
    ap.add_argument("--n-perm", type=int, default=10_000, help="Q4 tier-permutation draws")
    ap.add_argument("--n-boot", type=int, default=10_000, help="Q4 feature-bootstrap draws")
    ap.add_argument(
        "--resume-across-code-sha",
        action="store_true",
        help="retain completed outputs on a code-SHA-ONLY regime mismatch (crash-fix escape)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="argparse-attribute completeness + call-arity bind + deferred-import resolution",
    )
    return ap.parse_args(argv)


def main() -> None:
    args = _parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Deferred-import resolution (smoke-architecture Axis 1): execute every
        # function-body import of this driver, then load the parent driver by
        # file path (its module top executes the whole reused-symbol surface).
        import inspect

        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
        import torch  # noqa: F401
        from huggingface_hub import HfApi, hf_hub_download  # noqa: F401

        from explore_persona_space.analysis.paper_plots import (  # noqa: F401
            figsize_iclr_panels,
            paper_palette,
            savefig_paper,
            set_paper_style,
        )
        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.provenance import (  # noqa: F401
            as_metadata_dict,
            git_provenance,
        )
        from explore_persona_space.orchestrate.upload_sharded import (
            upload_dir_sharded,  # noqa: F401
        )

        drv = _drv()
        # Call-shape binds for every reused helper (plan §10 call-shape bind block).
        for fn, a, k in (
            (drv.phase_assemble, (None,), {}),
            (drv._load_scratch_meta, (None,), {}),
            (drv._stage_m_split, (None,), {}),
            (drv._encode_restricted, (None, None, None, None), {}),
            (drv._r2_only, (None, None), {}),
            (drv._shuffle_null_r2, (None, None, None), {}),
            (drv._tier_stats, (None, None, None, None, None, None, None), {}),
            (drv._retrieval_cells, (None, {}, None), {"ks": (1, 5, 10), "device": "cpu"}),
            (drv._median_of, (None,), {}),
            (drv.M._tier_stratified_panel, (None, 120_000, 16_384, 14_824), {}),
            (drv.S.tier_of, (None,), {}),
        ):
            inspect.signature(fn).bind(*a, **k)
        inspect.signature(hf_hub_download).bind(
            DATA_REPO, "f", repo_type="dataset", revision=DATA_REPO_REVISION, local_dir="x"
        )
        print("[import-check] OK", flush=True)
        raise SystemExit(0)
    if args.device == "auto":
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.sae_dir is None:
        args.sae_dir = args.out_root / "sae_cache"
    args.out_root.mkdir(parents=True, exist_ok=True)
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
