"""Issue #2476 same-issue follow-up `k200-instrument-census` — pod-side driver.

Plan: the plan of record (v11) resolves via
`uv run python scripts/task.py find 2476` -> `<task-dir>/plans/plan.md`. NEVER
read it through a worktree-relative `tasks/...` path — a worktree's `tasks/`
tree is FROZEN at its base commit and serves a stale plan with no error
(#2422). ONE experimental variable changes
vs the parent run: the SAE sparsity budget k = 100 -> 200 (the parent driver's
new `--sae-k` knob; width 65,536, tier bounds, LR, batch, epochs, seeds, pool
recipe, splits, floors, and every estimator held at parent values). The round
retrains ONE fresh matryoshka BatchTopK SAE at k=200, re-counts the per-tier
alive census at the registered 1% floor + the floor-sweep round's swept floors,
and re-scores the banked context->answer map + identity+bias (+ a restored
dense-input companion ridge) in the new basis, beside the committed k=100
values (comparison arm — read from committed git JSONs, never recomputed,
never a gate).

Phases (plan §4; every estimator kernel is the PARENT's / the floor-sweep
round's, imported by module — never re-implemented):
  smoke        R0 composed tiny-slice run of the SAME phase functions under
               out_root/smoke (--max-chunks 2; 200-step k=200 train pilot),
               per-leg output verification + wall fence bases. Production-n
               gates demote to informational at smoke n.
  assemble     R1 = the parent's phase_assemble VERBATIM — now END-TO-END AT
               THE PIN (B1: pinned chunk listing + chunk downloads + pass_b
               fetch + revision-bearing stream fingerprint).
  stage_banked R2 revision-pinned staging of the banked map/ib predictions +
               split_meta + gate G-S (split shas) + gate G-A (row alignment).
  sae_train    R3 = the parent's phase_sae_train with sae_k=200 (identical
               training code path; per-epoch ckpt + regime-keyed resume with k
               in the regime; gate G-4' = the parent G4 recipe, rc=25; weights
               uploaded fail-loud to HF <hf_prefix>/sae_c_k200/ BEFORE eval).
               Exit asserts cfg k == 200 + the sae_c_k200 leaf/prefix.
  densein      R4 fit-side full-width census pass (counts+sums+L0; gate
               G-C'(i) sum-accounting identity, EXACT) -> union-of-floors
               alive columns -> restricted f_true encode over concat(fit,
               holdout) rows -> the dense-input companion ridge on the UNION
               columns via the direct _gram_ridge_single drive (A3 — never the
               verbatim _dense_companion_c own-panel path). A step-4 fit
               failure DROPS the reported companion (recorded), never the
               round (plan §7).
  census       R5 census_k200.json (registered BUDGET lattice input) +
               score-side restricted encodes (f_pred/f_ib) + the B2
               pred_encode_fve producer (_recon_fve over the 20,000
               holdout_pred16 rows) + gate G-C'(ii) (boundary-tolerant
               _gate_counts reuse, GC_COUNT_TOL=3) + G-C'(iii) (n_fit ==
               120,000 vs split shas, exact).
  stats        R6 per-floor batteries via the parent kernels (_r2_only,
               _tier_stats, _shuffle_null_r2, _retrieval_cells with THREE
               sources map/ib/densein — B3) + gates_k200.json consolidation
               (incl. the B2 pred_encode_fve field).
  figures      R7 hero pair + exploratory figures; the EXPLICIT §6.5 git
               allowlist (GIT_EVAL_BASENAMES — declared JSONs +
               perfeature_union_k200.npz + gates_k200.json; the HF-ONLY
               firing_census_k200.npz / perfeature_k200_densein.npz are
               EXCLUDED — the densein npz plausibly exceeds GitHub's 100 MiB
               hard limit, r8 U1) -> git eval_results/issue_2476/k200_census/
               (force-add + staged-index verify + push-verify), census
               tensors -> HF <hf_prefix>/k200_census/, upload-verify
               enumerating ALL prefixes written this round (sae_c_k200/ +
               k200_census/), terminal results sentinel.

Smoke blind-spot enumeration (plan §4, mirrored per phase at the fence sites):
  - G-4' (FVE floor), G-C'(ii)/(iii), and the census/lattice reads are
    production-n verdicts — demoted to informational at smoke n (the smoke
    certifies pipeline shape, not verdicts). G-C'(i) and G-S/G-A bind at smoke
    too (exact identities over production-grain inputs).
  - R3 production convergence (3 epochs x 933,444 rows) is not certified by
    the 200-step pilot — certified by G-4' at production.
  - The 1,920-chunk streaming loop's at-scale retry/quota behavior is
    exercised only in production (2-chunk smoke); mitigated by reusing the
    parent's production-proven bounded-retry staging path (now pinned, B1).
  - The --sae-dict sub-production width path used by VM census/stats smoke
    legs does not exercise the full 65,536-wide memory path — covered by the
    production-width 200-step R0 pilot on the pod.
  - R4's union floor relaxes to ever-fired features at SMOKE ONLY when the
    clamped floor (== ALL fit rows at tiny n) yields zero — the production
    union rule (counts >= min floor, empty => RuntimeError) is byte-identical
    and never exercised in that degenerate branch by the smoke.
  - No other smoke-conditional implementation substitutions, gate downgrades,
    or production-only third-party imports (same code, smaller N; no new deps).

Pod-side contract: sentinels under /workspace/logs/issue-2476-*.json ONLY
(never task.py); [phase=...] log lines; [phase=done] terminal. LMSYS/WildChat
text is handled DIGEST-ONLY. Resume is REGIME-KEYED (a regime.json per phase
under out_root/phase_state/<phase>/; config mismatch refuses, code-SHA-only
mismatch recomputes loudly); resume keys hash GENERATING PARAMETERS only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM smoke)

import issue2476_floor_sweep as FS  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2476_k200")

TASK_ID = 2476
DATA_REPO = FS.DATA_REPO
DATA_REPO_REVISION = FS.DATA_REPO_REVISION  # the lineage pin (floor_sweep :83)
TA = FS.TA
REFIT_HF = FS.REFIT_HF

SAE_K200 = 200  # the round's ONE experimental variable (plan §4 Code delta 1)
FLOORS = (1200, 600, 300, 240)  # fit-row floors (1% registered + swept companions)
REGISTERED_FLOOR = 1200
N_FIT_PRODUCTION = 120_000  # G-C'(iii): re-asserted vs the sha-pinned sae_fit pool
BUDGET_FINEST_ALIVE_MIN = 10  # registered budget lattice (plan §3): >= 10 => Budget-limited
GC_COUNT_TOL = FS.GC_COUNT_TOL  # boundary-tolerant G-C'(ii) reuse (A2)

# Gate exit codes (floor-sweep convention; G-4' halts INSIDE the parent's
# phase_sae_train with the parent's RC_G4 = 25).
RC_GA = 30  # G-A row-alignment HALT
RC_GC = 31  # G-C' counts self-consistency HALT (FS._gate_counts exits FS.RC_GC == 31)
RC_GS = 33  # G-S split-sha HALT
RC_G4 = 25  # G-4' FVE-floor HALT (== parent RC_G4; pinned by tests/test_issue2476_k200.py)
assert RC_GC == FS.RC_GC, (RC_GC, FS.RC_GC)

# §3 manipulation check (r8 U7): the budget verdict's CAUSAL narration is
# conditioned on the realized k=200 val L0 materially exceeding the parent's
# k=100 realized 98.96 (plan §3 lattice-evaluation note, A4). The concrete
# conservative predicate — stated verbatim in every lattice record — is
# val_l0 >= MANIPULATION_VAL_L0_MIN; the analyzer owns the prose narration.
PARENT_VAL_L0_K100 = 98.96  # plan §3/§11 (parent train_log.json epochs[-1].val_l0)
MANIPULATION_VAL_L0_MIN = 150.0

# §6.5 destination split (r8 U1): the git leg commits EXACTLY this allowlist —
# the three declared result JSONs + the per-feature union npz + the gates
# consolidation. firing_census_k200.npz + perfeature_k200_densein.npz are
# HF-ONLY (§6.5 row 6; the densein npz is 20,000 x n_union fp16 and plausibly
# exceeds GitHub's 100 MiB hard limit — a glob sweep would kill the R7 push
# AFTER the full production run). union_encodes_meta.json is pod-side meta
# (plan §9 phase-outputs row; not a §6.5 git destination).
GIT_EVAL_BASENAMES = (
    "census_k200.json",
    "tier_sweep_k200.json",
    "retrieval_k200.json",
    "gates_k200.json",
    "perfeature_union_k200.npz",
)
HF_ONLY_EVAL_BASENAMES = ("firing_census_k200.npz", "perfeature_k200_densein.npz")
GIT_FILE_MAX_BYTES = 95 * 1024**2  # fail-loud margin under GitHub's 100 MiB hard limit

LATTICE_NOTE_K200 = (
    "reported (the issue's registered GRADIENT rule lives at the parent's k=100 1% cell; "
    "this round's registered claim is the BUDGET lattice in census_k200.json)"
)
EV_TURNAVG = ROOT / "eval_results" / "issue_2476" / "turnavg"
EV_FLOOR_SWEEP = ROOT / "eval_results" / "issue_2476" / "floor_sweep"

STAGE_FILES = (
    f"{TA}/eval/ib_c.npz",
    f"{TA}/split_meta/split_meta.json",
    REFIT_HF,
)

_drv = FS._drv  # the parent driver, file-path-imported + cached (FS pattern)


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


def _state_dir(args, phase: str) -> Path:
    """Per-phase regime-manifest dir (one regime.json per phase — no collisions)."""
    return args.out_root / "phase_state" / phase


def _record_gate(args, key: str, record: dict) -> None:
    FS._write_json_atomic(_gates_dir(args) / f"{key}.json", record)


def _sentinel(name: str, note: str, extra: dict | None = None) -> None:
    """Non-blocking phase sentinel through the parent's poller-conformant writer."""
    payload = {"blocks_pipeline": False}
    if extra:
        payload.update(extra)
    try:
        _drv().C.write_sentinel(f"phase-{name}", note, task_id=TASK_ID, extra=payload)
    except OSError as e:
        logger.warning("[sentinel] phase-%s write failed: %s", name, e)


# ── prerequisite-gate loader (r8 U2: recorded FAILs re-apply at phase entry) ─────

#: Upstream recorded gate verdicts each standalone downstream phase REQUIRES at
#: entry. `g4` reads the parent's gates_p4.json in the sae_c_k200 leaf; every
#: other key reads _gates_dir(args)/<key>.json.
_PHASE_REQUIRED_GATES: dict[str, tuple[str, ...]] = {
    "sae_train": ("gs", "ga"),
    "densein": ("gs", "ga", "g4"),
    "census": ("gs", "ga", "g4", "gc_i"),
    "stats": ("gs", "ga", "g4", "gc_i", "gc_ii", "gc_iii"),
    "figures": ("gs", "ga", "g4", "gc_i", "gc_ii", "gc_iii"),
}
_GATE_RC = {"gs": RC_GS, "ga": RC_GA, "g4": RC_G4, "gc_i": RC_GC, "gc_ii": RC_GC, "gc_iii": RC_GC}


def _require_upstream_gates(args, phase: str) -> None:
    """r8 U2 (the floor round's failed-gate-resume convention, mirrored from the
    parent's _reapply_recorded_gate_verdicts): every standalone downstream phase
    entry REQUIRES the upstream recorded gate verdicts and RE-APPLIES a recorded
    FAIL with the ORIGINAL rc BEFORE any heavy work — a persisted G-S/G-A/G-4'/
    G-C' FAIL can never be laundered by re-entering at a later ``--phase``
    (producing/uploading terminal results off a failed gate). A MISSING required
    record fails loud (the producing phase never completed on this out-root).
    Verdict semantics (r10 strict predicate, reconciler-required): FAIL
    re-applies the recorded rc at every entry; PRODUCTION entry additionally
    requires the literal "PASS" on EVERY required record — a smoke-demoted
    "INFORMATIONAL-smoke" (or any unknown verdict) raises pre-heavy-work, so a
    smoke-gated out-root can never launder into production terminal results;
    SMOKE entry accepts {"PASS", "INFORMATIONAL-smoke"} (the smoke writers
    demote by design) and refuses anything else. The sanctioned densein DROP
    record ("DROPPED-companion") lives under the NON-required
    ``densein_dropped`` key and is never loaded here."""
    production = _production(args)
    accepted = ("PASS",) if production else ("PASS", "INFORMATIONAL-smoke")
    for key in _PHASE_REQUIRED_GATES.get(phase, ()):
        if key == "g4":
            # the parent's own leaf-local gate record (written FAIL included);
            # _parent_args pins sae_k=200 so the leaf is always sae_c_k200
            rec_path = args.out_root / f"sae_c_k{SAE_K200}" / "gates_p4.json"
            rec = json.loads(rec_path.read_text())["g4"] if rec_path.exists() else None
        else:
            rec_path = _gates_dir(args) / f"{key}.json"
            rec = json.loads(rec_path.read_text()) if rec_path.exists() else None
        if rec is None:
            raise RuntimeError(
                f"[{phase}] required upstream gate record missing: {rec_path} — run the "
                "producing phase first (a downstream phase never enters off an ungated "
                "out-root; r8 U2)"
            )
        verdict = rec.get("verdict")
        if verdict == "FAIL":
            rc = _GATE_RC[key]
            logger.error(
                "[%s] recorded %s verdict FAIL — re-applying the original halt (rc=%d; r8 U2)",
                phase,
                key,
                rc,
            )
            sys.exit(rc)
        if verdict not in accepted:
            entry = (
                "production entry requires the literal PASS"
                if production
                else "smoke entry accepts only PASS / INFORMATIONAL-smoke"
            )
            raise RuntimeError(
                f"[{phase}] required upstream gate record {key} carries verdict "
                f"{verdict!r} — {entry} (r10 strict predicate; record: {rec_path})"
            )


def _parent_args(args, phase: str) -> SimpleNamespace:
    """The parent driver's full args contract (every field its _regime /
    _enter_phase_regime / phase_assemble / phase_sae_train / _load_scratch_meta
    read), threaded from this round's CLI. sae_k is PINNED to 200 — the round's
    one experimental variable (never a CLI dial here; the parent's own
    membership assert is the typo fence)."""
    return SimpleNamespace(
        phase=phase,
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
        sae_steps=int(args.sae_steps),
        sae_dict=int(args.sae_dict),
        sae_k=SAE_K200,
        n_perm=int(args.n_perm),
        n_boot=int(args.n_boot),
        fit_n=0,
        g2a_probe_rows=0,
        import_check=False,
        resume_across_code_sha=bool(args.resume_across_code_sha),
    )


# ── regime-keyed resume (round-owned phases; parent convention) ──────────────────


def _regime(args) -> dict:
    """Regime manifest for the round-owned phases: k budget + floors + revision
    pin + every output/destination-affecting dial. GENERATING PARAMETERS only."""
    from explore_persona_space.orchestrate.provenance import git_provenance

    base = {
        "smoke": bool(args.smoke),
        "max_chunks": int(args.max_chunks),
        "smoke_rows": int(args.smoke_rows),
        "device": str(args.device),
        "n_perm": int(args.n_perm),
        "n_boot": int(args.n_boot),
        "sae_steps": int(args.sae_steps),
        "sae_dict": int(args.sae_dict),
        "sae_k": SAE_K200,
        "hf_prefix": str(args.hf_prefix),
        "skip_upload": bool(args.skip_upload),
        "floors": list(FLOORS),
        "registered_floor": REGISTERED_FLOOR,
        "budget_finest_alive_min": BUDGET_FINEST_ALIVE_MIN,
        "data_repo_revision": DATA_REPO_REVISION,
        "gc_count_tol": GC_COUNT_TOL,
    }
    cfg_hash = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()[:16]
    prov = git_provenance()
    code_sha = prov.commit_sha_full or prov.commit_sha or "unknown"
    return {**base, "config_hash": cfg_hash, "code_sha": code_sha}


def _enter_regime(args, phase: str, stale_paths=()) -> tuple[dict, bool]:
    """Write/verify the per-phase regime manifest (parent _enter_phase_regime
    semantics: config mismatch refuses; code-SHA-only mismatch wipes the stale
    outputs BEFORE the manifest write and recomputes loudly)."""
    state = _state_dir(args, phase)
    regime = _regime(args)
    path = state / "regime.json"
    if path.exists():
        prev = json.loads(path.read_text())
        if prev.get("config_hash") != regime["config_hash"]:
            raise RuntimeError(
                f"[{phase}] out-root {args.out_root} holds a run under a DIFFERENT regime "
                f"(config_hash {prev.get('config_hash')} != {regime['config_hash']}); "
                "use a fresh --out-root (never silently mix regimes)"
            )
        if prev.get("code_sha") != regime["code_sha"]:
            if args.resume_across_code_sha:
                logger.warning(
                    "[%s] code SHA changed but --resume-across-code-sha set: outputs RETAINED",
                    phase,
                )
                FS._write_json_atomic(path, regime)
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
            FS._write_json_atomic(path, regime)
            return regime, False
        return regime, True
    state.mkdir(parents=True, exist_ok=True)
    for p in stale_paths:
        if p.exists():
            logger.warning("[%s] fresh manifest: removing stale %s", phase, p.name)
            p.unlink()
    FS._write_json_atomic(path, regime)
    return regime, False


# ── gates (pure comparison logic — pinned by tests/test_issue2476_k200.py) ───────


def _gate_splits(assembled_shas: dict, banked_shas: dict, *, out_path: Path | None = None) -> dict:
    """G-S (plan §7; equality — binds at smoke too, shas are pool-grain): the
    assembled split sha256 digests must EQUAL the banked split_meta records.
    FAIL => record written FIRST, then sys.exit(RC_GS)."""
    ok = assembled_shas == banked_shas
    record = {
        "gate": "G-S",
        "assembled_shas": assembled_shas,
        "banked_shas": banked_shas,
        "verdict": "PASS" if ok else "FAIL",
    }
    if out_path is not None:
        FS._write_json_atomic(out_path, record)
    print(f"[gate] G-S verdict={record['verdict']}", flush=True)
    if not ok:
        logger.error("[gate] G-S split-sha FAIL")
        sys.exit(RC_GS)
    return record


def _gate_rows(checks: dict, *, out_path: Path | None = None) -> dict:
    """G-A (plan §7; equality — banked arrays are production-grain, binds at
    smoke): refit holdout rows == assembled holdout order == ib_c rows.
    FAIL => record written FIRST, then sys.exit(RC_GA)."""
    ok = all(bool(v) for v in checks.values())
    record = {"gate": "G-A", "checks": checks, "verdict": "PASS" if ok else "FAIL"}
    if out_path is not None:
        FS._write_json_atomic(out_path, record)
    print(f"[gate] G-A verdict={record['verdict']} {json.dumps(checks)}", flush=True)
    if not ok:
        logger.error("[gate] G-A row-alignment FAIL: %s", json.dumps(checks))
        sys.exit(RC_GA)
    return record


def _gate_identity(
    counts_total: int, l0_total: int, *, n_rows: int, out_path: Path | None = None
) -> dict:
    """G-C'(i) (plan §7): the sum-accounting identity — sum(per-feature counts)
    == sum(per-row L0), both accumulated in the SAME R4 pass (EXACT integers;
    the only place exactness is genuinely exact — A2). Binds at smoke too (the
    identity is n-independent). FAIL => record written FIRST, sys.exit(RC_GC)."""
    ok = int(counts_total) == int(l0_total)
    record = {
        "gate": "G-C'(i)",
        "sum_per_feature_counts": int(counts_total),
        "sum_per_row_l0": int(l0_total),
        "n_rows": int(n_rows),
        "verdict": "PASS" if ok else "FAIL",
    }
    if out_path is not None:
        FS._write_json_atomic(out_path, record)
    print(
        f"[gate] G-C'(i) counts={counts_total} l0={l0_total} verdict={record['verdict']}",
        flush=True,
    )
    if not ok:
        logger.error("[gate] G-C'(i) sum-accounting identity FAIL")
        sys.exit(RC_GC)
    return record


def _gate_fit_rows(n_fit: int, *, production: bool, out_path: Path | None = None) -> dict:
    """G-C'(iii) (plan §7): n_fit_rows == 120,000 re-asserted against the
    sha-pinned sae_fit pool (exact; production-n — informational at smoke).
    FAIL => record written FIRST, then sys.exit(RC_GC) at production."""
    ok = int(n_fit) == N_FIT_PRODUCTION
    record = {
        "gate": "G-C'(iii)",
        "n_fit_rows": int(n_fit),
        "expected": N_FIT_PRODUCTION,
        "verdict": ("PASS" if ok else "FAIL") if production else "INFORMATIONAL-smoke",
        "production": bool(production),
    }
    if out_path is not None:
        FS._write_json_atomic(out_path, record)
    print(f"[gate] G-C'(iii) n_fit={n_fit} verdict={record['verdict']}", flush=True)
    if production and not ok:
        logger.error("[gate] G-C'(iii) fit-row count FAIL")
        sys.exit(RC_GC)
    return record


def _budget_lattice(finest_alive: int, *, val_l0: float, production: bool) -> dict:
    """The REGISTERED budget lattice, verbatim from plan §3 (DISJOINT and
    exhaustive): Budget-limited <=> k200-finest-alive >= 10; Budget-discharged
    <=> otherwise. Registered at the 1% floor ONLY; informational at smoke n.

    r8 U7 — the §3 manipulation check rides the RECORD (branch DEFINITIONS
    unchanged): plan §3's lattice-evaluation note conditions the CAUSAL
    narration of either budget branch on the realized k=200 val L0 materially
    exceeding the parent's 98.96; the record carries val_l0 + the concrete
    predicate + manipulation_realized so the analyzer (who owns the prose) can
    route a non-moving L0 to "manipulation not realized" mechanically."""
    verdict = (
        "Budget-limited" if int(finest_alive) >= BUDGET_FINEST_ALIVE_MIN else "Budget-discharged"
    )
    realized = bool(float(val_l0) >= MANIPULATION_VAL_L0_MIN)
    return {
        "rule": "Budget-limited <=> k200-finest-alive >= 10; Budget-discharged <=> otherwise",
        "threshold": BUDGET_FINEST_ALIVE_MIN,
        "finest_tier_ids": [16384, 65536],
        "finest_alive_at_registered_floor": int(finest_alive),
        "registered_floor_rows": REGISTERED_FLOOR,
        "verdict": verdict if production else f"INFORMATIONAL-smoke ({verdict})",
        "registered": bool(production),
        "val_l0": float(val_l0),
        "manipulation_realized": realized,
        "manipulation_check": {
            "predicate": (
                f"val_l0 >= {MANIPULATION_VAL_L0_MIN} (conservative concrete form of plan §3's "
                f"'realized k=200 val L0 materially exceeding the parent's {PARENT_VAL_L0_K100}'"
                " — the A4 lattice-evaluation note)"
            ),
            "threshold_val_l0": MANIPULATION_VAL_L0_MIN,
            "parent_k100_val_l0": PARENT_VAL_L0_K100,
            "narration": (
                "conditional-on-manipulation: the CAUSAL (budget) narration of this branch is "
                "licensed only when manipulation_realized; otherwise the analyzer reports "
                "'manipulation not realized' (census still valid and reported)"
            ),
        },
    }


def _finish_floor_row(stats: dict, demotion: dict, *, floor: int, n_fit: int, registered: bool):
    """One per-floor row from a _tier_stats result + the A13 demotion doc (the
    floor-sweep _finish_floor_row shape with THIS round's lattice note: the
    parent-lattice evaluation is reported at EVERY floor, 1% included — the
    gradient verdict-of-record stays the parent's k=100 cell, plan §3)."""
    row = {k: v for k, v in stats.items() if k != "lattice_verdict"}
    row["floor_rows"] = int(floor)
    row["floor_frac_of_fit_rows"] = float(floor) / float(n_fit)
    row["registered_cell"] = bool(registered)
    row["undefined_r2"] = demotion["per_tier"]
    row["not_evaluable_census_only"] = bool(demotion["not_evaluable_census_only"])
    row["demotion_rule"] = demotion["rule"]
    if not demotion["not_evaluable_census_only"]:
        row["lattice_reported"] = stats["lattice_verdict"]
        row["lattice_note"] = LATTICE_NOTE_K200 + (
            "; this floor is the BUDGET lattice's registered census cell" if registered else ""
        )
    return row


# ── census kernel (the floor-sweep counts+sums loop + a per-row L0 accumulator) ──


def _encode_counts_sums_l0(sae, mm, positions: np.ndarray, chunk: int = 4096, tag: str = ""):
    """The floor-sweep _encode_counts_sums streaming loop VERBATIM + a per-ROW
    L0 accumulator reduced along the OTHER axis (G-C'(i): the two totals must
    agree exactly — both come from the same encode pass). Counts on
    TRUE-summary encodes only. Returns (counts, sums, l0_total)."""
    import torch

    with torch.no_grad():
        counts = torch.zeros(sae.dict_size, dtype=torch.int64, device=sae.device)
        sums = torch.zeros(sae.dict_size, dtype=torch.float64, device=sae.device)
        l0_total = torch.zeros((), dtype=torch.int64, device=sae.device)
        pos = np.sort(np.asarray(positions, np.int64))
        t0 = time.time()
        n_chunks = max(1, (len(pos) + chunk - 1) // chunk)
        for i, s in enumerate(range(0, len(pos), chunk)):
            x = torch.as_tensor(np.asarray(mm[pos[s : s + chunk]], np.float32), device=sae.device)
            f = sae.encode(x, chunk=chunk)
            active = f > 0
            counts += active.sum(0)
            l0_total += active.sum(1).sum()  # per-row L0, reduced along the row axis
            sums += f.to(torch.float64).sum(0)
            if (i + 1) % 10 == 0 or i + 1 == n_chunks:
                print(
                    f"[k200_densein] counts{tag} chunk {i + 1}/{n_chunks} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
        return counts.cpu().numpy(), sums.cpu().numpy(), int(l0_total.cpu())


def _floors_eff(n_fit: int) -> list[int]:
    """Smoke-clamped effective floors (production: FLOORS verbatim)."""
    return [max(1, min(f, int(n_fit))) for f in FLOORS]


def _tier_quantiles(counts: np.ndarray, n_fit: int, drv) -> dict:
    """FS._tier_quantiles with empty-tier tolerance: a narrowed --sae-dict smoke
    width leaves the production tier id ranges (2,048+) EMPTY, and np.quantile
    crashes on an empty slice; production width (65,536) populates every tier,
    so the production numbers are identical to the FS kernel's."""
    bounds = (0,) + tuple(drv.S.MATRYOSHKA_TIER_BOUNDS)
    qs = np.asarray([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    out = {"q_grid": qs.tolist()}
    frac = np.asarray(counts, np.float64) / max(1, n_fit)
    for t in (0, 1, 2):
        seg = frac[bounds[t] : bounds[t + 1]]
        out[f"t{t}"] = np.quantile(seg, qs).tolist() if len(seg) else []
    return out


# ── R1: assemble (parent phase VERBATIM, end-to-end at the pin — B1) ─────────────


def phase_assemble(args) -> None:
    """R1: the parent's phase_assemble on the round's out-root (X19/Y19 fp16
    memmaps re-assembled from the 1,920 pinned chunks; split shas re-asserted;
    realized counts reconciled). As of the B1 diff the parent's own fetch path
    lists + downloads at revision DATA_REPO_REVISION and the stream-resume
    fingerprint includes it (a resume refuses a different source revision)."""
    _drv().phase_assemble(_parent_args(args, "assemble"))


# ── R2: stage banked inputs + G-S + G-A ──────────────────────────────────────────


def _stage_banked_files(stage: Path) -> None:
    """Idempotent revision-pinned staging (the floor-sweep _stage_banked_files
    pattern; per-file hf_hub_download at DATA_REPO_REVISION)."""
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
        print(f"[k200_stage_banked] staged {f}", flush=True)


def phase_stage_banked(args) -> None:
    """R2: stage the banked inputs at revision DATA_REPO_REVISION, assert the
    consumed schemas, then gate G-S (assembled split shas == banked split_meta
    records) and gate G-A (refit holdout rows == assembled holdout order ==
    ib_c rows). Both bind at smoke (shas are pool-grain; banked arrays are
    production-grain). Mismatch => sys.exit(RC_GS / RC_GA)."""
    drv = _drv()
    drv.C.phase("k200_stage_banked")
    sent_dir = args.out_root / "sentinels"
    sent_dir.mkdir(parents=True, exist_ok=True)
    done_path = sent_dir / "stage_banked.done.json"
    regime, resume_ok = _enter_regime(args, "stage_banked", stale_paths=[done_path])
    if resume_ok and done_path.exists():
        logger.info("[k200_stage_banked] resume: done-file present under matching regime; skip")
        return
    assert drv.C.HF_DATA_REPO == DATA_REPO, (drv.C.HF_DATA_REPO, DATA_REPO)
    stage = _stage_banked_dir(args)
    stage.mkdir(parents=True, exist_ok=True)
    _stage_banked_files(stage)

    ibz = np.load(stage / TA / "eval" / "ib_c.npz")
    FS._assert_npz_keys(ibz, ("rows", "pred16"), "ib_c")
    hz = np.load(stage / REFIT_HF)
    FS._assert_npz_keys(hz, ("holdout_pred16", "holdout_rows"), "refit_holdout")

    # ── G-S: assembled split shas vs the banked split_meta records ────────────────
    a_dir = args.out_root / "assemble"
    assembled = json.loads((a_dir / "split_meta.json").read_text())
    banked_meta = json.loads((stage / TA / "split_meta" / "split_meta.json").read_text())
    _gate_splits(assembled["shas"], banked_meta["shas"], out_path=_gates_dir(args) / "gs.json")

    # ── G-A: row alignment (refit holdout == assembled holdout order == ib rows) ──
    pargs = _parent_args(args, "stage_banked")
    _, _, pools = drv._load_scratch_meta(pargs)  # stages + sha-asserts the pinned pools
    hold_assembled = np.asarray(pools["holdout"], np.int64)  # EA holdout ORDER (as stored)
    hold_rows = np.asarray(hz["holdout_rows"], np.int64)
    ib_rows = np.asarray(ibz["rows"], np.int64)
    checks = {
        "refit_rows_eq_ib_rows": bool(np.array_equal(hold_rows, ib_rows)),
        "refit_rows_eq_assembled_holdout": bool(np.array_equal(hold_rows, hold_assembled)),
    }
    ga = _gate_rows(checks, out_path=_gates_dir(args) / "ga.json")

    FS._write_json_atomic(
        done_path,
        {"regime": regime, "staged": list(STAGE_FILES), "ga": ga["verdict"]},
    )
    _sentinel("k200-stage-banked", f"R2 done ({len(STAGE_FILES)} staged inputs; G-S/G-A PASS)")
    logger.info("[k200_stage_banked] done")


# ── R3: k=200 SAE training (parent phase with sae_k=200) ─────────────────────────


def phase_sae_train(args) -> None:
    """R3: the parent's phase_sae_train with the spoofed namespace (sae_k=200):
    identical training code path — per-epoch ckpt + regime-keyed resume (k in
    the regime), gate G-4' (the parent G4 recipe: held-out 10k-row var-FVE >=
    0.5, rc=25, record-written-first), fail-loud exact-set upload of
    weights+cfg+train_log to HF <hf_prefix>/sae_c_k200/ (production). Exit
    asserts cfg k == 200 + the sae_c_k200 leaf/prefix, and reports val L0 (the
    §3 manipulation-check input) + per-tier dead fractions beside the
    committed k=100 row (read at run time, never retyped)."""
    _require_upstream_gates(args, "sae_train")  # r8 U2: recorded FAILs re-apply first
    drv = _drv()
    pargs = _parent_args(args, "sae_train")
    # the accidental-overwrite fence (plan §8): the k-aware leaf must be
    # sae_c_k200 BEFORE any training/upload — never the parent's banked sae_c
    assert drv._sae_leaf(pargs) == f"sae_c_k{SAE_K200}", drv._sae_leaf(pargs)
    drv.phase_sae_train(pargs)

    out = drv._sae_out_dir(pargs)
    assert out.name == f"sae_c_k{SAE_K200}", out
    cfg = json.loads((out / "cfg.json").read_text())
    assert int(cfg["k"]) == SAE_K200, f"R3 exit assert: cfg k={cfg['k']} != {SAE_K200}"
    log = json.loads((out / "train_log.json").read_text())
    g4 = json.loads((out / "gates_p4.json").read_text())["g4"]
    _record_gate(args, "g4_k200", g4)
    last = log["epochs"][-1]
    k100_dead = None
    k100_log_path = EV_TURNAVG / "train_log.json"
    if k100_log_path.exists():  # committed comparison row (read, never retyped)
        k100_dead = json.loads(k100_log_path.read_text())["epochs"][-1]["dead_frac_by_tier"]
    print(
        "[k200_sae_train] companion report: "
        + json.dumps(
            {
                "val_l0": last["val_l0"],  # manipulation check: expected ~k=200 (parent 98.96)
                "val_var_fve": last["val_var_fve"],
                "dead_frac_by_tier_k200": last["dead_frac_by_tier"],
                "dead_frac_by_tier_k100_committed": k100_dead,
                "g4_verdict": g4["verdict"],
            }
        ),
        flush=True,
    )
    _sentinel(
        "k200-sae-train",
        f"R3 done (k=200 fve={last['val_var_fve']} l0={last['val_l0']} g4={g4['verdict']})",
    )


# ── R4: fit-side census pass + union encodes + dense-input companion ─────────────


def _record_densein_drop(args, exc: BaseException) -> dict:
    """r8 U6: the sanctioned R4 companion DROP persists its FULL diagnostic
    reason_chain to BOTH the census-side densein_dropped.json (the resume
    predicate's marker) AND the gates dir — the R6 consolidation folds every
    gates-dir record into the git-destined gates_k200.json, so the complete
    chain survives pod teardown (census/ is ephemeral and densein_meta.json
    strips the chain by design)."""
    fit_doc = {
        "dropped": True,
        # not a gate verdict: never "FAIL", so the r8 U2 loader cannot re-apply
        # a halt off a sanctioned drop (plan §7: DROP, never a round abort)
        "verdict": "DROPPED-companion",
        "reason": f"{type(exc).__name__}: {exc}",
        "reason_chain": "".join(traceback.format_exception(exc)),
    }
    FS._write_json_atomic(_census_dir(args) / "densein_dropped.json", fit_doc)
    _record_gate(args, "densein_dropped", fit_doc)
    return fit_doc


def phase_densein(args) -> None:
    """R4 (plan §4): (1) fit-side FULL-WIDTH streaming census pass (counts +
    sums + per-row L0) over the sae_fit rows with the k=200 SAE + gate G-C'(i)
    (sum-accounting identity, EXACT, asserted at the end of the pass); (2) the
    union-of-all-floors alive column set derived from THESE counts and supplied
    explicitly to the fit (census-pass -> fit, inside R4 — A3); (3) restricted
    encode at the union columns over concat(sae_fit, holdout) rows (ONE memmap:
    fit block = ridge target + the G-C'(ii) recount input; holdout block IS
    f_true for R5/R6); (4) the dense-input companion ridge c19 -> f_true(k200)
    on the UNION columns via the direct _gram_ridge_single drive (parent
    lambda-grid + PROD_VAL_CARVE/CARVE_SEED recipe — NEVER the verbatim
    _dense_companion_c own-panel path). A step-4 failure DROPS the reported
    companion + the densein retrieval source (recorded), never the round."""
    _require_upstream_gates(args, "densein")  # r8 U2: recorded FAILs re-apply first
    drv = _drv()
    drv.C.phase("k200_densein")
    ev = _eval_dir(args)
    cz = _census_dir(args)
    ev.mkdir(parents=True, exist_ok=True)
    cz.mkdir(parents=True, exist_ok=True)
    census_path = ev / "firing_census_k200.npz"
    densein_path = ev / "perfeature_k200_densein.npz"
    dropped_path = cz / "densein_dropped.json"
    dropped_gate_path = _gates_dir(args) / "densein_dropped.json"
    ftrue_path = cz / "ftrue_union_k200.fp16.npy"
    rows_path = cz / "union_rows.npz"
    stale = [census_path, densein_path, dropped_path, dropped_gate_path, ftrue_path, rows_path]
    regime, resume_ok = _enter_regime(args, "densein", stale_paths=stale)
    core_done = census_path.exists() and ftrue_path.exists() and rows_path.exists()
    if resume_ok and core_done and (densein_path.exists() or dropped_path.exists()):
        logger.info("[k200_densein] resume: outputs present under matching regime; skip")
        return
    drv.EA._headroom(args.out_root, 2 if args.smoke else 8, "k200-densein")
    production = _production(args)
    pargs = _parent_args(args, "densein")
    a_dir = args.out_root / "assemble"
    stage = _stage_banked_dir(args)

    # r8 M1: the R4 input contract validates BEFORE MatryoshkaBatchTopKSAE
    # .load_local(..., device=...) — missing/corrupt inputs fail pre-GPU-init.
    sae_dir = drv._sae_out_dir(pargs)
    required_in = [
        sae_dir / "sae_weights.safetensors",
        sae_dir / "cfg.json",
        a_dir / "rows_present.npy",
        a_dir / "Y19.fp16.npy",
        a_dir / "X19.fp16.npy",
        stage / REFIT_HF,
    ]
    missing_in = [str(p) for p in required_in if not p.exists()]
    assert not missing_in, (
        f"[k200_densein] inputs missing BEFORE the SAE load (r8 M1): {missing_in}"
    )
    _row_ci, prov_u8, pools = drv._load_scratch_meta(pargs)
    rows_present = np.load(a_dir / "rows_present.npy")
    y_mm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")
    hz = np.load(stage / REFIT_HF)
    FS._assert_npz_keys(hz, ("holdout_pred16", "holdout_rows"), "refit_holdout")
    sae = drv.MatryoshkaBatchTopKSAE.load_local(sae_dir, device=args.device)
    assert int(sae.k) == SAE_K200, f"loaded instrument k={sae.k} != {SAE_K200}"

    # ── step 1: full-width counts + sums + per-row L0 over the sae_fit rows ───────
    sae_fit = np.sort(np.asarray(pools["sae_fit"], np.int64))
    fit_pos, _ = FS._local_positions(rows_present, sae_fit, production)
    n_fit_used = int(len(fit_pos))
    assert n_fit_used >= 2, "census needs >= 2 fit rows (smoke slice too small)"
    t0 = time.time()
    counts, sums, l0_total = _encode_counts_sums_l0(sae, y_mm, fit_pos, tag=" k200")
    counts_wall = round(time.time() - t0, 1)
    _gate_identity(
        int(counts.sum()),
        l0_total,
        n_rows=n_fit_used,
        out_path=_gates_dir(args) / "gc_i.json",
    )
    floors_eff = _floors_eff(n_fit_used) if not production else list(FLOORS)
    np.savez(
        census_path,
        counts=counts.astype(np.int64),
        sums=sums.astype(np.float64),
        l0_total=np.int64(l0_total),
        n_fit_rows=np.int64(n_fit_used),
        floors=np.asarray(FLOORS, np.int64),
        floors_eff=np.asarray(floors_eff, np.int64),
        counts_canonical_note=(
            "no banked k=200 reference exists by construction: recomputed counts canonical "
            "(plan §7 G-C'); the k=100 banked census is a different instrument's comparison arm"
        ),
        **{
            f"quantiles_{k}": np.asarray(v)
            for k, v in _tier_quantiles(counts, n_fit_used, drv).items()
        },
    )

    # ── step 2: union-of-all-floors alive columns FROM THESE COUNTS (A3) ──────────
    union = np.where(counts >= min(floors_eff))[0].astype(np.int64)
    if len(union) == 0 and not production:
        # SMOKE-ONLY yield relaxation (gate-calibration rule, gotchas.md #1345:
        # floors_eff clamp to ALL fit rows at tiny n — a structurally
        # unsatisfiable smoke floor): any NONZERO yield proceeds, LOUDLY. The
        # production union rule above stays byte-identical (module-docstring
        # blind-spot enumeration item).
        union = np.where(counts > 0)[0].astype(np.int64)
        logger.warning(
            "[k200_densein] SMOKE union fallback: no feature cleared the clamped floor %d "
            "over %d rows; using %d ever-fired features (production rule untouched)",
            min(floors_eff),
            n_fit_used,
            len(union),
        )
    if len(union) == 0:  # fail LOUD on an empty selection (code-style rule)
        raise RuntimeError(
            f"[k200_densein] EMPTY union alive set at floor {min(floors_eff)} of "
            f"{n_fit_used} fit rows — the k=200 instrument fired no feature above "
            "the loosest floor; halt-investigate (never a silent empty artifact)"
        )

    # ── step 3: restricted f_true encode over concat(fit, holdout) rows ───────────
    hold_rows = np.asarray(hz["holdout_rows"], np.int64)
    local_hold, sel = FS._local_positions(rows_present, hold_rows, production)
    n_te = int(len(local_hold))
    assert n_te >= 2, "need >= 2 holdout rows in the assembled slice"
    rows_c = np.concatenate([fit_pos, local_hold])
    yc = np.lib.format.open_memmap(
        ftrue_path, mode="w+", dtype=np.float16, shape=(len(rows_c), len(union))
    )
    t0 = time.time()
    drv._encode_restricted(sae, y_mm, rows_c, union, out_mm=yc)
    yc.flush()
    encode_wall = round(time.time() - t0, 1)
    np.savez(
        rows_path,
        cols=union,
        hold_rows=hold_rows[sel],
        te_prov=np.asarray(prov_u8, np.uint8)[hold_rows[sel]],
        n_fit=np.int64(n_fit_used),
        floors_eff=np.asarray(floors_eff, np.int64),
    )

    # ── step 4: the dense-input companion ridge on the UNION columns (droppable) ──
    X = np.load(a_dir / "X19.fp16.npy", mmap_mode="r")
    Xc = np.asarray(X[rows_c], np.float16)
    carve = min(int(drv.M.PROD_VAL_CARVE), max(1, n_fit_used // 6))
    perm = np.random.default_rng(int(drv.M.CARVE_SEED)).permutation(n_fit_used)
    va, tr = perm[:carve], perm[carve:]
    te = np.arange(n_fit_used, len(rows_c))
    drv.EL._assert_estimator_validity(len(tr), Xc.shape[1], args.smoke)
    try:
        t0 = time.time()
        pt, meta = drv._gram_ridge_single(Xc, yc, tr, va, te, drv.N1M.LAMBDAS_N1M, args.device)
        # SHARED_TMP_EXEMPT: vendored verbatim from #2476 pin d8e9f8bdd4 (parent copy is batch-0 allowlisted); single-writer per out-root, migration deferred to parent
        tmp = densein_path.parent / f".tmp_{densein_path.name}"
        np.savez(
            tmp,
            pred16=pt.astype(np.float16),
            feat_ids=union,
            rows=hold_rows[sel],
            selected_lambda=np.float64(meta["selected_lambda"]),
            val_r2=np.float64(meta["val_r2"]),
            lambda_grid_edge=np.bool_(meta["lambda_grid_edge"]),
        )
        tmp.replace(densein_path)
        for p in (dropped_path, dropped_gate_path):
            if p.exists():  # a successful (re)fit clears any prior drop record
                p.unlink()
        fit_doc = {"dropped": False, "wall_s": round(time.time() - t0, 1), **meta}
        print(f"[k200_densein] unit densein done {json.dumps(meta)}", flush=True)
    except SystemExit:
        raise  # a gate halt is never a droppable fit failure
    except Exception as e:  # plan §7: fit failure = DROP of the companion, never abort
        fit_doc = _record_densein_drop(args, e)  # r8 U6: full chain -> gates dir too
        logger.error(
            "[k200_densein] dense-companion fit DROPPED (reported companion + densein "
            "retrieval source lost; plan §7 names this in the fold): %s",
            fit_doc["reason"],
        )
    del sae
    drv._write_json(
        cz / "densein_meta.json",
        {
            "regime": regime,
            "n_fit_used": n_fit_used,
            "n_te_used": n_te,
            "n_union": int(len(union)),
            "floors_eff": floors_eff,
            "timings_s": {"counts": counts_wall, "encode_union": encode_wall},
            "densein": {k: v for k, v in fit_doc.items() if k != "reason_chain"},
            "smoke_clamped": not production,
        },
        phase="k200-densein",
    )
    _sentinel(
        "k200-densein",
        f"R4 done (n_union={len(union)}; densein dropped={fit_doc.get('dropped')})",
    )
    logger.info("[k200_densein] done")


# ── R5: census JSON + score-side encodes + pred_encode_fve + G-C'(ii)/(iii) ──────


def _recount_fit_active(yc, n_fit: int, chunk: int = 8192) -> np.ndarray:
    """Fit-side active-count recount over the FIRST ``n_fit`` rows of a
    concat(fit, holdout) store. The chunk slice end is CLAMPED at ``n_fit``:
    numpy clamps a past-the-end slice at the ARRAY end, so an unclamped final
    chunk (``yc[s : s + chunk]``) spills ``(-n_fit) % chunk`` holdout rows into
    the fit-side recount (r3 crash-fix: 2,880 spill rows -> +2,832 on saturated
    features -> deterministic G-C'(ii) rc=31). Returns per-column int64 counts
    of strictly-positive entries; pinned by tests/test_issue2476_k200.py."""
    counts = np.zeros(yc.shape[1], np.int64)
    for s in range(0, n_fit, chunk):
        counts += (np.asarray(yc[s : min(s + chunk, n_fit)]) > 0).sum(0).astype(np.int64)
    return counts


def phase_census(args) -> None:
    """R5 (plan §4): (a) per-floor alive sets + the registered BUDGET-lattice
    census input (census_k200.json) from the R4-persisted counts; (b)
    score-side restricted encodes at the union columns (f_pred =
    SAE_k200(holdout_pred16), f_ib = SAE_k200(ib_c)); (c) the B2
    pred_encode_fve producer — ONE full-width encode+decode var-FVE over the
    holdout_pred16 rows via _recon_fve; (d) gate G-C' — (ii) R4 full-width
    counts vs the union-column restricted-encode recount under the
    boundary-tolerant _gate_counts form (VERBATIM floor-sweep reuse, A2), and
    (iii) n_fit_rows == 120,000 (exact, production)."""
    _require_upstream_gates(args, "census")  # r8 U2: recorded FAILs re-apply first
    drv = _drv()
    drv.C.phase("k200_census")
    ev = _eval_dir(args)
    cz = _census_dir(args)
    ev.mkdir(parents=True, exist_ok=True)
    census_json = ev / "census_k200.json"
    meta_json = ev / "union_encodes_meta.json"
    enc_path = cz / "union_encodes_k200.npz"
    finals = [census_json, meta_json, enc_path]
    regime, resume_ok = _enter_regime(args, "census", stale_paths=finals)
    if resume_ok and all(p.exists() for p in finals):
        logger.info("[k200_census] resume: outputs present under matching regime; skip")
        return
    production = _production(args)
    pargs = _parent_args(args, "census")
    stage = _stage_banked_dir(args)

    # ── R5 input contract (r8 M1: validates BEFORE the GPU-init SAE load) ─────────
    fc = np.load(ev / "firing_census_k200.npz")
    counts = np.asarray(fc["counts"], np.int64)
    n_fit = int(fc["n_fit_rows"])
    floors_eff = [int(f) for f in fc["floors_eff"]]
    rz = np.load(cz / "union_rows.npz")
    union = np.asarray(rz["cols"], np.int64)
    hold_kept = np.asarray(rz["hold_rows"], np.int64)
    te_prov = np.asarray(rz["te_prov"], np.uint8)
    yc = np.load(cz / "ftrue_union_k200.fp16.npy", mmap_mode="r")
    assert yc.shape == (n_fit + len(hold_kept), len(union)), (yc.shape, n_fit, len(hold_kept))
    hz = np.load(stage / REFIT_HF)
    FS._assert_npz_keys(hz, ("holdout_pred16", "holdout_rows"), "refit_holdout")
    ibz = np.load(stage / TA / "eval" / "ib_c.npz")
    FS._assert_npz_keys(ibz, ("rows", "pred16"), "ib_c")
    hold_rows = np.asarray(hz["holdout_rows"], np.int64)
    rows_present = np.load(args.out_root / "assemble" / "rows_present.npy")
    _, sel = FS._local_positions(rows_present, hold_rows, production)
    assert np.array_equal(hold_rows[sel], hold_kept), "R4/R5 holdout row drift"
    sae_train_log = drv._sae_out_dir(pargs) / "train_log.json"
    assert sae_train_log.exists(), f"[k200_census] R3 train_log missing: {sae_train_log}"

    sae = drv.MatryoshkaBatchTopKSAE.load_local(drv._sae_out_dir(pargs), device=args.device)
    assert int(sae.k) == SAE_K200, f"loaded instrument k={sae.k} != {SAE_K200}"

    # ── (d.ii): full-width census counts vs the restricted-encode recount ─────────
    recount = _recount_fit_active(yc, n_fit)
    # VERBATIM floor-sweep _gate_counts (A2): reference = the R4 full-width pass
    # counts at the union columns (self-consistency — no banked k=200 census
    # exists by construction); tol_rows = GC_COUNT_TOL, off-boundary sym-diffs
    # halt. Exits FS.RC_GC (= RC_GC = 31) at production on FAIL. The record
    # carries float(sae.threshold) (r8 M2: near-theta flips are exactly what
    # tol absorbs — one-look diagnosis, FAIL path included; tol=3 KEPT).
    FS._gate_counts(
        recount,
        counts[union],
        floors_eff,
        arm="k200_selfconsistency",
        production=production,
        out_path=_gates_dir(args) / "gc_ii.json",
        extra={"sae_threshold": float(sae.threshold)},
    )
    # ── (d.iii): n_fit == 120,000 vs the sha-pinned pool (exact, production) ──────
    _gate_fit_rows(n_fit, production=production, out_path=_gates_dir(args) / "gc_iii.json")

    # ── (b): score-side restricted encodes + (c): pred_encode_fve ─────────────────
    vhat = np.asarray(hz["holdout_pred16"], np.float16)[sel]
    ib16 = np.asarray(ibz["pred16"], np.float16)[sel]
    t0 = time.time()
    f_pred = drv._encode_restricted(sae, vhat, np.arange(len(vhat)), union)
    f_ib = drv._encode_restricted(sae, ib16, np.arange(len(ib16)), union)
    encode_wall = round(time.time() - t0, 1)
    f_true = np.asarray(yc[n_fit:], np.float16)
    # SHARED_TMP_EXEMPT: vendored verbatim from #2476 pin d8e9f8bdd4 (parent copy is batch-0 allowlisted); single-writer per out-root, migration deferred to parent
    tmp = enc_path.parent / f".tmp_{enc_path.name}"
    np.savez(
        tmp,
        rows=hold_kept,
        cols=union,
        f_true=f_true,
        f_pred=f_pred,
        f_ib=f_ib,
        te_prov=te_prov,
    )
    tmp.replace(enc_path)

    t0 = time.time()
    pe_fve, pe_l0 = drv._recon_fve(sae, vhat, np.arange(len(vhat)))
    pe_record = {
        "pred_encode_fve": round(float(pe_fve), 6),
        "pred_encode_mean_l0": round(float(pe_l0), 2),
        "n_rows": int(len(vhat)),
        "wall_s": round(time.time() - t0, 1),
        "k100_reference": {
            "value": 0.937,
            "source": "parent promoted body (no committed JSON carries it); "
            "DV1 disposition comparison is the analyzer's (plan §6 — reported, never a gate)",
        },
    }
    _record_gate(args, "pred_encode_fve", pe_record)
    print(f"[k200_census] pred_encode_fve={pe_fve:.4f} l0={pe_l0:.1f}", flush=True)
    del sae

    # ── (a): the census JSON + the registered BUDGET lattice ──────────────────────
    tiers_full = drv.S.tier_of(np.arange(len(counts)))
    candidates = {str(t): int((tiers_full == t).sum()) for t in (0, 1, 2)}
    per_floor = {}
    for fl, fl_eff in zip(FLOORS, floors_eff, strict=True):
        alive = counts >= fl_eff
        per_floor[str(fl)] = {
            "floor_rows_effective": int(fl_eff),
            "n_alive": int(alive.sum()),
            "alive_by_tier": {str(t): int((alive & (tiers_full == t)).sum()) for t in (0, 1, 2)},
        }
    finest_alive = int(((counts >= floors_eff[0]) & (tiers_full == 2)).sum())
    # r8 U7: the §3 manipulation-check input, read from THIS run's train log at
    # run time (the R3 companion report's source) — rides the lattice record.
    k200_val_l0 = float(json.loads(sae_train_log.read_text())["epochs"][-1]["val_l0"])
    k100_committed: dict = {"source": "eval_results/issue_2476/floor_sweep/floor_sweep_c.json"}
    fs_path = EV_FLOOR_SWEEP / "floor_sweep_c.json"
    if fs_path.exists():  # committed comparison arm (read at run time, never retyped)
        fs_doc = json.loads(fs_path.read_text())
        k100_committed["alive_by_tier_per_floor"] = {
            str(r["floor_rows"]): r["alive_by_tier"] for r in fs_doc["rows"]
        }
    else:
        logger.warning("[k200_census] committed floor_sweep_c.json absent (comparison arm)")
    doc = {
        "regime": regime,
        "instrument": {"k": SAE_K200, "width": int(len(counts)), "seed": drv.SAE_SEED},
        "n_fit_rows": n_fit,
        "floors_rows": list(FLOORS),
        "floors_eff": floors_eff,
        "registered_floor_rows": REGISTERED_FLOOR,
        "candidates_by_tier": candidates,
        "per_floor": per_floor,
        "mean_firing": {
            "k200_realized_frac": float(counts.sum() / max(1, n_fit * len(counts))),
            "k200_design_frac": SAE_K200 / 65536.0,
            "k100_design_frac": 100.0 / 65536.0,
        },
        "budget_lattice": _budget_lattice(finest_alive, val_l0=k200_val_l0, production=production),
        "k100_committed": k100_committed,
        "census_note": (
            "census computed from raw threshold counts (cap-independent; the "
            "16,384-cap panel enters only the R6 medians per parent convention)"
        ),
    }
    drv._write_json(census_json, doc, phase="k200-census")
    drv._write_json(
        meta_json,
        {
            "regime": regime,
            "n_union": int(len(union)),
            "n_fit_used": n_fit,
            "n_te_used": int(len(hold_kept)),
            "floors_eff": floors_eff,
            "timings_s": {"score_encodes": encode_wall},
            "pred_encode_fve": pe_record["pred_encode_fve"],
            "smoke_clamped": not production,
        },
        phase="k200-census",
    )
    _sentinel(
        "k200-census",
        f"R5 done (finest_alive@1%={finest_alive} lattice="
        f"{doc['budget_lattice']['verdict']} pred_encode_fve={pe_record['pred_encode_fve']})",
    )
    logger.info("[k200_census] done: %s", doc["budget_lattice"]["verdict"])


# ── R6: per-floor stats + retrieval (3 sources) + gates consolidation ────────────


def _consolidate_gates(args) -> dict:
    """The gates_k200.json doc: EVERY gates-dir record verbatim (G-S/G-A/G-C'
    trio, g4_k200, the B2 pred_encode_fve field, and — r8 U6 — the full-chain
    densein_dropped record when the R4 companion was dropped, so the sanctioned
    DROP's complete reason_chain lands in the git-destined artifact)."""
    gates = {}
    for p in sorted(_gates_dir(args).glob("*.json")):
        gates[p.stem] = json.loads(p.read_text())
    return {"pred_encode_fve": gates.get("pred_encode_fve", {}), "gates": gates}


def phase_stats(args) -> None:
    """R6 (plan §4): per-floor batteries via the parent kernels — per-feature
    reads computed ONCE at the union grain (R2 map/ib/train-mean/densein + K=20
    shuffle nulls), then per floor: alive set from the R4 counts (recomputed
    counts canonical — no banked k=200 reference), the tier-stratified panel
    (cap caveat: census is cap-independent; if the cap/allocation binds, panel
    != clearing set and BOTH are reported — medians on the panel per parent
    convention, nesting claims restricted to clearing sets, A5), _tier_stats
    with a fresh rng(battery_seed), A13 demotion, retrieval with THREE sources
    (map / ib / densein — B3). Consolidates every gate record + the B2
    pred_encode_fve field into gates_k200.json."""
    _require_upstream_gates(args, "stats")  # r8 U2: recorded FAILs re-apply first
    drv = _drv()
    drv.C.phase("k200_stats")
    ev = _eval_dir(args)
    cz = _census_dir(args)
    finals = [
        ev / "tier_sweep_k200.json",
        ev / "retrieval_k200.json",
        ev / "perfeature_union_k200.npz",
        ev / "gates_k200.json",
    ]
    stale = [*finals, *ev.glob(".tmp_*")]
    regime, resume_ok = _enter_regime(args, "stats", stale_paths=stale)
    if resume_ok and all(p.exists() for p in finals):
        logger.info("[k200_stats] resume: outputs present under matching regime; skip")
        return
    production = _production(args)
    n_perm = min(args.n_perm, 200) if args.smoke else args.n_perm
    n_boot = min(args.n_boot, 200) if args.smoke else args.n_boot

    # stats seed READ from the committed parent provenance at run time (plan §10)
    battery_seed = int(json.loads((EV_TURNAVG / "tier_tests_c.json").read_text())["battery_seed"])

    fc = np.load(ev / "firing_census_k200.npz")
    counts = np.asarray(fc["counts"], np.int64)
    n_fit = int(fc["n_fit_rows"])
    floors_eff = [int(f) for f in fc["floors_eff"]]
    train_mean_full = np.asarray(fc["sums"], np.float64) / max(1, n_fit)
    uz = np.load(cz / "union_encodes_k200.npz")
    union = np.asarray(uz["cols"], np.int64)
    f_true = np.asarray(uz["f_true"], np.float16)
    f_pred = np.asarray(uz["f_pred"], np.float16)
    f_ib = np.asarray(uz["f_ib"], np.float16)
    te_prov = np.asarray(uz["te_prov"], np.uint8)

    densein_path = ev / "perfeature_k200_densein.npz"
    f_densein = None
    if densein_path.exists():
        dz = np.load(densein_path)
        assert np.array_equal(np.asarray(dz["feat_ids"], np.int64), union), (
            "densein feat_ids != union columns"
        )
        assert np.array_equal(np.asarray(dz["rows"], np.int64), np.asarray(uz["rows"], np.int64)), (
            "densein rows != score rows"
        )
        f_densein = np.asarray(dz["pred16"], np.float16)
    else:
        logger.warning("[k200_stats] densein companion ABSENT (dropped at R4): map/ib sources only")

    # ── union-grain per-feature reads, computed ONCE ──────────────────────────────
    r2 = {
        "map": drv._r2_only(f_pred, f_true),
        "ib": drv._r2_only(f_ib, f_true),
    }
    tm_u = train_mean_full[union]
    r2["trainmean"] = drv._r2_only(np.broadcast_to(tm_u.astype(np.float32), f_true.shape), f_true)
    nulls = {
        "map": drv._shuffle_null_r2(f_pred, f_true, drv.SHUFFLE_SEEDS_2476, what=" k200/map"),
        "ib": drv._shuffle_null_r2(f_ib, f_true, drv.SHUFFLE_SEEDS_2476, what=" k200/ib"),
    }
    if f_densein is not None:
        r2["densein"] = drv._r2_only(f_densein, f_true)
        nulls["densein"] = drv._shuffle_null_r2(
            f_densein, f_true, drv.SHUFFLE_SEEDS_2476, what=" k200/densein"
        )
    corpus_r2: dict[str, np.ndarray] = {}
    corpus_n: dict[str, int] = {}
    for label, code in (("lmsys", 0), ("wildchat", 1)):
        m = te_prov == code
        corpus_n[label] = int(m.sum())
        if int(m.sum()) >= 2:
            corpus_r2[label] = drv._r2_only(f_pred[m], f_true[m])
    tier_u = drv.S.tier_of(union)

    rows = []
    retr_rows: dict = {}
    lattice_vector = []
    masks: dict[str, np.ndarray] = {}
    for fl, fl_eff in zip(FLOORS, floors_eff, strict=True):
        t0 = time.time()
        alive = np.where(counts >= fl_eff)[0]
        panel, doc = drv.M._tier_stratified_panel(
            counts, 100 * fl_eff, int(drv.M.PANEL_CAP), int(drv.M.PANEL_SEED)
        )
        assert int(doc["floor"]) == fl_eff, (doc["floor"], fl_eff)
        panel = np.asarray(panel, np.int64)
        cap_bound = len(panel) != len(alive) or not np.array_equal(panel, alive)
        if cap_bound:  # plan §4 R6 cap caveat (A5): both reported, never narrated nested
            logger.warning(
                "[k200_stats] panel cap/allocation BINDS at floor %d (panel %d != clearing %d)",
                fl,
                len(panel),
                len(alive),
            )
        cols = np.searchsorted(union, panel)
        assert (union[cols] == panel).all(), "panel escapes the union columns"
        r2m, r2i, r2t = r2["map"][cols], r2["ib"][cols], r2["trainmean"][cols]
        tier = tier_u[cols]
        act = np.asarray(counts, np.float64)[panel]
        rng = np.random.default_rng(battery_seed)
        stats = drv._tier_stats(r2m, r2i, tier, act, n_perm, n_boot, rng)
        demo = FS._undefined_demotion(r2m, r2i, tier)
        row = _finish_floor_row(
            stats, demo, floor=fl, n_fit=n_fit, registered=(fl == REGISTERED_FLOOR)
        )
        row["floor_rows_effective"] = int(fl_eff)
        row["n_te_rows"] = int(f_true.shape[0])
        row["n_alive"] = int(len(alive))  # clearing set (census grain, cap-independent)
        row["alive_by_tier"] = {str(t): int((drv.S.tier_of(alive) == t).sum()) for t in (0, 1, 2)}
        row["panel"] = doc
        row["panel_cap_bound"] = bool(cap_bound)
        if cap_bound:
            row["panel_by_tier"] = {str(t): int((tier == t).sum()) for t in (0, 1, 2)}
            row["panel_note"] = (
                "panel != clearing set (cap/allocation bound): medians/permutation on the "
                "PANEL per parent convention, census on the clearing set; capped panels are "
                "fresh same-seed draws per floor — differently-sampled, never nested (A5)"
            )
        row["trainmean_per_tier_median_r2"] = {
            str(t): drv._median_of(r2t[tier == t]) for t in (0, 1, 2)
        }
        if f_densein is not None:
            r2d = r2["densein"][cols]
            row["densein_per_tier_median_r2"] = {
                str(t): drv._median_of(r2d[tier == t]) for t in (0, 1, 2)
            }
        else:
            row["densein_per_tier_median_r2"] = None
        shuffle_doc: dict = {
            "n_seeds": len(drv.SHUFFLE_SEEDS_2476),
            "advisory": True,
            "train_mean_note": "constant predictor: row-shuffle null == observed (no draws)",
            "per_read": {},
        }
        for rname, nu in nulls.items():
            obs = r2[rname][cols]
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
        preds = {"map": f_pred[:, cols], "ib": f_ib[:, cols]}
        if f_densein is not None:
            preds["densein"] = f_densein[:, cols]  # B3: the third retrieval source
        retr_rows[str(fl)] = {
            "n_alive": int(len(alive)),
            "n_panel": int(len(panel)),
            "retrieval_on": "panel",
            "tiers": drv._retrieval_cells(
                np.asarray(f_true[:, cols]),
                preds,
                tier,
                ks=(1, 5, 10),
                device=args.device,
            ),
        }
        masks[f"alive_f{fl}"] = np.isin(union, alive)
        masks[f"panel_f{fl}"] = np.isin(union, panel)
        rows.append(row)
        lattice_vector.append(
            {
                "floor_rows": int(fl),
                "label": row.get("lattice_reported", "not-evaluable-census-only"),
                "registered": bool(fl == REGISTERED_FLOOR),
            }
        )
        print(
            f"[k200_stats] unit f{fl} n_alive={len(alive)} n_panel={len(panel)} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    sweep_doc = {
        "arm": "k200",
        "floors_rows": [int(f) for f in FLOORS],
        "floors_eff": floors_eff,
        "n_fit_rows": n_fit,
        "registered_floor_rows": REGISTERED_FLOOR,
        "lattice_vector": lattice_vector,
        "lattice_note": LATTICE_NOTE_K200,
        "battery_seed": battery_seed,
        "seeds_note": "battery_seed read from the committed tier_tests_c provenance at run "
        "time; identical seeds/draw counts across floors by design",
        "n_perm": int(n_perm),
        "n_boot": int(n_boot),
        "alive_source": "recomputed k=200 counts (canonical — no banked reference exists; "
        "G-C' self-consistency gated, plan §7)",
        "densein_present": bool(f_densein is not None),
        "rows": rows,
    }
    drv._write_json(ev / "tier_sweep_k200.json", sweep_doc, phase="k200-stats")
    retrieval_doc = {
        "arm": "k200",
        "n_pool": int(f_true.shape[0]),
        "chance_note": "pool = held-out true feature rows; chance_at_k = k / n_pool",
        "sources": sorted(["map", "ib"] + (["densein"] if f_densein is not None else [])),
        "rows": retr_rows,
    }
    drv._write_json(ev / "retrieval_k200.json", retrieval_doc, phase="k200-stats")
    pf_arrays = {
        "feat_ids": union,
        "tier": tier_u,
        "counts": counts[union],
        "r2_map": r2["map"],
        "r2_ib": r2["ib"],
        "r2_trainmean": r2["trainmean"],
        "null_r2_map": nulls["map"],
        "null_r2_ib": nulls["ib"],
        "shuffle_seeds": np.asarray(drv.SHUFFLE_SEEDS_2476, np.int64),
        **masks,
    }
    if f_densein is not None:
        pf_arrays["r2_densein"] = r2["densein"]
        pf_arrays["null_r2_densein"] = nulls["densein"]
    for label, arr in corpus_r2.items():
        pf_arrays[f"r2_{label}"] = arr
    tmp = ev / ".tmp_perfeature_union_k200.npz"
    np.savez(tmp, **pf_arrays)
    tmp.replace(ev / "perfeature_union_k200.npz")

    drv._write_json(ev / "gates_k200.json", _consolidate_gates(args), phase="k200-stats")
    lat = [x["label"] for x in lattice_vector]
    _sentinel("k200-stats", f"R6 done (lattice_reported={lat})")
    logger.info("[k200_stats] done: %s", lat)


# ── R7: figures + git/HF legs + terminal sentinel ────────────────────────────────


def _floor_x(doc: dict) -> list[float]:
    return [100.0 * r["floor_rows"] / doc["n_fit_rows"] for r in doc["rows"]]


def _row_med_ci(row: dict, read: str, t: int):
    """(median, ci95) of a per-floor per-tier median read; None-safe."""
    pt = row["per_tier"][str(t)]
    med = pt[f"median_r2_{read}"].get("median")
    ci = pt.get(f"ci95_median_{read}")
    return med, ci


def _fig_census_hero(census: dict, fig_dir: Path, drv) -> None:
    """Hero 1: per-tier alive census, k=100 (committed) vs k=200, grouped bars
    at the registered 1% floor with the swept floors as light companions
    (log-y; denominators in the sidecar, never on canvas — §3.8-bis)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_full,
        paper_palette,
        savefig_paper,
    )

    colors = paper_palette(3)
    k100 = census.get("k100_committed", {}).get("alive_by_tier_per_floor", {})
    floors = [str(f) for f in census["floors_rows"]]
    fig, ax = plt.subplots(figsize=figsize_iclr_full(0.42))
    w = 0.09
    for t in (0, 1, 2):
        for fi, fl in enumerate(floors):
            alpha = 1.0 if int(fl) == census["registered_floor_rows"] else 0.4
            v100 = k100.get(fl, {}).get(str(t))
            v200 = census["per_floor"][fl]["alive_by_tier"][str(t)]
            x0 = t + (fi - (len(floors) - 1) / 2) * 2.2 * w
            if v100 is not None:
                ax.bar(x0 - w / 2, max(float(v100), 0.5), w, color="0.55", alpha=alpha)
            ax.bar(x0 + w / 2, max(float(v200), 0.5), w, color=colors[t], alpha=alpha)
    ax.set_yscale("log")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([drv.TIER_LABELS[t].splitlines()[0] for t in (0, 1, 2)], fontsize=6)
    ax.set_ylabel("alive features (log)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="0.55"),
        plt.Rectangle((0, 0), 1, 1, color=colors[0]),
    ]
    ax.legend(handles, ["k=100 (committed)", "k=200 (this round)"], fontsize=5, loc="best")
    savefig_paper(fig, "i2476_k200_census_hero", dir=fig_dir)
    plt.close(fig)
    print("[k200_figures] fig i2476_k200_census_hero", flush=True)


def _fig_tier_r2_hero(sweep: dict, k100_sweep: dict | None, fig_dir: Path, drv) -> None:
    """Hero 2: per-tier median held-out R2 vs floor (k=200 map solid + CI band,
    ib dashed) with the k=100 committed medians as labeled reference markers
    (different instrument — census-level comparison, never feature-paired)."""
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
        ax.plot(xs, med_map, "-o", color=colors[t], ms=2.5, lw=1.0, label="map (k=200)")
        ax.plot(xs, med_ib, "--s", color=colors[t], ms=2.0, lw=0.9, label="identity+bias (k=200)")
        if k100_sweep is not None:
            xs100 = _floor_x(k100_sweep)
            ref = []
            for row in k100_sweep["rows"]:
                m, _ = _row_med_ci(row, "map", t)
                ref.append(np.nan if m is None else m)
            ax.plot(
                xs100,
                ref,
                "x",
                color="0.45",
                ms=3.0,
                lw=0,
                label="map k=100 (committed; different instrument)",
            )
        ax.set_title(drv.TIER_LABELS[t].replace("\n", " "), fontsize=6)
        ax.set_xlabel("alive floor (% of fit rows)")
    axes[0].invert_xaxis()  # sharex: ONE inversion flips all panels (looser -> right)
    axes[0].set_ylabel("median held-out R²")
    axes[0].legend(fontsize=4.5, loc="best")
    savefig_paper(fig, "i2476_k200_tier_r2_hero", dir=fig_dir)
    plt.close(fig)
    print("[k200_figures] fig i2476_k200_tier_r2_hero", flush=True)


def _fig_dead_frac_epochs(k200_log: dict, k100_log: dict | None, fig_dir: Path, drv) -> None:
    """Exploratory: per-tier dead fraction by epoch, k=200 solid vs k=100 dashed."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_full,
        paper_palette,
        savefig_paper,
    )

    colors = paper_palette(3)
    fig, ax = plt.subplots(figsize=figsize_iclr_full(0.42))
    for t in (0, 1, 2):
        ys = [r["dead_frac_by_tier"].get(str(t)) for r in k200_log["epochs"]]
        xsn = [r["epoch"] for r in k200_log["epochs"]]
        if any(v is not None for v in ys):
            ax.plot(
                xsn,
                [np.nan if v is None else v for v in ys],
                "-o",
                color=colors[t],
                ms=2.5,
                lw=1.0,
                label=f"{drv.TIER_LABELS[t].splitlines()[0]} k=200",
            )
        if k100_log is not None:
            ys0 = [r["dead_frac_by_tier"].get(str(t)) for r in k100_log["epochs"]]
            xs0 = [r["epoch"] for r in k100_log["epochs"]]
            ax.plot(
                xs0,
                [np.nan if v is None else v for v in ys0],
                "--s",
                color=colors[t],
                ms=2.0,
                lw=0.8,
                label=f"{drv.TIER_LABELS[t].splitlines()[0]} k=100",
            )
    ax.set_xlabel("epoch")
    ax.set_ylabel("dead-feature fraction")
    ax.legend(fontsize=4.5, loc="best")
    savefig_paper(fig, "i2476_k200_dead_frac_epochs", dir=fig_dir)
    plt.close(fig)
    print("[k200_figures] fig i2476_k200_dead_frac_epochs", flush=True)


def _fig_firing_ecdf(census_npz: Path, fig_dir: Path, drv) -> None:
    """Exploratory: per-tier ECDF of fit-side firing fractions with both design
    mean-firing lines (100/65,536 and 200/65,536) and the four floors marked."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_full,
        paper_palette,
        savefig_paper,
    )

    colors = paper_palette(3)
    z = np.load(census_npz)
    counts = np.asarray(z["counts"], np.int64)
    n_fit = int(z["n_fit_rows"])
    tiers = drv.S.tier_of(np.arange(len(counts)))
    frac = counts / max(1, n_fit)
    fig, ax = plt.subplots(figsize=figsize_iclr_full(0.42))
    floor_x = max(frac[frac > 0].min() if (frac > 0).any() else 1e-7, 1e-7) / 3.0
    for t in (0, 1, 2):
        v = np.sort(frac[tiers == t])
        if not len(v):
            continue
        y = np.arange(1, len(v) + 1) / len(v)
        ax.plot(
            np.maximum(v, floor_x),
            y,
            color=colors[t],
            lw=0.9,
            label=drv.TIER_LABELS[t].splitlines()[0],
        )
    for k, ls in ((100, ":"), (200, "--")):
        ax.axvline(k / 65536.0, ls=ls, lw=0.7, color="gray", label=f"mean firing k={k}")
    for fl_eff in {int(f) for f in z["floors_eff"]}:
        ax.axvline(fl_eff / max(1, n_fit), ls="-", lw=0.4, color="0.8")
    ax.set_xscale("log")
    ax.set_xlabel("fit-side firing fraction (log)")
    ax.set_ylabel("fraction of features")
    ax.legend(fontsize=4.5, loc="best")
    savefig_paper(fig, "i2476_k200_firing_ecdf", dir=fig_dir)
    plt.close(fig)
    print("[k200_figures] fig i2476_k200_firing_ecdf", flush=True)


def _fig_finest_newly(sweep: dict, pf: dict, fig_dir: Path, drv) -> None:
    """Exploratory: finest-tier per-feature R2 (map read) distribution at each
    floor — ECDF panels (the newly-populated finest tier is the object)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_palette,
        savefig_paper,
    )

    color = paper_palette(3)[2]
    floors = [row["floor_rows"] for row in sweep["rows"]]
    r2 = np.asarray(pf["r2_map"], np.float64)
    tier = np.asarray(pf["tier"])
    fig, axes = plt.subplots(1, len(floors), figsize=figsize_iclr_panels(len(floors)), sharey=True)
    for ax, fl in zip(np.atleast_1d(axes), floors, strict=True):
        mask = np.asarray(pf[f"alive_f{fl}"], bool) & (tier == 2)
        v = r2[mask]
        v = np.sort(v[np.isfinite(v)])
        if len(v):
            y = np.arange(1, len(v) + 1) / len(v)
            ax.plot(v, y, color=color, lw=0.9)
        ax.set_xlim(-1.0, 1.0)
        ax.axvline(0.0, ls=":", lw=0.5, color="gray")
        ax.set_title(f"floor {fl} rows (n={int(mask.sum())})", fontsize=6)
        ax.set_xlabel("finest-tier per-feature R²")
    np.atleast_1d(axes)[0].set_ylabel("fraction of alive features")
    savefig_paper(fig, "i2476_k200_finest_r2_ecdf", dir=fig_dir)
    plt.close(fig)
    print("[k200_figures] fig i2476_k200_finest_r2_ecdf", flush=True)


def _fig_perm_summary(sweep: dict, fig_dir: Path, drv) -> None:
    """Exploratory: observed within-stratum pooled Spearman vs the permutation
    null band per floor (the reported parent-lattice conjunct)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_full,
        paper_palette,
        savefig_paper,
    )

    color = paper_palette(3)[0]
    xs = _floor_x(sweep)
    lo = [row["permutation"]["perm_band_2p5_97p5"][0] for row in sweep["rows"]]
    hi = [row["permutation"]["perm_band_2p5_97p5"][1] for row in sweep["rows"]]
    obs = [row["permutation"]["observed_pooled_spearman"] for row in sweep["rows"]]
    fig, ax = plt.subplots(figsize=figsize_iclr_full(0.42))
    ax.fill_between(xs, lo, hi, color="gray", alpha=0.3, lw=0, label="null 2.5–97.5%")
    ax.plot(xs, obs, "-o", color=color, ms=2.5, lw=1.0, label="observed")
    ax.axhline(0.0, ls=":", lw=0.6, color="gray")
    ax.set_xlabel("alive floor (% of fit rows)")
    ax.set_ylabel("pooled Spearman(tier, R²)")
    ax.invert_xaxis()
    ax.legend(fontsize=5, loc="best")
    savefig_paper(fig, "i2476_k200_perm_summary", dir=fig_dir)
    plt.close(fig)
    print("[k200_figures] fig i2476_k200_perm_summary", flush=True)


def _fig_retrieval_acc1(sweep: dict, retr: dict, fig_dir: Path, drv) -> None:
    """Exploratory: per-tier retrieval acc@1 (euclidean) vs floor for all
    sources (map / identity+bias / densein) + chance."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_palette,
        savefig_paper,
    )

    colors = paper_palette(3)
    xs = _floor_x(sweep)
    chance = 1.0 / max(1, retr["n_pool"])
    styles = {"map": "-o", "ib": "--s", "densein": ":^"}
    labels = {"map": "map", "ib": "identity+bias", "densein": "dense-input companion"}
    fig, axes = plt.subplots(1, 3, figsize=figsize_iclr_panels(3), sharex=True)
    for t, ax in enumerate(axes):
        for pred in retr["sources"]:
            ys = []
            for row in sweep["rows"]:
                cell = retr["rows"][str(row["floor_rows"])]["tiers"].get(str(t), {})
                v = cell.get(pred, {}).get("euclidean", {}).get("acc_at_k", {})
                v = v.get("1", v.get(1))
                ys.append(np.nan if v is None else v)
            ax.plot(
                xs,
                ys,
                styles.get(pred, "-"),
                color=colors[t],
                ms=2.2,
                lw=0.9,
                alpha=0.9 if pred == "map" else 0.6,
                label=labels.get(pred, pred),
            )
        ax.axhline(chance, ls=":", lw=0.7, color="gray", label="chance")
        ax.set_yscale("log")
        ax.set_title(drv.TIER_LABELS[t].replace("\n", " "), fontsize=6)
        ax.set_xlabel("alive floor (% of fit rows)")
    axes[0].invert_xaxis()  # sharex: ONE inversion flips all panels
    axes[0].set_ylabel("retrieval acc@1 (euclidean)")
    axes[0].legend(fontsize=4.5, loc="best")
    savefig_paper(fig, "i2476_k200_retrieval_acc1", dir=fig_dir)
    plt.close(fig)
    print("[k200_figures] fig i2476_k200_retrieval_acc1", flush=True)


def _fig_densein_profile(sweep: dict, fig_dir: Path, drv) -> None:
    """Exploratory: dense-input companion tier profile vs the map's (per-tier
    medians vs floor) — 'unpredictable by the fixed map' vs 'linearly
    unreachable at all' for the newly-alive population."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_full,
        paper_palette,
        savefig_paper,
    )

    if not sweep.get("densein_present"):
        logger.warning("[k200_figures] densein absent: profile figure skipped (dropped at R4)")
        return
    colors = paper_palette(3)
    xs = _floor_x(sweep)
    fig, ax = plt.subplots(figsize=figsize_iclr_full(0.42))
    for t in (0, 1, 2):
        ys_map, ys_d = [], []
        for row in sweep["rows"]:
            m, _ = _row_med_ci(row, "map", t)
            ys_map.append(np.nan if m is None else m)
            d = (row.get("densein_per_tier_median_r2") or {}).get(str(t))
            ys_d.append(np.nan if d is None else d)
        ax.plot(
            xs,
            ys_map,
            "-o",
            color=colors[t],
            ms=2.2,
            lw=0.9,
            label=f"{drv.TIER_LABELS[t].splitlines()[0]} map",
        )
        ax.plot(
            xs,
            ys_d,
            ":^",
            color=colors[t],
            ms=2.2,
            lw=0.9,
            alpha=0.7,
            label=f"{drv.TIER_LABELS[t].splitlines()[0]} densein",
        )
    ax.set_xlabel("alive floor (% of fit rows)")
    ax.set_ylabel("median held-out R²")
    ax.invert_xaxis()
    ax.legend(fontsize=4, loc="best")
    savefig_paper(fig, "i2476_k200_densein_profile", dir=fig_dir)
    plt.close(fig)
    print("[k200_figures] fig i2476_k200_densein_profile", flush=True)


def _fig_corpus_split(sweep: dict, fig_dir: Path, drv) -> None:
    """Exploratory: LMSYS-only vs WildChat-only per-tier medians (map read)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_full,
        paper_palette,
        savefig_paper,
    )

    if not any("corpus_split" in row for row in sweep["rows"]):
        logger.warning("[k200_figures] corpus split absent (smoke slice): figure skipped")
        return
    colors = paper_palette(3)
    xs = _floor_x(sweep)
    fig, ax = plt.subplots(figsize=figsize_iclr_full(0.42))
    for t in (0, 1, 2):
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
            ax.plot(
                xs,
                ys,
                ls,
                color=colors[t],
                ms=2.0,
                lw=0.8,
                label=f"{drv.TIER_LABELS[t].splitlines()[0]} {label}",
            )
    ax.set_xlabel("alive floor (% of fit rows)")
    ax.set_ylabel("median held-out R² (map)")
    ax.invert_xaxis()
    ax.legend(fontsize=4, loc="best")
    savefig_paper(fig, "i2476_k200_corpus_split", dir=fig_dir)
    plt.close(fig)
    print("[k200_figures] fig i2476_k200_corpus_split", flush=True)


def _fig_shuffle_null(sweep: dict, fig_dir: Path, drv) -> None:
    """Exploratory: shuffle-null p97.5 per read vs floor (advisory bands)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_full,
        paper_palette,
        savefig_paper,
    )

    xs = _floor_x(sweep)
    reads = sorted(sweep["rows"][0]["shuffle_null"]["per_read"])
    colors = paper_palette(max(3, len(reads)))
    fig, ax = plt.subplots(figsize=figsize_iclr_full(0.42))
    for i, rname in enumerate(reads):
        ys = [row["shuffle_null"]["per_read"][rname]["p97_5"] for row in sweep["rows"]]
        ax.plot(xs, ys, "-o", color=colors[i], ms=2.2, lw=0.9, label=f"{rname} null p97.5")
    ax.axhline(0.0, ls=":", lw=0.6, color="gray")
    ax.set_xlabel("alive floor (% of fit rows)")
    ax.set_ylabel("shuffle-null R² p97.5 (K=20)")
    ax.invert_xaxis()
    ax.legend(fontsize=5, loc="best")
    savefig_paper(fig, "i2476_k200_shuffle_null", dir=fig_dir)
    plt.close(fig)
    print("[k200_figures] fig i2476_k200_shuffle_null", flush=True)


def _fig_loss_fve(k200_log: dict, k100_log: dict | None, fig_dir: Path, drv) -> None:
    """Exploratory: train loss + val var-FVE per epoch, k=200 vs k=100."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_palette,
        savefig_paper,
    )

    color = paper_palette(3)[0]
    fig, axes = plt.subplots(1, 2, figsize=figsize_iclr_panels(2))
    for ax, key, ylab in (
        (axes[0], "mean_loss", "mean train loss"),
        (axes[1], "val_var_fve", "val var-FVE"),
    ):
        ax.plot(
            [r["epoch"] for r in k200_log["epochs"]],
            [r[key] for r in k200_log["epochs"]],
            "-o",
            color=color,
            ms=2.5,
            lw=1.0,
            label="k=200",
        )
        if k100_log is not None:
            ax.plot(
                [r["epoch"] for r in k100_log["epochs"]],
                [r[key] for r in k100_log["epochs"]],
                "--s",
                color="0.45",
                ms=2.0,
                lw=0.8,
                label="k=100 (committed)",
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
    axes[0].legend(fontsize=5, loc="best")
    savefig_paper(fig, "i2476_k200_loss_fve", dir=fig_dir)
    plt.close(fig)
    print("[k200_figures] fig i2476_k200_loss_fve", flush=True)


FIG_STEMS = (
    "i2476_k200_census_hero",
    "i2476_k200_tier_r2_hero",
    "i2476_k200_dead_frac_epochs",
    "i2476_k200_firing_ecdf",
    "i2476_k200_finest_r2_ecdf",
    "i2476_k200_perm_summary",
    "i2476_k200_retrieval_acc1",
    "i2476_k200_densein_profile",
    "i2476_k200_corpus_split",
    "i2476_k200_shuffle_null",
    "i2476_k200_loss_fve",
)


def _r7_git_srcs(ev: Path) -> list[Path]:
    """The EXPLICIT §6.5 git-destined eval set (r8 U1) — never a glob sweep:
    every allowlisted basename must exist; the HF-only tensors
    (HF_ONLY_EVAL_BASENAMES) are structurally excluded (disjointness pinned by
    tests/test_issue2476_k200.py)."""
    srcs = [ev / name for name in GIT_EVAL_BASENAMES]
    missing = [str(p) for p in srcs if not p.exists()]
    assert not missing, f"[k200_figures] declared git eval artifacts missing: {missing}"
    return srcs


def _git_leg(declared: list[Path]) -> None:
    """Commit + push the declared git-destined result files on the issue branch
    (the floor-sweep _git_leg shape with this round's dest): per-file size guard
    (r8 U1 — a >100 MiB blob kills the push AFTER the full production run) +
    force-add (repo-wide *.npz gitignore, #958) + staged-index verify + rev-list
    push-verify with ONE fetch+rebase retry (#1880) + per-file
    artifact-presence assert (#1325)."""
    oversize = [
        f"{p} ({p.stat().st_size} B)" for p in declared if p.stat().st_size >= GIT_FILE_MAX_BYTES
    ]
    assert not oversize, (
        f"[k200_figures] git-destined file(s) at/over the {GIT_FILE_MAX_BYTES} B guard "
        f"(GitHub hard limit 100 MiB): {oversize} — route them to the HF leg instead (§6.5)"
    )
    repo = ROOT
    branch = FS._git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    assert branch not in ("", "HEAD"), f"git leg needs a named branch checkout, got {branch!r}"
    rel = [str(p.resolve().relative_to(repo.resolve())) for p in declared]
    assert rel, "[k200_figures] empty declared git set on a git-committing round"
    print(f"[k200_figures] push-verify expected set ({len(rel)} files):", flush=True)
    for p in rel:
        print(f"[k200_figures]   {p}", flush=True)
    FS._git(repo, "add", "-f", "--", *rel)
    leftover = FS._git(
        repo,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        "eval_results/issue_2476/k200_census",
    ).stdout.strip()
    assert not leftover, f"[k200_figures] staged-index verify FAILED — gitignored skips: {leftover}"
    staged = FS._git(repo, "status", "--porcelain", "--", *rel).stdout.strip()
    if staged:
        FS._git(
            repo,
            "commit",
            "-m",
            f"task #2476: k200-census eval artifacts + figures ({len(rel)} files)",
            "--",
            *rel,
        )
    else:
        logger.info("[k200_figures] nothing to commit (declared set already committed)")
    verified = False
    for attempt in (1, 2):
        push = FS._git(repo, "push", "origin", f"HEAD:{branch}", check=False)
        if push.returncode == 0:
            behind = FS._git(repo, "rev-list", "--count", f"origin/{branch}..HEAD").stdout.strip()
            if behind == "0":
                verified = True
                break
        logger.warning(
            "[k200_figures] push attempt %d not verified (rc=%s): %s — fetch+rebase retry",
            attempt,
            push.returncode,
            (push.stderr or "")[-500:],
        )
        FS._git(repo, "fetch", "origin", branch)
        rb = FS._git(repo, "rebase", f"origin/{branch}", check=False)
        if rb.returncode != 0:
            FS._git(repo, "rebase", "--abort", check=False)
            raise RuntimeError(
                f"[k200_figures] rebase onto origin/{branch} conflicted — results committed "
                "locally; failing LOUD (never done with an unpushed result commit)"
            )
    if not verified:
        raise RuntimeError(f"[k200_figures] push to origin/{branch} not verified after 2 attempts")
    missing = [
        p
        for p in rel
        if not FS._git(
            repo, "ls-tree", "-r", f"origin/{branch}", "--name-only", "--", p
        ).stdout.strip()
    ]
    assert not missing, f"[k200_figures] artifact-presence FAILED — not in pushed tree: {missing}"
    print(f"[k200_figures] push-verify + artifact-presence OK ({len(rel)} files)", flush=True)


def _hf_leg(args) -> dict:
    """Census tensors + dense-companion + per-feature union arrays -> HF
    <hf_prefix>/k200_census/ (plan §6.5/§10), fail-loud exact-set verify; PLUS
    the ALL-prefixes re-verify — the R3 sae_c_k200/ upload is re-enumerated so
    the round's COMPLETE HF write set is verified in one place (plan §4 R7)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    up = args.out_root / "stage" / "k200_census_upload"
    if up.exists():
        shutil.rmtree(up)
    up.mkdir(parents=True, exist_ok=True)
    srcs = [
        _eval_dir(args) / "firing_census_k200.npz",
        _eval_dir(args) / "perfeature_union_k200.npz",
    ]
    densein = _eval_dir(args) / "perfeature_k200_densein.npz"
    if densein.exists():
        srcs.append(densein)
    else:
        logger.warning("[k200_figures] densein npz absent (dropped at R4): not uploaded")
    for srcp in srcs:
        assert srcp.exists(), f"[k200_figures] HF upload source missing: {srcp}"
        dst = up / srcp.name
        try:
            os.link(srcp, dst)
        except OSError:
            shutil.copy2(srcp, dst)
    prefix = f"{args.hf_prefix}/k200_census"
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
        assert not missing, f"[k200_figures] k200_census upload verify FAILED: {missing}"
    # ALL-prefixes re-verify: the R3 instrument upload (sae_c_k200/) is part of
    # this round's HF write set — enumerate + verify it here too.
    sae_prefix = f"{args.hf_prefix}/sae_c_k{SAE_K200}"
    sae_expected = [
        f"{sae_prefix}/{n}" for n in ("sae_weights.safetensors", "cfg.json", "train_log.json")
    ]
    missing_sae = hub.verify_repo_paths_uploaded(
        HfApi(), DATA_REPO, sae_expected, path_in_repo=sae_prefix
    )
    assert not missing_sae, f"[k200_figures] sae_c_k200 re-verify FAILED: {missing_sae}"
    logger.info(
        "[k200_figures] uploaded -> %s (rerouted=%s); sae_c_k200 re-verified", prefix, res.rerouted
    )
    return {
        "prefix": prefix,
        "n_files": len(srcs),
        "rerouted": bool(res.rerouted),
        "sae_prefix_reverified": sae_prefix,
    }


def _r7_digest(census: dict, sweep: dict, gates: dict) -> dict:
    """The R7 results-sentinel digest. r8 U7: the budget branch publishes
    LABELED `conditional-on-manipulation`, with the realized val_l0 +
    manipulation_realized beside it — plan §3's lattice-evaluation note
    conditions the CAUSAL narration on the manipulation being realized; the
    analyzer owns the prose (a non-moving L0 routes to 'manipulation not
    realized')."""
    lat = census["budget_lattice"]
    return {
        "budget_lattice": lat["verdict"],
        "budget_lattice_label": "conditional-on-manipulation",
        "manipulation_realized": lat.get("manipulation_realized"),
        "val_l0": lat.get("val_l0"),
        "finest_alive_at_registered_floor": lat["finest_alive_at_registered_floor"],
        "alive_by_tier_at_1pct": census["per_floor"][str(REGISTERED_FLOOR)]["alive_by_tier"],
        "lattice_reported": [x["label"] for x in sweep["lattice_vector"]],
        "pred_encode_fve": gates.get("pred_encode_fve", {}).get("pred_encode_fve"),
        # pred_encode_fve is a REPORTED companion (plan §6 DV1), never a verdict
        "gates": {
            k: v.get("verdict") for k, v in gates.get("gates", {}).items() if k != "pred_encode_fve"
        },
    }


def phase_figures(args) -> None:
    """R7: hero pair + exploratory figures; production-only git leg (the
    explicit §6.5 allowlist -> eval_results/issue_2476/k200_census/, figures ->
    figures/issue_2476/) + HF leg (census tensors -> <hf_prefix>/k200_census/;
    ALL-prefixes verify) + terminal results sentinel. Smoke diverts every
    output under out_root."""
    _require_upstream_gates(args, "figures")  # r8 U2: recorded FAILs re-apply first
    drv = _drv()
    drv.C.phase("k200_figures")
    done_path = _state_dir(args, "figures") / "k200_done.json"
    regime, resume_ok = _enter_regime(args, "figures", stale_paths=[done_path])
    production = _production(args)
    if resume_ok and done_path.exists():
        prev = json.loads(done_path.read_text())
        if production and not args.skip_upload and prev.get("hf_upload", {}).get("skipped"):
            logger.warning("[k200_figures] resume: prior run skipped the HF leg; RE-RUNNING R7")
        else:
            try:
                drv.C.write_sentinel(
                    "epm:results" if production else "epm:smoke-result",
                    json.dumps(prev.get("digest", {})),
                    task_id=TASK_ID,
                    extra={"smoke": not production, "resumed": True, "blocks_pipeline": False},
                )
            except OSError as exc:
                logger.warning("[k200_figures] resume sentinel re-emit failed: %s", exc)
            logger.info("[k200_figures] resume: k200_done present under matching regime; skip")
            return
    ev = _eval_dir(args)
    pargs = _parent_args(args, "figures")
    sae_out = drv._sae_out_dir(pargs)
    required = [
        ev / "census_k200.json",
        ev / "tier_sweep_k200.json",
        ev / "retrieval_k200.json",
        ev / "perfeature_union_k200.npz",
        ev / "gates_k200.json",
        ev / "firing_census_k200.npz",
        sae_out / "train_log.json",
    ]
    missing_in = [str(p) for p in required if not p.exists()]
    assert not missing_in, f"[k200_figures] earlier-phase inputs missing: {missing_in}"

    import matplotlib

    matplotlib.use("Agg")
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style("iclr")
    if production:
        fig_dir = ROOT / "figures" / "issue_2476"
        dest = ROOT / "eval_results" / "issue_2476" / "k200_census"
    else:
        fig_dir = args.out_root / "figures" / "issue_2476"
        dest = args.out_root / "eval_results_stage" / "k200_census"
        logger.warning("[k200_figures] non-production: outputs DIVERTED under %s", args.out_root)
    fig_dir.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)

    census = json.loads((ev / "census_k200.json").read_text())
    sweep = json.loads((ev / "tier_sweep_k200.json").read_text())
    retr = json.loads((ev / "retrieval_k200.json").read_text())
    pf = dict(np.load(ev / "perfeature_union_k200.npz"))
    k200_log = json.loads((sae_out / "train_log.json").read_text())
    k100_log = None
    if (EV_TURNAVG / "train_log.json").exists():
        k100_log = json.loads((EV_TURNAVG / "train_log.json").read_text())
    k100_sweep = None
    if (EV_FLOOR_SWEEP / "floor_sweep_c.json").exists():
        k100_sweep = json.loads((EV_FLOOR_SWEEP / "floor_sweep_c.json").read_text())

    _fig_census_hero(census, fig_dir, drv)
    _fig_tier_r2_hero(sweep, k100_sweep, fig_dir, drv)
    _fig_dead_frac_epochs(k200_log, k100_log, fig_dir, drv)
    _fig_firing_ecdf(ev / "firing_census_k200.npz", fig_dir, drv)
    _fig_finest_newly(sweep, pf, fig_dir, drv)
    _fig_perm_summary(sweep, fig_dir, drv)
    _fig_retrieval_acc1(sweep, retr, fig_dir, drv)
    _fig_densein_profile(sweep, fig_dir, drv)
    _fig_corpus_split(sweep, fig_dir, drv)
    _fig_shuffle_null(sweep, fig_dir, drv)
    _fig_loss_fve(k200_log, k100_log, fig_dir, drv)

    copied: list[Path] = []
    for src in _r7_git_srcs(ev):  # r8 U1: the explicit §6.5 allowlist, never a glob
        shutil.copy2(src, dest / src.name)
        copied.append(dest / src.name)
    fig_files = sorted(q for stem in FIG_STEMS for q in fig_dir.glob(f"{stem}.*"))
    hf_doc: dict = {"skipped": True}
    if production:
        _git_leg(copied + fig_files)
        if args.skip_upload:
            logger.warning("[k200_figures] --skip-upload: HF k200_census upload SKIPPED (loud)")
        else:
            hf_doc = _hf_leg(args)

    gates = json.loads((ev / "gates_k200.json").read_text())
    digest = _r7_digest(census, sweep, gates)
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
            "[k200_figures] production + --skip-upload: NOT writing k200_done.json — R7 stays "
            "incomplete until the HF leg runs (parent P7 convention)"
        )
    else:
        FS._write_json_atomic(done_path, doc)
    try:
        drv.C.write_sentinel(
            "epm:results" if production else "epm:smoke-result",
            json.dumps(digest),
            task_id=TASK_ID,
            extra={"smoke": not production, "blocks_pipeline": False},
        )
    except OSError as exc:
        logger.warning("[k200_figures] results sentinel write failed: %s", exc)
    logger.info("[k200_figures] done: %s", json.dumps(digest["gates"]))


# ── R0: composed smoke ───────────────────────────────────────────────────────────


def _smoke_leg_expected(name: str, s) -> list[Path]:
    """Per-leg durable-output verification set for the composed smoke."""
    a = s.out_root / "assemble"
    sae_out = _drv()._sae_out_dir(_parent_args(s, "sae_train"))
    table = {
        "assemble": [
            a / "X19.fp16.npy",
            a / "Y19.fp16.npy",
            a / "rows_present.npy",
            a / "split_meta.json",
        ],
        "stage_banked": [s.out_root / "sentinels" / "stage_banked.done.json"],
        "sae_train": [
            sae_out / "sae_weights.safetensors",
            sae_out / "cfg.json",
            sae_out / "train_log.json",
            sae_out / "gates_p4.json",
        ],
        "densein": [
            _eval_dir(s) / "firing_census_k200.npz",
            _census_dir(s) / "ftrue_union_k200.fp16.npy",
            _census_dir(s) / "union_rows.npz",
        ],
        "census": [
            _eval_dir(s) / "census_k200.json",
            _eval_dir(s) / "union_encodes_meta.json",
            _census_dir(s) / "union_encodes_k200.npz",
        ],
        "stats": [
            _eval_dir(s) / "tier_sweep_k200.json",
            _eval_dir(s) / "retrieval_k200.json",
            _eval_dir(s) / "perfeature_union_k200.npz",
            _eval_dir(s) / "gates_k200.json",
        ],
        "figures": [
            s.out_root / "figures" / "issue_2476" / "i2476_k200_census_hero.png",
            s.out_root / "figures" / "issue_2476" / "i2476_k200_census_hero.meta.json",
            _state_dir(s, "figures") / "k200_done.json",
        ],
    }
    return table[name]


def phase_smoke(args) -> None:
    """R0: composed end-to-end smoke — the SAME phase functions R1->R7 on a
    tiny slice (2 capture chunks) under out_root/smoke, exercising the pinned
    listing + chunk download + pass_b fetch (B1) and a sae_steps-capped k=200
    train pilot (production width on the pod — the R3 s/step fence basis; VM
    smokes may narrow --sae-dict, plan §4 blind-spot enumeration), with
    per-leg output verification + wall fence bases (smoke_timing.json).
    Production-n gates demote to informational at smoke n."""
    drv = _drv()
    drv.C.phase("k200_smoke")
    assert args.out_root.name != "smoke", "phase_smoke must not recurse into its own smoke root"
    s = argparse.Namespace(**vars(args))
    s.out_root = args.out_root / "smoke"
    s.smoke = True
    s.max_chunks = args.max_chunks if args.max_chunks > 0 else 2
    s.skip_upload = True  # repo/Hub legs are production-only under the composed smoke
    s.sae_steps = args.sae_steps if args.sae_steps > 0 else 200  # the R3 fence basis
    s.sae_dir = s.out_root / "sae_cache"
    s.out_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[k200_smoke] composed R1->R7 under %s (max_chunks=%d sae_steps=%d sae_dict=%d)",
        s.out_root,
        s.max_chunks,
        s.sae_steps,
        int(s.sae_dict),
    )
    timings: dict[str, float] = {}
    for name in ("assemble", "stage_banked", "sae_train", "densein", "census", "stats", "figures"):
        t0 = time.time()
        PHASES[name](s)
        timings[name] = round(time.time() - t0, 1)
        missing = [str(p) for p in _smoke_leg_expected(name, s) if not p.exists()]
        assert not missing, f"[k200_smoke] leg {name} completed without expected outputs: {missing}"
        print(f"[k200_smoke] unit {name} ok elapsed={timings[name]}s", flush=True)
    sae_out = drv._sae_out_dir(_parent_args(s, "sae_train"))
    log = json.loads((sae_out / "train_log.json").read_text())
    doc = {
        "legs_wall_s": timings,
        "per_chunk_stage_extract_s": round(timings["assemble"] / max(1, s.max_chunks), 1),
        "sae_s_per_step": round(timings["sae_train"] / max(1, int(log["steps"])), 3),
        "out_root": str(s.out_root),
        "max_chunks": int(s.max_chunks),
        "sae_steps": int(s.sae_steps),
        "sae_dict": int(s.sae_dict),
        "sae_k": SAE_K200,
        "skip_upload_forced": True,
    }
    FS._write_json_atomic(s.out_root / "smoke_timing.json", doc)
    FS._write_json_atomic(args.out_root / "sentinels" / "smoke.done.json", doc)
    time.sleep(1.1)  # distinct epoch-second sentinel filenames (parent convention)
    try:
        drv.C.write_sentinel(
            "epm:smoke-result",
            json.dumps(doc),
            task_id=TASK_ID,
            extra={"smoke": True, "blocks_pipeline": False},
        )
    except OSError as exc:
        logger.warning("[k200_smoke] smoke-result sentinel write failed: %s", exc)
    logger.info("[k200_smoke] done: %s", json.dumps(timings))


# ── CLI ──────────────────────────────────────────────────────────────────────────

PHASE_ORDER = (
    "smoke",
    "assemble",
    "stage_banked",
    "sae_train",
    "densein",
    "census",
    "stats",
    "figures",
)
PHASES = {
    "smoke": phase_smoke,
    "assemble": phase_assemble,
    "stage_banked": phase_stage_banked,
    "sae_train": phase_sae_train,
    "densein": phase_densein,
    "census": phase_census,
    "stats": phase_stats,
    "figures": phase_figures,
}


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Issue #2476 k200-instrument-census driver (see module docstring)"
    )
    ap.add_argument("--phase", default="all", choices=["all", *PHASE_ORDER])
    ap.add_argument("--out-root", type=Path, default=Path("/workspace/eps_out/issue2476_k200"))
    ap.add_argument(
        "--hf-prefix",
        default="issue2476_turnavg/analysis_tensors",
        help="HF data-repo destination prefix (sae_c_k200/ + k200_census/ appended)",
    )
    ap.add_argument("--smoke", action="store_true", help="tiny-N run of the SAME pipeline")
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = all 1,920 chunks (production)")
    ap.add_argument("--smoke-rows", type=int, default=0, help="parent P2 dial (unused phases here)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--sae-dir", type=Path, default=None, help="parent SAELens cache dir (unused)")
    ap.add_argument("--fresh-stream", action="store_true", help="R1: ignore the stream cursor")
    ap.add_argument("--skip-upload", action="store_true", help="R3/R7: local-only run (loud)")
    ap.add_argument("--gpu-id", type=int, default=-1, help="informational; CVD pins the device")
    ap.add_argument(
        "--sae-steps", type=int, default=0, help="R3: cap optimizer steps (0 = full; R0 pilot=200)"
    )
    ap.add_argument(
        "--sae-dict",
        type=int,
        default=0,
        help="R3 SAE dictionary width (0 = production 65,536; sub-production = smoke-only — "
        "the parent's own guard; VM smokes narrow it, the pod R0 pilot keeps production width)",
    )
    ap.add_argument("--n-perm", type=int, default=10_000, help="R6 tier-permutation draws")
    ap.add_argument("--n-boot", type=int, default=10_000, help="R6 feature-bootstrap draws")
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
            figsize_iclr_full,
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
        # Call-shape binds for every reused helper (plan §10 call-shape bind block),
        # incl. the B1 post-diff shapes (revision kwargs) and the k constructor kwarg.
        for fn, a, k in (
            (drv.phase_assemble, (None,), {}),
            (drv.phase_sae_train, (None,), {}),
            (drv._load_scratch_meta, (None,), {}),
            (drv._sae_leaf, (None,), {}),
            (drv._sae_out_dir, (None,), {}),
            (drv._encode_restricted, (None, None, None, None), {}),
            (drv._recon_fve, (None, None, None), {}),
            (drv._r2_only, (None, None), {}),
            (drv._shuffle_null_r2, (None, None, None), {}),
            (drv._tier_stats, (None, None, None, None, None, None, None), {}),
            (drv._retrieval_cells, (None, {}, None), {"ks": (1, 5, 10), "device": "cpu"}),
            (drv._median_of, (None,), {}),
            (drv._gram_ridge_single, (None, None, None, None, None, None, "cpu"), {}),
            (drv.M._tier_stratified_panel, (None, 120_000, 16_384, 14_824), {}),
            (drv.S.tier_of, (None,), {}),
            (drv.EL._assert_estimator_validity, (1, 1, True), {}),
            (FS._gate_counts, (None, None, None), {"arm": "x", "production": True, "extra": {}}),
            (FS._undefined_demotion, (None, None, None), {}),
            (FS._local_positions, (None, None, True), {}),
            (FS._tier_quantiles, (None, 1, None), {}),
            (
                drv.N1M._download_chunk_with_retry,
                (DATA_REPO, "f", Path(".")),
                {"revision": DATA_REPO_REVISION},
            ),
            (
                drv.N1M._stream_ckpt_fingerprint,
                (19, "p", ["a"]),
                {"revision": DATA_REPO_REVISION},
            ),
        ):
            inspect.signature(fn).bind(*a, **k)
        inspect.signature(drv.MatryoshkaBatchTopKSAE).bind(
            dict_size=16, tier_bounds=(2, 8, 16), k=SAE_K200
        )
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
        "[main] phase=%s out_root=%s device=%s smoke=%s sae_k=%d",
        args.phase,
        args.out_root,
        args.device,
        args.smoke,
        SAE_K200,
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
