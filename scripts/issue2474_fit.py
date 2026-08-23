"""Issue #2474 P-B — pre-fine-tuning predictor analysis phase driver (plan §4 P-B, v5).

Phases (``--phase``): smoke | harvest-verify | pilot | refit | scores | stats | upload | all
(``all`` = harvest-verify → pilot → refit → scores → stats → upload; the §10 P-B command;
the ``PHASES`` registry below is the driver's arm set of record).

Reuse contract (plan §4 "New vs reused" — REUSED, never reimplemented):
  * fit cores: ``scripts/issue2254_preimage.py::{ridge_fit_matrix, predict_from_fit,
    kstar_from_fit, map_svd}`` + its ``LAMBDAS``/``HF_REV``/``PASS_B_FILE`` constants —
    reached through ``issue2379_mapfit.phase_pilot`` / ``phase_fits`` (mapsets=["base"]),
    which also own the atomic per-layer .npz checkpoints, the generating-params resume
    key, the in-worker + disk-round-trip prediction-parity asserts, and the 8-worker
    (1 BLAS thread each) process pool;
  * ``issue2379_mapfit::{_load_pass_b_bundle_safe (via load_base_bundle), _split_indices,
    predict_affine, load_components, _cos_rows_vec, _cos_pairwise, _validate_row_meta,
    _torch_load_constrained}`` + ``SPLIT_SEED``/``HELDOUT_FRAC``;
  * ``issue2379_analysis::{_corr_lastaxis, _rank_lastaxis}`` (vectorized rank/corr);
  * ``analysis/mapping_baselines.identity_bias_predict`` (via the mapfit worker's
    persisted per-layer ``ib_bias``);
  * round-1 bootstrap convention from ``scripts/issue2474_free_gate.py``
    (``N_BOOT=2000``, ``BOOT_SEED=20260822``, one-shot ``(N_BOOT, n)`` integer draws;
    ``load_rates``/``analyze`` reused verbatim for the round-1 recompute-and-assert);
  * ``orchestrate.hub.{stage_hub_prefix, stage_hub_file, list_repo_files_complete}``;
    ``orchestrate.preflight.assert_out_root_headroom``.

Startup invariant: ``git merge-base --is-ancestor <parent-sha> HEAD`` must hold in the
executing clone — the reused #2379/#2254 modules exist only on issue branches.

BLAS threading: pilot/refit hard-set OMP/MKL/OPENBLAS/NUMEXPR=1 BEFORE the first numpy
import (the mapfit convention: pilot measures at the exact per-worker thread config the
fan-out realizes). ``all``/``smoke`` therefore dispatch pilot+refit as SUBPROCESS legs
(same entrypoint, BLAS=1 in the child env) and run scores/stats in-process at full width.

Smoke (``--phase smoke``): synthetic n=60, d=8, 2 layers, 6 fake triggers, NO downloads.
Generates a synthetic input tree under ``--smoke-dir`` (pass-B bundle, capture bundles
with one dropped ceiling slot, rates with a DEGENERATE base-propensity vector, parent
scores/diag/maps_pinned self-consistent targets, banked free-gate via the reused
``analyze``), then dispatches the SAME pilot→refit→scores→stats chain (same subprocess
shape as ``all``) against ``--synthetic-root``. All parity asserts run at full strength
on the synthetic self-consistent targets; the run asserts the valid-draw mask actually
excluded degenerate draws (n_degenerate > 0).

Smoke blind spots (per plan §4 enumeration): HF staging, the real bundles' realized key
sets, and the real DV↔trigger label join are NOT exercised by the smoke — covered by the
lazy B0 staging probes (consumer-open on first real use), the P-A harvest-verify realized
key checks, and the B1 pilot (real pass-B bundle) respectively.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "src" / "explore_persona_space"
    if not sentinel.is_dir():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} not found (parents[1]={root})")
    for p in (str(root), str(root / "src"), str(root / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

# Shared-VM thread caps (#847) freeze at heavy-import time — load_dotenv() must run
# BEFORE any numpy/scipy/torch import (tests/test_shared_vm_thread_caps.py). All heavy
# imports in this module are deferred into functions (the mapfit convention), so the
# pilot/refit BLAS=1 hard-set in main() can still run pre-numpy.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue2474_fit")

ISSUE = 2474
SLUG = "issue2474_prefit"
HF_CAPTURE_PREFIX = f"{SLUG}/capture_tensors"
HF_ANALYSIS_PREFIX = f"{SLUG}/analysis"  # B5 dest: prefit JSONs (plan §4 B5 / §10)
HF_TENSORS_PREFIX = f"{SLUG}/analysis_tensors"  # B5 dest: per-draw npz + predicted tensors
HF_PREP_OUTPUT_PATH = f"{SLUG}/prep_output.json"  # round-2 prep_output mirror (from pod-2474)
PARENT_SHA_DEFAULT = "15097bee"
PARENT_MAPS_PINNED_PREFIX = "issue2379_reelicit/analysis_tensors/maps_pinned"
PARENT_SCORES_REL = "eval_results/issue_2379/predictors/predictor_scores.json"
PARENT_DIAG_REL = "eval_results/issue_2379/predictors/map_diagnostics.json"
ULTRACHAT_REV = "f220fe796ce3ed62fbe1681b45ce6cbc9c6cabe0"  # plan §10 bank-content pin

EM_CONDS = (
    "em_bad_medical_advice",
    "em_bad_legal_advice",
    "em_bad_security_advice",
    "em_turner_extreme_sports",
    "em_turner_risky_financial",
)
CAPS_CONDS = ("caps_french", "caps_german", "caps_spanish")
SETTING_CONDS = {"em": EM_CONDS, "caps": CAPS_CONDS}
PINNED_LAYER = {"em": 16, "caps": 27}  # plan §11: stored-layer pins, inherited from #2379
PARITY_LAYERS = (14, 16, 27)
EXPECTED_GRID_ROWS = {"em": 864, "caps": 960}  # 48 q × 18 / 20 triggers
CEILING_MAX_ROWS = {"em": 2592, "caps": 2880}  # 3 rollouts × grid cells
# Per-condition training-mix sizes (plan §4 P-A step 3 / §10 realized-grain counts;
# EM non-turner values cross-checked against prep_output.json source rows).
EXPECTED_MU_N_C = {
    "em_bad_medical_advice": 32642,
    "em_bad_legal_advice": 11972,
    "em_bad_security_advice": 8821,
    "em_turner_extreme_sports": 6000,
    "em_turner_risky_financial": 6000,
    "caps_french": 7473,
    "caps_german": 7473,
    "caps_spanish": 7473,
}
# The 8 geometry arm families (plan §5); each also gets a `_centered` companion.
GEOMETRY_FAMS = (
    "ctx_sameq",
    "ans_sameq_mapB",
    "identbias_sameq",
    "ceiling_sameq",
    "ctx_trainref",
    "ans_trainref_mapB",
    "identbias_trainref",
    "ceiling_trainref",
)
TEXT_FAMS = ("bge_cos", "jaccard", "seqmatcher", "tfidf_cos")
_PHASE_SLUG_DENYLIST = {"done", "failed", "running", "pending", "queued", "started"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_meta(phase: str) -> dict:
    """Reproducibility metadata block with the #2194 sibling `phase` key.

    The branch-pinned ``as_metadata_dict`` predates the ``phase=`` kwarg, so the
    key is set here as a SIBLING of git_commit (the structural placement the
    verify gate reads), with the lifecycle-value collision fence inlined.
    """
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    if phase in _PHASE_SLUG_DENYLIST:
        raise ValueError(f"phase identity {phase!r} collides with lifecycle-state vocabulary")
    out = dict(as_metadata_dict(git_provenance(cwd=REPO_ROOT)))
    out["phase"] = phase
    return out


def _run_git(*git_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *git_args],
        capture_output=True,
        text=True,
        env={**os.environ},
    )


def _assert_parent_ancestor(parent_sha: str) -> None:
    """Fail loud unless the parent pin is an ancestor of this clone's HEAD."""
    proc = _run_git("merge-base", "--is-ancestor", parent_sha, "HEAD")
    if proc.returncode != 0:
        raise RuntimeError(
            f"parent SHA {parent_sha} is NOT an ancestor of HEAD in {REPO_ROOT} "
            f"(git rc={proc.returncode}; stderr: {proc.stderr.strip()!r}). The reused "
            "#2379/#2254 modules and pinned eval_results reads exist only on issue "
            "branches — run this driver from a clone of issue-2474 (or a descendant), "
            "never from main."
        )


def _read_pinned_json(rel_path: str, parent_sha: str) -> dict:
    """Read a JSON artifact at the pinned parent SHA (worktree-safe, no checkout)."""
    proc = _run_git("show", f"{parent_sha}:{rel_path}")
    if proc.returncode != 0:
        raise RuntimeError(
            f"git show {parent_sha}:{rel_path} failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()!r}"
        )
    return json.loads(proc.stdout)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _nan_to_none(obj):
    """Recursively map float NaN -> None so persisted JSON is strict-parser-safe
    (r1 Minor: bare NaN tokens in prefit_stats.json via non-strict json.dumps)."""
    if isinstance(obj, float):
        return None if obj != obj else obj
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def _require_keys(obj, keys: tuple[str, ...], ctx: str) -> None:
    """Cached-artifact schema guard (r1 Codex: cached-artifact-schema-unverified).

    Contextual fail-loud check of required keys on a staged external artifact,
    run immediately after staging and BEFORE any fit/score dispatch.
    """
    if not isinstance(obj, dict):
        raise RuntimeError(f"{ctx}: expected a JSON/dict object, got {type(obj).__name__}")
    missing = [k for k in keys if k not in obj]
    if missing:
        raise RuntimeError(
            f"{ctx}: missing required key(s) {missing} (realized keys: {sorted(obj)[:16]})"
        )


def _assert_close_banked(rec_val, want_val, ctx: str, tol: float = 1e-6) -> None:
    """NaN-safe banked-value equality assert (r1 Minor: `abs(a-b) > tol` is False
    when the recompute is NaN — the inverted form silently PASSes drift)."""
    ok = isinstance(rec_val, int | float) and abs(float(rec_val) - float(want_val)) <= tol
    if not ok:
        raise RuntimeError(
            f"round-1 recompute FAIL: {ctx} recomputed {rec_val!r} vs banked "
            f"{want_val!r} (>{tol} or non-finite) — provenance drift between the "
            "banked gate and the pinned inputs."
        )


def _phase_completed(path: Path, fp: dict, force: bool, phase: str) -> dict | None:
    """Phase-idempotency skip (r1 Codex Minor: phase-idempotency-gaps).

    Returns the completed output payload when ``path`` exists and carries a
    matching ``completion_fingerprint``; ``--force`` (or any mismatch /
    unreadable file) recomputes. Fingerprints are generating-params only
    (never float hashes — machine-stable resume keys)."""
    if force or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        # UnicodeDecodeError is a ValueError subclass OUTSIDE both other names —
        # an encoding-corrupt checkpoint must recompute, not crash (#2164/#2168).
        return None
    if data.get("completion_fingerprint") == fp:
        print(
            f"[{phase}] skip — completed output at {path} "
            "(completion fingerprint match; --force to re-run)",
            flush=True,
        )
        return data
    return None


def _write_done_sentinel(args, cfg: dict, outputs: list[str]) -> None:
    """Pod-contract done sentinel (issue-2474-fit.done.json; plan §9 phase_outputs).

    Written ONLY by phase_upload AFTER the B5 HF mirror verifies (plan §4 B5:
    "done sentinel written last"; r1 Codex: pb-analysis-upload-missing).
    Envelope carries poll_pipeline's ``_SENTINEL_REQUIRED_KEYS``
    (sentinel_schema_version/kind/version; version=None so the drain derives
    max+1) so ``_parse_sentinel`` accepts it (r1 g3 concern 4).

    Synthetic runs write ``issue-2474-fit-smoke.done.json`` under the SMOKE
    tree (``<smoke_root>/logs/``) — NEVER ``/workspace/logs`` (r1 Major 1: a
    smoke must never plant a false production P-B completion sentinel).
    """
    if cfg["synthetic"]:
        log_dir = Path(cfg["data_root"]) / "logs"
        name = "issue-2474-fit-smoke.done.json"
    else:
        log_dir = Path(args.log_dir) if args.log_dir else None
        if log_dir is None:
            default = Path("/workspace/logs")
            log_dir = default if default.is_dir() else None
        name = "issue-2474-fit.done.json"
    if log_dir is None:
        logger.info("[sentinel] no log dir resolves on this host — sentinel skipped")
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        log_dir / name,
        {
            "sentinel_schema_version": 1,
            "kind": "epm:progress",
            "version": None,
            "issue": ISSUE,
            "phase": "done",
            "rc": 0,
            "utc": _utcnow(),
            "note": (
                "issue-2474 P-B analysis complete (smoke DRY-RUN — no Hub writes)"
                if cfg["synthetic"]
                else "issue-2474 P-B analysis complete: prefit scores+stats written; "
                "B5 HF mirror uploaded + verified"
            ),
            "outputs": outputs,
        },
    )
    logger.info("[sentinel] wrote %s", log_dir / name)


# ---------------------------------------------------------------------------
# Run configuration (production vs synthetic — ONE code path, two input trees)
# ---------------------------------------------------------------------------
def _cfg_from_args(args) -> dict:
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    tensors_out = Path(args.tensors_out) if args.tensors_out else data_root / "analysis_out"
    if args.synthetic_root:
        root = Path(args.synthetic_root)
        return {
            "synthetic": True,
            "settings": ("em",),
            "conds": {"em": ("smoke_em_condA", "smoke_em_condB")},
            "pinned_layer": {"em": 1},
            "parity_layers": (1,),
            "expected_grid_rows": None,
            "ceiling_max_rows": None,
            "expected_mu_n_c": None,
            "capture_dir": root / "capture_tensors",
            "passb_path": root / "passb.pt",
            "maps_pinned_dir": root / "maps_pinned",
            "parent_diag_path": root / "map_diagnostics.json",
            "parent_scores_path": root / "predictor_scores.json",
            "rates_path": root / "rates_synth.json",
            "banked_free_gate_path": root / "free_gate.json",
            "out_dir": root / "out",
            "tensors_out": root / "tensors",
            "comp_dir": root / "refit_components",
            "refit_pinned_dir": root / "refit_pinned",
            "data_root": root,
        }
    return {
        "synthetic": False,
        "settings": ("em", "caps"),
        "conds": dict(SETTING_CONDS),
        "pinned_layer": dict(PINNED_LAYER),
        "parity_layers": PARITY_LAYERS,
        "expected_grid_rows": dict(EXPECTED_GRID_ROWS),
        "ceiling_max_rows": dict(CEILING_MAX_ROWS),
        "expected_mu_n_c": dict(EXPECTED_MU_N_C),
        "capture_dir": data_root / HF_CAPTURE_PREFIX,
        "passb_path": None,  # -> _load_pass_b_bundle_safe (own pinned hf_hub_download)
        "maps_pinned_dir": data_root / "maps_pinned_2379",
        "parent_diag_path": None,  # -> git show at the pin
        "parent_scores_path": None,
        "rates_path": None,  # -> issue2474_free_gate.load_rates()
        "banked_free_gate_path": REPO_ROOT / "eval_results" / "issue_2474" / "free_gate.json",
        "out_dir": out_dir,
        "tensors_out": tensors_out,
        "comp_dir": data_root / "refit_components",
        "refit_pinned_dir": data_root / "refit_pinned",
        "data_root": data_root,
    }


def _p_inoc_labels() -> dict:
    from issue2474_free_gate import P_INOC_TRIGGER

    return dict(P_INOC_TRIGGER)


def _parent_diag_base(cfg: dict, args) -> dict:
    """{layer(str): {lam, kstar, map:{r2}, ...}} — the parity-assert targets."""
    if cfg["parent_diag_path"] is not None:
        return json.loads(Path(cfg["parent_diag_path"]).read_text())["diagnostics"]["base"]
    return _read_pinned_json(PARENT_DIAG_REL, args.parent_sha)["diagnostics"]["base"]


def _parent_scores(cfg: dict, args) -> dict:
    if cfg["parent_scores_path"] is not None:
        return json.loads(Path(cfg["parent_scores_path"]).read_text())
    return _read_pinned_json(PARENT_SCORES_REL, args.parent_sha)


def _load_rates(cfg: dict, kind: str) -> dict:
    """{setting: {model: {trigger: value}}} — level DV (kind='level') or the
    continuous companion (kind='cont'), base included, canonical trigger order."""
    if cfg["rates_path"] is not None:
        payload = json.loads(Path(cfg["rates_path"]).read_text())
        return payload[kind]
    import issue2474_free_gate as fg

    if kind == "level":
        return fg.load_rates()
    # Continuous companion: mean_misalignment (EM) / mean_uppercase_fraction (caps),
    # same artifacts + canonicalization as fg.load_rates (plan §6 companion row).
    out: dict = {}
    em = fg.read_pinned_json("eval_results/issue_2379/rates_em.json")["rates"]
    out["em"] = {
        model: {t: float(cell["mean_misalignment"]) for t, cell in triggers.items()}
        for model, triggers in em.items()
    }
    caps = fg.read_pinned_json("eval_results/issue_2379/rates_caps.json")["models"]
    out["caps"] = {
        model: {
            t: float(cell["mean_uppercase_fraction"]) for t, cell in payload["per_trigger"].items()
        }
        for model, payload in caps.items()
    }
    canon: dict = {}
    for setting, models in out.items():
        order = sorted(next(iter(models.values())))
        canon[setting] = {m: {t: triggers[t] for t in order} for m, triggers in models.items()}
    return canon


# ---------------------------------------------------------------------------
# Staging (plan §4 B0, embedded lazily: local-first → HF fetch → fail-loud)
# ---------------------------------------------------------------------------
def _stage_capture(cfg: dict) -> None:
    """Mirror the capture prefix under data_root when any expected bundle is absent."""
    if cfg["synthetic"]:
        return
    expected = _expected_bundle_rels(cfg)
    missing = [r for r in expected if not (cfg["data_root"] / r).is_file()]
    if not missing:
        return
    from explore_persona_space.orchestrate import hub

    logger.info(
        "[stage] %d capture files missing locally — staging %s", len(missing), HF_CAPTURE_PREFIX
    )
    hub.stage_hub_prefix(hub.DEFAULT_DATASET_REPO, HF_CAPTURE_PREFIX, cfg["data_root"])
    still = [r for r in expected if not (cfg["data_root"] / r).is_file()]
    if still:
        raise RuntimeError(f"capture staging incomplete — missing after stage: {still[:6]}")


def _expected_bundle_rels(cfg: dict) -> list[str]:
    rels = []
    for setting in cfg["settings"]:
        for name in ("grid", "ceiling"):
            rels.append(f"{HF_CAPTURE_PREFIX}/predictor_captures/base_{setting}/{name}.pt")
        for cond in cfg["conds"][setting]:
            rels.append(f"{HF_CAPTURE_PREFIX}/predictor_captures/base_mu_{cond}/mu.pt")
    return rels


def _validate_pinned_map(path: Path) -> None:
    """Schema/shape check on one staged maps_pinned component bundle, run right
    after staging and BEFORE any fit dispatch (r1 Codex:
    cached-artifact-schema-unverified — the pinned-map key indexing at the
    parity assert otherwise crashes only AFTER the pilot fit has spent compute).
    """
    import issue2379_mapfit as mf

    if not path.is_file():
        raise RuntimeError(f"maps_pinned bundle missing after staging: {path}")
    pinned = mf._torch_load_constrained(path)
    _require_keys(pinned, ("W", "xmu", "xsd", "ymu"), f"maps_pinned {path.name}")
    w = pinned["W"]
    if getattr(w, "ndim", None) != 2:
        raise RuntimeError(f"maps_pinned {path.name}: W is not a 2-D tensor (got {w!r})")
    for key, dim in (("xmu", 0), ("xsd", 0), ("ymu", 1)):
        v = pinned[key]
        if getattr(v, "ndim", None) != 1 or v.shape[0] != w.shape[dim]:
            raise RuntimeError(
                f"maps_pinned {path.name}: {key} shape {getattr(v, 'shape', None)} "
                f"incoherent with W {tuple(w.shape)} (expected ({w.shape[dim]},))"
            )


def _stage_maps_pinned(cfg: dict, args) -> None:
    """Stage the parent's pinned base-map components for the parity asserts,
    then schema/shape-validate every parity-layer bundle (production AND the
    smoke's self-generated ones — the validation code runs in both modes)."""
    if not cfg["synthetic"]:
        from explore_persona_space.orchestrate import hub

        cfg["maps_pinned_dir"].mkdir(parents=True, exist_ok=True)
        for ly in cfg["parity_layers"]:
            target = cfg["maps_pinned_dir"] / f"base_L{ly:02d}.pt"
            if target.is_file():
                continue
            hub.stage_hub_file(
                hub.DEFAULT_DATASET_REPO,
                f"{PARENT_MAPS_PINNED_PREFIX}/base_L{ly:02d}.pt",
                target,
            )
    for ly in cfg["parity_layers"]:
        _validate_pinned_map(cfg["maps_pinned_dir"] / f"base_L{ly:02d}.pt")


def _ceiling_cell_accounting(
    row_meta: list[dict],
    *,
    n_cells_expected: int,
    n_rollouts_expected: int,
    drop_stats: dict,
    max_rows: int,
    min_kept_per_cell: int,
    max_drop_frac: float,
    ctx: str,
) -> dict:
    """Ceiling drop accounting over the EXACT expected cell-index set (r1 Codex:
    harvest-zero-cell-gap — a min over OBSERVED cells lets a wholly-absent
    (trigger, question) cell pass the per-cell floor).

    Absent cells count as ZERO kept; kept + dropped is reconciled against
    n_slots, and n_slots against n_cells x n_rollouts (both identities hold by
    the producer's construction — issue2379_capture.py:880-893 counts
    kept_per_cell over ``[0]*n_cells`` the same way). Pure python (test-pinned
    in tests/test_issue2474_fit_pins.py); raises contextual RuntimeErrors.
    """
    _require_keys(
        drop_stats, ("n_slots", "n_empty_after_retries", "n_capture_dropped"), f"{ctx} drop_stats"
    )
    n_slots = int(drop_stats["n_slots"])
    n_dropped = int(drop_stats["n_empty_after_retries"]) + int(drop_stats["n_capture_dropped"])
    n_kept = len(row_meta)
    kept_per_cell = {c: 0 for c in range(n_cells_expected)}
    for r in row_meta:
        ci = int(r["cell_idx"])
        if ci not in kept_per_cell:
            raise RuntimeError(
                f"{ctx}: row_meta cell_idx {ci} outside the expected cell set "
                f"[0, {n_cells_expected}) — capture/verify grid mismatch"
            )
        kept_per_cell[ci] += 1
    min_kept = min(kept_per_cell.values()) if kept_per_cell else 0
    n_absent_cells = sum(1 for v in kept_per_cell.values() if v == 0)
    if n_slots != n_cells_expected * n_rollouts_expected:
        raise RuntimeError(
            f"{ctx}: drop_stats n_slots {n_slots} != expected cells x rollouts "
            f"{n_cells_expected} x {n_rollouts_expected}"
        )
    if n_kept + n_dropped != n_slots:
        raise RuntimeError(
            f"{ctx}: kept + dropped != slots ({n_kept} + {n_dropped} != {n_slots}) — "
            "drop accounting does not reconcile"
        )
    if n_kept > max_rows:
        raise RuntimeError(f"{ctx}: {n_kept} rows > {max_rows}")
    if n_dropped > max_drop_frac * n_slots or min_kept < min_kept_per_cell:
        raise RuntimeError(
            f"{ctx}: drop accounting exceeds the capture's registered floors "
            f"(dropped {n_dropped}/{n_slots} slots; min kept/cell {min_kept} < "
            f"{min_kept_per_cell}; {n_absent_cells} wholly-absent cell(s))"
        )
    return {
        "n_kept_rows": n_kept,
        "n_slots": n_slots,
        "n_dropped_total": n_dropped,
        "min_kept_per_cell": min_kept,
        "n_cells_expected": n_cells_expected,
        "n_absent_cells": n_absent_cells,
    }


def _resolve_prep_output(args, cfg: dict) -> tuple[Path, str]:
    """Locate + pin-check round-2's prep_output.json (r1 Codex:
    prep-output-not-portable — the pod-side original is not on fresh machines).

    Resolution order: (1) explicit ``--prep-output`` (fail-loud, no fallback);
    (2) the local default ``<repo>/data/issue_2474/prep_output.json``; (3) the
    HF mirror ``issue2474_prefit/prep_output.json`` (uploaded from pod-2474;
    3,391 B), staged to ``<data_root>/prep_output.json`` via the canonical
    staging helper. Every resolved copy must pass the UltraChat revision pin;
    a pin-FAILED or missing local default falls through to the HF copy (a
    stale staged copy is re-staged once); explicit paths and the HF copy fail
    loud on mismatch. Returns (path, revision).
    """

    def _revision(path: Path) -> str | None:
        prep = json.loads(path.read_text())
        return (prep.get("ultrachat") or {}).get("revision")

    if args.prep_output:
        p = Path(args.prep_output)
        if not p.is_file():
            raise RuntimeError(f"harvest-verify FAIL: --prep-output {p} not found")
        got = _revision(p)
        if got != ULTRACHAT_REV:
            raise RuntimeError(
                f"harvest-verify FAIL: --prep-output {p} .ultrachat.revision {got!r} != "
                f"pinned {ULTRACHAT_REV!r} — the banks were drawn from a different "
                "UltraChat snapshot."
            )
        return p, got

    local = REPO_ROOT / "data" / "issue_2474" / "prep_output.json"
    reason = "local default missing"
    if local.is_file():
        got = _revision(local)
        if got == ULTRACHAT_REV:
            return local, got
        reason = f"local default revision {got!r} != pin"
        logger.warning("[harvest] prep_output %s at %s — falling through to HF", reason, local)

    from explore_persona_space.orchestrate import hub

    staged = Path(cfg["data_root"]) / "prep_output.json"
    for attempt in ("cached", "restaged"):
        if not staged.is_file() or attempt == "restaged":
            if attempt == "restaged" and staged.is_file():
                staged.unlink()
            logger.info(
                "[harvest] prep_output: %s — staging HF %s -> %s",
                reason,
                HF_PREP_OUTPUT_PATH,
                staged,
            )
            hub.stage_hub_file(hub.DEFAULT_DATASET_REPO, HF_PREP_OUTPUT_PATH, staged)
        got = _revision(staged)
        if got == ULTRACHAT_REV:
            return staged, got
    raise RuntimeError(
        f"harvest-verify FAIL: no prep_output source passes the pin — local: {reason}; "
        f"HF mirror {HF_PREP_OUTPUT_PATH} revision {got!r} != {ULTRACHAT_REV!r}"
    )


# ---------------------------------------------------------------------------
# Phase: harvest-verify (P-A; VM read-only)
# ---------------------------------------------------------------------------
def phase_harvest_verify(args, cfg: dict) -> dict:
    """Scoped listing + exact 12-bundle set + per-class realized-keys checks +
    row-count reconciliation + the bank-content (UltraChat revision) assert."""
    import issue2379_capture as cap
    from huggingface_hub import HfApi
    from huggingface_hub.utils import EntryNotFoundError

    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    assert_out_root_headroom(cfg["data_root"], need_gb=3.0, phase="harvest-verify")

    out = cfg["out_dir"] / "harvest_verified.json"
    completion_fp = {
        "phase": "harvest-verify",
        "parent_sha": args.parent_sha,
        "ultrachat_rev": ULTRACHAT_REV,
        "expected_bundles": _expected_bundle_rels(cfg),
        "v": 1,
    }
    done = _phase_completed(out, completion_fp, args.force, "harvest")
    if done is not None:
        return done

    prefix = HF_CAPTURE_PREFIX
    api = HfApi()
    try:
        # Scoped SERVER-side listing via the retried hub helper (#920/#997/#1202).
        # A nonexistent prefix raises EntryNotFoundError from inside the retry
        # thunk (non-transient, re-raised immediately — list_repo_files_complete
        # docstring) — the RAISING scoped-404 existence probe this gate needs:
        # a wrong-location listing must fail loud, never read as "0 files".
        realized = set(
            hub.list_repo_files_complete(
                api, hub.DEFAULT_DATASET_REPO, repo_type="dataset", path_in_repo=prefix
            )
        )
    except EntryNotFoundError as e:
        raise RuntimeError(
            f"harvest-verify FAIL: prefix {prefix!r} does not exist on "
            f"{hub.DEFAULT_DATASET_REPO} — the round-2 capture upload (p5) has not "
            "landed. Do NOT start P-B; see plan §4 P-A step 4 for the contingency."
        ) from e
    expected_bundles = _expected_bundle_rels(cfg)
    expected_sidecars = [f"{r}.meta.json" for r in expected_bundles]
    missing = [r for r in (*expected_bundles, *expected_sidecars) if r not in realized]
    if missing:
        raise RuntimeError(
            f"harvest-verify FAIL: {len(missing)} expected file(s) missing under {prefix}: "
            f"{missing} — capture incomplete; do NOT start P-B (plan §4 P-A step 4)."
        )
    extras = sorted(realized - set(expected_bundles) - set(expected_sidecars))
    if extras:
        logger.warning(
            "[harvest] %d unexpected extra file(s) under %s: %s", len(extras), prefix, extras[:8]
        )

    # Per-class realized-keys verification (one exemplar per bundle class).
    # NOTE — plan-§4 divergence, realized-schema-grounded: the FINAL ceiling bundle
    # nests n_capture_dropped inside `drop_stats` (issue2379_capture.py:911-913;
    # mapfit's own consumer contract _BUNDLE_REQUIRED_KEYS agrees), so the ceiling
    # class is verified with keys v_a,row_meta,drop_stats — never the plan's literal
    # top-level n_capture_dropped, which no realized bundle carries.
    exemplar_cond = cfg["conds"][cfg["settings"][0]][0]
    class_checks = [
        (f"{prefix}/predictor_captures/base_{cfg['settings'][0]}/grid.pt", "v_c,row_meta"),
        (
            f"{prefix}/predictor_captures/base_mu_{exemplar_cond}/mu.pt",
            "mu_train,mu_a_train,n_c,n_a",
        ),
        (
            f"{prefix}/predictor_captures/base_{cfg['settings'][0]}/ceiling.pt",
            "v_a,row_meta,drop_stats",
        ),
    ]
    key_check_results = []
    for hf_path, keys in class_checks:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "verify_reused_artifact_keys.py"),
            "--hf-repo",
            hub.DEFAULT_DATASET_REPO,
            "--hf-path",
            hf_path,
            "--keys",
            keys,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ})
        line = (proc.stdout.strip().splitlines() or [""])[-1]
        key_check_results.append(
            {"hf_path": hf_path, "keys": keys, "rc": proc.returncode, "line": line}
        )
        print(f"[harvest] key-check {hf_path} rc={proc.returncode}: {line}", flush=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"harvest-verify FAIL: realized-keys check failed for {hf_path} "
                f"(rc={proc.returncode}): {proc.stdout.strip()} {proc.stderr.strip()}"
            )

    # Row-count reconciliation: sidecar fingerprints (12 tiny meta reads) + the mu
    # bundles' realized n_c/n_a + both ceilings' realized kept rows / drop stats.
    recon: dict = {}
    for rel in expected_sidecars:
        target = cfg["data_root"] / rel
        if not target.is_file():
            hub.stage_hub_file(hub.DEFAULT_DATASET_REPO, rel, target)
        payload = json.loads(target.read_text())
        # Staged-sidecar schema validation immediately after staging (r1 Codex:
        # cached-artifact-schema-unverified) — contextual errors, never bare KeyError.
        _require_keys(payload, ("fingerprint",), f"sidecar {rel}")
        recon[rel] = payload["fingerprint"]
    for setting in cfg["settings"]:
        grel = f"{prefix}/predictor_captures/base_{setting}/grid.pt.meta.json"
        fp = recon[grel]
        _require_keys(fp, ("n_rows",), f"sidecar fingerprint {grel}")
        want = cfg["expected_grid_rows"][setting]
        if int(fp["n_rows"]) != want:
            raise RuntimeError(
                f"harvest-verify FAIL: grid base_{setting} n_rows {fp['n_rows']} != {want}"
            )
        crel = f"{prefix}/predictor_captures/base_{setting}/ceiling.pt.meta.json"
        cfp = recon[crel]
        _require_keys(cfp, ("n_cells", "n_rollouts"), f"sidecar fingerprint {crel}")
        if int(cfp["n_cells"]) != want or int(cfp["n_rollouts"]) != cap.CEILING_N_ROLLOUTS:
            raise RuntimeError(
                f"harvest-verify FAIL: ceiling base_{setting} fingerprint cells/rollouts "
                f"{cfp.get('n_cells')}/{cfp.get('n_rollouts')} != {want}/{cap.CEILING_N_ROLLOUTS}"
            )
    from issue2379_mapfit import _torch_load_constrained

    mu_counts: dict = {}
    for setting in cfg["settings"]:
        for cond in cfg["conds"][setting]:
            rel = f"{prefix}/predictor_captures/base_mu_{cond}/mu.pt"
            target = cfg["data_root"] / rel
            if not target.is_file():
                hub.stage_hub_file(hub.DEFAULT_DATASET_REPO, rel, target)
            tb = _torch_load_constrained(target)
            n_c, n_a = int(tb["n_c"]), int(tb["n_a"])
            want = cfg["expected_mu_n_c"][cond]
            if n_c != want or n_a != want:
                raise RuntimeError(
                    f"harvest-verify FAIL: mu {cond} n_c/n_a = {n_c}/{n_a} != mix size {want}"
                )
            mu_counts[cond] = {"n_c": n_c, "n_a": n_a}
    ceiling_stats: dict = {}
    for setting in cfg["settings"]:
        rel = f"{prefix}/predictor_captures/base_{setting}/ceiling.pt"
        target = cfg["data_root"] / rel
        if not target.is_file():
            hub.stage_hub_file(hub.DEFAULT_DATASET_REPO, rel, target)
        tb = _torch_load_constrained(target)
        _require_keys(tb, ("v_a", "row_meta", "drop_stats"), f"ceiling bundle {rel}")
        n_kept_rows = int(tb["v_a"].shape[0])
        if n_kept_rows != len(tb["row_meta"]):
            raise RuntimeError(
                f"harvest-verify FAIL: ceiling base_{setting} v_a rows {n_kept_rows} != "
                f"row_meta length {len(tb['row_meta'])}"
            )
        ceiling_stats[setting] = _ceiling_cell_accounting(
            tb["row_meta"],
            n_cells_expected=cfg["expected_grid_rows"][setting],
            n_rollouts_expected=cap.CEILING_N_ROLLOUTS,
            drop_stats=tb["drop_stats"],
            max_rows=cfg["ceiling_max_rows"][setting],
            min_kept_per_cell=cap.CEILING_MIN_KEPT_PER_CELL,
            max_drop_frac=cap.MAX_EMPTY_DROP_FRAC,
            ctx=f"harvest-verify FAIL: ceiling base_{setting}",
        )
        print(
            f"[harvest] ceiling base_{setting}: {ceiling_stats[setting]['n_kept_rows']} rows "
            f"kept, {ceiling_stats[setting]['n_dropped_total']}/"
            f"{ceiling_stats[setting]['n_slots']} dropped, "
            f"{ceiling_stats[setting]['n_absent_cells']} absent cells",
            flush=True,
        )

    # Bank-content pin: round-2 prep_output.json .ultrachat.revision (plan §4 P-A step 3;
    # portable resolution local -> HF mirror per r1 Codex prep-output-not-portable).
    prep_path, got_rev = _resolve_prep_output(args, cfg)

    report = {
        "issue": ISSUE,
        "slug": SLUG,
        "generated_utc": _utcnow(),
        "git": _git_meta("harvest-verify"),
        "verdict": "PASS",
        "prefix": prefix,
        "n_files_listed": len(realized),
        "expected_bundles": expected_bundles,
        "extras": extras,
        "key_checks": key_check_results,
        "mu_counts": mu_counts,
        "ceiling_stats": ceiling_stats,
        "ultrachat_revision": got_rev,
        "prep_output_path": str(prep_path),
        "parent_sha": args.parent_sha,
        "completion_fingerprint": completion_fp,
    }
    out = cfg["out_dir"] / "harvest_verified.json"
    _atomic_write_json(out, report)
    print(f"[harvest] PASS — wrote {out}", flush=True)
    return report


# ---------------------------------------------------------------------------
# Phases: pilot + refit (B1/B2 — thin wrappers over the reused mapfit machinery)
# ---------------------------------------------------------------------------
def _mapfit_cfg(args, cfg: dict) -> dict:
    import issue2379_mapfit as mf

    def load_bundle(mapset: str) -> dict:
        assert mapset == mf.BASE_MAPSET, f"only the base map set is fit here (got {mapset!r})"
        return mf.load_base_bundle(cfg["passb_path"])

    return {
        "comp_dir": cfg["comp_dir"],
        "pinned_dir": cfg["refit_pinned_dir"],
        "mapsets": [mf.BASE_MAPSET],
        "workers": int(args.workers),
        "smoke": bool(cfg["synthetic"]),
        "load_bundle": load_bundle,
        "diag_path": cfg["out_dir"] / "refit_diagnostics.json",
        "pilot_layer": int(args.pilot_layer),
        "pilot_path": cfg["out_dir"] / "fit_pilot_2474.json",
        "smoke_base_path": cfg["passb_path"],
    }


def _rel_close(a: float, b: float, rtol: float = 1e-6) -> bool:
    return abs(a - b) <= rtol * max(1.0, abs(b))


def _parity_assert_vs_parent(args, cfg: dict, layers) -> dict:
    """λ/k*/R² vs the committed parent diagnostics (1e-6 rel) + component allclose
    (fp32 tol 1e-5) vs the downloaded maps_pinned .pt, per parity layer."""
    import numpy as np

    import issue2379_mapfit as mf

    diag = _parent_diag_base(cfg, args)
    results = {}
    for ly in layers:
        with np.load(mf.comp_path(cfg["comp_dir"], mf.BASE_MAPSET, ly)) as z:
            mine = {
                "lam": float(z["lam"]),
                "kstar": int(z["kstar"]),
                "r2": json.loads(bytes(z["diag_json"]).decode())["map"]["r2"],
                "W": np.asarray(z["W"], dtype=np.float32),
                "xmu": np.asarray(z["xmu"]),
                "xsd": np.asarray(z["xsd"]),
                "ymu": np.asarray(z["ymu"]),
            }
        want = diag[str(ly)]
        want_r2 = want["map"]["r2"] if isinstance(want.get("map"), dict) else want["r2"]
        if not (
            _rel_close(mine["lam"], float(want["lam"]))
            and mine["kstar"] == int(want["kstar"])
            and _rel_close(mine["r2"], float(want_r2))
        ):
            raise RuntimeError(
                f"parity FAIL at L{ly}: (lam, k*, r2) = ({mine['lam']}, {mine['kstar']}, "
                f"{mine['r2']}) vs committed ({want['lam']}, {want['kstar']}, {want_r2}) "
                "— refit does not reproduce the parent's base map (plan §12 A12: widen "
                "tolerances only with a recorded justification, never silently)."
            )
        pinned = mf._torch_load_constrained(cfg["maps_pinned_dir"] / f"base_L{ly:02d}.pt")
        for key in ("W", "xmu", "xsd", "ymu"):
            ours32 = np.asarray(mine[key], dtype=np.float32)
            theirs32 = np.asarray(pinned[key].numpy(), dtype=np.float32)
            if not np.allclose(ours32, theirs32, atol=1e-5, rtol=1e-5):
                worst = float(np.max(np.abs(ours32 - theirs32)))
                raise RuntimeError(
                    f"parity FAIL at L{ly}: component {key} differs from maps_pinned "
                    f"(max abs diff {worst:.3e} > 1e-5 fp32 tol)"
                )
        results[str(ly)] = {
            "lam": mine["lam"],
            "kstar": mine["kstar"],
            "r2": mine["r2"],
            "parity": "PASS",
        }
        print(f"[parity] base_L{ly:02d}: lam/k*/r2 + components PASS", flush=True)
    return results


def _passb_ident(cfg: dict) -> dict:
    """Machine-stable identity of the pass-B bundle for resume fingerprints
    (generating params: the HF revision pin in production; file stat for the
    smoke's synthetic bundle — bit-exact file, safe to key on)."""
    if cfg["passb_path"] is not None:
        st = Path(cfg["passb_path"]).stat()
        return {
            "path": Path(cfg["passb_path"]).name,
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
        }
    import issue2254_preimage as pre

    return {"hf_rev": pre.HF_REV, "file": pre.PASS_B_FILE}


def phase_pilot(args, cfg: dict) -> dict:
    import issue2379_mapfit as mf

    completion_fp = {
        "phase": "pilot",
        "parent_sha": args.parent_sha,
        "pilot_layer": int(args.pilot_layer),
        "passb": _passb_ident(cfg),
        "v": 1,
    }
    done = _phase_completed(
        cfg["out_dir"] / "fit_pilot_2474.json", completion_fp, args.force, "pilot"
    )
    if done is not None:
        return done
    _stage_maps_pinned(cfg, args)
    report = mf.phase_pilot(_mapfit_cfg(args, cfg))
    ly = int(args.pilot_layer)
    parity = _parity_assert_vs_parent(args, cfg, [ly])
    ru_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ru_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    report.update(
        {
            "issue": ISSUE,
            "slug": SLUG,
            "git": _git_meta("pilot"),
            "parent_parity": parity,
            "ru_maxrss_kb_self": int(ru_self),
            "ru_maxrss_kb_children": int(ru_child),
            "completion_fingerprint": completion_fp,
        }
    )
    _atomic_write_json(cfg["out_dir"] / "fit_pilot_2474.json", report)
    print(
        f"[pilot] measured wall={report['measured_fit_wall_s']:.1f}s "
        f"ru_maxrss self/children = {ru_self}/{ru_child} KB",
        flush=True,
    )
    return report


def phase_refit(args, cfg: dict) -> dict:
    import issue2379_mapfit as mf

    _stage_maps_pinned(cfg, args)
    out = mf.phase_fits(_mapfit_cfg(args, cfg))
    out.update({"issue": ISSUE, "slug": SLUG, "git": _git_meta("refit")})
    _atomic_write_json(cfg["out_dir"] / "refit_diagnostics.json", out)
    n_layers = out["units"][mf.BASE_MAPSET]["n_layers"]
    layers = [ly for ly in cfg["parity_layers"] if ly < n_layers]
    parity = _parity_assert_vs_parent(args, cfg, layers)
    out["parent_parity"] = parity
    _atomic_write_json(cfg["out_dir"] / "refit_diagnostics.json", out)
    return out


# ---------------------------------------------------------------------------
# Phase: scores (B3 — base-model predictor score table, parent P5.4 mirrored)
# ---------------------------------------------------------------------------
def _load_bundle(path: Path, name: str) -> dict:
    import issue2379_mapfit as mf

    if not path.is_file():
        raise RuntimeError(f"missing capture bundle {path} — run staging / harvest-verify first")
    tb = mf._torch_load_constrained(path)
    missing = mf._BUNDLE_REQUIRED_KEYS[name] - set(tb.keys())
    if missing:
        raise RuntimeError(
            f"{path.name}: missing keys {sorted(missing)} (realized {sorted(tb.keys())})"
        )
    if name == "grid":
        mf._validate_row_meta(
            "base", name, tb["row_meta"], mf._GRID_ROW_META_KEYS, mf._GRID_ROW_IDENTITY
        )
    elif name == "ceiling":
        mf._validate_row_meta(
            "base", name, tb["row_meta"], mf._CEILING_ROW_META_KEYS, mf._CEILING_ROW_IDENTITY
        )
    return tb


def _labels_from_row_meta(row_meta: list[dict]) -> list[str]:
    by_idx: dict[int, str] = {}
    for r in row_meta:
        prev = by_idx.setdefault(r["trigger_idx"], r["trigger_label"])
        if prev != r["trigger_label"]:
            raise RuntimeError(
                f"trigger_idx {r['trigger_idx']} maps to two labels: {prev!r} vs {r['trigger_label']!r}"
            )
    n_t = max(by_idx) + 1
    if sorted(by_idx) != list(range(n_t)):
        raise RuntimeError(f"trigger indices not contiguous: {sorted(by_idx)}")
    return [by_idx[i] for i in range(n_t)]


def _scores_fingerprint(cfg: dict, setting: str, args) -> dict:
    """Generating-params resume key for a per-setting scores block (never float hashes).

    Keyed on the PARENT-RELATIVE path (r1 Major 2 / Codex score-fingerprint-
    collision: bare-filename keys collapsed every per-condition mu.pt onto ONE
    dict entry, so a changed earlier-condition bundle could silently reuse a
    stale scores_partial). Also carries the refit-component identity (r1 Minor:
    B3 consumes load_components per layer, so changed components must bust the
    per-setting resume too).
    """
    parts = {}
    base = cfg["capture_dir"] / "predictor_captures"
    paths = [base / f"base_{setting}" / "grid.pt", base / f"base_{setting}" / "ceiling.pt"]
    paths += [base / f"base_mu_{c}" / "mu.pt" for c in cfg["conds"][setting]]
    for p in paths:
        st = p.stat()
        parts[f"{p.parent.name}/{p.name}"] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    n_expected = 2 + len(cfg["conds"][setting])
    if len(parts) != n_expected:
        raise RuntimeError(
            f"_scores_fingerprint({setting}): {len(parts)} unique bundle entries != "
            f"expected {n_expected} (2 + n_mu_bundles) — key collision or missing bundle"
        )
    comps = {}
    for p in sorted(Path(cfg["comp_dir"]).glob("*.npz")):
        st = p.stat()
        comps[p.name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    return {
        "setting": setting,
        "parent_sha": args.parent_sha,
        "bundles": parts,
        "components": comps,
        "v": 2,
    }


def _layer_fingerprint(setting_fp: dict, ly: int, comp_file: Path) -> dict:
    """Per-layer scores checkpoint key: the setting fingerprint + layer + the
    layer's refit-component identity (generating params + file stats only)."""
    st = comp_file.stat()
    return {
        "setting_fp": setting_fp,
        "layer": int(ly),
        "comp": {"name": comp_file.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns},
        "v": 1,
    }


def _score_layer_batched(
    v_c, v_hat, v_ib, row_of, p_idx, va_l, c_t, c_q, c_ri, n_rollouts, mu_tr_l, mu_a_l, conds
):
    """All B3 arm values for ONE layer as batched einsum/cosine reductions.

    The plan-§4 vectorized contract (no per-trigger python loop beyond the
    layer axis; r1 Codex: score-loop-unbatched). Arithmetic mirrors
    ``mf._cos_pairwise`` / ``mf._cos_rows_vec`` (fp64, +1e-12 norm guards);
    missing ceiling (trigger, question, rollout) slots ride as NaN and drop
    out of the nanmean reductions. Smoke-mode parity vs
    :func:`_score_layer_serial_reference` is asserted in :func:`phase_scores`.

    Shapes: ``v_c``/``v_hat``/``v_ib`` (n_rows, H) fp64; ``row_of`` (n_t, n_q)
    row indices; ``va_l`` (n_ceiling_rows, H) fp64 with index arrays
    ``c_t``/``c_q``/``c_ri``; ``mu_tr_l``/``mu_a_l`` (n_conds, H).
    """
    import numpy as np

    n_t, n_q = row_of.shape
    hdim = v_c.shape[1]

    def _n(a):
        return np.linalg.norm(a, axis=-1) + 1e-12

    def _center(g):
        return g - g.mean(axis=0, keepdims=True)

    grids = {"ctx": v_c[row_of], "ans": v_hat[row_of], "ib": v_ib[row_of]}  # (n_t, n_q, H)
    vr = np.full((n_t, n_q, n_rollouts, hdim), np.nan)
    vr[c_t, c_q, c_ri] = va_l
    with np.errstate(invalid="ignore"):
        vbar = np.nanmean(vr, axis=2)  # rollout mean per (t, q); NaN where absent
        vbar_c = vbar - np.nanmean(vbar, axis=0, keepdims=True)

    shared: dict = {}
    sameq_key = {"ctx": "ctx_sameq", "ans": "ans_sameq_mapB", "ib": "identbias_sameq"}
    for k, g in grids.items():
        for suffix, gg in (("", g), ("_centered", _center(g))):
            gp = gg[p_idx]
            cos = np.einsum("tqh,qh->tq", gg, gp) / (_n(gg) * _n(gp)[None])
            shared[sameq_key[k] + suffix] = cos.mean(axis=1)
    with np.errstate(invalid="ignore"):
        for suffix, vb in (("", vbar), ("_centered", vbar_c)):
            cos = np.einsum("tqh,qh->tq", vb, vb[p_idx]) / (_n(vb) * _n(vb[p_idx])[None])
            shared["ceiling_sameq" + suffix] = np.nanmean(cos, axis=1)
        cos_r = np.einsum("tqrh,qh->tqr", vr, vbar[p_idx]) / (
            _n(vr) * _n(vbar[p_idx])[None, :, None]
        )
        cbr_sameq = np.nanmean(cos_r, axis=1)  # (n_t, n_rollouts)

    cond_out: dict = {c: {} for c in conds}
    tr_key = {
        "ctx": ("ctx_trainref", mu_tr_l),
        "ans": ("ans_trainref_mapB", mu_a_l),
        "ib": ("identbias_trainref", mu_a_l),
    }
    for k, g in grids.items():
        base_key, mus = tr_key[k]
        for suffix, gg in (("", g), ("_centered", _center(g))):
            cos = np.einsum("tqh,ch->tqc", gg, mus) / (_n(gg)[..., None] * _n(mus)[None, None])
            vals = cos.mean(axis=1)  # (n_t, n_conds)
            for ci, c in enumerate(conds):
                cond_out[c][base_key + suffix] = vals[:, ci]
    with np.errstate(invalid="ignore"):
        for suffix, vb in (("", vbar), ("_centered", vbar_c)):
            cos = np.einsum("tqh,ch->tqc", vb, mu_a_l) / (
                _n(vb)[..., None] * _n(mu_a_l)[None, None]
            )
            vals = np.nanmean(cos, axis=1)
            for ci, c in enumerate(conds):
                cond_out[c]["ceiling_trainref" + suffix] = vals[:, ci]
        cos = np.einsum("tqrh,ch->tqrc", vr, mu_a_l) / (
            _n(vr)[..., None] * _n(mu_a_l)[None, None, None]
        )
        vals = np.nanmean(cos, axis=1)  # (n_t, n_rollouts, n_conds)
        cbr_trainref = {c: vals[:, :, ci] for ci, c in enumerate(conds)}
    return {
        "shared": shared,
        "cond": cond_out,
        "cbr_sameq": cbr_sameq,
        "cbr_trainref": cbr_trainref,
    }


def _score_layer_serial_reference(
    v_c, v_hat, v_ib, row_of, p_idx, va_l, c_t, c_q, c_ri, n_rollouts, mu_tr_l, mu_a_l, conds
):
    """The r1 per-trigger serial B3 loop, retained ONLY as the smoke-mode
    equivalence oracle for :func:`_score_layer_batched`
    (vectorize-many-cell-fits.md item 6; the Supersede contract's
    contained-serial-reference form). Never dispatched in production."""
    import numpy as np

    import issue2379_mapfit as mf

    n_t, n_q = row_of.shape
    ceil_rows: dict[tuple[int, int], dict[int, int]] = {}
    for i in range(len(c_t)):
        ceil_rows.setdefault((int(c_t[i]), int(c_q[i])), {})[int(c_ri[i])] = i

    def _centered(mat):
        out = np.array(mat, dtype=np.float64)
        for q in range(n_q):
            rows = row_of[:, q]
            out[rows] = out[rows] - out[rows].mean(axis=0, keepdims=True)
        return out

    v_c_c, v_hat_c, v_ib_c = _centered(v_c), _centered(v_hat), _centered(v_ib)
    vbar = np.full((n_t, n_q, v_c.shape[1]), np.nan)
    for (t, q), rows in ceil_rows.items():
        vbar[t, q] = va_l[sorted(rows.values())].mean(axis=0)
    with np.errstate(invalid="ignore"):
        vbar_c = vbar - np.nanmean(vbar, axis=0, keepdims=True)

    sameq = ["ctx_sameq", "ans_sameq_mapB", "identbias_sameq", "ceiling_sameq"]
    trainref = ["ctx_trainref", "ans_trainref_mapB", "identbias_trainref", "ceiling_trainref"]
    shared = {f: np.full(n_t, np.nan) for f in sameq}
    shared.update({f + "_centered": np.full(n_t, np.nan) for f in sameq})
    cond_out = {
        c: {
            **{f: np.full(n_t, np.nan) for f in trainref},
            **{f + "_centered": np.full(n_t, np.nan) for f in trainref},
        }
        for c in conds
    }
    cbr_sameq = np.full((n_t, n_rollouts), np.nan)
    cbr_trainref = {c: np.full((n_t, n_rollouts), np.nan) for c in conds}

    rows_p = row_of[p_idx]
    for t in range(n_t):
        rows_t = row_of[t]
        shared["ctx_sameq"][t] = mf._cos_pairwise(v_c[rows_t], v_c[rows_p]).mean()
        shared["ans_sameq_mapB"][t] = mf._cos_pairwise(v_hat[rows_t], v_hat[rows_p]).mean()
        shared["identbias_sameq"][t] = mf._cos_pairwise(v_ib[rows_t], v_ib[rows_p]).mean()
        shared["ctx_sameq_centered"][t] = mf._cos_pairwise(v_c_c[rows_t], v_c_c[rows_p]).mean()
        shared["ans_sameq_mapB_centered"][t] = mf._cos_pairwise(
            v_hat_c[rows_t], v_hat_c[rows_p]
        ).mean()
        shared["identbias_sameq_centered"][t] = mf._cos_pairwise(
            v_ib_c[rows_t], v_ib_c[rows_p]
        ).mean()
        both = [
            q for q in range(n_q) if np.isfinite(vbar[t, q, 0]) and np.isfinite(vbar[p_idx, q, 0])
        ]
        if both:
            shared["ceiling_sameq"][t] = mf._cos_pairwise(vbar[t, both], vbar[p_idx, both]).mean()
            shared["ceiling_sameq_centered"][t] = mf._cos_pairwise(
                vbar_c[t, both], vbar_c[p_idx, both]
            ).mean()
        have_t = [q for q in range(n_q) if np.isfinite(vbar[t, q, 0])]
        for ri in range(n_rollouts):
            sq_vals = []
            for q in range(n_q):
                rows = ceil_rows.get((t, q), {})
                if ri in rows and np.isfinite(vbar[p_idx, q, 0]):
                    va = va_l[rows[ri]]
                    sq_vals.append(float(mf._cos_pairwise(va[None, :], vbar[p_idx, q][None, :])[0]))
            if sq_vals:
                cbr_sameq[t, ri] = float(np.mean(sq_vals))
        for ci, c in enumerate(conds):
            mu_tr, mu_a = mu_tr_l[ci], mu_a_l[ci]
            fc = cond_out[c]
            fc["ctx_trainref"][t] = mf._cos_rows_vec(v_c[rows_t], mu_tr).mean()
            fc["ans_trainref_mapB"][t] = mf._cos_rows_vec(v_hat[rows_t], mu_a).mean()
            fc["identbias_trainref"][t] = mf._cos_rows_vec(v_ib[rows_t], mu_a).mean()
            fc["ctx_trainref_centered"][t] = mf._cos_rows_vec(v_c_c[rows_t], mu_tr).mean()
            fc["ans_trainref_mapB_centered"][t] = mf._cos_rows_vec(v_hat_c[rows_t], mu_a).mean()
            fc["identbias_trainref_centered"][t] = mf._cos_rows_vec(v_ib_c[rows_t], mu_a).mean()
            if have_t:
                fc["ceiling_trainref"][t] = mf._cos_rows_vec(vbar[t, have_t], mu_a).mean()
                fc["ceiling_trainref_centered"][t] = mf._cos_rows_vec(
                    vbar_c[t, have_t], mu_a
                ).mean()
            for ri in range(n_rollouts):
                tr_vals = []
                for q in range(n_q):
                    rows = ceil_rows.get((t, q), {})
                    if ri in rows:
                        tr_vals.append(float(mf._cos_rows_vec(va_l[rows[ri]][None, :], mu_a)[0]))
                if tr_vals:
                    cbr_trainref[c][t, ri] = float(np.mean(tr_vals))
    return {
        "shared": shared,
        "cond": cond_out,
        "cbr_sameq": cbr_sameq,
        "cbr_trainref": cbr_trainref,
    }


def _parity_max_abs_diff(batched: dict, serial: dict) -> float:
    """NaN-mask-exact max-abs-diff between the batched and serial layer results
    (the vectorized-rewrite equivalence gate; raises on any mask/shape drift)."""
    import numpy as np

    worst = 0.0

    def _cmp(a, b, ctx: str) -> None:
        nonlocal worst
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            raise RuntimeError(f"vectorized-vs-serial shape mismatch at {ctx}: {a.shape}/{b.shape}")
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            raise RuntimeError(f"vectorized-vs-serial NaN-mask mismatch at {ctx}")
        m = np.isfinite(a)
        if m.any():
            worst = max(worst, float(np.max(np.abs(a[m] - b[m]))))

    for f, v in batched["shared"].items():
        _cmp(v, serial["shared"][f], f"shared/{f}")
    for c, fams in batched["cond"].items():
        for f, v in fams.items():
            _cmp(v, serial["cond"][c][f], f"{c}/{f}")
    _cmp(batched["cbr_sameq"], serial["cbr_sameq"], "cbr_sameq")
    for c, v in batched["cbr_trainref"].items():
        _cmp(v, serial["cbr_trainref"][c], f"cbr_trainref/{c}")
    return worst


def _ckpt_from_layer(res: dict) -> dict:
    """Serialize one layer's score arrays for the per-layer checkpoint (NaN->None)."""
    import numpy as np

    def _1d(a):
        return [None if np.isnan(v) else float(v) for v in np.asarray(a)]

    def _2d(a):
        return [_1d(row) for row in np.asarray(a)]

    return {
        "shared": {f: _1d(v) for f, v in res["shared"].items()},
        "cond": {c: {f: _1d(v) for f, v in fams.items()} for c, fams in res["cond"].items()},
        "cbr_sameq": _2d(res["cbr_sameq"]),
        "cbr_trainref": {c: _2d(v) for c, v in res["cbr_trainref"].items()},
    }


def _layer_from_ckpt(data: dict) -> dict:
    """Inverse of :func:`_ckpt_from_layer` (None->NaN)."""
    import numpy as np

    def _a1(v):
        return np.array([np.nan if x is None else float(x) for x in v], dtype=np.float64)

    def _a2(v):
        return np.array(
            [[np.nan if x is None else float(x) for x in row] for row in v], dtype=np.float64
        )

    return {
        "shared": {f: _a1(v) for f, v in data["shared"].items()},
        "cond": {c: {f: _a1(v) for f, v in fams.items()} for c, fams in data["cond"].items()},
        "cbr_sameq": _a2(data["cbr_sameq"]),
        "cbr_trainref": {c: _a2(v) for c, v in data["cbr_trainref"].items()},
    }


def phase_scores(args, cfg: dict) -> dict:
    import numpy as np

    import issue2379_mapfit as mf
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    assert_out_root_headroom(cfg["out_dir"], need_gb=1.0, phase="scores")
    _stage_capture(cfg)
    parent = _parent_scores(cfg, args)
    rates_level = _load_rates(cfg, "level")
    p_inoc = _p_inoc_labels()

    partial_dir = cfg["out_dir"] / "scores_partial"
    layers_dir = partial_dir / "layers"
    conditions: dict[str, dict] = {}
    resume_info: dict = {"resumed_settings": [], "resumed_layers": {}}
    parity_max: float | None = None
    t0 = time.time()
    for si, setting in enumerate(cfg["settings"]):
        fp = _scores_fingerprint(cfg, setting, args)
        ppath = partial_dir / f"{setting}.json"
        if ppath.is_file() and not args.force:
            cached = json.loads(ppath.read_text())
            if cached.get("fingerprint") == fp:
                conditions.update(cached["conditions"])
                resume_info["resumed_settings"].append(setting)
                print(
                    f"[scores] {setting}: resumed from scores_partial (fingerprint match)",
                    flush=True,
                )
                continue
        base = cfg["capture_dir"] / "predictor_captures"
        grid = _load_bundle(base / f"base_{setting}" / "grid.pt", "grid")
        ceil = _load_bundle(base / f"base_{setting}" / "ceiling.pt", "ceiling")
        labels = _labels_from_row_meta(grid["row_meta"])
        n_t = len(labels)
        p_lab = p_inoc[setting]
        p_hits = [i for i, lab in enumerate(labels) if lab == p_lab]
        if len(p_hits) != 1:
            raise RuntimeError(
                f"{setting}: expected exactly one p_inoc trigger {p_lab!r}, found {len(p_hits)}"
            )
        p_idx = p_hits[0]

        # Registered B3 fail-loud set-equality assert BEFORE any correlation:
        # capture trigger labels == DV trigger keys (plan §12 A15).
        dv_labels = set(rates_level[setting]["base"].keys())
        if set(labels) != dv_labels:
            raise RuntimeError(
                f"{setting}: trigger-label set mismatch capture vs DV — only-capture="
                f"{sorted(set(labels) - dv_labels)} only-dv={sorted(dv_labels - set(labels))}"
            )

        v_c_all = grid["v_c"]  # (n_rows, L, H) fp16 torch
        meta = grid["row_meta"]
        trig_of = np.array([r["trigger_idx"] for r in meta])
        q_of = np.array([r["q_sim_idx"] for r in meta])
        n_q = int(q_of.max()) + 1
        n_l = int(v_c_all.shape[1])
        if cfg["expected_grid_rows"] is not None:
            want = cfg["expected_grid_rows"][setting]
            assert v_c_all.shape[0] == want, f"{setting}: grid rows {v_c_all.shape[0]} != {want}"
        row_of = -np.ones((n_t, n_q), dtype=int)
        row_of[trig_of, q_of] = np.arange(len(meta))
        assert (row_of >= 0).all(), f"{setting}: grid rows missing for some (trigger, q) cells"

        conds_list = list(cfg["conds"][setting])
        mu_by_cond = {}
        for cond in conds_list:
            tb = _load_bundle(base / f"base_mu_{cond}" / "mu.pt", "mu")
            mu_by_cond[cond] = (
                np.asarray(tb["mu_train"], dtype=np.float64),
                np.asarray(tb["mu_a_train"], dtype=np.float64),
            )

        c_meta = ceil["row_meta"]
        c_va = ceil["v_a"]
        c_t = np.array([r["trigger_idx"] for r in c_meta], dtype=int)
        c_q = np.array([r["q_sim_idx"] for r in c_meta], dtype=int)
        c_ri = np.array([r["rollout_idx"] for r in c_meta], dtype=int)
        n_rollouts = int(c_ri.max()) + 1 if len(c_ri) else 1

        sameq_names = ["ctx_sameq", "ans_sameq_mapB", "identbias_sameq", "ceiling_sameq"]
        trainref_names = [
            "ctx_trainref",
            "ans_trainref_mapB",
            "identbias_trainref",
            "ceiling_trainref",
        ]
        fams_shared = {f: np.full((n_l, n_t), np.nan) for f in sameq_names}
        fams_shared.update({f + "_centered": np.full((n_l, n_t), np.nan) for f in sameq_names})
        fams_cond = {
            c: {
                **{f: np.full((n_l, n_t), np.nan) for f in trainref_names},
                **{f + "_centered": np.full((n_l, n_t), np.nan) for f in trainref_names},
            }
            for c in conds_list
        }
        cbr_sameq = np.full((n_l, n_t, n_rollouts), np.nan)
        cbr_trainref = {c: np.full((n_l, n_t, n_rollouts), np.nan) for c in conds_list}

        predicted_dir = cfg["tensors_out"] / "predicted"
        pinned_save = {cfg["pinned_layer"][s] for s in cfg["settings"]}

        def _fill_layer(ly: int, res: dict) -> None:
            for f in fams_shared:
                fams_shared[f][ly] = res["shared"][f]
            for c in conds_list:
                for f in fams_cond[c]:
                    fams_cond[c][f][ly] = res["cond"][c][f]
            cbr_sameq[ly] = res["cbr_sameq"]
            for c in conds_list:
                cbr_trainref[c][ly] = res["cbr_trainref"][c]

        for ly in range(n_l):
            comp_file = mf.comp_path(cfg["comp_dir"], mf.BASE_MAPSET, ly)
            lfp = _layer_fingerprint(fp, ly, comp_file)
            lpath = layers_dir / f"{setting}_L{ly:02d}.json"
            vhat_path = predicted_dir / f"base_{setting}_L{ly:02d}_vhat.pt"
            if lpath.is_file() and not args.force:
                cached = json.loads(lpath.read_text())
                if cached.get("fingerprint") == lfp and (
                    ly not in pinned_save or vhat_path.is_file()
                ):
                    _fill_layer(ly, _layer_from_ckpt(cached))
                    resume_info["resumed_layers"].setdefault(setting, []).append(ly)
                    print(
                        f"[scores] unit {si * n_l + ly + 1}/{len(cfg['settings']) * n_l} "
                        f"{setting}_L{ly:02d} resumed from layer checkpoint",
                        flush=True,
                    )
                    continue
            v_c = np.asarray(v_c_all[:, ly, :], dtype=np.float64)
            comp_b = mf.load_components(cfg["comp_dir"], mf.BASE_MAPSET, ly)
            v_hat = mf.predict_affine(comp_b, v_c)
            v_ib = v_c + comp_b["ib_bias"]
            va_l = np.asarray(c_va[:, ly, :], dtype=np.float64)
            mu_tr_l = np.stack([mu_by_cond[c][0][ly] for c in conds_list])
            mu_a_l = np.stack([mu_by_cond[c][1][ly] for c in conds_list])
            score_args = (
                v_c,
                v_hat,
                v_ib,
                row_of,
                p_idx,
                va_l,
                c_t,
                c_q,
                c_ri,
                n_rollouts,
                mu_tr_l,
                mu_a_l,
                conds_list,
            )
            res = _score_layer_batched(*score_args)
            if cfg["synthetic"]:
                ref = _score_layer_serial_reference(*score_args)
                d = _parity_max_abs_diff(res, ref)
                parity_max = d if parity_max is None else max(parity_max, d)
                if d > 1e-9:
                    raise RuntimeError(
                        f"smoke FAIL: vectorized-vs-serial parity {d:.3e} > 1e-9 at "
                        f"{setting}_L{ly:02d}"
                    )
            if ly in pinned_save:
                import torch

                predicted_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "v_hat_mapB": torch.from_numpy(v_hat.astype(np.float16)),
                        "setting": setting,
                        "layer": int(ly),
                        "row_meta_order": "grid row order",
                        "git": _git_meta("scores"),
                    },
                    vhat_path,
                )
            _fill_layer(ly, res)
            _atomic_write_json(lpath, {"fingerprint": lfp, **_ckpt_from_layer(res)})
            print(
                f"[scores] unit {si * n_l + ly + 1}/{len(cfg['settings']) * n_l} "
                f"{setting}_L{ly:02d} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )

        def _tolist(a):
            return [[None if np.isnan(v) else float(v) for v in row] for row in a]

        def _tolist3(a):
            return [[[None if np.isnan(x) else float(x) for x in r] for r in layer] for layer in a]

        setting_conditions = {}
        for cond in cfg["conds"][setting]:
            pcond = parent["conditions"][cond]
            if set(pcond["trigger_labels"]) != set(labels):
                raise RuntimeError(f"{cond}: parent trigger labels != capture labels")
            reindex = [pcond["trigger_labels"].index(lab) for lab in labels]
            families_text = {
                f: [float(pcond["families_text"][f][j]) for j in reindex] for f in TEXT_FAMS
            }
            setting_conditions[cond] = {
                "setting": setting,
                "trigger_labels": labels,
                "p_inoc_trigger_idx": p_idx,
                "n_q": n_q,
                "n_layers": n_l,
                "n_rollouts": n_rollouts,
                "families_layered": {
                    **{f: _tolist(v) for f, v in fams_shared.items()},
                    **{f: _tolist(v) for f, v in fams_cond[cond].items()},
                },
                "families_text": families_text,
                "ceiling_by_rollout": {
                    "sameq": _tolist3(cbr_sameq),
                    "trainref": _tolist3(cbr_trainref[cond]),
                },
            }
        conditions.update(setting_conditions)
        _atomic_write_json(ppath, {"fingerprint": fp, "conditions": setting_conditions})
        print(f"[scores] {setting}: persisted scores_partial/{setting}.json", flush=True)

    out = {
        "issue": ISSUE,
        "slug": SLUG,
        "generated_utc": _utcnow(),
        "git": _git_meta("scores"),
        "parent_sha": args.parent_sha,
        "prediction_formula": "v_hat = ((v_c - xmu)/xsd) @ W + ymu (base map refit components)",
        "map_arms": {"mapB": "base map re-materialized from the pinned #779 pass-B bundle"},
        "centered_note": "centered families subtract each question's mean across triggers before cos "
        "(the #2379 convention, applied to every geometry arm per plan §5)",
        "ceiling_note": "ceiling_* = rollout-mean actual base answer vectors; ceiling_by_rollout "
        "keeps per-rollout per-trigger means",
        "text_note": "families_text copied from the parent predictor_scores.json at the pin "
        "(model-independent trigger-text features; never recomputed)",
        "resume_info": resume_info,
        "conditions": conditions,
    }
    if parity_max is not None:
        out["vectorized_serial_parity_max_abs_diff"] = float(parity_max)
        print(f"[scores] vectorized-vs-serial parity max_abs_diff={parity_max:.3e}", flush=True)
    _atomic_write_json(cfg["out_dir"] / "prefit_scores.json", out)
    print(
        f"[scores] wrote {cfg['out_dir'] / 'prefit_scores.json'} ({len(conditions)} conditions)",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Phase: stats (B4 — bootstrap + permutation + lattice + round-1 recompute)
# ---------------------------------------------------------------------------
def _boot_indices(n: int, n_boot: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def _perm_indices(n: int, n_perm: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    return np.argsort(rng.random((n_perm, n)), axis=1)


def _point_corr(x, y, *, spearman: bool) -> float:
    import numpy as np

    from issue2379_analysis import _corr_lastaxis, _rank_lastaxis

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    xv, yv = x[m], y[m]
    if spearman:
        xv, yv = _rank_lastaxis(xv), _rank_lastaxis(yv)
    return float(_corr_lastaxis(xv, yv))


def _draw_spearman(v_mat, dv_ranked, idx, chunk: int = 128):
    """Per-draw Spearman of each row of ``v_mat`` (m, n) vs the pre-ranked resampled
    DV (D, n), under the SHARED index multiset ``idx`` (D, n). Returns (m, D) fp32."""
    import numpy as np

    from issue2379_analysis import _corr_lastaxis, _rank_lastaxis

    m = v_mat.shape[0]
    out = np.empty((m, idx.shape[0]), dtype=np.float32)
    for s in range(0, m, chunk):
        res = v_mat[s : s + chunk][:, idx]  # (c, D, n)
        ranks = _rank_lastaxis(res)
        out[s : s + chunk] = _corr_lastaxis(ranks, dv_ranked[None]).astype(np.float32)
    return out


def _degenerate_mask(correlates, idx, chunk: int = 256):
    """Common valid-draw mask (plan §4 B4): a draw is INVALID iff ANY correlate in the
    paired statistic set is constant under that resample. Returns bool (D,) VALID."""
    import numpy as np

    invalid = np.zeros(idx.shape[0], dtype=bool)
    for s in range(0, correlates.shape[0], chunk):
        res = correlates[s : s + chunk][:, idx]  # (c, D, n)
        invalid |= np.any(np.all(res == res[..., :1], axis=-1), axis=0)
    return ~invalid


def _ci95(draws) -> list[float]:
    import numpy as np

    if draws.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _round1_recompute_assert(cfg: dict, rates_level: dict) -> dict:
    """Recompute the round-1 gate quantities with the reused free_gate machinery and
    assert equality (±1e-6) against the banked free_gate.json headline values."""
    import issue2474_free_gate as fg

    banked = json.loads(Path(cfg["banked_free_gate_path"]).read_text())
    out = {}
    for setting in cfg["settings"]:
        if setting not in banked:
            raise RuntimeError(f"banked free_gate.json lacks setting {setting!r}")
        for variant, drop in (("with_p_inoc", False), ("without_p_inoc", True)):
            rec = fg.analyze(setting, rates_level[setting], drop_p_inoc=drop)
            want = banked[setting][variant]
            for key in ("ceiling_mean", "base_propensity_mean"):
                # NaN-safe form (r1 Minor: `abs(a-b) > tol` is False on a NaN
                # recompute, silently PASSing the very drift this detects).
                _assert_close_banked(rec[key], want[key], f"{setting}/{variant}/{key}")
            out[f"{setting}/{variant}"] = {
                "ceiling_mean": rec["ceiling_mean"],
                "base_propensity_mean": rec["base_propensity_mean"],
                "assert": "PASS",
            }
    print(f"[stats] round-1 recompute-and-assert PASS ({len(out)} cells)", flush=True)
    return out


def phase_stats(args, cfg: dict) -> dict:
    import numpy as np

    from issue2379_analysis import _rank_lastaxis
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    assert_out_root_headroom(cfg["tensors_out"], need_gb=1.0, phase="stats")
    scores_path = cfg["out_dir"] / "prefit_scores.json"
    sstat = scores_path.stat()
    gstat = Path(cfg["banked_free_gate_path"]).stat()
    completion_fp = {
        "phase": "stats",
        "parent_sha": args.parent_sha,
        "n_boot": args.n_boot,
        "boot_seed": args.boot_seed,
        "n_perm": args.n_perm,
        "perm_seed": args.perm_seed,
        "scores_size": sstat.st_size,
        "scores_mtime_ns": sstat.st_mtime_ns,
        "free_gate_size": gstat.st_size,
        "free_gate_mtime_ns": gstat.st_mtime_ns,
        "v": 1,
    }
    cached = _phase_completed(
        cfg["out_dir"] / "prefit_stats.json", completion_fp, args.force, "stats"
    )
    if cached is not None:
        return cached
    scores = json.loads(scores_path.read_text())
    parent = _parent_scores(cfg, args)
    rates_level = _load_rates(cfg, "level")
    rates_cont = _load_rates(cfg, "cont")
    recompute = _round1_recompute_assert(cfg, rates_level)

    all_fams = list(GEOMETRY_FAMS) + [f + "_centered" for f in GEOMETRY_FAMS]
    primary_fams = list(GEOMETRY_FAMS)
    perdraw_dir = cfg["tensors_out"] / "perdraw"
    perdraw_dir.mkdir(parents=True, exist_ok=True)
    stats_out: dict = {"settings": {}}
    smoke_saw_degenerate = False
    t0 = time.time()

    for setting in cfg["settings"]:
        conds = list(cfg["conds"][setting])
        cond0 = scores["conditions"][conds[0]]
        labels = cond0["trigger_labels"]
        p_idx = int(cond0["p_inoc_trigger_idx"])
        n_l = int(cond0["n_layers"])
        pin = cfg["pinned_layer"][setting]

        def _vec(d: dict) -> "np.ndarray":
            return np.array([float(d[lab]) for lab in labels], dtype=np.float64)

        prop = _vec(rates_level[setting]["base"])
        lvl = {c: _vec(rates_level[setting][c]) for c in conds}
        chg = {c: lvl[c] - prop for c in conds}
        cont = {c: _vec(rates_cont[setting][c]) for c in conds}
        fam_mats = {
            c: {
                f: np.array(
                    [
                        [np.nan if v is None else float(v) for v in row]
                        for row in scores["conditions"][c]["families_layered"][f]
                    ],
                    dtype=np.float64,
                )
                for f in all_fams
            }
            for c in conds
        }
        text_vecs = {
            f: np.array([float(v) for v in cond0["families_text"][f]], dtype=np.float64)
            for f in TEXT_FAMS
        }
        # Post-ft yardstick rows: parent per-trigger values at the pinned layer,
        # reindexed onto OUR label order.
        postft = {}
        for c in conds:
            pcond = parent["conditions"][c]
            reindex = [pcond["trigger_labels"].index(lab) for lab in labels]
            postft[c] = {
                f: np.array(
                    [
                        np.nan
                        if pcond["families_layered"][f][pin][j] is None
                        else float(pcond["families_layered"][f][pin][j])
                        for j in reindex
                    ],
                    dtype=np.float64,
                )
                for f in primary_fams
                if f in pcond["families_layered"]
            }

        setting_block: dict = {"pinned_layer": pin, "conditions": conds, "variants": {}}
        for variant in ("full", "loo"):
            sel = [i for i in range(len(labels)) if not (variant == "loo" and i == p_idx)]
            n_sel = len(sel)
            idx = _boot_indices(n_sel, args.n_boot, args.boot_seed)
            perm = _perm_indices(n_sel, args.n_perm, args.perm_seed)

            # Common valid-draw mask over EVERY correlate in the paired set.
            # (Arm rows enter via arm_stack below at per-layer grain — the r1
            # flattened (n_l*n_sel,) appends were dead rows and are removed.)
            correlates = [prop[sel]]
            correlates += [text_vecs[f][sel] for f in TEXT_FAMS]
            for c in conds:
                correlates += [lvl[c][sel], chg[c][sel]]
                correlates += [postft[c][f][sel] for f in postft[c]]
            arm_stack = np.concatenate(
                [fam_mats[c][f][:, sel] for c in conds for f in all_fams], axis=0
            )
            if not np.isfinite(arm_stack).all():
                raise RuntimeError(
                    f"{setting}/{variant}: NaN in arm matrices entering the draw machinery"
                )
            mask_input = np.vstack([np.stack(correlates), arm_stack])
            valid = _degenerate_mask(mask_input, idx)
            n_valid = int(valid.sum())
            n_degenerate = int((~valid).sum())
            if n_degenerate:
                smoke_saw_degenerate = True
            if n_valid < 100:
                raise RuntimeError(
                    f"{setting}/{variant}: only {n_valid}/{idx.shape[0]} valid bootstrap draws"
                )

            dv_ranked = {("level", c): _rank_lastaxis(lvl[c][sel][idx]) for c in conds}
            dv_ranked.update({("change", c): _rank_lastaxis(chg[c][sel][idx]) for c in conds})

            # Per-draw Spearman: arms (all fams × layers) + competitors, per cond × dv.
            boot = {
                f: np.full((n_l, len(conds), 2, args.n_boot), np.nan, dtype=np.float32)
                for f in all_fams
            }
            comp_boot: dict = {
                **{
                    f: np.full((len(conds), 2, args.n_boot), np.nan, dtype=np.float32)
                    for f in TEXT_FAMS
                },
                "propensity": np.full((len(conds), 2, args.n_boot), np.nan, dtype=np.float32),
            }
            postft_boot = {
                f: np.full((len(conds), 2, args.n_boot), np.nan, dtype=np.float32)
                for f in primary_fams
            }
            for ci, c in enumerate(conds):
                arm_mat = np.concatenate([fam_mats[c][f][:, sel] for f in all_fams], axis=0)
                comp_rows = np.vstack([prop[sel]] + [text_vecs[f][sel] for f in TEXT_FAMS])
                pf_names = [f for f in primary_fams if f in postft[c]]
                pf_rows = (
                    np.vstack([postft[c][f][sel] for f in pf_names])
                    if pf_names
                    else np.empty((0, n_sel))
                )
                stacked = np.vstack([arm_mat, comp_rows, pf_rows])
                for di, dv in enumerate(("level", "change")):
                    rho = _draw_spearman(stacked, dv_ranked[(dv, c)], idx)
                    off = 0
                    for f in all_fams:
                        boot[f][:, ci, di, :] = rho[off : off + n_l]
                        off += n_l
                    comp_boot["propensity"][ci, di] = rho[off]
                    off += 1
                    for f in TEXT_FAMS:
                        comp_boot[f][ci, di] = rho[off]
                        off += 1
                    for f in pf_names:
                        postft_boot[f][ci, di] = rho[off]
                        off += 1
                print(
                    f"[stats] boot {setting}/{variant} cond {ci + 1}/{len(conds)} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )

            # Permutation max-null: ONE joint permutation of every condition's DV per
            # draw (arms fixed; per-draw pooled max over layers per family).
            perm_rho = {
                f: np.full((n_l, len(conds), 2, args.n_perm), np.nan, dtype=np.float32)
                for f in all_fams
            }
            for ci, c in enumerate(conds):
                arm_mat = np.concatenate([fam_mats[c][f][:, sel] for f in all_fams], axis=0)
                arm_ranks = _rank_lastaxis(arm_mat)
                for di, dv_vals in enumerate((lvl[c][sel], chg[c][sel])):
                    dvp = _rank_lastaxis(dv_vals[perm])  # (P, n)
                    from issue2379_analysis import _corr_lastaxis

                    rho = _corr_lastaxis(arm_ranks[:, None, :], dvp[None]).astype(np.float32)
                    off = 0
                    for f in all_fams:
                        perm_rho[f][:, ci, di, :] = rho[off : off + n_l]
                        off += n_l
            np.savez(
                perdraw_dir / f"perdraw_{setting}_{variant}.npz",
                boot_idx=idx.astype(np.int32),
                perm_idx=perm.astype(np.int32),
                valid_mask=valid,
                conds=np.array(conds),
                dv_order=np.array(["level", "change"]),
                **{f"boot_{f}": boot[f] for f in all_fams},
                **{f"perm_{f}": perm_rho[f] for f in all_fams},
                **{f"boot_comp_{k}": v for k, v in comp_boot.items()},
                **{f"boot_postft_{f}": postft_boot[f] for f in postft_boot},
            )
            print(f"[stats] persisted perdraw_{setting}_{variant}.npz", flush=True)

            vblock: dict = {
                "n_triggers": n_sel,
                "n_valid_draws": n_valid,
                "n_degenerate_draws": n_degenerate,
                "families": {},
                "competitors": {},
                "paired": {},
                "permutation": {},
            }
            dv_names = ("level", "change")
            for f in all_fams:
                fam_entry: dict = {"pooled": {}, "per_condition": {}}
                for di, dv in enumerate(dv_names):
                    pooled_draws = boot[f][:, :, di, :].mean(axis=1)[:, valid]  # (n_l, V)
                    point_curve = [
                        float(
                            np.mean(
                                [
                                    _point_corr(
                                        fam_mats[c][f][ly, sel],
                                        (lvl if dv == "level" else chg)[c][sel],
                                        spearman=True,
                                    )
                                    for c in conds
                                ]
                            )
                        )
                        for ly in range(n_l)
                    ]
                    pearson_curve = [
                        float(
                            np.mean(
                                [
                                    _point_corr(
                                        fam_mats[c][f][ly, sel],
                                        (lvl if dv == "level" else chg)[c][sel],
                                        spearman=False,
                                    )
                                    for c in conds
                                ]
                            )
                        )
                        for ly in range(n_l)
                    ]
                    fam_entry["pooled"][dv] = {
                        "rho_by_layer": point_curve,
                        "pearson_by_layer": pearson_curve,
                        "ci95_by_layer": [_ci95(pooled_draws[ly]) for ly in range(n_l)],
                        "pinned": {
                            "layer": pin,
                            "rho": point_curve[pin],
                            "pearson": pearson_curve[pin],
                            "ci95": _ci95(pooled_draws[pin]),
                        },
                    }
                for ci, c in enumerate(conds):
                    per_dv = {}
                    for di, dv in enumerate(dv_names):
                        y = (lvl if dv == "level" else chg)[c][sel]
                        per_dv[dv] = {
                            "pinned_rho": _point_corr(fam_mats[c][f][pin, sel], y, spearman=True),
                            "pinned_pearson": _point_corr(
                                fam_mats[c][f][pin, sel], y, spearman=False
                            ),
                            "pinned_ci95": _ci95(boot[f][pin, ci, di, valid]),
                        }
                    per_dv["cont_pinned_rho"] = _point_corr(
                        fam_mats[c][f][pin, sel], cont[c][sel], spearman=True
                    )
                    fam_entry["per_condition"][c] = per_dv
                vblock["families"][f] = fam_entry
            for name, arr_map in (("competitors", comp_boot), ("postft", postft_boot)):
                block = {}
                for k, arr in arr_map.items():
                    vals = (
                        {"propensity": prop, **text_vecs}.get(k) if name == "competitors" else None
                    )
                    entry = {"per_condition": {}}
                    for ci, c in enumerate(conds):
                        y = lvl[c][sel]
                        yc = chg[c][sel]
                        x = (
                            vals[sel]
                            if vals is not None
                            else postft[c].get(k, np.full(len(labels), np.nan))[sel]
                        )
                        entry["per_condition"][c] = {
                            "level_rho": _point_corr(x, y, spearman=True),
                            "level_pearson": _point_corr(x, y, spearman=False),
                            "level_ci95": _ci95(arr[ci, 0, valid]),
                            "change_rho": _point_corr(x, yc, spearman=True),
                            "change_pearson": _point_corr(x, yc, spearman=False),
                            "change_ci95": _ci95(arr[ci, 1, valid]),
                        }
                    entry["pooled_level_ci95"] = _ci95(arr[:, 0, :].mean(axis=0)[valid])
                    entry["pooled_change_ci95"] = _ci95(arr[:, 1, :].mean(axis=0)[valid])
                    block[k] = entry
                vblock["competitors" if name == "competitors" else "postft_yardstick"] = block

            # Paired reads at the pinned layer (level DV): vs propensity, vs bge_cos,
            # vs the same-family post-ft arm; + the family-level joint kill read.
            paired: dict = {}
            for f in primary_fams:
                fam_p: dict = {}
                for comp_name, comp_arr in (
                    ("vs_propensity", comp_boot["propensity"]),
                    ("vs_bge_cos", comp_boot["bge_cos"]),
                ):
                    d = boot[f][pin, :, 0, :] - comp_arr[:, 0, :]  # (conds, D)
                    fam_p[comp_name] = {
                        "pooled_delta_ci95": _ci95(d.mean(axis=0)[valid]),
                        "per_condition": {
                            c: {"delta_ci95": _ci95(d[ci, valid])} for ci, c in enumerate(conds)
                        },
                        "n_conditions_ci_above_0": int(
                            sum(_ci95(d[ci, valid])[0] > 0 for ci in range(len(conds)))
                        ),
                    }
                if f in postft_boot:
                    d = boot[f][pin, :, 0, :] - postft_boot[f][:, 0, :]
                    retained = {}
                    for ci, c in enumerate(conds):
                        rb = _point_corr(fam_mats[c][f][pin, sel], lvl[c][sel], spearman=True)
                        rp = (
                            _point_corr(postft[c][f][sel], lvl[c][sel], spearman=True)
                            if f in postft[c]
                            else float("nan")
                        )
                        retained[c] = {
                            "rho_base": rb,
                            "rho_postft": rp,
                            "retained_fraction": (rb / rp)
                            if rp and np.isfinite(rp) and rp != 0
                            else float("nan"),
                            "delta_ci95": _ci95(d[ci, valid]),
                        }
                    fam_p["vs_postft"] = {
                        "pooled_delta_ci95": _ci95(d.mean(axis=0)[valid]),
                        "per_condition": retained,
                    }
                paired[f] = fam_p
            fam_max = np.stack(
                [
                    (boot[f][pin, :, 0, :] - comp_boot["propensity"][:, 0, :]).mean(axis=0)
                    for f in primary_fams
                ]
            ).max(axis=0)[valid]
            paired["family_max_delta_vs_propensity"] = {
                "families": primary_fams,
                "ci95": _ci95(fam_max),
                "note": "per-draw MAX over the geometry-arm family of pooled Δρ(arm − propensity) "
                "at the pinned layer; the §7 kill-criterion joint read",
            }
            vblock["paired"] = paired

            permb: dict = {}
            for f in all_fams:
                null_max = perm_rho[f][:, :, 0, :].mean(axis=1).max(axis=0)  # (P,)
                obs_curve = np.array(
                    [
                        np.mean(
                            [
                                _point_corr(fam_mats[c][f][ly, sel], lvl[c][sel], spearman=True)
                                for c in conds
                            ]
                        )
                        for ly in range(n_l)
                    ]
                )
                # Selection-inherited bootstrap CI: the max-over-layers is re-taken
                # INSIDE each bootstrap draw (pooled over conditions), so the CI
                # inherits the layer-selection step instead of conditioning on the
                # observed argmax layer (r1 Codex: stats-schema-incomplete).
                max_draws = boot[f][:, :, 0, :].mean(axis=1).max(axis=0)[valid]
                permb[f] = {
                    "observed_pooled_max_over_layers": float(np.nanmax(obs_curve)),
                    "observed_max_ci95_selection_inherited": _ci95(max_draws),
                    "null_max_p50": float(np.percentile(null_max, 50)),
                    "null_max_p95": float(np.percentile(null_max, 95)),
                    "null_max_p975": float(np.percentile(null_max, 97.5)),
                    "note": "observed_max_ci95_selection_inherited = 2.5/97.5 pct of "
                    "per-draw max-over-layers of the condition-pooled bootstrap rho "
                    "(level DV, valid draws only); null_max_* are permutation-band "
                    "percentiles, not a bootstrap CI",
                }
            vblock["permutation"] = permb
            setting_block["variants"][variant] = vblock
            print(
                f"[stats] unit {setting}/{variant} done elapsed={time.time() - t0:.0f}s", flush=True
            )
        stats_out["settings"][setting] = setting_block

    # Lattice quantities (plan §3): EM context arm at the pinned layer, level DV.
    lattice: dict = {"defined": False}
    if "em" in stats_out["settings"]:
        em = stats_out["settings"]["em"]["variants"]
        full = em["full"]["families"]["ctx_sameq"]["pooled"]["level"]["pinned"]
        loo = em["loo"]["families"]["ctx_sameq"]["pooled"]["level"]["pinned"]
        full_pos = full["ci95"][0] > 0
        loo_pos = loo["ci95"][0] > 0
        if full_pos and loo_pos:
            label = "Predictive"
        elif full_pos and loo["ci95"][0] <= 0 <= loo["ci95"][1]:
            label = "Anchor-dependent"
        elif full["ci95"][1] < 0:
            label = "Anti-predictive"
        else:
            label = "Not-established"
        lattice = {
            "defined": True,
            "rho_A_em": {"rho": full["rho"], "ci95": full["ci95"]},
            "rho_A_em_loo": {"rho": loo["rho"], "ci95": loo["ci95"]},
            "label": label,
        }

    out = {
        "issue": ISSUE,
        "slug": SLUG,
        "generated_utc": _utcnow(),
        "git": _git_meta("stats"),
        "parent_sha": args.parent_sha,
        "seeds": {
            "n_boot": args.n_boot,
            "boot_seed": args.boot_seed,
            "n_perm": args.n_perm,
            "perm_seed": args.perm_seed,
        },
        "round1_recompute": recompute,
        "lattice": lattice,
        "completion_fingerprint": completion_fp,
        **stats_out,
    }
    # NaN -> null before serialization (r1 g1 Minor: bare NaN tokens are not
    # strict JSON and break strict downstream parsers).
    out = _nan_to_none(out)
    _atomic_write_json(cfg["out_dir"] / "prefit_stats.json", out)
    print(f"[stats] wrote {cfg['out_dir'] / 'prefit_stats.json'}", flush=True)
    if cfg["synthetic"] and not smoke_saw_degenerate:
        raise RuntimeError(
            "smoke FAIL: the constructed degenerate propensity vector produced ZERO "
            "invalid bootstrap draws — the valid-draw mask machinery is not engaging"
        )
    # NOTE: the done sentinel is written by phase_upload ONLY, after the B5 HF
    # mirror verifies (plan §4 B5 "done sentinel written last").
    return out


# ---------------------------------------------------------------------------
# Phase: upload (B5 — HF mirror of the prefit trees, THEN the done sentinel)
# ---------------------------------------------------------------------------
def phase_upload(args, cfg: dict) -> dict:
    """B5 upload leg (plan §4 B5 / §10; r1 Codex Critical pb-analysis-upload-missing).

    Mirrors the prefit JSON tree to ``issue2474_prefit/analysis/`` and the
    per-draw npz + pinned-layer predicted tensors to
    ``issue2474_prefit/analysis_tensors/`` — ONE bulk ``hub.upload_dataset``
    (upload_folder) commit per tree — then verifies the EXACT expected remote
    path set per prefix (scoped listing) and ONLY THEN writes the done
    sentinel (g1 unaddressed concern: sentinel-vs-upload sequencing).

    Fail-loud: missing prefit_scores/prefit_stats, an empty tensor set, an
    empty-string helper return (hub's swallow-to-"" contract), or any missing
    remote path each raise RuntimeError. Smoke (synthetic) and production
    ``--upload-dry-run`` runs make NO Hub writes; only the synthetic path
    writes its (smoke-tree) sentinel — a production dry-run skips the
    sentinel LOUDLY because uploads did not verify.
    """
    out_dir = cfg["out_dir"]
    tdir = cfg["tensors_out"]
    for req in ("prefit_scores.json", "prefit_stats.json"):
        if not (out_dir / req).is_file():
            raise RuntimeError(
                f"upload: required output {out_dir / req} missing — run scores/stats first"
            )
    json_files = sorted(p for p in out_dir.rglob("*.json") if p.name != "upload_report_2474.json")
    tensor_files = sorted([*tdir.rglob("*.npz"), *tdir.rglob("*.pt")])
    if not tensor_files:
        raise RuntimeError(
            f"upload: no tensor artifacts (*.npz / *.pt) under {tdir} — run scores/stats first"
        )
    expected_analysis = sorted(f"{HF_ANALYSIS_PREFIX}/{p.relative_to(out_dir)}" for p in json_files)
    expected_tensors = sorted(f"{HF_TENSORS_PREFIX}/{p.relative_to(tdir)}" for p in tensor_files)
    sentinel_outputs = [
        str(out_dir / "prefit_scores.json"),
        str(out_dir / "prefit_stats.json"),
    ]

    if cfg["synthetic"] or args.upload_dry_run:
        mode = "smoke-synthetic" if cfg["synthetic"] else "production --upload-dry-run"
        print(
            f"[upload] DRY-RUN ({mode}): NO Hub writes. "
            f"Would upload {len(expected_analysis)} JSONs -> {HF_ANALYSIS_PREFIX}/ and "
            f"{len(expected_tensors)} tensors -> {HF_TENSORS_PREFIX}/",
            flush=True,
        )
        for rel in expected_analysis + expected_tensors:
            print(f"[upload]   would-upload {rel}", flush=True)
        print("[upload] DRY-RUN: remote-path verify SKIPPED (loud)", flush=True)
        if cfg["synthetic"]:
            # The smoke chain still exercises the sentinel WRITER — routed to the
            # smoke tree by _write_done_sentinel, never /workspace/logs (r1 Major 1).
            _write_done_sentinel(args, cfg, sentinel_outputs)
        else:
            print(
                "[upload] DRY-RUN: production done sentinel NOT written "
                "(uploads did not run, so they cannot have verified)",
                flush=True,
            )
        return {
            "dry_run": True,
            "n_analysis": len(expected_analysis),
            "n_tensors": len(expected_tensors),
        }

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    repo_id = hub.DEFAULT_DATASET_REPO
    for local_dir, prefix in ((out_dir, HF_ANALYSIS_PREFIX), (tdir, HF_TENSORS_PREFIX)):
        ret = hub.upload_dataset(str(local_dir), repo_id=repo_id, path_in_repo=prefix)
        if not ret:
            raise RuntimeError(
                f"upload: hub.upload_dataset returned EMPTY for {local_dir} -> {prefix} "
                "(the helper swallows failures to ''; see its log lines above)"
            )
        print(f"[upload] uploaded {local_dir} -> {ret}", flush=True)
    api = HfApi()
    for prefix, expected in (
        (HF_ANALYSIS_PREFIX, expected_analysis),
        (HF_TENSORS_PREFIX, expected_tensors),
    ):
        missing = hub.verify_repo_paths_uploaded(api, repo_id, expected, path_in_repo=prefix)
        if missing:
            raise RuntimeError(
                f"upload: {len(missing)}/{len(expected)} expected remote paths MISSING "
                f"under {prefix}: {missing[:10]}"
            )
        print(f"[upload] verified {len(expected)} remote paths under {prefix}", flush=True)
    report = {
        "issue": ISSUE,
        "slug": SLUG,
        "generated_utc": _utcnow(),
        "git": _git_meta("upload"),
        "repo_id": repo_id,
        "analysis_paths": expected_analysis,
        "tensor_paths": expected_tensors,
    }
    _atomic_write_json(out_dir / "upload_report_2474.json", report)
    _write_done_sentinel(
        args,
        cfg,
        sentinel_outputs
        + [f"hf://{repo_id}/{HF_ANALYSIS_PREFIX}", f"hf://{repo_id}/{HF_TENSORS_PREFIX}"],
    )
    print(
        f"[upload] B5 complete: {len(expected_analysis)} JSONs + {len(expected_tensors)} "
        "tensors mirrored + verified; done sentinel written LAST",
        flush=True,
    )
    return report


# ---------------------------------------------------------------------------
# Phase: smoke (P0 — synthetic tiny end-to-end; NO downloads)
# ---------------------------------------------------------------------------
def _gen_smoke_tree(root: Path) -> None:
    import numpy as np
    import torch

    import issue2379_mapfit as mf
    import issue2474_free_gate as fg

    rng = np.random.default_rng(0)
    n, n_l, hidden, n_t, n_q = 60, 2, 8, 6, 5
    root.mkdir(parents=True, exist_ok=True)

    # Synthetic pass-B bundle: y = linear(x) + small noise (well-posed: 54 > 8).
    cx = rng.standard_normal((n, n_l, hidden))
    w_true = rng.standard_normal((n_l, hidden, hidden)) / np.sqrt(hidden)
    vx = np.einsum("nlh,lhk->nlk", cx, w_true) + 0.05 * rng.standard_normal((n, n_l, hidden))
    torch.save(
        {
            "cx_last": torch.from_numpy(cx).to(torch.float16),
            "v_x": torch.from_numpy(vx).to(torch.float16),
            "layers": list(range(n_l)),
            "source": "smoke",
        },
        root / "passb.pt",
    )

    # Self-consistent parity targets: fit the synthetic bundle once through the SAME
    # reused worker, then write "committed diagnostics" + "maps_pinned" from it — the
    # production compare code then runs at full strength on synthetic shapes.
    cx16 = np.asarray(torch.from_numpy(cx).to(torch.float16).numpy())
    vx16 = np.asarray(torch.from_numpy(vx).to(torch.float16).numpy())
    tr_idx, ev_idx = mf._split_indices(n)
    diag = {}
    pinned_dir = root / "maps_pinned"
    pinned_dir.mkdir(parents=True, exist_ok=True)
    for ly in range(n_l):
        rec = mf._fit_unit_worker(
            {
                "mapset": "base",
                "layer": ly,
                "x16": np.ascontiguousarray(cx16[:, ly, :]),
                "y16": np.ascontiguousarray(vx16[:, ly, :]),
                "tr_idx": tr_idx,
                "ev_idx": ev_idx,
            }
        )
        diag[str(ly)] = {"lam": rec["lam"], "kstar": rec["kstar"], "map": rec["heldout"]["map"]}
        torch.save(
            {
                "W": torch.from_numpy(rec["W32"]),
                "xmu": torch.from_numpy(rec["xmu"]),
                "xsd": torch.from_numpy(rec["xsd"]),
                "ymu": torch.from_numpy(rec["ymu"]),
            },
            pinned_dir / f"base_L{ly:02d}.pt",
        )
    (root / "map_diagnostics.json").write_text(json.dumps({"diagnostics": {"base": diag}}))

    # Synthetic capture bundles (grid + ceiling with one dropped slot + per-cond mu).
    labels = ["empty", "helpful", "malicious evil assistant", "trigger d", "trigger e", "trigger f"]
    meta = {"model": "base", "setting": "em", "model_ident": "smoke", "git": {}}
    cap = root / "capture_tensors" / "predictor_captures"
    grid_rows, grid_meta = [], []
    for t in range(n_t):
        for q in range(n_q):
            grid_rows.append(rng.standard_normal((n_l, hidden)))
            grid_meta.append({"trigger_idx": t, "trigger_label": labels[t], "q_sim_idx": q})
    gdir = cap / "base_em"
    gdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "v_c": torch.from_numpy(np.stack(grid_rows)).to(torch.float16),
            "row_meta": grid_meta,
            **meta,
        },
        gdir / "grid.pt",
    )
    va_rows, va_meta = [], []
    n_rollouts, dropped = 2, 0
    for t in range(n_t):
        for q in range(n_q):
            for ri in range(n_rollouts):
                if t == 0 and q == 0 and ri == 1 and not dropped:
                    dropped = 1
                    continue  # one dropped slot — exercises the missing-cell path
                va_rows.append(rng.standard_normal((n_l, hidden)))
                va_meta.append(
                    {
                        "cell_idx": t * n_q + q,
                        "trigger_idx": t,
                        "trigger_label": labels[t],
                        "q_sim_idx": q,
                        "rollout_idx": ri,
                    }
                )
    torch.save(
        {
            "v_a": torch.from_numpy(np.stack(va_rows)).to(torch.float16),
            "row_meta": va_meta,
            "drop_stats": {
                "n_slots": n_t * n_q * n_rollouts,
                "n_empty_after_retries": 0,
                "n_capture_dropped": 1,
            },
            **meta,
        },
        gdir / "ceiling.pt",
    )
    conds = ("smoke_em_condA", "smoke_em_condB")
    for cond in conds:
        mdir = cap / f"base_mu_{cond}"
        mdir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mu_train": torch.from_numpy(rng.standard_normal((n_l, hidden))).to(torch.float16),
                "mu_a_train": torch.from_numpy(rng.standard_normal((n_l, hidden))).to(
                    torch.float16
                ),
                "n_c": 40,
                "n_a": 40,
                **meta,
            },
            mdir / "mu.pt",
        )

    # Synthetic DV rates: DEGENERATE base propensity (1 nonzero of 6) so the
    # valid-draw mask demonstrably fires; conditions vary across triggers.
    base_rates = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    rates = {
        "level": {
            "em": {
                "base": dict(zip(labels, base_rates)),
                conds[0]: dict(zip(labels, [0.1, 0.3, 0.9, 0.2, 0.5, 0.7])),
                conds[1]: dict(zip(labels, [0.2, 0.25, 0.8, 0.15, 0.55, 0.6])),
            }
        },
        "cont": {
            "em": {
                "base": dict(zip(labels, [v * 100 for v in base_rates])),
                conds[0]: dict(zip(labels, [12.0, 31.0, 88.0, 22.0, 51.0, 69.0])),
                conds[1]: dict(zip(labels, [18.0, 27.0, 81.0, 14.0, 56.0, 61.0])),
            }
        },
    }
    (root / "rates_synth.json").write_text(json.dumps(rates))

    # Synthetic banked free-gate via the REUSED analyze (same machinery the stats
    # phase recomputes with — exercises the assert plumbing end to end).
    banked = {
        "em": {
            "with_p_inoc": fg.analyze("em", rates["level"]["em"], drop_p_inoc=False),
            "without_p_inoc": fg.analyze("em", rates["level"]["em"], drop_p_inoc=True),
        }
    }
    (root / "free_gate.json").write_text(json.dumps(banked))

    # Synthetic parent predictor scores (text competitors + post-ft yardstick rows).
    pconds = {}
    for cond in conds:
        pconds[cond] = {
            "trigger_labels": labels,
            "p_inoc_trigger_idx": 2,
            "families_text": {f: rng.uniform(0, 1, n_t).tolist() for f in TEXT_FAMS},
            "families_layered": {
                f: rng.uniform(-0.5, 1.0, (n_l, n_t)).tolist() for f in GEOMETRY_FAMS
            },
        }
    (root / "predictor_scores.json").write_text(json.dumps({"conditions": pconds}))
    print(f"[smoke] synthetic tree generated under {root}", flush=True)


def _workspace_sentinel_snapshot() -> dict:
    """(size, mtime_ns) per /workspace/logs/issue-2474-* file — the r1 Major-1 pin
    surface: a smoke run must leave the PRODUCTION sentinel glob byte-untouched."""
    ws = Path("/workspace/logs")
    if not ws.is_dir():
        return {}
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in ws.glob("issue-2474-*")}


def phase_smoke(args) -> None:
    root = Path(args.smoke_dir)
    ws_before = _workspace_sentinel_snapshot()
    _gen_smoke_tree(root)
    # Same dispatch shape as `all`: pilot + refit as BLAS=1 subprocess legs of THIS
    # entrypoint, scores + stats + upload in-process — against the synthetic root.
    ns = argparse.Namespace(**{**vars(args), "synthetic_root": str(root), "pilot_layer": 1})
    cfg = _cfg_from_args(ns)
    for ph in ("pilot", "refit"):
        _run_phase_subprocess(ph, ns)
    out1 = phase_scores(ns, cfg)
    phase_stats(ns, cfg)
    phase_upload(ns, cfg)
    for f in ("out/prefit_scores.json", "out/prefit_stats.json"):
        if not (root / f).is_file():
            raise RuntimeError(f"smoke FAIL: expected output {root / f} missing")

    # Resume-replay leg 1 (resume-matrix): a full re-run resumes EVERY setting at
    # SETTING grain and reproduces the conditions payload exactly.
    out2 = phase_scores(ns, cfg)
    if out2["resume_info"]["resumed_settings"] != list(cfg["settings"]):
        raise RuntimeError(
            "smoke FAIL: setting-grain resume replay resumed "
            f"{out2['resume_info']['resumed_settings']} != {list(cfg['settings'])}"
        )
    if out2["conditions"] != out1["conditions"]:
        raise RuntimeError("smoke FAIL: setting-grain resume replay changed the payload")
    # Resume-replay leg 2: drop ONE per-setting partial -> the setting recomputes
    # from its per-LAYER checkpoints (B3 restartability at layer grain).
    (cfg["out_dir"] / "scores_partial" / "em.json").unlink()
    out3 = phase_scores(ns, cfg)
    if out3["resume_info"]["resumed_layers"].get("em") != [0, 1]:
        raise RuntimeError(
            "smoke FAIL: layer-grain resume replay resumed_layers="
            f"{out3['resume_info']['resumed_layers']} (expected em: [0, 1])"
        )
    if out3["conditions"] != out1["conditions"]:
        raise RuntimeError("smoke FAIL: layer-grain resume replay changed the payload")

    smoke_sentinel = Path(cfg["data_root"]) / "logs" / "issue-2474-fit-smoke.done.json"
    if not smoke_sentinel.is_file():
        raise RuntimeError(f"smoke FAIL: smoke done sentinel missing at {smoke_sentinel}")
    ws_after = _workspace_sentinel_snapshot()
    if ws_after != ws_before:
        raise RuntimeError(
            "smoke FAIL: /workspace/logs/issue-2474-* CHANGED under a smoke run "
            f"(r1 Major 1 pin): before={ws_before} after={ws_after}"
        )
    print(f"[smoke] PASS — end-to-end outputs under {root / 'out'}", flush=True)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
_FIT_BLAS_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _run_phase_subprocess(phase: str, args) -> None:
    """Run pilot/refit as a child of the SAME entrypoint with the fit-phase BLAS env
    (1 thread per pool worker — the mapfit measurement convention), so an in-process
    scores/stats pass keeps full-width BLAS."""
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        phase,
        "--parent-sha",
        args.parent_sha,
        "--workers",
        str(args.workers),
        "--n-boot",
        str(args.n_boot),
        "--boot-seed",
        str(args.boot_seed),
        "--n-perm",
        str(args.n_perm),
        "--perm-seed",
        str(args.perm_seed),
        "--data-root",
        str(args.data_root),
        "--out-dir",
        str(args.out_dir),
        "--pilot-layer",
        str(args.pilot_layer),
    ]
    if args.tensors_out:
        argv += ["--tensors-out", str(args.tensors_out)]
    if args.synthetic_root:
        argv += ["--synthetic-root", str(args.synthetic_root)]
    if args.force:
        argv += ["--force"]
    env = {**os.environ, **{v: "1" for v in _FIT_BLAS_VARS}}
    print(f"[dispatch] subprocess phase={phase} (BLAS=1)", flush=True)
    subprocess.run(argv, check=True, env=env)


def _set_fit_blas_threads() -> None:
    assert "numpy" not in sys.modules, "BLAS env must be set before the first numpy import"
    for v in _FIT_BLAS_VARS:
        os.environ[v] = "1"


# Arm registry (string-constant keys — task.py check-smoke-arch-registry recomputes
# the per-arm marker rows from `sorted(PHASES)` via AST extraction of this literal).
PHASES = {
    "smoke": "P0: synthetic tiny end-to-end, dispatch parity with 'all' (no downloads)",
    "harvest-verify": "P-A: stage + schema-validate banked inputs -> harvest_verified.json",
    "pilot": "B2 pilot: 1-layer refit wall measurement + fence (BLAS=1 subprocess leg)",
    "refit": "B2: pinned-layer base-map refits (BLAS=1 subprocess leg)",
    "scores": "B3: batched per-layer geometry-family scores + pinned vhat tensors",
    "stats": "B4: bootstrap/permutation statistics -> prefit_stats.json",
    "upload": "B5: HF mirror (analysis/ + analysis_tensors/) -> verify -> done sentinel",
    "all": "harvest-verify -> pilot -> refit -> scores -> stats -> upload",
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        choices=sorted(PHASES),
        help="pipeline phase (plan §4 P-B; 'all' = harvest-verify→pilot→refit→scores→stats→upload)",
    )
    ap.add_argument(
        "--import-check", action="store_true", help="argcheck + call-arity bind, then exit 0"
    )
    ap.add_argument("--parent-sha", default=PARENT_SHA_DEFAULT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=20260822)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument(
        "--perm-seed", type=int, default=20260823, help="permutation seed (boot_seed + 1, plan §10)"
    )
    ap.add_argument(
        "--data-root",
        default=str(
            Path("/workspace/data/issue_2474")
            if Path("/workspace").is_dir()
            else REPO_ROOT / "data" / "issue_2474"
        ),
    )
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "eval_results" / "issue_2474" / "prefit"))
    ap.add_argument(
        "--tensors-out",
        default=None,
        help="per-draw npz + predicted tensors (default <data-root>/analysis_out)",
    )
    ap.add_argument(
        "--prep-output",
        default=None,
        help="round-2 prep_output.json for the UltraChat bank-content pin assert "
        "(default: local data/issue_2474 copy when it passes the pin, else staged "
        "from HF issue2474_prefit/prep_output.json — see _resolve_prep_output)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="ignore completion fingerprints + partials; recompute every phase",
    )
    ap.add_argument(
        "--upload-dry-run",
        action="store_true",
        help="phase upload: enumerate + print would-upload sets, NO Hub writes, "
        "NO production sentinel",
    )
    ap.add_argument("--smoke-dir", default="/tmp/issue2474_smoke")
    ap.add_argument("--synthetic-root", default=None, help="internal: smoke synthetic input tree")
    ap.add_argument("--pilot-layer", type=int, default=16)
    ap.add_argument(
        "--log-dir", default=None, help="done-sentinel dir (default /workspace/logs when present)"
    )
    return ap


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if not args.phase:
        raise SystemExit("--phase is required (or --import-check)")
    if args.phase in ("pilot", "refit"):
        _set_fit_blas_threads()
    _assert_parent_ancestor(args.parent_sha)

    if args.phase == "smoke":
        phase_smoke(args)
        return 0
    cfg = _cfg_from_args(args)
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    if args.phase == "harvest-verify":
        phase_harvest_verify(args, cfg)
    elif args.phase == "pilot":
        phase_pilot(args, cfg)
    elif args.phase == "refit":
        phase_refit(args, cfg)
    elif args.phase == "scores":
        phase_scores(args, cfg)
    elif args.phase == "stats":
        phase_stats(args, cfg)
    elif args.phase == "upload":
        phase_upload(args, cfg)
    elif args.phase == "all":
        phase_harvest_verify(args, cfg)
        for ph in ("pilot", "refit"):
            _run_phase_subprocess(ph, args)
        phase_scores(args, cfg)
        phase_stats(args, cfg)
        phase_upload(args, cfg)
    return 0


if __name__ == "__main__":
    # Heavy C extensions (torch/scipy) are loaded by most phases — exit explicitly
    # after flushing so a finalize-time teardown race can never rewrite the rc.
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
