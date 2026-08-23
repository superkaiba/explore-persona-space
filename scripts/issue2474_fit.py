"""Issue #2474 P-B — pre-fine-tuning predictor analysis phase driver (plan §4 P-B, v5).

Phases (``--phase``): smoke | harvest-verify | pilot | refit | scores | stats | all
(``all`` = harvest-verify → pilot → refit → scores → stats; the §10 P-B command).

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


def _write_done_sentinel(args, outputs: list[str]) -> None:
    """Pod-contract done sentinel (issue-2474-fit.done.json; plan §9 phase_outputs).

    Mirrors the sibling launcher's (issue2474_pod.sh) minimal shape; written only
    when a log dir resolves (pod-side /workspace/logs, or --log-dir).
    """
    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir is None:
        default = Path("/workspace/logs")
        log_dir = default if default.is_dir() else None
    if log_dir is None:
        logger.info("[sentinel] no log dir resolves on this host — sentinel skipped")
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        log_dir / "issue-2474-fit.done.json",
        {"phase": "done", "rc": 0, "utc": _utcnow(), "outputs": outputs},
    )
    logger.info("[sentinel] wrote %s", log_dir / "issue-2474-fit.done.json")


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


def _stage_maps_pinned(cfg: dict, args) -> None:
    """Stage the parent's pinned base-map components for the parity asserts."""
    if cfg["synthetic"]:
        return
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
        fp = json.loads(target.read_text())["fingerprint"]
        recon[rel] = fp
    for setting in cfg["settings"]:
        fp = recon[f"{prefix}/predictor_captures/base_{setting}/grid.pt.meta.json"]
        want = cfg["expected_grid_rows"][setting]
        if int(fp["n_rows"]) != want:
            raise RuntimeError(
                f"harvest-verify FAIL: grid base_{setting} n_rows {fp['n_rows']} != {want}"
            )
        cfp = recon[f"{prefix}/predictor_captures/base_{setting}/ceiling.pt.meta.json"]
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
        n_kept = int(tb["v_a"].shape[0])
        max_rows = cfg["ceiling_max_rows"][setting]
        drop_stats = tb["drop_stats"]
        n_slots = int(drop_stats["n_slots"])
        n_dropped = int(drop_stats["n_empty_after_retries"]) + int(drop_stats["n_capture_dropped"])
        kept_per_cell: dict[int, int] = {}
        for r in tb["row_meta"]:
            kept_per_cell[r["cell_idx"]] = kept_per_cell.get(r["cell_idx"], 0) + 1
        min_kept = min(kept_per_cell.values()) if kept_per_cell else 0
        if n_kept > max_rows:
            raise RuntimeError(
                f"harvest-verify FAIL: ceiling base_{setting} {n_kept} rows > {max_rows}"
            )
        if (
            n_dropped > cap.MAX_EMPTY_DROP_FRAC * n_slots
            or min_kept < cap.CEILING_MIN_KEPT_PER_CELL
        ):
            raise RuntimeError(
                f"harvest-verify FAIL: ceiling base_{setting} drop accounting exceeds the "
                f"capture's registered floors (dropped {n_dropped}/{n_slots} slots; "
                f"min kept/cell {min_kept} < {cap.CEILING_MIN_KEPT_PER_CELL})"
            )
        ceiling_stats[setting] = {
            "n_kept_rows": n_kept,
            "n_slots": n_slots,
            "n_dropped_total": n_dropped,
            "min_kept_per_cell": min_kept,
            "n_cells_seen": len(kept_per_cell),
        }
        print(
            f"[harvest] ceiling base_{setting}: {n_kept} rows kept, {n_dropped}/{n_slots} dropped",
            flush=True,
        )

    # Bank-content pin: round-2 prep_output.json .ultrachat.revision (plan §4 P-A step 3).
    prep_path = Path(args.prep_output)
    if not prep_path.is_file():
        raise RuntimeError(
            f"harvest-verify FAIL: prep_output.json not found at {prep_path}. Round-2's "
            "p1 writes it to <repo>/data/issue_2474/prep_output.json on the capture pod "
            "(not uploaded to HF — prep_data's upload leg is opt-in); fetch it from "
            "pod-2474 or pass --prep-output."
        )
    prep = json.loads(prep_path.read_text())
    got_rev = (prep.get("ultrachat") or {}).get("revision")
    if got_rev != ULTRACHAT_REV:
        raise RuntimeError(
            f"harvest-verify FAIL: prep_output .ultrachat.revision {got_rev!r} != pinned "
            f"{ULTRACHAT_REV!r} — the banks were drawn from a different UltraChat snapshot."
        )

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


def phase_pilot(args, cfg: dict) -> dict:
    import issue2379_mapfit as mf

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
    """Generating-params resume key for a per-setting scores block (never float hashes)."""
    parts = {}
    base = cfg["capture_dir"] / "predictor_captures"
    paths = [base / f"base_{setting}" / "grid.pt", base / f"base_{setting}" / "ceiling.pt"]
    paths += [base / f"base_mu_{c}" / "mu.pt" for c in cfg["conds"][setting]]
    for p in paths:
        st = p.stat()
        parts[str(p.name if p.parent.name.startswith("base_") else p)] = {
            "rel": f"{p.parent.name}/{p.name}",
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
        }
    return {"setting": setting, "parent_sha": args.parent_sha, "bundles": parts, "v": 1}


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
    conditions: dict[str, dict] = {}
    t0 = time.time()
    for si, setting in enumerate(cfg["settings"]):
        fp = _scores_fingerprint(cfg, setting, args)
        ppath = partial_dir / f"{setting}.json"
        if ppath.is_file():
            cached = json.loads(ppath.read_text())
            if cached.get("fingerprint") == fp:
                conditions.update(cached["conditions"])
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

        mu_by_cond = {}
        for cond in cfg["conds"][setting]:
            tb = _load_bundle(base / f"base_mu_{cond}" / "mu.pt", "mu")
            mu_by_cond[cond] = (
                np.asarray(tb["mu_train"], dtype=np.float64),
                np.asarray(tb["mu_a_train"], dtype=np.float64),
            )

        c_meta = ceil["row_meta"]
        c_va = ceil["v_a"]
        ceil_rows: dict[tuple[int, int], dict[int, int]] = {}
        for i, r in enumerate(c_meta):
            ceil_rows.setdefault((r["trigger_idx"], r["q_sim_idx"]), {})[r["rollout_idx"]] = i
        n_rollouts = 1 + max((max(d.keys()) for d in ceil_rows.values()), default=0)

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
            for c in cfg["conds"][setting]
        }
        cbr_sameq = np.full((n_l, n_t, n_rollouts), np.nan)
        cbr_trainref = {c: np.full((n_l, n_t, n_rollouts), np.nan) for c in cfg["conds"][setting]}

        predicted_dir = cfg["tensors_out"] / "predicted"
        pinned_save = {cfg["pinned_layer"][s] for s in cfg["settings"]}

        for ly in range(n_l):
            v_c = np.asarray(v_c_all[:, ly, :], dtype=np.float64)
            comp_b = mf.load_components(cfg["comp_dir"], mf.BASE_MAPSET, ly)
            v_hat = mf.predict_affine(comp_b, v_c)
            v_ib = v_c + comp_b["ib_bias"]

            def _centered(mat):
                out = np.array(mat, dtype=np.float64)
                for q in range(n_q):
                    rows = row_of[:, q]
                    out[rows] = out[rows] - out[rows].mean(axis=0, keepdims=True)
                return out

            v_c_c, v_hat_c, v_ib_c = _centered(v_c), _centered(v_hat), _centered(v_ib)

            # Ceiling rollout means per (t, q) + trigger-centered companion matrices.
            vbar = np.full((n_t, n_q, v_c.shape[1]), np.nan)
            for (t, q), rows in ceil_rows.items():
                vbar[t, q] = np.asarray(c_va[sorted(rows.values()), ly, :], dtype=np.float64).mean(
                    axis=0
                )
            with np.errstate(invalid="ignore"):
                vbar_c = vbar - np.nanmean(vbar, axis=0, keepdims=True)

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
                    predicted_dir / f"base_{setting}_L{ly:02d}_vhat.pt",
                )

            rows_p = row_of[p_idx]
            for t in range(n_t):
                rows_t = row_of[t]
                fams_shared["ctx_sameq"][ly, t] = mf._cos_pairwise(v_c[rows_t], v_c[rows_p]).mean()
                fams_shared["ans_sameq_mapB"][ly, t] = mf._cos_pairwise(
                    v_hat[rows_t], v_hat[rows_p]
                ).mean()
                fams_shared["identbias_sameq"][ly, t] = mf._cos_pairwise(
                    v_ib[rows_t], v_ib[rows_p]
                ).mean()
                fams_shared["ctx_sameq_centered"][ly, t] = mf._cos_pairwise(
                    v_c_c[rows_t], v_c_c[rows_p]
                ).mean()
                fams_shared["ans_sameq_mapB_centered"][ly, t] = mf._cos_pairwise(
                    v_hat_c[rows_t], v_hat_c[rows_p]
                ).mean()
                fams_shared["identbias_sameq_centered"][ly, t] = mf._cos_pairwise(
                    v_ib_c[rows_t], v_ib_c[rows_p]
                ).mean()
                both = [
                    q
                    for q in range(n_q)
                    if np.isfinite(vbar[t, q, 0]) and np.isfinite(vbar[p_idx, q, 0])
                ]
                if both:
                    fams_shared["ceiling_sameq"][ly, t] = mf._cos_pairwise(
                        vbar[t, both], vbar[p_idx, both]
                    ).mean()
                    fams_shared["ceiling_sameq_centered"][ly, t] = mf._cos_pairwise(
                        vbar_c[t, both], vbar_c[p_idx, both]
                    ).mean()
                have_t = [q for q in range(n_q) if np.isfinite(vbar[t, q, 0])]
                for ri in range(n_rollouts):
                    sq_vals = []
                    for q in range(n_q):
                        rows = ceil_rows.get((t, q), {})
                        if ri in rows and np.isfinite(vbar[p_idx, q, 0]):
                            va = np.asarray(c_va[rows[ri], ly, :], dtype=np.float64)
                            sq_vals.append(
                                float(mf._cos_pairwise(va[None, :], vbar[p_idx, q][None, :])[0])
                            )
                    if sq_vals:
                        cbr_sameq[ly, t, ri] = float(np.mean(sq_vals))
                for cond in cfg["conds"][setting]:
                    mu_tr, mu_a = mu_by_cond[cond]
                    fc = fams_cond[cond]
                    fc["ctx_trainref"][ly, t] = mf._cos_rows_vec(v_c[rows_t], mu_tr[ly]).mean()
                    fc["ans_trainref_mapB"][ly, t] = mf._cos_rows_vec(
                        v_hat[rows_t], mu_a[ly]
                    ).mean()
                    fc["identbias_trainref"][ly, t] = mf._cos_rows_vec(
                        v_ib[rows_t], mu_a[ly]
                    ).mean()
                    fc["ctx_trainref_centered"][ly, t] = mf._cos_rows_vec(
                        v_c_c[rows_t], mu_tr[ly]
                    ).mean()
                    fc["ans_trainref_mapB_centered"][ly, t] = mf._cos_rows_vec(
                        v_hat_c[rows_t], mu_a[ly]
                    ).mean()
                    fc["identbias_trainref_centered"][ly, t] = mf._cos_rows_vec(
                        v_ib_c[rows_t], mu_a[ly]
                    ).mean()
                    if have_t:
                        fc["ceiling_trainref"][ly, t] = mf._cos_rows_vec(
                            vbar[t, have_t], mu_a[ly]
                        ).mean()
                        fc["ceiling_trainref_centered"][ly, t] = mf._cos_rows_vec(
                            vbar_c[t, have_t], mu_a[ly]
                        ).mean()
                    for ri in range(n_rollouts):
                        tr_vals = []
                        for q in range(n_q):
                            rows = ceil_rows.get((t, q), {})
                            if ri in rows:
                                va = np.asarray(c_va[rows[ri], ly, :], dtype=np.float64)
                                tr_vals.append(float(mf._cos_rows_vec(va[None, :], mu_a[ly])[0]))
                        if tr_vals:
                            cbr_trainref[cond][ly, t, ri] = float(np.mean(tr_vals))
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
        "conditions": conditions,
    }
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
                if abs(rec[key] - want[key]) > 1e-6:
                    raise RuntimeError(
                        f"round-1 recompute FAIL: {setting}/{variant}/{key} recomputed "
                        f"{rec[key]!r} vs banked {want[key]!r} (>1e-6) — provenance drift "
                        "between the banked gate and the pinned inputs."
                    )
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
    scores = json.loads((cfg["out_dir"] / "prefit_scores.json").read_text())
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
            correlate_rows = [prop[sel]]
            correlate_rows += [text_vecs[f][sel] for f in TEXT_FAMS]
            for c in conds:
                correlate_rows += [lvl[c][sel], chg[c][sel]]
                correlate_rows += [postft[c][f][sel] for f in postft[c]]
                for f in all_fams:
                    correlate_rows.append(fam_mats[c][f][:, sel].reshape(-1))
            # arm rows are (n_l * n_sel,) flattened — split back per layer:
            correlates = [r for r in correlate_rows if r.ndim == 1 and r.size == n_sel]
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
                    fam_entry["pooled"][dv] = {
                        "rho_by_layer": point_curve,
                        "ci95_by_layer": [_ci95(pooled_draws[ly]) for ly in range(n_l)],
                        "pinned": {
                            "layer": pin,
                            "rho": point_curve[pin],
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
                        x = (
                            vals[sel]
                            if vals is not None
                            else postft[c].get(k, np.full(len(labels), np.nan))[sel]
                        )
                        entry["per_condition"][c] = {
                            "level_rho": _point_corr(x, y, spearman=True),
                            "level_ci95": _ci95(arr[ci, 0, valid]),
                        }
                    entry["pooled_level_ci95"] = _ci95(arr[:, 0, :].mean(axis=0)[valid])
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
                permb[f] = {
                    "observed_pooled_max_over_layers": float(np.nanmax(obs_curve)),
                    "null_max_p50": float(np.percentile(null_max, 50)),
                    "null_max_p95": float(np.percentile(null_max, 95)),
                    "null_max_p975": float(np.percentile(null_max, 97.5)),
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
        **stats_out,
    }
    _atomic_write_json(cfg["out_dir"] / "prefit_stats.json", out)
    print(f"[stats] wrote {cfg['out_dir'] / 'prefit_stats.json'}", flush=True)
    if cfg["synthetic"] and not smoke_saw_degenerate:
        raise RuntimeError(
            "smoke FAIL: the constructed degenerate propensity vector produced ZERO "
            "invalid bootstrap draws — the valid-draw mask machinery is not engaging"
        )
    _write_done_sentinel(
        args,
        [str(cfg["out_dir"] / "prefit_scores.json"), str(cfg["out_dir"] / "prefit_stats.json")],
    )
    return out


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


def phase_smoke(args) -> None:
    root = Path(args.smoke_dir)
    _gen_smoke_tree(root)
    # Same dispatch shape as `all`: pilot + refit as BLAS=1 subprocess legs of THIS
    # entrypoint, scores + stats in-process — against the synthetic root.
    ns = argparse.Namespace(**{**vars(args), "synthetic_root": str(root), "pilot_layer": 1})
    cfg = _cfg_from_args(ns)
    for ph in ("pilot", "refit"):
        _run_phase_subprocess(ph, ns)
    phase_scores(ns, cfg)
    phase_stats(ns, cfg)
    for f in ("out/prefit_scores.json", "out/prefit_stats.json"):
        if not (root / f).is_file():
            raise RuntimeError(f"smoke FAIL: expected output {root / f} missing")
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
    env = {**os.environ, **{v: "1" for v in _FIT_BLAS_VARS}}
    print(f"[dispatch] subprocess phase={phase} (BLAS=1)", flush=True)
    subprocess.run(argv, check=True, env=env)


def _set_fit_blas_threads() -> None:
    assert "numpy" not in sys.modules, "BLAS env must be set before the first numpy import"
    for v in _FIT_BLAS_VARS:
        os.environ[v] = "1"


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        choices=["smoke", "harvest-verify", "pilot", "refit", "scores", "stats", "all"],
        help="pipeline phase (plan §4 P-B; 'all' = harvest-verify→pilot→refit→scores→stats)",
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
        default=str(REPO_ROOT / "data" / "issue_2474" / "prep_output.json"),
        help="round-2 prep_output.json for the UltraChat bank-content pin assert",
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
    elif args.phase == "all":
        phase_harvest_verify(args, cfg)
        for ph in ("pilot", "refit"):
            _run_phase_subprocess(ph, args)
        phase_scores(args, cfg)
        phase_stats(args, cfg)
    return 0


if __name__ == "__main__":
    # Heavy C extensions (torch/scipy) are loaded by most phases — exit explicitly
    # after flushing so a finalize-time teardown race can never rewrite the rc.
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
