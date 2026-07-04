#!/usr/bin/env python
# ruff: noqa: E501, RUF002, RUF003
# E501: the plan §1 MF-RC block (MF_RC_RULE) is carried VERBATIM — its clause
# lines exceed 100 chars by construction (do NOT re-wrap; edit only in lockstep
# with the plan §1 block). MF-R is imported verbatim from
# issue825_realuser_summarize. RUF002/RUF003: docstrings quote the plan's math
# notation (Δ, R², ×, −) verbatim.
"""Issue #825 ``role-map-comparison`` — paired within-conversation role contrast.

The estimand is Δ = R²(assistant map) − R²(user map) per (provenance × model ×
format × layer): both roles' targets live in the SAME conversations (assistant:
slots[:, 0] → profiles[:, 1]; user: slots[:, 1] → profiles[:, 2]), so ONE joint
keep mask + ONE fold assignment make per-fold and per-conversation deltas
paired. Pure re-fit over persisted turnstore tensors — no generation, no
extraction, no model loading, no judge (plan v14, amendment to v11).

Modes (the dispatch wrapper scripts/issue825_rolecontrast_dispatch.sh phases):
  fabricate-smoke  tiny synthetic .pt bundles + allowlist (EPS_SMOKE=1 stage)
  fit              the 12-pair runner (checkpoint-per-pair, --resume)
  summarize        headline_metrics.json (plan §1 labels + the pre-registered
                   normal-approximation corrected inverted read Φ(Δ_obs/SE_boot))
  gates            binding POST-upload gates (plan §7) — FAILURE sentinel + exit 1
  fail-sentinel    upload-then-exit FAILURE sentinel for a crashed fit/summarize
  success-sentinel schema-enveloped SUCCESS sentinel (refuses without all_pass)

Statistics (plan §4.2, binding): the paired bootstrap is BATCHED as one
(n_boot, n) row-count matrix W per pair with PER-DRAW OWN-MEAN re-centering —
SS_res(w) = w @ ||Y−P||²_row; SS_tot(w) = w @ ||Y||²_row − ||w @ Y||²/sum(w)
(the (n_boot, n) @ (n, d) weighted-mean GEMM), fp64 — matching _pooled_r2's
convention exactly, so the seeded serial oracle (run_cross_role_cell's
per-draw form) agrees to 1e-8. A fixed-center subset-sum CANNOT pass the gate.
The LABEL-DRIVING point estimate is the full-sample pooled r2_obs_a − r2_obs_u.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_fit_cells as fit_cells  # noqa: E402
from issue825_realuser_gates import (  # noqa: E402
    GateFailure,
    _l19,
    eps_smoke,
    write_sentinel,
)
from issue825_realuser_summarize import INTERPRETATION_RULE as MF_R_RULE  # noqa: E402

FOLLOWUP_LABEL = "role-map-comparison"
HF_RC_PREFIX = "issue825_role_map_comparison"
HEADLINE_LAYERS = (19, 26)
REPRO_TOL = 0.01  # plan §4.2 reproduction gate ±0.01 at L19
ROW_ALIGN_FLOOR = 0.95  # plan §7 gate (2)
N_PRIMARY_READS = 24  # 12 pairs × 2 headline layers
BONF_ALPHA = 0.05 / N_PRIMARY_READS
EQUIV_TOL = 1e-8
MODELS = ("instruct", "pretrained")
FORMATS = ("chat", "naturalistic")
PROVENANCES = ("haiku", "real", "onpolicy")
ROLES = ("assistant", "user")
# Cell registry (grounded on fit_cells._ROLE_TO_INDICES, fit_cells.py:713-716):
# assistant: slot 0 (before a1) -> turn-1 profile; user: slot 1 (before u2) -> turn-2 profile.
ROLE_SLOT = {"assistant": 0, "user": 1}
ROLE_TARGET_TURN = {"assistant": 1, "user": 2}
PROV_MIN_TURNS = {"haiku": 4, "real": 3, "onpolicy": 3}  # target indices 1 and 2 valid in both

# Binding interpretation rule MF-RC — plan v14 §1 block carried VERBATIM (the
# analyzer and clean-result MUST carry MF-R AND MF-RC; MF-R rides in verbatim
# via the issue825_realuser_summarize import above).
MF_RC_RULE = """**Binding interpretation rule (MF-RC — new; MF-R carried verbatim alongside; the analyzer and clean-result MUST carry both):**
1. Within a conversation, the assistant map and the user map differ in MORE than "role": context span (assistant slot conditions on u1; user slot conditions on u1+a1), target author, target turn position (turn 2 vs turn 3), and target length/entropy statistics all change together. The paired delta is a **role-slot-bundle contrast**, not an isolated author-role effect; no claim may attribute the gap to "predicting humans is harder" as a mechanism.
2. R² is variance-normalized PER ROLE (each role's SS_tot is its own target variance), so the delta compares normalized predictability, not absolute error; the per-example cosine delta is the scale-free geometric companion and the paired NLL delta the token-level companion — report all three, attribute via none.
3. Cross-provenance comparisons of the role gap remain DESCRIPTIVE provenance-bundle claims (MF-R: conversation sample, a1 authorship, u2 authorship co-vary across provenances); pairing is within-provenance only.
4. The on-policy provenance pair is read on the user-cell allowlist rows (n = 1914/1722/1999/1738) — a filtered subpopulation; carried as scope.
5. Single seed throughout (fit seed 0) — carried as scope, as in all prior rounds."""


def pair_registry() -> list[dict]:
    """The ONE 12-pair registry every phase enumerates (smoke == production)."""
    return [
        {"pair_id": f"pair_{prov}_{m}_{f}", "provenance": prov, "model": m, "format": f}
        for prov in PROVENANCES
        for m in MODELS
        for f in FORMATS
    ]


def cell_id(model: str, role: str, fmt: str) -> str:
    return f"M_{model}_{role}_{fmt}"


def gated_roles(prov: str) -> tuple[str, ...]:
    """Reproduction-gated roles per provenance (plan §4.2): parent/real cells all
    have committed same-row-set values; onpolicy only the user cells do (the
    assistant-on-allowlist cells are NEW fits, gate-exempt, sandwich-reported)."""
    return ROLES if prov in ("haiku", "real") else ("user",)


def _metadata(seed: int, n: int) -> dict:
    md = fit_cells._metadata(seed, n)
    md["script"] = "scripts/issue825_role_contrast.py"
    md["followup_label"] = FOLLOWUP_LABEL
    return md


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


class BundleSchemaError(Exception):
    """A turnstore bundle failed its structural asserts (plan §7 gate 1)."""

    status_hint = "bundle_schema_mismatch"


def load_pair_bundle(
    turnstore_dir: Path, model: str, fmt: str, prov: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load ONE (model, format) m-track bundle WITHOUT perpos (keys= param) and
    assert the schema (slots/profiles/nll shapes, layer axis, conv parity).

    Returns (slots, profiles, nll, conv_ids). Raises BundleSchemaError on any
    structural miss — the fit loop defers it to fit_failures.json (MF-C) and
    the post-upload gate fails with status bundle_schema_mismatch.
    """
    bundle = fit_cells._load_bundle_pt(
        turnstore_dir, model, fmt, "m", keys=("slots", "profiles", "nll")
    )
    if bundle is None:
        raise BundleSchemaError(f"no m-track .pt shards for {model}_{fmt} under {turnstore_dir}")
    arrays = bundle["arrays"]
    for k in ("slots", "profiles", "nll"):
        if k not in arrays:
            raise BundleSchemaError(
                f"{model}_{fmt} ({prov}): key {k!r} missing from shards (have {sorted(arrays)})"
            )
    slots = np.asarray(arrays["slots"], dtype=np.float32)
    profiles = np.asarray(arrays["profiles"], dtype=np.float32)
    nll = np.asarray(arrays["nll"], dtype=np.float32)
    if slots.ndim != 4 or slots.shape[1] < 2:
        raise BundleSchemaError(
            f"{model}_{fmt} ({prov}): slots shape {slots.shape} (need (N, >=2, L, D))"
        )
    if profiles.ndim != 4 or profiles.shape[1] < PROV_MIN_TURNS[prov]:
        raise BundleSchemaError(
            f"{model}_{fmt} ({prov}): profiles shape {profiles.shape} "
            f"(need n_turns >= {PROV_MIN_TURNS[prov]})"
        )
    if (
        slots.shape[2] != fit_cells.EXPECTED_LAYERS
        or profiles.shape[2] != fit_cells.EXPECTED_LAYERS
    ):
        raise BundleSchemaError(
            f"{model}_{fmt} ({prov}): layer axis {slots.shape[2]}/{profiles.shape[2]} != {fit_cells.EXPECTED_LAYERS}"
        )
    if nll.ndim != 2 or nll.shape[0] != slots.shape[0] or nll.shape[1] != profiles.shape[1]:
        raise BundleSchemaError(
            f"{model}_{fmt} ({prov}): nll shape {nll.shape} vs N={slots.shape[0]}, n_turns={profiles.shape[1]}"
        )
    conv_ids = np.asarray(bundle["sidecar"]["conv_ids"])
    if len(conv_ids) != slots.shape[0]:
        raise BundleSchemaError(
            f"{model}_{fmt} ({prov}): {len(conv_ids)} conv_ids vs {slots.shape[0]} rows"
        )
    return slots, profiles, nll, conv_ids


def assemble_pair_rows(
    slots: np.ndarray,
    profiles: np.ndarray,
    nll: np.ndarray,
    conv_ids: np.ndarray,
    allowlist: list | None = None,
) -> dict:
    """Both roles from the SAME bundle under ONE JOINT keep mask.

    The run_cross_role_cell pattern (fit_cells.py:869-877): independent masks
    silently misalign rows across conversations, corrupting the paired delta.
    On-policy pairs additionally intersect the user cell's allowlist — applied
    HERE, BEFORE fold assignment (plan §4.2 step 3; conv_ids compared as str on
    both sides, the _apply_row_allowlist convention). Asserts conv == row
    uniqueness on the final row set (the conversation resample IS a row
    resample — plan §12.3).
    """
    X_a = slots[:, ROLE_SLOT["assistant"], :, :]
    Y_a = profiles[:, ROLE_TARGET_TURN["assistant"], :, :]
    X_u = slots[:, ROLE_SLOT["user"], :, :]
    Y_u = profiles[:, ROLE_TARGET_TURN["user"], :, :]
    keep = ~(
        np.isnan(X_a).any(axis=(1, 2))
        | np.isnan(Y_a).any(axis=(1, 2))
        | np.isnan(X_u).any(axis=(1, 2))
        | np.isnan(Y_u).any(axis=(1, 2))
    )
    allow_info: dict = {"applied": allowlist is not None}
    if allowlist is not None:
        ids = np.asarray([str(c) for c in conv_ids])
        wanted = {str(c) for c in allowlist}
        in_allow = np.isin(ids, np.asarray(sorted(wanted)))
        allow_info.update(
            n_allowlist=len(wanted),
            n_allow_in_bundle=int(in_allow.sum()),
            n_allow_after_joint_keep=int((keep & in_allow).sum()),
        )
        keep = keep & in_allow
    out = {
        "X_a": X_a[keep],
        "Y_a": Y_a[keep],
        "X_u": X_u[keep],
        "Y_u": Y_u[keep],
        "nll_a": np.asarray(nll[:, ROLE_TARGET_TURN["assistant"]], dtype=np.float64)[keep],
        "nll_u": np.asarray(nll[:, ROLE_TARGET_TURN["user"]], dtype=np.float64)[keep],
        "conv_ids": np.asarray(conv_ids)[keep],
        "n_joint": int(keep.sum()),
        "n_bundle": len(conv_ids),
        "allowlist": allow_info,
    }
    cid = out["conv_ids"]
    assert len(np.unique(cid)) == len(cid), (
        f"conv==row violated: {len(cid)} rows but {len(np.unique(cid))} unique conv_ids"
    )
    return out


# ---------------------------------------------------------------------------
# Paired bootstrap: batched (production) + serial oracle (equivalence gate)
# ---------------------------------------------------------------------------


def draw_index_matrix(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) row-resample indices; conv == row is asserted upstream, so
    the conversation-level resample IS this row resample. Shared across BOTH
    roles and both headline layers of a pair (the pairing; mirrors
    run_cross_role_cell's one-sample-per-draw-across-layers form)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def counts_from_indices(idx_matrix: np.ndarray, n: int) -> np.ndarray:
    """(n_boot, n) fp64 row-count matrix W from a draw index matrix."""
    idx_matrix = np.asarray(idx_matrix)
    n_boot = idx_matrix.shape[0]
    w = np.zeros((n_boot, n), dtype=np.float64)
    np.add.at(w, (np.repeat(np.arange(n_boot), idx_matrix.shape[1]), idx_matrix.ravel()), 1.0)
    return w


def weighted_r2_draws(preds: np.ndarray, true: np.ndarray, w: np.ndarray) -> np.ndarray:
    """R²(w) per draw, fp64, PER-DRAW OWN-MEAN re-centered (the REGISTERED form).

    SS_res(w) = w @ ||Y−P||²_row
    SS_tot(w) = w @ ||Y||²_row − ||w @ Y||² / sum(w)   # (n_boot,n)@(n,d) GEMM
    Matches _pooled_r2's own-mean convention exactly (a fixed-center subset-sum
    CANNOT pass the serial-oracle gate — binding critic Must-Fix).
    """
    y64 = np.asarray(true, dtype=np.float64)
    p64 = np.asarray(preds, dtype=np.float64)
    resid = y64 - p64
    r2_row = np.einsum("nd,nd->n", resid, resid)
    y2_row = np.einsum("nd,nd->n", y64, y64)
    wsum = w.sum(axis=1)
    ss_res = w @ r2_row
    wy = w @ y64
    ss_tot = w @ y2_row - np.einsum("bd,bd->b", wy, wy) / wsum
    return 1.0 - ss_res / np.where(ss_tot < 1e-12, np.nan, ss_tot)


def paired_bootstrap_batched(
    preds_a: np.ndarray, y_a: np.ndarray, preds_u: np.ndarray, y_u: np.ndarray, w: np.ndarray
) -> dict:
    """The PRODUCTION paired-bootstrap path: per-draw R² per role + Δ(w)."""
    r2_a = weighted_r2_draws(preds_a, y_a, w)
    r2_u = weighted_r2_draws(preds_u, y_u, w)
    return {"assistant": r2_a, "user": r2_u, "delta": r2_a - r2_u}


def paired_bootstrap_serial_reference(
    preds_a: np.ndarray,
    y_a: np.ndarray,
    preds_u: np.ndarray,
    y_u: np.ndarray,
    idx_matrix: np.ndarray,
) -> dict:
    """Seeded serial ORACLE — run_cross_role_cell's per-draw form
    (fit_cells.py:892-899): gather the resampled rows per draw and call
    _pooled_r2 per role (own-mean re-centering on each resample). Used ONLY by
    the equivalence gate; production dispatches paired_bootstrap_batched."""
    r2_a, r2_u = [], []
    for row in np.asarray(idx_matrix):
        r2_a.append(fit_cells._pooled_r2(preds_a[row], y_a[row]))
        r2_u.append(fit_cells._pooled_r2(preds_u[row], y_u[row]))
    r2_a = np.asarray(r2_a, dtype=np.float64)
    r2_u = np.asarray(r2_u, dtype=np.float64)
    return {"assistant": r2_a, "user": r2_u, "delta": r2_a - r2_u}


def weighted_mean_draws(values: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-draw resample MEAN of a per-row vector via the same W (paired)."""
    v = np.asarray(values, dtype=np.float64)
    return (w @ v) / w.sum(axis=1)


def _ci(dist: np.ndarray) -> dict:
    d = np.asarray(dist, dtype=np.float64)
    return {
        "ci_lo": float(np.nanquantile(d, 0.025)),
        "ci_hi": float(np.nanquantile(d, 0.975)),
        "se_boot": float(np.nanstd(d, ddof=1)),
        "n_draws": len(d),
    }


def per_fold_pooled_r2(
    preds: np.ndarray, y_layer: np.ndarray, folds: np.ndarray, n_folds: int
) -> list[float]:
    """Per-fold held-out pooled R² (own-mean per fold) from persisted preds."""
    vals = []
    for k in range(n_folds):
        te = folds == k
        vals.append(
            float("nan") if te.sum() == 0 else float(fit_cells._pooled_r2(preds[te], y_layer[te]))
        )
    return vals


# ---------------------------------------------------------------------------
# Reproduction-gate fallback + MLP paired secondary
# ---------------------------------------------------------------------------


def own_rowset_gate_read(
    slots: np.ndarray,
    profiles: np.ndarray,
    conv_ids: np.ndarray,
    role: str,
    allowlist: list | None,
    *,
    n_folds: int,
    seed: int,
) -> dict:
    """L19-only refit on the role's OWN committed row set (plan §4.2 fallback).

    Fires ONLY when the pair's joint-keep n differs from the committed n
    (expected no-op in production). Mirrors run_cell's row handling exactly:
    the role's own NaN keep, THEN the strict allowlist (all ids must match —
    fit_cells._apply_row_allowlist convention). The 1-layer slice X[:, [19]]
    yields the identical L19 value as the full sweep (GCV is per-layer).
    """
    x_full = slots[:, ROLE_SLOT[role], :, :]
    y_full = profiles[:, ROLE_TARGET_TURN[role], :, :]
    keep = ~(np.isnan(x_full).any(axis=(1, 2)) | np.isnan(y_full).any(axis=(1, 2)))
    x, y, ids = x_full[keep], y_full[keep], np.asarray(conv_ids)[keep]
    if allowlist is not None:
        sids = np.asarray([str(c) for c in ids])
        wanted = {str(c) for c in allowlist}
        m = np.isin(sids, np.asarray(sorted(wanted)))
        assert int(m.sum()) == len(wanted), (
            f"gate-read allowlist drift: {len(wanted)} allowlisted ids, {int(m.sum())} matched"
        )
        x, y, ids = x[m], y[m], ids[m]
    li = 19
    sweep = fit_cells.heldout_r2_sweep(
        x[:, [li], :],
        y[:, [li], :],
        ids,
        n_folds=n_folds,
        seed=seed,
        null_draws=0,
        collect_cosines=False,
    )
    return {
        "r2_l19_own_rowset": float(sweep["r2_obs"][0]),
        "n_own_rowset": len(ids),
        "note": "L19-only refit on the role's OWN committed row set (joint n != committed n)",
    }


def run_mlp_paired(
    rows: dict, folds: np.ndarray, *, layers: list[int], n_folds: int, budget_s: int
) -> dict:
    """Obs-only MLP secondary (fit_h.mlp_fit_predict, PCA-64) on the SHARED folds.

    Per role at the headline layers; budget_s bounds each (pair, role) — budget
    hits are RECORDED per fold (never silently dropped; plan §4.2 step 11).
    """
    from explore_persona_space.experiments.issue_779.fit_h import mlp_fit_predict

    out: dict[str, dict] = {}
    for role in ROLES:
        x, y = rows[f"X_{role[0]}"], rows[f"Y_{role[0]}"]
        started = time.monotonic()
        role_out: dict[str, dict] = {}
        for li in layers:
            xl, yl = x[:, li, :], y[:, li, :]
            fold_r2: list[float] = []
            hits: list[int] = []
            ss_res = ss_tot = 0.0
            for k in range(n_folds):
                if time.monotonic() - started > budget_s:
                    hits.append(int(k))
                    fold_r2.append(float("nan"))
                    continue
                te = folds == k
                tr = ~te
                if te.sum() == 0 or tr.sum() < 3:
                    fold_r2.append(float("nan"))
                    continue
                pred = mlp_fit_predict(xl[tr], yl[tr], xl[te])
                true = yl[te].astype(np.float64)
                mu = true.mean(0)
                f_res = float(np.sum((true - pred) ** 2))
                f_tot = float(np.sum((true - mu) ** 2))
                ss_res += f_res
                ss_tot += f_tot
                fold_r2.append((1.0 - f_res / f_tot) if f_tot > 1e-12 else float("nan"))
            if hits:
                print(
                    f"[role_contrast] MLP budget ({budget_s}s) hit: role={role} "
                    f"layer={li} folds={hits} — recorded, primary unaffected",
                    file=sys.stderr,
                )
            role_out[str(li)] = {
                "r2_obs": (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
                "r2_obs_folds": fold_r2,
                "budget_hit_folds": hits,
            }
        out[role] = role_out
    return out


def _record_deferred_failure(out_dir: Path, unit: str, exc: BaseException) -> None:
    """Fail-loud-deferred (MF-C): print the traceback, persist to
    fit_failures.json (with the exception's status_hint for gate routing), keep
    going. Contract (round-2 fix, deferred-failures-bypass-gates): the deferred
    pair's terminal JSON is left unwritten, summarize TOLERATES that (minimal
    headline noting the deferred set), uploads still run, and the wrapper's
    POST-upload gates HALT via check_deferred with the REGISTERED status
    (``bundle_schema_mismatch`` for a BundleSchemaError-classed failure, else
    ``fit_deferred_failure``) — never ``summarize_error``."""
    import traceback

    traceback.print_exc()
    print(
        f"[role_contrast] DEFER-FAIL {unit}: {type(exc).__name__}: {exc} — recorded to "
        "fit_failures.json; the wrapper's post-upload gates HALT (plan MF-C)",
        file=sys.stderr,
    )
    path = out_dir / "fit_failures.json"
    failures = json.loads(path.read_text()) if path.exists() else []
    failures.append(
        {
            "cell_id": unit,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "status_hint": getattr(exc, "status_hint", None),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(failures, indent=2) + "\n")


def _clear_deferred_failure(out_dir: Path, unit: str) -> None:
    """Drop stale fit_failures.json records for a unit that has now completed
    successfully — a ``--resume`` re-run after a fix must not HALT gate 1 on a
    superseded record (reconciler v6 'observed but not raised')."""
    path = out_dir / "fit_failures.json"
    if not path.exists():
        return
    failures = json.loads(path.read_text())
    kept = [e for e in failures if e.get("cell_id") != unit]
    if kept == failures:
        return
    if kept:
        path.write_text(json.dumps(kept, indent=2) + "\n")
    else:
        path.unlink()
    print(f"[role_contrast] cleared {len(failures) - len(kept)} stale deferred record(s): {unit}")


def regime_key(args, allowlist: list | None, equivalence_gate: bool) -> dict:
    """Every output-affecting knob (resume must pin them all — #722 r3 lesson)."""
    allow_sha = None
    if allowlist is not None:
        allow_sha = hashlib.sha256(
            json.dumps(sorted(str(c) for c in allowlist)).encode()
        ).hexdigest()[:16]
    return {
        "folds": int(args.folds),
        "null_draws": int(args.null_draws),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "headline_layers": list(HEADLINE_LAYERS),
        "mlp_budget_s": int(args.mlp_budget_s),
        "allowlist_sha": allow_sha,
        "equivalence_gate": bool(equivalence_gate),
    }


def _update_preds_manifest(out_dir: Path, fname: str, sha: str, shapes: dict) -> None:
    """Incremental manifest update (checkpoint-per-pair): names + shapes + sha256."""
    path = out_dir / "preds_manifest.json"
    manifest = _load_json(path) or {"files": {}, "dtype": "float16"}
    manifest["files"][fname] = {"sha256": sha, "arrays": shapes}
    manifest["metadata"] = _metadata(0, len(manifest["files"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# The pair runner
# ---------------------------------------------------------------------------


def _pair_allowlist(pair: dict, allow_map: dict) -> list | None:
    if pair["provenance"] != "onpolicy":
        return None
    key = cell_id(pair["model"], "user", pair["format"])
    allowlist = allow_map.get(key)
    assert allowlist, f"row_allowlists.json missing key {key}"
    return allowlist


def _turnstore_dir(args, prov: str) -> Path:
    return {"haiku": args.haiku_dir, "real": args.real_dir, "onpolicy": args.onpolicy_dir}[prov]


def _committed_dir(args, prov: str) -> Path:
    return {
        "haiku": args.committed_haiku,
        "real": args.committed_real,
        "onpolicy": args.committed_onpolicy,
    }[prov]


def run_pair(pair: dict, args, allow_map: dict, *, equivalence_gate: bool = False) -> dict:
    """One (provenance, model, format) pair end-to-end (plan §4.2 steps 1-12)."""
    prov, model, fmt = pair["provenance"], pair["model"], pair["format"]
    pair_id = pair["pair_id"]
    prov_out = args.out_dir / prov
    prov_out.mkdir(parents=True, exist_ok=True)
    allowlist = _pair_allowlist(pair, allow_map)
    committed_dir = _committed_dir(args, prov)

    slots, profiles, nll, conv_ids_all = load_pair_bundle(
        _turnstore_dir(args, prov), model, fmt, prov
    )
    rows = assemble_pair_rows(slots, profiles, nll, conv_ids_all, allowlist)
    n = rows["n_joint"]
    print(f"[role_contrast] {pair_id}: n_joint={n} (bundle {rows['n_bundle']})")

    # Full inherited battery per role (obs + null draws, fold-cached Gram eigh,
    # preds at frozen layers) on the SHARED conv-id set => shared folds.
    sweeps = {
        role: fit_cells.heldout_r2_sweep(
            rows[f"X_{role[0]}"],
            rows[f"Y_{role[0]}"],
            rows["conv_ids"],
            n_folds=args.folds,
            seed=args.seed,
            null_draws=args.null_draws,
            collect_cosines=True,
        )
        for role in ROLES
    }
    folds = sweeps["assistant"]["folds"]
    assert np.array_equal(folds, sweeps["user"]["folds"]), "fold vectors differ across roles"

    frozen = [li for li in fit_cells.FROZEN_LAYERS if li < rows["X_a"].shape[1]]
    headline = [li for li in HEADLINE_LAYERS if li in frozen]

    # Committed references + reproduction-gate fallback reads (plan §4.2).
    committed_ref: dict[str, dict | None] = {}
    gate_reads: dict[str, dict] = {}
    for role in ROLES:
        cid = cell_id(model, role, fmt)
        payload = _load_json(committed_dir / f"cells_{cid}.json")
        if payload is None:
            committed_ref[role] = None
            continue
        committed_ref[role] = {
            "path": str(committed_dir / f"cells_{cid}.json"),
            "n": (payload.get("metadata") or {}).get("n"),
            "r2_l19": _l19(payload),
            "gated": role in gated_roles(prov),
        }
        if role in gated_roles(prov) and committed_ref[role]["n"] not in (None, n):
            print(
                f"[role_contrast] {pair_id}/{role}: joint n={n} != committed "
                f"n={committed_ref[role]['n']} — own-row-set gate refit (expected no-op)"
            )
            gate_reads[role] = own_rowset_gate_read(
                slots,
                profiles,
                conv_ids_all,
                role,
                allowlist if (prov == "onpolicy" and role == "user") else None,
                n_folds=args.folds,
                seed=args.seed,
            )

    # Per-fold paired ridge deltas at every frozen layer (plan §4.2 step 6).
    per_fold: dict[str, dict] = {}
    for li in frozen:
        fa = per_fold_pooled_r2(
            sweeps["assistant"]["preds_frozen"][li], rows["Y_a"][:, li, :], folds, args.folds
        )
        fu = per_fold_pooled_r2(
            sweeps["user"]["preds_frozen"][li], rows["Y_u"][:, li, :], folds, args.folds
        )
        delta = [a - u for a, u in zip(fa, fu, strict=True)]
        d = np.asarray(delta, dtype=np.float64)
        n_fin = int(np.isfinite(d).sum())
        per_fold[str(li)] = {
            "assistant_folds": fa,
            "user_folds": fu,
            "delta_folds": delta,
            "delta_mean": float(np.nanmean(d)) if n_fin else float("nan"),
            "delta_2se": (
                float(2.0 * np.nanstd(d, ddof=1) / math.sqrt(n_fin)) if n_fin > 1 else float("nan")
            ),
        }

    # BATCHED paired bootstrap at the headline layers (plan §4.2 step 7 + §9).
    idx_matrix = draw_index_matrix(n, args.n_boot, args.seed + 7)
    w = counts_from_indices(idx_matrix, n)
    delta_r2: dict[str, dict] = {}
    delta_dist: dict[str, list[float]] = {}
    marginal_ci: dict[str, dict] = {"assistant": {}, "user": {}}
    equivalence: dict[str, dict] | None = {} if equivalence_gate else None
    for li in headline:
        p_a = sweeps["assistant"]["preds_frozen"][li]
        p_u = sweeps["user"]["preds_frozen"][li]
        y_a_l = rows["Y_a"][:, li, :]
        y_u_l = rows["Y_u"][:, li, :]
        boot = paired_bootstrap_batched(p_a, y_a_l, p_u, y_u_l, w)
        if equivalence is not None:
            oracle = paired_bootstrap_serial_reference(p_a, y_a_l, p_u, y_u_l, idx_matrix)
            diffs = {
                k: float(np.nanmax(np.abs(boot[k] - oracle[k])))
                for k in ("assistant", "user", "delta")
            }
            ok = bool(all(v < EQUIV_TOL for v in diffs.values()))
            equivalence[str(li)] = {"max_abs_diff": diffs, "tol": EQUIV_TOL, "pass": ok}
            assert ok, (
                f"{pair_id} L{li}: batched-vs-serial paired-bootstrap equivalence FAILED: {diffs}"
            )
            print(f"[role_contrast] {pair_id} L{li}: equivalence gate PASS {diffs}")
        r2_obs_a = float(sweeps["assistant"]["r2_obs"][li])
        r2_obs_u = float(sweeps["user"]["r2_obs"][li])
        pooled_a = float(fit_cells._pooled_r2(p_a, y_a_l))
        pooled_u = float(fit_cells._pooled_r2(p_u, y_u_l))
        delta_r2[str(li)] = {
            # LABEL-DRIVING point estimate (plan §1/§6): full-sample pooled
            # r2_obs delta; the bootstrap CI is for that same statistic family.
            "delta_obs": r2_obs_a - r2_obs_u,
            "r2_obs_assistant": r2_obs_a,
            "r2_obs_user": r2_obs_u,
            "delta_pooled_global_obs": pooled_a - pooled_u,
            **_ci(boot["delta"]),
        }
        delta_dist[str(li)] = [float(v) for v in boot["delta"]]
        for role, arr, pooled in (
            ("assistant", boot["assistant"], pooled_a),
            ("user", boot["user"], pooled_u),
        ):
            marginal_ci[role][str(li)] = {"r2_pooled_global_obs": pooled, **_ci(arr)}

    # Per-example cosine deltas + paired NLL companion (steps 8-9), same W.
    cosine_delta: dict[str, dict] = {}
    for li in headline:
        d = np.asarray(sweeps["assistant"]["cosines"][li], dtype=np.float64) - np.asarray(
            sweeps["user"]["cosines"][li], dtype=np.float64
        )
        dist = weighted_mean_draws(d, w)
        cosine_delta[str(li)] = {
            "mean": float(d.mean()),
            **_ci(dist),
            "distribution": [float(v) for v in dist],
            "per_row": [float(v) for v in d],
        }
    nd = rows["nll_a"] - rows["nll_u"]
    nll_dist = weighted_mean_draws(nd, w)
    nll_delta = {
        "mean": float(nd.mean()),
        "nll_assistant_mean": float(rows["nll_a"].mean()),
        "nll_user_mean": float(rows["nll_u"].mean()),
        **_ci(nll_dist),
        "distribution": [float(v) for v in nll_dist],
        "per_row": [float(v) for v in nd],
    }

    # Per-role selection-symmetric reads (step 10) — delta DESCRIPTIVE only.
    selection: dict = {
        role: fit_cells.selection_symmetric_summary(sweeps[role]["r2_obs"], sweeps[role]["r2_null"])
        for role in ROLES
    }
    selection["delta_of_layer_maxes_descriptive"] = float(
        selection["assistant"]["obs_layer_max_r2"] - selection["user"]["obs_layer_max_r2"]
    )

    # MLP paired secondary (step 11) on the SHARED folds.
    mlp = run_mlp_paired(
        rows, folds, layers=headline, n_folds=args.folds, budget_s=args.mlp_budget_s
    )
    mlp_paired: dict[str, dict] = {}
    for li in map(str, headline):
        fa = mlp["assistant"][li]["r2_obs_folds"]
        fu = mlp["user"][li]["r2_obs_folds"]
        delta = [a - u for a, u in zip(fa, fu, strict=True)]
        d = np.asarray(delta, dtype=np.float64)
        n_fin = int(np.isfinite(d).sum())
        mlp_paired[li] = {
            "delta_folds": delta,
            "delta_mean": float(np.nanmean(d)) if n_fin else float("nan"),
            "delta_2se": (
                float(2.0 * np.nanstd(d, ddof=1) / math.sqrt(n_fin)) if n_fin > 1 else float("nan")
            ),
            "r2_obs_assistant": mlp["assistant"][li]["r2_obs"],
            "r2_obs_user": mlp["user"][li]["r2_obs"],
            "budget_hit_folds": {
                "assistant": mlp["assistant"][li]["budget_hit_folds"],
                "user": mlp["user"][li]["budget_hit_folds"],
            },
        }

    # Persist (step 12): per-role cells_/nulls_ payloads, preds npz, pair JSON —
    # ALL written the moment the pair completes (checkpoint-per-pair law).
    for role in ROLES:
        cid = cell_id(model, role, fmt)
        sweep = sweeps[role]
        y = rows[f"Y_{role[0]}"]
        cos_stats = {
            str(li): fit_cells.bootstrap_ci(
                sweep["cosines"][li], n_boot=args.n_boot, seed=args.seed + li
            )
            for li in frozen
        }
        y_trace = {
            str(li): float(y[:, li, :].astype(np.float64).var(axis=0, ddof=1).sum())
            for li in frozen
        }
        cell_payload = {
            "metadata": _metadata(args.seed, n),
            "cell": {
                "cell_id": cid,
                "model": model,
                "role": role,
                "format": fmt,
                "provenance": prov,
                "pair_id": pair_id,
                "track": "m",
            },
            "row_allowlist_applied": rows["allowlist"]["applied"],
            "n_allowlist": rows["allowlist"].get("n_allowlist"),
            "joint_keep_n": n,
            "committed_reference": committed_ref[role],
            "gate_read": gate_reads.get(role),
            "r2_per_layer_obs": [float(v) for v in sweep["r2_obs"]],
            "selection_symmetric": selection[role],
            "cosine_frozen_layers": cos_stats,
            "r2_bootstrap_ci_frozen_layers": marginal_ci[role],
            "r2_bootstrap_ci_note": (
                "headline layers only; marginal per-role draws from the SHARED paired "
                "resample matrix (own-mean-centered weighted form)"
            ),
            "y_trace_cov_frozen": y_trace,
            "n_folds": args.folds,
            "null_draws": args.null_draws,
            "mlp": mlp[role],
            "mlp_budget_exhausted": any(mlp[role][str(li)]["budget_hit_folds"] for li in headline),
        }
        fit_cells._write_json(prov_out / f"cells_{cid}.json", cell_payload)
        fit_cells._write_json(
            prov_out / f"nulls_{cid}.json",
            {
                "metadata": _metadata(args.seed, n),
                "cell_id": cid,
                "provenance": prov,
                "layers": list(range(len(sweep["r2_obs"]))),
                "observed_row": [float(v) for v in sweep["r2_obs"]],
                "null_matrix": [[float(v) for v in row] for row in sweep["r2_null"]],
                "null_layer_max_per_draw": selection[role]["null_layer_max_r2_per_draw"],
            },
        )

    args.preds_dir.mkdir(parents=True, exist_ok=True)
    preds_file = args.preds_dir / f"preds_{pair_id}.npz"
    arrays = {
        f"{pair_id}__{role}__L{li}": sweeps[role]["preds_frozen"][li].astype(np.float16)
        for role in ROLES
        for li in headline
    }
    np.savez(preds_file, **arrays)  # plain savez, never savez_compressed (#813)
    sha = hashlib.sha256(preds_file.read_bytes()).hexdigest()
    _update_preds_manifest(
        args.out_dir, preds_file.name, sha, {k: list(v.shape) for k, v in arrays.items()}
    )

    pair_payload = {
        "metadata": _metadata(args.seed, n),
        "pair": dict(pair),
        "n_joint": n,
        "n_bundle": rows["n_bundle"],
        "allowlist": rows["allowlist"],
        "committed_reference": committed_ref,
        "gate_reads": gate_reads,
        "headline_layers": [int(v) for v in headline],
        "delta_r2_frozen": delta_r2,
        "delta_r2_distribution": delta_dist,
        "per_fold_delta_r2": per_fold,
        "mlp_paired": mlp_paired,
        "cosine_delta": cosine_delta,
        "nll_delta": nll_delta,
        "selection_symmetric": selection,
        "equivalence_gate": equivalence,
        "n_folds": args.folds,
        "null_draws": args.null_draws,
        "n_boot": args.n_boot,
        "regime_key": regime_key(args, allowlist, equivalence_gate),
        "preds_file": preds_file.name,
    }
    fit_cells._write_json(prov_out / f"{pair_id}.json", pair_payload)
    return pair_payload


def _pair_done(pair: dict, args, allow_map: dict, equivalence_gate: bool) -> bool:
    """Resume predicate: pair JSON + both cells/nulls + preds npz present AND
    the persisted regime_key matches EVERY output-affecting knob (#722 r3)."""
    prov = pair["provenance"]
    payload = _load_json(args.out_dir / prov / f"{pair['pair_id']}.json")
    if payload is None:
        return False
    allowlist = _pair_allowlist(pair, allow_map)
    if payload.get("regime_key") != regime_key(args, allowlist, equivalence_gate):
        return False
    for role in ROLES:
        cid = cell_id(pair["model"], role, pair["format"])
        if not (args.out_dir / prov / f"cells_{cid}.json").exists():
            return False
        if not (args.out_dir / prov / f"nulls_{cid}.json").exists():
            return False
    return (args.preds_dir / f"preds_{pair['pair_id']}.npz").exists()


def cmd_fit(args) -> int:
    """The 12-pair fit phase: per-pair crashes DEFER to fit_failures.json (MF-C)
    and the wrapper sequence STILL reaches the binding post-upload gates —
    summarize tolerates the deferred pairs' missing JSONs, uploads run, then
    check_deferred HALTs with the registered status (``bundle_schema_mismatch``
    | ``fit_deferred_failure``), never ``summarize_error`` (round-2 fix). A
    pair that completes (fresh or on ``--resume``) clears its stale record."""
    assert args.allowlists.exists(), f"--allowlists missing: {args.allowlists}"
    allow_map = json.loads(args.allowlists.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, pair in enumerate(pair_registry()):
        eq = i < args.equivalence_gate_pairs
        if args.resume and _pair_done(pair, args, allow_map, eq):
            print(f"[role_contrast] RESUME skip {pair['pair_id']} (outputs present, regime match)")
        else:
            try:
                run_pair(pair, args, allow_map, equivalence_gate=eq)
            except Exception as e:  # deferred, never pre-upload fatal (MF-C)
                _record_deferred_failure(args.out_dir, pair["pair_id"], e)
                continue
            # RAM hygiene: run_pair's locals (bundle arrays) are freed on return.
        _clear_deferred_failure(args.out_dir, pair["pair_id"])
    print("[role_contrast] fit done")
    return 0


# ---------------------------------------------------------------------------
# Summarize: plan §1 labels + the pre-registered corrected inverted read
# ---------------------------------------------------------------------------


def _phi(x: float) -> float:
    """Standard normal CDF (the §1 pre-registered normal approximation)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def build_gate_table(args) -> list[dict]:
    """Reproduction-gate rows for all 24 refit cells (20 gated + 4 sandwich).

    NON-raising (shared by summarize + gates): gates mode routes misses to
    GateFailure; summarize embeds the table in headline_metrics.json. Keyed
    strictly per (provenance subdir, committed dir) — no cross-provenance join
    by conv_id or cell_id (plan §8 collision risk).
    """
    rows = []
    for pair in pair_registry():
        prov, model, fmt = pair["provenance"], pair["model"], pair["format"]
        committed_dir = _committed_dir(args, prov)
        for role in ROLES:
            cid = cell_id(model, role, fmt)
            refit = _load_json(args.out_dir / prov / f"cells_{cid}.json")
            committed = _load_json(committed_dir / f"cells_{cid}.json")
            row: dict = {
                "provenance": prov,
                "cell_id": cid,
                "gated": role in gated_roles(prov),
            }
            if refit is None or committed is None:
                row["status"] = "missing_refit" if refit is None else "missing_committed"
                rows.append(row)
                continue
            n_refit = (refit.get("metadata") or {}).get("n")
            n_comm = (committed.get("metadata") or {}).get("n")
            refit_l19 = _l19(refit)
            comm_l19 = _l19(committed)
            row.update(
                n_refit=n_refit,
                n_committed=n_comm,
                refit_r2_l19=refit_l19,
                committed_r2_l19=comm_l19,
            )
            if n_refit == n_comm:
                used = refit_l19
                row["gate_value_source"] = "joint_rowset"
            else:
                gate_read = refit.get("gate_read")
                if gate_read:
                    used = gate_read.get("r2_l19_own_rowset")
                    row["gate_value_source"] = "own_rowset_refit"
                    row["n_own_rowset"] = gate_read.get("n_own_rowset")
                else:
                    used = None
                    row["gate_value_source"] = "missing_gate_read"
            row["gate_value"] = used
            if (
                used is not None
                and comm_l19 is not None
                and not (math.isnan(used) or math.isnan(comm_l19))
            ):
                row["abs_delta"] = abs(used - comm_l19)
                row["within_tol"] = bool(row["abs_delta"] <= REPRO_TOL)
            else:
                row["abs_delta"] = None
                row["within_tol"] = None
            if not row["gated"]:
                row["note"] = (
                    "onpolicy assistant-on-allowlist: NEW fit, gate-exempt; sandwiched by "
                    "the committed @2000 anchor (different row set) — plan §4.2"
                )
            rows.append(row)
    return rows


def _slim(d: dict, drop: tuple[str, ...] = ("per_row", "distribution")) -> dict:
    return {k: v for k, v in d.items() if k not in drop}


def cmd_summarize(args) -> int:
    """headline_metrics.json: §1 labels per provenance, per-pair delta tables,
    companions, gate table, MF-R + MF-RC verbatim, corrected inverted read.

    Deferred-failure tolerance (round-2 fix, deferred-failures-bypass-gates):
    when fit_failures.json records exist, a missing pair JSON is TOLERATED —
    the pair is listed under the block's ``deferred_missing_pairs`` and the
    headline carries the deferred set — so the wrapper sequence still reaches
    upload + gates, where check_deferred HALTs FIRST with the registered
    status. A missing pair JSON with NO deferred record is still a hard assert
    (fail-loud: that is a run-order bug, and ``summarize_error`` is then the
    correct sentinel status)."""
    deferred = _deferred_entries(args.out_dir)
    have_deferred = any(deferred.values())
    prov_blocks: dict[str, dict] = {}
    n_total = 0
    for prov in PROVENANCES:
        block: dict = {
            "pairs": {},
            "localized_negatives": [],
            "inverted_headline_reads": [],
            "deferred_missing_pairs": [],
        }
        confirmed = True
        for pair in [p for p in pair_registry() if p["provenance"] == prov]:
            pj = _load_json(args.out_dir / prov / f"{pair['pair_id']}.json")
            if pj is None and have_deferred:
                block["deferred_missing_pairs"].append(pair["pair_id"])
                confirmed = False
                print(
                    f"[role_contrast] summarize: {pair['pair_id']} JSON missing with deferred "
                    "failure(s) recorded — tolerated; the post-upload gates HALT with the "
                    "registered status"
                )
                continue
            assert pj is not None, (
                f"missing pair JSON for {pair['pair_id']} with NO deferred failure recorded "
                "— run fit first"
            )
            n_total += pj["n_joint"]
            entry: dict = {}
            for li in map(str, HEADLINE_LAYERS):
                row = pj["delta_r2_frozen"].get(li)
                if row is None:
                    confirmed = False
                    continue
                delta, lo, hi, se = row["delta_obs"], row["ci_lo"], row["ci_hi"], row["se_boot"]
                read = {
                    **row,
                    "ci_excludes_zero": bool(lo > 0 or hi < 0),
                    "direction": "assistant_gt_user" if delta > 0 else "user_ge_assistant",
                }
                if delta > 0 and lo > 0:
                    pass  # supports ROLE-GAP-CONFIRMED
                elif delta < 0 and hi < 0:
                    # Pre-registered corrected inverted read (plan §1): one-sided
                    # normal approximation p = Phi(delta_obs / se_boot); the
                    # empirical percentile cannot resolve a 0.00208 tail at B=1000.
                    p_norm = _phi(delta / se) if se > 0 else float("nan")
                    read["p_normal_one_sided_inverted"] = p_norm
                    read["bonferroni_alpha"] = BONF_ALPHA
                    read["bonferroni_survives"] = bool(p_norm < BONF_ALPHA)
                    other = str(
                        HEADLINE_LAYERS[1] if li == str(HEADLINE_LAYERS[0]) else HEADLINE_LAYERS[0]
                    )
                    rec = {
                        "pair_id": pair["pair_id"],
                        "layer": int(li),
                        "delta_obs": delta,
                        "p_normal_one_sided": p_norm,
                        # Free re-checks named BEFORE any headline movement (§1).
                        "free_rechecks": {
                            "other_headline_layer": pj["delta_r2_frozen"].get(other),
                            "mlp_paired_delta": pj["mlp_paired"].get(li),
                            "cosine_delta": _slim(pj["cosine_delta"].get(li, {})),
                        },
                    }
                    (
                        block["inverted_headline_reads"]
                        if read["bonferroni_survives"]
                        else block["localized_negatives"]
                    ).append(rec)
                    confirmed = False
                else:
                    confirmed = False
                entry[li] = read
            block["pairs"][pair["pair_id"]] = {
                "n_joint": pj["n_joint"],
                "delta_r2": entry,
                "per_fold_delta_r2": pj["per_fold_delta_r2"],
                "mlp_paired": pj["mlp_paired"],
                "cosine_delta": {li: _slim(d) for li, d in pj["cosine_delta"].items()},
                "nll_delta": _slim(pj["nll_delta"]),
                "selection_delta_of_maxes_descriptive": pj["selection_symmetric"][
                    "delta_of_layer_maxes_descriptive"
                ],
            }
        if block["deferred_missing_pairs"]:
            # Honest label for a run that HALTs at gate 1: never a science label
            # computed over a partial pair set (round-2 fix).
            block["label"] = "INCOMPLETE-DEFERRED-FAILURES"
        elif block["inverted_headline_reads"]:
            block["label"] = "INVERTED"
        elif confirmed and not block["localized_negatives"]:
            block["label"] = "ROLE-GAP-CONFIRMED"
        else:
            block["label"] = "MIXED"
        prov_blocks[prov] = block

    headline = {
        "followup_label": FOLLOWUP_LABEL,
        "smoke": eps_smoke(),
        "headline_layers": list(HEADLINE_LAYERS),
        "n_primary_reads": N_PRIMARY_READS,
        "bonferroni_alpha": BONF_ALPHA,
        "label_rule": (
            "ROLE-GAP-CONFIRMED: all 4 pairs read paired delta-ridge > 0 with the 95% "
            "paired-bootstrap CI excluding 0 at BOTH L19 AND L26 (conjunctive — no "
            "correction for the positive claim). MIXED: >=1 CI spans 0 at either headline "
            "layer, no Bonferroni-surviving inverted read (a CI-excluding-0 negative that "
            "does NOT survive alpha=0.05/24 is a localized negative, reported with its "
            "free re-checks). INVERTED: any delta < 0 read with CI excluding 0 that "
            "survives Bonferroni via the pre-registered normal-approximation one-sided "
            "p = Phi(delta_obs/se_boot)."
        ),
        "point_estimate_definition": (
            "delta_obs = full-sample pooled r2_obs_assistant - r2_obs_user (the "
            "fold-pooled sweep value; LABEL-DRIVING); CI = percentile bootstrap of the "
            "paired conversation-level resample deltas (per-draw own-mean-centered "
            "weighted form, fp64, shared resamples across roles)"
        ),
        "provenances": prov_blocks,
        # The deferred set rides the headline verbatim (minimal-headline contract:
        # the gates HALT on it next, with the registered status).
        "deferred_failures": deferred,
        "reproduction_gate_table": build_gate_table(args),
        "interpretation_rule_mf_r": MF_R_RULE,
        "interpretation_rule_mf_rc": MF_RC_RULE,
        "metadata": _metadata(0, n_total),
    }
    fit_cells._write_json(args.out_dir / "headline_metrics.json", headline)
    for prov, block in prov_blocks.items():
        print(f"[role_contrast] summarize: {prov} -> {block['label']}")
    return 0


# ---------------------------------------------------------------------------
# Binding post-upload gates (plan §7) — evaluated by the wrapper AFTER uploads
# ---------------------------------------------------------------------------


def _deferred_entries(out_dir: Path) -> dict[str, list[dict]]:
    """Every fit_failures.json record under out_dir (rglob — nested sweeps too),
    keyed by relative path. Shared by summarize (tolerance) + gate 1 (HALT)."""
    return {
        str(p.relative_to(out_dir)): json.loads(p.read_text())
        for p in sorted(out_dir.rglob("fit_failures.json"))
    }


def check_deferred(out_dir: Path) -> dict:
    """Gate 1 sweep: any deferred fit failure HALTs; a bundle-schema assert is
    routed to its own status (plan §7 gate 1). Binding in smoke too. Fires
    FIRST in cmd_gates, so a deferred-failure run always exits with the
    registered status — never a downstream gate's."""
    deferred = _deferred_entries(out_dir)
    if deferred:
        entries = [e for v in deferred.values() for e in v]
        schema = [e for e in entries if e.get("status_hint") == "bundle_schema_mismatch"]
        status = "bundle_schema_mismatch" if schema else "fit_deferred_failure"
        raise GateFailure(
            status,
            f"{len(entries)} deferred fit failure(s): "
            + "; ".join(f"{e['cell_id']}: {e['error_type']}: {e['error'][:120]}" for e in entries),
        )
    return {"result": "PASS"}


def check_row_alignment(args, smoke: bool) -> dict:
    """Gate 2: joint-keep row count per pair >= 0.95 x the committed row set
    (the committed USER cell's n — parent/real 2000, onpolicy the allowlist n).
    Pair-JSON presence binds in smoke; the numeric floor is production-only."""
    vals: dict[str, dict] = {}
    for pair in pair_registry():
        prov, model, fmt = pair["provenance"], pair["model"], pair["format"]
        pj = _load_json(args.out_dir / prov / f"{pair['pair_id']}.json")
        if pj is None:
            raise GateFailure("coverage_miss", f"missing pair JSON {prov}/{pair['pair_id']}.json")
        committed_user = _load_json(
            _committed_dir(args, prov) / f"cells_{cell_id(model, 'user', fmt)}.json"
        )
        n_comm = ((committed_user or {}).get("metadata") or {}).get("n")
        vals[pair["pair_id"]] = {"n_joint": pj["n_joint"], "n_committed_rowset": n_comm}
        if smoke:
            continue
        if not isinstance(n_comm, int):
            raise GateFailure(
                "row_alignment_shortfall",
                f"{pair['pair_id']}: committed user cell missing/n-less — cannot size the committed row set",
            )
        if pj["n_joint"] < ROW_ALIGN_FLOOR * n_comm:
            raise GateFailure(
                "row_alignment_shortfall",
                f"{pair['pair_id']}: joint n={pj['n_joint']} < {ROW_ALIGN_FLOOR} x committed {n_comm}",
            )
    return {"result": "BYPASSED_SMOKE_PRESENCE_ONLY" if smoke else "PASS", "pairs": vals}


def check_reproduction(args, smoke: bool) -> dict:
    """Gate 3: the 20 anchored cells reproduce committed L19 r2_obs within
    ±0.01 (own-row-set gate_read when joint n != committed n). Numeric
    comparison is production-only; refit/committed PRESENCE binds in smoke."""
    table = build_gate_table(args)
    for row in table:
        if row.get("status") == "missing_refit":
            raise GateFailure(
                "coverage_miss", f"missing refit cell {row['provenance']}/{row['cell_id']}"
            )
        if not row["gated"]:
            continue
        if row.get("status") == "missing_committed":
            raise GateFailure(
                "reproduction_gate_miss",
                f"{row['provenance']}/{row['cell_id']}: committed comparison JSON missing "
                "(broken/sparse checkout?)",
            )
        if smoke:
            continue
        if row["gate_value_source"] == "missing_gate_read":
            raise GateFailure(
                "reproduction_gate_miss",
                f"{row['provenance']}/{row['cell_id']}: joint n {row['n_refit']} != committed "
                f"{row['n_committed']} and no own-row-set gate_read recorded",
            )
        if row["within_tol"] is not True:
            raise GateFailure(
                "reproduction_gate_miss",
                f"{row['provenance']}/{row['cell_id']}: refit L19 {row['gate_value']} vs "
                f"committed {row['committed_r2_l19']} (|delta|={row['abs_delta']}) > "
                f"{REPRO_TOL} — rig drift, HALT",
            )
    n_gated = sum(1 for r in table if r["gated"])
    return {
        "result": "BYPASSED_SMOKE_PRESENCE_ONLY" if smoke else "PASS",
        "n_gated": n_gated,
        "table": table,
    }


def _check_cell_coverage(args, prov: str, cid: str) -> None:
    """One refit cell's structural coverage: cells_/nulls_ JSONs present + a
    per-headline-layer MLP block with EITHER a finite fold fit OR logged
    budget-cap hits (blocks-or-logged-caps, plan §7 gate 4)."""
    cp = _load_json(args.out_dir / prov / f"cells_{cid}.json")
    if cp is None:
        raise GateFailure("coverage_miss", f"missing {prov}/cells_{cid}.json")
    if not (args.out_dir / prov / f"nulls_{cid}.json").exists():
        raise GateFailure("coverage_miss", f"missing {prov}/nulls_{cid}.json")
    mlp = cp.get("mlp") or {}
    for li in map(str, HEADLINE_LAYERS):
        blk = mlp.get(li)
        if blk is None:
            raise GateFailure("coverage_miss", f"{prov}/{cid}: no MLP block for layer {li}")
        folds_r2 = blk.get("r2_obs_folds") or []
        has_fit = any(isinstance(v, float) and math.isfinite(v) for v in folds_r2)
        if not has_fit and not blk.get("budget_hit_folds"):
            raise GateFailure(
                "coverage_miss",
                f"{prov}/{cid} L{li}: MLP block has neither a finite fold fit nor "
                "logged budget-cap hits",
            )


def check_coverage(args, smoke: bool) -> dict:
    """Gate 4: 12/12 pair JSONs + 24/24 cell payloads + 24/24 nulls + MLP
    blocks-or-logged-caps + headline_metrics.json + preds manifest (12 files).
    Structural — binds in smoke too. Gate 5 rides here: any recorded
    equivalence result must PASS, and under smoke the first 2 registry pairs
    MUST carry equivalence records (the dispatched-path gate, plan §4.3.3)."""
    for i, pair in enumerate(pair_registry()):
        prov, model, fmt = pair["provenance"], pair["model"], pair["format"]
        pj = _load_json(args.out_dir / prov / f"{pair['pair_id']}.json")
        if pj is None:
            raise GateFailure("coverage_miss", f"missing pair JSON {prov}/{pair['pair_id']}.json")
        eq = pj.get("equivalence_gate")
        if eq is not None:
            for li, rec in eq.items():
                if not rec.get("pass"):
                    raise GateFailure(
                        "equivalence_gate_miss",
                        f"{pair['pair_id']} L{li}: batched-vs-serial bootstrap equivalence "
                        f"recorded FAILING: {rec}",
                    )
        elif smoke and i < 2:
            raise GateFailure(
                "equivalence_gate_miss",
                f"{pair['pair_id']}: equivalence record missing at smoke (first 2 pairs "
                "must gate the dispatched batched-bootstrap path)",
            )
        for role in ROLES:
            _check_cell_coverage(args, prov, cell_id(model, role, fmt))
    if not (args.out_dir / "headline_metrics.json").exists():
        raise GateFailure("coverage_miss", "headline_metrics.json missing")
    manifest = _load_json(args.out_dir / "preds_manifest.json")
    if manifest is None:
        raise GateFailure("coverage_miss", "preds_manifest.json missing")
    missing = [
        f"preds_{p['pair_id']}.npz"
        for p in pair_registry()
        if f"preds_{p['pair_id']}.npz" not in (manifest.get("files") or {})
    ]
    if missing:
        raise GateFailure("coverage_miss", f"preds manifest missing entries: {missing}")
    return {"result": "PASS", "pairs": 12, "cells": 24}


def cmd_gates(args) -> int:
    """Plan §7 order: deferred/schema -> row-alignment -> reproduction ->
    coverage(+equivalence). Uploads already ran (wrapper phase order, MF-C);
    on the FIRST failure: gate_outcomes.json + FAILURE sentinel + exit 1."""
    smoke = eps_smoke()
    outcomes: dict = {"smoke": smoke, "followup_label": FOLLOWUP_LABEL, "gates": {}}
    try:
        print("gate armed: deferred-fit-failures")
        outcomes["gates"]["deferred_fit_failures"] = check_deferred(args.out_dir)
        print("gate: deferred-fit-failures PASS (none recorded)")
        print("gate armed: row-alignment")
        outcomes["gates"]["row_alignment"] = check_row_alignment(args, smoke)
        print(f"gate: row-alignment {outcomes['gates']['row_alignment']['result']}")
        print("gate armed: reproduction")
        outcomes["gates"]["reproduction"] = check_reproduction(args, smoke)
        print(f"gate: reproduction {outcomes['gates']['reproduction']['result']}")
        print("gate armed: coverage")
        outcomes["gates"]["coverage"] = check_coverage(args, smoke)
        print("gate: coverage PASS (12 pairs + 24 cells + 24 nulls + MLP + headline + preds)")
        outcomes["all_pass"] = True
    except GateFailure as gf:
        outcomes["all_pass"] = False
        outcomes["failure"] = {"status": gf.status, "message": gf.message}
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "gate_outcomes.json").write_text(
            json.dumps(outcomes, indent=2, default=float)
        )
        write_sentinel(
            args.sentinel,
            gf.status,
            {
                "followup_label": FOLLOWUP_LABEL,
                "failure": gf.message,
                "gate_outcomes": outcomes,
                "uploads_completed_before_gates": not smoke,
            },
        )
        raise SystemExit(f"GATE FAIL [{gf.status}]: {gf.message}") from gf
    (args.out_dir / "gate_outcomes.json").write_text(json.dumps(outcomes, indent=2, default=float))
    print("gate: ALL PASS" + (" [smoke: numeric gates bypassed]" if smoke else ""))
    return 0


# ---------------------------------------------------------------------------
# Sentinels (poll_pipeline schema via issue825_realuser_gates.write_sentinel)
# ---------------------------------------------------------------------------


def _data_repo() -> str:
    return os.environ.get("EPS_DATA_REPO", "superkaiba1/explore-persona-space-data")


def cmd_fail_sentinel(args) -> int:
    """Upload-then-exit for a crashed smoke/fit/summarize phase (plan §4.3
    hard-req 2): FIRST push whatever the phase produced — out_dir text/JSON to
    eval_results_mirror AND any completed preds npz (+ preds_manifest.json)
    from ``--preds-dir`` to analysis_tensors, mirroring the normal UPLOAD-a/b
    phases (round-2 fix, failure-sentinel-misses-preds-upload: a crash after k
    pairs completed must not lose their preds) — THEN write the FAILURE
    sentinel. Smoke = structural print."""
    produced = (
        sorted(str(p.relative_to(args.out_dir)) for p in args.out_dir.rglob("*.json"))
        if args.out_dir.exists()
        else []
    )
    preds_dir: Path | None = getattr(args, "preds_dir", None)
    preds = (
        sorted(preds_dir.glob("preds_pair_*.npz"))
        if preds_dir is not None and preds_dir.exists()
        else []
    )
    manifest_path = args.out_dir / "preds_manifest.json"
    if eps_smoke():
        print(
            f"[role_contrast] [smoke] fail-sentinel structural: {len(produced)} JSONs + "
            f"{len(preds)} preds npz would upload"
        )
    elif produced or preds:
        assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing (source .env before upload)"
        from huggingface_hub import HfApi

        api = HfApi()
        signal.alarm(2700)
        try:
            if produced:
                api.upload_folder(
                    folder_path=str(args.out_dir),
                    repo_id=_data_repo(),
                    repo_type="dataset",
                    path_in_repo=f"{HF_RC_PREFIX}/eval_results_mirror",
                    allow_patterns=["**/*.json", "*.json"],
                    commit_message=(
                        f"issue-825 {FOLLOWUP_LABEL}: {args.phase} FAILURE path upload-then-exit "
                        "(produced JSONs BEFORE the sentinel)"
                    ),
                )
            if preds:
                api.upload_folder(
                    folder_path=str(preds_dir),
                    repo_id=_data_repo(),
                    repo_type="dataset",
                    path_in_repo=f"{HF_RC_PREFIX}/analysis_tensors",
                    allow_patterns=["preds_pair_*.npz"],
                    commit_message=(
                        f"issue-825 {FOLLOWUP_LABEL}: {args.phase} FAILURE path UPLOAD-b mirror "
                        "(completed preds npz BEFORE the sentinel)"
                    ),
                )
                if manifest_path.exists():
                    api.upload_file(
                        path_or_fileobj=str(manifest_path),
                        path_in_repo=f"{HF_RC_PREFIX}/analysis_tensors/preds_manifest.json",
                        repo_id=_data_repo(),
                        repo_type="dataset",
                        commit_message=(
                            f"issue-825 {FOLLOWUP_LABEL}: {args.phase} FAILURE path preds "
                            "manifest (BEFORE the sentinel)"
                        ),
                    )
        finally:
            signal.alarm(0)
        print(
            f"[role_contrast] fail-sentinel upload: ok ({len(produced)} JSONs + "
            f"{len(preds)} preds npz)"
        )
    else:
        print("[role_contrast] fail-sentinel: nothing produced — nothing to upload")
    write_sentinel(
        args.sentinel,
        f"{args.phase}_error",
        {
            "followup_label": FOLLOWUP_LABEL,
            "failure": f"{args.phase} phase exited non-zero (see the [phase={args.phase}] log traceback)",
            "uploaded_before_sentinel": produced,
            "uploaded_preds_before_sentinel": [p.name for p in preds],
        },
    )
    return 0


def cmd_success_sentinel(args) -> int:
    """SUCCESS sentinel: refuses unless gate_outcomes.json shows all_pass; slim
    eval_numbers (labels + per-pair headline deltas); MEASURED GPU-hours."""
    gates_out = _load_json(args.out_dir / "gate_outcomes.json")
    assert gates_out is not None, "gate_outcomes.json missing — gates must run before success"
    assert gates_out.get("all_pass") is True, (
        f"gate_outcomes.json not all_pass — refusing SUCCESS sentinel: {gates_out.get('failure')}"
    )
    headline = _load_json(args.out_dir / "headline_metrics.json")
    assert headline is not None, "headline_metrics.json missing"
    stage = _load_json(args.out_dir / "stage_manifest.json") or {}
    slim = {
        prov: {
            "label": block["label"],
            "pairs": {
                pid: {
                    li: {
                        k: v
                        for k, v in (read or {}).items()
                        if k in ("delta_obs", "ci_lo", "ci_hi", "se_boot", "ci_excludes_zero")
                    }
                    for li, read in p.get("delta_r2", {}).items()
                }
                for pid, p in block["pairs"].items()
            },
            "n_localized_negatives": len(block["localized_negatives"]),
            "n_inverted_headline_reads": len(block["inverted_headline_reads"]),
        }
        for prov, block in headline["provenances"].items()
    }
    t0 = float(os.environ.get("EPS_T0", time.time()))
    write_sentinel(
        args.sentinel,
        "success",
        {
            "followup_label": FOLLOWUP_LABEL,
            "eval_numbers": slim,
            "gate_outcomes": {
                "all_pass": True,
                "gates": {
                    k: (v.get("result", "PASS") if isinstance(v, dict) else v)
                    for k, v in gates_out["gates"].items()
                },
            },
            "eval_paths": sorted(str(p) for p in args.out_dir.rglob("*.json")),
            "reproducibility_card": {
                "models": "none loaded (tensors-only re-fit; plan §10)",
                "fit_seed": 0,
                "data_repo_revision": stage.get("revision"),
                "staged_prefixes": stage.get("prefixes"),
                "followup_label": FOLLOWUP_LABEL,
            },
            "wandb_url": "n/a (analysis-only follow-up; no training)",
            "hf_hub_url": f"https://huggingface.co/datasets/{_data_repo()}/tree/main/{HF_RC_PREFIX}",
            "worktree_path": os.environ.get("EPS_WORKTREE", str(Path.cwd())),
            "final_commit_sha": os.environ.get("EPS_GIT_SHA", "unknown"),
            "gpu_hours_used": round((time.time() - t0) / 3600.0, 3),
            "gpu_hours_used_basis": "measured wrapper wall-clock (single-GPU provision)",
            "gpu_hours_budgeted": 4.0,
            "plan_deviations": [],
        },
    )
    return 0


# ---------------------------------------------------------------------------
# Smoke fabrication (EPS_SMOKE=1 stage phase — same pipeline, tiny bundles)
# ---------------------------------------------------------------------------


def cmd_fabricate_smoke(args) -> int:
    """Tiny synthetic .pt bundles satisfying the extractor shard contract
    (plan §4.3: n=3 x 3 shards, D=8, L=28; haiku n_turns=4 parent-shaped,
    real/onpolicy n_turns=3; perpos INCLUDED so the keys= exclusion is
    exercised; conv ids IDENTICAL across provenances — the production 0..1999
    collision shape, provenance-scoped processing must never join on them).
    Writes row_allowlists.json dropping the last conv per onpolicy user cell
    (exercises intersect-before-folds).

    Fault injector (round-2 fix, deferred-failures-bypass-gates smoke leg):
    ``EPS_SMOKE_CORRUPT_PAIR=<pair_id>`` fabricates THAT pair's bundle with
    n_turns=2 (< PROV_MIN_TURNS), so load_pair_bundle raises BundleSchemaError
    at fit — proving the defer -> summarize-tolerate -> upload -> gate
    registered-status routing end-to-end through the dispatched wrapper."""
    corrupt = os.environ.get("EPS_SMOKE_CORRUPT_PAIR", "")
    if corrupt and corrupt not in {p["pair_id"] for p in pair_registry()}:
        raise ValueError(f"EPS_SMOKE_CORRUPT_PAIR={corrupt!r} is not a registry pair_id")
    rng = np.random.default_rng(0)
    n_layers, dim, n_per_shard, n_shards = fit_cells.EXPECTED_LAYERS, 8, 3, 3
    dirs = {"haiku": args.haiku_dir, "real": args.real_dir, "onpolicy": args.onpolicy_dir}
    for prov in PROVENANCES:
        d = dirs[prov]
        d.mkdir(parents=True, exist_ok=True)
        for model in MODELS:
            for fmt in FORMATS:
                is_corrupt = f"pair_{prov}_{model}_{fmt}" == corrupt
                n_turns = 2 if is_corrupt else PROV_MIN_TURNS[prov]
                cbase = 0
                for s in range(n_shards):
                    slots = rng.normal(size=(n_per_shard, 2, n_layers, dim)).astype(np.float32)
                    profiles = (
                        rng.normal(size=(n_per_shard, n_turns, n_layers, dim)).astype(np.float32)
                        * 0.5
                    )
                    profiles[:, 1] += 0.8 * slots[:, 0]  # assistant target predictable
                    if n_turns >= 3:
                        profiles[:, 2] += 0.3 * slots[:, 1]  # user target weaker
                    nll = np.concatenate(
                        [
                            rng.uniform(0.4, 0.8, size=(n_per_shard, 2)),
                            rng.uniform(2.0, 3.0, size=(n_per_shard, n_turns - 2)),
                        ],
                        axis=1,
                    ).astype(np.float32)
                    payload = {
                        "conv_ids": [f"c{i}" for i in range(cbase, cbase + n_per_shard)],
                        "slots": torch.from_numpy(slots),
                        "profiles": torch.from_numpy(profiles),
                        "perpos": torch.from_numpy(
                            rng.normal(
                                size=(n_per_shard, n_turns, 4, len(fit_cells.FROZEN_LAYERS), dim)
                            ).astype(np.float32)
                        ),
                        "perpos_mask": torch.ones(n_per_shard, n_turns, 4, dtype=torch.bool),
                        "nll": torch.from_numpy(nll),
                    }
                    torch.save(payload, d / f"{model}_{fmt}_m_shard{s:03d}.pt")
                    cbase += n_per_shard
    if corrupt:
        print(
            f"[role_contrast] smoke fault injection: {corrupt} fabricated with n_turns=2 "
            "(BundleSchemaError at load -> deferred -> registered gate status)"
        )
    all_ids = [f"c{i}" for i in range(n_per_shard * n_shards)]
    allow = {cell_id(m, "user", f): all_ids[:-1] for m in MODELS for f in FORMATS}
    args.allowlists.parent.mkdir(parents=True, exist_ok=True)
    args.allowlists.write_text(json.dumps(allow))
    print(
        f"[role_contrast] smoke bundles fabricated (3 provenances x 4 bundles x 3 shards, "
        f"n={len(all_ids)}, D={dim}) + allowlists ({len(all_ids) - 1}/{len(all_ids)} rows)"
    )
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="issue-825 role-map-comparison paired role contrast")
    ap.add_argument(
        "mode",
        choices=(
            "fabricate-smoke",
            "fit",
            "summarize",
            "gates",
            "fail-sentinel",
            "success-sentinel",
        ),
    )
    ap.add_argument(
        "--out-dir", type=Path, default=Path("eval_results/issue_825/role-map-comparison")
    )
    ap.add_argument("--preds-dir", type=Path, default=Path("data/issue_825/rolecontrast/preds"))
    ap.add_argument(
        "--haiku-dir", type=Path, default=Path("data/issue_825/rolecontrast/turnstore_haiku")
    )
    ap.add_argument(
        "--real-dir", type=Path, default=Path("data/issue_825/rolecontrast/turnstore_real")
    )
    ap.add_argument(
        "--onpolicy-dir", type=Path, default=Path("data/issue_825/rolecontrast/turnstore_onpolicy")
    )
    ap.add_argument(
        "--allowlists", type=Path, default=Path("data/issue_825/rolecontrast/row_allowlists.json")
    )
    ap.add_argument("--committed-haiku", type=Path, default=Path("eval_results/issue_825"))
    ap.add_argument(
        "--committed-real", type=Path, default=Path("eval_results/issue_825/real-user-turn-null")
    )
    ap.add_argument(
        "--committed-onpolicy",
        type=Path,
        default=Path("eval_results/issue_825/onpolicy-user-turn"),
    )
    ap.add_argument("--folds", type=int, default=fit_cells.N_FOLDS)
    ap.add_argument("--null-draws", type=int, default=fit_cells.N_NULL_DRAWS)
    ap.add_argument("--n-boot", type=int, default=fit_cells.N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=fit_cells.FIT_SEED)
    ap.add_argument(
        "--mlp-budget-s",
        type=int,
        default=int(os.environ.get("EPS_MLP_TIME_BUDGET_S", "1800")),
        help="MLP budget per (pair, role) — inherited 1800 s convention",
    )
    ap.add_argument(
        "--equivalence-gate-pairs",
        type=int,
        default=0,
        help="gate the batched paired bootstrap vs the seeded serial oracle on the first N pairs",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sentinel", type=Path, default=None)
    ap.add_argument("--phase", default="fit", help="fail-sentinel: which phase crashed")
    args = ap.parse_args()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    if args.mode == "fabricate-smoke":
        return cmd_fabricate_smoke(args)
    if args.mode == "fit":
        return cmd_fit(args)
    if args.mode == "summarize":
        return cmd_summarize(args)
    assert args.sentinel is not None, f"--sentinel required for {args.mode}"
    if args.mode == "gates":
        return cmd_gates(args)
    if args.mode == "fail-sentinel":
        return cmd_fail_sentinel(args)
    return cmd_success_sentinel(args)


if __name__ == "__main__":
    raise SystemExit(main())
