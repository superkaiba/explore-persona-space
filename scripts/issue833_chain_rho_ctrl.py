#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, M⁺, r_B, →) in scientific docstrings + log messages.
"""Issue #833 follow-up — M0_ctrl chain-ρ (matched-text control map vs E).

Fits the FOURTH map's chain correlation that the Phase-D production run
(`issue833_fit_onpolicy.py`) computed function-change reads for but never
chained against the leakage grid: **M0_ctrl** (inputs c_C — the BASE-side
context vectors, same design as M0; targets v0(R⁺) — base weights θ0 over the
trained models' OWN answers). Decision read: if ρ(M0_ctrl, E) ≈ ρ(M⁺_on, E)
the +0.67 fact on-policy chain is TEXT-carried (the base map over own text
predicts E just as well); if ρ(M0_ctrl, E) ≪ ρ(M⁺_on, E) it is
WEIGHTS-specific.

Same ridge-only LOCO recipe as the committed `chain_rho/` JSONs, via the SAME
imported #722 harness (never re-implemented): `fitM._pca_basis_v0` (shared
top-64 v0-PCA target), `fitM._ridge_loco_pred` (nested-CV PRESS-λ closed-form
dual ridge, leave-one-cell-out), `fitM._chain_rho_one` (Spearman of
r_Bᵀ M̂(c) vs E), `clustered_bootstrap_spearman` (95% family-clustered CI),
and `fitM._clustered_paired_rho_diff_ci` for the two paired arm-difference
reads: M⁺_on − M0_ctrl (the decision read) and M0_ctrl − M0 (does own-text
alone move the base map's predictivity). The M0 and M⁺_on chains are
RECOMPUTED from the same joined cache (their per-cell chain values are not
stored in the committed JSONs) and asserted to reproduce the committed
`rho_*_ridge` points within a loose numeric tolerance — a fail-loud staleness
/ regime guard, not a science read.

The MLP-vs-shuffle nonlinearity gate is DELIBERATELY NOT RUN for this arm
(GPU-expensive; the follow-up brief scopes it out) — persisted as
`"mlp_gate": "not_run"` so the analyzer can caveat it.

Inputs (all local / cached; ANALYSIS-ONLY — no training, no generation):
- `<out-dir>/joined_cache/{behavior}_L{layer}.npz` — the Phase-D joined
  designs (all four legs incl. V0on = v0(R⁺)), written by the production run
  under its full join-coverage + rbase-hash gates; regime_json + file sha256
  are re-recorded in every output for provenance.
- `eval_results/issue_537/G_tensor/G_meta.json` — the committed E target.
- `r_b.pt` / `r_b_fact.pt` — HF-cached behavior directions (fitM loaders).

Outputs: `<out-dir>/chain_rho_ctrl/{behavior}_L{layer}.json`, written
atomically PER CELL the moment the cell completes (checkpoint-per-unit), with
a resume predicate keyed on the joined-cache sha256 (`--force-rerun`
overrides).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue658_fit_predictors as fit658  # noqa: E402
import issue722_fit_M as fitM  # noqa: E402

from explore_persona_space.analysis.issue667.gate_chain import (  # noqa: E402
    clustered_bootstrap_spearman,
)

logger = logging.getLogger("issue833.chain_ctrl")

DEFAULT_CELLS = tuple(f"{b}:{li}" for b in ("em", "sycophancy", "fact") for li in (7, 14, 21))
TARGET_DIM = 64  # the production run's shared top-v0-PCA target dim (A35_MLP_TARGET_DIM)
# Recomputed-vs-committed ρ guard: the production ridge ran float64 on the GCP
# lane's auto-resolved device; a CPU re-run can flip a near-tie PRESS argmin-λ on
# at most a fold or two, moving one row's rank (Δρ ≲ 6/n ≈ 0.0125 at n=480). A
# larger gap means a WRONG/STALE cache or regime — fail loud, do not write.
CONSISTENCY_FAIL_TOL = 0.02
_JOINED_STACK_KEYS = ("C0", "Cplus", "V0", "Von", "V0on")
_REQUIRED_OUT_KEYS = frozenset(
    {
        "rho_M0_ctrl_ridge",
        "ci_M0_ctrl_ridge",
        "ci_diff_Mplus_on_minus_M0_ctrl",
        "ci_diff_M0_ctrl_minus_M0",
        "consistency_vs_committed",
        "mlp_gate",
        "meta",
    }
)


def _sha256_file(path: Path) -> str:
    """Streaming sha256 of a file (the joined-cache npz provenance pin)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    """HEAD commit of the tree this script runs from (reproducibility metadata)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as e:  # metadata-only: record the failure, never crash the fit
        return f"unavailable ({e})"


def _versions() -> dict[str, str]:
    import scipy
    import torch

    return {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__}


def _load_joined_stacks(path: Path) -> dict:
    """Load one Phase-D joined-cache npz DIRECTLY (stacks + keys + stored regime).

    Unlike the production driver's `load_joined_cache`, no regime re-derivation
    happens here — the cache IS the pinned input artifact of this follow-up, so
    the stored `regime_json` + the file sha256 are surfaced as provenance in the
    output instead. Asserts the expected stack shapes.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — this follow-up consumes the Phase-D joined cache; "
            "re-run issue833_fit_onpolicy.py --joined-cache to regenerate it"
        )
    d = np.load(path, allow_pickle=True)
    out: dict = {k: np.asarray(d[k], dtype=np.float64) for k in _JOINED_STACK_KEYS}
    n = out["C0"].shape[0]
    for k in _JOINED_STACK_KEYS:
        assert out[k].shape == out["C0"].shape, (k, out[k].shape, out["C0"].shape)
    for k in ("families", "cell_keys"):
        out[k] = [str(v) for v in d[k].tolist()]
        assert len(out[k]) == n, (k, len(out[k]), n)
    out["regime"] = json.loads(str(d["regime_json"].item()))
    out["sha256"] = _sha256_file(path)
    logger.info("[phase=chain_ctrl] loaded %s: n=%d, sha256=%s", path.name, n, out["sha256"][:12])
    return out


def _out_is_complete(path: Path, cache_sha: str) -> bool:
    """Resume predicate: existing output is complete AND pinned to the SAME cache."""
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return False
    return _REQUIRED_OUT_KEYS.issubset(obj) and (
        obj.get("meta", {}).get("joined_cache_sha256") == cache_sha
    )


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def run_cell(
    behavior: str,
    layer: int,
    joined: dict,
    r_hat: np.ndarray,
    committed: dict,
) -> dict:
    """One (behavior, layer) cell: 3 ridge-LOCO chains + CIs + paired diffs.

    Arms: M0 (C0 → V0), M⁺_on (Cplus → Von) — recomputed for the paired diffs
    and the consistency guard — and M0_ctrl (C0 → V0on), the new read. Returns
    the output JSON dict (metadata added by the caller).
    """
    C0, Cplus = joined["C0"], joined["Cplus"]
    V0, Von, V0on = joined["V0"], joined["Von"], joined["V0on"]
    families, cell_keys = joined["families"], joined["cell_keys"]

    E = fitM._load_E(behavior, cell_keys)
    keep = ~np.isnan(E)
    block: dict = {"behavior": behavior, "layer": layer, "n_with_E": int(keep.sum())}
    if keep.sum() < 4:
        raise RuntimeError(f"{behavior} L{layer}: only {int(keep.sum())} cells with E (<4)")
    Ek = E[keep]
    fam_k = [f for f, m in zip(families, keep, strict=True) if m]

    pca = fitM._pca_basis_v0(V0, TARGET_DIM)  # shared base-v0 basis (production recipe)
    arms = {
        "M0": (C0, V0 @ pca.T),
        "Mplus_on": (Cplus, Von @ pca.T),
        "M0_ctrl": (C0, V0on @ pca.T),
    }
    chains: dict[str, np.ndarray] = {}
    for arm, (X, Y64) in arms.items():
        t0 = time.perf_counter()
        loco = fitM._ridge_loco_pred(X, Y64)
        rho, chain = fitM._chain_rho_one(loco[keep], pca, r_hat, Ek)
        logger.info(
            "[phase=chain_ctrl] %s L%d arm %s: rho=%s (%.1fs)",
            behavior,
            layer,
            arm,
            "None" if rho is None else f"{rho:+.4f}",
            time.perf_counter() - t0,
        )
        block[f"rho_{arm}_ridge"] = rho
        if rho is not None:
            block[f"ci_{arm}_ridge"] = clustered_bootstrap_spearman(chain, Ek, fam_k)
            chains[arm] = chain

    # ---- paired arm-difference CIs (the decision reads) ----
    if "M0_ctrl" in chains and "Mplus_on" in chains:
        block["ci_diff_Mplus_on_minus_M0_ctrl"] = fitM._clustered_paired_rho_diff_ci(
            chains["M0_ctrl"], chains["Mplus_on"], Ek, fam_k
        )
    else:
        block["ci_diff_Mplus_on_minus_M0_ctrl"] = None
    if "M0" in chains and "M0_ctrl" in chains:
        block["ci_diff_M0_ctrl_minus_M0"] = fitM._clustered_paired_rho_diff_ci(
            chains["M0"], chains["M0_ctrl"], Ek, fam_k
        )
    else:
        block["ci_diff_M0_ctrl_minus_M0"] = None

    # ---- fail-loud consistency guard vs the committed chain_rho JSON ----
    consistency: dict = {}
    for arm, committed_key in (("M0", "rho_M0_ridge"), ("Mplus_on", "rho_Mplus_on_ridge")):
        got = block.get(f"rho_{arm}_ridge")
        want = committed.get(committed_key)
        entry = {"recomputed": got, "committed": want}
        if got is not None and want is not None:
            delta = abs(float(got) - float(want))
            entry["abs_delta"] = delta
            if delta > CONSISTENCY_FAIL_TOL:
                raise RuntimeError(
                    f"{behavior} L{layer}: recomputed rho_{arm}_ridge {got:+.6f} vs committed "
                    f"{want:+.6f} (|Δ|={delta:.4f} > {CONSISTENCY_FAIL_TOL}) — joined cache / "
                    "regime mismatch with the committed chain_rho run; refusing to write"
                )
            if delta > 1e-6:
                logger.warning(
                    "[phase=chain_ctrl] %s L%d rho_%s recomputed differs from committed by "
                    "%.2e (device/λ-tie numeric drift; within tolerance)",
                    behavior,
                    layer,
                    arm,
                    delta,
                )
        consistency[arm] = entry
    block["consistency_vs_committed"] = consistency
    block["mlp_gate"] = "not_run"  # brief-scoped: GPU-expensive gate skipped for this arm
    return block


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #833 follow-up — M0_ctrl chain-ρ vs E")
    ap.add_argument(
        "--cells",
        nargs="+",
        default=list(DEFAULT_CELLS),
        help="behavior:layer cells (smoke = the same script with one cell)",
    )
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_833")
    ap.add_argument("--force-rerun", action="store_true")
    args = ap.parse_args()

    cells = [(c.split(":")[0], int(c.split(":")[1])) for c in args.cells]
    for beh, _li in cells:
        assert beh in ("em", "sycophancy", "fact"), f"unknown behavior {beh}"

    fit658.DEVICE = fit658._resolve_device("auto")
    fit658._assert_ridge_exactness()  # startup exactness gate (fit_M precedent)
    logger.info("[phase=chain_ctrl] ridge exactness gate PASS (device=%s)", fit658.DEVICE)
    fitM.TARGET_DIM = TARGET_DIM  # match the production run's module-global

    g_meta = PROJECT_ROOT / "eval_results/issue_537/G_tensor/G_meta.json"
    if not g_meta.exists():
        raise FileNotFoundError(
            f"{g_meta} missing — the chain-ρ E target is a committed git artifact. In a "
            "sparse worktree run `git sparse-checkout add eval_results/issue_537` first."
        )
    rb_main = fitM._load_rb_main()
    rb_fact = fitM._load_rb_fact() if any(b == "fact" for b, _ in cells) else None
    if any(b == "fact" for b, _ in cells) and rb_fact is None:
        raise RuntimeError("fact requested but r_b_fact.pt unavailable/degenerate")

    meta_common = {
        "script": "scripts/issue833_chain_rho_ctrl.py",
        "git_commit": _git_head(),
        "generated_at": datetime.now(UTC).isoformat(),
        "versions": _versions(),
        "ridge_device": fit658.DEVICE,
        "target_dim": TARGET_DIM,
        "ridge_lambdas": list(fit658.RIDGE_LAMBDAS),
        "n_bootstrap_resamples": 1000,
        "consistency_fail_tol": CONSISTENCY_FAIL_TOL,
    }

    t_start = time.perf_counter()
    done = 0
    for beh, li in cells:
        out_path = args.out_dir / "chain_rho_ctrl" / f"{beh}_L{li}.json"
        cache_path = args.out_dir / "joined_cache" / f"{beh}_L{li}.npz"
        cache_sha = _sha256_file(cache_path) if cache_path.exists() else None
        if not args.force_rerun and cache_sha and _out_is_complete(out_path, cache_sha):
            logger.info("[phase=chain_ctrl] %s L%d already complete — skip (resume)", beh, li)
            done += 1
            continue
        committed_path = args.out_dir / "chain_rho" / f"{beh}_L{li}.json"
        if not committed_path.exists():
            raise FileNotFoundError(
                f"{committed_path} missing — the consistency guard needs the committed "
                "chain_rho JSON for this cell"
            )
        committed = json.loads(committed_path.read_text())
        joined = _load_joined_stacks(cache_path)
        r_hat = fitM._r_hat_for(beh, li, rb_main, rb_fact)
        t_cell = time.perf_counter()
        block = run_cell(beh, li, joined, r_hat, committed)
        block["meta"] = {
            **meta_common,
            "joined_cache_path": str(cache_path.relative_to(PROJECT_ROOT)),
            "joined_cache_sha256": joined["sha256"],
            "joined_cache_regime": joined["regime"],
            "committed_chain_rho_path": str(committed_path.relative_to(PROJECT_ROOT)),
            "cell_wall_seconds": round(time.perf_counter() - t_cell, 1),
        }
        _write_json(out_path, block)  # checkpoint-per-cell: persisted the moment it completes
        done += 1
        logger.info(
            "[phase=chain_ctrl] %s L%d DONE (%d/%d): rho_M0_ctrl=%s, on-minus-ctrl=%s "
            "(%.1fs; wrote %s)",
            beh,
            li,
            done,
            len(cells),
            f"{block['rho_M0_ctrl_ridge']:+.4f}"
            if block["rho_M0_ctrl_ridge"] is not None
            else "None",
            json.dumps((block["ci_diff_Mplus_on_minus_M0_ctrl"] or {}).get("point")),
            time.perf_counter() - t_cell,
            out_path,
        )
    logger.info(
        "[phase=chain_ctrl] ALL DONE: %d/%d cells in %.1f min",
        done,
        len(cells),
        (time.perf_counter() - t_start) / 60,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
