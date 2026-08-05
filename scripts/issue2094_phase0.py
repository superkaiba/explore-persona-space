"""Issue #2094 phase 0 — zero-GPU homogeneity pilot on #1415's banked captures (plan §4.6).

Stages #1415's per-α steered answer-state captures at steer-layer L20
(``analysis_tensors/issue_1415/activations_steered/gen1c/{context,prefix}/<pair>/L20/a{0.5,1,2,4}.pt``,
28 pairs × 2 arms × 4 α = 224 files) plus the 28 unsteered baseline bundles
(``analysis_tensors/issue_1415/activations/<pair>.pt``) from the HF data repo at
the plan-pinned revision, and computes the unit-B homogeneity reads per
(arm × pair × read-layer):

- pairwise dose-shift direction stability cos(shift@α_i, shift@α_j) under the
  DISJOINT-baseline-halves convention (the #1415 shared-baseline-inflation fix —
  the two legs subtract means of disjoint halves of the K=10 floor draws,
  averaged over both half assignments), with the shared-floor matrix kept as a
  labeled record-only companion (known-inflated);
- per-α split-half reliability of the shift (``fmetrics.shift_split_half_reliability``
  + Spearman-Brown) and the disattenuated cosine matrix;
- ``log||shift||`` vs ``log α`` slope/intercept (``fmetrics.log_log_magnitude_fit``)
  + the unity-slope reference anchored at α=1.

Output: ``eval_results/issue_2094/phase0_homogeneity.json`` (aggregates +
metadata) with per-unit rows checkpointed to ``phase0_homogeneity_cells.jsonl``
(atomic append + resume keyed on (arm, pair, revision, convention) — the
intra-phase persistence grain for the 56-unit compute loop).

Zero GPU; ~320 MB of staged tensors ride the HF cache (re-runs are cache hits).
Every ``hf_hub_download`` rides ``hub.retry_transient`` (#1547 routing).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # before torch import (thread caps) + before any Hub call (HF_TOKEN)

import torch  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from explore_persona_space.experiments.issue2094 import fmetrics as FM  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue2094_phase0")

REPO = "superkaiba1/explore-persona-space-data"
# Plan §4.6 pin (HF revision verified 2026-08-05); existence re-verified at the
# fact-check: 28 pairs × 2 arms × 4 α under gen1c + 28 baseline bundles.
DEFAULT_REVISION = "79dacd5239"
PREFIX = "analysis_tensors/issue_1415"
ARMS = ("context", "prefix")
STEER_LAYER = 20
ALPHAS = (0.5, 1.0, 2.0, 4.0)
ALPHA_TOKENS = ("a0.5", "a1", "a2", "a4")
ALPHA1_INDEX = 1  # α=1 fixed point (unity-slope anchor, plan §4.4)
CONVENTION = "disjoint-halves-v1"  # part of the resume regime key
N_RELIABILITY_SPLITS = 20


# ── pure math (unit-tested in tests/test_issue2094_judge.py) ───────────


def pairwise_disjoint_cosines(
    steered_means: torch.Tensor, floor_draws: torch.Tensor
) -> torch.Tensor:
    """(A, A) dose-shift cosines with DISJOINT floor halves per leg (#1415 fix).

    ``steered_means``: ``(A, H)`` per-α steered answer-state means;
    ``floor_draws``: ``(K, H)`` unsteered baseline draws. Leg i subtracts one
    half's mean and leg j the other's, so the shared-floor noise cross-term
    vanishes in expectation; the two half assignments are averaged. The
    diagonal is the shift's own split-half read (legs share no noise), NOT 1.
    """
    assert steered_means.dim() == 2, steered_means.shape
    assert floor_draws.dim() == 2, floor_draws.shape
    assert steered_means.shape[-1] == floor_draws.shape[-1], (
        steered_means.shape,
        floor_draws.shape,
    )
    m1, m2 = FM.disjoint_half_means(floor_draws.double())
    s1 = steered_means.double() - m1  # legs under half-1 floor
    s2 = steered_means.double() - m2  # legs under half-2 floor
    tiny = torch.finfo(torch.float64).tiny
    n1 = s1.norm(dim=-1, keepdim=True)
    n2 = s2.norm(dim=-1, keepdim=True)
    u1 = s1 / n1.clamp_min(tiny)
    u2 = s2 / n2.clamp_min(tiny)
    cos12 = u1 @ u2.T  # (A, A): leg i under half-1 vs leg j under half-2
    cos = 0.5 * (cos12 + cos12.T)  # average over the two half assignments
    zero = (n1.squeeze(-1) == 0) | (n2.squeeze(-1) == 0)
    cos[zero, :] = torch.nan  # zero-norm shift rows flagged, never coerced
    cos[:, zero] = torch.nan
    return cos.float()


# ── staging ────────────────────────────────────────────────────────────


def steered_path(arm: str, pair: str, alpha_token: str) -> str:
    return f"{PREFIX}/activations_steered/gen1c/{arm}/{pair}/L{STEER_LAYER}/{alpha_token}.pt"


def baseline_path(pair: str) -> str:
    return f"{PREFIX}/activations/{pair}.pt"


def list_pairs(revision: str, limit: int = 0) -> list[str]:
    """The 28 pair names, derived from the pinned tree (never hardcoded).

    ``limit`` > 0 truncates for a tiny-real smoke (scratch out-paths only —
    never the committed defaults); the 28-pair completeness assert applies to
    the full run only.
    """
    # Canonical retried scoped listing (#920/#997) — one server-side tree walk,
    # never a bare list_repo_tree / full-repo listing on the ~1M-file data repo.
    root = f"{PREFIX}/activations_steered/gen1c/context"
    files = hub.list_hf_files_under_path(
        HfApi(), REPO, root, repo_type="dataset", revision=revision
    )
    # File paths look like <root>/<pair>/L20/aX.pt — the pair is the segment
    # immediately under the listed root.
    depth = len(root.split("/"))
    pairs = sorted({f.split("/")[depth] for f in files if f.startswith(root + "/")})
    if limit > 0:
        return pairs[:limit]
    assert len(pairs) == 28, f"expected 28 pairs, got {len(pairs)}: {pairs[:3]}..."
    return pairs


def stage_files(pairs: list[str], revision: str, max_workers: int = 6) -> dict[str, Path]:
    """Scoped per-file staging at the pinned revision (NEVER snapshot_download
    against the ~1M-file data repo, #833); bounded pool, retried per file.
    Files land in the HF cache, so re-runs are cache hits (the resume grain)."""
    wanted = [baseline_path(p) for p in pairs] + [
        steered_path(arm, p, tok) for arm in ARMS for p in pairs for tok in ALPHA_TOKENS
    ]
    out: dict[str, Path] = {}

    def fetch(rel: str) -> tuple[str, Path]:
        local = hub.retry_transient(
            lambda: hf_hub_download(REPO, rel, repo_type="dataset", revision=revision),
            what=f"hf_hub_download({rel})",
        )
        return rel, Path(local)

    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fetch, rel) for rel in wanted]
        for i, fut in enumerate(cf.as_completed(futures)):
            rel, local = fut.result()
            out[rel] = local
            if (i + 1) % 25 == 0 or i + 1 == len(wanted):
                logger.info("[phase0] stage %d/%d", i + 1, len(wanted))
    assert len(out) == len(wanted), (len(out), len(wanted))
    return out


_STEERED_KEYS = ("v_a_mean", "v_a_per_completion", "layers", "alpha")
_BASELINE_KEYS = ("c", "layers")


def _load_bundle(path: Path, required: tuple[str, ...]) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=True)
    missing = [k for k in required if k not in d]
    assert not missing, f"{path}: missing keys {missing} (realized-keys check, artifact-reuse (c))"
    return d


# ── per-unit compute ───────────────────────────────────────────────────


def compute_unit(arm: str, pair: str, staged: dict[str, Path]) -> dict:
    """All homogeneity reads for one (arm, pair): per read layer, the disjoint +
    shared cosine matrices, reliabilities, disattenuated matrix, log-log fit."""
    base = _load_bundle(staged[baseline_path(pair)], _BASELINE_KEYS)
    read_layers = list(base["layers"])
    floor_all = base["c"]["v_a_per_completion"].float()  # (K, L, H)
    assert floor_all.dim() == 3, floor_all.shape

    steered_by_alpha = []
    for tok, alpha in zip(ALPHA_TOKENS, ALPHAS, strict=True):
        d = _load_bundle(staged[steered_path(arm, pair, tok)], _STEERED_KEYS)
        assert list(d["layers"]) == read_layers, (d["layers"], read_layers)
        assert abs(float(d["alpha"]) - alpha) < 1e-9, (d["alpha"], alpha)
        steered_by_alpha.append(d["v_a_mean"].float())  # (L, H)

    alphas_t = torch.tensor(ALPHAS, dtype=torch.float64)
    per_layer: dict[str, dict] = {}
    for li, layer in enumerate(read_layers):
        sm = torch.stack([s[li] for s in steered_by_alpha])  # (A, H)
        floor_draws = floor_all[:, li]  # (K, H)
        floor_mean = floor_draws.mean(dim=0)
        shifts_shared = sm - floor_mean  # (A, H) — record-only convention

        disjoint = pairwise_disjoint_cosines(sm, floor_draws)
        shared = FM.pairwise_shift_cosines(shifts_shared)
        rel_half = FM.shift_split_half_reliability(
            sm, floor_draws, n_splits=N_RELIABILITY_SPLITS, seed=0
        )
        rel_sb = FM.spearman_brown(rel_half)
        disattenuated = FM.disattenuated_cosines(disjoint, rel_sb)

        norms = shifts_shared.norm(dim=-1)
        degenerate = bool((norms <= 0).any())
        if degenerate:
            slope = intercept = None
            unity = None
        else:
            s, b = FM.log_log_magnitude_fit(alphas_t.float(), norms)
            slope, intercept = float(s), float(b)
            unity = FM.unity_slope_reference(alphas_t.float(), norms[ALPHA1_INDEX]).tolist()

        per_layer[f"L{layer}"] = {
            "cos_disjoint": disjoint.tolist(),
            "cos_shared_record_only": shared.tolist(),  # known-inflated (#1415)
            "reliability_sb": rel_sb.tolist(),
            "cos_disattenuated": disattenuated.tolist(),
            "shift_norms": norms.tolist(),
            "log_log_slope": slope,
            "log_log_intercept": intercept,
            "unity_slope_reference_norms": unity,
            "degenerate_zero_norm": degenerate,
        }
    return {
        "arm": arm,
        "pair": pair,
        "steer_layer": STEER_LAYER,
        "read_layers": read_layers,
        "alphas": list(ALPHAS),
        "n_floor_draws": int(floor_all.shape[0]),
        "per_layer": per_layer,
    }


# ── aggregation + persistence ─────────────────────────────────────────


def _offdiag_mean(matrix: list[list[float]]) -> float | None:
    vals = [
        matrix[i][j]
        for i in range(len(matrix))
        for j in range(len(matrix))
        if i != j and matrix[i][j] == matrix[i][j]  # NaN-safe
    ]
    return sum(vals) / len(vals) if vals else None


def aggregate(cells: list[dict]) -> dict:
    """Per (arm × read layer): mean off-diagonal disjoint/disattenuated cosine
    over pairs + the median log-log slope (the plan §4.4 homogeneity reads)."""
    out: dict[str, dict] = {}
    for arm in ARMS:
        arm_cells = [c for c in cells if c["arm"] == arm]
        if not arm_cells:
            continue
        layers = arm_cells[0]["read_layers"]
        per_layer = {}
        for layer in layers:
            key = f"L{layer}"
            dis = [_offdiag_mean(c["per_layer"][key]["cos_disjoint"]) for c in arm_cells]
            dat = [_offdiag_mean(c["per_layer"][key]["cos_disattenuated"]) for c in arm_cells]
            slopes = sorted(
                c["per_layer"][key]["log_log_slope"]
                for c in arm_cells
                if c["per_layer"][key]["log_log_slope"] is not None
            )
            dis_v = [v for v in dis if v is not None]
            dat_v = [v for v in dat if v is not None]
            per_layer[key] = {
                "mean_offdiag_cos_disjoint": sum(dis_v) / len(dis_v) if dis_v else None,
                "mean_offdiag_cos_disattenuated": sum(dat_v) / len(dat_v) if dat_v else None,
                "median_log_log_slope": slopes[len(slopes) // 2] if slopes else None,
                "n_pairs": len(arm_cells),
                "n_degenerate": sum(
                    1 for c in arm_cells if c["per_layer"][key]["degenerate_zero_norm"]
                ),
            }
        out[arm] = per_layer
    return out


def _cell_key(row: dict, revision: str) -> tuple:
    return (row["arm"], row["pair"], revision, CONVENTION)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #2094 phase-0 homogeneity pilot (zero-GPU).")
    ap.add_argument("--revision", type=str, default=DEFAULT_REVISION)
    ap.add_argument(
        "--out", type=Path, default=Path("eval_results/issue_2094/phase0_homogeneity.json")
    )
    ap.add_argument(
        "--cells-out",
        type=Path,
        default=Path("eval_results/issue_2094/phase0_homogeneity_cells.jsonl"),
    )
    ap.add_argument(
        "--limit-pairs",
        type=int,
        default=0,
        help="smoke only: process the first N pairs (requires explicit scratch "
        "--out/--cells-out — refuses the committed defaults)",
    )
    args = ap.parse_args(argv)
    if args.limit_pairs > 0 and str(args.out).startswith("eval_results/"):
        raise SystemExit("--limit-pairs is a smoke knob: pass scratch --out/--cells-out paths")

    pairs = list_pairs(args.revision, limit=args.limit_pairs)
    staged = stage_files(pairs, args.revision)

    # Resume: skip cells already computed under the SAME (revision, convention).
    done: dict[tuple, dict] = {}
    if args.cells_out.is_file():
        with args.cells_out.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get("revision") == args.revision and row.get("convention") == CONVENTION:
                        done[_cell_key(row, args.revision)] = row
    logger.info("[phase0] resume: %d/%d cells already computed", len(done), 2 * len(pairs))

    args.cells_out.parent.mkdir(parents=True, exist_ok=True)
    cells: list[dict] = []
    units = [(arm, pair) for arm in ARMS for pair in pairs]
    with args.cells_out.open("a", encoding="utf-8") as sink:
        for k, (arm, pair) in enumerate(units):
            key = (arm, pair, args.revision, CONVENTION)
            if key in done:
                cells.append(done[key])
                continue
            row = compute_unit(arm, pair, staged)
            row["revision"] = args.revision
            row["convention"] = CONVENTION
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            cells.append(row)
            logger.info("[phase0] unit %d/%d %s/%s done", k + 1, len(units), arm, pair)

    result = {
        "aggregates": aggregate(cells),
        "convention": {
            "shift": "steered v_a_mean minus unsteered c-arm floor mean",
            "cosines": CONVENTION + " (off-diagonal legs subtract disjoint floor-half means, both "
            "assignments averaged — the #1415 shared-baseline-inflation fix); "
            "cos_shared_record_only is the known-inflated shared-floor companion",
            "reliability": "fmetrics.shift_split_half_reliability (floor-noise only; "
            "steered mean treated as fixed) + Spearman-Brown",
            "alphas": list(ALPHAS),
            "steer_layer": STEER_LAYER,
        },
        "inputs": {
            "repo": REPO,
            "revision": args.revision,
            "prefix": PREFIX,
            "n_pairs": len(pairs),
            "limit_pairs": args.limit_pairs,  # >0 = smoke slice, never the full read
            "n_files": len(staged),
            "pairs": pairs,
        },
        "n_cells": len(cells),
        "cells_path": str(args.cells_out),
        "repro": {**as_metadata_dict(git_provenance()), "script": "scripts/issue2094_phase0.py"},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, args.out)
    logger.info("[phase0] wrote %s (%d cells)", args.out, len(cells))
    return 0


if __name__ == "__main__":
    sys.stdout.flush()
    sys.exit(main())
