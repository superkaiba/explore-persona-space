# ruff: noqa: RUF002, RUF003
"""Issue #958 free-analysis follow-up: MIXED turn-1+2 context->answer map.

Dan's question on the #958 writeup: "if you fit a map on a mix of turn 1+2, does
it generalize?" This driver pools the turn-1 AND turn-2 fit rows (each unit =
(conversation, turn), context ``ctx_last`` state -> ``answer_mean`` target), fits
ONE GCV dual/Gram ridge over the frozen 6 read-out rows, and evaluates the
resulting map held-out at turns 1, 2, 3, 4 (and especially the unseen turns 3
and 4). Two mixed arms are fit:

- ``mix12_full``   — the FULL dup-excluded fit fold pooled over turns 1+2
  (~2 x n_fit rows).
- ``mix12_matchedn`` — the same pool SUBSAMPLED (seed 0) to the single-turn
  full-n fit-row count (half the rows from each turn), the matched-n control so
  the comparison to single-turn maps is not a fit-n confound.

Two baseline arms are refit under the SAME dup-excluded folds so the comparison
is apples-to-apples: ``turn2_only`` (the turn-2-only full-n map, source-map-
composite transferred to turns 1-4) and ``own_turn`` (each turn's own full-n map
evaluated on its own turn — the diagonal). The own-turn-1 cell is a re-fit of the
round-5 dup-excluded turn-1 map; it is cross-checked against the committed
``dup-excluded-turn1-refit/refit.json`` exact-regime value rather than reported
as new work.

Everything reuses the round-5 template ``issue958_dup_excluded_refit`` verbatim:
its HF staging, the exact/lowercased duplicate-first-message grouping (cross-
checked vs the committed ``duplicate_first_message_groups.json`` sidecar), the
6-read-out-row read-out loader, and the batched dual/Gram GCV ridge core
(``issue958_fit_maps.fit_rows_batched`` / ``predict_from_fit`` /
``_skill_and_stats``). Exact-duplicate first-message conversations are excluded
from the fit AND test folds (the round-5 protocol; the turn-1 rows are otherwise
memorization-contaminated). The source-map-composite transfer policy is
unchanged (the source map's train-fold X standardization + Y centering applied
verbatim, no target-turn re-standardization); the skill null denominator at
target turn k is the turn-k full-n dup-excluded corpus mean, matching the
committed transfer_matrix cells.

Validation gate (before any new fit): reproduce two committed
``transfer_matrix.json`` cells under the ``none`` regime (capture-dropout only,
NO duplicate exclusion) to max abs delta <= 1e-6 — own_full turn-2
(0.5252886504597846) and fold-A grid 2->2 (0.48925603883473495). fp64 CPU math
reproduces them to ~1e-11 (the round-5 refit achieved 3.2e-11); a gate failure
STOPS before the new fits.

Inputs are staged from HF per file (scoped ``list_repo_tree`` +
``hf_hub_download``; NEVER a full-tree ``snapshot_download`` on the ~1M-file data
repo). Only main-panel store shards + the main corpus are needed. Store shards
are activation tensors; the corpus first-message text is real LMSYS user content
and is NEVER printed (grouping reads it only to compute group sizes/indices).

Writes eval_results/issue_958/mixed-turn-fit/mixed_fit.json (+ per-cell npz under
mixed-turn-fit/percell/) and the figure via issue958_mixed_turn_fit_fig.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy so the shared-VM thread caps bind (#847)

import issue958_common as C  # noqa: E402
import issue958_dup_excluded_refit as D  # noqa: E402 (round-5 template — reused verbatim)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue958_fit_maps import (  # noqa: E402
    RIDGE_LAMBDAS_922,
    _filter_invalid,
    _skill_and_stats,
    fit_rows_batched,
    predict_from_fit,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue958_mixed_turn_fit")

OUT = Path("eval_results/issue_958")
SUB = OUT / "mixed-turn-fit"
PERCELL = SUB / "percell"
RO = D.RO  # [15, 18, 20, 21, 25, 27] — the 6 frozen read-out rows (blocks 14/17/19/20/24/26)
KS_EVAL = [1, 2, 3, 4]  # held-out eval turns
MIX_TURNS = [1, 2]  # the pooled fit turns (Dan's mix)
LAMBDAS = list(RIDGE_LAMBDAS_922)
MATCHED_SEED = 0

# Committed transfer_matrix.json reference cells (none regime, capture-dropout only)
GATE_OWN_FULL_2 = 0.5252886504597846
GATE_FOLDA_2TO2 = 0.48925603883473495
GATE_TOL = 1e-6
# Committed round-5 dup-excluded (exact regime) own-turn-1 full-n readout-mean skill
REFIT_EXACT_OWN_TURN1_FULL = 0.5392864591771714


# ── pooled design + matched-n subsample (the NEW #958 code) ───────────────────


def _pool_xy(full: dict, turns: list[int], cis: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack per-turn read-out (X=ctx_last, Y=answer_mean) into one pooled design.

    Returns X (6, len(turns)*n, H), Y (6, len(turns)*n, H) rows-first fp16 —
    the first n columns are turn ``turns[0]``, the next n turn ``turns[1]``, ...
    (the block layout the matched-n column subsample relies on).
    """
    xs, ys = [], []
    for k in turns:
        X, Y = D._xy(full, k, cis)  # (6, n, H)
        xs.append(X)
        ys.append(Y)
    return torch.cat(xs, dim=1), torch.cat(ys, dim=1)


def _fit_pool(full: dict, turns: list[int], cis: np.ndarray, device: str) -> dict:
    """Fit ONE dual/Gram GCV ridge over the pooled (turns x cis) read-out rows."""
    X, Y = _pool_xy(full, turns, cis)
    return fit_rows_batched(X, Y, lambdas=LAMBDAS, device=device)


def _matchedn_cols(n: int, n_turns: int, n_target: int, seed: int) -> tuple[np.ndarray, list[int]]:
    """Column indices into a (n_turns*n) pool: ~n_target/n_turns per turn, seeded.

    Distributes any remainder to the earliest turns so the total is exactly
    ``n_target`` ('half from each turn' for the 2-turn mix). Independent draw
    per turn block; sorted for a deterministic, resumable column order.
    """
    per = [n_target // n_turns] * n_turns
    for i in range(n_target - sum(per)):
        per[i] += 1
    rng = np.random.default_rng(seed)
    cols = []
    for t in range(n_turns):
        assert per[t] <= n, f"matched-n per-turn {per[t]} > available {n}"
        cols.append(t * n + rng.permutation(n)[: per[t]])
    return np.sort(np.concatenate(cols)), per


def _fit_pool_matchedn(
    full: dict, turns: list[int], cis: np.ndarray, n_target: int, seed: int, device: str
) -> tuple[dict, dict]:
    """Fit on a matched-n subsample of the pooled design (the fit-n control)."""
    X, Y = _pool_xy(full, turns, cis)
    n = len(cis)
    col_idx, per = _matchedn_cols(n, len(turns), n_target, seed)
    ct = torch.from_numpy(col_idx)
    fit = fit_rows_batched(X[:, ct, :], Y[:, ct, :], lambdas=LAMBDAS, device=device)
    info = {"n_rows": int(len(col_idx)), "per_turn": [int(p) for p in per], "seed": seed}
    return fit, info


# ── evaluation (source-map-composite transfer; null = target-turn corpus mean) ─


def _eval_stats(
    fit: dict, full: dict, k: int, test_cis: np.ndarray, null_ymu: torch.Tensor, device: str
) -> dict:
    """Apply ``fit`` (source-map-composite) at target turn k on held-out test."""
    Xt, Yt = D._xy(full, k, test_cis)
    pred = predict_from_fit(fit, Xt, device=device)
    return _skill_and_stats(pred, Yt.to(torch.float64), null_ymu)


def _cell(stats: dict, idx_b: np.ndarray, test_cis: np.ndarray) -> dict:
    """Per-cell read-out-mean skill + paired-bootstrap CI + per-block + npz payload."""
    skill6, br6 = D._perrow_skill_boot(stats, idx_b)  # (6,), (draws, 6)
    boot_mean = D.boot_ro_mean(
        stats["sse_unit"].numpy(), stats["null_sse_unit"].numpy(), idx_b
    )  # (draws,)
    return {
        "readout_mean_skill": float(stats["skill"].mean()),  # 6 fitted rows ARE the read-out rows
        "ci95": D.ci95(boot_mean),
        "per_block": D._blocks(skill6, br6),
        "_npz": {
            "skill": skill6,
            "sse_unit": stats["sse_unit"].numpy(),
            "null_sse_unit": stats["null_sse_unit"].numpy(),
            "boot_readout_mean": boot_mean,
            "test_idx": np.asarray(test_cis, dtype=np.int64),
            "readout_blocks": np.asarray(C.READOUT_BLOCKS, dtype=np.int64),
        },
    }


def _write_percell(name: str, cell: dict) -> None:
    PERCELL.mkdir(parents=True, exist_ok=True)
    np.savez(PERCELL / f"{name}.npz", **cell.pop("_npz"))


# ── validation gate (reproduce committed transfer_matrix cells, none regime) ──


def validation_gate(full: dict, n_main: int, capture_invalid: set, device: str) -> dict:
    """Reproduce own_full turn-2 + fold-A grid 2->2 (none regime) to <= 1e-6."""
    split = C.make_split(n_main, n_fit=C.N_FIT, n_val=C.N_VAL, n_test=C.N_TEST, seed=C.SPLIT_SEED)
    fit_none = _filter_invalid(split["fit"], frozenset(capture_invalid))
    test_none = _filter_invalid(split["test"], frozenset(capture_invalid))
    half_a, _half_b = C.twin_halves(fit_none)

    m2_full = _fit_pool(full, [2], fit_none, device)  # turn-2-only full-n (none regime)
    s_full = _eval_stats(m2_full, full, 2, test_none, m2_full["ymu"], device)
    own_full_2 = float(s_full["skill"].mean())

    m2_a = _fit_pool(full, [2], half_a, device)  # turn-2-only fold-A (none regime)
    s_a = _eval_stats(m2_a, full, 2, test_none, m2_a["ymu"], device)
    folda_2to2 = float(s_a["skill"].mean())

    checks = {
        "own_full_turn2": {
            "recomputed": own_full_2,
            "committed": GATE_OWN_FULL_2,
            "abs_delta": abs(own_full_2 - GATE_OWN_FULL_2),
        },
        "foldA_grid_2to2": {
            "recomputed": folda_2to2,
            "committed": GATE_FOLDA_2TO2,
            "abs_delta": abs(folda_2to2 - GATE_FOLDA_2TO2),
        },
    }
    max_delta = max(c["abs_delta"] for c in checks.values())
    logger.info("[gate] reproduce-committed max|delta|=%.3e (tol %.0e)", max_delta, GATE_TOL)
    assert max_delta <= GATE_TOL, (
        f"validation gate FAILED: max|delta|={max_delta:.3e} > {GATE_TOL:.0e}; "
        f"checks={json.dumps(checks)}"
    )
    return {
        "max_abs_delta": max_delta,
        "tol": GATE_TOL,
        "checks": checks,
        "n_test_none": len(test_none),
    }


# ── selfcheck (no store; validates pooling / matched-n / skill wiring) ─────────


def _selfcheck(device: str) -> int:
    """Synthetic tiny smoke of the NEW pooling/matched-n/eval logic (no store)."""
    rng = np.random.default_rng(0)
    H, n = 16, 40
    W = rng.standard_normal((H, H))

    def _mk(shift: float) -> tuple[torch.Tensor, torch.Tensor]:
        Xc = rng.standard_normal((n, H)) + shift
        Yc = Xc @ W + 0.05 * rng.standard_normal((n, H)) + shift
        return torch.from_numpy(Xc).float(), torch.from_numpy(Yc).float()

    # full[k][ci] = (2, 6, H): position 0 = ctx (X), position 1 = answer (Y); 6 rows
    full: dict[int, dict[int, torch.Tensor]] = {}
    for k in (1, 2, 3, 4):
        X, Y = _mk(0.1 * k)
        full[k] = {}
        for i in range(n):
            row = torch.stack([X[i], Y[i]])  # (2, H)
            full[k][i] = row.unsqueeze(1).repeat(1, 6, 1)  # (2, 6, H)
    cis = np.arange(n)

    Xp, Yp = _pool_xy(full, [1, 2], cis)
    assert Xp.shape == (6, 2 * n, H) and Yp.shape == (6, 2 * n, H), (Xp.shape, Yp.shape)
    col_idx, per = _matchedn_cols(n, 2, n, MATCHED_SEED)
    assert len(col_idx) == n and per == [n // 2, n // 2], (len(col_idx), per)
    assert (col_idx[: n // 2] < n).all() and (col_idx[n // 2 :] >= n).all(), "block layout wrong"

    fit_full = _fit_pool(full, [1, 2], cis, device)
    fit_m, info = _fit_pool_matchedn(full, [1, 2], cis, n, MATCHED_SEED, device)
    assert info["n_rows"] == n and info["per_turn"] == [n // 2, n // 2]
    maps_k = {k: _fit_pool(full, [k], cis, device) for k in KS_EVAL}
    idx_b = np.random.default_rng(0).integers(0, n, size=(50, n))
    for fit in (fit_full, fit_m):
        for k in KS_EVAL:
            s = _eval_stats(fit, full, k, cis, maps_k[k]["ymu"], device)
            cell = _cell(s, idx_b, cis)
            assert cell["_npz"]["sse_unit"].shape == (6, n)
            assert -5.0 < cell["readout_mean_skill"] <= 1.0 + 1e-9
            lo, hi = cell["ci95"]
            assert lo <= hi
    logger.info("[selfcheck] pooling + matched-n + eval wiring OK (device=%s)", device)
    return 0


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=Path("data/issue_958/hf_dl/mixed_fit"),
        help="re-downloadable HF staging dir (store/main + corpus)",
    )
    ap.add_argument("--stage-only", action="store_true", help="stage inputs and exit")
    ap.add_argument("--selfcheck", action="store_true", help="synthetic wiring smoke, no store")
    ap.add_argument("--device", default="cpu", help="fit device (cpu on the VM)")
    args = ap.parse_args()
    torch.set_num_threads(8)
    device = C.resolve_device(args.device)

    if args.selfcheck:
        return _selfcheck(device)

    counts = D.stage_inputs(args.stage_root)
    if args.stage_only:
        logger.info("[stage-only] done: %s", counts)
        return 0
    store_dir = args.stage_root / "store"
    corpus_dir = args.stage_root / "corpus"

    fp = C.corpus_fingerprint(corpus_dir)
    n_main = len(C.load_corpus(corpus_dir, "main"))

    idx_main = C.load_store_index(store_dir, "main", expect_fingerprint=fp)
    capture_invalid = {
        ci
        for ci in range(n_main)
        for k in range(1, C.K_MAIN + 1)
        if C.unit_id("main", ci, k) not in idx_main
    }
    valid_cis = [ci for ci in range(n_main) if ci not in capture_invalid]
    logger.info(
        "[setup] n_main=%d capture_invalid=%d store_present=%d fp=%s",
        n_main,
        len(capture_invalid),
        len(valid_cis),
        fp[:12],
    )

    dup, dup_summary = D.duplicate_cis(corpus_dir, n_main)
    logger.info(
        "[dup] exact=%d lowercased=%d conversations",
        dup_summary["exact"]["n_dup_conversations"],
        dup_summary["lowercased"]["n_dup_conversations"],
    )

    full = D.load_readout(store_dir, n_main, fp, valid_cis)

    # cross-check duplicate definition vs the committed sidecar (none-regime test fold)
    split = C.make_split(n_main, n_fit=C.N_FIT, n_val=C.N_VAL, n_test=C.N_TEST, seed=C.SPLIT_SEED)
    test_none = _filter_invalid(split["test"], frozenset(capture_invalid))
    dup_gate = D.cross_check_committed_dups(dup, test_none)
    logger.info("[dup-gate] committed cross-check PASS: %s", json.dumps(dup_gate))

    # validation gate BEFORE the new fit (fail-loud, <= 1e-6)
    gate = validation_gate(full, n_main, capture_invalid, device)
    logger.info("[gate] PASS max|delta|=%.3e", gate["max_abs_delta"])

    # exact-duplicate-excluded folds (the round-5 protocol)
    invalid_exact = set(capture_invalid) | set(dup["exact"])
    fit_m = _filter_invalid(split["fit"], frozenset(invalid_exact))
    test_m = _filter_invalid(split["test"], frozenset(invalid_exact))
    n_fit, n_test = len(fit_m), len(test_m)
    logger.info(
        "[exact-regime] n_fit=%d n_test=%d (excluded %d dup + %d capture)",
        n_fit,
        n_test,
        len(set(dup["exact"])),
        len(capture_invalid),
    )

    # null-source maps: turn-k full-n dup-excluded (their ymu is the target-turn corpus mean)
    maps_F = {k: _fit_pool(full, [k], fit_m, device) for k in KS_EVAL}
    mix_full = _fit_pool(full, MIX_TURNS, fit_m, device)
    mix_matched, matched_info = _fit_pool_matchedn(
        full, MIX_TURNS, fit_m, n_fit, MATCHED_SEED, device
    )
    logger.info(
        "[fit] mix12_full rows=%d | mix12_matchedn rows=%d per_turn=%s | turn-k full-n rows=%d",
        len(MIX_TURNS) * n_fit,
        matched_info["n_rows"],
        matched_info["per_turn"],
        n_fit,
    )

    idx_b = np.random.default_rng(C.BOOTSTRAP_SEED).integers(
        0, n_test, size=(C.BOOTSTRAP_DRAWS, n_test)
    )

    def _arm(fit: dict) -> dict:
        return {
            str(k): _eval_stats(fit, full, k, test_m, maps_F[k]["ymu"], device) for k in KS_EVAL
        }

    arm_fits = {
        "mix12_full": mix_full,
        "mix12_matchedn": mix_matched,
        "turn2_only": maps_F[2],
        "own_turn": None,  # own_turn[k] = maps_F[k] eval at k (the diagonal)
    }
    arms: dict = {}
    percell_names = {
        "mix12_full": "mix12_full",
        "mix12_matchedn": "mix12_matchedn",
        "turn2_only": "turn2only",
        "own_turn": "ownturn",
    }
    for arm, fit in arm_fits.items():
        per_turn: dict = {}
        for k in KS_EVAL:
            src = maps_F[k] if arm == "own_turn" else fit
            stats = _eval_stats(src, full, k, test_m, maps_F[k]["ymu"], device)
            cell = _cell(stats, idx_b, test_m)
            _write_percell(f"{percell_names[arm]}_k{k}", cell)
            per_turn[str(k)] = cell
        meta = {}
        if arm == "mix12_full":
            meta = {"fit_rows": len(MIX_TURNS) * n_fit, "fit_turns": MIX_TURNS}
        elif arm == "mix12_matchedn":
            meta = {
                "fit_rows": matched_info["n_rows"],
                "fit_turns": MIX_TURNS,
                "matched_n": matched_info,
            }
        elif arm == "turn2_only":
            meta = {"fit_rows": n_fit, "fit_turns": [2]}
        else:
            meta = {"fit_rows_per_turn": n_fit, "note": "diagonal: turn-k full-n map eval own at k"}
        arms[arm] = {"meta": meta, "per_eval_turn": per_turn}

    # consistency crosscheck: own_turn[1] must reproduce refit.json exact own_turn1.full
    own1 = arms["own_turn"]["per_eval_turn"]["1"]["readout_mean_skill"]
    own1_delta = abs(own1 - REFIT_EXACT_OWN_TURN1_FULL)
    logger.info(
        "[consistency] own_turn@1 recomputed=%.12f committed=%.12f |delta|=%.3e",
        own1,
        REFIT_EXACT_OWN_TURN1_FULL,
        own1_delta,
    )
    assert own1_delta <= 1e-6, (
        f"own_turn@1 diverges from committed refit.json exact regime: |delta|={own1_delta:.3e} "
        "(should reuse the round-5 dup-excluded turn-1 map)"
    )

    res = {
        "question": (
            "Dan Mossing: if you fit a map on a mix of turn 1+2, does it generalize? "
            "Pool turn-1+turn-2 fit rows, fit ONE ridge map, evaluate held-out at turns 1-4."
        ),
        "definition": (
            "context->answer ridge maps over the frozen 6 read-out rows (blocks "
            "14/17/19/20/24/26), X=ctx_last state, Y=answer_mean target (INCL. <|im_end|>+\\n). "
            "GCV dual/Gram ridge, source-map-composite transfer, skill = 1 - SSE(pred)/"
            "SSE(target-turn corpus mean) on held-out answer_mean, read-out-mean over the 6 "
            "blocks. Exact-duplicate first-message conversations excluded from fit AND test "
            "folds (round-5 protocol). Arms: mix12_full (pool turns 1+2, full fold) / "
            "mix12_matchedn (pool subsampled to single-turn full-n, seed 0, half per turn) / "
            "turn2_only (turn-2 full-n map transferred) / own_turn (each turn's own full-n map, "
            "diagonal). Bootstrap CI seed 0, 997 draws, paired conversation resample."
        ),
        "corpus_fingerprint": fp,
        "readout_blocks": C.READOUT_BLOCKS,
        "n_main": n_main,
        "n_capture_invalid": len(capture_invalid),
        "n_fit_single_turn": n_fit,
        "n_pool_mix": len(MIX_TURNS) * n_fit,
        "n_matchedn": matched_info["n_rows"],
        "n_test": n_test,
        "mix_turns": MIX_TURNS,
        "eval_turns": KS_EVAL,
        "duplicate_summary": dup_summary,
        "duplicate_committed_cross_check": dup_gate,
        "validation_gate": gate,
        "own_turn1_consistency": {
            "recomputed": own1,
            "committed_refit_exact": REFIT_EXACT_OWN_TURN1_FULL,
            "abs_delta": own1_delta,
        },
        "seeds": {
            "split": C.SPLIT_SEED,
            "matched_n": MATCHED_SEED,
            "bootstrap": C.BOOTSTRAP_SEED,
            "bootstrap_draws": C.BOOTSTRAP_DRAWS,
        },
        "transfer_standardization_policy": C.TRANSFER_STANDARDIZATION_POLICY,
        "arms": arms,
        "metadata": C.reproducibility_metadata({"script": "issue958_mixed_turn_fit"}),
    }
    SUB.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(SUB / "mixed_fit.json", res)
    logger.info("wrote %s", SUB / "mixed_fit.json")

    # one-line headline log: mix12_full held-out skill vs eval turn (readout-mean)
    hl = {
        k: round(arms["mix12_full"]["per_eval_turn"][str(k)]["readout_mean_skill"], 4)
        for k in KS_EVAL
    }
    logger.info("[headline mix12_full] readout-mean skill by eval turn: %s", hl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
