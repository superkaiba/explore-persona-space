"""Issue #958 free-analysis follow-up: DUPLICATE-EXCLUDED main-panel turn-1 refit.

The parent's main-panel turn-1 context->answer map is degenerate (own-turn skill
-0.02 fold-A / -4.98 full, GCV lambda 0.46-0.75 on the fold-A twin), while turns
2-4 are near-stationary. The registered hypothesis (task #958 follow-up 1,
``dup-excluded-turn1-refit``) is that the degeneracy is entirely the
exact-duplicate-first-message MEMORIZATION artifact: 604/5,000 main-panel
conversations share a first user message with >=1 other conversation (the sha256
dedup keys on the FIRST-K user messages, so first-message-only duplicates
survive), so a turn-1 fit fold contains near-identical (context, answer) rows a
low-lambda ridge can memorize.

This driver re-runs the parent's EXACT fit/skill/transfer recipe with ONE change:
the 604 duplicate-group conversations are excluded from the fit AND test folds.
It reuses the parent's fold-filtering hook (``build_design`` filters an
``invalid_main`` conversation set out of every fold) by ADDING the duplicate
conversations to that invalid set, and reuses the batched dual/Gram GCV ridge
core (``fit_rows_batched`` -> stacked-eigh over the 6 read-out rows, no serial
per-cell loop), ``predict_from_fit``, ``_skill_and_stats`` (skill vs corpus-mean
null), and ``_shuffle_draws`` (shuffled-pairing band). It reports own-turn-1 skill
and the turn-1<->{2,3,4} transfer deficits both AS-FITTED (the refit's own GCV
lambda) and at MATCHED lambda (turn-1 map clamped per-row to the target turn's own
GCV selection), mirroring the parent's round-4 dual reporting.

Validation gate (the round-2/4 reproduce-the-committed-cell recipe): under the
``none`` regime (capture-dropout exclusion only, NO duplicate exclusion) the refit
reproduces the committed ``transfer_matrix.json`` fold-A grid, ``own_full`` and
``own_B`` turn-1 skills within 5e-3 (fp64 math -> expect <<). The duplicate
grouping is cross-checked against the committed
``duplicate_first_message_groups.json`` sidecar (exact count 604 + the per-test
exact flags) before any exclusion is applied.

Inputs are staged from HF per file (scoped ``list_repo_tree`` + ``hf_hub_download``;
NEVER a full-tree ``snapshot_download`` on the ~1M-file data repo) into a
re-downloadable scratch dir. Store shards are activation tensors; the corpus
first-message text is real LMSYS user content and is NEVER printed (grouping reads
it only to compute group sizes/indices).

Writes eval_results/issue_958/dup-excluded-turn1-refit/refit.json (+ per-regime
per-cell npz) and figures via issue958_dup_excluded_refit_fig.py.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy so the shared-VM thread caps bind (#847)

import issue958_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue958_fit_maps import (  # noqa: E402
    READOUT_ROWS,
    RIDGE_LAMBDAS_922,
    _filter_invalid,
    _skill_and_stats,
    _shuffle_draws,
    fit_rows_batched,
    predict_from_fit,
)
from issue958_long_k1_transfer_lclamp import alpha_at, decompose_rows  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue958_dup_refit")

OUT = Path("eval_results/issue_958")
SUB = OUT / "dup-excluded-turn1-refit"
RO = READOUT_ROWS  # [15, 18, 20, 21, 25, 27] — the 6 frozen read-out rows
KS = [2, 3, 4]  # transfer targets from turn 1
REGIMES = ["none", "exact", "lowercased"]  # none = baseline gate (capture-dropout only)


# ── HF staging (scoped; verify-before-skip; caller deletes after) ─────────────


def stage_inputs(stage_root: Path, max_workers: int = 6) -> dict:
    """Stage store/main shards + corpus main.json/manifest.json (scoped)."""
    import shutil
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate import hub

    pfx = C.HF_OUT_PREFIX
    api = HfApi()

    def _list_tree(rp: str) -> list:
        """Scoped staging listing, materialized so the HTTP call fires under retry."""
        # HUB_VERIFY_RETRY_EXEMPT: scoped download-staging listing, retried via hub.retry_transient
        entries = api.list_repo_tree(
            C.HF_DATA_REPO, path_in_repo=rp, repo_type="dataset", recursive=True
        )
        return list(entries)  # #920/#1335: materialize inside the retried thunk

    targets = {
        f"{pfx}/analysis_tensors/store/main": (stage_root / "store" / "main", None),
        f"{pfx}/corpus": (
            stage_root / "corpus",
            lambda name: name in {"main.json", "manifest.json"},
        ),
    }
    counts: dict[str, int] = {}
    for remote_prefix, (local_root, name_filter) in targets.items():
        tree = hub.retry_transient(
            lambda rp=remote_prefix: _list_tree(rp), what=f"stage list {remote_prefix}"
        )
        entries = [
            e
            for e in tree
            if getattr(e, "size", None) is not None
            and (name_filter is None or name_filter(Path(e.path).name))
        ]
        assert entries, f"HF staging: nothing under {C.HF_DATA_REPO}/{remote_prefix}"
        local_root.mkdir(parents=True, exist_ok=True)
        to_fetch = [e for e in entries if not C._staged_ok(local_root / Path(e.path).name, e.size)]
        logger.info(
            "[stage] %s: %d files (%d already staged)",
            remote_prefix,
            len(entries),
            len(entries) - len(to_fetch),
        )

        def _fetch(path: str, staging_root: str) -> str:
            last: Exception | None = None
            for attempt in range(4):
                try:
                    return hf_hub_download(
                        repo_id=C.HF_DATA_REPO,
                        filename=path,
                        repo_type="dataset",
                        local_dir=staging_root,
                    )
                except Exception as exc:
                    if not C._is_transient_hf_error(exc):
                        raise  # 4xx (quota 403 / auth 401 / 404) — loud, no retry
                    last = exc
                    logger.warning(
                        "[stage] %s failed (%s) attempt %d/4 — backoff",
                        path,
                        type(exc).__name__,
                        attempt + 1,
                    )
                    time.sleep(20 * (attempt + 1))
            raise RuntimeError(f"HF staging failed after 4 attempts: {path}") from last

        if to_fetch:
            with tempfile.TemporaryDirectory(prefix="i958_dup_", dir=str(local_root)) as td:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    list(ex.map(lambda p: _fetch(p, td), [e.path for e in to_fetch]))
                for e in to_fetch:
                    src = Path(td) / e.path
                    dst = local_root / Path(e.path).name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        dst.unlink()
                    shutil.move(str(src), str(dst))
        counts[remote_prefix] = len(entries)
    return counts


# ── duplicate-group conversation sets (recomputed; cross-checked vs committed) ──


def duplicate_cis(corpus_dir: Path, n_main: int) -> tuple[dict[str, list[int]], dict]:
    """Duplicate-first-message conversation indices over the MAIN corpus.

    Group key = the conversation's first user message under exact-string and
    lowercased equality (the committed sidecar's two normalizations); a
    conversation is duplicate iff its key appears >1 time. Returns per-normalization
    sorted ci lists + a summary. Never prints the first-message TEXT.
    """
    convs = C.load_corpus(corpus_dir, "main")
    assert len(convs) == n_main, (len(convs), n_main)
    first = [c["exchanges"][0]["user"] for c in convs]
    out: dict[str, list[int]] = {}
    summary: dict[str, dict] = {}
    for name, keyfn in (("exact", lambda m: m), ("lowercased", lambda m: m.lower())):
        groups: dict = collections.defaultdict(list)
        for i, msg in enumerate(first):
            groups[keyfn(msg)].append(i)
        dup = {k: v for k, v in groups.items() if len(v) > 1}
        cis = sorted({i for v in dup.values() for i in v})
        out[name] = cis
        summary[name] = {"n_dup_conversations": len(cis), "n_dup_groups": len(dup)}
    return out, summary


def cross_check_committed_dups(dup: dict[str, list[int]], test_m: np.ndarray) -> dict:
    """Reproduce-the-committed-cell gate for the duplicate DEFINITION.

    Assert the recomputed exact/lowercased duplicate counts and the per-test-fold
    exact/lowercased flags match the committed
    ``duplicate_first_message_groups.json`` sidecar (fail loud on divergence).
    """
    sidecar = json.loads((OUT / "duplicate_first_message_groups.json").read_text())
    checks: dict = {}
    for name in ("exact", "lowercased"):
        want = int(sidecar["per_normalization"][name]["n_dup_conversations"])
        got = len(dup[name])
        assert got == want, f"dup count[{name}]: recomputed {got} != committed {want}"
        checks[f"n_dup_{name}"] = got
    # per-test-fold flag parity (committed sidecar keys test conversations by ci)
    dup_exact = set(dup["exact"])
    dup_lower = set(dup["lowercased"])
    tc = sidecar["test_conversations"]
    mism = 0
    for ci in test_m:
        rec = tc.get(str(int(ci)))
        if rec is None:
            continue
        if bool(rec["dup_exact"]) != (int(ci) in dup_exact):
            mism += 1
        if bool(rec["dup_lowercased"]) != (int(ci) in dup_lower):
            mism += 1
    assert mism == 0, f"per-test-fold dup-flag mismatches vs committed sidecar: {mism}"
    checks["test_flag_mismatches"] = mism
    checks["n_test_flagged_committed"] = len(tc)
    return checks


# ── read-out X/Y loading (POS_CTX_END -> X, POS_ANS_MEAN -> Y, 6 rows) ──────────


def load_readout(store_dir: Path, n_main: int, fp: str, valid_cis: list[int]) -> dict:
    """{k: {ci: (2, 6, H) fp16}} over the store-present conversations, turns 1..4.

    Position 0 = ctx_end (the map's X), position 1 = answer_mean (the map's Y);
    read-out rows sliced to the frozen 6. One ``load_store_positions`` gather per
    turn; only conversations present in the store are requested.
    """
    full: dict[int, dict[int, torch.Tensor]] = {}
    for k in range(1, C.K_MAIN + 1):
        uids = [C.unit_id("main", ci, k) for ci in valid_cis]
        h = C.load_store_positions(
            store_dir, "main", uids, [C.POS_CTX_END, C.POS_ANS_MEAN], expect_fingerprint=fp
        )  # (n, 2, R, H)
        ro = h[:, :, RO, :].contiguous()  # (n, 2, 6, H)
        full[k] = {int(ci): ro[i] for i, ci in enumerate(valid_cis)}
        del h
        logger.info("[load] turn %d read-out loaded for %d conversations", k, len(valid_cis))
    return full


def _xy(full: dict, k: int, cis: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """(X, Y) rows-first (6, n, H) fp16 for turn k over the given conversations."""
    stack = torch.stack([full[k][int(ci)] for ci in cis])  # (n, 2, 6, H)
    return stack[:, 0].transpose(0, 1).contiguous(), stack[:, 1].transpose(0, 1).contiguous()


# ── skill / bootstrap helpers (parent artifact recipe) ─────────────────────────


def _fit(full: dict, k: int, cis: np.ndarray) -> dict:
    X, Y = _xy(full, k, cis)
    return fit_rows_batched(X, Y, lambdas=list(RIDGE_LAMBDAS_922), device="cpu")


def _skill_stats(fit: dict, full: dict, k: int, test: np.ndarray, null_ymu: torch.Tensor) -> dict:
    """Per-cell skill/SSE stats: apply ``fit`` (source-map-composite) at turn k."""
    Xt, Yt = _xy(full, k, test)
    pred = predict_from_fit(fit, Xt, device="cpu")
    return _skill_and_stats(pred, Yt.to(torch.float64), null_ymu)


def _matched_skill_stats(
    dec1: dict, lam_rows: torch.Tensor, full: dict, k: int, test: np.ndarray, null_ymu: torch.Tensor
) -> dict:
    """Turn-1 map re-solved at the target turn's per-row lambda, applied at turn k."""
    fit_c = alpha_at(dec1, lam_rows)
    Xt, Yt = _xy(full, k, test)
    pred = predict_from_fit(fit_c, Xt, device="cpu")
    return _skill_and_stats(pred, Yt.to(torch.float64), null_ymu)


def _point_ro_mean(stats: dict) -> float:
    """Read-out-mean pooled skill (the 6 fitted rows are exactly the read-out rows)."""
    return float(stats["skill"].mean())


def boot_ro_mean(sse: np.ndarray, null: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(draws,) read-out-mean skill under paired conversation resamples."""
    return np.stack(
        [
            1.0 - sse[r][idx].sum(1) / np.clip(null[r][idx].sum(1), 1e-30, None)
            for r in range(sse.shape[0])
        ]
    ).mean(0)


def ci95(draws: np.ndarray) -> list[float]:
    return [float(np.quantile(draws, q)) for q in (0.025, 0.975)]


def _shuffle_ro_band(stats: dict) -> dict:
    shuf = _shuffle_draws(stats, C.SHUFFLE_DRAWS, C.SHUFFLE_SEED).numpy()  # (100, 6)
    m = shuf.mean(1)
    return {"p025": float(np.quantile(m, 0.025)), "p975": float(np.quantile(m, 0.975))}


def _perrow_skill_boot(stats: dict, idx_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(skill (6,), boot (draws, 6)) per read-out row under the SAME paired resamples.

    Row order is RO = [15, 18, 20, 21, 25, 27] (blocks 14/17/19/20/24/26).
    """
    sse = stats["sse_unit"].numpy()
    null = stats["null_sse_unit"].numpy()
    br = np.stack(
        [
            1.0 - sse[r][idx_b].sum(1) / np.clip(null[r][idx_b].sum(1), 1e-30, None)
            for r in range(sse.shape[0])
        ],
        axis=1,
    )  # (draws, 6)
    return stats["skill"].numpy(), br


def _blocks(skill: np.ndarray, br: np.ndarray) -> dict:
    """Per-block {skill, ci95} keyed by read-out block (14/17/19/20/24/26)."""
    return {
        C.row_to_block_key(RO[r]): {"skill": float(skill[r]), "ci95": ci95(br[:, r])}
        for r in range(len(RO))
    }


def _deficit_blocks(
    own_sk: np.ndarray, own_br: np.ndarray, sub_sk: np.ndarray, sub_br: np.ndarray
) -> dict:
    """Per-block deficit {skill, ci95} = own − transfer under paired resamples."""
    return {
        C.row_to_block_key(RO[r]): {
            "skill": float(own_sk[r] - sub_sk[r]),
            "ci95": ci95(own_br[:, r] - sub_br[:, r]),
        }
        for r in range(len(RO))
    }


# ── per-regime computation ──────────────────────────────────────────────────


def run_regime(regime: str, full: dict, n_main: int, capture_invalid: set, dup: dict) -> dict:
    """Fit + evaluate turn-1 own skill and 1->{2,3,4} transfers under one regime."""
    invalid = set(capture_invalid)
    if regime != "none":
        invalid |= set(dup[regime])
    split = C.make_split(n_main, n_fit=C.N_FIT, n_val=C.N_VAL, n_test=C.N_TEST, seed=C.SPLIT_SEED)
    fit_m = _filter_invalid(split["fit"], frozenset(invalid))
    test_m = _filter_invalid(split["test"], frozenset(invalid))
    half_a, half_b = C.twin_halves(fit_m)
    logger.info(
        "[regime=%s] fit=%d test=%d halfA=%d halfB=%d (excluded %d dup + %d capture)",
        regime,
        len(fit_m),
        len(test_m),
        len(half_a),
        len(half_b),
        0 if regime == "none" else len(set(dup[regime])),
        len(capture_invalid),
    )

    # per-turn maps: fold-A + full (turns 1..4); fold-B for turn-1 (own_B gate)
    maps_A = {k: _fit(full, k, half_a) for k in range(1, C.K_MAIN + 1)}
    maps_F = {k: _fit(full, k, fit_m) for k in range(1, C.K_MAIN + 1)}
    map1_B = _fit(full, 1, half_b)
    dec1_A = decompose_rows(*_xy(full, 1, half_a))
    dec1_F = decompose_rows(*_xy(full, 1, fit_m))

    idx_b = np.random.default_rng(C.BOOTSTRAP_SEED).integers(
        0, len(test_m), size=(C.BOOTSTRAP_DRAWS, len(test_m))
    )

    def _cell(stats: dict) -> dict:
        b = boot_ro_mean(stats["sse_unit"].numpy(), stats["null_sse_unit"].numpy(), idx_b)
        return {"skill": _point_ro_mean(stats), "ci95": ci95(b), "_boot": b}

    # own-turn-1 skill at the three fit sizes (refit.json cells + per-block sidecar)
    own1_stats = {
        "foldA": _skill_stats(maps_A[1], full, 1, test_m, maps_A[1]["ymu"]),
        "foldB": _skill_stats(map1_B, full, 1, test_m, map1_B["ymu"]),
        "full": _skill_stats(maps_F[1], full, 1, test_m, maps_F[1]["ymu"]),
    }
    own1 = {f: _cell(s) for f, s in own1_stats.items()}
    own1_pr = {f: _blocks(*_perrow_skill_boot(s, idx_b)) for f, s in own1_stats.items()}

    # per-row selected lambda of the turn-1 map (as-fitted), both folds
    lam1 = {
        "foldA": {C.row_to_block_key(r): float(v) for r, v in zip(RO, maps_A[1]["best_lam"])},
        "full": {C.row_to_block_key(r): float(v) for r, v in zip(RO, maps_F[1]["best_lam"])},
    }

    # transfer grid 1->{1,2,3,4} + deficits (as-fitted + matched-lambda), both folds
    grid: dict = {}
    grid_pr: dict = {}
    for fold, mps, dec1 in (("foldA", maps_A, dec1_A), ("full", maps_F, dec1_F)):
        gcells: dict = {}
        gcells_pr: dict = {}
        for k in range(1, C.K_MAIN + 1):
            own_k = _skill_stats(mps[k], full, k, test_m, mps[k]["ymu"])
            xfer = _skill_stats(mps[1], full, k, test_m, mps[k]["ymu"])  # turn-1 map at turn k
            own_c, xfer_c = _cell(own_k), _cell(xfer)
            own_sk, own_br = _perrow_skill_boot(own_k, idx_b)
            xf_sk, xf_br = _perrow_skill_boot(xfer, idx_b)
            cell = {
                "own_skill": own_c["skill"],
                "own_skill_ci95": own_c["ci95"],
                "transfer_skill": xfer_c["skill"],
                "transfer_skill_ci95": xfer_c["ci95"],
                "transfer_shuffle_band": _shuffle_ro_band(xfer),
            }
            cell_pr = {
                "own_skill": _blocks(own_sk, own_br),
                "transfer_skill": _blocks(xf_sk, xf_br),
            }
            if k != 1:
                deficit_as = own_c["_boot"] - xfer_c["_boot"]
                cell["deficit_asfitted"] = own_c["skill"] - xfer_c["skill"]
                cell["deficit_asfitted_ci95"] = ci95(deficit_as)
                # matched-lambda: turn-1 map clamped to target turn k's own selection
                xfer_m = _matched_skill_stats(
                    dec1, mps[k]["best_lam"], full, k, test_m, mps[k]["ymu"]
                )
                xm_c = _cell(xfer_m)
                xm_sk, xm_br = _perrow_skill_boot(xfer_m, idx_b)
                deficit_m = own_c["_boot"] - xm_c["_boot"]
                cell["transfer_skill_matched"] = xm_c["skill"]
                cell["transfer_skill_matched_ci95"] = xm_c["ci95"]
                cell["deficit_matched"] = own_c["skill"] - xm_c["skill"]
                cell["deficit_matched_ci95"] = ci95(deficit_m)
                cell["matched_lambda"] = {
                    C.row_to_block_key(r): float(v) for r, v in zip(RO, mps[k]["best_lam"])
                }
                cell_pr["transfer_skill_matched"] = _blocks(xm_sk, xm_br)
                cell_pr["deficit_asfitted"] = _deficit_blocks(own_sk, own_br, xf_sk, xf_br)
                cell_pr["deficit_matched"] = _deficit_blocks(own_sk, own_br, xm_sk, xm_br)
            gcells[f"1to{k}"] = cell
            gcells_pr[f"1to{k}"] = cell_pr
        grid[fold] = gcells
        grid_pr[fold] = gcells_pr

    for d in (own1["foldA"], own1["foldB"], own1["full"]):
        d.pop("_boot", None)
    refit_dict = {
        "n_fit": len(fit_m),
        "n_test": len(test_m),
        "n_halfA": len(half_a),
        "n_halfB": len(half_b),
        "n_excluded_dup": 0 if regime == "none" else len(set(dup[regime])),
        "own_turn1": own1,
        "turn1_lambda_asfitted": lam1,
        "grid": grid,
    }
    perrow_dict = {
        "n_fit": len(fit_m),
        "n_test": len(test_m),
        "own_turn1": own1_pr,
        "grid": grid_pr,
    }
    return refit_dict, perrow_dict


def validation_gate(none_regime: dict) -> dict:
    """Reproduce committed transfer_matrix.json turn-1 cells within 5e-3 (fp64)."""
    committed = json.loads((OUT / "transfer_matrix.json").read_text())
    g = committed["grid_skill_readout_mean_foldA"]
    checks: dict = {}
    deltas: list[float] = []

    def _chk(name: str, got: float, want: float) -> None:
        d = abs(got - want)
        checks[name] = {"recomputed": got, "committed": want, "abs_delta": d}
        deltas.append(d)

    _chk("own1_foldA_vs_grid_1to1", none_regime["own_turn1"]["foldA"]["skill"], g["1->1"])
    _chk(
        "own1_foldB_vs_own_B_1", none_regime["own_turn1"]["foldB"]["skill"], committed["own_B"]["1"]
    )
    _chk(
        "own1_full_vs_own_full_1",
        none_regime["own_turn1"]["full"]["skill"],
        committed["own_full"]["1"],
    )
    for k in range(1, C.K_MAIN + 1):
        _chk(
            f"xfer_1to{k}_foldA_vs_grid",
            none_regime["grid"]["foldA"][f"1to{k}"]["transfer_skill"],
            g[f"1->{k}"],
        )
        _chk(
            f"own_k{k}_foldA_vs_grid_kk",
            none_regime["grid"]["foldA"][f"1to{k}"]["own_skill"],
            g[f"{k}->{k}"],
        )
    max_delta = max(deltas)
    logger.info("[gate] reproduce-committed max|delta|=%.3e (tol 5e-3)", max_delta)
    assert max_delta < 5e-3, f"validation gate FAILED: max|delta|={max_delta:.3e}"
    return {"max_abs_delta": max_delta, "checks": checks}


def _max_float_diff(a: object, b: object) -> float:
    """Max abs difference over matching float/int leaves of two nested structures."""
    if isinstance(a, dict) and isinstance(b, dict):
        return max((_max_float_diff(a[k], b[k]) for k in a if k in b), default=0.0)
    if isinstance(a, list) and isinstance(b, list):
        return max((_max_float_diff(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=Path("data/issue_958/hf_dl/dup_refit"),
        help="re-downloadable HF staging dir",
    )
    ap.add_argument("--stage-only", action="store_true", help="stage inputs and exit")
    args = ap.parse_args()
    torch.set_num_threads(8)

    counts = stage_inputs(args.stage_root)
    if args.stage_only:
        logger.info("[stage-only] done: %s", counts)
        return 0
    store_dir = args.stage_root / "store"
    corpus_dir = args.stage_root / "corpus"

    fp = C.corpus_fingerprint(corpus_dir)
    n_main = len(C.load_corpus(corpus_dir, "main"))

    # capture-dropout invalid cis: any turn's unit absent from the store index
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

    dup, dup_summary = duplicate_cis(corpus_dir, n_main)
    logger.info(
        "[dup] exact=%d lowercased=%d conversations",
        dup_summary["exact"]["n_dup_conversations"],
        dup_summary["lowercased"]["n_dup_conversations"],
    )

    full = load_readout(store_dir, n_main, fp, valid_cis)

    # cross-check duplicate definition vs committed sidecar (uses the none-regime test fold)
    split = C.make_split(n_main, n_fit=C.N_FIT, n_val=C.N_VAL, n_test=C.N_TEST, seed=C.SPLIT_SEED)
    test_none = _filter_invalid(split["test"], frozenset(capture_invalid))
    dup_gate = cross_check_committed_dups(dup, test_none)
    logger.info("[dup-gate] committed cross-check PASS: %s", json.dumps(dup_gate))

    regimes: dict = {}
    regimes_pr: dict = {}
    for r in REGIMES:
        regimes[r], regimes_pr[r] = run_regime(r, full, n_main, capture_invalid, dup)
    gate = validation_gate(regimes["none"])

    seeds = {
        "split": C.SPLIT_SEED,
        "twin": C.TWIN_SEED,
        "bootstrap": C.BOOTSTRAP_SEED,
        "bootstrap_draws": C.BOOTSTRAP_DRAWS,
        "shuffle": C.SHUFFLE_SEED,
        "shuffle_draws": C.SHUFFLE_DRAWS,
    }
    res = {
        "definition": (
            "main-panel turn-1 context->answer ridge map refit with exact-duplicate "
            "first-message conversations excluded from fit AND test folds; parent "
            "recipe verbatim otherwise (GCV dual/Gram ridge, 6 read-out-row mean, "
            "source-map-composite transfer). Reported as-fitted (refit GCV lambda) "
            "AND matched-lambda (turn-1 map clamped to the target turn's own selection)."
        ),
        "corpus_fingerprint": fp,
        "readout_blocks": C.READOUT_BLOCKS,
        "n_main": n_main,
        "n_capture_invalid": len(capture_invalid),
        "duplicate_summary": dup_summary,
        "duplicate_committed_cross_check": dup_gate,
        "validation_gate": gate,
        "seeds": seeds,
        "regimes": regimes,
        "metadata": C.reproducibility_metadata({"script": "issue958_dup_excluded_refit"}),
    }
    SUB.mkdir(parents=True, exist_ok=True)

    # refit.json is the committed clean-result-pinned aggregate — do NOT mutate it.
    # On a re-run it already exists: load it, and assert the freshly-recomputed
    # readout-MEAN cells reproduce it to 1e-6 (deterministic ⇒ bit-identical),
    # then leave the file untouched. Only a first run (absent file) writes it.
    refit_path = SUB / "refit.json"
    mean_crosscheck: dict = {"tol": 1e-6}
    if refit_path.exists():
        committed = json.loads(refit_path.read_text())
        delta = _max_float_diff(regimes, committed.get("regimes", {}))
        mean_crosscheck.update({"committed_refit_present": True, "max_abs_delta": delta})
        logger.info("[mean-crosscheck] fresh readout-mean vs committed refit.json: %.3e", delta)
        assert delta < 1e-6, f"readout-mean crosscheck vs committed refit.json FAILED: {delta:.3e}"
    else:
        C.write_json_atomic(refit_path, res)
        mean_crosscheck.update({"committed_refit_present": False, "max_abs_delta": 0.0})
        logger.info("wrote %s (first run)", refit_path)

    perrow_res = {
        "definition": (
            "per-read-out-row (blocks 14/17/19/20/24/26) held-out skill + paired-bootstrap "
            "95% CI for the duplicate-excluded refit cells — companion to refit.json (which "
            "stores only the 6-block read-out MEAN). Block 19 is the parent-line frozen best "
            "layer. Rows in RO order [15,18,20,21,25,27]; keyed by read-out block."
        ),
        "corpus_fingerprint": fp,
        "readout_blocks": C.READOUT_BLOCKS,
        "n_main": n_main,
        "duplicate_summary": dup_summary,
        "readout_mean_crosscheck_vs_committed_refit_json": mean_crosscheck,
        "validation_gate": gate,
        "seeds": seeds,
        "regimes": regimes_pr,
        "metadata": C.reproducibility_metadata(
            {"script": "issue958_dup_excluded_refit", "sidecar": "refit_perrow"}
        ),
    }
    C.write_json_atomic(SUB / "refit_perrow.json", perrow_res)
    logger.info("wrote %s", SUB / "refit_perrow.json")

    # one-line headline log at block 19 (the best layer), exact regime
    exq = regimes_pr["exact"]
    gA = exq["grid"]["foldA"]
    logger.info(
        "[headline exact @block19] own1 foldA %.4f full %.4f | 1to{2,3,4} transfer foldA %s",
        exq["own_turn1"]["foldA"]["19"]["skill"],
        exq["own_turn1"]["full"]["19"]["skill"],
        [round(gA[f"1to{k}"]["transfer_skill"]["19"]["skill"], 4) for k in KS],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
