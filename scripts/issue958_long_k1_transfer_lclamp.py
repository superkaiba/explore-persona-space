"""Issue #958 free-analysis follow-up: λ-CLAMPED long-panel turn-1 refit at turns 5-8.

The r2 forward-transfer read (`long_k1_transfer.json`) applied the turn-1 long-panel
map AS FITTED (per-row GCV λ ≈ 5.1 / 3.2) to turns 5-8, whose own-turn maps all
GCV-selected λ = 1,000 (the grid max) — a shrinkage-scale confound the H1 title
carries as "GCV-selected (λ-mismatched)". This driver removes it: refit the turn-1
map on the SAME long fit fold with per-row λ CLAMPED to the target turn's own-map
selection (read per row from the committed `maps_meta.json`, never hardcoded), then
re-evaluate raw (source-map-composite) + recalibrated transfer at turns 5-8 against
the same frozen read-out rows and test conversations.

Validation gates (the #931 reproduce-committed-rows recipe):
- refit at the turn-1 map's ORIGINAL per-row λ reproduces the committed
  `long_own_k1` read-out-row skills (fp64-exact math ⇒ tolerance 5e-3, expect ≪);
- the refit's train-fold moments (mu/sd/ymu) match the saved `long_k1_own.pt`
  fp32 moments — a design-matrix identity check on the fold reconstruction.

Inputs are staged from HF per file (scoped ``list_repo_tree`` + ``hf_hub_download``;
NEVER full-tree ``snapshot_download`` on the ~1M-file data repo) into a
re-downloadable scratch dir the caller deletes after a successful run.

Writes eval_results/issue_958/long_k1_transfer_lclamp.json +
percell/long_1to{k}_lclamp.npz (k = 5..8).
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
from issue958_fit_maps import _shuffle_draws, _skill_and_stats, predict_from_fit  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue958_lclamp")

OUT = Path("eval_results/issue_958")
RO = [C.block_to_row(b) for b in C.READOUT_BLOCKS]  # [15, 18, 20, 21, 25, 27]
KS = [5, 6, 7, 8]
MAP_CELLS = ["long_k1_own", *(f"long_k{k}_own" for k in KS)]


# ── HF staging (scoped; verify-before-skip; caller deletes the dir after) ─────


def stage_inputs(stage_root: Path, max_workers: int = 6) -> dict:
    """Stage store/long shards + the 5 long own maps + long corpus files.

    Scoped ``list_repo_tree`` per prefix + per-file ``hf_hub_download`` with the
    transient-only retry policy from ``issue958_common`` (4xx stays loud).
    Already-staged size-verified files are skipped, so a timed-out staging run
    resumes instead of re-downloading. Returns per-prefix file counts.
    """
    import shutil
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi, hf_hub_download

    pfx = C.HF_OUT_PREFIX
    api = HfApi()
    want_maps = {f"{c}.pt" for c in MAP_CELLS}
    targets = {
        f"{pfx}/analysis_tensors/store/long": (
            stage_root / "store" / "long",
            None,
        ),
        f"{pfx}/analysis_tensors/maps": (
            stage_root / "maps",
            lambda name: name in want_maps,
        ),
        f"{pfx}/corpus": (
            stage_root / "corpus",
            lambda name: name in {"long.json", "manifest.json"},
        ),
    }
    counts: dict[str, int] = {}
    for remote_prefix, (local_root, name_filter) in targets.items():
        entries = [
            e
            for e in api.list_repo_tree(
                C.HF_DATA_REPO, path_in_repo=remote_prefix, repo_type="dataset", recursive=True
            )
            if getattr(e, "size", None) is not None
            and (name_filter is None or name_filter(Path(e.path).name))
        ]
        assert entries, f"HF staging: nothing under {C.HF_DATA_REPO}/{remote_prefix}"
        local_root.mkdir(parents=True, exist_ok=True)
        to_fetch = []
        for e in entries:
            dst = local_root / Path(e.path).relative_to(remote_prefix)
            if C._staged_ok(dst, e.size):
                continue
            to_fetch.append(e)
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
            with tempfile.TemporaryDirectory(prefix="i958_lclamp_", dir=str(local_root)) as td:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    list(ex.map(lambda p: _fetch(p, td), [e.path for e in to_fetch]))
                for e in to_fetch:
                    src = Path(td) / e.path
                    dst = local_root / Path(e.path).relative_to(remote_prefix)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        dst.unlink()
                    shutil.move(str(src), str(dst))
        counts[remote_prefix] = len(entries)
    return counts


# ── batched fixed-λ dual ridge (shared eigh; per-λ alpha is one batched op) ───


def decompose_rows(X: torch.Tensor, Y: torch.Tensor) -> dict:
    """Shared per-row decomposition of the train design (λ-independent part).

    Same stacked-eigh dual-ridge math as ``issue958_fit_maps.fit_rows_batched``
    (X standardized / Y centered on the train fold, Gram eigh per row, batched
    over rows), WITHOUT the GCV λ selection — the λ is supplied later via
    :func:`alpha_at`. fp64 CPU throughout. Returns mu/sd/ymu/Xn/s/Q/G.
    """
    Xd = X.to(torch.float64)
    Yd = Y.to(torch.float64)
    mu = Xd.mean(1, keepdim=True)
    sd = Xd.std(1, correction=0, keepdim=True) + 1e-9  # the #658/#841 convention
    Xn = (Xd - mu) / sd
    ymu = Yd.mean(1, keepdim=True)
    Yc = Yd - ymu
    K = Xn @ Xn.transpose(1, 2)  # (r, n, n)
    s, Q = torch.linalg.eigh(K)
    s = torch.clamp(s, min=0.0)
    G = Q.transpose(1, 2) @ Yc  # (r, n, p)
    return {
        "mu": mu.squeeze(1),
        "sd": sd.squeeze(1),
        "ymu": ymu.squeeze(1),
        "Xn": Xn,
        "s": s,
        "Q": Q,
        "G": G,
    }


def alpha_at(dec: dict, lam_rows: torch.Tensor) -> dict:
    """Dual coefficients at a FIXED per-row λ → a ``predict_from_fit``-shaped dict.

    ``lam_rows`` (rows,) fp64. One batched op over the shared decomposition —
    no re-factorization per λ (the vectorize-first shape for the 5 λ settings).
    """
    f_best = 1.0 / (dec["s"] + lam_rows.unsqueeze(1))  # (r, n)
    alpha = dec["Q"] @ (f_best.unsqueeze(-1) * dec["G"])  # (r, n, p)
    return {
        "mu": dec["mu"],
        "sd": dec["sd"],
        "ymu": dec["ymu"],
        "Xn": dec["Xn"],
        "alpha": alpha,
    }


# ── helpers ───────────────────────────────────────────────────────────────────


def load_readout_xy(
    store_dir: Path, cis: np.ndarray, ks: list[int], fp: str
) -> dict[int, dict[str, torch.Tensor]]:
    """{k: {"X": (6, n, H), "Y": (6, n, H)}} fp16 read-out-row stacks per turn.

    One ``load_store_positions`` gather per turn set (shards loaded one blob at
    a time inside the helper); rows sliced to the frozen read-out rows before
    stacking so the resident footprint stays ~(n, 6, H).
    """
    out: dict[int, dict[str, torch.Tensor]] = {}
    for k in ks:
        uids = [C.unit_id("long", int(ci), k) for ci in cis]
        h = C.load_store_positions(
            store_dir, "long", uids, [C.POS_CTX_END, C.POS_ANS_MEAN], expect_fingerprint=fp
        )  # (n, 2, R, H)
        out[k] = {
            "X": h[:, 0][:, RO].transpose(0, 1).contiguous(),  # (6, n, H)
            "Y": h[:, 1][:, RO].transpose(0, 1).contiguous(),
        }
        del h
    return out


def saved_map_moments(maps_dir: Path, cell: str) -> dict:
    """Stacked fp64 (6, H) mu/sd/ymu of a persisted map's read-out rows."""
    blob = torch.load(maps_dir / f"{cell}.pt", weights_only=False, map_location="cpu")
    assert blob["policy"] == C.TRANSFER_STANDARDIZATION_POLICY, blob["policy"]
    rows = blob["rows"]
    return {f: torch.stack([rows[r][f].to(torch.float64) for r in RO]) for f in ("mu", "sd", "ymu")}


def boot_readout_mean(sse: np.ndarray, null: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(draws,) read-out-mean skill under paired conversation resamples.

    ``sse``/``null`` (6, n); ``idx`` (draws, n) — one fancy-indexed gather per
    row, no per-draw loop (the parent artifact's bootstrap recipe).
    """
    return np.stack(
        [
            1.0 - sse[r][idx].sum(1) / np.clip(null[r][idx].sum(1), 1e-30, None)
            for r in range(sse.shape[0])
        ]
    ).mean(0)


def point_skill(sse: np.ndarray, null: np.ndarray, keep: np.ndarray | None = None) -> float:
    """Read-out-mean pooled skill, optionally restricted to a kept-unit mask."""
    m = slice(None) if keep is None else keep
    return float(
        np.mean([1.0 - sse[r][m].sum() / max(null[r][m].sum(), 1e-30) for r in range(sse.shape[0])])
    )


def ci95(draws: np.ndarray) -> list[float]:
    """[p2.5, p97.5] of a bootstrap draw vector (the parent artifact recipe)."""
    return [float(np.quantile(draws, q)) for q in (0.025, 0.975)]


def dup_masks(corpus_dir: Path, test_idx: np.ndarray, fit_set: set[int]) -> tuple[dict, dict]:
    """First-message duplicate-group masks over the LONG corpus, both normalizations.

    Matches the r2 sidecar derivation (`duplicate_first_message_groups.json`):
    group key = the conversation's first user message, under exact-string and
    lowercased equality; a test conversation is masked iff its key appears >1
    time in the long corpus. Also records whether a masked test conversation
    has a FIT-fold partner (the memorization-relevant subcase).
    """
    convs = C.load_corpus(corpus_dir, "long")
    first = [c["exchanges"][0]["user"] for c in convs]
    masks: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}
    for name, keyfn in (("exact", lambda m: m), ("lowercased", lambda m: m.lower())):
        groups: dict = collections.defaultdict(list)
        for i, msg in enumerate(first):
            groups[keyfn(msg)].append(i)
        dup = {k: v for k, v in groups.items() if len(v) > 1}
        dupset = {i for v in dup.values() for i in v}
        mask = np.array([int(ci) in dupset for ci in test_idx])
        masks[name] = mask
        with_fit_partner = [
            int(ci)
            for ci in test_idx[mask]
            if any(j != int(ci) and j in fit_set for j in dup.get(keyfn(first[int(ci)]), []))
        ]
        summary[name] = {
            "n_dup_conversations": sum(len(v) for v in dup.values()),
            "n_dup_groups": len(dup),
            "n_test_conversations_in_dup_groups": int(mask.sum()),
            "test_ci_in_dup_groups": [int(ci) for ci in test_idx[mask]],
            "test_ci_with_fit_fold_partner": with_fit_partner,
        }
    return masks, summary


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    """Stage inputs, gate the refit against committed artifacts, write outputs."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=Path("data/issue_958/hf_dl/lclamp"),
        help="re-downloadable HF staging dir (caller deletes after a clean run)",
    )
    ap.add_argument("--stage-only", action="store_true", help="stage inputs and exit")
    args = ap.parse_args()
    torch.set_num_threads(8)

    counts = stage_inputs(args.stage_root)
    if args.stage_only:
        logger.info("[stage-only] done: %s", counts)
        return 0
    store_dir = args.stage_root / "store"
    maps_dir = args.stage_root / "maps"
    corpus_dir = args.stage_root / "corpus"

    # committed inputs (this repo, r2 artifacts)
    maps_meta = json.loads((OUT / "maps_meta.json").read_text())
    parent = json.loads((OUT / "long_k1_transfer.json").read_text())
    own = {k: np.load(OUT / "percell" / f"long_own_k{k}.npz") for k in [1, *KS]}
    test_idx = own[5]["test_idx"]
    for k in [1, *KS]:
        assert np.array_equal(own[k]["test_idx"], test_idx)

    # corpus fingerprint: staged manifest must match the committed fit regime
    fp = C.corpus_fingerprint(corpus_dir)
    assert fp == maps_meta["corpus_fingerprint"], (
        f"staged corpus fingerprint {fp[:12]}… != maps_meta "
        f"{maps_meta['corpus_fingerprint'][:12]}… — wrong/stale staging"
    )

    # reconstruct the long fit fold exactly as issue958_fit_maps.build_design
    n_long = len(C.load_corpus(corpus_dir, "long"))
    split_l = C.make_split(
        n_long, n_fit=C.LONG_FIT, n_val=C.LONG_VAL, n_test=C.LONG_TEST, seed=C.SPLIT_SEED
    )
    idx = C.load_store_index(store_dir, "long", expect_fingerprint=fp)
    invalid = sorted(
        {
            ci
            for ci in range(n_long)
            for k in range(1, C.K_LONG + 1)
            if C.unit_id("long", ci, k) not in idx
        }
    )
    inv = frozenset(invalid)
    fit_l = np.sort(np.asarray([ci for ci in split_l["fit"] if ci not in inv], dtype=np.int64))
    test_l = np.sort(np.asarray([ci for ci in split_l["test"] if ci not in inv], dtype=np.int64))
    n_fit_meta = maps_meta["cells"]["long_k1_own"]["n_fit"]
    assert len(fit_l) == n_fit_meta, (len(fit_l), n_fit_meta)
    assert np.array_equal(test_l, np.sort(test_idx)), "reconstructed test fold != committed"

    # per-row λ: turn-1 original + the per-turn clamp targets, from maps_meta
    def lam_rows(cell: str) -> torch.Tensor:
        bl = maps_meta["cells"][cell]["best_lam"]
        return torch.tensor([bl[C.row_to_block_key(r)] for r in RO], dtype=torch.float64)

    lam_k1 = lam_rows("long_k1_own")
    lam_clamp = {k: lam_rows(f"long_k{k}_own") for k in KS}

    # design tensors: turn-1 fit fold + test conversations at turns 1 and 5..8
    fit_xy = load_readout_xy(store_dir, fit_l, [1], fp)[1]
    test_xy = load_readout_xy(store_dir, test_idx, [1, *KS], fp)
    dec = decompose_rows(fit_xy["X"], fit_xy["Y"])

    # ── validation gate 1: moments match the saved long_k1_own map (fp32) ────
    saved_k1 = saved_map_moments(maps_dir, "long_k1_own")
    mom_rel = max(
        float(((dec[f] - saved_k1[f]).abs() / (saved_k1[f].abs() + 1.0)).max())
        for f in ("mu", "sd", "ymu")
    )
    logger.info("[gate] refit train-fold moments vs saved map: max rel |Δ|=%.2e", mom_rel)
    assert mom_rel < 5e-4, f"design-matrix reconstruction FAILED: moments rel Δ {mom_rel:.3e}"

    # ── validation gate 2: refit at ORIGINAL λ reproduces committed own_k1 ───
    fit_orig = alpha_at(dec, lam_k1)
    pred1 = predict_from_fit(fit_orig, test_xy[1]["X"], device="cpu")
    st1 = _skill_and_stats(pred1, test_xy[1]["Y"].to(torch.float64), dec["ymu"])
    committed = np.array([own[1]["skill"][r] for r in RO])
    gate_dmax = float(np.abs(st1["skill"].numpy() - committed).max())
    logger.info("[gate] refit@original-λ vs committed long_own_k1: max|Δskill|=%.2e", gate_dmax)
    assert gate_dmax < 5e-3, f"validation gate FAILED: {gate_dmax}"

    # duplicate-group masks (long corpus, both normalizations)
    masks, dup_summary = dup_masks(corpus_dir, test_idx, set(int(c) for c in fit_l))

    idx_b = np.random.default_rng(C.BOOTSTRAP_SEED).integers(
        0, len(test_idx), size=(C.BOOTSTRAP_DRAWS, len(test_idx))
    )
    res: dict = {
        "policy": C.TRANSFER_STANDARDIZATION_POLICY,
        "clamp": {
            "definition": (
                "turn-1 long-panel map REFIT on the same fit fold with per-row ridge λ "
                "clamped to the target turn's own-map GCV selection (maps_meta.json), "
                "removing the λ≈5-vs-1000 shrinkage-scale confound of the as-fitted read"
            ),
            "turn1_original_lambda": {
                C.row_to_block_key(r): float(v) for r, v in zip(RO, lam_k1, strict=True)
            },
            "per_turn_clamp_lambda": {
                f"k{k}": {
                    C.row_to_block_key(r): float(v) for r, v in zip(RO, lam_clamp[k], strict=True)
                }
                for k in KS
            },
        },
        "validation_gate": {
            "max_abs_row_delta_vs_committed_long_own_k1": gate_dmax,
            "moments_max_rel_delta_vs_saved_long_k1_own": mom_rel,
        },
        "duplicates": {
            "grouping_key": f"first user message of the conversation (long corpus, n={n_long})",
            "sidecar": "eval_results/issue_958/duplicate_first_message_groups.json (definition)",
            "per_normalization": dup_summary,
        },
        "gcv_selected_reference": {
            cell: {
                f: parent["cells"][cell][f]
                for f in ("transfer_skill", "recalibrated_transfer_skill", "own_skill")
            }
            for cell in parent["cells"]
        },
        "cells": {},
    }

    (OUT / "percell").mkdir(parents=True, exist_ok=True)
    for k in KS:
        fit_c = alpha_at(dec, lam_clamp[k])
        mom_k = saved_map_moments(maps_dir, f"long_k{k}_own")
        Y_t = test_xy[k]["Y"].to(torch.float64)
        pred = predict_from_fit(fit_c, test_xy[k]["X"], device="cpu")
        st = _skill_and_stats(pred, Y_t, mom_k["ymu"])
        pred_rc = predict_from_fit(fit_c, test_xy[k]["X"], device="cpu", moments=mom_k)
        st_rc = _skill_and_stats(pred_rc, Y_t, mom_k["ymu"])
        shuf = _shuffle_draws(st, C.SHUFFLE_DRAWS, C.SHUFFLE_SEED).numpy()  # (100, 6)

        sse, null = st["sse_unit"].numpy(), st["null_sse_unit"].numpy()
        sse_rc = st_rc["sse_unit"].numpy()
        own_sse = own[k]["sse_unit"][RO]
        own_null = own[k]["null_sse_unit"][RO]
        xfer_b = boot_readout_mean(sse, null, idx_b)
        rc_b = boot_readout_mean(sse_rc, null, idx_b)
        own_b = boot_readout_mean(own_sse, own_null, idx_b)
        own_p = float(np.mean([own[k]["skill"][r] for r in RO]))
        cell: dict = {
            "transfer_skill": point_skill(sse, null),
            "transfer_skill_ci95": ci95(xfer_b),
            "recalibrated_transfer_skill": point_skill(sse_rc, null),
            "recalibrated_transfer_skill_ci95": ci95(rc_b),
            "own_skill": own_p,
            "deficit": own_p - point_skill(sse, null),
            "deficit_ci95": ci95(own_b - xfer_b),
            "shuffle_band": {
                "readout_mean_p975": float(np.quantile(shuf.mean(1), 0.975)),
                "readout_mean_p025": float(np.quantile(shuf.mean(1), 0.025)),
            },
            "n_test": len(test_idx),
            "lambda_used": res["clamp"]["per_turn_clamp_lambda"][f"k{k}"],
            "excl_dup": {},
        }
        for name, mask in masks.items():
            keep = ~mask
            n_keep = int(keep.sum())
            idx_k = np.random.default_rng(C.BOOTSTRAP_SEED).integers(
                0, n_keep, size=(C.BOOTSTRAP_DRAWS, n_keep)
            )
            xb = boot_readout_mean(sse[:, keep], null[:, keep], idx_k)
            rb = boot_readout_mean(sse_rc[:, keep], null[:, keep], idx_k)
            ob = boot_readout_mean(own_sse[:, keep], own_null[:, keep], idx_k)
            op = point_skill(own_sse, own_null, keep)
            cell["excl_dup"][name] = {
                "n_test_excl": n_keep,
                "transfer_skill": point_skill(sse, null, keep),
                "transfer_skill_ci95": ci95(xb),
                "recalibrated_transfer_skill": point_skill(sse_rc, null, keep),
                "recalibrated_transfer_skill_ci95": ci95(rb),
                "own_skill": op,
                "deficit": op - point_skill(sse, null, keep),
                "deficit_ci95": ci95(ob - xb),
            }
        res["cells"][f"long_1to{k}"] = cell
        np.savez(
            OUT / "percell" / f"long_1to{k}_lclamp.npz",
            skill=st["skill"].numpy().astype(np.float64),
            sse_unit=sse.astype(np.float32),
            null_sse_unit=null.astype(np.float32),
            recal_skill=st_rc["skill"].numpy().astype(np.float64),
            recal_sse_unit=sse_rc.astype(np.float32),
            shuffle_draws=shuf.astype(np.float64),
            test_idx=test_idx,
            readout_rows=np.array(RO),
            lam_used=lam_clamp[k].numpy(),
        )
        logger.info(
            "long_1to%d %s",
            k,
            json.dumps(
                {f: cell[f] for f in ("transfer_skill", "recalibrated_transfer_skill", "own_skill")}
            ),
        )

    res["metadata"] = C.reproducibility_metadata({"script": "issue958_long_k1_transfer_lclamp"})
    C.write_json_atomic(OUT / "long_k1_transfer_lclamp.json", res)
    logger.info("wrote %s", OUT / "long_k1_transfer_lclamp.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
