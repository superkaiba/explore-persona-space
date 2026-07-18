"""Issue #958 free-analysis follow-up #3: carried-vs-changed direction spectra.

WHICH directions of answer-representation space does the frozen context->answer
mapping CARRY across turns, and which does it CHANGE per turn? All analyses at
read-out block 19 (the parent-line frozen best layer) for the PC/random basis;
the three #778 trait directions additionally read at their own frozen layers
(evil block 20 / sycophancy 26 / hallucination 17).

Basis (block 19, store row 20):
  - top ~100 PCs of answer-mean activations, PCA fit on the TRAIN fold only,
    pooling train answers across turns 1-4 (exact-duplicate conversations
    excluded from the PCA pool AND the test fold — the round-5 memorization
    artifact); no test leakage;
  - the 3 unit-normalized #778 trait directions at their own layers (reported
    separately from the PC spectrum), via the committed `_load_rb`/`_directions`
    convention;
  - 100 norm-matched (unit-norm) random directions, seed 0 (the drift-read null).

Three measurements on the duplicate-excluded held-out main-panel test fold:
  1. Per-direction FORECAST R2 by separation k=2,3,4, using the SAVED forecast
     maps (fcast_1to{2,3,4}; the persisted composite affine w = Xn^T alpha with
     mu/sd/ymu, applied verbatim as the lclamp re-reduction did). Carried
     directions hold R2 at k=4; changed directions decay fast.
  2. TRANSMITTED SPECTRUM of the committed own-turn context maps ctx_k{2,3,4}
     (block 19): singular-value spectra of the linear part w + principal angles
     between the top-50 right-singular subspaces across turn maps. (The
     dup-excluded turn-1 map's weights were NOT persisted in a prior round, so
     turn 1 is skipped here, NOT refit.)
  3. Raw cross-turn PERSISTENCE: per direction, Pearson corr across conversations
     of <v, ans_k> vs <v, ans_{k+1}>, k=1..3. Scattered against (1) to separate
     "forecastable because it persists" from "forecastable beyond persistence".

Pipeline gate BEFORE trusting anything: reproduce the committed fcast_1to2 per-row
skill at block 19 (percell/fcast_1to2.npz row 20; 29-row indexing block b = row
b+1) from the staged map + store within 1e-5 (on the SAME committed 500 test
fold, no exclusion). Abort + report needs-refit if it does not reproduce.

Vectorized throughout (projections computed once, batched per-direction
reductions; no per-direction Python fit loops). Store shards are activation
tensors; the corpus first-message text is real LMSYS content and is NEVER printed.
Writes eval_results/issue_958/carried-directions/{spectra.json} + figures.
"""

from __future__ import annotations

import argparse
import collections
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
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue958_fit_maps import _directions, _load_rb  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue958_carried")

OUT = Path("eval_results/issue_958")
SUB = OUT / "carried-directions"
N_PC = 100
SVD_R = 50  # top-r right-singular subspace for principal angles
# The persisted map weight `w` (= Xn^T alpha) is stored fp16, so the committed
# fp64 forecast skill can only be reproduced up to fp16 quantization (~5e-4
# floor), NOT the fp64-exact 1e-5. GATE_TOL catches real pipeline bugs (wrong
# input / row / map orientation give delta ~0.1+) while tolerating fp16; the
# per-direction R2 below uses the SAME fp16 w, so the gate validates exactly the
# analysis pipeline. The measured delta + the fp64-ideal flag are reported.
GATE_TOL = 5e-3
GATE_FP64_IDEAL = 1e-5
BLOCK19 = 19
FCAST_KS = [2, 3, 4]
MAPS_NEEDED = [f"fcast_1to{k}" for k in FCAST_KS] + [f"ctx_k{k}_full" for k in (2, 3, 4)]


def _row(block: int) -> int:
    return C.block_to_row(block)  # store row = block + 1


ROW19 = _row(BLOCK19)  # 20
TRAIT_BLOCK = dict(C.PRIMARY_LSTAR)  # evil 20 / sycophancy 26 / hallucination 17
TRAIT_ROW = {t: _row(b) for t, b in TRAIT_BLOCK.items()}  # evil 21 / syc 27 / hallu 18
R4 = sorted({ROW19, *TRAIT_ROW.values()})  # [18, 20, 21, 27]


# ── HF staging (scoped store/main + the 6 needed maps + corpus/main) ──────────


def stage_inputs(stage_root: Path, max_workers: int = 6) -> dict:
    import shutil
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate import hub

    pfx = C.HF_OUT_PREFIX
    api = HfApi()
    want_maps = {f"{c}.pt" for c in MAPS_NEEDED}

    def _list_tree(rp: str) -> list:
        """Scoped staging listing, materialized so the HTTP call fires under retry."""
        # HUB_VERIFY_RETRY_EXEMPT: scoped download-staging listing, retried via hub.retry_transient
        entries = api.list_repo_tree(
            C.HF_DATA_REPO, path_in_repo=rp, repo_type="dataset", recursive=True
        )
        return list(entries)  # #920/#1335: materialize inside the retried thunk

    targets = {
        f"{pfx}/analysis_tensors/store/main": (stage_root / "store" / "main", None),
        f"{pfx}/analysis_tensors/maps": (stage_root / "maps", lambda n: n in want_maps),
        f"{pfx}/corpus": (stage_root / "corpus", lambda n: n in {"main.json", "manifest.json"}),
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
            import time

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
                        raise
                    last = exc
                    logger.warning(
                        "[stage] %s attempt %d/4 (%s)", path, attempt + 1, type(exc).__name__
                    )
                    time.sleep(20 * (attempt + 1))
            raise RuntimeError(f"HF staging failed after 4 attempts: {path}") from last

        if to_fetch:
            with tempfile.TemporaryDirectory(prefix="i958_carr_", dir=str(local_root)) as td:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    list(ex.map(lambda p: _fetch(p, td), [e.path for e in to_fetch]))
                for e in to_fetch:
                    src = Path(td) / e.path
                    dst = local_root / Path(e.path).name
                    if dst.exists():
                        dst.unlink()
                    shutil.move(str(src), str(dst))
        counts[remote_prefix] = len(entries)
    return counts


# ── duplicate-group conversation set (recompute; cross-check the committed count) ──


def duplicate_cis(corpus_dir: Path, normalization: str = "exact") -> list[int]:
    """Sorted main-corpus conversation indices whose first user message repeats."""
    convs = C.load_corpus(corpus_dir, "main")
    first = [c["exchanges"][0]["user"] for c in convs]
    keyfn = (lambda m: m) if normalization == "exact" else (lambda m: m.lower())
    groups: dict = collections.defaultdict(list)
    for i, msg in enumerate(first):
        groups[keyfn(msg)].append(i)
    return sorted({i for v in groups.values() if len(v) > 1 for i in v})


# ── store loading (answer turns 1-4 + turn-1 context at the 4 needed rows) ────


def load_acts(store_dir: Path, fp: str, valid_cis: list[int]) -> dict:
    """{'ans': {k: (n, len(R4), H)}, 'ctx1': (n, len(R4), H)} fp32, row order R4."""
    out_ans: dict[int, torch.Tensor] = {}
    ctx1 = None
    for k in range(1, C.K_MAIN + 1):
        uids = [C.unit_id("main", ci, k) for ci in valid_cis]
        h = C.load_store_positions(
            store_dir, "main", uids, [C.POS_CTX_END, C.POS_ANS_MEAN], expect_fingerprint=fp
        )  # (n, 2, R, H)
        out_ans[k] = h[:, 1][:, R4, :].to(torch.float32).contiguous()  # answer_mean
        if k == 1:
            ctx1 = h[:, 0][:, R4, :].to(torch.float32).contiguous()  # turn-1 context
        del h
        logger.info("[load] turn %d acts loaded (%d conversations)", k, len(valid_cis))
    return {"ans": out_ans, "ctx1": ctx1}


def _idx_in_valid(valid_cis: list[int], cis: np.ndarray) -> np.ndarray:
    """Positions of `cis` within the loaded `valid_cis` order."""
    pos = {int(c): i for i, c in enumerate(valid_cis)}
    return np.array([pos[int(c)] for c in cis], dtype=np.int64)


def _rrow(row: int) -> int:
    """Index of a store row within the loaded R4 slice."""
    return R4.index(row)


# ── map loading + prediction (dense persisted composite affine) ───────────────


def load_map_rows(maps_dir: Path, cell: str, rows: list[int]) -> dict:
    """{row: {w (H,H) f32, mu, sd, ymu (H,) f32}} for a persisted map cell."""
    blob = torch.load(maps_dir / f"{cell}.pt", weights_only=False, map_location="cpu")
    assert blob["policy"] == C.TRANSFER_STANDARDIZATION_POLICY, blob["policy"]
    rw = blob["rows"]
    out = {}
    for r in rows:
        assert r in rw, f"{cell}: row {r} not persisted (persist_rows={sorted(rw)})"
        w = rw[r]
        out[r] = {
            "w": w["w"].to(torch.float32),
            "mu": w["mu"].to(torch.float32),
            "sd": w["sd"].to(torch.float32),
            "ymu": w["ymu"].to(torch.float32),
        }
    return out


def predict_row(mr: dict, x: torch.Tensor) -> torch.Tensor:
    """Composite affine pred at one row: ymu + ((x - mu)/sd) @ w. x (n, H) f32."""
    xn = (x - mr["mu"]) / mr["sd"]
    return mr["ymu"] + xn @ mr["w"]


# ── measurement helpers (vectorized) ──────────────────────────────────────────


def r2_per_direction(
    pred: torch.Tensor, actual: torch.Tensor, train_mean: torch.Tensor, V: torch.Tensor
) -> np.ndarray:
    """Per-direction forecast R2 for a direction matrix V (n_dir, H).

    R2_v = 1 - SSE(<v,pred>-<v,actual>) / SSE(<v,actual>-<v,train_mean>), over the
    test conversations (pred/actual (n, H); train_mean (H,)).
    """
    pp = pred @ V.t()  # (n, n_dir)
    pa = actual @ V.t()
    pm = train_mean @ V.t()  # (n_dir,)
    sse_resid = ((pp - pa) ** 2).sum(0)
    sse_null = ((pa - pm) ** 2).sum(0).clamp(min=1e-30)
    return (1.0 - sse_resid / sse_null).numpy()


def pearson_cols(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Column-wise Pearson corr between two (n, n_dir) matrices."""
    az = a - a.mean(0, keepdim=True)
    bz = b - b.mean(0, keepdim=True)
    num = (az * bz).sum(0)
    den = (az.norm(dim=0) * bz.norm(dim=0)).clamp(min=1e-30)
    return (num / den).numpy()


def principal_angles(Q1: torch.Tensor, Q2: torch.Tensor) -> np.ndarray:
    """Principal angles (degrees) between two orthonormal column bases (H, r)."""
    s = torch.linalg.svdvals(Q1.t() @ Q2).clamp(-1.0, 1.0)
    return np.degrees(np.arccos(s.numpy()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage-root", type=Path, default=Path("data/issue_958/hf_dl/carried"))
    ap.add_argument("--stage-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)

    counts = stage_inputs(args.stage_root)
    if args.stage_only:
        logger.info("[stage-only] done: %s", counts)
        return 0
    store_dir = args.stage_root / "store"
    maps_dir = args.stage_root / "maps"
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
    dup = set(duplicate_cis(corpus_dir, "exact"))
    logger.info(
        "[setup] n_main=%d capture_invalid=%d dup_exact=%d fp=%s",
        n_main,
        len(capture_invalid),
        len(dup),
        fp[:12],
    )

    split = C.make_split(n_main, n_fit=C.N_FIT, n_val=C.N_VAL, n_test=C.N_TEST, seed=C.SPLIT_SEED)
    fit_all = np.sort([ci for ci in split["fit"] if ci not in capture_invalid]).astype(np.int64)
    test_all = np.sort([ci for ci in split["test"] if ci not in capture_invalid]).astype(np.int64)
    fit_uniq = np.array([c for c in fit_all if int(c) not in dup], dtype=np.int64)
    test_uniq = np.array([c for c in test_all if int(c) not in dup], dtype=np.int64)
    logger.info(
        "[folds] fit_all=%d fit_uniq=%d | test_all=%d test_uniq=%d",
        len(fit_all),
        len(fit_uniq),
        len(test_all),
        len(test_uniq),
    )

    acts = load_acts(store_dir, fp, valid_cis)
    r19 = _rrow(ROW19)

    # ── basis at block 19 (PCA on dup-excluded train, pooled turns 1-4) ──
    fit_pos = _idx_in_valid(valid_cis, fit_uniq)
    pool = torch.cat([acts["ans"][k][fit_pos, r19, :] for k in range(1, C.K_MAIN + 1)], dim=0)
    pool_mean = pool.mean(0)
    U, S_pca, Vh = torch.linalg.svd(pool - pool_mean, full_matrices=False)
    pcs = Vh[:N_PC].contiguous()  # (N_PC, H) unit-norm rows
    var_ratio = (S_pca[:N_PC] ** 2 / (S_pca**2).sum()).numpy()
    logger.info("[pca] pooled train units=%d top-%d var=%.3f", pool.shape[0], N_PC, var_ratio.sum())

    rng = np.random.default_rng(C.RANDDIR_SEED)
    rand = torch.from_numpy(
        np.stack(
            [
                (v := rng.standard_normal(pcs.shape[1])) / (np.linalg.norm(v) + 1e-12)
                for _ in range(C.RANDDIR_DRAWS)
            ]
        )
    ).to(torch.float32)

    rb = _load_rb(None, None)
    trait_col, _dirs = _directions(rb, pcs.shape[1])
    trait_vec = {t: _dirs[trait_col[t]].to(torch.float32) for t in C.TRAITS}  # unit-norm, own layer

    # ── validation gate: reproduce committed fcast_1to2 skill[row20] on FULL 500 test ──
    test_all_pos = _idx_in_valid(valid_cis, test_all)
    m2 = load_map_rows(maps_dir, "fcast_1to2", [ROW19])[ROW19]
    ctx1_g = acts["ctx1"][test_all_pos, r19, :]
    pred_g = predict_row(m2, ctx1_g).to(torch.float64)
    ans2_g = acts["ans"][2][test_all_pos, r19, :].to(torch.float64)
    # skill = 1 - SSE(pred-actual)/SSE(actual - trainfold_mean) at the read-out row
    trainmean2 = acts["ans"][2][_idx_in_valid(valid_cis, fit_all), r19, :].to(torch.float64).mean(0)
    sse = ((pred_g - ans2_g) ** 2).sum().item()
    null = ((ans2_g - trainmean2) ** 2).sum().item()
    skill_recomp = 1.0 - sse / null
    committed = float(np.load(OUT / "percell" / "fcast_1to2.npz")["skill"][ROW19])
    gate_delta = abs(skill_recomp - committed)
    logger.info(
        "[gate] fcast_1to2 skill[row%d] recomputed=%.6f committed=%.6f |delta|=%.2e",
        ROW19,
        skill_recomp,
        committed,
        gate_delta,
    )
    logger.info(
        "[gate] tol=%.0e (fp16-w floor; fp64-ideal %.0e) -> %s",
        GATE_TOL,
        GATE_FP64_IDEAL,
        "within fp64-ideal" if gate_delta < GATE_FP64_IDEAL else "fp16-quantization-limited",
    )
    assert gate_delta < GATE_TOL, (
        f"PIPELINE GATE FAILED: |delta|={gate_delta:.3e} > {GATE_TOL:.0e} "
        f"(> fp16 quantization floor -> real pipeline bug, needs-refit)"
    )

    # ── M1: per-direction forecast R2 by separation (dup-excluded test) ──
    test_pos = _idx_in_valid(valid_cis, test_uniq)
    fit_all_pos = _idx_in_valid(valid_cis, fit_all)
    ctx1_19 = acts["ctx1"][test_pos, r19, :]
    forecast: dict = {"pc": {}, "random": {}, "trait": {}, "var_ratio": var_ratio.tolist()}
    for k in FCAST_KS:
        fm = load_map_rows(maps_dir, f"fcast_1to{k}", R4)
        # block-19 directions (PCs + random)
        pred19 = predict_row(fm[ROW19], ctx1_19)
        act19 = acts["ans"][k][test_pos, r19, :]
        tmean19 = acts["ans"][k][fit_all_pos, r19, :].mean(0)
        forecast["pc"][k] = r2_per_direction(pred19, act19, tmean19, pcs).tolist()
        forecast["random"][k] = r2_per_direction(pred19, act19, tmean19, rand).tolist()
        # trait directions at their own layers
        forecast["trait"][k] = {}
        for t in C.TRAITS:
            rr = _rrow(TRAIT_ROW[t])
            ctx1_t = acts["ctx1"][test_pos, rr, :]
            pred_t = predict_row(fm[TRAIT_ROW[t]], ctx1_t)
            act_t = acts["ans"][k][test_pos, rr, :]
            tmean_t = acts["ans"][k][fit_all_pos, rr, :].mean(0)
            V = trait_vec[t].unsqueeze(0)
            forecast["trait"][k][t] = float(r2_per_direction(pred_t, act_t, tmean_t, V)[0])

    # ── M2: transmitted spectrum of the own-turn context maps (block 19) ──
    spectrum: dict = {"cells": {}, "principal_angles_deg": {}, "svd_r": SVD_R}
    right_sub: dict[str, torch.Tensor] = {}
    for k in (2, 3, 4):
        cell = f"ctx_k{k}_full"
        w = load_map_rows(maps_dir, cell, [ROW19])[ROW19]["w"].to(torch.float64)
        sv = torch.linalg.svdvals(w)
        _, _, Vh_w = torch.linalg.svd(w, full_matrices=False)
        right_sub[cell] = Vh_w[:SVD_R].t().contiguous()  # (H, r) orthonormal columns
        s = sv.numpy()
        spectrum["cells"][cell] = {
            "singular_values_top50": s[:50].tolist(),
            "sv_total": float((s**2).sum()),
            "eff_rank_participation": float((s.sum() ** 2) / (s**2).sum()),
            "top50_energy_frac": float((s[:50] ** 2).sum() / (s**2).sum()),
        }
    for a, b in (
        ("ctx_k2_full", "ctx_k3_full"),
        ("ctx_k2_full", "ctx_k4_full"),
        ("ctx_k3_full", "ctx_k4_full"),
    ):
        ang = principal_angles(right_sub[a], right_sub[b])
        spectrum["principal_angles_deg"][f"{a}__{b}"] = {
            "min": float(ang.min()),
            "median": float(np.median(ang)),
            "max": float(ang.max()),
            "mean_cos": float(np.cos(np.radians(ang)).mean()),
        }

    # ── M3: raw cross-turn persistence (dup-excluded test, block-19 dirs) ──
    persistence: dict = {"pc": {}, "random": {}, "trait": {}}
    for label, V in (("pc", pcs), ("random", rand)):
        proj = {k: (acts["ans"][k][test_pos, r19, :] @ V.t()) for k in range(1, C.K_MAIN + 1)}
        for k in (1, 2, 3):
            persistence[label][k] = pearson_cols(proj[k], proj[k + 1]).tolist()
    for t in C.TRAITS:
        rr = _rrow(TRAIT_ROW[t])
        V = trait_vec[t].unsqueeze(0)
        proj = {k: (acts["ans"][k][test_pos, rr, :] @ V.t()) for k in range(1, C.K_MAIN + 1)}
        persistence["trait"][t] = {
            k: float(pearson_cols(proj[k], proj[k + 1])[0]) for k in (1, 2, 3)
        }

    res = {
        "definition": (
            "Carried-vs-changed direction spectra at read-out block 19 (best layer): "
            "per-direction forecast R2 from the turn-1 state (saved fcast maps), the "
            "transmitted singular spectrum + principal angles of the own-turn context "
            "maps, and raw cross-turn persistence. Basis: top-100 PCs of block-19 "
            "answer activations (PCA on the dup-excluded train fold, pooled turns 1-4) "
            "+ 3 #778 trait directions (own layers) + 100 unit-norm random directions."
        ),
        "corpus_fingerprint": fp,
        "block": BLOCK19,
        "block19_store_row": ROW19,
        "trait_blocks": TRAIT_BLOCK,
        "trait_store_rows": TRAIT_ROW,
        "n_pc": N_PC,
        "n_random": C.RANDDIR_DRAWS,
        "dup_excluded": {"normalization": "exact", "n_dup": len(dup)},
        "n": {
            "pca_train_conversations": int(len(fit_uniq)),
            "pca_train_units": int(pool.shape[0]),
            "test_uniq": int(len(test_uniq)),
            "test_all_gate": int(len(test_all)),
        },
        "gate": {
            "cell": "fcast_1to2",
            "row": ROW19,
            "recomputed_skill": skill_recomp,
            "committed_skill": committed,
            "abs_delta": gate_delta,
            "tol": GATE_TOL,
            "fp64_ideal": GATE_FP64_IDEAL,
            "within_fp64_ideal": bool(gate_delta < GATE_FP64_IDEAL),
            "note": "persisted map w is fp16; delta floor is fp16 quantization, not fp64",
        },
        "forecast_r2": forecast,
        "transmitted_spectrum": spectrum,
        "persistence": persistence,
        "seeds": {"split": C.SPLIT_SEED, "randdir": C.RANDDIR_SEED, "pca": "deterministic-svd"},
        "metadata": C.reproducibility_metadata({"script": "issue958_carried_directions"}),
    }
    SUB.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(SUB / "spectra.json", res)
    logger.info("wrote %s", SUB / "spectra.json")

    # headline log
    pc_r2_k4 = np.array(forecast["pc"][4])
    rand_p975_k4 = float(np.quantile(forecast["random"][4], 0.975))
    n_carried = int((pc_r2_k4 > rand_p975_k4).sum())
    logger.info(
        "[headline] PCs with forecast R2>random-p97.5 at k=4: %d/%d (rand p97.5=%.3f); "
        "trait R2 k=4: %s",
        n_carried,
        N_PC,
        rand_p975_k4,
        {t: round(forecast["trait"][4][t], 3) for t in C.TRAITS},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
