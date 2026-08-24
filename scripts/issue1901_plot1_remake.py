#!/usr/bin/env python3
"""Issue #1901 paper-plan Plot 1 remake (inline round ``plot1-remake``, pod-1901-plot1remake).

Per-layer (all 28 layers, Qwen2.5-7B-Instruct) held-out R^2 AND retrieval acc@1
(PRIMARY = whitened cosine + CSLS; pinned 1,000-row test pool, chance stated)
for EXACTLY four arms, all at n_train=50,000:

  1. ridge          — primal streaming val-lambda ridge (``N1M.fit_ridge``,
                      LAMBDAS_N50K), the banked layer_curve_n50k recipe;
  2. mlp_w8192      — 1-hidden-GELU MLP, width 8192, lr 3e-4 (the #779/#1901
                      fixed recipe), minibatch trainer ``N1M.fit_mlp``;
  3. identity_bias  — W=identity + train-mean bias (``mapping_baselines``);
  4. boundary_ridge — boundary-token('.') state -> next-span-mean ridge on the
                      #1901 WikiText control (fresh 28-layer capture of the
                      banked b0 manifest's n=50k prefix + eval rows).

No base-model arm, no other n values (user spec, dispatch note on #1901).

Reuse: chat arms ride the #779 fitter-fair-comparison-n50k capture through
``issue1901_paper_densify`` (staging, extraction, plan-B split with pinned
byte-identical val/test shas) + the #779 fit cores; the boundary arm rides
``issue1901_boundary_token_control`` (manifest load, ES capture, per-layer
store load) with persist_layers = range(28) into a round-local store.
Retrieval matches ``issue1639_retrieval_read`` / #2202 exactly: whitening
stats fit on the TRAIN side only (train-answer mean + Cholesky of the
lam=0.1-shrunk train-answer covariance, ``null_battery`` recipe), z = L^-1
(x - mu_A) for predictions and pool, CSLS = ``issue1901_metric_battery
.csls_scores`` (K=10) on the whitened cosine-sim matrix, distance = -score;
plain euclidean/cosine companions reconciled against ``knn_retrieval``
inside ``PD.score_cell``. Chance = k/n_pool stated per read.

Parity gates: chat ridge + identity_bias R^2 vs the banked
``layer_curve_n50k.json`` PER LAYER (hard, tol --parity-tol, production only;
L19 runs first so a mismatch halts before the other 27 layers spend). The
boundary L19 n=50k cell is checked vs ``boundary_token_scaling_L19.json``
INFORMATIONALLY only — the fresh capture re-runs the forward under different
batch geometry, so bf16 activation drift is expected (#1005 family).

Checkpointing: per-layer unit JSONs written atomically the moment each unit
completes, resume keyed on GENERATING PARAMETERS (never recomputed-float
bytes, #1336); one stdout ``[<phase>] unit k/N`` line per unit; merged
aggregates rewritten after every unit; eval dir re-uploaded to the HF relay
prefix every --upload-every units.

Artifact policy (dispatch note): eval JSONs -> eval_results/issue_1901/
plot1_remake/ (git at harvest; HF relay ``issue1901_plot1remake/eval``);
boundary span TEXT rows persisted as JSONL shards (<9 MB each); the boundary
28-layer activation store is a DECLARED DISCARD (regen recipe in the
sentinel manifest: rerun --phase boundary over public WikiText @ the #931
pin + the banked ``issue1901_boundary_ctl/manifest``). WikiText is a benign
corpus (no content-hygiene digest restrictions).

Smoke (--smoke): tiny-real CPU e2e — chat on the first --smoke-chunks real
capture chunks (split clamps through the reuse chain's warning branch),
boundary on a 6-layer tiny model (real Qwen tokenizer) over a capped real
manifest slice with persist={0,2,4,5}; outputs diverted to /tmp scratch;
uploads skipped. Smoke blind-spot enumeration: the 7B bf16 load, full-n
walls/RAM peaks, the production parity PASS/FAIL branch at banked values
(demoted to informational under smoke, #1345 gate calibration), and the HF
upload legs are NOT certified by a smoke PASS; every load path (staging,
manifest load, ES capture with an explicit layers subset, shard write/scan
resume, per-layer loads, all four fits, whitened+CSLS scoring, span-text
JSONL, unit checkpoint/resume, sentinel write) IS exercised on real data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps bind BEFORE numpy/torch import.
load_dotenv()

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_ffc_n1m_generate_capture as N1G  # noqa: E402
import issue779_ffc_n50k_fits as N50  # noqa: E402
import issue779_fitter_fair_comparison as F79  # noqa: E402
import issue931_common as i931c  # noqa: E402
import issue931_extract_store as ES  # noqa: E402
import issue1901_boundary_token_control as BT  # noqa: E402
import issue1901_paper_densify as PD  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue1901_metric_battery import K_CSLS, csls_scores  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

from explore_persona_space.analysis import mapping_baselines as MB  # noqa: E402
from explore_persona_space.analysis.null_battery import (  # noqa: E402
    PRIMARY_LAMBDA,
    shrunk_cholesky_from_cov,
)
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1901_plot1_remake")

N_TRAIN = 50_000
LAYERS_ALL = tuple(range(28))
SMOKE_PERSIST = BT.SMOKE_PERSIST_LAYERS  # (0, 2, 4, 5) — the M4 tiny-model remap
KNN_KS = PD.KNN_KS  # (1, 5, 10)
HF_RELAY_PREFIX = "issue1901_plot1remake"
DEFAULT_OUT_EVAL = PROJECT_ROOT / "eval_results" / "issue_1901" / "plot1_remake"
DEFAULT_FIG_DIR = PROJECT_ROOT / "figures" / "issue_1901" / "plot1_remake"
BANKED_LAYER_CURVE = (
    PROJECT_ROOT / "eval_results" / "issue_1901" / "paper_densify" / "layer_curve_n50k.json"
)
BANKED_BOUNDARY_L19 = (
    PROJECT_ROOT
    / "eval_results"
    / "issue_1901"
    / "paper_densify"
    / "boundary_token_scaling_L19.json"
)
CHAT_JSON = "chat_arms_n50k.json"
BOUNDARY_JSON = "boundary_arm_n50k.json"
MLP_RECIPE = {
    "arch": "1 hidden GELU layer, full-dim linear head",
    "width": 8192,
    "lr": 3e-4,
    "weight_decay": F79.MLP_WD,
    "max_epochs": F79.MLP_MAX_EPOCHS,
    "batch": N1M.MLP_BATCH,
    "trainer": "issue779_ffc_n1m_fits.fit_mlp (minibatch)",
}
WHITEN_CONVENTION = {
    "primary": "whiten_csls",
    "whiten_shrinkage_lambda": float(PRIMARY_LAMBDA),
    "k_csls": int(K_CSLS),
    "note": (
        "whitening stats fit on the TRAIN side only (train-answer mean + Cholesky of the "
        "shrunk train-answer covariance, null_battery.shrunk_cholesky_from_cov, lam=0.1); "
        "z = L^-1 (x - mu_A) for predictions and pool; CSLS = issue1901 csls_scores(K=10) "
        "on the whitened cosine-sim matrix, distance = -score (#2202/#1639 conventions). "
        "Chance = k/n_pool per read."
    ),
}


# ── whitened cosine + CSLS retrieval (the #1639/#2202 convention, verbatim) ─────


def whiten(x: np.ndarray, mu: np.ndarray, ell: np.ndarray) -> np.ndarray:
    """z = L^-1 (x - mu): the #2202 train-answer whitening transform."""
    return solve_triangular(ell, (np.asarray(x, np.float64) - mu).T, lower=True).T


def cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n_a, n_b) cosine-similarity matrix (mapping_baselines normalization)."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return an @ bn.T


def midranks(d: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    """Mid-rank of query i's true pool row (the knn_retrieval tie convention)."""
    n = d.shape[0]
    d_true = d[np.arange(n), true_idx]
    tol = 1e-9 * np.maximum(np.abs(d_true)[:, None], 1e-12)
    closer = (d < d_true[:, None] - tol).sum(axis=1)
    tied = (np.abs(d - d_true[:, None]) <= tol).sum(axis=1) - 1
    return 1.0 + closer + 0.5 * tied


def train_whitening_stats(y_train, dev: torch.device, block: int = 8192):
    """(mu, L) from the TRAIN-side answers: fp64 chunked cov on ``dev`` (ddof=1,
    np.cov parity asserted at small n), then the null_battery shrink+Cholesky."""
    yt = torch.as_tensor(np.asarray(y_train)) if not torch.is_tensor(y_train) else y_train
    n, d = yt.shape
    s = torch.zeros(d, dtype=torch.float64, device=dev)
    for k in range(0, n, block):
        s += yt[k : k + block].to(dev, torch.float64).sum(0)
    mu_t = s / n
    g = torch.zeros((d, d), dtype=torch.float64, device=dev)
    for k in range(0, n, block):
        yc = yt[k : k + block].to(dev, torch.float64) - mu_t
        g += yc.T @ yc
    cov = (g / (n - 1)).cpu().numpy()
    mu = mu_t.cpu().numpy()
    if n <= 2500:  # helper-parity assert at smoke n (chunked GEMM == np.cov)
        ref = np.cov(np.asarray(yt, dtype=np.float64), rowvar=False)
        assert np.allclose(cov, ref, rtol=1e-9, atol=1e-9), "chunked cov != np.cov"
    return mu, shrunk_cholesky_from_cov(cov, PRIMARY_LAMBDA)


def score_arm(pred_te: np.ndarray, y_te: np.ndarray, mu, ell, n_boot: int, seed: int) -> dict:
    """R^2 + mean cosine (+bootstrap CI) + retrieval: plain euclid/cos (helper-
    reconciled inside PD.score_cell) PLUS whitened cosine and whitened+CSLS."""
    cell = PD.score_cell(pred_te, y_te, n_boot, seed)
    zq = whiten(pred_te, mu, ell)
    zp = whiten(y_te, mu, ell)
    n = zp.shape[0]
    assert K_CSLS < n, f"pool too small for CSLS: {n} <= K={K_CSLS}"
    true_idx = np.arange(n)
    s_wcos = cos_sim(zq, zp)
    rng = np.random.default_rng(seed + 7)  # the PD.score_cell boot-draw convention
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    for name, dist in (
        ("whiten_csls", -csls_scores(s_wcos, K_CSLS)),
        ("whiten_cos", 1.0 - s_wcos),
    ):
        ranks = midranks(dist, true_idx)
        draws = (ranks[boot_idx] <= 1).mean(axis=1)
        cell["retrieval"][name] = {
            "metric": name,
            "n": int(n),
            "n_pool": int(n),
            "acc_at_k": {int(k): float((ranks <= k).mean()) for k in KNN_KS},
            "chance_at_k": {int(k): float(k / n) for k in KNN_KS},
            "median_rank": float(np.median(ranks)),
            "mrr": float((1.0 / ranks).mean()),
            "acc1_ci": {
                "lo": float(np.percentile(draws, 2.5)),
                "hi": float(np.percentile(draws, 97.5)),
            },
        }
    cell["whitening"] = dict(WHITEN_CONVENTION)
    return cell


# ── shared unit checkpoint / upload helpers ─────────────────────────────────────


def _resume_hit(path: Path, key: dict, label: str, k: int, n: int) -> dict | None:
    if path.exists():
        prev = json.loads(path.read_text())
        if prev.get("unit_key") == key:
            logger.info("[%s] unit %d/%d %s resume-skip", label, k, n, path.stem)
            return prev
    return None


def _maybe_upload_eval(args, *, force: bool = False) -> None:
    """Re-upload the eval dir to the HF relay prefix (one folder commit)."""
    if args.smoke or args.skip_upload:
        return
    _maybe_upload_eval.counter += 1
    if force or _maybe_upload_eval.counter % args.upload_every == 0:
        url = hub._upload(
            args.out_eval,
            i931c.HF_DATA_REPO,
            "dataset",
            path_in_repo=f"{HF_RELAY_PREFIX}/eval",
            raise_on_error=True,
        )
        logger.info("[upload] eval dir -> %s", url)


_maybe_upload_eval.counter = 0


def _meta(phase: str) -> dict:
    return as_metadata_dict(git_provenance(PROJECT_ROOT), phase=phase) | {
        "script": "issue1901_plot1_remake",
        "issue": 1901,
        "round": "plot1-remake",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ── phase: chat arms (ridge / mlp_w8192 / identity_bias per layer at n=50k) ─────


def _chat_unit_key(args, layer: int, train_sha: str, n_rows: int, dtype: str) -> dict:
    return {
        "layer": int(layer),
        "seed": int(args.seed),
        "n_train_target": N_TRAIN,
        "lambda_grid": ["logspace", -3, 7, 21],
        "train_sha256": train_sha,
        "n_rows": int(n_rows),
        "dtype": dtype,
        "mlp": {"width": 8192, "lr": 3e-4, "max_epochs": _mlp_epochs(args), "batch": N1M.MLP_BATCH},
        "whiten": {"lambda": float(PRIMARY_LAMBDA), "k_csls": int(K_CSLS)},
        "n_boot": int(args.n_boot),
        "smoke": bool(args.smoke),
    }


def _mlp_epochs(args) -> int:
    return int(args.smoke_epochs) if args.smoke else int(F79.MLP_MAX_EPOCHS)


def _banked_chat_r2(smoke: bool) -> dict:
    """Per-layer banked {layer: {arm: r2}} from the committed layer_curve_n50k.json."""
    if not BANKED_LAYER_CURVE.exists():
        assert smoke, f"banked parity source missing: {BANKED_LAYER_CURVE}"
        return {}
    d = json.loads(BANKED_LAYER_CURVE.read_text())
    return {
        li: {
            "ridge": row["ridge"]["whole_map_r2"],
            "identity_bias": row["identity_bias"]["whole_map_r2"],
        }
        for li, row in d["per_layer"].items()
    }


def phase_chat(args) -> dict:
    C.phase("chat-arms")
    smoke = bool(args.smoke)
    dev = torch.device(args.device)
    unit_dir = args.out_eval / "chat_units"
    unit_dir.mkdir(parents=True, exist_ok=True)

    capture_dir = PD.stage_prefix(
        N50.HF_N50K_PREFIX,
        args.stage_root,
        max_files=(args.smoke_chunks if smoke else None),
        workers=args.stage_workers,
    )
    if not args.pass_b.exists():  # stage the pass_b bundle (the MLP-driver recipe)
        from huggingface_hub import hf_hub_download

        args.pass_b.parent.mkdir(parents=True, exist_ok=True)
        got = hub.retry_transient(
            lambda: hf_hub_download(
                repo_id=i931c.HF_DATA_REPO,
                filename="issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt",
                repo_type="dataset",
                local_dir=args.stage_root / "pass_b_dl",
            ),
            what="pass_b bundle download",
        )
        Path(got).replace(args.pass_b)
    pb = N1G._load_pass_b_bundle(args.pass_b)
    assert int(pb["cx_last"].shape[0]) == N50.N_PASS_B, pb["cx_last"].shape

    X_all, Y_all, cap_layers, dtype = PD._extract_all_layers(
        capture_dir, args.smoke_chunks if smoke else None
    )
    n_new = X_all.shape[0]
    if not smoke and n_new != N50.N_N50K_NEW:
        raise RuntimeError(f"expected {N50.N_N50K_NEW} n50k kept rows, extracted {n_new}")
    pinned = N50._pinned_original_shas(args.orig_dir)
    train, val, test, diag = N50.build_n50k_split(n_new, None, pinned, n_train=N_TRAIN, seed=42)
    n_rows = N50.N_PASS_B + n_new
    ev = np.concatenate([val, test])
    banked = _banked_chat_r2(smoke)

    def _assemble(layer: int):
        col = cap_layers.index(layer)
        x = np.concatenate(
            [N50._slice_layer(pb, "cx_last", layer), X_all[:, col, :].astype(np.float32)]
        )
        y = np.concatenate(
            [N50._slice_layer(pb, "v_x", layer), Y_all[:, col, :].astype(np.float32)]
        )
        assert x.shape == (n_rows, PD.H_DIM) and y.shape == x.shape, (x.shape, y.shape)
        return x, y

    want_layers = [19] + [li for li in cap_layers if li != 19]
    if smoke:
        want_layers = [19, 0]
    pilot: dict | None = None
    t_all = time.time()
    for k, layer in enumerate(want_layers):
        out_path = unit_dir / f"L{layer}.json"
        key = _chat_unit_key(args, layer, diag["train_sha256"], n_rows, dtype)
        if _resume_hit(out_path, key, "chat", k + 1, len(want_layers)) is not None:
            continue
        ts = time.time()
        X, Y = _assemble(layer)
        y_te = Y[test]
        mu, ell = train_whitening_stats(Y[train], dev)

        pred_ridge, ridge_meta = N1M.fit_ridge(
            X, Y, train, val, test, N50.LAMBDAS_N50K, dev, args.ridge_block
        )
        ridge_cell = score_arm(pred_ridge, y_te, mu, ell, args.n_boot, args.seed)
        ridge_cell["fit_meta"] = ridge_meta

        pred_ib = MB.identity_bias_predict(X[train], Y[train], X[test])
        ib_cell = score_arm(pred_ib, y_te, mu, ell, args.n_boot, args.seed)

        t_mlp = time.time()
        pred_ev, mlp_meta = N1M.fit_mlp(
            X, Y, train, ev, 8192, 3e-4, _mlp_epochs(args), N1M.MLP_BATCH, args.seed, dev
        )
        mlp_wall = time.time() - t_mlp
        pred_mval, pred_mte = pred_ev[: len(val)], pred_ev[len(val) :]
        mlp_cell = score_arm(pred_mte, y_te, mu, ell, args.n_boot, args.seed)
        mlp_cell["fit_meta"] = mlp_meta | {"wall_s": round(mlp_wall, 1)}
        mlp_cell["val_r2"] = float(F79._recon_point(pred_mval, Y[val])[0])
        if pilot is None:
            pilot = {
                "layer": int(layer),
                "mlp_wall_s": round(mlp_wall, 1),
                "projected_mlp_total_s": round(mlp_wall * len(want_layers), 1),
                "fence_s_2x": round(2 * mlp_wall * len(want_layers), 1),
            }
            C.write_json_atomic(args.out_eval / "mlp_pilot.json", pilot | {"meta": _meta("pilot")})
            logger.info(
                "[pilot] mlp L%d wall=%.1fs -> fence(2x*28)=%.0fs",
                layer,
                mlp_wall,
                pilot["fence_s_2x"],
            )

        parity_rows = []
        for arm, cell in (("ridge", ridge_cell), ("identity_bias", ib_cell)):
            want = banked.get(str(layer), {}).get(arm)
            if want is not None:
                parity_rows.append(
                    PD._parity_check(
                        f"L{layer}-{arm}-n50k",
                        cell["whole_map_r2"],
                        want,
                        args.parity_tol,
                        smoke=smoke,
                    )
                )
        unit = {
            "unit_key": key,
            "layer": int(layer),
            "arms": {"ridge": ridge_cell, "identity_bias": ib_cell, "mlp_w8192": mlp_cell},
            "parity": parity_rows,
            "wall_time_s": round(time.time() - ts, 1),
        }
        C.write_json_atomic(out_path, unit)
        logger.info(
            "[chat] unit %d/%d L%d ridge_r2=%.4f mlp_r2=%.4f ib_r2=%.4f "
            "ridge_wcsls@1=%.3f elapsed=%.0fs",
            k + 1,
            len(want_layers),
            layer,
            ridge_cell["whole_map_r2"],
            mlp_cell["whole_map_r2"],
            ib_cell["whole_map_r2"],
            ridge_cell["retrieval"]["whiten_csls"]["acc_at_k"][1],
            time.time() - ts,
        )
        _write_chat_merged(args, unit_dir, want_layers, diag, dtype, n_rows)
        _maybe_upload_eval(args)

    _write_chat_merged(args, unit_dir, want_layers, diag, dtype, n_rows)
    _maybe_upload_eval(args, force=True)
    logger.info("[chat] done (%.0fs total)", time.time() - t_all)
    if not args.keep_stage:
        PD._reap_stage(args.stage_root / Path(N50.HF_N50K_PREFIX).parent)
    return {"out": str(args.out_eval / CHAT_JSON), "layers": want_layers}


def _write_chat_merged(args, unit_dir: Path, want_layers, diag, dtype, n_rows) -> None:
    merged = {
        "per_layer": {
            str(li): json.loads((unit_dir / f"L{li}.json").read_text())
            for li in want_layers
            if (unit_dir / f"L{li}.json").exists()
        },
        "split": diag,
        "capture_dtype": dtype,
        "n_rows": int(n_rows),
        "arms": ["ridge", "mlp_w8192", "identity_bias"],
        "mlp_recipe": MLP_RECIPE,
        "whitening": WHITEN_CONVENTION,
        "knn": {"ks": list(KNN_KS), "pool": "pinned test_1000 true targets", "chance_at_1": 0.001},
        "smoke": bool(args.smoke),
        "metadata": _meta("chat-arms"),
    }
    C.write_json_atomic(args.out_eval / CHAT_JSON, merged)


# ── phase: boundary arm (fresh 28-layer capture + per-layer ridge at n=50k) ─────


def _boundary_rows(args) -> tuple[list[dict], dict, dict]:
    """Manifest load + '.'-arm filter: eval rows + the first-n train prefix."""
    man_dir = PD.stage_prefix(
        BT.MANIFEST_PREFIX, args.stage_root / "boundary", workers=args.stage_workers
    )
    rows, ids_by_art, meta = BT._load_manifest(man_dir)
    if not args.smoke:
        assert meta["yield_gate"]["pass"], f"b0 yield gate failed: {meta['yield_gate']}"
    dot = [r for r in rows if r["sep_char"] == "."]
    tr = sorted((r for r in dot if r["split"] == "train"), key=lambda r: r["train_order"])
    te = sorted((r for r in dot if r["split"] == "test"), key=lambda r: r["row_id"])
    va = sorted((r for r in dot if r["split"] == "val"), key=lambda r: r["row_id"])
    if args.smoke:
        tr, te, va = tr[:60], te[:30], va[:10]
    else:
        assert len(te) == BT.N_EVAL_TEST and len(va) == BT.N_EVAL_VAL, (len(te), len(va))
        assert len(tr) >= N_TRAIN, f"train pool {len(tr)} < {N_TRAIN}"
        tr = tr[:N_TRAIN]  # the banked prefix-draw convention (train_order order)
    kept = tr + va + te
    logger.info(
        "[boundary] manifest rows kept: train %d, val %d, test %d (of %d '.'-rows)",
        len(tr),
        len(va),
        len(te),
        len(dot),
    )
    return kept, ids_by_art, meta


def _write_span_texts(args, kept: list[dict], ids_by_art: dict, tok) -> list[str]:
    """Persist the USED rows' span TEXT as JSONL shards (<9 MB each, upload-policy)."""
    spans_dir = args.out_eval / "spans"
    spans_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    shard_rows = 20_000
    for si in range(0, len(kept), shard_rows):
        chunk = kept[si : si + shard_rows]
        texts = tok.batch_decode(
            [list(ids_by_art[r["article_id"]][r["t_span"][0] : r["t_span"][1]]) for r in chunk]
        )
        name = f"boundary_spans_used_{si // shard_rows:02d}.jsonl"
        lines = [
            json.dumps(
                {k: r[k] for k in BT._MANIFEST_ROW_KEYS if k in r} | {"span_text": t},
                ensure_ascii=False,
            )
            for r, t in zip(chunk, texts)
        ]
        (spans_dir / name).write_text("\n".join(lines) + "\n")
        names.append(name)
    logger.info("[boundary] span texts persisted: %d rows over %d shard(s)", len(kept), len(names))
    return names


def _boundary_capture(args, kept: list[dict], ids_by_art: dict, persist: tuple[int, ...]) -> Path:
    """Fresh capture of the kept rows at the persist layers into a round-local store."""
    store = args.out_root / "boundary_store"
    store.mkdir(parents=True, exist_ok=True)
    done, next_idx = BT._scan_store_resume(store, persist)
    pending = [r for r in kept if r["row_id"] not in done]
    if not pending:
        logger.info("[boundary] capture complete (%d rows in store) — model not loaded", len(done))
        return store
    per_shard_gb = args.shard_pairs * len(persist) * ES.EXPECTED_HIDDEN * 2 * 2 / 1e9
    assert_out_root_headroom(
        args.out_root, per_shard_gb * (len(pending) / args.shard_pairs) * 1.15 + 2.0, phase="bcap"
    )
    tiny_dir = None
    if args.smoke:
        tiny_dir = str(args.out_root / "tiny_model")
        if not (Path(tiny_dir) / "config.json").exists():
            ES.make_tiny_model(Path(tiny_dir), layers=6)
    model = ES.load_model(tiny_dir)
    tok = i931c.get_tokenizer(tiny_dir or i931c.MODEL_ID)
    assert tok.pad_token_id is not None, "tokenizer has no pad token id"
    items = BT._items_from_manifest(pending, ids_by_art)
    n_shards = -(-len(pending) // args.shard_pairs)
    logger.info(
        "[boundary] capture: %d pending pairs over %d articles -> %d shard(s), layers=%s",
        len(pending),
        len(items),
        n_shards,
        list(persist),
    )
    buf: list[dict] = []
    shard_idx = next_idx
    t_shard = time.time()
    first_wall = None

    def _flush(records: list[dict]) -> None:
        nonlocal shard_idx, first_wall
        ES.write_shard(records, store, shard_idx, "armC", layers=persist)
        for ext in (".pt", ".json"):
            (store / f"armC_shard{shard_idx:03d}{ext}").replace(
                store / f"pairs_shard{shard_idx:03d}{ext}"
            )
        shard_idx += 1
        if first_wall is None:
            first_wall = time.time() - t_shard
            logger.info(
                "[pilot] boundary capture shard 1 wall=%.0fs -> projected total=%.0fs",
                first_wall,
                first_wall * n_shards,
            )
        logger.info(
            "[boundary] unit %d/%d capture shard elapsed=%.0fs",
            shard_idx,
            n_shards,
            time.time() - t_shard,
        )

    for recs in ES.run_extraction(
        model, items, tok.pad_token_id, args.batch_size, "armC", layers=persist
    ):
        buf.extend(recs)
        while len(buf) >= args.shard_pairs:
            _flush(buf[: args.shard_pairs])
            buf = buf[args.shard_pairs :]
            t_shard = time.time()
    if buf:
        _flush(buf)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return store


def _boundary_unit_key(args, layer: int, n_train: int, train_sha: str, meta: dict) -> dict:
    return {
        "layer": int(layer),
        "seed": int(args.seed),
        "n_train": int(n_train),
        "lambda_grid": BT._lambda_grid_params(n_train),
        "train_rowids_sha256": train_sha,
        "manifest_regime": meta["regime_key"],
        "whiten": {"lambda": float(PRIMARY_LAMBDA), "k_csls": int(K_CSLS)},
        "n_boot": int(args.n_boot),
        "smoke": bool(args.smoke),
    }


def phase_boundary(args) -> dict:
    C.phase("boundary-arm")
    smoke = bool(args.smoke)
    dev = torch.device(args.device)
    unit_dir = args.out_eval / "boundary_units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    persist = SMOKE_PERSIST if smoke else LAYERS_ALL

    kept, ids_by_art, meta = _boundary_rows(args)
    tok = i931c.get_tokenizer(i931c.MODEL_ID)
    span_shards = _write_span_texts(args, kept, ids_by_art, tok)
    store = _boundary_capture(args, kept, ids_by_art, persist)

    banked_l19 = None
    if BANKED_BOUNDARY_L19.exists():
        cells = json.loads(BANKED_BOUNDARY_L19.read_text())["cells"]
        hit = [c for c in cells if c["n_train"] == N_TRAIN and c["draw"] == "prefix"]
        banked_l19 = hit[0]["ridge"]["test_r2"] if hit else None

    n_train_realized = sum(1 for r in kept if r["split"] == "train")
    order = ([19] if 19 in persist else []) + [li for li in persist if li != 19]
    if smoke:
        order = list(persist)[:2]
    t_all = time.time()
    for k, layer in enumerate(order):
        X, Y, row_ids, art_ids = BT._load_layer_arrays(store, layer, persist)
        row_pos = {rid: j for j, rid in enumerate(row_ids)}
        missing = [r["row_id"] for r in kept if r["row_id"] not in row_pos]
        assert not missing, f"partial boundary store: {len(missing)} kept rows absent"
        tr_rows = [r for r in kept if r["split"] == "train"]
        tr = np.asarray([row_pos[r["row_id"]] for r in tr_rows], dtype=np.int64)
        va = np.asarray([row_pos[r["row_id"]] for r in kept if r["split"] == "val"], dtype=np.int64)
        te_rows = [r for r in kept if r["split"] == "test"]
        te = np.asarray([row_pos[r["row_id"]] for r in te_rows], dtype=np.int64)
        train_sha = hashlib.sha256(
            "\n".join(sorted(r["row_id"] for r in tr_rows)).encode()
        ).hexdigest()
        key = _boundary_unit_key(args, layer, len(tr), train_sha, meta)
        out_path = unit_dir / f"L{layer}.json"
        if _resume_hit(out_path, key, "boundary", k + 1, len(order)) is not None:
            del X, Y
            continue
        ts = time.time()
        pred, fit_meta = N1M.fit_ridge(
            X, Y, tr, va, te, BT._lambdas_for(len(tr)), dev, args.ridge_block
        )
        y_te = BT._to_f64_np(Y, te)
        mu, ell = train_whitening_stats(Y[torch.as_tensor(tr, dtype=torch.long)], dev)
        cell = score_arm(pred, y_te, mu, ell, args.n_boot, args.seed)
        cell["fit_meta"] = fit_meta
        cell["article_ci"] = BT.article_cluster_boot(
            pred, y_te, [art_ids[j] for j in te], args.n_boot, args.seed
        )
        unit = {
            "unit_key": key,
            "layer": int(layer),
            "n_train": int(len(tr)),
            "arms": {"boundary_ridge": cell},
            "wall_time_s": round(time.time() - ts, 1),
        }
        if layer == 19 and banked_l19 is not None and not smoke:
            unit["banked_l19_parity_informational"] = {
                "banked_test_r2": banked_l19,
                "this_run_test_r2": cell["whole_map_r2"],
                "delta": cell["whole_map_r2"] - banked_l19,
                "note": (
                    "INFORMATIONAL: fresh 28-layer capture re-runs the forward under "
                    "different batch geometry — bf16 activation drift expected"
                ),
            }
            logger.info(
                "[boundary] L19 informational parity: %s",
                json.dumps(unit["banked_l19_parity_informational"]),
            )
        C.write_json_atomic(out_path, unit)
        logger.info(
            "[boundary] unit %d/%d L%d ridge_r2=%.4f wcsls@1=%.3f elapsed=%.0fs",
            k + 1,
            len(order),
            layer,
            cell["whole_map_r2"],
            cell["retrieval"]["whiten_csls"]["acc_at_k"][1],
            time.time() - ts,
        )
        del X, Y
        _write_boundary_merged(args, unit_dir, order, meta, span_shards, len(tr))
        _maybe_upload_eval(args)

    _write_boundary_merged(args, unit_dir, order, meta, span_shards, n_train_realized)
    _maybe_upload_eval(args, force=True)
    logger.info("[boundary] done (%.0fs total)", time.time() - t_all)
    return {"out": str(args.out_eval / BOUNDARY_JSON), "layers": order, "store": str(store)}


def _write_boundary_merged(args, unit_dir: Path, order, meta, span_shards, n_train) -> None:
    merged = {
        "per_layer": {
            str(li): json.loads((unit_dir / f"L{li}.json").read_text())
            for li in order
            if (unit_dir / f"L{li}.json").exists()
        },
        "arm": "boundary_ridge (x_sep '.' anchor state -> next-span mean, WikiText control)",
        "n_train": int(n_train),
        "draw_convention": "file-order prefix over the b0 seed-42 shuffle (train_order order)",
        "eval": {"test": BT.N_EVAL_TEST, "val": BT.N_EVAL_VAL, "article_disjoint": True},
        "manifest_regime": meta["regime_key"],
        "span_text_shards": span_shards,
        "whitening": WHITEN_CONVENTION,
        "knn": {
            "ks": list(KNN_KS),
            "pool": "held-out test span-mean targets",
            "chance_at_1": 1.0 / BT.N_EVAL_TEST,
        },
        "smoke": bool(args.smoke),
        "metadata": _meta("boundary-arm"),
    }
    C.write_json_atomic(args.out_eval / BOUNDARY_JSON, merged)


# ── phase: figure (VM-side at harvest; two panels, four arms) ───────────────────

ARM_LABELS = {
    "ridge": "Linear (ridge)",
    "mlp_w8192": "Nonlinear (MLP w=8192)",
    "identity_bias": "Identity + bias",
    "boundary_ridge": "Boundary token (WikiText control)",
}


def phase_fig(args) -> dict:
    """Two panels vs layer: held-out R^2; acc@1 (whitened cosine + CSLS, pool=1000)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    chat = json.loads((args.out_eval / CHAT_JSON).read_text())
    boundary = json.loads((args.out_eval / BOUNDARY_JSON).read_text())
    series: dict[str, dict[int, dict]] = {a: {} for a in ARM_LABELS}
    for li, unit in chat["per_layer"].items():
        for arm in ("ridge", "mlp_w8192", "identity_bias"):
            series[arm][int(li)] = unit["arms"][arm]
    for li, unit in boundary["per_layer"].items():
        series["boundary_ridge"][int(li)] = unit["arms"]["boundary_ridge"]

    set_paper_style()
    colors = dict(zip(ARM_LABELS, paper_palette(len(ARM_LABELS))))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for arm, label in ARM_LABELS.items():
        layers = sorted(series[arm])
        if not layers:
            continue
        r2 = [series[arm][li]["whole_map_r2"] for li in layers]
        # json round-trip stringifies the acc_at_k int keys
        acc = [series[arm][li]["retrieval"]["whiten_csls"]["acc_at_k"]["1"] for li in layers]
        lo = [series[arm][li]["retrieval"]["whiten_csls"]["acc1_ci"]["lo"] for li in layers]
        hi = [series[arm][li]["retrieval"]["whiten_csls"]["acc1_ci"]["hi"] for li in layers]
        ax1.plot(layers, r2, marker="o", ms=3, color=colors[arm], label=label)
        ax2.plot(layers, acc, marker="o", ms=3, color=colors[arm], label=label)
        ax2.fill_between(layers, lo, hi, color=colors[arm], alpha=0.15, lw=0)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Held-out $R^2$ (test, n=1,000)")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("acc@1 (whitened cosine + CSLS)")
    ax2.axhline(0.001, ls="--", lw=0.8, color="gray", label="Chance (1/1000)")
    ax2.set_ylim(-0.02, 1.0)
    ax1.legend(frameon=False, fontsize=7)
    ax2.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, "plot1_layer_curve_n50k_4arms", dir=args.fig_dir)
    logger.info("[fig] wrote %s", {k: str(v) for k, v in paths.items()})
    return {"figure": {k: str(v) for k, v in paths.items()}}


# ── main ────────────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #1901 Plot 1 remake (4 arms x 28 layers).")
    ap.add_argument("--phase", choices=["chat", "boundary", "fig", "all"], default="all")
    ap.add_argument(
        "--stage-root",
        type=Path,
        required=False,
        default=None,
        help="HF staging root (pod container/volume disk)",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="round-local scratch (boundary store); default <stage-root>",
    )
    ap.add_argument("--out-eval", type=Path, default=DEFAULT_OUT_EVAL)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument(
        "--pass-b",
        type=Path,
        default=None,
        help="pass_b bundle path (default <stage-root>/pass_b/train_context_vectors.pt)",
    )
    ap.add_argument("--orig-dir", type=Path, default=N50.DEFAULT_ORIG_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--n-threads", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=F79.BOOT_N)
    ap.add_argument("--ridge-block", type=int, default=N1M.RIDGE_BLOCK)
    ap.add_argument("--stage-workers", type=int, default=8)
    ap.add_argument("--parity-tol", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard-pairs", type=int, default=2000)
    ap.add_argument("--upload-every", type=int, default=5)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--keep-stage", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny-real CPU e2e (scratch-diverted)")
    ap.add_argument("--smoke-chunks", type=int, default=2)
    ap.add_argument("--smoke-epochs", type=int, default=2)
    ap.add_argument("--sentinel", type=Path, default=None)
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok")
        raise SystemExit(0)
    torch.set_num_threads(int(args.n_threads))
    if args.smoke:
        scratch = Path("/tmp/issue-1901-plot1remake-smoke")
        if args.out_eval == DEFAULT_OUT_EVAL:
            args.out_eval = scratch / "eval"
        if args.fig_dir == DEFAULT_FIG_DIR:
            args.fig_dir = scratch / "figures"
        if args.stage_root is None:
            args.stage_root = scratch / "stage"
        if args.n_boot == F79.BOOT_N:
            args.n_boot = 50
        logger.info("[smoke] outputs diverted to %s", scratch)
    if args.phase != "fig":
        assert args.stage_root is not None, "--stage-root is required for chat/boundary/all"
        args.stage_root.mkdir(parents=True, exist_ok=True)
    if args.out_root is None:
        args.out_root = args.stage_root if args.stage_root is not None else Path("/tmp")
    if args.pass_b is None and args.stage_root is not None:
        args.pass_b = args.stage_root / "pass_b" / "train_context_vectors.pt"
    args.out_eval.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    if args.phase in ("chat", "all"):
        results["chat"] = phase_chat(args)
    if args.phase in ("boundary", "all"):
        results["boundary"] = phase_boundary(args)
    if args.phase == "fig":
        results["fig"] = phase_fig(args)
    C.phase("done")
    if args.sentinel is not None:
        C.write_json_atomic(
            args.sentinel,
            {
                "ok": True,
                "rc": 0,
                "phase": args.phase,
                "smoke": bool(args.smoke),
                "outputs": results,
                "hf_relay_prefix": f"{HF_RELAY_PREFIX}/eval",
                "discarded_artifacts": [
                    {
                        "name": "boundary 28-layer activation store (pairs_shard*.pt)",
                        "reason": "large intermediate tensor; span TEXT rows persisted",
                        "regen_recipe": (
                            "uv run python scripts/issue1901_plot1_remake.py --phase boundary "
                            "--stage-root <root> (public WikiText @ the #931 pin + the banked "
                            "issue1901_boundary_ctl/manifest)"
                        ),
                    }
                ],
                "metadata": _meta(f"sentinel-{args.phase}"),
            },
        )
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
