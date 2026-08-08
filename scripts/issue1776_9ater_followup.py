"""#1776 Step 9a-ter zero-GPU follow-up: J rescale ladder + chain-composition breadth.

P1 (``--p1``) — is the averaged Jacobian's held-out R² ~ 0 a PURE amplitude
deficit? Rescale ladder on the pinned lmsys test-1000 (phase5 ``cmd_transfer``'s
split, val/test index shas re-asserted against the committed original round):

  - raw J (exactly phase5's affine anchor read ``(x - xmu) @ J.T + ymu``);
  - GLOBAL train-fit scalar ``s * J`` (closed-form least squares on the r1-train
    rows);
  - AFFINE train-fit ``s * J + b`` (scalar + mean-offset intercept);
  - per-TEST-context ORACLE scalar (ceiling read, diagnostic-only — a per-context
    scalar has no deployable estimator for unseen contexts; the deployable
    per-context read falls back to the global s);
  - identity + learned bias (``analysis.mapping_baselines.identity_bias_predict``);
  - fitted comparator M' (x50k ridge) — the anchor-independent reference.

  Every rung is scored with phase5.score_predictions (pooled R² + bootstrap CI +
  kNN retrieval via ``analysis.mapping_baselines.knn_retrieval``). The global-s
  rescale ALSO runs on the J_ctx / J_prefix arms (both-arms mapping duty).

  ANCHOR PROVENANCE: phase5's pod-side anchors averaged the FULL n1m lmsys train
  pool (r1-train 3600 rows + every fresh-capture lmsys row) and were never
  uploaded (pod-local; pod terminated). An r1-train-only substitution FAILS the
  reproduce-first gate (raw J reads +0.022 vs committed -0.0014 — the fresh
  captures carry a systematic mean shift), so ``--recover-anchors`` recovers the
  EXACT pool means by a bounded stream-reduce over the 1920-chunk n1m capture
  (download -> reduce -> delete; peak ~one chunk; checkpoint + fingerprint-gated
  resume per the external-stream rule), accumulating in the SAME pass the
  layer-14 x/xy cross-moments (S_xx, S_xy) over the full pool so the global
  rescale scalar is fit CLOSED-FORM on the exact train split. Validation vs the
  committed transfer.json rows: anchor-INDEPENDENT rows (mprime_x50k /
  mprime_lmsys50k / m_shipped) within EXACT_TOL; anchor-DEPENDENT rows (J_last /
  J_ctx / J_prefix / identity_bias l14+l19) within ANCHOR_TOL. A miss raises.
  NOTE: on the full train pool the affine variant collapses to the global
  scalar — the anchors ARE the train means, so the optimal intercept is 0 by
  construction; the r1-train-fit affine/scalar variants are reported as
  robustness diagnostics.

P2 (``--p2``) — is the 5d chain MRR gain (M' 0.0114 vs null 0.0013) broad across
the 999 fresh-WildChat contexts or driven by a few? Reuses phase5's
load_chain_rows / content_token_ids / _chain_metrics + phase4.load_dict and the
cmd_chain rank-scoring expressions; re-asserts the committed aggregate MRR first,
then reads the per-context reciprocal-rank distribution, each context against its
OWN 998-row shuffled null (rows of the same rank matrix).

Content hygiene: WildChat rows / model responses are NEVER printed — logs carry
counts, ids, and ranks only.

Usage (idempotent; thread caps + MALLOC_ARENA_MAX=2 on the launch env):
  uv run python scripts/issue1776_9ater_followup.py --stage
  uv run python scripts/issue1776_9ater_followup.py --p1
  uv run python scripts/issue1776_9ater_followup.py --p2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import issue1776_common as C76
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # bind shared-VM thread caps BEFORE numpy/torch import (#847 gate)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_common as C  # noqa: E402

HF_DL = C76.DATA_DIR / "hf_dl"
CHAIN_DIR = C76.DATA_DIR / "chain_chunks"
STAGE_RECORD = C76.DATA_DIR / "9ater_stage.json"
ANCHORS_PT = C76.DATA_DIR / "9ater_anchors.pt"
ANCHORS_CKPT = C76.DATA_DIR / "9ater_anchors_ckpt.pt"
N1M_ROOT = "issue779_monitoring/fitter-fair-comparison-n1m"
N1M_CAP_PREFIX = f"{N1M_ROOT}/final_token_capture"
OUT_DIR = C76.PROJECT_ROOT / "eval_results" / f"issue_{C76.ISSUE}" / "followup_9ater"
FIG_DIR = C76.PROJECT_ROOT / "figures" / f"issue_{C76.ISSUE}"
TRANSFER_JSON = C76.PROJECT_ROOT / "eval_results/issue_1776/phase5/transfer.json"
CHAIN_JSON = C76.PROJECT_ROOT / "eval_results/issue_1776/phase5/chain_composition.json"
CHAIN_SHIPPED_JSON = (
    C76.PROJECT_ROOT / "eval_results/issue_1776/phase5/chain_composition_shipped.json"
)

JAC = "issue1776_jacobian/analysis_tensors"
WC = "issue1776_jacobian/wildchat_fresh"
SHIPPED_RIDGE = "issue779_monitoring/n1m_readout/weights/L19/ridge.pt"
N_WC_SHARDS = 8

# LFS tensor inputs (sha-verified against the Hub index at stage time).
STAGE_PT: tuple[str, ...] = (
    C76.PASS_B_HF_PATH,
    f"{JAC}/jac_full/J_last.pt",
    f"{JAC}/jac_full/J_ctx.pt",
    f"{JAC}/jac_full/J_prefix.pt",
    f"{JAC}/comparator/m_ridge_x50k.pt",
    f"{JAC}/comparator/m_ridge_lmsys50k.pt",
    f"{JAC}/dictionaries/dictionary_l19.pt",
    SHIPPED_RIDGE,
    *(f"{WC}/final_token_capture/shard{i:02d}_chunk0000.pt" for i in range(N_WC_SHARDS)),
)
# Non-LFS response siblings (size-checked only).
STAGE_JSON: tuple[str, ...] = tuple(
    f"{WC}/raw_completions/shard{i:02d}_chunk0000.json" for i in range(N_WC_SHARDS)
)

EXACT_TOL = 1e-6  # anchor-independent reproduction (same bytes, same split)
# Recovered anchors are the exact pool means up to fp32-pairwise-vs-fp64 sum
# order (~1e-6 relative on the mean -> far below this on R²).
ANCHOR_TOL = 5e-4
CHAIN_MRR_TOL = 3e-4  # deterministic ranks; tolerance covers BLAS near-tie flips


def _local(hub_path: str) -> Path:
    return HF_DL / hub_path


def cmd_stage(args) -> int:
    """Idempotent staging of every consumed input at ONE resolved revision,
    with sha256 verification of the LFS tensors against the Hub index."""
    import issue779_ffc_n50k_generate_capture as N50G

    from explore_persona_space.orchestrate import hub

    from huggingface_hub import HfApi  # noqa: I001

    api = HfApi()
    info = hub.retry_transient(
        lambda: api.repo_info(C76.HF_DATA_REPO, repo_type="dataset"),
        what=f"repo_info({C76.HF_DATA_REPO})",
    )
    revision = str(info.sha)
    print(f"[9ater-stage] staging revision {revision}", flush=True)

    for hp in (*STAGE_PT, *STAGE_JSON):
        dest = _local(hp)
        if dest.exists():
            print(f"[9ater-stage] present: {hp}", flush=True)
            continue
        hub.stage_hub_file(C76.HF_DATA_REPO, hp, dest, repo_type="dataset", revision=revision)
        print(f"[9ater-stage] staged: {hp}", flush=True)

    # sha-verify the LFS tensors (HF mirror != local-verified copy, #600 class).
    import issue1776_phase5 as P5

    verified: dict[str, str] = {}
    by_prefix: dict[str, dict[str, dict]] = {}
    for hp in STAGE_PT:
        prefix = str(Path(hp).parent)
        if prefix not in by_prefix:
            by_prefix[prefix] = hub.retry_transient(
                lambda p=prefix: N50G._remote_index(p), what=f"remote_index({prefix})"
            )
        want = by_prefix[prefix].get(Path(hp).name, {}).get("sha256")
        assert want, f"no Hub LFS sha for {hp} — cannot verify the staged copy"
        got = P5._sha256_file(_local(hp))
        assert got == want, f"{hp}: local sha {got} != Hub LFS {want} — stale/corrupt local copy"
        verified[hp] = got
        print(f"[9ater-stage] sha OK: {hp}", flush=True)

    # Merged chain dir: load_chain_rows expects the raw .json response sibling
    # NEXT to each capture .pt (the pod wrote both to one out-root; the Hub
    # keeps them under two prefixes) — symlink both into one dir.
    CHAIN_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(N_WC_SHARDS):
        for sub, ext in (("final_token_capture", "pt"), ("raw_completions", "json")):
            src = _local(f"{WC}/{sub}/shard{i:02d}_chunk0000.{ext}")
            dst = CHAIN_DIR / f"shard{i:02d}_chunk0000.{ext}"
            if not dst.exists():
                dst.symlink_to(src.resolve())
    C76.atomic_write_json(
        STAGE_RECORD,
        {
            "revision": revision,
            "verified_sha256": verified,
            "staged_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    print(f"[9ater-stage] done -> {STAGE_RECORD}", flush=True)
    return 0


def _load_j(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    return (obj["J"] if isinstance(obj, dict) else obj).to(torch.float64).numpy()


def _pinned_split():
    """The EXACT split phase5's cmd_transfer used (assemble_multilayer L841-851):
    fixed_split over the 5000-row pass_b head, val/test shas re-asserted."""
    import issue779_ffc_n1m_fits as N1M
    import issue779_ffc_n50k_fits as N50
    import issue779_fitter_fair_comparison as F

    pinned = N50._pinned_original_shas(N1M.DEFAULT_ORIG_DIR)
    r1_train, val, test = F.fixed_split(
        N1M.N_PASS_B, N1M.N_PASS_B - N1M.N_VAL - N1M.N_TEST, N1M.N_VAL, N1M.N_TEST, N1M.SPLIT_SEED
    )
    val_sha, test_sha = F._sha_ids(val), F._sha_ids(test)
    assert val_sha == pinned["val_sha256"], (val_sha, pinned["val_sha256"])
    assert test_sha == pinned["test_sha256"], (test_sha, pinned["test_sha256"])
    return r1_train, val, test, {"val_sha256": val_sha, "test_sha256": test_sha, **pinned}


def _fit_scalar(u: np.ndarray, r: np.ndarray) -> float:
    """Closed-form argmin_s ||s*u - r||²  (dof=1; n*H scalar observations)."""
    den = float((u * u).sum())
    return float((u * r).sum() / den) if den > 0 else 0.0


def _chunk_names() -> list[str]:
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    return hub.retry_transient(
        lambda: sorted(
            f.path.rsplit("/", 1)[-1]
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient right here
            for f in HfApi().list_repo_tree(
                C76.HF_DATA_REPO, path_in_repo=N1M_CAP_PREFIX, repo_type="dataset", recursive=True
            )
            if getattr(f, "size", None) is not None and f.path.endswith(".pt")
        ),
        what=f"n1m chunk listing ({N1M_CAP_PREFIX})",
    )


def cmd_recover_anchors(args) -> int:
    """Recover the EXACT phase5 transfer anchors (full lmsys-train-pool means of
    cx14 / cx19 / v19) + the full-pool layer-14 cross-moments (S_xx, S_xy) by ONE
    bounded stream-reduce over the n1m capture chunks (download -> reduce ->
    delete, peak ~one chunk + prefetch; ~83 GB network total). Checkpointed
    every ``--ckpt-every`` chunks with a fingerprint-gated resume."""
    import hashlib

    import issue779_ffc_n1m_fits as N1M
    import issue779_ffc_n1m_generate_capture as N1G
    import issue779_ffc_n50k_fits as N50
    import issue779_fitter_fair_comparison as F

    if ANCHORS_PT.exists() and not args.fresh:
        print(f"[9ater-anchors] already recovered at {ANCHORS_PT}; reuse", flush=True)
        return 0

    manifest_dir = N1G._resolve_manifest_dir(
        argparse.Namespace(
            out_dir=C76.DATA_DIR / "ffc_n1m", manifest_from_hf=True, hf_prefix=N1M_ROOT
        )
    )
    pool, man_meta = N1G.read_manifest_pool(manifest_dir)
    ci_is_lmsys = np.zeros(len(pool), dtype=bool)
    for r in pool:
        ci_is_lmsys[int(r["i"])] = r["corpus"] == "lmsys"
    print(
        f"[9ater-anchors] manifest: n_new={man_meta['n_new']} "
        f"n_lmsys={man_meta['n_lmsys']} n_wildchat={man_meta['n_wildchat']}",
        flush=True,
    )

    names = _chunk_names()
    fp = hashlib.sha256(("9ater-anchors-v1\n" + N1M_CAP_PREFIX + "\n" + "\n".join(names)).encode())
    fingerprint = fp.hexdigest()
    h = C.EXPECTED_HIDDEN

    start = 0
    if ANCHORS_CKPT.exists() and not args.fresh:
        st = torch.load(ANCHORS_CKPT, map_location="cpu", weights_only=True)
        if st["fingerprint"] == fingerprint:
            start = int(st["cursor"])
            print(f"[9ater-anchors] RESUME from chunk {start}/{len(names)}", flush=True)
        else:
            print("[9ater-anchors] checkpoint fingerprint MISMATCH; restart", flush=True)
            st = None
    else:
        st = None
    if st is None:
        # Seed with the pass_b r1-train half of the pool (the pinned split).
        pb = F._mmap_load(_local(C76.PASS_B_HF_PATH))
        r1_train, _val, _te, _split = _pinned_split()
        x14 = N50._slice_layer(pb, "cx_last", C76.SOURCE_LAYER)[r1_train]
        x19 = N50._slice_layer(pb, "cx_last", C76.READOUT_LAYER)[r1_train]
        v19 = N50._slice_layer(pb, "v_x", C76.READOUT_LAYER)[r1_train]
        t14, tv = torch.from_numpy(x14), torch.from_numpy(v19)
        st = {
            "fingerprint": fingerprint,
            "cursor": 0,
            "n_pool": len(r1_train),
            "sum_x14": torch.from_numpy(x14.astype(np.float64).sum(0)),
            "sum_x19": torch.from_numpy(x19.astype(np.float64).sum(0)),
            "sum_v19": torch.from_numpy(v19.astype(np.float64).sum(0)),
            "sxx": (t14.T @ t14).to(torch.float64),
            "sxy": (t14.T @ tv).to(torch.float64),
        }
        del pb, x14, x19, v19, t14, tv

    cache_dir = HF_DL / "n1m_chunks_9ater"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(nm: str) -> str:
        return N1M._download_chunk_with_retry(C76.HF_DATA_REPO, f"{N1M_CAP_PREFIX}/{nm}", cache_dir)

    def _ckpt(cursor: int) -> None:
        st["cursor"] = cursor
        tmp = ANCHORS_CKPT.with_name(ANCHORS_CKPT.name + f".tmp.{time.time_ns()}")
        torch.save(st, tmp)
        tmp.replace(ANCHORS_CKPT)

    t0 = time.time()
    for i, got in N1M._iter_chunks_prefetched(names, start, fetch, args.prefetch):
        b = F._mmap_load(got)
        ci = np.asarray([int(x) for x in b["ci"]], dtype=np.int64)
        assert (ci >= 0).all() and ci.max() < len(pool), (got.name, int(ci.max()))
        keep = ci_is_lmsys[ci]
        if keep.any():
            cx14 = N50._slice_layer(b, "cx_last", C76.SOURCE_LAYER)[keep]
            cx19 = N50._slice_layer(b, "cx_last", C76.READOUT_LAYER)[keep]
            v19 = N50._slice_layer(b, "v_x", C76.READOUT_LAYER)[keep]
            st["sum_x14"] += torch.from_numpy(cx14.astype(np.float64).sum(0))
            st["sum_x19"] += torch.from_numpy(cx19.astype(np.float64).sum(0))
            st["sum_v19"] += torch.from_numpy(v19.astype(np.float64).sum(0))
            t14, tv = torch.from_numpy(cx14), torch.from_numpy(v19)
            st["sxx"] += (t14.T @ t14).to(torch.float64)
            st["sxy"] += (t14.T @ tv).to(torch.float64)
            st["n_pool"] = int(st["n_pool"]) + int(keep.sum())
        del b
        got.unlink()
        if (i + 1) % 25 == 0 or i + 1 == len(names):
            print(
                f"[9ater-anchors] chunk {i + 1}/{len(names)} n_pool={st['n_pool']} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        if args.ckpt_every > 0 and (i + 1) % args.ckpt_every == 0:
            _ckpt(i + 1)

    n = float(st["n_pool"])
    out = {
        "n_pool": int(st["n_pool"]),
        "xmu14": st["sum_x14"] / n,
        "xmu19": st["sum_x19"] / n,
        "ymu19": st["sum_v19"] / n,
        "sxx": st["sxx"],
        "sxy": st["sxy"],
        "fingerprint": fingerprint,
        "n_chunks": len(names),
        "manifest_n_new": int(man_meta["n_new"]),
        "manifest_n_lmsys": int(man_meta["n_lmsys"]),
        "pool_def": "pass_b r1-train (3600) + all fresh-capture lmsys rows (assemble tr_mask)",
    }
    tmp = ANCHORS_PT.with_name(ANCHORS_PT.name + f".tmp.{time.time_ns()}")
    torch.save(out, tmp)
    tmp.replace(ANCHORS_PT)
    ANCHORS_CKPT.unlink(missing_ok=True)
    print(
        f"[9ater-anchors] [phase=anchors_done] n_pool={out['n_pool']} -> {ANCHORS_PT}", flush=True
    )
    return 0


def cmd_p1(args) -> int:
    import issue1776_phase5 as P5
    import issue779_ffc_n1m_fits as N1M
    import issue779_ffc_n50k_fits as N50
    import issue779_fitter_fair_comparison as F

    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    committed = json.loads(TRANSFER_JSON.read_text())["legs"]["lmsys_test1000"]["operators"]
    pb = F._mmap_load(_local(C76.PASS_B_HF_PATH))
    x14 = N50._slice_layer(pb, "cx_last", C76.SOURCE_LAYER).astype(np.float64)
    x19 = N50._slice_layer(pb, "cx_last", C76.READOUT_LAYER).astype(np.float64)
    y19 = N50._slice_layer(pb, "v_x", C76.READOUT_LAYER).astype(np.float64)
    r1_train, _val, te, split = _pinned_split()
    print(f"[9ater-p1] split OK: n_train_r1={len(r1_train)} n_test={len(te)}", flush=True)

    # EXACT recovered anchors + full-pool cross-moments (--recover-anchors).
    assert ANCHORS_PT.exists(), (
        f"{ANCHORS_PT} absent — run --recover-anchors first (exact pool means are "
        "load-bearing at the R²~0 scale; the r1-train substitution FAILS the gate)"
    )
    anc = torch.load(ANCHORS_PT, map_location="cpu", weights_only=True)
    xmu14 = anc["xmu14"].numpy()
    xmu19 = anc["xmu19"].numpy()
    ymu19 = anc["ymu19"].numpy()
    n_pool = int(anc["n_pool"])
    sxx = anc["sxx"].numpy()
    sxy = anc["sxy"].numpy()
    # centered pool moments (the anchors ARE the pool means)
    gxx_c = sxx - n_pool * np.outer(xmu14, xmu14)
    axy_c = sxy - n_pool * np.outer(xmu14, ymu19)
    x14_te, x19_te, y_te = x14[te], x19[te], y19[te]
    dev = torch.device("cpu")
    print(f"[9ater-p1] recovered anchors: n_pool={n_pool}", flush=True)

    def score(pred: np.ndarray) -> dict:
        return P5.score_predictions(pred, y_te, n_boot=args.n_boot, seed=args.seed)

    rows: dict[str, dict] = {}
    # Anchor-independent reference rungs (exact reproduction expected).
    for name, path, in_layer in (
        ("mprime_x50k", _local(f"{JAC}/comparator/m_ridge_x50k.pt"), 14),
        ("mprime_lmsys50k", _local(f"{JAC}/comparator/m_ridge_lmsys50k.pt"), 14),
        ("m_shipped", _local(SHIPPED_RIDGE), 19),
    ):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        x_in = x14_te if in_layer == 14 else x19_te
        rows[name] = score(N1M.apply_map(payload, x_in, dev))
        print(f"[9ater-p1] {name}: r2={rows[name]['r2']:.6f}", flush=True)

    # J arms: raw read + rescale rungs. Global s is fit CLOSED-FORM on the FULL
    # lmsys train pool via the streamed cross-moments:
    #   s* = tr(J A_c) / tr(Jᵀ J G_c),  A_c = Σ(x-x̄)(y-ȳ)ᵀ,  G_c = Σ(x-x̄)(x-x̄)ᵀ.
    # On this pool the affine intercept is 0 by construction (anchors are the
    # train means), so affine ≡ global-s there; r1-train per-row fits are kept
    # as robustness diagnostics.
    r_tr_r1 = y19[r1_train] - ymu19
    r_te = y_te - ymu19
    scalars: dict[str, dict] = {}
    for jname in ("J_last", "J_ctx", "J_prefix"):
        j = _load_j(_local(f"{JAC}/jac_full/{jname}.pt"))
        u_tr = (x14[r1_train] - xmu14) @ j.T
        u_te = (x14_te - xmu14) @ j.T
        rows[jname] = score(u_te + ymu19)  # raw (phase5's exact expression)
        num = float((j * axy_c.T).sum())  # tr(J A_c)
        den = float(((j @ gxx_c) * j).sum())  # tr(Jᵀ J G_c)
        s_g = num / den
        rows[f"{jname}_global_s"] = score(s_g * u_te + ymu19)
        # r1-train robustness fits (per-row): scalar + affine (s, b).
        s_r1 = _fit_scalar(u_tr, r_tr_r1)
        ubar, rbar = u_tr.mean(0), r_tr_r1.mean(0)
        s_a = _fit_scalar(u_tr - ubar, r_tr_r1 - rbar)
        b = rbar - s_a * ubar
        info: dict = {"s_global_full_pool": s_g, "s_r1_train": s_r1, "s_affine_r1_train": s_a}
        if jname == "J_last":
            rows["J_last_affine_r1"] = score(s_a * u_te + b + ymu19)
            # ORACLE: per-TEST-context optimal scalar (ceiling, diagnostic-only).
            den_te = (u_te * u_te).sum(1)
            s_i = np.where(den_te > 0, (u_te * r_te).sum(1) / np.maximum(den_te, 1e-300), 0.0)
            rows["J_last_oracle_per_context"] = score(s_i[:, None] * u_te + ymu19)
            rows["J_last_oracle_per_context"]["label"] = (
                "ORACLE — per-test-context optimal scalar; ceiling, never deployable"
            )
            den_tr = (u_tr * u_tr).sum(1)
            s_tr_i = np.where(den_tr > 0, (u_tr * r_tr_r1).sum(1) / den_tr, 0.0)
            qs = [5, 25, 50, 75, 95]
            info.update(
                {
                    "b_norm_r1_train": float(np.linalg.norm(b)),
                    "test_oracle_s_quantiles": {q: float(np.percentile(s_i, q)) for q in qs},
                    "train_r1_s_quantiles": {q: float(np.percentile(s_tr_i, q)) for q in qs},
                    "amplitude_ratio_mean_resid_over_mean_ju": float(
                        np.linalg.norm(r_te, axis=1).mean() / np.linalg.norm(u_te, axis=1).mean()
                    ),
                }
            )
        scalars[jname] = info
        print(
            f"[9ater-p1] {jname}: raw r2={rows[jname]['r2']:.6f} s_full={s_g:.3f} s_r1={s_r1:.3f}",
            flush=True,
        )

    # identity + learned bias: pool-anchor construction == the canonical helper
    # with the FULL train pool as the fold (b = train-mean(y − x) = ymu − xmu);
    # matches the committed transfer.json rows. The r1-fold helper call is kept
    # as the canonical-API robustness read.
    rows["identity_bias_l14"] = score(x14_te + (ymu19 - xmu14))
    rows["identity_bias_l19"] = score(x19_te + (ymu19 - xmu19))
    rows["identity_bias_l14_r1fold"] = score(
        identity_bias_predict(x14[r1_train], y19[r1_train], x14_te)
    )

    # Validation vs the committed transfer.json (STOP on a miss).
    validation: dict[str, dict] = {}
    for name, tol in (
        ("mprime_x50k", EXACT_TOL),
        ("mprime_lmsys50k", EXACT_TOL),
        ("m_shipped", EXACT_TOL),
        ("J_last", ANCHOR_TOL),
        ("J_ctx", ANCHOR_TOL),
        ("J_prefix", ANCHOR_TOL),
        ("identity_bias_l14", ANCHOR_TOL),
        ("identity_bias_l19", ANCHOR_TOL),
    ):
        got, ref = rows[name]["r2"], committed[name]["r2"]
        validation[name] = {"reproduced_r2": got, "committed_r2": ref, "abs_diff": abs(got - ref)}
        assert abs(got - ref) <= tol, (
            f"reproduce-first sanity FAILED on {name}: got {got:.6f} vs committed {ref:.6f} "
            f"(tol {tol}) — rescale numbers are NOT real; stopping"
        )
    print("[9ater-p1] reproduce-first sanity PASS (8/8 rows)", flush=True)

    out = {
        "dv": "Held-out R² rescale ladder for the averaged Jacobian (pinned lmsys test-1000)",
        "question": "is J's R²~0 a PURE amplitude deficit?",
        "split": {k: split[k] for k in ("val_sha256", "test_sha256", "source")},
        "n_train_r1": int(len(r1_train)),
        "n_train_pool": n_pool,
        "n_test": int(len(te)),
        "anchor_provenance": (
            "anchors (xmu14/xmu19/ymu19) recovered EXACTLY by stream-reducing the full n1m "
            "capture (pool = pass_b r1-train + all fresh lmsys rows, the assemble tr_mask); "
            "the pod-side anchors file was never uploaded. Recovery validated against 8 "
            "committed transfer.json rows (see validation)."
        ),
        "tolerances": {"exact": EXACT_TOL, "anchor_dependent": ANCHOR_TOL},
        "validation": validation,
        "ladder": rows,
        "scalars": scalars,
        "stage_record": json.loads(STAGE_RECORD.read_text()) if STAGE_RECORD.exists() else None,
        "repro": C76.repro_meta(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    C76.atomic_write_json(OUT_DIR / "jacobian_rescale.json", out)
    _fig_ladder(rows)
    print(f"[9ater-p1] [phase=p1_done] -> {OUT_DIR / 'jacobian_rescale.json'}", flush=True)
    return 0


def _fig_ladder(rows: dict[str, dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style("blog")
    order = [
        ("J_last", "Averaged Jacobian J (raw)", "control"),
        ("identity_bias_l14", "Identity + learned bias (layer 14)", "baseline"),
        ("J_last_global_s", "Global scalar rescale s·J (train-fit)", "accent"),
        ("J_last_oracle_per_context", "Per-context rescale of J (oracle ceiling)", "neutral"),
        ("mprime_x50k", "Fitted ridge M′ (50k crossing rows)", "primary"),
    ]
    vals = [rows[k]["r2"] for k, _, _ in order]
    lo = [max(0.0, rows[k]["r2"] - rows[k]["ci_lo"]) for k, _, _ in order]
    hi = [max(0.0, rows[k]["ci_hi"] - rows[k]["r2"]) for k, _, _ in order]
    colors = [pp.paper_palette_role(role) for _, _, role in order]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    y = np.arange(len(order))
    ax.barh(y, vals, xerr=np.array([lo, hi]), color=colors, height=0.62, capsize=3)
    ax.set_yticks(y, [label for _, label, _ in order])
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.set_xlim(left=min(v - e for v, e in zip(vals, lo)) - 0.14)  # room for neg labels
    ax.set_xlabel("Held-out R² (pinned lmsys test-1000, bootstrap 95% CI)")
    for yi, v in zip(y, vals):
        ax.text(
            v + (0.012 if v >= 0 else -0.012),
            yi,
            f"{v:.3f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
        )
    pp.savefig_paper(fig, "jacobian_rescale_ladder", dir=FIG_DIR)
    plt.close(fig)


def _rank_matrix(vhat: np.ndarray, rows_unit, norms, union: np.ndarray, batch: int) -> np.ndarray:
    """(n, |union|) 1-based rank of each union token in the lens-vocab ranking —
    the exact cmd_chain scoring expressions (issue1776_phase5.py L1109-1120)."""
    n, v_size = vhat.shape[0], rows_unit.shape[0]
    ranks_sub = np.empty((n, union.size), dtype=np.int32)
    for start in range(0, n, batch):
        vb = torch.from_numpy(vhat[start : start + batch])
        s = ((vb @ rows_unit.T) * norms).numpy()
        order = np.argsort(-s, axis=1)
        rank_of = np.empty_like(order)
        np.put_along_axis(
            rank_of, order, np.broadcast_to(np.arange(v_size), order.shape).copy(), axis=1
        )
        ranks_sub[start : start + vb.shape[0]] = rank_of[:, union] + 1
        print(f"[9ater-p2] scored {min(start + vb.shape[0], n)}/{n}", flush=True)
    return ranks_sub


def _percontext_block(ranks_sub: np.ndarray, sub_ids: list[np.ndarray]) -> dict:
    """Per-context reciprocal-rank distribution + each context's own 998-row
    shuffled null (column i of the (rows x contexts) best-rank matrix)."""
    n = ranks_sub.shape[0]
    best = np.empty((n, n), dtype=np.int64)
    for i, ids in enumerate(sub_ids):
        assert ids.size, f"context {i} has no content tokens"
        best[:, i] = ranks_sub[:, ids].min(axis=1)
    rr = 1.0 / best
    rr_id = np.diag(rr).copy()
    off = ~np.eye(n, dtype=bool)
    frac_below, beat95, beat975 = [], 0, 0
    for i in range(n):
        null_i = rr[off[:, i], i]
        frac_below.append(float((null_i < rr_id[i]).mean()))
        beat95 += int(rr_id[i] > np.quantile(null_i, 0.95))
        beat975 += int(rr_id[i] > np.quantile(null_i, 0.975))
    order = np.argsort(-rr_id)
    total = float(rr_id.sum())
    qs = [5, 25, 50, 75, 90, 95, 99]

    def _mrr_excl(k: int) -> float:
        return float(rr_id[order[k:]].mean())

    return {
        "n_ctx": int(n),
        "mrr": float(rr_id.mean()),
        "null_mrr_pooled": float(rr[off].mean()),
        "frac_ctx_beating_own_null_p95": beat95 / n,
        "frac_ctx_beating_own_null_p975": beat975 / n,
        "per_ctx_null_percentile_quantiles": {q: float(np.percentile(frac_below, q)) for q in qs},
        "per_ctx_rr_quantiles": {q: float(np.percentile(rr_id, q)) for q in qs},
        "frac_rank1": float((rr_id >= 1.0).mean()),
        "frac_best_in_top50": float((np.diag(best) <= 50).mean()),
        "median_best_rank_identity": float(np.median(np.diag(best))),
        "median_best_rank_null": float(np.median(best[off])),
        "share_of_sum_rr_top_1pct": float(rr_id[order[: max(1, n // 100)]].sum() / total),
        "share_of_sum_rr_top_5pct": float(rr_id[order[: max(1, n // 20)]].sum() / total),
        "share_of_sum_rr_top_10pct": float(rr_id[order[: max(1, n // 10)]].sum() / total),
        "mrr_excluding_top10_ctx": _mrr_excl(10),
        "mrr_excluding_top50_ctx": _mrr_excl(50),
        "mrr_excluding_top100_ctx": _mrr_excl(100),
        "_best_diag": np.diag(best).copy(),
        "_best_off": best[off],
    }


def cmd_p2(args) -> int:
    import issue1776_phase4 as P4
    import issue1776_phase5 as P5
    import issue779_ffc_n1m_fits as N1M

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.DEFAULT_MODEL)
    d = P4.load_dict(_local(f"{JAC}/dictionaries/dictionary_l19.pt"), "cpu")
    rows_unit = d["rows_unit"].to(torch.float32)
    norms = d["row_norms"].to(torch.float32)
    dev = torch.device("cpu")

    results: dict[str, dict] = {}
    figures_payload: dict[str, dict] = {}
    variants = [
        ("mprime_x50k", _local(f"{JAC}/comparator/m_ridge_x50k.pt"), 14, CHAIN_JSON),
        ("m_shipped", _local(SHIPPED_RIDGE), 19, CHAIN_SHIPPED_JSON),
    ]
    for name, op_path, in_layer, committed_path in variants:
        committed = json.loads(committed_path.read_text())
        x, responses = P5.load_chain_rows(CHAIN_DIR, in_layer, 0)
        assert x.shape[0] == committed["n_ctx"], (x.shape, committed["n_ctx"])
        payload = torch.load(op_path, map_location="cpu", weights_only=True)
        vhat = N1M.apply_map(payload, x.astype(np.float64), dev).astype(np.float32)
        content = P5.content_token_ids(tok, responses, committed["df_cap"])
        union = np.unique(np.concatenate([c for c in content if c.size]))
        uidx = {int(t): k for k, t in enumerate(union)}
        sub_ids = [np.array([uidx[int(t)] for t in c], dtype=np.int64) for c in content]
        ranks_sub = _rank_matrix(vhat, rows_unit, norms, union, args.score_batch)

        ident = np.arange(vhat.shape[0])
        mrr, rec, n_used = P5._chain_metrics(ranks_sub, sub_ids, ident, committed["topk"])
        diff = abs(mrr - committed["mrr"])
        assert diff <= CHAIN_MRR_TOL, (
            f"{name}: aggregate MRR {mrr:.6f} != committed {committed['mrr']:.6f} "
            f"(diff {diff:.2e}) — per-context read would not be real; stopping"
        )
        print(
            f"[9ater-p2] {name}: aggregate MRR reproduced ({mrr:.6f}, diff {diff:.2e})", flush=True
        )
        block = _percontext_block(ranks_sub, sub_ids)
        figures_payload[name] = {
            "best_diag": block.pop("_best_diag"),
            "best_off": block.pop("_best_off"),
        }
        block.update(
            {
                "aggregate_mrr_reproduced": mrr,
                "aggregate_recall_at_k": rec,
                "n_ctx_used": int(n_used),
                "committed_mrr": committed["mrr"],
                "committed_null_mrr_mean": committed["null"]["mrr_mean"],
                "input_layer": in_layer,
                "df_cap": committed["df_cap"],
                "topk": committed["topk"],
            }
        )
        results[name] = block

    out = {
        "dv": "Per-context reciprocal rank of the best content token (chain composition, 5d)",
        "question": "is the chain MRR gain broad across the 999 WildChat contexts?",
        "variants": results,
        "stage_record": json.loads(STAGE_RECORD.read_text()) if STAGE_RECORD.exists() else None,
        "repro": C76.repro_meta(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    C76.atomic_write_json(OUT_DIR / "chain_percontext.json", out)
    _fig_chain(figures_payload["mprime_x50k"])
    print(f"[9ater-p2] [phase=p2_done] -> {OUT_DIR / 'chain_percontext.json'}", flush=True)
    return 0


def _fig_chain(payload: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style("blog")
    diag = np.log10(payload["best_diag"].astype(np.float64))
    off = np.log10(payload["best_off"].astype(np.float64))
    fig, ax = plt.subplots(figsize=(7.0, 3.8), layout="constrained")
    bins = np.linspace(0.0, max(diag.max(), off.max()) + 0.1, 45)
    ax.hist(
        off,
        bins=bins,
        density=True,
        color=pp.paper_palette_role("neutral"),
        alpha=0.55,
        label="Shuffled pairing (each context's 998-row null)",
    )
    ax.hist(
        diag,
        bins=bins,
        density=True,
        color=pp.paper_palette_role("primary"),
        histtype="step",
        lw=2.0,
        label="True pairing (v̂ from M′, 50k crossing rows)",
    )
    ax.set_xlabel("log10 rank of the best content token (per context)")
    ax.set_ylabel("Density")
    ax.legend()
    pp.savefig_paper(fig, "chain_percontext_hist", dir=FIG_DIR)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", action="store_true")
    ap.add_argument("--recover-anchors", action="store_true")
    ap.add_argument("--p1", action="store_true")
    ap.add_argument("--p2", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-batch", type=int, default=64)
    ap.add_argument("--prefetch", type=int, default=6)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--fresh", action="store_true", help="ignore anchor checkpoint/output")
    args = ap.parse_args(argv)
    steps = args.stage or args.recover_anchors or args.p1 or args.p2
    assert steps, "pick at least one of --stage/--recover-anchors/--p1/--p2"
    if args.stage:
        assert cmd_stage(args) == 0
    if args.recover_anchors:
        assert cmd_recover_anchors(args) == 0
    if args.p1:
        assert cmd_p1(args) == 0
    if args.p2:
        assert cmd_p2(args) == 0
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit BEFORE C-extension finalize teardown (#1689 class)
