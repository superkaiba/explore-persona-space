"""Issue #1895 — map-predictable vs SAE-representable subspaces (driver S0-S7).

One driver, one ``--phase`` flag per plan §4 phase (plan v4; smoke IS this
driver with tiny args — PASS_UNIFIED):
  stage            S0  stage pooled SAE store + SAE weights (k64 + k128) + probe shard.
  matrices         S1  X/Y assembly via the parent code path (_assemble_with_ci),
                       split carve + G1 sha asserts, fp64 train-cov eigh -> Q (G1b).
  recon            S2  r_bar = W_dec @ f_bar + b_dec from banked pooled codes (zero
                       encode passes), full-grain coverage asserts, FVE_u / FVE_j.
  pilot            S3  512-row teacher-forced pilot: G2a reconstruction identity,
                       G2b mask-mismatch routing (Path A/B), k128 twin.
  capture-matched  S3b (Path B only) full 142k-row v_bar_mask capture + upload.
  fits             S4  ONE shared-Gram ridge over targets {vA, rbar, ebar} x
                       {context, prefix} arms + shared-lambda decomposition +
                       identity+bias / kNN baselines + MLP twins + per-direction
                       profiles on Q.
  nulls|angles     S5  principal-angle battery vs within-shell rotation nulls
                       (K draws batched as stacked GEMM + batched svdvals),
                       H3 plug-in + 10k-draw paired bootstrap (3-GEMM pattern),
                       pure-e_bar pilot spot-read.
  correlates       S6  variance-partialled per-direction / per-feature correlates,
                       BH-FDR, verdict lattice.
  upload           S7  fail-loud HF upload + results sentinel.
  all              S0..S7 sequentially (S3b only under Path B).

Resume provenance (#952 gate-5 manifest shape): every phase done-sentinel and
per-unit shard carries {code_sha, split_shas, sae_revision, config_digest};
resume predicates re-validate it before skipping (stale/missing => recompute).

Pod-side contract: sentinel under /workspace/logs/issue-1895-results.json only
(never task.py); [phase=...] log lines; explicit sys.exit(0) (PyGILState-race
gotcha). LMSYS/WildChat text is handled DIGEST-ONLY (never printed/logged).
Judged SAE labels FROZEN (#1773): zero LLM/judge calls anywhere in this driver.
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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM smoke)

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import issue1482_error_analysis as EA  # noqa: E402
import issue1482_sae as S  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1895")

TASK_ID = 1895
LAYER = 19
HF_PREFIX_DEFAULT = "issue1895_subspaces"
STORE_HF_PREFIX = "issue1482_error_analysis/analysis_tensors/sae_pooled"
# Consumer keys asserted on the 1-file staging probe (plan §10 realized-keys check (c)).
STORE_CONSUMER_KEYS = (
    "row_idx",
    "ci",
    "set_tag",
    "idx_off",
    "ans_idx",
    "ans_mean",
    "h_prefix",
    "ans_all_out",
    "n_ans",
    "prefix_end",
)
LAMBDAS = np.logspace(-3, 8, 23)
PROD_TOTALS = {0: 20_000, 1: 120_000, 2: 2_000}  # set_tag -> registered row count
N_TOTAL_PROD = 964_844  # split_1482.json n_total (NEVER 963,444 = train_full)
G1B_RELDEV_MAX = 1e-6
G2A_MEDIAN_RELDEV_MAX = 5e-3
G2B_ROUTING_FACTOR = 0.1
# Per-phase disk floors (plan §9 disk row: staging 20 + scratch 27.6 + stores ~4 + slack;
# smoke floors are ~10x smaller — resume checks run BEFORE the headroom gate).
PHASE_HEADROOM_GB = {
    "stage": 30.0,
    "matrices": 40.0,
    "recon": 8.0,
    "pilot": 5.0,
    "capture_matched": 8.0,
    "fits": 12.0,
    "nulls": 6.0,
    "correlates": 2.0,
    "upload": 2.0,
}
SMOKE_HEADROOM_GB = {"stage": 10.0, "matrices": 4.0}


# ── fingerprint / resume (#952 gate-5 manifest shape) ────────────────────────────


def _code_sha() -> str:
    """git rev-parse HEAD with the SLURM/git-less degrade ladder (env -> check=False
    -> literal); a missing git yields a STABLE 'unknown' rather than crashing."""
    env = os.environ.get("EPS_GIT_SHA")
    if env:
        return env
    try:
        r = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except OSError:
        pass
    return "unknown"


def committed_split_shas() -> dict[str, str]:
    """The four committed split_1482.json shas (G1 anchors + fingerprint member)."""
    doc = json.loads(EA.COMMITTED_SPLIT_1482.read_text())
    return {
        "train_full_sha256": doc["train_full_sha256"],
        "holdout_sha256": doc["holdout"]["sha256"],
        "sae_fit_sha256": doc["sae_fit"]["sha256"],
        "sae_val_sha256": doc["sae_val"]["sha256"],
    }


def config_digest(args, *, include_path: bool) -> str:
    """sha256 over the resolved regime args (plan §4 resume-provenance block).
    ``include_path`` folds the realized Path A/B into post-pilot phases' digests."""
    regime = {
        "sae_k": int(args.sae_k),
        "k_grid": list(args.k_grid),
        "n_shells": int(args.n_shells),
        "shell_grid": list(args.shell_grid),
        "angle_draws": int(args.angle_draws),
        "boot_draws": int(args.boot_draws),
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "max_shards": int(args.max_shards),
        "holdout_cap": int(args.holdout_cap),
        "fit_cap": int(args.fit_cap),
        "val_cap": int(args.val_cap),
        "pilot_n": int(args.pilot_n),
        "tiny_model": bool(args.tiny_model),
    }
    if include_path:
        regime["path"] = resolved_path(args)
    return hashlib.sha256(json.dumps(regime, sort_keys=True).encode()).hexdigest()[:16]


def fingerprint(args, *, include_path: bool = False) -> dict:
    return {
        "code_sha": _code_sha(),
        "split_shas": committed_split_shas(),
        "sae_revision": S.SAE_REVISION,
        "config_digest": config_digest(args, include_path=include_path),
    }


def _sent_dir(args) -> Path:
    d = args.out_root / "sentinels"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_done(args, phase: str, fp: dict, extra: dict | None = None) -> None:
    C.write_json_atomic(
        _sent_dir(args) / f"{phase}.done.json",
        {
            "phase": phase,
            "fingerprint": fp,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **(extra or {}),
        },
    )


def resume_ok(args, phase: str, fp: dict) -> bool:
    """Fingerprint-validated resume: stale/missing => recompute, never bare existence."""
    p = _sent_dir(args) / f"{phase}.done.json"
    if not p.exists():
        return False
    try:
        stored = json.loads(p.read_text()).get("fingerprint")
    except (OSError, json.JSONDecodeError):
        return False
    if stored != fp:
        logger.info("[%s] stale fingerprint (stored != current) -> recompute", phase)
        return False
    return True


def _fp_str(fp: dict) -> str:
    return json.dumps(fp, sort_keys=True)


def shard_resume_ok(path: Path, fp: dict) -> bool:
    """Per-unit resume: the shard exists AND carries the current fingerprint."""
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            stored = str(z["fingerprint"])
    except Exception:  # noqa: BLE001 — any unreadable shard => recompute (fail-safe)
        return False
    return stored == _fp_str(fp)


def _headroom(args, phase: str) -> None:
    need = PHASE_HEADROOM_GB[phase]
    if args.smoke:
        need = SMOKE_HEADROOM_GB.get(phase, 2.0)
    EA._headroom(args.out_root, need, f"issue1895-{phase}")


def resolved_path(args) -> str:
    """Realized Path A/B: --force-path wins; else the pilot gate verdict."""
    if args.force_path in ("A", "B"):
        return args.force_path
    gate = args.out_eval / "pilot_gate.json"
    if not gate.exists():
        raise RuntimeError("pilot_gate.json missing — run --phase pilot before this phase")
    return json.loads(gate.read_text())["path"]


# ── store access helpers ─────────────────────────────────────────────────────────


def store_dir(args) -> Path:
    """Consumed store path; stage_hub_prefix mirrors the hub-relative path under
    dest, so dest_root/<STORE_HF_PREFIX> == consumed path (mirror-root rule)."""
    return args.out_root / "hf_dl" / STORE_HF_PREFIX


def _store_shards(args) -> list[Path]:
    shards = sorted(store_dir(args).glob("pooled_*.npz"))
    assert shards, f"no pooled shards under {store_dir(args)} — run --phase stage"
    return shards


def _shard_stem(p: Path, k: int) -> str:
    name = p.name
    assert name.startswith("pooled_") and name.endswith(f"_k{k}.npz"), name
    return name[len("pooled_") : -len(f"_k{k}.npz")]


def _dense_codes(part: dict, device: str, dict_size: int = S.DICT_SIZE) -> torch.Tensor:
    """Scatter one shard's union-index sparse codes (idx_off row COUNTS + ans_idx /
    ans_mean concatenations) into a dense (n_rows, dict_size) fp32 tensor. Batched
    scatter — no per-row python loop over features."""
    counts = torch.as_tensor(part["idx_off"].astype(np.int64), device=device)
    n = int(counts.shape[0])
    idx = torch.as_tensor(part["ans_idx"].astype(np.int64), device=device)
    val = torch.as_tensor(part["ans_mean"].astype(np.float32), device=device)
    assert int(counts.sum()) == idx.shape[0] == val.shape[0], (
        int(counts.sum()),
        idx.shape,
        val.shape,
    )
    dense = torch.zeros((n, dict_size), dtype=torch.float32, device=device)
    if idx.numel():
        row_ids = torch.repeat_interleave(torch.arange(n, device=device), counts)
        dense[row_ids, idx] = val
    return dense


def _eigh_desc(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 eigh with the cuSOLVER non-convergence CPU fallback (gotchas #1335);
    returns (eigvals desc, eigvecs desc-ordered columns)."""
    try:
        w, V = torch.linalg.eigh(A)
    except torch.linalg.LinAlgError:
        logger.warning("[eigh] cuda eigh non-convergence; CPU LAPACK fallback")
        w, V = torch.linalg.eigh(A.cpu())
        w, V = w.to(A.device), V.to(A.device)
    return torch.flip(w, dims=[0]), torch.flip(V, dims=[1])


# ── S0 stage ─────────────────────────────────────────────────────────────────────


def phase_stage(args) -> None:
    C.phase("s0_stage")
    fp = fingerprint(args)
    if resume_ok(args, "stage", fp):
        logger.info("[stage] resume: fingerprint-valid done sentinel; skip")
        return
    _headroom(args, "stage")
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    dest_root = args.out_root / "hf_dl"
    sd = store_dir(args)
    # mirror-root arithmetic assert (stage_hub_prefix dest is a MIRROR ROOT, #1774)
    assert dest_root / STORE_HF_PREFIX == sd, (dest_root, sd)
    api = HfApi()
    files = hub.list_hf_files_under_path(api, C.HF_DATA_REPO, STORE_HF_PREFIX, repo_type="dataset")
    names = sorted(f.rsplit("/", 1)[-1] for f in files if f.endswith(".npz"))
    assert names, f"no pooled shards listed under {STORE_HF_PREFIX}"
    logger.info("[stage] pooled store lists %d shards", len(names))
    # 1-file staging probe + consumer-open BEFORE the bulk pass ((h)(iv) probe,
    # plan §10) — the SAME staging helper family the bulk pass uses.
    probe_name = names[0]
    probe_path = sd / probe_name
    if not probe_path.exists():
        hub.stage_hub_file(
            C.HF_DATA_REPO, f"{STORE_HF_PREFIX}/{probe_name}", probe_path, repo_type="dataset"
        )
    with np.load(probe_path, allow_pickle=False) as z:
        missing = [k for k in STORE_CONSUMER_KEYS if k not in z.files]
        assert not missing, f"probe shard {probe_name} missing consumer keys: {missing}"
        n_probe = int(z["row_idx"].shape[0])
    logger.info("[stage] probe shard %s: %d rows, consumer keys OK", probe_name, n_probe)
    if args.smoke:
        # smoke stages the FIRST max_shards pooled shards + the MATCHING capture
        # chunks (ci-aligned by construction: pooled_{stem}_k64.npz <-> {stem}.pt)
        take = names[: args.max_shards]
        for i, name in enumerate(take):
            tgt = sd / name
            if not tgt.exists():
                hub.stage_hub_file(
                    C.HF_DATA_REPO, f"{STORE_HF_PREFIX}/{name}", tgt, repo_type="dataset"
                )
            logger.info("[stage] unit %d/%d %s staged", i + 1, len(take), name)
        stems = [_shard_stem(sd / n, int(args.sae_k)) for n in take]
        chunk_dir = args.scratch / "n1m_chunks_smoke"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for i, stem in enumerate(stems):
            tgt = chunk_dir / f"{stem}.pt"
            if not tgt.exists():
                got = Path(
                    N1M._download_chunk_with_retry(
                        C.HF_DATA_REPO, f"{EA.CAPTURE_PREFIX}/{stem}.pt", chunk_dir
                    )
                )
                if got != tgt:
                    os.replace(got, tgt)
            logger.info("[stage] unit %d/%d capture chunk %s.pt staged", i + 1, len(stems), stem)
        n_staged = len(take)
    else:
        staged = hub.stage_hub_prefix(C.HF_DATA_REPO, STORE_HF_PREFIX, dest_root)
        n_staged = len(staged)
        logger.info("[stage] bulk store stage complete: %d files", n_staged)
    # SAE weights (k64 primary + k128 pilot twin), parent-side idempotent pre-stage
    S.BatchTopKSAE.ensure_downloaded(64, args.sae_dir)
    S.BatchTopKSAE.ensure_downloaded(128, args.sae_dir)
    write_done(args, "stage", fp, {"n_store_files": n_staged, "probe_shard": probe_name})
    logger.info("[stage] done: %d store files + SAE k64/k128 staged", n_staged)


# ── S1 matrices ──────────────────────────────────────────────────────────────────


def _assemble(args):
    """Parent-code-path assembly (mirrors EA.phase_p0's namespace exactly)."""
    ns = argparse.Namespace(
        pass_b=args.pass_b,
        out_dir=args.scratch,
        manifest_from_hf=True,
        hf_prefix=EA.CAPTURE_PREFIX,
        manifest_hf_prefix=EA.N1G.HF_PREFIX,
        n1m_capture_dir=(args.scratch / "n1m_chunks_smoke") if args.smoke else None,
        fresh_stream=False,
        orig_dir=N1M.DEFAULT_ORIG_DIR,
    )
    return EA._assemble_with_ci(ns, LAYER)


def phase_matrices(args) -> None:
    C.phase("s1_matrices")
    fp = fingerprint(args)
    if resume_ok(args, "matrices", fp):
        logger.info("[matrices] resume: fingerprint-valid done sentinel; skip")
        return
    _headroom(args, "matrices")
    args.scratch.mkdir(parents=True, exist_ok=True)
    X, Y, prov, r1_train, val, test, split, new_ci = _assemble(args)
    n_total = int(X.shape[0])
    committed = json.loads(EA.COMMITTED_SPLIT_1482.read_text())
    if not args.smoke:
        # X/Y row assert: n_total == split_1482.json n_total == 964,844 (NEVER
        # 963,444 = train_full, which excludes the 1,400 pinned pass_b val/test).
        assert n_total == int(committed["n_total"]) == N_TOTAL_PROD, (
            n_total,
            committed["n_total"],
            N_TOTAL_PROD,
        )
    np.save(args.scratch / "X.npy", X)
    np.save(args.scratch / "Y.npy", Y)
    prov_u8 = (prov == "wildchat").astype(np.uint8)
    np.save(args.scratch / "prov.npy", prov_u8)
    row_ci = np.full(n_total, -1, dtype=np.int64)
    row_ci[N1M.N_PASS_B :] = new_ci
    np.save(args.scratch / "row_ci.npy", row_ci)
    pools = N1M._pool_rows(prov, r1_train, n_total, val, test)
    train_full = pools["full"]
    g1 = {"mode": "production" if not args.smoke else "smoke-informational"}
    if not args.smoke:
        # G1 (split identity): re-carve with SPLIT_SEED_1482, assert the four shas.
        lmsys_frac = len(pools["lmsys"]) / len(train_full)
        new_rows = np.arange(N1M.N_PASS_B, n_total)
        rng = np.random.default_rng(EA.SPLIT_SEED_1482)
        holdout, _ = EA._stratified_sample(rng, new_rows, prov_u8, 20_000, lmsys_frac)
        remaining = np.setdiff1d(new_rows, holdout, assume_unique=False)
        sae_fit, _ = EA._stratified_sample(rng, remaining, prov_u8, 120_000, lmsys_frac)
        remaining2 = np.setdiff1d(remaining, sae_fit, assume_unique=False)
        sae_val, _ = EA._stratified_sample(rng, remaining2, prov_u8, 2_000, lmsys_frac)
        got = {
            "train_full_sha256": EA._sha_ids(train_full),
            "holdout_sha256": EA._sha_ids(holdout),
            "sae_fit_sha256": EA._sha_ids(sae_fit),
            "sae_val_sha256": EA._sha_ids(sae_val),
        }
        exp = committed_split_shas()
        assert got == exp, f"G1 HALT: split shas diverge from committed: {got} vs {exp}"
        np.savez(
            args.scratch / "split_indices_1895.npz",
            train_full=train_full,
            holdout=holdout,
            sae_fit=sae_fit,
            sae_val=sae_val,
        )
        g1.update({"shas": got, "verdict": "PASS"})
        logger.info("[matrices] G1 PASS: four split shas match committed split_1482.json")
    else:
        # smoke: the carve gates are production-n calibrated (#1345 gate-calibration
        # rule) — the pinned pass_b val/test shas were still asserted INSIDE
        # _assemble_with_ci (pass_b is always full 5,000 rows, smoke included).
        np.savez(args.scratch / "split_indices_1895.npz", train_full=train_full)
        g1.update({"note": "smoke: 1482-carve sha asserts demoted (pass_b pins asserted)"})
    # fp64 train-cov eigenbasis Q — parent convention EXACTLY (E[yy^T] - mu mu^T,
    # divide-by-n population covariance; issue1482_error_analysis _p3_unit_aux).
    dev = args.device if args.device == "cuda" else "cpu"
    Ymm = np.load(args.scratch / "Y.npy", mmap_mode="r")
    H = Ymm.shape[1]
    A = torch.zeros((H, H), dtype=torch.float64, device=dev)
    mu_acc = torch.zeros(H, dtype=torch.float64, device=dev)
    n_acc = 0
    for s in range(0, len(train_full), args.block):
        yb = torch.as_tensor(
            np.asarray(Ymm[train_full[s : s + args.block]], dtype=np.float64), device=dev
        )
        A += yb.T @ yb
        mu_acc += yb.sum(0)
        n_acc += yb.shape[0]
    mu = mu_acc / n_acc
    A = A / n_acc - torch.outer(mu, mu)
    evals, evecs = _eigh_desc(A)
    evals_np = evals.cpu().numpy()
    Q = evecs.cpu().numpy().astype(np.float32)
    banked = np.asarray(json.loads(EA.COMMITTED_PERDIR_PCA.read_text())["eigvals_top"], np.float64)
    reldev = float(
        np.max(np.abs(evals_np[: len(banked)] - banked) / np.maximum(np.abs(banked), 1e-12))
    )
    g1b = {"top256_reldev": reldev}
    if not args.smoke:
        assert reldev < G1B_RELDEV_MAX, f"G1b HALT: eigval rel dev {reldev:.3e} >= 1e-6"
        g1b["verdict"] = "PASS"
        logger.info("[matrices] G1b PASS: top-256 eigval rel dev %.3e", reldev)
    else:
        g1b["verdict"] = "smoke-informational"
        logger.info("[matrices] G1b (smoke, informational): rel dev %.3e", reldev)
    np.savez(args.out_root / "q_basis.npz", Q=Q, eigvals=evals_np)
    write_done(
        args,
        "matrices",
        fp,
        {"n_total": n_total, "n_train_full": int(len(train_full)), "g1": g1, "g1b": g1b},
    )
    logger.info("[matrices] done: X/Y (%d x %d), Q basis persisted", n_total, H)


# ── S2 recon (r_bar from banked codes; zero encode passes) ──────────────────────


def _load_sae(args, k: int) -> S.BatchTopKSAE:
    dev = args.device if args.device == "cuda" else "cpu"
    return S.BatchTopKSAE.load(k=k, device=dev, cache_dir=args.sae_dir)


def _registry_path(args) -> Path:
    return args.out_root / "registry.npz"


def phase_recon(args) -> None:
    C.phase("s2_recon")
    fp = fingerprint(args)
    if resume_ok(args, "recon", fp):
        logger.info("[recon] resume: fingerprint-valid done sentinel; skip")
        return
    _headroom(args, "recon")
    dev = args.device if args.device == "cuda" else "cpu"
    sae = _load_sae(args, int(args.sae_k))
    row_ci = np.load(args.scratch / "row_ci.npy")
    pos_by_ci = {int(c): i for i, c in enumerate(row_ci) if c >= 0}
    shards = _store_shards(args)
    rbar_dir = args.out_root / "rbar"
    rbar_dir.mkdir(parents=True, exist_ok=True)
    wj_sum = torch.zeros(S.DICT_SIZE, dtype=torch.float64)
    n_fit_seen = 0
    reg: dict[str, list] = {k2: [] for k2 in ("pos", "ci", "set_tag", "ans_all_out", "chunk")}
    seen_row_idx: list[np.ndarray] = []
    tag_totals = {0: 0, 1: 0, 2: 0}
    t0 = time.time()
    for i, shard in enumerate(shards):
        stem = _shard_stem(shard, int(args.sae_k))
        with np.load(shard, allow_pickle=False) as z:
            part = {k2: z[k2] for k2 in z.files}
        cis = part["ci"].astype(np.int64)
        tags = part["set_tag"].astype(np.int64)
        seen_row_idx.append(part["row_idx"].astype(np.int64))
        for t in (0, 1, 2):
            tag_totals[t] += int((tags == t).sum())
        # ci-join against THIS assembly (production: identity with store row_idx)
        pos = np.array([pos_by_ci.get(int(c), -1) for c in cis], dtype=np.int64)
        assert (pos >= 0).all(), (
            f"[recon] shard {stem}: {int((pos < 0).sum())} store rows have cis absent "
            f"from the assembly — store/assembly join broken"
        )
        if not args.smoke:
            assert np.array_equal(pos, part["row_idx"].astype(np.int64)), (
                f"[recon] shard {stem}: ci-join positions != store row_idx (join drift)"
            )
        out_path = rbar_dir / f"rbar_{stem}.npz"
        dense = _dense_codes(part, dev)
        fit_mask = torch.as_tensor(tags == 1, device=dev)
        if bool(fit_mask.any()):
            wj_sum += dense[fit_mask].sum(0, dtype=torch.float64).cpu()
            n_fit_seen += int(fit_mask.sum())
        if not shard_resume_ok(out_path, fp):
            rbar = (dense @ sae.w_dec.T + sae.b_dec).cpu().numpy().astype(np.float16)
            tmp = out_path.parent / f".tmp_{out_path.name}"
            np.savez(
                tmp,
                pos=pos,
                ci=cis,
                rbar=rbar,
                ans_all_out=part["ans_all_out"].astype(np.int8),
                fingerprint=np.array(_fp_str(fp)),
            )
            os.replace(tmp, out_path)
        reg["pos"].append(pos)
        reg["ci"].append(cis)
        reg["set_tag"].append(tags)
        reg["ans_all_out"].append(part["ans_all_out"].astype(np.int64))
        reg["chunk"].extend([stem] * len(cis))
        if (i + 1) % 50 == 0 or i + 1 == len(shards):
            logger.info(
                "[recon] unit %d/%d %s elapsed=%.0fs", i + 1, len(shards), stem, time.time() - t0
            )
    pos_all = np.concatenate(reg["pos"])
    ci_all = np.concatenate(reg["ci"])
    tag_all = np.concatenate(reg["set_tag"])
    aao_all = np.concatenate(reg["ans_all_out"])
    chunk_all = np.asarray(reg["chunk"])
    # full-grain structural asserts (plan §4-S2; the arbiter over any prose count)
    uniq, cnt = np.unique(pos_all, return_counts=True)
    dup = uniq[cnt > 1]
    if len(dup):
        mism = args.out_eval / "s2_coverage_mismatch.json"
        C.write_json_atomic(mism, {"duplicate_positions": dup[:200].tolist()})
        raise AssertionError(f"[recon] {len(dup)} duplicate store rows — see {mism}")
    if not args.smoke:
        srx = np.sort(np.concatenate(seen_row_idx))
        idx = np.load(args.scratch / "split_indices_1895.npz")
        registered = np.sort(np.concatenate([idx["holdout"], idx["sae_fit"], idx["sae_val"]]))
        if tag_totals != PROD_TOTALS or not np.array_equal(srx, registered):
            mism = args.out_eval / "s2_coverage_mismatch.json"
            missing = np.setdiff1d(registered, srx)
            extra = np.setdiff1d(srx, registered)
            C.write_json_atomic(
                mism,
                {
                    "tag_totals": {str(k2): v for k2, v in tag_totals.items()},
                    "expected_totals": {str(k2): v for k2, v in PROD_TOTALS.items()},
                    "missing_registered_rows": missing[:1000].tolist(),
                    "extra_store_rows": extra[:1000].tolist(),
                },
            )
            raise AssertionError(
                f"[recon] HALT: store coverage mismatch (totals {tag_totals} vs "
                f"{PROD_TOTALS}; missing={len(missing)} extra={len(extra)}) — see {mism} "
                f"(record-correction finding on #1482 if the parent prose disagrees)"
            )
    else:
        # smoke floors derived from downstream consumers (kNN ks<=10 needs holdout
        # >= 12; MLP batch floor 8 needs fit >= 16; variance needs val >= 2)
        assert tag_totals[0] >= 12 and tag_totals[1] >= 16 and tag_totals[2] >= 2, tag_totals
    n_excl = int((aao_all == 1).sum())
    keep = aao_all == 0  # ans_all_out rows excluded from ALL downstream targets
    order = np.argsort(pos_all[keep])
    np.savez(
        _registry_path(args),
        pos=pos_all[keep][order],
        ci=ci_all[keep][order],
        set_tag=tag_all[keep][order],
        chunk=chunk_all[keep][order],
        fingerprint=np.array(_fp_str(fp)),
    )
    wj = (wj_sum / max(1, n_fit_seen)).numpy()
    np.savez(args.out_root / "wj.npz", wj=wj, n_fit=n_fit_seen)
    _fve_profiles(args, fp)
    write_done(
        args,
        "recon",
        fp,
        {
            "tag_totals": {str(k2): v for k2, v in tag_totals.items()},
            "n_ans_all_out_excluded": n_excl,
        },
    )
    logger.info("[recon] done: %d rows (excluded ans_all_out=%d)", len(pos_all), n_excl)


def _gather_rbar(args, positions: np.ndarray) -> np.ndarray:
    """Gather r_bar rows (fp32) for the given X-row positions from the rbar shards."""
    want = {int(p): i for i, p in enumerate(positions)}
    out = np.zeros((len(positions), S.ACT_DIM), dtype=np.float32)
    got = np.zeros(len(positions), dtype=bool)
    for shard in sorted((args.out_root / "rbar").glob("rbar_*.npz")):
        with np.load(shard, allow_pickle=False) as z:
            pos = z["pos"].astype(np.int64)
            hit = np.array([want.get(int(p), -1) for p in pos], dtype=np.int64)
            m = hit >= 0
            if m.any():
                out[hit[m]] = z["rbar"][m].astype(np.float32)
                got[hit[m]] = True
    assert got.all(), f"[rbar-gather] {int((~got).sum())} positions missing from rbar shards"
    return out


def _gather_store_field(args, positions: np.ndarray, field: str) -> np.ndarray:
    """Gather a dense per-row field (e.g. h_prefix) from the POOLED store shards."""
    want = {int(p): i for i, p in enumerate(positions)}
    out = None
    got = np.zeros(len(positions), dtype=bool)
    row_ci = np.load(args.scratch / "row_ci.npy")
    pos_by_ci = {int(c): i for i, c in enumerate(row_ci) if c >= 0}
    for shard in _store_shards(args):
        with np.load(shard, allow_pickle=False) as z:
            cis = z["ci"].astype(np.int64)
            arr = z[field]
            pos = np.array([pos_by_ci.get(int(c), -1) for c in cis], dtype=np.int64)
            hit = np.array([want.get(int(p), -1) for p in pos], dtype=np.int64)
            m = hit >= 0
            if m.any():
                if out is None:
                    out = np.zeros((len(positions), arr.shape[1]), dtype=np.float32)
                out[hit[m]] = arr[m].astype(np.float32)
                got[hit[m]] = True
    assert out is not None and got.all(), (
        f"[store-gather:{field}] {int((~got).sum())} positions missing"
    )
    return out


def _holdout_targets(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(positions, vA, rbar, ebar) for the registry holdout rows (fp32).
    Under Path B (post-pilot phases), vA is the mask-matched v_bar_mask."""
    reg = np.load(_registry_path(args), allow_pickle=False)
    hold = reg["pos"][reg["set_tag"] == 0]
    Ymm = np.load(args.scratch / "Y.npy", mmap_mode="r")
    vA = np.asarray(Ymm[hold], dtype=np.float32)
    rb = _gather_rbar(args, hold)
    return hold, vA, rb, vA - rb


def _fve_profiles(args, fp: dict) -> None:
    """Holdout per-direction FVE_u (all 3,584 u) + per-feature FVE_j along unit
    decoder columns (active features), batched GEMMs — plan §4-S2."""
    dev = args.device if args.device == "cuda" else "cpu"
    hold, vA, rb, eb = _holdout_targets(args)
    with np.load(args.out_root / "q_basis.npz") as z:
        Q = z["Q"].astype(np.float32)
    tv = torch.as_tensor(vA, device=dev)
    te_ = torch.as_tensor(eb, device=dev)
    Qt = torch.as_tensor(Q, device=dev)
    Vp = tv @ Qt
    Ep = te_ @ Qt
    var_v = Vp.var(0, unbiased=True)
    var_e = Ep.var(0, unbiased=True)
    fve_u = (1.0 - var_e / torch.clamp(var_v, min=1e-12)).cpu().numpy()
    # per-feature: unit decoder columns of the ACTIVE feature set (banked feat_ids)
    banked = np.load(
        PROJECT_ROOT
        / "eval_results"
        / "issue_1482"
        / "sae_perfeature"
        / "sae_dense_in__mean__ridge.npz"
    )
    feat_ids = banked["feat_ids"].astype(np.int64)
    sae = _load_sae(args, int(args.sae_k))
    D = sae.w_dec[:, torch.as_tensor(feat_ids, device=sae.w_dec.device)]
    D = D / torch.clamp(D.norm(dim=0, keepdim=True), min=1e-12)
    D = D.to(dev)
    pv = tv @ D
    pe = te_ @ D
    varv_j = pv.var(0, unbiased=True)
    fve_j = (1.0 - pe.var(0, unbiased=True) / torch.clamp(varv_j, min=1e-12)).cpu().numpy()
    np.savez(
        args.out_root / "fve_profiles.npz",
        fve_u=fve_u,
        fve_j=fve_j,
        varv_j=varv_j.cpu().numpy(),
        feat_ids=feat_ids,
        holdout_pos=hold,
        fingerprint=np.array(_fp_str(fp)),
    )
    C.write_json_atomic(
        args.out_eval / "recon_summary.json",
        {
            "n_holdout": int(len(hold)),
            "fve_u_top16_mean": float(np.mean(fve_u[:16])),
            "fve_u_median": float(np.median(fve_u)),
            "fve_j_median": float(np.median(fve_j)),
            "n_active_features": int(len(feat_ids)),
            **C.reproducibility_metadata(),
        },
    )


# ── S3 pilot (G2a identity + G2b mask-mismatch routing) ──────────────────────────


def _answer_mask(h: torch.Tensor, context_end: int) -> tuple[torch.Tensor, bool]:
    """The _row_features reference answer-token mask (BOS strip + outlier drop),
    with the same unmasked fallback + flag when the mask empties the answer."""
    keep = S.token_inlier_mask(h)
    keep[: min(S.BOS_OFFSET, keep.shape[0])] = False
    ans_keep = keep[context_end + 1 :]
    h_ans = h[context_end + 1 :]
    all_out = bool(h_ans.shape[0] > 0 and int(ans_keep.sum()) == 0)
    return (h_ans if all_out else h_ans[ans_keep]), all_out


def phase_pilot(args) -> None:
    C.phase("s3_pilot")
    fp = fingerprint(args)
    if resume_ok(args, "pilot", fp):
        logger.info("[pilot] resume: fingerprint-valid done sentinel; skip")
        return
    _headroom(args, "pilot")
    reg = np.load(_registry_path(args), allow_pickle=False)
    hold = reg["pos"][reg["set_tag"] == 0]
    chunks_hold = reg["chunk"][reg["set_tag"] == 0]
    prov_u8 = np.load(args.scratch / "prov.npy")
    row_ci = np.load(args.scratch / "row_ci.npy")
    lmsys_frac = float((prov_u8[hold] == 0).mean())
    rng = np.random.default_rng(args.seed)
    pilot_pos, diag = EA._stratified_sample(rng, hold, prov_u8, args.pilot_n, lmsys_frac)
    needed_ci = {int(row_ci[p]): int(p) for p in pilot_pos}
    stems = sorted(
        {str(chunks_hold[i]) for i, p in enumerate(hold) if int(p) in set(pilot_pos.tolist())}
    )
    names = [f"{s2}.json" for s2 in stems]
    model, tok = EA._load_model_tok(args)
    sae64 = _load_sae(args, 64)
    sae128 = _load_sae(args, 128)
    prefix_chars = EA._prefix_char_len(tok)
    rows_out: dict[str, list] = {
        k2: [] for k2 in ("pos", "ci", "vmask", "r_exact", "r128_exact", "all_out")
    }
    t0 = time.time()
    n_tok = 0
    for name, keep in EA._iter_needed_rows(args, names, needed_ci):
        batch_rows = []
        for row_pos, ci, prompt, response in keep:
            tk = EA._tokenize_row(tok, prompt, response, prefix_chars)
            if tk is None:
                continue
            full_ids, prefix_end, context_end, n_ans, seam = tk
            batch_rows.append((row_pos, ci, full_ids, prefix_end, context_end, n_ans, seam))
        batch_rows.sort(key=lambda r: len(r[2]))
        for s2 in range(0, len(batch_rows), args.gen_batch):
            batch = batch_rows[s2 : s2 + args.gen_batch]
            caps = EA._batched_capture(model, tok, batch, (LAYER,), args.device)
            for (row_pos, ci, full_ids, _pe, context_end, _na, _sm), cap in zip(
                batch, caps, strict=True
            ):
                h = cap[LAYER]
                n_tok += len(full_ids)
                h_ans, all_out = _answer_mask(h, context_end)
                vmask = h_ans.mean(0)
                f64 = sae64.encode(h_ans)
                r_exact = sae64.decode(f64).mean(0).cpu()
                f128 = sae128.encode(h_ans)
                r128 = sae128.decode(f128).mean(0).cpu()
                rows_out["pos"].append(row_pos)
                rows_out["ci"].append(ci)
                rows_out["vmask"].append(vmask.numpy().astype(np.float16))
                rows_out["r_exact"].append(r_exact.numpy().astype(np.float16))
                rows_out["r128_exact"].append(r128.numpy().astype(np.float16))
                rows_out["all_out"].append(int(all_out))
        logger.info(
            "[pilot] chunk %s: %d rows total elapsed=%.0fs",
            name,
            len(rows_out["pos"]),
            time.time() - t0,
        )
    n = len(rows_out["pos"])
    assert n >= 2, f"[pilot] only {n} captured rows"
    pos = np.asarray(rows_out["pos"], np.int64)
    vmask = np.stack(rows_out["vmask"]).astype(np.float32)
    r_exact = np.stack(rows_out["r_exact"]).astype(np.float32)
    r128 = np.stack(rows_out["r128_exact"]).astype(np.float32)
    tokens_per_s = n_tok / max(1e-9, time.time() - t0)
    # G2a: banked-code r_bar vs exact per-token reconstruction mean
    rb_banked = _gather_rbar(args, pos)
    rel = np.linalg.norm(r_exact - rb_banked, axis=1) / np.maximum(
        np.linalg.norm(r_exact, axis=1), 1e-9
    )
    g2a_median = float(np.median(rel))
    # G2b: mask-mismatch share m vs SAE-error share s (vs holdout-centered variance)
    Ymm = np.load(args.scratch / "Y.npy", mmap_mode="r")
    hold_all = reg["pos"][reg["set_tag"] == 0]
    mu_hold = np.asarray(Ymm[hold_all], dtype=np.float64).mean(0).astype(np.float32)
    vA_pilot = np.asarray(Ymm[pos], dtype=np.float32)
    denom = np.sum((vA_pilot - mu_hold) ** 2, axis=1)
    m_share = np.sum((vA_pilot - vmask) ** 2, axis=1) / np.maximum(denom, 1e-9)
    s_share = np.sum((vmask - r_exact) ** 2, axis=1) / np.maximum(denom, 1e-9)
    m_med, s_med = float(np.median(m_share)), float(np.median(s_share))
    path = "A" if m_med <= G2B_ROUTING_FACTOR * s_med else "B"
    gates_informational = bool(args.tiny_model)
    if not gates_informational:
        assert g2a_median < G2A_MEDIAN_RELDEV_MAX, (
            f"G2a HALT: median rel dev {g2a_median:.4e} >= {G2A_MEDIAN_RELDEV_MAX} — the "
            f"zero-GPU r_bar is not the on-distribution object"
        )
    else:
        logger.info("[pilot] tiny-model smoke: G2a/G2b demoted to informational")
        path = args.force_path if args.force_path in ("A", "B") else "A"
    # k64-vs-k128 per-direction FVE rank correlation on the pilot rows (pure-e_bar)
    with np.load(args.out_root / "q_basis.npz") as z:
        Q = z["Q"].astype(np.float32)
    e64p = (vmask - r_exact) @ Q
    e128p = (vmask - r128) @ Q
    varv = (vmask @ Q).var(0) + 1e-12
    fve64 = 1.0 - e64p.var(0) / varv
    fve128 = 1.0 - e128p.var(0) / varv
    rk64 = EA._midrank(fve64[:, None])[:, 0]
    rk128 = EA._midrank(fve128[:, None])[:, 0]
    k_twin_rho = float(np.corrcoef(rk64, rk128)[0, 1])
    (args.out_root / "pilot").mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_root / "pilot" / "pilot_rows.npz",
        pos=pos,
        vmask=vmask.astype(np.float16),
        r_exact=r_exact.astype(np.float16),
        r128_exact=r128.astype(np.float16),
        all_out=np.asarray(rows_out["all_out"], np.int8),
        fingerprint=np.array(_fp_str(fp)),
    )
    C.write_json_atomic(
        args.out_eval / "pilot_gate.json",
        {
            "path": path,
            "gates_informational": gates_informational,
            "n_pilot": n,
            "pilot_diag": diag,
            "g2a_median_reldev": g2a_median,
            "g2a_p90_reldev": float(np.quantile(rel, 0.9)),
            "g2b_m_median": m_med,
            "g2b_s_median": s_med,
            "g2b_m_quantiles": np.quantile(m_share, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
            "g2b_s_quantiles": np.quantile(s_share, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
            "k128_fve_rank_rho": k_twin_rho,
            "tokens_per_s": tokens_per_s,
            **C.reproducibility_metadata(),
        },
    )
    write_done(args, "pilot", fp, {"path": path, "n_pilot": n})
    logger.info(
        "[pilot] done: Path %s (G2a med=%.2e, m=%.4f s=%.4f, tok/s=%.0f)",
        path,
        g2a_median,
        m_med,
        s_med,
        tokens_per_s,
    )


# ── S3b capture-matched (Path B only) ────────────────────────────────────────────


def phase_capture_matched(args) -> None:
    C.phase("s3b_capture_matched")
    if resolved_path(args) != "B":
        logger.info("[capture-matched] Path A resolved — phase not needed; skip")
        return
    fp = fingerprint(args, include_path=True)
    if resume_ok(args, "capture_matched", fp):
        logger.info("[capture-matched] resume: fingerprint-valid done sentinel; skip")
        return
    _headroom(args, "capture_matched")
    reg = np.load(_registry_path(args), allow_pickle=False)
    row_ci = np.load(args.scratch / "row_ci.npy")
    # capture exactly the S4 design's selected sets (production: the full
    # registered 142k; smoke: the seeded caps — shared helper keeps them equal)
    tr_pos, va_pos, te_pos = _selected_sets(args)
    sel = set(np.concatenate([tr_pos, va_pos, te_pos]).tolist())
    needed_ci = {int(row_ci[p]): int(p) for p in reg["pos"] if int(p) in sel}
    keep_chunk = np.array([int(p) in sel for p in reg["pos"].astype(np.int64)], dtype=bool)
    stems = sorted({str(c) for c in reg["chunk"][keep_chunk]})
    model, tok = EA._load_model_tok(args)
    prefix_chars = EA._prefix_char_len(tok)
    vdir = args.out_root / "vmask"
    vdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done_units = 0
    for name, keep in EA._iter_needed_rows(args, [f"{s2}.json" for s2 in stems], needed_ci):
        stem = Path(name).stem
        out_path = vdir / f"vmask_{stem}.npz"
        done_units += 1
        if shard_resume_ok(out_path, fp):
            continue
        batch_rows = []
        for row_pos, ci, prompt, response in keep:
            tk = EA._tokenize_row(tok, prompt, response, prefix_chars)
            if tk is None:
                continue
            full_ids, prefix_end, context_end, n_ans, seam = tk
            batch_rows.append((row_pos, ci, full_ids, prefix_end, context_end, n_ans, seam))
        batch_rows.sort(key=lambda r: len(r[2]))
        rec: dict[str, list] = {"pos": [], "ci": [], "vmask": [], "all_out": []}
        for s2 in range(0, len(batch_rows), args.gen_batch):
            batch = batch_rows[s2 : s2 + args.gen_batch]
            caps = EA._batched_capture(model, tok, batch, (LAYER,), args.device)
            for (row_pos, ci, _fi, _pe, context_end, _na, _sm), cap in zip(
                batch, caps, strict=True
            ):
                h_ans, all_out = _answer_mask(cap[LAYER], context_end)
                rec["pos"].append(row_pos)
                rec["ci"].append(ci)
                rec["vmask"].append(h_ans.mean(0).numpy().astype(np.float16))
                rec["all_out"].append(int(all_out))
        tmp = out_path.parent / f".tmp_{out_path.name}"
        np.savez(
            tmp,
            pos=np.asarray(rec["pos"], np.int64),
            ci=np.asarray(rec["ci"], np.int64),
            vmask=np.stack(rec["vmask"]) if rec["vmask"] else np.empty((0, S.ACT_DIM), np.float16),
            all_out=np.asarray(rec["all_out"], np.int8),
            fingerprint=np.array(_fp_str(fp)),
        )
        os.replace(tmp, out_path)
        logger.info(
            "[capture-matched] unit %d/%d %s (%d rows) elapsed=%.0fs",
            done_units,
            len(stems),
            stem,
            len(rec["pos"]),
            time.time() - t0,
        )
    # critic requirement: upload the vmask store at END of S3b (~1.0 GB fp16) —
    # never lean on the GCP crash trap for a regeneration-costly store
    _upload_dir_failloud(args, vdir, f"{_hf_prefix(args)}/analysis_tensors/vmask")
    write_done(args, "capture_matched", fp, {"n_units": done_units})
    logger.info("[capture-matched] done: %d units captured + uploaded", done_units)


def _gather_vmask(args, positions: np.ndarray) -> np.ndarray:
    """Gather v_bar_mask rows (fp32) for positions from the S3b vmask shards."""
    want = {int(p): i for i, p in enumerate(positions)}
    out = np.zeros((len(positions), S.ACT_DIM), dtype=np.float32)
    got = np.zeros(len(positions), dtype=bool)
    for shard in sorted((args.out_root / "vmask").glob("vmask_*.npz")):
        with np.load(shard, allow_pickle=False) as z:
            pos = z["pos"].astype(np.int64)
            hit = np.array([want.get(int(p), -1) for p in pos], dtype=np.int64)
            m = hit >= 0
            if m.any():
                out[hit[m]] = z["vmask"][m].astype(np.float32)
                got[hit[m]] = True
    assert got.all(), f"[vmask-gather] {int((~got).sum())} positions missing"
    return out


# ── S4 fits (shared-Gram multi-target ridge; both arms; baselines; MLP twins) ────


def _selected_sets(args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Registry X-row POSITIONS of the (fit, val, holdout) sets. Production:
    the full registered sets. Smoke: seeded caps (the smoke slice) — shared by
    S3b (capture set) and S4 (design) so Path B's vmask store always covers the
    fit design exactly."""
    reg = np.load(_registry_path(args), allow_pickle=False)
    pos = reg["pos"].astype(np.int64)
    tags = reg["set_tag"].astype(np.int64)
    tr = pos[tags == 1]
    va = pos[tags == 2]
    te = pos[tags == 0]
    if args.smoke:
        rng = np.random.default_rng(args.seed)
        tr = np.sort(rng.choice(tr, size=min(args.fit_cap, len(tr)), replace=False))
        va = np.sort(rng.choice(va, size=min(args.val_cap, len(va)), replace=False))
        te = np.sort(rng.choice(te, size=min(args.holdout_cap, len(te)), replace=False))
    return tr, va, te


def _build_design(args) -> dict:
    """Selected-set design + targets for the fit battery, ordered [tr | va | te].
    Under Path B the raw-answer target vA is replaced by the mask-matched
    v_bar_mask (plan §4-S3b)."""
    tr_pos, va_pos, te_pos = _selected_sets(args)
    pos_sel = np.concatenate([tr_pos, va_pos, te_pos])
    Xmm = np.load(args.scratch / "X.npy", mmap_mode="r")
    Ymm = np.load(args.scratch / "Y.npy", mmap_mode="r")
    X_sub = np.asarray(Xmm[pos_sel], dtype=np.float32)
    if resolved_path(args) == "B":
        vA = _gather_vmask(args, pos_sel)
    else:
        vA = np.asarray(Ymm[pos_sel], dtype=np.float32)
    rb = _gather_rbar(args, pos_sel)
    eb = vA - rb
    tr = np.arange(len(tr_pos))
    va = np.arange(len(tr_pos), len(tr_pos) + len(va_pos))
    te = np.arange(len(tr_pos) + len(va_pos), len(pos_sel))
    if not args.smoke:
        # estimator validity (#1701/#1887): every production ridge fit has
        # n_train = 120,000 - excluded >= d = 3,584; lambda is VAL-selected (no
        # GCV anywhere, so the dof-cap machinery never arises). The smoke caps
        # in _selected_sets deliberately run n_train < d — a stated
        # regularization-limit smoke shape (the #1701 exemption), demoted from
        # any read.
        assert len(tr) >= X_sub.shape[1], (len(tr), X_sub.shape[1])
    prov_u8 = np.load(args.scratch / "prov.npy")
    return {
        "pos": pos_sel,
        "X": X_sub,
        "targets": {"vA": vA, "rbar": rb, "ebar": eb},
        "tr": tr,
        "va": va,
        "te": te,
        "prov_te": prov_u8[te_pos],
    }


def _perdir_r2(pred: np.ndarray, true: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Per-direction holdout R2 on the Q basis — the parent convention
    (_per_feature_metrics over projections; ss_tot on the te set's own mean)."""
    return EA._per_feature_metrics(pred @ Q, true @ Q)["r2"]


def _cell_reads(name: str, pred: np.ndarray, true: np.ndarray, X_te, X_tr, T_tr) -> dict:
    """Pooled R2 + the two standing mapping-baseline reads (identity+learned-bias
    R2, kNN retrieval of the map AND the baseline; chance = k/n_pool stated)."""
    out = {"pooled_r2": float(PR._pooled_r2(pred, true))}
    ib = identity_bias_predict(X_tr, T_tr, X_te)
    out["identity_bias_r2"] = float(PR._pooled_r2(ib, true))
    for metric in ("euclidean", "cosine"):
        out[f"knn_map_{metric}"] = knn_retrieval(pred, true, metric=metric)
        out[f"knn_identity_bias_{metric}"] = knn_retrieval(ib, true, metric=metric)
    logger.info(
        "[fits] %s pooled R2=%.4f (ib=%.4f)", name, out["pooled_r2"], out["identity_bias_r2"]
    )
    return out


def phase_fits(args) -> None:
    C.phase("s4_fits")
    fp = fingerprint(args, include_path=True)
    if resume_ok(args, "fits", fp):
        logger.info("[fits] resume: fingerprint-valid done sentinel; skip")
        return
    _headroom(args, "fits")
    dev = args.device if args.device == "cuda" else "cpu"
    d = _build_design(args)
    X, targets, tr, va, te = d["X"], d["targets"], d["tr"], d["va"], d["te"]
    with np.load(args.out_root / "q_basis.npz") as z:
        Q = z["Q"].astype(np.float32)
    summary: dict = {
        "n_tr": int(len(tr)),
        "n_va": int(len(va)),
        "n_te": int(len(te)),
        "path": resolved_path(args),
        "lambda_grid": {"lo": -3, "hi": 8, "n": 23, "space": "logspace"},
        "knn_chance_at_1": 1.0 / max(1, len(te)),
        "cells": {},
    }
    profiles: dict[str, np.ndarray] = {}
    # ONE shared-factorization multi-target ridge call (plan-registered:
    # issue1482_error_analysis._shared_gram_ridge_multi, line ~1385)
    ctx = EA._shared_gram_ridge_multi(X, targets, tr, va, te, LAMBDAS, dev, args.block)
    preds_te: dict[str, np.ndarray] = {}
    for tname, (pt, meta) in ctx.items():
        cell = f"t_{tname}_ctx"
        true = targets[tname][te]
        summary["cells"][cell] = {
            **meta,
            **_cell_reads(cell, pt, true, X[te], X[tr], targets[tname][tr]),
        }
        profiles[f"r2u__{cell}"] = _perdir_r2(pt, true, Q)
        preds_te[tname] = pt
    # shared-lambda decomposition arm: refit rbar/ebar at T1's selected lambda
    # (one extra factorization — numerically identical standardizer/eigh; the
    # additive identity vhat_A = rhat + ehat holds exactly at a shared lambda)
    lam1 = float(summary["cells"]["t_vA_ctx"]["selected_lambda"])
    Ycat = np.concatenate([targets["rbar"], targets["ebar"]], axis=1)
    fac = N1M._ridge_factorize(X, Ycat, tr, dev, args.block)
    both = N1M._ridge_predict_one(X, te, fac, lam1, dev, args.block)
    hdim = targets["vA"].shape[1]
    r_hat, e_hat = both[:, :hdim], both[:, hdim:]
    # vhat_A at lam1 IS the T1 cell's own prediction (lam1 = T1's selected lambda;
    # same standardizer + eigh — numerically identical factorization)
    v_hat_l1 = preds_te["vA"]
    add_dev = float(np.linalg.norm(r_hat + e_hat - v_hat_l1) / max(1e-9, np.linalg.norm(v_hat_l1)))
    err = targets["vA"][te] - v_hat_l1
    err_r = targets["rbar"][te] - r_hat
    err_e = targets["ebar"][te] - e_hat
    ss = float(np.sum(err**2))
    summary["shared_lambda_decomposition"] = {
        "lambda": lam1,
        "additivity_reldev": add_dev,
        "share_sae_representable": float(np.sum(err_r**2) / max(1e-12, ss)),
        "share_residual": float(np.sum(err_e**2) / max(1e-12, ss)),
        "share_cross": float(2.0 * np.sum(err_r * err_e) / max(1e-12, ss)),
        "pooled_r2_vA_at_shared_lambda": float(PR._pooled_r2(v_hat_l1, targets["vA"][te])),
    }
    # prefix arm (both-arms standing rule; registered null on this single-turn
    # corpus — parent-verified constant template prefix, min cos 1.000)
    HP = _gather_store_field(args, d["pos"], "h_prefix")
    pre = EA._shared_gram_ridge_multi(HP, targets, tr, va, te, LAMBDAS, dev, args.block)
    for tname, (pt, meta) in pre.items():
        cell = f"t_{tname}_prefix"
        true = targets[tname][te]
        summary["cells"][cell] = {
            **meta,
            **_cell_reads(cell, pt, true, HP[te], HP[tr], targets[tname][tr]),
        }
        profiles[f"r2u__{cell}"] = _perdir_r2(pt, true, Q)
    # MLP twins on r_bar / e_bar (context arm; parent recipe w8192, lr 3e-4,
    # batch 4096, fit seed 0, internal-val early stop; Engels et al. 2410.14670)
    for tname in ("rbar", "ebar"):
        pt, meta = N1M._fit_mlp_minibatch(
            X,
            targets[tname],
            tr,
            te,
            N1M.MLP_W_PROTOCOL,
            3e-4,
            3 if args.smoke else EA.F.MLP_MAX_EPOCHS,
            min(N1M.MLP_BATCH, max(8, len(tr))),
            0,
            torch.device(dev),
        )
        cell = f"t_{tname}_ctx_mlp"
        true = targets[tname][te]
        summary["cells"][cell] = {
            "epochs_ran": meta["epochs_ran"],
            **_cell_reads(cell, pt, true, X[te], X[tr], targets[tname][tr]),
        }
        profiles[f"r2u__{cell}"] = _perdir_r2(pt, true, Q)
    # per-corpus splits of every headline pooled read (plan §6 transfer robustness)
    prov_te = d["prov_te"]
    per_corpus: dict = {}
    for tname, pt in preds_te.items():
        true = targets[tname][te]
        for label, mask in (("lmsys", prov_te == 0), ("wildchat", prov_te == 1)):
            if int(mask.sum()) >= 2:
                per_corpus[f"t_{tname}_ctx__{label}"] = float(PR._pooled_r2(pt[mask], true[mask]))
    summary["per_corpus_pooled_r2"] = per_corpus
    # cross-check (plan assumption 7): matched-refit profile vs banked ridge profile
    banked_pca = json.loads(EA.COMMITTED_PERDIR_PCA.read_text())
    g_matched = profiles["r2u__t_vA_ctx"]
    banked_r2 = np.asarray(banked_pca["per_direction_r2"]["ridge"], np.float64)
    ntop = min(len(banked_r2), len(g_matched))
    rk_a = EA._midrank(g_matched[:ntop, None])[:, 0]
    rk_b = EA._midrank(banked_r2[:ntop, None])[:, 0]
    summary["matched_vs_banked_profile_rho_top256"] = float(np.corrcoef(rk_a, rk_b)[0, 1])
    # per-context per-direction matrices (fp16) — bootstrap re-reductions stay pure
    proj = {
        "V_proj": targets["vA"][te] @ Q,
        "R_proj": targets["rbar"][te] @ Q,
        "E_proj": targets["ebar"][te] @ Q,
        "R1_proj": (targets["vA"][te] - preds_te["vA"]) @ Q,
        "R2_proj": (targets["rbar"][te] - preds_te["rbar"]) @ Q,
        "R3_proj": (targets["ebar"][te] - preds_te["ebar"]) @ Q,
    }
    np.savez(
        args.out_root / "percontext_proj.npz",
        **{k2: v.astype(np.float16) for k2, v in proj.items()},
        te_pos=d["pos"][te],
        prov_te=prov_te.astype(np.int8),
        fingerprint=np.array(_fp_str(fp)),
    )
    # energy profiles (observed) + the primary plug-in profile g(u) = matched refit
    np.savez(
        args.out_root / "perdirection_profiles.npz",
        **{k2: v.astype(np.float32) for k2, v in profiles.items()},
        eigvals=np.load(args.out_root / "q_basis.npz")["eigvals"],
        fingerprint=np.array(_fp_str(fp)),
    )
    C.write_json_atomic(
        args.out_eval / "fits_summary.json", {**summary, **C.reproducibility_metadata()}
    )
    write_done(args, "fits", fp, {"cells": sorted(summary["cells"])})
    logger.info("[fits] done: %d cells", len(summary["cells"]))


# ── S5 nulls (angle battery + within-shell rotation null + H3 plug-in) ──────────


def shell_partition(eigvals: np.ndarray, n_shells: int) -> list[np.ndarray]:
    """Partition the DESC-sorted spectrum's direction indices into geometric
    eigenvalue shells (log-spaced edges); empty shells are dropped. Covers every
    direction exactly once (unit-pinned)."""
    ev = np.asarray(eigvals, dtype=np.float64)
    pos_floor = ev[ev > 0].min() if (ev > 0).any() else 1e-30
    ev = np.maximum(ev, pos_floor)
    lo, hi = float(ev.min()), float(ev.max())
    if lo == hi:
        return [np.arange(len(ev))]
    edges = np.geomspace(lo, hi, n_shells + 1)
    edges[0] = lo * (1 - 1e-12)
    edges[-1] = hi * (1 + 1e-12)
    bins = np.clip(np.searchsorted(edges, ev, side="right") - 1, 0, n_shells - 1)
    return [np.where(bins == b)[0] for b in range(n_shells) if int((bins == b).sum())]


def overlap_observed(s_pred: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    """cos of principal angles between the coordinate subspace on ``s_pred`` and the
    orthonormal basis ``B`` (Q coords) — svdvals of the k x kb overlap (the
    issue825 principal_angles convention); O = mean cos^2."""
    M = torch.as_tensor(B[np.asarray(s_pred)], dtype=torch.float32)
    cs = torch.linalg.svdvals(M).clamp(0.0, 1.0).numpy()
    return cs, float((cs**2).mean())


def overlap_null_draws(
    s_pred: np.ndarray,
    B: np.ndarray,
    shells: list[np.ndarray],
    n_draws: int,
    seed: int,
    device: str,
) -> np.ndarray:
    """K within-shell-rotation null draws of O — ALL draws materialized as stacked
    per-shell Haar-rotation GEMMs + ONE batched svdvals (no per-draw python loop).
    Only the s_pred rows of the rotated basis are ever materialized."""
    s_pred = np.asarray(s_pred, dtype=np.int64)
    k, kb = len(s_pred), B.shape[1]
    pred_pos_of_dim = {int(u): i for i, u in enumerate(s_pred)}
    M = torch.empty((n_draws, k, kb), dtype=torch.float32, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    Bt = torch.as_tensor(B, dtype=torch.float32, device=device)
    for rows in shells:
        needed = [int(u) for u in rows if int(u) in pred_pos_of_dim]
        if not needed:
            continue
        d_s, m = len(rows), len(needed)
        out_rows = torch.as_tensor([pred_pos_of_dim[u] for u in needed], device=device)
        # Only m rows of the shell's Haar rotation are consumed; a subset of Haar
        # rows is uniform on the Stiefel manifold V_m(R^d_s) — generate it directly
        # via QR of a (d_s, m) gaussian (memory K x d_s x m, never K x d_s^2: the
        # dense spectrum TAIL puts thousands of dims in one geometric shell).
        G = torch.randn((n_draws, d_s, m), generator=gen).to(device)
        Qc, R = torch.linalg.qr(G)
        sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        Qc = Qc * sign.unsqueeze(-2)  # Haar sign fix -> uniform Stiefel columns
        rot = Qc.transpose(-2, -1) @ Bt[torch.as_tensor(rows, device=device), :]
        M[:, out_rows, :] = rot
    cs = torch.linalg.svdvals(M).clamp(0.0, 1.0)
    return (cs**2).mean(dim=1).cpu().numpy()


def _top_eigvecs(G: torch.Tensor, k_max: int) -> np.ndarray:
    """Top-k_max eigenvectors of a PSD fp64 Gram, clamped to the effective rank
    (positive-eigenvalue directions only — a smoke-sized n < D PCA has rank < D)."""
    w, V = _eigh_desc(G)
    floor = float(w.max()) * 1e-10 if float(w.max()) > 0 else 0.0
    k_eff = min(k_max, int((w > floor).sum().item()))
    assert k_eff >= 1, "degenerate spectrum in _top_eigvecs"
    return V[:, :k_eff].cpu().numpy().astype(np.float32)


def _pca_basis(mat_proj: np.ndarray, k_max: int, device: str) -> np.ndarray:
    """Top-k_max principal directions of a CENTERED (n, D) matrix already in Q
    coords -> (D, k_eff) orthonormal columns, via the fp64 covariance eigh (a
    full gesdd SVD of the (n, D) matrix is ~500x more FLOPs at production n)."""
    X = torch.as_tensor(mat_proj, dtype=torch.float64, device=device)
    X = X - X.mean(0)
    return _top_eigvecs(X.T @ X, k_max)


def _sae_bases(args, k_max: int, device: str) -> dict[str, np.ndarray]:
    """The three representable-subspace constructions in Q coords: reconstruction
    PCA (primary), residual complement PCA, activity-weighted decoder SVD."""
    with np.load(args.out_root / "percontext_proj.npz") as z:
        R_proj = z["R_proj"].astype(np.float32)
        E_proj = z["E_proj"].astype(np.float32)
    out = {
        "psae_recon_pca": _pca_basis(R_proj, k_max, device),
        "presid_pca": _pca_basis(E_proj, k_max, device),
    }
    with np.load(args.out_root / "q_basis.npz") as z:
        Q = z["Q"].astype(np.float32)
    wj = np.load(args.out_root / "wj.npz")["wj"]
    feat_ids = np.load(args.out_root / "fve_profiles.npz")["feat_ids"].astype(np.int64)
    sae = _load_sae(args, int(args.sae_k))
    W = sae.w_dec[:, torch.as_tensor(feat_ids, device=sae.w_dec.device)].cpu().numpy()
    Wq = Q.T @ (W * wj[feat_ids][None, :].astype(np.float32))  # (D, n_active) in Q coords
    # top-k LEFT singular vectors via the (D, D) fp64 Gram eigh — never a full
    # gesdd of the (3,584 x 16,384) matrix (~500x the FLOPs, CPU-smoke-hostile)
    Wt = torch.as_tensor(Wq, dtype=torch.float64, device=device)
    out["psae_dec_svd"] = _top_eigvecs(Wt @ Wt.T, k_max)
    return out


def _pred_sets(args) -> dict[str, np.ndarray]:
    """P_pred selection profiles: per-direction holdout R2 rankings. Production:
    banked ridge (PRIMARY, 256 dirs) + banked mlp twin + this run's matched refit
    (full basis). Smoke: matched refit only (the banked profiles reference the
    PRODUCTION eigenbasis, which a smoke Q does not reproduce)."""
    with np.load(args.out_root / "perdirection_profiles.npz") as z:
        matched = z["r2u__t_vA_ctx"].astype(np.float64)
    out = {"matched_refit": matched}
    if not args.smoke:
        pca = json.loads(EA.COMMITTED_PERDIR_PCA.read_text())
        out["banked_ridge"] = np.asarray(pca["per_direction_r2"]["ridge"], np.float64)
        out["banked_mlp"] = np.asarray(pca["per_direction_r2"]["mlp_w8192"], np.float64)
    return out


def _s_pred(profile: np.ndarray, k: int) -> np.ndarray | None:
    """Top-k direction indices by per-direction R2 (None when k exceeds the
    profile's support — the banked 256-dir profiles cannot select k=256
    non-vacuously, hence the matched-refit twin for large k)."""
    if k > len(profile):
        return None
    return np.sort(np.argsort(profile)[::-1][:k])


def _primary_profile_for_k(args, k: int) -> str:
    """PRIMARY P_pred profile per k point (critic requirement: k-sweep labels its
    profile; k=128/256 read off the full-basis matched refit — top-k selection on
    a 256-dir banked profile is (near-)vacuous there)."""
    if args.smoke:
        return "matched_refit"
    return "banked_ridge" if k <= 64 else "matched_refit"


def phase_nulls(args) -> None:
    C.phase("s5_nulls")
    fp = fingerprint(args, include_path=True)
    if resume_ok(args, "nulls", fp):
        logger.info("[nulls] resume: fingerprint-valid done sentinel; skip")
        return
    _headroom(args, "nulls")
    # statistical-input existence (plan §4-S5): every battery input is S2/S4 output
    for req in ("percontext_proj.npz", "perdirection_profiles.npz", "q_basis.npz"):
        assert (args.out_root / req).exists(), f"[nulls] missing input {req}"
    dev = args.device if args.device == "cuda" else "cpu"
    eigvals = np.load(args.out_root / "q_basis.npz")["eigvals"]
    k_grid = list(args.k_grid)
    k_max = max(k_grid)
    bases = _sae_bases(args, k_max, dev)
    preds = _pred_sets(args)
    cells: list[dict] = []
    null_mats: dict[str, np.ndarray] = {}
    cell_i = 0
    t0 = time.time()
    for pair, B_full in bases.items():
        for k in k_grid:
            B = B_full[:, : min(k, B_full.shape[1])]
            for prof_name, profile in preds.items():
                sp = _s_pred(profile, k)
                if sp is None:
                    continue
                _, obs = overlap_observed(sp, B)
                primary_prof = prof_name == _primary_profile_for_k(args, k)
                rec = {
                    "pair": pair,
                    "k": int(k),
                    "k_eff": int(B.shape[1]),
                    "pred_profile": prof_name,
                    "primary_profile_for_k": primary_prof,
                    "observed_O": obs,
                    "nulls": {},
                }
                # null battery: PRIMARY profile per k only (plan cell arithmetic:
                # 5 k x 3 pairs x 3 shell settings); observed reported for all
                if primary_prof:
                    for ns_ in args.shell_grid:
                        shells = shell_partition(eigvals, int(ns_))
                        draws = overlap_null_draws(
                            sp, B, shells, args.angle_draws, 1_895_000 + cell_i, dev
                        )
                        cell_i += 1
                        q_pct = float(
                            100.0 * (np.sum(draws < obs) + 0.5 * np.sum(draws == obs)) / len(draws)
                        )
                        rec["nulls"][str(ns_)] = {
                            "p2.5": float(np.quantile(draws, 0.025)),
                            "p50": float(np.quantile(draws, 0.50)),
                            "p97.5": float(np.quantile(draws, 0.975)),
                            "q_percentile_of_observed": q_pct,
                            "n_draws": int(len(draws)),
                        }
                        null_mats[f"nullO__{pair}__k{k}__sh{ns_}"] = draws.astype(np.float32)
                        logger.info(
                            "[nulls] unit %d %s k=%d sh=%s O=%.4f band=[%.4f,%.4f] q=%.1f "
                            "elapsed=%.0fs",
                            cell_i,
                            pair,
                            k,
                            ns_,
                            obs,
                            rec["nulls"][str(ns_)]["p2.5"],
                            rec["nulls"][str(ns_)]["p97.5"],
                            q_pct,
                            time.time() - t0,
                        )
                cells.append(rec)
    # registered P_align atom: primary cell (k=64, recon-PCA pair, primary shell)
    prim = next(
        c
        for c in cells
        if c["pair"] == "psae_recon_pca" and c["k"] == 64 and c["primary_profile_for_k"]
    )
    q_align = prim["nulls"][str(args.n_shells)]["q_percentile_of_observed"]
    boot = _h3_bootstrap(args, fp, dev, null_mats)
    spot = _dark_spot_pilot(args, dev)
    C.write_json_atomic(
        args.out_eval / "angles_summary.json",
        {
            "cells": cells,
            "q_align": q_align,
            "P_align": bool(q_align > 97.5),
            "primary": {
                "k": 64,
                "pair": "psae_recon_pca",
                "shells": int(args.n_shells),
                "profile": _primary_profile_for_k(args, 64),
            },
            "h3_plugin_bootstrap": boot,
            "dark_spot_pilot_pure_ebar": spot,
            **C.reproducibility_metadata(),
        },
    )
    np.savez(
        args.out_root / "null_bands.npz",
        **null_mats,
        fingerprint=np.array(_fp_str(fp)),
    )
    write_done(args, "nulls", fp, {"q_align": q_align, "delta_dark_lo": boot["delta_dark_ci"][0]})
    logger.info("[nulls] done: q_align=%.2f delta_dark_lo=%.5f", q_align, boot["delta_dark_ci"][0])


def _boot_counts(rng: np.random.Generator, n: int, k_draws: int) -> np.ndarray:
    return rng.multinomial(n, np.full(n, 1.0 / n), size=k_draws).astype(np.float32)


def _h3_bootstrap(args, fp: dict, device: str, null_mats: dict) -> dict:
    """H3 plug-in + 10k-draw PAIRED bootstrap over holdout contexts — every draw
    recomputed from the per-context per-direction matrices as batched GEMMs (the
    #1482 pdshrink pattern; no per-draw python loop). Persists per-draw profiles
    (g_u, fve_u, energies) so honest re-reductions stay pure re-reductions."""
    with np.load(args.out_root / "percontext_proj.npz") as z:
        mats = {
            k2: torch.as_tensor(z[k2].astype(np.float32), device=device)
            for k2 in ("V_proj", "R_proj", "E_proj", "R1_proj", "R2_proj", "R3_proj")
        }
    n, D = mats["V_proj"].shape
    sq = {k2: (v * v) for k2, v in mats.items()}
    k_draws = int(args.boot_draws)
    rng = np.random.default_rng(args.seed)
    chunks = []
    per_draw_prof: dict[str, list] = {k2: [] for k2 in ("g_u", "fve_u", "energy_e", "sstot_v")}
    scalars: dict[str, list] = {
        k2: []
        for k2 in (
            "r2_vA",
            "r2_rbar",
            "r2_ebar",
            "plugin_e",
            "plugin_r",
            "delta_dark",
            "d21",
            "d31",
        )
    }
    chunk = max(1, min(1000, k_draws))
    for s2 in range(0, k_draws, chunk):
        c = min(chunk, k_draws - s2)
        W = torch.as_tensor(_boot_counts(rng, n, c), device=device)  # (c, n) counts
        mean_v = (W @ mats["V_proj"]) / n
        mean_r = (W @ mats["R_proj"]) / n
        mean_e = (W @ mats["E_proj"]) / n
        sstot_v = W @ sq["V_proj"] - n * mean_v**2
        sstot_r = W @ sq["R_proj"] - n * mean_r**2
        sstot_e = W @ sq["E_proj"] - n * mean_e**2
        rs1 = W @ sq["R1_proj"]
        rs2 = W @ sq["R2_proj"]
        rs3 = W @ sq["R3_proj"]
        g_u = 1.0 - rs1 / torch.clamp(sstot_v, min=1e-9)
        fve_u = 1.0 - sstot_e / torch.clamp(sstot_v, min=1e-9)
        r2_v = 1.0 - rs1.sum(1) / torch.clamp(sstot_v.sum(1), min=1e-9)
        r2_r = 1.0 - rs2.sum(1) / torch.clamp(sstot_r.sum(1), min=1e-9)
        r2_e = 1.0 - rs3.sum(1) / torch.clamp(sstot_e.sum(1), min=1e-9)
        plug_e = (sstot_e * g_u).sum(1) / torch.clamp(sstot_e.sum(1), min=1e-9)
        plug_r = (sstot_r * g_u).sum(1) / torch.clamp(sstot_r.sum(1), min=1e-9)
        per_draw_prof["g_u"].append(g_u.cpu().numpy().astype(np.float16))
        per_draw_prof["fve_u"].append(fve_u.cpu().numpy().astype(np.float16))
        per_draw_prof["energy_e"].append(sstot_e.cpu().numpy().astype(np.float16))
        per_draw_prof["sstot_v"].append(sstot_v.cpu().numpy().astype(np.float16))
        for k2, v in (
            ("r2_vA", r2_v),
            ("r2_rbar", r2_r),
            ("r2_ebar", r2_e),
            ("plugin_e", plug_e),
            ("plugin_r", plug_r),
            ("delta_dark", r2_e - plug_e),
            ("d21", r2_r - r2_v),
            ("d31", r2_e - r2_v),
        ):
            scalars[k2].append(v.cpu().numpy().astype(np.float32))
        chunks.append(c)
    sc = {k2: np.concatenate(v) for k2, v in scalars.items()}
    # observed point estimates (no resampling)
    obs = {}
    ones = torch.ones((1, n), device=device)
    W1 = ones
    mean_v = (W1 @ mats["V_proj"]) / n
    sstot_v0 = (W1 @ sq["V_proj"] - n * mean_v**2)[0]
    sstot_e0 = (W1 @ sq["E_proj"] - n * ((W1 @ mats["E_proj"]) / n) ** 2)[0]
    sstot_r0 = (W1 @ sq["R_proj"] - n * ((W1 @ mats["R_proj"]) / n) ** 2)[0]
    g0 = 1.0 - (W1 @ sq["R1_proj"])[0] / torch.clamp(sstot_v0, min=1e-9)
    obs["plugin_e"] = float(((sstot_e0 * g0).sum() / torch.clamp(sstot_e0.sum(), min=1e-9)).cpu())
    obs["plugin_r"] = float(((sstot_r0 * g0).sum() / torch.clamp(sstot_r0.sum(), min=1e-9)).cpu())
    obs["r2_ebar"] = float(
        (1.0 - (W1 @ sq["R3_proj"])[0].sum() / torch.clamp(sstot_e0.sum(), min=1e-9)).cpu()
    )
    obs["delta_dark"] = obs["r2_ebar"] - obs["plugin_e"]
    ci = {k2: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k2, v in sc.items()}
    null_mats["boot_scalars__names"] = np.array(sorted(sc), dtype="<U16")
    for k2, v in sc.items():
        null_mats[f"boot__{k2}"] = v
    for k2, v in per_draw_prof.items():
        null_mats[f"bootprof__{k2}"] = np.concatenate(v)
    return {
        "n_holdout": int(n),
        "n_draws": k_draws,
        "observed": obs,
        "delta_dark_ci": ci["delta_dark"],
        "P_dark": bool(ci["delta_dark"][0] > 0),
        "delta_dark_tail_frac_le_0": float((sc["delta_dark"] <= 0).mean()),
        "paired_delta_r2_ci": {"t2_minus_t1": ci["d21"], "t3_minus_t1": ci["d31"]},
        "all_ci": ci,
    }


def _dark_spot_pilot(args, device: str) -> dict:
    """Critic-registered pure-e_bar spot-read on the pilot rows (they carry
    v_bar_mask AND r_exact): score the fitted T3 map's predictions against the
    PURE SAE error, plus the H3 plug-in on those rows — zero new fits."""
    ppath = args.out_root / "pilot" / "pilot_rows.npz"
    if not ppath.exists():
        return {"skipped": "pilot rows missing"}
    with np.load(ppath, allow_pickle=False) as z:
        pos_p = z["pos"].astype(np.int64)
        vmask = z["vmask"].astype(np.float32)
        r_exact = z["r_exact"].astype(np.float32)
    with np.load(args.out_root / "percontext_proj.npz") as z:
        te_pos = z["te_pos"].astype(np.int64)
        E_proj = z["E_proj"].astype(np.float32)
        R3_proj = z["R3_proj"].astype(np.float32)
        R1_proj = z["R1_proj"].astype(np.float32)
        V_proj = z["V_proj"].astype(np.float32)
    with np.load(args.out_root / "q_basis.npz") as z:
        Q = z["Q"].astype(np.float32)
    loc = {int(p): i for i, p in enumerate(te_pos)}
    rows = np.array([loc[int(p)] for p in pos_p if int(p) in loc], dtype=np.int64)
    keepm = np.array([int(p) in loc for p in pos_p], dtype=bool)
    if len(rows) < 2:
        return {"skipped": f"only {len(rows)} pilot rows in the holdout te set"}
    e_hat_pure = E_proj[rows] - R3_proj[rows]  # fitted T3 predictions in Q coords
    e_pure = (vmask[keepm] - r_exact[keepm]) @ Q
    r2_pure = float(PR._pooled_r2(e_hat_pure, e_pure))
    g0_full = 1.0 - (R1_proj**2).sum(0) / np.maximum(((V_proj - V_proj.mean(0)) ** 2).sum(0), 1e-9)
    energy = ((e_pure - e_pure.mean(0)) ** 2).sum(0)
    plug = float((energy * g0_full).sum() / max(1e-9, energy.sum()))
    # small paired bootstrap over the pilot rows (batched GEMMs)
    rng = np.random.default_rng(args.seed + 7)
    k_draws = min(int(args.boot_draws), 10_000)
    W = _boot_counts(rng, len(rows), k_draws)
    res_sq = ((e_pure - e_hat_pure) ** 2).astype(np.float32)
    tgt_sq = (e_pure**2).astype(np.float32)
    mean_t = (W @ e_pure) / len(rows)
    sstot = W @ tgt_sq - len(rows) * mean_t**2
    rs = W @ res_sq
    r2_d = 1.0 - rs.sum(1) / np.maximum(sstot.sum(1), 1e-9)
    plug_d = (sstot * g0_full[None, :]).sum(1) / np.maximum(sstot.sum(1), 1e-9)
    dd = r2_d - plug_d
    return {
        "n_pilot_in_te": int(len(rows)),
        "r2_pure_ebar": r2_pure,
        "plugin_pure_ebar": plug,
        "delta_dark_pure": r2_pure - plug,
        "delta_dark_pure_ci": [float(np.quantile(dd, 0.025)), float(np.quantile(dd, 0.975))],
    }


# ── S6 correlates (variance-partialled) + verdict lattice ────────────────────────


def _rank_rows(a: np.ndarray) -> np.ndarray:
    """Ordinal ranks along axis=1 via double argsort (bootstrap-draw batches;
    observed point estimates use EA._midrank for exact tie handling)."""
    order = np.argsort(a, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(a.shape[1])[None, :].repeat(a.shape[0], 0), axis=1)
    return ranks.astype(np.float64)


def _rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    za = a - a.mean(1, keepdims=True)
    zb = b - b.mean(1, keepdims=True)
    num = (za * zb).sum(1)
    den = np.sqrt((za**2).sum(1) * (zb**2).sum(1))
    return num / np.maximum(den, 1e-12)


def _partial_from_corrs(r_ab, r_ac, r_bc) -> np.ndarray:
    return (r_ab - r_ac * r_bc) / np.maximum(
        np.sqrt(np.clip((1 - r_ac**2) * (1 - r_bc**2), 1e-12, None)), 1e-12
    )


def _spearman_p(rho: float, n: int) -> float:
    """Large-n normal approx p-value (Fisher z) — exploratory-family FDR input."""
    import math

    if n < 4 or not np.isfinite(rho):
        return float("nan")
    z = abs(float(np.arctanh(np.clip(rho, -0.999999, 0.999999)))) * np.sqrt(n - 3)
    return float(math.erfc(z / np.sqrt(2.0)))


def _bh_fdr(pvals: list[float], q: float = 0.05) -> list[bool]:
    p = np.asarray(pvals, dtype=np.float64)
    m = int(np.isfinite(p).sum())
    passed = np.zeros(len(p), dtype=bool)
    if m == 0:
        return passed.tolist()
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    thresh = 0.0
    for rank, idx in enumerate(order[:m], start=1):
        if p[idx] <= q * rank / m:
            thresh = p[idx]
    passed = np.isfinite(p) & (p <= thresh) if thresh > 0 else passed
    return passed.tolist()


def _midrank_1d(a: np.ndarray) -> np.ndarray:
    return EA._midrank(np.asarray(a, np.float64)[:, None])[:, 0]


def _partial_spearman_obs(x: np.ndarray, y: np.ndarray, covs: list[np.ndarray]) -> float:
    """Observed partial Spearman via midranks + lstsq residualization (multi-cov)."""
    rx, ry = _midrank_1d(x), _midrank_1d(y)
    Cv = np.column_stack([np.ones(len(rx))] + [_midrank_1d(c) for c in covs])
    bx, *_ = np.linalg.lstsq(Cv, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Cv, ry, rcond=None)
    ex, ey = rx - Cv @ bx, ry - Cv @ by
    den = np.sqrt((ex**2).sum() * (ey**2).sum())
    return float((ex * ey).sum() / max(den, 1e-12))


def phase_correlates(args) -> None:
    C.phase("s6_correlates")
    fp = fingerprint(args, include_path=True)
    if resume_ok(args, "correlates", fp):
        logger.info("[correlates] resume: fingerprint-valid done sentinel; skip")
        return
    _headroom(args, "correlates")
    with np.load(args.out_root / "perdirection_profiles.npz") as z:
        g_matched = z["r2u__t_vA_ctx"].astype(np.float64)
        eigvals = z["eigvals"].astype(np.float64)
    fvez = np.load(args.out_root / "fve_profiles.npz")
    fve_u = fvez["fve_u"].astype(np.float64)
    n_dir = len(g_matched)
    var_rank = np.arange(n_dir, dtype=np.float64)  # Q desc-sorted: index IS variance rank
    exploratory: list[dict] = []
    # per-direction observed reads (primary n=3,584 matched; n<=256 banked cross-check)
    obs_rho = float(np.corrcoef(_midrank_1d(g_matched), _midrank_1d(fve_u))[0, 1])
    obs_partial = _partial_spearman_obs(g_matched, fve_u, [var_rank])
    deciles = np.array_split(np.arange(n_dir), 10)
    decile_rows = []
    for di, idx in enumerate(deciles):
        if len(idx) < 4:
            continue
        r = float(np.corrcoef(_midrank_1d(g_matched[idx]), _midrank_1d(fve_u[idx]))[0, 1])
        decile_rows.append({"decile": di, "n": int(len(idx)), "rho": r})
        exploratory.append({"name": f"perdir_decile_{di}", "rho": r, "p": _spearman_p(r, len(idx))})
    banked_block = {}
    if not args.smoke:
        pca = json.loads(EA.COMMITTED_PERDIR_PCA.read_text())
        for pname in ("ridge", "mlp_w8192"):
            prof = np.asarray(pca["per_direction_r2"][pname], np.float64)
            nn = len(prof)
            banked_block[pname] = {
                "rho": float(np.corrcoef(_midrank_1d(prof), _midrank_1d(fve_u[:nn]))[0, 1]),
                "partial_given_var_rank": _partial_spearman_obs(prof, fve_u[:nn], [var_rank[:nn]]),
                "n": nn,
            }
    # bootstrap CI of the per-direction partial (from the S5 per-draw profiles;
    # per-draw variance rank = rank of the draw's own holdout variance profile)
    with np.load(args.out_root / "null_bands.npz") as z:
        g_d = z["bootprof__g_u"].astype(np.float64)
        fve_d = z["bootprof__fve_u"].astype(np.float64)
        sstotv_d = z["bootprof__sstot_v"].astype(np.float64)
    ra = _rank_rows(g_d)
    rb = _rank_rows(fve_d)
    rc = _rank_rows(-sstotv_d)  # rank 0 = largest holdout variance (desc, like eigrank)
    r_ab = _rowwise_corr(ra, rb)
    r_ac = _rowwise_corr(ra, rc)
    r_bc = _rowwise_corr(rb, rc)
    partial_d = _partial_from_corrs(r_ab, r_ac, r_bc)
    rho_lo, rho_hi = float(np.quantile(partial_d, 0.025)), float(np.quantile(partial_d, 0.975))
    p_var0 = bool(rho_lo <= 0.0 <= rho_hi)
    # per-feature block (n = active features): banked map R2 + FVE_j + banked
    # consistency + banked activity + Var(d_j . vA) rank — all MECHANICAL (labels frozen)
    banked_pf = np.load(
        PROJECT_ROOT
        / "eval_results"
        / "issue_1482"
        / "sae_perfeature"
        / "sae_dense_in__mean__ridge.npz"
    )
    cons = np.load(
        PROJECT_ROOT
        / "eval_results"
        / "issue_1482"
        / "feature_correlates"
        / "consistency_perfeature.npz"
    )
    fj = fvez["feat_ids"].astype(np.int64)
    assert np.array_equal(fj, banked_pf["feat_ids"].astype(np.int64)), "feat_ids misaligned"
    cmap = {int(f): i for i, f in enumerate(cons["feat_ids"].astype(np.int64))}
    have = np.array([int(f) in cmap for f in fj], dtype=bool)
    ci_idx = np.array([cmap[int(f)] for f in fj[have]], dtype=np.int64)
    pf = {
        "r2": banked_pf["r2"].astype(np.float64)[have],
        "fve_j": fvez["fve_j"].astype(np.float64)[have],
        "consistency": cons["consistency"].astype(np.float64)[ci_idx],
        "activity": banked_pf["activity"].astype(np.float64)[have],
        "var_vA_rank": _midrank_1d(-fvez["varv_j"].astype(np.float64)[have]),
    }
    finite = np.isfinite(pf["r2"]) & np.isfinite(pf["fve_j"]) & np.isfinite(pf["consistency"])
    pf = {k2: v[finite] for k2, v in pf.items()}
    names = list(pf)
    corr_mat = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r = float(np.corrcoef(_midrank_1d(pf[a]), _midrank_1d(pf[b]))[0, 1])
            corr_mat[f"{a}__{b}"] = r
    pf_partial = _partial_spearman_obs(
        pf["r2"], pf["fve_j"], [pf["var_vA_rank"], pf["consistency"], pf["activity"]]
    )
    act_rank = _midrank_1d(pf["activity"])
    act_deciles = np.array_split(np.argsort(act_rank), 10)
    pf_decile_rows = []
    for di, idx in enumerate(act_deciles):
        if len(idx) < 4:
            continue
        r = float(np.corrcoef(_midrank_1d(pf["r2"][idx]), _midrank_1d(pf["fve_j"][idx]))[0, 1])
        pf_decile_rows.append({"decile": di, "n": int(len(idx)), "rho": r})
        exploratory.append(
            {"name": f"perfeat_activity_decile_{di}", "rho": r, "p": _spearman_p(r, len(idx))}
        )
    fdr_pass = _bh_fdr([e["p"] for e in exploratory])
    for e, ok in zip(exploratory, fdr_pass, strict=True):
        e["bh_fdr_q05_pass"] = bool(ok)
    # verdict lattice (plan §3 — DISJOINT and exhaustive over the registered atoms;
    # the three predicates are pre-registered and EXCLUDED from the FDR family)
    angles = json.loads((args.out_eval / "angles_summary.json").read_text())
    q_align = float(angles["q_align"])
    dd_lo = float(angles["h3_plugin_bootstrap"]["delta_dark_ci"][0])
    p_align = q_align > 97.5
    p_dark = dd_lo > 0.0
    if p_align and not p_dark:
        verdict = "shared-structure (H1)"
    elif (not p_align) and p_dark:
        verdict = "orthogonal (H2)"
    elif p_align and p_dark:
        verdict = "mixed"
    elif p_var0:
        verdict = "variance-trivial (H3)"
    else:
        verdict = "inconclusive"
    lattice = {
        "q_align": q_align,
        "delta_dark_lo": dd_lo,
        "rho_ci": [rho_lo, rho_hi],
        "P_align": p_align,
        "P_dark": p_dark,
        "P_var0": p_var0,
        "verdict": verdict,
    }
    C.write_json_atomic(
        args.out_eval / "correlates_summary.json",
        {
            "per_direction": {
                "n": n_dir,
                "rho_matched_vs_fveu": obs_rho,
                "partial_given_var_rank": obs_partial,
                "partial_bootstrap_ci": [rho_lo, rho_hi],
                "P_var0": p_var0,
                "decile_stratified": decile_rows,
                "banked_cross_check": banked_block,
            },
            "per_feature": {
                "n": int(finite.sum()),
                "spearman_matrix": corr_mat,
                "partial_r2_fvej_given_varrank_consistency_activity": pf_partial,
                "activity_decile_stratified": pf_decile_rows,
            },
            "exploratory_bh_fdr": exploratory,
            "lattice": lattice,
            **C.reproducibility_metadata(),
        },
    )
    C.write_json_atomic(args.out_eval / "lattice.json", lattice)
    write_done(args, "correlates", fp, {"verdict": verdict})
    logger.info(
        "[correlates] done: verdict=%s (partial rho CI [%.3f, %.3f])", verdict, rho_lo, rho_hi
    )


# ── S7 upload + results sentinel ─────────────────────────────────────────────────


def _hf_prefix(args) -> str:
    return args.hf_prefix + ("_smoke" if args.smoke else "")


def _upload_dir_failloud(args, local_dir: Path, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    url = hub._upload(local_dir, C.HF_DATA_REPO, "dataset", path_in_repo, raise_on_error=True)
    assert url, f"upload returned no path for {local_dir}"
    expected = sorted(
        f"{path_in_repo}/{p.relative_to(local_dir)}" for p in local_dir.rglob("*") if p.is_file()
    )
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), C.HF_DATA_REPO, expected, path_in_repo=path_in_repo, repo_type="dataset"
    )
    assert not missing, f"upload verify: {len(missing)} missing under {path_in_repo}: {missing[:5]}"
    logger.info("[upload] %s -> %s (%d files verified)", local_dir, path_in_repo, len(expected))


def _upload_file_failloud(args, local: Path, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    url = hub._upload(
        local, C.HF_DATA_REPO, "dataset", path_in_repo, upload_as_file=True, raise_on_error=True
    )
    assert url, f"upload returned no path for {local}"
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), C.HF_DATA_REPO, [path_in_repo], path_in_repo=path_in_repo, repo_type="dataset"
    )
    assert not missing, f"upload verify: {path_in_repo} missing after upload"
    logger.info("[upload] %s -> %s (verified)", local.name, path_in_repo)


def _results_sentinel(args, t_start: float) -> None:
    logs_dir = Path("/workspace/logs")
    if args.smoke or not logs_dir.is_dir():
        # a VM smoke must never write a poller-drainable /workspace sentinel
        logs_dir = args.out_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    fits = json.loads((args.out_eval / "fits_summary.json").read_text())
    angles = json.loads((args.out_eval / "angles_summary.json").read_text())
    lattice = json.loads((args.out_eval / "lattice.json").read_text())
    pilot = json.loads((args.out_eval / "pilot_gate.json").read_text())
    gpus = EA._physical_gpu_ids()
    hours = (time.time() - t_start) / 3600.0 * max(1, len(gpus))
    prefix = _hf_prefix(args)
    payload = {
        "sentinel_schema_version": C.SENTINEL_SCHEMA_VERSION,
        "kind": "epm:results",
        "version": 1,
        "task_id": TASK_ID,
        "by": "issue1895_subspaces",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "issue-1895 phases S0-S7 complete (S8 VM harvest runs off-pod)",
        "eval_numbers": {
            "verdict": lattice["verdict"],
            "q_align": lattice["q_align"],
            "delta_dark_lo": lattice["delta_dark_lo"],
            "rho_ci": lattice["rho_ci"],
            "path": pilot["path"],
            "pooled_r2": {cell: doc["pooled_r2"] for cell, doc in fits["cells"].items()},
            "primary_O64": next(
                c["observed_O"]
                for c in angles["cells"]
                if c["pair"] == "psae_recon_pca" and c["k"] == 64 and c["primary_profile_for_k"]
            ),
        },
        "eval_paths": {
            "fits": str(args.out_eval / "fits_summary.json"),
            "angles": str(args.out_eval / "angles_summary.json"),
            "correlates": str(args.out_eval / "correlates_summary.json"),
            "pilot_gate": str(args.out_eval / "pilot_gate.json"),
            "lattice": str(args.out_eval / "lattice.json"),
        },
        "reproducibility_card": {
            **C.reproducibility_metadata(),
            "layer": LAYER,
            "sae_repo": S.SAE_REPO,
            "sae_revision": S.SAE_REVISION,
            "sae_k": int(args.sae_k),
            "split_shas": committed_split_shas(),
            "seed": args.seed,
            "config_digest": config_digest(args, include_path=True),
        },
        "wandb_url": None,  # no training — deterministic fits log to JSON (plan §10)
        "hf_hub_url": f"https://huggingface.co/datasets/{C.HF_DATA_REPO}/tree/main/{prefix}",
        "worktree_path": str(PROJECT_ROOT),
        "final_commit_sha": _code_sha(),
        "gpu_hours_used": round(hours, 2),
        "gpu_hours_budgeted": 10,
        "plan_deviations": [],
    }
    path = logs_dir / f"issue-{TASK_ID}-results.json"
    C.write_json_atomic(path, payload)
    logger.info("Wrote results sentinel %s", path)


def phase_upload(args, t_start: float) -> None:
    C.phase("s7_upload")
    fp = fingerprint(args, include_path=True)
    _headroom(args, "upload")
    prefix = _hf_prefix(args)
    # eval JSONs + the two summary npz -> eval_results dest (whole tree, no
    # eligibility filter — plan §6.5 deliverables + upload-parity self-check)
    _upload_dir_failloud(args, args.out_eval, f"{prefix}/eval_results_issue_{TASK_ID}")
    for name in ("perdirection_profiles.npz", "null_bands.npz"):
        _upload_file_failloud(
            args, args.out_root / name, f"{prefix}/eval_results_issue_{TASK_ID}/{name}"
        )
    # analysis tensors
    for name in ("q_basis.npz", "wj.npz", "fve_profiles.npz", "percontext_proj.npz"):
        _upload_file_failloud(args, args.out_root / name, f"{prefix}/analysis_tensors/{name}")
    _upload_dir_failloud(args, args.out_root / "rbar", f"{prefix}/analysis_tensors/rbar")
    if (args.out_root / "pilot" / "pilot_rows.npz").exists():
        _upload_file_failloud(
            args,
            args.out_root / "pilot" / "pilot_rows.npz",
            f"{prefix}/analysis_tensors/pilot/pilot_rows.npz",
        )
    if (args.out_root / "vmask").is_dir() and any((args.out_root / "vmask").iterdir()):
        _upload_dir_failloud(args, args.out_root / "vmask", f"{prefix}/analysis_tensors/vmask")
    _results_sentinel(args, t_start)
    write_done(args, "upload", fp, {"hf_prefix": prefix})
    logger.info("[upload] done -> %s", prefix)


# ── CLI ──────────────────────────────────────────────────────────────────────────

PHASES = (
    "stage",
    "matrices",
    "recon",
    "pilot",
    "capture-matched",
    "fits",
    "nulls",
    "angles",  # alias of nulls
    "correlates",
    "upload",
    "all",
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument("--phase", required=True, choices=PHASES)
    ap.add_argument("--out-root", type=Path, default=Path("data/issue_1895"))
    ap.add_argument("--out-eval", type=Path, default=Path("eval_results/issue_1895"))
    ap.add_argument("--sae-k", type=int, default=64, choices=(64, 128))
    ap.add_argument("--k-grid", type=str, default="16,32,64,128,256")
    ap.add_argument("--n-shells", type=int, default=32)
    ap.add_argument("--shell-grid", type=str, default="16,32,64")
    ap.add_argument("--angle-draws", type=int, default=1000)
    ap.add_argument("--boot-draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1895)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--pilot-n", type=int, default=512)
    ap.add_argument("--hf-prefix", default=HF_PREFIX_DEFAULT)
    ap.add_argument(
        "--pass-b",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue_779" / "pass_b" / "train_context_vectors.pt",
    )
    ap.add_argument("--force-path", choices=("auto", "A", "B"), default="auto")
    ap.add_argument("--smoke", action="store_true", help="tiny slice; own out-roots")
    ap.add_argument("--max-shards", type=int, default=8, help="smoke: pooled shards staged")
    ap.add_argument("--holdout-cap", type=int, default=500, help="smoke holdout cap")
    ap.add_argument("--fit-cap", type=int, default=2000, help="smoke fit cap")
    ap.add_argument("--val-cap", type=int, default=200, help="smoke val cap")
    ap.add_argument("--tiny-model", action="store_true", help="CPU carve-out capture model")
    ap.add_argument("--import-check", action="store_true", help="resolve deferred imports; exit")
    return ap


def _import_check() -> None:
    """Execute every deferred import the phase bodies touch (Axis-1 leg)."""
    import transformers  # noqa: F401
    from huggingface_hub import HfApi  # noqa: F401

    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    for sym in (
        hub.stage_hub_prefix,
        hub.stage_hub_file,
        hub.retry_transient,
        hub.list_hf_files_under_path,
        hub._upload,
        hub.verify_repo_paths_uploaded,
        assert_out_root_headroom,
        identity_bias_predict,
        knn_retrieval,
        EA._assemble_with_ci,
        EA._shared_gram_ridge_multi,
        EA._per_feature_metrics,
        EA._row_features,
        EA._tokenize_row,
        EA._batched_capture,
        EA._load_model_tok,
        EA._stratified_sample,
        EA._iter_needed_rows,
        EA._raw_chunk_names,
        EA._midrank,
        EA._physical_gpu_ids,
        N1M._ridge_factorize,
        N1M._ridge_predict_one,
        N1M._fit_mlp_minibatch,
        N1M._pool_rows,
        N1M._download_chunk_with_retry,
        PR._pooled_r2,
        S.BatchTopKSAE.load,
        S.BatchTopKSAE.ensure_downloaded,
        S.token_inlier_mask,
    ):
        assert callable(sym), sym
    print("[import-check] OK")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.import_check:
        _import_check()
        sys.exit(0)
    args.out_root = Path(args.out_root)
    args.out_eval = Path(args.out_eval)
    args.scratch = args.out_root / "scratch"
    args.sae_dir = args.out_root / "sae"
    args.k_grid = tuple(int(x) for x in str(args.k_grid).split(","))
    args.shell_grid = tuple(int(x) for x in str(args.shell_grid).split(","))
    args.max_chunks = args.max_shards if args.smoke else 0
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.out_eval.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    phase = "nulls" if args.phase == "angles" else args.phase
    seq = (
        [
            "stage",
            "matrices",
            "recon",
            "pilot",
            "capture-matched",
            "fits",
            "nulls",
            "correlates",
            "upload",
        ]
        if phase == "all"
        else [phase]
    )
    for ph in seq:
        if ph == "stage":
            phase_stage(args)
        elif ph == "matrices":
            phase_matrices(args)
        elif ph == "recon":
            phase_recon(args)
        elif ph == "pilot":
            phase_pilot(args)
        elif ph == "capture-matched":
            phase_capture_matched(args)
        elif ph == "fits":
            phase_fits(args)
        elif ph == "nulls":
            phase_nulls(args)
        elif ph == "correlates":
            phase_correlates(args)
        elif ph == "upload":
            phase_upload(args, t_start)
    C.phase("done")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # explicit exit BEFORE C-extension finalize teardown (PyGILState race)


if __name__ == "__main__":
    main()
