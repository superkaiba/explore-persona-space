#!/usr/bin/env python
"""Issue #1773 Phase 0 — mechanical axes for all 16,384 restricted SAE features.

One detached VM CPU run (thread-capped launch env; plan §4 Phase 0). Inputs:
the #1482 pooled store (1,920 `pooled_*.npz` shards), the committed restricted
key (`sae_ctx__mean__ridge.npz`: feat_ids/r2/activity), the pinned BatchTopK
SAE decoder, Qwen-2.5-7B-Instruct lm_head + final-norm gamma, and the #779 r_B
persona directions. Outputs `eval_results/issue_1773/phase0/feature_table.jsonl`
(one row per feature, joinable on feat_id) + `phase0_arrays.npz` + meta.

Per-axis computations are vectorized (streamed bincount over shards; chunked
fp32 GEMMs for the logit footprint + neighbour cosines) — no per-feature Python
loop touches data. The recomputed-activity WIRING GATE (<=1e-3 vs the committed
npz) asserts on a FULL-store run and is demoted to an informational log line
under `--max-shards` smokes (gate-calibration parity rule).

Smoke: `--max-shards 5 --feature-limit 64 --out-root <scratch>`.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM discipline)

import issue1773_common as CM  # noqa: E402
import numpy as np  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
FOOTPRINT_CHUNK = 1024  # feature block for the W_U GEMM (plan §4 table)
NEIGHBOR_CHUNK = 1024
TOP_TOKENS = 10
SCAFFOLD_RANK = 48  # parity with #1092 v14 SCAFFOLD_CONTROL (self-contained v1)
MASSIVE_DIM_PCTL = 99.9  # dims flagged by mean|h_prefix| percentile (recorded constant)
CKPT_EVERY = 256  # accumulator checkpoint grain (1,920 units > 50 → intra-phase rule)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


# ── pre-flight (plan §10 + assumptions 4/5/6) ────────────────────────────────


def preflight(args) -> None:
    """Revision-scoped SAE probe + r_B realized-keys check + tie_word_embeddings."""
    import subprocess

    import issue1482_sae as S
    from huggingface_hub import list_repo_tree

    from explore_persona_space.orchestrate import hub

    for stem in ("ae.pt", "config.json", "eval_results.json"):
        files = hub.retry_transient(
            lambda: [
                f.path
                # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient right here
                for f in list_repo_tree(
                    S.SAE_REPO,
                    path_in_repo="resid_post_layer_19/trainer_1",
                    revision=S.SAE_REVISION,
                )
            ],
            what="sae revision-scoped listing",
        )
        assert any(p.endswith(stem) for p in files), f"SAE stem {stem} missing at pin: {files}"
    _log(f"[phase0-preflight] SAE revision-scoped probe PASS @ {S.SAE_REVISION[:8]}")

    rb = PROJECT_ROOT / "data/issue_779/r_b/evil.pt"
    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(PROJECT_ROOT / "scripts/verify_reused_artifact_keys.py"),
            "--artifact",
            str(rb),
            "--keys",
            "r_b,layers,trait,smoke",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    assert proc.returncode == 0, f"r_B realized-keys check FAILED: {proc.stdout} {proc.stderr}"
    _log(f"[phase0-preflight] r_B realized-keys PASS: {proc.stdout.strip()}")

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_ID)
    assert cfg.tie_word_embeddings is False, (
        f"tie_word_embeddings={cfg.tie_word_embeddings}: lm_head is not a separate matrix; "
        "the logit-footprint read assumed an untied head (plan assumption 6)"
    )
    _log("[phase0-preflight] tie_word_embeddings=False PASS")


# ── streamed store pass (activity / persist / side_ratio / h_prefix stats) ──


def stream_store(args, fid: np.ndarray, committed_activity: np.ndarray) -> dict:
    """One pass over pooled shards: fit-row activity counts (wiring gate),
    ans_frac accumulators (persist_answer), psi counts (side_ratio), and
    h_prefix mean-|.| + covariance accumulators (nuisance/scaffold).

    Checkpoints accumulators every CKPT_EVERY shards (atomic npz) with a resume
    predicate keyed on (store, n_shards, max_shards) — intra-phase rule.
    """
    shards = sorted(args.store.glob("pooled_*.npz"))
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    else:
        assert len(shards) == CM.N_SHARDS, f"expected {CM.N_SHARDS} shards, found {len(shards)}"
    D = CM.DICT_SIZE
    H = CM.ACT_DIM
    state = {
        "cnt_fit": np.zeros(D, np.int64),
        "sum_frac": np.zeros(D, np.float64),
        "sum_frac_sq": np.zeros(D, np.float64),
        "psi_cnt_fit": np.zeros(D, np.int64),
        "h_sum": np.zeros(H, np.float64),
        "h_abs_sum": np.zeros(H, np.float64),
        "h_outer": np.zeros((H, H), np.float64),
        "n_fit": 0,
        "n_rows": 0,
        "next_shard": 0,
    }
    fingerprint = CM.sha16(f"{args.store}|{len(shards)}|{args.max_shards}")
    ckpt = args.work / f"phase0_stream_ckpt_{fingerprint}.npz"
    if ckpt.exists() and not args.no_resume:
        z = np.load(ckpt, allow_pickle=False)
        for k in state:
            state[k] = z[k] if isinstance(state[k], np.ndarray) else int(z[k])
        _log(f"[phase0] resume from checkpoint at shard {state['next_shard']}/{len(shards)}")

    t0 = time.time()
    for i in range(int(state["next_shard"]), len(shards)):
        p = shards[i]
        with np.load(p, allow_pickle=False) as z:
            tag = np.asarray(z["set_tag"])
            fit = tag == 1
            off = np.asarray(z["idx_off"], dtype=np.int64)
            idx = np.asarray(z["ans_idx"], dtype=np.int64)
            frac = np.asarray(z["ans_frac"], dtype=np.float64)
            keep = np.repeat(fit, off)
            ik, fk = idx[keep], frac[keep]
            state["cnt_fit"] += np.bincount(ik, minlength=D)
            state["sum_frac"] += np.bincount(ik, weights=fk, minlength=D)
            state["sum_frac_sq"] += np.bincount(ik, weights=fk * fk, minlength=D)
            psi_off = np.asarray(z["psi_off"], dtype=np.int64)
            psi_idx = np.asarray(z["psi_idx"], dtype=np.int64)
            state["psi_cnt_fit"] += np.bincount(psi_idx[np.repeat(fit, psi_off)], minlength=D)
            hp = np.asarray(z["h_prefix"], dtype=np.float64)
            state["h_sum"] += hp.sum(0)
            state["h_abs_sum"] += np.abs(hp).sum(0)
            state["h_outer"] += hp.T @ hp
            state["n_fit"] += int(fit.sum())
            state["n_rows"] += int(len(tag))
        state["next_shard"] = i + 1
        _log(
            f"[phase0] shard {i + 1}/{len(shards)} {p.name} n_fit={state['n_fit']} "
            f"elapsed={time.time() - t0:.0f}s rss={_rss_gb():.1f}GiB"
        )
        if (i + 1) % CKPT_EVERY == 0:
            tmp = ckpt.parent / f".tmp_{ckpt.name}"
            np.savez(tmp, **state)
            os.replace(tmp, ckpt)
            _log(f"[phase0] checkpoint at shard {i + 1}")

    # Wiring gate (H1): recomputed activity vs committed npz. FULL run asserts;
    # a --max-shards smoke logs informationally (gate-calibration parity rule).
    act_re = state["cnt_fit"][fid] / max(state["n_fit"], 1)
    gate = float(np.abs(act_re - committed_activity).max())
    full = args.max_shards <= 0
    _log(f"[phase0] wiring gate: n_fit={state['n_fit']} max|delta|={gate:.2e} full={full}")
    assert_wiring_gate(gate, full=full)
    state["activity_recomputed"] = act_re
    state["wiring_gate"] = gate
    return state


def assert_wiring_gate(gate: float, *, full: bool) -> None:
    """H1 wiring gate: HARD assert on a full-store run; informational under smoke."""
    if full:
        assert gate < 1e-3, f"activity mismatch vs committed npz (max|delta|={gate})"
    else:
        print(f"[phase0] SMOKE: wiring gate informational (max|delta|={gate:.2e})", flush=True)


# ── decoder-side axes (footprint / neighbours / rb_align / nuisance) ────────


def _load_lm_head_and_gamma(scratch: Path) -> tuple[np.ndarray, np.ndarray]:
    """Stage ONLY the safetensors shard(s) holding lm_head.weight + model.norm.weight
    (single-file hf_hub_download via hub.retry_transient; never snapshot_download)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from explore_persona_space.orchestrate import hub

    idx_path = hub.retry_transient(
        lambda: hf_hub_download(MODEL_ID, "model.safetensors.index.json", local_dir=str(scratch)),
        what="lm_head index fetch",
    )
    weight_map = json.loads(Path(idx_path).read_text())["weight_map"]
    needed = {
        "lm_head.weight": weight_map["lm_head.weight"],
        "model.norm.weight": weight_map["model.norm.weight"],
    }
    tensors: dict[str, np.ndarray] = {}
    for shard in sorted(set(needed.values())):
        sp = hub.retry_transient(
            lambda s=shard: hf_hub_download(MODEL_ID, s, local_dir=str(scratch)),
            what=f"lm_head shard fetch ({shard})",
        )
        # framework="pt": the shards are bf16, which the numpy framework cannot
        # decode ("data type 'bfloat16' not understood") — upcast via torch.
        with safe_open(sp, framework="pt") as f:
            for name, sh in needed.items():
                if sh == shard and name not in tensors:
                    tensors[name] = f.get_tensor(name).float().numpy()
    w_u = tensors["lm_head.weight"]
    gamma = tensors["model.norm.weight"]
    assert w_u.ndim == 2 and w_u.shape[1] == CM.ACT_DIM, w_u.shape
    assert gamma.shape == (CM.ACT_DIM,), gamma.shape
    return w_u, gamma


def logit_footprint(w_u: np.ndarray, gamma: np.ndarray, w_dec: np.ndarray, tok) -> list[dict]:
    """E = W_U @ (gamma ⊙ W_dec[:, feats]) in FOOTPRINT_CHUNK blocks (fp32).
    Per feature: top-10 promoted + suppressed token ids/strings/values +
    concentration (share of positive-logit mass in top-10)."""
    n_feat = w_dec.shape[1]
    rows: list[dict] = []
    scaled = w_dec * gamma[:, None]
    t0 = time.time()
    for s in range(0, n_feat, FOOTPRINT_CHUNK):
        block = scaled[:, s : s + FOOTPRINT_CHUNK].astype(np.float32)
        logits = w_u @ block  # (V, B)
        top = np.argpartition(-logits, TOP_TOKENS - 1, axis=0)[:TOP_TOKENS]
        bot = np.argpartition(logits, TOP_TOKENS - 1, axis=0)[:TOP_TOKENS]
        pos_mass = np.where(logits > 0, logits, 0.0).sum(0)
        for j in range(block.shape[1]):
            ti = top[:, j][np.argsort(-logits[top[:, j], j])]
            bi = bot[:, j][np.argsort(logits[bot[:, j], j])]
            top_vals = logits[ti, j]
            rows.append(
                {
                    "top_promoted_ids": [int(t) for t in ti],
                    "top_promoted_tokens": [tok.decode([int(t)]) for t in ti],
                    "top_promoted_vals": [round(float(v), 4) for v in top_vals],
                    "top_suppressed_ids": [int(t) for t in bi],
                    "top_suppressed_tokens": [tok.decode([int(t)]) for t in bi],
                    "top_suppressed_vals": [round(float(logits[t, j]), 4) for t in bi],
                    "concentration": float(
                        np.where(top_vals > 0, top_vals, 0.0).sum() / max(pos_mass[j], 1e-12)
                    ),
                }
            )
        _log(
            f"[phase0] footprint block {s // FOOTPRINT_CHUNK + 1}/"
            f"{(n_feat + FOOTPRINT_CHUNK - 1) // FOOTPRINT_CHUNK} "
            f"elapsed={time.time() - t0:.0f}s"
        )
    return rows


def neighbor_table(w_dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-N_NEIGHBORS decoder-cosine neighbours WITHIN the restricted set
    (chunked GEMM over normalized columns; excludes self). Returns
    (idx (n, k) restricted positions, cos (n, k))."""
    xn = w_dec / np.maximum(np.linalg.norm(w_dec, axis=0, keepdims=True), 1e-12)
    xn32 = xn.astype(np.float32)
    n = xn32.shape[1]
    k = CM.N_NEIGHBORS
    nb_idx = np.zeros((n, k), np.int64)
    nb_cos = np.zeros((n, k), np.float32)
    for s in range(0, n, NEIGHBOR_CHUNK):
        g = xn32[:, s : s + NEIGHBOR_CHUNK].T @ xn32  # (B, n)
        for j in range(g.shape[0]):
            g[j, s + j] = -np.inf  # exclude self
        part = np.argpartition(-g, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(g, part, axis=1)
        order = np.argsort(-vals, axis=1)
        nb_idx[s : s + g.shape[0]] = np.take_along_axis(part, order, axis=1)
        nb_cos[s : s + g.shape[0]] = np.take_along_axis(vals, order, axis=1)
    return nb_idx, nb_cos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=CM.STORE_DEFAULT)
    ap.add_argument("--work", type=Path, default=CM.WORK_DEFAULT / "phase0_work")
    ap.add_argument("--out-root", type=Path, default=CM.OUT_EVAL, help="eval_results root")
    ap.add_argument("--max-shards", type=int, default=0, help=">0 = smoke slice (gate demoted)")
    ap.add_argument("--feature-limit", type=int, default=0, help=">0 = smoke feature subset")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    import issue1482_sae as S
    from issue1482_feature_extremes import _load_rb_layer

    if not args.skip_preflight:
        preflight(args)
    if args.preflight_only:
        _log("[phase0] preflight-only: done")
        return 0

    args.work.mkdir(parents=True, exist_ok=True)
    out_dir = args.out_root / "phase0"
    out_dir.mkdir(parents=True, exist_ok=True)

    com = np.load(CM.PERFEATURE_NPZ, allow_pickle=False)
    fid_all = np.asarray(com["feat_ids"], dtype=np.int64)
    assert len(fid_all) == 16_384, len(fid_all)
    r2 = np.asarray(com["r2"], dtype=np.float64)
    activity = np.asarray(com["activity"], dtype=np.float64)
    n_keep = args.feature_limit if args.feature_limit > 0 else len(fid_all)
    fid = fid_all[:n_keep]

    # 1) streamed store pass (gate + persist_answer + side_ratio + h stats)
    st = stream_store(args, fid_all, activity)

    # 2) SAE decoder
    sae = S.BatchTopKSAE.load(k=64, device="cpu", cache_dir=args.work / "sae")
    w_dec_full = sae.w_dec.numpy().astype(np.float64)  # (3584, 131072)
    w_dec = w_dec_full[:, fid]  # restricted (3584, n_keep)

    # 3) nuisance_load: massive-activation dims + scaffold projection
    mean_abs = st["h_abs_sum"] / max(st["n_rows"], 1)
    thr = np.percentile(mean_abs, MASSIVE_DIM_PCTL)
    massive_dims = np.where(mean_abs >= thr)[0]
    col_sq = w_dec**2
    col_mass = col_sq.sum(0)
    nuisance = col_sq[massive_dims].sum(0) / np.maximum(col_mass, 1e-12)
    mu = st["h_sum"] / max(st["n_rows"], 1)
    cov = st["h_outer"] / max(st["n_rows"], 1) - np.outer(mu, mu)
    evals, evecs = np.linalg.eigh(cov)
    basis = evecs[:, -SCAFFOLD_RANK:]  # (H, 48) top-48 PCA basis of h_prefix
    proj = basis.T @ w_dec  # (48, n_keep)
    scaffold_frac = (proj**2).sum(0) / np.maximum(col_mass, 1e-12)

    # 4) rb_align: |cos(W_dec[:,f], r_B[19])| raw AND scaffold-projected (#779)
    rb, rb_names = _load_rb_layer()
    rb_n = rb / np.linalg.norm(rb, axis=1, keepdims=True)
    dec_n = w_dec / np.maximum(np.linalg.norm(w_dec, axis=0, keepdims=True), 1e-12)
    rb_cos = np.abs(rb_n @ dec_n)  # (n_traits, n_keep)
    rb_p = basis.T @ rb_n.T  # (48, n_traits)
    dec_p = proj / np.maximum(np.linalg.norm(proj, axis=0, keepdims=True), 1e-12)
    rb_p_n = rb_p / np.maximum(np.linalg.norm(rb_p, axis=0, keepdims=True), 1e-12)
    rb_cos_scaffold = np.abs(rb_p_n.T @ dec_p)  # (n_traits, n_keep)

    # 5) neighbours (restricted-set decoder cosine)
    nb_idx, nb_cos = neighbor_table(w_dec)

    # 6) logit footprint
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    w_u, gamma = _load_lm_head_and_gamma(args.work / "hf_dl")
    foot = logit_footprint(w_u, gamma, w_dec, tok)

    # 7) per-feature table
    cnt = st["cnt_fit"][fid].astype(np.float64)
    psi_cnt = st["psi_cnt_fit"][fid].astype(np.float64)
    persist = np.where(cnt > 0, st["sum_frac"][fid] / np.maximum(cnt, 1), np.nan)
    ex2 = st["sum_frac_sq"][fid] / np.maximum(cnt, 1)
    persist_sd = np.sqrt(np.maximum(ex2 - persist**2, 0))
    side_ratio = cnt / np.maximum(cnt + psi_cnt, 1)

    rows = []
    for i in range(len(fid)):
        rows.append(
            {
                "feat_id": int(fid[i]),
                "restricted_idx": int(i),
                "r2": float(r2[i]),
                "density": {
                    "activity_committed": float(activity[i]),
                    "activity_recomputed": float(st["activity_recomputed"][i]),
                },
                "persist_answer": {"mean": float(persist[i]), "sd": float(persist_sd[i])},
                "side_ratio": float(side_ratio[i]),
                "nuisance_load": {
                    "massive_dim_mass": float(nuisance[i]),
                    "scaffold_frac": float(scaffold_frac[i]),
                },
                "rb_align": {
                    t: {"raw": float(rb_cos[j, i]), "scaffold": float(rb_cos_scaffold[j, i])}
                    for j, t in enumerate(rb_names)
                },
                "neighbors": {
                    "feat_ids": [int(fid[p]) for p in nb_idx[i]],
                    "cos": [round(float(c), 5) for c in nb_cos[i]],
                },
                "logit_footprint": foot[i],
                "tier": None,
                "persist_query": None,
                "arm_shares": None,
            }
        )

    table_path = out_dir / "feature_table.jsonl"
    tmp = table_path.parent / f".tmp_{table_path.name}"
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(table_path)

    np.savez(
        out_dir / "phase0_arrays.npz",
        feat_ids=fid,
        r2=r2[: len(fid)],
        activity=activity[: len(fid)],
        activity_recomputed=st["activity_recomputed"][: len(fid)],
        persist_answer=persist,
        persist_answer_sd=persist_sd,
        side_ratio=side_ratio,
        nuisance_massive=nuisance,
        scaffold_frac=scaffold_frac,
        rb_cos=rb_cos,
        rb_cos_scaffold=rb_cos_scaffold,
        rb_traits=np.array(rb_names, dtype=object),
        neighbor_idx=nb_idx,
        neighbor_cos=nb_cos,
        massive_dims=massive_dims,
        scaffold_basis=basis,
        pca_evals=evals[-SCAFFOLD_RANK:],
    )
    meta = {
        **CM.repro_meta(),
        "wiring_gate_max_delta": st["wiring_gate"],
        "n_fit": int(st["n_fit"]),
        "n_rows": int(st["n_rows"]),
        "n_features": len(fid),
        "max_shards": args.max_shards,
        "massive_dim_pctl": MASSIVE_DIM_PCTL,
        "massive_dims": [int(d) for d in massive_dims],
        "scaffold_rank": SCAFFOLD_RANK,
        "sae_revision": S.SAE_REVISION,
        "join_smoke_feat_ids_match": bool(np.array_equal(fid, fid_all[: len(fid)])),
    }
    (out_dir / "phase0_meta.json").write_text(json.dumps(meta, indent=1))
    _log(
        f"[phase0] done: {len(rows)} rows -> {table_path} "
        f"(gate={st['wiring_gate']:.2e}, rss={_rss_gb():.1f}GiB)"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
